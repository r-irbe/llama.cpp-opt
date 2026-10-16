#include "ggml-cpu-numa.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <atomic>
#include <filesystem>
#include <map>
#include <mutex>
#include <unordered_map>

namespace fs = std::filesystem;

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#if defined(__gnu_linux__)
#include <cerrno>
#include <cstring>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include "ggml.h"

// from linux/mempolicy.h, not exposed by glibc
#ifndef MPOL_DEFAULT
#define MPOL_DEFAULT 0
#endif
#ifndef MPOL_PREFERRED
#define MPOL_PREFERRED 1
#endif
#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_INTERLEAVE
#define MPOL_INTERLEAVE 3
#endif
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE 1
#endif
#ifndef MPOL_MF_STRICT
#define MPOL_MF_STRICT 2
#endif
#endif

namespace ggml::cpu::numa {

static std::atomic<bool> g_numa_expert{false};

bool is_numa_expert() {
    return g_numa_expert.load(std::memory_order_relaxed);
}

void set_numa_expert(bool enable) {
    g_numa_expert.store(enable, std::memory_order_relaxed);
}

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

std::vector<int> parse_list(const std::string & list) {
    std::vector<int> ids;

    size_t pos = 0;
    while (pos < list.size()) {
        const size_t end = list.find(',', pos);
        std::string  tok = list.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        pos = end == std::string::npos ? list.size() : end + 1;

        tok.erase(std::remove_if(tok.begin(), tok.end(), [](unsigned char c) { return std::isspace(c); }), tok.end());
        if (tok.empty()) {
            continue;
        }

        const size_t dash = tok.find('-');
        try {
            if (dash == std::string::npos) {
                ids.push_back(std::stoi(tok));
            } else {
                const int lo = std::stoi(tok.substr(0, dash));
                const int hi = std::stoi(tok.substr(dash + 1));
                for (int i = lo; i <= hi; i++) {
                    ids.push_back(i);
                }
            }
        } catch (const std::exception &) {
            return {};
        }
    }

    return ids;
}

// value of a "Node 0 MemTotal:  123 kB" style line, in bytes
static size_t parse_meminfo(const std::string & meminfo, const std::string & key) {
    std::istringstream ss(meminfo);
    std::string        line;
    while (std::getline(ss, line)) {
        const size_t k = line.find(key + ":");
        if (k == std::string::npos) {
            continue;
        }
        try {
            return (size_t) std::stoull(line.substr(k + key.size() + 1)) * 1024;
        } catch (const std::exception &) {
            return 0;
        }
    }
    return 0;
}

static std::vector<int> get_physical_cores(const std::string & sysfs_root, const std::vector<int> & cpus) {
    std::vector<int> cores;
    cores.reserve(cpus.size());

    for (int cpu : cpus) {
        std::string siblings;
        if (!read_file(sysfs_root + "/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list", siblings)) {
            // no topology information, assume the CPU is its own core
            cores.push_back(cpu);
            continue;
        }
        const std::vector<int> ids = parse_list(siblings);
        cores.push_back(ids.empty() ? cpu : *std::min_element(ids.begin(), ids.end()));
    }

    std::sort(cores.begin(), cores.end());
    cores.erase(std::unique(cores.begin(), cores.end()), cores.end());

    return cores;
}

std::vector<node> parse_topology(const std::string & sysfs_root, const std::vector<int> & cpu_mask) {
    const std::string node_root = sysfs_root + "/devices/system/node";

    std::string online;
    if (!read_file(node_root + "/online", online)) {
        return {};
    }

    std::vector<node> nodes;
    for (int id : parse_list(online)) {
        const std::string node_dir = node_root + "/node" + std::to_string(id);

        std::string cpulist;
        if (!read_file(node_dir + "/cpulist", cpulist)) {
            continue;
        }

        node n;
        n.id = id;

        for (int cpu : parse_list(cpulist)) {
            if (cpu_mask.empty() || std::find(cpu_mask.begin(), cpu_mask.end(), cpu) != cpu_mask.end()) {
                n.cpus.push_back(cpu);
            }
        }

        std::string meminfo;
        read_file(node_dir + "/meminfo", meminfo);
        n.mem_total     = parse_meminfo(meminfo, "MemTotal");
        // there is no per node MemAvailable; free plus reclaimable page cache is the usable estimate,
        // MemFree alone would report a node full of cached file pages as having no memory
        n.mem_available = parse_meminfo(meminfo, "MemFree") + parse_meminfo(meminfo, "Inactive(file)");

        // a node is only usable if it can both run threads and hold weights
        if (n.cpus.empty() || n.mem_total == 0) {
            continue;
        }

        n.cores   = get_physical_cores(sysfs_root, n.cpus);
        n.n_cores = (int) n.cores.size();

        n.socket_id = -1;
        if (!n.cpus.empty()) {
            std::string pkg_str;
            const std::string pkg_path = sysfs_root + "/devices/system/cpu/cpu" + std::to_string(n.cpus[0]) + "/topology/physical_package_id";
            if (read_file(pkg_path, pkg_str)) {
                try { n.socket_id = std::stoi(pkg_str); } catch (...) { n.socket_id = -1; }
            }
        }

        nodes.push_back(std::move(n));
    }

    // Fallback if physical_package_id is unavailable (e.g. inside virtualized environments):
    if (!nodes.empty() && nodes[0].socket_id < 0) {
        const size_t total = nodes.size();
        for (size_t i = 0; i < total; ++i) {
            if (total == 1) {
                nodes[i].socket_id = 0;
            } else if (total == 2) {
                nodes[i].socket_id = (int)i;
            } else {
                nodes[i].socket_id = (i >= total / 2) ? 1 : 0;
            }
        }
    }

    return nodes;
}

void refresh_memory(node & n) {
    std::string meminfo;
    if (!read_file("/sys/devices/system/node/node" + std::to_string(n.id) + "/meminfo", meminfo)) {
        return;
    }
    n.mem_total     = parse_meminfo(meminfo, "MemTotal");
    n.mem_available = parse_meminfo(meminfo, "MemFree") + parse_meminfo(meminfo, "Inactive(file)");
}

#if defined(__gnu_linux__)

static size_t page_size() {
    static const size_t size = (size_t) sysconf(_SC_PAGESIZE);
    return size;
}

static size_t detect_hugepage_size() {
    static size_t hpsz = 0;
    if (hpsz == 0) {
        std::string val;
        if (read_file("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size", val)) {
            try {
                hpsz = std::stoull(val);
            } catch (...) {
                hpsz = 0;
            }
        }
        if (hpsz == 0) {
            hpsz = 2u * 1024u * 1024u; // 2MB fallback
        }
    }
    return hpsz;
}

static size_t get_node_free_hugepages(int node_id, size_t page_size_kb) {
    std::string path = "/sys/devices/system/node/node" + std::to_string(node_id) +
                       "/hugepages/hugepages-" + std::to_string(page_size_kb) + "kB/free_hugepages";
    std::string content;
    if (read_file(path, content)) {
        try {
            return std::stoull(content);
        } catch (...) {}
    }
    return 0;
}

static size_t page_align(size_t size) {
    const size_t hpsz = detect_hugepage_size();
    if (size >= hpsz) {
        return (size + hpsz - 1) & ~(hpsz - 1);
    }
    const size_t ps = page_size();
    return (size + ps - 1) & ~(ps - 1);
}

int page_node(const void * addr) {
    void * pages[1] = { const_cast<void *>(addr) };
    int    status[1] = { -1 };

    if (syscall(__NR_move_pages, 0, 1, pages, nullptr, status, 0) != 0) {
        return -1;
    }

    return status[0];
}

// write to one byte of every page in [first, last) so that the pages are really allocated
static void touch_pages(char * addr, size_t first, size_t last, size_t stride) {
    for (size_t off = first; off < last; off += stride) {
        addr[off] = 0;
    }
}

// fault the mapping from a thread pinned to the node, then check where the pages ended up
static bool touch_and_verify(void * addr, size_t len, int node_id, bool bound, std::string & error) {
    const std::vector<node> & nodes = topology();

    const auto it = std::find_if(nodes.begin(), nodes.end(), [node_id](const struct node & n) { return n.id == node_id; });
    if (it == nodes.end()) {
        error = "node " + std::to_string(node_id) + " is not usable";
        return false;
    }

    cpu_set_t prev;
    CPU_ZERO(&prev);
    const bool have_prev = sched_getaffinity(0, sizeof(prev), &prev) == 0;

    cpu_set_t on_node;
    CPU_ZERO(&on_node);
    for (int cpu : it->cpus) {
        CPU_SET(cpu, &on_node);
    }
    const bool pinned = sched_setaffinity(0, sizeof(on_node), &on_node) == 0;
    if (!pinned && !bound) {
        error = "cannot pin a thread to node " + std::to_string(node_id) + " to place its memory: " + strerror(errno);
        return false;
    }

    // when mbind took, the policy places every page, so only a sample has to be faulted here.
    // without it first touch is the only placement mechanism, so every page has to be faulted now
    const size_t ps      = page_size();
    const size_t n_pages = len / ps;
    const size_t stride  = bound ? std::max<size_t>(1, n_pages / 64) * ps : ps;

    touch_pages((char *) addr, 0, len, stride);

    // check a sample spread over the whole mapping, independent of how much was faulted above -
    // sampling only the beginning would miss pages that landed elsewhere further in
    const size_t vstride = std::max<size_t>(1, n_pages / 64) * ps;

    std::vector<void *> pages;
    std::vector<int>    status;
    for (size_t off = 0; off < len && pages.size() < 64; off += vstride) {
        pages .push_back((char *) addr + off);
        status.push_back(-1);
    }

    bool ok = true;
    if (syscall(__NR_move_pages, 0, pages.size(), pages.data(), nullptr, status.data(), 0) == 0) {
        size_t n_remote = 0;
        int    seen     = -1;
        for (size_t i = 0; i < status.size(); i++) {
            if (status[i] >= 0 && status[i] != node_id) {
                n_remote++;
                seen = status[i];
            }
        }
        if (n_remote > 0) {
            error = "asked for node " + std::to_string(node_id) + " but " + std::to_string(n_remote) + " of " +
                    std::to_string(status.size()) + " sampled pages are on node " + std::to_string(seen);
            ok = false;
        }
    }

    if (have_prev) {
        sched_setaffinity(0, sizeof(prev), &prev);
    }

    return ok;
}

void * alloc_onnode(size_t size, int node_id, std::string & error) {
    if (size == 0) {
        error = "zero size allocation";
        return nullptr;
    }

    const size_t len = page_align(size);

    // MPOL_BIND reports an over-commit as an OOM when the pages are faulted, which would be a crash
    // in the middle of loading, so an allocation that cannot fit in the node at all is refused here.
    // merely exceeding the free + reclaimable estimate is not an error: the estimate is imprecise
    // and the kernel reclaims cache as the pages fault in (the caller warns about it)
    struct node n;
    n.id = node_id;
    refresh_memory(n);
    if (n.mem_total > 0 && len > n.mem_total) {
        error = "node " + std::to_string(node_id) + " has " + std::to_string(n.mem_total >> 20) +
                " MiB in total, need " + std::to_string(len >> 20) + " MiB";
        return nullptr;
    }

    void * addr = nullptr;
    const size_t hpsz = detect_hugepage_size();
    constexpr size_t GIGAPAGE_SIZE = 1024u * 1024u * 1024u;

#if defined(MAP_HUGETLB) && defined(MAP_HUGE_1GB)
    // Attempt 1 GB hugetlbfs allocation if available and requested
    if (len >= GIGAPAGE_SIZE && (len % GIGAPAGE_SIZE == 0) &&
        get_node_free_hugepages(node_id, 1048576) >= (len / GIGAPAGE_SIZE)) {
        void * h1g_addr = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0);
        if (h1g_addr != MAP_FAILED) {
            addr = h1g_addr;
        }
    }
#endif

#if defined(MAP_HUGETLB) && defined(MAP_HUGE_2MB)
    // Attempt 2 MB hugetlbfs allocation if available
    if (!addr && len >= hpsz && (len % hpsz == 0) &&
        get_node_free_hugepages(node_id, hpsz / 1024) >= (len / hpsz)) {
        void * h2m_addr = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (h2m_addr != MAP_FAILED) {
            addr = h2m_addr;
        }
    }
#endif

    // Fallback: 2MB-aligned anonymous mmap with transparent hugepage advice
    if (!addr) {
        if (len >= hpsz) {
            const size_t raw_len = len + hpsz;
            void * raw_addr = mmap(nullptr, raw_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (raw_addr == MAP_FAILED) {
                error = std::string("mmap failed: ") + strerror(errno);
                return nullptr;
            }
            uintptr_t uaddr = (uintptr_t) raw_addr;
            uintptr_t aligned_uaddr = (uaddr + hpsz - 1) & ~(hpsz - 1);
            size_t prefix = (size_t)(aligned_uaddr - uaddr);
            size_t suffix = hpsz - prefix;

            if (prefix > 0) {
                munmap(raw_addr, prefix);
            }
            if (suffix > 0) {
                munmap((void *)(aligned_uaddr + len), suffix);
            }
            addr = (void *) aligned_uaddr;
        } else {
            addr = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (addr == MAP_FAILED) {
                error = std::string("mmap failed: ") + strerror(errno);
                return nullptr;
            }
        }
    }

    // bind before the first fault, the policy is applied when the pages are allocated
    const size_t                bits_per_word = 8 * sizeof(unsigned long);
    std::vector<unsigned long>  nodemask(node_id / bits_per_word + 1, 0);
    nodemask[node_id / bits_per_word] = 1UL << (node_id % bits_per_word);

    bool bound = syscall(__NR_mbind, addr, len, MPOL_BIND, nodemask.data(), nodemask.size() * bits_per_word, 0) == 0;
    if (!bound) {
        const int err = errno;
        if (err != ENOSYS && err != EPERM) {
            error = std::string("mbind failed: ") + strerror(err);
            munmap(addr, len);
            return nullptr;
        }

        // without mbind the only way to place pages is to fault them from the node_id, which the kernel
        // only honours if the process did not inherit a different policy
        int mode = MPOL_DEFAULT;
        if (syscall(__NR_get_mempolicy, &mode, nullptr, 0UL, nullptr, 0UL) != 0 || mode != MPOL_DEFAULT) {
            error = std::string("mbind is unavailable (") + strerror(err) +
                    ") and the process runs under a non default memory policy, so pages cannot be placed";
            munmap(addr, len);
            return nullptr;
        }
    }

    // best effort, large pages help a lot at these sizes
    madvise(addr, len, MADV_HUGEPAGE);

    if (!touch_and_verify(addr, len, node_id, bound, error)) {
        munmap(addr, len);
        return nullptr;
    }

    return addr;
}

void free_onnode(void * ptr, size_t size) {
    if (ptr != nullptr) {
        munmap(ptr, page_align(size));
    }
}

bool bind_current_thread(const std::vector<int> & cpus) {
    if (cpus.empty()) {
        return false;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) {
            CPU_SET(cpu, &set);
        }
    }

    return sched_setaffinity(0, sizeof(set), &set) == 0;
}

bool bind_current_thread_to_node(int node_id, int thread_idx) {
    const auto & nodes = topology();
    for (const auto & n : nodes) {
        if (n.id == node_id) {
            if (n.cpus.empty()) return false;
            cpu_set_t set;
            CPU_ZERO(&set);
            if (thread_idx >= 0) {
                int target_cpu = -1;
                if (!n.cores.empty()) {
                    if (thread_idx < (int) n.cores.size()) {
                        target_cpu = n.cores[thread_idx];
                    } else {
                        // Gather SMT siblings not in n.cores
                        std::vector<int> smt_siblings;
                        for (int cpu : n.cpus) {
                            if (std::find(n.cores.begin(), n.cores.end(), cpu) == n.cores.end()) {
                                smt_siblings.push_back(cpu);
                            }
                        }
                        if (!smt_siblings.empty()) {
                            int total_pool = (int) (n.cores.size() + smt_siblings.size());
                            int rem = (thread_idx - (int) n.cores.size()) % total_pool;
                            if (rem < (int) smt_siblings.size()) {
                                target_cpu = smt_siblings[rem];
                            } else {
                                target_cpu = n.cores[rem - (int) smt_siblings.size()];
                            }
                        } else {
                            target_cpu = n.cores[thread_idx % n.cores.size()];
                        }
                    }
                } else {
                    target_cpu = n.cpus[thread_idx % n.cpus.size()];
                }
                if (target_cpu >= 0 && target_cpu < CPU_SETSIZE) {
                    CPU_SET(target_cpu, &set);
                }
            } else {
                for (int cpu : n.cpus) {
                    if (cpu >= 0 && cpu < CPU_SETSIZE) {
                        CPU_SET(cpu, &set);
                    }
                }
            }
            return sched_setaffinity(0, sizeof(set), &set) == 0;
        }
    }
    return false;
}

static std::vector<int> process_cpu_mask() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) {
        return {};
    }

    std::vector<int> cpus;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &set)) {
            cpus.push_back(i);
        }
    }

    return cpus;
}

const std::vector<node> & topology() {
    static const std::vector<node> nodes = parse_topology("/sys", process_cpu_mask());
    return nodes;
}
#else

const std::vector<node> & topology() {
    static const std::vector<node> nodes;
    return nodes;
}

int page_node(const void * addr) {
    return -1;

    (void) addr;
}

void * alloc_onnode(size_t size, int node, std::string & error) {
    error = "NUMA memory placement is only implemented on Linux";
    return nullptr;

    (void) size;
    (void) node;
}

void free_onnode(void * ptr, size_t size) {
    (void) ptr;
    (void) size;
}

bool bind_current_thread(const std::vector<int> & cpus) {
    return false;

    (void) cpus;
}

bool bind_current_thread_to_node(int node_id, int thread_idx) {
    (void) node_id;
    (void) thread_idx;
    return false;
}

#endif

int socket_count(const std::vector<node> & nodes) {
    std::vector<int> sockets;
    for (const auto & n : nodes) {
        if (n.socket_id >= 0 && std::find(sockets.begin(), sockets.end(), n.socket_id) == sockets.end()) {
            sockets.push_back(n.socket_id);
        }
    }
    return (int)sockets.size();
}

int socket_count() {
    return socket_count(topology());
}

int node_socket(const std::vector<node> & nodes, int node_id) {
    for (const auto & n : nodes) {
        if (n.id == node_id) return n.socket_id;
    }
    return -1;
}

int node_socket(int node_id) {
    return node_socket(topology(), node_id);
}

std::vector<int> socket_nodes(const std::vector<node> & nodes, int socket_id) {
    std::vector<int> res;
    for (const auto & n : nodes) {
        if (n.socket_id == socket_id) res.push_back(n.id);
    }
    return res;
}

std::vector<int> socket_nodes(int socket_id) {
    return socket_nodes(topology(), socket_id);
}

int get_optimal_draft_node(const std::vector<node> & nodes, int target_node) {
    if (nodes.size() <= 1) return -1;

    if (target_node < 0) {
#ifdef GGML_USE_CUDA
        target_node = ggml_backend_cuda_get_device_numa_node(0);
#endif
        if (target_node < 0) target_node = nodes[0].id;
    }

    int target_socket = node_socket(nodes, target_node);
    if (target_socket < 0) target_socket = (nodes.empty() ? 0 : nodes[0].socket_id);

    int n_sockets = socket_count(nodes);
    if (n_sockets >= 2) {
        int draft_socket = -1;
        for (const auto & n : nodes) {
            if (n.socket_id != target_socket && n.socket_id >= 0) {
                draft_socket = n.socket_id;
                break;
            }
        }
        if (draft_socket >= 0) {
            auto d_nodes = socket_nodes(nodes, draft_socket);
            if (!d_nodes.empty()) return d_nodes[0];
        }
    }

    // Single-socket multi-NUMA (e.g. Single EPYC NPS2/NPS4 or Intel SNC-2/SNC-4)
    int idx = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].id == target_node) { idx = (int)i; break; }
    }
    return nodes[(idx + (int)nodes.size() / 2) % (int)nodes.size()].id;
}

int get_optimal_draft_node(int target_node) {
    return get_optimal_draft_node(topology(), target_node);
}

static int s_storage_node = -1;

int get_storage_numa_node() {
    if (s_storage_node < 0) {
        s_storage_node = detect_path_numa_node(nullptr);
    }
    return s_storage_node;
}

void set_storage_numa_node(int node) {
    s_storage_node = node;
}

int detect_dev_numa_node(uint64_t dev, const std::string & sysfs_root) {
#if defined(__gnu_linux__)
    unsigned int maj = major(dev);
    unsigned int min = minor(dev);

    // Special handling for major == 0: Pseudo-filesystem / ZFS / Btrfs subvolumes
    if (maj == 0) {
        std::ifstream minfo("/proc/self/mountinfo");
        if (minfo.is_open()) {
            std::string line;
            std::string target_dev = "0:" + std::to_string(min);
            while (std::getline(minfo, line)) {
                std::istringstream iss(line);
                std::vector<std::string> tokens;
                std::string t;
                while (iss >> t) tokens.push_back(t);
                if (tokens.size() >= 5 && tokens[2] == target_dev) {
                    auto hyp = std::find(tokens.begin(), tokens.end(), "-");
                    if (hyp != tokens.end() && (hyp + 2) < tokens.end()) {
                        std::string fstype = *(hyp + 1);
                        std::string source = *(hyp + 2);
                        if (fstype == "zfs") {
                            std::string pool_name = source.substr(0, source.find('/'));
                            // Check /etc/zfs/zpool.cache for member devices of this specific pool
                            std::string zcache;
                            if (read_file("/etc/zfs/zpool.cache", zcache)) {
                                size_t pool_pos = zcache.find(pool_name);
                                if (pool_pos != std::string::npos) {
                                    // Search within this pool's section in zpool.cache
                                    size_t next_pool = zcache.find("pool_guid", pool_pos + pool_name.size() + 16);
                                    size_t limit = (next_pool != std::string::npos) ? (next_pool + 4096) : zcache.size();
                                    size_t pos = pool_pos;
                                    while ((pos = zcache.find("pci-", pos)) != std::string::npos && pos < limit) {
                                        pos += 4;
                                        if (pos + 12 <= zcache.size()) {
                                            std::string pci = zcache.substr(pos, 12);
                                            std::string numa_f = sysfs_root + "/bus/pci/devices/" + pci + "/numa_node";
                                            std::string val;
                                            if (fs::exists(numa_f) && read_file(numa_f, val)) {
                                                try { int n = std::stoi(val); if (n >= 0) return n; } catch (...) {}
                                            }
                                        }
                                    }
                                    // Also check for /dev/ disk identifiers in this pool's section
                                    pos = pool_pos;
                                    while ((pos = zcache.find("/dev/", pos)) != std::string::npos && pos < limit) {
                                        size_t end = zcache.find('\0', pos);
                                        if (end == std::string::npos) end = zcache.find(' ', pos);
                                        if (end != std::string::npos) {
                                            std::string dev_path = zcache.substr(pos, end - pos);
                                            struct stat dst;
                                            if (stat(dev_path.c_str(), &dst) == 0 && (S_ISBLK(dst.st_mode) || S_ISCHR(dst.st_mode))) {
                                                int n = detect_dev_numa_node(dst.st_rdev, sysfs_root);
                                                if (n >= 0) return n;
                                            }
                                            pos = end;
                                        } else {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    std::string block_dev_dir = sysfs_root + "/dev/block/" + std::to_string(maj) + ":" + std::to_string(min);
    std::error_code ec;
    fs::path p = fs::canonical(block_dev_dir, ec);
    if (!ec) {
        // Check for virtual/MD RAID devices with member slaves
        fs::path slaves_dir = p / "slaves";
        if (fs::exists(slaves_dir)) {
            for (const auto & slave : fs::directory_iterator(slaves_dir)) {
                fs::path slave_canon = fs::canonical(slave.path(), ec);
                if (!ec) {
                    for (fs::path cur = slave_canon; !cur.empty() && cur != cur.root_path(); cur = cur.parent_path()) {
                        fs::path numa_f = cur / "numa_node";
                        std::string val;
                        if (fs::exists(numa_f) && read_file(numa_f.string(), val)) {
                            try { int n = std::stoi(val); if (n >= 0) return n; } catch (...) {}
                        }
                    }
                }
            }
        }

        // Direct device inspection
        for (fs::path cur = p; !cur.empty() && cur != cur.root_path(); cur = cur.parent_path()) {
            fs::path numa_f = cur / "numa_node";
            std::string val;
            if (fs::exists(numa_f) && read_file(numa_f.string(), val)) {
                try { int n = std::stoi(val); if (n >= 0) return n; } catch (...) {}
            }
        }
    }
#else
    (void)dev; (void)sysfs_root;
#endif
    return -1;
}

int detect_path_numa_node(const char * path, const std::string & sysfs_root) {
    const char * env = getenv("GGML_NUMA_STORAGE_NODE");
    if (!env) env = getenv("LLAMA_NVME_NUMA_NODE");
    if (env) {
        try { int n = std::stoi(env); if (n >= 0) return n; } catch (...) {}
    }

    if (path && path[0] != '\0') {
#if defined(__gnu_linux__)
        struct stat st;
        if (stat(path, &st) == 0) {
            int n = detect_dev_numa_node(st.st_dev, sysfs_root);
            if (n >= 0) return n;
        }
#endif
    }

    // Fallback: scan /sys/block/nvme*
    // Prioritize high-bandwidth storage arrays (NUMA node with largest aggregate NVMe capacity) over boot drives
    try {
        fs::path bdir = fs::path(sysfs_root) / "block";
        if (fs::exists(bdir)) {
            std::map<int, uint64_t> node_capacity;
            std::map<int, int>      node_count;

            for (const auto & entry : fs::directory_iterator(bdir)) {
                std::string fname = entry.path().filename().string();
                if (fname.rfind("nvme", 0) == 0 && fname.find('n') != std::string::npos) {
                    fs::path numa_f = entry.path() / "device/numa_node";
                    fs::path size_f = entry.path() / "size";
                    std::string n_val;
                    std::string s_val;
                    int n_id = -1;
                    uint64_t cap = 0;
                    if (fs::exists(numa_f) && read_file(numa_f.string(), n_val)) {
                        try { n_id = std::stoi(n_val); } catch (...) {}
                    }
                    if (n_id < 0) {
                        // Traverse parent hierarchy to find parent PCIe numa_node
                        std::error_code ec_canon;
                        fs::path canon = fs::canonical(entry.path(), ec_canon);
                        if (!ec_canon) {
                            for (fs::path cur = canon; !cur.empty() && cur != cur.root_path(); cur = cur.parent_path()) {
                                fs::path parent_numa = cur / "numa_node";
                                if (fs::exists(parent_numa) && read_file(parent_numa.string(), n_val)) {
                                    try { int parsed = std::stoi(n_val); if (parsed >= 0) { n_id = parsed; break; } } catch (...) {}
                                }
                            }
                        }
                    }
                    if (n_id >= 0) {
                        if (fs::exists(size_f) && read_file(size_f.string(), s_val)) {
                            try { cap = std::stoull(s_val) * 512ULL; } catch (...) {}
                        }
                        node_capacity[n_id] += cap;
                        node_count[n_id]++;
                    }
                }
            }

            int best_node = -1;
            uint64_t max_cap = 0;
            int max_drives = 0;
            for (const auto & kv : node_capacity) {
                int nid = kv.first;
                uint64_t cap = kv.second;
                int drives = node_count[nid];
                if (cap > max_cap || (cap == max_cap && drives > max_drives)) {
                    max_cap = cap;
                    max_drives = drives;
                    best_node = nid;
                }
            }
            if (best_node >= 0) {
                return best_node;
            }
        }
    } catch (...) {}

    return 0; // Default to Node 0
}

bool bind_tensor_experts_round_robin(struct ggml_tensor * tensor, int n_numa_nodes) {
#if defined(__gnu_linux__)
    if (!tensor || !tensor->data || tensor->ne[2] <= 1 || n_numa_nodes <= 1) {
        return false;
    }

    const int n_experts = (int)tensor->ne[2];
    const size_t expert_stride = tensor->nb[2];
    const uintptr_t page_mask = ~(4096ULL - 1);
    const size_t bits_per_word = 8 * sizeof(unsigned long);

    for (int e = 0; e < n_experts; ++e) {
        int target_node = e % n_numa_nodes;
        uintptr_t start_addr = (uintptr_t)tensor->data + e * expert_stride;
        uintptr_t end_addr   = start_addr + expert_stride;

        uintptr_t page_start = start_addr & page_mask;
        uintptr_t page_end   = (end_addr + 4095ULL) & page_mask;
        size_t len           = page_end - page_start;
        if (len == 0) continue;

        std::vector<unsigned long> nodemask(target_node / bits_per_word + 1, 0);
        nodemask[target_node / bits_per_word] = 1UL << (target_node % bits_per_word);

        long ret = syscall(__NR_mbind, (void *)page_start, len, MPOL_BIND, nodemask.data(), nodemask.size() * bits_per_word, MPOL_MF_MOVE);
        if (ret != 0) {
            syscall(__NR_mbind, (void *)page_start, len, MPOL_PREFERRED, nodemask.data(), nodemask.size() * bits_per_word, 0);
        }
        madvise((void *)page_start, len, MADV_HUGEPAGE);
    }
    return true;
#else
    (void)tensor;
    (void)n_numa_nodes;
    return false;
#endif
}

static std::atomic<bool> g_numa_replicate_shared{false};
static std::mutex g_replica_mutex;
static std::unordered_map<const struct ggml_tensor *, std::vector<void *>> g_tensor_replicas;

bool is_numa_replicate_shared() {
    static int env_checked = -1;
    if (env_checked == -1) {
        const char * env = getenv("GGML_NUMA_REPLICATE_SHARED");
        if (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0 || strcmp(env, "yes") == 0)) {
            g_numa_replicate_shared.store(true, std::memory_order_relaxed);
        }
        env_checked = 1;
    }
    return g_numa_replicate_shared.load(std::memory_order_relaxed);
}

void set_numa_replicate_shared(bool enable) {
    g_numa_replicate_shared.store(enable, std::memory_order_relaxed);
}

bool bind_tensor_shared_interleaved(struct ggml_tensor * tensor, int n_numa_nodes) {
#if defined(__gnu_linux__)
    if (!tensor || !tensor->data || n_numa_nodes <= 1) {
        return false;
    }

    const uintptr_t page_mask = ~(4096ULL - 1);
    uintptr_t start_addr = (uintptr_t)tensor->data;
    size_t total_size = ggml_nbytes(tensor);
    uintptr_t end_addr = start_addr + total_size;

    uintptr_t page_start = start_addr & page_mask;
    uintptr_t page_end   = (end_addr + 4095ULL) & page_mask;
    size_t len           = page_end - page_start;
    if (len < 4096) return false;

    const size_t bits_per_word = 8 * sizeof(unsigned long);
    std::vector<unsigned long> nodemask((n_numa_nodes + bits_per_word - 1) / bits_per_word, 0);
    for (int i = 0; i < n_numa_nodes; ++i) {
        nodemask[i / bits_per_word] |= (1UL << (i % bits_per_word));
    }

    long ret = syscall(__NR_mbind, (void *)page_start, len, MPOL_INTERLEAVE, nodemask.data(), nodemask.size() * bits_per_word, MPOL_MF_MOVE);
    madvise((void *)page_start, len, MADV_HUGEPAGE);
    return ret == 0;
#else
    (void)tensor;
    (void)n_numa_nodes;
    return false;
#endif
}

bool replicate_tensor_numa(struct ggml_tensor * tensor, int n_numa_nodes) {
#if defined(__gnu_linux__)
    if (!tensor || !tensor->data || n_numa_nodes <= 1) {
        return false;
    }
    size_t size = ggml_nbytes(tensor);
    if (size < 4096) return false;

    std::lock_guard<std::mutex> lock(g_replica_mutex);
    auto it = g_tensor_replicas.find(tensor);
    if (it != g_tensor_replicas.end() && !it->second.empty()) {
        return true;
    }

    std::vector<void *> replicas(n_numa_nodes, nullptr);
    replicas[0] = tensor->data;

    const size_t bits_per_word = 8 * sizeof(unsigned long);
    for (int node = 1; node < n_numa_nodes; ++node) {
        void * ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr == MAP_FAILED) {
            continue;
        }

        std::vector<unsigned long> nodemask((node / bits_per_word) + 1, 0);
        nodemask[node / bits_per_word] = 1UL << (node % bits_per_word);
        syscall(__NR_mbind, ptr, size, MPOL_BIND, nodemask.data(), nodemask.size() * bits_per_word, MPOL_MF_MOVE | MPOL_MF_STRICT);
        madvise(ptr, size, MADV_HUGEPAGE);

        memcpy(ptr, tensor->data, size);
        replicas[node] = ptr;
    }

    g_tensor_replicas[tensor] = std::move(replicas);
    return true;
#else
    (void)tensor;
    (void)n_numa_nodes;
    return false;
#endif
}

void * get_tensor_numa_replica(const struct ggml_tensor * tensor, int node_id) {
    if (!tensor || node_id < 0) return tensor ? tensor->data : nullptr;
    std::lock_guard<std::mutex> lock(g_replica_mutex);
    auto it = g_tensor_replicas.find(tensor);
    if (it != g_tensor_replicas.end() && (size_t)node_id < it->second.size() && it->second[node_id] != nullptr) {
        return it->second[node_id];
    }
    return tensor->data;
}

} // namespace ggml::cpu::numa

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_cpu_numa_bind_current_thread(int node_id, int thread_idx) {
    return ggml::cpu::numa::bind_current_thread_to_node(node_id, thread_idx);
}

bool ggml_cpu_numa_bind_tensor_experts(struct ggml_tensor * tensor, int n_numa_nodes) {
    return ggml::cpu::numa::bind_tensor_experts_round_robin(tensor, n_numa_nodes);
}

bool ggml_cpu_numa_bind_tensor_shared(struct ggml_tensor * tensor, int n_numa_nodes) {
    return ggml::cpu::numa::bind_tensor_shared_interleaved(tensor, n_numa_nodes);
}

bool ggml_cpu_numa_replicate_tensor(struct ggml_tensor * tensor, int n_numa_nodes) {
    return ggml::cpu::numa::replicate_tensor_numa(tensor, n_numa_nodes);
}

void * ggml_cpu_numa_get_tensor_replica(const struct ggml_tensor * tensor, int node_id) {
    return ggml::cpu::numa::get_tensor_numa_replica(tensor, node_id);
}

#ifdef __cplusplus
}
#endif
