// tests for --numa split: the topology parser, node-local allocation, and NUMA backends
//
// the parser is pure, so it runs against generated sysfs trees and needs no NUMA hardware

#include "ggml-cpu-numa.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "spsc-ring.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>

#if defined(__gnu_linux__) || defined(__linux__)
#include <sys/sysmacros.h>
#endif

using namespace ggml::cpu::numa;

namespace fs = std::filesystem;

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("  %-58s %s\n", what.c_str(), ok ? "OK" : "FAILED");
    if (!ok) {
        n_fail++;
    }
}

static void write_file(const fs::path & path, const std::string & content) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path);
    f << content;
}

struct fake_node {
    int         id;
    std::string cpulist;
    size_t      mem_total_kb;
};

// build a sysfs tree, every CPU gets a sibling at cpu+n_cpus_per_core_offset to emulate SMT
static fs::path make_sysfs(const std::string & name, const std::string & online, const std::vector<fake_node> & nodes, int smt_offset) {
    const fs::path root = fs::temp_directory_path() / ("ggml-numa-test-" + name);
    fs::remove_all(root);

    write_file(root / "devices/system/node/online", online + "\n");

    for (const auto & n : nodes) {
        const fs::path dir = root / ("devices/system/node/node" + std::to_string(n.id));
        write_file(dir / "cpulist", n.cpulist + "\n");
        write_file(dir / "meminfo",
                "Node " + std::to_string(n.id) + " MemTotal:       " + std::to_string(n.mem_total_kb) + " kB\n"
                "Node " + std::to_string(n.id) + " MemFree:        " + std::to_string(n.mem_total_kb / 4) + " kB\n"
                "Node " + std::to_string(n.id) + " Active(file):   " + std::to_string(n.mem_total_kb / 8) + " kB\n"
                "Node " + std::to_string(n.id) + " Inactive(file): " + std::to_string(n.mem_total_kb / 4) + " kB\n");

        for (int cpu : parse_list(n.cpulist)) {
            const int sibling = cpu < smt_offset ? cpu + smt_offset : cpu - smt_offset;
            write_file(root / ("devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list"),
                    std::to_string(std::min(cpu, sibling)) + "," + std::to_string(std::max(cpu, sibling)) + "\n");
        }
    }

    return root;
}

int main() {
    printf("parse_list\n");
    {
        check(parse_list("0-3") == std::vector<int>({0, 1, 2, 3}), "range");
        check(parse_list("0,2,4") == std::vector<int>({0, 2, 4}), "comma separated");
        check(parse_list("0-1,4,6-7\n") == std::vector<int>({0, 1, 4, 6, 7}), "mixed with newline");
        check(parse_list("5") == std::vector<int>({5}), "single value");
        check(parse_list("").empty(), "empty");
        check(parse_list("garbage").empty(), "garbage");
    }

    printf("parse_topology\n");
    {
        // emulated topology: 2 nodes, interleaved CPU numbering, hyperthreads at +48
        const fs::path root = make_sysfs("2node", "0-1", {{0, "0-1,48-49", 8000}, {1, "2-3,50-51", 8000}}, 48);

        const auto nodes = parse_topology(root.string(), {});
        check(nodes.size() == 2, "two nodes found");
        if (nodes.size() == 2) {
            check(nodes[0].id == 0 && nodes[1].id == 1, "node ids");
            check(nodes[0].cpus == std::vector<int>({0, 1, 48, 49}), "node 0 cpus");
            check(nodes[1].cpus == std::vector<int>({2, 3, 50, 51}), "node 1 cpus");
            check(nodes[0].n_cores == 2 && nodes[1].n_cores == 2, "hyperthreads counted as one core");
            check(nodes[0].mem_total == 8000ull * 1024, "memory in bytes");
            // MemFree plus Inactive(file), the reclaimable page cache; Active(file) is not counted
            check(nodes[0].mem_available == (2000ull + 2000ull) * 1024, "available includes reclaimable cache");
        }
    }
    {
        const fs::path root  = make_sysfs("1node", "0", {{0, "0-3", 8000}}, 2);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 1, "single node");
    }
    {
        // node 1 offline, so the ids have a hole
        const fs::path root  = make_sysfs("hole", "0,2", {{0, "0-1", 8000}, {2, "4-5", 8000}}, 2);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 2, "hole in node numbering");
        if (nodes.size() == 2) {
            check(nodes[1].id == 2, "second node keeps its id");
        }
    }
    {
        // a memory-only node, e.g. CXL, cannot run threads
        const fs::path root  = make_sysfs("nocpu", "0-1", {{0, "0-1", 8000}, {1, "", 8000}}, 2);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 1 && nodes[0].id == 0, "node without cpus dropped");
    }
    {
        const fs::path root  = make_sysfs("nomem", "0-1", {{0, "0-1", 8000}, {1, "2-3", 0}}, 2);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 1 && nodes[0].id == 0, "node without memory dropped");
    }
    {
        const fs::path root = make_sysfs("affinity", "0-1", {{0, "0-1", 8000}, {1, "2-3", 8000}}, 2);

        const auto pinned = parse_topology(root.string(), {2, 3});
        check(pinned.size() == 1 && pinned[0].id == 1, "affinity mask leaves one node");

        const auto partial = parse_topology(root.string(), {0, 2});
        check(partial.size() == 2 && partial[0].cpus == std::vector<int>({0}), "affinity mask trims cpus");
    }
    {
        const auto nodes = parse_topology("/nonexistent-sysfs", {});
        check(nodes.empty(), "missing sysfs is not an error");
    }
    {
        std::vector<fake_node> many;
        for (int i = 0; i < 16; i++) {
            many.push_back({i, std::to_string(2 * i) + "-" + std::to_string(2 * i + 1), 8000});
        }
        const fs::path root  = make_sysfs("16node", "0-15", many, 1);
        const auto     nodes = parse_topology(root.string(), {});
        check(nodes.size() == 16, "sixteen nodes");
    }

    printf("node local allocation\n");
    if (topology().size() < 2) {
        printf("  skipped, this machine has fewer than 2 usable NUMA nodes\n");
    } else {
        const size_t size = 64ull << 20;

        for (const auto & n : topology()) {
            std::string error;

            void * data = alloc_onnode(size, n.id, error);
            check(data != nullptr, "allocated 64 MiB on node " + std::to_string(n.id) + (data ? "" : ": " + error));
            if (data == nullptr) {
                continue;
            }

            // the allocator only samples pages, so check the whole range here
            size_t n_remote = 0;
            for (size_t off = 0; off < size; off += 2u << 20) {
                ((char *) data)[off] = 1;
                if (page_node((char *) data + off) != n.id) {
                    n_remote++;
                }
            }
            check(n_remote == 0, "every page of node " + std::to_string(n.id) + " is local");

            free_onnode(data, size);
        }

        std::string error;
        check(alloc_onnode(size, 999, error) == nullptr && !error.empty(), "allocation on an unusable node fails");
    }

    printf("numa expert registration & backend verification\n");
    {
        check(ggml_numa_get_affinity() == GGML_NUMA_STRATEGY_DISABLED, "initial ggml_numa_get_affinity is disabled");
        const enum llama_numa_init_status status = llama_numa_init_ex(GGML_NUMA_STRATEGY_EXPERT);

        if (topology().size() < 2) {
            check(status == LLAMA_NUMA_INIT_STATUS_UNAVAILABLE, "unavailable with fewer than 2 nodes");
        } else {
            check(status == LLAMA_NUMA_INIT_STATUS_SUCCESS, "expert initialized");
            check(is_numa_expert(), "is_numa_expert reports true");
            check(ggml_numa_get_affinity() == GGML_NUMA_STRATEGY_EXPERT, "ggml_numa_get_affinity reports expert");
            check(ggml_numa_get_strategy() == GGML_NUMA_STRATEGY_EXPERT, "ggml_numa_get_strategy reports expert");

            for (const auto & n : topology()) {
                const std::string expected = "CPU" + std::to_string(n.id);

                ggml_backend_buffer_type_t buft = ggml_backend_cpu_numa_buffer_type(n.id);
                check(buft != nullptr, expected + " buffer type found");
                if (!buft) continue;

                check(std::string(ggml_backend_buft_name(buft)) == expected, expected + " has correct name");
                check(ggml_backend_buft_is_host(buft), expected + " buffer type is host memory");

                ggml_backend_t backend = ggml_backend_cpu_numa_init(n.id);
                check(backend != nullptr, expected + " backend initialized");
                if (!backend) continue;

                check(ggml_backend_is_cpu(backend), expected + " is CPU backend");
                check(ggml_backend_is_cpu_numa(backend), expected + " is CPU NUMA backend");
                check(ggml_backend_cpu_numa_get_node(backend) == n.id, expected + " reports node id");
                check(ggml_backend_supports_buft(backend, buft), expected + " supports its own buffer type");

                // Test buffer allocation on node
                const size_t buf_size = 2 * 1024 * 1024;
                ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, buf_size);
                check(buf != nullptr, expected + " allocated 2 MiB buffer");
                if (buf) {
                    void * base = ggml_backend_buffer_get_base(buf);
                    check(base != nullptr, expected + " buffer base ptr valid");
                    if (base) {
                        memset(base, 0x42, buf_size);
                        check(page_node(base) == n.id, expected + " buffer memory is verified local");
                    }
                    ggml_backend_buffer_free(buf);
                }

                // Verify cross-node rejection
                for (const auto & other : topology()) {
                    if (other.id == n.id) continue;
                    ggml_backend_buffer_type_t other_buft = ggml_backend_cpu_numa_buffer_type(other.id);
                    check(!ggml_backend_supports_buft(backend, other_buft),
                          expected + " rejects buffer type of CPU" + std::to_string(other.id));
                }

                ggml_backend_free(backend);
            }

            check(llama_numa_init_ex(GGML_NUMA_STRATEGY_EXPERT) == LLAMA_NUMA_INIT_STATUS_SUCCESS, "second call is idempotent");
        }
    }

    printf("hardware topology simulations (nps0, nps1, nps2, nps4, naples, intel snc)\n");
    {
        // 1. NPS0 / UMA: 1 node total (Node 0, Socket 0)
        std::vector<node> topo_nps0 = {
            { 0, 0, {0,1,2,3}, {0,1,2,3}, 4, 32ULL<<30, 30ULL<<30 }
        };
        check(socket_count(topo_nps0) == 1, "NPS0: socket_count == 1");
        check(node_socket(topo_nps0, 0) == 0, "NPS0: node 0 socket is 0");
        check(get_optimal_draft_node(topo_nps0, 0) == -1, "NPS0: draft node is -1 (no secondary NUMA domain)");

        // 2. NPS1 (Dual Socket): 2 nodes (Node 0 on Socket 0, Node 1 on Socket 1)
        std::vector<node> topo_nps1 = {
            { 0, 0, {0,1,2,3}, {0,1,2,3}, 4, 64ULL<<30, 60ULL<<30 },
            { 1, 1, {4,5,6,7}, {4,5,6,7}, 4, 64ULL<<30, 60ULL<<30 }
        };
        check(socket_count(topo_nps1) == 2, "NPS1: socket_count == 2");
        check(node_socket(topo_nps1, 0) == 0, "NPS1: node 0 socket is 0");
        check(node_socket(topo_nps1, 1) == 1, "NPS1: node 1 socket is 1");
        check(get_optimal_draft_node(topo_nps1, 0) == 1, "NPS1: target 0 selects draft node 1 on Socket 1");
        check(get_optimal_draft_node(topo_nps1, 1) == 0, "NPS1: target 1 selects draft node 0 on Socket 0");

        // 3. NPS2 (Dual Socket): 4 nodes total (Nodes 0,1 on Socket 0; Nodes 2,3 on Socket 1)
        std::vector<node> topo_nps2 = {
            { 0, 0, {0,1}, {0,1}, 2, 32ULL<<30, 30ULL<<30 },
            { 1, 0, {2,3}, {2,3}, 2, 32ULL<<30, 30ULL<<30 },
            { 2, 1, {4,5}, {4,5}, 2, 32ULL<<30, 30ULL<<30 },
            { 3, 1, {6,7}, {6,7}, 2, 32ULL<<30, 30ULL<<30 }
        };
        check(socket_count(topo_nps2) == 2, "NPS2: socket_count == 2");
        check(get_optimal_draft_node(topo_nps2, 0) == 2, "NPS2: target 0 selects draft node 2 on Socket 1");
        check(get_optimal_draft_node(topo_nps2, 1) == 2, "NPS2: target 1 selects draft node 2 on Socket 1");
        check(get_optimal_draft_node(topo_nps2, 2) == 0, "NPS2: target 2 selects draft node 0 on Socket 0");
        check(get_optimal_draft_node(topo_nps2, 3) == 0, "NPS2: target 3 selects draft node 0 on Socket 0");

        // 4. NPS4 (Dual Socket EPYC Milan - target machine configuration): 8 nodes total
        // Nodes 0-3 on Socket 0; Nodes 4-7 on Socket 1
        std::vector<node> topo_nps4;
        for (int i = 0; i < 8; ++i) {
            topo_nps4.push_back({ i, i < 4 ? 0 : 1, {i*2, i*2+1}, {i*2, i*2+1}, 2, 128ULL<<30, 120ULL<<30 });
        }
        check(socket_count(topo_nps4) == 2, "NPS4: socket_count == 2");
        check(get_optimal_draft_node(topo_nps4, 0) == 4, "NPS4: target 0 selects draft node 4 on Socket 1");
        check(get_optimal_draft_node(topo_nps4, 2) == 4, "NPS4: target 2 selects draft node 4 on Socket 1");
        check(get_optimal_draft_node(topo_nps4, 4) == 0, "NPS4: target 4 selects draft node 0 on Socket 0");
        check(get_optimal_draft_node(topo_nps4, 7) == 0, "NPS4: target 7 selects draft node 0 on Socket 0");

        // 5. AMD Naples / Zen 1 Single Socket: 4 NUMA nodes on Socket 0
        std::vector<node> topo_naples = {
            { 0, 0, {0,1}, {0,1}, 2, 16ULL<<30, 15ULL<<30 },
            { 1, 0, {2,3}, {2,3}, 2, 16ULL<<30, 15ULL<<30 },
            { 2, 0, {4,5}, {4,5}, 2, 16ULL<<30, 15ULL<<30 },
            { 3, 0, {6,7}, {6,7}, 2, 16ULL<<30, 15ULL<<30 }
        };
        check(socket_count(topo_naples) == 1, "Naples single socket: socket_count == 1");
        check(get_optimal_draft_node(topo_naples, 0) == 2, "Naples single socket: target 0 selects opposite quadrant 2");
        check(get_optimal_draft_node(topo_naples, 1) == 3, "Naples single socket: target 1 selects opposite quadrant 3");

        // 6. Intel Xeon Quad-Socket (4 sockets, 1 node each)
        std::vector<node> topo_intel_quad = {
            { 0, 0, {0,1,2,3}, {0,1,2,3}, 4, 32ULL<<30, 30ULL<<30 },
            { 1, 1, {4,5,6,7}, {4,5,6,7}, 4, 32ULL<<30, 30ULL<<30 },
            { 2, 2, {8,9,10,11}, {8,9,10,11}, 4, 32ULL<<30, 30ULL<<30 },
            { 3, 3, {12,13,14,15}, {12,13,14,15}, 4, 32ULL<<30, 30ULL<<30 }
        };
        check(socket_count(topo_intel_quad) == 4, "Intel Quad-Socket: socket_count == 4");
        check(get_optimal_draft_node(topo_intel_quad, 0) == 1, "Intel Quad-Socket: target 0 selects remote socket 1");
        check(get_optimal_draft_node(topo_intel_quad, 1) == 0, "Intel Quad-Socket: target 1 selects remote socket 0");

        // 7. AMD Genoa / Turin Single-Socket NPS4 (4 quadrants, 12 memory channels on Socket 0)
        std::vector<node> topo_genoa_1s = {
            { 0, 0, {0,1,2}, {0,1,2}, 3, 48ULL<<30, 45ULL<<30 },
            { 1, 0, {3,4,5}, {3,4,5}, 3, 48ULL<<30, 45ULL<<30 },
            { 2, 0, {6,7,8}, {6,7,8}, 3, 48ULL<<30, 45ULL<<30 },
            { 3, 0, {9,10,11}, {9,10,11}, 3, 48ULL<<30, 45ULL<<30 }
        };
        check(socket_count(topo_genoa_1s) == 1, "AMD Genoa 1S NPS4: socket_count == 1");
        check(get_optimal_draft_node(topo_genoa_1s, 0) == 2, "AMD Genoa 1S NPS4: target 0 selects opposite quadrant 2");

        // 8. Intel Sapphire Rapids / Emerald Rapids Dual-Socket SNC-4 (8 sub-NUMA clusters across 2 sockets)
        std::vector<node> topo_spr_snc4;
        for (int i = 0; i < 8; ++i) {
            topo_spr_snc4.push_back({ i, i < 4 ? 0 : 1, {i*4, i*4+1}, {i*4, i*4+1}, 2, 64ULL<<30, 60ULL<<30 });
        }
        check(socket_count(topo_spr_snc4) == 2, "Intel Sapphire Rapids SNC-4: socket_count == 2");
        check(get_optimal_draft_node(topo_spr_snc4, 0) == 4, "Intel Sapphire Rapids SNC-4: target 0 selects draft node 4 on Socket 1");
        check(get_optimal_draft_node(topo_spr_snc4, 5) == 0, "Intel Sapphire Rapids SNC-4: target 5 selects draft node 0 on Socket 0");
    }

    printf("storage numa node resolution\n");
    {
        const fs::path root = fs::temp_directory_path() / "ggml-numa-storage-test";
        fs::remove_all(root);

        // 1. Direct NVMe device: /sys/dev/block/259:0 -> /sys/devices/pci0000:c0/0000:c0:01.1/nvme/nvme0/nvme0n1
        fs::path pci_nvme_dir = root / "devices/pci0000:c0/0000:c0:01.1";
        fs::path nvme_dev_dir = pci_nvme_dir / "nvme/nvme0/nvme0n1";
        fs::create_directories(nvme_dev_dir);
        write_file(pci_nvme_dir / "numa_node", "2\n");

        fs::path dev_block_dir = root / "dev/block";
        fs::create_directories(dev_block_dir);
        fs::create_directory_symlink(nvme_dev_dir, dev_block_dir / "259:0");

#if defined(__gnu_linux__) || defined(__linux__)
        int direct_node = detect_dev_numa_node(makedev(259, 0), root.string());
        check(direct_node == 2, "direct NVMe resolves parent PCI numa_node 2");
#endif

        // 2. MD RAID0 device: /sys/dev/block/9:0 -> /sys/devices/virtual/block/md0
        // with member slave pointing to nvme0n1
        fs::path md0_dir = root / "devices/virtual/block/md0";
        fs::path slaves_dir = md0_dir / "slaves";
        fs::create_directories(slaves_dir);
        fs::create_directory_symlink(nvme_dev_dir, slaves_dir / "nvme0n1");
        fs::create_directory_symlink(md0_dir, dev_block_dir / "9:0");

#if defined(__gnu_linux__) || defined(__linux__)
        int raid_node = detect_dev_numa_node(makedev(9, 0), root.string());
        check(raid_node == 2, "MD RAID0 resolves member slave NVMe numa_node 2");

        // 3. Live host environment storage resolution check
        int host_storage_auto = detect_path_numa_node(nullptr);
        check(host_storage_auto >= 0, "host storage auto-discovery detects valid storage NUMA node");

        // Dynamic storage path resolution on current working directory
        int cwd_node = detect_path_numa_node(".");
        check(cwd_node >= 0, "storage path auto-discovery resolves current working directory to valid NUMA node");
#endif

        fs::remove_all(root);
    }

    printf("wait-free spsc ring buffer multi-threaded stress test\n");
    {
        common_spsc_ring_buffer<int, 1024> ring;
        check(ring.capacity() == 1024, "ring buffer capacity is 1024");
        check(ring.empty(), "ring buffer is initially empty");

        const int n_items = 50000;
        std::atomic<bool> producer_done{false};
        std::vector<int> consumed;
        consumed.reserve(n_items);

        std::thread consumer([&]() {
            int val = 0;
            while (consumed.size() < (size_t)n_items) {
                if (ring.pop(val)) {
                    consumed.push_back(val);
                } else if (producer_done.load(std::memory_order_acquire) && ring.empty()) {
                    break;
                } else {
                    std::this_thread::yield();
                }
            }
        });

        std::thread producer([&]() {
            for (int i = 0; i < n_items; ++i) {
                while (!ring.push(i)) {
                    std::this_thread::yield();
                }
            }
            producer_done.store(true, std::memory_order_release);
        });

        producer.join();
        consumer.join();

        check(consumed.size() == (size_t)n_items, "all 50,000 items consumed");
        bool strictly_ordered = true;
        for (int i = 0; i < n_items; ++i) {
            if (consumed[i] != i) {
                strictly_ordered = false;
                break;
            }
        }
        check(strictly_ordered, "items consumed in strictly sequential FIFO order with zero loss");

        // Batch operations and underflow/overflow check
        std::vector<int> batch_in = {10, 20, 30, 40, 50};
        size_t pushed = ring.push_batch(batch_in.data(), batch_in.size());
        check(pushed == 5, "batch push 5 items succeeded");
        check(ring.size() == 5, "ring size reports 5");

        std::vector<int> batch_out(5, 0);
        size_t popped = ring.pop_batch(batch_out.data(), batch_out.size());
        check(popped == 5, "batch pop 5 items succeeded");
        check(batch_in == batch_out, "batch pop data matches batch push data");
        check(ring.empty(), "ring buffer is empty after popping batch");

        // Underflow guard check
        int dummy = -1;
        check(!ring.pop(dummy), "pop on empty ring returns false");
        check(ring.pop_batch(&dummy, 1) == 0, "pop_batch on empty ring returns 0");

        ring.clear();
        check(ring.empty(), "ring buffer clear succeeds");
    }

    printf("thread pinning and SMT / physical core binding tests\n");
    {
        // 1. Verify GPU node resolution
        int gpu_node = llama_numa_get_gpu_node(0);
#ifdef GGML_USE_CUDA
        check(gpu_node >= 0, "GPU 0 resolves to valid NUMA node");
#else
        check(gpu_node == -1, "GPU 0 returns -1 without CUDA");
#endif

        // 2. Verify binding to physical core vs SMT sibling
        const auto & nodes = topology();
        if (!nodes.empty()) {
            const auto & n0 = nodes[0];
            // Test node-wide binding (thread_idx = -1)
            check(llama_numa_bind_current_thread(n0.id, -1), "bind_current_thread to node-wide succeeds");

            cpu_set_t set;
            CPU_ZERO(&set);
            if (sched_getaffinity(0, sizeof(set), &set) == 0) {
                bool has_all_cpus = true;
                for (int cpu : n0.cpus) {
                    if (!CPU_ISSET(cpu, &set)) {
                        has_all_cpus = false;
                        break;
                    }
                }
                check(has_all_cpus, "node-wide binding covers all logical CPUs on the node");
            }

            // Test physical core primary binding (thread_idx = 0)
            if (!n0.cores.empty()) {
                check(llama_numa_bind_current_thread(n0.id, 0), "bind_current_thread to primary core 0 succeeds");
                CPU_ZERO(&set);
                if (sched_getaffinity(0, sizeof(set), &set) == 0) {
                    check(CPU_ISSET(n0.cores[0], &set), "thread 0 is pinned to physical core 0");
                }
            }

            // Test SMT sibling binding if SMT is active (thread_idx = n_cores)
            if (n0.cpus.size() > (size_t) n0.n_cores && n0.n_cores > 0) {
                check(llama_numa_bind_current_thread(n0.id, n0.n_cores), "bind_current_thread to SMT sibling succeeds");
                CPU_ZERO(&set);
                if (sched_getaffinity(0, sizeof(set), &set) == 0) {
                    // Verify it is NOT pinned to physical core 0 primary, but to an SMT sibling
                    check(!CPU_ISSET(n0.cores[0], &set), "SMT thread is not mapped to physical core 0 primary");
                }
            }
        }
    }

    printf("ggml_numa_init GGML_NUMA_STRATEGY_EXPERT and contiguous compute affinity\n");
    {
        ggml_numa_init(GGML_NUMA_STRATEGY_EXPERT);
        check(ggml_numa_get_strategy() == GGML_NUMA_STRATEGY_EXPERT, "ggml_numa_get_strategy() == GGML_NUMA_STRATEGY_EXPERT");
        struct ggml_init_params params = { 16 * 1024 * 1024, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        check(ctx != NULL, "ggml_init for expert NUMA test succeeds");
        if (ctx) {
            struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
            struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
            memset(a->data, 0, ggml_nbytes(a));
            memset(b->data, 0, ggml_nbytes(b));
            struct ggml_tensor * c = ggml_mul_mat(ctx, a, b);
            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, c);
            struct ggml_cplan plan = ggml_graph_plan(gf, 8);
            std::vector<uint8_t> work(plan.work_size + 64);
            plan.work_data = work.data();
            enum ggml_status st = ggml_graph_compute(gf, &plan);
            check(st == GGML_STATUS_SUCCESS, "ggml_graph_compute under GGML_NUMA_STRATEGY_EXPERT succeeds");
            ggml_free(ctx);
        }
    }

    printf("topology of this machine\n");
    for (const auto & n : topology()) {
        printf("  node %d: %zu cpus, %d cores, %zu MiB total, %zu MiB free\n",
                n.id, n.cpus.size(), n.n_cores, n.mem_total >> 20, n.mem_available >> 20);
    }

    printf("%s\n", n_fail == 0 ? "test-numa: all tests OK" : "test-numa: FAILED");

    return n_fail == 0 ? 0 : 1;
}
