#pragma once

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// NUMA topology discovery for the CPU backend, used by --numa split.
// Linux only, everything reports an empty topology on other platforms.

struct ggml_tensor;

namespace ggml::cpu::numa {

// Bind individual expert slices within a 3D tensor to NUMA nodes round-robin via mbind(MPOL_BIND)
bool bind_tensor_experts_round_robin(struct ggml_tensor * tensor, int n_numa_nodes);

// Bind individual expert slices within a 3D tensor to physical CPU sockets (e % n_sockets)
// with MPOL_INTERLEAVE across all NUMA nodes belonging to that socket. This eliminates
// cross-socket xGMI traffic while pooling all memory channels of the home socket per expert.
bool bind_tensor_experts_socket_striped(struct ggml_tensor * tensor);
int  get_tensor_expert_home_socket(const struct ggml_tensor * tensor, int expert_idx);
int  get_tensor_expert_home_node(const struct ggml_tensor * tensor, int expert_idx);
bool has_expert_socket_binding(const struct ggml_tensor * tensor);
bool has_any_expert_binding();

// Install a process-wide default memory policy, set_mempolicy(MPOL_INTERLEAVE), over the
// given number of nodes. Every allocation made by this thread from now on -- and by every
// thread it creates afterwards -- is then spread page-round-robin across those nodes.
//
// This is the deterministic baseline placement for CPU-resident weights: without it, page
// placement is whatever the loader's first touch happened to produce, which on a dual
// socket box can collapse most of the model onto one socket and cost ~40% of decode
// throughput (measured: TG 10.5 -> 17.1 tok/s on 2xEPYC NPS4 with 470 GB of weights).
// Per-expert binding (above) is an optional upgrade attempted on top of this baseline.
// Returns false when NUMA is unusable, the syscall is refused, or fewer than 2 nodes.
bool set_default_interleave(int n_nodes);

// whether set_default_interleave() installed a process-wide policy
bool default_interleave_active();

// Collapse one host-resident tensor into 2 MiB pages (MADV_HUGEPAGE + MADV_COLLAPSE).
// Returns the number of bytes the kernel accepted (0 when nothing was done).
size_t collapse_tensor_hugepages(struct ggml_tensor * tensor);

struct node {
    int id = -1;
    int socket_id = -1;           // physical package/socket ID from /sys/devices/system/cpu/cpu*/topology/physical_package_id

    std::vector<int> cpus;        // usable logical CPUs, after the process affinity mask
    std::vector<int> cores;       // usable physical core primary CPUs (excluding SMT siblings)
    int              n_cores = 0; // usable physical cores

    size_t mem_total     = 0;
    size_t mem_available = 0; // free + reclaimable page cache; per node meminfo has no MemAvailable
};

// nodes that are online and have both usable CPUs and memory, empty if NUMA is not usable here
const std::vector<node> & topology();

// re-read the memory counters of a node, they change while the process runs
void refresh_memory(node & n);

// parse a sysfs tree directly, only the CPUs in cpu_mask are considered (empty means all).
// exposed for tests, use topology() otherwise
std::vector<node> parse_topology(const std::string & sysfs_root, const std::vector<int> & cpu_mask);

// parse a sysfs "0,2-4" style list
std::vector<int> parse_list(const std::string & list);

// Allocate memory that is guaranteed to be on the given node, or fail.
// The pages are bound with mbind, faulted by a thread pinned to the node and then checked, so that
// --numa split can never end up running on memory that silently landed somewhere else.
// Returns nullptr and fills error on failure, the caller is expected to log it and give up.
void * alloc_onnode(size_t size, int node, std::string & error);
void   free_onnode (void * ptr, size_t size);

// node the page at addr is on, -1 if that cannot be determined
int page_node(const void * addr);

// pin the calling thread to the given CPUs, best effort
bool bind_current_thread(const std::vector<int> & cpus);
bool bind_current_thread_to_node(int node_id, int thread_idx);

// whether numa expert mode is enabled
bool is_numa_expert();
void set_numa_expert(bool enable);

// Dynamic topology querying: physical socket count and node-to-socket mapping
int socket_count();
int socket_count(const std::vector<node> & nodes);
int node_socket(int node_id);
int node_socket(const std::vector<node> & nodes, int node_id);
std::vector<int> socket_nodes(int socket_id);
std::vector<int> socket_nodes(const std::vector<node> & nodes, int socket_id);
int get_optimal_draft_node(int target_node = -1);
int get_optimal_draft_node(const std::vector<node> & nodes, int target_node = -1);

// NVMe / Storage NUMA node detection
int  get_storage_numa_node();
void set_storage_numa_node(int node);
int  detect_dev_numa_node(uint64_t dev, const std::string & sysfs_root = "/sys");
int  detect_path_numa_node(const char * path, const std::string & sysfs_root = "/sys");

// NUMA Shared Tensor Interleaving and Replication
bool is_numa_replicate_shared();
void set_numa_replicate_shared(bool enable);
bool bind_tensor_shared_interleaved(struct ggml_tensor * tensor, int n_numa_nodes);

// --engram-resident: interleave the tensor pages across NUMA nodes, advise
// THP, then prefault the whole range with parallel MADV_POPULATE_READ so the
// table is fully resident before the first serving request (Linux only).
// Returns the bytes successfully touched (0 on refusal/failure).
size_t prefault_tensor_interleaved(struct ggml_tensor * tensor, int n_numa_nodes, int n_threads);
bool replicate_tensor_numa(struct ggml_tensor * tensor, int n_numa_nodes);
// Replicate a tensor once per physical CPU socket (1 copy interleaved across each socket's
// NUMA nodes), populating g_tensor_replicas[tensor][node] with that node's socket copy.
// On a 2-socket NPS4 machine (8 nodes), this uses 2 copies (1x extra RAM) instead of 8
// copies (7x extra RAM) while eliminating 100% of cross-socket xGMI traffic and pooling
// all 4 memory controllers (8 DDR4 channels) within each socket.
bool replicate_tensor_socket(struct ggml_tensor * tensor);
void * get_tensor_numa_replica(const struct ggml_tensor * tensor, int node_id);
void free_tensor_numa_replicas(const struct ggml_tensor * tensor);
void free_all_numa_replicas();

// True when at least one tensor owns per-node replicas. Hot-path safe: one
// relaxed atomic load, no lock, no syscall.
bool has_tensor_replicas();

// Number of times a matmul actually addressed a node-local replica (test/debug).
size_t replica_hit_count();
void reset_replica_hit_count();

// NUMA node / physical socket the calling thread is currently running on (-1 when unavailable).
int current_thread_numa_node();
int current_thread_socket();

// Translate a tensor-relative pointer to the calling thread's node-local replica.
// Returns `ptr` unchanged when the tensor has no replicas, when the thread cannot
// report a node, or when the original is used for that node.
const void * numa_local_ptr(const struct ggml_tensor * tensor, const void * ptr);

} // namespace ggml::cpu::numa
#else
#include <stdbool.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

// C-compatible wrapper for binding the current thread during graph compute
bool ggml_cpu_numa_bind_current_thread(int node_id, int thread_idx);

// C-compatible wrapper for binding expert tensor slices round-robin to NUMA nodes
bool ggml_cpu_numa_bind_tensor_experts(struct ggml_tensor * tensor, int n_numa_nodes);
bool ggml_cpu_numa_bind_tensor_experts_socket(struct ggml_tensor * tensor);
int  ggml_cpu_numa_expert_home_socket(const struct ggml_tensor * tensor, int expert_idx);
int  ggml_cpu_numa_expert_home_node(const struct ggml_tensor * tensor, int expert_idx);
bool ggml_cpu_numa_has_expert_socket_binding(const struct ggml_tensor * tensor);
bool ggml_cpu_numa_has_any_expert_binding(void);
int  ggml_cpu_numa_socket_count(void);
int  ggml_cpu_numa_node_count(void);
int  ggml_cpu_numa_current_socket(void);
int  ggml_cpu_numa_thread_socket(int thread_idx, int n_threads);
int  ggml_cpu_numa_thread_node(int thread_idx, int n_threads);

// C-compatible wrapper for installing the process-wide MPOL_INTERLEAVE default policy
// (deterministic even placement of host tensors across NUMA nodes)
bool ggml_cpu_numa_set_default_interleave(int n_numa_nodes);

// C-compatible wrapper for collapsing a host tensor into 2 MiB pages (MADV_COLLAPSE)
size_t ggml_cpu_numa_collapse_tensor_hugepages(struct ggml_tensor * tensor);
bool ggml_cpu_numa_default_interleave_active(void);

// C-compatible wrapper for interleaving shared tensor pages across NUMA nodes
bool ggml_cpu_numa_bind_tensor_shared(struct ggml_tensor * tensor, int n_numa_nodes);

// C-compatible wrapper for --engram-resident: interleave + THP + parallel
// MADV_POPULATE_READ prefault; returns bytes touched (see namespace wrapper)
size_t ggml_cpu_numa_prefault_tensor_interleaved(struct ggml_tensor * tensor, int n_numa_nodes, int n_threads);

// C-compatible wrapper for replicating shared tensors across NUMA nodes or sockets
bool ggml_cpu_numa_replicate_tensor(struct ggml_tensor * tensor, int n_numa_nodes);
bool ggml_cpu_numa_replicate_tensor_socket(struct ggml_tensor * tensor);
void * ggml_cpu_numa_get_tensor_replica(const struct ggml_tensor * tensor, int node_id);
void ggml_cpu_numa_free_tensor_replicas(const struct ggml_tensor * tensor);
void ggml_cpu_numa_free_all_replicas(void);

// Replica-aware base pointer for the CPU matmul paths: returns the calling
// thread's node-local copy of `ptr` (tensor-relative), or `ptr` when the tensor
// has no replicas. Cheap enough to call once per matmul op per thread.
const void * ggml_cpu_numa_local_ptr(const struct ggml_tensor * tensor, const void * ptr);
bool   ggml_cpu_numa_has_replicas(void);
size_t ggml_cpu_numa_replica_hits(void);
void   ggml_cpu_numa_reset_replica_hits(void);
int    ggml_cpu_numa_current_node(void);

#ifdef __cplusplus
}
#endif
