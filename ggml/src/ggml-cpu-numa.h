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
bool replicate_tensor_numa(struct ggml_tensor * tensor, int n_numa_nodes);
void * get_tensor_numa_replica(const struct ggml_tensor * tensor, int node_id);
void free_tensor_numa_replicas(const struct ggml_tensor * tensor);
void free_all_numa_replicas();

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

// C-compatible wrapper for interleaving shared tensor pages across NUMA nodes
bool ggml_cpu_numa_bind_tensor_shared(struct ggml_tensor * tensor, int n_numa_nodes);

// C-compatible wrapper for replicating shared tensors across NUMA nodes
bool ggml_cpu_numa_replicate_tensor(struct ggml_tensor * tensor, int n_numa_nodes);
void * ggml_cpu_numa_get_tensor_replica(const struct ggml_tensor * tensor, int node_id);
void ggml_cpu_numa_free_tensor_replicas(const struct ggml_tensor * tensor);
void ggml_cpu_numa_free_all_replicas(void);

#ifdef __cplusplus
}
#endif
