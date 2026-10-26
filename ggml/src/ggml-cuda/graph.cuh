#pragma once

#include "ggml.h"

struct ggml_graph_node_properties {
    void * node_address;
    ggml_op   node_op;
    ggml_type type;
    void * view_src;
    size_t view_offs;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    void * src_address[GGML_MAX_SRC];
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
};

// PolyGraph Plan-Buffer (28-byte fixed metadata struct)
// Allows dynamic speculative draft verification (n in {0, 1, 2, 4, 8}) with fixed launch grids
// and skips inactive draft tiles with zero CUDA graph invalidation or re-instantiation.
struct alignas(4) ggml_cuda_polygraph_plan {
    uint32_t active_draft_len;  // n in [0, n_max]
    uint32_t active_tiles;      // Number of active execution tiles
    uint32_t tile_offset;       // Base tile index in verification batch
    uint32_t flags;             // Bitflags (e.g. 0x1: target-only baseline)
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;         // 7 x 4 bytes = 28 bytes
};
static_assert(sizeof(ggml_cuda_polygraph_plan) == 28, "PolyGraph plan buffer must be exactly 28 bytes");

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
    ~ggml_cuda_graph() {
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
        }
        if (graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    uint64_t uid = 0;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    size_t num_nodes = 0;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> params;
    bool disable_due_to_gpu_arch = false;
    bool disable_due_to_too_many_updates = false;
    bool disable_due_to_failed_graph_capture = false;
    int number_consecutive_updates = 0;
    std::vector<ggml_graph_node_properties> ggml_graph_properties;
    bool use_cpy_indirection = false;
    std::vector<char *> cpy_dest_ptrs;
    char ** dest_ptrs_d;
    int dest_ptrs_size = 0;
    // Index to allow each cpy kernel to be aware of it's position within the graph
    // relative to other cpy nodes.
    int graph_cpynode_index = -1;

    // PolyGraph plan-buffer support
    bool has_polygraph_plan = false;
    ggml_cuda_polygraph_plan polygraph_plan = {};
    void * polygraph_plan_d = nullptr;
#endif
};

