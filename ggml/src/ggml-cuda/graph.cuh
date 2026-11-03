#pragma once

#include <cstdint>
#include <cstring>
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

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
    ~ggml_cuda_graph() {
        release();
    }
    // The owner synchronizes and selects the owning device before releasing.
    void release() {
        if (instance != nullptr) {
            CUDA_CHECK(cudaGraphExecDestroy(instance));
            instance = nullptr;
        }
        if (graph != nullptr) {
            CUDA_CHECK(cudaGraphDestroy(graph));
            graph = nullptr;
        }
        if (dest_ptrs_d != nullptr) {
            CUDA_CHECK(cudaFree(dest_ptrs_d));
            dest_ptrs_d = nullptr;
        }
        if (polygraph_plan_d != nullptr) {
            CUDA_CHECK(cudaFree(polygraph_plan_d));
            polygraph_plan_d = nullptr;
        }
        std::memset(polygraph_plan_h, 0, sizeof(polygraph_plan_h));
        dest_ptrs_size = 0;
        use_cpy_indirection = false;
    }
    ggml_cuda_graph() = default;
    ggml_cuda_graph(const ggml_cuda_graph &) = delete;
    ggml_cuda_graph & operator=(const ggml_cuda_graph &) = delete;
    uint64_t last_used = 0;
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
    char ** dest_ptrs_d = nullptr;
    int dest_ptrs_size = 0;
    // Index to allow each cpy kernel to be aware of it's position within the graph
    // relative to other cpy nodes.
    int graph_cpynode_index = -1;
    // 28-byte device-resident PolyGraph plan buffer and host shadow words
    uint32_t * polygraph_plan_d = nullptr;
    uint32_t polygraph_plan_h[7] = {0};
#endif
};

