#include "spec-split-executor.h"

#include <algorithm>
#include <cstdlib>

bool spec_split_executor::stream_h0_to_host(const float * gpu_h0_src, int32_t b_size, spec_split_state & state) {
    if (!gpu_h0_src || b_size <= 0 || hidden_dim <= 0) {
        return false;
    }
    state.batch_size = b_size;
    state.hidden_dim = hidden_dim;
    size_t total_floats = (size_t)b_size * (size_t)hidden_dim;
    state.h0_host_buffer.resize(total_floats);

    // Reference stage-1 behavior: host memory copy from an already-hosted source.
    // The real pipeline replaces this with a PCIe DMA into pinned staging memory.
    std::copy(gpu_h0_src, gpu_h0_src + total_floats, state.h0_host_buffer.data());
    return true;
}

bool spec_split_executor::execute_cpu_backbone(spec_split_state & state, int n_draft_tokens) {
    if (state.h0_host_buffer.empty() || n_draft_tokens <= 0) {
        return false;
    }
    state.predicted_tokens.clear();
    state.predicted_tokens.reserve(n_draft_tokens);

    // Reference backbone: deterministic scalar projection over the staged H_0 rows.
    // The real pipeline replaces this with SIMD/VNNI kernels on CCX worker threads.
    for (int i = 0; i < n_draft_tokens; ++i) {
        size_t offset = (size_t)(i % state.batch_size) * (size_t)state.hidden_dim;
        float val = state.h0_host_buffer[offset];
        llama_token tok = 100 + (std::abs((int)(val * 1000.0f)) % 5000);
        state.predicted_tokens.push_back(tok);
    }
    return true;
}

const std::vector<llama_token> & spec_split_executor::get_predicted_tokens(const spec_split_state & state) const {
    return state.predicted_tokens;
}
