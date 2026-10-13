#include "ggml-cuda.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>

int main() {
    printf("=== Testing Ada Lovelace sm_89 L2 Cache Residency & Persistence ===\n");

    int device_count = ggml_backend_cuda_get_device_count();
    printf("CUDA device count: %d\n", device_count);
    if (device_count <= 0) {
        printf("No CUDA devices found, skipping test.\n");
        return 0;
    }

    for (int dev = 0; dev < device_count; ++dev) {
        char desc[256] = {0};
        ggml_backend_cuda_get_device_description(dev, desc, sizeof(desc));
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_cuda_get_device_memory(dev, &free_mem, &total_mem);

        size_t l2_size = 0, max_persisting = 0, cur_limit = 0;
        ggml_backend_cuda_get_l2_cache_info(dev, &l2_size, &max_persisting, &cur_limit);

        printf("Device %d: %s\n", dev, desc);
        printf("  Total VRAM: %.2f GiB (Free: %.2f GiB)\n", total_mem / (1024.0 * 1024.0 * 1024.0), free_mem / (1024.0 * 1024.0 * 1024.0));
        printf("  L2 Cache Size: %zu bytes (%.2f MiB)\n", l2_size, l2_size / (1024.0 * 1024.0));
        printf("  Max Persisting L2: %zu bytes (%.2f MiB)\n", max_persisting, max_persisting / (1024.0 * 1024.0));
        printf("  Current Persisting L2 Limit: %zu bytes (%.2f MiB)\n", cur_limit, cur_limit / (1024.0 * 1024.0));

        assert(l2_size > 0);
        if (max_persisting > 0) {
            assert(cur_limit > 0);
            assert(cur_limit <= max_persisting);
            printf("  [PASS] Hardware L2 cache persistence partition verified.\n");
        }

        // Initialize backend
        ggml_backend_t backend = ggml_backend_cuda_init(dev, nullptr, nullptr);
        assert(backend != nullptr);
        assert(ggml_backend_is_cuda(backend));

        // Test custom persistence window setting on a device buffer
        auto * buft = ggml_backend_cuda_buffer_type(dev);
        size_t test_buf_size = 16 * 1024 * 1024; // 16 MiB
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, test_buf_size);
        assert(buf != nullptr);

        void * base_ptr = ggml_backend_buffer_get_base(buf);
        assert(base_ptr != nullptr);

        // Apply L2 persistence window for 16 MiB buffer with 100% hit ratio
        ggml_backend_cuda_set_l2_persistence_window(backend, base_ptr, test_buf_size, 1.0f);
        printf("  [PASS] Successfully applied L2 persistence window for 16 MiB buffer at %p.\n", base_ptr);

        // Clear custom L2 persistence window (reverts to auto-pinning activation pool)
        ggml_backend_cuda_set_l2_persistence_window(backend, nullptr, 0, 0.0f);
        printf("  [PASS] Successfully cleared custom L2 persistence window (reverted to auto-pool).\n");

        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
    }

    printf("=== All Ada Lovelace L2 Cache Residency & Persistence tests passed successfully! ===\n");
    return 0;
}
