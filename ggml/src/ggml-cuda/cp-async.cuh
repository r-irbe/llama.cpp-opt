// Simplified API for asynchronous data loading.

#include "common.cuh"

static __device__ __forceinline__ unsigned int ggml_cuda_cvta_generic_to_shared(void * generic_ptr) {
#ifdef CP_ASYNC_AVAILABLE
    return __cvta_generic_to_shared(generic_ptr);
#else
    GGML_UNUSED(generic_ptr);
    NO_DEVICE_CODE;
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// Copies data from global to shared memory, cg == cache global.
// Both the src and dst pointers must be aligned to 16 bit.
// Shared memory uses 32 bit addressing, the pointer is passed as unsigned int.
// Generic pointers can be converted to 32 bit shared memory pointers using __cvta_generic_to_shared.
// Only the 16 bit copy is exposed because 4 and 8 bit copies did not yield performance improvements.
template <int preload>
static __device__ __forceinline__ void cp_async_cg_16(const unsigned int dst, const void * src) {
    static_assert(preload == 0 || preload == 64 || preload == 128 || preload == 256, "bad preload");
#ifdef CP_ASYNC_AVAILABLE
#if CUDART_VERSION >= 11040
    if (preload == 256) {
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 128) {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 64) {
        asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else
#endif // CUDART_VERSION >= 11040
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    }
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Copies data from global to shared memory, ca == cache all.
template <int preload>
static __device__ __forceinline__ void cp_async_ca_16(const unsigned int dst, const void * src) {
    static_assert(preload == 0 || preload == 64 || preload == 128 || preload == 256, "bad preload");
#ifdef CP_ASYNC_AVAILABLE
#if CUDART_VERSION >= 11040
    if (preload == 256) {
        asm volatile("cp.async.ca.shared.global.L2::256B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 128) {
        asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 64) {
        asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else
#endif // CUDART_VERSION >= 11040
    {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    }
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

template <int preload = 128>
static __device__ __forceinline__ void cp_async_cg_16(void * dst_shared, const void * src_global) {
    cp_async_cg_16<preload>(ggml_cuda_cvta_generic_to_shared(dst_shared), src_global);
}

template <int preload = 128>
static __device__ __forceinline__ void cp_async_ca_16(void * dst_shared, const void * src_global) {
    cp_async_ca_16<preload>(ggml_cuda_cvta_generic_to_shared(dst_shared), src_global);
}

static __device__ __forceinline__ void cp_async_ca_4(const unsigned int dst, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(dst), "l"(src));
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

static __device__ __forceinline__ void cp_async_ca_4(void * dst_shared, const void * src_global) {
    cp_async_ca_4(ggml_cuda_cvta_generic_to_shared(dst_shared), src_global);
}

static __device__ __forceinline__ void cp_async_ca_8(const unsigned int dst, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" : : "r"(dst), "l"(src));
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

static __device__ __forceinline__ void cp_async_ca_8(void * dst_shared, const void * src_global) {
    cp_async_ca_8(ggml_cuda_cvta_generic_to_shared(dst_shared), src_global);
}

// PTX ISA specification restricts .cg (cache-global) copies to 16 bytes.
// 4-byte and 8-byte copies use .ca (cache-all) under the hood.
static __device__ __forceinline__ void cp_async_cg_4(const unsigned int dst, const void * src) {
    cp_async_ca_4(dst, src);
}

static __device__ __forceinline__ void cp_async_cg_4(void * dst_shared, const void * src_global) {
    cp_async_ca_4(dst_shared, src_global);
}

static __device__ __forceinline__ void cp_async_cg_8(const unsigned int dst, const void * src) {
    cp_async_ca_8(dst, src);
}

static __device__ __forceinline__ void cp_async_cg_8(void * dst_shared, const void * src_global) {
    cp_async_ca_8(dst_shared, src_global);
}

// Commits all prior uncommitted cp.async instructions into a group.
static __device__ __forceinline__ void cp_async_commit_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Waits until N or fewer cp.async groups are pending.
template <int N>
static __device__ __forceinline__ void cp_async_wait_group() {
    static_assert(N >= 0 && N <= 8, "wait group count must be between 0 and 8");
#ifdef CP_ASYNC_AVAILABLE
    if constexpr (N == 0) {
        asm volatile("cp.async.wait_group 0;");
    } else if constexpr (N == 1) {
        asm volatile("cp.async.wait_group 1;");
    } else if constexpr (N == 2) {
        asm volatile("cp.async.wait_group 2;");
    } else if constexpr (N == 3) {
        asm volatile("cp.async.wait_group 3;");
    } else if constexpr (N == 4) {
        asm volatile("cp.async.wait_group 4;");
    } else if constexpr (N == 5) {
        asm volatile("cp.async.wait_group 5;");
    } else if constexpr (N == 6) {
        asm volatile("cp.async.wait_group 6;");
    } else if constexpr (N == 7) {
        asm volatile("cp.async.wait_group 7;");
    } else if constexpr (N == 8) {
        asm volatile("cp.async.wait_group 8;");
    }
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Makes each thread wait until its asynchronous data copies are done.
// This does NOT provide any additional synchronization.
// In particular, when copying data with multiple warps a call to __syncthreads will be needed.
static __device__ __forceinline__ void cp_async_wait_all() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_all;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}
