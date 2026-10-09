#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

/**
 * High-Throughput Linux io_uring + O_DIRECT Zero-Copy DMA Subsystem for ik_llama.cpp
 *
 * Implements high-speed model weight streaming directly from NVMe controllers into
 * pre-allocated HugeTLB / anonymous memory buffers:
 * - O_DIRECT: Bypasses Linux page cache double-buffering and memory reclamation stalls.
 * - IORING_SETUP_SQPOLL: Eliminates per-I/O system call overhead via kernel polling thread.
 * - IORING_REGISTER_BUFFERS: Pre-pins HugeTLB pages in IOMMU, eliminating page table walks.
 * - Pretoucher Pattern: MADV_POPULATE_WRITE eliminates soft page-fault jitter from hot path.
 * - Automatic Fallback: Gracefully falls back to standard POSIX I/O if io_uring is unsupported.
 */

struct llama_io_uring_stats {
    size_t total_bytes = 0;
    double elapsed_seconds = 0.0;
    double throughput_gb_s = 0.0;
    bool used_direct_io = false;
    bool used_sqpoll = false;
    bool used_registered_buffers = false;
    std::string mode_str;
};

struct llama_io_uring_config {
    bool enable = true;                   // Master toggle
    bool try_direct_io = true;            // Attempt O_DIRECT direct DMA
    bool try_sqpoll = true;               // Attempt kernel SQPOLL thread
    bool try_register_buffers = true;     // Attempt IOMMU buffer pre-registration
    size_t chunk_size = 4 * 1024 * 1024;  // 4 MB chunks (optimal for NVMe queues)
    unsigned queue_depth = 64;            // SQ ring submission queue depth
    bool populate_pages = true;           // Proactive page fault elimination via madvise
};

// Returns true if io_uring syscalls are supported by the running Linux kernel
bool llama_io_uring_available();

// Reads an entire file into dest_addr using asynchronous io_uring DMA streaming
// dest_capacity specifies the allocated capacity of dest_addr (must be >= file_size)
bool llama_io_uring_read_file(
    const char * path,
    void * dest_addr,
    size_t file_size,
    size_t dest_capacity,
    const llama_io_uring_config & config = {},
    llama_io_uring_stats * stats_out = nullptr
);

// Reads an open file descriptor into dest_addr using io_uring DMA streaming
bool llama_io_uring_read_fd(
    int fd,
    void * dest_addr,
    size_t file_size,
    size_t dest_capacity,
    const llama_io_uring_config & config = {},
    llama_io_uring_stats * stats_out = nullptr
);
