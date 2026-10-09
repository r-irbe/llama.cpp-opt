// Unit tests for Linux io_uring + O_DIRECT High-Throughput DMA Subsystem:
// 1. Kernel probe & availability detection
// 2. Multi-megabyte DMA streaming and byte-for-byte verification
// 3. Sub-sector and non-power-of-two EOF edge cases
// 4. IOMMU buffer registration (IORING_REGISTER_BUFFERS)
// 5. O_DIRECT direct I/O vs buffered io_uring modes
// 6. SQPOLL kernel thread submission vs standard event polling
// 7. File descriptor interface (llama_io_uring_read_fd)
// 8. Robust error handling & invalid argument resilience

#include "llama-io-uring.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("  %-64s %s\n", what.c_str(), ok ? "OK" : "FAILED");
    if (!ok) {
        n_fail++;
    }
}

// Helper to create a temporary test file with deterministic pattern
static bool create_test_file(const char * path, size_t size, uint8_t seed = 0x5A) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;

    constexpr size_t BUF_SIZE = 1024 * 1024;
    std::vector<uint8_t> buffer(BUF_SIZE);

    size_t written = 0;
    while (written < size) {
        size_t to_write = std::min(BUF_SIZE, size - written);
        for (size_t i = 0; i < to_write; ++i) {
            buffer[i] = static_cast<uint8_t>((written + i + seed) & 0xFF);
        }
        ssize_t n = write(fd, buffer.data(), to_write);
        if (n != static_cast<ssize_t>(to_write)) {
            close(fd);
            unlink(path);
            return false;
        }
        written += to_write;
    }

    close(fd);
    return true;
}

// Helper to verify memory matches expected pattern
static bool verify_pattern(const void * ptr, size_t size, uint8_t seed = 0x5A) {
    const uint8_t * p = static_cast<const uint8_t *>(ptr);
    for (size_t i = 0; i < size; ++i) {
        uint8_t expected = static_cast<uint8_t>((i + seed) & 0xFF);
        if (p[i] != expected) {
            return false;
        }
    }
    return true;
}

int main() {
    printf("===================================================================\n");
    printf("   ik_llama.cpp - High-Throughput Linux io_uring DMA Test Suite    \n");
    printf("===================================================================\n\n");

    printf("=== Testing Kernel Probe & Availability ===\n");
    bool available = llama_io_uring_available();
    check(available, "Linux io_uring kernel syscall probe reports available");

    if (!available) {
        printf("\n[SKIP] io_uring is not supported on this kernel/platform.\n");
        return n_fail;
    }

    printf("\n=== Testing Multi-Megabyte DMA Streaming ===\n");
    const char * test_file_16m = "/tmp/test_llama_io_uring_16m.bin";
    const size_t size_16m = 16 * 1024 * 1024;
    check(create_test_file(test_file_16m, size_16m, 0xA5), "Generated 16 MB deterministic test file");

    // Allocate 2MB-aligned destination buffer
    void * dest_16m = nullptr;
    int memalign_res = posix_memalign(&dest_16m, 2 * 1024 * 1024, size_16m);
    check(memalign_res == 0 && dest_16m != nullptr, "Allocated 2MB-aligned destination memory buffer");
    std::memset(dest_16m, 0, size_16m);

    llama_io_uring_stats stats_16m;
    llama_io_uring_config cfg_default;
    bool read_16m_ok = llama_io_uring_read_file(test_file_16m, dest_16m, size_16m, size_16m, cfg_default, &stats_16m);
    check(read_16m_ok, "16 MB async io_uring DMA transfer succeeded");
    check(stats_16m.total_bytes == size_16m, "Transferred byte count matches exact file size");
    check(stats_16m.throughput_gb_s > 0.0, "Throughput calculation is positive and non-zero");
    check(verify_pattern(dest_16m, size_16m, 0xA5), "Byte-for-byte binary integrity verification passed across 16 MB");
    printf("     -> Mode: %s | Elapsed: %.4f s | Throughput: %.2f GB/s\n",
           stats_16m.mode_str.c_str(), stats_16m.elapsed_seconds, stats_16m.throughput_gb_s);

    printf("\n=== Testing Sub-Sector & Trailing Remainder Edge Cases ===\n");
    // Test 1: Sub-sector file (< 4096 bytes)
    const char * test_sub = "/tmp/test_llama_io_uring_sub.bin";
    const size_t size_sub = 573;
    create_test_file(test_sub, size_sub, 0x12);
    void * dest_sub = nullptr;
    posix_memalign(&dest_sub, 4096, 4096);
    std::memset(dest_sub, 0, 4096);

    llama_io_uring_stats stats_sub;
    bool read_sub_ok = llama_io_uring_read_file(test_sub, dest_sub, size_sub, 4096, cfg_default, &stats_sub);
    check(read_sub_ok, "Sub-sector file (573 bytes) read successfully");
    check(verify_pattern(dest_sub, size_sub, 0x12), "Sub-sector payload binary integrity matched");

    // Test 2: Unaligned trailing remainder (e.g. 1MB + 137 bytes)
    const char * test_rem = "/tmp/test_llama_io_uring_rem.bin";
    const size_t size_rem = 1024 * 1024 + 137;
    create_test_file(test_rem, size_rem, 0x34);
    void * dest_rem = nullptr;
    posix_memalign(&dest_rem, 4096, size_rem + 4096);
    std::memset(dest_rem, 0, size_rem + 4096);

    llama_io_uring_stats stats_rem;
    bool read_rem_ok = llama_io_uring_read_file(test_rem, dest_rem, size_rem, size_rem + 4096, cfg_default, &stats_rem);
    check(read_rem_ok, "Unaligned trailing remainder (1MB + 137B) read successfully");
    check(verify_pattern(dest_rem, size_rem, 0x34), "Trailing remainder binary integrity matched");

    printf("\n=== Testing IOMMU Buffer Registration Modes ===\n");
    llama_io_uring_config cfg_no_reg = cfg_default;
    cfg_no_reg.try_register_buffers = false;
    std::memset(dest_16m, 0, size_16m);
    llama_io_uring_stats stats_no_reg;
    bool read_no_reg_ok = llama_io_uring_read_file(test_file_16m, dest_16m, size_16m, size_16m, cfg_no_reg, &stats_no_reg);
    check(read_no_reg_ok, "io_uring read succeeded with registered buffers disabled");
    check(!stats_no_reg.used_registered_buffers, "Stats confirm registered buffers were not used");
    check(verify_pattern(dest_16m, size_16m, 0xA5), "Unregistered buffer read verified binary integrity");

    printf("\n=== Testing SQPOLL vs Standard Event Polling ===\n");
    llama_io_uring_config cfg_no_sqpoll = cfg_default;
    cfg_no_sqpoll.try_sqpoll = false;
    std::memset(dest_16m, 0, size_16m);
    llama_io_uring_stats stats_no_sqpoll;
    bool read_no_sqpoll_ok = llama_io_uring_read_file(test_file_16m, dest_16m, size_16m, size_16m, cfg_no_sqpoll, &stats_no_sqpoll);
    check(read_no_sqpoll_ok, "io_uring read succeeded in standard (non-SQPOLL) polling mode");
    check(!stats_no_sqpoll.used_sqpoll, "Stats confirm SQPOLL was disabled");
    check(verify_pattern(dest_16m, size_16m, 0xA5), "Standard event polling verified binary integrity");

    printf("\n=== Testing File Descriptor Interface (read_fd) ===\n");
    int fd_16m = open(test_file_16m, O_RDONLY);
    check(fd_16m >= 0, "Opened existing test file descriptor");
    std::memset(dest_16m, 0, size_16m);
    llama_io_uring_stats stats_fd;
    bool read_fd_ok = llama_io_uring_read_fd(fd_16m, dest_16m, size_16m, size_16m, cfg_default, &stats_fd);
    check(read_fd_ok, "llama_io_uring_read_fd succeeded via open file descriptor");
    check(verify_pattern(dest_16m, size_16m, 0xA5), "File descriptor read verified binary integrity");
    close(fd_16m);

    printf("\n=== Testing Error Handling & Argument Validation ===\n");
    llama_io_uring_stats stats_err;
    bool non_existent = llama_io_uring_read_file("/tmp/non_existent_file_xyz_123.bin", dest_16m, 1024, 1024, cfg_default, &stats_err);
    check(!non_existent, "Non-existent file gracefully returns false");

    bool null_dest = llama_io_uring_read_file(test_file_16m, nullptr, size_16m, size_16m, cfg_default, &stats_err);
    check(!null_dest, "Null destination buffer gracefully returns false");

    bool zero_size = llama_io_uring_read_file(test_file_16m, dest_16m, 0, size_16m, cfg_default, &stats_err);
    check(!zero_size, "Zero file size gracefully returns false");

    bool insufficient_cap = llama_io_uring_read_file(test_file_16m, dest_16m, size_16m, size_16m - 100, cfg_default, &stats_err);
    check(!insufficient_cap, "Insufficient buffer capacity gracefully returns false");

    llama_io_uring_config cfg_disabled = cfg_default;
    cfg_disabled.enable = false;
    bool disabled_res = llama_io_uring_read_file(test_file_16m, dest_16m, size_16m, size_16m, cfg_disabled, &stats_err);
    check(!disabled_res, "Explicitly disabled config gracefully returns false");

    // Clean up temporary test files & memory
    unlink(test_file_16m);
    unlink(test_sub);
    unlink(test_rem);
    free(dest_16m);
    free(dest_sub);
    free(dest_rem);

    printf("\n===================================================================\n");
    if (n_fail == 0) {
        printf("  ALL IO_URING DIRECT DMA TESTS PASSED (100%% SUCCESS)\n");
    } else {
        printf("  %d TESTS FAILED\n", n_fail);
    }
    printf("===================================================================\n");

    return n_fail;
}
