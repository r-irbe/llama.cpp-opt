#include "llama-io-uring.h"

#if defined(__linux__)

#include <linux/io_uring.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <cerrno>
#include <cstdio>
#include <algorithm>
#include <stdexcept>

#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 426
#endif
#ifndef __NR_io_uring_register
#define __NR_io_uring_register 427
#endif
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif

bool llama_io_uring_available() {
    static int cached_status = -1;
    if (cached_status != -1) {
        return cached_status == 1;
    }

    struct io_uring_params p;
    std::memset(&p, 0, sizeof(p));
    int ring_fd = (int)syscall(__NR_io_uring_setup, 1, &p);
    if (ring_fd >= 0) {
        close(ring_fd);
        cached_status = 1;
        return true;
    }

    cached_status = 0;
    return false;
}

namespace {

struct io_uring_session {
    int ring_fd = -1;
    struct io_uring_params params;
    void * sq_ptr = MAP_FAILED;
    void * cq_ptr = MAP_FAILED;
    struct io_uring_sqe * sqes = (struct io_uring_sqe *)MAP_FAILED;
    size_t sring_sz = 0;
    size_t cring_sz = 0;
    size_t sqes_sz = 0;

    __u32 * sq_tail = nullptr;
    __u32 * sq_mask = nullptr;
    __u32 * sq_array = nullptr;
    __u32 * sq_flags = nullptr;

    __u32 * cq_head = nullptr;
    __u32 * cq_tail = nullptr;
    __u32 * cq_mask = nullptr;
    struct io_uring_cqe * cqes = nullptr;

    bool is_sqpoll = false;
    bool buffers_registered = false;

    ~io_uring_session() {
        cleanup();
    }

    void cleanup() {
        if (ring_fd < 0) return;

        if (buffers_registered) {
            syscall(__NR_io_uring_register, ring_fd, IORING_UNREGISTER_BUFFERS, nullptr, 0);
            buffers_registered = false;
        }

        if (sqes != MAP_FAILED && sqes != nullptr) {
            munmap(sqes, sqes_sz);
            sqes = (struct io_uring_sqe *)MAP_FAILED;
        }

        if (params.features & IORING_FEAT_SINGLE_MMAP) {
            if (sq_ptr != MAP_FAILED && sq_ptr != nullptr) {
                munmap(sq_ptr, sring_sz);
                sq_ptr = MAP_FAILED;
                cq_ptr = MAP_FAILED;
            }
        } else {
            if (sq_ptr != MAP_FAILED && sq_ptr != nullptr) {
                munmap(sq_ptr, sring_sz);
                sq_ptr = MAP_FAILED;
            }
            if (cq_ptr != MAP_FAILED && cq_ptr != nullptr) {
                munmap(cq_ptr, cring_sz);
                cq_ptr = MAP_FAILED;
            }
        }

        close(ring_fd);
        ring_fd = -1;
    }

    bool init(unsigned depth, bool try_sqpoll, void * dest_addr, size_t dest_capacity, bool try_reg_buffers) {
        std::memset(&params, 0, sizeof(params));

        if (try_sqpoll) {
            params.flags = IORING_SETUP_SQPOLL;
            params.sq_thread_idle = 2000; // 2 seconds polling idle timeout
            ring_fd = (int)syscall(__NR_io_uring_setup, depth, &params);
            if (ring_fd >= 0) {
                is_sqpoll = true;
            }
        }

        if (ring_fd < 0) {
            std::memset(&params, 0, sizeof(params));
            ring_fd = (int)syscall(__NR_io_uring_setup, depth, &params);
            is_sqpoll = false;
        }

        if (ring_fd < 0) {
            return false;
        }

        sring_sz = params.sq_off.array + params.sq_entries * sizeof(__u32);
        cring_sz = params.cq_off.cqes + params.cq_entries * sizeof(struct io_uring_cqe);
        if (params.features & IORING_FEAT_SINGLE_MMAP) {
            if (cring_sz > sring_sz) sring_sz = cring_sz;
            cring_sz = sring_sz;
        }

        sq_ptr = mmap(0, sring_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQ_RING);
        if (sq_ptr == MAP_FAILED) {
            cleanup();
            return false;
        }

        cq_ptr = (params.features & IORING_FEAT_SINGLE_MMAP) ? sq_ptr :
                 mmap(0, cring_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_CQ_RING);
        if (cq_ptr == MAP_FAILED) {
            cleanup();
            return false;
        }

        sqes_sz = params.sq_entries * sizeof(struct io_uring_sqe);
        sqes = (struct io_uring_sqe *)mmap(0, sqes_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQES);
        if (sqes == MAP_FAILED) {
            cleanup();
            return false;
        }

        sq_tail  = (__u32 *)((char *)sq_ptr + params.sq_off.tail);
        sq_mask  = (__u32 *)((char *)sq_ptr + params.sq_off.ring_mask);
        sq_array = (__u32 *)((char *)sq_ptr + params.sq_off.array);
        sq_flags = (__u32 *)((char *)sq_ptr + params.sq_off.flags);

        cq_head = (__u32 *)((char *)cq_ptr + params.cq_off.head);
        cq_tail = (__u32 *)((char *)cq_ptr + params.cq_off.tail);
        cq_mask = (__u32 *)((char *)cq_ptr + params.cq_off.ring_mask);
        cqes    = (struct io_uring_cqe *)((char *)cq_ptr + params.cq_off.cqes);

        if (try_reg_buffers && dest_addr != nullptr && dest_capacity > 0) {
            struct iovec iov;
            iov.iov_base = dest_addr;
            iov.iov_len = dest_capacity;
            int ret = (int)syscall(__NR_io_uring_register, ring_fd, IORING_REGISTER_BUFFERS, &iov, 1);
            if (ret == 0) {
                buffers_registered = true;
            }
        }

        return true;
    }
};

} // namespace

bool llama_io_uring_read_fd(
    int fd,
    void * dest_addr,
    size_t file_size,
    size_t dest_capacity,
    const llama_io_uring_config & config,
    llama_io_uring_stats * stats_out
) {
    if (!config.enable || file_size == 0 || dest_addr == nullptr || dest_capacity < file_size) {
        return false;
    }

    // Check direct I/O alignment constraints
    int fcntl_flags = fcntl(fd, F_GETFL);
    bool is_direct = (fcntl_flags != -1) && (fcntl_flags & O_DIRECT);
    uintptr_t addr_val = reinterpret_cast<uintptr_t>(dest_addr);

    // O_DIRECT requires 512-byte (or 4096-byte) alignment
    if (is_direct && (addr_val % 512 != 0)) {
        is_direct = false;
    }

    // Proactive memory pre-touching to eliminate page table soft faults
    if (config.populate_pages) {
        madvise(dest_addr, dest_capacity, MADV_POPULATE_WRITE);
    }

    unsigned depth = std::max(8u, std::min(config.queue_depth, 512u));
    io_uring_session session;
    if (!session.init(depth, config.try_sqpoll, dest_addr, dest_capacity, config.try_register_buffers)) {
        return false;
    }

    size_t chunk_size = std::max<size_t>(4096, config.chunk_size);
    if (is_direct) {
        chunk_size = (chunk_size / 4096) * 4096;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    size_t submitted_offset = 0;
    size_t completed_bytes = 0;
    size_t inflight = 0;
    bool had_error = false;

    while (completed_bytes < file_size && !had_error) {
        // Enqueue SQEs up to ring capacity
        unsigned to_submit = 0;
        while (submitted_offset < file_size && inflight < session.params.sq_entries) {
            size_t off = submitted_offset;
            size_t remaining = file_size - off;
            size_t len = std::min(chunk_size, remaining);

            size_t read_len = len;
            if (is_direct) {
                // If trailing block is not 4KB aligned, pad to 4KB if buffer allows
                if (read_len % 4096 != 0) {
                    size_t padded_len = ((read_len + 4095) / 4096) * 4096;
                    if (off + padded_len <= dest_capacity) {
                        read_len = padded_len;
                    }
                }
            }

            unsigned tail = *session.sq_tail;
            unsigned index = tail & *session.sq_mask;
            struct io_uring_sqe * sqe = &session.sqes[index];
            std::memset(sqe, 0, sizeof(*sqe));

            if (session.buffers_registered) {
                sqe->opcode = IORING_OP_READ_FIXED;
                sqe->buf_index = 0;
                sqe->addr = (uintptr_t)((char *)dest_addr + off);
            } else {
                sqe->opcode = IORING_OP_READ;
                sqe->addr = (uintptr_t)((char *)dest_addr + off);
            }
            sqe->fd = fd;
            sqe->off = off;
            sqe->len = (uint32_t)read_len;
            sqe->user_data = (uint64_t)off;

            session.sq_array[index] = index;
            __atomic_store_n(session.sq_tail, tail + 1, __ATOMIC_RELEASE);

            submitted_offset += len;
            inflight++;
            to_submit++;
        }

        // Notify SQPOLL thread or submit events
        if (to_submit > 0) {
            if (session.is_sqpoll) {
                unsigned fl = __atomic_load_n(session.sq_flags, __ATOMIC_ACQUIRE);
                if (fl & IORING_SQ_NEED_WAKEUP) {
                    syscall(__NR_io_uring_enter, session.ring_fd, to_submit, 0, IORING_ENTER_SQ_WAKEUP, nullptr);
                }
            } else {
                syscall(__NR_io_uring_enter, session.ring_fd, to_submit, 0, 0, nullptr);
            }
        }

        // Harvest CQE completions
        unsigned head = *session.cq_head;
        unsigned tail = __atomic_load_n(session.cq_tail, __ATOMIC_ACQUIRE);

        if (head == tail && inflight > 0) {
            // Block waiting for at least 1 event
            syscall(__NR_io_uring_enter, session.ring_fd, 0, 1, IORING_ENTER_GETEVENTS, nullptr);
            tail = __atomic_load_n(session.cq_tail, __ATOMIC_ACQUIRE);
        }

        while (head != tail) {
            struct io_uring_cqe * cqe = &session.cqes[head & *session.cq_mask];
            if (cqe->res < 0) {
                had_error = true;
                break;
            }

            completed_bytes += cqe->res;
            inflight--;
            head++;
        }
        __atomic_store_n(session.cq_head, head, __ATOMIC_RELEASE);
    }

    if (had_error || completed_bytes < file_size) {
        return false;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    double elapsed_s = diff.count();
    double gb = (double)file_size / (1024.0 * 1024.0 * 1024.0);
    double throughput = elapsed_s > 0.0 ? (gb / elapsed_s) : 0.0;

    if (stats_out) {
        stats_out->total_bytes = file_size;
        stats_out->elapsed_seconds = elapsed_s;
        stats_out->throughput_gb_s = throughput;
        stats_out->used_direct_io = is_direct;
        stats_out->used_sqpoll = session.is_sqpoll;
        stats_out->used_registered_buffers = session.buffers_registered;

        std::string mode = "[";
        if (is_direct) mode += "O_DIRECT";
        else mode += "buffered";
        if (session.is_sqpoll) mode += " + SQPOLL";
        if (session.buffers_registered) mode += " + registered HugeTLB";
        mode += "]";
        stats_out->mode_str = mode;
    }

    return true;
}

bool llama_io_uring_read_file(
    const char * path,
    void * dest_addr,
    size_t file_size,
    size_t dest_capacity,
    const llama_io_uring_config & config,
    llama_io_uring_stats * stats_out
) {
    if (!path || !config.enable || file_size == 0 || dest_addr == nullptr) {
        return false;
    }

    int fd = -1;
    bool is_direct = false;

    // Try O_DIRECT first if requested
    if (config.try_direct_io && (reinterpret_cast<uintptr_t>(dest_addr) % 512 == 0)) {
        fd = open(path, O_RDONLY | O_DIRECT | O_NOATIME);
        if (fd < 0 && errno == EPERM) {
            // O_NOATIME may require file ownership; retry without it
            fd = open(path, O_RDONLY | O_DIRECT);
        }
        if (fd >= 0) {
            is_direct = true;
        }
    }

    // Fall back to standard open if direct I/O failed or was not requested
    if (fd < 0) {
        fd = open(path, O_RDONLY | O_NOATIME);
        if (fd < 0 && errno == EPERM) {
            fd = open(path, O_RDONLY);
        }
        if (fd < 0) {
            return false;
        }
        is_direct = false;
    }

    llama_io_uring_config active_cfg = config;
    active_cfg.try_direct_io = is_direct;

    bool ok = llama_io_uring_read_fd(fd, dest_addr, file_size, dest_capacity, active_cfg, stats_out);
    close(fd);
    return ok;
}

#else

// Non-Linux Stubs
bool llama_io_uring_available() {
    return false;
}

bool llama_io_uring_read_fd(
    int, void *, size_t, size_t,
    const llama_io_uring_config &,
    llama_io_uring_stats *
) {
    return false;
}

bool llama_io_uring_read_file(
    const char *, void *, size_t, size_t,
    const llama_io_uring_config &,
    llama_io_uring_stats *
) {
    return false;
}

#endif
