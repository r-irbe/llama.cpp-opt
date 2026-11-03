#pragma once

// Disposable, per-process checkpoint blobs. Not a durable session format.
// No source ownership is relinquished until verified atomic publication.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace common_spill {
struct metadata {
    int64_t pos_min = 0;
    int64_t pos_max = 0;
    int64_t n_tokens = 0;
    uint64_t bytes = 0;
    uint64_t checksum = 0;
};

inline uint64_t checksum(const uint8_t * data, size_t n, uint64_t h = UINT64_C(14695981039346656037)) {
    // Accidental-corruption detection, not authentication or encryption.
    for (size_t i = 0; i < n; ++i) {
        h = (h ^ data[i]) * UINT64_C(1099511628211);
    }
    return h;
}

inline std::array<uint8_t, 48> header(const metadata & m) {
    const uint64_t words[] = {UINT64_C(0x31504c4c49505343), uint64_t(m.pos_min),
        uint64_t(m.pos_max), uint64_t(m.n_tokens), m.bytes, m.checksum};
    std::array<uint8_t, 48> result {};
    for (size_t i = 0; i < 6; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            result[8 * i + j] = uint8_t(words[i] >> (8 * j));
        }
    }
    return result;
}

#ifndef _WIN32
inline std::string system_error(const char * op) {
    const int saved_errno = errno;
    return std::string(op) + ": " + std::strerror(saved_errno);
}

inline std::string create_process_directory(const std::string & root, std::string & error) {
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    struct stat st {};
    if (ec || ::lstat(root.c_str(), &st) != 0) {
        error = ec ? ec.message() : system_error("stat spill root");
        return {};
    }
    if (!S_ISDIR(st.st_mode) || st.st_uid != ::geteuid() || (st.st_mode & (S_IWGRP | S_IWOTH))) {
        error = "spill root must be an owned directory, not a symlink or group/world-writable";
        return {};
    }
    const std::string pattern = (std::filesystem::path(root) / "run-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    if (::mkdtemp(name.data()) == nullptr) {
        error = system_error("create private spill directory");
        return {};
    }
    return name.data(); // mkdtemp creates mode 0700, independently of umask.
}

using write_fn = ssize_t (*)(int, const void *, size_t);

inline bool write_exact(int fd, const uint8_t * p, size_t n, std::string & error, write_fn writer) {
    while (n > 0) {
        const size_t chunk = std::min<size_t>(n, 1024 * 1024);
        const ssize_t done = writer(fd, p, chunk);
        if (done < 0 && errno == EINTR) {
            continue;
        }
        if (done <= 0 || size_t(done) > chunk) {
            error = done < 0 ? system_error("write checkpoint") : "short/invalid checkpoint write";
            return false;
        }
        p += done;
        n -= size_t(done);
    }
    return true;
}

inline bool read_exact(int fd, uint8_t * p, size_t n, std::string & error) {
    while (n > 0) {
        const size_t chunk = std::min<size_t>(n, 1024 * 1024);
        const ssize_t done = ::read(fd, p, chunk);
        if (done < 0 && errno == EINTR) {
            continue;
        }
        if (done <= 0) {
            error = done < 0 ? system_error("read checkpoint") : "truncated checkpoint";
            return false;
        }
        p += done;
        n -= size_t(done);
    }
    return true;
}

// Validate against trusted in-memory metadata before allocation. During store,
// use a fixed-size buffer rather than a second full checkpoint copy.
inline bool read_file(const std::string & path, const metadata & expected,
        std::vector<uint8_t> * output, std::string & error) {
    if (expected.bytes == 0 || expected.bytes > uint64_t(std::numeric_limits<size_t>::max()) - 48 ||
            expected.bytes > uint64_t(std::numeric_limits<off_t>::max()) - 48) {
        error = "invalid expected checkpoint length";
        return false;
    }
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) {
        error = system_error("open checkpoint");
        return false;
    }
    struct closer { int fd; ~closer() { ::close(fd); } } close_fd {fd};
    struct stat st {};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_uid != ::geteuid() ||
            (st.st_mode & (S_IRWXG | S_IRWXO)) || st.st_size < 0 || uint64_t(st.st_size) != expected.bytes + 48) {
        error = "checkpoint type, ownership, permissions or length mismatch";
        return false;
    }
    std::array<uint8_t, 48> stored_header {};
    if (!read_exact(fd, stored_header.data(), stored_header.size(), error) || stored_header != header(expected)) {
        error = "checkpoint identity/header mismatch";
        return false;
    }
    std::vector<uint8_t> restored;
    if (output != nullptr) {
        try { restored.resize(size_t(expected.bytes)); }
        catch (const std::bad_alloc &) { error = "checkpoint restore allocation failed"; return false; }
        catch (const std::length_error &) { error = "checkpoint restore length exceeds vector limit"; return false; }
    }
    std::array<uint8_t, 64 * 1024> scratch {};
    uint64_t h = UINT64_C(14695981039346656037);
    for (uint64_t offset = 0; offset < expected.bytes;) {
        const size_t n = size_t(std::min<uint64_t>(scratch.size(), expected.bytes - offset));
        uint8_t * p = output ? restored.data() + size_t(offset) : scratch.data();
        if (!read_exact(fd, p, n, error)) { return false; }
        h = checksum(p, n, h);
        offset += n;
    }
    if (h != expected.checksum) { error = "checkpoint checksum mismatch"; return false; }
    // A concurrent truncation/extension cannot silently publish a partial blob.
    uint8_t extra;
    ssize_t tail;
    do { tail = ::read(fd, &extra, 1); } while (tail < 0 && errno == EINTR);
    if (tail != 0) { error = "checkpoint changed during read"; return false; }
    if (output) { output->swap(restored); }
    return true;
}

inline bool store(const std::string & path, metadata & m, std::vector<uint8_t> & data,
        std::string & error, write_fn writer = ::write) {
    if (data.empty()) { error = "cannot spill an empty checkpoint"; return false; }
    metadata candidate = m;
    candidate.bytes = data.size();
    candidate.checksum = checksum(data.data(), data.size());
    const auto bytes = header(candidate);
    const std::string pattern = path + ".tmp-XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const int fd = ::mkstemp(name.data()); // 0600, exclusive, same filesystem as destination
    if (fd < 0) { error = system_error("create checkpoint temporary file"); return false; }
    bool ok = write_exact(fd, bytes.data(), bytes.size(), error, writer) &&
        write_exact(fd, data.data(), data.size(), error, writer);
    // Surface delayed write errors before discarding the RAM source. No directory
    // fsync: this is explicitly disposable cache, not power-loss persistence.
    if (ok && ::fsync(fd) != 0) { error = system_error("flush checkpoint"); ok = false; }
    if (::close(fd) != 0) { error = system_error("close checkpoint"); ok = false; }
    if (ok) { ok = read_file(name.data(), candidate, nullptr, error); }
    if (ok && ::rename(name.data(), path.c_str()) != 0) { error = system_error("publish checkpoint"); ok = false; }
    if (!ok) { ::unlink(name.data()); return false; } // only this operation's exclusive temp file
    m = candidate;
    std::vector<uint8_t>().swap(data);
    return true;
}

inline bool restore(const std::string & path, const metadata & expected,
        std::vector<uint8_t> & data, std::string & error) {
    return read_file(path, expected, &data, error);
}
#else
inline std::string create_process_directory(const std::string &, std::string & error) {
    error = "private checkpoint spill requires a supported POSIX backend";
    return {};
}
inline bool store(const std::string &, metadata &, std::vector<uint8_t> &, std::string & error) {
    error = "private checkpoint spill unsupported"; return false;
}
inline bool restore(const std::string &, const metadata &, std::vector<uint8_t> &, std::string & error) {
    error = "private checkpoint spill unsupported"; return false;
}
#endif
} // namespace common_spill
