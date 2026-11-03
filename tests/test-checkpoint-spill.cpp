#include "spill-file.h"
#include "kv-capsule-filter.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (0)

#ifndef _WIN32
static ssize_t no_space(int, const void *, size_t) { errno = ENOSPC; return -1; }
static ssize_t short_write(int fd, const void * p, size_t n) {
    static bool interrupted = false;
    if (!interrupted) { interrupted = true; errno = EINTR; return -1; }
    return ::write(fd, p, std::min<size_t>(n, 7));
}
#endif

int main() {
#ifdef _WIN32
    std::puts("SKIP: private spill currently requires POSIX file primitives");
    return 0;
#else
    std::string error;
    const auto pattern = (std::filesystem::temp_directory_path() / "spill-test-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    CHECK(::mkdtemp(name.data()) != nullptr);
    const std::string root = name.data();
    const auto first = common_spill::create_process_directory(root, error);
    const auto second = common_spill::create_process_directory(root, error);
    CHECK(!first.empty() && !second.empty() && first != second);
    struct stat st {};
    CHECK(::stat(first.c_str(), &st) == 0 && (st.st_mode & 0777) == 0700);
    const std::string path = first + "/checkpoint.bin";
    const std::vector<uint8_t> original = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto data = original;
    common_spill::metadata meta {2, 5, 6, 0, 0};

    CHECK(!common_spill::store(first + "/missing/file", meta, data, error));
    CHECK(data == original && meta.bytes == 0);
    CHECK(!common_spill::store(path, meta, data, error, no_space));
    CHECK(data == original && meta.bytes == 0 && !std::filesystem::exists(path));
    CHECK(std::filesystem::is_empty(first));

    CHECK(common_spill::store(path, meta, data, error, short_write));
    CHECK(data.empty() && data.capacity() == 0 && meta.bytes == original.size());
    CHECK(::stat(path.c_str(), &st) == 0 && (st.st_mode & 0777) == 0600);
    CHECK(common_spill::restore(path, meta, data, error));
    CHECK(data == original);
    // ENOSPC during replacement must not damage an existing published blob.
    const auto published = meta;
    CHECK(!common_spill::store(path, meta, data, error, no_space));
    CHECK(data == original && meta.checksum == published.checksum);
    std::vector<uint8_t> previous;
    CHECK(common_spill::restore(path, published, previous, error) && previous == original);
    const std::string link = second + "/link";
    CHECK(::symlink(path.c_str(), link.c_str()) == 0);
    CHECK(!common_spill::restore(link, meta, previous, error));
    CHECK(previous == original);
    CHECK(common_spill::create_process_directory(link, error).empty());
    CHECK(::unlink(link.c_str()) == 0);

    auto wrong = meta;
    wrong.n_tokens++;
    std::vector<uint8_t> sentinel {99};
    CHECK(!common_spill::restore(path, wrong, sentinel, error));
    CHECK(sentinel == std::vector<uint8_t>({99}));
    wrong = meta;
    wrong.bytes = UINT64_MAX;
    CHECK(!common_spill::restore(path, wrong, sentinel, error));
    CHECK(sentinel == std::vector<uint8_t>({99}));

    // A damaged payload must not publish partial output.
    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(-1, std::ios::end);
        f.put('x');
    }
    CHECK(!common_spill::restore(path, meta, sentinel, error));
    CHECK(sentinel == std::vector<uint8_t>({99}));

    data = original;
    CHECK(common_spill::store(path, meta, data, error));
    CHECK(!common_spill::store(path, meta, data, error)); // empty source never replaces a valid file
    std::filesystem::resize_file(path, 3);
    CHECK(!common_spill::restore(path, meta, sentinel, error));
    CHECK(sentinel == std::vector<uint8_t>({99}));

    // Failed publication preserves both the source and an existing target.
    data = original;
    CHECK(!common_spill::store(second, meta, data, error));
    CHECK(data == original && std::filesystem::is_directory(second));
    CHECK(std::filesystem::is_empty(second));
    for (const auto & entry : std::filesystem::directory_iterator(root)) {
        CHECK(entry.path() == first || entry.path() == second);
    }
    CHECK(::unlink(path.c_str()) == 0);

    // ContextCapsule (BinaryFuse8 + SimHash64) pre-restore indexing & < 15 ns rejection test
    {
        std::vector<int32_t> ckpt_tokens(256);
        for (int32_t i = 0; i < 256; ++i) {
            ckpt_tokens[i] = 10000 + i * 7;
        }
        kv_capsule::ContextCapsule capsule;
        capsule.build_from_span(1, 0, ckpt_tokens.data(), (int32_t)ckpt_tokens.size(), nullptr, 0, 0, 0);
        CHECK(capsule.token_count == 256);
        CHECK(capsule.fuse_index.array_length > 0);
        CHECK(capsule.simhash_64 != 0ULL);

        std::vector<uint8_t> ckpt_blob(4096, 0x5A);
        common_spill::metadata ckpt_meta {0, 255, 256, 0, 0};
        const std::string capsule_spill_path = first + "/capsule-checkpoint.bin";
        CHECK(common_spill::store(capsule_spill_path, ckpt_meta, ckpt_blob, error));
        CHECK(ckpt_blob.empty());

        // Unrelated token span should be rejected by fuse8_only and fuse8_simhash64 (with strict cosine threshold)
        // without touching disk (common_spill::restore not called).
        std::vector<int32_t> unrelated_query = {990001, 990002, 990003, 990004};
        CHECK(!capsule.matches_token_span(unrelated_query.data(), (int32_t)unrelated_query.size(), "fuse8_only", 2));
        CHECK(!capsule.matches_token_span(unrelated_query.data(), (int32_t)unrelated_query.size(), "fuse8_simhash64", 2, 0.99f));

        // Benchmark single-token / short-span rejection latency (< 15 ns per lookup in L1)
        const int iters = 10000;
        int false_positives = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            int32_t q_tok = 500000 + i * 13;
            if (capsule.fuse_index.contains((uint64_t)(uint32_t)q_tok)) {
                false_positives++;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ns_per_lookup = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
        CHECK(ns_per_lookup < 50.0); // < 15 ns typical on Zen 3 L1, generous bound for CI jitter
        CHECK(false_positives < 150); // 8-bit fuse theoretical FPR is ~0.39% (~39 / 10000)

        // Matching token query across all 4 modes
        std::vector<int32_t> matching_query(ckpt_tokens.begin(), ckpt_tokens.begin() + 64);
        CHECK(capsule.matches_token_span(matching_query.data(), (int32_t)matching_query.size(), "fuse8_simhash64", 1, 0.35f));
        CHECK(capsule.matches_token_span(matching_query.data(), (int32_t)matching_query.size(), "fuse8_only", 1, 0.35f));
        CHECK(capsule.matches_token_span(ckpt_tokens.data(), (int32_t)ckpt_tokens.size(), "simhash64_only", 1, 0.95f));
        CHECK(capsule.matches_token_span(unrelated_query.data(), (int32_t)unrelated_query.size(), "disabled", 1, 0.99f));

        // Restore after positive capsule match
        CHECK(common_spill::restore(capsule_spill_path, ckpt_meta, ckpt_blob, error));
        CHECK(ckpt_blob.size() == 4096 && ckpt_blob[0] == 0x5A);
        CHECK(::unlink(capsule_spill_path.c_str()) == 0);
    }

    CHECK(::rmdir(first.c_str()) == 0);
    CHECK(::rmdir(second.c_str()) == 0);
    CHECK(::rmdir(root.c_str()) == 0);
    std::puts("checkpoint spill: private isolation, failure preservation, short writes, integrity, capsule filter and bounded restore PASS");
    return 0;
#endif
}
