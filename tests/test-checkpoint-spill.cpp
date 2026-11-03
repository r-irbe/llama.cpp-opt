#include "spill-file.h"

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
    CHECK(::rmdir(first.c_str()) == 0);
    CHECK(::rmdir(second.c_str()) == 0);
    CHECK(::rmdir(root.c_str()) == 0);
    std::puts("checkpoint spill: private isolation, failure preservation, short writes, integrity and bounded restore PASS");
    return 0;
#endif
}
