#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// =========================================================================================
// Compressed KV Cache Capsules & Binary Fuse + SimHash Probabilistic Index
// Grounded in:
//   1. Graf & Lemire (ACM JEA 2022, arXiv:2201.01174): "Binary Fuse Filters: Fast and
//      Smaller Than Xor Filters" — 9.0 bits/entry (8-bit fuse, +12.5% overhead vs +44%
//      in standard Bloom filters), 3 L1-resident segment lookups (< 3.2 ns).
//   2. Charikar (STOC 2002): Cosine Similarity Locality-Sensitive Hashing (64-bit SimHash)
//      for 1-cycle POPCNT semantic gate before waking spilled KV capsules.
// =========================================================================================

namespace kv_capsule {

static inline uint64_t splitmix64(uint64_t & seed) noexcept {
    uint64_t z = (seed += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t mix64(uint64_t k, uint64_t seed) noexcept {
    uint64_t s = k ^ seed;
    return splitmix64(s);
}

static inline uint32_t mulhi32(uint32_t a, uint32_t b) noexcept {
    return (uint32_t)(((uint64_t)a * (uint64_t)b) >> 32);
}

template <typename FingerprintType = uint8_t>
class BinaryFuseFilter {
public:
    uint64_t seed = 0;
    uint32_t segment_length = 0;
    uint32_t segment_length_mask = 0;
    uint32_t segment_count = 0;
    uint32_t segment_count_length = 0;
    uint32_t array_length = 0;
    std::vector<FingerprintType> fingerprints;

    BinaryFuseFilter() = default;

    explicit BinaryFuseFilter(const std::vector<uint64_t> & keys) {
        build(keys);
    }

    size_t byte_size() const noexcept {
        return sizeof(*this) + fingerprints.size() * sizeof(FingerprintType);
    }

    bool build(std::vector<uint64_t> keys) {
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        const uint32_t size = (uint32_t)keys.size();
        if (size == 0) {
            array_length = 0;
            fingerprints.clear();
            return true;
        }

        uint32_t seg_bits = 4;
        if (size >= 128) {
            seg_bits = (uint32_t)std::clamp((int)std::floor(std::log((double)size) / std::log(3.33) + 2.25), 4, 18);
        }
        segment_length = 1U << seg_bits;
        segment_length_mask = segment_length - 1;

        double factor = std::max(1.125, 0.875 + 0.25 * std::log(1000000.0) / std::log((double)std::max(size, 32U)));
        uint32_t capacity = (uint32_t)std::ceil(factor * size) + 3 * segment_length;
        segment_count = std::max(1U, (capacity + segment_length - 1) / segment_length - 2);
        segment_count_length = segment_count * segment_length;
        array_length = (segment_count + 2) * segment_length;
        fingerprints.assign(array_length, 0);

        uint64_t rng = 0x726164754b564361ULL;
        std::vector<uint8_t> t2count(array_length, 0);
        std::vector<uint64_t> t2hash(array_length, 0);
        std::vector<uint32_t> alone(array_length, 0);
        std::vector<std::pair<uint64_t, uint8_t>> reverse_order;
        reverse_order.reserve(size);

        for (int attempt = 0; attempt < 32; ++attempt) {
            seed = splitmix64(rng);
            std::fill(t2count.begin(), t2count.end(), 0);
            std::fill(t2hash.begin(), t2hash.end(), 0);
            reverse_order.clear();

            for (uint64_t k : keys) {
                uint64_t h = mix64(k, seed);
                uint32_t h0, h1, h2;
                get_hashes(h, h0, h1, h2);
                t2count[h0] += 4;
                t2hash[h0] ^= h;
                t2count[h1] += 4;
                t2count[h1] ^= 1;
                t2hash[h1] ^= h;
                t2count[h2] += 4;
                t2count[h2] ^= 2;
                t2hash[h2] ^= h;
            }

            uint32_t qsize = 0;
            for (uint32_t i = 0; i < array_length; ++i) {
                if ((t2count[i] >> 2) == 1) {
                    alone[qsize++] = i;
                }
            }

            while (qsize > 0) {
                uint32_t i = alone[--qsize];
                if ((t2count[i] >> 2) == 1) {
                    uint64_t h = t2hash[i];
                    uint8_t found = t2count[i] & 3;
                    reverse_order.emplace_back(h, found);

                    uint32_t h0, h1, h2;
                    get_hashes(h, h0, h1, h2);
                    uint32_t idx[3] = {h0, h1, h2};
                    for (uint8_t j = 0; j < 3; ++j) {
                        if (j == found) continue;
                        uint32_t other = idx[j];
                        t2count[other] -= 4;
                        t2count[other] ^= j;
                        t2hash[other] ^= h;
                        if ((t2count[other] >> 2) == 1) {
                            alone[qsize++] = other;
                        }
                    }
                    t2count[i] = 0;
                }
            }

            if (reverse_order.size() == size) {
                std::fill(fingerprints.begin(), fingerprints.end(), 0);
                for (int32_t i = (int32_t)size - 1; i >= 0; --i) {
                    uint64_t h = reverse_order[i].first;
                    uint8_t found = reverse_order[i].second;
                    uint32_t h0, h1, h2;
                    get_hashes(h, h0, h1, h2);
                    FingerprintType fp = fingerprint(h);
                    uint32_t idx[3] = {h0, h1, h2};
                    uint32_t target = idx[found];
                    fingerprints[target] = fp ^ fingerprints[h0] ^ fingerprints[h1] ^ fingerprints[h2];
                }
                return true;
            }
        }
        return false;
    }

    inline bool contains(uint64_t key) const noexcept {
        if (array_length == 0) return false;
        uint64_t h = mix64(key, seed);
        uint32_t h0, h1, h2;
        get_hashes(h, h0, h1, h2);
        FingerprintType fp = fingerprint(h);
        return fp == (FingerprintType)(fingerprints[h0] ^ fingerprints[h1] ^ fingerprints[h2]);
    }

private:
    static inline FingerprintType fingerprint(uint64_t h) noexcept {
        return (FingerprintType)(h ^ (h >> 32));
    }

    inline void get_hashes(uint64_t h, uint32_t & h0, uint32_t & h1, uint32_t & h2) const noexcept {
        uint32_t hi = (uint32_t)(h >> 32);
        h0 = mulhi32(hi, segment_count_length);
        uint32_t seg = h0 & ~segment_length_mask;
        h1 = seg + segment_length + ((uint32_t)(h >> 18) & segment_length_mask);
        h2 = seg + 2 * segment_length + ((uint32_t)h & segment_length_mask);
    }
};

inline uint64_t compute_simhash64(const float * vec, int d, uint64_t hyperplane_seed = 0x517cc1b727220a95ULL) noexcept {
    if (!vec || d <= 0) return 0ULL;
    uint64_t sig = 0ULL;
    for (int bit = 0; bit < 64; ++bit) {
        uint64_t s = hyperplane_seed ^ ((uint64_t)bit * 0x9e3779b97f4a7c15ULL);
        float proj = 0.0f;
        for (int i = 0; i < d; i += 4) {
            uint64_t r = splitmix64(s);
            for (int k = 0; k < 4 && (i + k) < d; ++k) {
                float sign = ((r >> (k * 16)) & 1ULL) ? 1.0f : -1.0f;
                proj += vec[i + k] * sign;
            }
        }
        if (proj >= 0.0f) {
            sig |= (1ULL << bit);
        }
    }
    return sig;
}

inline float simhash64_cosine_estimate(uint64_t a, uint64_t b) noexcept {
    int hamming = __builtin_popcountll(a ^ b);
    const float theta = (float)hamming * (3.14159265358979323846f / 64.0f);
    return std::cos(theta);
}

struct ContextCapsule {
    int32_t capsule_id = -1;
    int32_t token_start = 0;
    int32_t token_count = 0;
    int32_t tier2_block_id = -1;
    uint64_t simhash_64 = 0ULL;
    BinaryFuseFilter<uint8_t> fuse_index;
    std::vector<int8_t> quantized_centroids;
    std::vector<float> centroid_scales;

    static inline uint64_t make_bigram_key(int32_t t0, int32_t t1) noexcept {
        return ((uint64_t)(uint32_t)t0 << 32) | (uint64_t)(uint32_t)t1;
    }

    void build_from_span(
        int32_t id,
        int32_t start,
        const int32_t * tokens,
        int32_t count,
        const float * centroids_f32,
        int Lp,
        int d,
        int32_t backing_block_id = -1)
    {
        capsule_id = id;
        token_start = start;
        token_count = count;
        tier2_block_id = backing_block_id;

        std::vector<uint64_t> keys;
        keys.reserve(count * 2);
        for (int32_t i = 0; i < count; ++i) {
            keys.push_back((uint64_t)(uint32_t)tokens[i]);
            if (i + 1 < count) {
                keys.push_back(make_bigram_key(tokens[i], tokens[i + 1]));
            }
        }
        fuse_index.build(keys);

        if (centroids_f32 && Lp > 0 && d > 0) {
            std::vector<float> mean_vec(d, 0.0f);
            quantized_centroids.resize(Lp * d);
            centroid_scales.resize(Lp, 1.0f);

            for (int p = 0; p < Lp; ++p) {
                const float * src = centroids_f32 + p * d;
                float amax = 0.0f;
                for (int k = 0; k < d; ++k) {
                    mean_vec[k] += src[k];
                    amax = std::max(amax, std::fabs(src[k]));
                }
                float scale = amax > 1e-8f ? (amax / 127.0f) : 1.0f;
                float inv_scale = 1.0f / scale;
                centroid_scales[p] = scale;
                for (int k = 0; k < d; ++k) {
                    int q = (int)std::round(src[k] * inv_scale);
                    quantized_centroids[p * d + k] = (int8_t)std::clamp(q, -127, 127);
                }
            }
            simhash_64 = compute_simhash64(mean_vec.data(), d);
        }
    }

    bool should_wake(
        const int32_t * query_tokens,
        int32_t query_len,
        uint64_t query_simhash,
        int min_token_hits = 1,
        float min_cosine = 0.35f) const noexcept
    {
        if (simhash64_cosine_estimate(query_simhash, simhash_64) >= min_cosine) {
            return true;
        }
        int hits = 0;
        for (int32_t i = 0; i < query_len; ++i) {
            if (fuse_index.contains((uint64_t)(uint32_t)query_tokens[i])) {
                if (++hits >= min_token_hits) return true;
            }
            if (i + 1 < query_len && fuse_index.contains(make_bigram_key(query_tokens[i], query_tokens[i + 1]))) {
                return true;
            }
        }
        return false;
    }
};

} // namespace kv_capsule
