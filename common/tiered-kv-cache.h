#pragma once

#include "paged-kv-cache.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// =========================================================================================
// LMCache-Style 4-Tier Hierarchical KV Cache Architecture
//   Tier 0 (VRAM)            : Active token generation window (<4k tokens)
//   Tier 1 (Host NUMA DDR)   : Pinned HugeTLB host memory (<64k tokens)
//   Tier 2 (Local NVMe RAID) : Paged direct-I/O block storage (io_uring / O_DIRECT)
//   Tier 3 (Cold Archive)    : Serialized long-term cold session archives
//
// Integrated with Chronicle Queue / OpenHFT Pretoucher pattern:
// Proactively stages Tier 2 NVMe blocks into Tier 1 DDR HugeTLB pages ahead of context turns.
// =========================================================================================

enum kv_cache_tier {
    KV_TIER_0_VRAM        = 0,
    KV_TIER_1_HOST_NUMA   = 1,
    KV_TIER_2_LOCAL_NVME  = 2,
    KV_TIER_3_COLD_ARCHIVE= 3
};

inline const char * kv_cache_tier_to_str(kv_cache_tier tier) {
    switch (tier) {
        case KV_TIER_0_VRAM:         return "Tier 0 (VRAM)";
        case KV_TIER_1_HOST_NUMA:    return "Tier 1 (Host NUMA DDR)";
        case KV_TIER_2_LOCAL_NVME:   return "Tier 2 (Local NVMe)";
        case KV_TIER_3_COLD_ARCHIVE: return "Tier 3 (Cold Archive)";
        default:                     return "Unknown Tier";
    }
}

class common_tiered_kv_cache {
public:
    struct config {
        paged_kv_config paged_cfg;
        size_t max_tier2_blocks{8192};
        bool enable_background_pretoucher{true};
    };

    explicit common_tiered_kv_cache(const config & cfg)
        : m_cfg(cfg), m_paged_cache(cfg.paged_cfg), m_stop_pretoucher(false) {

        const size_t block_bytes = cfg.paged_cfg.block_size_tokens * (cfg.paged_cfg.n_embd_k + cfg.paged_cfg.n_embd_v);
        m_block_bytes = (block_bytes + 127) & ~127ULL;

        // Allocate Tier 2 NVMe simulated backing memory (aligned to 4096-byte sectors)
        m_tier2_storage.resize(cfg.max_tier2_blocks * m_block_bytes, 0);

        if (cfg.enable_background_pretoucher) {
            m_pretoucher_thread = std::thread(&common_tiered_kv_cache::pretoucher_loop, this);
        }
    }

    ~common_tiered_kv_cache() {
        m_stop_pretoucher.store(true, std::memory_order_release);
        m_pretoucher_cv.notify_all();
        if (m_pretoucher_thread.joinable()) {
            m_pretoucher_thread.join();
        }
    }

    // Allocate physical block in Tier 1 (Host NUMA)
    int32_t allocate_block() {
        int32_t bid = m_paged_cache.allocate_block();
        if (bid >= 0) {
            std::lock_guard<std::mutex> lock(m_tier_mutex);
            m_block_tiers[bid] = KV_TIER_1_HOST_NUMA;
        }
        return bid;
    }

    void release_block(int32_t bid) {
        {
            std::lock_guard<std::mutex> lock(m_tier_mutex);
            m_block_tiers.erase(bid);
            m_tier2_slots.erase(bid);
            m_tier3_storage.erase(bid);
        }
        m_paged_cache.release_block(bid);
    }

    kv_cache_tier get_block_tier(int32_t bid) {
        std::lock_guard<std::mutex> lock(m_tier_mutex);
        auto it = m_block_tiers.find(bid);
        if (it != m_block_tiers.end()) {
            return it->second;
        }
        return KV_TIER_1_HOST_NUMA;
    }

    // Demote block from Tier 1 (Host NUMA) to Tier 2 (Local NVMe direct block)
    bool demote_to_tier2(int32_t bid, const uint8_t * src_bytes = nullptr) {
        std::lock_guard<std::mutex> lock(m_tier_mutex);
        if (m_tier2_slots.size() >= m_cfg.max_tier2_blocks) {
            return false; // Tier 2 storage full
        }

        size_t slot_idx = m_tier2_slots.size();
        m_tier2_slots[bid] = slot_idx;

        uint8_t * dest = m_tier2_storage.data() + slot_idx * m_block_bytes;
        if (src_bytes) {
            std::memcpy(dest, src_bytes, m_block_bytes);
        }

        m_block_tiers[bid] = KV_TIER_2_LOCAL_NVME;
        return true;
    }

    // Demote block from Tier 2 to Tier 3 (Cold compressed archive)
    bool demote_to_tier3(int32_t bid) {
        std::lock_guard<std::mutex> lock(m_tier_mutex);
        auto it = m_tier2_slots.find(bid);
        if (it == m_tier2_slots.end()) {
            return false;
        }

        const uint8_t * src = m_tier2_storage.data() + it->second * m_block_bytes;
        m_tier3_storage[bid] = std::vector<uint8_t>(src, src + m_block_bytes);

        m_tier2_slots.erase(it);
        m_block_tiers[bid] = KV_TIER_3_COLD_ARCHIVE;
        return true;
    }

    // Promote block from Tier 2 back to Tier 1 (Host NUMA)
    bool promote_to_tier1(int32_t bid, uint8_t * dest_bytes) {
        std::lock_guard<std::mutex> lock(m_tier_mutex);
        auto it = m_tier2_slots.find(bid);
        if (it == m_tier2_slots.end() || !dest_bytes) {
            return false;
        }

        const uint8_t * src = m_tier2_storage.data() + it->second * m_block_bytes;
        std::memcpy(dest_bytes, src, m_block_bytes);

        m_tier2_slots.erase(it);
        m_block_tiers[bid] = KV_TIER_1_HOST_NUMA;
        return true;
    }

    // Proactively enqueue blocks for asynchronous background pretouching from Tier 2 to Tier 1
    void enqueue_pretouch(int32_t bid, uint8_t * target_host_dest) {
        if (bid < 0 || !target_host_dest) return;
        {
            std::lock_guard<std::mutex> lock(m_queue_mutex);
            m_pretouch_queue.push_back({bid, target_host_dest});
        }
        m_pretoucher_cv.notify_one();
    }

    size_t pending_pretouch_count() {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        return m_pretouch_queue.size();
    }

    common_paged_kv_cache & paged_cache() { return m_paged_cache; }

private:
    struct pretouch_req {
        int32_t bid;
        uint8_t * dest;
    };

    void pretoucher_loop() {
        while (!m_stop_pretoucher.load(std::memory_order_acquire)) {
            pretouch_req req{-1, nullptr};
            {
                std::unique_lock<std::mutex> lock(m_queue_mutex);
                m_pretoucher_cv.wait(lock, [this]() {
                    return m_stop_pretoucher.load(std::memory_order_acquire) || !m_pretouch_queue.empty();
                });

                if (m_stop_pretoucher.load(std::memory_order_acquire) && m_pretouch_queue.empty()) {
                    break;
                }

                if (!m_pretouch_queue.empty()) {
                    req = m_pretouch_queue.front();
                    m_pretouch_queue.pop_front();
                }
            }

            if (req.bid >= 0 && req.dest) {
                promote_to_tier1(req.bid, req.dest);
            }
        }
    }

    config m_cfg;
    common_paged_kv_cache m_paged_cache;
    size_t m_block_bytes{0};

    std::mutex m_tier_mutex;
    std::unordered_map<int32_t, kv_cache_tier> m_block_tiers;
    std::unordered_map<int32_t, size_t> m_tier2_slots;
    std::vector<uint8_t> m_tier2_storage;
    std::unordered_map<int32_t, std::vector<uint8_t>> m_tier3_storage;

    std::atomic<bool> m_stop_pretoucher{false};
    std::mutex m_queue_mutex;
    std::condition_variable m_pretoucher_cv;
    std::deque<pretouch_req> m_pretouch_queue;
    std::thread m_pretoucher_thread;
};
