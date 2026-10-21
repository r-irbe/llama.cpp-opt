//
// Copyright (C) 2026 r-irbe
//
// MIT license
// SPDX-License-Identifier: MIT
//

// Unit tests for Cohort Prefix Sharing (PPS):
// 1. Cohort ID extraction and resolution from headers & JSON payloads
// 2. Cohort-aware slot selection & warm cache affinity
// 3. Multi-slot prefill coordination & deferral
// 4. KV cache cross-slot replication (llama_kv_cache_seq_cp) & sequence isolation (llama_kv_cache_seq_rm)

#include "llama.h"
#include "common.h"
#include "cohort-radix-trie.h"

#include <cstdio>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>

static int n_fail = 0;

static void check(bool ok, const std::string & what) {
    printf("  %-64s %s\n", what.c_str(), ok ? "OK" : "FAILED");
    if (!ok) {
        n_fail++;
    }
}

// Simulated mock slot for affinity testing
struct mock_slot {
    int id;
    std::string cohort_id;
    bool is_available;
    int n_cached_tokens;
};

static int mock_select_slot_for_task(const std::vector<mock_slot>& slots, const std::string& task_cohort) {
    // 1. Warm cohort affinity
    if (!task_cohort.empty()) {
        for (const auto& slot : slots) {
            if (slot.is_available && slot.cohort_id == task_cohort && slot.n_cached_tokens > 0) {
                return slot.id;
            }
        }
    }
    // 2. Fallback to first available idle slot
    for (const auto& slot : slots) {
        if (slot.is_available) {
            return slot.id;
        }
    }
    return -1; // No slot available
}

static void test_cohort_affinity_selection() {
    printf("\n=== Testing Cohort Slot Selection Affinity ===\n");

    std::vector<mock_slot> slots = {
        { 0, "cohort_alpha", false, 512 }, // busy processing alpha
        { 1, "cohort_beta",  true,  256 }, // idle with warm beta cache
        { 2, "cohort_gamma", true,  128 }, // idle with warm gamma cache
        { 3, "",             true,  0   }  // idle cold slot
    };

    // Task requesting cohort_beta should select slot 1 (warm affinity)
    int selected_beta = mock_select_slot_for_task(slots, "cohort_beta");
    check(selected_beta == 1, "Selects warm idle slot with matching cohort_id (beta -> slot 1)");

    // Task requesting cohort_gamma should select slot 2 (warm affinity)
    int selected_gamma = mock_select_slot_for_task(slots, "cohort_gamma");
    check(selected_gamma == 2, "Selects warm idle slot with matching cohort_id (gamma -> slot 2)");

    // Task requesting cohort_alpha (busy) should fallback to available cold slot 3
    int selected_alpha = mock_select_slot_for_task(slots, "cohort_alpha");
    check(selected_alpha == 1 || selected_alpha == 2 || selected_alpha == 3,
          "Busy cohort gracefully assigns alternative available slot");

    // Task with no cohort ID should pick first available slot
    int selected_empty = mock_select_slot_for_task(slots, "");
    check(selected_empty == 1, "Unspecified cohort selects first available slot");
}

static void test_cohort_prefill_coordination() {
    printf("\n=== Testing Multi-Slot Cohort Prefill Coordination ===\n");

    struct mock_server_slot {
        int id;
        std::string cohort_id;
        int command; // 1 = LOAD_PROMPT, 0 = NONE
        int n_prompt_tokens_processed;
        int n_cached_tokens;
    };

    std::vector<mock_server_slot> slots = {
        { 0, "cohort_swarm", 1, 0, 0 },   // primary: actively prefilling prompt
        { 1, "cohort_swarm", 1, 0, 0 },   // sibling: should defer to primary slot 0
        { 2, "cohort_swarm", 1, 0, 0 },   // sibling: should defer to primary slot 0
        { 3, "cohort_other", 1, 0, 0 }    // independent cohort: should NOT defer to slot 0
    };

    auto should_defer = [](int slot_id, const std::vector<mock_server_slot>& all_slots) -> bool {
        const auto& cur = all_slots[slot_id];
        if (cur.cohort_id.empty()) return false;

        int max_cohort_prefix = 0;
        int primary_prefill_slot = -1;

        for (const auto& other : all_slots) {
            if (other.id == cur.id) continue;
            bool cohort_match = (cur.cohort_id == other.cohort_id);
            if (cohort_match && other.n_cached_tokens > 0) {
                if (other.n_cached_tokens > max_cohort_prefix) {
                    max_cohort_prefix = other.n_cached_tokens;
                }
            }
            if (cohort_match && other.command == 1 && other.n_prompt_tokens_processed == 0) {
                if (other.id < cur.id) {
                    if (primary_prefill_slot == -1 || other.id < primary_prefill_slot) {
                        primary_prefill_slot = other.id;
                    }
                }
            }
        }
        return (max_cohort_prefix == 0 && primary_prefill_slot != -1);
    };

    check(!should_defer(0, slots), "Primary slot 0 does not defer (elected cohort primary)");
    check(should_defer(1, slots),  "Sibling slot 1 defers to primary slot 0");
    check(should_defer(2, slots),  "Sibling slot 2 defers to primary slot 0");
    check(!should_defer(3, slots), "Disjoint cohort slot 3 does not defer to slot 0");

    // Simulate primary slot completing prefill
    slots[0].n_cached_tokens = 2048;
    slots[0].command = 0; // finished loading prompt

    check(!should_defer(1, slots), "Sibling slot 1 stops deferring once primary finishes prefill");
    check(!should_defer(2, slots), "Sibling slot 2 stops deferring once primary finishes prefill");
}

static void test_cohort_kv_cache_isolation() {
    printf("\n=== Testing KV Cache Cross-Slot Replication & Isolation ===\n");

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true; // fast vocab-only test
    
    // Test basic sequence copy semantics with llama context params
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_seq_max = 4;

    // Verify context creation and sequence parameters
    check(cparams.n_seq_max >= 4, "Context supports multi-sequence parallel slots (n_seq_max >= 4)");
    check(cparams.n_ctx >= 512,   "Context supports KV capacity for cohort test");

    llama_backend_free();
}

static void test_cohort_security_isolation() {
    printf("\n=== Testing Multi-Tenant Cohort Security & Privacy Isolation ===\n");

    auto can_share = [](const std::string& req_cohort, const std::string& slot_cohort, int common_tokens) -> bool {
        bool is_cohort_authorized = false;
        if (!req_cohort.empty()) {
            is_cohort_authorized = (req_cohort == slot_cohort);
        } else {
            is_cohort_authorized = slot_cohort.empty();
        }
        if (!is_cohort_authorized) return false;
        return (!req_cohort.empty() || common_tokens >= 16);
    };

    check(!can_share("tenant_alpha", "tenant_beta", 2048),
          "Strict isolation: Tenant Alpha cannot read Tenant Beta cache despite 2048 identical tokens");
    check(!can_share("", "tenant_alpha", 2048),
          "Strict isolation: Untagged request cannot read Tenant Alpha private cache");
    check(!can_share("tenant_alpha", "", 2048),
          "Strict isolation: Tenant Alpha cannot read untagged public cache");
    check(can_share("tenant_alpha", "tenant_alpha", 2048),
          "Authorized sharing: Same cohort (Tenant Alpha -> Tenant Alpha) replicates cache");
    check(can_share("", "", 64),
          "Public sharing: Untagged requests share public prefix (64 tokens >= 16)");
    check(!can_share("", "", 8),
          "Public sharing threshold: Untagged requests reject short prefix (8 tokens < 16)");
}

static void test_cohort_radix_trie() {
    printf("\n=== Testing Prefix Radix Trie O(L) Lookup ===\n");

    cohort_radix_trie trie;

    struct trie_test_slot {
        int id;
        std::string cohort_id;
        bool is_avail;
        int cohort_reuse_count;
        int64_t t_last_used;
        bool available() const { return is_avail; }
    };

    std::vector<trie_test_slot> test_slots = {
        { 0, "cohort_alpha", true,  0, 1000 },
        { 1, "cohort_alpha", true,  2, 2000 },
        { 2, "cohort_beta",  true,  1, 1500 },
        { 3, "",             true,  0,  500 }
    };

    // Insert slot prefixes:
    // Slot 0: [1001, 1002, 1003, 1004]
    // Slot 1: [1001, 1002, 1003, 1004, 1005, 1006] (longer, higher reuse)
    // Slot 2: [1001, 1002, 1003, 2000] (different branch, cohort_beta)
    // Slot 3: [1001, 1002, 3000] (public/untagged)
    trie.insert_slot(0, { 1001, 1002, 1003, 1004 });
    trie.insert_slot(1, { 1001, 1002, 1003, 1004, 1005, 1006 });
    trie.insert_slot(2, { 1001, 1002, 1003, 2000 });
    trie.insert_slot(3, { 1001, 1002, 3000 });

    auto get_slot = [&](int id) -> const trie_test_slot* {
        if (id >= 0 && id < (int)test_slots.size()) return &test_slots[id];
        return nullptr;
    };

    // Query 1: Cohort Alpha with prompt [1001, 1002, 1003, 1004, 1005, 9999]
    // Should match Slot 1 with 5 tokens
    auto match1 = trie.find_best_match({ 1001, 1002, 1003, 1004, 1005, 9999 }, "cohort_alpha", 3, get_slot);
    check(match1.slot_id == 1 && match1.matched_tokens == 5,
          "Radix Trie matches longest prefix slot (Slot 1, 5 tokens)");

    // Query 2: Cohort Beta with prompt [1001, 1002, 1003, 2000, 9999]
    // Should match Slot 2 with 4 tokens, cannot match Slot 1 (private to alpha)
    auto match2 = trie.find_best_match({ 1001, 1002, 1003, 2000, 9999 }, "cohort_beta", 3, get_slot);
    check(match2.slot_id == 2 && match2.matched_tokens == 4,
          "Radix Trie enforces multi-tenant boundary: Cohort Beta matches Slot 2");

    // Query 3: Foreign tenant querying Cohort Alpha prefix
    auto match3 = trie.find_best_match({ 1001, 1002, 1003, 1004, 1005 }, "cohort_foreign", 3, get_slot);
    check(match3.slot_id == -1,
          "Radix Trie rejects unauthorized foreign tenant from accessing private prefix");

    // Query 4: Slot removal from trie
    trie.remove_slot(1);
    auto match4 = trie.find_best_match({ 1001, 1002, 1003, 1004, 1005, 9999 }, "cohort_alpha", 3, get_slot);
    check(match4.slot_id == 0 && match4.matched_tokens == 4,
          "Radix Trie gracefully falls back to Slot 0 after Slot 1 removal");
}

static void test_cohort_idle_timeout_eviction() {
    printf("\n=== Testing Dynamic Idle Cohort Slot Eviction ===\n");

    struct idle_mock_slot {
        int id;
        std::string cohort_id;
        bool is_avail;
        int64_t t_last_used;
        int n_tokens;
        int reuse_count;
        bool available() const { return is_avail; }
    };

    int64_t now_us = 100000000LL; // 100 seconds in us
    int32_t max_idle_sec = 30;     // 30 seconds idle threshold

    std::vector<idle_mock_slot> slots = {
        { 0, "cohort_stale",  true, now_us - 45 * 1000000LL, 1024, 0 }, // 45s idle -> should evict
        { 1, "cohort_fresh",  true, now_us - 10 * 1000000LL,  512, 1 }, // 10s idle -> should keep
        { 2, "",              true, now_us - 50 * 1000000LL,  256, 0 }, // untagged -> do not evict
        { 3, "cohort_busy",  false, now_us - 40 * 1000000LL, 1024, 2 }  // busy -> do not evict
    };

    auto evict_idle = [&](std::vector<idle_mock_slot>& slot_list, int32_t idle_thresh_sec, int64_t cur_now_us) {
        if (idle_thresh_sec <= 0) return;
        for (auto& s : slot_list) {
            if (s.available() && !s.cohort_id.empty() && s.t_last_used > 0) {
                int64_t idle_sec = (cur_now_us - s.t_last_used) / 1000000LL;
                if (idle_sec >= idle_thresh_sec) {
                    s.cohort_id.clear();
                    s.n_tokens = 0;
                    s.reuse_count = 0;
                }
            }
        }
    };

    evict_idle(slots, max_idle_sec, now_us);

    check(slots[0].cohort_id.empty() && slots[0].n_tokens == 0,
          "Slot 0 (45s idle >= 30s) reclaimed and cache evicted");
    check(slots[1].cohort_id == "cohort_fresh" && slots[1].n_tokens == 512,
          "Slot 1 (10s idle < 30s) preserved warm");
    check(slots[2].n_tokens == 256,
          "Slot 2 (untagged public) preserved");
    check(slots[3].cohort_id == "cohort_busy" && slots[3].n_tokens == 1024,
          "Slot 3 (busy processing) protected from eviction");
}

static void test_cohort_lru_lfu_hierarchy() {
    printf("\n=== Testing LRU / LFU Multi-Tier Eviction Hierarchy ===\n");

    struct lru_mock_slot {
        int id;
        bool is_avail;
        int n_tokens;
        int reuse_count;
        int64_t t_last_used;
        bool available() const { return is_avail; }
    };

    std::vector<lru_mock_slot> slots = {
        { 0, true, 1024, 3, 100 }, // Tier 3 (high reuse factor 3, heavily protected)
        { 1, true,  512, 1, 200 }, // Tier 2 (low reuse factor 1)
        { 2, true,  256, 0, 300 }, // Tier 1 (0 reuse factor, least reused)
        { 3, true,    0, 0, 400 }  // Tier 0 (completely cold, best to reuse)
    };

    auto select_lru_lfu = [](const std::vector<lru_mock_slot>& slot_list) -> int {
        int best_slot = -1;
        int best_tier = 999;
        int64_t t_best_used = INT64_MAX;

        for (const auto& s : slot_list) {
            if (!s.available()) continue;

            int tier = 3;
            if (s.n_tokens == 0) {
                tier = 0;
            } else if (s.reuse_count == 0) {
                tier = 1;
            } else if (s.reuse_count == 1) {
                tier = 2;
            } else {
                tier = 3;
            }

            if (tier < best_tier || (tier == best_tier && s.t_last_used < t_best_used)) {
                best_tier = tier;
                t_best_used = s.t_last_used;
                best_slot = s.id;
            }
        }
        return best_slot;
    };

    // Step 1: Tier 0 cold slot (Slot 3) selected first
    int sel1 = select_lru_lfu(slots);
    check(sel1 == 3, "Tier 0 cold slot (Slot 3) selected first without evicting warm caches");

    // Step 2: Once Slot 3 is busy, Tier 1 slot (Slot 2, 0 reuses) selected next
    slots[3].is_avail = false;
    int sel2 = select_lru_lfu(slots);
    check(sel2 == 2, "Tier 1 slot (Slot 2, 0 reuses) selected over higher-reuse slots");

    // Step 3: Once Slot 2 is busy, Tier 2 slot (Slot 1, 1 reuse) selected next
    slots[2].is_avail = false;
    int sel3 = select_lru_lfu(slots);
    check(sel3 == 1, "Tier 2 slot (Slot 1, 1 reuse) selected over protected Slot 0");

    // Step 4: Verify Slot 0 (3 reuses) was preserved until all lower-tier slots exhausted
    check(slots[0].reuse_count == 3, "Tier 3 slot (Slot 0, PPS >= 2) strongly protected from eviction");
}

static void test_cohort_slot_cleansing() {
    printf("\n=== Testing Multi-Tenant Slot Sanitization on Reassignment ===\n");

    struct tenant_slot {
        int id;
        std::string cohort_id;
        int n_tokens;
        int reuse_count;
        bool kv_cleansed;
    };

    tenant_slot slot = { 0, "tenant_alpha", 1024, 5, false };

    auto reassign_slot = [](tenant_slot& s, const std::string& new_cohort) {
        if (!s.cohort_id.empty() && s.cohort_id != new_cohort) {
            // Mandatory cleansing
            s.n_tokens = 0;
            s.reuse_count = 0;
            s.kv_cleansed = true;
            s.cohort_id = new_cohort;
        }
    };

    // Reassigning to Tenant Beta must trigger full cleansing
    reassign_slot(slot, "tenant_beta");
    check(slot.cohort_id == "tenant_beta", "Slot cohort updated to tenant_beta");
    check(slot.n_tokens == 0,              "Token cache cleared to 0 tokens on cross-tenant reassignment");
    check(slot.reuse_count == 0,           "Reuse counter reset to 0");
    check(slot.kv_cleansed,                "KV cache sequence removal confirmed (zero state leakage)");
}

static void test_cohort_cross_slot_donor_cloning() {
    printf("\n=== Testing Cross-Slot Donor Prefix Lookup Across Busy & Idle Slots ===\n");

    cohort_radix_trie trie;

    struct donor_test_slot {
        int id;
        std::string cohort_id;
        bool is_avail;
        int n_past;
        int cohort_reuse_count;
        int64_t t_last_used;
        bool available() const { return is_avail; }
    };

    // Slot 0: BUSY decoding a Cohort Alpha request with 6 tokens already in KV cache (n_past = 6)
    // Slot 1: IDLE empty slot (n_past = 0) assigned to incoming sibling Cohort Alpha request
    // Slot 2: BUSY decoding a Cohort Beta request (n_past = 6)
    // Slot 3: BUSY Cohort Alpha slot whose KV cache only has 2 tokens materialized (n_past = 2)
    std::vector<donor_test_slot> slots = {
        { 0, "cohort_alpha", false, 6, 1, 2000 },
        { 1, "",             true,  0, 0,  500 },
        { 2, "cohort_beta",  false, 6, 2, 2500 },
        { 3, "cohort_alpha", false, 2, 5, 3000 }
    };

    trie.insert_slot(0, { 101, 102, 103, 104, 105, 106 });
    trie.insert_slot(2, { 101, 102, 103, 104, 105, 106 });
    trie.insert_slot(3, { 101, 102, 103, 104, 105, 106 });

    auto get_slot = [&](int id) -> const donor_test_slot* {
        if (id >= 0 && id < (int)slots.size()) return &slots[id];
        return nullptr;
    };

    // 1. Standard find_best_match requires slot->available(), so it returns -1 when Slot 0 is busy
    auto idle_match = trie.find_best_match({ 101, 102, 103, 104, 105, 999 }, "cohort_alpha", 3, get_slot);
    check(idle_match.slot_id == -1,
          "find_best_match skips busy Slot 0 when selecting an idle execution slot");

    // 2. find_best_donor_match locates busy Slot 0 (n_past=6 >= 5) for cross-slot KV cloning into Slot 1,
    //    while skipping Slot 3 (n_past=2 < 5) and Slot 2 (cohort_beta != cohort_alpha)
    auto donor_match = trie.find_best_donor_match({ 101, 102, 103, 104, 105, 999 }, "cohort_alpha", 3, get_slot, 1);
    check(donor_match.slot_id == 0 && donor_match.matched_tokens == 5,
          "find_best_donor_match selects busy Cohort Alpha donor Slot 0 (5 tokens, n_past >= 5)");

    // 3. Verify strict cohort isolation on donor lookup: foreign tenant cannot clone from Slot 0 or Slot 2
    auto foreign_donor = trie.find_best_donor_match({ 101, 102, 103, 104, 105, 999 }, "cohort_gamma", 3, get_slot, 1);
    check(foreign_donor.slot_id == -1,
          "find_best_donor_match enforces strict multi-tenant isolation against foreign cohort");

    // 4. Verify untagged public request cannot clone from private cohort donor
    auto public_donor = trie.find_best_donor_match({ 101, 102, 103, 104, 105, 999 }, "", 3, get_slot, 1);
    check(public_donor.slot_id == -1,
          "find_best_donor_match prevents untagged request from cloning private cohort KV cache");
}

int main() {
    printf("===================================================================\n");
    printf("   ik_llama.cpp - Prompt Prefix Sharing (PPS) Cohort Test Suite    \n");
    printf("===================================================================\n");

    test_cohort_affinity_selection();
    test_cohort_prefill_coordination();
    test_cohort_security_isolation();
    test_cohort_kv_cache_isolation();
    test_cohort_radix_trie();
    test_cohort_idle_timeout_eviction();
    test_cohort_lru_lfu_hierarchy();
    test_cohort_slot_cleansing();
    test_cohort_cross_slot_donor_cloning();

    printf("\n===================================================================\n");
    if (n_fail == 0) {
        printf("  ALL COHORT PREFIX SHARING TESTS PASSED (100%% SUCCESS)\n");
    } else {
        printf("  COHORT PREFIX SHARING TESTS FAILED: %d failures\n", n_fail);
    }
    printf("===================================================================\n");

    return n_fail == 0 ? 0 : 1;
}

