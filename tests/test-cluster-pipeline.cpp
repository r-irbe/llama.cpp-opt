#include "distributed-cluster-pipeline.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// =========================================================================================
// Test Suite: Distributed Multi-Node & Cluster Scale-Out Pipeline Architecture
// =========================================================================================

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  TEST: Distributed Multi-Node & Cluster Scale-Out Pipeline     \n";
    std::cout << "=================================================================\n\n";

    // -------------------------------------------------------------------------------------
    // Test 1: DeepSeek 3FS / RDMA Zero-Copy Checkpoint Ingestion Engine (400 Gbps)
    // -------------------------------------------------------------------------------------
    std::cout << "[1] Testing DeepSeek 3FS RDMA Zero-Copy Memory Region & Slice Ingestion...\n";
    rdma_3fs_client client("roce://10.0.0.1:4791/3fs-cluster", 128);

    const size_t test_slice_size = 512 * 1024; // 512 KB
    std::vector<uint8_t> dst_buffer(test_slice_size, 0);
    std::vector<uint8_t> src_payload(test_slice_size, 0xEF);

    auto mr = client.register_memory_region(dst_buffer.data(), test_slice_size, 0x1001, 0x2002);
    if (!mr.is_registered) {
        std::cerr << "FAIL: RDMA memory region registration failed!\n";
        return 1;
    }

    int rc = client.stream_checkpoint_slice(mr, 0, src_payload.data(), test_slice_size);
    if (rc != 0 || client.get_total_bytes_streamed() != test_slice_size) {
        std::cerr << "FAIL: 3FS checkpoint slice streaming failed!\n";
        return 1;
    }

    if (dst_buffer[0] != 0xEF || dst_buffer[test_slice_size - 1] != 0xEF) {
        std::cerr << "FAIL: 3FS payload byte integrity mismatch!\n";
        return 1;
    }
    std::cout << "  Streamed 512 KB checkpoint slice via 3FS RDMA client (ops: " 
              << client.get_transfer_ops() << ")\n";
    std::cout << "  DeepSeek 3FS RDMA Ingestion PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 2: Multi-Node MoE Pipeline & Tensor Parallel Router (256 Experts)
    // -------------------------------------------------------------------------------------
    std::cout << "[2] Testing Multi-Node MoE Routing across 4 Cluster Ranks (256 Experts)...\n";
    cluster_moe_router router(0 /* local_rank = 0 */, 4 /* world_size = 4 */, 256 /* total_experts */);

    // Selected top-8 experts:
    // Rank 0 owns [0..64):   10, 45
    // Rank 1 owns [64..128): 70, 115
    // Rank 2 owns [128..192): 140
    // Rank 3 owns [192..256): 195, 210, 250
    std::vector<int> selected_top8 = {10, 45, 70, 115, 140, 195, 210, 250};
    std::vector<int> local_experts;
    std::unordered_map<int, std::vector<int>> remote_experts;

    router.partition_expert_batch(selected_top8, local_experts, remote_experts);

    if (local_experts.size() != 2 || local_experts[0] != 10 || local_experts[1] != 45) {
        std::cerr << "FAIL: Local expert partitioning mismatch on Rank 0!\n";
        return 1;
    }

    if (remote_experts[1].size() != 2 || remote_experts[2].size() != 1 || remote_experts[3].size() != 3) {
        std::cerr << "FAIL: Remote expert partitioning mismatch across cluster ranks!\n";
        return 1;
    }
    std::cout << "  Local Rank 0: " << local_experts.size() << " experts\n";
    for (const auto & pair : remote_experts) {
        std::cout << "  Remote Rank " << pair.first << ": " << pair.second.size() << " experts\n";
    }
    std::cout << "  Multi-Node MoE Routing PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 3: Decentralized Multi-Tenant Cohort Prefix Synchronization
    // -------------------------------------------------------------------------------------
    std::cout << "[3] Testing Decentralized Cohort Prefix Synchronization across Nodes...\n";
    distributed_cohort_sync node0_sync(0);
    distributed_cohort_sync node1_sync(1);

    // Node 0 computes and publishes warm cohort prefix
    node0_sync.publish_cohort_prefix("tenant_finance", "system_prompt_q3", 4096);

    cohort_cluster_descriptor desc_lookup;
    bool found_on_node0 = node0_sync.lookup_cohort_prefix("tenant_finance", "system_prompt_q3", desc_lookup);
    if (!found_on_node0 || desc_lookup.hosting_node_rank != 0 || desc_lookup.prefix_length != 4096) {
        std::cerr << "FAIL: Node 0 failed to look up its own published prefix!\n";
        return 1;
    }

    // Broadcast announcement to Node 1
    node1_sync.ingest_remote_cohort_announcement(desc_lookup);

    cohort_cluster_descriptor node1_lookup;
    bool found_on_node1 = node1_sync.lookup_cohort_prefix("tenant_finance", "system_prompt_q3", node1_lookup);
    if (!found_on_node1 || node1_lookup.hosting_node_rank != 0 || node1_lookup.prefix_length != 4096) {
        std::cerr << "FAIL: Node 1 failed to locate remote cohort prefix on Node 0!\n";
        return 1;
    }

    std::cout << "  Discovered warm prefix on remote Node " << node1_lookup.hosting_node_rank 
              << " (" << node1_lookup.prefix_length << " tokens, cohort: " << node1_lookup.cohort_id << ")\n";
    std::cout << "  Decentralized Cohort Prefix Sync PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 4: Network-Oblivious D-BSP Hierarchical Token Dispatch (Bilardi et al. JACM 2016)
    // -------------------------------------------------------------------------------------
    std::cout << "[4] Testing Network-Oblivious D-BSP Hierarchical Token Dispatch...\n";
    std::vector<int> dbsp_batch = {0, 10, 45, 200};
    auto plan = router.partition_expert_batch_dbsp(dbsp_batch, 0 /* local_socket */, 0 /* local_numa */, 0 /* local_ccx */);

    if (plan.tier_0_intra_ccx.size() != 1 || plan.tier_0_intra_ccx[0].expert_id != 0) {
        std::cerr << "FAIL: Tier 0 Intra-CCX routing mismatch!\n";
        return 1;
    }
    if (plan.tier_1_intra_socket.size() != 1 || plan.tier_1_intra_socket[0].expert_id != 10) {
        std::cerr << "FAIL: Tier 1 Intra-Socket routing mismatch!\n";
        return 1;
    }
    if (plan.tier_2_inter_socket.size() != 1 || plan.tier_2_inter_socket[0].expert_id != 45) {
        std::cerr << "FAIL: Tier 2 Inter-Socket routing mismatch!\n";
        return 1;
    }
    if (plan.tier_3_inter_node.size() != 1 || plan.tier_3_inter_node[0].expert_id != 200) {
        std::cerr << "FAIL: Tier 3 Inter-Node routing mismatch!\n";
        return 1;
    }

    std::cout << "  D-BSP Sub-Superstep Plan (total tokens: " << plan.total_tokens() << "):\n";
    std::cout << "    Tier 0 (Intra-CCX):      " << plan.tier_0_intra_ccx.size() << " tokens (expert " << plan.tier_0_intra_ccx[0].expert_id << ")\n";
    std::cout << "    Tier 1 (Intra-Socket):   " << plan.tier_1_intra_socket.size() << " tokens (expert " << plan.tier_1_intra_socket[0].expert_id << ")\n";
    std::cout << "    Tier 2 (Inter-Socket):   " << plan.tier_2_inter_socket.size() << " tokens (expert " << plan.tier_2_inter_socket[0].expert_id << ")\n";
    std::cout << "    Tier 3 (Inter-Node):     " << plan.tier_3_inter_node.size() << " tokens (expert " << plan.tier_3_inter_node[0].expert_id << ")\n";
    std::cout << "  Network-Oblivious D-BSP Hierarchical Token Dispatch PASSED OK.\n\n";

    std::cout << "=================================================================\n";
    std::cout << "  ALL DISTRIBUTED CLUSTER SCALE-OUT TESTS PASSED (100% SUCCESS)  \n";
    std::cout << "=================================================================\n";
    return 0;
}
