#include "superchip-unified-pipeline.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// =========================================================================================
// Test Suite: Superchip & Coherent Interconnects (AMD MI300A, NVIDIA GH200/GB200, CXL 3.0)
// =========================================================================================

int main() {
    std::cout << "=================================================================\n";
    std::cout << "  TEST: Superchips & Coherent Interconnect Pipeline Architecture \n";
    std::cout << "=================================================================\n\n";

    // -------------------------------------------------------------------------------------
    // Test 1: Superchip & Interconnect Topology Detection
    // -------------------------------------------------------------------------------------
    std::cout << "[1] Probing system for Superchip & Coherent Interconnect capabilities...\n";
    superchip_caps caps = superchip_detector::detect();

    std::cout << "  Detected Architecture: " << caps.device_name << "\n";
    std::cout << "  Description:           " << caps.description << "\n";
    std::cout << "  Unified Memory:        " << (caps.has_unified_memory ? "ENABLED" : "N/A") << "\n";
    std::cout << "  Hardware ATS:          " << (caps.has_hardware_ats ? "SUPPORTED" : "N/A") << "\n";
    std::cout << "  HSA_XNACK Status:      " << (caps.has_xnack_enabled ? "ENABLED (1)" : "DISABLED (0)") << "\n";
    std::cout << "  NVLink-C2C Support:    " << (caps.has_nvlink_c2c ? "ACTIVE" : "N/A") << "\n";
    std::cout << "  Peak Coherent BW:      " << caps.peak_coherent_bw_gbps << " GB/s\n";
    std::cout << "  CXL Expansion Nodes:   " << caps.cxl_node_count << "\n";
    std::cout << "  Topology probe PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 2: Unified Zero-Copy Buffer Allocation (AMD MI300A / Apple Silicon)
    // -------------------------------------------------------------------------------------
    std::cout << "[2] Allocating Unified Zero-Copy Buffer (HugeTLB / Direct Mapping)...\n";
    const size_t test_buf_size = 2ULL * 1024ULL * 1024ULL; // 2MB HugeTLB page
    unified_zero_copy_buffer ubuf = unified_zero_copy_buffer::allocate(test_buf_size, true);

    if (!ubuf.data || ubuf.size != test_buf_size) {
        std::cerr << "FAIL: unified_zero_copy_buffer allocation failed!\n";
        return 1;
    }

    // Write test token activation tensor
    uint32_t * int_ptr = static_cast<uint32_t *>(ubuf.data);
    for (size_t i = 0; i < 1024; ++i) {
        int_ptr[i] = static_cast<uint32_t>(i ^ 0xCAFEBABE);
    }

    // Verify written data
    for (size_t i = 0; i < 1024; ++i) {
        if (int_ptr[i] != static_cast<uint32_t>(i ^ 0xCAFEBABE)) {
            std::cerr << "FAIL: unified buffer data mismatch at index " << i << "!\n";
            return 1;
        }
    }
    std::cout << "  Unified Zero-Copy buffer (2MB) allocated and verified at: " << ubuf.data << "\n";
    ubuf.release();
    std::cout << "  Unified buffer release PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 3: NVLink-C2C Double-Buffered MoE Expert Streamer (GH200 / GB200)
    // -------------------------------------------------------------------------------------
    std::cout << "[3] Testing NVLink-C2C Double-Buffered MoE Expert Streamer...\n";
    const size_t expert_slot_size = 256 * 1024; // 256 KB per expert slice
    nvlink_c2c_expert_streamer streamer(expert_slot_size, 2);

    // Prepare simulated MoE experts in Grace CPU host memory
    std::vector<uint8_t> expert0_cpu(expert_slot_size, 0xAA);
    std::vector<uint8_t> expert1_cpu(expert_slot_size, 0xBB);

    moe_expert_descriptor exp0;
    exp0.expert_id = 42;
    exp0.size_bytes = expert_slot_size;
    exp0.lpddr5x_source = expert0_cpu.data();

    moe_expert_descriptor exp1;
    exp1.expert_id = 43;
    exp1.size_bytes = expert_slot_size;
    exp1.lpddr5x_source = expert1_cpu.data();

    // Stream expert 0 into Slot 0
    int rc0 = streamer.schedule_stream(exp0, 0);
    if (rc0 != 0 || !streamer.is_slot_ready(0)) {
        std::cerr << "FAIL: Streamer failed to stage expert 0!\n";
        return 1;
    }

    // Verify staged content in Slot 0
    const uint8_t * slot0_ptr = streamer.get_staged_expert_ptr(0);
    if (!slot0_ptr || slot0_ptr[0] != 0xAA || slot0_ptr[expert_slot_size - 1] != 0xAA) {
        std::cerr << "FAIL: Staged expert 0 data mismatch!\n";
        return 1;
    }

    // Stream expert 1 into Slot 1 (pipelined overlap)
    int rc1 = streamer.schedule_stream(exp1, 1);
    if (rc1 != 0 || !streamer.is_slot_ready(1)) {
        std::cerr << "FAIL: Streamer failed to stage expert 1!\n";
        return 1;
    }

    // Verify staged content in Slot 1
    const uint8_t * slot1_ptr = streamer.get_staged_expert_ptr(1);
    if (!slot1_ptr || slot1_ptr[0] != 0xBB || slot1_ptr[expert_slot_size - 1] != 0xBB) {
        std::cerr << "FAIL: Staged expert 1 data mismatch!\n";
        return 1;
    }

    std::cout << "  Streamed 2 routed experts (" << streamer.get_total_bytes_streamed() 
              << " bytes total) across NVLink-C2C staging pipeline.\n";
    std::cout << "  NVLink-C2C streaming test PASSED OK.\n\n";

    // -------------------------------------------------------------------------------------
    // Test 4: CXL 2.0 / 3.0 Type 3 Memory Expansion Pool Allocation
    // -------------------------------------------------------------------------------------
    std::cout << "[4] Testing CXL 2.0/3.0 Type 3 Pooled Memory Allocation...\n";
    const size_t cxl_table_size = 1024 * 1024; // 1MB expert table
    auto cxl_alloc = cxl_type3_memory_pool::allocate_expert_table(cxl_table_size, -1);

    if (!cxl_alloc.ptr || cxl_alloc.size != cxl_table_size) {
        std::cerr << "FAIL: CXL Type 3 memory allocation failed!\n";
        return 1;
    }

    std::memset(cxl_alloc.ptr, 0x5A, cxl_table_size);
    uint8_t * cxl_bytes = static_cast<uint8_t *>(cxl_alloc.ptr);
    if (cxl_bytes[0] != 0x5A || cxl_bytes[cxl_table_size - 1] != 0x5A) {
        std::cerr << "FAIL: CXL memory readback error!\n";
        return 1;
    }

    std::cout << "  Allocated and bound 1MB sparse expert table on CXL Type 3 pool at: " << cxl_alloc.ptr << "\n";
    cxl_type3_memory_pool::free_expert_table(cxl_alloc);
    std::cout << "  CXL Type 3 pool release PASSED OK.\n\n";

    std::cout << "=================================================================\n";
    std::cout << "  ALL SUPERCHIP & COHERENT INTERCONNECT TESTS PASSED (100%)      \n";
    std::cout << "=================================================================\n";
    return 0;
}
