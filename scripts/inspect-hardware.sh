#!/usr/bin/env bash
#
# scripts/inspect-hardware.sh - Hardware Inspection and NUMA Topology Synthesizer for ik_llama.cpp
#
# Probes CPU packages, physical cores, SMT threads, NUMA memory, PCIe GPUs,
# NVMe root complexes, MD RAID arrays, ZFS pools, and synthesizes recommended
# ik_llama.cpp runtime flags (--numa split, -sm graph, -ot, draft/prefetch node).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/inspect-hardware.py"

FORCE_FALLBACK=0
if [ "${1:-}" = "--fallback" ]; then
    FORCE_FALLBACK=1
    shift
fi

# Delegate to Python 3 probe if available and not explicitly disabled
if [ "$FORCE_FALLBACK" -eq 0 ] && command -v python3 >/dev/null 2>&1; then
    exec python3 "$PYTHON_SCRIPT" "$@"
fi

# --- Native Bash / Sysfs / Procfs Fallback Inspection ---

echo "================================================================================"
echo "          ik_llama.cpp Hardware Topology & Architecture Synthesis Report        "
echo "================================================================================"
echo " [!] PRE-ALPHA ARCHITECTURE NOTICE: NUMA split & low-latency streaming are experimental."
echo " [!] Running native sysfs/procfs shell inspection fallback."
echo ""

# 1. CPU & NUMA Nodes
echo "[1] CPU & NUMA Topology"
MODEL_NAME=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^[ \t]*//' || echo "Unknown CPU")
VENDOR_ID=$(grep -m1 "vendor_id" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^[ \t]*//' || echo "Unknown")
echo "  • Processor Model     : $MODEL_NAME ($VENDOR_ID)"

if [ -d /sys/devices/system/cpu ]; then
    TOTAL_CPUS=$(find /sys/devices/system/cpu/ -maxdepth 1 -name 'cpu[0-9]*' | wc -l)
    SOCKET_COUNT=$(cat /sys/devices/system/cpu/cpu*/topology/physical_package_id 2>/dev/null | sort -u | wc -l)
    [ "$SOCKET_COUNT" -eq 0 ] && SOCKET_COUNT=1
    PHYSICAL_CORES=$(paste -d: <(cat /sys/devices/system/cpu/cpu[0-9]*/topology/physical_package_id) <(cat /sys/devices/system/cpu/cpu[0-9]*/topology/core_id) | sort -u | wc -l)
    [ "$PHYSICAL_CORES" -eq 0 ] && PHYSICAL_CORES=$TOTAL_CPUS
    SMT_THREADS=$(( TOTAL_CPUS / PHYSICAL_CORES ))
    [ "$SMT_THREADS" -eq 0 ] && SMT_THREADS=1

    echo "  • Physical Sockets    : $SOCKET_COUNT"
    echo "  • Physical Cores      : $PHYSICAL_CORES"
    echo "  • Logical Threads     : $TOTAL_CPUS ($SMT_THREADS SMT threads/core)"
fi

NODE_COUNT=1
if [ -d /sys/devices/system/node ]; then
    NODE_COUNT=$(find /sys/devices/system/node/ -maxdepth 1 -name 'node[0-9]*' | wc -l)
    echo "  • Total NUMA Nodes    : $NODE_COUNT"
    for node in /sys/devices/system/node/node[0-9]*; do
        [ -e "$node" ] || continue
        NODE_NAME=$(basename "$node")
        CPUS=$(cat "$node/cpulist" 2>/dev/null || echo "unknown")
        MEM=$(grep MemTotal "$node/meminfo" 2>/dev/null | awk '{printf "%.1f GiB", $4/1048576}' || echo "unknown")
        FREE=$(grep MemFree "$node/meminfo" 2>/dev/null | awk '{printf "%.1f GiB", $4/1048576}' || echo "unknown")
        echo "    - $NODE_NAME: $MEM total, $FREE free | CPUs: $CPUS"
    done
else
    echo "  • Flat UMA system (no /sys/devices/system/node)."
fi
echo ""

# 2. Transparent Huge Pages (THP)
echo "[2] Memory & THP Status"
TOTAL_RAM=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{printf "%.1f GiB", $2/1048576}' || echo "unknown")
FREE_RAM=$(grep MemFree /proc/meminfo 2>/dev/null | awk '{printf "%.1f GiB", $2/1048576}' || echo "unknown")
echo "  • Total System RAM    : $TOTAL_RAM ($FREE_RAM free)"
if [ -f /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo "  • THP Mode            : $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
fi
echo ""

# 3. PCIe Topology & GPU Accelerators
echo "[3] PCIe Topology & GPU Accelerators"
FOUND_GPU=0
if [ -d /sys/bus/pci/devices ]; then
    for dev in /sys/bus/pci/devices/*; do
        [ -e "$dev" ] || continue
        CLASS=$(cat "$dev/class" 2>/dev/null || echo "")
        VENDOR=$(cat "$dev/vendor" 2>/dev/null || echo "")
        # Class 0x03* is Display Controller / 3D Accelerator
        if [[ "$CLASS" =~ ^0x03 ]] || [[ "$VENDOR" =~ ^(0x10de|0x1002) && "$CLASS" =~ ^(0x03|0x12) ]]; then
            PCI_ADDR=$(basename "$dev")
            NUMA=$(cat "$dev/numa_node" 2>/dev/null || echo "-1")
            DEVICE_ID=$(cat "$dev/device" 2>/dev/null || echo "")
            echo "  • GPU / Accelerator @ $PCI_ADDR: Vendor $VENDOR Device $DEVICE_ID -> NUMA Node: $NUMA"
            FOUND_GPU=1
        fi
    done
fi
if [ "$FOUND_GPU" -eq 0 ]; then
    echo "  • No discrete GPU accelerators found (CPU-only NUMA execution)."
fi
echo ""

# 4. Storage & Storage Pools
echo "[4] Storage Subsystem & NUMA Root Complex Discovery"
# ZFS Pools
if [ -f /proc/mounts ] && grep -q " zfs " /proc/mounts; then
    echo "  • OpenZFS Storage Pools:"
    grep " zfs " /proc/mounts | awk '{print "    - Pool: " $1 " mounted at " $2}' | sort -u
fi

# MD RAID
if [ -d /sys/block ]; then
    MD_COUNT=0
    for md in /sys/block/md*; do
        [ -e "$md" ] || continue
        MD_NAME=$(basename "$md")
        LEVEL=$(cat "$md/md/level" 2>/dev/null || echo "raid")
        SLAVES=$(ls "$md/slaves" 2>/dev/null | tr '\n' ' ' || echo "none")
        echo "  • Linux Software MD RAID /dev/$MD_NAME: Level $LEVEL | Member Slaves: $SLAVES"
        MD_COUNT=1
    done
fi

# NVMe Drives
for dev in /sys/block/nvme*n*; do
    [ -e "$dev" ] || continue
    DEV_NAME=$(basename "$dev")
    MODEL=$(cat "$dev/device/model" 2>/dev/null || echo "NVMe SSD")
    NUMA=$(cat "$dev/device/numa_node" 2>/dev/null || echo "-1")
    SECTORS=$(cat "$dev/size" 2>/dev/null || echo "0")
    CAP_GIB=$(awk "BEGIN {printf \"%.1f GiB\", $SECTORS * 512 / 1073741824}")
    echo "  • $DEV_NAME ($MODEL, $CAP_GIB) -> NUMA Node: $NUMA"
done
echo "  • Storage Capacity Note : Manufacturers specify capacity in decimal (e.g. 4x 4 TB = 16 TB raw), while OS/ZFS tools report in binary TiB (14.88 TiB = 16.36 TB)."
echo ""

# 5. Recommendation
echo "[5] Recommended ik_llama.cpp Flags"
CMD_FLAGS=()
if [ "$NODE_COUNT" -gt 1 ]; then
    CMD_FLAGS+=("--numa split")
fi
if [ "$FOUND_GPU" -eq 1 ]; then
    CMD_FLAGS+=("-sm graph")
fi
CMD_FLAGS+=("-t 7" "-tb 7" "-thp")

echo "  • Recommended Flags: ${CMD_FLAGS[*]}"
echo ""
echo "Ready-to-Run Server Launch Command:"
echo "./build/bin/llama-server \\"
echo "    -m /path/to/target-model.gguf \\"
if [ "$NODE_COUNT" -gt 1 ]; then
    echo "    -md /path/to/draft-model.gguf \\"
fi
for ((i=0; i<${#CMD_FLAGS[@]}; i++)); do
    if [ "$i" -lt $(( ${#CMD_FLAGS[@]} - 1 )) ]; then
        echo "    ${CMD_FLAGS[$i]} \\"
    else
        echo "    ${CMD_FLAGS[$i]}"
    fi
done
echo ""
echo "================================================================================"
