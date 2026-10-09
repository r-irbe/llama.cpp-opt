#!/usr/bin/env python3
"""
scripts/inspect-hardware.py - Comprehensive Hardware Inspection and NUMA Topology Synthesizer for ik_llama.cpp

Probes:
  - CPU packages, physical sockets, physical cores, SMT threads, NUMA node assignments
  - System RAM per NUMA node, free memory, and Transparent Huge Pages (THP) status
  - PCIe topology and GPU NUMA root complex association
  - Block devices: NVMe drives, PCI root complexes, drive models, capacities, MD RAID arrays (/slaves/), and ZFS storage pools
  - Inter-socket interconnect bandwidth and draft model recommendations
  - Synthesizes optimal ik_llama.cpp runtime flags (--numa split, -sm graph, -ot, prefetch node, draft node)

Zero external dependencies - uses standard Linux /sys, /proc, /etc files, and optional zpool CLI.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# --- ANSI Formatting Utilities ---
class Colors:
    def __init__(self, enabled=True):
        if enabled and sys.stdout.isatty():
            self.RESET = "\033[0m"
            self.BOLD = "\033[1m"
            self.DIM = "\033[2m"
            self.RED = "\033[31m"
            self.GREEN = "\033[32m"
            self.YELLOW = "\033[33m"
            self.BLUE = "\033[34m"
            self.MAGENTA = "\033[35m"
            self.CYAN = "\033[36m"
            self.WHITE = "\033[37m"
            self.BG_BLUE = "\033[44m"
        else:
            self.RESET = ""
            self.BOLD = ""
            self.DIM = ""
            self.RED = ""
            self.GREEN = ""
            self.YELLOW = ""
            self.BLUE = ""
            self.MAGENTA = ""
            self.CYAN = ""
            self.WHITE = ""
            self.BG_BLUE = ""

def read_file(path):
    """Safely read string from file, return empty string on failure."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except Exception:
        return ""

def read_int(path, default=-1):
    """Safely read integer from file."""
    s = read_file(path)
    try:
        return int(s)
    except Exception:
        return default

def format_bytes(n_bytes, dual=True):
    """Format bytes into human-readable string (GiB/TiB with decimal TB)."""
    if n_bytes <= 0:
        return "0 B"
    tb_bin = n_bytes / (1024 ** 4)
    tb_dec = n_bytes / (1000 ** 4)
    gb_bin = n_bytes / (1024 ** 3)
    gb_dec = n_bytes / (1000 ** 3)
    mb_bin = n_bytes / (1024 ** 2)
    if tb_bin >= 1.0:
        if dual and abs(tb_bin - tb_dec) >= 0.1:
            return f"{tb_bin:.2f} TiB ({tb_dec:.2f} TB)"
        return f"{tb_bin:.2f} TiB"
    if gb_bin >= 1.0:
        if dual and abs(gb_bin - gb_dec) >= 2.0:
            return f"{gb_bin:.1f} GiB ({gb_dec:.1f} GB)"
        return f"{gb_bin:.1f} GiB"
    return f"{mb_bin:.0f} MiB"

def parse_cpu_list(cpulist_str):
    """Parse Linux CPU list syntax (e.g. '0-3,7,10-12') into sorted list of ints."""
    cpus = []
    if not cpulist_str:
        return cpus
    for part in cpulist_str.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = part.split("-")
                cpus.extend(range(int(start), int(end) + 1))
            except Exception:
                pass
        else:
            try:
                cpus.append(int(part))
            except Exception:
                pass
    return sorted(cpus)

# --- Hardware Probing ---

def probe_cpu_topology(sysfs_root="/sys"):
    """Probe CPU packages, sockets, physical cores, threads, and NUMA node mappings."""
    cpu_dir = os.path.join(sysfs_root, "devices/system/cpu")
    node_dir = os.path.join(sysfs_root, "devices/system/node")
    proc_cpuinfo = os.path.join(sysfs_root, "proc/cpuinfo") if os.path.exists(os.path.join(sysfs_root, "proc/cpuinfo")) else "/proc/cpuinfo"
    
    # 1. Parse /proc/cpuinfo for model and microarchitecture
    model_name = "Unknown CPU"
    vendor_id = "Unknown"
    cpu_family = -1
    cpu_model = -1
    flags = []
    try:
        with open(proc_cpuinfo, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if ":" in line:
                    k, v = [x.strip() for x in line.split(":", 1)]
                    if k == "model name" and model_name == "Unknown CPU":
                        model_name = v
                    elif k == "vendor_id" and vendor_id == "Unknown":
                        vendor_id = v
                    elif k == "cpu family" and cpu_family == -1:
                        try: cpu_family = int(v)
                        except Exception: pass
                    elif k == "model" and cpu_model == -1:
                        try: cpu_model = int(v)
                        except Exception: pass
                    elif k == "flags" and not flags:
                        flags = v.split()
    except Exception:
        pass

    # Microarchitecture identification
    uarch = "Generic x86-64"
    if vendor_id == "AuthenticAMD":
        if cpu_family == 23: # Zen 1 / Zen 2
            uarch = "AMD Zen 1 / Zen 2"
        elif cpu_family == 25: # Zen 3 / Zen 4
            if cpu_model >= 0x10 and cpu_model <= 0x1f:
                uarch = "AMD Zen 4 (Genoa / Bergamo)"
            else:
                uarch = "AMD Zen 3 (Milan / Chagall)"
        elif cpu_family == 26:
            uarch = "AMD Zen 5 (Turin)"
        else:
            uarch = f"AMD Family {cpu_family}"
    elif vendor_id == "GenuineIntel":
        if "avx512_bf16" in flags or "amx_tile" in flags:
            uarch = "Intel Sapphire Rapids / Emerald Rapids / Granite Rapids"
        elif "avx512f" in flags:
            uarch = "Intel Skylake-SP / Cascade Lake / Ice Lake"
        else:
            uarch = "Intel Core / Xeon"

    # 2. Enumerate CPUs
    online_cpus = parse_cpu_list(read_file(os.path.join(cpu_dir, "online")))
    if not online_cpus:
        online_cpus = []
        for p in glob.glob(os.path.join(cpu_dir, "cpu[0-9]*")):
            b = os.path.basename(p)
            if b.startswith("cpu") and b[3:].isdigit():
                online_cpus.append(int(b[3:]))
        online_cpus.sort()

    packages = set()
    core_map = {} # (package_id, core_id) -> list of cpu_ids
    cpu_to_pkg = {}
    cpu_to_core = {}

    for cpu_id in online_cpus:
        topo_path = os.path.join(cpu_dir, f"cpu{cpu_id}/topology")
        pkg_id = read_int(os.path.join(topo_path, "physical_package_id"), 0)
        core_id = read_int(os.path.join(topo_path, "core_id"), cpu_id)
        packages.add(pkg_id)
        cpu_to_pkg[cpu_id] = pkg_id
        cpu_to_core[cpu_id] = (pkg_id, core_id)
        core_key = (pkg_id, core_id)
        if core_key not in core_map:
            core_map[core_key] = []
        core_map[core_key].append(cpu_id)

    num_sockets = max(len(packages), 1)
    total_physical_cores = len(core_map)
    total_logical_cpus = len(online_cpus)
    smt_threads_per_core = total_logical_cpus // total_physical_cores if total_physical_cores > 0 else 1

    # 3. NUMA Nodes
    numa_nodes = []
    node_paths = sorted(glob.glob(os.path.join(node_dir, "node[0-9]*")))
    if not node_paths:
        numa_nodes.append({
            "id": 0,
            "socket_id": 0,
            "cpus": online_cpus,
            "core_count": total_physical_cores,
            "total_bytes": 0,
            "free_bytes": 0,
        })
    else:
        for np in node_paths:
            node_name = os.path.basename(np)
            try:
                node_id = int(node_name.replace("node", ""))
            except Exception:
                continue
            cpulist_str = read_file(os.path.join(np, "cpulist"))
            node_cpus = parse_cpu_list(cpulist_str)
            node_pkg = 0
            if node_cpus:
                node_pkg = cpu_to_pkg.get(node_cpus[0], 0)
            node_cores = set(cpu_to_core.get(c) for c in node_cpus if c in cpu_to_core)

            # Parse node meminfo
            meminfo_content = read_file(os.path.join(np, "meminfo"))
            tot_bytes = 0
            free_bytes = 0
            for line in meminfo_content.splitlines():
                if "MemTotal:" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        tot_bytes = int(parts[3]) * 1024
                elif "MemFree:" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        free_bytes = int(parts[3]) * 1024

            is_cxl = (len(node_cpus) == 0 and tot_bytes > 0)
            numa_nodes.append({
                "id": node_id,
                "socket_id": node_pkg,
                "cpus": node_cpus,
                "core_count": len(node_cores),
                "total_bytes": tot_bytes,
                "free_bytes": free_bytes,
                "is_cxl": is_cxl,
            })

    total_numa_nodes = len(numa_nodes)
    cxl_node_ids = [n["id"] for n in numa_nodes if n.get("is_cxl")]
    nodes_per_socket = total_numa_nodes // num_sockets if num_sockets > 0 else 1
    nps_mode = f"NPS{nodes_per_socket}" if total_numa_nodes > 1 else "UMA / NPS0"

    return {
        "model_name": model_name,
        "vendor_id": vendor_id,
        "uarch": uarch,
        "flags": flags,
        "num_sockets": num_sockets,
        "total_physical_cores": total_physical_cores,
        "cores_per_socket": total_physical_cores // num_sockets if num_sockets > 0 else total_physical_cores,
        "total_logical_cpus": total_logical_cpus,
        "smt_threads_per_core": smt_threads_per_core,
        "numa_nodes": numa_nodes,
        "cxl_node_ids": cxl_node_ids,
        "total_numa_nodes": total_numa_nodes,
        "nodes_per_socket": nodes_per_socket,
        "nps_mode": nps_mode,
    }

def probe_memory_and_thp(sysfs_root="/sys"):
    """Probe global RAM capacity, per-node distribution, and Transparent Huge Pages (THP)."""
    mem_total = 0
    mem_free = 0
    mem_available = 0
    proc_meminfo = os.path.join(sysfs_root, "proc/meminfo") if os.path.exists(os.path.join(sysfs_root, "proc/meminfo")) else "/proc/meminfo"
    try:
        with open(proc_meminfo, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = parts[1].strip().split()[0]
                    if k == "MemTotal":
                        mem_total = int(v) * 1024
                    elif k == "MemFree":
                        mem_free = int(v) * 1024
                    elif k == "MemAvailable":
                        mem_available = int(v) * 1024
    except Exception:
        pass

    thp_enabled_raw = read_file(os.path.join(sysfs_root, "kernel/mm/transparent_hugepage/enabled"))
    thp_defrag_raw = read_file(os.path.join(sysfs_root, "kernel/mm/transparent_hugepage/defrag"))
    thp_shmem_raw = read_file(os.path.join(sysfs_root, "kernel/mm/transparent_hugepage/shmem_enabled"))

    def extract_active(s):
        m = re.search(r"\[([a-zA-Z]+)\]", s)
        return m.group(1) if m else s.strip()

    thp_enabled = extract_active(thp_enabled_raw) or "unknown"
    thp_defrag = extract_active(thp_defrag_raw) or "unknown"
    thp_shmem = extract_active(thp_shmem_raw) or "unknown"

    if thp_enabled in ["always", "madvise"]:
        thp_recommendation = "Optimal (THP is active and available for mmap/madvise 2MB huge pages)"
    else:
        thp_recommendation = "Suboptimal (THP is 'never'; enable via 'echo madvise > /sys/kernel/mm/transparent_hugepage/enabled')"

    return {
        "total_bytes": mem_total,
        "free_bytes": mem_free,
        "available_bytes": mem_available,
        "thp_enabled": thp_enabled,
        "thp_defrag": thp_defrag,
        "thp_shmem": thp_shmem,
        "thp_recommendation": thp_recommendation,
    }

def probe_pci_gpus(sysfs_root="/sys"):
    """Probe PCIe bus devices for display and compute GPU accelerators and their NUMA nodes."""
    devices = []
    pci_dir = os.path.join(sysfs_root, "bus/pci/devices")
    
    known_devices = {
        ("0x10de", "0x2684"): "NVIDIA GeForce RTX 4090 (AD102)",
        ("0x10de", "0x2685"): "NVIDIA RTX 4090 D",
        ("0x10de", "0x2704"): "NVIDIA GeForce RTX 4080 (AD103)",
        ("0x10de", "0x2204"): "NVIDIA GeForce RTX 3090 (GA102)",
        ("0x10de", "0x20b0"): "NVIDIA A100-SXM4-40GB / 80GB",
        ("0x10de", "0x20f1"): "NVIDIA A100 PCIe 40GB / 80GB",
        ("0x10de", "0x2330"): "NVIDIA H100 PCIe 80GB",
        ("0x10de", "0x2322"): "NVIDIA H100 SXM5",
        ("0x1a03", "0x2000"): "ASPEED AST2500 BMC Graphics",
        ("0x1002", "0x7408"): "AMD Radeon RX 7900 XTX (Navi 31)",
        ("0x1002", "0x740c"): "AMD Instinct MI300A APU (128GB HBM3 @ 5.3 TB/s)",
        ("0x1002", "0x7448"): "AMD Instinct MI300X OAM",
        ("0x10de", "0x2340"): "NVIDIA Grace Hopper (GH200 NVLink-C2C 900 GB/s)",
        ("0x10de", "0x2341"): "NVIDIA Grace Hopper (GH200 NVLink-C2C 900 GB/s)",
        ("0x10de", "0x2941"): "NVIDIA Grace Blackwell (GB200 NVLink-C2C 1,800 GB/s)",
    }

    for dev_path in sorted(glob.glob(os.path.join(pci_dir, "*"))):
        pci_addr = os.path.basename(dev_path)
        class_str = read_file(os.path.join(dev_path, "class"))
        vendor_id = read_file(os.path.join(dev_path, "vendor")).lower()
        device_id = read_file(os.path.join(dev_path, "device")).lower()
        numa_node = read_int(os.path.join(dev_path, "numa_node"), -1)

        is_gpu = False
        if class_str.startswith("0x03"):
            is_gpu = True
        elif vendor_id in ["0x10de", "0x1002"]:
            if class_str.startswith("0x03") or class_str.startswith("0x12"):
                is_gpu = True

        if not is_gpu:
            continue

        is_bmc = (vendor_id == "0x1a03")

        max_speed = read_file(os.path.join(dev_path, "max_link_speed"))
        cur_speed = read_file(os.path.join(dev_path, "current_link_speed"))
        max_width = read_file(os.path.join(dev_path, "max_link_width"))
        cur_width = read_file(os.path.join(dev_path, "current_link_width"))

        vendor_name = "NVIDIA" if vendor_id == "0x10de" else ("AMD" if vendor_id == "0x1002" else ("ASPEED" if vendor_id == "0x1a03" else vendor_id))
        dev_name = known_devices.get((vendor_id, device_id), f"{vendor_name} Device [{device_id}]")

        devices.append({
            "pci_address": pci_addr,
            "vendor_id": vendor_id,
            "device_id": device_id,
            "name": dev_name,
            "numa_node": numa_node,
            "is_bmc": is_bmc,
            "is_compute_gpu": not is_bmc,
            "pcie_link": f"{cur_speed} {cur_width} (Max: {max_speed} {max_width})",
        })

    return devices

def probe_block_devices_and_storage(sysfs_root="/sys"):
    """Probe NVMe drives, MD RAID arrays, and ZFS storage pools for NUMA locality."""
    block_dir = os.path.join(sysfs_root, "block")
    
    nvme_drives = []
    md_arrays = []
    zfs_pools = []

    # 1. Inspect direct NVMe drives
    for dev_path in sorted(glob.glob(os.path.join(block_dir, "nvme*n*"))):
        dev_name = os.path.basename(dev_path)
        size_sectors = read_int(os.path.join(dev_path, "size"), 0)
        size_bytes = size_sectors * 512
        model = read_file(os.path.join(dev_path, "device/model"))
        if not model:
            model = read_file(os.path.join(dev_path, "device/device/model"))
        
        numa_node = -1
        pci_address = ""
        pci_addresses = []
        try:
            real_path = os.path.realpath(dev_path)
            pci_matches = re.findall(r"([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])", real_path)
            if pci_matches:
                pci_addresses = pci_matches
                pci_address = pci_matches[-1] # Endpoint controller address
            
            cur = Path(real_path)
            while cur != cur.parent:
                numa_f = cur / "numa_node"
                if numa_f.exists():
                    n = read_int(str(numa_f), -1)
                    if n >= 0:
                        numa_node = n
                        break
                cur = cur.parent
        except Exception:
            pass

        nvme_drives.append({
            "device": dev_name,
            "model": model or "NVMe SSD",
            "size_bytes": size_bytes,
            "size_str": format_bytes(size_bytes, dual=True),
            "pci_address": pci_address,
            "pci_addresses": pci_addresses,
            "numa_node": numa_node,
        })

    # 2. Inspect Linux MD RAID arrays (/sys/block/md*)
    for md_path in sorted(glob.glob(os.path.join(block_dir, "md*"))):
        md_name = os.path.basename(md_path)
        size_sectors = read_int(os.path.join(md_path, "size"), 0)
        size_bytes = size_sectors * 512
        level = read_file(os.path.join(md_path, "md/level")) or "raid"
        
        slaves = []
        member_nodes = set()
        slaves_dir = os.path.join(md_path, "slaves")
        if os.path.exists(slaves_dir):
            for s in sorted(os.listdir(slaves_dir)):
                slaves.append(s)
                for d in nvme_drives:
                    if d["device"] == s and d["numa_node"] >= 0:
                        member_nodes.add(d["numa_node"])

        resolved_numa = list(member_nodes)[0] if len(member_nodes) == 1 else (-1 if not member_nodes else list(member_nodes))

        md_arrays.append({
            "device": md_name,
            "level": level,
            "size_bytes": size_bytes,
            "size_str": format_bytes(size_bytes, dual=True),
            "slaves": slaves,
            "numa_node": resolved_numa,
        })

    # 3. Inspect OpenZFS Storage Pools
    proc_mounts_file = os.path.join(sysfs_root, "proc/mounts") if os.path.exists(os.path.join(sysfs_root, "proc/mounts")) else "/proc/mounts"
    zfs_mounts = {}
    try:
        with open(proc_mounts_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[2] == "zfs":
                    source = parts[0]
                    mount_point = parts[1]
                    pool_name = source.split("/")[0]
                    if pool_name not in zfs_mounts:
                        zfs_mounts[pool_name] = []
                    zfs_mounts[pool_name].append(mount_point)
    except Exception:
        pass

    # Check for zpool CLI binary
    zpool_bin = None
    for cand in ["/sbin/zpool", "/usr/sbin/zpool", "zpool"]:
        if shutil.which(cand) or (os.path.isabs(cand) and os.access(cand, os.X_OK)):
            zpool_bin = cand
            break

    zpool_cli_info = {}
    if zpool_bin:
        try:
            res = subprocess.run([zpool_bin, "list", "-Hp", "-o", "name,size,alloc,free"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            for line in res.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) >= 4:
                    zpool_cli_info[parts[0]] = {
                        "size": int(parts[1]),
                        "alloc": int(parts[2]),
                        "free": int(parts[3]),
                        "devices": [],
                    }
            for pool in (set(zfs_mounts.keys()) | set(zpool_cli_info.keys())):
                res_st = subprocess.run([zpool_bin, "status", "-P", pool], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                in_config = False
                for line in res_st.stdout.splitlines():
                    if "config:" in line:
                        in_config = True
                        continue
                    if in_config:
                        parts = line.split()
                        if parts and parts[0].startswith("/dev/"):
                            dev_path = parts[0]
                            real = os.path.realpath(dev_path)
                            bname = os.path.basename(real)
                            m = re.match(r"^(nvme\d+n\d+)p\d+$", bname)
                            if m:
                                bname = m.group(1)
                            else:
                                m2 = re.match(r"^([a-z]+)\d+$", bname)
                                if m2:
                                    bname = m2.group(1)
                            if pool not in zpool_cli_info:
                                zpool_cli_info[pool] = {"size": 0, "alloc": 0, "free": 0, "devices": []}
                            if bname not in zpool_cli_info[pool]["devices"]:
                                zpool_cli_info[pool]["devices"].append(bname)
        except Exception:
            pass

    zfs_cache_file = os.path.join(sysfs_root, "etc/zfs/zpool.cache") if os.path.exists(os.path.join(sysfs_root, "etc/zfs/zpool.cache")) else "/etc/zfs/zpool.cache"
    all_pools = sorted(list(set(zfs_mounts.keys()) | set(zpool_cli_info.keys())))
    
    for pool in all_pools:
        mounts = zfs_mounts.get(pool, [])
        member_devices = []
        pool_numa = -1
        total_pool_bytes = 0

        if pool in zpool_cli_info and zpool_cli_info[pool]["devices"]:
            cli_info = zpool_cli_info[pool]
            member_devices = cli_info["devices"]
            total_pool_bytes = cli_info["size"]
            numa_candidates = []
            for mdev in member_devices:
                for d in nvme_drives:
                    if d["device"] == mdev and d["numa_node"] >= 0:
                        numa_candidates.append(d["numa_node"])
                b_numa = read_int(os.path.join(block_dir, f"{mdev}/device/numa_node"), -1)
                if b_numa >= 0:
                    numa_candidates.append(b_numa)
            if numa_candidates:
                pool_numa = max(set(numa_candidates), key=numa_candidates.count)
        else:
            # Parse from /etc/zfs/zpool.cache partition-by-pool
            if os.path.exists(zfs_cache_file):
                try:
                    with open(zfs_cache_file, "rb") as f:
                        c_data = f.read().decode("latin1", errors="ignore")
                    p_pos = c_data.find(pool)
                    if p_pos != -1:
                        next_pos = c_data.find("pool_guid", p_pos + len(pool) + 16)
                        limit = (next_pos + 4096) if next_pos != -1 else len(c_data)
                        pool_section = c_data[p_pos:limit]
                        pcis = re.findall(r"pci-([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])", pool_section)
                        dev_refs = re.findall(r"/dev/([a-zA-Z0-9_\-\/]+)", pool_section)
                        
                        for pci in set(pcis):
                            for d in nvme_drives:
                                if pci in d.get("pci_addresses", []) or pci == d["pci_address"]:
                                    if d["device"] not in member_devices:
                                        member_devices.append(d["device"])
                                        total_pool_bytes += d["size_bytes"]
                                        if d["numa_node"] >= 0 and pool_numa < 0:
                                            pool_numa = d["numa_node"]

                        for dref in set(dev_refs):
                            real = os.path.realpath("/dev/" + dref)
                            bname = os.path.basename(real)
                            m = re.match(r"^(nvme\d+n\d+)p\d+$", bname)
                            if m:
                                bname = m.group(1)
                            else:
                                m2 = re.match(r"^([a-z]+)\d+$", bname)
                                if m2:
                                    bname = m2.group(1)
                            if bname not in member_devices:
                                member_devices.append(bname)
                                for d in nvme_drives:
                                    if d["device"] == bname:
                                        total_pool_bytes += d["size_bytes"]
                                        if d["numa_node"] >= 0 and pool_numa < 0:
                                            pool_numa = d["numa_node"]
                                if pool_numa < 0:
                                    b_numa = read_int(os.path.join(block_dir, f"{bname}/device/numa_node"), -1)
                                    if b_numa >= 0:
                                        pool_numa = b_numa
                except Exception:
                    pass

        if pool_numa < 0:
            pool_numa = 0

        cap_str = format_bytes(total_pool_bytes, dual=True)

        zfs_pools.append({
            "pool_name": pool,
            "mount_points": mounts,
            "member_drives": member_devices,
            "raw_capacity_bytes": total_pool_bytes,
            "raw_capacity_str": cap_str,
            "numa_node": pool_numa,
        })

    # Optimal prefetch node: discover NUMA node with largest aggregate NVMe storage capacity
    optimal_prefetch_node = 0
    nvme_node_caps = {}
    for d in nvme_drives:
        if d["numa_node"] >= 0:
            nvme_node_caps[d["numa_node"]] = nvme_node_caps.get(d["numa_node"], 0) + d["size_bytes"]

    if nvme_node_caps:
        optimal_prefetch_node = max(nvme_node_caps.items(), key=lambda x: x[1])[0]
    elif zfs_pools and zfs_pools[0]["numa_node"] >= 0:
        optimal_prefetch_node = zfs_pools[0]["numa_node"]
    elif md_arrays and isinstance(md_arrays[0]["numa_node"], int) and md_arrays[0]["numa_node"] >= 0:
        optimal_prefetch_node = md_arrays[0]["numa_node"]

    return {
        "nvme_drives": nvme_drives,
        "md_arrays": md_arrays,
        "zfs_pools": zfs_pools,
        "optimal_prefetch_node": optimal_prefetch_node,
    }

def synthesize_speculative_and_interconnect(cpu_info, pci_gpus):
    """Synthesize inter-socket interconnect characteristics and draft model affinity."""
    num_sockets = cpu_info["num_sockets"]
    total_nodes = cpu_info["total_numa_nodes"]
    vendor = cpu_info["vendor_id"]

    if num_sockets >= 2:
        if vendor == "AuthenticAMD":
            interconnect_name = "AMD Infinity Fabric (xGMI3)"
            interconnect_bw = "64 - 128 GB/s bidirectional (3-4 links @ 16 GT/s)"
        else:
            interconnect_name = "Intel Ultra Path Interconnect (UPI)"
            interconnect_bw = "64 - 128 GB/s bidirectional (3-4 UPI links @ 16 GT/s)"
    elif total_nodes > 1:
        interconnect_name = "On-Package Die Interconnect (IFOP / Sub-NUMA Mesh)"
        interconnect_bw = "> 200 GB/s on-die crossbar"
    else:
        interconnect_name = "Monolithic UMA Bus (Single Memory Controller Domain)"
        interconnect_bw = "Local Memory Channels"

    target_node = 0

    draft_node = -1
    draft_reason = ""
    if total_nodes <= 1:
        draft_node = -1
        draft_reason = "Single NUMA domain (UMA/NPS0); draft model runs colocated with target model"
    elif num_sockets >= 2:
        target_socket = cpu_info["numa_nodes"][target_node]["socket_id"]
        opposite_nodes = [n["id"] for n in cpu_info["numa_nodes"] if n["socket_id"] != target_socket]
        if opposite_nodes:
            draft_node = opposite_nodes[0]
            draft_reason = f"Cross-socket isolation: Socket {target_socket} -> Socket {cpu_info['numa_nodes'][draft_node]['socket_id']} (Node {draft_node}) prevents L3 cache evictions and memory bus contention"
        else:
            draft_node = (target_node + total_nodes // 2) % total_nodes
            draft_reason = f"Strided NUMA quadrant (Node {draft_node})"
    else:
        draft_node = (target_node + total_nodes // 2) % total_nodes
        draft_reason = f"Opposite NUMA quadrant on single package (Node {draft_node}) maximizes DDR channel parallelism"

    return {
        "interconnect_name": interconnect_name,
        "interconnect_bandwidth": interconnect_bw,
        "target_primary_node": target_node,
        "optimal_draft_node": draft_node,
        "draft_selection_reason": draft_reason,
    }

def synthesize_llama_flags(cpu_info, mem_info, storage_info, pci_gpus, spec_info):
    """Synthesize recommended ik_llama.cpp runtime flags and launch command."""
    flags = []
    
    if cpu_info["total_numa_nodes"] > 1:
        flags.append("--numa split")
    
    has_gpu = any(g["is_compute_gpu"] for g in pci_gpus)
    if has_gpu:
        flags.append("-sm graph")

    if has_gpu and cpu_info["total_numa_nodes"] > 1:
        ot_rule = (
            '^.*\\.attn.*=CUDA0,'
            '.*\\.router.*=CUDA0,'
            '.*\\.ffn_shared.*=CUDA0,'
            '.*\\.nextn\\..*=CPU,'
            '^blk\\..*\\.ffn_.*_exps.*=CPU'
        )
        flags.append(f'-ot "{ot_rule}"')

    cores_per_node = cpu_info["numa_nodes"][0]["core_count"] if cpu_info["numa_nodes"] else 8
    threads_recommendation = max(cores_per_node, 4)
    flags.append(f"-t {threads_recommendation}")
    flags.append(f"-tb {threads_recommendation}")
    flags.append("-thp")

    prefetch_node = storage_info["optimal_prefetch_node"]
    if prefetch_node >= 0 and cpu_info["total_numa_nodes"] > 1:
        prefetch_hint = f"Node {prefetch_node} (Auto-discovered via sysfs root complex)"
    else:
        prefetch_hint = "Node 0 (Default)"

    draft_node = spec_info["optimal_draft_node"]

    cmd = [
        "./build/bin/llama-server",
        "-m /path/to/target-model.gguf",
    ]
    if draft_node >= 0:
        cmd.append("-md /path/to/draft-model.gguf")
    cmd.extend(flags)

    return {
        "recommended_flags": flags,
        "recommended_command": " \\\n    ".join(cmd),
        "prefetch_node_hint": prefetch_hint,
        "draft_node_hint": f"Node {draft_node}" if draft_node >= 0 else "Colocated",
        "thread_count": threads_recommendation,
    }

# --- Human-Readable Terminal Report Formatter ---

def render_terminal_report(report_data, colors):
    c = colors
    cpu = report_data["cpu"]
    mem = report_data["memory"]
    gpus = report_data["pci_gpus"]
    storage = report_data["storage"]
    spec = report_data["speculative_interconnect"]
    llama = report_data["llama_synthesis"]

    lines = []
    lines.append(f"{c.BOLD}{c.CYAN}================================================================================{c.RESET}")
    lines.append(f"{c.BOLD}{c.CYAN}          ik_llama.cpp Hardware Topology & Architecture Synthesis Report        {c.RESET}")
    lines.append(f"{c.BOLD}{c.CYAN}================================================================================{c.RESET}")
    lines.append(f"{c.YELLOW} [!] PRE-ALPHA ARCHITECTURE NOTICE:{c.RESET} NUMA split & low-latency streaming are experimental.")
    lines.append("")

    # 1. System & CPU Topology
    lines.append(f"{c.BOLD}{c.GREEN}[1] CPU Topology & Microarchitecture{c.RESET}")
    lines.append(f"  • Processor Model     : {c.WHITE}{cpu['model_name']}{c.RESET}")
    lines.append(f"  • Microarchitecture   : {c.WHITE}{cpu['uarch']}{c.RESET} ({cpu['vendor_id']})")
    lines.append(f"  • Physical Sockets    : {c.WHITE}{cpu['num_sockets']}{c.RESET}")
    lines.append(f"  • Physical Cores      : {c.WHITE}{cpu['total_physical_cores']}{c.RESET} ({cpu['cores_per_socket']} cores/socket)")
    lines.append(f"  • Logical Threads     : {c.WHITE}{cpu['total_logical_cpus']}{c.RESET} ({cpu['smt_threads_per_core']} SMT threads/core)")
    lines.append(f"  • NUMA Partitioning   : {c.WHITE}{cpu['nps_mode']}{c.RESET} ({cpu['total_numa_nodes']} total NUMA nodes, {cpu['nodes_per_socket']} nodes/socket)")
    lines.append("")

    # 2. NUMA Memory Distribution & THP
    lines.append(f"{c.BOLD}{c.GREEN}[2] NUMA Memory Distribution & Transparent Huge Pages (THP){c.RESET}")
    lines.append(f"  • System Total RAM    : {c.WHITE}{format_bytes(mem['total_bytes'])}{c.RESET} ({format_bytes(mem['free_bytes'])} free)")
    lines.append(f"  • THP Configuration   : [{c.WHITE}{mem['thp_enabled']}{c.RESET}] (defrag: [{mem['thp_defrag']}], shmem: [{mem['thp_shmem']}])")
    lines.append(f"  • THP Status          : {mem['thp_recommendation']}")
    lines.append("  • Per-Node Allocation :")
    for n in cpu["numa_nodes"]:
        n_id = n["id"]
        s_id = n["socket_id"]
        tot_str = format_bytes(n["total_bytes"], dual=False)
        free_str = format_bytes(n["free_bytes"], dual=False)
        cores = n["core_count"]
        cpus_span = f"cpu{n['cpus'][0]}-{n['cpus'][-1]}" if len(n["cpus"]) > 1 else f"cpu{n['cpus'][0]}"
        lines.append(f"    - {c.BOLD}Node {n_id}{c.RESET} (Socket {s_id}): {tot_str} total, {free_str} free | {cores} physical cores ({cpus_span})")
    lines.append("")

    # 3. PCIe Topology & GPU Accelerators
    lines.append(f"{c.BOLD}{c.GREEN}[3] PCIe Topology & GPU Accelerators{c.RESET}")
    compute_gpus = [g for g in gpus if g["is_compute_gpu"]]
    bmc_adapters = [g for g in gpus if g["is_bmc"]]
    if compute_gpus:
        for g in compute_gpus:
            lines.append(f"  • {c.WHITE}{g['name']}{c.RESET} @ {c.CYAN}{g['pci_address']}{c.RESET}")
            lines.append(f"    - PCIe Affinity     : {c.BOLD}NUMA Node {g['numa_node']}{c.RESET}")
            lines.append(f"    - Bus Link Speed    : {g['pcie_link']}")
    else:
        lines.append(f"  • {c.YELLOW}No dedicated discrete GPU accelerators found (running CPU-only NUMA).{c.RESET}")
    if bmc_adapters:
        for b in bmc_adapters:
            lines.append(f"  • Console Adapter   : {b['name']} @ {b['pci_address']} (NUMA Node {b['numa_node']})")
    lines.append("")

    # 4. Storage Subsystem & NUMA Locality
    lines.append(f"{c.BOLD}{c.GREEN}[4] Storage Subsystem & NUMA Root Complex Discovery{c.RESET}")
    if storage["zfs_pools"]:
        lines.append("  • OpenZFS Storage Pools:")
        for z in storage["zfs_pools"]:
            lines.append(f"    - Pool '{c.WHITE}{z['pool_name']}{c.RESET}': Raw Capacity: {c.BOLD}{z['raw_capacity_str']}{c.RESET} | NUMA Affinity: {c.BOLD}Node {z['numa_node']}{c.RESET}")
            lines.append(f"      Member Drives     : {', '.join(z['member_drives']) if z['member_drives'] else 'NVMe carrier card backing vdevs'}")
            lines.append(f"      Mount Points      : {', '.join(z['mount_points'])}")
    if storage["md_arrays"]:
        lines.append("  • Linux Software MD RAID Arrays:")
        for m in storage["md_arrays"]:
            lines.append(f"    - Device '/dev/{m['device']}': Level: {m['level']} | Capacity: {m['size_str']} | NUMA Affinity: {c.BOLD}Node {m['numa_node']}{c.RESET}")
            lines.append(f"      Member Slaves     : {', '.join(m['slaves'])}")
    if storage["nvme_drives"]:
        lines.append("  • Physical NVMe Drives:")
        for d in storage["nvme_drives"]:
            lines.append(f"    - {d['device']} : {d['model']} ({d['size_str']}) | PCI: {d['pci_address']} -> {c.BOLD}NUMA Node {d['numa_node']}{c.RESET}")
    lines.append(f"  • Optimal Prefetch Node : {c.BOLD}{c.CYAN}Node {storage['optimal_prefetch_node']}{c.RESET} (Binds MoE DMA async page-in threads to storage local channels)")
    lines.append(f"  • Storage Capacity Note : Manufacturers define 1 TB = 1,000,000,000,000 bytes (e.g. 4x 4 TB = 16 TB raw), while OS/ZFS tools report in binary TiB (14.88 TiB = 16.36 TB).")
    lines.append("")

    # 5. Inter-Socket Interconnect & Speculative Drafting
    lines.append(f"{c.BOLD}{c.GREEN}[5] Inter-Socket Interconnect & Speculative Decoding Placement{c.RESET}")
    lines.append(f"  • Interconnect Link   : {c.WHITE}{spec['interconnect_name']}{c.RESET}")
    lines.append(f"  • Est. Bandwidth      : {spec['interconnect_bandwidth']}")
    lines.append(f"  • Target Model Anchor : {c.BOLD}NUMA Node {spec['target_primary_node']}{c.RESET} (Socket {cpu['numa_nodes'][spec['target_primary_node']]['socket_id']})")
    lines.append(f"  • Optimal Draft Node  : {c.BOLD}{c.CYAN}NUMA Node {spec['optimal_draft_node']}{c.RESET}")
    lines.append(f"  • Placement Rationale : {spec['draft_selection_reason']}")
    lines.append("")

    # 6. Recommended ik_llama.cpp Runtime Configuration
    lines.append(f"{c.BOLD}{c.GREEN}[6] Synthesized ik_llama.cpp Runtime Launch Configuration{c.RESET}")
    lines.append(f"  Recommended Flags : {c.WHITE}{' '.join(llama['recommended_flags'])}{c.RESET}")
    lines.append("")
    lines.append(f"{c.BOLD}Ready-to-Run Server Launch Command:{c.RESET}")
    lines.append(f"{c.CYAN}{llama['recommended_command']}{c.RESET}")
    lines.append("")
    lines.append(f"{c.BOLD}{c.CYAN}================================================================================{c.RESET}")

    return "\n".join(lines)

# --- Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Hardware Inspection and NUMA Topology Synthesizer for ik_llama.cpp")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON structure")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI terminal colors")
    parser.add_argument("--sysfs-root", default="/sys", help="Custom sysfs root path for testing or emulation (default: /sys)")
    args = parser.parse_args()

    colors = Colors(enabled=not args.no_color)

    # 1. Probing
    cpu_info = probe_cpu_topology(sysfs_root=args.sysfs_root)
    mem_info = probe_memory_and_thp(sysfs_root=args.sysfs_root)
    pci_gpus = probe_pci_gpus(sysfs_root=args.sysfs_root)
    storage_info = probe_block_devices_and_storage(sysfs_root=args.sysfs_root)
    spec_info = synthesize_speculative_and_interconnect(cpu_info, pci_gpus)
    llama_info = synthesize_llama_flags(cpu_info, mem_info, storage_info, pci_gpus, spec_info)

    report_data = {
        "cpu": cpu_info,
        "memory": mem_info,
        "pci_gpus": pci_gpus,
        "storage": storage_info,
        "speculative_interconnect": spec_info,
        "llama_synthesis": llama_info,
    }

    if args.json:
        print(json.dumps(report_data, indent=2))
    else:
        print(render_terminal_report(report_data, colors))

if __name__ == "__main__":
    main()
