"""
scripts/benchmarks/common.py - Shared Provenance, Metrics, and Subprocess Engine

Extracts and unifies duplicated benchmarking utilities across:
  - scripts/bench_matrix.py
  - scripts/bench_multicontext.py
  - scripts/bench_concurrent_server.py
"""

import hashlib
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

MODE_LIVE = "live"
MODE_SIMULATED = "simulated"
MODE_FAILED = "failed"
RUN_MODES = (MODE_LIVE, MODE_SIMULATED, MODE_FAILED)

_GIT_SHA_CACHE: Optional[str] = None


def get_git_sha(repo_root: Optional[Path] = None) -> str:
    """Return current git HEAD SHA for provenance tracking."""
    global _GIT_SHA_CACHE
    if _GIT_SHA_CACHE is not None:
        return _GIT_SHA_CACHE
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        _GIT_SHA_CACHE = out if out else "unknown"
        return _GIT_SHA_CACHE
    except Exception:
        _GIT_SHA_CACHE = "unknown"
        return _GIT_SHA_CACHE


def get_binary_sha256(path: Any) -> str:
    """Compute and cache SHA256 of binary file for provenance integrity."""
    if not path:
        return "missing"
    p = Path(path)
    if not p.is_file():
        resolved = shutil.which(str(path))
        if not resolved:
            return "missing"
        p = Path(resolved)
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "error"


def detect_platform_info() -> Dict[str, Any]:
    """Probe hardware platform, CPU, NUMA, GPU, and memory characteristics."""
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "hostname": platform.node(),
        "numa_nodes": 1,
        "cpu_cores": os.cpu_count() or 1,
        "is_apple_silicon": False,
        "is_dual_epyc": False,
        "gpu_name": "None",
        "gpu_vram_gb": 0.0,
    }

    if info["os"] == "Darwin" and info["arch"] == "arm64":
        info["is_apple_silicon"] = True
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            info["cpu_brand"] = brand
            mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            info["total_ram_gb"] = round(int(mem) / (1024 ** 3), 1)
            info["p_cores"] = int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.logicalcpu"]).decode().strip())
            info["e_cores"] = int(subprocess.check_output(["sysctl", "-n", "hw.perflevel1.logicalcpu"]).decode().strip())
            info["gpu_name"] = f"{brand} Metal 3.1"
            info["gpu_vram_gb"] = info["total_ram_gb"]
        except Exception:
            info["p_cores"] = 8
            info["e_cores"] = 4
        return info

    if info["os"] == "Linux":
        # Check NUMA nodes
        try:
            node_dirs = list(Path("/sys/devices/system/node").glob("node[0-9]*"))
            if node_dirs:
                info["numa_nodes"] = len(node_dirs)
        except Exception:
            pass

        # Check CPU model
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
                m = re.search(r"model name\s*:\s*(.+)", content)
                if m:
                    info["cpu_brand"] = m.group(1).strip()
                    if "EPYC" in info["cpu_brand"]:
                        info["is_dual_epyc"] = True
        except Exception:
            pass

        # Check Total RAM
        try:
            with open("/proc/meminfo", "r") as f:
                content = f.read()
                m = re.search(r"MemTotal:\s*(\d+)\s*kB", content)
                if m:
                    info["total_ram_gb"] = round(int(m.group(1)) / (1024 ** 2), 1)
        except Exception:
            pass

        # Check NVIDIA GPU via nvidia-smi if available
        if shutil.which("nvidia-smi"):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                lines = [l for l in out.splitlines() if l.strip()]
                if lines:
                    info["gpu_count"] = len(lines)
                    parts = lines[0].split(",")
                    info["gpu_name"] = parts[0].strip()
                    info["gpu_vram_gb"] = round(float(parts[1].strip()) / 1024.0, 1)
            except Exception:
                info["gpu_count"] = 0

    return info


def get_hardware_fingerprint(plat: Optional[Dict[str, Any]] = None) -> str:
    """Return a concise, deterministic hardware topology fingerprint."""
    if plat is None:
        plat = detect_platform_info()
    return (
        f"{plat.get('arch', 'unknown')}|{plat.get('cpu_brand', 'unknown')}|"
        f"{plat.get('cpu_cores', 1)}c|numa={plat.get('numa_nodes', 1)}|"
        f"ram={plat.get('total_ram_gb', 0.0)}G|gpu={plat.get('gpu_name', 'None')}({plat.get('gpu_vram_gb', 0.0)}G)"
    )


def read_proc_vmhwm_kb(pid: int) -> int:
    """Read VmHWM (Peak Resident Set Size) in kB from /proc/<pid>/status."""
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        pass
    return 0


class RusagePopen(subprocess.Popen):
    """Popen subclass that captures per-child resource usage via os.wait4 on POSIX."""

    def __init__(self, *args, **kwargs):
        self.rusage = None
        super().__init__(*args, **kwargs)

    def _try_wait(self, wait_flags):
        if hasattr(os, "wait4"):
            try:
                pid, sts, ru = os.wait4(self.pid, wait_flags)
                if pid == self.pid:
                    self.rusage = ru
                return (pid, sts)
            except ChildProcessError:
                return (self.pid, 0)
        return super()._try_wait(wait_flags)


def run_with_vmhwm_monitor(cmd, env=None, timeout=None) -> Tuple[int, str, str, int]:
    """Run subprocess while polling /proc/<pid>/status and os.wait4 for peak VmHWM in kB."""
    proc = RusagePopen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        env=env,
    )
    peak_kb = [0]
    stop_evt = threading.Event()

    def _poll():
        while not stop_evt.is_set():
            hwm = read_proc_vmhwm_kb(proc.pid)
            if hwm > peak_kb[0]:
                peak_kb[0] = hwm
            stop_evt.wait(0.01)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    finally:
        stop_evt.set()
        t.join(timeout=1.0)

    if getattr(proc, "rusage", None) is not None and getattr(proc.rusage, "ru_maxrss", 0) > 0:
        ru_kb = int(proc.rusage.ru_maxrss)
        if ru_kb > peak_kb[0]:
            peak_kb[0] = ru_kb
    elif peak_kb[0] == 0:
        try:
            ru = resource.getrusage(resource.RUSAGE_CHILDREN)
            if ru.ru_maxrss > 0:
                peak_kb[0] = int(ru.ru_maxrss)
        except Exception:
            pass

    return proc.returncode, stdout, stderr, peak_kb[0]


class VmHwmMonitor:
    """Background thread that tracks VmHWM from /proc/<pid>/status."""
    def __init__(self, pid: int, poll_interval_s: float = 0.05):
        self.pid = pid
        self.poll_interval_s = poll_interval_s
        self.max_vmhwm_kb = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self) -> int:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        return self.max_vmhwm_kb

    def _run(self):
        status_path = Path(f"/proc/{self.pid}/status")
        while not self._stop_event.is_set():
            if status_path.is_file():
                try:
                    for line in status_path.read_text().splitlines():
                        if line.startswith("VmHWM:"):
                            val = int(line.split()[1])
                            if val > self.max_vmhwm_kb:
                                self.max_vmhwm_kb = val
                            break
                except Exception:
                    pass
            time.sleep(self.poll_interval_s)
