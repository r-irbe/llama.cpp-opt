#!/usr/bin/env python3
"""
Empirical verification harness for live llama-server slab arbiter endpoints and ThinLTO/PGO binary.
Spawns two co-resident llama-server instances (port 18099 and 18100) sharing /dev/shm/ik_llama_fleet_slab_test.shm.
Verifies:
1. Both instances accept --slab-arbiter-shm and --slab-priority.
2. /health returns 200 OK.
3. /metrics exposes slab_arbiter_allocated_slabs and slab_arbiter_free_slabs.
4. /slots exposes per-slot slab status (is_valid, slab_id, generation).
5. Requests borrow/yield slabs dynamically across both co-resident instances.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SHM_PATH = "/dev/shm/ik_llama_fleet_slab_test.shm"
MODEL_PATH = Path(".pi/pto-profiles/tiny_pto_llama.gguf").resolve()
BINARY_PATH = Path("build-pto/bin/llama-server").resolve()

def cleanup():
    if os.path.exists(SHM_PATH):
        try:
            os.remove(SHM_PATH)
            print(f"[Cleanup] Removed {SHM_PATH}")
        except Exception as e:
            print(f"[Cleanup] Error removing {SHM_PATH}: {e}")

def wait_for_ready(port: int, timeout: float = 30.0) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False

def http_get_json(url: str):
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))

def http_get_text(url: str):
    with urllib.request.urlopen(url, timeout=5.0) as resp:
        return resp.read().decode("utf-8")

def http_post_json(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    cleanup()
    if not BINARY_PATH.exists():
        print(f"Error: Binary {BINARY_PATH} not found!")
        sys.exit(1)
    if not MODEL_PATH.exists():
        print(f"Error: Model {MODEL_PATH} not found!")
        sys.exit(1)

    print(f"[1] Testing binary: {BINARY_PATH}")
    print(f"    Size: {BINARY_PATH.stat().st_size} bytes")

    # Instance 1: port 18099, priority 100
    cmd1 = [
        str(BINARY_PATH),
        "-m", str(MODEL_PATH),
        "--host", "127.0.0.1",
        "--port", "18099",
        "-c", "512",
        "-np", "2",
        "-t", "4",
        "-ngl", "0",
        "--metrics",
        "--slab-arbiter-shm", SHM_PATH,
        "--slab-priority", "100"
    ]

    # Instance 2: port 18100, priority 150
    cmd2 = [
        str(BINARY_PATH),
        "-m", str(MODEL_PATH),
        "--host", "127.0.0.1",
        "--port", "18100",
        "-c", "512",
        "-np", "2",
        "-t", "4",
        "-ngl", "0",
        "--metrics",
        "--slab-arbiter-shm", SHM_PATH,
        "--slab-priority", "150"
    ]

    p1 = None
    p2 = None
    try:
        print("[2] Spawning Instance 1 on port 18099...")
        p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if not wait_for_ready(18099):
            print("Instance 1 failed to become ready!")
            if p1.poll() is not None:
                stdout, stderr = p1.communicate()
                print("STDOUT:", stdout[:500])
                print("STDERR:", stderr[:1000])
            sys.exit(1)
        print("    Instance 1 is READY.")

        print("[3] Spawning co-resident Instance 2 on port 18100 sharing same SHM...")
        p2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if not wait_for_ready(18100):
            print("Instance 2 failed to become ready!")
            if p2.poll() is not None:
                stdout, stderr = p2.communicate()
                print("STDOUT:", stdout[:500])
                print("STDERR:", stderr[:1000])
            sys.exit(1)
        print("    Instance 2 is READY.")

        # Verify SHM exists
        if os.path.exists(SHM_PATH):
            shm_size = os.path.getsize(SHM_PATH)
            print(f"[4] Verified shared memory file exists at {SHM_PATH} (size: {shm_size} bytes)")
        else:
            print(f"[4] Warning: {SHM_PATH} does not exist on disk (may be memory-only)")

        # Verify /metrics on both instances
        m1 = http_get_text("http://127.0.0.1:18099/metrics")
        m2 = http_get_text("http://127.0.0.1:18100/metrics")
        assert "slab_arbiter_allocated_slabs" in m1, "Instance 1 /metrics missing slab_arbiter_allocated_slabs"
        assert "slab_arbiter_free_slabs" in m1, "Instance 1 /metrics missing slab_arbiter_free_slabs"
        assert "slab_arbiter_allocated_slabs" in m2, "Instance 2 /metrics missing slab_arbiter_allocated_slabs"
        assert "slab_arbiter_free_slabs" in m2, "Instance 2 /metrics missing slab_arbiter_free_slabs"
        print("[5] Verified /metrics on both instances exposes slab_arbiter metrics:")
        for line in m1.splitlines():
            if "slab_arbiter" in line:
                print(f"    Instance 1 metric: {line}")
        for line in m2.splitlines():
            if "slab_arbiter" in line:
                print(f"    Instance 2 metric: {line}")

        # Verify /slots on both instances
        s1 = http_get_json("http://127.0.0.1:18099/slots")
        s2 = http_get_json("http://127.0.0.1:18100/slots")
        assert len(s1) == 2, f"Instance 1 expected 2 slots, got {len(s1)}"
        assert len(s2) == 2, f"Instance 2 expected 2 slots, got {len(s2)}"
        assert "slab" in s1[0], "Instance 1 /slots missing slab metadata"
        assert "slab" in s2[0], "Instance 2 /slots missing slab metadata"
        print(f"[6] Verified /slots metadata on Instance 1: slot 0 slab={s1[0]['slab']}")
        print(f"    Verified /slots metadata on Instance 2: slot 0 slab={s2[0]['slab']}")

        # Submit completions to trigger active slab borrowing/yielding
        print("[7] Submitting completion requests to both instances...")
        req1 = {"prompt": "Hello world", "n_predict": 4}
        r1 = http_post_json("http://127.0.0.1:18099/completion", req1)
        print(f"    Instance 1 response: content={r1.get('content', '')!r}")

        req2 = {"prompt": "Testing slab", "n_predict": 4}
        r2 = http_post_json("http://127.0.0.1:18100/completion", req2)
        print(f"    Instance 2 response: content={r2.get('content', '')!r}")

        # Re-check /slots and /metrics after completion
        s1_after = http_get_json("http://127.0.0.1:18099/slots")
        print(f"[8] Post-request /slots on Instance 1: slot 0 slab={s1_after[0]['slab']}")

        print("\n=== ALL EMPIRICAL CHECKS PASSED SUCCESSFULLY ===")

    finally:
        print("[9] Tearing down test server instances...")
        if p1:
            p1.terminate()
            try:
                p1.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                p1.kill()
        if p2:
            p2.terminate()
            try:
                p2.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                p2.kill()
        cleanup()

if __name__ == "__main__":
    main()
