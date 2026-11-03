#!/usr/bin/env python3
"""Linux-only, sampled containment for one directly launched test fixture.

stdout is UTF-8 JSONL: ready, episode_begin(id), episode_end(id, ok: bool).
Other records require a string event and are opaque, bounded measurements.
Artifacts are INTERNAL: raw streams are not sanitized for public publication.
This is not a kernel resource limit or capacity-calibration instrumentation.
Campaign acceptance additionally requires --reviewed-no-subprocess-contract: a
reviewer-approved digest from contract_identity(argv), binding executable bytes
and exact argv. A digest is an attestation, not a technical prohibition on fork.
The operator must keep the executable immutable and use the reviewed environment
and dynamic dependencies. Missing attestation allows containment validation only.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import select
import selectors
import signal
import subprocess
import sys
import time

MIB = 1024 * 1024
RESERVE = 16384
RECORD_LIMIT = 16384
READ_SIZE = 4096
CADENCE = 0.1
HEADROOM = 30.0


class Stop(Exception):
    pass


def finite_number(value):
    value = float(value)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("finite number required")
    return value


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=600)
    parser.add_argument("--wall-seconds", type=finite_number, default=1800.0)
    parser.add_argument("--cpu-seconds", type=finite_number, default=1800.0)
    parser.add_argument("--rss-mib", type=finite_number, default=768.0)
    parser.add_argument("--log-mib", type=finite_number, default=128.0)
    parser.add_argument("--reviewed-no-subprocess-contract",
                        help="reviewed SHA256 binding executable content and exact argv; not proof of no fork")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if args.reviewed_no_subprocess_contract is not None:
        digest = args.reviewed_no_subprocess_contract
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            parser.error("reviewed contract must be a lowercase SHA256 digest")
    if not 1 <= args.max_episodes <= 600:
        parser.error("episode limit must be in [1, 600]")
    if not HEADROOM < args.wall_seconds <= 1800 or not HEADROOM < args.cpu_seconds <= 1800:
        parser.error("wall/CPU limits must exceed the fixed 30s reserve and be <=1800")
    if not 0 < args.rss_mib <= 768 or not RESERVE < args.log_mib * MIB <= 128 * MIB:
        parser.error("RSS/log limits must be positive and cannot raise approved ceilings; logs need summary reserve")
    if not args.command or not os.path.isabs(args.command[0]):
        parser.error("an absolute executable path after -- is required (no PATH search)")
    if not os.path.isfile(args.command[0]) or not os.access(args.command[0], os.X_OK):
        parser.error("executable is unavailable")
    return args


def contract_identity(command):
    """Versioned binding only; callers must independently review no-subprocess behavior.

    SHA256(canonical JSON {version:1, executable_sha256:SHA256(bytes), argv:[...]}).
    Never publish argv; it can contain paths or secrets. Immutable executable,
    reviewed environment and dependency assumptions remain outside this digest.
    """
    digest = hashlib.sha256()
    with open(command[0], "rb") as stream:
        before = os.fstat(stream.fileno())
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    identity = {"device": before.st_dev, "inode": before.st_ino, "size": before.st_size,
                "mtime_ns": before.st_mtime_ns, "sha256": digest.hexdigest()}
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("executable changed during identity observation")
    payload = {"version": 1, "executable_sha256": digest.hexdigest(), "argv": command}
    binding = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True).encode("ascii")).hexdigest()
    return identity, binding


def child_setup():
    # Only the launched child changes its core limit; no host configuration.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def read_proc(pid):
    with open(f"/proc/{pid}/stat", "rb") as stream:
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError("oversized stat")
    fields = raw[raw.rfind(b") ") + 2:].split()
    return {"state": fields[0].decode("ascii"), "parent": int(fields[1]), "group": int(fields[2]),
            "start": int(fields[19]),
            "cpu": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
            "rss": int(fields[21]) * os.sysconf("SC_PAGE_SIZE")}


def scoped_children(pid):
    """Only inspect tasks belonging to a known process; never enumerate host PIDs."""
    children: set[int] = set()
    with os.scandir(f"/proc/{pid}/task") as tasks:
        for count, task in enumerate(tasks):
            if count >= 4096:
                raise ValueError("task observation bound")
            try:
                with open(task.path + "/children", "rb") as stream:
                    raw = stream.read(RECORD_LIMIT + 1)
            except FileNotFoundError:
                continue  # A thread may finish between task enumeration and open.
            if len(raw) > RECORD_LIMIT:
                raise ValueError("child observation bound")
            children.update(int(value) for value in raw.split())
            if len(children) > 256:
                raise ValueError("child observation bound")
    return children


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


@contextlib.contextmanager
def artifact_stream(path):
    with open(path, "xb", buffering=0) as stream:
        yield stream


def terminal_record(data, deadline):
    """A stalled report consumer must not hold the campaign open indefinitely."""
    fd = sys.stdout.fileno()
    blocking = os.get_blocking(fd)
    try:
        os.set_blocking(fd, False)
        offset = 0
        while offset < len(data) and time.monotonic() < deadline:
            if not select.select([], [fd], [], min(0.05, max(0, deadline - time.monotonic())))[1]:
                continue
            try:
                offset += os.write(fd, data[offset:])
            except BlockingIOError:
                continue
        return offset == len(data)
    except OSError:
        return False
    finally:
        os.set_blocking(fd, blocking)


class Runner:
    def __init__(self, args):
        self.args = args
        self.start = time.monotonic()
        # Charge interpreter/import/argument startup too, not only the loop.
        self.cpu_start = 0.0
        self.proc = None
        self.known = {}
        self.final_root_cpu = None
        self.root_usage = None
        self.ready = False
        self.seen = set()
        self.active = {}
        self.completed = 0
        self.failed = 0
        self.attempts = 0
        self.reason = None
        self.buffer = bytearray()
        self.selector = selectors.DefaultSelector()
        self.files = {}
        self.file_stack = contextlib.ExitStack()
        self.log_bytes = 0
        self.stream_bytes = {"stdout": 0, "stderr": 0, "samples": 0}
        self.samples = 0
        self.first_sample = None
        self.last_sample = None
        self.max_gap = 0.0
        self.peak_rss = 0
        self.last_cpu = None
        self.sample_overhead = 0.0
        self.sample_attempts = 0
        self.max_sample_duration = 0.0
        self.residual = []
        self.unwaitable = []
        self.cleaning = False
        self.interrupted = False
        self.exit_code = None
        self.old_handlers = {}
        self.executable_identity = None
        self.descendants_observed = False
        self.group_cleanup_unconfirmed = False
        self.pidfds = {}
        self.contract_digest = None
        self.contract_attested = False

    def elapsed(self):
        return time.monotonic() - self.start

    def stop(self, reason):
        if self.reason is None:
            self.reason = reason
        raise Stop()

    def retain(self, name, data):
        remaining = int(self.args.log_mib * MIB) - RESERVE - self.log_bytes
        kept = data[:max(0, remaining)]
        if kept:
            self.files[name].write(kept)
            self.log_bytes += len(kept)
            self.stream_bytes[name] += len(kept)
        if len(kept) != len(data):
            self.stop("log_limit")

    def event(self, raw):
        try:
            record = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                                parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            if not isinstance(record, dict) or not isinstance(record.get("event"), str):
                raise ValueError()
            kind = record["event"]
            if not kind or len(kind) > 128:
                raise ValueError()
        except (ValueError, UnicodeError, RecursionError):
            self.stop("malformed_record")
        if kind == "ready":
            if not self.ready and self.elapsed() >= 20:
                self.stop("ready_timeout")
            if self.ready or self.seen:
                self.stop("duplicate_or_late_ready")
            self.ready = True
        elif kind in ("episode_begin", "episode_end"):
            identity = record.get("id")
            if not self.ready or not isinstance(identity, (str, int)) or isinstance(identity, bool):
                self.stop("invalid_episode")
            if isinstance(identity, str) and (not identity or len(identity) > 128):
                self.stop("invalid_episode")
            # Preserve integer/string identity without collisions.
            identity = (type(identity).__name__, identity)
            if kind == "episode_begin":
                if self.elapsed() >= self.args.wall_seconds - HEADROOM:
                    self.stop("wall_cleanup_reserve")
                self.attempts += 1
                if identity in self.seen:
                    self.stop("duplicate_episode")
                if len(self.seen) >= self.args.max_episodes:
                    self.stop("episode_limit")
                self.seen.add(identity)
                self.active[identity] = time.monotonic()
            else:
                if identity not in self.active or type(record.get("ok")) is not bool:
                    self.stop("invalid_episode_end")
                if time.monotonic() - self.active[identity] >= 10:
                    self.stop("episode_timeout")
                del self.active[identity]
                if not record["ok"]:
                    self.failed += 1
                    self.stop("episode_failure")
                self.completed += 1

    def drain_once(self, timeout: float = 0.0):
        for key, _ in self.selector.select(timeout):
            try:
                data = os.read(key.fd, READ_SIZE)
            except BlockingIOError:
                continue
            if not data:
                self.selector.unregister(key.fileobj)
                continue
            name = key.data
            self.retain(name, data)
            if name == "stdout":
                self.buffer.extend(data)
                while b"\n" in self.buffer:
                    line, _, rest = self.buffer.partition(b"\n")
                    self.buffer = bytearray(rest)
                    if len(line) > RECORD_LIMIT:
                        self.stop("record_limit")
                    self.event(line)
                if len(self.buffer) > RECORD_LIMIT:
                    self.stop("record_limit")

    def aggregate_cpu(self):
        root = self.known.get(self.proc.pid, {}) if self.proc else {}
        fixture = self.final_root_cpu if self.final_root_cpu is not None else root.get("cpu", 0.0)
        root_pid = self.proc.pid if self.proc is not None else None
        return time.process_time() - self.cpu_start + fixture + sum(
            item["cpu"] for pid, item in self.known.items() if pid != root_pid)

    def sample(self):
        before = time.process_time()
        started = time.monotonic()
        self.sample_attempts += 1
        try:
            return self._sample()
        finally:
            self.sample_overhead += time.process_time() - before
            self.max_sample_duration = max(self.max_sample_duration, time.monotonic() - started)

    def _sample(self):
        now = self.elapsed()
        rss = 0
        assert self.proc is not None
        pending = list(self.known)
        if self.proc.pid not in self.known:
            pending.append(self.proc.pid)
        visited = set()
        unexpected = False
        while pending:
            pid = pending.pop()
            if pid in visited:
                continue
            visited.add(pid)
            if pid != self.proc.pid:
                self.descendants_observed = True
            if len(visited) > 256:
                self.stop("process_observation_bound")
            try:
                item = read_proc(pid)
                old = self.known.get(pid)
                if old is not None and old["start"] != item["start"]:
                    self.stop("process_identity_changed")
                self.known[pid] = item
                rss += item["rss"]
                if pid != self.proc.pid:
                    unexpected = True
                if item["state"] not in ("Z", "X"):
                    for child in scoped_children(pid):
                        if child not in self.known:
                            self.capture_descendant(pid, item, child)
                        pending.append(child)
            except (OSError, ValueError, IndexError):
                # Even a short-lived descendant prevents complete attribution.
                self.stop("required_metric_unavailable")
        try:
            own = read_proc(os.getpid())
        except (OSError, ValueError, IndexError):
            self.stop("required_metric_unavailable")
        rss += own["rss"]  # process RSS once, not summed over its threads
        self.peak_rss = max(self.peak_rss, rss)
        self.last_cpu = self.aggregate_cpu()
        if self.first_sample is None:
            self.first_sample = now
        if self.last_sample is not None:
            self.max_gap = max(self.max_gap, now - self.last_sample)
        self.last_sample = now
        self.samples += 1
        self.retain("samples", (json.dumps({"t": now, "rss_bytes": rss,
                    "cpu_seconds": self.last_cpu}, separators=(",", ":")) + "\n").encode())
        if unexpected:
            self.stop("unexpected_child")
        if rss >= self.args.rss_mib * MIB:
            self.stop("rss_limit")
        if self.last_cpu >= self.args.cpu_seconds - HEADROOM:
            self.stop("cpu_cleanup_reserve")
        return self.known[self.proc.pid]["state"]

    def capture_descendant(self, parent_pid, parent, pid):
        """Capture only a task/children-discovered, parent/start-verified identity."""
        self.descendants_observed = True
        if len(self.known) >= 256:
            self.stop("process_observation_bound")
        fd = None
        try:
            first = read_proc(pid)
            if first["parent"] != parent_pid:
                self.residual.append({"pid": pid, "start": first["start"], "state": "ownership_unconfirmed"})
                self.stop("descendant_ownership_unconfirmed")
            self.known[pid] = first
            if not callable(getattr(os, "pidfd_open", None)) or not callable(getattr(signal, "pidfd_send_signal", None)):
                self.stop("pidfd_capability_unavailable")
            fd = os.pidfd_open(pid, 0)
            current = read_proc(pid)
            owner = read_proc(parent_pid)
            if (current["start"] != first["start"] or current["parent"] != parent_pid or
                    owner["start"] != parent["start"]):
                self.stop("descendant_identity_unconfirmed")
            self.pidfds[pid] = (fd, first["start"])
            fd = None
        except (OSError, ValueError, IndexError):
            self.stop("descendant_handle_unavailable")
        finally:
            if fd is not None:
                os.close(fd)

    def signal_descendants(self, sig):
        for pid, (fd, start) in self.pidfds.items():
            try:
                current = read_proc(pid)
                if current["start"] != start or current["state"] in ("Z", "X"):
                    continue
                # Never use kill(pid): a pidfd refers to the captured identity,
                # including after setsid/setpgid or reparenting.
                signal.pidfd_send_signal(fd, sig, None, 0)
            except ProcessLookupError:
                continue
            except (OSError, ValueError, IndexError):
                self.reason = self.reason or "descendant_cleanup_unconfirmed"

    def close_pidfds(self):
        for fd, _ in self.pidfds.values():
            with contextlib.suppress(OSError):
                os.close(fd)
        self.pidfds.clear()

    def aggregate_cpu_available(self):
        # Absence in samples cannot exclude an unwaited, short-lived orphan.
        return self.contract_attested and self.final_root_cpu is not None and not self.descendants_observed

    def interrupt(self, signum, frame):
        # Do not raise between Popen returning and assignment of its owned PID.
        # The foreground loop observes this flag; cleanup cannot be interrupted.
        self.interrupted = True

    def group_signal(self, sig):
        # The unreaped direct child anchors this new PGID against reuse.
        assert self.proc is not None
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, sig)

    def cleanup(self):
        self.cleaning = True
        if self.proc is None:
            return
        self.group_signal(signal.SIGTERM)
        self.signal_descendants(signal.SIGTERM)
        grace = time.monotonic() + 1.0
        next_sample = time.monotonic()
        while time.monotonic() < grace:
            try:
                if time.monotonic() >= next_sample:
                    self.sample()
                    self.signal_descendants(signal.SIGTERM)
                    next_sample = time.monotonic() + CADENCE
                self.drain_once(min(0.05, max(0, grace - time.monotonic())))
            except (Stop, OSError, ValueError) as error:
                # Retention/protocol failure cannot prevent the KILL/reap path.
                if not isinstance(error, Stop):
                    self.reason = self.reason or "cleanup_observation_or_log_failure"
                self.signal_descendants(signal.SIGTERM)
                next_sample = time.monotonic() + CADENCE
                time.sleep(0.01)
        self.group_signal(signal.SIGKILL)
        self.signal_descendants(signal.SIGKILL)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            pid, status, usage = os.wait4(self.proc.pid, os.WNOHANG)
            if pid:
                self.exit_code = os.waitstatus_to_exitcode(status)
                self.proc.returncode = self.exit_code
                self.final_root_cpu = usage.ru_utime + usage.ru_stime
                self.root_usage = {"user_seconds": usage.ru_utime,
                                   "system_seconds": usage.ru_stime,
                                   "maxrss_kib": usage.ru_maxrss}
                break
            time.sleep(0.01)
        if self.final_root_cpu is None:
            self.residual.append({"pid": self.proc.pid, "state": "unreaped"})
        else:
            # Non-signalling query only: after reap the PGID is no longer an
            # identity-protected signal target. Any surviving group is ambiguous.
            try:
                os.killpg(self.proc.pid, 0)
                self.group_cleanup_unconfirmed = True
            except ProcessLookupError:
                pass
        # SIGKILL delivery is asynchronous; allow known identities to become
        # gone/zombie within the same bounded cleanup deadline before judging.
        while time.monotonic() < deadline:
            live = False
            for pid, old in self.known.items():
                if pid == self.proc.pid:
                    continue
                try:
                    item = read_proc(pid)
                    live |= item["start"] == old["start"] and item["state"] not in ("Z", "X")
                except (OSError, ValueError, IndexError):
                    continue
            if not live:
                break
            time.sleep(0.01)
        # Only known identities are inspected. Zombies are not running, but are
        # explicitly not claimed to have been reaped by this supervisor.
        for pid, old in self.known.items():
            if pid == self.proc.pid and self.final_root_cpu is not None:
                continue
            try:
                current = read_proc(pid)
            except FileNotFoundError:
                if pid != self.proc.pid:
                    self.unwaitable.append({"pid": pid, "start": old["start"], "state": "gone"})
                continue
            except (OSError, ValueError, IndexError):
                self.residual.append({"pid": pid, "start": old["start"], "state": "unavailable"})
                continue
            if current["start"] != old["start"]:
                continue  # The recorded identity has exited; never signal its replacement.
            entry = {"pid": pid, "start": old["start"], "state": current["state"]}
            if current["state"] in ("Z", "X"):
                self.unwaitable.append(entry)
            else:
                self.residual.append(entry)
        # Drain at most the budget and a bounded number of pipe chunks after KILL.
        for _ in range(64):
            if not self.selector.get_map():
                break
            try:
                self.drain_once(0)
            except (Stop, OSError, ValueError) as error:
                if not isinstance(error, Stop):
                    self.reason = self.reason or "cleanup_observation_or_log_failure"
                break
        for key in list(self.selector.get_map().values()):
            self.selector.unregister(key.fileobj)
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream is not None:
                stream.close()

    def run(self):
        # mkdir is deliberately exclusive: existing directories/symlinks fail.
        os.mkdir(self.args.output_dir, 0o700)
        try:
            for name in ("stdout", "stderr", "samples"):
                self.files[name] = self.file_stack.enter_context(
                    artifact_stream(os.path.join(self.args.output_dir, name + ".jsonl")))
            self.executable_identity, self.contract_digest = contract_identity(self.args.command)
            attestation = self.args.reviewed_no_subprocess_contract
            if attestation is not None and attestation != self.contract_digest:
                self.stop("no_subprocess_contract_mismatch")
            self.contract_attested = attestation is not None
            if not callable(getattr(os, "pidfd_open", None)) or not callable(getattr(signal, "pidfd_send_signal", None)):
                self.stop("pidfd_capability_unavailable")
            for sig in (signal.SIGINT, signal.SIGTERM):
                self.old_handlers[sig] = signal.signal(sig, self.interrupt)
            self.proc = subprocess.Popen(self.args.command, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         start_new_session=True, preexec_fn=child_setup,
                                         close_fds=True, shell=False)
            if self.interrupted:
                self.stop("interrupt")
            try:
                capability_fd = os.pidfd_open(self.proc.pid, 0)
                os.close(capability_fd)
            except OSError:
                self.stop("pidfd_capability_unavailable")
            for stream, name in ((self.proc.stdout, "stdout"), (self.proc.stderr, "stderr")):
                assert stream is not None
                os.set_blocking(stream.fileno(), False)
                self.selector.register(stream, selectors.EVENT_READ, name)
            next_sample = time.monotonic()
            while True:
                now = time.monotonic()
                if self.interrupted:
                    self.stop("interrupt")
                if self.elapsed() >= self.args.wall_seconds - HEADROOM:
                    self.stop("wall_cleanup_reserve")
                if not self.ready and self.elapsed() >= 20:
                    self.stop("ready_timeout")
                if any(now - began >= 10 for began in self.active.values()):
                    self.stop("episode_timeout")
                if now >= next_sample:
                    state = self.sample()
                    next_sample = time.monotonic() + CADENCE
                    if state in ("Z", "X"):
                        # Finish already buffered records before judging protocol.
                        for _ in range(64):
                            if not self.selector.get_map():
                                break
                            self.drain_once(0)
                        break
                self.drain_once(min(0.05, max(0, next_sample - time.monotonic())))
        except Stop:
            pass
        except KeyboardInterrupt:
            self.reason = self.reason or "interrupt"
        except (OSError, ValueError, IndexError, RuntimeError):
            self.reason = self.reason or "supervisor_error"
        finally:
            try:
                self.cleanup()
            except (OSError, ValueError, IndexError, RuntimeError):
                self.reason = self.reason or "cleanup_unconfirmed"
                if self.proc:
                    self.residual.append({"pid": self.proc.pid, "state": "cleanup_unconfirmed"})
            self.close_pidfds()
            for sig, handler in self.old_handlers.items():
                signal.signal(sig, handler)
            self.selector.close()
        if self.residual or self.group_cleanup_unconfirmed:
            self.reason = self.reason or "cleanup_unconfirmed"
        if self.interrupted:
            self.reason = self.reason or "interrupt"
        if self.buffer:
            self.reason = self.reason or "truncated_record"
        if not self.ready:
            self.reason = self.reason or "missing_ready"
        if self.active:
            self.reason = self.reason or "unfinished_episode"
        if not self.seen:
            self.reason = self.reason or "missing_episode"
        if self.exit_code != 0:
            self.reason = self.reason or "fixture_exit_failure"
        final_cpu = self.aggregate_cpu() if self.proc else time.process_time() - self.cpu_start
        wall = self.elapsed()
        if wall > self.args.wall_seconds or final_cpu > self.args.cpu_seconds:
            self.reason = self.reason or "final_budget_exceeded"
        if not self.contract_attested:
            self.reason = self.reason or "no_subprocess_contract_unattested"
        valid_cpu = self.aggregate_cpu_available()
        summary = {
            "event": "terminal", "ok": self.reason is None, "reason": self.reason or "completed",
            "classification": "reviewed_fixture_sampled_containment" if self.contract_attested else "containment_validation_only",
            "executable_identity": self.executable_identity,
            "no_subprocess_contract": {"version": 1, "digest": self.contract_digest,
                                       "attested": self.contract_attested,
                                       "descendants_observed": self.descendants_observed},
            "episodes": {"begun": len(self.seen), "attempts": self.attempts,
                         "completed": self.completed, "failed": self.failed, "unfinished": len(self.active)},
            "limits": {"wall_seconds": self.args.wall_seconds, "cpu_seconds": self.args.cpu_seconds,
                       "rss_bytes": int(self.args.rss_mib * MIB), "log_bytes": int(self.args.log_mib * MIB),
                       "max_episodes": self.args.max_episodes, "cleanup_reserve_seconds": HEADROOM,
                       "ready_seconds": 20, "episode_seconds": 10},
            "wall_seconds_through_report_preparation": wall,
            "supervisor_cpu_seconds": time.process_time() - self.cpu_start,
            "final_aggregate_cpu_seconds": final_cpu if valid_cpu else None,
            "provisional_aggregate_cpu_seconds": self.last_cpu,
            "root_wait4": self.root_usage, "exit_code": self.exit_code,
            "peak_sampled_rss_bytes": self.peak_rss if self.samples else None,
            "sampling": {"cadence_seconds": CADENCE, "count": self.samples,
                         "attempts": self.sample_attempts, "maximum_probe_duration_seconds": self.max_sample_duration,
                         "first_seconds": self.first_sample, "last_seconds": self.last_sample,
                         "maximum_gap_seconds": self.max_gap,
                         "final_observation_gap_seconds": None if self.last_sample is None else wall - self.last_sample,
                         "sampler_cpu_seconds": self.sample_overhead},
            "observed_overshoot": {"wall_seconds": max(0, wall - self.args.wall_seconds),
                                   "cpu_seconds": max(0, final_cpu - self.args.cpu_seconds) if valid_cpu else None,
                                   "rss_bytes": max(0, self.peak_rss - int(self.args.rss_mib * MIB)) if self.samples else None},
            "stream_bytes": self.stream_bytes, "retained_stream_bytes": self.log_bytes,
            "group_cleanup_unconfirmed": self.group_cleanup_unconfirmed,
            "residual_identities": self.residual[:256],
            "unwaitable_descendants": self.unwaitable[:256],
            "limitations": ["sampled thresholds, not zero-overshoot hard bounds",
                            "raw streams are internal and may contain secrets or private paths",
                            "unwaitable descendant final CPU unavailable; no valid aggregate claim",
                            "cleanup cannot survive supervisor SIGKILL or uninterruptible kernel stalls",
                            "wall/CPU reported through report preparation; report I/O adds overhead",
                            "run-local single invocation; operator must prevent overlapping campaigns",
                            "short-lived or escaping descendants can evade sampled discovery",
                            "contract digest binds review, not technical fork prevention or environment behavior",
                            "aggregate completeness assumes reviewed fixture/argv, immutable executable and approved dependencies/environment",
                            "episode totals are observed protocol records, not inferred HTTP attempts",
                            "fixture instrumentation must count controls, faults, recovery and parent attempts"]}
        data = (json.dumps(summary, separators=(",", ":"), allow_nan=False) + "\n").encode()
        # Keep summary bounded even after a descendant-observation failure.
        if len(data) > RESERVE:
            summary["residual_identities"] = [{"state": "detail_bound_exceeded"}]
            summary["unwaitable_descendants"] = [{"state": "detail_bound_exceeded"}]
            data = (json.dumps(summary, separators=(",", ":")) + "\n").encode()
        try:
            with open(os.path.join(self.args.output_dir, "summary.json"), "xb") as stream:
                stream.write(data)
            delivered = terminal_record(data, min(time.monotonic() + 1.0, self.start + self.args.wall_seconds))
            if not delivered:
                # The file remains the authoritative terminal report if its
                # consumer disconnected or applied backpressure to stdout.
                summary["ok"] = False
                summary["reason"] = self.reason or "terminal_report_unavailable"
                data = (json.dumps(summary, separators=(",", ":")) + "\n").encode()
                with open(os.path.join(self.args.output_dir, "summary.json"), "wb") as stream:
                    stream.write(data)
        finally:
            self.file_stack.close()
        return 0 if summary["ok"] else 2


def main():
    try:
        return Runner(arguments()).run()
    except (OSError, ValueError):
        # Do not reflect paths, command arguments, environment or exception text.
        terminal_record(b'{"event":"terminal","ok":false,"reason":"artifact_or_launch_error"}\n',
                        time.monotonic() + 1.0)
        return 2


if __name__ == "__main__":
    sys.exit(main())
