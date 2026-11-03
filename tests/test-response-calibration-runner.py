#!/usr/bin/env python3
"""Purpose-built sequential containment fixtures; no server/model/GPU needed.

Run explicitly with Python after independent review. Each low validation wall
limit retains the full 30s cleanup reserve but normally runs for about 1-3s.
These tests do not establish production calibration or hard resource bounds.
"""

import importlib.util
import io
import itertools
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


RUNNER = Path(__file__).with_name("response-calibration-runner.py")
LOADER = SourceFileLoader("response_calibration_runner", str(RUNNER))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
LOADER.exec_module(MODULE)

HEADER = """\
import json, os, resource, signal, time

def event(name, **fields):
    print(json.dumps(dict(event=name, **fields)), flush=True)

event('measurement', pid=os.getpid(), core_limit=list(resource.getrlimit(resource.RLIMIT_CORE)))
"""
NORMAL = """\
event('ready')
event('episode_begin', id='one')
event('episode_end', id='one', ok=True)
"""


class ResponseCalibrationRunnerTests(unittest.TestCase):
    def assert_not_running(self, pid):
        try:
            state = MODULE.read_proc(pid)["state"]
        except FileNotFoundError:
            return
        self.assertIn(state, ("Z", "X"), "fixture PID is still running")

    def run_fixture(self, body, *, options=(), interrupt=False, expected_ok=False, attest=True):
        with tempfile.TemporaryDirectory(prefix="response-runner-test-") as temporary:
            output = Path(temporary) / "new"
            fixture = [sys.executable, "-c", HEADER + textwrap.dedent(body)]
            # Test-local binding, never an approval of arbitrary production
            # executables. Deliberate fork fixtures must invalidate even if bound.
            contract = (["--reviewed-no-subprocess-contract", MODULE.contract_identity(fixture)[1]]
                        if attest else [])
            command = [sys.executable, str(RUNNER), "--output-dir", str(output),
                       "--wall-seconds", "34", "--max-episodes", "3", *options,
                       *contract, "--", *fixture]
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as supervisor:
                try:
                    if interrupt:
                        deadline = time.monotonic() + 2
                        while time.monotonic() < deadline:
                            stream = output / "stdout.jsonl"
                            if stream.exists() and b'"ready"' in stream.read_bytes():
                                break
                            if supervisor.poll() is not None:
                                self.fail("supervisor exited before interrupt fixture became ready")
                            time.sleep(0.01)
                        else:
                            self.fail("interrupt fixture did not become ready")
                        supervisor.send_signal(signal.SIGINT)
                    stdout, stderr = supervisor.communicate(timeout=8)
                except BaseException:
                    # Do not kill the supervisor: its own finally must clean up.
                    supervisor.send_signal(signal.SIGTERM)
                    supervisor.communicate(timeout=5)
                    raise
                self.assertEqual(stderr, b"", "runner must not leak exception text/paths")
                records = stdout.splitlines()
                self.assertEqual(len(records), 1, "exactly one terminal record required")
                summary = json.loads(records[0])
                self.assertEqual(summary, json.loads((output / "summary.json").read_text()))
                self.assertEqual(summary["ok"], expected_ok, summary)
                self.assertEqual(supervisor.returncode, 0 if expected_ok else 2)
                self.assertNotIn(str(output), stdout.decode())
                self.assertNotIn(sys.executable, stdout.decode())
                self.assertLessEqual(sum(path.stat().st_size for path in output.iterdir()),
                                     summary["limits"]["log_bytes"])
                self.assertLessEqual(summary["retained_stream_bytes"] + (output / "summary.json").stat().st_size,
                                     summary["limits"]["log_bytes"])
                self.assertLessEqual(summary["sampling"]["cadence_seconds"], 0.1)
                self.assertIsNotNone(summary["sampling"]["last_seconds"])
                # Only fixture-emitted identities are checked, never a host scan.
                for line in (output / "stdout.jsonl").read_bytes().splitlines():
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if isinstance(record, dict):
                        for key in ("pid", "child_pid"):
                            if key in record:
                                self.assert_not_running(record[key])
                self.assertFalse(summary["residual_identities"], summary)
                return summary

    def test_normal_and_final_wait4_replaces_live_cpu(self):
        summary = self.run_fixture("""\
            event('ready')
            event('episode_begin', id='one')
            until = time.process_time() + 0.15
            while time.process_time() < until:
                pass
            event('episode_end', id='one', ok=True)
            """, expected_ok=True)
        self.assertEqual(summary["reason"], "completed")
        self.assertEqual(summary["episodes"]["completed"], 1)
        root = summary["root_wait4"]
        final = summary["final_aggregate_cpu_seconds"]
        self.assertAlmostEqual(final, root["user_seconds"] + root["system_seconds"] +
                               summary["supervisor_cpu_seconds"], delta=0.01)
        self.assertFalse(summary["group_cleanup_unconfirmed"])

    def test_unattested_executable_is_validation_only(self):
        summary = self.run_fixture(NORMAL, attest=False)
        self.assertEqual(summary["reason"], "no_subprocess_contract_unattested")
        self.assertIsNone(summary["final_aggregate_cpu_seconds"])
        self.assertEqual(summary["classification"], "containment_validation_only")

    def test_cpu_loop(self):
        summary = self.run_fixture("""\
            event('ready')
            event('episode_begin', id='cpu')
            while True:
                pass
            """, options=("--cpu-seconds", "30.2"))
        self.assertEqual(summary["reason"], "cpu_cleanup_reserve")
        self.assertGreaterEqual(summary["final_aggregate_cpu_seconds"], 0.2)

    def test_sleeping_wall_timeout(self):
        summary = self.run_fixture("""\
            event('ready')
            event('episode_begin', id='sleep')
            time.sleep(60)
            """, options=("--wall-seconds", "30.4"))
        self.assertEqual(summary["reason"], "wall_cleanup_reserve")
        self.assertEqual(summary["episodes"]["unfinished"], 1)

    def test_ignore_term_requires_kill(self):
        summary = self.run_fixture("""\
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            event('ready')
            event('episode_begin', id='ignore')
            time.sleep(60)
            """, options=("--wall-seconds", "30.4"))
        self.assertEqual(summary["reason"], "wall_cleanup_reserve")
        self.assertEqual(summary["exit_code"], -signal.SIGKILL)

    def test_unexpected_same_group_child_survives_immediate_parent(self):
        summary = self.run_fixture("""\
            event('ready')
            event('episode_begin', id='fork')
            child = os.fork()
            if child == 0:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                time.sleep(60)
                os._exit(0)
            event('measurement', child_pid=child)
            # Let at least one ordinary observation see the owned child before
            # the immediate parent exits; cleanup still has to kill the child.
            time.sleep(0.25)
            os._exit(0)
            """)
        self.assertEqual(summary["reason"], "unexpected_child")
        self.assertIsNone(summary["final_aggregate_cpu_seconds"])
        self.assertTrue(summary["unwaitable_descendants"])

    def test_log_flood_is_bounded(self):
        summary = self.run_fixture("""\
            event('ready')
            event('episode_begin', id='flood')
            while True:
                os.write(2, b'x' * 4096)
            """, options=("--log-mib", "0.0625"))
        self.assertEqual(summary["reason"], "log_limit")
        self.assertLessEqual(summary["retained_stream_bytes"], 65536 - MODULE.RESERVE)

    def test_small_allocation_with_low_rss_threshold(self):
        summary = self.run_fixture("""\
            event('ready')
            event('episode_begin', id='rss')
            allocation = bytearray(32 * 1024 * 1024)
            time.sleep(60)
            """, options=("--rss-mib", "40"))
        self.assertEqual(summary["reason"], "rss_limit")
        self.assertGreaterEqual(summary["peak_sampled_rss_bytes"], 40 * MODULE.MIB)

    def test_malformed_record(self):
        summary = self.run_fixture("print('not-json', flush=True)\ntime.sleep(60)\n")
        self.assertEqual(summary["reason"], "malformed_record")

    def test_missing_episode_end(self):
        summary = self.run_fixture("event('ready')\nevent('episode_begin', id='unfinished')\n")
        self.assertEqual(summary["reason"], "unfinished_episode")

    def test_duplicate_episode(self):
        summary = self.run_fixture(NORMAL + "event('episode_begin', id='one')\n")
        self.assertEqual(summary["reason"], "duplicate_episode")

    def test_episode_limit_pre_admission(self):
        summary = self.run_fixture(NORMAL + "event('episode_begin', id='two')\n",
                                   options=("--max-episodes", "1"))
        self.assertEqual(summary["reason"], "episode_limit")
        self.assertEqual(summary["episodes"]["begun"], 1)
        self.assertEqual(summary["episodes"]["attempts"], 2)

    def test_false_episode_end_is_failure(self):
        summary = self.run_fixture("""\
            event('ready')
            event('episode_begin', id='failure')
            event('episode_end', id='failure', ok=False)
            """)
        self.assertEqual(summary["reason"], "episode_failure")
        self.assertEqual(summary["episodes"]["failed"], 1)

    def test_truncated_record(self):
        summary = self.run_fixture(NORMAL + "os.write(1, b'{')\n")
        self.assertEqual(summary["reason"], "truncated_record")

    def test_interrupt_uses_same_cleanup(self):
        summary = self.run_fixture("""\
            event('ready')
            event('episode_begin', id='interrupt')
            time.sleep(60)
            """, interrupt=True)
        self.assertEqual(summary["reason"], "interrupt")

    def test_child_only_core_limit(self):
        # The first measurement is retained as raw internal data. Verify the
        # child limit without changing or asserting any workstation setting.
        summary = self.run_fixture(NORMAL + """\
if resource.getrlimit(resource.RLIMIT_CORE) != (0, 0):
    os._exit(9)
""", expected_ok=True)
        self.assertEqual(summary["exit_code"], 0)

    def test_ready_and_episode_deadlines_without_long_sleep(self):
        args = type("Args", (), {"wall_seconds": 1800})()
        runner = MODULE.Runner(args)
        try:
            with mock.patch.object(runner, "elapsed", return_value=20), self.assertRaises(MODULE.Stop):
                runner.event(b'{"event":"ready"}')
            self.assertEqual(runner.reason, "ready_timeout")
            runner.reason = None
            runner.ready = True
            runner.active[("str", "late")] = time.monotonic() - 10.1
            with self.assertRaises(MODULE.Stop):
                runner.event(b'{"event":"episode_end","id":"late","ok":true}')
            self.assertEqual(runner.reason, "episode_timeout")
        finally:
            runner.selector.close()

    def test_missing_required_metric_stops(self):
        args = type("Args", (), {})()
        runner = MODULE.Runner(args)
        runner.proc = type("OwnedProcess", (), {"pid": os.getpid()})()
        with mock.patch.object(MODULE, "read_proc", side_effect=PermissionError()), self.assertRaises(MODULE.Stop):
            runner.sample()
        self.assertEqual(runner.reason, "required_metric_unavailable")
        runner.selector.close()

    def test_limits_cannot_be_raised_or_reserve_removed(self):
        for flag, value in (("--wall-seconds", "1801"), ("--cpu-seconds", "1801"),
                            ("--rss-mib", "769"), ("--log-mib", "129"),
                            ("--max-episodes", "601"), ("--wall-seconds", "30"),
                            ("--cpu-seconds", "30"), ("--wall-seconds", "nan")):
            with self.subTest(flag=flag, value=value), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                MODULE.arguments(["--output-dir", "unused", flag, value, "--", sys.executable])

    def mocked_runner(self):
        args = MODULE.arguments(["--output-dir", "unused", "--", sys.executable, "-c", NORMAL])
        runner = MODULE.Runner(args)
        self.addCleanup(runner.selector.close)
        return runner

    @staticmethod
    def proc_item(start=1, parent=0, group=111, state="R", cpu=0.0):
        return {"start": start, "parent": parent, "group": group, "state": state,
                "cpu": cpu, "rss": 4096}

    def test_escaped_verified_descendant_cleanup_uses_pidfd(self):
        runner = self.mocked_runner()
        runner.proc = SimpleNamespace(pid=111, stdout=None, stderr=None, returncode=None)
        parent = self.proc_item()
        child = self.proc_item(start=2, parent=111, group=222)
        runner.known[111] = parent
        clock = itertools.count(0.0, 0.2)
        usage = SimpleNamespace(ru_utime=1.0, ru_stime=0.5, ru_maxrss=10)

        def send(fd, sig, info, flags):
            if sig == signal.SIGKILL:
                child["state"] = "Z"

        with (mock.patch.object(MODULE, "read_proc", side_effect=lambda pid: child if pid == 222 else parent),
              mock.patch.object(MODULE.os, "pidfd_open", return_value=55, create=True),
              mock.patch.object(MODULE.signal, "pidfd_send_signal", side_effect=send, create=True) as send_signal,
              mock.patch.object(MODULE.os, "close") as close,
              mock.patch.object(runner, "group_signal") as group_signal,
              mock.patch.object(runner, "sample", return_value="Z"),
              mock.patch.object(runner, "drain_once"),
              mock.patch.object(MODULE.time, "monotonic", side_effect=lambda: next(clock)),
              mock.patch.object(MODULE.os, "wait4", return_value=(111, 0, usage)),
              mock.patch.object(MODULE.os, "killpg", side_effect=ProcessLookupError)):
            runner.capture_descendant(111, parent, 222)
            runner.cleanup()
            group_signal.assert_has_calls([mock.call(signal.SIGTERM), mock.call(signal.SIGKILL)])
            send_signal.assert_any_call(55, signal.SIGTERM, None, 0)
            send_signal.assert_any_call(55, signal.SIGKILL, None, 0)
            self.assertFalse(runner.residual)
            self.assertTrue(runner.descendants_observed)
            self.assertFalse(runner.aggregate_cpu_available())
            runner.close_pidfds()
            close.assert_called_once_with(55)

    def test_replaced_descendant_identity_is_never_signalled(self):
        runner = self.mocked_runner()
        runner.pidfds[222] = (55, 2)
        with (mock.patch.object(MODULE, "read_proc", return_value=self.proc_item(start=3)),
              mock.patch.object(MODULE.signal, "pidfd_send_signal", create=True) as send,
              mock.patch.object(MODULE.os, "close")):
            runner.signal_descendants(signal.SIGTERM)
            runner.signal_descendants(signal.SIGKILL)
            send.assert_not_called()
            runner.close_pidfds()

    def test_discovery_replacement_closes_handle_without_signalling(self):
        runner = self.mocked_runner()
        with (mock.patch.object(MODULE, "read_proc", side_effect=[
                  self.proc_item(start=2, parent=111), self.proc_item(start=3, parent=111), self.proc_item()]),
              mock.patch.object(MODULE.os, "pidfd_open", return_value=55, create=True),
              mock.patch.object(MODULE.signal, "pidfd_send_signal", create=True) as send,
              mock.patch.object(MODULE.os, "close") as close,
              self.assertRaises(MODULE.Stop)):
            try:
                runner.capture_descendant(111, self.proc_item(), 222)
            finally:
                self.assertFalse(runner.pidfds)
                send.assert_not_called()
                close.assert_called_once_with(55)
        self.assertEqual(runner.reason, "descendant_identity_unconfirmed")

    def test_missing_pidfd_has_no_bare_pid_fallback(self):
        runner = self.mocked_runner()
        with (mock.patch.object(MODULE, "read_proc", return_value=self.proc_item(start=2, parent=111)),
              mock.patch.object(MODULE.os, "pidfd_open", side_effect=OSError(), create=True),
              mock.patch.object(MODULE.signal, "pidfd_send_signal", create=True) as send,
              mock.patch.object(MODULE.os, "kill") as kill,
              self.assertRaises(MODULE.Stop)):
            runner.capture_descendant(111, self.proc_item(), 222)
        self.assertEqual(runner.reason, "descendant_handle_unavailable")
        self.assertTrue(runner.descendants_observed)
        send.assert_not_called()
        kill.assert_not_called()
        self.assertFalse(runner.pidfds)

    def test_cpu_replacement_arithmetic_is_exact(self):
        runner = self.mocked_runner()
        runner.proc = SimpleNamespace(pid=111)
        runner.known[111] = self.proc_item(cpu=113)
        runner.cpu_start = 3
        runner.final_root_cpu = 7
        with mock.patch.object(MODULE.time, "process_time", return_value=13):
            self.assertEqual(runner.aggregate_cpu(), 17)
            runner.final_root_cpu = None
            self.assertEqual(runner.aggregate_cpu(), 123)

    def test_unsampled_orphan_cannot_establish_aggregate_completeness(self):
        runner = self.mocked_runner()
        runner.proc = SimpleNamespace(pid=111)
        runner.known[111] = self.proc_item(state="Z")
        runner.final_root_cpu = 7
        runner.files["samples"] = io.BytesIO()
        with (mock.patch.object(MODULE, "read_proc", return_value=self.proc_item(state="Z")),
              mock.patch.object(MODULE, "scoped_children") as children):
            self.assertEqual(runner.sample(), "Z")
        children.assert_not_called()
        self.assertFalse(runner.descendants_observed)
        self.assertFalse(runner.aggregate_cpu_available())
        runner.contract_attested = True
        self.assertTrue(runner.aggregate_cpu_available())
        runner.descendants_observed = True
        self.assertFalse(runner.aggregate_cpu_available())

    def test_missing_metrics_independently_fail_closed(self):
        for missing in ("root", "supervisor", "children"):
            with self.subTest(missing=missing):
                runner = self.mocked_runner()
                runner.proc = SimpleNamespace(pid=111)

                def read(pid, missing=missing):
                    if (pid == 111 and missing == "root") or (pid != 111 and missing == "supervisor"):
                        raise PermissionError()
                    return self.proc_item()

                with (mock.patch.object(MODULE, "read_proc", side_effect=read),
                      mock.patch.object(MODULE, "scoped_children", side_effect=(
                          PermissionError() if missing == "children" else lambda pid: set())),
                      self.assertRaises(MODULE.Stop)):
                    runner.sample()
                self.assertEqual(runner.reason, "required_metric_unavailable")

    def test_shared_stdout_stderr_retention_and_record_bounds(self):
        runner = self.mocked_runner()
        runner.args.log_mib = 0.0625
        runner.files = {name: io.BytesIO() for name in ("stdout", "stderr", "samples")}
        runner.retain("stdout", b"x" * 30000)
        with self.assertRaises(MODULE.Stop):
            runner.retain("stderr", b"y" * 30000)
        self.assertEqual(runner.reason, "log_limit")
        self.assertEqual(runner.log_bytes, 65536 - MODULE.RESERVE)
        self.assertEqual(sum(len(stream.getvalue()) for stream in runner.files.values()), runner.log_bytes)
        for name in ("stdout", "stderr"):
            with self.subTest(stream=name):
                runner = self.mocked_runner()
                runner.args.log_mib = 0.0625
                runner.files = {stream: io.BytesIO() for stream in ("stdout", "stderr", "samples")}
                line = b'{"event":"measurement"}\n'
                chunk = line * (MODULE.READ_SIZE // len(line)) if name == "stdout" else b"x" * MODULE.READ_SIZE
                key = SimpleNamespace(fd=123, data=name, fileobj=None)
                with (mock.patch.object(runner.selector, "select", return_value=[(key, None)]),
                      mock.patch.object(MODULE.os, "read", return_value=chunk),
                      self.assertRaises(MODULE.Stop)):
                    for _ in range(64):
                        runner.drain_once()
                self.assertEqual(runner.reason, "log_limit")
                self.assertEqual(runner.log_bytes, 65536 - MODULE.RESERVE)
                self.assertLessEqual(len(runner.buffer), MODULE.RECORD_LIMIT + MODULE.READ_SIZE)
        for terminated in (False, True):
            with self.subTest(terminated=terminated):
                runner = self.mocked_runner()
                runner.files["stdout"] = io.BytesIO()
                raw = b"x" * (MODULE.RECORD_LIMIT + 1) + (b"\n" if terminated else b"")
                chunks = iter(raw[offset:offset + MODULE.READ_SIZE] for offset in range(0, len(raw), MODULE.READ_SIZE))
                key = SimpleNamespace(fd=123, data="stdout", fileobj=None)
                with (mock.patch.object(runner.selector, "select", return_value=[(key, None)]),
                      mock.patch.object(MODULE.os, "read", side_effect=lambda *args, chunks=chunks: next(chunks)),
                      self.assertRaises(MODULE.Stop)):
                    for _ in range(6):
                        runner.drain_once()
                self.assertEqual(runner.reason, "record_limit")
                self.assertLessEqual(len(runner.buffer), MODULE.RECORD_LIMIT + MODULE.READ_SIZE)

    def test_launch_interrupt_and_contract_mismatch_always_reach_cleanup(self):
        for mode in ("interrupt", "mismatch", "unattested", "root_metric", "supervisor_metric", "children_metric"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                runner = self.mocked_runner()
                runner.args.output_dir = str(Path(temporary) / "new")
                runner.args.reviewed_no_subprocess_contract = (
                    "b" * 64 if mode == "mismatch" else None if mode == "unattested" else "a" * 64)
                fixture = SimpleNamespace(pid=111, stdout=mock.Mock(), stderr=mock.Mock(), returncode=None)
                fixture.stdout.fileno.return_value = 123
                fixture.stderr.fileno.return_value = 124

                def launch(*args, mode=mode, runner=runner, fixture=fixture, **kwargs):
                    if mode == "interrupt":
                        # Arrives inside Popen, before its return value is assigned.
                        runner.interrupt(signal.SIGINT, None)
                    return fixture

                def read(pid, mode=mode):
                    if (mode == "root_metric" and pid == 111) or (mode == "supervisor_metric" and pid != 111):
                        raise PermissionError()
                    return self.proc_item()

                def children(pid, mode=mode):
                    if mode == "children_metric":
                        raise PermissionError()
                    return set()

                def observe(runner=runner, mode=mode):
                    if mode.endswith("_metric"):
                        return MODULE.Runner.sample(runner)
                    runner.ready = True
                    runner.seen.add(("str", "one"))
                    runner.completed = 1
                    runner.known[111] = self.proc_item(state="Z")
                    return "Z"

                def clean(runner=runner, fixture=fixture, mode=mode):
                    self.assertIs(runner.proc, None if mode == "mismatch" else fixture)
                    if runner.proc is not None:
                        runner.final_root_cpu = 0.01
                        runner.exit_code = 0
                        runner.proc.returncode = 0

                with (mock.patch.object(MODULE, "contract_identity", return_value=({"sha256": "c" * 64}, "a" * 64)),
                      mock.patch.object(MODULE.subprocess, "Popen", side_effect=launch) as popen,
                      mock.patch.object(MODULE.os, "pidfd_open", return_value=55, create=True),
                      mock.patch.object(MODULE.signal, "pidfd_send_signal", create=True),
                      mock.patch.object(MODULE.os, "close"),
                      mock.patch.object(MODULE.os, "set_blocking"),
                      mock.patch.object(MODULE, "read_proc", side_effect=read),
                      mock.patch.object(MODULE, "scoped_children", side_effect=children),
                      mock.patch.object(runner.selector, "register"),
                      mock.patch.object(runner, "sample", side_effect=observe),
                      mock.patch.object(runner, "cleanup", side_effect=clean) as cleanup,
                      mock.patch.object(MODULE, "terminal_record", return_value=True)):
                    self.assertEqual(runner.run(), 2)
                    cleanup.assert_called_once()
                    if mode == "mismatch":
                        popen.assert_not_called()
                    else:
                        popen.assert_called_once()
                summary = json.loads((Path(runner.args.output_dir) / "summary.json").read_text())
                self.assertFalse(summary["ok"])
                self.assertEqual(summary["reason"], {
                    "interrupt": "interrupt", "mismatch": "no_subprocess_contract_mismatch",
                    "unattested": "no_subprocess_contract_unattested", "root_metric": "required_metric_unavailable",
                    "supervisor_metric": "required_metric_unavailable", "children_metric": "required_metric_unavailable"}[mode])
                if mode == "unattested":
                    self.assertFalse(summary["no_subprocess_contract"]["descendants_observed"])
                    self.assertIsNone(summary["final_aggregate_cpu_seconds"])

    def test_contract_binds_executable_bytes_and_exact_argv(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "fixture"
            executable.write_bytes(b"one")
            first = MODULE.contract_identity([str(executable), "argument-one"])[1]
            changed_argv = MODULE.contract_identity([str(executable), "argument-two"])[1]
            executable.write_bytes(b"two")
            changed_executable = MODULE.contract_identity([str(executable), "argument-one"])[1]
        self.assertNotEqual(first, changed_argv)
        self.assertNotEqual(first, changed_executable)

    def test_output_directory_must_be_new(self):
        with tempfile.TemporaryDirectory() as existing:
            args = MODULE.arguments(["--output-dir", existing, "--", sys.executable, "-c", NORMAL])
            runner = MODULE.Runner(args)
            with self.assertRaises(FileExistsError):
                runner.run()
            runner.selector.close()


if __name__ == "__main__":
    unittest.main()
