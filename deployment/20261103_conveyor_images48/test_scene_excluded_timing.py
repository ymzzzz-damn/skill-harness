from __future__ import annotations
import copy
import datetime as dt
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import scene_excluded_timing as timing


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out = self.root / "full/conditions/layout_03/episode_001"
        self.epoch = dt.datetime(2026, 11, 2, 12, tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp()
        self.result = {"status": "completed", "task": timing.TASK, "seed": 3,
            "tool_set": "pyrua", "task_conditions": True, "error": None,
            "task_wall_s": 100.0, "task_card_ready_at": self.epoch + 17,
            "startup_s": 120.0, "total_wall_s": 225.0,
            "task_timing": {"end_observed": True}, "native_episode_over": True,
            "environment_success": False, "native_outcome": {"valid": True, "success": False}}
        self.receipt = {"first_observation_seen": True, "first_rgb_seen": True,
            "native_boot_completed": True, "model_phase_entered": True,
            "native_client_cleanup_confirmed": True, "final_result_status": "completed",
            "first_rgb_at": "2026-11-02T12:00:00+08:00", "pod_uid": "pod-uid", "node": "node"}
        self.manifest = {"arm": "conditions", "phase": "full", "layout": 3,
            "task_conditions": True, "pod": {"uid": "pod-uid", "node": "node"}}
        self.native = {"eval_time": 1, "score": 0, "success_rate": 0.0,
            "details": {"0": {"layout_id": 3, "success": False, "score": 0.0}}}
        self.save()

    def save(self):
        write(self.out / "result.json", self.result)
        write(self.out / "initialization-receipt.json", self.receipt)
        write(self.out / "experiment.json", self.manifest)
        write(self.out / "bench/eval_result/RoboDojo" / timing.TASK / "_result.json", self.native)

    def row(self, official=None):
        return timing.inspect(self.root, "conditions", 3, official)

    def test_formula_preserves_original_duration_and_includes_failed_native_task(self):
        before = {str(path): path.read_bytes() for path in self.out.rglob("*") if path.is_file()}
        row = self.row({"valid": True, "result_sha256": timing.sha(self.out / "result.json")})
        self.assertTrue(row["timing_valid_completed"])
        self.assertEqual(row["task_wall_s"], 100)
        self.assertEqual(row["tool_ready_overhead_s"], 17)
        self.assertEqual(row["scene_excluded_total_s"], 117)
        report = timing.summarize([row])
        self.assertEqual(report["completed_n"], 1)
        self.assertEqual(report["protocol_valid_completed_n"], 1)
        self.assertEqual(report["native_failure_n"], 1)
        self.assertEqual(report["timing_valid_completed_metrics"]["scene_excluded_total_s"],
                         {"n": 1, "sum": 117, "mean": 117, "median": 117})
        self.assertEqual(before, {str(path): path.read_bytes() for path in self.out.rglob("*") if path.is_file()})

    def test_stale_or_missing_official_validity_is_unknown_not_valid(self):
        for official in (None, {"valid": True, "result_sha256": "stale"}):
            with self.subTest(official=official):
                row = self.row(official)
                self.assertTrue(row["timing_valid_completed"])
                self.assertIsNone(row["protocol_valid"])
                counts = timing.summarize([row])
                self.assertEqual(counts["protocol_valid_completed_n"], 0)
                self.assertEqual(counts["protocol_validity_unavailable_n"], 1)

    def test_negative_nonfinite_missing_or_inconsistent_times_do_not_produce_estimate(self):
        cases = (("task_wall_s", -1), ("task_wall_s", float("nan")),
            ("task_card_ready_at", float("inf")), ("task_card_ready_at", True),
            ("task_card_ready_at", self.epoch - 1),
            ("task_card_ready_at", self.epoch + 130), ("task_card_ready_at", None),
            ("startup_s", -1), ("total_wall_s", 100))
        original = copy.deepcopy(self.result)
        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.result = copy.deepcopy(original)
                self.result[key] = value
                self.save()
                row = self.row()
                self.assertFalse(row["timing_valid_completed"])
                self.assertIsNone(row["tool_ready_overhead_s"])
                self.assertIsNone(row["scene_excluded_total_s"])

    def test_receipt_flags_timezone_and_identity_are_required(self):
        cases = (("first_rgb_seen", False), ("native_client_cleanup_confirmed", False),
            ("first_rgb_at", "2026-11-02T12:00:00"), ("first_rgb_at", "invalid"),
            ("first_rgb_at", None), ("pod_uid", "other"), ("final_result_status", "running"))
        original = copy.deepcopy(self.receipt)
        for key, value in cases:
            with self.subTest(key=key):
                self.receipt = copy.deepcopy(original)
                self.receipt[key] = value
                self.save()
                self.assertIsNone(self.row()["scene_excluded_total_s"])

    def test_running_unobserved_end_or_invalid_native_report_never_yields_total(self):
        for mutation in ("running", "end", "native", "protocol"):
            with self.subTest(mutation=mutation):
                self.result["status"] = "completed"
                self.result["task_timing"]["end_observed"] = True
                self.native["details"]["0"]["layout_id"] = 3
                official = None
                if mutation == "running": self.result["status"] = "running"
                if mutation == "end": self.result["task_timing"]["end_observed"] = False
                if mutation == "native": self.native["details"]["0"]["layout_id"] = 4
                self.save()
                if mutation == "protocol":
                    official = {"valid": False, "result_sha256": timing.sha(self.out / "result.json")}
                self.assertIsNone(self.row(official)["scene_excluded_total_s"])

    def test_denominators_do_not_turn_pending_or_incomplete_episodes_into_zero_seconds(self):
        write(self.root / "batch_status.json", {"status": "running"})
        report = timing.build(self.root)
        self.assertEqual(report["arms"]["conditions"]["expected_n"], 55)
        self.assertEqual(report["arms"]["conditions"]["attempted_n"], 1)
        self.assertEqual(report["arms"]["conditions"]["pending_n"], 54)
        self.assertEqual(report["arms"]["control"]["completed_n"], 0)
        self.assertIsNone(report["arms"]["control"]["timing_valid_completed_metrics"]["scene_excluded_total_s"]["sum"])
        self.assertIn("Mixed wall-clock", report["method"]["limitations"][0])
        json.dumps(report, allow_nan=False)

    def test_once_cli_writes_only_supplement(self):
        target = self.root / "scene-excluded-timing.json"
        completed = subprocess.run([timing.sys.executable, timing.__file__, "--root", str(self.root),
                                    "--out", str(target)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(target.read_text())["schema"], "scene-excluded-timing-supplement/v1")
        protected = subprocess.run([timing.sys.executable, timing.__file__, "--root", str(self.root),
                                   "--out", str(self.out / "result.json")], capture_output=True, text=True)
        self.assertNotEqual(protected.returncode, 0)

    def test_watch_stops_on_batch_terminal_state_without_requiring_a_live_finished_pod(self):
        target = self.root / "supplement.json"
        for state, expected in (("finished", 0), ("needs_attention", 2)):
            write(self.root / "batch_status.json", {"status": state})
            code = timing.watch(self.root, target, liveness=lambda *args: self.fail("terminal batch queried Pod"))
            self.assertEqual(code, expected)
            self.assertEqual(json.loads(target.read_text())["watch"]["status"], state)

    def test_watch_is_30_seconds_bounded_and_exits_on_inactive_pod(self):
        write(self.root / "batch_status.json", {"status": "running"})
        target = self.root / "supplement.json"
        ticks = iter((0, 0, 86400))
        sleeps = []
        code = timing.watch(self.root, target, monotonic=lambda: next(ticks), sleep=sleeps.append,
                            liveness=lambda *args: {"phase": "Running"})
        self.assertEqual(code, 3)
        self.assertEqual(sleeps, [30])
        self.assertIn("24-hour", json.loads(target.read_text())["watch"]["error"])
        write(self.root / "batch_status.json", {"status": "finished"})
        ticks = iter((0, 86401))
        self.assertEqual(timing.watch(self.root, target, monotonic=lambda: next(ticks)), 3)
        write(self.root / "batch_status.json", {"status": "running"})
        def inactive(*args): raise RuntimeError("inactive Pod")
        code = timing.watch(self.root, target, liveness=inactive)
        self.assertEqual(code, 3)
        self.assertIn("inactive", json.loads(target.read_text())["watch"]["error"])

    def test_live_pod_check_is_read_only_and_rejects_a_replacement(self):
        identity = {"pod": "example", "uid": "pod-uid", "namespace": "test", "node": "node"}
        write(self.root / "preflight/pod-identity.json", identity)
        pod = {"metadata": {"name": "example", "uid": "pod-uid", "namespace": "test"},
               "spec": {"nodeName": "node"}, "status": {"phase": "Running"}}
        response = types.SimpleNamespace(returncode=0, stdout=json.dumps(pod), stderr="")
        with patch.object(timing.subprocess, "run", return_value=response) as command:
            self.assertEqual(timing.check_live_pod(self.root)["uid"], "pod-uid")
            self.assertEqual(command.call_args.args[0], ["kubectl", "-n", "test", "get", "pod", "example", "-o", "json"])
        pod["metadata"]["uid"] = "replacement"
        response.stdout = json.dumps(pod)
        with patch.object(timing.subprocess, "run", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "inactive, replaced"):
                timing.check_live_pod(self.root)


if __name__ == "__main__":
    unittest.main()
