from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from harnesslab import runner

RECONNECT = {"type": "error", "message": "Reconnecting... 1/10 (rate limit exceeded)"}
CONTENT_RECONNECT = {"type": "error", "message":
                     "Reconnecting... 1/10 (stream disconnected: content_filter)"}
PROGRESS = {"type": "item.completed", "item": {
            "type": "mcp_tool_call", "status": "completed", "error": None, "tool": "python"}}

class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[2] / "experiments/cpu-tests/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.order = []

    def run_fake(self, native_success=False, failure=False, *,
                 generation=None, events=None, agent_finish=True,
                 native_end=False, report=True, report_layout=0, budget=False):
        order, root = self.order, self.root
        class Ledger:
            records = []
            def __len__(self): return 0
            def stateful_calls(self): return 0
            def env_steps(self): return 0
            def write_jsonl(self, path): path.write_text("")
        class Native:
            pass
        native = Native()
        class Backend:
            over = False
            def conclude(self):
                order.append("native_conclude")
                self.over = True
            def solved(self): order.append("native_score"); return native_success
            def env_steps(self): return 7
            def episode_over(self): return self.over
        backend = Backend()
        class Robot:
            def __init__(self, current_backend, **kw):
                if not getattr(native, "installed", False):
                    raise AssertionError("strong_grasp must install before Robot")
                order.append("robot")
        class Host:
            def boot(self, cell, out, **kw):
                order.append("native_boot")
                return native, []
            def make_backend(self, current): return backend
            def dump_card(self, current, cell, out):
                order.append("initial_observation")
                return types.SimpleNamespace(task_language="lift object", max_env_steps=200,
                                             images={"head": "head.png"})
        host = Host()
        adapter = types.SimpleNamespace(host_module=lambda: host, robot_cls=Robot)
        class Policy:
            finish = None
            turns = []
            sandbox_flags = []
            turn_budget_exhausted = budget
            def __init__(self, robo, workspace, **kw):
                order.append("fresh_code_policy")
                self.robo = robo
        class Server:
            def __init__(self, toolkit, **kw): order.append("mcp")
            def start(self): return "http://127.0.0.1:1/mcp/"
            def stop(self): order.append("mcp_stop")
        def install(current, robot_cls):
            order.append("install_grasp")
            current.installed = True
        def agent(prompt, images, **kw):
            order.append("agent")
            if failure:
                raise RuntimeError("provider disconnected")
            if events is not None:
                (root / "episode/codex_events.jsonl").write_text(
                    "\n".join(json.dumps(event) for event in events) + "\n")
            if agent_finish:
                kw["toolkit"].finish = {"status": "success", "summary": "agent claim"}
            backend.over = native_end
            kw["should_stop"]()
            return copy.deepcopy(generation or {"status": "completed", "returncode": 0, "errors": []})
        def cleanup(*args):
            order.append("native_cleanup")
            # Actual client writes the final report while stopping. This fixture
            # catches regressions that classify before that write is complete.
            if report:
                path = root / "episode/bench/eval_result/RoboDojo/general_pickup/run/_result.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"eval_time": 1, "score": 100 if native_success else 0,
                    "success_rate": float(native_success),
                    "details": {"0": {"layout_id": report_layout, "success": native_success,
                                       "score": float(native_success)}}}))
        def module(name, **attrs):
            result = types.ModuleType(name)
            result.__file__ = str(root / name.replace(".", "_") / "__init__.py")
            Path(result.__file__).parent.mkdir(exist_ok=True)
            Path(result.__file__).write_text("")
            result.__dict__.update(attrs)
            return result
        modules = {
            "pyrualean": module("pyrualean"),
            "pyrualean._backend": module("pyrualean._backend", Ledger=Ledger),
            "pyrualean.hosts.common": module("pyrualean.hosts.common", stop_all=cleanup),
            "pyrualean.mcp_server": module("pyrualean.mcp_server", ArmMcpServer=Server),
            "pyrualean.robots": module("pyrualean.robots", get_adapter=lambda _: adapter),
            "strong_grasp": module("strong_grasp", install=install, grasp=lambda *a, **kw: None),
        }
        cell = types.SimpleNamespace(task="general_pickup", suite="arx_x5", seed=0,
                                     cuda_device=0)
        with patch.dict("sys.modules", modules), patch.object(runner, "CodePolicy", Policy), \
             patch.object(runner, "build_prompt", lambda *a, **kw: "prompt"), \
             patch.object(runner, "run_codex", agent):
            return runner.run_episode(cell, root / "episode", model="fixed-model")

    def test_native_success_is_separate_from_agent_finish(self):
        result = self.run_fake(native_success=False)
        self.assertEqual(result["finish"]["status"], "success")
        self.assertFalse(result["environment_success"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["env_steps_total"], 7)
        self.assertTrue(result["native_outcome"]["valid"])
        self.assertEqual(self.order[:5],
                         ["native_boot", "install_grasp", "initial_observation",
                          "robot", "fresh_code_policy"])
        self.assertLess(self.order.index("agent"), self.order.index("native_conclude"))
        self.assertLess(self.order.index("native_conclude"), self.order.index("native_score"))
        self.assertIn("native_cleanup", self.order)
        saved = json.loads((self.root / "episode/result.json").read_text())
        self.assertEqual(saved["system"], "harness")
        self.assertTrue(saved["native_outcome"]["valid"])

    def test_agent_error_finalizes_result_without_fake_success(self):
        result = self.run_fake(failure=True)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["message"], "provider disconnected")
        self.assertFalse(result["environment_success"])
        self.assertIn("mcp_stop", self.order)
        self.assertIn("native_cleanup", self.order)

    def test_no_implicit_retry_or_episode_directory_reuse(self):
        self.run_fake()
        with self.assertRaises(FileExistsError):
            self.run_fake()
        self.assertEqual(self.order.count("agent"), 1)

    def test_invalid_budget_fails_before_native_boot(self):
        cell = types.SimpleNamespace()
        for kwargs in ({"budget_s": float("inf")}, {"max_turns": True}):
            with self.assertRaises(ValueError):
                runner.run_episode(cell, self.root / "episode", model="fixed", **kwargs)
        self.assertFalse(self.order)

    def test_unsupported_runtime_fails_before_native_boot(self):
        with self.assertRaises(ValueError):
            runner.run_episode(types.SimpleNamespace(), self.root / "episode",
                               model="fixed", runtime="openai")
        self.assertFalse(self.order)

    def test_recovered_retry_native_success_and_expected_sigterm_complete(self):
        result = self.run_fake(native_success=True, agent_finish=False, native_end=True,
            generation={"status": "completed", "returncode": -15, "errors": [RECONNECT]},
            events=[RECONNECT, PROGRESS])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["environment_success"])
        self.assertTrue(result["native_end_observed_during_agent"])
        self.assertEqual(result["generation"]["errors"], [RECONNECT])  # Raw history preserved.
        diagnostics = result["planner_diagnostics"]
        self.assertEqual(len(diagnostics["transport_warnings"]), 1)
        self.assertFalse(diagnostics["fatal_errors"])
        self.assertEqual(diagnostics["transport_warnings"][0]["recovered_by_event_line"], 2)

    def test_recovered_retry_native_failure_is_also_completed(self):
        result = self.run_fake(native_success=False, agent_finish=False, native_end=True,
            generation={"status": "completed", "returncode": -15, "errors": [CONTENT_RECONNECT]},
            events=[CONTENT_RECONNECT, PROGRESS])
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["environment_success"])
        self.assertEqual(len(result["planner_diagnostics"]["transport_warnings"]), 1)

    def test_unrecovered_retry_not_hidden_by_native_success(self):
        result = self.run_fake(native_success=True, agent_finish=False, native_end=True,
            generation={"status": "completed", "returncode": -15, "errors": [RECONNECT]},
            events=[PROGRESS, RECONNECT])
        self.assertEqual(result["status"], "planner_error")
        self.assertTrue(result["environment_success"])
        self.assertEqual(result["planner_diagnostics"]["fatal_errors"][0]["reason"],
                         "unrecovered_reconnect")

    def test_fatal_turn_failed_stays_error_despite_native_success_or_agent_finish(self):
        fatal = {"type": "turn.failed", "error": {"message": "provider terminal failure"}}
        result = self.run_fake(native_success=True, native_end=True,
            generation={"status": "completed", "returncode": 0, "errors": [fatal]},
            events=[fatal, PROGRESS])
        self.assertEqual(result["status"], "planner_error")
        self.assertTrue(result["planner_diagnostics"]["fatal_errors"])

    def test_unknown_error_with_later_progress_is_not_silently_downgraded(self):
        fatal = {"type": "error", "message": "malformed response fatal"}
        result = self.run_fake(native_success=False,
            generation={"status": "completed", "returncode": 0, "errors": [fatal]},
            events=[fatal, PROGRESS])
        self.assertEqual(result["status"], "planner_error")
        self.assertFalse(result["planner_diagnostics"]["transport_warnings"])

    def test_missing_report_does_not_qualify_recovered_retry(self):
        result = self.run_fake(native_success=True, report=False, native_end=True,
            generation={"status": "completed", "returncode": 0, "errors": [RECONNECT]},
            events=[RECONNECT, PROGRESS])
        self.assertEqual(result["status"], "native_result_error")
        self.assertFalse(result["native_outcome"]["valid"])

    def test_wrong_layout_report_is_not_trusted(self):
        result = self.run_fake(report_layout=3)
        self.assertEqual(result["status"], "native_result_error")
        self.assertIn("native layout_id mismatch", result["native_outcome"]["errors"])

    def test_sigterm_without_observed_native_end_or_finish_is_unexpected(self):
        result = self.run_fake(native_success=True, agent_finish=False, native_end=False,
            generation={"status": "completed", "returncode": -15, "errors": []})
        self.assertEqual(result["status"], "planner_error")
        self.assertEqual(result["planner_diagnostics"]["fatal_errors"][0]["reason"],
                         "unexpected_planner_process_exit")

    def test_nonzero_process_exit_not_hidden_by_native_success(self):
        result = self.run_fake(native_success=True, native_end=True,
            generation={"status": "completed", "returncode": 1, "errors": []})
        self.assertEqual(result["status"], "planner_error")

    def test_timeout_is_reported_separately_from_native_task_outcome(self):
        result = self.run_fake(native_success=False, agent_finish=False,
            generation={"status": "timeout", "returncode": -15, "errors": []})
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["native_outcome"]["valid"])

    def test_code_budget_is_reported_without_a_success_score_gate(self):
        result = self.run_fake(native_success=True, budget=True)
        self.assertEqual(result["status"], "call_budget")
        self.assertTrue(result["environment_success"])

if __name__ == "__main__":
    unittest.main()
