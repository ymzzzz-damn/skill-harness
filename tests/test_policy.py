from __future__ import annotations
import builtins
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from harnesslab.policy import CodePolicy
from harnesslab.serialization import jsonable

@dataclass
class GraspReport:
    success: bool
    lifted_m: float
    reason: str

class Record:
    def __init__(self, tool="move", stateful=True):
        self.tool, self.stateful = tool, stateful
    def to_dict(self):
        return {"tool": self.tool, "stateful": self.stateful}

class Ledger:
    def __init__(self):
        self.records = []
    def __len__(self):
        return len(self.records)

class Backend:
    def __init__(self):
        self.steps = 0
    def env_steps(self):
        return self.steps

class Robo:
    ON_MOTION_IMAGES = ("head.png", "left.png")
    def __init__(self):
        self._ledger, self._backend = Ledger(), Backend()
        self.requested, self.stopped = [], []
    def move(self):
        self._ledger.records.append(Record())
        self._backend.steps += 3
        return {"position": [1, 2, 3]}
    def _summary_dict(self):
        return {"position": [1, 2, 3]}
    def _stop_episode(self, reason):
        self.stopped.append(reason)
    def show(self, camera):
        self.requested.append(camera)
    def _take_show_requests(self):
        result, self.requested = self.requested, []
        return result
    def _show_artifacts(self, requested):
        return [name + ".png" for name in requested]
    def artifact(self, name):
        return b"fake png: " + name.encode()

def policy(robo, workspace, **kwargs):
    return CodePolicy(robo, workspace, guard_factory=lambda _: vars(builtins).copy(),
                      audit=lambda code, **_: [], blocked=lambda code: [], **kwargs)

class PolicyTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[2] / "experiments/cpu-tests/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / "workspace"
        self.robo = Robo()
        self.policy = policy(self.robo, self.workspace)

    def execute(self, source):
        return self.policy.execute_tool("python", {"code": source})

    def test_dataclass_feedback_and_persistent_variables(self):
        self.policy._globals["report"] = GraspReport(True, .12, "visible object lift")
        result = self.execute("counter = 4\nreport")
        self.assertEqual(result.result["value"], {"success": True, "lifted_m": .12,
                                                "reason": "visible object lift"})
        self.assertEqual(self.execute("counter + 5").result["value"], 9)
        self.assertTrue((self.workspace.parent / "tool_feedback/001.json").exists())
        self.assertEqual(len(self.policy.turns), 2)

    def test_partial_action_exception_keeps_state_and_images(self):
        result = self.execute("robo.move()\nraise ValueError('failed grasp')")
        self.assertEqual(result.result["exception"]["type"], "ValueError")
        self.assertEqual(result.result["physical"]["native_actions_used"], 3)
        self.assertEqual(result.result["physical"]["primitive_calls"][0]["tool"], "move")
        self.assertEqual(len(result.images), 2)
        self.assertTrue((self.workspace.parent / "tool_feedback/001_00.png").exists())
        self.assertFalse(self.policy.turns[0]["ok"])

    def test_budget_does_not_execute_extra_actions_and_stops_once(self):
        self.policy = policy(self.robo, self.workspace, max_turns=1)
        self.execute("robo.move()")
        self.assertIn("error", self.execute("robo.move()").result)
        self.execute("robo.move()")
        self.assertEqual(self.robo._backend.steps, 3)
        self.assertEqual(self.robo.stopped, ["call_budget"])
        self.assertEqual(len(self.policy.turns), 1)

    def test_finish_is_agent_claim_and_blocks_later_actions(self):
        finish = self.policy.execute_tool("finish", {"status": "success", "summary": "done"})
        self.assertEqual(finish.result["agent_finish"]["status"], "success")
        self.assertNotEqual(finish.result["native_success"], True)
        self.assertIn("error", self.execute("robo.move()").result)
        self.assertEqual(self.robo._backend.steps, 0)

    def test_pure_feedback_still_returns_grasp_report_and_requested_images(self):
        self.policy = policy(self.robo, self.workspace, feedback="pure", images="on-demand")
        result = self.execute("robo.show('head')\n{'success': False, 'reason': 'empty'}")
        self.assertEqual(result.result["value"]["reason"], "empty")
        self.assertNotIn("physical", result.result)
        self.assertEqual(result.result["images"], ["head.png"])

    def test_namespace_does_not_transfer_between_scenes(self):
        self.execute("private_memory = 42")
        second = policy(Robo(), self.workspace.parent / "other" / "workspace")
        result = second.execute_tool("python", {"code": "private_memory"})
        self.assertEqual(result.result["exception"]["type"], "NameError")

    def test_display_does_not_leak_or_break_on_second_cell(self):
        first = self.execute("display(b'first')")
        second = self.execute("display(b'second')")
        third = self.execute("1 + 1")
        self.assertEqual(first.images, [b"first"])
        self.assertEqual(second.images, [b"second"])
        self.assertEqual(third.images, [])

    def test_invalid_finish_does_not_end_scene(self):
        result = self.policy.execute_tool("finish", {"status": "invented"})
        self.assertIn("error", result.result)
        self.assertIsNone(self.policy.finish)

    def test_json_serialization_does_not_emit_nan(self):
        self.assertEqual(jsonable({"x": float("nan"), "y": float("inf")}),
                         {"x": None, "y": None})

    def test_sandbox_refusal_does_not_run_physics(self):
        self.policy.blocked = lambda _: ["__globals__"]
        result = self.execute("robo.move()")
        self.assertEqual(result.result["exception"]["type"], "PermissionError")
        self.assertEqual(self.robo._backend.steps, 0)

    def test_syntax_error_is_feedback_and_next_cell_can_recover(self):
        self.assertEqual(self.execute("if").result["exception"]["type"], "SyntaxError")
        self.assertEqual(self.execute("3").result["value"], 3)

if __name__ == "__main__":
    unittest.main()
