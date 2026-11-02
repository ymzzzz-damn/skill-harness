"""CPU proof that condition enforcement lives below robot aliases/helpers.

These fixtures have the same native-toolkit -> backend -> robot dispatch shape;
no primitive implementation is replaced and no simulator/model is imported.
"""
from __future__ import annotations
import builtins
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from harnesslab.policy import CodePolicy
from harnesslab.runner import _guard_native_actions


class Record:
    def __init__(self, name): self.tool, self.stateful = name, True
    def to_dict(self): return {"tool": self.tool, "stateful": True}


class Ledger:
    def __init__(self): self.records = []
    def __len__(self): return len(self.records)


class EpisodeFinished(BaseException):
    def __init__(self): self.reason = "native_success"


class NativeToolkit:
    def __init__(self):
        self.steps = 0
        self.calls = []
        self.failure = None

    def execute_tool(self, name, kwargs):
        self.calls.append((name, kwargs))
        if name == "move_pose":
            # A native handler can alter physical state before raising.
            self.steps += 2
            if self.failure is not None:
                raise self.failure
            return {"reached": True}
        return {"camera": "head", "visible": "basket"}


class Backend:
    def __init__(self, native): self.native = native
    def env_steps(self): return self.native.steps
    def call(self, name, kwargs): return self.native.execute_tool(name, kwargs)


class Robot:
    ON_MOTION_IMAGES = ()
    def __init__(self, backend):
        self._backend, self._ledger = backend, Ledger()
    def move(self, value):
        # The pristine SDK-style primitive delegates to the backend. The guard
        # must catch this even if a caller already saved the bound method.
        try:
            return self._backend.call("move_pose", {"position": value})
        finally:
            self._ledger.records.append(Record("move_pose"))
    def observe(self): return self._backend.call("camera_image", {})
    def _summary_dict(self): return {"env_steps": self._backend.env_steps()}
    def _take_show_requests(self): return []


def plan():
    return {"conditions": [{"id": "placed", "description": "target in basket",
        "kind": "final", "mode": "state", "source": "instruction",
        "source_text": "put the target in the basket",
        "evidence_required": "current RGB shows the target inside the basket"}],
        "subtasks": [{"id": "place", "description": "place target in basket",
                      "preconditions": [], "completion": ["placed"], "maintain": []}],
        "simultaneous": [], "ambiguities": []}


class NativeGuardTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[2] / "experiments/cpu-tests/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.native = NativeToolkit()
        self.backend = Backend(self.native)
        self.robot = Robot(self.backend)
        self.original_move = Robot.move
        self.alias_before_hook = self.robot.move
        self.policy = CodePolicy(self.robot, Path(self.temp.name) / "workspace",
            task_conditions=True, images="none",
            guard_factory=lambda _: vars(builtins).copy(),
            audit=lambda code, **_: [], blocked=lambda code: [])
        self.policy._globals["move_alias"] = self.alias_before_hook
        native_module = types.ModuleType("pyrualean.hosts.robodojo_native")
        native_module.STATEFUL = {"move_pose", "gripper"}
        with patch.dict("sys.modules", {native_module.__name__: native_module}):
            _guard_native_actions(self.native, self.backend, self.policy)

    def execute(self, code, **metadata):
        return self.policy.execute_tool("python", {"code": code, **metadata}).result

    @staticmethod
    def intent(): return {"subtask": "place", "purpose": "place object", "preserve": []}

    def events(self, kind):
        return [event for event in self.policy.contract.events if event["event"] == kind]

    def test_observations_are_allowed_before_any_task_plan(self):
        result = self.execute("robo.observe()")
        self.assertNotIn("exception", result)
        self.assertEqual(result["value"]["visible"], "basket")
        self.assertEqual(self.native.steps, 0)
        self.assertFalse(self.events("physical_dispatch"))
        self.assertFalse(self.events("physical_return"))

    def test_saved_alias_and_indirect_helper_cannot_bypass_missing_plan(self):
        result = self.execute("def indirect():\n    return move_alias([1, 2, 3])\nindirect()")
        self.assertEqual(result["exception"]["type"], "ContractError")
        self.assertIn("register task_plan first", result["exception"]["message"])
        self.assertEqual(self.native.calls, [])
        self.assertEqual(self.native.steps, 0)
        self.assertFalse(self.events("physical_return"))
        self.assertIs(Robot.move, self.original_move)
        self.assertIs(self.robot.move.__func__, self.alias_before_hook.__func__)

    def test_plan_without_intent_rejects_before_native_handler_or_after_hook(self):
        result = self.execute("move_alias([1, 2, 3])", task_plan=plan())
        self.assertEqual(result["exception"]["type"], "ContractError")
        self.assertIn("task_intent", result["exception"]["message"])
        self.assertEqual(self.native.calls, [])
        self.assertFalse(self.events("physical_dispatch"))
        self.assertFalse(self.events("physical_return"))

    def test_authorized_alias_and_helper_execute_pristine_robot_primitives(self):
        result = self.execute("def indirect():\n    return move_alias([1, 2, 3])\n"
                              "indirect()\nrobo.move([4, 5, 6])",
                              task_plan=plan(), task_intent=self.intent())
        self.assertNotIn("exception", result)
        self.assertEqual(self.native.steps, 4)
        self.assertEqual(len(self.events("physical_dispatch")), 2)
        self.assertEqual(len(self.events("physical_return")), 2)
        self.assertEqual([(event["steps_before"], event["steps_after"])
                          for event in self.events("physical_return")], [(0, 2), (2, 4)])
        self.assertEqual(result["physical"]["native_actions_used"], 4)
        self.assertIs(Robot.move, self.original_move)
        self.assertEqual(self.policy.contract.epoch, 2)

    def test_partial_native_handler_exception_runs_after_hook_and_preserves_evidence(self):
        self.native.failure = RuntimeError("native motion failed after moving")
        result = self.execute("move_alias([1, 2, 3])", task_plan=plan(), task_intent=self.intent())
        self.assertEqual(result["exception"]["type"], "RuntimeError")
        self.assertEqual(self.native.steps, 2)
        self.assertEqual(result["physical"]["native_actions_used"], 2)
        self.assertTrue(result["physical"]["moved"])
        self.assertEqual(len(self.events("physical_return")), 1)
        self.assertEqual(self.events("physical_return")[0]["steps_after"], 2)
        self.assertEqual(self.policy.contract.assessments["placed"]["status"], "unknown")
        self.assertNotIn("exception", self.execute("robo.observe()"))

    def test_native_success_stop_is_not_relabelled_as_budget_or_control_exception(self):
        self.native.failure = EpisodeFinished()
        result = self.execute("move_alias([1, 2, 3])", task_plan=plan(), task_intent=self.intent())
        self.assertEqual(result["episode_stopped"], "native_success")
        self.assertNotIn("exception", result)
        self.assertEqual(self.native.steps, 2)
        self.assertEqual(len(self.events("physical_return")), 1)
        self.assertFalse(self.policy.turn_budget_exhausted)


if __name__ == "__main__":
    unittest.main()
