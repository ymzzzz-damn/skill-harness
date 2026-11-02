from __future__ import annotations
import copy
import sys
import types
import unittest
from unittest.mock import patch
from harnesslab.task_contract import ContractError, TaskContract


def example_plan():
    def condition(key, kind, mode="state", **kw):
        return {"id": key, "description": key.replace("_", " "), "kind": kind,
                "mode": mode, "source": "instruction" if kind == "final" else "inferred",
                "source_text": "task wording" if kind == "final" else "necessary support for the task",
                "evidence_required": "relevant public observation", **kw}
    return {"conditions": [condition("goal", "final"), condition("ready", "precondition"),
                            condition("achieved", "completion", "event"),
                            condition("support", "maintain"),
                            condition("tidy", "cleanup", required=False)],
            "subtasks": [{"id": "prepare", "description": "establish support",
                          "preconditions": [], "completion": ["achieved"], "maintain": []},
                         {"id": "place", "description": "establish final relation",
                          "preconditions": ["ready", "achieved"], "completion": ["goal"],
                          "maintain": ["support"]}],
            "simultaneous": [["goal", "support"]],
            "ambiguities": ["Retain necessary support until the final relation has been checked."]}


def assessment(key, cell=0, status="satisfied"):
    return {"id": key, "status": status,
            "evidence": [] if status == "unknown" else [{"cell": cell, "detail": "public observation supports this assessment"}]}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = TaskContract()
        self.contract.record_evidence(0, 0, usable=True, images=["initial RGB"])
        self.contract.register(example_plan(), cell=1, step=0)

    def test_register_all_categories_and_record_ambiguity(self):
        snapshot = self.contract.snapshot()
        self.assertEqual({x["kind"] for x in snapshot["plan"]["conditions"]},
                         {"final", "precondition", "completion", "maintain", "cleanup"})
        self.assertEqual(snapshot["plan"]["ambiguities"], example_plan()["ambiguities"])
        self.assertTrue(all(x["status"] == "unknown" for x in snapshot["assessments"].values()))

    def test_failed_plan_replacement_is_atomic(self):
        before = copy.deepcopy(self.contract.snapshot())
        plan = example_plan()
        plan["subtasks"][0]["completion"] = ["invented"]
        with self.assertRaises(ContractError):
            self.contract.register(plan, cell=2, step=0)
        self.assertEqual(before, self.contract.snapshot())

    def test_events_persist_but_current_state_expires_on_dispatch(self):
        self.contract.update([assessment("goal"), assessment("achieved")], cell=1, step=0)
        self.contract.motion(cell=2, step=0, tool="native_move")
        self.assertEqual(self.contract.assessments["goal"]["status"], "unknown")
        self.assertTrue(self.contract.current("achieved", 3))
        self.assertFalse(self.contract.current("goal", 3))
        with self.assertRaisesRegex(ContractError, "stale"):
            self.contract.update([assessment("goal")], cell=3, step=3)

    def test_same_step_partial_dispatch_still_expires_old_evidence(self):
        self.contract.motion(cell=2, step=0, tool="failed_move")
        with self.assertRaisesRegex(ContractError, "stale"):
            self.contract.update([assessment("goal")], cell=3, step=0)

    def test_unknown_or_empty_evidence_cannot_satisfy_a_condition(self):
        for item in ({"id": "goal", "status": "satisfied", "evidence": []},
                     assessment("goal", cell=99)):
            with self.assertRaises(ContractError):
                self.contract.update([item], cell=1, step=0)
        self.contract.record_evidence(2, 0, usable=False)
        with self.assertRaisesRegex(ContractError, "observation-bearing"):
            self.contract.update([assessment("goal", cell=2)], cell=3, step=0)

    def test_updates_are_atomic_and_tri_state(self):
        with self.assertRaises(ContractError):
            self.contract.update([assessment("goal"), assessment("invented")], cell=1, step=0)
        self.assertEqual(self.contract.assessments["goal"]["status"], "unknown")
        self.contract.update([assessment("goal", status="unsatisfied")], cell=1, step=0)
        self.assertEqual(self.contract.assessments["goal"]["status"], "unsatisfied")
        self.contract.update([assessment("goal", status="unknown")], cell=1, step=0)
        self.assertEqual(self.contract.assessments["goal"]["status"], "unknown")

    def test_subtask_authorization_checks_prerequisites_and_maintain(self):
        intent = {"subtask": "place", "purpose": "place while keeping support", "preserve": ["support"]}
        with self.assertRaisesRegex(ContractError, "current evidence"):
            self.contract.authorize(intent, cell=1, step=0)
        self.contract.update([assessment(x) for x in ("ready", "achieved", "support")], cell=1, step=0)
        self.contract.authorize(intent, cell=1, step=0)
        intent["preserve"] = []
        with self.assertRaisesRegex(ContractError, "acknowledge"):
            self.contract.authorize(intent, cell=1, step=0)

    def test_success_needs_current_simultaneous_conditions(self):
        self.contract.update([assessment("goal")], cell=1, step=0)
        self.assertEqual(len(self.contract.success_errors(0)), 1)
        self.contract.update([assessment("support")], cell=1, step=0)
        self.assertEqual(self.contract.success_errors(0), [])
        self.contract.motion(cell=2, step=0, tool="move")
        self.contract.record_evidence(2, 3, usable=True)
        self.contract.update([assessment("goal", cell=2)], cell=3, step=3)
        self.assertTrue(any("support" in error for error in self.contract.success_errors(3)))

    def test_only_explicitly_required_cleanup_is_gated(self):
        self.contract.update([assessment("goal"), assessment("support")], cell=1, step=0)
        self.assertEqual(self.contract.success_errors(0), [])
        plan = example_plan()
        plan["conditions"][-1]["required"] = True
        self.contract.register(plan, cell=2, step=0)
        self.contract.update([assessment("goal"), assessment("support")], cell=2, step=0)
        self.assertTrue(any("tidy" in error for error in self.contract.success_errors(0)))

    def test_simultaneous_group_cannot_use_a_historical_event(self):
        plan = example_plan()
        plan["simultaneous"] = [["goal", "achieved"]]
        with self.assertRaisesRegex(ContractError, "current states"):
            self.contract.register(plan, cell=2, step=0)

    def test_prompt_prioritizes_conditions_and_leaves_original_tools_unmodified(self):
        from harnesslab.prompt import build_prompt
        dependency = types.ModuleType("pyrualean.prompt")
        dependency.api_reference = lambda *a, **kw: "ORIGINAL ROBOT API"
        dependency.knowledge_text = lambda *a, **kw: "GENERIC KNOWLEDGE: release and home."
        adapter = types.SimpleNamespace(robot_cls=object, knowledge="robot")
        card = types.SimpleNamespace(render=lambda: "NATIVE TASK INSTRUCTION")
        with patch.dict(sys.modules, {"pyrualean.prompt": dependency}):
            prompt = build_prompt(adapter, card, max_turns=100, images="on-demand",
                                  feedback="pure", task_conditions=True, tool_set="pyrua")
            control = build_prompt(adapter, card, max_turns=100, images="on-demand",
                                   feedback="pure", task_conditions=False, tool_set="pyrua")
        self.assertLess(prompt.index("TASK CONDITION PROTOCOL"), prompt.index("GENERIC KNOWLEDGE"))
        self.assertIn("Explicit task conditions take priority", prompt)
        self.assertIn("GENERIC KNOWLEDGE: release and home.", prompt)
        self.assertNotIn("strong_grasp", prompt)
        self.assertNotIn("robo.grasp", prompt)
        self.assertNotIn("TASK CONDITION PROTOCOL", control)


if __name__ == "__main__":
    unittest.main()
