"""Agent-assessed task conditions, kept separate from the native verdict.

This module never imports a simulator, reads evaluator code, or infers object
truth. It validates the structure and freshness of the agent's cited evidence.
"""
from __future__ import annotations
import copy
from typing import Any


class ContractError(ValueError):
    pass


PLAN_SCHEMA = {
    "type": "object", "properties": {
        "conditions": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "description": {"type": "string"},
            "kind": {"type": "string", "enum": ["final", "precondition", "completion", "maintain", "cleanup"]},
            "mode": {"type": "string", "enum": ["state", "event"]},
            "source": {"type": "string", "enum": ["instruction", "inferred"]},
            "source_text": {"type": "string"}, "evidence_required": {"type": "string"},
            "required": {"type": "boolean"}},
            "required": ["id", "description", "kind", "mode", "source", "source_text", "evidence_required"]}},
        "subtasks": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "description": {"type": "string"},
            "preconditions": {"type": "array", "items": {"type": "string"}},
            "completion": {"type": "array", "items": {"type": "string"}},
            "maintain": {"type": "array", "items": {"type": "string"}}},
            "required": ["id", "description", "preconditions", "completion", "maintain"]}},
        "simultaneous": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
        "ambiguities": {"type": "array", "items": {"type": "string"}}},
    "required": ["conditions", "subtasks", "simultaneous", "ambiguities"]}

UPDATE_SCHEMA = {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "string"},
    "status": {"type": "string", "enum": ["satisfied", "unsatisfied", "unknown"]},
    "evidence": {"type": "array", "items": {"type": "object", "properties": {
        "cell": {"type": "integer"}, "detail": {"type": "string"}}, "required": ["cell", "detail"]}},
    "note": {"type": "string"}}, "required": ["id", "status", "evidence"]}}

INTENT_SCHEMA = {"type": "object", "properties": {
    "subtask": {"type": "string"}, "purpose": {"type": "string"},
    "preserve": {"type": "array", "items": {"type": "string"}}},
    "required": ["subtask", "purpose", "preserve"]}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be nonempty text")
    return value


def _ids(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise ContractError(f"{field} must be a list of condition IDs")
    if len(value) != len(set(value)):
        raise ContractError(f"{field} contains duplicate IDs")
    return value


class TaskContract:
    """Per-episode bookkeeping; physical state evidence expires on every action.

    An event means "this happened" and survives subsequent motion. A state means
    "this is true now" and must be reassessed from evidence after motion. Both
    are agent assessments, including the ones accepted by the finish gate.
    """
    def __init__(self) -> None:
        self.plan: dict[str, Any] | None = None
        self.conditions: dict[str, dict[str, Any]] = {}
        self.subtasks: dict[str, dict[str, Any]] = {}
        self.assessments: dict[str, dict[str, Any]] = {}
        self.evidence: dict[int, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.epoch = 0
        self.revision = 0

    def _event(self, kind: str, **fields: Any) -> None:
        self.events.append({"event": kind, "epoch": self.epoch, "revision": self.revision, **fields})

    def register(self, plan: dict[str, Any], *, cell: int, step: int | None) -> None:
        if not isinstance(plan, dict):
            raise ContractError("task_plan must be an object")
        conditions, subtasks = {}, {}
        entries = plan.get("conditions")
        if not isinstance(entries, list) or not entries or len(entries) > 128:
            raise ContractError("task_plan needs 1..128 conditions")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ContractError("each condition must be an object")
            for field in ("id", "description", "source_text", "evidence_required"):
                _text(entry.get(field), "condition." + field)
            key = entry["id"]
            if key in conditions:
                raise ContractError("duplicate condition: " + key)
            if entry.get("kind") not in ("final", "precondition", "completion", "maintain", "cleanup"):
                raise ContractError("invalid condition kind")
            if entry.get("mode") not in ("state", "event"):
                raise ContractError("condition mode must distinguish state from event")
            if entry.get("source") not in ("instruction", "inferred"):
                raise ContractError("condition source must be instruction or inferred")
            if "required" in entry and type(entry["required"]) is not bool:
                raise ContractError("condition.required must be boolean")
            conditions[key] = copy.deepcopy(entry)
        if not any(x["kind"] == "final" for x in conditions.values()):
            raise ContractError("at least one final condition is required")
        entries = plan.get("subtasks")
        if not isinstance(entries, list) or not entries or len(entries) > 64:
            raise ContractError("task_plan needs 1..64 subtasks")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ContractError("each subtask must be an object")
            key = _text(entry.get("id"), "subtask.id")
            _text(entry.get("description"), "subtask.description")
            if key in subtasks:
                raise ContractError("duplicate subtask: " + key)
            for field in ("preconditions", "completion", "maintain"):
                refs = _ids(entry.get(field), "subtask." + field)
                if not set(refs) <= conditions.keys():
                    raise ContractError("subtask refers to an unknown condition")
            if not entry["completion"]:
                raise ContractError("each subtask requires a completion condition")
            subtasks[key] = copy.deepcopy(entry)
        groups = plan.get("simultaneous")
        if not isinstance(groups, list):
            raise ContractError("simultaneous must be a list of condition-ID lists")
        for group in groups:
            refs = _ids(group, "simultaneous group")
            if len(refs) < 2 or not set(refs) <= conditions.keys():
                raise ContractError("simultaneous groups need >=2 known conditions")
            if any(conditions[x]["mode"] != "state" for x in refs):
                raise ContractError("simultaneous conditions must describe current states, not historical events")
        ambiguities = plan.get("ambiguities")
        if not isinstance(ambiguities, list) or any(not isinstance(x, str) for x in ambiguities):
            raise ContractError("ambiguities must be a list of text assumptions/questions")
        # Commit only a wholly valid plan; a bad replacement cannot corrupt it.
        self.plan = copy.deepcopy(plan)
        self.conditions, self.subtasks = conditions, subtasks
        self.assessments = {key: {"status": "unknown", "evidence": [], "epoch": self.epoch,
                                  "step": step} for key in conditions}
        self.revision += 1
        self._event("plan_registered", cell=cell, step=step, plan=self.plan)

    def record_evidence(self, cell: int, step: int | None, *, usable: bool,
                        images: list[str] | None = None) -> None:
        self.evidence[cell] = {"step": step, "epoch": self.epoch, "usable": bool(usable),
                               "images": list(images or [])}

    def update(self, updates: list[dict[str, Any]], *, cell: int, step: int | None) -> None:
        if self.plan is None:
            raise ContractError("register task_plan before task_update")
        if not isinstance(updates, list):
            raise ContractError("task_update must be an assessment list")
        staged, seen = {}, set()
        for update in updates:
            if not isinstance(update, dict) or update.get("id") not in self.conditions:
                raise ContractError("assessment refers to an unknown condition")
            key = update["id"]
            if key in seen:
                raise ContractError("duplicate condition in task_update")
            seen.add(key)
            status, refs = update.get("status"), update.get("evidence")
            if status not in ("satisfied", "unsatisfied", "unknown") or not isinstance(refs, list):
                raise ContractError("assessment requires tri-state status and an evidence list")
            if status != "unknown" and not refs:
                raise ContractError("determinate assessments require cited observations")
            for ref in refs:
                if not isinstance(ref, dict) or type(ref.get("cell")) is not int:
                    raise ContractError("evidence requires an integer cell reference")
                _text(ref.get("detail"), "evidence.detail")
                observed = self.evidence.get(ref["cell"])
                if observed is None or not observed["usable"]:
                    raise ContractError("evidence must cite an existing observation-bearing cell (0 = initial RGB)")
                if self.conditions[key]["mode"] == "state" and (
                        observed["epoch"] != self.epoch or observed["step"] != step):
                    raise ContractError("current-state assessment cites stale evidence; observe again")
            staged[key] = {"status": status, "evidence": copy.deepcopy(refs),
                           "note": str(update.get("note", "")), "epoch": self.epoch,
                           "step": step, "assessed_in_cell": cell,
                           "semantics": "agent assessment, not ground truth"}
        self.assessments.update(staged)
        self._event("conditions_assessed", cell=cell, step=step, assessments=staged)

    def current(self, key: str, step: int | None) -> bool:
        assessment = self.assessments[key]
        return (assessment["status"] == "satisfied" and
                (self.conditions[key]["mode"] == "event" or
                 (assessment["epoch"] == self.epoch and assessment["step"] == step)))

    def authorize(self, intent: dict[str, Any] | None, *, cell: int, step: int | None) -> None:
        if self.plan is None:
            raise ContractError("physical action blocked: register task_plan first")
        if not isinstance(intent, dict) or intent.get("subtask") not in self.subtasks:
            raise ContractError("physical action requires task_intent naming a registered subtask")
        _text(intent.get("purpose"), "task_intent.purpose")
        preserve = _ids(intent.get("preserve"), "task_intent.preserve")
        if not set(preserve) <= self.conditions.keys():
            raise ContractError("task_intent preserves an unknown condition")
        task = self.subtasks[intent["subtask"]]
        if not set(task["maintain"]) <= set(preserve):
            raise ContractError("task_intent must acknowledge all this subtask's maintain conditions")
        missing = [key for key in set(task["preconditions"] + task["maintain"])
                   if not self.current(key, step)]
        if missing:
            raise ContractError("subtask prerequisites/maintain conditions need current evidence: " + ", ".join(sorted(missing)))
        self._event("action_authorized", cell=cell, step=step, intent=copy.deepcopy(intent))

    def motion(self, *, cell: int, step: int | None, tool: str) -> None:
        self.epoch += 1
        invalidated = []
        for key, condition in self.conditions.items():
            if condition["mode"] == "state":
                invalidated.append(key)
                self.assessments[key] = {"status": "unknown", "evidence": [], "epoch": self.epoch,
                                         "step": step, "note": "physical action invalidated old state evidence"}
        self._event("physical_dispatch", cell=cell, step=step, tool=tool, invalidated=invalidated)

    def success_errors(self, step: int | None) -> list[str]:
        if self.plan is None:
            return ["task_plan was not registered"]
        required = [key for key, condition in self.conditions.items()
                    if condition["kind"] == "final" or
                    (condition["kind"] == "cleanup" and condition.get("required", False))]
        # Simultaneous groups are evaluated together against the same physical
        # revision, never by accumulating earlier states.
        required.extend(key for group in self.plan["simultaneous"] for key in group)
        return ["condition needs current satisfied assessment: " + key
                for key in sorted(set(required)) if not self.current(key, step)]

    def snapshot(self) -> dict[str, Any]:
        return {"schema": "harness-task-contract-v1", "registered": self.plan is not None,
                "revision": self.revision, "motion_epoch": self.epoch, "plan": self.plan,
                "assessments": self.assessments, "evidence": self.evidence,
                "verification_owner": "agent evidence assessment; native evaluator is independent"}
