"""Own persistent code policy and automatic, auditable physical feedback.

The optional PyRUA dependency supplies sandbox guards, not the orchestration.
The native backend retains physics, action limits, scoring and video ownership.
"""
from __future__ import annotations
import ast
import base64
import contextlib
import io
import math
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from .serialization import jsonable, write_json
from .task_contract import (ContractError, TaskContract, PLAN_SCHEMA,
                            UPDATE_SCHEMA, INTENT_SCHEMA)

@dataclass
class Feedback:
    result: dict[str, Any]
    images: list[bytes] = field(default_factory=list)

    @property
    def content_blocks(self) -> list[dict[str, Any]]:
        import json
        text = json.dumps(jsonable(self.result), ensure_ascii=False, allow_nan=False)
        if len(text) > 20000:
            text = text[:20000] + "\n[feedback truncated; complete record retained on host]"
        blocks = [{"type": "text", "text": text}]
        blocks.extend({"type": "image", "source": {"type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(data).decode("ascii")}}
                      for data in self.images)
        return blocks

class CodePolicy:
    """A fresh instance is created for every native Test scene.

    Code cells persist within one scene. No Python state, user code or findings
    transfer to another scene; the frozen strong_grasp package is the only new tool.
    """
    def __init__(self, robo: Any, workspace: Path, *, max_turns: int = 100,
                 images: str = "on-motion", feedback: str = "rich",
                 grasp: Callable[..., Any] | None = None,
                 task_conditions: bool = False, initial_evidence: bool = True,
                 guard_factory: Callable[[Path], dict[str, Any]] | None = None,
                 audit: Callable[..., list[str]] | None = None,
                 blocked: Callable[[str], list[str]] | None = None) -> None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if images not in ("on-motion", "on-demand", "none"):
            raise ValueError("invalid images policy")
        if feedback not in ("rich", "pure"):
            raise ValueError("invalid feedback mode")
        if guard_factory is None or audit is None or blocked is None:
            from pyrualean.sandbox import make_builtins, audit_source, blocked_names
            guard_factory = guard_factory or make_builtins
            audit = audit or audit_source
            blocked = blocked or blocked_names
        self.robo, self.workspace = robo, Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.max_turns, self.images, self.feedback = max_turns, images, feedback
        self.audit, self.blocked = audit, blocked
        self.finish: dict[str, str] | None = None
        self.turns: list[dict[str, Any]] = []
        self.sandbox_flags: list[dict[str, Any]] = []
        self.turn_budget_exhausted = False
        self._cells = 0
        self._lock = threading.Lock()
        self._displayed: list[Any] = []
        self._has_grasp = grasp is not None
        self.contract = TaskContract() if task_conditions else None
        self._active_cell: int | None = None
        self._intent: dict[str, Any] | None = None
        self._cell_authorized = False
        self._physical_dispatches = 0
        if self.contract is not None:
            self.contract.record_evidence(0, self.robo._backend.env_steps(),
                                          usable=initial_evidence, images=["initial RGB"])
            self._save_contract()
        self._globals = {"__name__": "__harness_policy__",
                         "__builtins__": guard_factory(self.workspace),
                         "robo": robo, "math": math, "time": time,
                         "log": print, "display": self._displayed.append}
        if grasp is not None:
            self._globals["grasp"] = lambda target, **kw: grasp(robo, target, **kw)
        try:
            import numpy as np
            self._globals.update(np=np, numpy=np)
        except ImportError:
            pass

    def get_tools_spec(self) -> list[dict[str, Any]]:
        tools = [
            {"name": "python", "description":
             "Execute Python in this scene's persistent namespace with robo. "
             "Returns structured trailing results, stdout/errors and requested camera images. "
             + ("robo.grasp(target, ...) is the supplied bounded grasp tool." if self._has_grasp else ""),
             "input_schema": {"type": "object", "properties":
                              {"code": {"type": "string"}}, "required": ["code"]}},
            {"name": "finish", "description":
             "End this scene and let the native evaluator judge its final state. "
             "Your declared status is recorded separately from native success.",
             "input_schema": {"type": "object", "properties":
                 {"status": {"type": "string", "enum": ["success", "failure", "stuck"]},
                  "summary": {"type": "string"}}, "required": ["status", "summary"]}},
        ]
        if self.contract is not None:
            tools[0]["input_schema"]["properties"].update(
                task_plan=PLAN_SCHEMA, task_update=UPDATE_SCHEMA,
                task_update_after=UPDATE_SCHEMA, task_intent=INTENT_SCHEMA)
            tools[0]["description"] += (
                " Register task_plan before physical actions. task_update assesses completed "
                "observation cells before code; task_update_after is applied after code. "
                "Motion requires task_intent. A trailing dict task_update is a code-derived assessment.")
            tools[1]["input_schema"]["properties"]["task_update"] = UPDATE_SCHEMA
            tools[1]["description"] += " Success requires current, evidence-cited final conditions."
        return tools

    def _save_contract(self) -> None:
        if self.contract is not None:
            write_json(self.workspace.parent / "task_contract.json", self.contract.snapshot())
            write_json(self.workspace.parent / "task_condition_events.json", self.contract.events)

    def before_native_action(self, name: str, kwargs: dict[str, Any]) -> None:
        """Called by the runner at NativeToolkit's actual stateful dispatch boundary.

        Does not wrap or replace a robot primitive. One bounded code cell has one
        declared subtask; its prerequisites are checked before its first action.
        Every dispatch, including a potentially partial failed one, expires state.
        """
        if self.contract is None:
            return
        if self._active_cell is None:
            raise ContractError("physical action outside an active policy cell")
        step = self.robo._backend.env_steps()
        if not self._cell_authorized:
            self.contract.authorize(self._intent, cell=self._active_cell, step=step)
            self._cell_authorized = True
        self.contract.motion(cell=self._active_cell, step=step, tool=name)
        self._physical_dispatches += 1
        self._save_contract()

    def after_native_action(self, name: str, steps_before: int | None,
                            steps_after: int | None) -> None:
        if self.contract is not None:
            self.contract._event("physical_return", cell=self._active_cell, tool=name,
                                  steps_before=steps_before, steps_after=steps_after)
            self._save_contract()

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Feedback:
        name = name.split("__")[-1] if name.startswith("mcp__") else name
        with self._lock:
            if self.finish is not None:
                return Feedback({"error": "episode already finished", "finish": self.finish})
            if name == "finish":
                status = str(input_dict.get("status", ""))
                if status not in ("success", "failure", "stuck"):
                    return Feedback({"error": "invalid finish status"})
                if self.contract is not None:
                    try:
                        if "task_update" in input_dict:
                            self.contract.update(input_dict["task_update"], cell=self._cells + 1,
                                                 step=self.robo._backend.env_steps())
                        errors = self.contract.success_errors(self.robo._backend.env_steps()) if status == "success" else []
                        if errors:
                            self.contract._event("finish_rejected", errors=errors)
                            self._save_contract()
                            return Feedback({"error": "success declaration lacks current task evidence",
                                             "conditions": errors,
                                             "native_success": "independent native evaluator"})
                    except ContractError as exc:
                        return Feedback({"error": str(exc)})
                    self._save_contract()
                self.finish = {"status": status, "summary": str(input_dict.get("summary", ""))}
                write_json(self.workspace.parent / "agent_finish.json", self.finish)
                return Feedback({"ok": True, "agent_finish": self.finish,
                                 "native_success": "judged by evaluator after conclusion"})
            if name != "python":
                return Feedback({"error": "unknown tool: " + name})
            if self._cells >= self.max_turns:
                if not self.turn_budget_exhausted:
                    self.turn_budget_exhausted = True
                    self.robo._stop_episode("call_budget")
                return Feedback({"error": "code decision budget exhausted; call finish",
                                 "remaining_code_calls": 0})
            self._cells += 1
            return self._execute_cell(str(input_dict.get("code", "")), input_dict)

    def _execute_cell(self, code: str, metadata: dict[str, Any] | None = None) -> Feedback:
        index, started = self._cells, time.perf_counter()
        location = self.workspace / f"cell_{index:03d}.py"
        location.write_text(code, encoding="utf-8")
        if self.contract is not None:
            folder = self.workspace.parent / "task_metadata"
            folder.mkdir(exist_ok=True)
            write_json(folder / f"{index:03d}.json", {key: value for key, value in (metadata or {}).items()
                       if key.startswith("task_")})
        flags = self.audit(code, workspace=self.workspace)
        if flags:
            self.sandbox_flags.append({"cell": index, "flags": flags})
        before = len(self.robo._ledger)
        steps_before = self.robo._backend.env_steps()
        dispatches_before = self._physical_dispatches
        self._active_cell = index
        self._intent = (metadata or {}).get("task_intent")
        self._cell_authorized = False
        stdout, stderr = io.StringIO(), io.StringIO()
        result: dict[str, Any] = {"cell": index,
                                  "remaining_code_calls": self.max_turns - index}
        blocked = self.blocked(code)
        try:
            if self.contract is not None:
                if "task_plan" in (metadata or {}):
                    self.contract.register(metadata["task_plan"], cell=index, step=steps_before)
                if "task_update" in (metadata or {}):
                    self.contract.update(metadata["task_update"], cell=index, step=steps_before)
            if blocked:
                raise PermissionError("sandbox blocked names: " + ", ".join(blocked))
            tree = ast.parse(code, filename=str(location))
            tail = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(tree, str(location), "exec"), self._globals)
                if tail is not None:
                    value = eval(compile(ast.Expression(tail.value), str(location), "eval"),
                                 self._globals)
                    if value is not None:
                        result["value"] = jsonable(value)
        except BaseException as exc:
            # EpisodeFinished deliberately inherits BaseException. Preserve the
            # native stop reason instead of losing the partial tool evidence.
            if type(exc).__name__ == "EpisodeFinished":
                result["episode_stopped"] = str(getattr(exc, "reason", str(exc)))
            elif isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            else:
                result["exception"] = {"type": type(exc).__name__, "message": str(exc),
                                       "traceback": traceback.format_exc(limit=8)}
                if self.contract is not None and isinstance(exc, ContractError):
                    self.contract._event("condition_operation_rejected", cell=index, error=str(exc))
        result["stdout"], result["stderr"] = stdout.getvalue(), stderr.getvalue()
        records = self.robo._ledger.records[before:]
        moved = (self._physical_dispatches > dispatches_before if self.contract is not None
                 else any(record.stateful for record in records))
        steps_after = self.robo._backend.env_steps()
        # Fail closed if integration forgot the native boundary hook. The runner
        # also verifies this hook before starting the model; never silently treat
        # ledger inspection as pre-dispatch enforcement.
        if self.contract is not None and steps_after != steps_before and not moved:
            result["exception"] = {"type": "ContractError", "message": "native action guard was not installed"}
        if self.feedback == "rich":
            result["physical"] = {
                "moved": moved,
                "native_actions_used": (steps_after - steps_before
                    if steps_before is not None and steps_after is not None else None),
                "primitive_calls": [jsonable(record.to_dict()) for record in records
                                    if record.tool != "artifact"][-40:],
            }
            try:
                result["state"] = jsonable(self.robo._summary_dict())
            except Exception as exc:
                result["state_error"] = f"{type(exc).__name__}: {exc}"
        images, names = self._images(moved, result)
        if names:
            result["images"] = names
        if self.contract is not None:
            self.contract.record_evidence(index, steps_after, usable=bool(
                images or result.get("stdout") or "value" in result or
                result.get("state") or (moved and records)), images=names)
            if "exception" not in result:
                try:
                    value_updates = (result.get("value", {}).get("task_update")
                                     if isinstance(result.get("value"), dict) else None)
                    if value_updates is not None:
                        self.contract.update(value_updates, cell=index, step=steps_after)
                    if "task_update_after" in (metadata or {}):
                        self.contract.update(metadata["task_update_after"], cell=index, step=steps_after)
                except ContractError as exc:
                    result["exception"] = {"type": "ContractError", "message": str(exc)}
            result["task_conditions"] = {"registered": self.contract.plan is not None,
                "revision": self.contract.revision, "motion_epoch": self.contract.epoch,
                "assessments": self.contract.assessments,
                "success_gate_errors": self.contract.success_errors(steps_after),
                "semantics": "agent evidence assessment; native verdict remains independent"}
            self._save_contract()
        self._active_cell = None
        self._intent = None
        elapsed = time.perf_counter() - started
        summary = {"index": index, "tool": "python", "elapsed_s": elapsed,
                   "ok": "exception" not in result, "moved": moved,
                   "env_steps": (steps_after - steps_before
                     if steps_before is not None and steps_after is not None else None),
                   "sandbox_flags": flags, "feedback": f"tool_feedback/{index:03d}.json"}
        self.turns.append(summary)
        folder = self.workspace.parent / "tool_feedback"
        folder.mkdir(exist_ok=True)
        for image_index, data in enumerate(images):
            (folder / f"{index:03d}_{image_index:02d}.png").write_bytes(data)
        write_json(folder / f"{index:03d}.json", result)
        write_json(self.workspace.parent / "tool_turns.json", self.turns)
        return Feedback(result, images)

    def _images(self, moved: bool, result: dict[str, Any]) -> tuple[list[bytes], list[str]]:
        wanted = self.robo._take_show_requests()
        if self.images == "on-motion" and moved:
            artifacts = list(type(self.robo).ON_MOTION_IMAGES)
        elif self.images == "on-demand":
            artifacts = list(self.robo._show_artifacts(wanted))
        else:
            artifacts = []
        output, names = [], []
        for artifact in artifacts:
            try:
                output.append(self.robo.artifact(artifact))
                names.append(artifact)
            except Exception as exc:
                result.setdefault("image_errors", []).append(f"{artifact}: {exc}")
        # Explicit display() is supported regardless of automatic camera policy,
        # matching the code-cell contract and preserving mask/overlay diagnostics.
        displayed = self._displayed[:]
        self._displayed.clear()
        for index, value in enumerate(displayed):
            try:
                if isinstance(value, (bytes, bytearray)):
                    data = bytes(value)
                else:
                    from PIL import Image
                    image = value if hasattr(value, "save") else Image.fromarray(value)
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    data = buffer.getvalue()
                output.append(data)
                names.append(f"display_{index}")
            except Exception as exc:
                result.setdefault("image_errors", []).append(f"display_{index}: {exc}")
        return output, names
