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
        return [
            {"name": "python", "description":
             "Execute Python in this scene's persistent namespace with robo. "
             "Returns structured trailing results, stdout/errors and requested camera images. "
             "robo.grasp(target, ...) is the supplied bounded grasp tool.",
             "input_schema": {"type": "object", "properties":
                              {"code": {"type": "string"}}, "required": ["code"]}},
            {"name": "finish", "description":
             "End this scene and let the native evaluator judge its final state. "
             "Your declared status is recorded separately from native success.",
             "input_schema": {"type": "object", "properties":
                 {"status": {"type": "string", "enum": ["success", "failure", "stuck"]},
                  "summary": {"type": "string"}}, "required": ["status", "summary"]}},
        ]

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Feedback:
        name = name.split("__")[-1] if name.startswith("mcp__") else name
        with self._lock:
            if self.finish is not None:
                return Feedback({"error": "episode already finished", "finish": self.finish})
            if name == "finish":
                status = str(input_dict.get("status", ""))
                if status not in ("success", "failure", "stuck"):
                    return Feedback({"error": "invalid finish status"})
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
            return self._execute_cell(str(input_dict.get("code", "")))

    def _execute_cell(self, code: str) -> Feedback:
        index, started = self._cells, time.perf_counter()
        location = self.workspace / f"cell_{index:03d}.py"
        location.write_text(code, encoding="utf-8")
        flags = self.audit(code, workspace=self.workspace)
        if flags:
            self.sandbox_flags.append({"cell": index, "flags": flags})
        before = len(self.robo._ledger)
        steps_before = self.robo._backend.env_steps()
        stdout, stderr = io.StringIO(), io.StringIO()
        result: dict[str, Any] = {"cell": index,
                                  "remaining_code_calls": self.max_turns - index}
        blocked = self.blocked(code)
        try:
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
        result["stdout"], result["stderr"] = stdout.getvalue(), stderr.getvalue()
        records = self.robo._ledger.records[before:]
        moved = any(record.stateful for record in records)
        steps_after = self.robo._backend.env_steps()
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
