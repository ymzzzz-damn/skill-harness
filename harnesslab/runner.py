"""Own native Test episode orchestration; never calls pyrualean.play_episode."""
from __future__ import annotations
import contextlib
import hashlib
import importlib
import inspect
import json
import math
import os
import re
from pathlib import Path
import sys
import time
from typing import Any
from .agent import DISABLED_FEATURES, run_codex
from .policy import CodePolicy
from .prompt import build_prompt
from .serialization import jsonable, write_json


def _guard_native_actions(toolkit: Any, backend: Any, policy: Any) -> None:
    """Check task conditions at the real primitive dispatch boundary.

    The SDK, action implementations and observations remain unchanged. This
    instance-only wrapper also covers aliased robot methods and helper functions.
    """
    from pyrualean.hosts.robodojo_native import STATEFUL
    execute = toolkit.execute_tool

    def guarded(name: str, input_dict: dict[str, Any]) -> Any:
        if name not in STATEFUL:
            return execute(name, input_dict)
        policy.before_native_action(name, input_dict)
        before = backend.env_steps()
        try:
            return execute(name, input_dict)
        finally:
            policy.after_native_action(name, before, backend.env_steps())

    toolkit.execute_tool = guarded

def _read_native_outcome(out_dir: Path, task: str, layout: int) -> dict[str, Any]:
    """Read one finalized native trial; called after the client has stopped.

    The trial index is always "0"; layout_id identifies the selected scene.
    This proof is independent of whether the task succeeded.
    """
    paths = sorted((out_dir / "bench/eval_result").rglob("_result.json"))
    proof: dict[str, Any] = {"valid": False, "success": None, "errors": [],
                             "path": str(paths[0]) if len(paths) == 1 else None}
    if len(paths) != 1:
        proof["errors"].append(f"expected one native result, found {len(paths)}")
        return proof
    path = paths[0]
    if task not in path.parts:
        proof["errors"].append("native task path mismatch")
    try:
        native = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        proof["errors"].append(f"native result unreadable: {exc}")
        return proof
    if not isinstance(native, dict):
        proof["errors"].append("native result must be an object")
        return proof
    details = native.get("details")
    trial = details.get("0") if isinstance(details, dict) and set(details) == {"0"} else None
    if not isinstance(trial, dict):
        proof["errors"].append("native details must contain exactly trial 0")
        return proof
    if type(native.get("eval_time")) is not int or native["eval_time"] != 1:
        proof["errors"].append("native eval_time must be integer 1")
    if type(trial.get("layout_id")) is not int or trial["layout_id"] != layout:
        proof["errors"].append("native layout_id mismatch")
    if type(trial.get("success")) is not bool:
        proof["errors"].append("native success must be an explicit boolean")
    def number(value: Any) -> bool:
        return type(value) in (int, float) and math.isfinite(value)
    score, detail_score, rate = native.get("score"), trial.get("score"), native.get("success_rate")
    if not number(score) or not 0 <= score <= 100:
        proof["errors"].append("native aggregate score must be finite 0..100")
    if not number(detail_score) or not 0 <= detail_score <= 1:
        proof["errors"].append("native trial score must be finite 0..1")
    if number(score) and number(detail_score) and not math.isclose(score, detail_score * 100, abs_tol=1e-5):
        proof["errors"].append("native aggregate/trial scores disagree")
    if (not number(rate) or type(trial.get("success")) is not bool or
            rate != float(trial["success"])):
        proof["errors"].append("native success rate disagrees with verdict")
    proof.update(valid=not proof["errors"], success=trial.get("success"),
                 score=score, detail_score=detail_score)
    return proof


def _planner_diagnostics(generation: dict[str, Any], events_path: Path) -> dict[str, Any]:
    """Separate proven recovered CLI reconnects from fatal/unknown planner events.

    A retry notice is a warning only when a later successful MCP completion or
    completed model turn proves that this same agent resumed. Raw errors remain
    in generation.errors and the original event stream for historical auditing.
    """
    events = []
    if events_path.is_file():
        for line_number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append((line_number, event))
    progress = []
    candidates = []
    for line_number, event in events:
        item = event.get("item")
        item = item if isinstance(item, dict) else {}
        if event.get("type") == "turn.completed" or (
                event.get("type") == "item.completed" and
                item.get("type") in ("mcp_tool_call", "mcpToolCall") and
                item.get("status") == "completed" and not item.get("error")):
            progress.append(line_number)
        if event.get("type") in ("error", "turn.failed"):
            candidates.append((line_number, event))
        elif event.get("type") == "item.completed" and item.get("type") == "error":
            candidates.append((line_number, item))
    warnings, fatal, used = [], [], set()
    raw_errors = generation.get("errors") or []
    if not isinstance(raw_errors, list):
        raw_errors = [raw_errors]
    for error in raw_errors:
        match = next(((index, line, event) for index, (line, event) in enumerate(candidates)
                      if index not in used and event == error), None)
        if match is None:
            fatal.append({"event": error, "reason": "unmatched_or_unclassified_planner_error"})
            continue
        index, line_number, event = match
        used.add(index)
        recovery = next((line for line in progress if line > line_number), None)
        message = event.get("message", "")
        reconnect = (event.get("type") == "error" and isinstance(message, str) and
                     re.match(r"^Reconnecting(?:\.\.\.|…)\s+\d+/\d+\s+\(", message))
        if reconnect and recovery is not None:
            warnings.append({"event": error, "event_line": line_number,
                             "recovered_by_event_line": recovery,
                             "reason": "CLI reconnect followed by completed agent/tool progress"})
        else:
            fatal.append({"event": error, "event_line": line_number,
                          "reason": "unrecovered_reconnect" if reconnect else "fatal_or_unknown_planner_event"})
    return {"transport_warnings": warnings, "fatal_errors": fatal,
            "raw_error_count": len(raw_errors)}


def _finalize_runtime_status(result: dict[str, Any], generation: dict[str, Any],
                             policy: Any, native: dict[str, Any], *,
                             native_end_observed: bool, episode_over: bool,
                             events_path: Path) -> None:
    """Classify transport/process completion without using success as a gate."""
    diagnostics = _planner_diagnostics(generation, events_path)
    result["planner_diagnostics"] = diagnostics
    generation["transport_warnings"] = diagnostics["transport_warnings"]
    generation["fatal_errors"] = diagnostics["fatal_errors"]
    result["native_outcome"] = native
    result["native_episode_over"] = episode_over
    result["native_end_observed_during_agent"] = native_end_observed
    if diagnostics["transport_warnings"]:
        result["notes"].append("Recovered transport retry notices retained separately from fatal planner errors.")
    if diagnostics["fatal_errors"]:
        result["status"] = "planner_error"
        return
    if not native["valid"] or not episode_over:
        result["status"] = "native_result_error"
        return
    if result.get("environment_success") != native["success"]:
        native["valid"] = False
        native["errors"].append("backend and finalized native verdict disagree")
        result["status"] = "native_result_error"
        return
    # Budgets are runtime terminal reasons, not inferred from the native score.
    if generation.get("status") == "timeout":
        result["status"] = "timeout"
        return
    if policy.turn_budget_exhausted:
        result["status"] = "call_budget"
        return
    returncode = generation.get("returncode")
    expected_stop = returncode == -15 and (native_end_observed or policy.finish is not None)
    if generation.get("status") != "completed" or (returncode != 0 and not expected_stop):
        result["status"] = "planner_error"
        diagnostics["fatal_errors"].append({"event": {"returncode": returncode},
                                            "reason": "unexpected_planner_process_exit"})
        return
    result["status"] = "completed"


def run_episode(cell: Any, out_dir: Path, *, arm: str = "cells", model: str,
                reasoning_effort: str = "xhigh", budget_s: float = 5400,
                max_turns: int = 100, codex_bin: str = "codex",
                codex_home: str | None = None, base_url: str | None = None,
                api_key_env: str | None = None, robot: str = "robodojo_xpl_rgb",
                boot_options: dict[str, Any] | None = None, runtime: str = "codex",
                primitives: str = "no-vla", images: str = "on-motion",
                feedback: str = "rich", tool_set: str = "strong-grasp",
                task_conditions: bool = False,
                **compatibility: Any) -> dict[str, Any]:
    """Fresh native scene, fresh policy/agent, frozen tool, native scoring/video.

    API matches the deployment runner's PyRUA keyword surface. Only explicit
    harmless defaults are accepted; unknown options fail instead of drifting.
    No model calls or GPU work occur until this function is explicitly invoked.
    """
    allowed = {"rpent_root": None, "max_programs": None, "max_attempts": 1,
               "guides": False, "guides_dir": None, "claude_cli": None,
               "openai_options": None}
    for key, value in compatibility.items():
        if key not in allowed or value != allowed[key]:
            raise ValueError(f"unsupported harness option {key}={value!r}")
    if arm != "cells" or runtime != "codex" or primitives != "no-vla":
        raise ValueError("harness v0.1 requires cells/codex/no-vla")
    if robot != "robodojo_xpl_rgb":
        raise ValueError("this controlled Test runner requires RGB-only RoboDojo")
    if tool_set not in ("pyrua", "strong-grasp"):
        raise ValueError("tool_set must be pyrua or strong-grasp")
    if type(task_conditions) is not bool:
        raise ValueError("task_conditions must be boolean")
    if (not math.isfinite(float(budget_s)) or float(budget_s) <= 0 or
            not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1):
        raise ValueError("positive budget_s and max_turns required")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Instrumentation may precreate OUT. Refuse any previous policy/result evidence.
    evidence = ("result.json", "workspace", "codex_events.jsonl", "tool_turns.json", "bench")
    if any((out_dir / name).exists() for name in evidence):
        raise FileExistsError(f"refusing to reuse an episode directory: {out_dir}")
    workspace = out_dir / "workspace"
    workspace.mkdir()
    from pyrualean._backend import Ledger
    from pyrualean.hosts.common import stop_all
    from pyrualean.mcp_server import ArmMcpServer
    from pyrualean.robots import get_adapter
    adapter = get_adapter(robot)
    if tool_set == "pyrua" and hasattr(adapter.robot_cls, "grasp"):
        raise RuntimeError("original PyRUA tools require a clean, unpatched robot class")
    install, grasp = None, None
    if tool_set == "strong-grasp":
        from strong_grasp import install, grasp
    host = adapter.host_module()
    ledger = Ledger()
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema": "harness-native-test-v1", "system": "harness",
        "framework_version": "0.2.0", "arm": arm, "runtime": runtime,
        "tool_set": tool_set, "task_conditions": task_conditions,
        "agent_features": {"disabled": list(DISABLED_FEATURES)},
        "robot": robot, "backend": robot, "task": cell.task, "suite": cell.suite,
        "seed": cell.seed, "model": model, "reasoning_effort": reasoning_effort,
        "max_turns": max_turns, "timeout_s": budget_s, "images": images,
        "feedback": feedback, "environment_success": False,
        "status": "boot_error", "finish": None, "error": None,
        "generation": None, "startup_s": None, "policy_wall_s": None,
        "total_wall_s": None, "tool_turns": 0, "calls": 0,
        "task_wall_s": None,
        "task_timing": {"start": "initial task card, RGB and robot tools ready",
                        "end": "first observed native episode end or explicit conclusion",
                        "excludes": ["GPU queue", "native scene initialization", "robot tool initialization", "teardown"],
                        "includes": ["framework planning", "model reasoning", "robot tools", "feedback"]},
        "env_steps": 0, "env_steps_total": None, "notes": [],
        "protocol": {"fresh_agent_per_scene": True, "test_reset": False,
                     "cross_scene_learning": False, "automatic_replays": False,
                     "success_owner": "native evaluator", "observation": "RGB only"},
        "host": {"python": sys.executable, "cuda_device": cell.cuda_device,
                 "primitive_set": primitives, "boot_options": jsonable(boot_options or {})},
        "dependencies": {},
    }
    native_toolkit, daemons, server, policy, backend = None, [], None, None, None
    old_cwd = Path.cwd()
    generation = None
    native_end_observed = False
    task_started, task_ended = None, None
    try:
        native_toolkit, daemons = host.boot(cell, out_dir, **(boot_options or {}))
        if install is not None:
            install(native_toolkit, robot_cls=adapter.robot_cls)
        backend = host.make_backend(native_toolkit)
        card = host.dump_card(native_toolkit, cell, out_dir)
        task_started = time.perf_counter()
        result["task_card_ready_at"] = time.time()
        robo = adapter.robot_cls(backend, ledger=ledger, primitives=primitives)
        policy = CodePolicy(robo, workspace, max_turns=max_turns, images=images,
                            feedback=feedback, grasp=grasp,
                            task_conditions=task_conditions, initial_evidence=True)
        if task_conditions:
            _guard_native_actions(native_toolkit, backend, policy)
        result["robot_tools"] = {"native_handlers": sorted(getattr(native_toolkit, "_tools", {})),
                                  "grasp_installed": hasattr(robo, "grasp"),
                                  "global_grasp_supplied": grasp is not None}
        if tool_set == "pyrua" and (result["robot_tools"]["grasp_installed"] or
                                      "strong_grasp" in getattr(native_toolkit, "_tools", {})):
            raise RuntimeError("unexpected strong grasp injection in original-tool mode")
        result["task_language"] = card.task_language
        result["max_episode_steps"] = card.max_env_steps
        result["startup_s"] = task_started - started
        for name in (("pyrualean", "strong_grasp", "harnesslab") if grasp is not None
                     else ("pyrualean", "harnesslab")):
            module = importlib.import_module(name)
            root = Path(inspect.getfile(module)).resolve().parent
            result["dependencies"][name] = {"path": str(root),
                "sha256": {str(path.relative_to(root)):
                           hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(root.rglob("*.py"))}}
        prompt = build_prompt(adapter, card, max_turns=max_turns,
                              images=images, feedback=feedback,
                              tool_set=tool_set, task_conditions=task_conditions)
        (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        result["prompt"] = {"sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                            "images": list(card.images.values())}
        # Host-native simulator/action protocol is shared, while policy execution,
        # feedback, prompt, agent process, budgets and result finalization are ours.
        server = ArmMcpServer(policy, annotations={"readOnlyHint": True,
                              "destructiveHint": False, "openWorldHint": False})
        mcp_url = server.start()
        os.chdir(workspace)
        result["status"] = "running"
        write_json(out_dir / "result.json", result)
        policy_started = time.perf_counter()
        def observe_native_end():
            nonlocal native_end_observed, task_ended
            callback = getattr(backend, "episode_over", None)
            over = bool(callback()) if callable(callback) else False
            native_end_observed = native_end_observed or over
            if over and task_ended is None:
                task_ended = time.perf_counter()
            return over
        generation = run_codex(prompt, list(card.images.values()), toolkit=policy,
            model=model, reasoning_effort=reasoning_effort, budget_s=budget_s,
            out_dir=out_dir, mcp_url=mcp_url, codex_bin=codex_bin,
            codex_home=codex_home, base_url=base_url, api_key_env=api_key_env,
            should_stop=observe_native_end)
        result["generation"], result["status"] = generation, generation["status"]
        result["codex_returncode"] = generation["returncode"]
        result["policy_wall_s"] = time.perf_counter() - policy_started
        result["finish"] = policy.finish
        # The same native client used by PyRUA supplies final success and video.
        backend.conclude()
        if task_ended is None:
            task_ended = time.perf_counter()
        result["environment_success"] = bool(backend.solved())
        result["env_steps_total"] = backend.env_steps()
        result["native_reports"] = [str(path.relative_to(out_dir))
                                   for path in sorted(out_dir.rglob("*"))
                                   if path.is_file() and "eval_result" in path.parts]
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if result["status"] != "boot_error":
            result["status"] = "error"
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if task_started is not None:
            result["task_wall_s"] = max(0.0, (task_ended or time.perf_counter()) - task_started)
            result["task_timing"]["end_observed"] = task_ended is not None
        if policy is not None:
            result["tool_turns"], result["finish"] = len(policy.turns), policy.finish
            result["sandbox_flags"] = policy.sandbox_flags
            result["tool_wall_s"] = sum(turn["elapsed_s"] for turn in policy.turns)
            if result["policy_wall_s"] is not None:
                result["non_tool_agent_s"] = max(0, result["policy_wall_s"] - result["tool_wall_s"])
            write_json(out_dir / "tool_turns.json", policy.turns)
        result["calls"], result["stateful_calls"] = len(ledger), ledger.stateful_calls()
        result["env_steps"] = ledger.env_steps()
        ledger.write_jsonl(out_dir / "calls.jsonl")
        if server is not None:
            with contextlib.suppress(Exception):
                server.stop()
        with contextlib.suppress(Exception):
            stop_all(native_toolkit, daemons)
        # _Client.stop waits for the native client to finish writing reports.
        # Classify only after that cleanup, avoiding a report-write race.
        if generation is not None and policy is not None and result["error"] is None:
            native = _read_native_outcome(out_dir, cell.task, cell.seed)
            try:
                episode_over = bool(backend.episode_over())
            except Exception:
                episode_over = False
            _finalize_runtime_status(result, generation, policy, native,
                native_end_observed=native_end_observed, episode_over=episode_over,
                events_path=out_dir / "codex_events.jsonl")
        result["native_reports"] = [str(path.relative_to(out_dir))
            for path in sorted(out_dir.rglob("*"))
            if path.is_file() and "eval_result" in path.parts]
        os.chdir(old_cwd)
        result["total_wall_s"] = time.perf_counter() - started
        write_json(out_dir / "result.json", result)
    return result

# Familiar spelling for callers, with our implementation and lifecycle.
play_episode = run_episode
