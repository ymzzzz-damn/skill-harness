"""Offline timing supplement; never imports or changes experiment code.

One shot:
  python scene_excluded_timing.py --root EXPERIMENT --out REPORT.json
Auxiliary reporting process for this batch only:
  python scene_excluded_timing.py --root EXPERIMENT --out REPORT.json --watch

The supplement adds the observed RGB->Tool/task-card readiness wall-clock gap
to the original monotonic task duration. It is a mixed-clock estimate, not an
exact monotonic reconstruction. Original native scores and reports stay intact.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

TASK = "pick_from_conveyor_by_image"
ARMS = ("conditions", "control")
EXPECTED = 55
POLL_SECONDS = 30
MAX_WATCH_SECONDS = 24 * 60 * 60
COMPLETED_STATUSES = ("completed", "call_budget", "timeout")
TERMINAL_STATUSES = (*COMPLETED_STATUSES, "error", "boot_error", "planner_error",
                     "native_result_error", "launcher_error", "infrastructure_blocked")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    """Distinguish missing/incomplete files from a valid empty object."""
    path = Path(path)
    if not path.is_file():
        return None, "missing " + path.name
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeError) as exc:
        return None, "unreadable " + path.name + ": " + type(exc).__name__
    if not isinstance(value, dict):
        return None, "non-object " + path.name
    return value, None


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("first_rgb_at must be an ISO datetime with explicit timezone")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("first_rgb_at lacks an explicit timezone")
    value = parsed.timestamp()
    if not finite_nonnegative(value):
        raise ValueError("first_rgb_at must be finite and nonnegative")
    return value


def native_verdict(out, layout):
    """Audit identity only; do not change or substitute the native evaluator."""
    files = sorted((out / "bench/eval_result").rglob("_result.json"))
    errors, success = [], None
    if len(files) != 1:
        return {"valid": False, "success": None,
                "errors": ["expected exactly one finalized native report"]}
    native, error = read(files[0])
    if error:
        return {"valid": False, "success": None, "errors": [error]}
    details = native.get("details")
    trial = details.get("0") if isinstance(details, dict) and set(details) == {"0"} else None
    if TASK not in files[0].parts or type(native.get("eval_time")) is not int or native["eval_time"] != 1:
        errors.append("native task/trial identity mismatch")
    if not isinstance(trial, dict):
        errors.append("native report must contain exactly trial 0")
    else:
        success = trial.get("success")
        if type(trial.get("layout_id")) is not int or trial["layout_id"] != layout:
            errors.append("native layout identity mismatch")
        if type(success) is not bool:
            errors.append("native success is not boolean")
        score, detail, rate = native.get("score"), trial.get("score"), native.get("success_rate")
        if not finite_nonnegative(score) or score > 100 or not finite_nonnegative(detail) or detail > 1:
            errors.append("invalid native score scale")
        elif not math.isclose(score, detail * 100, abs_tol=1e-5):
            errors.append("native aggregate/trial scores disagree")
        if not finite_nonnegative(rate) or type(success) is not bool or rate != float(success):
            errors.append("native success rate disagrees with verdict")
    return {"valid": not errors, "success": success if type(success) is bool else None,
            "errors": errors, "path": str(files[0]), "sha256": sha(files[0])}


def official_rows(root):
    """Use existing protocol-validity proof only when its result hash matches.

    Between stage boundaries the original full-report can legitimately lag.
    Missing/stale proof is unknown, never silently counted as a valid episode.
    """
    report, _ = read(root / "full-report.json")
    rows = report.get("episodes", []) if report else []
    return {(r.get("arm"), r.get("layout")): r for r in rows
            if isinstance(r, dict)} if isinstance(rows, list) else {}


def inspect(root, arm, layout, official=None):
    out = root / "full" / arm / f"layout_{layout:02d}" / "episode_001"
    result, result_error = read(out / "result.json")
    receipt, receipt_error = read(out / "initialization-receipt.json")
    manifest, manifest_error = read(out / "experiment.json")
    launch, _ = read(out.parent / "launch_status.json")
    result, receipt, manifest, launch = result or {}, receipt or {}, manifest or {}, launch or {}
    status = result.get("status", launch.get("status", "pending"))
    attempted = out.exists() or bool(result or receipt or manifest or launch)
    completed = status in COMPLETED_STATUSES
    row = {"arm": arm, "layout": layout, "episode_directory": str(out),
           "attempted": attempted, "status": status, "terminal": status in TERMINAL_STATUSES,
           "completed": completed, "timing_valid_completed": False,
           "protocol_valid": None, "protocol_validity_source": "existing full-report.json only",
           "task_wall_s": None, "tool_ready_overhead_s": None,
           "scene_excluded_total_s": None, "errors": []}
    if not attempted:
        return row
    errors = row["errors"]
    for error in (result_error, receipt_error, manifest_error):
        if error:
            errors.append(error)
    row["result_sha256"] = sha(out / "result.json") if (out / "result.json").is_file() else None
    row["initialization_receipt_sha256"] = sha(out / "initialization-receipt.json") if (out / "initialization-receipt.json").is_file() else None
    if official and official.get("result_sha256") == row["result_sha256"] and type(official.get("valid")) is bool:
        row["protocol_valid"] = official["valid"]
    if not completed:
        errors.append("task attempt is not finalized with completed/call_budget/timeout status")
    if result.get("error") or launch.get("error"):
        errors.append("framework/launcher error present")
    if result.get("task") != TASK or type(result.get("seed")) is not int or result["seed"] != layout:
        errors.append("framework task/layout identity mismatch")
    if result.get("tool_set") != "pyrua" or result.get("task_conditions") is not (arm == "conditions"):
        errors.append("framework tool/condition arm mismatch")
    if (manifest.get("arm") != arm or manifest.get("phase") != "full"
            or type(manifest.get("layout")) is not int or manifest["layout"] != layout
            or manifest.get("task_conditions") is not (arm == "conditions")):
        errors.append("episode manifest identity mismatch")
    pod = manifest.get("pod", {})
    if (not isinstance(pod, dict) or not pod.get("uid") or not pod.get("node")
            or receipt.get("pod_uid") != pod.get("uid") or receipt.get("node") != pod.get("node")):
        errors.append("first-RGB receipt and episode Pod identity disagree")
    for flag in ("first_observation_seen", "first_rgb_seen", "native_boot_completed",
                 "model_phase_entered", "native_client_cleanup_confirmed"):
        if receipt.get(flag) is not True:
            errors.append("initialization/finalization receipt flag missing: " + flag)
    if receipt.get("final_result_status") != status:
        errors.append("initialization receipt does not yet confirm this finalized result")
    timing = result.get("task_timing", {})
    if not isinstance(timing, dict) or timing.get("end_observed") is not True:
        errors.append("task timing has no observed native/conclusion end")
    if result.get("native_episode_over") is not True:
        errors.append("framework did not confirm native episode end")
    native = native_verdict(out, layout)
    row["native_outcome"] = native
    if not native["valid"]:
        errors.extend(native["errors"])
    elif result.get("environment_success") is not native["success"]:
        errors.append("framework/native verdict disagree")
    native_proof = result.get("native_outcome", {})
    if not isinstance(native_proof, dict) or native_proof.get("valid") is not True:
        errors.append("runner has no finalized valid native outcome proof")
    elif native["valid"] and native_proof.get("success") is not native["success"]:
        errors.append("runner native-outcome proof disagrees with native report")
    if row["protocol_valid"] is False:
        errors.append("original report marks this episode protocol-invalid")
    duration = result.get("task_wall_s")
    if finite_nonnegative(duration):
        # Preserve the original measured interval separately even when the
        # added interval cannot be reconstructed or the episode is incomplete.
        row["task_wall_s"] = duration
    else:
        errors.append("task_wall_s must be finite and nonnegative")
    ready = result.get("task_card_ready_at")
    first = None
    try:
        first = timestamp(receipt.get("first_rgb_at"))
    except (ValueError, OverflowError, OSError) as exc:
        errors.append(str(exc))
    if not finite_nonnegative(ready):
        errors.append("task_card_ready_at must be finite and nonnegative")
    gap = ready - first if finite_nonnegative(ready) and first is not None else None
    startup, total = result.get("startup_s"), result.get("total_wall_s")
    if not finite_nonnegative(startup) or not finite_nonnegative(total):
        errors.append("original startup_s/total_wall_s must be finite and nonnegative")
    elif finite_nonnegative(duration) and startup + duration > total + 0.01:
        errors.append("original startup and task durations exceed total runtime")
    if gap is not None:
        if not finite_nonnegative(gap):
            errors.append("first-RGB timestamp is later than Tool/task-card readiness")
        elif finite_nonnegative(startup) and gap > startup + 0.01:
            errors.append("RGB-to-ready wall-clock gap exceeds monotonic startup duration")
    if not errors and finite_nonnegative(gap) and finite_nonnegative(duration):
        added = duration + gap
        if finite_nonnegative(added):
            row.update(timing_valid_completed=True, tool_ready_overhead_s=gap,
                       scene_excluded_total_s=added, first_rgb_timestamp=first,
                       task_card_ready_at=ready)
        else:
            errors.append("combined scene-excluded duration is non-finite")
    return row


def stats(values):
    values = [v for v in values if finite_nonnegative(v)]
    return {"n": len(values), "sum": sum(values) if values else None,
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None}


def summarize(rows):
    attempted = [r for r in rows if r["attempted"]]
    completed = [r for r in rows if r["completed"]]
    timing = [r for r in rows if r["timing_valid_completed"]]
    valid = [r for r in timing if r["protocol_valid"] is True]
    native = [r for r in attempted if r.get("native_outcome", {}).get("valid")]
    fields = ("task_wall_s", "tool_ready_overhead_s", "scene_excluded_total_s")
    return {"expected_n": len(rows), "attempted_n": len(attempted),
            "pending_n": len(rows) - len(attempted), "completed_n": len(completed),
            "terminal_n": sum(r["terminal"] for r in attempted),
            "incomplete_n": sum(not r["terminal"] for r in attempted),
            "execution_failure_n": sum(r["terminal"] and not r["completed"] for r in attempted),
            "native_outcomes_n": len(native),
            "native_failure_n": sum(r["native_outcome"]["success"] is False for r in native),
            "native_success_n": sum(r["native_outcome"]["success"] is True for r in native),
            "timing_valid_completed_n": len(timing),
            "completed_without_valid_timing_n": len(completed) - len(timing),
            "protocol_valid_completed_n": len(valid),
            "protocol_validity_unavailable_n": sum(r["protocol_valid"] is None for r in completed),
            "protocol_invalid_completed_n": sum(r["protocol_valid"] is False for r in completed),
            "timing_valid_completed_metrics": {key: stats(r[key] for r in timing) for key in fields},
            "protocol_valid_completed_metrics": {key: stats(r[key] for r in valid) for key in fields}}


def build(root):
    root = Path(root).resolve()
    existing = official_rows(root)
    rows = [inspect(root, arm, layout, existing.get((arm, layout)))
            for arm in ARMS for layout in range(EXPECTED)]
    batch, error = read(root / "batch_status.json")
    return {"schema": "scene-excluded-timing-supplement/v1", "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "supplement_script_sha256": sha(__file__),
            "experiment_root": str(root), "batch_status": batch.get("status") if batch else None,
            "batch_status_read_error": error,
            "method": {"formula": "scene_excluded_total_s = task_wall_s + (task_card_ready_at - timestamp(first_rgb_at))",
                "tool_ready_overhead_s": "task_card_ready_at - timestamp(first_rgb_at)",
                "task_wall_s": "original, unchanged monotonic perf_counter interval: task card/RGB/Tool ready -> first observed native end or explicit conclusion",
                "first_rgb_at": "first RGB noticed by initialization observer in this episode process; ISO wall clock with timezone",
                "task_card_ready_at": "time.time() in the same episode process, after native SDK/SAM3 and task card preparation",
                "limitations": ["Mixed wall-clock and monotonic intervals; no precise clock-adjustment correction can be reconstructed from these records.",
                    "A wall-clock change between first RGB and task-card readiness can bias the added interval; impossible/negative/out-of-startup intervals are rejected but small undetectable adjustments remain possible.",
                    "First RGB is the observer timestamp, not the simulator render timestamp. End is observed native end or conclusion, not a reconstructed physical goal-crossing instant.",
                    "Parallel episode-duration sums are GPU workload totals, not elapsed batch wall time. Fixed treatment-first order is not randomized.",
                    "No native success verdict, frozen source, experiment control, or original timing value is changed."],
                "completed_statuses": list(COMPLETED_STATUSES),
                "denominators": "Timing-valid completed attempts are reported separately from episodes also proven protocol-valid by the original full-report and matching raw result SHA256. Missing/stale official validity is unknown, never counted as valid.",
                "native_failure_note": "Native task failures and execution failures are separate; timing includes native task failures in the eligible completed cohort."},
            "arms": {arm: summarize([r for r in rows if r["arm"] == arm]) for arm in ARMS},
            "episodes": rows}


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_live_pod(root, kubectl="kubectl", kubeconfig=None):
    identity, error = read(Path(root) / "preflight/pod-identity.json")
    if error or not all(isinstance(identity.get(k), str) and identity[k] for k in ("pod", "uid", "namespace", "node")):
        raise RuntimeError("verified Pod identity is missing or incomplete")
    command = [kubectl]
    if kubeconfig:
        command += ["--kubeconfig", str(kubeconfig)]
    command += ["-n", identity["namespace"], "get", "pod", identity["pod"], "-o", "json"]
    response = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    if response.returncode:
        raise RuntimeError("read-only Pod liveness query failed: " + response.stderr.strip()[:500])
    pod = json.loads(response.stdout)
    meta, status, spec = pod.get("metadata", {}), pod.get("status", {}), pod.get("spec", {})
    if (meta.get("uid") != identity["uid"] or meta.get("name") != identity["pod"]
            or meta.get("namespace") != identity["namespace"] or spec.get("nodeName") != identity["node"]
            or meta.get("deletionTimestamp") or status.get("phase") != "Running"):
        raise RuntimeError("experiment Pod is inactive, replaced, deleting, or moved")
    return {"pod": meta["name"], "uid": meta["uid"], "node": spec["nodeName"], "phase": status["phase"]}


def watch(root, out, *, kubectl="kubectl", kubeconfig=None,
          monotonic=time.monotonic, sleep=time.sleep, liveness=check_live_pod):
    started = monotonic()
    while True:
        report = build(root)
        elapsed = monotonic() - started
        report["watch"] = {"elapsed_s": elapsed, "poll_seconds": POLL_SECONDS,
                           "max_seconds": MAX_WATCH_SECONDS,
                           "kind": "batch-local offline reporting process; not a Codex automation"}
        state = report["batch_status"]
        if elapsed >= MAX_WATCH_SECONDS:
            report["watch"].update(status="error", error="report watch exceeded its 24-hour bound")
            write_report(out, report)
            return 3
        if state in ("finished", "needs_attention"):
            report["watch"]["status"] = state
            write_report(out, report)
            return 0 if state == "finished" else 2
        try:
            report["watch"]["pod"] = liveness(root, kubectl, kubeconfig)
        except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
            report["watch"].update(status="error", error=str(exc))
            write_report(out, report)
            return 3
        report["watch"]["status"] = "watching"
        write_report(out, report)
        sleep(POLL_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--watch", action="store_true", help="Read-only 30-second batch/Pod watch; stops within 24 hours")
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--kubeconfig", type=Path)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if not root.is_dir():
        parser.error("--root must be the existing experiment output directory")
    protected = ("full", "smoke", "preflight", "logs", "workers", "boot_attempts")
    if out == root or any(out.is_relative_to(root / directory) for directory in protected) or out.name in (
            "batch_status.json", "full-report.json", "smoke-report.json", "source-freeze.json", "stage_times.json"):
        parser.error("--out must be a new supplementary reporting path, outside raw experiment evidence")
    if args.watch:
        kubeconfig = args.kubeconfig
        default_config = Path("/root/.kube/p6000-user.conf")
        if kubeconfig is None:
            try:
                if default_config.is_file():
                    kubeconfig = default_config
            except OSError:
                pass  # Ordinary local users may not inspect /root.
        return watch(root, out, kubectl=args.kubectl, kubeconfig=kubeconfig)
    write_report(out, build(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
