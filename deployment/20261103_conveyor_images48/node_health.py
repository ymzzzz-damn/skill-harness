"""One-shot, read-only host guardian for this Job's selected node.

The SSH entry host keeps its kubeconfig. The Pod writes a request marker only
after CPU preflight, then reads a UID-bound health receipt before GPU/model work.
The guardian never mutates Kubernetes resources, retries an experiment, or waits
for an unhealthy node to recover after publishing its negative receipt.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

MAX_WAIT_S = 24 * 60 * 60
POLL_S = 30
SOURCE = "ssh-host kubectl direct GET Node; no Kubernetes mutations"
_FAULT_WORDS = ("health", "pressure", "fault", "problem", "error", "unavailable",
                "corrupt", "deadlock", "lockup", "hung", "restart", "readonly", "read_only")


class NodeHealthError(ValueError):
    pass


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _text(value, field):
    if not isinstance(value, str) or not value:
        raise NodeHealthError("missing or invalid " + field)
    return value


def _uid(value):
    value = _text(value, "Pod UID")
    if not re.fullmatch(r"[A-Za-z0-9-]+", value):
        raise NodeHealthError("invalid Pod UID path component")
    return value


def _epoch(value):
    value = time.time() if value is None else value
    if not _number(value):
        raise NodeHealthError("current epoch must be finite and nonnegative")
    return value


def _binding(proof, job_uid=None):
    if not isinstance(proof, dict):
        raise NodeHealthError("Pod proof must be an object")
    actual_job = proof.get("job_uid")
    if job_uid is not None and actual_job is not None and actual_job != job_uid:
        raise NodeHealthError("Pod proof Job UID mismatch")
    return {"job_uid": _text(job_uid or actual_job, "Job UID"),
            "job_name": _text(proof.get("job_name"), "Job name"),
            "pod_uid": _uid(proof.get("uid")), "pod": _text(proof.get("pod"), "Pod name"),
            "namespace": _text(proof.get("namespace"), "namespace"),
            "node": _text(proof.get("node"), "selected node")}


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError, UnicodeError):
        raise NodeHealthError("health marker/receipt unreadable") from None
    if not isinstance(value, dict):
        raise NodeHealthError("health marker/receipt must be an object")
    return value


def node_reasons(node):
    """Evaluate only node readiness and present negative health/fault signals."""
    reasons = []
    if not isinstance(node, dict):
        return ["node object is invalid"]
    if node.get("spec", {}).get("unschedulable") is True:
        reasons.append("node is unschedulable")
    conditions = node.get("status", {}).get("conditions")
    if not isinstance(conditions, list):
        return reasons + ["node conditions unavailable"]
    ready = [c for c in conditions if isinstance(c, dict) and c.get("type") == "Ready"]
    if len(ready) != 1 or ready[0].get("status") != "True":
        reasons.append("node Ready must be True")
    for condition in conditions:
        if not isinstance(condition, dict):
            reasons.append("invalid node condition")
            continue
        kind, status = condition.get("type"), condition.get("status")
        if kind == "Ready":
            continue
        if not isinstance(kind, str):
            reasons.append("node condition has no type")
        elif any(word in kind.casefold() for word in _FAULT_WORDS) and status != "False":
            reasons.append(kind + "=" + str(status))
    return reasons


def validate_receipt(receipt, podproof, job_uid=None, *, max_age_s=180, now_epoch=None):
    """No file/network reads: validate a supplied receipt against live Pod proof.

    Pass now_epoch explicitly for deterministic/pure validation. When omitted,
    the only implicit input is the current clock; no state is mutated.
    """
    expected, current = _binding(podproof, job_uid), _epoch(now_epoch)
    if not _number(max_age_s) or max_age_s <= 0:
        raise NodeHealthError("max_age_s must be finite and positive")
    if not isinstance(receipt, dict) or receipt.get("schema") != "harness-node-health-v1":
        raise NodeHealthError("invalid node-health receipt schema")
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise NodeHealthError("node-health receipt identity mismatch")
    for key in ("node_uid", "resource_version"):
        _text(receipt.get(key), key)
    checked, requested = receipt.get("checked_at"), receipt.get("requested_at")
    if not _number(checked) or not _number(requested) or requested > checked:
        raise NodeHealthError("invalid node-health receipt timestamps")
    if checked > current:
        raise NodeHealthError("node-health receipt is from the future")
    if current - checked > max_age_s:
        raise NodeHealthError("node-health receipt is stale")
    if receipt.get("source") != SOURCE:
        raise NodeHealthError("node-health receipt is not from the host guardian")
    if receipt.get("healthy") is not True or receipt.get("reasons") != []:
        raise NodeHealthError("selected node failed host health validation")
    if node_reasons({"spec": {"unschedulable": receipt.get("node_unschedulable")},
                     "status": {"conditions": receipt.get("conditions")}}):
        raise NodeHealthError("receipt conditions contradict healthy status")
    return receipt


def request_health(podproof, job_uid=None, *, receipt_dir, now_epoch=None):
    marker = {"schema": "harness-node-health-request-v1", **_binding(podproof, job_uid),
              "requested_at": _epoch(now_epoch)}
    atomic(Path(receipt_dir) / (marker["pod_uid"] + ".request.json"), marker)
    return marker


write_health_request = request_health


def wait_for_health(podproof, job_uid=None, *, receipt_dir, timeout_s=300, max_age_s=180, poll_s=1):
    if not _number(timeout_s) or timeout_s <= 0 or not _number(poll_s) or poll_s <= 0:
        raise NodeHealthError("positive finite health wait durations required")
    binding = _binding(podproof, job_uid)
    root = Path(receipt_dir)
    marker = load(root / (binding["pod_uid"] + ".request.json"))
    if any(marker.get(k) != v for k, v in binding.items()) or not _number(marker.get("requested_at")):
        raise NodeHealthError("node-health request identity mismatch")
    deadline = time.monotonic() + timeout_s
    path = root / (binding["pod_uid"] + ".json")
    while time.monotonic() < deadline:
        if path.exists():
            receipt = load(path)
            if receipt.get("requested_at") != marker["requested_at"]:
                raise NodeHealthError("node-health receipt belongs to a different request")
            return validate_receipt(receipt, podproof, job_uid, max_age_s=max_age_s, now_epoch=time.time())
        time.sleep(min(poll_s, max(0, deadline - time.monotonic())))
    raise NodeHealthError("host node-health receipt wait timed out")


def kubectl_get(kubeconfig, kind, name=None, *, namespace=None, selector=None):
    command = ["kubectl", "--kubeconfig", str(kubeconfig), "--request-timeout=15s"]
    if namespace:
        command += ["-n", namespace]
    command += ["get", kind]
    if name:
        command.append(name)
    if selector:
        command += ["-l", selector]
    command += ["-o", "json"]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        raise NodeHealthError("host Kubernetes read failed or timed out: " + kind) from None
    if completed.returncode:
        # Never copy kubeconfig, command output, or authentication diagnostics
        # into a Pod-visible receipt or stdout/stderr.
        raise NodeHealthError("host Kubernetes read failed: " + kind)
    try:
        value = json.loads(completed.stdout)
    except ValueError:
        raise NodeHealthError("host Kubernetes read returned invalid JSON: " + kind) from None
    if not isinstance(value, dict):
        raise NodeHealthError("host Kubernetes read returned non-object: " + kind)
    return value


def select_pod(pods, job_name, job_uid, namespace):
    if not isinstance(pods.get("items"), list):
        raise NodeHealthError("Pod list is invalid")
    if not pods["items"]:
        return None
    if len(pods["items"]) != 1:
        raise NodeHealthError("expected exactly one Pod for the pinned Job UID")
    pod = pods["items"][0]
    meta = pod.get("metadata", {})
    owners = [owner for owner in meta.get("ownerReferences", [])
              if owner.get("kind") == "Job" and owner.get("controller") is True]
    if (len(owners) != 1 or owners[0].get("uid") != job_uid or owners[0].get("name") != job_name
            or meta.get("namespace") != namespace):
        raise NodeHealthError("Pod controller Job owner UID/name/namespace mismatch")
    if meta.get("deletionTimestamp") or pod.get("status", {}).get("phase") in ("Succeeded", "Failed"):
        raise NodeHealthError("selected Pod is terminal or deleting")
    return pod


def run_guardian(job_name, job_uid, namespace, kubeconfig, receipt_dir, *,
                 read=kubectl_get, clock=time.time, monotonic=time.monotonic, sleep=time.sleep):
    root = Path(receipt_dir)
    started, deadline = monotonic(), monotonic() + MAX_WAIT_S
    base = {"job_name": job_name, "job_uid": job_uid, "namespace": namespace,
            "source": SOURCE, "kubernetes_mutations": 0, "model_calls": 0, "gpu_calls": 0}
    def status(state, **values):
        atomic(root / "guardian-status.json", {**base, "status": state, "updated_at": clock(),
            "elapsed_s": monotonic() - started, **values})
    try:
        while monotonic() < deadline:
            job = read(kubeconfig, "job", job_name, namespace=namespace)
            meta = job.get("metadata", {})
            if meta.get("uid") != job_uid or meta.get("name") != job_name or meta.get("namespace") != namespace:
                raise NodeHealthError("tracked Job identity changed")
            if meta.get("deletionTimestamp") or any(c.get("status") == "True" and c.get("type") in ("Complete", "Failed")
                    for c in job.get("status", {}).get("conditions", [])):
                raise NodeHealthError("tracked Job is terminal or deleting")
            pods = read(kubeconfig, "pods", namespace=namespace, selector="job-name=" + job_name)
            pod = select_pod(pods, job_name, job_uid, namespace)
            if pod is None:
                status("waiting_for_pod")
            else:
                meta, spec = pod["metadata"], pod.get("spec", {})
                if not spec.get("nodeName") or pod.get("status", {}).get("phase") != "Running":
                    status("waiting_for_running_assignment", pod_uid=meta.get("uid"), node=spec.get("nodeName"))
                else:
                    proof = {"pod": meta.get("name"), "uid": meta.get("uid"), "namespace": namespace,
                             "node": spec["nodeName"], "job_uid": job_uid, "job_name": job_name}
                    binding = _binding(proof, job_uid)
                    request_path = root / (binding["pod_uid"] + ".request.json")
                    if not request_path.exists():
                        status("waiting_for_bootstrap_request", pod_uid=binding["pod_uid"], node=binding["node"])
                    else:
                        marker = load(request_path)
                        if (marker.get("schema") != "harness-node-health-request-v1"
                                or any(marker.get(k) != v for k, v in binding.items())
                                or not _number(marker.get("requested_at")) or marker["requested_at"] > clock()):
                            raise NodeHealthError("bootstrap health request identity/time mismatch")
                        node = read(kubeconfig, "node", binding["node"])
                        node_meta = node.get("metadata", {})
                        if node_meta.get("name") != binding["node"]:
                            raise NodeHealthError("direct Node identity mismatch")
                        node_uid = _text(node_meta.get("uid"), "Node UID")
                        version = _text(node_meta.get("resourceVersion"), "Node resourceVersion")
                        reasons = node_reasons(node)
                        receipt = {"schema": "harness-node-health-v1", **binding,
                            "node_uid": node_uid, "resource_version": version,
                            "requested_at": marker["requested_at"], "checked_at": clock(),
                            "healthy": not reasons, "reasons": reasons, "source": SOURCE,
                            "node_unschedulable": node.get("spec", {}).get("unschedulable", False),
                            "conditions": [{"type": c.get("type"), "status": c.get("status"), "reason": c.get("reason")}
                                           for c in node.get("status", {}).get("conditions", []) if isinstance(c, dict)]}
                        atomic(root / (binding["pod_uid"] + ".json"), receipt)
                        status("healthy" if receipt["healthy"] else "unhealthy", receipt=str(root / (binding["pod_uid"] + ".json")),
                               pod_uid=binding["pod_uid"], node=binding["node"], reasons=reasons)
                        return 0 if receipt["healthy"] else 2
            sleep(min(POLL_S, max(0, deadline - monotonic())))
        raise NodeHealthError("guardian exceeded 24-hour scheduling/request wait")
    except Exception as exc:
        # Only our sanitized error text is surfaced; unknown errors use the type.
        error = str(exc) if isinstance(exc, NodeHealthError) else type(exc).__name__
        status("error", error=error)
        return 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("job-name", "job-uid", "namespace", "kubeconfig", "receipt-dir"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args()
    return run_guardian(args.job_name, args.job_uid, args.namespace, args.kubeconfig, args.receipt_dir)


if __name__ == "__main__":
    raise SystemExit(main())
