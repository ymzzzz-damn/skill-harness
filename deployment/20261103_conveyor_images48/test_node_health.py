import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import node_health as h


class NodeHealthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proof = {"pod": "pod", "uid": "pod-uid", "namespace": "ns", "node": "node",
                      "job_uid": "job-uid", "job_name": "job"}
        self.node = {"metadata": {"name": "node", "uid": "node-uid", "resourceVersion": "123"},
            "spec": {}, "status": {"conditions": [{"type": "Ready", "status": "True"},
                {"type": "GPUUnhealthy", "status": "False"}, {"type": "MemoryPressure", "status": "False"}]}}
        self.job = {"metadata": {"name": "job", "uid": "job-uid", "namespace": "ns"}}
        self.pod = {"metadata": {"name": "pod", "uid": "pod-uid", "namespace": "ns",
            "ownerReferences": [{"kind": "Job", "controller": True, "name": "job", "uid": "job-uid"}]},
            "spec": {"nodeName": "node"}, "status": {"phase": "Running"}}

    def reader(self, kubeconfig, kind, name=None, **kwargs):
        return copy.deepcopy({"job": self.job, "pods": {"items": [self.pod]}, "node": self.node}[kind])

    def guardian(self):
        return h.run_guardian("job", "job-uid", "ns", "/not-copied/kubeconfig", self.root,
            read=self.reader, clock=lambda: 100.0, monotonic=lambda: 0.0,
            sleep=lambda delay: self.fail("guardian did not exit after requested one-shot check"))

    def receipt(self):
        h.request_health(self.proof, receipt_dir=self.root, now_epoch=99)
        self.assertEqual(self.guardian(), 0)
        return json.loads((self.root / "pod-uid.json").read_text())

    def test_matching_receipt_binds_live_job_pod_node_and_request(self):
        receipt = self.receipt()
        self.assertEqual(receipt["node_uid"], "node-uid")
        self.assertEqual(receipt["resource_version"], "123")
        self.assertIs(h.validate_receipt(receipt, self.proof, "job-uid", now_epoch=101), receipt)
        with patch.object(h.time, "time", return_value=101):
            self.assertEqual(h.wait_for_health(self.proof, receipt_dir=self.root)["healthy"], True)

    def test_stale_future_nonfinite_and_unhealthy_receipts_rejected(self):
        receipt = self.receipt()
        for field, value, current in (("checked_at", 100, 281), ("checked_at", 102, 101),
                ("checked_at", float("nan"), 101), ("healthy", False, 101),
                ("requested_at", 105, 101)):
            changed = copy.deepcopy(receipt)
            changed[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(h.NodeHealthError):
                h.validate_receipt(changed, self.proof, now_epoch=current)

    def test_owner_uid_mismatch_never_checks_or_writes_node_receipt(self):
        self.pod["metadata"]["ownerReferences"][0]["uid"] = "other-job"
        self.assertEqual(self.guardian(), 3)
        self.assertFalse((self.root / "pod-uid.json").exists())
        status = json.loads((self.root / "guardian-status.json").read_text())
        self.assertIn("owner UID", status["error"])

    def test_gpu_unhealthy_and_unknown_pressure_rejected_despite_ready_true(self):
        for index, status in ((1, "True"), (1, "Unknown"), (2, "Unknown")):
            self.node["status"]["conditions"][1]["status"] = "False"
            self.node["status"]["conditions"][2]["status"] = "False"
            self.node["status"]["conditions"][index]["status"] = status
            h.request_health(self.proof, receipt_dir=self.root, now_epoch=99)
            self.assertEqual(self.guardian(), 2)
            receipt = json.loads((self.root / "pod-uid.json").read_text())
            self.assertFalse(receipt["healthy"])
            with self.assertRaises(h.NodeHealthError):
                h.validate_receipt(receipt, self.proof, now_epoch=101)

    def test_ready_false_unknown_or_missing_rejected(self):
        for value in ("False", "Unknown", None):
            node = copy.deepcopy(self.node)
            node["status"]["conditions"][0]["status"] = value
            self.assertTrue(h.node_reasons(node))
        node = copy.deepcopy(self.node)
        node["status"]["conditions"] = []
        self.assertTrue(h.node_reasons(node))

    def test_wrong_job_pod_node_or_request_cannot_reuse_receipt(self):
        receipt = self.receipt()
        for key in ("job_uid", "pod_uid", "node", "namespace"):
            bad = dict(receipt, **{key: "different"})
            with self.subTest(key=key), self.assertRaises(h.NodeHealthError):
                h.validate_receipt(bad, self.proof, now_epoch=101)
        h.request_health(self.proof, receipt_dir=self.root, now_epoch=100)
        with self.assertRaisesRegex(h.NodeHealthError, "different request"):
            h.wait_for_health(self.proof, receipt_dir=self.root)

    def test_terminal_pod_and_changed_job_exit_without_receipt(self):
        self.pod["status"]["phase"] = "Failed"
        self.assertEqual(self.guardian(), 3)
        self.pod["status"]["phase"] = "Running"
        self.job["metadata"]["uid"] = "new-job"
        self.assertEqual(self.guardian(), 3)
        self.assertFalse((self.root / "pod-uid.json").exists())

    def test_waits_for_running_and_cpu_preflight_marker_before_node_read(self):
        self.pod["status"]["phase"] = "Pending"
        sleeps, kinds = [], []
        def reader(*args, **kwargs):
            kinds.append(args[1])
            return self.reader(*args, **kwargs)
        def sleep(delay):
            sleeps.append(delay)
            if len(sleeps) == 1:
                self.pod["status"]["phase"] = "Running"
            else:
                h.request_health(self.proof, receipt_dir=self.root, now_epoch=99)
        code = h.run_guardian("job", "job-uid", "ns", "not-copied", self.root,
            read=reader, clock=lambda: 100, monotonic=lambda: 0, sleep=sleep)
        self.assertEqual(code, 0)
        self.assertEqual(sleeps, [30, 30])
        self.assertEqual(kinds.count("node"), 1)

    def test_guardian_24_hour_bound_and_sanitized_api_error(self):
        ticks = iter((0, 0, 86400, 86400))
        code = h.run_guardian("job", "job-uid", "ns", "secret-path", self.root,
            read=self.reader, clock=lambda: 100, monotonic=lambda: next(ticks), sleep=lambda _: None)
        self.assertEqual(code, 3)
        status = (self.root / "guardian-status.json").read_text()
        self.assertIn("24-hour", status)
        self.assertNotIn("secret-path", status)
        with patch.object(h.subprocess, "run", side_effect=OSError("SECRET_CREDENTIAL")):
            with self.assertRaises(h.NodeHealthError) as caught:
                h.kubectl_get("SECRET_CREDENTIAL", "node", "node")
        self.assertNotIn("SECRET_CREDENTIAL", str(caught.exception))


if __name__ == "__main__": unittest.main()
