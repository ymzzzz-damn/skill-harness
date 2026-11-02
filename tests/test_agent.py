from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from harnesslab.agent import audit_events, agent_environment

class AgentTests(unittest.TestCase):
    def test_usage_sums_completed_turns_and_preserves_errors(self):
        temporary_root = Path(__file__).resolve().parents[2] / "experiments/cpu-tests/tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as root:
            path = Path(root) / "events.jsonl"
            events = [
                {"type": "thread.started", "thread_id": "fresh"},
                {"type": "turn.completed", "usage": {"input_tokens": 11, "output_tokens": 3}},
                {"type": "turn.completed", "usage": {"input_tokens": 19, "output_tokens": 7}},
                {"type": "error", "message": "provider dropped"},
            ]
            path.write_text("\n".join(json.dumps(event) for event in events) + "\ninvalid")
            result = audit_events(path)
            self.assertEqual(result["input_tokens"], 30)
            self.assertEqual(result["output_tokens"], 10)
            self.assertEqual(result["thread_ids"], ["fresh"])
            self.assertEqual(len(result["errors"]), 1)

    def test_agent_environment_does_not_inherit_other_project_paths(self):
        with patch.dict(os.environ, {"PATH": "/bin", "CODEX_HOME": "/private/old",
                        "PYTHONPATH": "/hidden", "API_KEY": "test-fixture-key",
                        "UNRELATED_SECRET": "test-unrelated"}, clear=True):
            env = agent_environment("/fresh/home", "API_KEY")
            self.assertEqual(env["CODEX_HOME"], "/fresh/home")
            self.assertEqual(env["API_KEY"], "test-fixture-key")
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("UNRELATED_SECRET", env)
            self.assertIn("127.0.0.1", env["NO_PROXY"])

if __name__ == "__main__":
    unittest.main()
