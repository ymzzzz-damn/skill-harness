from pathlib import Path
import io
import json
import os
import tempfile
import unittest
import types
from unittest.mock import patch
from harnesslab.agent import DISABLED_FEATURES, audit_events, agent_environment, run_codex

class AgentTests(unittest.TestCase):
    def test_agent_cli_disables_shell_in_both_experiment_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace").mkdir()
            class Process:
                stdin = io.BytesIO()
                def poll(self): return 0
                def wait(self): return 0
            compatibility = types.ModuleType("pyrualean.generate")
            compatibility.codex_catalog_flags = lambda model: []
            compatibility.provider_overrides = lambda base, key: []
            with patch.dict("sys.modules", {"pyrualean.generate": compatibility}), \
                 patch("harnesslab.agent.subprocess.Popen", return_value=Process()) as popen:
                result = run_codex("test", [], toolkit=types.SimpleNamespace(finish=None),
                    model="fixed", reasoning_effort="high", budget_s=100,
                    out_dir=root, mcp_url="http://127.0.0.1:1/mcp")
            arguments = popen.call_args.args[0]
            for feature in DISABLED_FEATURES:
                self.assertEqual(arguments[arguments.index(feature) - 1], "--disable")
            self.assertIn("shell_tool", result["disabled_features"])

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
