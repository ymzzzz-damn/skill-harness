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
            self.assertFalse(result["image_context"]["enabled"])
            self.assertEqual(result["image_context"]["reason"], "no_explicit_responses_provider")

    def test_explicit_provider_uses_bounded_proxy_and_ephemeral_child_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace").mkdir()
            class Process:
                stdin = io.BytesIO()
                def poll(self): return 0
                def wait(self): return 0
            class Proxy:
                base_url = "http://127.0.0.1:1234/v1"
                client_api_key = "local-fixture-token"
                closed = False
                def __enter__(self): return self
                def __exit__(self, *args): self.closed = True
                def summary(self):
                    assert self.closed
                    return {"enabled": True, "max_images": 48, "requests": 0}
            compatibility = types.ModuleType("pyrualean.generate")
            compatibility.codex_catalog_flags = lambda model: []
            compatibility.provider_overrides = lambda base, key: ["-c", "proxy_base=" + str(base),
                                                                   "-c", "proxy_key=" + str(key)]
            original_key = "upstream-test-fixture-key"
            with patch.dict("sys.modules", {"pyrualean.generate": compatibility}), \
                 patch.dict(os.environ, {"TEST_PROVIDER_KEY": original_key}), \
                 patch("harnesslab.agent.ImageContextProxy", return_value=Proxy()) as factory, \
                 patch("harnesslab.agent.subprocess.Popen", return_value=Process()) as popen:
                result = run_codex("test", ["/initial/reference.png"],
                    toolkit=types.SimpleNamespace(finish=None), model="fixed", reasoning_effort="high",
                    budget_s=100, out_dir=root, mcp_url="http://127.0.0.1:1/mcp",
                    base_url="https://provider.invalid", api_key_env="TEST_PROVIDER_KEY")
                self.assertEqual(os.environ["TEST_PROVIDER_KEY"], original_key)
            factory.assert_called_once_with("https://provider.invalid/v1",
                upstream_api_key=original_key, max_images=48,
                reference_image_count=1,
                log_path=root / "image_context_requests.jsonl")
            argv, env = popen.call_args.args[0], popen.call_args.kwargs["env"]
            self.assertIn("proxy_base=http://127.0.0.1:1234/v1", argv)
            self.assertIn("proxy_key=HARNESS_IMAGE_CONTEXT_TOKEN", argv)
            self.assertIn("features.enable_request_compression=false", argv)
            self.assertIn("model_providers.pyrualean.supports_websockets=false", argv)
            self.assertNotIn("TEST_PROVIDER_KEY", env)
            self.assertNotIn(original_key, env.values())
            self.assertEqual(env["HARNESS_IMAGE_CONTEXT_TOKEN"], "local-fixture-token")
            self.assertEqual(argv[argv.index("-i") + 1], "/initial/reference.png")
            self.assertTrue(result["image_context"]["enabled"])
            self.assertEqual(json.loads((root / "image_context.json").read_text()), result["image_context"])

    def test_missing_configured_key_fails_before_starting_model(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {}, clear=True), \
             patch("harnesslab.agent.subprocess.Popen") as popen, \
             patch("harnesslab.agent.ImageContextProxy") as proxy:
            with self.assertRaisesRegex(ValueError, "environment variable is missing"):
                run_codex("test", [], toolkit=types.SimpleNamespace(finish=None), model="fixed",
                    reasoning_effort="high", budget_s=100, out_dir=Path(directory),
                    mcp_url="http://127.0.0.1:1/mcp", base_url="https://provider.invalid/v1",
                    api_key_env="MISSING_PROVIDER_KEY")
            popen.assert_not_called()
            proxy.assert_not_called()

    def test_invalid_image_budgets_fail_before_starting_model(self):
        with patch("harnesslab.agent.subprocess.Popen") as popen:
            for value in (True, 0, -1, 51, 1.5):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    run_codex("test", [], toolkit=None, model="fixed", reasoning_effort="high",
                        budget_s=100, out_dir=Path("unused"), mcp_url="http://127.0.0.1:1/mcp",
                        max_context_images=value)
            popen.assert_not_called()

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
