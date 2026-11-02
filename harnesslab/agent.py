"""Independent agent process lifecycle with no automatic model replays."""
from __future__ import annotations
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Callable

def audit_events(path: Path) -> dict[str, Any]:
    total_usage: dict[str, int] = {}
    tool_calls: dict[str, int] = {}
    errors, thread_ids = [], []
    turns = 0
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started" and event.get("thread_id"):
            thread_ids.append(str(event["thread_id"]))
        if event.get("type") == "turn.completed":
            turns += 1
            for key, value in (event.get("usage") or {}).items():
                if isinstance(value, int):
                    total_usage[key] = total_usage.get(key, 0) + value
        if event.get("type") in ("error", "turn.failed"):
            errors.append(event)
        item = event.get("item") or {}
        if event.get("type") == "item.completed":
            if item.get("type") in ("mcp_tool_call", "mcpToolCall"):
                name = str(item.get("tool", "?"))
                tool_calls[name] = tool_calls.get(name, 0) + 1
            elif item.get("type") == "error":
                errors.append(item)
    return {"runtime": "codex", "turns": turns, "thread_ids": thread_ids,
            "tool_calls": tool_calls, "errors": errors, **total_usage}

def agent_environment(codex_home: str | None, api_key_env: str | None) -> dict[str, str]:
    keep = {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "TMPDIR",
            "TZ", "XDG_CACHE_HOME", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"}
    env = {key: value for key, value in os.environ.items() if key in keep}
    if api_key_env and api_key_env in os.environ:
        env[api_key_env] = os.environ[api_key_env]
    env["SHELL"] = "/bin/bash"
    if codex_home:
        env["CODEX_HOME"] = str(Path(codex_home).resolve())
    loopback = "localhost,127.0.0.1,::1"
    upstream = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
    env["NO_PROXY"] = env["no_proxy"] = ",".join(filter(None, [upstream, loopback]))
    return env

def run_codex(prompt: str, images: list[str], *, toolkit: Any, model: str,
              reasoning_effort: str, budget_s: float, out_dir: Path,
              mcp_url: str, codex_bin: str = "codex", codex_home: str | None = None,
              base_url: str | None = None, api_key_env: str | None = None,
              should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
    # Only provider/catalog argument compatibility comes from the pinned SDK.
    from pyrualean.generate import codex_catalog_flags, provider_overrides
    workspace = out_dir / "workspace"
    argv = [codex_bin, "exec", "-m", model,
            "-c", f"model_reasoning_effort={json.dumps(reasoning_effort)}",
            "-c", f"mcp_servers.harness.url={json.dumps(mcp_url)}",
            "-c", "project_doc_max_bytes=0",
            *codex_catalog_flags(model), "--disable", "shell_snapshot",
            "--disable", "memories", *provider_overrides(base_url, api_key_env),
            "-s", "workspace-write", "-C", str(workspace),
            "--skip-git-repo-check", "--json", "-o", str(out_dir / "last_message.md")]
    for image in images:
        argv.extend(["-i", str(image)])
    argv.append("-")
    events = out_dir / "codex_events.jsonl"
    started, status, returncode = time.perf_counter(), "completed", None
    with events.open("wb") as output, (out_dir / "codex_stderr.txt").open("wb") as error:
        process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=output, stderr=error,
                                   cwd=workspace, env=agent_environment(codex_home, api_key_env),
                                   start_new_session=True)
        assert process.stdin is not None
        process.stdin.write(prompt.encode("utf-8"))
        process.stdin.close()
        reason = None
        while process.poll() is None:
            if time.perf_counter() - started >= budget_s:
                status, reason = "timeout", "timeout"
                break
            # Finish or native end should not leave the model spinning until
            # the episode ceiling. Give it a short chance to complete its reply.
            if toolkit.finish is not None or (should_stop is not None and should_stop()):
                reason = "native_end" if toolkit.finish is None else "agent_finish"
                try:
                    process.wait(timeout=min(5.0, max(0.1, budget_s -
                                                   (time.perf_counter() - started))))
                except subprocess.TimeoutExpired:
                    pass
                break
            time.sleep(0.1)
        if reason is not None and process.poll() is None:
            if reason == "timeout":
                toolkit.robo._stop_episode("timeout")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        returncode = process.wait()
    generation = audit_events(events)
    generation.update(status=status, returncode=returncode, attempts=1,
                      wall_s=time.perf_counter() - started)
    return generation
