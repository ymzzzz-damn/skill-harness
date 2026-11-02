"""Deployment-only native-startup isolation and conservative retry receipts.

Adapted from first-party grasp r6 deployment; see ORIGINS.json. Only paths,
receipt names and environment names change; native scoring/control is unchanged.

Kit path/CLI references:
https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/tokens.html
https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/Kernel/CommandLineOptions.html
"""
from __future__ import annotations
import datetime as dt
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import tempfile
import threading
import time

PIP_ARCHIVE = Path("/mnt/auwomo-data/RoboDojo/qwenpaw-harness/isaac-cache/exts/v2/omni.kit.pip_archive-d38fa9ecd1fb6df4")

def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)

def observation_rgb_shapes(obs):
    result = {}
    for name, camera in (obs.get("vision") or {}).items():
        if not isinstance(camera, dict):
            continue
        color = camera.get("color")
        shape = getattr(color, "shape", None)
        if shape is not None and len(shape) == 3 and shape[0] > 0 and shape[1] > 0 and shape[2] >= 3:
            result[str(name)] = [int(x) for x in shape]
    return result

def live_group_pids(pgid):
    live = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                live.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return live

def retry_decision(state, evidence, cleanup_ok, status):
    known = state.get("native_boot_error", {}).get("retryable_initialization_error") is True
    return bool(status == "boot_error" and state.get("native_boot_started") is True
        and known and state.get("first_observation_seen") is False
        and state.get("first_rgb_seen") is False and state.get("native_boot_completed") is False
        and state.get("model_phase_entered") is False and cleanup_ok is True
        and not any(evidence.values()))

class NativeStartup:
    def __init__(self, out, runtime_root, proof, workspace, shared_script, *, slots=2, timeout_s=900):
        if not 1 <= slots <= 8 or not 30 <= timeout_s <= 1800:
            raise ValueError("Native startup slots must be 1..8 and timeout 30..1800s.")
        self.out, self.runtime_root = Path(out), Path(runtime_root)
        self.proof, self.workspace = proof, Path(workspace)
        self.shared_script, self.slots, self.timeout_s = Path(shared_script), slots, timeout_s
        self.receipt_path = self.out / "initialization-receipt.json"
        self.lock = threading.RLock()
        self.slot_handle = None
        self.clients = []
        self.finalized = False
        self.state = {"schema": "harness-native-initialization/v1", "at": now(),
            "first_observation_seen": False, "first_rgb_seen": False,
            "native_boot_started": False, "native_boot_completed": False,
            "model_phase_entered": False, "native_action_requests": 0,
            "model_calls": 0, "native_actions": 0, "retry_eligible": False,
            "native_boot_error": {}, "native_client_cleanup_confirmed": False,
            "startup_concurrency_limit": slots, "native_boot_timeout_s": timeout_s,
            "pod_uid": proof["uid"], "node": proof["node"]}
        self.save()

    def save(self):
        with self.lock:
            self.state["updated_at"] = now()
            atomic(self.receipt_path, self.state)

    def prepare_paths(self):
        kit = self.out / "kit_runtime"
        config = kit / "config"
        data = kit / "data"
        logs = kit / "logs"
        cache = self.runtime_root / "omni-cache"
        portable = self.runtime_root / "kit-portable"
        for path in (config, data, logs, cache, portable):
            path.mkdir(parents=True, exist_ok=True)
        manifest = PIP_ARCHIVE / "config/extension.toml"
        if not manifest.is_file() or not (PIP_ARCHIVE / "pip_prebundle").is_dir():
            raise FileNotFoundError("The complete existing read-only pip_archive is unavailable.")
        toml = "[paths]\n" + "\n".join(key + " = " + json.dumps(str(value))
            for key, value in (("data_root", data), ("cache_root", cache), ("logs_root", logs))) + "\n"
        (config / "omniverse.toml").write_text(toml)
        os.environ["OMNI_CONFIG_PATH"] = str(config)
        self.kit_log = logs / "kit.log"
        self.kit_args = ["--portable-root", str(portable), "--ext-path", str(PIP_ARCHIVE), "--/log/file=" + str(self.kit_log),
            "--/log/includeProcessId=true", "--/app/userConfigPath=" + str(kit / "user.config.json")]
        if any(any(char.isspace() for char in argument) for argument in self.kit_args):
            raise ValueError("Deployment Kit paths must not contain whitespace.")
        self.state["kit_isolation"] = {"omni_config_path": str(config), "data_root": str(data),
            "cache_root": str(cache), "logs_root": str(logs), "log_file": str(self.kit_log),
            "portable_root": str(portable), "portable_scope": "per lane; explicit Kit log and user config per episode",
            "extension_path": str(PIP_ARCHIVE), "extension_manifest_sha256": sha(manifest),
            "kit_args": self.kit_args, "shared_assets_modified": False}
        self.save()

    def patch_bench(self, bench):
        bench = Path(bench)
        bench_real, out_real = bench.resolve(strict=True), self.out.resolve(strict=True)
        if bench_real == out_real or not bench_real.is_relative_to(out_real):
            raise RuntimeError("Benchmark launch overlay must remain within this episode's bench.")
        script = bench / "scripts/eval_policy.sh"
        parent_real = script.parent.resolve(strict=True)
        if not parent_real.is_relative_to(bench_real):
            raise RuntimeError("Benchmark scripts directory escapes the episode bench.")
        if script.absolute() == self.shared_script.absolute():
            raise RuntimeError("Refusing to replace the shared benchmark launch script.")
        shared_before_sha = sha(self.shared_script)
        before = script.read_text()
        if hashlib.sha256(before.encode()).hexdigest() != shared_before_sha:
            raise RuntimeError("Unexpected preexisting benchmark launch-script modification.")
        marker = 'MAX_BASH_RETRIES="${ROBODOJO_MAX_BASH_RETRIES:-10}"'
        if before.count(marker) != 1:
            raise RuntimeError("Native launch-script insertion point drifted.")
        addition = "# r6 deployment-only Kit paths; no benchmark/control change.\nKIT_ARGS+=" + shlex.quote(" " + " ".join(self.kit_args)) + "\n\n"
        after = before.replace(marker, addition + marker)
        # Always replace the directory entry with a new regular file. Writing the
        # existing path could follow a symlink or mutate a hard-linked source.
        original_is_symlink = script.is_symlink()
        mode = stat.S_IMODE(script.stat().st_mode)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".eval_policy.", suffix=".tmp", dir=parent_real)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(after)
            temporary.chmod(mode)
            os.replace(temporary, script)
        finally:
            temporary.unlink(missing_ok=True)
        if script.is_symlink() or not script.resolve(strict=True).is_relative_to(bench_real):
            raise RuntimeError("Generated launch overlay did not remain an independent bench file.")
        shared_after_sha = sha(self.shared_script)
        if shared_after_sha != shared_before_sha or os.path.samefile(script, self.shared_script):
            raise RuntimeError("Shared launch script changed or generated overlay aliases its source.")
        diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile="shared-original/scripts/eval_policy.sh", tofile="episode-overlay/scripts/eval_policy.sh"))
        (self.out / "native-launch-overlay.diff").write_text(diff)
        self.state["launch_overlay"] = {"script": str(script), "shared_source": str(self.shared_script),
            "original_sha256": hashlib.sha256(before.encode()).hexdigest(), "patched_sha256": sha(script),
            "shared_source_sha256_before": shared_before_sha, "shared_source_sha256_after": shared_after_sha,
            "original_was_symlink": original_is_symlink, "independent_regular_file": True,
            "diff": str(self.out / "native-launch-overlay.diff"), "scope": "generated episode bench only"}
        self.save()

    def acquire_slot(self):
        root = self.workspace / "harness/runtime/20261102_conveyor_conditions/native-boot-slots" / self.proof["uid"]
        root.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        deadline = start + float(os.environ.get("HARNESS_NATIVE_BOOT_QUEUE_TIMEOUT_S", "3600"))
        while time.monotonic() < deadline:
            for index in range(self.slots):
                handle = (root / f"slot_{index}.lock").open("a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.slot_handle = handle
                    self.state["startup_slot"] = index
                    self.state["startup_queue_wait_s"] = time.monotonic() - start
                    self.save()
                    return
                except BlockingIOError:
                    handle.close()
            time.sleep(0.25)
        raise TimeoutError("Native startup concurrency slot was not available.")

    def release_slot(self):
        with self.lock:
            if self.slot_handle is not None:
                fcntl.flock(self.slot_handle, fcntl.LOCK_UN)
                self.slot_handle.close()
                self.slot_handle = None
                self.state["startup_slot_released_at"] = now()
                self.save()

    def observe(self, obs):
        with self.lock:
            self.state["first_observation_seen"] = True
            self.state["model_calls"] = None
            self.state.setdefault("first_observation_at", now())
            shapes = observation_rgb_shapes(obs if isinstance(obs, dict) else {})
            if shapes:
                self.state["first_rgb_seen"] = True
                self.state.setdefault("first_rgb_at", now())
                self.state["rgb_shapes"] = shapes
                self.state["kit_explicit_log_exists"] = self.kit_log.is_file()
                if not self.state["kit_explicit_log_exists"]:
                    self.state["deployment_warning"] = "First RGB arrived before the explicit Kit log was visible; inspect the native log for path relocation or delayed flush."
                else:
                    self.state["read_only_extension_path_seen_in_kit_log"] = str(PIP_ARCHIVE) in self.kit_log.read_text(errors="replace")
            self.save()
            if shapes:
                self.release_slot()

    def note_action(self):
        self.state["native_action_requests"] += 1
        self.state["native_actions"] = self.state["native_action_requests"]
        self.save()

    def note_model(self):
        self.state["model_phase_entered"] = True
        self.state["model_calls"] = None
        self.save()

    def install(self, host, bridge_cls):
        original_layout, original_boot = host.lay_out, host.boot
        original_update, original_first = bridge_cls.update_obs, bridge_cls.first_obs
        original_client_init = host._Client.__init__
        def layout(*args, **kwargs):
            bench = original_layout(*args, **kwargs)
            self.patch_bench(bench)
            return bench
        def update(bridge, obs):
            self.observe(obs)
            return original_update(bridge, obs)
        def first(bridge, *args, **kwargs):
            obs = original_first(bridge, *args, **kwargs)
            self.observe(obs)
            return obs
        def client_init(client, proc, *args, **kwargs):
            original_client_init(client, proc, *args, **kwargs)
            self.clients.append(proc)
            self.state["native_client_pids"] = [p.pid for p in self.clients]
            self.save()
        def boot(*args, **kwargs):
            self.acquire_slot()
            self.state["native_boot_started"] = True
            self.state["native_boot_started_at"] = now()
            self.save()
            kwargs["boot_timeout_s"] = self.timeout_s
            try:
                toolkit, daemons = original_boot(*args, **kwargs)
                self.state["native_boot_completed"] = True
                self.save()
                return toolkit, daemons
            except BaseException as exc:
                known = isinstance(exc, TimeoutError) or (isinstance(exc, RuntimeError)
                    and "evaluation client exited before its first observation" in str(exc))
                self.state["native_boot_error"] = {"type": type(exc).__name__, "message": str(exc),
                    "retryable_initialization_error": known}
                self.save()
                raise
            finally:
                self.release_slot()
        host.lay_out, host.boot = layout, boot
        bridge_cls.update_obs, bridge_cls.first_obs = update, first
        host._Client.__init__ = client_init

    def cleanup_failed_boot(self):
        groups = [proc.pid for proc in self.clients]
        # SDK launches each tracked native shell with start_new_session=True.
        # Its own normal stop has already run before the boot exception returns.
        for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
            pending = [group for group in groups if live_group_pids(group)]
            if not pending:
                break
            for group in pending:
                if group != os.getpgrp():
                    try:
                        os.killpg(group, sig)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline and any(live_group_pids(group) for group in pending):
                time.sleep(0.1)
        left = {str(group): live_group_pids(group) for group in groups if live_group_pids(group)}
        self.state["native_client_cleanup_remaining"] = left
        return not left

    def finalize(self, result=None):
        if self.finalized:
            return
        self.finalized = True
        result = result or {}
        if not self.state["native_boot_completed"]:
            cleanup_ok = self.cleanup_failed_boot()
        else:
            cleanup_ok = not any(live_group_pids(proc.pid) for proc in self.clients)
        self.state["native_client_cleanup_confirmed"] = cleanup_ok
        evidence = {"native_action_requests": self.state["native_action_requests"],
            "model_phase_entered": self.state["model_phase_entered"],
            "steps_present": any((self.out / "steps").glob("*")),
            "native_report_present": any((self.out / "bench/eval_result").rglob("_result.json")),
            "session_present": any((self.out / "codex-home/sessions").rglob("*.jsonl"))}
        for name in ("calls.jsonl", "native_actions.jsonl", "codex_events.jsonl", "tool_spans.jsonl"):
            path = self.out / name
            evidence[name] = path.is_file() and path.stat().st_size > 0
        for key in ("env_steps", "env_steps_total", "tool_turns"):
            evidence[key] = result.get(key) not in (0, None)
        evidence["generation_present"] = bool(result.get("generation"))
        self.state["final_evidence"] = evidence
        self.state["retry_eligible"] = retry_decision(self.state, evidence, cleanup_ok, result.get("status"))
        self.state["final_result_status"] = result.get("status")
        self.release_slot()
        self.save()
