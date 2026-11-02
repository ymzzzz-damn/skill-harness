"""One committed native episode. Use a fresh process; no measured episode replay."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback
from identity import cuda_mapping, gpu_inventory, own_pod
from protocol import (ARMS, ASSETS, CODEX, CONTROLS, DEPLOYMENT, OUTPUT, PROJECT,
                      PYRUA, RUNTIME, SOURCE, TASK, WORKSPACE, episode_dir,
                      load, now, sha, verify_freeze, write)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--phase', choices=('smoke', 'full'), required=True)
    parser.add_argument('--layout', type=int, required=True)
    parser.add_argument('--gpu', type=int, choices=range(8), required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--allow-model-calls', action='store_true')
    parser.add_argument('--resource-wait-s', type=float, default=120)
    args = parser.parse_args()
    if not args.allow_model_calls:
        parser.error('Activation flag required after experiment authorization')
    if not 0 <= args.layout < 55 or args.gpu != args.layout % 8:
        parser.error('Each layout must use the same layout%8 GPU lane in both stages')
    frozen = json.loads(args.freeze.read_text())
    verify_freeze(frozen)
    proof = own_pod()  # API read precedes CUDA, simulator and model execution.
    out = episode_dir(args.phase, args.arm, args.layout)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = (out.parent / 'launch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if list(out.parent.glob('episode_[0-9][0-9][0-9]')):
        raise FileExistsError('Refusing a duplicate/measured episode replay: ' + str(out))
    out.mkdir()
    runtime = RUNTIME / 'lanes' / str(args.gpu)
    leases, startup, result = [], None, None

    def state(status, **values):
        write(out.parent / 'launch_status.json', {'status': status, 'at': now(),
            'pid': os.getpid(), 'arm': args.arm, 'phase': args.phase,
            'layout': args.layout, 'gpu': args.gpu, **values})

    state('preflight', pod=proof)
    try:
        paths = [RUNTIME / 'python-deps', SOURCE, PYRUA / 'source/src']
        os.environ.pop('PYTHONHOME', None)
        os.environ.update({'PATH': '/opt/conda/envs/RoboDojo/bin:' + os.environ.get('PATH', ''),
            'PYTHONPATH': os.pathsep.join(map(str, paths)), 'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONNOUSERSITE': '1', 'SAM3_CHECKPOINT_PATH': str(PYRUA / 'runtime/checkpoints/sam3.pt'),
            'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'OMNI_KIT_ACCEPT_EULA': 'YES', 'ACCEPT_EULA': 'Y',
            'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1'})
        for env, directory in (('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'cache'), ('HF_HOME', 'huggingface'),
            ('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'), ('TORCH_HOME', 'torch'),
            ('NUMBA_CACHE_DIR', 'numba'), ('CUDA_CACHE_PATH', 'cuda-cache'),
            ('TRITON_CACHE_DIR', 'triton'), ('TORCH_EXTENSIONS_DIR', 'torch-extensions')):
            path = runtime / directory
            path.mkdir(parents=True, exist_ok=True)
            os.environ[env] = str(path)
        for path in reversed(paths):
            sys.path.insert(0, str(path))
        mapping = cuda_mapping(gpu_inventory(), args.gpu)
        selected = mapping['devices'][args.gpu]
        record_path = RUNTIME / 'gpu-mapping' / proof['uid'] / f'lane_{args.gpu}.json'
        previous = load(record_path)
        if previous and [r['uuid'] for r in previous['devices']] != [r['uuid'] for r in mapping['devices']]:
            raise RuntimeError('Same-GPU comparison mapping changed within the allocated Pod')
        write(record_path, mapping)
        deadline = time.monotonic() + args.resource_wait_s
        while True:
            row = next(r for r in gpu_inventory() if r['uuid'] == selected['uuid'])
            lease_path = RUNTIME / 'gpu-leases' / proof['node'] / (selected['uuid'] + '.lock')
            lease_path.parent.mkdir(parents=True, exist_ok=True)
            lease = lease_path.open('a')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if row['free_mib'] < 22000:
                    raise BlockingIOError('Less than 22000 MiB free on the allocated lane')
                leases.append(lease)
                break
            except BlockingIOError:
                lease.close()
                if time.monotonic() >= deadline:
                    state('infrastructure_blocked', reason='Allocated GPU resource check failed',
                          model_calls=0, native_actions=0)
                    return 3
                time.sleep(min(5, max(0, deadline - time.monotonic())))
        write(out / 'pod-identity.json', proof)
        write(out / 'gpu-mapping.json', mapping)
        from pyrualean.hosts import robodojo_native, xpolicylab_bridge
        from pyrualean.hosts import xpolicylab_robodojo as native_host
        from pyrualean.hosts.xpolicylab_robodojo import Cell
        import harnesslab.runner as runner
        from native_startup import NativeStartup
        startup = NativeStartup(out, runtime, proof, WORKSPACE,
            PYRUA / 'runtime/robodojo-source/scripts/eval_policy.sh',
            slots=CONTROLS['native_boot_concurrency'], timeout_s=CONTROLS['native_boot_timeout_s'])
        startup.prepare_paths()
        startup.install(native_host, xpolicylab_bridge.Bridge)
        append_lock = threading.Lock()

        def append(path, packet):
            with append_lock:
                with Path(path).open('a') as handle:
                    handle.write(json.dumps(packet, ensure_ascii=False, default=str) + '\n')

        original_record = robodojo_native.Steps.record
        def observe_record(self, obs, status):
            step = original_record(self, obs, status)
            write(self.root / f'{step:04d}/state.json', self.records[step])
            write(self.root / f'{step:04d}/native_status.json', status)
            return step
        robodojo_native.Steps.record = observe_record
        original_step = xpolicylab_bridge.Bridge.step
        def observe_step(self, action, *a, **kw):
            startup.note_action()
            append(out / 'native_actions.jsonl', {'event': 'request', 'at': now(), 'action': action})
            value = original_step(self, action, *a, **kw)
            append(out / 'native_actions.jsonl', {'event': 'observed', 'at': now(), 'status': self.status()})
            return value
        xpolicylab_bridge.Bridge.step = observe_step
        original_codex = runner.run_codex
        def observe_codex(*a, **kw):
            startup.note_model()
            return original_codex(*a, **kw)
        runner.run_codex = observe_codex
        if 'strong_grasp' in sys.modules:
            raise RuntimeError('Original-PyRUA-only experiment must not import the custom grasp package')
        # The private file is an explicit shared authentication dependency. Never
        # serialize its contents or include its values in exceptions/receipts.
        config_path = Path(os.environ['HARNESS_PLANNER_CONFIG'])
        config = json.loads(config_path.read_text())
        if config.get('model') != CONTROLS['model']:
            raise RuntimeError('Private provider configuration model does not match frozen controls')
        os.environ['HARNESS_EPISODE_API_KEY'] = config['api_key']
        codex_home = out / 'codex-home'
        codex_home.mkdir(mode=0o700)
        (codex_home / 'config.toml').write_text('approval_policy = "never"\nsandbox_mode = "workspace-write"\n')
        write(out / 'experiment.json', {'controls': CONTROLS, 'arm': args.arm,
            'task_conditions': args.arm == 'conditions', 'phase': args.phase, 'layout': args.layout,
            'gpu': args.gpu, 'gpu_uuid': selected['uuid'], 'pod': proof,
            'layout_sha256': sha(ASSETS / f'{TASK}_{args.layout}.json'),
            'source_freeze_sha256': sha(args.freeze), 'source_freeze': str(args.freeze),
            'runtime_directory': str(runtime), 'started_at': now(),
            'no_custom_grasp_import': True})
        state('booting')
        def terminate(signum, frame):
            raise KeyboardInterrupt('Experiment process received signal ' + str(signum))
        signal.signal(signal.SIGTERM, terminate)
        result = runner.run_episode(Cell(suite='arx_x5', task=TASK, seed=args.layout, cuda_device=args.gpu), out,
            arm='cells', model=CONTROLS['model'], reasoning_effort=CONTROLS['reasoning_effort'],
            budget_s=CONTROLS['budget_s'], max_turns=CONTROLS['max_turns'],
            codex_bin=str(CODEX), codex_home=str(codex_home), base_url=config['base_url'],
            api_key_env='HARNESS_EPISODE_API_KEY', robot=CONTROLS['robot'],
            runtime='codex', primitives='no-vla', images='on-demand', feedback='pure',
            max_attempts=1, tool_set='pyrua', task_conditions=args.arm == 'conditions',
            boot_options={'source_root': str(PYRUA / 'runtime/robodojo-source'),
                          'xpolicylab_root': '/opt/robodojo/XPolicyLab', 'eval_seed': 0})
        if 'strong_grasp' in sys.modules:
            raise RuntimeError('Custom grasp package unexpectedly imported during the experiment')
        state('finished', result_status=result.get('status'), success=result.get('environment_success'),
              env_steps=result.get('env_steps'), task_wall_s=result.get('task_wall_s'), error=result.get('error'))
        return 0 if result.get('status') == 'completed' else 2
    except BaseException as exc:
        state('launcher_error', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if startup is not None:
            startup.finalize(result)
        os.environ.pop('HARNESS_EPISODE_API_KEY', None)
        for handle in leases:
            handle.close()
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
