"""Archive only provably unstarted native boots; never replay a measured episode.

First-party r6 deployment adaptation; provenance is recorded in ORIGINS.json.
"""
from pathlib import Path
import hashlib
import json
import shutil

MAX_BOOT_ATTEMPTS = 3

def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default

def boot_retry_allowed(out):
    out = Path(out)
    result = load(out / 'result.json', {})
    receipt = load(out / 'initialization-receipt.json', {})
    if receipt.get('retry_eligible') is not True:
        return False
    if receipt.get('first_observation_seen') is not False or receipt.get('first_rgb_seen') is not False:
        return False
    if receipt.get('model_phase_entered') is not False or receipt.get('native_action_requests') != 0:
        return False
    if receipt.get('native_client_cleanup_confirmed') is not True:
        return False
    if result.get('status') != 'boot_error':
        return False
    if result.get('env_steps') != 0 or result.get('env_steps_total', 0) not in (None, 0):
        return False
    if result.get('tool_turns', 0) != 0 or result.get('generation'):
        return False
    # Receiving even the first observation commits this episode: no replay.
    if any((out / 'steps').glob('*')) or any((out / 'bench/eval_result').rglob('_result.json')):
        return False
    for name in ('calls.jsonl', 'native_actions.jsonl', 'codex_events.jsonl', 'tool_spans.jsonl'):
        path = out / name
        if path.exists() and path.stat().st_size:
            return False
    if load(out / 'tool_turns.json', []):
        return False
    if any((out / 'codex-home/sessions').rglob('*.jsonl')):
        return False
    return True

def archive_unstarted_boot(out, attempt, elapsed_s):
    out = Path(out)
    if type(attempt) is not int or not 1 <= attempt < MAX_BOOT_ATTEMPTS:
        raise ValueError('Only bounded initialization retries are allowed')
    if not boot_retry_allowed(out):
        raise RuntimeError('Refusing to replay an episode with observation/model/action/result evidence')
    destination = out.parent / 'boot_attempts' / f'attempt_{attempt:03d}'
    if destination.exists():
        raise RuntimeError('Existing initialization evidence cannot be overwritten')
    result_sha = hashlib.sha256((out / 'result.json').read_bytes()).hexdigest()
    result = load(out / 'result.json')
    history_path = out.parent / 'boot-attempts.json'
    history = load(history_path, [])
    if not isinstance(history, list) or len(history) != attempt - 1:
        raise RuntimeError('Initialization history is inconsistent')
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.rename(destination)
    launch_status = out.parent / 'launch_status.json'
    if launch_status.is_file():
        shutil.copy2(launch_status, destination / 'launch_status-before-retry.json')
    receipt = {'attempt': attempt, 'original_episode_directory': str(out),
        'archive_directory': str(destination), 'result_sha256': result_sha,
        'worker_wall_s': elapsed_s, 'error': result.get('error'),
        'first_observation_received': False, 'model_calls': 0, 'native_actions': 0,
        'reason': 'Unstarted native initialization failed; every byte retained before a fresh process retry.'}
    history.append(receipt)
    temporary = history_path.with_name(history_path.name + '.tmp')
    temporary.write_text(json.dumps(history, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(history_path)
    return receipt
