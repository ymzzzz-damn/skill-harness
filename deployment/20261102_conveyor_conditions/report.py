"""Read native evidence, paired uncertainty and time accounting without simulation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from protocol import (ARMS, CONTROLS, ORIGINAL_NATIVE_HANDLERS, OUTPUT, TASK,
                      episode_dir, load, now, sha, write)


def stats(values):
    values = sorted(v for v in values if type(v) in (int, float) and math.isfinite(v) and v >= 0)
    if not values:
        return {'n': 0, 'sum': None, 'mean': None, 'median': None}
    return {'n': len(values), 'sum': sum(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def native_outcome(out, layout):
    paths = sorted((Path(out) / 'bench/eval_result').rglob('_result.json'))
    proof = {'valid': False, 'success': None, 'errors': [], 'path': str(paths[0]) if len(paths) == 1 else None}
    if len(paths) != 1:
        proof['errors'].append('expected exactly one native result')
        return proof
    native = load(paths[0], {})
    if not isinstance(native, dict):
        proof['errors'].append('native result is not an object')
        return proof
    details = native.get('details', {})
    trial = details.get('0') if isinstance(details, dict) and set(details) == {'0'} else None
    if not isinstance(trial, dict):
        proof['errors'].append('native details must contain exactly trial0')
        return proof
    if TASK not in paths[0].parts or type(native.get('eval_time')) is not int or native['eval_time'] != 1:
        proof['errors'].append('native task identity or trial count mismatch')
    if type(trial.get('layout_id')) is not int or trial['layout_id'] != layout:
        proof['errors'].append('native layout identity mismatch')
    if type(trial.get('success')) is not bool:
        proof['errors'].append('native verdict must be boolean')
    def number(value):
        return type(value) in (int, float) and math.isfinite(value)
    score, detail, rate = native.get('score'), trial.get('score'), native.get('success_rate')
    if not number(score) or not 0 <= score <= 100 or not number(detail) or not 0 <= detail <= 1:
        proof['errors'].append('invalid native score scale')
    elif not math.isclose(score, detail * 100, abs_tol=1e-5):
        proof['errors'].append('native scores disagree')
    if not number(rate) or rate != float(trial.get('success') is True):
        proof['errors'].append('native aggregate and trial verdict disagree')
    proof.update(valid=not proof['errors'], success=trial.get('success'), score=score,
                 sha256=sha(paths[0]))
    return proof


def model_accounting(out):
    ids, sessions, totals = set(), 0, []
    def packets(path):
        with path.open(errors='replace') as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    for path in sorted((Path(out) / 'codex-home/sessions').rglob('*.jsonl')):
        sessions += 1
        last_usage = None
        for packet in packets(path):
            def walk(value):
                nonlocal last_usage
                if isinstance(value, dict):
                    response = value.get('response_id')
                    if isinstance(response, str) and response.startswith('resp_'):
                        ids.add(response)
                    usage = value.get('total_token_usage')
                    if isinstance(usage, dict):
                        last_usage = {key: item for key, item in usage.items() if type(item) is int}
                    for item in value.values():
                        walk(item)
                elif isinstance(value, list):
                    for item in value:
                        walk(item)
            walk(packet)
        if last_usage is not None:
            totals.append(last_usage)
    keys = {key for item in totals for key in item}
    return {'model_responses_observed': len(ids) if ids else None,
            'response_metric': 'unique response_id from Codex sessions; absence is unavailable, not zero',
            'session_files': sessions,
            'token_usage_observed': {key: sum(item.get(key, 0) for item in totals) for key in keys} if totals else None}


def implementation_proof(out, result, manifest, arm, output):
    """Compare actual runner receipts to prepared pristine dependency hashes."""
    errors = []
    scalar = {'tool_set': 'pyrua', 'task_conditions': arm == 'conditions',
        'model': CONTROLS['model'], 'reasoning_effort': 'high', 'max_turns': 100,
        'timeout_s': 5400, 'max_episode_steps': 700, 'images': 'on-demand', 'feedback': 'pure',
        'robot': 'robodojo_xpl_rgb', 'runtime': 'codex', 'arm': 'cells', 'suite': 'arx_x5'}
    for key, expected in scalar.items():
        if type(result.get(key)) is not type(expected) or result.get(key) != expected:
            errors.append('actual runner control mismatch: ' + key)
    disabled = CONTROLS['disabled_agent_features']
    if (result.get('agent_features', {}).get('disabled') != disabled
            or (result.get('generation') or {}).get('disabled_features') != disabled):
        errors.append('actual Agent process lacks the shared disabled-shell/memory feature policy')
    host = result.get('host', {})
    options = host.get('boot_options', {})
    if (host.get('primitive_set') != 'no-vla' or type(options.get('eval_seed')) is not int
            or options.get('eval_seed') != 0 or options.get('xpolicylab_root') != '/opt/robodojo/XPolicyLab'):
        errors.append('actual native boot/no-vla/eval-seed controls mismatch')
    tools = result.get('robot_tools', {})
    if (tools.get('native_handlers') != ORIGINAL_NATIVE_HANDLERS
            or tools.get('grasp_installed') is not False or tools.get('global_grasp_supplied') is not False):
        errors.append('actual robot handler surface is not original PyRUA-only')
    frozen_path = Path(output) / 'source-freeze.json'
    if not frozen_path.is_file() and manifest.get('source_freeze'):
        frozen_path = Path(manifest['source_freeze'])
    frozen = load(frozen_path, {})
    provenance = {}
    if (not frozen_path.is_file() or sha(frozen_path) != manifest.get('source_freeze_sha256')
            or frozen.get('controls') != CONTROLS):
        errors.append('pristine immutable source manifest unavailable or hash mismatch')
    else:
        for package, root_name, prefix in (('pyrualean','pyrua','src/pyrualean/'),
                                          ('harnesslab','harness','harnesslab/')):
            record = frozen.get('sources', {}).get(root_name, {})
            expected = {name[len(prefix):]: digest for name, digest in record.get('files', {}).items()
                        if name.startswith(prefix) and name.endswith('.py')}
            actual = result.get('dependencies', {}).get(package, {})
            expected_path = str(Path(record.get('root','/missing')) / prefix.rstrip('/'))
            if not expected or actual.get('sha256') != expected or actual.get('path') != expected_path:
                errors.append('actual dependency origin/hash differs from pristine source: ' + package)
            provenance[package] = actual.get('sha256')
        expected_native = str(Path(frozen['sources']['pyrua']['root']).parent / 'runtime/robodojo-source')
        if options.get('source_root') != expected_native:
            errors.append('actual native source-root mismatch')
    provenance.update(native_handlers=tools.get('native_handlers'), grasp_installed=tools.get('grasp_installed'),
                      global_grasp_supplied=tools.get('global_grasp_supplied'))
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest() if not errors else None
    timing = result.get('task_timing', {})
    duration = result.get('task_wall_s')
    timing_valid = (type(duration) in (int,float) and math.isfinite(duration) and duration >= 0
                    and timing.get('end_observed') is True)
    if not timing_valid:
        errors.append('complete finite task timing with observed native/conclusion end required')
    contract = load(out / 'task_contract.json', {})
    events = load(out / 'task_condition_events.json', [])
    names = [e.get('event') for e in events if isinstance(e, dict)] if isinstance(events,list) else []
    registered = (contract.get('schema') == 'harness-task-contract-v1' and contract.get('registered') is True
                  and isinstance(contract.get('plan'),dict) and bool(contract['plan'].get('conditions'))
                  and 'plan_registered' in names)
    actions = result.get('env_steps_total', result.get('env_steps',0))
    acted = type(actions) in (int,float) and actions > 0
    guarded = not acted or all(name in names for name in ('action_authorized','physical_dispatch','physical_return'))
    if arm == 'conditions' and acted and (not registered or not guarded):
        errors.append('physical actions lack registered condition plan or native guard events')
    if arm == 'control' and (contract or events):
        errors.append('conditions-disabled control unexpectedly produced a task contract')
    return {'errors': errors, 'tool_provenance_sha256': digest,
        'task_timing_valid': timing_valid, 'contract_registered': registered,
        'condition_guard_evidence_valid': guarded, 'condition_event_counts': {name:names.count(name) for name in set(names)}}


def inspect_episode(phase, arm, layout, output=OUTPUT):
    out = episode_dir(phase, arm, layout, output)
    result = load(out / 'result.json', {})
    manifest = load(out / 'experiment.json', {})
    launch = load(out.parent / 'launch_status.json', {})
    native = native_outcome(out, layout)
    errors = []
    if manifest:
        if (manifest.get('controls') != CONTROLS or manifest.get('arm') != arm
                or manifest.get('layout') != layout or manifest.get('phase') != phase
                or manifest.get('task_conditions') is not (arm == 'conditions')):
            errors.append('episode control/identity manifest mismatch')
        if manifest.get('gpu') != layout % 8 or not str(manifest.get('gpu_uuid', '')).startswith('GPU-'):
            errors.append('layout to actual GPU mapping unavailable')
    elif result:
        errors.append('missing experiment identity manifest')
    if result and (result.get('task') != TASK or result.get('seed') != layout):
        errors.append('framework result task/layout mismatch')
    if native['valid'] and result and result.get('environment_success') is not native['success']:
        errors.append('framework and native verdict disagree')
    if 'strong_grasp' in result.get('dependencies', {}) or (out / 'strong_grasp').exists():
        errors.append('custom grasp dependency/artifact present')
    implementation = implementation_proof(out,result,manifest,arm,output) if result else {
        'errors': [], 'tool_provenance_sha256': None, 'task_timing_valid': False,
        'contract_registered': False, 'condition_guard_evidence_valid': False, 'condition_event_counts': {}}
    errors.extend(implementation['errors'])
    runtime_valid = (result.get('status') in ('completed', 'call_budget', 'timeout')
                     and not result.get('error') and not launch.get('error') and not errors)
    histories = load(out.parent / 'boot-attempts.json', [])
    boot_ok = isinstance(histories, list)
    prior_boot_s = 0
    if boot_ok:
        from boot_recovery import boot_retry_allowed
        for index, history in enumerate(histories, 1):
            archive = out.parent / 'boot_attempts' / f'attempt_{index:03d}'
            archived_result = archive / 'result.json'
            boot_ok = boot_ok and boot_retry_allowed(archive) and archived_result.is_file()
            if archived_result.is_file() and sha(archived_result) != history.get('result_sha256'):
                boot_ok = False
            prior_boot_s += history.get('worker_wall_s', 0)
    if not boot_ok:
        errors.append('invalid initialization retry history')
    attempted = bool(result or manifest or launch or out.exists())
    closed = bool((result and result.get('status') not in ('running',None))
                  or launch.get('status') in ('launcher_error','infrastructure_blocked','finished'))
    record = {'phase': phase, 'arm': arm, 'layout': layout, 'attempted': attempted,
        'closed': closed,
        'status': result.get('status', launch.get('status', 'pending')), 'error': result.get('error') or launch.get('error'),
        'native_record_valid': native['valid'], 'native': native,
        'valid': bool(native['valid'] and runtime_valid and boot_ok),
        'success': native['success'] if native['valid'] else None,
        'identity_errors': errors, 'gpu_lane': manifest.get('gpu'), 'gpu_uuid': manifest.get('gpu_uuid'),
        'pod_uid': manifest.get('pod', {}).get('uid'), 'node': manifest.get('pod', {}).get('node'),
        'layout_sha256': manifest.get('layout_sha256'), 'freeze_sha256': manifest.get('source_freeze_sha256'),
        'task_wall_s': result.get('task_wall_s'), 'startup_s': result.get('startup_s'),
        'policy_wall_s': result.get('policy_wall_s'), 'tool_wall_s': result.get('tool_wall_s'),
        'non_tool_agent_s': result.get('non_tool_agent_s'), 'total_wall_s': result.get('total_wall_s'),
        'prior_initialization_wall_s': prior_boot_s,
        'total_wall_s_including_initialization_retries': result.get('total_wall_s', 0) + prior_boot_s if result else None,
        'native_actions': result.get('env_steps_total', result.get('env_steps')),
        'python_decisions': len(list((out / 'workspace').glob('cell_*.py'))),
        'finish': result.get('finish'), 'episode_directory': str(out),
        'result_sha256': sha(out / 'result.json') if (out / 'result.json').is_file() else None}
    record.update({key:value for key,value in implementation.items() if key != 'errors'})
    record.update(model_accounting(out))
    return record


def wilson(successes, n):
    if not n:
        return None
    z, p = 1.959963984540054, successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [center - radius, center + radius]


def arm_summary(records):
    attempted = [r for r in records if r['attempted']]
    valid = [r for r in records if r['valid']]
    native = [r for r in records if r['native_record_valid']]
    success = sum(r['success'] is True for r in native)
    fields = ('task_wall_s', 'startup_s', 'policy_wall_s', 'tool_wall_s', 'non_tool_agent_s',
              'total_wall_s', 'prior_initialization_wall_s', 'native_actions', 'python_decisions',
              'model_responses_observed')
    return {'expected': len(records), 'attempted': len(attempted), 'pending': len(records)-len(attempted),
        'closed': sum(r.get('closed',False) for r in records),
        'final': bool(records) and all(r.get('closed',False) for r in records),
        'native_outcomes': len(native), 'native_successes': success,
        'observed_success_fraction_of_all_expected': success / len(records) if records else None,
        'observed_success_fraction_of_all_attempted': success / len(attempted) if attempted else None,
        'missing_or_invalid_native_outcomes': len(attempted)-len(native),
        'valid': len(valid), 'valid_successes': sum(r['success'] is True for r in valid),
        'valid_success_rate': sum(r['success'] is True for r in valid)/len(valid) if valid else None,
        'valid_success_rate_wilson95': wilson(sum(r['success'] is True for r in valid), len(valid)),
        'execution_or_identity_errors': sum(not r['valid'] for r in attempted),
        'all_attempted_metrics': {key: stats(r.get(key) for r in attempted) for key in fields},
        'valid_only_metrics': {key: stats(r.get(key) for r in valid) for key in fields},
        'timing_note': 'task_wall_s runs from initial task card and robot tools ready to first observed native end/conclusion, including planning, LLM and controls. Scene/Tool initialization and teardown are excluded; first simulator RGB receipt is a separate initialization timestamp. Sums of parallel episode durations are not stage elapsed wall time.'}


def paired(records, replicates=10000):
    by = {(r['arm'], r['layout']): r for r in records}
    pairs, excluded = [], []
    for layout in range(55):
        control, treatment = by.get(('control', layout)), by.get(('conditions', layout))
        if not control or not treatment or not control['valid'] or not treatment['valid']:
            excluded.append(layout)
            continue
        keys = ('gpu_lane', 'gpu_uuid', 'pod_uid', 'node', 'layout_sha256', 'freeze_sha256', 'tool_provenance_sha256')
        if any(control.get(k) is None or control.get(k) != treatment.get(k) for k in keys):
            excluded.append(layout)
            continue
        pairs.append({'layout': layout, 'control': control['success'], 'conditions': treatment['success'],
            'success_difference': int(treatment['success'])-int(control['success']),
            'task_wall_difference_s': (treatment['task_wall_s']-control['task_wall_s']
                if type(treatment.get('task_wall_s')) in (int,float) and type(control.get('task_wall_s')) in (int,float) else None)})
    gained = sum(p['conditions'] and not p['control'] for p in pairs)
    lost = sum(p['control'] and not p['conditions'] for p in pairs)
    discordant = gained + lost
    p_value = min(1, 2*sum(math.comb(discordant, k) for k in range(min(gained,lost)+1)) / 2**discordant) if discordant else 1.0
    ci = None
    if pairs:
        rng, differences = random.Random(0), [p['success_difference'] for p in pairs]
        samples = sorted(sum(rng.choice(differences) for _ in differences)/len(differences) for _ in range(replicates))
        ci = [samples[int(.025*(replicates-1))], samples[int(.975*(replicates-1))]]
    return {'n': len(pairs), 'excluded_layouts': excluded, 'gained': gained, 'lost': lost,
        'success_rate_difference': (gained-lost)/len(pairs) if pairs else None,
        'paired_bootstrap95': ci, 'bootstrap_replicates': replicates, 'bootstrap_seed': 0,
        'exact_mcnemar_two_sided_p': p_value if pairs else None,
        'task_wall_difference_observations': sum(p['task_wall_difference_s'] is not None for p in pairs),
        'mean_task_wall_difference_s': statistics.mean(p['task_wall_difference_s'] for p in pairs
            if p['task_wall_difference_s'] is not None) if any(p['task_wall_difference_s'] is not None for p in pairs) else None,
        'median_task_wall_difference_s': statistics.median(p['task_wall_difference_s'] for p in pairs
            if p['task_wall_difference_s'] is not None) if any(p['task_wall_difference_s'] is not None for p in pairs) else None,
        'pairs': pairs,
        'interpretation': 'Fixed treatment-first stage order is user requested, not randomized. Same seed0 layouts and physical GPU mapping support this cohort comparison, not unseen-layout generalization.'}


def build_report(output=OUTPUT, phase='full'):
    layouts = range(55) if phase == 'full' else (3, 21)
    arms = ARMS if phase == 'full' else ('conditions',)
    rows = [inspect_episode(phase, arm, layout, output) for arm in arms for layout in layouts]
    return {'schema': 'conveyor-condition-report/v1', 'at': now(), 'phase': phase, 'controls': CONTROLS,
        'arms': {arm: arm_summary([r for r in rows if r['arm'] == arm]) for arm in arms},
        'paired': paired(rows) if phase == 'full' else None, 'episodes': rows,
        'stage_elapsed_wall': load(Path(output) / 'stage_times.json', {}),
        'historical_pyrua_reference': {'task': TASK, 'successes': 11, 'all_layouts': 55,
            'valid': 52, 'valid_successes': 11, 'paired_control': False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--phase', choices=('smoke', 'full'), default='full')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = build_report(args.output, args.phase)
    if args.write:
        write(args.output / (args.phase + '-report.json'), result)
    print(json.dumps({key: result[key] for key in ('phase', 'arms', 'paired', 'stage_elapsed_wall')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
