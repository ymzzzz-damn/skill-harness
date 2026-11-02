"""Frozen controls and filesystem helpers; importing this module starts no work."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

WORKSPACE = Path('/mnt/auwomo-data/yumuzhen')
PROJECT = WORKSPACE / 'harness'
PYRUA = WORKSPACE / 'PyRUA-Lean'
SOURCE = PROJECT / 'source'
DEPLOYMENT = PROJECT / 'deployment/20261102_conveyor_conditions'
OUTPUT = PROJECT / 'experiments/20261102_conveyor_conditions'
RUNTIME = PROJECT / 'runtime/20261102_conveyor_conditions'
TASK = 'pick_from_conveyor_by_image'
ALLOWED_GPU_PRODUCTS = ('NVIDIA-RPBZZZ6', 'NVIDIA-GeForce-RTX-5090')
NAMESPACE = 'dreamzero-training'
PYRUA_COMMIT = 'ffad6d439f926eea450e9e3d3b3a106438b141b1'
SAM3_COMMIT = 'c1f6b1e83967c912f675d8975d437ccdc41e17d8'
SAM3_SOURCE = PYRUA / 'runtime/vendor' / ('sam3-' + SAM3_COMMIT)
BASELINE_COMMIT = 'ec2b94afb0161e6aab91d213221d33f5797fee86'
IMAGE = ('registry.d-robotics.cc/auwomo-auwomo/robodojo-tswam-full:20260901-f2'
         '@sha256:834a60f7a9e36254b5ac8fc1ec4689b19d9e4eb413836e65812e4047940499a3')
PIP_ARCHIVE = Path('/mnt/auwomo-data/RoboDojo/qwenpaw-harness/isaac-cache/exts/v2/'
                   'omni.kit.pip_archive-d38fa9ecd1fb6df4')
ASSETS = PYRUA / 'runtime/robodojo-source/Assets/Eval_Layout/RoboDojo/arx_x5/0'
PYTHON = PYRUA / 'runtime/policy-venv/bin/python'
CODEX = PYRUA / 'runtime/bin/codex-0.159.2'
CODEX_SHA256 = '1748767b230ebfc3d4ab7e4e254920d0c0ad9691fd8c11f190e7d44511a4a92e'
OPENCV_WHEEL = WORKSPACE / ('exp/grasp_tool_20261029/runtime/wheels/'
    'opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')
OPENCV_SHA256 = '377d08a7e48a1405b5e84afcbe4798464ce7ee17081c1c23619c8b398ff18295'
ARMS = ('control', 'conditions')
ORIGINAL_NATIVE_HANDLERS = ['back_project', 'grip', 'home', 'locate', 'move_pose', 'segment']
GPU_COUNT = 8
SMOKE_LAYOUTS = (3, 21)
CONTROLS = {
    'task': TASK, 'eval_seed': 0, 'layouts': list(range(55)),
    'model': 'gpt-6-astra', 'reasoning_effort': 'high', 'max_turns': 100,
    'budget_s': 5400, 'native_step_limit': 700, 'runtime': 'codex',
    'robot': 'robodojo_xpl_rgb', 'primitives': 'no-vla',
    'images': 'on-demand', 'feedback': 'pure', 'tool_set': 'pyrua',
    'pyrua_commit': PYRUA_COMMIT, 'planner_attempt_limit': 1,
    'baseline_snapshot_commit': BASELINE_COMMIT,
    'native_boot_concurrency': 2, 'native_boot_timeout_s': 900,
    'fresh_agent_per_scene': True, 'training': False,
    'cross_scene_memory': False, 'measured_episode_replay': False,
    'stage_order': ['conditions', 'control'],
    'stage_order_reason': 'User requested treatment first, then baseline/ablation',
    'scheduling': 'shared P6000 / RTX5090 queue, eight-GPU allocation',
    'disabled_agent_features': ['shell_snapshot', 'memories', 'shell_tool'],
}


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stream_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    temporary.replace(path)


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def pair_plan(layout, gpu):
    if type(layout) is not int or not 0 <= layout < 55:
        raise ValueError('Layout must be 0..54')
    if type(gpu) is not int or not 0 <= gpu < GPU_COUNT:
        raise ValueError('GPU lane must be 0..7')
    return [{'layout': layout, 'arm': arm, 'gpu': gpu} for arm in CONTROLS['stage_order']]


def full_plan(gpu, arm):
    if type(gpu) is not int or not 0 <= gpu < GPU_COUNT or arm not in ARMS:
        raise ValueError('Invalid stage/lane')
    return [{'layout': layout, 'arm': arm, 'gpu': gpu} for layout in range(gpu, 55, GPU_COUNT)]


def episode_dir(phase, arm, layout, output=OUTPUT):
    if phase not in ('smoke', 'full') or arm not in ARMS or not 0 <= layout < 55:
        raise ValueError('Invalid experiment identity')
    return Path(output) / phase / arm / f'layout_{layout:02d}' / 'episode_001'


def source_files(root):
    root = Path(root)
    omitted = {'.git', '__pycache__', '.pytest_cache', 'build', 'dist', '.secrets'}
    extensions = {'.py', '.md', '.txt', '.toml'}
    return {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob('*'))
            if path.is_file() and not omitted.intersection(path.relative_to(root).parts)
            and (path.suffix in extensions or path.name == 'ORIGINS.json')}


def verify_freeze(frozen):
    if frozen.get('controls') != CONTROLS or frozen.get('allowed_gpu_products') != list(ALLOWED_GPU_PRODUCTS):
        raise RuntimeError('Experiment controls changed after preparation')
    for name, record in frozen['sources'].items():
        actual = source_files(record['root'])
        if actual != record['files']:
            raise RuntimeError('Frozen source/dependency changed: ' + name)
    for entry in frozen['files']:
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('Frozen runtime/native file changed: ' + entry['path'])
    checkpoint = frozen.get('checkpoint', {})
    if checkpoint:
        stat = Path(checkpoint['path']).stat()
        if stat.st_size != checkpoint['size'] or stat.st_mtime_ns != checkpoint['mtime_ns']:
            raise RuntimeError('Reviewed SAM3 checkpoint changed after preparation')
