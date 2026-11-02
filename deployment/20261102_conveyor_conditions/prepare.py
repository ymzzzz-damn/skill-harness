"""Prepare immutable inputs and queue manifests. This command never submits a Job."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import subprocess
import urllib.parse
from protocol import (ALLOWED_GPU_PRODUCTS, ASSETS, BASELINE_COMMIT, CODEX, CODEX_SHA256, CONTROLS,
                      DEPLOYMENT, GPU_COUNT, IMAGE, NAMESPACE,
                      OPENCV_SHA256, OPENCV_WHEEL, OUTPUT, PIP_ARCHIVE, PROJECT, PYRUA, PYRUA_COMMIT, PYTHON,
                      RUNTIME, SAM3_COMMIT, SOURCE, now, sha, source_files, stream_sha, verify_freeze, write)

NAME = 'yumuzhen-conveyor-conditions-20261102'


def resources():
    return {'requests': {'cpu': '24', 'memory': '128Gi', 'ephemeral-storage': '64Gi',
                         'nvidia.com/gpu': '8', 'cce.baidubce.com/ip': '1'},
            'limits': {'cpu': '64', 'memory': '384Gi', 'ephemeral-storage': '128Gi',
                       'nvidia.com/gpu': '8', 'cce.baidubce.com/ip': '1'}}


def manifests():
    labels = {'app': NAME, 'owner': 'yumuzhen', 'experiment': 'conveyor-conditions-20261102'}
    account = {'apiVersion': 'v1', 'kind': 'ServiceAccount',
               'metadata': {'name': NAME, 'namespace': NAMESPACE, 'labels': labels}}
    role = {'apiVersion': 'rbac.authorization.k8s.io/v1', 'kind': 'Role',
            'metadata': {'name': NAME, 'namespace': NAMESPACE, 'labels': labels},
            'rules': [{'apiGroups': [''], 'resources': ['pods'], 'verbs': ['get']}]}
    binding = {'apiVersion': 'rbac.authorization.k8s.io/v1', 'kind': 'RoleBinding',
               'metadata': {'name': NAME, 'namespace': NAMESPACE, 'labels': labels},
               'subjects': [{'kind': 'ServiceAccount', 'name': NAME, 'namespace': NAMESPACE}],
               'roleRef': {'apiGroup': 'rbac.authorization.k8s.io', 'kind': 'Role', 'name': NAME}}
    env = [{'name': name, 'valueFrom': {'fieldRef': {'fieldPath': field}}}
           for name, field in (('HARNESS_POD_NAME', 'metadata.name'),
                               ('HARNESS_POD_UID', 'metadata.uid'),
                               ('HARNESS_POD_NAMESPACE', 'metadata.namespace'),
                               ('HARNESS_NODE_NAME', 'spec.nodeName'))]
    env += [{'name': name, 'value': value} for name, value in (
        ('HARNESS_ALLOWED_GPU_PRODUCTS', ','.join(ALLOWED_GPU_PRODUCTS)), ('NVIDIA_DRIVER_CAPABILITIES', 'all'),
        ('CUDA_DEVICE_ORDER', 'PCI_BUS_ID'), ('PYTHONDONTWRITEBYTECODE', '1'),
        ('PYTHONNOUSERSITE', '1'),
        ('HARNESS_PLANNER_CONFIG', str(PYRUA / '.secrets/planner.json')))]
    mount = lambda name, path, **kw: {'name': name, 'mountPath': str(path), **kw}
    job = {'apiVersion': 'batch/v1', 'kind': 'Job',
        'metadata': {'name': NAME, 'namespace': NAMESPACE, 'labels': labels},
        'spec': {'backoffLimit': 0, 'template': {'metadata': {'labels': labels}, 'spec': {
            'restartPolicy': 'Never', 'schedulerName': 'default-scheduler',
            'preemptionPolicy': 'Never',
            'serviceAccountName': NAME, 'automountServiceAccountToken': True,
            'imagePullSecrets': [{'name': 'registry-secret'}],
            'enableServiceLinks': False,
            'affinity': {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
                'nodeSelectorTerms': [{'matchExpressions': [{'key': 'nvidia.com/gpu.product',
                    'operator': 'In', 'values': list(ALLOWED_GPU_PRODUCTS)}]}]}}},
            'dnsPolicy': 'None', 'dnsConfig': {
                'nameservers': ['169.254.20.10', '172.16.0.10'],
                'searches': [NAMESPACE + '.svc.cluster.local', 'svc.cluster.local', 'cluster.local'],
                'options': [{'name': 'ndots', 'value': '5'}, {'name': 'attempts', 'value': '2'},
                            {'name': 'timeout', 'value': '1'}]},
            'containers': [{'name': 'runner', 'image': IMAGE, 'imagePullPolicy': 'IfNotPresent',
                'command': ['/opt/conda/envs/RoboDojo/bin/python', '-B', '-u',
                            str(DEPLOYMENT / 'bootstrap.py'),
                            '--freeze', str(DEPLOYMENT / 'freeze.json')],
                'securityContext': {'allowPrivilegeEscalation': False},
                'resources': resources(), 'env': env,
                'volumeMounts': [mount('shared', '/mnt/auwomo-data', readOnly=True),
                    mount('workspace', PROJECT), mount('shm', '/dev/shm'),
                    mount('extension-cache', '/root/.local/share/ov/data/exts/v2/' + PIP_ARCHIVE.name,
                          readOnly=True),
                    mount('legacy-robot-assets', '/personal/Web/RoboDojo/Assets', readOnly=True)]}],
            'volumes': [{'name': 'shared', 'hostPath': {'path': '/mnt/auwomo-data', 'type': 'Directory'}},
                {'name': 'workspace', 'hostPath': {'path': str(PROJECT), 'type': 'Directory'}},
                {'name': 'shm', 'emptyDir': {'medium': 'Memory', 'sizeLimit': '32Gi'}},
                {'name': 'extension-cache', 'hostPath': {'path': str(PIP_ARCHIVE), 'type': 'Directory'}},
                {'name': 'legacy-robot-assets', 'hostPath': {
                    'path': '/mnt/auwomo-data/auwomo-datasets/raw-data/manipulation/RoboDojo/Assets',
                    'type': 'Directory'}}]}}}}
    return [account, role, binding, job]


def validate_job(job):
    spec = job['spec']['template']['spec']
    assert 'nodeSelector' not in spec
    assert 'nodeName' not in spec and not spec.get('tolerations')
    assert spec['schedulerName'] == 'default-scheduler'
    assert spec['preemptionPolicy'] == 'Never'
    assert spec['containers'][0]['resources'] == resources()
    assert spec['containers'][0]['image'] == IMAGE
    assert any(e.get('name') == 'HARNESS_ALLOWED_GPU_PRODUCTS' and e.get('value') == ','.join(ALLOWED_GPU_PRODUCTS)
               for e in spec['containers'][0]['env'])
    assert spec['affinity']['nodeAffinity']['requiredDuringSchedulingIgnoredDuringExecution'] == {
        'nodeSelectorTerms': [{'matchExpressions': [{'key': 'nvidia.com/gpu.product',
            'operator': 'In', 'values': list(ALLOWED_GPU_PRODUCTS)}]}]}
    assert job['spec']['backoffLimit'] == 0 and spec['restartPolicy'] == 'Never'
    return True


def sam3_registration(pyr=PYRUA):
    """Read editable-install metadata without executing its Python or .pth code."""
    pyr = Path(pyr)
    expected = pyr / 'runtime/vendor' / ('sam3-' + SAM3_COMMIT)
    candidates = sorted((pyr / 'runtime/policy-venv/lib').glob(
        'python*/site-packages/sam3-0.1.0.dist-info/direct_url.json'))
    if len(candidates) != 1:
        raise RuntimeError('Expected exactly one registered SAM3 editable install')
    metadata = candidates[0]
    direct = json.loads(metadata.read_text())
    url = urllib.parse.urlparse(direct.get('url', ''))
    root = Path(urllib.parse.unquote(url.path))
    if (url.scheme != 'file' or url.netloc or direct.get('dir_info', {}).get('editable') is not True
            or root != expected or not (root / 'sam3/__init__.py').is_file()):
        raise RuntimeError('SAM3 editable registration differs from the reviewed pinned source')
    site = metadata.parent.parent
    pth = site / '__editable__.sam3-0.1.0.pth'
    finder = site / '__editable___sam3_0_1_0_finder.py'
    if pth.read_text().strip() != 'import __editable___sam3_0_1_0_finder; __editable___sam3_0_1_0_finder.install()':
        raise RuntimeError('Unexpected SAM3 editable .pth loader')
    mapping = None
    for node in ast.parse(finder.read_text()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'MAPPING':
            mapping = ast.literal_eval(node.value)
    if not isinstance(mapping, dict) or mapping.get('sam3') != str(root / 'sam3'):
        raise RuntimeError('SAM3 editable finder mapping differs from registered source')
    return {'root': root, 'package': root / 'sam3', 'metadata_files': [metadata, pth, finder]}


def freeze():
    commit = subprocess.check_output(['git', '-C', str(PYRUA / 'source'), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(PYRUA / 'source'), 'status', '--porcelain'], text=True)
    if commit != PYRUA_COMMIT or dirty:
        raise RuntimeError('Original PyRUA dependency must be the unchanged pinned commit')
    task = PYRUA / 'runtime/robodojo-source/task/RoboDojo/tasks' / (CONTROLS['task'] + '.py')
    if 'self.step_lim = 700' not in task.read_text():
        raise RuntimeError('Unexpected native task step budget')
    if sha(CODEX) != CODEX_SHA256:
        raise RuntimeError('Codex executable drift')
    if sha(OPENCV_WHEEL) != OPENCV_SHA256:
        raise RuntimeError('Reviewed local OpenCV compatibility wheel changed')
    sam3 = sam3_registration()
    files = [task, CODEX, OPENCV_WHEEL, PYRUA / 'source/LICENSE',
             PYRUA / 'runtime/robodojo-source/scripts/eval_policy.sh',
             PYRUA / 'runtime/robodojo-source/env/reward_manager/reward_manager.py',
             PYRUA / 'runtime/robodojo-source/env/reward_manager/func_parser.py',
             PIP_ARCHIVE / 'config/extension.toml']
    files += sam3['metadata_files']
    files += [ASSETS / f'{CONTROLS["task"]}_{layout}.json' for layout in range(55)]
    checkpoint = PYRUA / 'runtime/checkpoints/sam3.pt'
    if checkpoint.stat().st_size != 3450062241:
        raise RuntimeError('SAM3 checkpoint size no longer matches reviewed runtime')
    roots = {'harness': SOURCE, 'pyrua': PYRUA / 'source', 'deployment': DEPLOYMENT,
             'sam3': sam3['root']}
    for name, root in roots.items():
        if not root.is_dir() or not source_files(root):
            raise RuntimeError('Required source/dependency tree is missing or empty: ' + name + ' at ' + str(root))
    checkpoint_hash = stream_sha(checkpoint)
    if checkpoint_hash != '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e':
        raise RuntimeError('SAM3 checkpoint integrity check failed')
    return {'schema': 'conveyor-condition-experiment/v1', 'prepared_at': now(),
        'controls': CONTROLS, 'allowed_gpu_products': list(ALLOWED_GPU_PRODUCTS),
        'sources': {name: {'root': str(root), 'files': source_files(root)} for name, root in roots.items()},
        'files': [{'path': str(path), 'sha256': sha(path)} for path in files],
        'checkpoint': {'path': str(checkpoint), 'size': checkpoint.stat().st_size,
                       'mtime_ns': checkpoint.stat().st_mtime_ns, 'sha256': checkpoint_hash},
        'dependencies': {'runtime_python': str(PYTHON), 'sam3_checkpoint': str(PYRUA / 'runtime/checkpoints/sam3.pt'),
            'sam3_source': str(sam3['root']), 'sam3_commit': SAM3_COMMIT,
            'sam3_registration_files': list(map(str, sam3['metadata_files'])),
            'sam3_checkpoint_sha256': '9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e',
            'native_source': str(PYRUA / 'runtime/robodojo-source'),
            'xpolicylab_root': '/opt/robodojo/XPolicyLab', 'pyrua_license': 'Apache-2.0',
            'credential_source': 'HARNESS_PLANNER_CONFIG environment path, runtime read only; values never recorded'},
        'comparison': {'control': 'baseline harness behavior, task_conditions=False, tool_set=pyrua',
            'conditions': 'same harness source, task_conditions=True, tool_set=pyrua',
            'baseline_snapshot_commit': BASELINE_COMMIT,
            'historical_pyrua_comparison': '11/55 is historical context, not the paired control arm',
            'all_outputs': str(OUTPUT), 'runtime': str(RUNTIME)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Validate/print the plan without reading remote dependencies or writing')
    parser.add_argument('--output-dir', type=Path, default=DEPLOYMENT)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    docs = manifests()
    validate_job(docs[-1])
    if args.verify:
        verify_freeze(json.loads(args.verify.read_text()))
        print(json.dumps({'verified': True, 'model_calls': 0, 'native_actions': 0}))
        return 0
    if args.dry_run:
        print(json.dumps({'controls': CONTROLS, 'allowed_gpu_products': ALLOWED_GPU_PRODUCTS,
            'allocated_gpus': GPU_COUNT, 'full_episodes': 110, 'smoke_layouts': [3, 21],
            'job_manifest_valid': True, 'submitted': False, 'model_calls': 0, 'native_actions': 0}, indent=2))
        return 0
    if any((args.output_dir / name).exists() for name in ('freeze.json', 'rbac-and-job.json')):
        raise FileExistsError('Prepared immutable deployment already exists; do not overwrite it')
    frozen = freeze()
    write(args.output_dir / 'freeze.json', frozen)
    write(args.output_dir / 'rbac-and-job.json', {'apiVersion': 'v1', 'kind': 'List', 'items': docs})
    print(json.dumps({'prepared': str(args.output_dir), 'submitted': False, 'model_calls': 0, 'native_actions': 0}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
