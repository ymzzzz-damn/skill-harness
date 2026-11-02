"""Read the running Pod from Kubernetes and verify its real allocation."""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import urllib.parse
import urllib.request
from protocol import ALLOWED_GPU_PRODUCTS, GPU_COUNT, IMAGE, NAMESPACE, now


def validate_pod(pod, env):
    meta, spec, status = pod['metadata'], pod['spec'], pod['status']
    for key, actual in (('HARNESS_POD_NAME', meta['name']), ('HARNESS_POD_UID', meta['uid']),
                        ('HARNESS_POD_NAMESPACE', meta['namespace']),
                        ('HARNESS_NODE_NAME', spec.get('nodeName'))):
        if not actual or env.get(key) != actual:
            raise RuntimeError('Actual Pod identity disagrees with Downward API: ' + key)
    if meta['namespace'] != NAMESPACE or status.get('phase') != 'Running' or meta.get('deletionTimestamp'):
        raise RuntimeError('Experiment Pod is not a live Running allocation')
    if env.get('HOSTNAME') != meta['name'] or env.get('HARNESS_ALLOWED_GPU_PRODUCTS') != ','.join(ALLOWED_GPU_PRODUCTS):
        raise RuntimeError('Container identity or allowed hardware policy mismatch')
    container = [c for c in spec['containers'] if c['name'] == 'runner']
    if len(container) != 1 or container[0]['image'] != IMAGE:
        raise RuntimeError('Unexpected experiment runner image')
    resources = container[0].get('resources', {})
    if any(str(resources.get(k, {}).get('nvidia.com/gpu')) != str(GPU_COUNT) for k in ('requests', 'limits')):
        raise RuntimeError('The actual Pod does not own an eight-GPU allocation')
    expressions = spec.get('affinity', {}).get('nodeAffinity', {}).get(
        'requiredDuringSchedulingIgnoredDuringExecution', {}).get('nodeSelectorTerms', [])
    if expressions != [{'matchExpressions': [{'key': 'nvidia.com/gpu.product',
            'operator': 'In', 'values': list(ALLOWED_GPU_PRODUCTS)}]}]:
        raise RuntimeError('Pod shared-queue hardware policy drift')
    if spec.get('schedulerName') != 'default-scheduler' or spec.get('nodeSelector') or spec.get('tolerations'):
        # Kubernetes may inject its ordinary NotReady/Unreachable NoExecute tolerations.
        tolerations = spec.get('tolerations', [])
        ordinary = all(t.get('key') in ('node.kubernetes.io/not-ready', 'node.kubernetes.io/unreachable')
                       and t.get('effect') == 'NoExecute' for t in tolerations)
        if spec.get('schedulerName') != 'default-scheduler' or spec.get('nodeSelector') or not ordinary:
            raise RuntimeError('Unexpected fixed-node/taint/priority scheduling override')
    if spec.get('priority', 0) != 0:
        raise RuntimeError('Experiment must keep ordinary queue priority')
    if spec.get('preemptionPolicy') != 'Never':
        raise RuntimeError('Experiment allocation must not preempt other workloads')
    return {'pod': meta['name'], 'uid': meta['uid'], 'namespace': meta['namespace'],
        'node': spec['nodeName'], 'phase': status['phase'], 'gpu_request': GPU_COUNT,
        'allowed_gpu_products': list(ALLOWED_GPU_PRODUCTS), 'verified_at': now(),
        'identity_source': 'own live Kubernetes Pod API, matched against Downward API',
        'image': container[0]['image'],
        'container_image_ids': [s.get('imageID') for s in status.get('containerStatuses', [])]}


def own_pod():
    base = Path('/var/run/secrets/kubernetes.io/serviceaccount')
    context = ssl.create_default_context(cafile=str(base / 'ca.crt'))
    host = os.environ.get('KUBERNETES_SERVICE_HOST', 'kubernetes.default.svc')
    port = os.environ.get('KUBERNETES_SERVICE_PORT_HTTPS', '443')
    namespace, name = (os.environ[k] for k in ('HARNESS_POD_NAMESPACE', 'HARNESS_POD_NAME'))
    path = '/api/v1/namespaces/' + urllib.parse.quote(namespace, safe='') + '/pods/' + urllib.parse.quote(name, safe='')
    request = urllib.request.Request('https://' + host + ':' + port + path,
        headers={'Authorization': 'Bearer ' + (base / 'token').read_text().strip()})
    with urllib.request.urlopen(request, context=context, timeout=20) as response:
        payload = response.read(2 * 1024 * 1024)
    return validate_pod(json.loads(payload), os.environ)


def gpu_inventory():
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,pci.bus_id,name,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    rows = []
    for fields in csv.reader(output.splitlines()):
        index, gpu_uuid, pci, name, free, utilization = [v.strip() for v in fields]
        rows.append({'physical_nvml_index': int(index), 'uuid': gpu_uuid, 'pci_bus_id': pci,
                     'name': name, 'free_mib': int(free), 'utilization_percent': int(utilization)})
    if len(rows) != GPU_COUNT or len({r['uuid'] for r in rows}) != GPU_COUNT:
        raise RuntimeError('Exactly eight distinct allocated physical GPUs must be exposed')
    models = {r['name'].replace(' ', '').replace('-', '').casefold() for r in rows}
    if not all(name.startswith(('nvidiageforcertx5090', 'nvidiarpbzzz6')) for name in models):
        raise RuntimeError('Actual GPU hardware is outside the authorized shared queue')
    if any(not r['uuid'].startswith('GPU-') for r in rows):
        raise RuntimeError('Missing physical GPU UUID')
    return rows


def cuda_mapping(rows, lane):
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('Episode parent must not inherit a CUDA mask; each SDK lane applies its own mask')
    script = r'''
import json, os, sys, torch
rows, lane = json.loads(sys.argv[1]), int(sys.argv[2])
assert torch.cuda.device_count() == 8 and torch.cuda.is_available()
props = [torch.cuda.get_device_properties(i) for i in range(8)]
def norm(value):
    value = value.hex() if isinstance(value, bytes) else str(value)
    return value.casefold().removeprefix('gpu-').replace('-', '')
by_uuid = {norm(r['uuid']): r for r in rows}
uuids = [getattr(p, 'uuid', None) for p in props]
if all(v is not None and norm(v) in by_uuid for v in uuids):
    mapped = [by_uuid[norm(v)] for v in uuids]
    method = 'CUDA device-properties UUID'
else:
    def key(row):
        domain, bus, rest = row['pci_bus_id'].split(':')
        device, function = rest.split('.')
        return tuple(int(x,16) for x in (domain,bus,device,function))
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID' and 'CUDA_VISIBLE_DEVICES' not in os.environ
    mapped, method = sorted(rows,key=key), 'PCI_BUS_ID without inherited CUDA mask'
assert len({r['uuid'] for r in mapped}) == 8
probe = torch.ones(16,device='cuda:'+str(lane))
assert float((probe+1).sum().item()) == 32
torch.cuda.synchronize(lane)
print(json.dumps({'method':method,'devices':[dict(row,cuda_local_index=i,cuda_name=props[i].name)
    for i,row in enumerate(mapped)],'probe_lane':lane,'torch':torch.__version__,'cuda':torch.version.cuda}))
'''
    completed = subprocess.run([sys.executable, '-B', '-c', script, json.dumps(rows), str(lane)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode:
        raise RuntimeError('GPU identity probe failed: ' + completed.stderr[-2000:])
    proof = json.loads(completed.stdout)
    proof['probe_kind'] = '16-element CUDA arithmetic; no model or simulation'
    return proof
