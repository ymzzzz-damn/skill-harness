"""Copy a verified same-version extension once and create eight unused cache entrances."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
from protocol import OUTPUT,RUNTIME,URDF_SOURCE,URDF_TREE_SHA256,URDF_VERSION,now,write


def tree_manifest(root):
    root=Path(root)
    if not root.is_dir():
        raise FileNotFoundError('Required reviewed extension cache is missing: '+str(root))
    hashes={}
    sizes={}
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':
            continue
        digest=hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
        key=str(path.relative_to(root))
        hashes[key]=digest.hexdigest();sizes[key]=path.stat().st_size
    canonical=json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()
    return {'tree_sha256':hashlib.sha256(canonical).hexdigest(),'files':len(hashes),
            'bytes':sum(sizes.values()),'file_hashes':hashes}


def prewarm(source=URDF_SOURCE,runtime=RUNTIME,output=OUTPUT,expected=URDF_TREE_SHA256,version=URDF_VERSION):
    source,runtime,output=map(Path,(source,runtime,output))
    if list(output.glob('full/*/layout_*/episode_001')):
        raise RuntimeError('Cache prewarm must precede every native episode')
    manifest=tree_manifest(source)
    if manifest['tree_sha256']!=expected:
        raise RuntimeError('Extension bytes differ from the pinned, verified package')
    destination=runtime/'extension-cache'/version
    destination.parent.mkdir(parents=True,exist_ok=True)
    copied=False
    if os.path.lexists(destination):
        if not destination.is_dir() or tree_manifest(destination)!=manifest:
            raise RuntimeError('Existing extension cache differs; refusing overwrite')
    else:
        temporary=destination.with_name('.'+destination.name+'.'+str(os.getpid())+'.tmp')
        shutil.copytree(source,temporary,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        if tree_manifest(temporary)!=manifest:
            raise RuntimeError('Extension copy verification failed')
        if os.path.lexists(destination):
            raise RuntimeError('Concurrent cache destination appeared; refusing overwrite')
        os.rename(temporary,destination)
        copied=True
    entries=[]
    for lane in range(8):
        entry=runtime/f'lanes/{lane}/kit-portable/data/Kit/Isaac-Sim/5.1/exts/3'/version
        if os.path.lexists(entry):
            if not entry.is_dir() or tree_manifest(entry)!=manifest:
                raise RuntimeError('Existing lane cache is not the exact pinned version; refusing replacement')
            action='existing_verified_entry_preserved'
        else:
            entry.parent.mkdir(parents=True,exist_ok=True)
            os.symlink(str(destination),str(entry))
            action='exclusive_atomic_symlink_created'
        entries.append({'gpu_lane':lane,'path':str(entry),'target':str(entry.resolve()),'action':action})
    receipt={'schema':'harness-extension-prewarm/v1','at':now(),'source':str(source),
        'destination':str(destination),'copied':copied,'version':version,'manifest':manifest,
        'lanes':entries,'model_calls':0,'native_actions':0,
        'scope':'Same extension bytes only; no SDK/task/budget changes or existing-entry replacement'}
    write(output/'preflight/extension-prewarm.json',receipt)
    return receipt
