"""Validate the live allocation and frozen dependencies, then activate the batch."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from identity import own_pod
from protocol import (OPENCV_SHA256,OPENCV_WHEEL,OUTPUT,PYRUA,PYTHON,RUNTIME,SOURCE,
                      now,sha,verify_freeze,write)

D=Path(__file__).resolve().parent


def run(command,log,env):
    with Path(log).open('w') as handle:
        result=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,env=env)
    if result.returncode:
        raise RuntimeError('Preflight/batch failed; see '+str(log))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True)
    args=parser.parse_args()
    frozen=json.loads(args.freeze.read_text())
    verify_freeze(frozen)
    proof=own_pod()  # No import of simulator/model packages precedes identity.
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'batch_status.json').exists():
        raise FileExistsError('Existing batch evidence must not be replayed')
    logs=OUTPUT/'preflight'
    logs.mkdir(exist_ok=True)
    write(logs/'pod-identity.json',proof)
    shutil.copyfile(args.freeze,OUTPUT/'source-freeze.json')
    deps=RUNTIME/'python-deps'
    deps.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ)
    env.pop('PYTHONHOME',None)
    env.pop('CUDA_VISIBLE_DEVICES',None)
    env.update(PYTHONPATH=os.pathsep.join(map(str,(deps,SOURCE,PYRUA/'source/src'))),
               PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    os.environ.pop('CUDA_VISIBLE_DEVICES',None)
    for name in ('tmp','cache','config','data'):
        (RUNTIME/name).mkdir(parents=True,exist_ok=True)
    env.update(TMPDIR=str(RUNTIME/'tmp'),XDG_CACHE_HOME=str(RUNTIME/'cache'),
               XDG_CONFIG_HOME=str(RUNTIME/'config'),XDG_DATA_HOME=str(RUNTIME/'data'))
    if sha(OPENCV_WHEEL)!=OPENCV_SHA256:
        raise RuntimeError('OpenCV wheel integrity check failed')
    if not (deps/'cv2').exists():
        run([str(PYTHON),'-m','pip','install','--no-deps','--no-index','--target',str(deps),str(OPENCV_WHEEL)],
            logs/'install-opencv.log',env)
    run([str(PYTHON),'-B','-c',
         'import sys,cv2,numpy,torch,sam3,mcp,harnesslab; '
         'assert "strong_grasp" not in sys.modules; '
         'print({"cv2":cv2.__version__,"numpy":numpy.__version__,"torch":torch.__version__})'],
        logs/'dependency-imports.log',env)
    run([str(PYTHON),'-B','-m','unittest','discover','-s',str(SOURCE/'tests'),'-v'],logs/'harness-tests.log',env)
    run([str(PYTHON),'-B','-m','unittest','discover','-s',str(D),'-p','test_*.py','-v'],logs/'deployment-tests.log',env)
    write(logs/'preflight.json',{'status':'passed','at':now(),'pod':proof,'model_calls':0,'native_actions':0,
        'checks':['live own Pod API','immutable sources/native inputs','isolated OpenCV','dependency imports','CPU tests'],
        'opencv_wheel':str(OPENCV_WHEEL),'opencv_sha256':OPENCV_SHA256,
        'opencv_install_directory':str(deps)})
    run([str(PYTHON),'-B','-u',str(D/'batch.py'),'--freeze',str(args.freeze),'--allow-model-calls'],
        OUTPUT/'batch.log',env)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Do not serialize private config or environment values in receipts.
        if OUTPUT.exists():
            write(OUTPUT/'bootstrap-error.json',{'at':now(),'error':type(exc).__name__+': '+str(exc)})
        raise
