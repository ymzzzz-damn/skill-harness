"""Treatment smoke, all treatment layouts, then all paired baseline layouts."""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from boot_recovery import MAX_BOOT_ATTEMPTS, archive_unstarted_boot, boot_retry_allowed
from protocol import (ARMS, CONTROLS, OUTPUT, PYTHON, SMOKE_LAYOUTS, episode_dir,
                      full_plan, load, now, verify_freeze, write)
from report import build_report, inspect_episode

D = Path(__file__).resolve().parent


def video_check(out):
    import cv2
    files = sorted((out / 'bench/eval_result').rglob('*.mp4'))
    rows = []
    for path in files:
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        first, _ = cap.read() if count else (False, None)
        last = False
        if first:
            cap.set(cv2.CAP_PROP_POS_FRAMES, count-1)
            last, _ = cap.read()
        cap.release()
        rows.append({'path': str(path), 'frames': count, 'first_decodes': bool(first), 'last_decodes': bool(last)})
    cameras = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
    valid = len(files) == 3 and all(sum(name in p.name for p in files) == 1 for name in cameras)
    return {'valid': valid and all(r['first_decodes'] and r['last_decodes'] for r in rows), 'videos': rows}


def smoke_qualifies(rows):
    return (len(rows) == len(SMOKE_LAYOUTS)
        and {(r['arm'],r['layout']) for r in rows} == {('conditions',i) for i in SMOKE_LAYOUTS}
        and all(r['valid'] and r['status'] == 'completed' and not r.get('error')
                and r.get('contract_registered') is True
                and r.get('condition_guard_evidence_valid') is True
                and r.get('task_timing_valid') is True
                and r.get('video_check',{}).get('valid') is True for r in rows))


class Batch:
    def __init__(self, frozen):
        self.frozen = Path(frozen)
        self.lock = threading.RLock()
        self.records, self.stage_times = [], {}

    def update(self, status, **values):
        with self.lock:
            write(OUTPUT / 'batch_status.json', {'status': status, 'at': now(),
                'controls': CONTROLS, 'expected_full': 110, 'records': list(self.records),
                'stage_elapsed_wall': dict(self.stage_times), **values})

    def one(self, phase, item):
        arm, layout, gpu = (item[k] for k in ('arm','layout','gpu'))
        out = episode_dir(phase,arm,layout)
        logroot = OUTPUT / 'logs' / phase / arm
        logroot.mkdir(parents=True,exist_ok=True)
        started, archives = time.monotonic(), []
        worker = OUTPUT / 'workers' / f'{phase}_{arm}_gpu{gpu}.json'
        write(worker, {'status':'running','at':now(),**item})
        command = [str(PYTHON),'-B','-u',str(D/'run_episode.py'),'--phase',phase,'--arm',arm,
            '--layout',str(layout),'--gpu',str(gpu),'--freeze',str(self.frozen),'--allow-model-calls']
        exception, returncode, attempt = None, None, 0
        for attempt in range(1,MAX_BOOT_ATTEMPTS+1):
            before = time.monotonic()
            logfile = logroot / f'layout{layout:02d}_boot{attempt}.log'
            try:
                with logfile.open('w') as log:
                    completed = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ))
                returncode = completed.returncode
                retry = attempt < MAX_BOOT_ATTEMPTS and boot_retry_allowed(out)
                if retry:
                    archives.append(archive_unstarted_boot(out,attempt,time.monotonic()-before))
                else:
                    break
            except Exception as exc:
                exception = type(exc).__name__+': '+str(exc)
                break
        try:
            record = inspect_episode(phase,arm,layout)
        except Exception as exc:
            record = {'phase':phase,'arm':arm,'layout':layout,'attempted':True,
                'valid':False,'native_record_valid':False,'success':None,'status':'inspection_error',
                'error':type(exc).__name__+': '+str(exc)}
        record.update(returncode=returncode,boot_attempts_executed=attempt,
                      worker_wall_s=time.monotonic()-started,unstarted_boot_archives=archives)
        if exception:
            record.update(valid=False,status='launcher_error',error=exception)
        if phase=='smoke':
            try:
                record['video_check']=video_check(out)
            except Exception as exc:
                record['video_check']={'valid':False,'error':type(exc).__name__+': '+str(exc)}
        with self.lock:
            self.records.append(record)
        write(worker,{'status':'finished','at':now(),'record':record})
        self.update('running',current_phase=phase,current_arm=arm)
        return record

    def lane(self,arm,gpu):
        return [self.one('full',item) for item in full_plan(gpu,arm)]

    def stage(self,phase,arm):
        verify_freeze(json.loads(self.frozen.read_text()))
        key=phase+'_'+arm
        start=time.monotonic()
        self.stage_times[key]={'started_at':now(),'status':'running'}
        self.update('running',current_phase=phase,current_arm=arm)
        if phase=='smoke':
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(self.one,phase,{'arm':arm,'layout':layout,'gpu':layout%8})
                         for layout in SMOKE_LAYOUTS]
                rows=[future.result() for future in futures]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures=[pool.submit(self.lane,arm,gpu) for gpu in range(8)]
                rows=[row for future in futures for row in future.result()]
        self.stage_times[key].update(ended_at=now(),elapsed_wall_s=time.monotonic()-start,
            status='finished',attempted=len(rows),valid=sum(r['valid'] for r in rows))
        write(OUTPUT/'stage_times.json',self.stage_times)
        write(OUTPUT/(phase+'-report.json'),build_report(phase=phase))
        return rows

    def run(self):
        pilot=self.stage('smoke','conditions')
        passed=smoke_qualifies(pilot)
        write(OUTPUT/'smoke-qualification.json',{'passed':passed,'at':now(),
            'criterion':'Both treatment episodes completed with valid native identity and three decodable camera videos; task success is never a gate.',
            'records':pilot})
        if not passed:
            self.update('needs_attention',reason='Treatment smoke evidence/runtime validation failed; no full stage started')
            return 2
        # The user explicitly prioritizes treatment. Complete all 55 attempts
        # before any control episode, keeping layout%8 and physical UUID fixed.
        for arm in CONTROLS['stage_order']:
            self.stage('full',arm)
        self.update('finished')
        return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--allow-model-calls',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.dry_run:
        print(json.dumps({'stages':[{'phase':'smoke','arm':'conditions','layouts':SMOKE_LAYOUTS},
            *[{'phase':'full','arm':arm,'episodes':55,'lanes':{str(g):[i['layout'] for i in full_plan(g,arm)] for g in range(8)}}
              for arm in CONTROLS['stage_order']]],'model_calls':0,'native_actions':0},indent=2))
        return 0
    if not args.allow_model_calls:
        parser.error('Explicit experiment activation flag is required')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUTPUT/'batch_status.json').exists():
            raise FileExistsError('Refusing to resume/replay an existing batch automatically')
        batch=Batch(args.freeze)
        try:
            return batch.run()
        except BaseException as exc:
            batch.update('needs_attention',error=type(exc).__name__+': '+str(exc))
            raise


if __name__=='__main__':
    raise SystemExit(main())
