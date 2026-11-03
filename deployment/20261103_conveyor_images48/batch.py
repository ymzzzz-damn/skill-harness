"""Eight formal treatment workers, then hardware-matched control; no separate smoke."""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
from boot_recovery import MAX_BOOT_ATTEMPTS,archive_unstarted_boot,boot_retry_allowed
from protocol import CONTROLS,OUTPUT,PYTHON,control_plan,episode_dir,load,now,verify_freeze,write
from report import build_report,inspect_episode

D=Path(__file__).resolve().parent


def video_check(out):
    import cv2
    files=sorted((out/'bench/eval_result').rglob('*.mp4'));rows=[]
    for path in files:
        cap=cv2.VideoCapture(str(path));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        first,_=cap.read() if count else (False,None);last=False
        if first:
            cap.set(cv2.CAP_PROP_POS_FRAMES,count-1);last,_=cap.read()
        cap.release()
        rows.append({'path':str(path),'frames':count,'first_decodes':bool(first),'last_decodes':bool(last)})
    cameras=('cam_head','cam_left_wrist','cam_right_wrist')
    valid=len(files)==3 and all(sum(name in p.name for p in files)==1 for name in cameras)
    return {'valid':valid and all(r['first_decodes'] and r['last_decodes'] for r in rows),'videos':rows}


def execution_problem(record):
    """A zero native score is an ordinary outcome, never a dispatch stop signal."""
    return (not record.get('valid') or bool(record.get('error'))
            or record.get('status') not in ('completed','timeout','call_budget')
            or record.get('video_check',{}).get('valid') is not True)


def cohort_problems(out):
    """Only shared-invariant or unsafe-cleanup failures stop the whole cohort."""
    launch=load(Path(out).parent/'launch_status.json',{})
    receipt=load(Path(out)/'initialization-receipt.json',{})
    reasons=[]
    if not launch or launch.get('failure_scope')=='global_invariant':
        reasons.append('launcher/global source, identity, allocation or mapping invariant unverified')
    if receipt:
        if receipt.get('native_client_cleanup_confirmed') is not True:
            reasons.append('native client process cleanup is unconfirmed')
    elif launch.get('native_not_started') is not True:
        reasons.append('native start/cleanup boundary is unconfirmed')
    return reasons


class Batch:
    def __init__(self,frozen):
        self.frozen=Path(frozen);self.lock=threading.RLock();self.stop=threading.Event()
        self.records=[];self.stage_times={};self.stage_name=None
        self.workers={str(gpu):{'status':'idle'} for gpu in range(8)}
        self.allocation=load(OUTPUT/'allocation.json',{})
        if len(self.allocation.get('devices',[]))!=8:
            raise RuntimeError('Verified eight-GPU allocation proof is required')
        self.mapping={'schema':'conveyor-images48-assignments/v1','allocation':self.allocation,'layouts':{}}
        self.pending=queue.Queue()
        for layout in range(55):self.pending.put(layout)

    def update(self,status='running',**values):
        with self.lock:
            live={'stage':self.stage_name,'at':now(),'workers':dict(self.workers)}
            write(OUTPUT/'worker-live.json',live)
            write(OUTPUT/'batch_status.json',{'status':status,'at':now(),'controls':CONTROLS,
                'expected_full':110,'records':list(self.records),'stage_elapsed_wall':dict(self.stage_times),
                'dispatch_stopped_for_global_error':self.stop.is_set(),
                'case_execution_errors':sum(bool(r.get('execution_problem')) for r in self.records),**live,**values})

    def reserve(self,gpu):
        """Reserve and persist mapping before a child episode can receive an action."""
        with self.lock:
            if self.stop.is_set():return None
            try:layout=self.pending.get_nowait()
            except queue.Empty:return None
            if str(layout) in self.mapping['layouts']:raise RuntimeError('Duplicate treatment assignment')
            device=self.allocation['devices'][gpu];proof=self.allocation['pod']
            if device.get('cuda_local_index')!=gpu:raise RuntimeError('Allocation lane ordering mismatch')
            item={'layout':layout,'arm':'conditions','gpu':gpu}
            self.mapping['layouts'][str(layout)]={**item,'gpu_uuid':device['uuid'],
                'pod_uid':proof['uid'],'node':proof['node'],'assigned_at':now(),'status':'assigned'}
            write(OUTPUT/'treatment-mapping.json',self.mapping)
            return item

    def one(self,item):
        arm,layout,gpu=(item[k] for k in ('arm','layout','gpu'))
        out=episode_dir('full',arm,layout);logroot=OUTPUT/'logs/full'/arm
        logroot.mkdir(parents=True,exist_ok=True);started=time.monotonic();archives=[]
        with self.lock:self.workers[str(gpu)]={'status':'running','layout':layout,'arm':arm,'at':now()}
        self.update()
        try:verify_freeze(json.loads(self.frozen.read_text()))
        except Exception:
            self.stop.set();raise
        command=[str(PYTHON),'-B','-u',str(D/'run_episode.py'),'--phase','full','--arm',arm,
                 '--layout',str(layout),'--gpu',str(gpu),'--freeze',str(self.frozen),'--allow-model-calls']
        exception,returncode,attempt=None,None,0
        for attempt in range(1,MAX_BOOT_ATTEMPTS+1):
            before=time.monotonic();logfile=logroot/f'layout{layout:02d}_boot{attempt}.log'
            try:
                with logfile.open('w') as log:
                    completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ))
                returncode=completed.returncode
                retry=attempt<MAX_BOOT_ATTEMPTS and boot_retry_allowed(out)
                if retry:archives.append(archive_unstarted_boot(out,attempt,time.monotonic()-before))
                else:break
            except Exception as exc:
                exception=type(exc).__name__+': '+str(exc);break
        try:record=inspect_episode('full',arm,layout)
        except Exception as exc:
            record={'phase':'full','arm':arm,'layout':layout,'attempted':True,'closed':True,
                'valid':False,'native_record_valid':False,'success':None,'status':'inspection_error',
                'error':type(exc).__name__+': '+str(exc)}
        record.update(returncode=returncode,boot_attempts_executed=attempt,
            worker_wall_s=time.monotonic()-started,unstarted_boot_archives=archives)
        if exception:record.update(valid=False,status='launcher_error',error=exception)
        try:record['video_check']=video_check(out)
        except Exception as exc:record['video_check']={'valid':False,'error':type(exc).__name__+': '+str(exc)}
        record['execution_problem']=execution_problem(record)
        record['global_stop_reasons']=cohort_problems(out)
        with self.lock:
            self.records.append(record)
            if arm=='conditions':
                entry=self.mapping['layouts'][str(layout)]
                entry.update(status='closed',treatment_task_wall_s=record.get('task_wall_s'),
                             treatment_worker_wall_s=record['worker_wall_s'])
                write(OUTPUT/'treatment-mapping.json',self.mapping)
            self.workers[str(gpu)]={'status':'idle','last_layout':layout,'at':now()}
            if record['global_stop_reasons']:self.stop.set()
            write(OUTPUT/'workers'/f'{arm}_layout{layout:02d}.json',record)
        self.update(reason='Global invariant or unsafe cleanup stopped new dispatch; active cases may finish' if self.stop.is_set() else None)
        return record

    def treatment_lane(self,gpu,initial=None):
        rows=[]
        if initial is not None:rows.append(self.one(initial))
        while True:
            item=self.reserve(gpu)
            if item is None:return rows
            rows.append(self.one(item))

    def control_lane(self,gpu):
        rows=[]
        for item in control_plan(gpu,self.mapping):
            with self.lock:
                if self.stop.is_set():break
            rows.append(self.one(item))
        return rows

    def stage(self,arm):
        verify_freeze(json.loads(self.frozen.read_text()));self.stage_name=arm
        self.stage_times[arm]={'started_at':now(),'status':'running'}
        if arm=='control':
            plans={str(g):control_plan(g,self.mapping) for g in range(8)}
            write(OUTPUT/'control-plan.json',{'at':now(),'lanes':plans,
                'rule':'Descending treatment duration within each unchanged assigned GPU; no cross-GPU rebalance'})
        started=time.monotonic();self.update()
        initial={gpu:self.reserve(gpu) for gpu in range(8)} if arm=='conditions' else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures=([pool.submit(self.treatment_lane,gpu,initial[gpu]) for gpu in range(8)]
                     if arm=='conditions' else [pool.submit(self.control_lane,gpu) for gpu in range(8)])
            rows=[row for future in futures for row in future.result()]
        self.stage_times[arm].update(ended_at=now(),elapsed_wall_s=time.monotonic()-started,
            status='needs_attention' if self.stop.is_set() else ('finished_with_case_errors' if any(r.get('execution_problem') for r in rows) else 'finished'),attempted=len(rows),
            valid=sum(r['valid'] for r in rows))
        write(OUTPUT/'stage_times.json',self.stage_times)
        write(OUTPUT/'full-report.json',build_report())
        self.scene_timing()
        return rows

    def scene_timing(self):
        try:
            subprocess.run([str(PYTHON),'-B',str(D/'scene_excluded_timing.py'),'--root',str(OUTPUT),
                '--out',str(OUTPUT/'scene-excluded-timing.json')],check=True,
                stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        except Exception as exc:
            write(OUTPUT/'scene-timing-error.json',{'at':now(),'error':type(exc).__name__+': '+str(exc)})

    def run(self):
        if (OUTPUT/'treatment-mapping.json').exists():raise FileExistsError('Existing assignment map cannot be resumed implicitly')
        for arm in CONTROLS['stage_order']:
            rows=self.stage(arm)
            if self.stop.is_set() or len(rows)!=55:
                self.update('needs_attention',reason='Global invariant/cleanup error or incomplete dispatch; no measured replay or next stage')
                return 2
        self.update('finished_with_case_errors' if any(r.get('execution_problem') for r in self.records) else 'finished');return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--allow-model-calls',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.dry_run:
        print(json.dumps({'stages':CONTROLS['stage_order'],'separate_smoke':False,'formal_episodes_per_stage':55,
            'workers':8,'treatment_dispatch':'dynamic FIFO','mapping_persisted_before_launch':True,
            'control_dispatch':'same lane/UUID; descending treatment duration','boot_concurrency':4,
            'max_context_images':48,'monitor_interval_s':30,'model_calls':0,'native_actions':0},indent=2));return 0
    if not args.allow_model_calls:parser.error('Explicit activation is required')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUTPUT/'batch_status.json').exists():raise FileExistsError('Batch cannot be resumed/replayed implicitly')
        batch=Batch(args.freeze)
        try:return batch.run()
        except BaseException as exc:
            batch.update('needs_attention',error=type(exc).__name__+': '+str(exc))
            write(OUTPUT/'full-report.json',build_report())
            batch.scene_timing();raise


if __name__=='__main__':raise SystemExit(main())
