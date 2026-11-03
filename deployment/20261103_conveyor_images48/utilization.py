"""Lightweight metadata sampling. This monitor never creates GPU compute work."""
from __future__ import annotations
import csv
import datetime as dt
import json
import math
from pathlib import Path
import subprocess
import threading
import time
from protocol import CONTROLS,OUTPUT,load,now,write


def parse_gpu_csv(value):
    def metric(text,maximum=None):
        # NVML may return N/A for one field while all other metadata remains
        # usable. Unknown values are missing, never a measured zero.
        try:result=int(text)
        except (ValueError,TypeError):return None
        return result if result>=0 and (maximum is None or result<=maximum) else None
    rows=[]
    for fields in csv.reader(value.splitlines()):
        if not fields:continue
        index,uuid,utilization,used,total=[x.strip() for x in fields]
        rows.append({'nvml_index':int(index),'uuid':uuid,'gpu_utilization_percent':metric(utilization,100),
                     'memory_used_mib':metric(used),'memory_total_mib':metric(total)})
    return rows


def measured(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0


def read_cgroup(root=Path('/sys/fs/cgroup')):
    result={}
    for name in ('memory.current','memory.max','memory.events','cpu.stat','cpu.max'):
        path=Path(root)/name
        if not path.is_file():continue
        value=path.read_text().strip()
        if name in ('memory.events','cpu.stat'):
            result[name]={line.split()[0]:int(line.split()[1]) for line in value.splitlines() if len(line.split())==2}
        elif value.isdigit():result[name]=int(value)
        else:result[name]=value
    return result


def enrich(rows,allocation,live,output=OUTPUT,previous_actions=None):
    previous_actions=previous_actions if previous_actions is not None else {}
    by_uuid={r['uuid']:r for r in rows}
    if set(by_uuid)!={r['uuid'] for r in allocation['devices']}:
        raise RuntimeError('Monitor device UUIDs differ from the admitted allocation')
    devices=[]
    for device in allocation['devices']:
        lane=device['cuda_local_index'];worker=live.get('workers',{}).get(str(lane),{})
        row=dict(by_uuid[device['uuid']],gpu_lane=lane,stage=live.get('stage'),
                 case=worker.get('layout'),worker_assigned=worker.get('status')=='running',
                 worker_state=worker.get('status','idle'))
        if row['worker_assigned']:
            out=Path(output)/'full'/live['stage']/f'layout_{worker["layout"]:02d}'/'episode_001'
            receipt=load(out/'initialization-receipt.json',{})
            actions=receipt.get('native_action_requests')
            key=(lane,live['stage'],worker['layout'])
            if receipt.get('model_phase_entered'):
                activity=load(out/'activity.json',{})
                row['runtime_phase']=activity.get('phase','agent_or_non_native_tool')
                row['activity_at']=activity.get('at')
                row['no_new_native_action_since_previous_sample']=previous_actions.get(key)==actions
            elif receipt.get('first_rgb_seen'):row['runtime_phase']='tool_initialization'
            elif receipt.get('native_boot_started'):row['runtime_phase']='native_scene_loading'
            else:row['runtime_phase']='worker_preflight_or_boot_slot_wait'
            row['native_actions_observed']=actions
            previous_actions[key]=actions
        else:row['runtime_phase']='no_case_assigned'
        devices.append(row)
    return devices


class HourlySummary:
    def __init__(self):self.groups={}
    def add(self,sample):
        hour=sample['utc'][:13]+':00:00Z'
        group=self.groups.setdefault(hour,{'samples':0,'assigned_worker_sum':0,'gpus':{}})
        group['samples']+=1
        group['assigned_worker_sum']+=sum(d['worker_assigned'] for d in sample['devices'])
        for d in sample['devices']:
            row=group['gpus'].setdefault(d['uuid'],{'gpu_lane':d['gpu_lane'],'samples':0,
                'utilization_sum':0,'utilization_samples':0,'memory_mib_sum':0,'memory_samples':0,
                'active_gt0':0,'busy_ge10':0,'worker_assigned':0})
            row['samples']+=1
            utilization,memory=d.get('gpu_utilization_percent'),d.get('memory_used_mib')
            if measured(utilization):
                row['utilization_sum']+=utilization;row['utilization_samples']+=1
                row['active_gt0']+=utilization>0;row['busy_ge10']+=utilization>=10
            if measured(memory):row['memory_mib_sum']+=memory;row['memory_samples']+=1
            row['worker_assigned']+=d['worker_assigned']
    def result(self):
        result={}
        for hour,group in self.groups.items():
            devices={}
            for uuid,row in group['gpus'].items():
                n,u,m=row['samples'],row['utilization_samples'],row['memory_samples']
                devices[uuid]={'gpu_lane':row['gpu_lane'],'samples':n,
                    'gpu_utilization_valid_samples':u,'gpu_utilization_missing_samples':n-u,
                    'memory_used_valid_samples':m,'memory_used_missing_samples':n-m,
                    'mean_gpu_utilization_percent':row['utilization_sum']/u if u else None,
                    'mean_memory_used_mib':row['memory_mib_sum']/m if m else None,
                    'gpu_active_fraction_util_gt0':row['active_gt0']/u if u else None,
                    'gpu_busy_fraction_util_ge10':row['busy_ge10']/u if u else None,
                    'worker_assigned_fraction':row['worker_assigned']/n}
            result[hour]={'samples':group['samples'],
                'mean_assigned_worker_count':group['assigned_worker_sum']/group['samples'],'gpus':devices}
        return result


class Monitor:
    def __init__(self,allocation,output=OUTPUT,interval=CONTROLS['gpu_sample_interval_s'],query=None,cgroup=None):
        self.output=Path(output);self.allocation=allocation;self.interval=interval
        self.query=query or self.nvidia_query;self.cgroup=cgroup or read_cgroup
        self.stop=threading.Event();self.thread=None;self.hourly=HourlySummary();self.previous_actions={}
        self.started=time.monotonic();self.sample_count=0;self.error_count=0
    @staticmethod
    def nvidia_query():
        packet=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5)
        if packet.returncode:raise RuntimeError('nvidia-smi metadata query failed')
        return parse_gpu_csv(packet.stdout)
    def once(self):
        self.output.joinpath('utilization').mkdir(parents=True,exist_ok=True)
        timestamp=dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            devices=enrich(self.query(),self.allocation,load(self.output/'worker-live.json',{}),self.output,self.previous_actions)
            group=self.cgroup();current,limit=group.get('memory.current'),group.get('memory.max')
            utilization=[d['gpu_utilization_percent'] for d in devices if measured(d.get('gpu_utilization_percent'))]
            sample={'utc':timestamp,'elapsed_s':time.monotonic()-self.started,'devices':devices,
                'assigned_worker_count':sum(d['worker_assigned'] for d in devices),
                'gpu_utilization_valid_devices':len(utilization),
                'gpu_utilization_missing_devices':len(devices)-len(utilization),
                'gpu_busy_count_util_ge10':sum(value>=10 for value in utilization) if utilization else None,
                'cgroup':group,'memory_pressure_above90percent':bool(type(current) is int and type(limit) is int and current>.9*limit)}
            previous=getattr(self,'previous_cgroup',{})
            sample['cgroup_counter_deltas']={name:{key:max(0,value-previous.get(name,{}).get(key,value))
                for key,value in group.get(name,{}).items()} for name in ('memory.events','cpu.stat')}
            self.previous_cgroup=group
            self.hourly.add(sample);self.sample_count+=1
        except Exception as exc:
            self.error_count+=1
            sample={'utc':timestamp,'error':type(exc).__name__+': '+str(exc)}
        with (self.output/'utilization/samples.jsonl').open('a') as handle:handle.write(json.dumps(sample)+'\n')
        write(self.output/'utilization/latest.json',sample)
        self.save()
    def save(self):
        write(self.output/'utilization/summary.json',{'at':now(),'interval_s':self.interval,
            'samples':self.sample_count,'sampling_errors':self.error_count,'hourly':self.hourly.result(),
            'notes':['Periodic sampled averages/fractions, not continuous instrumentation.',
                'Unavailable GPU fields are null; each hourly metric uses its own valid-sample denominator. Busy counts cover measured GPUs only, with missing device counts reported separately.',
                'An assigned worker may be waiting for LLM/provider or inspecting observations while GPU utilization is low.',
                'native_control is an in-flight native step; agent_or_non_native_tool includes LLM wait, Python and perception, not proof of a specific model request.',
                'No artificial GPU load is generated; memory pressure/OOM/throttling counters are recorded without changing threads or budgets.']})
    def _run(self):
        while not self.stop.is_set():
            self.once();self.stop.wait(self.interval)
    def start(self):
        if self.thread is not None:raise RuntimeError('Monitor may only start once')
        self.thread=threading.Thread(target=self._run,name='gpu-metadata-monitor',daemon=True)
        self.thread.start();return self
    def close(self):
        self.stop.set()
        if self.thread is not None:self.thread.join(timeout=10)
        self.save()
