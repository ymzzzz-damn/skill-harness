import copy
import json
from pathlib import Path
import tempfile
import unittest
from protocol import CONTROLS, ORIGINAL_NATIVE_HANDLERS, TASK, episode_dir, sha, write
from report import arm_summary, inspect_episode, model_accounting, native_outcome, paired, stats


class ReportTests(unittest.TestCase):
    def make_episode(self,root,arm='conditions',layout=3,success=False):
        out=episode_dir('full',arm,layout,root)
        frozen=Path(root)/'source-freeze.json'
        write(frozen,{'controls':CONTROLS,'sources':{
            'pyrua':{'root':'/fake/PyRUA-Lean/source','files':{'src/pyrualean/__init__.py':'a'*64}},
            'harness':{'root':'/fake/harness/source','files':{'harnesslab/__init__.py':'b'*64}}}})
        write(out/'experiment.json',{'controls':CONTROLS,'arm':arm,'phase':'full','layout':layout,
            'task_conditions':arm=='conditions','gpu':layout%8,'gpu_uuid':'GPU-'+str(layout%8),
            'pod':{'uid':'test-pod','node':'test-node'},'layout_sha256':'layout-sha',
            'source_freeze_sha256':sha(frozen),'source_freeze':str(frozen)})
        write(out/'result.json',{'task':TASK,'seed':layout,'status':'completed','error':None,
            'environment_success':success,'task_wall_s':100,'startup_s':40,'policy_wall_s':95,
            'tool_wall_s':20,'non_tool_agent_s':75,'total_wall_s':145,'env_steps_total':200,
            'tool_set':'pyrua','task_conditions':arm=='conditions','model':'gpt-6-astra',
            'reasoning_effort':'high','max_turns':100,'timeout_s':5400,'max_episode_steps':700,
            'images':'on-demand','feedback':'pure','robot':'robodojo_xpl_rgb','runtime':'codex',
            'arm':'cells','suite':'arx_x5','host':{'primitive_set':'no-vla','boot_options':{
                'eval_seed':0,'source_root':'/fake/PyRUA-Lean/runtime/robodojo-source',
                'xpolicylab_root':'/opt/robodojo/XPolicyLab'}},
            'robot_tools':{'native_handlers':ORIGINAL_NATIVE_HANDLERS,'grasp_installed':False,'global_grasp_supplied':False},
            'dependencies':{'pyrualean':{'path':'/fake/PyRUA-Lean/source/src/pyrualean','sha256':{'__init__.py':'a'*64}},
                            'harnesslab':{'path':'/fake/harness/source/harnesslab','sha256':{'__init__.py':'b'*64}}},
            'agent_features':{'disabled':['shell_snapshot','memories','shell_tool']},
            'generation':{'disabled_features':['shell_snapshot','memories','shell_tool']},
            'task_timing':{'end_observed':True}})
        if arm=='conditions':
            write(out/'task_contract.json',{'schema':'harness-task-contract-v1','registered':True,
                  'plan':{'conditions':[{'id':'goal'}]}})
            write(out/'task_condition_events.json',[{'event':name} for name in
                  ('plan_registered','action_authorized','physical_dispatch','physical_return')])
        write(out/'bench/eval_result/RoboDojo'/TASK/'_result.json',{'eval_time':1,'score':100*int(success),
            'success_rate':float(success),'details':{'0':{'layout_id':layout,'success':success,'score':float(success)}}})
        return out

    def test_native_identity_mismatch_and_missing_report_not_silent_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.make_episode(temp)
            self.assertTrue(native_outcome(out,3)['valid'])
            self.assertFalse(native_outcome(out,4)['valid'])
            record=inspect_episode('full','conditions',3,temp)
            self.assertTrue(record['valid'])
            self.assertIs(record['success'],False)
            missing=inspect_episode('full','conditions',4,temp)
            self.assertFalse(missing['attempted'])
            self.assertIsNone(missing['success'])

    def test_task_wall_includes_controls_and_agent_but_load_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            self.make_episode(temp)
            record=inspect_episode('full','conditions',3,temp)
            summary=arm_summary([record])
            metrics=summary['valid_only_metrics']
            self.assertEqual(metrics['task_wall_s']['sum'],100)
            self.assertEqual(metrics['startup_s']['mean'],40)
            self.assertEqual(metrics['policy_wall_s']['mean'],95)
            self.assertEqual(summary['valid'],1)

    def test_custom_tool_and_manifest_drift_invalidates_result(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.make_episode(temp)
            path=out/'result.json';data=json.loads(path.read_text());data['dependencies']['strong_grasp']={}
            write(path,data)
            self.assertFalse(inspect_episode('full','conditions',3,temp)['valid'])
            del data['dependencies']['strong_grasp'];write(path,data)
            path=out/'experiment.json';data=json.loads(path.read_text());data['task_conditions']=False
            write(path,data)
            self.assertFalse(inspect_episode('full','conditions',3,temp)['valid'])

    def test_actual_tool_code_hash_and_controls_override_manifest_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.make_episode(temp)
            path=out/'result.json';original=json.loads(path.read_text())
            for field,value in (('reasoning_effort','xhigh'),('tool_set','strong-grasp'),('max_episode_steps',200),
                                ('agent_features',{'disabled':['shell_snapshot','memories']})):
                changed=copy.deepcopy(original);changed[field]=value;write(path,changed)
                self.assertFalse(inspect_episode('full','conditions',3,temp)['valid'])
            changed=copy.deepcopy(original)
            changed['dependencies']['pyrualean']['sha256']['__init__.py']='c'*64
            write(path,changed)
            record=inspect_episode('full','conditions',3,temp)
            self.assertFalse(record['valid'])
            self.assertIsNone(record['tool_provenance_sha256'])

    def test_missing_guard_or_task_end_invalidates_smoke_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.make_episode(temp)
            write(out/'task_condition_events.json',[{'event':'plan_registered'}])
            record=inspect_episode('full','conditions',3,temp)
            self.assertFalse(record['valid'])
            self.assertFalse(record['condition_guard_evidence_valid'])
            path=out/'result.json';data=json.loads(path.read_text());data['task_timing']['end_observed']=False
            write(path,data)
            self.assertFalse(inspect_episode('full','conditions',3,temp)['task_timing_valid'])

    def test_native_success_with_runtime_error_stays_visible_but_not_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.make_episode(temp,success=True)
            path=out/'result.json';data=json.loads(path.read_text());data['status']='planner_error'
            write(path,data)
            record=inspect_episode('full','conditions',3,temp)
            summary=arm_summary([record])
            self.assertEqual(summary['native_successes'],1)
            self.assertEqual(summary['valid'],0)
            self.assertEqual(summary['execution_or_identity_errors'],1)

    def test_pairing_checks_physical_gpu_and_keeps_signed_time_delta(self):
        rows=[]
        with tempfile.TemporaryDirectory() as temp:
            for arm in ('control','conditions'):
                self.make_episode(temp,arm,3,success=arm=='conditions')
                rows.append(inspect_episode('full',arm,3,temp))
            rows[0]['task_wall_s']=120
            result=paired(rows,100)
            self.assertEqual(result['n'],1)
            self.assertEqual(result['gained'],1)
            self.assertEqual(result['mean_task_wall_difference_s'],-20)
            rows[0]['gpu_uuid']='GPU-other'
            self.assertEqual(paired(rows,100)['n'],0)

    def test_missing_values_not_zero_and_response_ids_deduplicated(self):
        self.assertIsNone(stats([None])['mean'])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.assertIsNone(model_accounting(root)['model_responses_observed'])
            path=root/'codex-home/sessions/session.jsonl'
            path.parent.mkdir(parents=True)
            path.write_text('\n'.join(json.dumps(x) for x in (
                {'response_id':'resp_one'},{'response_id':'resp_one'},
                {'response_id':'resp_two','total_token_usage':{'input_tokens':40,'output_tokens':5}})))
            accounting=model_accounting(root)
            self.assertEqual(accounting['model_responses_observed'],2)
            self.assertEqual(accounting['token_usage_observed']['input_tokens'],40)


if __name__=='__main__':unittest.main()
