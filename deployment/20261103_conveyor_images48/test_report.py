import copy
import json
from pathlib import Path
import tempfile
import unittest
from protocol import CONTROLS, ORIGINAL_NATIVE_HANDLERS, TASK, episode_dir, sha, write
from report import arm_summary,build_report,inspect_episode,model_accounting,native_outcome,paired,stats


class ReportTests(unittest.TestCase):
    def make_episode(self,root,arm='conditions',layout=3,success=False):
        out=episode_dir('full',arm,layout,root)
        assignment={'layout':layout,'gpu':2,'gpu_uuid':'GPU-2','pod_uid':'test-pod','node':'test-node'}
        mapping_path=Path(root)/'treatment-mapping.json'
        mapping=json.loads(mapping_path.read_text()) if mapping_path.is_file() else {'layouts':{}}
        mapping['layouts'][str(layout)]=assignment;write(mapping_path,mapping)
        row={'request':1,'upstream_requested':True,'images_before':60,'images_after':48,'images_removed':12,
             'max_images':48,'initial_images_preserved':3,'latest_observation_images_preserved':3}
        image_context={'schema':'harness-image-context-v1','enabled':True,'max_images':48,'reference_image_count':3,
            'transport':CONTROLS['image_transport'],'automatic_replays':False,'original_artifacts_modified':False,
            'requests_started':1,'requests':1,'pending_requests':0,'upstream_requests':1,'requests_pruned':1,'images_removed':12,'max_images_after':48,
            'max_images_before':60,'audit':[row],'errors':[]}
        write(out/'initialization-receipt.json',{'native_client_cleanup_confirmed':True})
        write(out/'image_context.json',image_context)
        (out/'image_context_requests.jsonl').write_text(json.dumps(row)+'\n')
        frozen=Path(root)/'source-freeze.json'
        write(frozen,{'controls':CONTROLS,'sources':{
            'pyrua':{'root':'/fake/PyRUA-Lean/source','files':{'src/pyrualean/__init__.py':'a'*64}},
            'harness':{'root':'/fake/harness/source','files':{'harnesslab/__init__.py':'b'*64}}}})
        write(out/'experiment.json',{'controls':CONTROLS,'arm':arm,'phase':'full','layout':layout,
            'task_conditions':arm=='conditions','gpu':2,'gpu_uuid':'GPU-2','treatment_assignment':assignment,
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
            'generation':{'disabled_features':['shell_snapshot','memories','shell_tool'],'attempts':1,'image_context':image_context},
            'max_context_images':48,
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

    def test_missing_guard_or_task_end_invalidates_evidence(self):
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

    def test_unconfirmed_native_cleanup_invalidates_but_preserves_native_success(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.make_episode(temp,success=True)
            write(out/'initialization-receipt.json',{'native_client_cleanup_confirmed':False})
            record=inspect_episode('full','conditions',3,temp)
            self.assertFalse(record['valid']);self.assertFalse(record['native_cleanup_confirmed'])
            self.assertTrue(record['native_record_valid']);self.assertTrue(record['success'])

    def test_final_report_requires_result_bound_three_camera_decode_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.make_episode(temp)
            self.assertFalse(inspect_episode('full','conditions',3,temp,require_video=True)['valid'])
            worker={'arm':'conditions','layout':3,'result_sha256':sha(out/'result.json'),
                'video_check':{'valid':True,'videos':[{'path':f'/episode/{cam}.mp4','frames':12,
                    'first_decodes':True,'last_decodes':True} for cam in ('cam_head','cam_left_wrist','cam_right_wrist')]}}
            write(Path(temp)/'workers/conditions_layout03.json',worker)
            self.assertTrue(inspect_episode('full','conditions',3,temp,require_video=True)['valid'])
            worker['result_sha256']='stale';write(Path(temp)/'workers/conditions_layout03.json',worker)
            self.assertFalse(inspect_episode('full','conditions',3,temp,require_video=True)['valid'])

    def update_image_context(self,out,mutate,extra=None):
        result=json.loads((out/'result.json').read_text())
        saved=json.loads((out/'image_context.json').read_text())
        mutate(saved)
        result['generation']['image_context']=saved
        write(out/'result.json',result);write(out/'image_context.json',saved)
        rows=saved['audit']+(extra or [])
        (out/'image_context_requests.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')

    def test_actual_cap_initial_references_transport_and_single_agent_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            for mutation in (
                lambda s:s['audit'][0].update(images_after=49,images_removed=11),
                lambda s:s['audit'][0].update(initial_images_preserved=2),
                lambda s:s.update(transport={'wire_api':'responses','supports_websockets':True,'request_compression':False}),
                lambda s:s.update(max_images=64),
                lambda s:s.update(enabled=False),
            ):
                out=self.make_episode(temp);self.update_image_context(out,mutation)
                self.assertFalse(inspect_episode('full','conditions',3,temp)['image_context_valid'])
            out=self.make_episode(temp)
            path=out/'result.json';result=json.loads(path.read_text());result['generation']['attempts']=2;write(path,result)
            self.assertFalse(inspect_episode('full','conditions',3,temp)['image_context_valid'])

    def test_native_stop_disconnect_and_pending_late_terminal_metadata_remain_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.make_episode(temp,success=True)
            late={'request':2,'error':'client_disconnected','upstream_requested':True,
                  'images_before':70,'images_after':48,'images_removed':22,'max_images':48,'initial_images_preserved':3}
            def mutation(saved):
                saved['audit'][0]['error']='client_disconnected'
                saved.update(requests_started=2,pending_requests=1,errors=['client_disconnected'])
            self.update_image_context(out,mutation,[late])
            record=inspect_episode('full','conditions',3,temp)
            self.assertTrue(record['valid']);self.assertTrue(record['image_context_valid'])
            self.assertEqual(record['proxy_late_terminal_audit_records'],1)
            self.assertEqual(record['proxy_terminal_errors'][0]['error'],'client_disconnected')
            late['images_after']=49;late['images_removed']=21
            self.update_image_context(out,lambda saved:None,[late])
            self.assertFalse(inspect_episode('full','conditions',3,temp)['image_context_valid'])

    def test_saved_audit_and_actual_gpu_mapping_cannot_be_forged_by_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.make_episode(temp)
            (out/'image_context_requests.jsonl').write_text('{}\n')
            self.assertFalse(inspect_episode('full','conditions',3,temp)['image_context_valid'])
            out=self.make_episode(temp)
            mapping=json.loads((Path(temp)/'treatment-mapping.json').read_text())
            mapping['layouts']['3']['gpu']=7;write(Path(temp)/'treatment-mapping.json',mapping)
            self.assertFalse(inspect_episode('full','conditions',3,temp)['valid'])

    def test_aggregate_inspection_exception_keeps_all_cases_and_no_invented_failure_or_time(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.make_episode(temp,success=True)
            write(Path(temp)/'workers/conditions_layout03.json',{'arm':'conditions','layout':3,
                'attempted':True,'closed':True,'status':'inspection_error','success':True,'task_wall_s':999})
            write(out/'result.json',[])  # Reproduces a per-case inspect exception.
            report=build_report(temp)
            self.assertEqual(len(report['episodes']),110)
            row=next(r for r in report['episodes'] if r['arm']=='conditions' and r['layout']==3)
            self.assertTrue(row['attempted']);self.assertTrue(row['closed']);self.assertFalse(row['valid'])
            self.assertEqual(row['status'],'inspection_error');self.assertIsNone(row['success'])
            self.assertIsNone(row['task_wall_s']);self.assertIsNone(row['native_actions'])
            self.assertEqual(row['gpu_uuid'],'GPU-2')
            self.assertEqual(report['arms']['conditions']['attempted'],1)
            self.assertEqual(report['arms']['conditions']['native_outcomes'],0)
            self.assertIsNone(report['arms']['conditions']['all_attempted_metrics']['task_wall_s']['sum'])
            self.assertEqual(report['arms']['control']['pending'],55)
            self.assertEqual(report['inspection_errors'][0]['layout'],3)
            self.assertIsNone(report['paired']['success_rate_difference'])

    def test_inspection_error_without_worker_uses_terminal_launch_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            out=episode_dir('full','control',7,temp)
            write(out/'result.json',[])  # no trustworthy result/worker object
            write(out.parent/'launch_status.json',{'status':'launcher_error'})
            report=build_report(temp)
            row=next(r for r in report['episodes'] if r['arm']=='control' and r['layout']==7)
            self.assertTrue(row['attempted']);self.assertTrue(row['closed'])
            self.assertIsNone(row['success']);self.assertFalse(row['native_record_valid'])
            self.assertEqual(report['arms']['control']['execution_or_identity_errors'],1)

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
