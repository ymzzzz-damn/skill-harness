import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import batch
from protocol import load,write


def allocation():
    return {'pod':{'uid':'pod','node':'node'},
            'devices':[{'cuda_local_index':i,'uuid':f'GPU-{i}'} for i in range(8)]}


class StageOrderTests(unittest.TestCase):
    def instance(self,root):
        write(root/'allocation.json',allocation())
        return batch.Batch(root/'freeze.json')

    def test_reserve_is_persisted_before_launch_and_not_modulo(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(batch,'OUTPUT',Path(temp)):
            root=Path(temp);instance=self.instance(root)
            first=instance.reserve(7)
            self.assertEqual(first,{'layout':0,'gpu':7,'arm':'conditions'})
            entry=load(root/'treatment-mapping.json')['layouts']['0']
            self.assertEqual((entry['gpu'],entry['gpu_uuid'],entry['pod_uid']),(7,'GPU-7','pod'))
            instance.stop.set();self.assertIsNone(instance.reserve(1))

    def test_eight_initial_formal_lanes_then_dynamic_queue_and_full_stage_barrier(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(batch,'OUTPUT',Path(temp)):
            root=Path(temp);instance=self.instance(root);calls=[]
            def one(item):
                with instance.lock:
                    calls.append(dict(item))
                    if item['arm']=='conditions':
                        entry=load(root/'treatment-mapping.json')['layouts'][str(item['layout'])]
                        self.assertEqual(entry['gpu'],item['gpu'])
                        instance.mapping['layouts'][str(item['layout'])].update(treatment_task_wall_s=item['layout'],status='closed')
                return {'valid':True}
            with patch.object(instance,'one',side_effect=one),patch.object(batch,'verify_freeze'), \
                 patch.object(batch,'build_report',return_value={}),patch.object(instance,'scene_timing'):
                (root/'freeze.json').write_text('{}')
                self.assertEqual(instance.run(),0)
            self.assertEqual(len(calls),110)
            self.assertEqual([r['arm'] for r in calls[:55]],['conditions']*55)
            self.assertEqual([r['arm'] for r in calls[55:]],['control']*55)
            mapping=instance.mapping['layouts']
            for layout in range(8):self.assertEqual(mapping[str(layout)]['gpu'],layout)
            self.assertEqual(set(mapping),{str(i) for i in range(55)})
            for row in calls[55:]:self.assertEqual(row['gpu'],mapping[str(row['layout'])]['gpu'])
            for gpu in range(8):
                self.assertEqual([r['layout'] for r in calls[55:] if r['gpu']==gpu],
                    sorted([int(k) for k,v in mapping.items() if v['gpu']==gpu],reverse=True))

    def test_task_zero_does_not_stop_but_infra_failure_does(self):
        record={'valid':True,'success':False,'status':'completed','error':None,'video_check':{'valid':True}}
        self.assertFalse(batch.execution_problem(record))
        self.assertTrue(batch.execution_problem(dict(record,valid=False)))
        self.assertTrue(batch.execution_problem(dict(record,status='planner_error')))
        with tempfile.TemporaryDirectory() as temp,patch.object(batch,'OUTPUT',Path(temp)):
            instance=self.instance(Path(temp));calls=[]
            def stage(arm):
                calls.append(arm);instance.stop.set();return [record]*8
            with patch.object(instance,'stage',side_effect=stage):self.assertEqual(instance.run(),2)
            self.assertEqual(calls,['conditions'])

    def test_case_inspection_video_error_does_not_stop_next_assignment(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(batch,'OUTPUT',Path(temp)):
            root=Path(temp);instance=self.instance(root);instance.frozen.write_text('{}')
            item=instance.reserve(5)
            out=root/'full/conditions/layout_00/episode_001'
            write(out.parent/'launch_status.json',{'failure_scope':'case_execution','native_not_started':False})
            write(out/'initialization-receipt.json',{'native_client_cleanup_confirmed':True})
            def inspect(*a):raise ValueError('case inspection error')
            with patch.object(batch,'episode_dir',return_value=out),patch.object(batch,'verify_freeze'), \
                 patch.object(batch.subprocess,'run',return_value=type('Done',(),{'returncode':2})()), \
                 patch.object(batch,'inspect_episode',side_effect=inspect), \
                 patch.object(batch,'video_check',return_value={'valid':False,'videos':[]}):
                record=instance.one(item)
            self.assertTrue(record['execution_problem']);self.assertFalse(record['valid'])
            self.assertEqual(record['global_stop_reasons'],[]);self.assertFalse(instance.stop.is_set())
            self.assertEqual(instance.reserve(5)['layout'],1)
            write(out/'initialization-receipt.json',{'native_client_cleanup_confirmed':False})
            self.assertTrue(batch.cohort_problems(out))
            write(out/'initialization-receipt.json',{'native_client_cleanup_confirmed':True})
            write(out.parent/'launch_status.json',{'failure_scope':'global_invariant','native_not_started':True})
            self.assertTrue(batch.cohort_problems(out))

    def test_all_case_errors_still_attempt_both_full_stages(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(batch,'OUTPUT',Path(temp)):
            instance=self.instance(Path(temp));calls=[]
            def stage(arm):calls.append(arm);return [{'valid':False,'execution_problem':True}]*55
            with patch.object(instance,'stage',side_effect=stage):self.assertEqual(instance.run(),0)
            self.assertEqual(calls,['conditions','control'])

    def test_existing_measured_mapping_cannot_resume(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(batch,'OUTPUT',Path(temp)):
            root=Path(temp);instance=self.instance(root);instance.reserve(0)
            with self.assertRaises(FileExistsError):instance.run()


if __name__=='__main__':unittest.main()
