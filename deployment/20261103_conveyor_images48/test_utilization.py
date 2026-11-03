from pathlib import Path
import tempfile
import unittest
from protocol import load,write
from utilization import Monitor,enrich,parse_gpu_csv,read_cgroup


class UtilizationTests(unittest.TestCase):
    def test_csv_and_honest_worker_vs_compute_metrics(self):
        rows=parse_gpu_csv('0, GPU-a, 0, 12000, 32768\n1, GPU-b, 50, 4000, 32768')
        allocation={'devices':[{'uuid':'GPU-a','cuda_local_index':0},{'uuid':'GPU-b','cuda_local_index':1}]}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write(root/'worker-live.json',{'stage':'conditions','workers':{'0':{'status':'running','layout':9}}})
            out=root/'full/conditions/layout_09/episode_001'
            write(out/'initialization-receipt.json',{'native_boot_started':True,'first_rgb_seen':True,'model_phase_entered':True,'native_action_requests':8})
            write(out/'activity.json',{'phase':'native_control','at':'time'})
            monitor=Monitor(allocation,root,query=lambda:rows,cgroup=lambda:{'memory.current':95,'memory.max':100,'cpu.stat':{'nr_throttled':2},'memory.events':{'oom':0}})
            monitor.once();monitor.once();monitor.close()
            latest=load(root/'utilization/latest.json')
            self.assertEqual(latest['assigned_worker_count'],1)
            self.assertEqual(latest['devices'][0]['runtime_phase'],'native_control')
            self.assertTrue(latest['memory_pressure_above90percent'])
            self.assertEqual(latest['cgroup_counter_deltas']['cpu.stat']['nr_throttled'],0)
            group=next(iter(load(root/'utilization/summary.json')['hourly'].values()))
            self.assertEqual(group['gpus']['GPU-a']['worker_assigned_fraction'],1)
            self.assertEqual(group['gpus']['GPU-a']['gpu_active_fraction_util_gt0'],0)
            self.assertEqual(group['gpus']['GPU-b']['gpu_active_fraction_util_gt0'],1)
            self.assertEqual(group['mean_assigned_worker_count'],1)
            with self.assertRaises(RuntimeError):enrich(rows[:1],allocation,{})

    def test_metadata_error_is_logged_and_cgroup_limits_read_without_compute(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'cpu.stat').write_text('nr_throttled 4\nthrottled_usec 99\n')
            (root/'memory.events').write_text('oom 2\noom_kill 1\n')
            (root/'memory.max').write_text('max')
            self.assertEqual(read_cgroup(root)['memory.events']['oom_kill'],1)
            self.assertEqual(read_cgroup(root)['memory.max'],'max')
            def fail():raise RuntimeError('metadata unavailable')
            monitor=Monitor({'devices':[]},root,query=fail);monitor.once();monitor.close()
            self.assertEqual(load(root/'utilization/summary.json')['sampling_errors'],1)
            self.assertIn('metadata unavailable',load(root/'utilization/latest.json')['error'])

    def test_na_fields_are_independent_missing_values_not_zero(self):
        rows=parse_gpu_csv('0, GPU-a, N/A, 12000, 32768\n1, GPU-b, 0, [Not Supported], N/A\n')
        self.assertIsNone(rows[0]['gpu_utilization_percent'])
        self.assertEqual(rows[0]['memory_used_mib'],12000)
        self.assertEqual(rows[0]['memory_total_mib'],32768)
        self.assertEqual(rows[1]['gpu_utilization_percent'],0)
        self.assertIsNone(rows[1]['memory_used_mib'])
        self.assertIsNone(rows[1]['memory_total_mib'])

    def test_missing_utilization_keeps_other_gpu_cgroup_and_worker_association(self):
        allocation={'devices':[{'uuid':'GPU-a','cuda_local_index':0},{'uuid':'GPU-b','cuda_local_index':1}]}
        rows=parse_gpu_csv('0, GPU-a, N/A, 12000, 32768\n1, GPU-b, 50, 4000, 32768')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            write(root/'worker-live.json',{'stage':'conditions','workers':{'0':{'status':'running','layout':9}}})
            out=root/'full/conditions/layout_09/episode_001'
            write(out/'initialization-receipt.json',{'model_phase_entered':True,'native_action_requests':8})
            write(out/'activity.json',{'phase':'agent_or_non_native_tool'})
            group={'memory.current':95,'memory.max':100,'cpu.stat':{'nr_throttled':2},'memory.events':{'oom_kill':0}}
            monitor=Monitor(allocation,root,query=lambda:rows,cgroup=lambda:group)
            monitor.once();monitor.close()
            sample=load(root/'utilization/latest.json')
            self.assertNotIn('error',sample)
            self.assertEqual(sample['cgroup'],group)
            self.assertIsNone(sample['devices'][0]['gpu_utilization_percent'])
            self.assertEqual(sample['devices'][0]['runtime_phase'],'agent_or_non_native_tool')
            self.assertEqual(sample['devices'][0]['case'],9)
            self.assertEqual(sample['devices'][1]['gpu_utilization_percent'],50)
            self.assertEqual(sample['assigned_worker_count'],1)
            self.assertEqual(sample['gpu_utilization_valid_devices'],1)
            self.assertEqual(sample['gpu_utilization_missing_devices'],1)
            self.assertEqual(sample['gpu_busy_count_util_ge10'],1)
            self.assertEqual(load(root/'utilization/summary.json')['sampling_errors'],0)

    def test_hourly_each_metric_uses_its_own_valid_denominator(self):
        allocation={'devices':[{'uuid':'GPU-a','cuda_local_index':0}]}
        packets=iter([parse_gpu_csv('0, GPU-a, 20, N/A, 32768'),
                      parse_gpu_csv('0, GPU-a, N/A, 4000, N/A'),
                      parse_gpu_csv('0, GPU-a, 0, 0, 32768')])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            monitor=Monitor(allocation,root,query=lambda:next(packets),cgroup=lambda:{})
            for _ in range(3):monitor.once()
            monitor.close()
            hourly=next(iter(load(root/'utilization/summary.json')['hourly'].values()))['gpus']['GPU-a']
            self.assertEqual(hourly['samples'],3)
            self.assertEqual(hourly['gpu_utilization_valid_samples'],2)
            self.assertEqual(hourly['gpu_utilization_missing_samples'],1)
            self.assertEqual(hourly['memory_used_valid_samples'],2)
            self.assertEqual(hourly['memory_used_missing_samples'],1)
            self.assertEqual(hourly['mean_gpu_utilization_percent'],10)
            self.assertEqual(hourly['mean_memory_used_mib'],2000)
            self.assertEqual(hourly['gpu_active_fraction_util_gt0'],.5)
            self.assertEqual(hourly['gpu_busy_fraction_util_ge10'],.5)

    def test_all_missing_fields_report_null_aggregates_and_preserve_cgroup_deltas(self):
        allocation={'devices':[{'uuid':'GPU-a','cuda_local_index':0}]}
        packets=iter([{'memory.current':90,'cpu.stat':{'nr_throttled':2},'memory.events':{'oom_kill':0}},
                      {'memory.current':92,'cpu.stat':{'nr_throttled':5},'memory.events':{'oom_kill':1}}])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            monitor=Monitor(allocation,root,query=lambda:parse_gpu_csv('0, GPU-a, N/A, N/A, N/A'),
                            cgroup=lambda:next(packets))
            monitor.once();monitor.once();monitor.close()
            sample=load(root/'utilization/latest.json')
            self.assertIsNone(sample['gpu_busy_count_util_ge10'])
            self.assertEqual(sample['gpu_utilization_valid_devices'],0)
            self.assertEqual(sample['gpu_utilization_missing_devices'],1)
            self.assertEqual(sample['cgroup_counter_deltas']['cpu.stat']['nr_throttled'],3)
            self.assertEqual(sample['cgroup_counter_deltas']['memory.events']['oom_kill'],1)
            result=load(root/'utilization/summary.json')
            self.assertEqual(result['samples'],2)
            self.assertEqual(result['sampling_errors'],0)
            hourly=next(iter(result['hourly'].values()))['gpus']['GPU-a']
            for key in ('mean_gpu_utilization_percent','mean_memory_used_mib',
                        'gpu_active_fraction_util_gt0','gpu_busy_fraction_util_ge10'):
                self.assertIsNone(hourly[key])
            self.assertEqual(hourly['gpu_utilization_valid_samples'],0)
            self.assertEqual(hourly['worker_assigned_fraction'],0)


if __name__=='__main__':unittest.main()
