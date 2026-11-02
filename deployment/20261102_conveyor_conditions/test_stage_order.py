from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import batch


class StageOrderTests(unittest.TestCase):
    def rows(self):
        return [{'arm':'conditions','layout':layout,'valid':True,'status':'completed',
                 'contract_registered':True,'condition_guard_evidence_valid':True,'task_timing_valid':True,
                 'success':False,'video_check':{'valid':True}} for layout in (3,21)]

    def test_smoke_checks_evidence_never_success(self):
        self.assertTrue(batch.smoke_qualifies(self.rows()))
        rows=self.rows();rows[0]['video_check']['valid']=False
        self.assertFalse(batch.smoke_qualifies(rows))

    def test_all_treatment_stage_returns_before_control_starts(self):
        with tempfile.TemporaryDirectory() as temp:
            instance=batch.Batch(Path(temp)/'freeze.json')
            order=[]
            def stage(phase,arm):
                order.append((phase,arm))
                return self.rows()
            with patch.object(batch,'OUTPUT',Path(temp)),patch.object(instance,'stage',side_effect=stage):
                self.assertEqual(instance.run(),0)
            self.assertEqual(order,[('smoke','conditions'),('full','conditions'),('full','control')])

    def test_failed_smoke_starts_neither_full_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            instance=batch.Batch(Path(temp)/'freeze.json')
            rows=self.rows();rows[0]['valid']=False
            with patch.object(batch,'OUTPUT',Path(temp)),patch.object(instance,'stage',return_value=rows) as stage:
                self.assertEqual(instance.run(),2)
            stage.assert_called_once_with('smoke','conditions')


if __name__=='__main__':unittest.main()
