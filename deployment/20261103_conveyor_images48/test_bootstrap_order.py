import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import bootstrap


class BootstrapOrderTests(unittest.TestCase):
    def test_cpu_cache_and_tests_then_fresh_guard_then_cuda_and_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);freeze=root/'freeze.json'
            freeze.write_text(json.dumps({'sources':{'sam3':{'root':'/sam3'}}}))
            order=[]
            class Monitor:
                def __init__(self,allocation):order.append('monitor_created')
                def start(self):order.append('monitor_start');return self
                def close(self):order.append('monitor_close')
            def run(cmd,*a):order.append('batch' if 'batch.py' in cmd[3] else 'cpu')
            with patch.object(bootstrap,'OUTPUT',root/'output'),patch.object(bootstrap,'RUNTIME',root/'runtime'), \
                 patch.object(bootstrap,'verify_freeze',side_effect=lambda f:order.append('freeze')), \
                 patch.object(bootstrap,'own_pod',side_effect=lambda:order.append('pod') or {}), \
                 patch.object(bootstrap,'prewarm',side_effect=lambda:order.append('cache')), \
                 patch.object(bootstrap,'sha',return_value=bootstrap.OPENCV_SHA256), \
                 patch.object(bootstrap,'run',side_effect=run), \
                 patch.object(bootstrap,'request_health',side_effect=lambda *a,**k:order.append('request') or {}), \
                 patch.object(bootstrap,'wait_for_health',side_effect=lambda *a,**k:order.append('healthy') or {}), \
                 patch.object(bootstrap,'gpu_inventory',side_effect=lambda:order.append('nvml') or []), \
                 patch.object(bootstrap,'cuda_mapping',side_effect=lambda *a:order.append('cuda') or {'devices':[]}), \
                 patch.object(bootstrap,'Monitor',Monitor), \
                 patch('sys.argv',['bootstrap.py','--freeze',str(freeze)]):
                self.assertEqual(bootstrap.main(),0)
            self.assertLess(order.index('pod'),order.index('cache'))
            self.assertLess(max(i for i,x in enumerate(order) if x=='cpu'),order.index('request'))
            self.assertEqual(order.count('request'),1);self.assertEqual(order.count('healthy'),1)
            self.assertLess(order.index('healthy'),order.index('cuda'))
            self.assertLess(order.index('cuda'),order.index('batch'))
            self.assertEqual(order[-1],'monitor_close')


if __name__=='__main__':unittest.main()
