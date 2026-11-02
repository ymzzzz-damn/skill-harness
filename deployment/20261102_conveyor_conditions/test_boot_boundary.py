from pathlib import Path
import tempfile
import unittest
from boot_recovery import archive_unstarted_boot, boot_retry_allowed
from native_startup import retry_decision
from protocol import load, write


class BootBoundaryTests(unittest.TestCase):
    def fixture(self,out):
        write(out/'result.json',{'status':'boot_error','env_steps':0,'tool_turns':0,'generation':None})
        write(out/'initialization-receipt.json',{'retry_eligible':True,
            'first_observation_seen':False,'first_rgb_seen':False,'model_phase_entered':False,
            'native_action_requests':0,'native_client_cleanup_confirmed':True})

    def test_observation_model_action_and_unknown_error_prevent_retry(self):
        state={'native_boot_error':{'retryable_initialization_error':True},'native_boot_started':True,
            'first_observation_seen':False,'first_rgb_seen':False,'native_boot_completed':False,
            'model_phase_entered':False}
        self.assertTrue(retry_decision(state,{},True,'boot_error'))
        for key in ('first_observation_seen','first_rgb_seen','native_boot_completed','model_phase_entered'):
            changed=dict(state,**{key:True})
            self.assertFalse(retry_decision(changed,{},True,'boot_error'))
        self.assertFalse(retry_decision(state,{'actions':1},True,'boot_error'))
        self.assertFalse(retry_decision(state,{},False,'boot_error'))
        self.assertFalse(retry_decision(state,{},True,'completed'))

    def test_archive_retains_every_byte_and_measured_episode_never_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'layout_03/episode_001'
            self.fixture(out)
            (out/'native.log').write_text('retained error evidence\n')
            self.assertTrue(boot_retry_allowed(out))
            receipt=archive_unstarted_boot(out,1,35)
            archive=Path(receipt['archive_directory'])
            self.assertEqual((archive/'native.log').read_text(),'retained error evidence\n')
            self.assertFalse(out.exists())
            self.assertEqual(len(load(out.parent/'boot-attempts.json')),1)
            self.fixture(out)
            (out/'steps/0000').mkdir(parents=True)
            self.assertFalse(boot_retry_allowed(out))
            with self.assertRaises(RuntimeError):archive_unstarted_boot(out,2,35)


if __name__=='__main__':unittest.main()
