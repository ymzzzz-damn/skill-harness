import os
from pathlib import Path
import tempfile
import unittest
from warm_cache import prewarm,tree_manifest


class WarmCacheTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'reviewed';(source/'bin').mkdir(parents=True)
        (source/'bin/plugin.so').write_bytes(b'fixed-version-plugin')
        (source/'extension.toml').write_text('version="2.4.31"')
        return source,tree_manifest(source)['tree_sha256']

    def test_verified_copy_all_eight_entries_and_preserves_valid_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,digest=self.fixture(root)
            receipt=prewarm(source,root/'runtime',root/'output',digest,'version')
            self.assertTrue(receipt['copied']);self.assertEqual(len(receipt['lanes']),8)
            for entry in receipt['lanes']:
                self.assertTrue(Path(entry['path']).is_symlink())
                self.assertEqual(tree_manifest(Path(entry['path']))['tree_sha256'],digest)
            again=prewarm(source,root/'runtime',root/'output',digest,'version')
            self.assertFalse(again['copied'])
            self.assertTrue(all(e['action']=='existing_verified_entry_preserved' for e in again['lanes']))

    def test_mismatch_broken_entry_and_started_episode_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,digest=self.fixture(root)
            with self.assertRaises(RuntimeError):prewarm(source,root/'runtime',root/'output','0'*64,'version')
            entry=root/'runtime/lanes/0/kit-portable/data/Kit/Isaac-Sim/5.1/exts/3/version'
            entry.parent.mkdir(parents=True);entry.symlink_to(root/'missing')
            with self.assertRaises(RuntimeError):prewarm(source,root/'runtime',root/'output',digest,'version')
            self.assertTrue(os.path.lexists(entry));self.assertEqual(os.readlink(entry),str(root/'missing'))
            (root/'output/full/conditions/layout_00/episode_001').mkdir(parents=True)
            with self.assertRaises(RuntimeError):prewarm(source,root/'runtime',root/'output',digest,'version')


if __name__=='__main__':unittest.main()
