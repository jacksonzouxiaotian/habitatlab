from pathlib import Path
import tempfile
import unittest
from prepare_migration import validate_output, is_inside
from verify_migration import remap
from restore_external_sources import target_for

class SafeMigration(unittest.TestCase):
    def test_output_inside_source_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):validate_output(Path(t)/'backup',{'src':Path(t)})

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):validate_output(Path(t),{})

    def test_symlink_recursion_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); (root/'src').mkdir(); (root/'alias').symlink_to(root/'src',target_is_directory=True)
            with self.assertRaises(ValueError):validate_output(root/'alias'/'out',{'src':root/'src'})

    def test_new_external_output_allowed(self):
        with tempfile.TemporaryDirectory() as t:
            validate_output(Path(t)/'out',{'src':Path(t)/'source'})

    def test_remap_uses_most_specific_source(self):
        sources={'root':'/old', 'data':'/old/data'}
        self.assertEqual(remap('/old/data/scene/file', sources, Path('/new/files')),
                         Path('/new/files/data/scene/file'))

    def test_remap_rejects_untracked_root(self):
        with self.assertRaises(ValueError):
            remap('/old_other/file', {'root':'/old'}, Path('/new/files'))

    def test_external_nested_source_target(self):
        self.assertEqual(target_for({'path':'/home/xiaotian/vla/Open-Nav/SpatialBot_repo'}, Path('/new')),
                         Path('/new/vla/Open-Nav/SpatialBot_repo'))

    def test_external_model_source_target(self):
        source='/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/reproduction/sources/NaVid-VLN-CE'
        self.assertEqual(target_for({'path':source}, Path('/new')),
                         Path('/new/vln_sources/NaVid-VLN-CE'))

    def test_external_unknown_source_rejected(self):
        with self.assertRaises(ValueError):
            target_for({'path':'/untracked/source'}, Path('/new'))

if __name__=='__main__':unittest.main(verbosity=2)
