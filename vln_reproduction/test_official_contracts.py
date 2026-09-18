"""Execute actual upstream method ASTs with synthetic observations for unit tests.

No model is mocked into a benchmark: these are parser/queue characterization
tests only. Known-bug tests describe upstream behavior, not model quality.
"""
import ast
from pathlib import Path
import random
import re
import subprocess
from types import SimpleNamespace
import unittest

SRC = Path('/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/reproduction/sources/NaVid-VLN-CE')

def method(file, cls, name, patched=False):
    source = (SRC / file).read_text() if patched else subprocess.check_output(['git', '-C', str(SRC), 'show', 'HEAD:' + file], text=True)
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'re': re, 'random': random}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SRC / file), 'exec'), namespace)
    return namespace[name]

def agent(file, name, output, patched=False):
    obj = SimpleNamespace(require_map=False, rgb_list=[], topdown_map_list=[],
                          pending_action_list=[], promt_template='{}',
                          predict_inference=lambda prompt: output)
    obj.act = lambda obs: method(file, name, 'act', patched)(obj, obs, {}, 'unit-test-only')
    obj.extract_result = lambda text: method(file, name, 'extract_result', patched)(obj, text)
    return obj

OBS = {'rgb': 'synthetic-observation-unit-test-only', 'instruction': {'text': 'test'}}

class OfficialContracts(unittest.TestCase):
    def test_patched_navid_bare_stop(self):
        self.assertEqual(agent('agent_navid.py', 'NaVid_Agent', 'stop', True).act(OBS)['action'], 0)

    def test_patched_navid_invalid_terminates_without_action(self):
        a = agent('agent_navid.py', 'NaVid_Agent', 'unparseable', True)
        with self.assertRaisesRegex(ValueError, 'invalid_model_output'):
            a.act(OBS)
        self.assertEqual(a.pending_action_list, [])

    def test_patched_valid_action_prefix_matches_upstream(self):
        for output in ['move forward 75 cm.', 'turn left 60 degree.', 'turn right 90 degree.']:
            before = agent('agent_navid.py', 'NaVid_Agent', output)
            after = agent('agent_navid.py', 'NaVid_Agent', output, True)
            for _ in range(6):
                self.assertEqual(before.act(OBS), after.act(OBS))

    def test_navid_parser_units(self):
        a = agent('agent_navid.py', 'NaVid_Agent', '')
        self.assertEqual(a.extract_result('move forward 75 cm.'), (1, 75.))
        self.assertEqual(a.extract_result('turn left 60 degree.'), (2, 60.))

    def test_navid_native_queue_preserves_25cm_primitives(self):
        a = agent('agent_navid.py', 'NaVid_Agent', 'move forward 75 cm.')
        self.assertEqual([a.act(OBS)['action'] for _ in range(3)], [1, 1, 1])
        self.assertEqual(len(a.rgb_list), 3)

    def test_navid_native_limit_three_primitives(self):
        a = agent('agent_navid.py', 'NaVid_Agent', 'turn right 180 degree.')
        self.assertEqual(a.act(OBS)['action'], 3)
        self.assertEqual(a.pending_action_list, [3, 3])

    def test_characterize_navid_bare_stop_truncation_bug(self):
        a = agent('agent_navid.py', 'NaVid_Agent', 'stop')
        self.assertEqual(a.extract_result('stop'), (0, None))
        self.assertIn(a.act(OBS)['action'], [1, 2, 3])

    def test_navid_punctuated_stop_survives(self):
        self.assertEqual(agent('agent_navid.py', 'NaVid_Agent', 'stop.').act(OBS)['action'], 0)

    def test_characterize_navid_invalid_random_motion(self):
        self.assertIn(agent('agent_navid.py', 'NaVid_Agent', 'unparseable').act(OBS)['action'], [1, 2, 3])

    def test_uninavid_native_executes_two_then_reobserves(self):
        a = agent('agent_uninavid.py', 'UniNaVid_Agent', 'forward left right stop')
        self.assertEqual(a.act(OBS)['action'], 1)
        self.assertEqual(a.pending_action_list, [2])
        self.assertEqual(a.act(OBS)['action'], 2)
        self.assertEqual(a.pending_action_list, [])
        self.assertEqual(len(a.rgb_list), 2)

    def test_uninavid_invalid_does_not_become_stop(self):
        a = agent('agent_uninavid.py', 'UniNaVid_Agent', 'invalid')
        with self.assertRaises(ValueError):
            a.act(OBS)

    def test_uninavid_reset_clears_all_online_caches(self):
        a = agent('agent_uninavid.py', 'UniNaVid_Agent', 'forward left')
        calls = []
        core = SimpleNamespace(initialize_online_inference_nav_feat_cache=lambda: calls.append('reset'), new_frames=7)
        a.model = SimpleNamespace(config=SimpleNamespace(run_type='train'), get_model=lambda: core)
        a.count_id = 1
        a.act(OBS)
        method('agent_uninavid.py', 'UniNaVid_Agent', 'reset')(a)
        self.assertEqual(a.pending_action_list, [])
        self.assertEqual(a.rgb_list, [])
        self.assertEqual(core.new_frames, 0)
        self.assertEqual(a.model.config.run_type, 'eval')
        self.assertEqual(calls, ['reset'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
