"""Actual Habitat 0.1.7 render/action smoke, NOT a navigation policy evaluation.

Fixed LEFT, RIGHT, FORWARD, STOP checks the interface only. Ground truth is
recorded by the evaluator and never used to select these diagnostic actions.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path('/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/reproduction')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['navid', 'uninavid'], default='navid')
    args = parser.parse_args()
    src = ROOT / 'sources' / 'NaVid-VLN-CE'
    sys.path.insert(0, str(src))
    os.chdir(src)
    import habitat
    import habitat_sim
    from habitat import Env
    from habitat.datasets import make_dataset
    from VLN_CE.vlnce_baselines.config.default import get_config
    import imageio.v2 as imageio
    config = get_config('VLN_CE/vlnce_baselines/config/r2r_baselines/%s_r2r.yaml' % args.method)
    config.defrost()
    tc = config.TASK_CONFIG
    data = ROOT.parent / 'data'
    tc.DATASET.DATA_PATH = str(data / 'datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz')
    tc.DATASET.SCENES_DIR = str(data / 'scene_datasets')
    tc.TASK.NDTW.GT_PATH = str(data / 'datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}_gt.json.gz')
    tc.TASK.NDTW.SPLIT = 'val_unseen'
    tc.SEED = 42
    tc.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.freeze()
    audit = sorted((ROOT / 'audits').glob('*/r2r_val_unseen_1_tasks.json'))[-1]
    target = str(json.loads(audit.read_text())['episodes'][0]['episode_id'])
    dataset = make_dataset(tc.DATASET.TYPE, config=tc.DATASET)
    dataset.episodes = [ep for ep in dataset.episodes if str(ep.episode_id) == target]
    assert len(dataset.episodes) == 1
    out = ROOT / 'smoke' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + args.method)
    out.mkdir(parents=True)
    (out / 'config.yaml').write_text(config.dump())
    def serial(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        return str(x)
    records = []
    frames = []
    with Env(tc, dataset=dataset) as env:
        obs = env.reset()
        for step, action in enumerate([None, 2, 3, 1, 0]):
            if action is not None:
                obs = env.step({'action': action})
            state = env.sim.get_agent_state()
            metrics = {k:v for k,v in env.get_metrics().items() if k != 'top_down_map_vlnce'}
            rgb = obs['rgb']
            frames.append(rgb)
            imageio.imwrite(out / ('rgb_%03d.png' % step), rgb)
            records.append({'step':step, 'action':action, 'position':state.position.tolist(), 'rotation_xyzw':[state.rotation.x,state.rotation.y,state.rotation.z,state.rotation.w], 'rgb_shape':list(rgb.shape), 'metrics':metrics})
        result = {'kind':'simulator_interface_diagnostic_not_model_episode', 'method_config':args.method, 'seed':42,
                  'habitat_path':habitat.__file__, 'habitat_sim_path':habitat_sim.__file__, 'episode_id':target,
                  'scene_id':env.current_episode.scene_id, 'instruction':env.current_episode.instruction.instruction_text,
                  'records':records, 'policy_receives_gt':False}
    delta = np.array(records[3]['position']) - np.array(records[2]['position'])
    result['measured_forward_m'] = float(np.linalg.norm(delta))
    result['turn_pair_returns_to_start_rotation'] = bool(np.allclose(records[0]['rotation_xyzw'], records[2]['rotation_xyzw'], atol=1e-5))
    (out / 'result.json').write_text(json.dumps(result, default=serial, ensure_ascii=False, indent=2))
    imageio.mimsave(out / 'interface_smoke.mp4', frames, fps=1)
    print('SIMULATOR_SMOKE', out, result['measured_forward_m'], result['turn_pair_returns_to_start_rotation'])

if __name__ == '__main__':
    main()
