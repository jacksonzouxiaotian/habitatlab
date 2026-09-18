"""Recover the exact previous ten tasks; no new episode sampling."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from eagor_repro.config import load_config
from eagor_repro.evaluation.habitat_evaluator import build_habitat_config
from eagor_repro.spherical.rotation import habitat_rotation_world_from_eagor_body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', type=Path, default=Path('eagor_outputs/failure_attribution_10scenes_corrected_20260910'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = OmegaConf.merge(load_config('configs/eagor/base.yaml'), OmegaConf.load('configs/eagor/mp3d_failure_attribution.yaml'))
    hc = build_habitat_config(config)
    import habitat
    import quaternion
    from habitat_sim.utils.common import quat_from_coeffs
    dataset = habitat.make_dataset(hc.habitat.dataset.type, config=hc.habitat.dataset)
    lookup = {(Path(e.scene_id).stem, str(e.episode_id)): e for e in dataset.episodes}
    rows = []
    for path in sorted((args.old_root / 'A_eagor_direct_area/eagor/raw').glob('episode_*.csv')):
        with path.open() as f:
            row = next(csv.DictReader(f))
        ep = lookup[(Path(row['scene_id']).stem, row['episode_id'])]
        pos = [float(row['agent_position_'+a]) for a in 'xyz']
        np.testing.assert_allclose(pos, ep.start_position, atol=1e-5)
        rot = habitat_rotation_world_from_eagor_body(quat_from_coeffs(ep.start_rotation))
        forward = rot[:, 0]
        yaw = float(np.arctan2(-forward[0], -forward[2]))
        assert abs(np.arctan2(np.sin(yaw-float(row['agent_yaw'])), np.cos(yaw-float(row['agent_yaw'])))) < 1e-5
        # Dataset owns task truth; it is stored for audit, never handed to planner.
        goals = json.loads(dataset.to_json())['goals_by_category']
        goal_key = ep.goals_key
        payload = goals[goal_key]
        rows.append(dict(scene_id=ep.scene_id, episode_id=str(ep.episode_id),
            key=Path(ep.scene_id).stem+'_'+str(ep.episode_id), start_position=ep.start_position,
            start_rotation=ep.start_rotation, target_query=ep.object_category,
            goals_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            evaluation_only_goals=payload, source_csv=str(path),
            source_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if len(rows) != 10:
        raise RuntimeError(f'Expected 10 recovered tasks, found {len(rows)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(seed=int(config.seed), recovery='old step-0 position/yaw verified against original dataset full quaternion and goals; historical full configuration snapshot absent',
        protocol=OmegaConf.to_container(hc, resolve=True), base_config=OmegaConf.to_container(config, resolve=True), episodes=rows), indent=2))
    print(f'Verified and recovered {len(rows)} episodes: {args.output}')


if __name__ == '__main__':
    main()
