#!/usr/bin/env python3
"""Mine narrow-passage PointNav anchors from real HM3D scenes.

For each scene, samples (start, goal) pairs on the navmesh, keeps ones whose
geodesic path is short and reasonably straight (so the existing
NarrowPassageNav-v0 heading/lateral measures, which score against the
straight start->goal line, stay meaningful), then walks the path and queries
PathFinder.distance_to_closest_obstacle to find the tightest point.

Output is an anchors CSV in the same schema generate_habitat_episodes.py
already consumes (see examples/narrow_passage_rl/HABITAT_INTEGRATION.md) --
this script does not write the final dataset .json.gz itself.

Run inside the `habitat` conda env (needs habitat_sim).
"""

import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional, Tuple

import habitat_sim
import numpy as np

# distance_to_closest_obstacle returns the raw distance from a path point
# (treated as zero-radius) to the nearest obstacle surface. Assuming the
# geodesic path roughly hugs the corridor centerline, that's an estimate of
# half the passage width. Subtracting the robot radius converts it to
# "body_margin": clearance between the robot's body edge and the wall, the
# same convention used by examples/narrow_passage_rl/procedural_env.py and
# narrow_passage_fix_files/.../params.yaml on the real robot.
DIFFICULTY_BOUNDS = {
    "narrow": (0.0, 0.15),
    "normal": (0.15, 0.40),
    "wide": (0.40, float("inf")),
}
BUCKET_TARGET_FRACTIONS = {"narrow": 0.5, "normal": 0.35, "wide": 0.15}


def classify(body_margin: float) -> Optional[str]:
    for name, (lo, hi) in DIFFICULTY_BOUNDS.items():
        if lo <= body_margin < hi:
            return name
    return None


def resample_path(points, step: float) -> Tuple[List[np.ndarray], List[float]]:
    pts = [np.array(p, dtype=np.float64) for p in points]
    seg_vecs = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
    seg_lens = [float(np.linalg.norm(v)) for v in seg_vecs]
    total = sum(seg_lens)
    if total < 1e-6:
        return [pts[0]], [0.0]

    samples, arclens = [], []
    dist = 0.0
    seg_idx = 0
    seg_progress = 0.0
    while dist <= total:
        while seg_idx < len(seg_lens) - 1 and seg_progress > seg_lens[seg_idx]:
            seg_progress -= seg_lens[seg_idx]
            seg_idx += 1
        seg_len = seg_lens[seg_idx]
        t = 0.0 if seg_len < 1e-9 else min(1.0, seg_progress / seg_len)
        samples.append(pts[seg_idx] + seg_vecs[seg_idx] * t)
        arclens.append(dist)
        dist += step
        seg_progress += step
    samples.append(pts[-1])
    arclens.append(total)
    return samples, arclens


def path_tangent(samples: List[np.ndarray], idx: int) -> np.ndarray:
    lo = max(0, idx - 1)
    hi = min(len(samples) - 1, idx + 1)
    vec = samples[hi] - samples[lo]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return vec / norm


def mine_scene(
    sim: "habitat_sim.Simulator",
    scene_id_rel: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> List[dict]:
    pf = sim.pathfinder
    if not pf.is_loaded:
        return []

    candidates: List[dict] = []
    attempts = 0
    while attempts < args.attempts_per_scene and len(candidates) < args.max_per_scene:
        attempts += 1
        p1 = pf.get_random_navigable_point()
        p2 = pf.get_random_navigable_point()
        if pf.get_island(p1) != pf.get_island(p2):
            continue

        sp = habitat_sim.ShortestPath()
        sp.requested_start = p1
        sp.requested_end = p2
        if not pf.find_path(sp) or len(sp.points) < 2:
            continue

        geo = sp.geodesic_distance
        if geo < args.min_geodesic or geo > args.max_geodesic:
            continue
        euclid = float(np.linalg.norm(np.array(p1) - np.array(p2)))
        if euclid / geo < args.min_straightness:
            continue

        samples, arclens = resample_path(sp.points, args.step)
        trim = args.trim
        profile = [
            (i, s, pf.distance_to_closest_obstacle(samples[i], args.max_search_radius))
            for i, s in enumerate(arclens)
            if trim <= s <= geo - trim
        ]
        if len(profile) < 3:
            continue

        idx, min_s, point_clearance = min(profile, key=lambda x: x[2])
        body_margin = point_clearance - args.robot_radius
        difficulty = classify(body_margin)
        if difficulty is None or args.bucket_quota.get(difficulty, 0) <= 0:
            continue

        thresh = point_clearance * args.passage_length_factor
        in_band = [s for _, s, c in profile if c <= thresh]
        passage_length = max(0.3, max(in_band) - min(in_band)) if in_band else 0.3
        direction = path_tangent(samples, idx)

        candidates.append(
            {
                "scene_id": scene_id_rel,
                "start": samples[0],
                "goal": samples[-1],
                "passage_center": samples[idx],
                "passage_direction": direction,
                "passage_width": round(2.0 * point_clearance, 4),
                "passage_length": round(passage_length, 4),
                "difficulty": difficulty,
                "body_margin": round(body_margin, 4),
                "min_clearance": round(point_clearance, 4),
            }
        )
        args.bucket_quota[difficulty] -= 1

    return candidates


def make_sim(scene_path: str) -> "habitat_sim.Simulator":
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = False
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = []
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def write_anchors_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_id", "scene_id",
        "start_x", "start_y", "start_z",
        "goal_x", "goal_y", "goal_z",
        "passage_x", "passage_y", "passage_z",
        "passage_direction", "passage_width", "passage_length",
        "difficulty", "false_feasible", "body_margin",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            start, goal, center = row["start"], row["goal"], row["passage_center"]
            direction = row["passage_direction"]
            writer.writerow(
                {
                    "episode_id": f"hm3d_narrow_{i:06d}",
                    "scene_id": row["scene_id"],
                    "start_x": start[0], "start_y": start[1], "start_z": start[2],
                    "goal_x": goal[0], "goal_y": goal[1], "goal_z": goal[2],
                    "passage_x": center[0], "passage_y": center[1], "passage_z": center[2],
                    "passage_direction": f"{direction[0]},{direction[1]},{direction[2]}",
                    "passage_width": row["passage_width"],
                    "passage_length": row["passage_length"],
                    "difficulty": row["difficulty"],
                    "false_feasible": 0,
                    "body_margin": row["body_margin"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-dir", type=Path,
                         default=Path("data/scene_datasets/hm3d/val"))
    parser.add_argument("--num-scenes", type=int, default=0,
                         help="0 = use all scenes found under --scenes-dir")
    parser.add_argument("--train-frac", type=float, default=0.8,
                         help="Fraction of scenes used for the train anchors file")
    parser.add_argument("--target-episodes", type=int, default=200)
    parser.add_argument("--max-per-scene", type=int, default=6)
    parser.add_argument("--attempts-per-scene", type=int, default=300)
    parser.add_argument("--robot-radius", type=float, default=0.21)
    parser.add_argument("--step", type=float, default=0.12)
    parser.add_argument("--trim", type=float, default=0.3)
    parser.add_argument("--min-geodesic", type=float, default=1.5)
    parser.add_argument("--max-geodesic", type=float, default=6.0)
    parser.add_argument("--min-straightness", type=float, default=0.8)
    parser.add_argument("--max-search-radius", type=float, default=2.0)
    parser.add_argument("--passage-length-factor", type=float, default=1.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-train", type=Path,
                         default=Path("data/datasets/narrow_passage/anchors_train.csv"))
    parser.add_argument("--out-val", type=Path,
                         default=Path("data/datasets/narrow_passage/anchors_val.csv"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    scene_paths = sorted(args.scenes_dir.glob("*/*.basis.glb"))
    if not scene_paths:
        raise SystemExit(f"No scenes found under {args.scenes_dir}")
    rng.shuffle(scene_paths)
    if args.num_scenes > 0:
        scene_paths = scene_paths[: args.num_scenes]

    split_idx = max(1, int(len(scene_paths) * args.train_frac))
    scene_splits = {"train": scene_paths[:split_idx], "val": scene_paths[split_idx:]}

    for split, scenes in scene_splits.items():
        target = int(args.target_episodes * (args.train_frac if split == "train" else 1 - args.train_frac))
        bucket_quota = {
            name: max(1, int(target * frac)) for name, frac in BUCKET_TARGET_FRACTIONS.items()
        }
        args.bucket_quota = bucket_quota
        rows: List[dict] = []
        counts = {"narrow": 0, "normal": 0, "wide": 0}
        for scene_path in scenes:
            if sum(bucket_quota.values()) <= 0:
                break
            scene_id_rel = str(scene_path)
            sim = make_sim(str(scene_path))
            try:
                found = mine_scene(sim, scene_id_rel, args, rng)
            finally:
                sim.close()
            rows.extend(found)
            for r in found:
                counts[r["difficulty"]] += 1
            print(f"[{split}] {scene_path.name}: +{len(found)} "
                  f"(remaining quota narrow={bucket_quota['narrow']} "
                  f"normal={bucket_quota['normal']} wide={bucket_quota['wide']})")

        out_path = args.out_train if split == "train" else args.out_val
        write_anchors_csv(out_path, rows)
        print(f"== {split}: {len(rows)} episodes written to {out_path} "
              f"(narrow={counts['narrow']} normal={counts['normal']} wide={counts['wide']}) ==")


if __name__ == "__main__":
    main()
