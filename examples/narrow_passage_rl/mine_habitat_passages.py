#!/usr/bin/env python3
"""Mine narrow-passage PointNav anchors from real scanned Habitat scenes.

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
import gzip
import json
import math
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
    "false_feasible": (-float("inf"), 0.0),
    "narrow": (0.0, 0.15),
    "normal": (0.15, 0.40),
    "wide": (0.40, float("inf")),
}
BUCKET_TARGET_FRACTIONS = {"narrow": 0.5, "normal": 0.35, "wide": 0.15}


def allocate_exact_quotas(target: int, fractions: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation whose bucket counts sum exactly to target."""
    raw = {name: target * fraction for name, fraction in fractions.items()}
    quotas = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        fractions,
        key=lambda name: (raw[name] - quotas[name], name),
        reverse=True,
    )
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


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
                "robot_radius": float(args.robot_radius),
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


def write_anchors_csv(
    path: Path,
    rows: List[dict],
    episode_prefix: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_id", "scene_id",
        "start_x", "start_y", "start_z",
        "goal_x", "goal_y", "goal_z",
        "passage_x", "passage_y", "passage_z",
        "passage_direction", "passage_width", "passage_length",
        "difficulty", "false_feasible", "body_margin", "robot_radius",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            start, goal, center = row["start"], row["goal"], row["passage_center"]
            direction = row["passage_direction"]
            writer.writerow(
                {
                    "episode_id": f"{episode_prefix}_{i:06d}",
                    "scene_id": row["scene_id"],
                    "start_x": start[0], "start_y": start[1], "start_z": start[2],
                    "goal_x": goal[0], "goal_y": goal[1], "goal_z": goal[2],
                    "passage_x": center[0], "passage_y": center[1], "passage_z": center[2],
                    "passage_direction": f"{direction[0]},{direction[1]},{direction[2]}",
                    "passage_width": row["passage_width"],
                    "passage_length": row["passage_length"],
                    "difficulty": row["difficulty"],
                    "false_feasible": int(
                        row["difficulty"] == "false_feasible"
                    ),
                    "body_margin": row["body_margin"],
                    "robot_radius": row["robot_radius"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-dir", type=Path,
                         default=Path("data/scene_datasets/hm3d/val"))
    parser.add_argument(
        "--scene-glob",
        default="*/*.basis.glb",
        help=(
            "Scene pattern below --scenes-dir. Use '*/*.glb' for MP3D and "
            "'*/*.basis.glb' for HM3D."
        ),
    )
    parser.add_argument(
        "--episode-prefix",
        default="hm3d_narrow",
        help="Prefix used for stable episode IDs written to both splits.",
    )
    parser.add_argument("--num-scenes", type=int, default=0,
                         help="0 = use all scenes found under --scenes-dir")
    parser.add_argument("--train-frac", type=float, default=0.8,
                         help="Fraction of scenes used for the train anchors file")
    parser.add_argument(
        "--train-scene-content-dir",
        type=Path,
        default=None,
        help=(
            "Optional official PointNav content directory. When paired with "
            "--val-scene-content-dir, its *.json.gz stems define the train "
            "scene split instead of randomly splitting scenes."
        ),
    )
    parser.add_argument(
        "--val-scene-content-dir",
        type=Path,
        default=None,
        help="Official PointNav validation content directory paired with train.",
    )
    parser.add_argument(
        "--train-scene-dataset",
        type=Path,
        default=None,
        help="Optional PointNav .json.gz whose episodes define train scene IDs.",
    )
    parser.add_argument(
        "--val-scene-dataset",
        type=Path,
        default=None,
        help="Optional PointNav .json.gz whose episodes define validation scene IDs.",
    )
    parser.add_argument("--target-episodes", type=int, default=200)
    parser.add_argument(
        "--train-target-episodes",
        type=int,
        default=None,
        help="Explicit train quota; requires --val-target-episodes.",
    )
    parser.add_argument(
        "--val-target-episodes",
        type=int,
        default=None,
        help="Explicit validation quota; requires --train-target-episodes.",
    )
    parser.add_argument("--max-per-scene", type=int, default=6)
    parser.add_argument("--attempts-per-scene", type=int, default=300)
    parser.add_argument("--robot-radius", type=float, default=0.21)
    parser.add_argument(
        "--false-feasible-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of anchors whose measured point clearance is below "
            "--robot-radius. These remain point-navmesh reachable but are "
            "labelled morphology-infeasible."
        ),
    )
    parser.add_argument("--step", type=float, default=0.12)
    parser.add_argument("--trim", type=float, default=0.3)
    parser.add_argument("--min-geodesic", type=float, default=1.5)
    parser.add_argument("--max-geodesic", type=float, default=6.0)
    parser.add_argument("--min-straightness", type=float, default=0.8)
    parser.add_argument("--max-search-radius", type=float, default=2.0)
    parser.add_argument("--passage-length-factor", type=float, default=1.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require-target",
        action="store_true",
        help="Exit non-zero unless every split reaches its exact target quota.",
    )
    parser.add_argument("--out-train", type=Path,
                         default=Path("data/datasets/narrow_passage/anchors_train.csv"))
    parser.add_argument("--out-val", type=Path,
                         default=Path("data/datasets/narrow_passage/anchors_val.csv"))
    args = parser.parse_args()
    if not 0.0 <= args.false_feasible_fraction < 1.0:
        raise ValueError("--false-feasible-fraction must be in [0, 1)")

    rng = random.Random(args.seed)
    scene_paths = sorted(args.scenes_dir.glob(args.scene_glob))
    if not scene_paths:
        raise SystemExit(f"No scenes found under {args.scenes_dir}")
    official_split_requested = (
        args.train_scene_content_dir is not None
        or args.val_scene_content_dir is not None
        or args.train_scene_dataset is not None
        or args.val_scene_dataset is not None
    )
    if official_split_requested:
        train_sources = sum(
            value is not None
            for value in (args.train_scene_content_dir, args.train_scene_dataset)
        )
        val_sources = sum(
            value is not None
            for value in (args.val_scene_content_dir, args.val_scene_dataset)
        )
        if train_sources != 1 or val_sources != 1:
            raise ValueError(
                "provide exactly one content directory or dataset file per split"
            )

        def content_scene_ids(directory: Path) -> set[str]:
            if not directory.is_dir():
                raise ValueError(f"missing PointNav content directory: {directory}")
            # Path.stem removes .gz and the second stem removes .json.
            return {path.stem.rsplit(".json", 1)[0] for path in directory.glob("*.json.gz")}

        def dataset_scene_ids(path: Path) -> set[str]:
            if not path.is_file():
                raise ValueError(f"missing PointNav dataset: {path}")
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                episodes = json.load(handle).get("episodes", [])
            ids = {
                Path(str(episode["scene_id"])).stem
                for episode in episodes
            }
            if not ids:
                raise ValueError(f"PointNav dataset contains no scene IDs: {path}")
            return ids

        split_ids = {
            "train": content_scene_ids(args.train_scene_content_dir)
            if args.train_scene_content_dir is not None
            else dataset_scene_ids(args.train_scene_dataset),
            "val": content_scene_ids(args.val_scene_content_dir)
            if args.val_scene_content_dir is not None
            else dataset_scene_ids(args.val_scene_dataset),
        }
        scene_by_id = {path.parent.name: path for path in scene_paths}
        missing = {
            split: sorted(ids - scene_by_id.keys())
            for split, ids in split_ids.items()
            if ids - scene_by_id.keys()
        }
        if missing:
            raise ValueError(f"scene assets missing for official split: {missing}")
        overlap = split_ids["train"] & split_ids["val"]
        if overlap:
            raise ValueError(f"official train/val split overlaps: {sorted(overlap)}")
        scene_splits = {
            split: [scene_by_id[name] for name in sorted(ids)]
            for split, ids in split_ids.items()
        }
        for split in scene_splits:
            rng.shuffle(scene_splits[split])
            if args.num_scenes > 0:
                scene_splits[split] = scene_splits[split][: args.num_scenes]
    else:
        rng.shuffle(scene_paths)
        if args.num_scenes > 0:
            scene_paths = scene_paths[: args.num_scenes]
        split_idx = max(1, int(len(scene_paths) * args.train_frac))
        scene_splits = {
            "train": scene_paths[:split_idx],
            "val": scene_paths[split_idx:],
        }

    explicit_targets = (
        args.train_target_episodes is not None
        or args.val_target_episodes is not None
    )
    if explicit_targets:
        if args.train_target_episodes is None or args.val_target_episodes is None:
            raise ValueError(
                "--train-target-episodes and --val-target-episodes must be paired"
            )
        split_targets = {
            "train": args.train_target_episodes,
            "val": args.val_target_episodes,
        }
    else:
        train_target = int(round(args.target_episodes * args.train_frac))
        split_targets = {
            "train": train_target,
            "val": args.target_episodes - train_target,
        }
    shortfalls = []
    for split, scenes in scene_splits.items():
        target = split_targets[split]
        feasible_scale = 1.0 - args.false_feasible_fraction
        fractions = {
            name: frac * feasible_scale
            for name, frac in BUCKET_TARGET_FRACTIONS.items()
        }
        fractions["false_feasible"] = args.false_feasible_fraction
        bucket_quota = allocate_exact_quotas(target, fractions)
        args.bucket_quota = bucket_quota
        rows: List[dict] = []
        counts = {name: 0 for name in fractions}
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
            remaining = " ".join(
                f"{name}={bucket_quota[name]}" for name in fractions
            )
            print(
                f"[{split}] {scene_path.name}: +{len(found)} "
                f"(remaining quota {remaining})"
            )

        out_path = args.out_train if split == "train" else args.out_val
        write_anchors_csv(out_path, rows, f"{args.episode_prefix}_{split}")
        count_text = " ".join(f"{name}={counts[name]}" for name in fractions)
        print(
            f"== {split}: {len(rows)} episodes written to {out_path} "
            f"({count_text}) =="
        )
        if len(rows) != target or any(bucket_quota.values()):
            shortfalls.append(
                f"{split}: expected={target}, actual={len(rows)}, "
                + ", ".join(
                    f"missing_{name}={bucket_quota[name]}" for name in fractions
                )
            )

    if args.require_target and shortfalls:
        raise RuntimeError(
            "Exact mining target was not reached: " + "; ".join(shortfalls)
        )


if __name__ == "__main__":
    main()
