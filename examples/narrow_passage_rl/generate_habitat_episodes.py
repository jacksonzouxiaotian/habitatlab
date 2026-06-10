#!/usr/bin/env python3

import argparse
import csv
import gzip
import json
import math
from pathlib import Path


def yaw_to_quat(yaw):
    half = 0.5 * yaw
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def normalize(v):
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-8:
        return [0.0, 0.0, 1.0]
    return [x / norm for x in v]


def yaw_from_start_goal(start, goal):
    dx = goal[0] - start[0]
    dz = goal[2] - start[2]
    return math.atan2(dx, dz)


def read_anchors(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())

    anchors = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            anchors.append(row)
    return anchors


def as_float_list(value, keys, row):
    if value:
        return [float(x.strip()) for x in value.split(",")]
    return [float(row[key]) for key in keys]


def build_episode(row, idx, split):
    scene_id = row["scene_id"]
    start = as_float_list(row.get("start_position"), ["start_x", "start_y", "start_z"], row)
    goal = as_float_list(row.get("goal_position"), ["goal_x", "goal_y", "goal_z"], row)
    center = as_float_list(
        row.get("passage_center"),
        ["passage_x", "passage_y", "passage_z"],
        row,
    )
    direction = row.get("passage_direction", "")
    if direction:
        passage_dir = normalize([float(x.strip()) for x in direction.split(",")])
    else:
        passage_dir = normalize([goal[i] - start[i] for i in range(3)])

    yaw = float(row["start_yaw"]) if row.get("start_yaw") else yaw_from_start_goal(start, goal)
    episode_id = row.get("episode_id") or f"{split}_narrow_passage_{idx:06d}"
    difficulty = row.get("difficulty", "hard")
    passage_width = float(row.get("passage_width", 0.75))
    passage_length = float(row.get("passage_length", 2.0))

    return {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "start_position": start,
        "start_rotation": yaw_to_quat(yaw),
        "goals": [{"position": goal, "radius": float(row.get("goal_radius", 0.3))}],
        "info": {
            "passage_center": center,
            "passage_direction": passage_dir,
            "passage_width": passage_width,
            "passage_length": passage_length,
            "difficulty": difficulty,
            "false_feasible": bool(int(row.get("false_feasible", 0))),
            "body_margin": float(row.get("body_margin", passage_width * 0.5 - 0.18)),
        },
    }


def write_dataset(path, episodes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = {
        "episodes": episodes,
        "category_to_task_category_id": {},
        "category_to_scene_annotation_category_id": {},
        "category_to_mp3d_category_id": {},
    }
    with gzip.open(path, "wt") as f:
        json.dump(dataset, f)


def write_template(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_id",
        "scene_id",
        "start_x",
        "start_y",
        "start_z",
        "goal_x",
        "goal_y",
        "goal_z",
        "passage_x",
        "passage_y",
        "passage_z",
        "passage_width",
        "passage_length",
        "difficulty",
        "false_feasible",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "episode_id": "demo_gap_000001",
                "scene_id": "data/scene_datasets/hm3d/example/example.basis.glb",
                "start_x": 0.0,
                "start_y": 0.0,
                "start_z": -1.0,
                "goal_x": 0.0,
                "goal_y": 0.0,
                "goal_z": 2.0,
                "passage_x": 0.0,
                "passage_y": 0.0,
                "passage_z": 0.5,
                "passage_width": 0.75,
                "passage_length": 2.0,
                "difficulty": "hard",
                "false_feasible": 0,
            }
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchors", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/narrow_passage/train/train.json.gz"),
    )
    parser.add_argument("--write-template", type=Path, default=None)
    args = parser.parse_args()

    if args.write_template is not None:
        write_template(args.write_template)
        print(f"template: {args.write_template}")
        return
    if args.anchors is None:
        raise ValueError("Provide --anchors or --write-template.")

    anchors = read_anchors(args.anchors)
    episodes = [build_episode(row, idx, args.split) for idx, row in enumerate(anchors)]
    write_dataset(args.output, episodes)
    print(f"episodes: {len(episodes)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
