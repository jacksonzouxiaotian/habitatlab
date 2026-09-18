#!/usr/bin/env python3
"""Collect raw visual demonstrations for DEGNAV-E2E on Habitat scans.

The 19-D geometry vector is consulted only by the privileged teacher and saved
under an explicitly named training-only key.  Actor observations contain raw
Depth (optionally RGB), PointGoal, previous mode/outcome state, and recurrent
history.  The optional task-memory field is off by default because it is
constructed from hand-designed state bins and is not part of the clean E2E run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import habitat
import numpy as np
from habitat.config import read_write

from narrow_passage.selector.contract import MEMORY_DIM, Mode
from narrow_passage.selector.habitat_controller import (
    mode_action,
    privileged_teacher_mode,
    stop_action,
)
from narrow_passage.selector.visual_model import ACTION_OUTCOME_DIM


DEFAULT_DATA_ROOT = Path(
    "/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/pointnav_assets/"
    "mp3d_narrow_v1"
)


def make_config(args: argparse.Namespace):
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "habitat/task=narrow_passage",
            f"habitat.dataset.data_path={args.dataset}",
            f"habitat.dataset.split={args.split}",
            "habitat.dataset.type=PointNav-v1",
            "habitat.simulator.habitat_sim_v0.allow_sliding=False",
            f"habitat.simulator.agents.main_agent.radius={args.agent_radius}",
        ],
    )
    with read_write(config):
        config.habitat.seed = args.seed
        config.habitat.environment.max_episode_steps = args.max_steps
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.sim_gpu_id
    return config


def resize_depth(depth: np.ndarray, size: int) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32).squeeze(-1)
    value = cv2.resize(value, (size, size), interpolation=cv2.INTER_AREA)
    value = np.nan_to_num(value, nan=1.0, posinf=1.0, neginf=0.0)
    return np.rint(np.clip(value, 0.0, 1.0) * 65535.0).astype(np.uint16)


def resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.uint8)
    return cv2.resize(value, (size, size), interpolation=cv2.INTER_AREA)


def actor_observation(
    observations: dict[str, Any],
    action_outcome: np.ndarray,
    size: int,
    input_type: str,
    *,
    include_privileged: bool = True,
    include_memory: bool = False,
) -> dict[str, np.ndarray]:
    required = {
        "depth",
        "pointgoal_with_gps_compass",
    }
    if include_privileged:
        required.add("narrow_passage_features")
    if include_memory:
        required.add("narrow_passage_memory")
    missing = sorted(required - observations.keys())
    if missing:
        raise KeyError(f"Habitat observation is missing: {missing}")
    output = {
        "depth": resize_depth(observations["depth"], size),
        "pointgoal": np.array(
            observations["pointgoal_with_gps_compass"], dtype=np.float32
        ).reshape(2).copy(),
        # ``action_outcome`` is mutated in place after each environment step.
        # Store a snapshot: appending a view here would overwrite the complete
        # episode history with its final value and leak future outcome state.
        "action_outcome": np.array(
            action_outcome, dtype=np.float32, copy=True
        ).reshape(ACTION_OUTCOME_DIM),
    }
    if include_memory:
        output["memory_context"] = np.array(
            observations["narrow_passage_memory"], dtype=np.float32
        ).reshape(MEMORY_DIM).copy()
    if include_privileged:
        output["privileged_19d"] = np.array(
            observations["narrow_passage_features"], dtype=np.float32
        ).reshape(19).copy()
    if input_type == "rgbd":
        if "rgb" not in observations:
            raise KeyError("RGB-D collection requested but Habitat returned no rgb")
        output["rgb"] = resize_rgb(observations["rgb"], size)
    return output


def write_episode(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.stem}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def collect_episode(
    env: habitat.Env,
    observations: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode = env.current_episode
    info = dict(getattr(episode, "info", {}) or {})
    morphology_infeasible = bool(info.get("false_feasible", False))
    action_outcome = np.zeros(ACTION_OUTCOME_DIM, dtype=np.float32)
    storage: dict[str, list[np.ndarray | int | bool]] = {
        "depth": [],
        "pointgoal": [],
        "action_outcome": [],
        "privileged_19d": [],
        "mode": [],
        "episode_start": [],
    }
    if args.input_type == "rgbd":
        storage["rgb"] = []
    if args.store_memory_context:
        storage["memory_context"] = []
    modes: Counter[str] = Counter()
    collision = False
    rejected = False
    steps = 0
    no_progress_steps = 0
    started = time.monotonic()

    while not env.episode_over and steps < args.max_steps:
        visual = actor_observation(
            observations,
            action_outcome,
            args.image_size,
            args.input_type,
            include_memory=args.store_memory_context,
        )
        features = visual["privileged_19d"]
        current_distance = float(visual["pointgoal"][0])
        current_heading = float(visual["pointgoal"][1])
        # STOP is a task termination action, not one of the four DEGNAV modes.
        if current_distance < args.stop_distance:
            observations = env.step(stop_action())
            steps += 1
            break

        mode = privileged_teacher_mode(
            features,
            current_heading,
            morphology_infeasible=morphology_infeasible,
        )
        for key, value in visual.items():
            storage[key].append(value)
        storage["mode"].append(int(mode))
        storage["episode_start"].append(len(storage["mode"]) == 1)
        modes[mode.name] += 1

        if mode is Mode.REJECT:
            rejected = True
            break

        # Deployment-equivalent controller: PointGoal only.  The teacher's
        # privileged lateral offset chooses the mode but never realizes action.
        action = mode_action(mode, current_heading, 0.0)
        if action is None:
            raise AssertionError("only REJECT may omit a Habitat action")
        previous_distance = current_distance
        observations = env.step(action)
        steps += 1
        next_pointgoal = np.asarray(
            observations["pointgoal_with_gps_compass"], dtype=np.float32
        )
        progress = float(previous_distance - next_pointgoal[0])
        no_progress_steps = no_progress_steps + 1 if abs(progress) < 1e-3 else 0
        previous_sim_observation = getattr(env.sim, "_prev_sim_obs", {})
        step_collision = bool(
            isinstance(previous_sim_observation, dict)
            and previous_sim_observation.get("collided", False)
        )
        action_outcome.fill(0.0)
        action_outcome[int(mode)] = 1.0
        action_outcome[4] = float(step_collision)
        action_outcome[5] = float(no_progress_steps >= 14)
        action_outcome[6] = float(np.clip(progress, -1.0, 1.0))
        collision = collision or step_collision

    if not storage["mode"]:
        raise RuntimeError(f"episode {episode.episode_id} produced no teacher samples")
    metrics = env.get_metrics()
    success = float(metrics.get("narrow_passage_success", 0.0))
    stacked = {
        key: np.stack(value)
        if key not in {"mode", "episode_start"}
        else np.asarray(value, dtype=np.uint8 if key == "mode" else np.bool_)
        for key, value in storage.items()
    }
    stacked.update(
        {
            "episode_id": np.asarray(str(episode.episode_id)),
            "scene_id": np.asarray(str(episode.scene_id)),
            "morphology_infeasible": np.asarray(morphology_infeasible),
        }
    )
    row = {
        "episode_id": str(episode.episode_id),
        "scene_id": str(episode.scene_id),
        "difficulty": str(info.get("difficulty", "unknown")),
        "morphology_infeasible": int(morphology_infeasible),
        "frames": len(storage["mode"]),
        "environment_steps": steps,
        "success": success,
        "collision": int(collision),
        "rejected": int(rejected),
        "commit_frames": modes[Mode.COMMIT.name],
        "explore_frames": modes[Mode.EXPLORE.name],
        "recover_frames": modes[Mode.RECOVER.name],
        "reject_frames": modes[Mode.REJECT.name],
        "elapsed_seconds": time.monotonic() - started,
    }
    return stacked, row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_ROOT / "e2e_teacher")
    parser.add_argument("--input-type", choices=("depth", "rgbd"), default="depth")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--agent-radius", type=float, default=0.18)
    parser.add_argument("--stop-distance", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--num-episodes", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument(
        "--store-memory-context",
        action="store_true",
        help="Store the task's hand-designed 4-D memory for a separate ablation.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = DEFAULT_DATA_ROOT / args.split / f"{args.split}.json.gz"
    if not args.dataset.is_file():
        parser.error(f"missing derived MP3D dataset: {args.dataset}")

    split_dir = args.output_dir / args.input_type / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(args)
    rows: list[dict[str, Any]] = []
    with habitat.Env(config=config) as env:
        requested = env.number_of_episodes
        if args.num_episodes >= 0:
            requested = min(requested, args.num_episodes)
        for index in range(requested):
            observations = env.reset()
            episode_id = str(env.current_episode.episode_id)
            output = split_dir / f"{episode_id}.npz"
            if output.exists() and args.resume:
                with np.load(output, allow_pickle=False) as existing:
                    modes = np.asarray(existing["mode"])
                    row = {
                        "episode_id": episode_id,
                        "scene_id": str(existing["scene_id"]),
                        "difficulty": "resume_unknown",
                        "morphology_infeasible": int(existing["morphology_infeasible"]),
                        "frames": len(modes),
                        "environment_steps": -1,
                        "success": float("nan"),
                        "collision": -1,
                        "rejected": int(np.any(modes == int(Mode.REJECT))),
                        **{
                            f"{mode.name.lower()}_frames": int(np.sum(modes == int(mode)))
                            for mode in Mode
                        },
                        "elapsed_seconds": 0.0,
                    }
            else:
                arrays, row = collect_episode(env, observations, args)
                write_episode(output, arrays)
            rows.append(row)
            print(
                f"[{args.split}] {index + 1}/{requested} {episode_id} "
                f"frames={row['frames']} success={row['success']} "
                f"reject={row['rejected']}",
                flush=True,
            )

    manifest = split_dir / "episodes.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mode_counts = {
        mode.name: sum(int(row[f"{mode.name.lower()}_frames"]) for row in rows)
        for mode in Mode
    }
    missing_modes = [name for name, count in mode_counts.items() if count <= 0]
    if missing_modes:
        raise RuntimeError(
            "teacher collection has no supervision for modes: "
            f"{missing_modes}; refusing to train a nominal four-mode policy"
        )
    metadata = {
        "dataset": str(args.dataset),
        "source": "official MP3D scans and official PointNav train/val scene split",
        "split": args.split,
        "episodes": len(rows),
        "frames": sum(int(row["frames"]) for row in rows),
        "input_type": args.input_type,
        "image_size": args.image_size,
        "agent_radius": args.agent_radius,
        "history_storage_version": 2,
        "history_storage_contract": "frame t stores a copy of outcome from action t-1",
        "actor_inputs": [
            "raw depth",
            *( ["raw rgb"] if args.input_type == "rgbd" else [] ),
            "pointgoal",
            "previous mode/outcome",
            "GRU history",
            *( ["task memory context"] if args.store_memory_context else [] ),
        ],
        "actor_excludes": [
            "narrow_passage_features (19-D)",
            "feasibility label",
            *( [] if args.store_memory_context else ["hand-designed task memory"] ),
        ],
        "privileged_training_only": ["narrow_passage_features (19-D)", "false_feasible"],
        "teacher_rule": {
            "explore_if_body_margin_below_m": 0.30,
            "reject_if_infeasible_and_observed_failure": True,
            "observed_failure": "collision or stuck_score > 0.80",
        },
        "mode_counts": mode_counts,
    }
    (split_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {manifest}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
