#!/usr/bin/env python3
"""Evaluate a trained DEGNAV-E2E selector on held-out Habitat scenes."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import habitat
import numpy as np
import torch
from habitat.config import read_write

from collect_habitat_e2e_teacher import actor_observation
from narrow_passage.selector.contract import MODE_NAMES, Mode
from narrow_passage.selector.habitat_controller import (
    mode_action,
    stop_action,
)
from narrow_passage.selector.visual_model import VisualFourModeGRUPolicy


DEFAULT_DATASET = Path(
    "/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/pointnav_assets/"
    "mp3d_narrow_v1/val/val.json.gz"
)


def make_config(args: argparse.Namespace, include_memory: bool):
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "habitat/task=narrow_passage",
            f"habitat.dataset.data_path={args.dataset}",
            "habitat.dataset.split=val",
            "habitat.dataset.type=PointNav-v1",
            "habitat.simulator.habitat_sim_v0.allow_sliding=False",
            f"habitat.simulator.agents.main_agent.radius={args.agent_radius}",
        ],
    )
    with read_write(config):
        config.habitat.seed = args.seed
        config.habitat.environment.max_episode_steps = args.max_steps
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.sim_gpu_id
        # Keep the geometry sensor in the environment because the custom task's
        # success/collision/stuck measures consume it.  ``actor_observation``
        # below is the deployment boundary and, with ``include_privileged=False``,
        # never forwards this evaluator-only tensor to the policy.  Removing the
        # sensor would silently force NarrowPassageSuccess to zero.
        sensors = config.habitat.task.lab_sensors
        if "narrow_passage_geometry_sensor" not in sensors:
            raise RuntimeError(
                "narrow_passage_geometry_sensor is required by task measures"
            )
        # The optional memory sensor is retained only for an explicitly
        # memory-conditioned actor ablation.
        if not include_memory and "narrow_passage_memory_sensor" in sensors:
            del sensors.narrow_passage_memory_sensor
    return config


def load_policy(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    policy = VisualFourModeGRUPolicy(**model_config)
    policy.load_state_dict(checkpoint["model"], strict=True)
    policy.to(device).eval()
    return policy, checkpoint


def evaluate_episode(
    env: habitat.Env,
    observations: dict[str, Any],
    policy: VisualFourModeGRUPolicy,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    episode = env.current_episode
    episode_info = dict(getattr(episode, "info", {}) or {})
    infeasible = bool(episode_info.get("false_feasible", False))
    action_outcome = np.zeros(7, dtype=np.float32)
    hidden = policy.initial_hidden(1, device)
    first_step = True
    rejected = False
    stop_triggered = False
    collision = False
    steps = 0
    no_progress_steps = 0
    mode_counts: Counter[str] = Counter()
    confidence_sum = 0.0
    started = time.monotonic()

    while not env.episode_over and steps < args.max_steps:
        current_pointgoal = np.asarray(
            observations["pointgoal_with_gps_compass"], dtype=np.float32
        )
        current_distance = float(current_pointgoal[0])
        current_heading = float(current_pointgoal[1])
        if current_distance < args.stop_distance:
            observations = env.step(stop_action())
            steps += 1
            stop_triggered = True
            break

        actor = actor_observation(
            observations,
            action_outcome,
            args.image_size,
            policy.input_type,
            include_privileged=False,
            include_memory=bool(policy.memory_dim),
        )
        depth = torch.from_numpy(actor["depth"]).to(device).float()[None, ..., None]
        depth /= 65535.0
        pointgoal = torch.from_numpy(actor["pointgoal"]).to(device)[None]
        outcome = torch.from_numpy(actor["action_outcome"]).to(device)[None]
        starts = torch.as_tensor([first_step], device=device, dtype=torch.float32)
        rgb = (
            torch.from_numpy(actor["rgb"]).to(device)[None]
            if policy.input_type == "rgbd"
            else None
        )
        memory = (
            torch.from_numpy(actor["memory_context"]).to(device)[None]
            if policy.memory_dim
            else None
        )
        with torch.inference_mode():
            logits, _values, hidden = policy.forward_sequence(
                depth,
                pointgoal,
                outcome,
                hidden,
                rgb=rgb,
                memory_context=memory,
                episode_starts=starts,
            )
            probabilities = torch.softmax(logits[0, 0], dim=-1)
            if args.sample_actions:
                mode_index = int(torch.multinomial(probabilities, 1).item())
            else:
                mode_index = int(probabilities.argmax().item())
        first_step = False
        mode = Mode(mode_index)
        mode_counts[mode.name] += 1
        confidence_sum += float(probabilities[mode_index].item())
        if mode is Mode.REJECT:
            rejected = True
            break

        action = mode_action(mode, current_heading, 0.0)
        if action is None:
            raise AssertionError("non-REJECT mode returned no action")
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
        action_outcome[mode_index] = 1.0
        action_outcome[4] = float(step_collision)
        action_outcome[5] = float(no_progress_steps >= 14)
        action_outcome[6] = float(np.clip(progress, -1.0, 1.0))
        collision = collision or step_collision

    metrics = env.get_metrics()
    success = float(metrics.get("narrow_passage_success", 0.0))
    diagnostic_state = env.task.get_narrow_passage_state()
    if success > 0.5:
        termination_reason = "success"
    elif rejected:
        termination_reason = "reject"
    elif steps >= args.max_steps:
        termination_reason = "timeout"
    elif stop_triggered:
        termination_reason = "stop_without_task_success"
    elif env.episode_over:
        termination_reason = "environment_end"
    else:
        termination_reason = "unknown"
    decision_count = sum(mode_counts.values())
    return {
        "episode_id": str(episode.episode_id),
        "scene_id": str(episode.scene_id),
        "difficulty": str(episode_info.get("difficulty", "unknown")),
        "morphology_infeasible": int(infeasible),
        "success": success,
        "collision": int(collision),
        "rejected": int(rejected),
        "correct_reject": int(rejected and infeasible),
        "false_reject": int(rejected and not infeasible),
        "steps": steps,
        "termination_reason": termination_reason,
        "final_distance_to_goal": float(diagnostic_state.distance_to_local_goal),
        "final_heading_error": float(diagnostic_state.heading_error),
        "final_lateral_offset": float(diagnostic_state.lateral_offset),
        "final_stuck_score": float(diagnostic_state.stuck_score),
        "mean_selected_probability": confidence_sum / max(1, decision_count),
        **{
            f"{name.lower()}_steps": mode_counts[name] for name in MODE_NAMES
        },
        "elapsed_seconds": time.monotonic() - started,
    }


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.fmean(float(row[key]) for row in rows)) if rows else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--agent-radius", type=float, default=0.18)
    parser.add_argument("--stop-distance", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--policy-gpu-id", type=int, default=0)
    parser.add_argument("--cpu-policy", action="store_true")
    parser.add_argument("--sample-actions", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"missing E2E checkpoint: {args.checkpoint}")
    if not args.dataset.is_file():
        parser.error(f"missing validation dataset: {args.dataset}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cpu"
        if args.cpu_policy or not torch.cuda.is_available()
        else f"cuda:{args.policy_gpu_id}"
    )
    policy, checkpoint = load_policy(args.checkpoint, device)
    config = make_config(args, include_memory=bool(policy.memory_dim))
    rows: list[dict[str, Any]] = []
    with habitat.Env(config=config) as env:
        requested = env.number_of_episodes
        if args.num_episodes >= 0:
            requested = min(requested, args.num_episodes)
        for index in range(requested):
            observations = env.reset()
            row = evaluate_episode(env, observations, policy, args, device)
            rows.append(row)
            print(
                f"[eval] {index + 1}/{requested} success={row['success']:.0f} "
                f"collision={row['collision']} reject={row['rejected']}",
                flush=True,
            )

    feasible = [row for row in rows if not row["morphology_infeasible"]]
    infeasible = [row for row in rows if row["morphology_infeasible"]]
    summary = {
        "method": "DEGNAV-E2E",
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset": str(args.dataset),
        "episodes": len(rows),
        "feasible_episodes": len(feasible),
        "infeasible_episodes": len(infeasible),
        "success_rate_feasible": rate(feasible, "success"),
        "collision_rate_feasible": rate(feasible, "collision"),
        "false_reject_rate_feasible": rate(feasible, "false_reject"),
        "correct_reject_rate_infeasible": rate(infeasible, "correct_reject"),
        "mean_steps": rate(rows, "steps"),
        "sample_actions": args.sample_actions,
        "actor_uses_19d": False,
        "actor_uses_handdesigned_task_memory": bool(policy.memory_dim),
        "evaluator_uses_19d_for_task_measures": True,
        "protocol": "MP3D-derived narrow passage; official train/val scene split",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "episodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
