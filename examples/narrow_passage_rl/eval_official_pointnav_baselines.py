#!/usr/bin/env python3
"""Evaluate official Habitat PointNav baselines on one frozen episode split.

The script intentionally keeps the stock PointNav task, discrete action space,
sensor resolution, agent radius, sliding behavior, and success definition.  It
only adds collision and step-count measurements so that every method produces
the same episode-level audit table.

Supported methods, in the requested paper order, are:

* ``pointnav_ppo``: official ResNet18+GRU PointNav PPO checkpoint;
* ``ddppo``: official ResNet50+LSTM DD-PPO checkpoint;
* ``forward_only``, ``random``, ``random_forward``, ``goal_follower``:
  official Habitat simple agents;
* ``shortest_path_follower``: Habitat NavMesh oracle upper bound.

The learned checkpoints are evaluated with sampled actions by default because
that is the protocol documented with the released DD-PPO weights.  Pass
``--deterministic`` for an additional argmax diagnostic, not as a replacement
for the sampled official run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import statistics
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from gym.spaces import Box, Dict as SpaceDict, Discrete

import habitat
from habitat.config import read_write
from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat_baselines.agents.simple_agents import (
    ForwardOnlyAgent,
    GoalFollower,
    RandomAgent,
    RandomForwardAgent,
)
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from habitat_baselines.utils.common import batch_obs


DEFAULT_DATA_PATH = Path(
    "/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/pointnav_assets/"
    "mp3d_v1/val/val.json.gz"
)
DEFAULT_SCENES_DIR = Path(
    "/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/data/scene_datasets"
)
DEFAULT_PPO_CKPT = Path(
    "/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/pointnav_assets/"
    "checkpoints/mp3d-rgbd-best.pth"
)
DEFAULT_DDPPO_CKPT = Path(
    "/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln/data/"
    "ddppo-models/gibson-2plus-resnet50.pth"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "official_pointnav_mp3d_v1_20260904"
)

METHOD_ORDER = (
    "pointnav_ppo",
    "ddppo",
    "forward_only",
    "random",
    "random_forward",
    "goal_follower",
    "shortest_path_follower",
)
STOCHASTIC_SIMPLE_METHODS = {"random", "random_forward"}
LEARNED_METHODS = {"pointnav_ppo", "ddppo"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _observation_space(input_type: str, resolution: int) -> SpaceDict:
    # The released 2021 PPO RGB-D checkpoints were trained with the v0.2.x
    # encoder, which concatenated visual tensors in the fixed order RGB then
    # Depth.  Habitat-Baselines 0.3.x instead follows observation-space order.
    # Keep the legacy order here so that the checkpoint sees the same channel
    # semantics it saw during training.  Depth-only DD-PPO is unaffected.
    # gym.spaces.Dict sorts a plain mapping by key.  OrderedDict is required
    # here; otherwise the intended RGB-before-Depth insertion order is lost.
    spaces: Dict[str, Any] = OrderedDict()
    spaces["pointgoal_with_gps_compass"] = Box(
        low=np.finfo(np.float32).min,
        high=np.finfo(np.float32).max,
        shape=(2,),
        dtype=np.float32,
    )
    if input_type in {"rgb", "rgbd"}:
        spaces["rgb"] = Box(
            low=0,
            high=255,
            shape=(resolution, resolution, 3),
            dtype=np.uint8,
        )
    if input_type in {"depth", "rgbd"}:
        spaces["depth"] = Box(
            low=0.0,
            high=1.0,
            shape=(resolution, resolution, 1),
            dtype=np.float32,
        )
    return SpaceDict(spaces)


class ReleasedPointNavPolicyAgent(Agent):
    """Adapter for both released PPO and DD-PPO PointNav checkpoints."""

    def __init__(
        self,
        checkpoint: Path,
        input_type: str,
        device: torch.device,
        seed: int,
        deterministic: bool,
        *,
        backbone: Optional[str] = None,
        rnn_type: Optional[str] = None,
        hidden_size: Optional[int] = None,
        num_recurrent_layers: Optional[int] = None,
    ) -> None:
        checkpoint_data = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        model_args = checkpoint_data.get("model_args")
        if model_args is not None:
            backbone = backbone or getattr(model_args, "backbone", None)
            rnn_type = rnn_type or getattr(model_args, "rnn_type", None)
            hidden_size = hidden_size or getattr(model_args, "hidden_size", None)
            num_recurrent_layers = num_recurrent_layers or getattr(
                model_args, "num_recurrent_layers", None
            )

        self.device = device
        self.deterministic = deterministic
        self.hidden_size = int(hidden_size or 512)
        self.input_type = input_type

        observation_space = _observation_space(input_type, 256)
        if input_type == "rgbd":
            actual_visual_order = [
                key for key in observation_space.spaces if key in {"rgb", "depth"}
            ]
            if actual_visual_order != ["rgb", "depth"]:
                raise RuntimeError(
                    "Released PPO checkpoint requires RGB then Depth channels; "
                    f"got {actual_visual_order}"
                )

        self.actor_critic = PointNavResNetPolicy(
            observation_space=observation_space,
            action_space=Discrete(4),
            hidden_size=self.hidden_size,
            backbone=str(backbone or "resnet18"),
            rnn_type=str(rnn_type or "GRU"),
            num_recurrent_layers=int(num_recurrent_layers or 1),
            normalize_visual_inputs=input_type in {"rgb", "rgbd"},
        ).to(device)
        if input_type == "rgbd":
            # PointNavResNetNet builds an internal spaces.Dict from a plain
            # comprehension, which sorts the visual keys again in Gym 0.23.
            # The tensor concatenation list is the only order-bearing state;
            # restore the legacy checkpoint convention explicitly.
            self.actor_critic.net.visual_encoder.visual_keys = ["rgb", "depth"]
            if self.actor_critic.net.visual_encoder.visual_keys != [
                "rgb",
                "depth",
            ]:
                raise RuntimeError("Could not restore legacy RGB-D channel order")

        state_dict = checkpoint_data["state_dict"]
        prefix = "actor_critic."
        if any(key.startswith(prefix) for key in state_dict):
            state_dict = {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
        self.actor_critic.load_state_dict(state_dict, strict=True)
        self.actor_critic.eval()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(
            1, 1, device=self.device, dtype=torch.bool
        )
        self.prev_actions = torch.zeros(
            1, 1, device=self.device, dtype=torch.long
        )

    def act(self, observations: Observations) -> Dict[str, int]:
        assert self.recurrent_hidden_states is not None
        assert self.not_done_masks is not None
        assert self.prev_actions is not None
        selected = {
            key: observations[key]
            for key in self.actor_critic.net.visual_encoder.visual_keys
            if key in observations
        }
        selected["pointgoal_with_gps_compass"] = observations[
            "pointgoal_with_gps_compass"
        ]
        batch = batch_obs([selected], device=self.device)
        with torch.inference_mode():
            action_data = self.actor_critic.act(
                batch,
                self.recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=self.deterministic,
            )
        self.recurrent_hidden_states = action_data.rnn_hidden_states
        self.not_done_masks.fill_(True)
        self.prev_actions.copy_(action_data.actions)
        return {"action": int(action_data.env_actions[0][0].item())}


class ShortestPathFollowerAgent(Agent):
    def __init__(self, env: habitat.Env, goal_radius: float) -> None:
        self.env = env
        self.follower = ShortestPathFollower(
            env.sim,
            goal_radius,
            return_one_hot=False,
            stop_on_error=True,
        )

    def reset(self) -> None:
        return None

    def act(self, observations: Observations) -> Dict[str, int]:
        del observations
        action = self.follower.get_next_action(
            self.env.current_episode.goals[0].position
        )
        if action is None:
            action = HabitatSimActions.stop
        return {"action": int(action)}


def make_config(args: argparse.Namespace, seed: int):
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_mp3d.yaml",
        overrides=[
            "+habitat/task/measurements@habitat.task.measurements.collisions=collisions",
            "+habitat/task/measurements@habitat.task.measurements.num_steps=num_steps",
        ],
    )
    with read_write(config):
        config.habitat.seed = seed
        config.habitat.dataset.split = args.split
        config.habitat.dataset.data_path = str(args.data_path)
        config.habitat.dataset.scenes_dir = str(args.scenes_dir)
        config.habitat.environment.max_episode_steps = args.max_steps
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.sim_gpu_id
        if args.method == "ddppo":
            # The released DD-PPO checkpoint is Depth-only.
            del config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        elif args.method not in LEARNED_METHODS:
            # PointGoal-only agents and the NavMesh oracle do not consume
            # RGB.  Keep Depth even though the agent does not consume it:
            # Habitat-Sim 0.3.3 can segfault while reconfiguring between MP3D
            # scenes when an agent has no visual sensor at all.
            sensors = config.habitat.simulator.agents.main_agent.sim_sensors
            if "rgb_sensor" in sensors:
                del sensors.rgb_sensor
    return config


def build_agent(
    method: str,
    args: argparse.Namespace,
    env: habitat.Env,
    config: Any,
    seed: int,
) -> Agent:
    goal_radius = float(
        config.habitat.task.measurements.success.success_distance
    )
    goal_sensor = config.habitat.task.goal_sensor_uuid
    if method == "pointnav_ppo":
        return ReleasedPointNavPolicyAgent(
            args.ppo_checkpoint,
            "rgbd",
            args.device,
            seed,
            args.deterministic,
            backbone="resnet18",
            rnn_type="GRU",
            hidden_size=512,
            num_recurrent_layers=1,
        )
    if method == "ddppo":
        return ReleasedPointNavPolicyAgent(
            args.ddppo_checkpoint,
            "depth",
            args.device,
            seed,
            args.deterministic,
        )
    if method == "forward_only":
        return ForwardOnlyAgent(goal_radius, goal_sensor)
    if method == "random":
        return RandomAgent(goal_radius, goal_sensor)
    if method == "random_forward":
        return RandomForwardAgent(goal_radius, goal_sensor)
    if method == "goal_follower":
        return GoalFollower(goal_radius, goal_sensor)
    if method == "shortest_path_follower":
        return ShortestPathFollowerAgent(env, goal_radius)
    raise ValueError(f"unknown method: {method}")


def _collision_metrics(metrics: Mapping[str, Any]) -> tuple[float, float]:
    collisions = metrics.get("collisions", {})
    if isinstance(collisions, Mapping):
        return float(collisions.get("count", 0.0)), float(
            collisions.get("is_collision", False)
        )
    return 0.0, 0.0


def evaluate_one(
    method: str, args: argparse.Namespace, seed: int
) -> list[dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = make_config(args, seed)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()

    with habitat.Env(config=config) as env:
        total = env.number_of_episodes
        requested = total if args.num_episodes < 0 else min(args.num_episodes, total)
        agent = build_agent(method, args, env, config, seed)
        for episode_index in range(requested):
            observations = env.reset()
            episode = env.current_episode
            agent.reset()
            while not env.episode_over:
                observations = env.step(agent.act(observations))

            metrics = env.get_metrics()
            collision_count, final_collision = _collision_metrics(metrics)
            rows.append(
                {
                    "method": method,
                    "eval_seed": seed,
                    "episode_index": episode_index,
                    "episode_id": str(episode.episode_id),
                    "scene_id": str(episode.scene_id),
                    "success": float(metrics.get("success", 0.0)),
                    "spl": float(metrics.get("spl", 0.0)),
                    "distance_to_goal": float(
                        metrics.get("distance_to_goal", float("nan"))
                    ),
                    "collision_count": collision_count,
                    "final_step_collision": final_collision,
                    "num_steps": int(metrics.get("num_steps", 0)),
                }
            )
            if (episode_index + 1) % args.progress_every == 0:
                elapsed = time.monotonic() - started
                successes = sum(row["success"] for row in rows)
                print(
                    f"[{method} seed={seed}] {episode_index + 1}/{requested} "
                    f"SR={successes / len(rows):.4f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    return rows


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(statistics.fmean(values)) if values else float("nan")


def write_outputs(
    method: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    seeds: list[int],
) -> None:
    method_dir = args.output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    episode_path = method_dir / "episodes.csv"
    with episode_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_seed = []
    for seed in seeds:
        selected = [row for row in rows if int(row["eval_seed"]) == seed]
        per_seed.append(
            {
                "method": method,
                "eval_seed": seed,
                "episodes": len(selected),
                "success": _mean(selected, "success"),
                "spl": _mean(selected, "spl"),
                "distance_to_goal": _mean(selected, "distance_to_goal"),
                "collision_rate": float(
                    statistics.fmean(
                        float(row["collision_count"] > 0.0) for row in selected
                    )
                ),
                "mean_collision_count": _mean(selected, "collision_count"),
                "mean_steps": _mean(selected, "num_steps"),
            }
        )

    summary_path = method_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)

    metadata = {
        "method": method,
        "dataset": "official MP3D PointNav v1",
        "split": args.split,
        "data_path": str(args.data_path),
        "data_sha256": _sha256(args.data_path),
        "scenes_dir": str(args.scenes_dir),
        "max_steps": args.max_steps,
        "sampled_actions": not args.deterministic,
        "seeds": seeds,
        "ppo_checkpoint": str(args.ppo_checkpoint)
        if method == "pointnav_ppo"
        else None,
        "ppo_checkpoint_sha256": _sha256(args.ppo_checkpoint)
        if method == "pointnav_ppo"
        else None,
        "ddppo_checkpoint": str(args.ddppo_checkpoint)
        if method == "ddppo"
        else None,
        "ddppo_checkpoint_sha256": _sha256(args.ddppo_checkpoint)
        if method == "ddppo"
        else None,
        "ddppo_train_domain": "Gibson 2+"
        if method == "ddppo"
        else None,
        "official_pointnav_defaults_preserved": {
            "agent_radius": 0.1,
            "forward_step_size": 0.25,
            "turn_angle_degrees": 10,
            "allow_sliding": True,
            "success_distance": 0.2,
        },
        "policy_visual_inputs": {
            "pointnav_ppo": ["rgb", "depth"],
            "ddppo": ["depth"],
        }.get(method, []),
        "simulator_visual_sensors": (
            ["rgb", "depth"]
            if method == "pointnav_ppo"
            else ["depth"]
        ),
        "legacy_checkpoint_visual_channel_order": (
            ["rgb", "depth"] if method == "pointnav_ppo" else ["depth"]
            if method == "ddppo"
            else []
        ),
        "legacy_rgbd_channel_adapter_applied": method == "pointnav_ppo",
        "software": {
            name: importlib.metadata.version(name)
            for name in ("habitat-lab", "habitat-sim", "torch")
        },
    }
    (method_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {episode_path}", flush=True)
    print(f"[write] {summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHOD_ORDER, required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--scenes-dir", type=Path, default=DEFAULT_SCENES_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument("--ppo-checkpoint", type=Path, default=DEFAULT_PPO_CKPT)
    parser.add_argument(
        "--ddppo-checkpoint", type=Path, default=DEFAULT_DDPPO_CKPT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-episodes", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--sim-gpu-id", type=int, default=0)
    parser.add_argument("--policy-gpu-id", type=int, default=0)
    parser.add_argument("--cpu-policy", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    if not args.data_path.is_file():
        parser.error(f"missing PointNav dataset: {args.data_path}")
    if not args.scenes_dir.is_dir():
        parser.error(f"missing scene directory: {args.scenes_dir}")
    if args.method == "pointnav_ppo" and not args.ppo_checkpoint.is_file():
        parser.error(f"missing PPO checkpoint: {args.ppo_checkpoint}")
    if args.method == "ddppo" and not args.ddppo_checkpoint.is_file():
        parser.error(f"missing DD-PPO checkpoint: {args.ddppo_checkpoint}")

    if args.cpu_policy or not torch.cuda.is_available():
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda", args.policy_gpu_id)

    requested_seeds = args.seed or [1701, 1702, 1703]
    if args.method not in STOCHASTIC_SIMPLE_METHODS and args.method not in LEARNED_METHODS:
        requested_seeds = requested_seeds[:1]
    args.seeds = requested_seeds
    return args


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    completed_seeds: list[int] = []
    for seed in args.seeds:
        all_rows.extend(evaluate_one(args.method, args, seed))
        completed_seeds.append(seed)
        # Persist at every seed boundary so a later interruption never erases
        # already completed full-split runs.
        write_outputs(args.method, args, all_rows, completed_seeds)
    if not all_rows:
        raise RuntimeError("evaluation produced no episode rows")


if __name__ == "__main__":
    main()
