#!/usr/bin/env python3
"""Initialize a standard RecurrentPPO actor from a trained direct PPO policy.

The student remains an SB3-Contrib RecurrentPPO policy.  Only its actor is
supervised here; subsequent on-policy PPO training supplies the value targets
and adapts the recurrent state to student-induced trajectories.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import HarderNarrowPassageEnv


CTYPES = [
    "straight",
    "l_shaped",
    "s_shaped",
    "narrow_exit",
    "narrow_entry",
    "asymmetric",
    "false_feasible",
]


def collect_teacher_trajectories(model, episodes: int, seed: int):
    env = HarderNarrowPassageEnv(
        {
            "corridor_types": CTYPES,
            "seed": seed,
            "width_range": (0.45, 0.90),
            "max_steps": 400,
        }
    )
    trajectories = []
    successes = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed * 1_000_000 + episode)
        trajectory = []
        info = {}
        for _ in range(400):
            trajectory.append(np.asarray(obs, dtype=np.float32).copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        trajectories.append(np.asarray(trajectory, dtype=np.float32))
        successes.append(float(info.get("success", 0.0)))
        if episode < 3 or (episode + 1) % 50 == 0:
            print(
                f"[collect] ep={episode + 1}/{episodes} "
                f"teacher_sr={np.mean(successes):.3f}",
                flush=True,
            )
    return trajectories


def sample_windows(trajectories, batch_size: int, sequence_length: int, rng):
    eligible = [trajectory for trajectory in trajectories if len(trajectory) > 0]
    windows = []
    for _ in range(batch_size):
        trajectory = eligible[int(rng.integers(0, len(eligible)))]
        if len(trajectory) >= sequence_length:
            start = int(rng.integers(0, len(trajectory) - sequence_length + 1))
            window = trajectory[start : start + sequence_length]
        else:
            indices = np.minimum(np.arange(sequence_length), len(trajectory) - 1)
            window = trajectory[indices]
        windows.append(window)
    # RecurrentPPO expects sequence-major flattened batches: all timesteps for
    # sequence 0, then all timesteps for sequence 1, and so on.
    return np.concatenate(windows, axis=0)


def distill(
    teacher,
    student,
    trajectories,
    updates: int,
    batch_size: int,
    sequence_length: int,
    learning_rate: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    th.manual_seed(seed)
    policy = student.policy
    device = policy.device
    actor_parameters = list(policy.lstm_actor.parameters())
    actor_parameters += list(policy.mlp_extractor.policy_net.parameters())
    actor_parameters += list(policy.action_net.parameters())
    optimizer = th.optim.Adam(actor_parameters, lr=learning_rate)

    losses = []
    for update in range(updates):
        obs_np = sample_windows(
            trajectories, batch_size, sequence_length, rng
        )
        obs = th.as_tensor(obs_np, device=device)
        with th.no_grad():
            teacher_obs = obs.to(teacher.device)
            teacher_dist = teacher.policy.get_distribution(teacher_obs)
            target_mean = teacher_dist.distribution.mean.to(device)

        layers = policy.lstm_actor.num_layers
        hidden = policy.lstm_actor.hidden_size
        zeros = th.zeros((layers, batch_size, hidden), device=device)
        starts = th.zeros(batch_size * sequence_length, device=device)
        starts[::sequence_length] = 1.0
        student_dist, _ = policy.get_distribution(
            obs, (zeros, zeros.clone()), starts
        )
        predicted_mean = student_dist.distribution.mean
        loss = th.mean((predicted_mean - target_mean) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if update < 3 or (update + 1) % 100 == 0:
            print(
                f"[distill] update={update + 1}/{updates} "
                f"mse={losses[-1]:.6f} mean100={np.mean(losses[-100:]):.6f}",
                flush=True,
            )
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    teacher = PPO.load(str(args.teacher_model), device=args.device)
    student = RecurrentPPO.load(str(args.student_model), device=args.device)
    trajectories = collect_teacher_trajectories(
        teacher, args.episodes, args.seed
    )
    distill(
        teacher,
        student,
        trajectories,
        args.updates,
        args.batch_size,
        args.sequence_length,
        args.learning_rate,
        args.seed,
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    student.save(str(args.output_model))
    print(f"[saved] {args.output_model}.zip")


if __name__ == "__main__":
    random.seed(0)
    main()
