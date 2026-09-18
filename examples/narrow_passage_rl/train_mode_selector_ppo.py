#!/usr/bin/env python3
"""Recurrent PPO for the strict 19-D, four-mode selector.

The implementation keeps complete `[rollout, env]` sequences during PPO
updates, so the GRU is optimized through time instead of treating cached hidden
states as unrelated flat samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import yaml
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import HarderNarrowPassageEnv
from narrow_passage.envs.four_mode_selector_env import (
    REQUIRED_STEP_LOG_FIELDS,
    FourModeSelectorEnv,
)
from narrow_passage.selector.contract import MODE_NAMES, assert_selector_contract
from narrow_passage.selector.expert import training_rule_mode
from narrow_passage.selector.model import FourModeGRUPolicy
from narrow_passage.selector.sampling import BalancedScenarioSampler


DEFAULTS = {
    "seed": 1701,
    "total_steps": 20_000,
    "num_envs": 16,
    "rollout_steps": 128,
    "learning_rate": 2.5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_param": 0.2,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "ppo_epochs": 4,
    "num_mini_batch": 4,
    "hidden_size": 128,
    "max_episode_steps": 160,
    "with_memory": False,
    "device": "cpu",
    "supervised_aux_coef": 0.0,
}


def load_config(path: Path | None) -> dict:
    config = dict(DEFAULTS)
    if path is not None:
        loaded = yaml.safe_load(path.read_text()) or {}
        if "ppo" in loaded:
            loaded = {**loaded, **loaded.pop("ppo")}
        config.update(loaded)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_gae(rewards, dones, values, last_values, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        next_values = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * nonterminal - values[step]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[step] = gae
    return advantages, advantages + values


def ppo_update(policy, optimizer, rollout, advantages, returns, cfg, device):
    policy.train()
    num_envs = rollout["obs"].shape[1]
    envs_per_batch = max(1, num_envs // int(cfg["num_mini_batch"]))
    diagnostics = defaultdict(list)
    final_grad_by_mode = [0.0] * 4
    for _epoch in range(int(cfg["ppo_epochs"])):
        permutation = np.random.permutation(num_envs)
        for start in range(0, num_envs, envs_per_batch):
            mb = permutation[start : start + envs_per_batch]
            obs = torch.as_tensor(rollout["obs"][:, mb], dtype=torch.float32, device=device)
            memory = torch.as_tensor(rollout["memory"][:, mb], dtype=torch.float32, device=device)
            starts = torch.as_tensor(rollout["starts"][:, mb], dtype=torch.float32, device=device)
            actions = torch.as_tensor(rollout["actions"][:, mb], dtype=torch.long, device=device)
            expert_actions = torch.as_tensor(
                rollout["expert_actions"][:, mb], dtype=torch.long, device=device
            )
            old_logp = torch.as_tensor(rollout["logp"][:, mb], dtype=torch.float32, device=device)
            adv = torch.as_tensor(advantages[:, mb], dtype=torch.float32, device=device)
            ret = torch.as_tensor(returns[:, mb], dtype=torch.float32, device=device)
            hidden = torch.as_tensor(rollout["initial_hidden"][:, mb], dtype=torch.float32, device=device)
            dist, values, _ = policy.distribution_and_value(
                obs,
                hidden,
                memory if policy.memory_dim else None,
                starts,
            )
            logp = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(logp - old_logp)
            policy_loss = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1.0 - cfg["clip_param"], 1.0 + cfg["clip_param"]) * adv,
            ).mean()
            value_loss = 0.5 * (ret - values).square().mean()
            auxiliary_loss = F.cross_entropy(
                dist.logits.reshape(-1, 4), expert_actions.reshape(-1)
            )
            loss = (
                policy_loss
                + cfg["value_loss_coef"] * value_loss
                - cfg["entropy_coef"] * entropy
                + float(cfg["supervised_aux_coef"]) * auxiliary_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            if policy.actor.weight.grad is not None:
                final_grad_by_mode = policy.actor.weight.grad.detach().norm(dim=1).cpu().tolist()
            optimizer.step()
            diagnostics["policy_loss"].append(float(policy_loss.item()))
            diagnostics["value_loss"].append(float(value_loss.item()))
            diagnostics["entropy"].append(float(entropy.item()))
            diagnostics["supervised_aux_loss"].append(float(auxiliary_loss.item()))
            diagnostics["approx_kl"].append(float((old_logp - logp).mean().item()))
    output = {key: float(np.mean(values)) for key, values in diagnostics.items()}
    output["actor_head_grad_norm"] = {
        name: float(value) for name, value in zip(MODE_NAMES, final_grad_by_mode)
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bc-checkpoint", type=Path, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for name in ("total_steps", "seed", "device"):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value
    cfg["output_dir"] = str(args.output_dir)
    cfg["bc_checkpoint"] = str(args.bc_checkpoint) if args.bc_checkpoint else None
    seed_everything(int(cfg["seed"]))
    device = torch.device(cfg["device"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True)
    )

    envs, samplers, episode_numbers = [], [], []
    for rank in range(int(cfg["num_envs"])):
        base = HarderNarrowPassageEnv(
            {
                "seed": int(cfg["seed"]) + rank,
                "max_steps": int(cfg["max_episode_steps"]),
            }
        )
        envs.append(FourModeSelectorEnv(base))
        samplers.append(BalancedScenarioSampler(int(cfg["seed"]) + 10_000 + rank))
        episode_numbers.append(0)

    current_specs = [None] * len(envs)

    def reset_env(index: int):
        spec = samplers[index].sample()
        current_specs[index] = spec
        envs[index].set_memory_context(spec.memory_context if cfg["with_memory"] else None)
        seed = int(cfg["seed"]) + index * 1_000_003 + episode_numbers[index]
        episode_numbers[index] += 1
        return envs[index].reset(seed=seed, options=spec.reset_options)

    reset_results = [reset_env(i) for i in range(len(envs))]
    observations = np.stack([item[0] for item in reset_results])
    for obs, env in zip(observations, envs):
        assert_selector_contract(obs, env.action_space)
    episode_starts = np.ones(len(envs), dtype=np.float32)

    policy = FourModeGRUPolicy(
        hidden_size=int(cfg["hidden_size"]),
        memory_dim=4 if cfg["with_memory"] else 0,
    ).to(device)
    initialization = "scratch"
    if args.bc_checkpoint is not None:
        checkpoint = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
        expected = checkpoint["model_config"]
        if int(expected["hidden_size"]) != int(cfg["hidden_size"]):
            raise ValueError("BC checkpoint hidden_size does not match PPO config")
        if int(expected["memory_dim"]) != policy.memory_dim:
            raise ValueError("BC checkpoint memory_dim does not match PPO config")
        policy.load_state_dict(checkpoint["model"])
        initialization = "behavior_cloning"
    else:
        policy.update_normalization(torch.as_tensor(observations, device=device))
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(cfg["learning_rate"]))

    step_log_path = args.output_dir / "steps.csv"
    episode_log_path = args.output_dir / "episodes.csv"
    step_handle = step_log_path.open("w", newline="")
    episode_handle = episode_log_path.open("w", newline="")
    step_fields = [
        "global_step", "env_rank", "episode_number", "scenario_category",
        "reward", "termination_reason", *REQUIRED_STEP_LOG_FIELDS,
    ]
    episode_fields = [
        "env_rank", "episode_number", "scenario_category", "corridor_type",
        "feasible", "termination_reason", "success", "collision", "stuck",
        "timeout", "correct_reject", "false_reject", "episode_return",
        "selector_steps",
    ]
    step_writer = csv.DictWriter(step_handle, fieldnames=step_fields)
    episode_writer = csv.DictWriter(episode_handle, fieldnames=episode_fields)
    step_writer.writeheader()
    episode_writer.writeheader()

    total_steps = 0
    action_counts: Counter[str] = Counter()
    recent_actions: deque[str] = deque(maxlen=10_000)
    terminal_counts: Counter[str] = Counter()
    reward_by_mode: defaultdict[str, list[float]] = defaultdict(list)
    advantage_by_mode: defaultdict[str, list[float]] = defaultdict(list)
    update_rows = []
    hidden = policy.initial_hidden(len(envs), device)

    while total_steps < int(cfg["total_steps"]):
        rollout_steps = min(
            int(cfg["rollout_steps"]),
            math.ceil((int(cfg["total_steps"]) - total_steps) / len(envs)),
        )
        policy.update_normalization(torch.as_tensor(observations, device=device))
        initial_hidden = hidden.detach().cpu().numpy()
        storage = defaultdict(list)
        infos_by_step = []
        for rollout_step in range(rollout_steps):
            obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
            memory_np = np.stack([env.memory_context for env in envs])
            expert_actions_np = np.asarray(
                [
                    int(
                        training_rule_mode(
                            observations[rank],
                            memory_np[rank],
                            feasible_label=envs[rank].training_feasible_label,
                        )
                    )
                    for rank in range(len(envs))
                ],
                dtype=np.int64,
            )
            memory_tensor = torch.as_tensor(memory_np, dtype=torch.float32, device=device)
            starts_tensor = torch.as_tensor(episode_starts, dtype=torch.float32, device=device)
            with torch.no_grad():
                dist, values, next_hidden = policy.distribution_and_value(
                    obs_tensor,
                    hidden,
                    memory_tensor if policy.memory_dim else None,
                    starts_tensor,
                )
                actions = dist.sample().squeeze(0)
                probabilities = dist.probs.squeeze(0)
                logp = dist.log_prob(actions.unsqueeze(0)).squeeze(0)
                values = values.squeeze(0)
            next_observations, rewards, dones, step_infos = [], [], [], []
            next_starts = []
            for rank, env in enumerate(envs):
                action = int(actions[rank].item())
                probability = float(probabilities[rank, action].item())
                env.set_policy_metadata(probability=probability)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                action_counts[MODE_NAMES[action]] += 1
                recent_actions.append(MODE_NAMES[action])
                reward_by_mode[MODE_NAMES[action]].append(float(reward))
                step_writer.writerow(
                    {
                        "global_step": total_steps + rollout_step * len(envs) + rank,
                        "env_rank": rank,
                        "episode_number": episode_numbers[rank] - 1,
                        "scenario_category": current_specs[rank].category,
                        "reward": float(reward),
                        "termination_reason": info["termination_reason"],
                        **{field: info[field] for field in REQUIRED_STEP_LOG_FIELDS},
                    }
                )
                if done:
                    terminal_counts[info["termination_reason"]] += 1
                    episode_writer.writerow(
                        {
                            "env_rank": rank,
                            "episode_number": episode_numbers[rank] - 1,
                            "scenario_category": current_specs[rank].category,
                            "corridor_type": info.get("corridor_type", "unknown"),
                            "feasible": info["ground_truth_feasible_for_training_only"],
                            "termination_reason": info["termination_reason"],
                            "success": info["success"],
                            "collision": info["collision"],
                            "stuck": info["stuck"],
                            "timeout": info["timeout"],
                            "correct_reject": info.get("correct_reject", 0.0),
                            "false_reject": info.get("false_reject", 0.0),
                            "episode_return": info["episode_return"],
                            "selector_steps": info["selector_step"],
                        }
                    )
                    next_obs, _ = reset_env(rank)
                next_observations.append(next_obs)
                rewards.append(float(reward))
                dones.append(float(done))
                next_starts.append(float(done))
                step_infos.append(info)
            storage["obs"].append(observations.copy())
            storage["memory"].append(memory_np.copy())
            storage["starts"].append(episode_starts.copy())
            storage["actions"].append(actions.cpu().numpy())
            storage["expert_actions"].append(expert_actions_np)
            storage["logp"].append(logp.cpu().numpy())
            storage["values"].append(values.cpu().numpy())
            storage["rewards"].append(np.asarray(rewards, dtype=np.float32))
            storage["dones"].append(np.asarray(dones, dtype=np.float32))
            infos_by_step.append(step_infos)
            observations = np.stack(next_observations)
            episode_starts = np.asarray(next_starts, dtype=np.float32)
            hidden = next_hidden.detach()
            done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=device).reshape(1, -1, 1)
            hidden = hidden * (1.0 - done_tensor)

        with torch.no_grad():
            next_memory = torch.as_tensor(
                np.stack([env.memory_context for env in envs]), dtype=torch.float32, device=device
            )
            next_dist, last_values, _ = policy.distribution_and_value(
                torch.as_tensor(observations, dtype=torch.float32, device=device),
                hidden,
                next_memory if policy.memory_dim else None,
                torch.as_tensor(episode_starts, dtype=torch.float32, device=device),
            )
            del next_dist
        rollout = {key: np.asarray(value) for key, value in storage.items()}
        rollout["initial_hidden"] = initial_hidden
        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["dones"],
            rollout["values"],
            last_values.squeeze(0).cpu().numpy(),
            float(cfg["gamma"]),
            float(cfg["gae_lambda"]),
        )
        for mode_index, mode_name in enumerate(MODE_NAMES):
            mask = rollout["actions"] == mode_index
            if mask.any():
                advantage_by_mode[mode_name].extend(advantages[mask].tolist())
        advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
        diagnostics = ppo_update(
            policy, optimizer, rollout, advantages, returns, cfg, device
        )
        total_steps += rollout_steps * len(envs)
        action_total = max(1, sum(action_counts.values()))
        distribution = {
            name: action_counts[name] / action_total for name in MODE_NAMES
        }
        diagnostics.update(
            {
                "total_steps": total_steps,
                "action_distribution": distribution,
                "terminal_counts": dict(terminal_counts),
            }
        )
        update_rows.append(diagnostics)
        print(json.dumps(diagnostics, sort_keys=True))
        step_handle.flush()
        episode_handle.flush()

    step_handle.close()
    episode_handle.close()
    action_total = max(1, sum(action_counts.values()))
    recent_counts = Counter(recent_actions)
    recent_total = max(1, len(recent_actions))
    summary = {
        "initialization": initialization,
        "total_steps_actual": total_steps,
        "mode_counts": {name: action_counts[name] for name in MODE_NAMES},
        "mode_distribution": {
            name: action_counts[name] / action_total for name in MODE_NAMES
        },
        "last_10k_mode_counts": {
            name: recent_counts[name] for name in MODE_NAMES
        },
        "last_10k_mode_distribution": {
            name: recent_counts[name] / recent_total for name in MODE_NAMES
        },
        "mean_immediate_reward_by_mode": {
            name: float(np.mean(reward_by_mode[name])) if reward_by_mode[name] else None
            for name in MODE_NAMES
        },
        "mean_raw_advantage_by_mode": {
            name: float(np.mean(advantage_by_mode[name])) if advantage_by_mode[name] else None
            for name in MODE_NAMES
        },
        "terminal_counts": dict(terminal_counts),
        "collapse_over_95_percent": (
            max(recent_counts.values(), default=0) / recent_total > 0.95
        ),
        "all_modes_sampled": all(action_counts[name] > 0 for name in MODE_NAMES),
        "final_actor_head_grad_norm": update_rows[-1]["actor_head_grad_norm"],
        "ground_truth_in_actor_observation": False,
        "supervised_aux_coef": float(cfg["supervised_aux_coef"]),
    }
    torch.save(
        {
            "model": policy.state_dict(),
            "model_config": {
                "hidden_size": int(cfg["hidden_size"]),
                "memory_dim": 4 if cfg["with_memory"] else 0,
                "privileged_dim": 0,
            },
            "training_config": cfg,
            "training_summary": summary,
        },
        args.output_dir / "ppo_final.pt",
    )
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[saved] {args.output_dir / 'ppo_final.pt'}")


if __name__ == "__main__":
    main()
