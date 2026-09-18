#!/usr/bin/env python3
"""Fixed-episode evaluation with numerator/denominator selector metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import HarderNarrowPassageEnv
from narrow_passage.envs.four_mode_selector_env import REQUIRED_STEP_LOG_FIELDS, FourModeSelectorEnv
from narrow_passage.models.belief_state import BeliefState
from narrow_passage.models.policy import choose_mode
from narrow_passage.selector.contract import MODE_NAMES, Mode, assert_selector_contract
from narrow_passage.selector.expert import training_rule_mode
from narrow_passage.selector.model import FourModeGRUPolicy
from narrow_passage.selector.sampling import BalancedScenarioSampler, deterministic_category
from train_mode_selector_bc import confusion_metrics


def ratio(numerator: int, denominator: int) -> dict[str, float | int]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": float(numerator / denominator) if denominator else 0.0,
    }


def rule_action(obs, memory) -> tuple[int, float]:
    belief = BeliefState.from_obs(obs, memory_risk=float(memory[3]))
    risk = float(np.clip(1.0 - belief.p_feas + belief.memory_risk, 0.0, 1.0))
    decision = choose_mode(belief.p_feas, risk, belief.stuck_score, belief.memory_risk)
    mode = Mode[decision.mode.name]
    return int(mode), 1.0


def load_policy(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_cfg = checkpoint["model_config"]
    policy = FourModeGRUPolicy(**model_cfg).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    return policy, checkpoint


def evaluate(args):
    device = torch.device(args.device)
    policy = checkpoint = None
    if args.method != "rule_19d":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for learned methods")
        policy, checkpoint = load_policy(args.checkpoint, device)
    sampler = BalancedScenarioSampler(args.seed)
    env = FourModeSelectorEnv(
        HarderNarrowPassageEnv({"seed": args.seed, "max_steps": args.max_steps})
    )
    episode_rows, step_rows = [], []
    target_labels, predictions, feasibility_steps = [], [], []
    global_modes: Counter[str] = Counter()

    for episode_index in range(args.episodes):
        category = deterministic_category(episode_index)
        spec = sampler.sample_category(category)
        use_memory = args.method == "rule_19d_memory" or (
            policy is not None and policy.memory_dim == 4
        )
        env.set_memory_context(spec.memory_context if use_memory else None)
        episode_seed = args.seed + episode_index * 1009
        obs, reset_info = env.reset(seed=episode_seed, options=spec.reset_options)
        assert_selector_contract(obs, env.action_space)
        feasible = bool(reset_info["ground_truth_feasible_for_training_only"])
        hidden = policy.initial_hidden(1, device) if policy is not None else None
        episode_start = torch.ones(1, device=device)
        done = False
        mode_counts: Counter[str] = Counter()
        step_id = 0
        path_length = 0.0
        start_distance = float(obs[12])
        previous_position = env.env.pose[:2].copy()
        last_info = {}
        repeated_commit = 0
        recovered = False

        while not done and step_id < args.max_steps:
            teacher = training_rule_mode(
                obs, env.memory_context, feasible_label=feasible
            )
            if args.method.startswith("rule_19d"):
                action, probability = rule_action(obs, env.memory_context)
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                memory_tensor = torch.as_tensor(env.memory_context, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    dist, _value, hidden = policy.distribution_and_value(
                        obs_tensor,
                        hidden,
                        memory_tensor if policy.memory_dim else None,
                        episode_start,
                    )
                    probs = dist.probs.squeeze(0).squeeze(0)
                    action = int(probs.argmax().item())
                    probability = float(probs[action].item())
                episode_start.zero_()
            env.set_policy_metadata(probability=probability)
            next_obs, reward, terminated, truncated, info = env.step(action)
            position = env.env.pose[:2].copy()
            path_length += float(np.linalg.norm(position - previous_position))
            previous_position = position
            mode_name = MODE_NAMES[action]
            mode_counts[mode_name] += 1
            global_modes[mode_name] += 1
            recovered = recovered or action == int(Mode.RECOVER)
            repeated_commit += int(action == int(Mode.COMMIT) and env.memory_context[1] > 0)
            target_labels.append(int(teacher))
            predictions.append(action)
            feasibility_steps.append(int(feasible))
            step_rows.append(
                {
                    "episode_index": episode_index,
                    "step_id": step_id,
                    "category": category,
                    "target_mode": teacher.name,
                    "reward": float(reward),
                    "termination_reason": info["termination_reason"],
                    **{field: info[field] for field in REQUIRED_STEP_LOG_FIELDS},
                }
            )
            obs = next_obs
            last_info = info
            done = bool(terminated or truncated)
            step_id += 1

        success = int(float(last_info.get("success", 0.0)) > 0.5)
        spl = success * start_distance / max(start_distance, path_length, 1e-8)
        episode_rows.append(
            {
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "method": args.method,
                "category": category,
                "corridor_type": spec.reset_options["corridor_type"],
                "feasible": int(feasible),
                "success": success,
                "correct_reject": int(float(last_info.get("correct_reject", 0.0)) > 0.5),
                "false_reject": int(float(last_info.get("false_reject", 0.0)) > 0.5),
                "collision": int(float(last_info.get("collision", 0.0)) > 0.5),
                "stuck": int(float(last_info.get("stuck", 0.0)) > 0.5),
                "timeout": int(float(last_info.get("timeout", 0.0)) > 0.5),
                "recovery_attempted": int(recovered),
                "recovery_success": int(recovered and success),
                "repeated_commitment_count": repeated_commit,
                "episode_return": float(last_info.get("episode_return", 0.0)),
                "steps": step_id,
                "path_length": path_length,
                "spl": spl,
                "termination_reason": last_info.get("termination_reason", "limit"),
                **{f"{name.lower()}_count": mode_counts[name] for name in MODE_NAMES},
            }
        )

    feasible_rows = [row for row in episode_rows if row["feasible"] == 1]
    infeasible_rows = [row for row in episode_rows if row["feasible"] == 0]
    recover_rows = [row for row in episode_rows if row["recovery_attempted"]]
    total_steps = sum(global_modes.values())
    metrics = {
        "method": args.method,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "seed": args.seed,
        "episodes": len(episode_rows),
        "feasible_success": ratio(sum(row["success"] for row in feasible_rows), len(feasible_rows)),
        "correct_rejection": ratio(sum(row["correct_reject"] for row in infeasible_rows), len(infeasible_rows)),
        "false_rejection": ratio(sum(row["false_reject"] for row in feasible_rows), len(feasible_rows)),
        "collision": ratio(sum(row["collision"] for row in episode_rows), len(episode_rows)),
        "timeout": ratio(sum(row["timeout"] for row in episode_rows), len(episode_rows)),
        "stuck": ratio(sum(row["stuck"] for row in episode_rows), len(episode_rows)),
        "recovery_success": ratio(sum(row["recovery_success"] for row in recover_rows), len(recover_rows)),
        "repeated_commitment_count": int(sum(row["repeated_commitment_count"] for row in episode_rows)),
        "mode_usage": {name: ratio(global_modes[name], total_steps) for name in MODE_NAMES},
        "mean_episode_return": float(np.mean([row["episode_return"] for row in episode_rows])),
        "mean_spl": float(np.mean([row["spl"] for row in episode_rows])),
        "mode_classification": confusion_metrics(
            np.asarray(target_labels), np.asarray(predictions), np.asarray(feasibility_steps)
        ),
        "domain": "procedural geometry simulation",
        "actor_ground_truth_input": False,
    }
    return episode_rows, step_rows, metrics


def write_csv(rows, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(metrics, path):
    lines = [
        f"# Selector evaluation: {metrics['method']}",
        "",
        f"Domain: {metrics['domain']}. Ground-truth feasibility is not an Actor input.",
        "",
        "| Metric | Numerator | Denominator | Rate |",
        "|:---|---:|---:|---:|",
    ]
    for name in ("feasible_success", "correct_rejection", "false_rejection", "collision", "timeout", "stuck", "recovery_success"):
        row = metrics[name]
        lines.append(f"| {name} | {row['numerator']} | {row['denominator']} | {row['rate']:.4f} |")
    lines.extend(["", "| Mode | Count | Total steps | Usage |", "|:---|---:|---:|---:|"])
    for name in MODE_NAMES:
        row = metrics["mode_usage"][name]
        lines.append(f"| {name} | {row['numerator']} | {row['denominator']} | {row['rate']:.4f} |")
    lines.extend(
        [
            "",
            f"Mean episode return: {metrics['mean_episode_return']:.4f}",
            f"Mean SPL (auxiliary): {metrics['mean_spl']:.4f}",
            f"Repeated commitment count: {metrics['repeated_commitment_count']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["rule_19d", "rule_19d_memory", "bc_gru", "ppo_gru", "bc_ppo_gru", "bc_ppo_gru_memory"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=90001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows, step_rows, metrics = evaluate(args)
    write_csv(episode_rows, args.output_dir / "episodes.csv")
    write_csv(step_rows, args.output_dir / "steps.csv")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    write_markdown(metrics, args.output_dir / "metrics.md")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"[write] {args.output_dir}")


if __name__ == "__main__":
    main()
