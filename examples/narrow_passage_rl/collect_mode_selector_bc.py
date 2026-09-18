#!/usr/bin/env python3
"""Collect leakage-safe four-mode BC supervision from procedural-v2.

The simulator feasibility bit is saved as a training/evaluation label only.  It
is never concatenated with `obs_19d` or the independent four-value memory input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import HarderNarrowPassageEnv
from narrow_passage.envs.four_mode_selector_env import FourModeSelectorEnv
from narrow_passage.selector.contract import MODE_NAMES, assert_selector_contract
from narrow_passage.selector.expert import training_rule_mode
from narrow_passage.selector.sampling import (
    BalancedScenarioSampler,
    SAMPLING_RATIOS,
    deterministic_category,
)


def group_split(category_occurrence: int) -> str:
    """80/10/10 split at whole-episode level within each scenario stratum."""

    slot = int(category_occurrence) % 10
    if slot == 8:
        return "validation"
    if slot == 9:
        return "test"
    return "train"


def collect(args) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict]:
    sampler = BalancedScenarioSampler(args.seed)
    base = HarderNarrowPassageEnv({"seed": args.seed, "max_steps": args.max_steps})
    env = FourModeSelectorEnv(base)
    category_occurrences: defaultdict[str, int] = defaultdict(int)
    transition_rows: list[dict[str, object]] = []
    arrays: dict[str, list] = defaultdict(list)
    mode_counts: Counter[str] = Counter()
    episode_counts: Counter[str] = Counter()

    for episode_index in range(args.episodes):
        category = deterministic_category(episode_index)
        spec = sampler.sample_category(category)
        occurrence = category_occurrences[category]
        category_occurrences[category] += 1
        split = group_split(occurrence)
        episode_counts[split] += 1
        episode_seed = int(args.seed + 1009 * episode_index)
        episode_id = f"procedural_selector_{args.seed}_{episode_index:06d}"
        env.set_memory_context(spec.memory_context if args.with_memory else None)
        obs, reset_info = env.reset(seed=episode_seed, options=spec.reset_options)
        assert_selector_contract(obs, env.action_space)
        feasible_value = int(reset_info["ground_truth_feasible_for_training_only"])
        feasible = None if feasible_value < 0 else bool(feasible_value)
        scene_id = f"procedural_v2/{spec.reset_options['corridor_type']}"
        done = False
        step_id = 0

        while not done and step_id < args.max_steps:
            rule_mode = training_rule_mode(
                obs,
                env.memory_context,
                feasible_label=feasible,
            )
            arrays["obs_19d"].append(obs.copy())
            arrays["memory_context"].append(env.memory_context)
            arrays["rule_mode"].append(int(rule_mode))
            arrays["episode_index"].append(episode_index)
            arrays["step_id"].append(step_id)
            arrays["feasibility_label"].append(feasible_value)
            arrays["split"].append(split)
            arrays["episode_id"].append(episode_id)
            arrays["scene_id"].append(scene_id)
            arrays["category"].append(category)
            mode_counts[rule_mode.name] += 1

            env.set_policy_metadata(probability=1.0, override_reason="rule_teacher")
            next_obs, reward, terminated, truncated, info = env.step(int(rule_mode))
            transition_rows.append(
                {
                    "episode_id": episode_id,
                    "scene_id": scene_id,
                    "split": split,
                    "episode_index": episode_index,
                    "step_id": step_id,
                    "category": category,
                    "corridor_type": spec.reset_options["corridor_type"],
                    "feasibility_label": feasible_value,
                    "rule_mode": rule_mode.name,
                    "rule_mode_index": int(rule_mode),
                    "reward": float(reward),
                    "termination_reason": info["termination_reason"],
                    **{name: info[name] for name in (
                        "raw_policy_mode",
                        "executed_mode",
                        "mode_probability",
                        "mode_override_reason",
                        "ground_truth_feasible_for_training_only",
                        "estimated_margin",
                        "alignment_ready",
                        "memory_support",
                        "collision",
                        "stuck",
                        "timeout",
                        "success",
                    )},
                }
            )
            obs = next_obs
            done = bool(terminated or truncated)
            step_id += 1

    packed = {
        "obs_19d": np.asarray(arrays["obs_19d"], dtype=np.float32),
        "memory_context": np.asarray(arrays["memory_context"], dtype=np.float32),
        "rule_mode": np.asarray(arrays["rule_mode"], dtype=np.int64),
        "episode_index": np.asarray(arrays["episode_index"], dtype=np.int64),
        "step_id": np.asarray(arrays["step_id"], dtype=np.int64),
        "feasibility_label": np.asarray(arrays["feasibility_label"], dtype=np.int8),
        "split": np.asarray(arrays["split"]),
        "episode_id": np.asarray(arrays["episode_id"]),
        "scene_id": np.asarray(arrays["scene_id"]),
        "category": np.asarray(arrays["category"]),
    }
    metadata = {
        "seed": args.seed,
        "episodes": args.episodes,
        "transitions": len(packed["rule_mode"]),
        "max_steps": args.max_steps,
        "with_memory": bool(args.with_memory),
        "mode_mapping": {str(i): name for i, name in enumerate(MODE_NAMES)},
        "mode_counts": dict(mode_counts),
        "episode_split_counts": dict(episode_counts),
        "sampling_ratios": SAMPLING_RATIOS,
        "split_unit": "whole procedural episode, stratified by scenario category",
        "actor_ground_truth_input": False,
    }
    return packed, transition_rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--with-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/narrow_passage_rl/results/mode_selector/bc_dataset_v1"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packed, rows, metadata = collect(args)
    np.savez_compressed(args.output_dir / "mode_selector_bc.npz", **packed)
    with (args.output_dir / "transitions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"[write] {args.output_dir}")


if __name__ == "__main__":
    main()
