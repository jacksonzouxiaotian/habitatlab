#!/usr/bin/env python3
"""Evaluate DEGNAV-RL belief-mode PPO on the procedural v2 benchmark.

This script is intentionally procedural-only.  Habitat evaluation for
DEGNAV-RL should be added separately once the mode-selection policy is wired to
the Habitat controller interface.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import HarderNarrowPassageEnv
from narrow_passage.envs.belief_mode_env import (
    VALID_BELIEF_ABLATIONS,
    BeliefModeEnv,
    MODE_ORDER,
)
from narrow_passage.metrics.strict_metrics import compute_strict_metrics


CTYPE_CONFIGS = {
    "full": None,
    "no_ff": [
        "STRAIGHT",
        "L_SHAPED",
        "S_SHAPED",
        "NARROW_EXIT",
        "NARROW_ENTRY",
        "ASYMMETRIC",
    ],
    "straight_only": ["STRAIGHT"],
    "hard": ["L_SHAPED", "S_SHAPED", "NARROW_EXIT", "NARROW_ENTRY", "ASYMMETRIC"],
    "false_feasible_only": ["FALSE_FEASIBLE"],
}

CSV_FIELDS = [
    "episode",
    "seed",
    "ablation",
    "corridor_type",
    "success",
    "strict_success",
    "collision",
    "near_collision",
    "reject",
    "correct_reject",
    "false_reject",
    "timeout",
    "min_clearance",
    "final_p_feas",
    "final_delta_mean",
    "final_delta_var",
    "final_d_hat",
    "final_w_req_cons",
    "final_memory_risk",
    "final_risk",
    "d_hat",
    "w_req_cons",
    "delta_mean",
    "delta_var",
    "p_feas",
    "mode",
    "risk",
    "mode_commit_count",
    "mode_explore_count",
    "mode_recover_count",
    "mode_reject_count",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def env_config(ctypes: str, max_steps: int | None = None) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    corridor_types = CTYPE_CONFIGS[ctypes]
    if corridor_types is not None:
        cfg["corridor_types"] = corridor_types
    if max_steps is not None:
        cfg["max_steps"] = int(max_steps)
    return cfg


def make_env(ctypes: str, max_steps: int, ablation: str = "full") -> BeliefModeEnv:
    return BeliefModeEnv(
        HarderNarrowPassageEnv(env_config(ctypes, max_steps)),
        ablation=ablation,
    )


def _corridor_type(env: BeliefModeEnv, info: dict[str, Any]) -> str:
    if "corridor_type" in info:
        return str(info["corridor_type"])
    ctype = getattr(env.env, "corridor_type", None)
    if ctype is None:
        return "?"
    return str(getattr(ctype, "value", ctype))


def _float(info: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(info.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _mode_counter_row(mode_counts: Counter[str]) -> dict[str, int]:
    return {
        "mode_commit_count": int(mode_counts.get("commit", 0)),
        "mode_explore_count": int(mode_counts.get("explore", 0)),
        "mode_recover_count": int(mode_counts.get("recover", 0)),
        "mode_reject_count": int(mode_counts.get("reject", 0)),
    }


def evaluate_model(
    model,
    episodes: int,
    seed: int = 0,
    ctypes: str = "full",
    max_steps: int = 400,
    deterministic: bool = True,
    ablation: str = "full",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run procedural-v2 evaluation and return per-episode rows + summary."""

    env = make_env(ctypes, max_steps, ablation)
    rows: list[dict[str, Any]] = []
    global_mode_counts: Counter[str] = Counter()

    for ep in range(int(episodes)):
        ep_seed = int(seed + ep)
        obs, _ = env.reset(seed=ep_seed)
        done = False
        steps = 0
        last_info: dict[str, Any] = {}
        mode_counts: Counter[str] = Counter()
        min_clearance = float("inf")
        any_reject = False
        any_correct_reject = False
        any_false_reject = False
        timeout = False
        truncated = False

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, info = env.step(action)
            last_info = dict(info)
            steps += 1

            mode_name = str(last_info.get("mode_name", "?")).lower()
            mode_counts[mode_name] += 1
            global_mode_counts[mode_name] += 1
            any_reject = any_reject or _float(last_info, "reject") > 0.5
            any_correct_reject = (
                any_correct_reject or _float(last_info, "correct_reject") > 0.5
            )
            any_false_reject = (
                any_false_reject or _float(last_info, "false_reject") > 0.5
            )
            min_clearance = min(min_clearance, _float(last_info, "min_clearance", np.inf))
            done = bool(terminated or truncated)

        if not done and steps >= max_steps:
            timeout = True
        else:
            timeout = bool(
                truncated
                or (
                    steps >= max_steps
                    and _float(last_info, "success") <= 0.5
                    and not any_reject
                )
            )
        if not np.isfinite(min_clearance):
            min_clearance = _float(last_info, "body_margin")
        strict_metrics = compute_strict_metrics(
            success=_float(last_info, "success"),
            collision=_float(last_info, "collision"),
            stuck=_float(last_info, "stuck"),
            min_clearance=min_clearance,
        )

        row = {
            "episode": ep,
            "seed": ep_seed,
            "ablation": ablation,
            "corridor_type": _corridor_type(env, last_info),
            "success": _float(last_info, "success"),
            "strict_success": strict_metrics["strict_success"],
            "collision": _float(last_info, "collision"),
            "near_collision": strict_metrics["near_collision"],
            "reject": float(any_reject),
            "correct_reject": float(any_correct_reject),
            "false_reject": float(any_false_reject),
            "timeout": float(timeout),
            "min_clearance": float(min_clearance),
            "final_p_feas": _float(last_info, "p_feas"),
            "final_delta_mean": _float(last_info, "delta_mean"),
            "final_delta_var": _float(last_info, "delta_var"),
            "final_d_hat": _float(last_info, "d_hat"),
            "final_w_req_cons": _float(last_info, "w_req_cons"),
            "final_memory_risk": _float(last_info, "memory_risk"),
            "final_risk": _float(last_info, "risk"),
            # Common aliases used by cross-method margin/phase visualizations.
            "d_hat": _float(last_info, "d_hat"),
            "w_req_cons": _float(last_info, "w_req_cons"),
            "delta_mean": _float(last_info, "delta_mean"),
            "delta_var": _float(last_info, "delta_var"),
            "p_feas": _float(last_info, "p_feas"),
            "mode": str(last_info.get("mode_name", "")),
            "risk": _float(last_info, "risk"),
            **_mode_counter_row(mode_counts),
        }
        rows.append(row)

    return rows, summarize(rows, global_mode_counts)


def summarize(
    rows: list[dict[str, Any]],
    mode_counts: Counter[str] | None = None,
) -> dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "per_corridor": {},
            "mode_counts": {},
            "mode_distribution": {},
        }

    mode_counts = mode_counts or Counter()
    if not mode_counts:
        for row in rows:
            mode_counts.update(
                {
                    "commit": int(row["mode_commit_count"]),
                    "explore": int(row["mode_explore_count"]),
                    "recover": int(row["mode_recover_count"]),
                    "reject": int(row["mode_reject_count"]),
                }
            )

    def mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    by_corridor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_corridor[str(row["corridor_type"])].append(row)

    per_corridor = {}
    for ctype, vals in sorted(by_corridor.items()):
        per_corridor[ctype] = {
            "episodes": len(vals),
            "success": float(np.mean([float(v["success"]) for v in vals])),
            "strict_success": float(np.mean([float(v["strict_success"]) for v in vals])),
        }

    total_modes = max(1, sum(mode_counts.values()))
    return {
        "episodes": len(rows),
        "success": mean("success"),
        "strict_success": mean("strict_success"),
        "collision": mean("collision"),
        "near_collision": mean("near_collision"),
        "reject": mean("reject"),
        "correct_reject": mean("correct_reject"),
        "false_reject": mean("false_reject"),
        "timeout": mean("timeout"),
        "avg_min_clearance": mean("min_clearance"),
        "per_corridor": per_corridor,
        "mode_counts": dict(mode_counts),
        "mode_distribution": {
            mode: count / total_modes for mode, count in sorted(mode_counts.items())
        },
        "ablation": str(rows[0].get("ablation", "full")),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n=== DEGNAV-RL belief-mode PPO procedural v2 ===")
    print(f"  episodes:             {summary.get('episodes', 0)}")
    print(f"  ablation:             {summary.get('ablation', 'full')}")
    print(f"  success_rate:         {summary.get('success', 0.0):.3f}")
    print(f"  strict_success_rate:  {summary.get('strict_success', 0.0):.3f}")
    print(f"  collision_rate:       {summary.get('collision', 0.0):.3f}")
    print(f"  near_collision_rate:  {summary.get('near_collision', 0.0):.3f}")
    print(f"  reject_rate:          {summary.get('reject', 0.0):.3f}")
    print(f"  correct_reject_rate:  {summary.get('correct_reject', 0.0):.3f}")
    print(f"  false_reject_rate:    {summary.get('false_reject', 0.0):.3f}")
    print(f"  timeout_rate:         {summary.get('timeout', 0.0):.3f}")
    print(f"  avg_min_clearance:    {summary.get('avg_min_clearance', 0.0):.3f}")

    print("\nPer-corridor success:")
    for ctype, stats in summary.get("per_corridor", {}).items():
        print(
            f"  {ctype:20s}: SR={stats['success']:.3f} "
            f"strict={stats['strict_success']:.3f} ({stats['episodes']} eps)"
        )

    print("\nMode distribution:")
    for mode, frac in summary.get("mode_distribution", {}).items():
        count = summary.get("mode_counts", {}).get(mode, 0)
        print(f"  {mode:8s}: {count:6d}  {frac:.3f}")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DEGNAV-RL Procedural v2 Evaluation",
        "",
        "| Metric | Value |",
        "|:---|---:|",
    ]
    for key in [
        "episodes",
        "ablation",
        "success",
        "strict_success",
        "collision",
        "near_collision",
        "reject",
        "correct_reject",
        "false_reject",
        "timeout",
        "avg_min_clearance",
    ]:
        value = summary.get(key, 0.0)
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Per-Corridor Success",
            "",
            "| Corridor | Episodes | Success | Strict success |",
            "|:---|---:|---:|---:|",
        ]
    )
    for ctype, stats in summary.get("per_corridor", {}).items():
        lines.append(
            f"| {ctype} | {stats['episodes']} | "
            f"{stats['success']:.4f} | {stats['strict_success']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Mode Distribution",
            "",
            "| Mode | Count | Fraction |",
            "|:---|---:|---:|",
        ]
    )
    for mode, frac in summary.get("mode_distribution", {}).items():
        count = summary.get("mode_counts", {}).get(mode, 0)
        lines.append(f"| {mode} | {count} | {frac:.4f} |")

    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ctypes", choices=list(CTYPE_CONFIGS), default="full")
    parser.add_argument(
        "--ablation",
        choices=list(VALID_BELIEF_ABLATIONS),
        default="full",
        help="Belief-state ablation used by the trained model.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "examples/narrow_passage_rl/results/narrow_passage_rl/"
            "degnav_rl_belief_mode_ppo_eval.csv"
        ),
    )
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic policy actions. Use --no-deterministic for stochastic rollout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError("Install stable-baselines3 to evaluate DEGNAV-RL") from exc

    model = PPO.load(str(args.model), device="cpu")
    rows, summary = evaluate_model(
        model=model,
        episodes=args.episodes,
        seed=args.seed,
        ctypes=args.ctypes,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        ablation=args.ablation,
    )
    print_summary(summary)

    output_md = args.output_md
    if output_md is None:
        output_md = args.output_csv.with_suffix(".md")
    write_csv(rows, args.output_csv)
    write_markdown(summary, output_md)


if __name__ == "__main__":
    main()
