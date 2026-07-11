#!/usr/bin/env python3
"""Memory transfer/interference experiment for failure-memory navigation.

This experiment extends the repeated false-feasible test by separating:

  A. same-passage repeated false-feasible passages,
  B. similar-but-new false-feasible passages, and
  C. similar-but-feasible passages.

The goal is to test whether geometry-anchored failure memory transfers beyond
identical repeated passages while avoiding false rejection on feasible passages
that look geometrically similar.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from cross_episode_memory import CrossEpisodeMemory, MemoryConfig, _fingerprint, _fp_distance
from eval_harder_benchmark import TurnCommitFSM
from eval_memory_baselines import (
    KNNFailureMemory,
    VanillaEpisodicMemory,
    _geometry_features,
)
from eval_multi_agent_memory import FF_CTYPES, PASSABLE_CTYPES, _actual_passage_width
from failure_memory import PassageFailureMemory
from procedural_env_v2 import HarderNarrowPassageEnv


RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"
DEFAULT_CSV = RESULTS / "memory_transfer_interference.csv"
DEFAULT_MD = RESULTS / "paper_table_memory_transfer_interference.md"
DEFAULT_TEX = RESULTS / "paper_table_memory_transfer_interference.tex"

METHODS = [
    "no_memory",
    "local_intra_episode_memory",
    "vanilla_episodic_memory",
    "knn_failure_memory",
    "geometry_guided_cross_episode_failure_memory",
]

PRESETS = {
    "smoke": {
        "num_seeds": 1,
        "n_rounds": 3,
        "n_same_ff": 8,
        "n_transfer_ff": 8,
        "n_feasible": 8,
        "candidate_pool": 96,
        "max_steps": 140,
    },
    "paper": {
        "num_seeds": 5,
        "n_rounds": 8,
        "n_same_ff": 40,
        "n_transfer_ff": 40,
        "n_feasible": 40,
        "candidate_pool": 600,
        "max_steps": 300,
    },
}


def _make_memory(method: str, args):
    if method == "knn_failure_memory":
        return KNNFailureMemory(
            min_neighbors=args.min_similar,
            reject_threshold=args.reject_threshold,
        )
    if method == "vanilla_episodic_memory":
        return VanillaEpisodicMemory(
            min_neighbors=args.min_similar,
            reject_threshold=args.reject_threshold,
        )
    if method == "geometry_guided_cross_episode_failure_memory":
        return CrossEpisodeMemory(
            cfg=MemoryConfig(
                fp_radius=args.geometry_fp_radius,
                min_similar_for_reject=args.min_similar,
                min_similar_for_cautious=max(1, args.min_similar),
                sr_reject_threshold=1.0 - args.reject_threshold,
            )
        )
    return None


def _entry_descriptor(env: HarderNarrowPassageEnv, seed: int) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    width = _actual_passage_width(env)
    obs_fp = obs.copy()
    obs_fp[8] = width
    ctype = env._params.ctype.value
    return {
        "seed": int(seed),
        "obs": obs_fp,
        "width": float(width),
        "ctype": ctype,
        "fingerprint": _fingerprint(obs_fp, ctype),
        "geometry": _geometry_features(obs_fp, width),
    }


def _select_nearest_seeds(
    env: HarderNarrowPassageEnv,
    reference_desc: list[dict[str, Any]],
    n: int,
    start_seed: int,
    candidate_pool: int,
    exclude_seeds: set[int] | None = None,
) -> list[int]:
    exclude_seeds = exclude_seeds or set()
    refs = [d["geometry"] for d in reference_desc]
    scored = []
    for seed in range(start_seed, start_seed + candidate_pool):
        if seed in exclude_seeds:
            continue
        desc = _entry_descriptor(env, seed)
        dist = min(float(np.linalg.norm(desc["geometry"] - ref)) for ref in refs)
        scored.append((dist, seed))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [seed for _dist, seed in scored[:n]]


def _should_reject(method: str, memory, obs: np.ndarray, width: float, ctype: str, args) -> bool:
    if method == "knn_failure_memory":
        return bool(memory.should_reject(obs, width))
    if method == "vanilla_episodic_memory":
        return bool(memory.should_reject(obs, width))
    if method == "geometry_guided_cross_episode_failure_memory":
        fp_q = _fingerprint(obs, ctype)
        similar = [
            rec
            for rec in memory._records
            if _fp_distance(rec.fingerprint, fp_q) <= args.geometry_fp_radius
        ]
        if len(similar) < args.min_similar:
            return False
        success_rate = float(sum(rec.success for rec in similar)) / len(similar)
        return success_rate < (1.0 - args.reject_threshold)
    return False


def _record_memory(method: str, memory, obs: np.ndarray, width: float, ctype: str, success: bool, steps: int) -> None:
    if method == "knn_failure_memory":
        memory.record(obs, width, success)
    elif method == "vanilla_episodic_memory":
        memory.record(obs, width, success)
    elif method == "geometry_guided_cross_episode_failure_memory":
        memory.record_episode(
            obs,
            success,
            passage_width=width,
            steps=steps,
            corridor_type=ctype,
        )


def _run_episode(
    env: HarderNarrowPassageEnv,
    seed: int,
    method: str,
    memory,
    args,
    record: bool,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    entry_obs = obs.copy()
    width = _actual_passage_width(env)
    entry_obs[8] = width
    ctype = env._params.ctype.value
    passable = ctype != "false_feasible"

    rejected = False
    if method not in ("no_memory", "local_intra_episode_memory"):
        rejected = _should_reject(method, memory, entry_obs, width, ctype, args)

    if rejected:
        if record:
            # Rejections are decisions, not traversal evidence; do not write them
            # as successful/failed traversal attempts.
            pass
        return {
            "seed": seed,
            "ctype": ctype,
            "passable": passable,
            "rejected": True,
            "attempted": False,
            "success": 0.0,
            "steps": 0,
        }

    local_mem = PassageFailureMemory() if method == "local_intra_episode_memory" else None
    fsm = TurnCommitFSM(variant="full", local_mem=local_mem, cross_mem=None)
    fsm.reset()

    steps = 0
    done = False
    any_collision = False
    info: dict[str, Any] = {"success": 0.0}
    while not done and steps < args.max_steps:
        if float(obs[16]) > 0.5:
            any_collision = True
        action, _mode = fsm.step(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        if local_mem is not None and any_collision:
            local_mem.add_failure(obs)
        steps += 1

    success = bool(float(info.get("success", 0.0)) > 0.5)
    if record and method not in ("no_memory", "local_intra_episode_memory"):
        _record_memory(method, memory, entry_obs, width, ctype, success, steps)

    return {
        "seed": seed,
        "ctype": ctype,
        "passable": passable,
        "rejected": False,
        "attempted": True,
        "success": float(success),
        "steps": int(steps),
    }


def _rate(stats: list[dict[str, Any]], key: str) -> float:
    if not stats:
        return float("nan")
    return float(np.mean([float(s[key]) for s in stats]))


def _round_summary(stats: list[dict[str, Any]]) -> dict[str, float]:
    attempted = [s for s in stats if s["attempted"]]
    rejected = [s for s in stats if s["rejected"]]
    return {
        "reject_rate": len(rejected) / max(1, len(stats)),
        "wasted_attempts": len(attempted),
        "wasted_steps": int(sum(int(s["steps"]) for s in attempted)),
    }


def _build_groups(seed: int, args) -> dict[str, list[int]]:
    same_ff_env = HarderNarrowPassageEnv(
        {"corridor_types": FF_CTYPES, "width_range": args.width_range}
    )
    transfer_ff_env = HarderNarrowPassageEnv(
        {"corridor_types": FF_CTYPES, "width_range": args.width_range}
    )
    feasible_env = HarderNarrowPassageEnv(
        {"corridor_types": PASSABLE_CTYPES, "width_range": args.width_range}
    )

    same_start = args.same_seed_offset + seed * 100_000
    transfer_start = args.transfer_seed_offset + seed * 100_000
    feasible_start = args.feasible_seed_offset + seed * 100_000

    same_ff = list(range(same_start, same_start + args.n_same_ff))
    reference = [_entry_descriptor(same_ff_env, s) for s in same_ff]
    transfer_ff = _select_nearest_seeds(
        transfer_ff_env,
        reference,
        args.n_transfer_ff,
        transfer_start,
        args.candidate_pool,
        exclude_seeds=set(same_ff),
    )
    feasible = _select_nearest_seeds(
        feasible_env,
        reference,
        args.n_feasible,
        feasible_start,
        args.candidate_pool,
    )

    for env in (same_ff_env, transfer_ff_env, feasible_env):
        if hasattr(env, "close"):
            env.close()
    return {
        "same_ff": same_ff,
        "transfer_ff": transfer_ff,
        "feasible": feasible,
    }


def run_seed_method(seed: int, method: str, args) -> dict[str, Any]:
    groups = _build_groups(seed, args)
    same_ff_env = HarderNarrowPassageEnv(
        {"corridor_types": FF_CTYPES, "width_range": args.width_range}
    )
    transfer_ff_env = HarderNarrowPassageEnv(
        {"corridor_types": FF_CTYPES, "width_range": args.width_range}
    )
    feasible_env = HarderNarrowPassageEnv(
        {"corridor_types": PASSABLE_CTYPES, "width_range": args.width_range}
    )
    memory = _make_memory(method, args)

    round_stats = []
    for round_idx in range(args.n_rounds):
        stats = [
            _run_episode(same_ff_env, s, method, memory, args, record=True)
            for s in groups["same_ff"]
        ]
        summary = _round_summary(stats)
        round_stats.append(summary)
        print(
            f"  seed={seed:02d} [{method:45s}] round={round_idx + 1:02d} "
            f"same_FF_reject={summary['reject_rate']:.3f} "
            f"wasted_attempts={summary['wasted_attempts']:3d} "
            f"wasted_steps={summary['wasted_steps']:5d}"
        )

    transfer_stats = [
        _run_episode(transfer_ff_env, s, method, memory, args, record=False)
        for s in groups["transfer_ff"]
    ]
    feasible_stats = [
        _run_episode(feasible_env, s, method, memory, args, record=False)
        for s in groups["feasible"]
    ]

    r1 = round_stats[0]
    rf = round_stats[-1]
    passable_rejects = [s for s in feasible_stats if s["rejected"]]
    transfer_rejects = [s for s in transfer_stats if s["rejected"]]
    row = {
        "preset": args.preset,
        "seed": seed,
        "method": method,
        "n_rounds": args.n_rounds,
        "n_same_false_feasible": len(groups["same_ff"]),
        "n_transfer_false_feasible": len(groups["transfer_ff"]),
        "n_similar_feasible": len(groups["feasible"]),
        "passable_success_rate": _rate(feasible_stats, "success"),
        "passable_false_reject_rate": len(passable_rejects) / max(1, len(feasible_stats)),
        "false_feasible_reject_rate_round1": r1["reject_rate"],
        "false_feasible_reject_rate_final_round": rf["reject_rate"],
        "wasted_false_feasible_attempts_round1": int(r1["wasted_attempts"]),
        "wasted_false_feasible_attempts_final_round": int(rf["wasted_attempts"]),
        "wasted_false_feasible_steps_round1": int(r1["wasted_steps"]),
        "wasted_false_feasible_steps_final_round": int(rf["wasted_steps"]),
        "transfer_reject_rate_on_similar_new_false_feasible": len(transfer_rejects)
        / max(1, len(transfer_stats)),
        "interference_false_reject_rate_on_similar_feasible": len(passable_rejects)
        / max(1, len(feasible_stats)),
    }

    for env in (same_ff_env, transfer_ff_env, feasible_env):
        if hasattr(env, "close"):
            env.close()
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    summary = []
    metric_keys = [
        "passable_success_rate",
        "passable_false_reject_rate",
        "false_feasible_reject_rate_round1",
        "false_feasible_reject_rate_final_round",
        "wasted_false_feasible_attempts_round1",
        "wasted_false_feasible_attempts_final_round",
        "transfer_reject_rate_on_similar_new_false_feasible",
        "interference_false_reject_rate_on_similar_feasible",
    ]
    for method in METHODS:
        vals = by_method.get(method, [])
        if not vals:
            continue
        out = {"method": method, "num_seeds": len(vals)}
        for key in metric_keys:
            arr = np.asarray([float(v[key]) for v in vals], dtype=np.float32)
            out[f"{key}_mean"] = float(np.mean(arr))
            out[f"{key}_std"] = float(np.std(arr))
        summary.append(out)
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def _fmt_mean_std(row: dict[str, Any], key: str, percent: bool = True) -> str:
    mean = float(row[f"{key}_mean"])
    std = float(row[f"{key}_std"])
    if percent:
        return f"{100.0 * mean:.1f} +/- {100.0 * std:.1f}"
    return f"{mean:.1f} +/- {std:.1f}"


def _fmt_mean_std_latex(row: dict[str, Any], key: str, percent: bool = True) -> str:
    mean = float(row[f"{key}_mean"])
    std = float(row[f"{key}_std"])
    if percent:
        return f"{100.0 * mean:.1f} $\\pm$ {100.0 * std:.1f}"
    return f"{mean:.1f} $\\pm$ {std:.1f}"


def write_markdown(summary: list[dict[str, Any]], path: Path, preset: str) -> None:
    lines = [
        "# Table: Memory Transfer and Interference",
        "",
        "This table tests whether failure memory transfers from repeated false-feasible passages to similar new false-feasible passages, and whether that transfer causes false rejection on similar feasible passages.",
        "",
        f"Preset: `{preset}`. Use `--preset paper` for the paper-scale run; `smoke` is a quick reproducibility check.",
        "",
        "| Method | Seeds | Passable SR (up) | Passable false reject (down) | FF reject R1 (up) | FF reject final (up) | Wasted FF attempts R1 (down) | Wasted FF attempts final (down) | Transfer reject on new FF (up) | Interference false reject on feasible (down) |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {method} | {seeds} | {psr} | {pfr} | {ffr1} | {ffrf} | {wa1} | {waf} | {tr} | {ifr} |".format(
                method=row["method"],
                seeds=row["num_seeds"],
                psr=_fmt_mean_std(row, "passable_success_rate"),
                pfr=_fmt_mean_std(row, "passable_false_reject_rate"),
                ffr1=_fmt_mean_std(row, "false_feasible_reject_rate_round1"),
                ffrf=_fmt_mean_std(row, "false_feasible_reject_rate_final_round"),
                wa1=_fmt_mean_std(row, "wasted_false_feasible_attempts_round1", percent=False),
                waf=_fmt_mean_std(row, "wasted_false_feasible_attempts_final_round", percent=False),
                tr=_fmt_mean_std(row, "transfer_reject_rate_on_similar_new_false_feasible"),
                ifr=_fmt_mean_std(row, "interference_false_reject_rate_on_similar_feasible"),
            )
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_latex(summary: list[dict[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Method & Seeds & Pass SR & Pass FR & FF rej R1 & FF rej final & Waste R1 & Waste final & Transfer & Interference \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _tex_escape(row["method"]),
                row["num_seeds"],
                _fmt_mean_std_latex(row, "passable_success_rate"),
                _fmt_mean_std_latex(row, "passable_false_reject_rate"),
                _fmt_mean_std_latex(row, "false_feasible_reject_rate_round1"),
                _fmt_mean_std_latex(row, "false_feasible_reject_rate_final_round"),
                _fmt_mean_std_latex(row, "wasted_false_feasible_attempts_round1", percent=False),
                _fmt_mean_std_latex(row, "wasted_false_feasible_attempts_final_round", percent=False),
                _fmt_mean_std_latex(row, "transfer_reject_rate_on_similar_new_false_feasible"),
                _fmt_mean_std_latex(row, "interference_false_reject_rate_on_similar_feasible"),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))
    print(f"[write] {path}")


def _apply_preset(args) -> None:
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.width_range = (args.width_min, args.width_max)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--n-rounds", type=int, default=None)
    parser.add_argument("--n-same-ff", type=int, default=None)
    parser.add_argument("--n-transfer-ff", type=int, default=None)
    parser.add_argument("--n-feasible", type=int, default=None)
    parser.add_argument("--candidate-pool", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--min-similar", type=int, default=2)
    parser.add_argument("--reject-threshold", type=float, default=0.80)
    parser.add_argument("--geometry-fp-radius", type=int, default=0)
    parser.add_argument("--width-min", type=float, default=0.55)
    parser.add_argument("--width-max", type=float, default=0.80)
    parser.add_argument("--same-seed-offset", type=int, default=0)
    parser.add_argument("--transfer-seed-offset", type=int, default=10_000)
    parser.add_argument("--feasible-seed-offset", type=int, default=20_000)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_TEX)
    args = parser.parse_args()
    _apply_preset(args)
    return args


def main() -> None:
    args = parse_args()
    print(
        f"[memory_transfer_interference] preset={args.preset} "
        f"seeds={args.seed}..{args.seed + args.num_seeds - 1} "
        f"rounds={args.n_rounds} same_ff={args.n_same_ff} "
        f"transfer_ff={args.n_transfer_ff} feasible={args.n_feasible}"
    )
    rows = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        for method in args.methods:
            print(f"\n=== seed={seed} method={method} ===")
            rows.append(run_seed_method(seed, method, args))

    write_csv(rows, args.output_csv)
    summary = summarize(rows)
    write_markdown(summary, args.output_md, args.preset)
    write_latex(summary, args.output_tex)


if __name__ == "__main__":
    main()
