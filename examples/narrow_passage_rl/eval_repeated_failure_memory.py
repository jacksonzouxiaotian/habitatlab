#!/usr/bin/env python3
"""Repeated false-feasible passage experiment for failure memory.

The memory contribution is not one-shot nominal success.  This experiment
replays the same passable and false-feasible corridor seeds across multiple
rounds and measures whether a method suppresses repeated commitment to passages
that already failed.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from cross_episode_memory import CrossEpisodeMemory, MemoryConfig
from eval_harder_benchmark import TurnCommitFSM
from eval_memory_baselines import (
    KNNFailureMemory,
    VanillaEpisodicMemory,
    _run_episode_with_memory,
)
from eval_multi_agent_memory import (
    FF_CTYPES,
    PASSABLE_CTYPES,
    run_one_episode,
)
from failure_memory import PassageFailureMemory
from procedural_env_v2 import HarderNarrowPassageEnv


RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"
DEFAULT_CSV = RESULTS / "repeated_failure_memory.csv"
DEFAULT_MD = RESULTS / "paper_table_repeated_failure_memory.md"
DEFAULT_TEX = RESULTS / "paper_table_repeated_failure_memory.tex"

METHODS = [
    "no_memory",
    "local_intra_episode_memory",
    "knn_failure_memory",
    "vanilla_episodic_memory",
    "geometry_guided_cross_episode_failure_memory",
]


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
                fp_radius=0,
                min_similar_for_reject=args.min_similar,
                min_similar_for_cautious=max(1, args.min_similar),
                sr_reject_threshold=1.0 - args.reject_threshold,
            )
        )
    return None


def _run_nomemory_episode(env, seed: int, max_steps: int, use_local: bool) -> Dict:
    local_mem = PassageFailureMemory() if use_local else None
    fsm = TurnCommitFSM(variant="full", local_mem=local_mem, cross_mem=None)
    return run_one_episode(
        env,
        seed,
        fsm,
        local_mem=local_mem,
        cross_mem=None,
        max_steps=max_steps,
    )


def _run_memory_episode(env, seed: int, method: str, memory, max_steps: int) -> Dict:
    adapter_method = method
    if method == "geometry_guided_cross_episode_failure_memory":
        adapter_method = "geometry_guided_failure_memory"
    return _run_episode_with_memory(env, seed, adapter_method, memory, max_steps)


def run_condition(method: str, args) -> List[Dict]:
    passable_env = HarderNarrowPassageEnv({"corridor_types": PASSABLE_CTYPES})
    ff_env = HarderNarrowPassageEnv({"corridor_types": FF_CTYPES})
    memory = _make_memory(method, args)
    passable_seeds = list(range(args.n_passable))
    ff_seeds = list(range(args.n_ff))
    rows = []

    for round_idx in range(args.n_rounds):
        pass_stats = []
        ff_stats = []

        for seed in passable_seeds:
            seed_i = args.seed_offset + seed
            if method == "no_memory":
                stat = _run_nomemory_episode(passable_env, seed_i, args.max_steps, use_local=False)
            elif method == "local_intra_episode_memory":
                stat = _run_nomemory_episode(passable_env, seed_i, args.max_steps, use_local=True)
            else:
                stat = _run_memory_episode(passable_env, seed_i, method, memory, args.max_steps)
            pass_stats.append(stat)

        for seed in ff_seeds:
            seed_i = args.seed_offset + seed
            if method == "no_memory":
                stat = _run_nomemory_episode(ff_env, seed_i, args.max_steps, use_local=False)
            elif method == "local_intra_episode_memory":
                stat = _run_nomemory_episode(ff_env, seed_i, args.max_steps, use_local=True)
            else:
                stat = _run_memory_episode(ff_env, seed_i, method, memory, args.max_steps)
            ff_stats.append(stat)

        pass_rejects = [s for s in pass_stats if s.get("rejected", False)]
        ff_rejects = [s for s in ff_stats if s.get("rejected", False)]
        ff_attempts = [s for s in ff_stats if not s.get("rejected", False)]
        rejected_total = len(pass_rejects) + len(ff_rejects)
        retrieval_precision = (
            len(ff_rejects) / rejected_total if rejected_total > 0 else float("nan")
        )
        row = {
            "method": method,
            "round": round_idx + 1,
            "n_passable": len(pass_stats),
            "n_false_feasible": len(ff_stats),
            "passable_success_rate": float(np.mean([s["success"] for s in pass_stats])),
            "passable_false_reject_rate": len(pass_rejects) / max(1, len(pass_stats)),
            "false_feasible_reject_rate": len(ff_rejects) / max(1, len(ff_stats)),
            "false_feasible_attempt_rate": len(ff_attempts) / max(1, len(ff_stats)),
            "wasted_steps_false_feasible": int(sum(s["steps"] for s in ff_attempts)),
            "memory_retrieval_precision": retrieval_precision,
            "passable_rejections": len(pass_rejects),
            "false_feasible_rejections": len(ff_rejects),
        }
        rows.append(row)
        print(
            f"  [{method:45s}] round={round_idx + 1:02d} "
            f"pass_SR={row['passable_success_rate']:.3f} "
            f"pass_FR={row['passable_false_reject_rate']:.3f} "
            f"FF_reject={row['false_feasible_reject_rate']:.3f} "
            f"wasted={row['wasted_steps_false_feasible']}"
        )

    passable_env.close() if hasattr(passable_env, "close") else None
    ff_env.close() if hasattr(ff_env, "close") else None
    return rows


def summarize(rows: List[Dict]) -> List[Dict]:
    by_method: Dict[str, List[Dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    no_memory_wasted = sum(
        r["wasted_steps_false_feasible"]
        for r in by_method.get("no_memory", [])
    )
    summary = []
    for method, vals in by_method.items():
        wasted = sum(r["wasted_steps_false_feasible"] for r in vals)
        precision_vals = [
            r["memory_retrieval_precision"]
            for r in vals
            if np.isfinite(r["memory_retrieval_precision"])
        ]
        summary.append(
            {
                "method": method,
                "rounds": len(vals),
                "passable_success_rate": float(np.mean([r["passable_success_rate"] for r in vals])),
                "passable_false_reject_rate": float(np.mean([r["passable_false_reject_rate"] for r in vals])),
                "final_false_feasible_reject_rate": float(vals[-1]["false_feasible_reject_rate"]),
                "wasted_steps_false_feasible": int(wasted),
                "steps_saved_vs_no_memory": int(no_memory_wasted - wasted)
                if method != "no_memory"
                else 0,
                "memory_retrieval_precision": float(np.mean(precision_vals))
                if precision_vals
                else float("nan"),
            }
        )
    return summary


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "-"
    return f"{x:.3f}"


def write_markdown(summary: List[Dict], path: Path) -> None:
    lines = [
        "# Table: Repeated Failure Memory",
        "",
        "Failure memory is evaluated by repeated exposure to passable and false-feasible passages. "
        "The contribution is reduced repeated infeasible commitment and wasted attempts, not higher one-shot nominal Habitat success.",
        "",
        "| Method | Passable SR ↑ | Passable false reject ↓ | Final false-feasible reject ↑ | Wasted FF steps ↓ | Steps saved vs no memory ↑ | Retrieval precision ↑ |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {method} | {psr:.3f} | {pfr:.3f} | {ffr:.3f} | {wasted} | {saved} | {prec} |".format(
                method=row["method"],
                psr=row["passable_success_rate"],
                pfr=row["passable_false_reject_rate"],
                ffr=row["final_false_feasible_reject_rate"],
                wasted=row["wasted_steps_false_feasible"],
                saved=row["steps_saved_vs_no_memory"],
                prec=_fmt(row["memory_retrieval_precision"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_latex(summary: List[Dict], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Pass SR & Pass FR & FF reject & Wasted FF & Saved & Precision \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            "{} & {:.3f} & {:.3f} & {:.3f} & {} & {} & {} \\\\".format(
                _tex_escape(row["method"]),
                row["passable_success_rate"],
                row["passable_false_reject_rate"],
                row["final_false_feasible_reject_rate"],
                row["wasted_steps_false_feasible"],
                row["steps_saved_vs_no_memory"],
                _fmt(row["memory_retrieval_precision"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))
    print(f"[write] {path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--n-rounds", type=int, default=8)
    ap.add_argument("--n-passable", type=int, default=60)
    ap.add_argument("--n-ff", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--min-similar", type=int, default=2)
    ap.add_argument("--reject-threshold", type=float, default=0.80)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--output-tex", type=Path, default=DEFAULT_TEX)
    return ap.parse_args()


def main():
    args = parse_args()
    rows = []
    for method in args.methods:
        print(f"\n=== {method} ===")
        rows.extend(run_condition(method, args))
    write_csv(rows, args.output_csv)
    summary = summarize(rows)
    write_markdown(summary, args.output_md)
    write_latex(summary, args.output_tex)


if __name__ == "__main__":
    main()
