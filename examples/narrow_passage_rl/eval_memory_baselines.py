#!/usr/bin/env python3
"""Evaluate memory/history baselines on the v2 multi-agent passage benchmark.

This script isolates the paper's memory claim:

  - no_memory: geometry FSM retries every false-feasible passage.
  - knn_failure_memory: kNN over explicit hand geometry features.
  - vanilla_episodic_memory: cosine retrieval over a normalized episode embedding.
  - geometry_guided_failure_memory: existing geometry/type-aware failure memory.

All methods use the same TurnCommitFSM controller for actual traversal.  Memory
only decides whether to reject a passage before entering it, which keeps the
comparison focused on cross-episode failure retrieval instead of controller
differences.
"""

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from cross_episode_memory import CrossEpisodeMemory, MemoryConfig, _fingerprint, _fp_distance
from eval_harder_benchmark import TurnCommitFSM
from eval_multi_agent_memory import FF_CTYPES, PASSABLE_CTYPES, _actual_passage_width, run_one_episode
from failure_memory import PassageFailureMemory
from procedural_env_v2 import HarderNarrowPassageEnv

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"


def _geometry_features(obs: np.ndarray, width: float) -> np.ndarray:
    robot_body_width = 0.36
    left = float(obs[6])
    right = float(obs[7])
    min_clearance = min(left, right)
    width_ratio = width / robot_body_width
    asymmetry = (left - right) / max(abs(left) + abs(right), 1e-3)
    return np.array(
        [
            width_ratio,
            abs(float(obs[10])),
            abs(float(obs[11])),
            min_clearance,
            asymmetry,
        ],
        dtype=np.float32,
    )


def _vanilla_embedding(obs: np.ndarray, width: float) -> np.ndarray:
    emb = obs.astype(np.float32).copy()
    emb[8] = width
    scale = np.array(
        [
            5, 5, 5, 5, 5, 5,
            2, 2, 1.5, 1.0,
            math.pi, 1.0, 6.0,
            0.35, 0.8, 1.0, 1.0, 0.35, 0.8,
        ],
        dtype=np.float32,
    )
    emb = emb / scale
    norm = float(np.linalg.norm(emb))
    return emb / max(norm, 1e-6)


@dataclass
class MemoryRecord:
    feature: np.ndarray
    success: bool


class KNNFailureMemory:
    """Simple geometry kNN memory with failure-label rejection."""

    def __init__(self, k: int = 5, min_neighbors: int = 2, reject_threshold: float = 0.80):
        self.k = k
        self.min_neighbors = min_neighbors
        self.reject_threshold = reject_threshold
        self.records: List[MemoryRecord] = []

    def should_reject(self, obs: np.ndarray, width: float) -> bool:
        if len(self.records) < self.min_neighbors:
            return False
        q = _geometry_features(obs, width)
        ranked = sorted(
            self.records,
            key=lambda r: float(np.linalg.norm(q - r.feature)),
        )[: self.k]
        if len(ranked) < self.min_neighbors:
            return False
        failure_rate = 1.0 - float(np.mean([r.success for r in ranked]))
        return failure_rate >= self.reject_threshold

    def record(self, obs: np.ndarray, width: float, success: bool):
        self.records.append(MemoryRecord(_geometry_features(obs, width), success))


class VanillaEpisodicMemory:
    """Embedding-only episodic memory without geometry-aware constraints."""

    def __init__(self, k: int = 5, min_neighbors: int = 2, reject_threshold: float = 0.80):
        self.k = k
        self.min_neighbors = min_neighbors
        self.reject_threshold = reject_threshold
        self.records: List[MemoryRecord] = []

    def should_reject(self, obs: np.ndarray, width: float) -> bool:
        if len(self.records) < self.min_neighbors:
            return False
        q = _vanilla_embedding(obs, width)
        ranked = sorted(
            self.records,
            key=lambda r: -float(np.dot(q, r.feature)),
        )[: self.k]
        if len(ranked) < self.min_neighbors:
            return False
        failure_rate = 1.0 - float(np.mean([r.success for r in ranked]))
        return failure_rate >= self.reject_threshold

    def record(self, obs: np.ndarray, width: float, success: bool):
        self.records.append(MemoryRecord(_vanilla_embedding(obs, width), success))


def _run_episode_with_memory(env, seed, method, memory, max_steps):
    obs, _ = env.reset(seed=seed)
    entry_obs = obs.copy()
    actual_width = _actual_passage_width(env)
    entry_obs_fp = entry_obs.copy()
    entry_obs_fp[8] = actual_width
    ctype = env._params.ctype.value
    passable = ctype != "false_feasible"

    rejected = False
    if method == "knn_failure_memory":
        rejected = memory.should_reject(entry_obs_fp, actual_width)
    elif method == "vanilla_episodic_memory":
        rejected = memory.should_reject(entry_obs_fp, actual_width)
    elif method == "geometry_guided_failure_memory":
        fp_q = _fingerprint(entry_obs_fp, ctype)
        similar = [
            r for r in memory._records
            if _fp_distance(r.fingerprint, fp_q) == 0
        ]
        if len(similar) >= memory.cfg.min_similar_for_reject:
            success_rate = float(sum(r.success for r in similar)) / len(similar)
            rejected = success_rate < memory.cfg.sr_reject_threshold

    if rejected:
        success = False
        steps = 0
    else:
        fsm = TurnCommitFSM(variant="full", local_mem=None, cross_mem=None)
        stat = run_one_episode(
            env,
            seed,
            fsm,
            local_mem=None,
            cross_mem=None,
            max_steps=max_steps,
        )
        success = bool(stat["success"] > 0.5)
        steps = int(stat["steps"])

    if method == "knn_failure_memory":
        memory.record(entry_obs_fp, actual_width, success)
    elif method == "vanilla_episodic_memory":
        memory.record(entry_obs_fp, actual_width, success)
    elif method == "geometry_guided_failure_memory":
        memory.record_episode(
            entry_obs_fp,
            success,
            passage_width=actual_width,
            steps=steps,
            corridor_type=ctype,
        )

    return {
        "ctype": ctype,
        "passable": passable,
        "rejected": rejected,
        "success": float(success),
        "steps": steps,
    }


def _make_memory(method: str, min_similar: int, reject_threshold: float):
    if method == "knn_failure_memory":
        return KNNFailureMemory(min_neighbors=min_similar, reject_threshold=reject_threshold)
    if method == "vanilla_episodic_memory":
        return VanillaEpisodicMemory(min_neighbors=min_similar, reject_threshold=reject_threshold)
    if method == "geometry_guided_failure_memory":
        return CrossEpisodeMemory(
            cfg=MemoryConfig(
                fp_radius=0,
                min_similar_for_reject=min_similar,
                sr_reject_threshold=1.0 - reject_threshold,
            )
        )
    return None


def run_condition(method: str, args) -> List[dict]:
    passable_env = HarderNarrowPassageEnv({"corridor_types": PASSABLE_CTYPES})
    ff_env = HarderNarrowPassageEnv({"corridor_types": FF_CTYPES})
    memory = _make_memory(method, args.min_similar, args.reject_threshold)
    rows = []

    for round_idx in range(args.n_rounds):
        passable_stats = []
        ff_stats = []
        for seed in range(args.n_passable):
            if method == "no_memory":
                fsm = TurnCommitFSM(variant="full", local_mem=None, cross_mem=None)
                stat = run_one_episode(passable_env, seed, fsm, None, None, args.max_steps)
            else:
                stat = _run_episode_with_memory(
                    passable_env, seed, method, memory, args.max_steps
                )
            passable_stats.append(stat)

        for seed in range(args.n_ff):
            if method == "no_memory":
                fsm = TurnCommitFSM(variant="full", local_mem=None, cross_mem=None)
                stat = run_one_episode(ff_env, seed, fsm, None, None, args.max_steps)
            else:
                stat = _run_episode_with_memory(ff_env, seed, method, memory, args.max_steps)
            ff_stats.append(stat)

        ff_attempts = [s for s in ff_stats if not s["rejected"]]
        ff_rejects = [s for s in ff_stats if s["rejected"]]
        passable_rejects = [s for s in passable_stats if s["rejected"]]
        row = {
            "method": method,
            "round": round_idx + 1,
            "n_passable": len(passable_stats),
            "n_ff": len(ff_stats),
            "sr_passable": round(float(np.mean([s["success"] for s in passable_stats])), 4),
            "passable_reject_rate": round(len(passable_rejects) / max(len(passable_stats), 1), 4),
            "ff_attempt_rate": round(len(ff_attempts) / max(len(ff_stats), 1), 4),
            "ff_reject_rate": round(len(ff_rejects) / max(len(ff_stats), 1), 4),
            "wasted_steps_ff": int(sum(s["steps"] for s in ff_attempts)),
        }
        rows.append(row)
        print(
            f"  [{method:30s}] round {round_idx + 1:02d} "
            f"pass_SR={row['sr_passable']:.3f} pass_rej={row['passable_reject_rate']:.2f} "
            f"FF_rej={row['ff_reject_rate']:.2f} wasted={row['wasted_steps_ff']}"
        )

    return rows


def write_markdown(rows: List[dict], path: Path):
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    lines = [
        "| Method | Passable SR ↑ | Passable Reject ↓ | Final FF Reject ↑ | Wasted FF Steps ↓ | Steps Saved vs No Memory ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    no_mem_wasted = sum(r["wasted_steps_ff"] for r in rows if r["method"] == "no_memory")
    for method in methods:
        mr = [r for r in rows if r["method"] == method]
        pass_sr = float(np.mean([r["sr_passable"] for r in mr]))
        pass_rej = float(np.mean([r["passable_reject_rate"] for r in mr]))
        final_ff_rej = float(mr[-1]["ff_reject_rate"])
        wasted = int(sum(r["wasted_steps_ff"] for r in mr))
        saved = no_mem_wasted - wasted if method != "no_memory" else 0
        lines.append(
            f"| {method} | {pass_sr:.3f} | {pass_rej:.3f} | "
            f"{final_ff_rej:.3f} | {wasted} | {saved} |"
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--methods",
        nargs="+",
        default=[
            "no_memory",
            "knn_failure_memory",
            "vanilla_episodic_memory",
            "geometry_guided_failure_memory",
        ],
    )
    ap.add_argument("--n-rounds", type=int, default=8)
    ap.add_argument("--n-passable", type=int, default=60)
    ap.add_argument("--n-ff", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--min-similar", type=int, default=2)
    ap.add_argument("--reject-threshold", type=float, default=0.80)
    ap.add_argument("--output-csv", type=Path, default=RESULTS / "memory_baselines.csv")
    ap.add_argument("--output-table", type=Path, default=RESULTS / "paper_table_memory_baselines.md")
    args = ap.parse_args()

    all_rows = []
    for method in args.methods:
        print(f"\n=== {method} ===")
        all_rows.extend(run_condition(method, args))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[write] {args.output_csv}")
    write_markdown(all_rows, args.output_table)


if __name__ == "__main__":
    main()
