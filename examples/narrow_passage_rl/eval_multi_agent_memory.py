#!/usr/bin/env python3
"""Multi-agent cross-episode memory experiment (Experiment ①).

N agents (rounds) explore the SAME fixed set of corridors.
Condition A — cross_memory:   shared CrossEpisodeMemory across agents.
Condition B — local_memory:   PassageFailureMemory reset each round (intra-episode only).
Condition C — no_memory:      pure geometry FSM, no memory.

Key story: false_feasible corridors are impassable (always fail). Cross-episode
memory learns to REJECT them after round 1, saving wasted steps. Local memory
and no-memory re-attempt every round.

Metrics per round
-----------------
  sr_passable      — SR on navigable corridors (cross-memory should NOT hurt this)
  attempt_rate_ff  — fraction of false_feasible corridors the agent actually enters
  reject_rate_ff   — fraction the memory correctly rejects (1 - attempt_rate_ff)
  wasted_steps_ff  — total steps spent inside false_feasible corridors
  steps_saved      — cumulative reduction vs no_memory (shown from round 2+)

Usage
-----
    python examples/narrow_passage_rl/eval_multi_agent_memory.py
    python examples/narrow_passage_rl/eval_multi_agent_memory.py \
        --n-rounds 10 --n-passable 80 --n-ff 40 --max-steps 300
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import HarderNarrowPassageEnv, CorridorType
from failure_memory import PassageFailureMemory
from cross_episode_memory import CrossEpisodeMemory, MemoryConfig, FSMMode, _fingerprint, _fp_distance
from eval_harder_benchmark import TurnCommitFSM, fsm_action
from dmin_calibrator import CalibConfig

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"
PASSABLE_CTYPES = ["STRAIGHT", "L_SHAPED", "S_SHAPED", "NARROW_EXIT",
                   "NARROW_ENTRY", "ASYMMETRIC"]
FF_CTYPES = ["FALSE_FEASIBLE"]


# ── Corridor set builder ──────────────────────────────────────────────────────

def build_fixed_corridors(n_passable: int, n_ff: int):
    """Return two lists of seeds: passable and false_feasible.

    We use separate HarderNarrowPassageEnvs (one per type group) so that seeds
    map deterministically to corridor configs — the same seed always produces
    the same corridor geometry when the env is created with the same config.
    """
    passable_env = HarderNarrowPassageEnv({"corridor_types": PASSABLE_CTYPES})
    ff_env = HarderNarrowPassageEnv({"corridor_types": FF_CTYPES})

    passable_seeds = list(range(n_passable))
    ff_seeds = list(range(n_ff))

    passable_env.close() if hasattr(passable_env, "close") else None
    ff_env.close() if hasattr(ff_env, "close") else None

    return passable_seeds, ff_seeds


# ── Episode runner ────────────────────────────────────────────────────────────

def _actual_passage_width(env: HarderNarrowPassageEnv) -> float:
    """True minimum corridor width from env params (not the far-field obs[8]=10.0)."""
    return min(w for _, w in env._params.width_profile)


def run_one_episode(
    env: HarderNarrowPassageEnv,
    seed: int,
    fsm: TurnCommitFSM,
    local_mem: Optional[PassageFailureMemory],
    cross_mem: Optional[CrossEpisodeMemory],
    max_steps: int,
) -> dict:
    """Reset env with fixed seed, run one episode, return stats."""
    obs, _ = env.reset(seed=seed)
    entry_obs = obs.copy()
    ctype = env._params.ctype.value
    passable = ctype != "false_feasible"

    # Use the actual passage width (not far-field obs[8]=10.0) so fingerprints
    # are informative.  obs[8] at reset is always 10.0 (robot hasn't entered
    # the corridor yet), which collapses all episodes into the same width bucket.
    actual_width = _actual_passage_width(env)
    entry_obs_fp = entry_obs.copy()
    entry_obs_fp[8] = actual_width   # patch obs[8] for fingerprinting only

    # Cross-episode memory: decide before attempting.
    # We use fingerprint-only rejection (fp_radius=0, exact type_class match) so that
    # false_feasible failures (type_class=2) never bleed into passable corridors
    # (type_class=0/1/3), and we bypass the DMinCalibrator which mistakes FF failures
    # for "passage too narrow" and wrongly rejects passable corridors.
    rejected = False
    if cross_mem is not None:
        cross_mem.reset_local()
        fp_q = _fingerprint(entry_obs_fp, ctype)
        similar = [
            r for r in cross_mem._records
            if _fp_distance(r.fingerprint, fp_q) == 0  # exact match only
        ]
        n_sim = len(similar)
        if n_sim >= cross_mem.cfg.min_similar_for_reject:
            sr = float(sum(r.success for r in similar)) / n_sim
            if sr < cross_mem.cfg.sr_reject_threshold:
                rejected = True

    if rejected:
        return {
            "seed": seed,
            "ctype": ctype,
            "passable": passable,
            "rejected": True,
            "success": 0.0,
            "steps": 0,
            "min_bm": float("nan"),
        }

    if local_mem is not None:
        local_mem.reset()
    fsm.reset()

    steps = 0
    done = False
    min_bm = float("inf")
    any_collision = False

    while not done and steps < max_steps:
        bm = float(obs[9])
        min_bm = min(min_bm, bm)
        if float(obs[16]) > 0.5:
            any_collision = True

        action, _ = fsm.step(obs)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc

        if local_mem is not None and any_collision:
            local_mem.add_failure(obs)

        steps += 1

    success = float(info.get("success", 0.0))

    if cross_mem is not None:
        cross_mem.record_episode(
            entry_obs_fp, bool(success > 0.5),
            passage_width=actual_width,
            steps=steps,
            corridor_type=ctype,
        )

    return {
        "seed": seed,
        "ctype": ctype,
        "passable": passable,
        "rejected": False,
        "success": success,
        "steps": steps,
        "min_bm": min_bm if np.isfinite(min_bm) else float(obs[9]),
    }


# ── Per-round runner ──────────────────────────────────────────────────────────

def run_round(
    passable_env: HarderNarrowPassageEnv,
    ff_env: HarderNarrowPassageEnv,
    passable_seeds: list,
    ff_seeds: list,
    fsm: TurnCommitFSM,
    local_mem: Optional[PassageFailureMemory],
    cross_mem: Optional[CrossEpisodeMemory],
    max_steps: int,
    round_idx: int,
    condition: str,
) -> dict:
    """Run all corridors for one agent (round)."""
    stats_passable = []
    stats_ff = []

    for seed in passable_seeds:
        stat = run_one_episode(
            passable_env, seed, fsm, local_mem, cross_mem, max_steps
        )
        stats_passable.append(stat)

    for seed in ff_seeds:
        stat = run_one_episode(
            ff_env, seed, fsm, local_mem, cross_mem, max_steps
        )
        stats_ff.append(stat)

    n_pass = len(stats_passable)
    n_ff = len(stats_ff)

    sr_passable = float(
        np.mean([s["success"] for s in stats_passable])
    ) if n_pass else 0.0

    # False-feasible metrics
    ff_attempts = [s for s in stats_ff if not s["rejected"]]
    ff_rejects = [s for s in stats_ff if s["rejected"]]
    attempt_rate_ff = len(ff_attempts) / n_ff if n_ff else 0.0
    reject_rate_ff = len(ff_rejects) / n_ff if n_ff else 0.0
    wasted_steps_ff = sum(s["steps"] for s in ff_attempts)

    print(
        f"  [{condition:14s}] round {round_idx + 1:2d} | "
        f"pass SR={sr_passable:.3f}  "
        f"FF attempt={attempt_rate_ff:.2f}  reject={reject_rate_ff:.2f}  "
        f"wasted_steps={wasted_steps_ff:5d}"
    )

    return {
        "condition": condition,
        "round": round_idx + 1,
        "n_passable": n_pass,
        "n_ff": n_ff,
        "sr_passable": round(sr_passable, 4),
        "sr_ff": 0.0,
        "attempt_rate_ff": round(attempt_rate_ff, 4),
        "reject_rate_ff": round(reject_rate_ff, 4),
        "wasted_steps_ff": wasted_steps_ff,
        "ff_attempts": len(ff_attempts),
        "ff_rejects": len(ff_rejects),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rounds", type=int, default=8,
                    help="Number of agents (rounds) to simulate")
    ap.add_argument("--n-passable", type=int, default=60,
                    help="Number of passable corridor instances per round")
    ap.add_argument("--n-ff", type=int, default=40,
                    help="Number of false_feasible corridor instances per round")
    ap.add_argument("--max-steps", type=int, default=300,
                    help="Max steps per episode")
    ap.add_argument("--output-csv", type=Path, default=None)
    ap.add_argument("--conditions", nargs="+",
                    default=["cross_memory", "local_memory", "no_memory"],
                    choices=["cross_memory", "local_memory", "no_memory"])
    ap.add_argument("--min-similar", type=int, default=2,
                    help="min_similar_for_reject in MemoryConfig")
    ap.add_argument("--sr-reject-threshold", type=float, default=0.10,
                    help="sr_reject_threshold in MemoryConfig")
    args = ap.parse_args()

    print(f"[multi_agent_memory] {args.n_rounds} rounds × "
          f"({args.n_passable} passable + {args.n_ff} false_feasible) corridors")
    print(f"  conditions: {args.conditions}")

    passable_seeds = list(range(args.n_passable))
    ff_seeds = list(range(args.n_ff))

    all_rows = []

    for condition in args.conditions:
        print(f"\n=== Condition: {condition} ===")

        passable_env = HarderNarrowPassageEnv({"corridor_types": PASSABLE_CTYPES})
        ff_env = HarderNarrowPassageEnv({"corridor_types": FF_CTYPES})

        # Set up memory for this condition.
        # For the multi-agent experiment: exact fingerprint matching (fp_radius=0)
        # so false_feasible records (type_class=2) never match passable corridors.
        # min_similar_for_reject=2: reject from round 3 (after 2 identical failures).
        cross_mem = CrossEpisodeMemory(
            cfg=MemoryConfig(fp_radius=0,
                             min_similar_for_reject=args.min_similar,
                             sr_reject_threshold=args.sr_reject_threshold)
        ) if condition == "cross_memory" else None
        local_mem_base = PassageFailureMemory() if condition == "local_memory" else None

        for round_idx in range(args.n_rounds):
            # Fresh local_mem each round (intra-episode only, resets per episode inside run_round)
            local_mem = local_mem_base  # passthrough; reset happens inside run_one_episode

            # Do NOT pass cross_mem to the FSM: rejection decisions are handled
            # explicitly in run_one_episode using patched entry obs.  Passing it
            # would cause fsm_action() to call cross_mem.fsm_mode(live_obs) on
            # every step, which uses unpatched obs[8] (live clearance) and can
            # spuriously trigger mid-episode REJECT for passable corridors whose
            # live obs fingerprint collides with recorded failures.
            fsm = TurnCommitFSM(
                variant="full",
                local_mem=local_mem,
                cross_mem=None,
            )

            row = run_round(
                passable_env, ff_env,
                passable_seeds, ff_seeds,
                fsm, local_mem, cross_mem,
                args.max_steps, round_idx, condition,
            )
            all_rows.append(row)

    # Summary table
    print("\n=== Summary: wasted steps on false_feasible per round ===")
    print(f"{'Condition':15s} | " + " | ".join(
        f"R{r+1:02d}" for r in range(args.n_rounds)
    ))
    print("-" * (17 + 6 * args.n_rounds))
    for cond in args.conditions:
        rows = [r for r in all_rows if r["condition"] == cond]
        wasted = [str(r["wasted_steps_ff"]).rjust(4) for r in rows]
        print(f"{cond:15s} | " + " | ".join(wasted))

    print("\n=== Summary: reject_rate_ff per round ===")
    print(f"{'Condition':15s} | " + " | ".join(
        f"R{r+1:02d}" for r in range(args.n_rounds)
    ))
    print("-" * (17 + 6 * args.n_rounds))
    for cond in args.conditions:
        rows = [r for r in all_rows if r["condition"] == cond]
        reject = [f"{r['reject_rate_ff']:.2f}" for r in rows]
        print(f"{cond:15s} | " + " | ".join(reject))

    print("\n=== Summary: SR on passable corridors per round ===")
    print(f"{'Condition':15s} | " + " | ".join(
        f"R{r+1:02d}" for r in range(args.n_rounds)
    ))
    print("-" * (17 + 6 * args.n_rounds))
    for cond in args.conditions:
        rows = [r for r in all_rows if r["condition"] == cond]
        sr = [f"{r['sr_passable']:.3f}" for r in rows]
        print(f"{cond:15s} | " + " | ".join(sr))

    # Compute cumulative steps saved vs no_memory
    no_mem_rows = {r["round"]: r for r in all_rows if r["condition"] == "no_memory"}
    if no_mem_rows and "cross_memory" in args.conditions:
        print("\n=== Steps saved by cross_memory vs no_memory (cumulative) ===")
        cumulative_saved = 0
        for r in sorted({r["round"] for r in all_rows}):
            nm = no_mem_rows.get(r, {}).get("wasted_steps_ff", 0)
            cm_rows = [x for x in all_rows if x["condition"] == "cross_memory" and x["round"] == r]
            cm = cm_rows[0]["wasted_steps_ff"] if cm_rows else nm
            saved = nm - cm
            cumulative_saved += saved
            print(f"  round {r:2d}: saved {saved:5d} steps  (cumulative={cumulative_saved:6d})")

    # Write CSV
    out = args.output_csv or (RESULTS / "multi_agent_memory.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
