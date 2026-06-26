#!/usr/bin/env python3
"""Cross-episode memory differentiation experiment.

Design
------
We run N_PASSAGES episodes repeated across N_ROUNDS rounds to show that
CrossEpisodeMemory learns to *reject* known-bad passages while a memoryless
agent keeps wasting attempts.

Passage mix (per round):
  - 15 narrow / passable  (true_narrow):  body_margin ∈ (0.05, 0.20m)
  - 15 false_feasible:                    corridor looks passable but bm < 0
  - 10 dynamic_block:                     passable in rounds 1-2, blocked in 3+

Methods:
  - no_memory   — full Geometry-FSM, CrossEpisodeMemory disabled
  - with_memory — full Geometry-FSM + CrossEpisodeMemory

Per-round metrics
-----------------
  sr_navigable      : success rate on true_narrow passages
  attempt_rate_ff   : fraction of false_feasible passages actually attempted
  reject_rate_ff    : fraction rejected outright (memory only)
  wasted_steps_ff   : avg steps spent on false_feasible per episode
  wasted_steps_dyn  : avg steps spent on dynamic_block in rounds 3-5
  collision_rate    : fraction of episodes with any_collision or timeout

Run
---
    python examples/narrow_passage_rl/eval_memory_differentiation.py
    python examples/narrow_passage_rl/eval_memory_differentiation.py --rounds 6
    python examples/narrow_passage_rl/eval_memory_differentiation.py --seeds 3
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import HarderNarrowPassageEnv
from eval_harder_benchmark import fsm_action, _act
from cross_episode_memory import CrossEpisodeMemory, MemoryConfig

RESULTS_DIR = Path(__file__).parent / "results" / "narrow_passage_rl"
MAX_STEPS = 400

# ── Episode seeds per category ────────────────────────────────────────────────

TRUE_NARROW_SEEDS   = list(range(0, 15))   # passable narrow corridors
FALSE_FEASIBLE_SEEDS = list(range(0, 15))  # impassable passages
DYNAMIC_BLOCK_SEEDS  = list(range(0, 10))  # passable → blocked in round 3+

DYNAMIC_BLOCK_CUTOFF = 3  # passage becomes blocked from this round onwards


def _run_episode(env, obs_init, memory=None, corridor_type="true_narrow",
                 max_steps=MAX_STEPS):
    """Run one episode. Returns dict of stats."""
    obs = obs_init
    steps = 0
    any_collision = False
    min_bm = float("inf")

    while steps < max_steps:
        bm = float(obs[9])
        if bm < min_bm:
            min_bm = bm
        collision_flag = float(obs[16]) > 0.5
        if collision_flag:
            any_collision = True

        action, mode = fsm_action(obs, variant="full",
                                  local_mem=None,
                                  cross_mem=memory)
        result = env.step(action)
        obs = result[0]
        done = result[2] or result[3]
        info = result[4]
        steps += 1

        if done:
            break

    success = bool(info.get("success", False))
    if memory is not None:
        memory.record_episode(
            entry_obs=obs_init,
            success=success,
            passage_width=float(obs_init[8]),
            steps=steps,
            corridor_type=corridor_type,
        )

    return {
        "success": success,
        "steps": steps,
        "any_collision": any_collision,
        "min_bm": min_bm if np.isfinite(min_bm) else 0.0,
    }


def _run_round(narrow_env, ff_env, dyn_env_open, dyn_env_blocked,
               round_idx, memory=None):
    """Run one round through all passages. Returns per-category stats."""

    stats = {
        "true_narrow": [],
        "false_feasible": [],
        "dynamic_block": [],
    }

    # ---- true narrow ----
    for seed in TRUE_NARROW_SEEDS:
        obs, _ = narrow_env.reset(seed=seed)
        if memory is not None:
            memory.reset_local()
            if not memory.should_attempt(float(obs[8]), corridor_type="narrow", entry_obs=obs):
                stats["true_narrow"].append({
                    "success": False, "steps": 0, "rejected": True,
                    "any_collision": False, "min_bm": 0.0
                })
                continue
        result = _run_episode(narrow_env, obs, memory, "narrow")
        result["rejected"] = False
        stats["true_narrow"].append(result)

    # ---- false feasible ----
    for seed in FALSE_FEASIBLE_SEEDS:
        obs, _ = ff_env.reset(seed=seed)
        if memory is not None:
            memory.reset_local()
            if not memory.should_attempt(float(obs[8]), corridor_type="false_feasible", entry_obs=obs):
                stats["false_feasible"].append({
                    "success": False, "steps": 0, "rejected": True,
                    "any_collision": False, "min_bm": 0.0
                })
                # Still record this as a failure to keep memory consistent
                memory.record_episode(obs, False, float(obs[8]),
                                      steps=0, corridor_type="false_feasible")
                continue
        result = _run_episode(ff_env, obs, memory, "false_feasible")
        result["rejected"] = False
        stats["false_feasible"].append(result)

    # ---- dynamic block ----
    # Passable in rounds 0..CUTOFF-1, blocked in rounds CUTOFF+
    dyn_env = dyn_env_open if round_idx < DYNAMIC_BLOCK_CUTOFF else dyn_env_blocked
    for seed in DYNAMIC_BLOCK_SEEDS:
        obs, _ = dyn_env.reset(seed=seed)
        if memory is not None:
            memory.reset_local()
            dyn_ctype = "dynamic_passable" if round_idx < DYNAMIC_BLOCK_CUTOFF else "dynamic_blocked"
            if not memory.should_attempt(float(obs[8]), corridor_type=dyn_ctype, entry_obs=obs):
                stats["dynamic_block"].append({
                    "success": False, "steps": 0, "rejected": True,
                    "any_collision": False, "min_bm": 0.0
                })
                memory.record_episode(obs, False, float(obs[8]),
                                      steps=0, corridor_type=dyn_ctype)
                continue
        ctype = "dynamic_passable" if round_idx < DYNAMIC_BLOCK_CUTOFF else "dynamic_blocked"
        result = _run_episode(dyn_env, obs, memory, ctype)
        result["rejected"] = False
        stats["dynamic_block"].append(result)

    return stats


def _summarise_round(stats):
    def agg(records, key):
        vals = [r[key] for r in records]
        return float(np.mean(vals)) if vals else 0.0

    out = {}
    for cat, records in stats.items():
        n = len(records)
        n_attempted = sum(1 for r in records if not r["rejected"])
        n_rejected  = sum(1 for r in records if r["rejected"])
        n_success   = sum(1 for r in records if r["success"])
        wasted = [r["steps"] for r in records if not r["rejected"] and not r["success"]]
        out[cat] = {
            "n": n,
            "n_attempted": n_attempted,
            "n_rejected": n_rejected,
            "attempt_rate": n_attempted / n if n else 0.0,
            "reject_rate":  n_rejected  / n if n else 0.0,
            "sr":  n_success / n_attempted if n_attempted else 0.0,
            "wasted_steps_mean": float(np.mean(wasted)) if wasted else 0.0,
            "collision_rate": agg(records, "any_collision"),
        }
    return out


def run_experiment(n_rounds=6, n_seeds=1, verbose=True):
    """Run the full differentiation experiment."""
    # Build four environments:
    #   narrow_env    : true narrow corridors (passable)
    #   ff_env        : false_feasible (impassable)
    #   dyn_open_env  : narrow corridor, wide enough (passable)
    #   dyn_block_env : same geometry but too narrow (blocked)
    narrow_env   = HarderNarrowPassageEnv({"corridor_types": ["narrow_exit"]})
    ff_env       = HarderNarrowPassageEnv({"corridor_types": ["false_feasible"]})
    dyn_open_env = HarderNarrowPassageEnv({
        "corridor_types": ["narrow_exit"],
        "passage_width_range": [0.54, 0.62],   # comfortably passable
    })
    dyn_block_env = HarderNarrowPassageEnv({
        "corridor_types": ["narrow_entry"],
        "passage_width_range": [0.32, 0.38],   # too narrow (robot body ~0.42m)
    })

    all_seed_results = []

    for seed_offset in range(n_seeds):
        round_results = {"no_memory": [], "with_memory": []}

        # Independent memory per seed run
        mem_cfg = MemoryConfig()
        mem_cfg.min_similar_for_reject = 3   # faster learning for experiment
        mem_cfg.min_similar_for_cautious = 2
        mem_cfg.sr_reject_threshold = 0.10
        memory = CrossEpisodeMemory(cfg=mem_cfg)

        # Offset seeds per run to get different passages
        global TRUE_NARROW_SEEDS, FALSE_FEASIBLE_SEEDS, DYNAMIC_BLOCK_SEEDS
        _tn   = [s + seed_offset * 100 for s in TRUE_NARROW_SEEDS]
        _ff   = [s + seed_offset * 100 for s in FALSE_FEASIBLE_SEEDS]
        _dyn  = [s + seed_offset * 100 for s in DYNAMIC_BLOCK_SEEDS]

        # Patch seeds for this run
        import eval_memory_differentiation as _self
        orig_tn, orig_ff, orig_dyn = (
            _self.TRUE_NARROW_SEEDS, _self.FALSE_FEASIBLE_SEEDS,
            _self.DYNAMIC_BLOCK_SEEDS
        )
        _self.TRUE_NARROW_SEEDS   = _tn
        _self.FALSE_FEASIBLE_SEEDS = _ff
        _self.DYNAMIC_BLOCK_SEEDS  = _dyn

        for method in ["no_memory", "with_memory"]:
            m = memory if method == "with_memory" else None
            if m is not None:
                m.__init__(cfg=mem_cfg)   # reset for each method

            for r in range(n_rounds):
                stats = _run_round(
                    narrow_env, ff_env, dyn_open_env, dyn_block_env,
                    round_idx=r, memory=m
                )
                summary = _summarise_round(stats)
                summary["round"] = r + 1
                summary["method"] = method
                summary["seed_offset"] = seed_offset
                round_results[method].append(summary)

        _self.TRUE_NARROW_SEEDS   = orig_tn
        _self.FALSE_FEASIBLE_SEEDS = orig_ff
        _self.DYNAMIC_BLOCK_SEEDS  = orig_dyn

        all_seed_results.append(round_results)

    return all_seed_results


def _print_table(all_seed_results, n_rounds):
    print("\n=== Memory Differentiation Experiment ===")
    print(f"{'Round':>6}  {'Method':>12}  {'Narrow SR':>9}  "
          f"{'FF attempt%':>11}  {'FF reject%':>10}  "
          f"{'FF wasted':>9}  {'Dyn attempt%':>12}  {'Dyn SR':>7}")
    print("-" * 90)

    for r in range(n_rounds):
        for method in ["no_memory", "with_memory"]:
            # Average over seeds
            rows = [s[method][r] for s in all_seed_results]
            tn_sr   = np.mean([x["true_narrow"]["sr"]            for x in rows])
            ff_att  = np.mean([x["false_feasible"]["attempt_rate"] for x in rows])
            ff_rej  = np.mean([x["false_feasible"]["reject_rate"]  for x in rows])
            ff_wst  = np.mean([x["false_feasible"]["wasted_steps_mean"] for x in rows])
            dyn_att = np.mean([x["dynamic_block"]["attempt_rate"]  for x in rows])
            dyn_sr  = np.mean([x["dynamic_block"]["sr"]            for x in rows])

            print(f"{r+1:>6}  {method:>12}  {tn_sr:>9.3f}  "
                  f"{ff_att:>11.3f}  {ff_rej:>10.3f}  "
                  f"{ff_wst:>9.1f}  {dyn_att:>12.3f}  {dyn_sr:>7.3f}")
        if r < n_rounds - 1:
            print()


def _save_csv(all_seed_results, n_rounds, out_path):
    rows = []
    for seed_idx, seed_data in enumerate(all_seed_results):
        for method in ["no_memory", "with_memory"]:
            for r in range(n_rounds):
                d = seed_data[method][r]
                row = {
                    "seed_offset": seed_idx,
                    "method": method,
                    "round": r + 1,
                    # true narrow
                    "narrow_sr":      d["true_narrow"]["sr"],
                    "narrow_attempt": d["true_narrow"]["attempt_rate"],
                    # false feasible
                    "ff_attempt":     d["false_feasible"]["attempt_rate"],
                    "ff_reject":      d["false_feasible"]["reject_rate"],
                    "ff_wasted":      d["false_feasible"]["wasted_steps_mean"],
                    "ff_collision":   d["false_feasible"]["collision_rate"],
                    # dynamic block
                    "dyn_sr":         d["dynamic_block"]["sr"],
                    "dyn_attempt":    d["dynamic_block"]["attempt_rate"],
                    "dyn_reject":     d["dynamic_block"]["reject_rate"],
                    "dyn_wasted":     d["dynamic_block"]["wasted_steps_mean"],
                }
                rows.append(row)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seeds",  type=int, default=3,
                    help="number of independent seed runs (for mean±std)")
    args = ap.parse_args()

    print(f"Running memory differentiation: {args.rounds} rounds × {args.seeds} seeds")
    results = run_experiment(n_rounds=args.rounds, n_seeds=args.seeds)
    _print_table(results, args.rounds)

    out = RESULTS_DIR / "memory_differentiation.csv"
    _save_csv(results, args.rounds, out)

    # Print summary comparison: round 1 vs round 6
    print("\n=== Key comparison: Round 1 vs Round 6 ===")
    print(f"{'':30s}  {'Round 1':>10}  {'Round 6':>10}")
    for method in ["no_memory", "with_memory"]:
        for metric, label in [
            ("false_feasible.attempt_rate", "FF attempt rate"),
            ("false_feasible.wasted_steps_mean", "FF wasted steps"),
            ("false_feasible.collision_rate", "FF collision rate"),
        ]:
            cat, key = metric.split(".")
            r1 = np.mean([s[method][0][cat][key]           for s in results])
            r6 = np.mean([s[method][args.rounds - 1][cat][key] for s in results])
            print(f"  {method:12s} {label:20s}  {r1:>10.3f}  {r6:>10.3f}")


if __name__ == "__main__":
    main()
