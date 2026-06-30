#!/usr/bin/env python3
"""Inference-time comparison: Geometry-FSM vs RL policy (Experiment ⑥).

FSM is a handful of arithmetic operations on a 19-dim vector; RL requires a
forward pass through a neural network.  This script measures wall-clock latency
for each method on CPU (10 000 calls, report μ ± σ in microseconds).

Methods timed
-------------
  fsm_action           — stateless FSM (one call, no TurnCommitFSM state)
  fsm_turn_commit      — TurnCommitFSM.step() (adds branching state machine overhead)
  ppo_sb3_v1           — SB3 PPO trained on v1 env (MLP 64×64)
  ppo_sb3_v2           — SB3 PPO trained on v2 env (if checkpoint exists)
  sac_sb3_v2           — SB3 SAC trained on v2 env (if checkpoint exists)
  td3_sb3_v2           — SB3 TD3 trained on v2 env (if checkpoint exists)

Usage
-----
    python examples/narrow_passage_rl/eval_inference_speed.py
    python examples/narrow_passage_rl/eval_inference_speed.py --n-calls 50000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from failure_memory import PassageFailureMemory
from cross_episode_memory import CrossEpisodeMemory
from eval_harder_benchmark import fsm_action, TurnCommitFSM

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"

_FSM_CHECKPOINTS = {
    "ppo_sb3_v1":  Path("data/narrow_passage_sb3_hard/ppo_narrow_passage.zip"),
    "ppo_sb3_v2":  Path("data/narrow_passage_sb3_v2_ppo/ppo_narrow_passage_v2.zip"),
    "sac_sb3_v2":  Path("data/narrow_passage_sb3_v2_sac/sac_narrow_passage_v2.zip"),
    "td3_sb3_v2":  Path("data/narrow_passage_sb3_v2_td3/td3_narrow_passage_v2.zip"),
}


def _random_obs(rng: np.random.Generator) -> np.ndarray:
    """Generate a plausible 19-dim obs for timing (values don't need to be realistic)."""
    obs = rng.uniform(-1.0, 1.0, size=19).astype(np.float32)
    obs[8] = rng.uniform(0.4, 1.2)    # passage_width > robot_diameter
    obs[9] = rng.uniform(-0.1, 0.3)   # body_margin
    obs[12] = rng.uniform(0.5, 4.0)   # dist_to_goal > 0
    return obs


def time_fn(fn, obs_batch, n_warmup=200):
    """Run fn on each obs in obs_batch, return per-call latency in µs."""
    for obs in obs_batch[:n_warmup]:
        fn(obs)

    latencies = []
    for obs in obs_batch:
        t0 = time.perf_counter()
        fn(obs)
        latencies.append((time.perf_counter() - t0) * 1e6)
    return np.array(latencies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calls", type=int, default=10_000,
                    help="Number of timed calls per method")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-csv", type=Path, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    obs_batch = [_random_obs(rng) for _ in range(args.n_calls)]

    results = []

    # FSM (stateless)
    print("Timing fsm_action ...")
    lats = time_fn(lambda obs: fsm_action(obs), obs_batch)
    results.append({
        "method": "fsm_action",
        "mean_us": round(float(np.mean(lats)), 3),
        "std_us":  round(float(np.std(lats)),  3),
        "p50_us":  round(float(np.percentile(lats, 50)), 3),
        "p99_us":  round(float(np.percentile(lats, 99)), 3),
    })

    # TurnCommitFSM (stateful)
    print("Timing TurnCommitFSM.step ...")
    fsm = TurnCommitFSM()
    lats = time_fn(lambda obs: fsm.step(obs), obs_batch)
    results.append({
        "method": "fsm_turn_commit",
        "mean_us": round(float(np.mean(lats)), 3),
        "std_us":  round(float(np.std(lats)),  3),
        "p50_us":  round(float(np.percentile(lats, 50)), 3),
        "p99_us":  round(float(np.percentile(lats, 99)), 3),
    })

    # TurnCommitFSM + CrossEpisodeMemory
    print("Timing TurnCommitFSM + CrossEpisodeMemory ...")
    cross_mem = CrossEpisodeMemory()
    fsm_mem = TurnCommitFSM(cross_mem=cross_mem)
    lats = time_fn(lambda obs: fsm_mem.step(obs), obs_batch)
    results.append({
        "method": "fsm_cross_memory",
        "mean_us": round(float(np.mean(lats)), 3),
        "std_us":  round(float(np.std(lats)),  3),
        "p50_us":  round(float(np.percentile(lats, 50)), 3),
        "p99_us":  round(float(np.percentile(lats, 99)), 3),
    })

    # SB3 RL models
    for name, ckpt in _FSM_CHECKPOINTS.items():
        if not ckpt.exists():
            print(f"  [{name}] checkpoint not found: {ckpt} — skipping")
            continue
        print(f"Timing {name} ...")
        try:
            from stable_baselines3 import PPO, SAC, TD3
            if "sac" in name:
                algo_cls = SAC
            elif "td3" in name:
                algo_cls = TD3
            else:
                algo_cls = PPO
            model = algo_cls.load(str(ckpt), device="cpu")
            lats = time_fn(
                lambda obs, m=model: m.predict(obs, deterministic=True),
                obs_batch,
            )
            results.append({
                "method": name,
                "mean_us": round(float(np.mean(lats)), 3),
                "std_us":  round(float(np.std(lats)),  3),
                "p50_us":  round(float(np.percentile(lats, 50)), 3),
                "p99_us":  round(float(np.percentile(lats, 99)), 3),
            })
        except Exception as exc:
            print(f"  [{name}] error: {exc}")

    # Print table
    print(f"\n=== Inference latency (CPU, n={args.n_calls} calls) ===")
    print(f"{'Method':22s}  {'Mean (µs)':>10}  {'Std':>8}  {'P50':>8}  {'P99':>8}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['method']:22s}  {r['mean_us']:10.2f}  {r['std_us']:8.2f}"
            f"  {r['p50_us']:8.2f}  {r['p99_us']:8.2f}"
        )

    # Relative speedup vs slowest RL method
    rl_results = [r for r in results if "fsm" not in r["method"]]
    if rl_results:
        slowest_rl = max(rl_results, key=lambda r: r["mean_us"])
        print(f"\n  Speedup (vs {slowest_rl['method']} @ {slowest_rl['mean_us']:.1f} µs):")
        for r in results:
            if r["mean_us"] > 0:
                speedup = slowest_rl["mean_us"] / r["mean_us"]
                print(f"    {r['method']:22s}:  {speedup:.1f}×")

    # Write CSV
    import csv
    out = args.output_csv or (RESULTS / "inference_speed.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
