#!/usr/bin/env python3
"""D_min self-calibration experiment (synthetic env).

The robot's true capability boundary is D_min = 2×robot_radius = 0.36m.
We start with a wrong (pessimistic) estimate D_hat_init = 0.56m, which causes
the agent to REJECT feasible passages with width W ∈ (0.36, 0.56)m.

Three agents share the same episode sequence:
  oracle      — uses D_true = 0.36m (knows robot radius exactly)
  fixed_wrong — uses D_hat  = 0.56m, never calibrates
  calibrated  — starts at 0.56m, updates from traversal outcomes

All three use the same trained PPO passage model for actual traversal.

The calibrator updates only from SUCCESSES (reliable signal: success at W
proves D_min ≤ W).  With epsilon-exploration (attempt some rejected passages),
the posterior converges toward D_true as the robot accumulates evidence.

Outputs
-------
  results/narrow_passage_rl/dmin_calib_convergence.csv  — D_hat per episode
  results/narrow_passage_rl/dmin_calib_episodes.csv     — per-episode SR table

Run
---
  python examples/narrow_passage_rl/eval_dmin_calibration.py
  python examples/narrow_passage_rl/eval_dmin_calibration.py --n-episodes 500
"""

import argparse, csv, sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent))
from procedural_env import ProceduralNarrowPassageEnv
from dmin_calibrator import DMinCalibrator, CalibConfig

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"
MODEL_PATH = Path("data/narrow_passage_sb3_hard/ppo_narrow_passage.zip")

ROBOT_RADIUS_TRUE = 0.18
D_MIN_TRUE  = 2 * ROBOT_RADIUS_TRUE   # 0.36 m — true capability boundary
D_MIN_WRONG = 0.56                    # pessimistic initial guess (55% too large)


def run_episode_with_ppo(env, model, seed: int, attempt: bool, max_steps: int = 300) -> dict:
    """Run one episode.  If not attempted (entry-rejected), returns immediately."""
    obs, _ = env.reset(seed=seed)
    W = float(env.params.width)
    bm = (W - D_MIN_TRUE) / 2

    if not attempt:
        return {"passage_width": W, "body_margin": bm,
                "success": 0.0, "rejected": 1.0, "collision": 0.0, "steps": 0}

    done = False; steps = 0; info = {}
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        steps += 1
        done = term or trunc

    return {"passage_width": W, "body_margin": bm,
            "success":   float(info.get("success",  False)),
            "rejected":  0.0,
            "collision": float(info.get("collision", False)),
            "steps":     steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes",   type=int,   default=300)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--d-min-wrong",  type=float, default=D_MIN_WRONG)
    ap.add_argument("--width-lo",     type=float, default=0.36,
                    help="Minimum passage width (m). Set ≈ D_true for all-feasible.")
    ap.add_argument("--width-hi",     type=float, default=0.90)
    ap.add_argument("--explore-prob", type=float, default=0.25,
                    help="Epsilon-exploration: try even when D_hat says reject")
    ap.add_argument("--noise-sigma",  type=float, default=0.025,
                    help="Likelihood noise σ for Bayesian update (m)")
    args = ap.parse_args()

    N      = args.n_episodes
    rng    = np.random.default_rng(args.seed)
    seeds  = rng.integers(0, 2**31, size=N)

    env_cfg = {"width_range": [args.width_lo, args.width_hi],
               "false_feasible_prob": 0.0,
               "max_steps": 300}
    env   = ProceduralNarrowPassageEnv(env_cfg)
    model = PPO.load(MODEL_PATH, device="cpu")

    calib_cfg = CalibConfig(
        explore_prob  = args.explore_prob,
        noise_sigma   = args.noise_sigma,
        conservative_pct = 0.15,   # 85th-percentile for cautious reject
    )
    calibrator = DMinCalibrator(d_init=args.d_min_wrong, cfg=calib_cfg)

    ep_rows, conv_rows = [], []

    print(f"D_true={D_MIN_TRUE:.3f}m  D_init={args.d_min_wrong:.3f}m  "
          f"explore={args.explore_prob}  N={N}")
    print(f"{'ep':>4}  {'W':>5}  {'bm':>6}  "
          f"{'ora':>3} {'fix':>3} {'cal':>3}  "
          f"{'D_hat':>5} {'±std':>4} {'err':>5}")

    for ep_idx in range(N):
        seed_i = int(seeds[ep_idx])

        # ── oracle ───────────────────────────────────────────────────────────
        env.reset(seed=seed_i)
        W = float(env.params.width)
        oracle_attempt = (W >= D_MIN_TRUE)
        r_ora = run_episode_with_ppo(env, model, seed_i, oracle_attempt)

        # ── fixed wrong ──────────────────────────────────────────────────────
        fixed_attempt = (W >= args.d_min_wrong)
        r_fix = run_episode_with_ppo(env, model, seed_i, fixed_attempt)

        # ── calibrated ───────────────────────────────────────────────────────
        calib_attempt = calibrator.should_attempt(W, rng)
        r_cal = run_episode_with_ppo(env, model, seed_i, calib_attempt)
        # Update posterior from this episode's outcome
        if calib_attempt:
            calibrator.update(W, success=bool(r_cal["success"] > 0.5),
                              attempted=True)

        d_hat = calibrator.d_hat
        d_std = calibrator.std
        d_err = abs(d_hat - D_MIN_TRUE)
        ci_lo, ci_hi = calibrator.posterior_interval(0.90)
        bm = (W - D_MIN_TRUE) / 2

        conv_rows.append({"episode": ep_idx, "passage_width": round(W, 4),
                          "body_margin": round(bm, 4),
                          "d_hat": round(d_hat, 4), "d_std": round(d_std, 4),
                          "d_err": round(d_err, 4),
                          "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
                          "calib_attempted": int(calib_attempt)})
        for agent, r in [("oracle", r_ora), ("fixed", r_fix), ("calib", r_cal)]:
            ep_rows.append({"episode": ep_idx, "agent": agent,
                            "passage_width": round(W, 4),
                            "body_margin": round(bm, 4),
                            "success": r["success"], "rejected": r["rejected"],
                            "collision": r["collision"], "steps": r["steps"]})

        if ep_idx % 25 == 0 or ep_idx < 5:
            print(f"{ep_idx:4d}  {W:.3f}  {bm:+.3f}  "
                  f"{int(r_ora['success'] if oracle_attempt else -1):>3} "
                  f"{int(r_fix['success'] if fixed_attempt else -1):>3} "
                  f"{int(r_cal['success'] if calib_attempt else -1):>3}  "
                  f"{d_hat:.3f} ±{d_std:.3f} Δ={d_err:.3f}",
                  flush=True)

    # ── summary ────────────────────────────────────────────────────────────
    print("\n=== Per-agent summary ===")
    for agent in ["oracle", "fixed", "calib"]:
        rows = [r for r in ep_rows if r["agent"] == agent]
        sr  = np.mean([r["success"]  for r in rows])
        rjr = np.mean([r["rejected"] for r in rows])
        fnr = np.mean([r["rejected"] for r in rows
                       if r["body_margin"] > 0])  # false-neg among feasible
        print(f"  {agent:10s}: SR={sr:.3f}  reject_rate={rjr:.3f}  "
              f"false_neg_rate={fnr:.3f}")

    final = conv_rows[-1]
    print(f"\n  D_true={D_MIN_TRUE:.3f}  D_hat_final={final['d_hat']:.3f}  "
          f"err={final['d_err']:.4f}m ({final['d_err']/D_MIN_TRUE*100:.1f}%)")
    print(f"  90% CI: [{final['ci_lo']:.3f}, {final['ci_hi']:.3f}]")

    # ── rolling SR (window=50) for convergence plot ─────────────────────────
    window = 50
    for agent in ["oracle", "fixed", "calib"]:
        rows = [r for r in ep_rows if r["agent"] == agent]
        rolling = [np.mean([rows[j]["success"] for j in range(max(0, i-window+1), i+1)])
                   for i in range(len(rows))]
        for i, row in enumerate(conv_rows):
            row[f"rolling_sr_{agent}"] = round(rolling[i], 4)

    # ── save ─────────────────────────────────────────────────────────────────
    RESULTS.mkdir(parents=True, exist_ok=True)
    ep_path   = RESULTS / "dmin_calib_episodes.csv"
    conv_path = RESULTS / "dmin_calib_convergence.csv"
    with open(ep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys()))
        w.writeheader(); w.writerows(ep_rows)
    with open(conv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(conv_rows[0].keys()))
        w.writeheader(); w.writerows(conv_rows)
    print(f"\nSaved {ep_path}")
    print(f"Saved {conv_path}")


if __name__ == "__main__":
    main()
