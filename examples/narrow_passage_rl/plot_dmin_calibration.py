#!/usr/bin/env python3
"""Plot D_min self-calibration results.

Two-panel figure:
  Left:  D_hat(t) ± 90% CI converging toward D_true = 0.36m
  Right: Rolling success rate (window=50) for oracle / fixed / calibrated

Usage:
    python examples/narrow_passage_rl/plot_dmin_calibration.py
"""
import os, sys, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results", "narrow_passage_rl")
D_TRUE  = 0.36   # m

AGENT_STYLES = {
    "oracle": dict(color="tab:blue",   ls="-",  lw=2.2, label="Oracle (D_true = 0.36m)"),
    "fixed":  dict(color="tab:orange", ls="--", lw=1.8, label="Fixed wrong (D_hat = 0.56m)"),
    "calib":  dict(color="tab:green",  ls="-",  lw=2.2, label="Calibrated (ours)"),
}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv-csv", default=os.path.join(RESULTS, "dmin_calib_convergence.csv"))
    ap.add_argument("--ep-csv",   default=os.path.join(RESULTS, "dmin_calib_episodes.csv"))
    ap.add_argument("--out",      default=os.path.join(RESULTS, "dmin_calibration.png"))
    ap.add_argument("--window",   type=int, default=50)
    args = ap.parse_args()

    for p in [args.conv_csv, args.ep_csv]:
        if not os.path.exists(p):
            sys.exit(f"Missing: {p}")

    conv = load_csv(args.conv_csv)
    eps  = load_csv(args.ep_csv)

    episodes = [int(r["episode"])    for r in conv]
    d_hat    = [float(r["d_hat"])    for r in conv]
    d_err    = [float(r["d_err"])    for r in conv]
    d_std    = [float(r["d_std"])    for r in conv]
    ci_lo    = [float(r["ci_lo"])    for r in conv]
    ci_hi    = [float(r["ci_hi"])    for r in conv]

    # Rolling SR from convergence CSV
    rolling = {
        agent: [float(r[f"rolling_sr_{agent}"]) for r in conv]
        for agent in ["oracle", "fixed", "calib"]
    }

    # ── per-difficulty summary from ep CSV ───────────────────────────────
    def ep_sr(agent):
        rows = [r for r in eps if r["agent"] == agent]
        return (np.mean([float(r["success"]) for r in rows]),
                np.mean([float(r["rejected"]) for r in rows]))

    print("=== Final summary ===")
    for agent in ["oracle", "fixed", "calib"]:
        sr, rjr = ep_sr(agent)
        print(f"  {agent:10s}: SR={sr:.3f}  reject_rate={rjr:.3f}")
    print(f"  D_hat_final: {d_hat[-1]:.4f}m  err={d_err[-1]:.4f}m ({d_err[-1]/D_TRUE*100:.1f}%)")

    # ── figure ────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"$D_\mathrm{min}$ Self-Calibration: Learning the Robot's Capability Boundary",
                 fontsize=13)

    # ── Left: D_hat convergence ───────────────────────────────────────────
    ax1.plot(episodes, d_hat,  color="tab:green", lw=2.5, label=r"$\hat{D}_\mathrm{min}$ (posterior mean)")
    ax1.fill_between(episodes, ci_lo, ci_hi, alpha=0.20, color="tab:green",
                     label="90% credible interval")
    ax1.axhline(D_TRUE, color="tab:blue",   lw=1.8, ls="--",
                label=f"$D_{{\\mathrm{{true}}}}$ = {D_TRUE:.2f}m")
    ax1.axhline(d_hat[0], color="tab:orange", lw=1.5, ls=":",
                label=f"$D_{{\\mathrm{{init}}}}$ = {d_hat[0]:.2f}m (wrong guess)")

    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Estimated $D_\\mathrm{min}$ (m)", fontsize=12)
    ax1.set_ylim(0.20, 0.70)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(alpha=0.3)
    ax1.set_title(r"Posterior $\hat{D}_\mathrm{min}$ Converging to $D_\mathrm{true}$", fontsize=11)

    # annotate final error
    ax1.annotate(
        f"Final err = {d_err[-1]:.3f}m\n({d_err[-1]/D_TRUE*100:.0f}% of D_true)",
        xy=(episodes[-1], d_hat[-1]),
        xytext=(episodes[-1]*0.65, d_hat[-1] + 0.08),
        fontsize=8, color="tab:green",
        arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.2),
    )

    # ── Right: rolling success rate ────────────────────────────────────────
    for agent, style in AGENT_STYLES.items():
        ax2.plot(episodes, [r * 100 for r in rolling[agent]], **style)

    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel(f"Rolling Success Rate (%) [window={args.window}]", fontsize=12)
    ax2.set_ylim(-5, 108)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_title("SR Improvement as $\\hat{D}_\\mathrm{min}$ Converges", fontsize=11)

    # shade the "SR gap" between oracle and fixed
    ax2.fill_between(episodes,
                     [r * 100 for r in rolling["fixed"]],
                     [r * 100 for r in rolling["oracle"]],
                     alpha=0.12, color="gray", label="Gap closed by calibration")

    # annotate gap closure
    mid = len(episodes) // 2
    gap = rolling["oracle"][mid]*100 - rolling["fixed"][mid]*100
    ax2.annotate(f"gap = {gap:.0f}%",
                 xy=(episodes[mid], (rolling["oracle"][mid] + rolling["fixed"][mid]) * 50),
                 fontsize=8, color="gray", ha="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
