#!/usr/bin/env python3
"""Plot ΔD = 2×body_margin phase-transition curve: FSM vs PPO success rate.

Usage:
    python3 examples/narrow_passage_rl/plot_delta_d_phase.py
    python3 examples/narrow_passage_rl/plot_delta_d_phase.py --out results/narrow_passage_rl/delta_d_phase.png

Inputs (auto-discovered in results/narrow_passage_rl/):
    habitat_fsm_v2_episodes.csv       — FSM per-episode results
    habitat_ppo_v2_episodes.csv       — PPO per-episode results (written by eval_ppo_per_episode.py)
    val.json.gz                       — episode metadata (body_margin)

Both CSVs must have 'episode_id' and 'success' columns.
FSM CSV may lack 'body_margin'; it is joined from val.json.gz.
PPO CSV already contains 'body_margin' (written by the eval script).
"""
import os, sys, gzip, json, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results", "narrow_passage_rl")
VAL_JSON = os.path.join(ROOT, "..", "..", "data", "datasets",
                         "narrow_passage", "val", "val.json.gz")


def load_val_meta():
    with gzip.open(VAL_JSON) as f:
        d = json.load(f)
    return {
        ep["episode_id"]: {
            "body_margin":   ep["info"]["body_margin"],
            "delta_d":       2.0 * ep["info"]["body_margin"],
            "difficulty":    ep["info"]["difficulty"],
            "passage_width": ep["info"]["passage_width"],
        }
        for ep in d["episodes"]
    }


def load_csv(path, meta):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            eid = r["episode_id"]
            bm = float(r.get("body_margin", "nan"))
            if np.isnan(bm) and eid in meta:
                bm = meta[eid]["body_margin"]
            rows.append({
                "episode_id": eid,
                "body_margin": bm,
                "delta_d":     2.0 * bm,
                "difficulty":  r.get("difficulty") or meta.get(eid, {}).get("difficulty", "?"),
                "success":     float(r["success"]),
            })
    return rows


def bin_success(rows, bins):
    """Bin episodes by delta_d and compute success rate per bin."""
    centers, rates, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        eps = [r for r in rows if lo <= r["delta_d"] < hi]
        if eps:
            centers.append((lo + hi) / 2)
            rates.append(np.mean([r["success"] for r in eps]))
            counts.append(len(eps))
    return np.array(centers), np.array(rates), np.array(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsm-csv",  default=os.path.join(RESULTS, "habitat_fsm_v2_episodes.csv"))
    ap.add_argument("--ppo-csv",  default=os.path.join(RESULTS, "habitat_ppo_v2_episodes.csv"))
    ap.add_argument("--out",      default=os.path.join(RESULTS, "delta_d_phase.png"))
    ap.add_argument("--bin-width", type=float, default=0.15,
                    help="ΔD bin width in metres (default 0.15)")
    args = ap.parse_args()

    for p in [args.fsm_csv, args.ppo_csv]:
        if not os.path.exists(p):
            sys.exit(f"Missing: {p}")

    meta = load_val_meta()
    fsm_rows = load_csv(args.fsm_csv, meta)
    ppo_rows = load_csv(args.ppo_csv, meta)

    all_dd = [r["delta_d"] for r in fsm_rows + ppo_rows if not np.isnan(r["delta_d"])]
    dd_max = max(all_dd) + 0.01
    bins = np.arange(0.0, dd_max + args.bin_width, args.bin_width)

    fsm_c, fsm_r, fsm_n = bin_success(fsm_rows, bins)
    ppo_c, ppo_r, ppo_n = bin_success(ppo_rows, bins)

    # ── per-difficulty summary ──────────────────────────────────────────
    print("\n=== Per-difficulty summary ===")
    for diff in ["narrow", "normal", "wide"]:
        fn = [r for r in fsm_rows if r["difficulty"] == diff]
        pn = [r for r in ppo_rows if r["difficulty"] == diff]
        if fn:
            print(f"  FSM  {diff:6s}: {np.mean([r['success'] for r in fn]):.3f} ({sum(r['success']>0.5 for r in fn)}/{len(fn)})")
        if pn:
            print(f"  PPO  {diff:6s}: {np.mean([r['success'] for r in pn]):.3f} ({sum(r['success']>0.5 for r in pn)}/{len(pn)})")

    print(f"\n=== Overall ===")
    print(f"  FSM: {np.mean([r['success'] for r in fsm_rows]):.3f} ({sum(r['success']>0.5 for r in fsm_rows)}/{len(fsm_rows)})")
    print(f"  PPO: {np.mean([r['success'] for r in ppo_rows]):.3f} ({sum(r['success']>0.5 for r in ppo_rows)}/{len(ppo_rows)})")

    # ── plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ΔD Phase Transition: FSM vs PPO\n"
                 r"$\Delta D = D_\mathrm{passage} - D_\mathrm{robot} = 2 \times \mathrm{body\_margin}$",
                 fontsize=13)

    # ── left: line plot ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(fsm_c, fsm_r * 100, "o-", color="tab:blue",  linewidth=2, label="FSM (ours)")
    ax.plot(ppo_c, ppo_r * 100, "s--", color="tab:orange", linewidth=2, label="PPO baseline")

    # shade "narrow" zone (ΔD < 0.30m, i.e. body_margin < 0.15m)
    ax.axvspan(0, 0.30, alpha=0.08, color="red",   label="narrow zone (ΔD < 0.30m)")
    ax.axvspan(0.30, 0.80, alpha=0.05, color="yellow", label="normal zone")
    ax.axvspan(0.80, bins[-1], alpha=0.05, color="green", label="wide zone")

    ax.set_xlabel("ΔD = 2 × body_margin  (m)", fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_ylim(-5, 105)
    ax.set_xlim(left=0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Success Rate vs ΔD", fontsize=11)

    # annotate bin counts
    for x, n in zip(fsm_c, fsm_n):
        ax.annotate(f"n={n}", xy=(x, -4), ha="center", fontsize=7, color="gray")

    # ── right: bar chart grouped by difficulty ───────────────────────────
    ax2 = axes[1]
    diff_order = ["narrow", "normal", "wide"]
    diff_colors = {"narrow": "#e74c3c", "normal": "#f39c12", "wide": "#27ae60"}

    x_ticks = np.arange(len(diff_order))
    bar_w = 0.35

    fsm_bars, ppo_bars, fsm_ns, ppo_ns = [], [], [], []
    for diff in diff_order:
        fn = [r for r in fsm_rows if r["difficulty"] == diff]
        pn = [r for r in ppo_rows if r["difficulty"] == diff]
        fsm_bars.append(np.mean([r["success"] for r in fn]) * 100 if fn else 0)
        ppo_bars.append(np.mean([r["success"] for r in pn]) * 100 if pn else 0)
        fsm_ns.append(len(fn))
        ppo_ns.append(len(pn))

    bars1 = ax2.bar(x_ticks - bar_w/2, fsm_bars, bar_w, label="FSM", color="tab:blue", alpha=0.85)
    bars2 = ax2.bar(x_ticks + bar_w/2, ppo_bars, bar_w, label="PPO", color="tab:orange", alpha=0.85)

    for bar, n in zip(bars1, fsm_ns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"n={n}", ha="center", va="bottom", fontsize=8)
    for bar, n in zip(bars2, ppo_ns):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"n={n}", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f"{d}\n(ΔD {'<0.30' if d=='narrow' else '0.30–0.80' if d=='normal' else '>0.80'}m)"
                         for d in diff_order], fontsize=9)
    ax2.set_ylabel("Success Rate (%)", fontsize=12)
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("Success Rate by Difficulty Bucket", fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
