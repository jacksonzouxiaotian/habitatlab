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
    ap.add_argument("--fsm-csv",    default=os.path.join(RESULTS, "habitat_fsm_v2_episodes.csv"))
    ap.add_argument("--ppo-csv",    default=os.path.join(RESULTS, "habitat_ppo_v2_episodes.csv"))
    ap.add_argument("--apf-csv",    default=os.path.join(RESULTS, "habitat_apf_gap_episodes.csv"))
    ap.add_argument("--policy-csv", default=os.path.join(RESULTS, "habitat_ppo_policy_episodes.csv"),
                    help="Per-episode CSV from eval_habitat_ppo_policy.py (optional)")
    ap.add_argument("--out",      default=os.path.join(RESULTS, "delta_d_phase.png"))
    ap.add_argument("--bin-width", type=float, default=0.15,
                    help="ΔD bin width in metres (default 0.15)")
    args = ap.parse_args()

    for p in [args.fsm_csv, args.ppo_csv]:
        if not os.path.exists(p):
            sys.exit(f"Missing: {p}")

    meta = load_val_meta()
    fsm_rows    = load_csv(args.fsm_csv, meta)
    ppo_rows    = load_csv(args.ppo_csv, meta)
    apf_rows    = load_csv(args.apf_csv, meta) if os.path.exists(args.apf_csv) else []
    policy_rows = load_csv(args.policy_csv, meta) if os.path.exists(args.policy_csv) else []

    all_dd = [r["delta_d"] for r in fsm_rows + ppo_rows + apf_rows + policy_rows
              if not np.isnan(r["delta_d"])]
    dd_max = max(all_dd) + 0.01
    bins = np.arange(0.0, dd_max + args.bin_width, args.bin_width)

    fsm_c,    fsm_r,    fsm_n    = bin_success(fsm_rows, bins)
    ppo_c,    ppo_r,    ppo_n    = bin_success(ppo_rows, bins)
    apf_c,    apf_r,    apf_n    = bin_success(apf_rows, bins)    if apf_rows    else (np.array([]), np.array([]), np.array([]))
    policy_c, policy_r, policy_n = bin_success(policy_rows, bins) if policy_rows else (np.array([]), np.array([]), np.array([]))

    # ── per-difficulty summary ──────────────────────────────────────────
    print("\n=== Per-difficulty summary ===")
    methods = [("FSM", fsm_rows), ("APF+Gap", apf_rows), ("PPO-SB3", ppo_rows)]
    if policy_rows:
        methods.append(("PPO-Policy", policy_rows))
    for diff in ["narrow", "normal", "wide"]:
        for name, rws in methods:
            eps = [r for r in rws if r["difficulty"] == diff]
            if eps:
                print(f"  {name:12s} {diff:6s}: {np.mean([r['success'] for r in eps]):.3f}"
                      f" ({sum(r['success']>0.5 for r in eps)}/{len(eps)})")

    print(f"\n=== Overall ===")
    for name, rws in methods:
        if rws:
            print(f"  {name:12s}: {np.mean([r['success'] for r in rws]):.3f}"
                  f" ({sum(r['success']>0.5 for r in rws)}/{len(rws)})")

    # ── plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("ΔD Phase Transition: Geometry-FSM vs Baselines\n"
                 r"$\Delta D = D_\mathrm{passage} - D_\mathrm{robot} = 2 \times \mathrm{body\_margin}$",
                 fontsize=13)

    # ── left: line plot ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(fsm_c, fsm_r * 100, "o-", color="tab:blue",  linewidth=2.5, label="Geometry-FSM (ours)")
    if len(apf_c):
        ax.plot(apf_c, apf_r * 100, "^-", color="tab:green", linewidth=2, label="APF+Gap (classical)")
    if len(policy_c):
        ax.plot(policy_c, policy_r * 100, "D-", color="tab:purple", linewidth=2, label="PPO w/ geometry sensor")
    ax.plot(ppo_c, ppo_r * 100, "s--", color="tab:orange", linewidth=1.8, label="PPO baseline (SB3)")

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

    x_ticks = np.arange(len(diff_order))

    def diff_bars(rows):
        bars, ns = [], []
        for diff in diff_order:
            grp = [r for r in rows if r["difficulty"] == diff]
            bars.append(np.mean([r["success"] for r in grp]) * 100 if grp else 0)
            ns.append(len(grp))
        return bars, ns

    fsm_bars,    fsm_ns    = diff_bars(fsm_rows)
    apf_bars,    apf_ns    = diff_bars(apf_rows)
    policy_bars, policy_ns = diff_bars(policy_rows)
    ppo_bars,    ppo_ns    = diff_bars(ppo_rows)

    # Dynamically choose bar width based on how many methods have data
    active = [(fsm_bars, fsm_ns, "Geometry-FSM (ours)", "tab:blue"),
              (apf_bars, apf_ns, "APF+Gap", "tab:green") if apf_rows else None,
              (policy_bars, policy_ns, "PPO w/ geometry", "tab:purple") if policy_rows else None,
              (ppo_bars, ppo_ns, "PPO-SB3 baseline", "tab:orange")]
    active = [m for m in active if m is not None]
    n_methods = len(active)
    bar_w = 0.8 / n_methods
    offsets = np.linspace(-(n_methods - 1) / 2 * bar_w, (n_methods - 1) / 2 * bar_w, n_methods)

    for (bars, ns, label, color), offset in zip(active, offsets):
        b = ax2.bar(x_ticks + offset, bars, bar_w * 0.92, label=label, color=color, alpha=0.85)
        for bar, n in zip(b, ns):
            if n > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f"{n}", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([f"{d}\n(ΔD {'<0.30' if d=='narrow' else '0.30–0.80' if d=='normal' else '>0.80'}m)"
                         for d in diff_order], fontsize=9)
    ax2.set_ylabel("Success Rate (%)", fontsize=12)
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("Success Rate by Difficulty Bucket", fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
