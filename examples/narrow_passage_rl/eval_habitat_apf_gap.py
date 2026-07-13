#!/usr/bin/env python3
"""APF+Gap classical reactive navigation baseline for NarrowPassageNav-v0.

Uses ONLY raw depth image + GPS/compass (same inputs as DD-PPO would use).
No trained model, no geometry sensor.

Reference:
  Potential field: Khatib 1986.
  Gap navigation: Meng & Burdick, "Narrow Passage Problem" 2002.

Key design (verified against Habitat convention):
  GPS gps[1] > 0  →  goal is to the LEFT  →  need positive angular velocity
  Depth pixel[0]  →  leftmost column  →  angle = -HFOV/2 (to the LEFT)
  sin(angle)      →  right obstacle (angle>0) gives positive torque (turn left) ✓
"""
import argparse, csv, gzip, json, math, sys
from pathlib import Path
import numpy as np

import habitat

from evaluation.logging_schema import fieldnames_for_rows, finalize_episode_row

RESULTS_DIR = Path("examples/narrow_passage_rl/results/narrow_passage_rl")
OUT_CSV = RESULTS_DIR / "habitat_apf_gap_episodes.csv"
VAL_JSON = Path("data/datasets/narrow_passage/val/val.json.gz")

# ── velocity action constants ────────────────────────────────────────────────
_LIN_MIN, _LIN_MAX = -0.15, 0.35
_ANG_MIN, _ANG_MAX = -45.0, 45.0


def _norm_lin(vx_ms: float) -> float:
    return max(-1.0, min(1.0, (vx_ms - _LIN_MIN) / (_LIN_MAX - _LIN_MIN) * 2.0 - 1.0))


def _norm_ang(wz_rads: float) -> float:
    return max(-1.0, min(1.0, (math.degrees(wz_rads) - _ANG_MIN) / (_ANG_MAX - _ANG_MIN) * 2.0 - 1.0))


def _make_action(lin_norm: float, ang_norm: float) -> dict:
    return {"action": "velocity_control",
            "action_args": {"linear_velocity": float(lin_norm),
                            "angular_velocity": float(ang_norm)}}


# ── APF+Gap controller ────────────────────────────────────────────────────────
HFOV = math.pi / 2          # 90° confirmed
SAFETY_DIST = 0.80          # m — repulsion onset
GAP_REL_RATIO = 0.60        # gap = depth > max_depth * ratio
GAP_MIN_ABS = 0.35          # m — absolute floor for gap threshold
MIN_GAP_PIX = 10            # minimum gap width in pixels (~3.5°)
K_ATT = 0.75                # goal attraction gain
K_GAP = 0.55                # gap alignment gain
K_REP = 0.45                # repulsion gain
STOP_DIST = 0.35            # m


def apf_gap_act(depth: np.ndarray, gps: np.ndarray) -> tuple:
    """Return (lin_norm, ang_norm) ∈ [-1, 1].

    gps[0] = distance_to_goal (m)
    gps[1] = angle_to_goal (rad), CCW-positive: positive = goal to LEFT ✓
    depth = (H, W, 1) float32 in metres
    """
    goal_dist  = float(gps[0])
    goal_angle = float(gps[1])   # CCW+, confirmed empirically

    # ── STOP ────────────────────────────────────────────────────────────────
    if goal_dist < STOP_DIST:
        # _norm_lin(0.0) = -0.4 → physical 0.0 m/s → triggers is_stop_called
        return _norm_lin(0.0), _norm_ang(0.0)

    H, W = depth.shape[:2]
    scan = depth[H // 2, :, 0].astype(np.float64)  # mid-height horizontal
    pix_angles = np.linspace(-HFOV / 2, HFOV / 2, W)  # left=-π/4, right=+π/4

    # ── Relative gap threshold ───────────────────────────────────────────────
    # Use max depth in scan so threshold adapts to narrow/wide scenes.
    max_d = max(float(scan.max()), GAP_MIN_ABS)
    gap_thr = max(GAP_MIN_ABS, max_d * GAP_REL_RATIO)
    is_open = scan > gap_thr

    # ── APF repulsion ────────────────────────────────────────────────────────
    w_rep = np.clip((SAFETY_DIST - scan) / SAFETY_DIST, 0.0, 1.0)
    w_rep[scan >= SAFETY_DIST] = 0.0
    n_close = max(1, int((scan < SAFETY_DIST).sum()))
    # sin(a) > 0 for right obstacles → positive angular (turn left, away) ✓
    repulsive_ang = float(np.sum(w_rep * np.sin(pix_angles))) / n_close

    # ── Gap detection ─────────────────────────────────────────────────────────
    gaps = []
    i = 0
    while i < W:
        if is_open[i]:
            j = i
            while j < W and is_open[j]:
                j += 1
            if j - i >= MIN_GAP_PIX:
                cx = pix_angles[(i + j) // 2]
                gaps.append((abs(cx - goal_angle), cx, j - i))
            i = j
        else:
            i += 1

    # ── Angular velocity ─────────────────────────────────────────────────────
    if gaps:
        gaps.sort(key=lambda x: x[0])
        gap_angle = gaps[0][1]
        ang_raw = K_ATT * goal_angle + K_GAP * gap_angle + K_REP * repulsive_ang
    else:
        # No clear gap: goal-directed with repulsion to escape corners
        ang_raw = K_ATT * goal_angle + 1.2 * K_REP * repulsive_ang

    # ── Linear velocity ──────────────────────────────────────────────────────
    # Check fraction of "open" pixels in forward ±30° cone
    fwd_mask = np.abs(pix_angles) < (math.pi / 6)
    fwd_open_frac = float(is_open[fwd_mask].mean())

    if fwd_open_frac > 0.25:
        lin_raw = 0.22    # clear ahead — full speed
    elif fwd_open_frac > 0.05:
        lin_raw = 0.08    # some opening — cautious
    elif abs(ang_raw) > 0.6:
        lin_raw = 0.0     # need to turn significantly — rotate in place
    else:
        lin_raw = 0.06    # slowly creep forward

    return _norm_lin(lin_raw), _norm_ang(ang_raw)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-episodes", type=int, default=-1)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    with gzip.open(VAL_JSON) as f:
        val_data = json.load(f)
    ep_meta = {e["episode_id"]: e["info"] for e in val_data["episodes"]}

    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "habitat/task=narrow_passage",
            "habitat.dataset.data_path=data/datasets/narrow_passage/val/val.json.gz",
            "habitat.dataset.split=val",
            "habitat.dataset.type=PointNav-v1",
            "habitat.simulator.habitat_sim_v0.allow_sliding=False",
        ],
    )

    rows = []
    with habitat.Env(config=config) as env:
        total = env.number_of_episodes
        n_eps = total if args.num_episodes <= 0 else min(args.num_episodes, total)
        print(f"[apf_gap] evaluating {n_eps} episodes")

        for ep_idx in range(n_eps):
            obs = env.reset()
            eid = env.current_episode.episode_id
            meta = ep_meta.get(eid, {})
            bm   = meta.get("body_margin", float("nan"))
            done = False; steps = 0

            while not done and steps < args.max_steps:
                depth = np.array(obs["depth"], dtype=np.float32)
                gps   = np.array(obs["pointgoal_with_gps_compass"], dtype=np.float32)
                lin_n, ang_n = apf_gap_act(depth, gps)
                obs = env.step(_make_action(lin_n, ang_n))
                steps += 1
                if env.episode_over:
                    done = True

            m = env.get_metrics()
            success = float(m.get("narrow_passage_success", 0.0))
            row = finalize_episode_row({
                "episode_id":  eid,
                "scene_id": str(env.current_episode.scene_id).split("/")[-2],
                "split": "val",
                "seed": "deterministic",
                "method": "APF+Gap",
                "body_margin": bm,
                "delta_d":     2.0 * bm,
                "difficulty":  meta.get("difficulty", "unknown"),
                "success":     success,
                "collision": float(m.get("narrow_passage_collision", float("nan"))),
                "steps":       steps,
            }, belief_available=False, mode_interface_available=False, final_mode="apf")
            rows.append(row)
            print(f"  ep {ep_idx:3d}  [{meta.get('difficulty','?'):6s}]"
                  f"  bm={bm:.3f}  ΔD={2*bm:.3f}"
                  f"  success={int(success)}  steps={steps}", flush=True)

    n_succ = sum(r["success"] > 0.5 for r in rows)
    print(f"\nTotal: {n_succ}/{len(rows)} = {n_succ/len(rows):.3f}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_for_rows(rows))
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
