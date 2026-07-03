#!/usr/bin/env python3
"""FSM ablation study on Habitat NarrowPassageNav-v0.

Runs FSM variants to isolate the contribution of each controller component:
  full                  — complete geometry-FSM
  no_recovery           — disable backward recovery on collision/stuck
  no_alignment          — disable both heading and lateral alignment
  no_heading_alignment  — keep lateral centering, remove heading-to-goal alignment
  no_lateral_alignment  — keep heading-to-goal alignment, remove lateral centering

Usage:
    conda run -n habitat python3 examples/narrow_passage_rl/eval_habitat_fsm_ablations.py
    conda run -n habitat python3 examples/narrow_passage_rl/eval_habitat_fsm_ablations.py \
        --variants full no_recovery no_alignment no_heading_alignment no_lateral_alignment
    conda run -n habitat python3 examples/narrow_passage_rl/eval_habitat_fsm_ablations.py \
        --split extreme_narrow --heading-perturb-deg 60 --lateral-perturb-m 0.2
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

import habitat

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"

_LIN_MIN, _LIN_MAX = -0.15, 0.35
_ANG_MIN, _ANG_MAX = -45.0, 45.0


def _norm_lin(vx: float) -> float:
    return float(np.clip((vx - _LIN_MIN) / (_LIN_MAX - _LIN_MIN) * 2.0 - 1.0, -1.0, 1.0))

def _norm_ang(wz_rad: float) -> float:
    wz_deg = wz_rad * 180.0 / math.pi
    return float(np.clip((wz_deg - _ANG_MIN) / (_ANG_MAX - _ANG_MIN) * 2.0 - 1.0, -1.0, 1.0))

def _act(lin: float, ang_rad: float) -> dict:
    return {"action": "velocity_control",
            "action_args": {"linear_velocity": _norm_lin(lin),
                            "angular_velocity": _norm_ang(ang_rad)}}

def _stop_act() -> dict:
    return {"action": "velocity_control",
            "action_args": {"linear_velocity": _norm_lin(0.0),
                            "angular_velocity": _norm_ang(0.0)}}


def _dist_to_goal(env) -> float:
    ap = np.array(env.sim.get_agent_state().position, dtype=np.float32)
    gp = np.array(env.current_episode.goals[0].position, dtype=np.float32)
    return float(np.linalg.norm((gp - ap)[[0, 2]]))


def _heading_error(env) -> float:
    state = env.sim.get_agent_state()
    ap = np.array(state.position, dtype=np.float32)
    gp = np.array(env.current_episode.goals[0].position, dtype=np.float32)
    delta = gp - ap
    goal_yaw = math.atan2(-float(delta[0]), -float(delta[2]))
    rot = state.rotation
    yaw = math.atan2(2.0 * (rot.real * rot.y + rot.x * rot.z),
                     1.0 - 2.0 * (rot.y**2 + rot.z**2))
    err = goal_yaw - yaw
    return float((err + math.pi) % (2 * math.pi) - math.pi)


def _yaw_from_rotation(rot) -> float:
    return math.atan2(2.0 * (rot.real * rot.y + rot.x * rot.z),
                      1.0 - 2.0 * (rot.y**2 + rot.z**2))


def _yaw_to_quat(yaw: float) -> list:
    half = 0.5 * yaw
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _apply_heading_perturb(env, degrees: float) -> None:
    if abs(degrees) < 1e-6:
        return
    state = env.sim.get_agent_state()
    yaw = _yaw_from_rotation(state.rotation) + math.radians(degrees)
    env.sim.set_agent_state(
        state.position,
        _yaw_to_quat(yaw),
        reset_sensors=False,
    )


def _apply_lateral_perturb(env, meters: float) -> None:
    """Shift start pose sideways in the agent frame.

    Positive values move to the agent's right.  This stress-tests lateral
    centering without changing the episode goal.
    """
    if abs(meters) < 1e-6:
        return
    state = env.sim.get_agent_state()
    yaw = _yaw_from_rotation(state.rotation)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float32)
    pos = np.array(state.position, dtype=np.float32) + meters * right
    env.sim.set_agent_state(pos, state.rotation, reset_sensors=False)


def _apply_start_distance_shift(env, meters: float) -> None:
    """Move start pose along the current start-goal line.

    Positive values move the robot closer to the goal; negative values move it
    farther from the entrance.  Large values may leave the navigable region and
    should be used as stress tests, not dataset regeneration.
    """
    if abs(meters) < 1e-6:
        return
    state = env.sim.get_agent_state()
    ap = np.array(state.position, dtype=np.float32)
    gp = np.array(env.current_episode.goals[0].position, dtype=np.float32)
    direction = gp - ap
    direction[1] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return
    pos = ap + meters * direction / norm
    env.sim.set_agent_state(pos, state.rotation, reset_sensors=False)


def _stress_features(features: np.ndarray, args, rng: np.random.Generator) -> np.ndarray:
    """Apply feature-level sensor stress to the 19-D geometry vector.

    This simulates noisy depth/clearance estimates while leaving simulator state
    and success/collision metrics unchanged.
    """
    out = np.array(features, dtype=np.float32).copy()
    if args.feature_noise_std > 0.0:
        # Depth sectors, clearances, passage width, body margin, and lateral offset.
        idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11])
        out[idx] += rng.normal(0.0, args.feature_noise_std, size=len(idx)).astype(np.float32)
    if args.depth_dropout_prob > 0.0:
        mask = rng.random(6) < args.depth_dropout_prob
        depth = out[:6].copy()
        depth[mask] = 0.0
        out[:6] = depth
    out[:10] = np.clip(out[:10], -1.0, 10.0)
    return out


def _wz(heading_error: float, lateral_offset: float, gain: float = 1.0) -> float:
    return float(np.clip((0.9 * heading_error - 1.2 * lateral_offset) * gain, -1.2, 1.2))


def _wz_variant(heading_error: float, lateral_offset: float, gain: float, variant: str) -> float:
    if variant == "no_alignment":
        return 0.0
    if variant == "no_heading_alignment":
        return float(np.clip((-1.2 * lateral_offset) * gain, -1.2, 1.2))
    if variant == "no_lateral_alignment":
        return float(np.clip((0.9 * heading_error) * gain, -1.2, 1.2))
    return _wz(heading_error, lateral_offset, gain)


def run_episode(env, features, variant: str, args, rng: np.random.Generator) -> dict:
    steps = 0
    any_collision = False
    min_bm = float("inf")
    near_collision = False
    recover_triggers = 0
    done = False

    max_steps = args.max_steps
    max_recover = args.max_recover_steps

    while not done and steps < max_steps:
        bm = float(features[9])
        if bm < min_bm:
            min_bm = bm
        if bm < 0.05:
            near_collision = True
        if float(features[16]) > 0.5:
            any_collision = True

        he = _heading_error(env)
        dist = _dist_to_goal(env)
        lat = float(features[11])
        stuck = float(features[15])
        collision_flag = float(features[16]) > 0.5

        # ── decide mode ───────────────────────────────────────────────────
        if dist < 0.25:
            obs = env.step(_stop_act())
            steps += 1
            features = _stress_features(
                np.array(obs.get("narrow_passage_features", features), np.float32),
                args, rng,
            )
            if env.episode_over:
                done = True
            continue

        # Shared logic for "full" and "no_recovery"
        if variant != "no_alignment" and abs(he) > 0.7:
            mode = "ALIGN"
        elif collision_flag or stuck > 0.80:
            mode = "RECOVER" if variant != "no_recovery" else "COMMIT"
        elif abs(he) > 0.45 or abs(lat) > 0.25:
            mode = "EXPLORE"
        else:
            mode = "COMMIT"

        if mode == "ALIGN":
            obs = env.step(_act(0.0, _wz_variant(he, lat, 1.8, variant)))
        elif mode == "COMMIT":
            obs = env.step(_act(0.20, _wz_variant(he, lat, 1.0, variant)))
        elif mode == "EXPLORE":
            obs = env.step(_act(0.08, _wz_variant(he, lat, 1.6, variant)))
        elif mode == "RECOVER":
            # back up + reorient for up to max_recover steps
            recover_triggers += 1
            for _ in range(max_recover):
                he_r = _heading_error(env)
                lat_r = float(features[11])
                obs = env.step(_act(-0.12, _wz_variant(he_r, lat_r, 0.4, variant)))
                steps += 1
                features = _stress_features(
                    np.array(obs.get("narrow_passage_features", features), np.float32),
                    args, rng,
                )
                if float(features[9]) < min_bm:
                    min_bm = float(features[9])
                if float(features[16]) > 0.5:
                    any_collision = True
                if env.episode_over or steps >= max_steps:
                    done = True
                    break
                he_r = _heading_error(env)
                if (abs(he_r) < 0.35 and abs(float(features[11])) < 0.20
                        and float(features[15]) < 0.30):
                    break
            if not done:
                obs = env.step(_act(0.0, 0.0))  # flush
                steps += 1
                features = _stress_features(
                    np.array(obs.get("narrow_passage_features", features), np.float32),
                    args, rng,
                )
                if env.episode_over:
                    done = True
            continue
        else:
            obs = env.step(_stop_act())

        steps += 1
        features = _stress_features(
            np.array(obs.get("narrow_passage_features", features), np.float32),
            args, rng,
        )
        if env.episode_over:
            done = True

    metrics = env.get_metrics()
    if not np.isfinite(min_bm):
        min_bm = float(features[9])
    return {
        "steps": steps,
        "success": float(metrics.get("narrow_passage_success", 0.0)),
        "collision": float(any_collision),
        "stuck": float(metrics.get("narrow_passage_stuck", 0.0)),
        "near_collision": float(near_collision),
        "min_clearance": float(min_bm),
        "recover_triggers": recover_triggers,
    }


def eval_variant(variant: str, args) -> list:
    data_path = args.data_path.format(split=args.split)
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "habitat/task=narrow_passage",
            f"habitat.dataset.data_path={data_path}",
            f"habitat.dataset.split={args.split}",
            "habitat.dataset.type=PointNav-v1",
            "habitat.simulator.habitat_sim_v0.allow_sliding=False",
        ],
    )

    all_stats = []
    rng = np.random.default_rng(args.seed)
    with habitat.Env(config=config) as env:
        total = env.number_of_episodes
        n = total if args.num_episodes <= 0 else min(args.num_episodes, total)
        print(f"\n[{variant}] {n} episodes", flush=True)

        for ep_idx in range(n):
            obs = env.reset()
            episode = env.current_episode
            info = episode.info if hasattr(episode, "info") and episode.info else {}
            _apply_heading_perturb(env, args.heading_perturb_deg)
            _apply_lateral_perturb(env, args.lateral_perturb_m)
            _apply_start_distance_shift(env, args.start_distance_shift_m)
            features = np.array(
                obs.get("narrow_passage_features", np.zeros(19, np.float32)), np.float32
            )
            features = _stress_features(features, args, rng)
            stat = run_episode(env, features, variant, args, rng)
            stat["episode_id"] = str(episode.episode_id)
            stat["difficulty"] = str(info.get("difficulty", "?"))
            stat["body_margin"] = round(float(info.get("body_margin", float("nan"))), 4)
            stat["heading_perturb_deg"] = float(args.heading_perturb_deg)
            stat["lateral_perturb_m"] = float(args.lateral_perturb_m)
            stat["start_distance_shift_m"] = float(args.start_distance_shift_m)
            stat["feature_noise_std"] = float(args.feature_noise_std)
            stat["depth_dropout_prob"] = float(args.depth_dropout_prob)
            stat["variant"] = variant
            all_stats.append(stat)

            if ep_idx % 20 == 0 or ep_idx < 5:
                print(f"  ep {ep_idx:3d}  {stat['difficulty']:6s}  "
                      f"success={stat['success']:.0f}  steps={stat['steps']:3d}", flush=True)

    sr = np.mean([s["success"] for s in all_stats])
    cr = np.mean([s["collision"] for s in all_stats])
    print(f"[{variant}] SR={sr:.3f}  collision={cr:.3f}")
    for diff in ["narrow", "normal", "wide"]:
        grp = [s for s in all_stats if s["difficulty"] == diff]
        if grp:
            dsr = np.mean([s["success"] for s in grp])
            print(f"  {diff}: {dsr:.3f} ({sum(s['success']>0.5 for s in grp)}/{len(grp)})")
    return all_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=[
                        "full",
                        "no_recovery",
                        "no_alignment",
                        "no_heading_alignment",
                        "no_lateral_alignment",
                    ],
                    choices=[
                        "full",
                        "no_recovery",
                        "no_alignment",
                        "no_heading_alignment",
                        "no_lateral_alignment",
                    ])
    ap.add_argument("--data-path", default="data/datasets/narrow_passage/{split}/{split}.json.gz")
    ap.add_argument("--split", default="val")
    ap.add_argument("--num-episodes", type=int, default=-1)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--max-recover-steps", type=int, default=30)
    ap.add_argument("--heading-perturb-deg", type=float, default=0.0,
                    help="Rotate the start pose after reset; stress-tests alignment modules")
    ap.add_argument("--lateral-perturb-m", type=float, default=0.0,
                    help="Shift start pose sideways in agent frame; + = right")
    ap.add_argument("--start-distance-shift-m", type=float, default=0.0,
                    help="Move start along start-goal line; + = closer to goal")
    ap.add_argument("--feature-noise-std", type=float, default=0.0,
                    help="Gaussian noise std applied to depth/clearance geometry features")
    ap.add_argument("--depth-dropout-prob", type=float, default=0.0,
                    help="Probability of dropping each of the six depth-sector features")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for feature noise/dropout stress tests")
    ap.add_argument("--output-csv", type=Path,
                    default=RESULTS / "habitat_fsm_ablation_episodes.csv")
    args = ap.parse_args()

    all_stats = []
    for variant in args.variants:
        all_stats.extend(eval_variant(variant, args))

    # ── summary table ─────────────────────────────────────────────────────
    print("\n=== Ablation Summary ===")
    print(f"{'Variant':22s}  {'SR':>5}  {'Narrow':>6}  {'Normal':>6}  {'Wide':>6}  {'Col':>5}  {'Steps':>6}")
    for variant in args.variants:
        rows = [s for s in all_stats if s["variant"] == variant]
        sr = np.mean([s["success"] for s in rows])
        cr = np.mean([s["collision"] for s in rows])
        avg_steps = np.mean([s["steps"] for s in rows])
        def dsr(diff):
            g = [s for s in rows if s["difficulty"] == diff]
            return np.mean([s["success"] for s in g]) if g else float("nan")
        print(f"{variant:22s}  {sr:.3f}  {dsr('narrow'):6.3f}  "
              f"{dsr('normal'):6.3f}  {dsr('wide'):6.3f}  {cr:.3f}  {avg_steps:6.1f}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_stats[0].keys()))
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"\n[write] {args.output_csv}")


if __name__ == "__main__":
    main()
