#!/usr/bin/env python3
"""Habitat HM3D stress validation for narrow-passage controllers.

The nominal HM3D mined anchors are intentionally kept as anchor validation.  This
script adds controlled perturbations so the table can separate alignment,
recovery, sensor robustness, and clearance safety.
"""

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

import habitat

from eval_habitat_apf_gap import apf_gap_act


RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"
DEFAULT_CSV = RESULTS / "habitat_stress_validation.csv"
DEFAULT_MD = RESULTS / "paper_table_habitat_stress.md"
DEFAULT_TEX = RESULTS / "paper_table_habitat_stress.tex"

FEATURE_DIM = 19
_LIN_MIN, _LIN_MAX = -0.15, 0.35
_ANG_MIN, _ANG_MAX = -45.0, 45.0


@dataclass(frozen=True)
class StressCase:
    name: str
    yaw_deg: float = 0.0
    lateral_m: float = 0.0
    depth_dropout: float = 0.0
    feature_noise: float = 0.0
    extreme_narrow_only: bool = False


def _norm_lin(vx: float) -> float:
    return float(np.clip((vx - _LIN_MIN) / (_LIN_MAX - _LIN_MIN) * 2.0 - 1.0, -1.0, 1.0))


def _norm_ang(wz_rad: float) -> float:
    wz_deg = math.degrees(wz_rad)
    return float(np.clip((wz_deg - _ANG_MIN) / (_ANG_MAX - _ANG_MIN) * 2.0 - 1.0, -1.0, 1.0))


def _act(lin: float, ang_rad: float) -> Dict:
    return {
        "action": "velocity_control",
        "action_args": {
            "linear_velocity": _norm_lin(lin),
            "angular_velocity": _norm_ang(ang_rad),
        },
    }


def _stop_act() -> Dict:
    return _act(0.0, 0.0)


def _yaw_from_rotation(rot) -> float:
    return math.atan2(
        2.0 * (rot.real * rot.y + rot.x * rot.z),
        1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z),
    )


def _yaw_to_quat(yaw: float) -> List[float]:
    half = 0.5 * yaw
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _goal_position(env) -> np.ndarray:
    return np.array(env.current_episode.goals[0].position, dtype=np.float32)


def _agent_position(env) -> np.ndarray:
    return np.array(env.sim.get_agent_state().position, dtype=np.float32)


def _dist_to_goal(env) -> float:
    delta = _goal_position(env) - _agent_position(env)
    return float(np.linalg.norm(delta[[0, 2]]))


def _heading_error(env) -> float:
    state = env.sim.get_agent_state()
    delta = _goal_position(env) - np.array(state.position, dtype=np.float32)
    goal_yaw = math.atan2(-float(delta[0]), -float(delta[2]))
    yaw = _yaw_from_rotation(state.rotation)
    err = goal_yaw - yaw
    return float((err + math.pi) % (2.0 * math.pi) - math.pi)


def _lateral_offset(env) -> float:
    episode = env.current_episode
    start = np.array(getattr(episode, "start_position", _agent_position(env)), dtype=np.float32)
    goal = _goal_position(env)
    pos = _agent_position(env)
    line = goal[[0, 2]] - start[[0, 2]]
    norm = float(np.linalg.norm(line))
    if norm < 1e-6:
        return 0.0
    line = line / norm
    rel = pos[[0, 2]] - start[[0, 2]]
    return float(rel[0] * line[1] - rel[1] * line[0])


def _refresh_state_features(env, features: np.ndarray) -> np.ndarray:
    out = np.asarray(features, dtype=np.float32).copy()
    if out.shape[0] < FEATURE_DIM:
        out = np.zeros(FEATURE_DIM, dtype=np.float32)
    out[10] = _heading_error(env)
    out[11] = _lateral_offset(env)
    out[12] = _dist_to_goal(env)
    return out


def _apply_heading_perturb(env, degrees: float) -> None:
    if abs(degrees) < 1e-6:
        return
    state = env.sim.get_agent_state()
    yaw = _yaw_from_rotation(state.rotation) + math.radians(degrees)
    env.sim.set_agent_state(state.position, _yaw_to_quat(yaw), reset_sensors=False)


def _apply_lateral_perturb(env, meters: float) -> None:
    if abs(meters) < 1e-6:
        return
    state = env.sim.get_agent_state()
    yaw = _yaw_from_rotation(state.rotation)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float32)
    pos = np.array(state.position, dtype=np.float32) + meters * right
    env.sim.set_agent_state(pos, state.rotation, reset_sensors=False)


def _features_from_obs(env, obs: Dict, stress: StressCase, rng: np.random.Generator) -> np.ndarray:
    features = np.asarray(
        obs.get("narrow_passage_features", np.zeros(FEATURE_DIM, dtype=np.float32)),
        dtype=np.float32,
    )
    features = _refresh_state_features(env, features)
    if stress.feature_noise > 0.0:
        # Perturb interpretable geometry estimates, not previous action flags.
        idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11], dtype=int)
        features[idx] += rng.normal(0.0, stress.feature_noise, size=len(idx)).astype(np.float32)
    if stress.depth_dropout > 0.0:
        depth_sectors = features[:6].copy()
        mask = rng.random(6) < stress.depth_dropout
        depth_sectors[mask] = 0.0
        features[:6] = depth_sectors
    features[:10] = np.clip(features[:10], -1.0, 10.0)
    return features


def _depth_from_obs(obs: Dict, stress: StressCase, rng: np.random.Generator) -> np.ndarray:
    depth = np.asarray(obs.get("depth"), dtype=np.float32).copy()
    if stress.feature_noise > 0.0:
        depth = depth + rng.normal(0.0, stress.feature_noise, size=depth.shape).astype(np.float32)
    if stress.depth_dropout > 0.0:
        h, w = depth.shape[:2]
        thirds = [(0, w // 3), (w // 3, 2 * w // 3), (2 * w // 3, w)]
        for lo, hi in thirds:
            if rng.random() < stress.depth_dropout:
                depth[:, lo:hi, ...] = 0.0
    return np.clip(depth, 0.0, 10.0)


def _wz(heading_error: float, lateral_offset: float, gain: float, method: str) -> float:
    if method == "fsm_no_heading_alignment":
        raw = -1.2 * lateral_offset
    elif method == "fsm_no_lateral_alignment":
        raw = 0.9 * heading_error
    else:
        raw = 0.9 * heading_error - 1.2 * lateral_offset
    return float(np.clip(raw * gain, -1.2, 1.2))


def _fsm_action(env, features: np.ndarray, method: str):
    heading_error = _heading_error(env)
    lateral = float(features[11])
    stuck = float(features[15])
    collision = float(features[16]) > 0.5
    dist = _dist_to_goal(env)

    if dist < 0.25:
        return _stop_act(), "STOP"
    if method != "fsm_no_heading_alignment" and abs(heading_error) > 0.7:
        return _act(0.0, _wz(heading_error, lateral, 1.8, method)), "ALIGN"
    if collision or stuck > 0.80:
        if method == "fsm_no_recovery":
            return _act(0.20, _wz(heading_error, lateral, 1.0, method)), "COMMIT"
        return _act(-0.12, np.clip(0.4 * heading_error, -0.8, 0.8)), "RECOVER"
    if abs(heading_error) > 0.45 or abs(lateral) > 0.25:
        return _act(0.08, _wz(heading_error, lateral, 1.6, method)), "EXPLORE"
    return _act(0.20, _wz(heading_error, lateral, 1.0, method)), "COMMIT"


def _apf_action(env, obs: Dict, stress: StressCase, rng: np.random.Generator) -> Dict:
    depth = _depth_from_obs(obs, stress, rng)
    gps = np.array([_dist_to_goal(env), _heading_error(env)], dtype=np.float32)
    lin, ang = apf_gap_act(depth, gps)
    return {
        "action": "velocity_control",
        "action_args": {"linear_velocity": float(lin), "angular_velocity": float(ang)},
    }


def make_env(data_path: str, split: str):
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "habitat/task=narrow_passage",
            f"habitat.dataset.data_path={data_path}",
            f"habitat.dataset.split={split}",
            "habitat.dataset.type=PointNav-v1",
            "habitat.simulator.habitat_sim_v0.allow_sliding=False",
        ],
    )
    return habitat.Env(config=config)


def stress_cases(args) -> List[StressCase]:
    if args.preset == "paper":
        return [
            StressCase("nominal"),
            StressCase("yaw30", yaw_deg=30.0),
            StressCase("yaw60", yaw_deg=60.0),
            StressCase("lat010", lateral_m=0.10),
            StressCase("lat020", lateral_m=0.20),
            StressCase("drop010", depth_dropout=0.10),
            StressCase("drop030", depth_dropout=0.30),
            StressCase("noise002", feature_noise=0.02),
            StressCase("noise005", feature_noise=0.05),
            StressCase("extreme", extreme_narrow_only=True),
            StressCase("extreme_yaw60", yaw_deg=60.0, extreme_narrow_only=True),
        ]
    if args.preset == "single":
        return [
            StressCase(
                "single",
                yaw_deg=args.yaw_deg,
                lateral_m=args.lateral_m,
                depth_dropout=args.depth_dropout,
                feature_noise=args.feature_noise,
                extreme_narrow_only=args.extreme_narrow_only,
            )
        ]

    cases = []
    for yaw in args.yaw_degs:
        for lat in args.lateral_offsets:
            for drop in args.depth_dropouts:
                for noise in args.feature_noises:
                    name = f"yaw{int(yaw)}_lat{lat:.2f}_drop{drop:.2f}_noise{noise:.2f}"
                    cases.append(StressCase(name, yaw, lat, drop, noise, args.extreme_narrow_only))
    return cases


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _stable_int(*parts: str) -> int:
    key = "::".join(parts).encode("utf-8")
    return int(hashlib.sha1(key).hexdigest()[:8], 16)


def run_episode(env, obs: Dict, method: str, stress: StressCase, args, rng: np.random.Generator) -> Dict:
    features = _features_from_obs(env, obs, stress, rng)
    steps = 0
    any_collision = False
    near_collision = False
    min_clearance = float("inf")
    recover_triggers = 0
    done = False

    while not done and steps < args.max_steps:
        bm = float(features[9])
        min_clearance = min(min_clearance, bm)
        near_collision = near_collision or bm < args.near_collision_threshold
        any_collision = any_collision or float(features[16]) > 0.5

        if method == "apf_gap":
            action = _apf_action(env, obs, stress, rng)
            mode = "APF"
        else:
            action, mode = _fsm_action(env, features, method)
            if mode == "RECOVER":
                recover_triggers += 1

        obs = env.step(action)
        steps += 1
        features = _features_from_obs(env, obs, stress, rng)

        metrics = env.get_metrics()
        if (
            env.episode_over
            or float(metrics.get("narrow_passage_success", 0.0)) > 0.5
            or float(metrics.get("narrow_passage_collision", 0.0)) > 0.5
            or float(metrics.get("narrow_passage_stuck", 0.0)) > 0.5
        ):
            done = True

    metrics = env.get_metrics()
    if not np.isfinite(min_clearance):
        min_clearance = float(features[9])
    success = float(metrics.get("narrow_passage_success", 0.0))
    collision = max(float(any_collision), float(metrics.get("narrow_passage_collision", 0.0)))
    stuck = float(metrics.get("narrow_passage_stuck", 0.0))
    timeout = float(steps >= args.max_steps and success < 0.5 and collision < 0.5 and stuck < 0.5)
    clearance_safe = float(min_clearance >= args.strict_clearance_threshold)
    strict_success = float(success > 0.5 and collision < 0.5 and stuck < 0.5 and clearance_safe > 0.5)
    return {
        "steps": steps,
        "success": success,
        "strict_success": strict_success,
        "collision": collision,
        "stuck": stuck,
        "near_collision": float(near_collision),
        "min_clearance": float(min_clearance),
        "timeout": timeout,
        "recover_triggers": recover_triggers,
    }


def evaluate_method(method: str, stress: StressCase, args) -> List[Dict]:
    data_path = args.data_path.format(split=args.split)
    rows = []
    rng = np.random.default_rng(args.seed + _stable_int(method, stress.name) % 100000)
    with make_env(data_path, args.split) as env:
        total = env.number_of_episodes
        target = total if args.num_episodes <= 0 else min(args.num_episodes, total)
        seen = 0
        evaluated = 0
        while seen < total and evaluated < target:
            obs = env.reset()
            episode = env.current_episode
            seen += 1
            info = getattr(episode, "info", {}) or {}
            body_margin = float(info.get("body_margin", float("nan")))
            if stress.extreme_narrow_only and not (body_margin < args.extreme_margin):
                continue

            _apply_heading_perturb(env, stress.yaw_deg)
            _apply_lateral_perturb(env, stress.lateral_m)

            stat = run_episode(env, obs, method, stress, args, rng)
            scene_parts = str(episode.scene_id).split("/")
            scene_id = scene_parts[-2] if len(scene_parts) > 1 else scene_parts[0]
            stat.update(
                {
                    "method": method,
                    "stress": stress.name,
                    "yaw_deg": stress.yaw_deg,
                    "lateral_m": stress.lateral_m,
                    "depth_dropout": stress.depth_dropout,
                    "feature_noise": stress.feature_noise,
                    "extreme_narrow_only": float(stress.extreme_narrow_only),
                    "episode_id": str(episode.episode_id),
                    "scene_id": scene_id,
                    "difficulty": str(info.get("difficulty", "?")),
                    "body_margin": body_margin,
                }
            )
            rows.append(stat)
            evaluated += 1
            if args.verbose and (evaluated <= 5 or evaluated % 25 == 0):
                print(
                    f"  {method:24s} {stress.name:14s} ep={evaluated:3d} "
                    f"success={stat['success']:.0f} strict={stat['strict_success']:.0f} "
                    f"steps={stat['steps']:3d}",
                    flush=True,
                )
    return rows


def summarize(rows: List[Dict]) -> List[Dict]:
    groups: Dict[tuple, List[Dict]] = {}
    for row in rows:
        groups.setdefault((row["stress"], row["method"]), []).append(row)
    summary = []
    for (stress, method), vals in sorted(groups.items()):
        n = len(vals)
        summary.append(
            {
                "stress": stress,
                "method": method,
                "episodes": n,
                "success_rate": np.mean([v["success"] for v in vals]),
                "strict_success_rate": np.mean([v["strict_success"] for v in vals]),
                "collision_rate": np.mean([v["collision"] for v in vals]),
                "near_collision_rate": np.mean([v["near_collision"] for v in vals]),
                "avg_min_clearance": np.mean([v["min_clearance"] for v in vals]),
                "avg_steps": np.mean([v["steps"] for v in vals]),
                "timeout_rate": np.mean([v["timeout"] for v in vals]),
            }
        )
    return summary


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def write_markdown(summary: List[Dict], path: Path) -> None:
    lines = [
        "# Table: Habitat Stress Validation",
        "",
        "Nominal HM3D anchors are mostly well aligned; this table applies controlled stressors to test module robustness.",
        "",
        "| Stress | Method | Episodes | SR | Strict SR | Collision | Near collision | Avg min clearance | Avg steps | Timeout |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {stress} | {method} | {episodes} | {sr} | {strict} | {collision} | {near} | {clearance:.3f} | {steps:.1f} | {timeout} |".format(
                stress=row["stress"],
                method=row["method"],
                episodes=row["episodes"],
                sr=_fmt_pct(row["success_rate"]),
                strict=_fmt_pct(row["strict_success_rate"]),
                collision=_fmt_pct(row["collision_rate"]),
                near=_fmt_pct(row["near_collision_rate"]),
                clearance=row["avg_min_clearance"],
                steps=row["avg_steps"],
                timeout=_fmt_pct(row["timeout_rate"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def write_latex(summary: List[Dict], path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Stress & Method & N & SR & Strict SR & Collision & Near collision & Min clearance & Timeout \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {:.3f} & {} \\\\".format(
                _tex_escape(row["stress"]),
                _tex_escape(row["method"]),
                row["episodes"],
                _fmt_pct(row["success_rate"]).replace("%", r"\%"),
                _fmt_pct(row["strict_success_rate"]).replace("%", r"\%"),
                _fmt_pct(row["collision_rate"]).replace("%", r"\%"),
                _fmt_pct(row["near_collision_rate"]).replace("%", r"\%"),
                row["avg_min_clearance"],
                _fmt_pct(row["timeout_rate"]).replace("%", r"\%"),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines))
    print(f"[write] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=[
        "fsm_full",
        "fsm_no_heading_alignment",
        "fsm_no_recovery",
        "fsm_no_lateral_alignment",
        "apf_gap",
    ])
    parser.add_argument("--preset", choices=["paper", "single", "full-grid"], default="paper")
    parser.add_argument("--yaw-degs", type=parse_float_list, default=parse_float_list("0,30,60"))
    parser.add_argument("--lateral-offsets", type=parse_float_list, default=parse_float_list("0,0.10,0.20"))
    parser.add_argument("--depth-dropouts", type=parse_float_list, default=parse_float_list("0,0.10,0.30"))
    parser.add_argument("--feature-noises", type=parse_float_list, default=parse_float_list("0,0.02,0.05"))
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--lateral-m", type=float, default=0.0)
    parser.add_argument("--depth-dropout", type=float, default=0.0)
    parser.add_argument("--feature-noise", type=float, default=0.0)
    parser.add_argument("--extreme-narrow-only", action="store_true")
    parser.add_argument("--extreme-margin", type=float, default=0.05)
    parser.add_argument("--data-path", default="data/datasets/narrow_passage/{split}/{split}.json.gz")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-episodes", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--near-collision-threshold", type=float, default=0.05)
    parser.add_argument("--strict-clearance-threshold", type=float, default=-0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--summary-tex", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    valid_methods = {
        "fsm_full",
        "fsm_no_heading_alignment",
        "fsm_no_recovery",
        "fsm_no_lateral_alignment",
        "apf_gap",
    }
    bad = sorted(set(args.methods) - valid_methods)
    if bad:
        raise ValueError(f"Unknown methods: {bad}. Valid: {sorted(valid_methods)}")

    all_rows: List[Dict] = []
    cases = stress_cases(args)
    print(f"[stress] preset={args.preset} cases={len(cases)} methods={args.methods}")
    for stress in cases:
        print(f"\n[stress] {stress}")
        for method in args.methods:
            rows = evaluate_method(method, stress, args)
            all_rows.extend(rows)
            if rows:
                sr = np.mean([r["success"] for r in rows])
                strict = np.mean([r["strict_success"] for r in rows])
                print(f"  [{method}] n={len(rows)} SR={sr:.3f} strict={strict:.3f}")

    if not all_rows:
        raise RuntimeError("No episodes were evaluated. Check dataset path or extreme-narrow filter.")
    write_csv(all_rows, args.output_csv)
    summary = summarize(all_rows)
    write_markdown(summary, args.summary_md)
    write_latex(summary, args.summary_tex)


if __name__ == "__main__":
    main()
