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
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import habitat

from evaluation.logging_schema import (
    fieldnames_for_rows,
    finalize_episode_row,
    mode_bucket,
)
from eval_habitat_apf_gap import apf_gap_act
from narrow_passage.models.belief_state import BeliefState


RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"
DEFAULT_CSV = RESULTS / "habitat_stress_validation.csv"
DEFAULT_MD = RESULTS / "paper_table_habitat_stress.md"
DEFAULT_TEX = RESULTS / "paper_table_habitat_stress.tex"
DEFAULT_RAW_ALL = RESULTS / "raw" / "habitat_stress_all.csv"
TABLE_DIR = RESULTS / "tables"
STRESS_NOMINAL_MD = TABLE_DIR / "paper_table_habitat_stress_nominal.md"
STRESS_NOMINAL_TEX = TABLE_DIR / "paper_table_habitat_stress_nominal.tex"
CLEARANCE_DIAG_MD = TABLE_DIR / "paper_table_habitat_clearance_diagnostic.md"
CLEARANCE_DIAG_TEX = TABLE_DIR / "paper_table_habitat_clearance_diagnostic.tex"
KEY_SLICES_MD = TABLE_DIR / "paper_table_habitat_stress_key_slices.md"
KEY_SLICES_TEX = TABLE_DIR / "paper_table_habitat_stress_key_slices.tex"

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


def _belief_metrics(features: np.ndarray) -> Dict[str, float]:
    belief = BeliefState.from_obs(features, memory_risk=0.0)
    risk = float(np.clip(1.0 - belief.p_feas + belief.memory_risk, 0.0, 1.0))
    return {
        "d_hat": belief.d_hat,
        "w_req_cons": belief.w_req_cons,
        "delta_mean": belief.delta_mean,
        "delta_var": belief.delta_var,
        "p_feas": belief.p_feas,
        "risk": risk,
    }


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
    mode_counts: Counter[str] = Counter()
    belief_samples: List[Dict[str, float]] = []
    last_mode = "apf" if method == "apf_gap" else "unavailable"

    while not done and steps < args.max_steps:
        if method != "apf_gap":
            belief_samples.append(_belief_metrics(features))
        bm = float(features[9])
        min_clearance = min(min_clearance, bm)
        near_collision = near_collision or bm < args.near_collision_threshold
        any_collision = any_collision or float(features[16]) > 0.5

        if method == "apf_gap":
            action = _apf_action(env, obs, stress, rng)
            mode = "APF"
        else:
            action, mode = _fsm_action(env, features, method)
            last_mode = str(mode)
            mode_counts[mode_bucket(last_mode)] += 1
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
    if method != "apf_gap":
        belief_samples.append(_belief_metrics(features))
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
        "_mode_counts": dict(mode_counts),
        "_belief_samples": belief_samples,
        "_last_mode": last_mode,
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
            mode_counts = stat.pop("_mode_counts", {})
            belief_samples = stat.pop("_belief_samples", [])
            last_mode = stat.pop("_last_mode", "apf" if method == "apf_gap" else "unavailable")
            scene_parts = str(episode.scene_id).split("/")
            scene_id = scene_parts[-2] if len(scene_parts) > 1 else scene_parts[0]
            stat.update(
                {
                    "method": method,
                    "split": args.split,
                    "seed": args.seed,
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
            stat = finalize_episode_row(
                stat,
                belief_samples=belief_samples,
                belief_available=method != "apf_gap",
                mode_counts=mode_counts if method != "apf_gap" else None,
                mode_interface_available=method != "apf_gap",
                final_mode=last_mode if method != "apf_gap" else "apf",
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
                "success_but_unsafe_rate": np.mean([
                    float(float(v["success"]) > 0.5 and float(v["strict_success"]) < 0.5)
                    for v in vals
                ]),
            }
        )
    return summary


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_for_rows(rows))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def copy_raw_stress(rows: List[Dict], path: Path = DEFAULT_RAW_ALL) -> None:
    """Write a provenance-preserving copy for table-generation scripts."""

    write_csv(rows, path)


def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _fmt_num(x: float, digits: int = 1) -> str:
    return f"{float(x):.{digits}f}"


def write_markdown(summary: List[Dict], path: Path) -> None:
    lines = [
        "# Table: Habitat Stress Validation",
        "",
        "Module sensitivity under controlled Habitat perturbations. Clearance-aware metrics are reported as diagnostic indicators and are not interpreted as calibrated physical safety measurements.",
        "",
        "The Habitat clearance-related metrics are derived from depth observations and an approximate robot body-margin model. They are used as clearance-aware diagnostic indicators rather than calibrated physical safety measurements.",
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


def _table_rows(summary: List[Dict], stresses: set[str] | None = None,
                methods: set[str] | None = None) -> List[Dict]:
    rows = []
    for row in summary:
        if stresses is not None and row["stress"] not in stresses:
            continue
        if methods is not None and row["method"] not in methods:
            continue
        rows.append(row)
    return rows


def write_stress_nominal_table(summary: List[Dict], md_path: Path, tex_path: Path) -> None:
    """Write stress table with only nominal task metrics, no clearance diagnostics."""

    rows = _table_rows(summary)
    md_lines = [
        "# Table: Habitat Stress Validation - Nominal Metrics",
        "",
        "Nominal task metrics under controlled Habitat perturbations. Clearance-aware diagnostics are split into `paper_table_habitat_clearance_diagnostic.md`.",
        "",
        "| Stress | Method | Episodes | Success rate | Collision | Timeout | Avg steps |",
        "|:---|:---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {stress} | {method} | {episodes} | {sr} | {collision} | {timeout} | {steps} |".format(
                stress=row["stress"],
                method=row["method"],
                episodes=row["episodes"],
                sr=_fmt_pct(row["success_rate"]),
                collision=_fmt_pct(row["collision_rate"]),
                timeout=_fmt_pct(row["timeout_rate"]),
                steps=_fmt_num(row["avg_steps"], 1),
            )
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[write] {md_path}")

    tex_lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Stress & Method & N & Success & Collision & Timeout & Avg steps \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            "{} & {} & {} & {} & {} & {} & {:.1f} \\\\".format(
                _tex_escape(row["stress"]),
                _tex_escape(row["method"]),
                row["episodes"],
                _fmt_pct(row["success_rate"]).replace("%", r"\%"),
                _fmt_pct(row["collision_rate"]).replace("%", r"\%"),
                _fmt_pct(row["timeout_rate"]).replace("%", r"\%"),
                row["avg_steps"],
            )
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(tex_lines))
    print(f"[write] {tex_path}")


def write_clearance_diagnostic_table(summary: List[Dict], md_path: Path, tex_path: Path) -> None:
    rows = _table_rows(summary)
    note = (
        "The strict clearance metric is a depth-derived body-margin proxy. "
        "Negative values and high near-collision rates may reflect scanned-scene, "
        "navmesh, or depth-proxy artifacts. We report it as a diagnostic metric, "
        "not as a direct physical contact measurement."
    )
    md_lines = [
        "# Table: Habitat Clearance Diagnostic Metrics",
        "",
        note,
        "",
        "| Stress | Method | Strict success | Near collision | Avg min clearance | Success-but-unsafe |",
        "|:---|:---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {stress} | {method} | {strict} | {near} | {clearance:.3f} | {sbu} |".format(
                stress=row["stress"],
                method=row["method"],
                strict=_fmt_pct(row["strict_success_rate"]),
                near=_fmt_pct(row["near_collision_rate"]),
                clearance=row["avg_min_clearance"],
                sbu=_fmt_pct(row.get("success_but_unsafe_rate", 0.0)),
            )
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[write] {md_path}")

    tex_lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Stress & Method & Strict success & Near collision & Min clearance & Success-but-unsafe \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            "{} & {} & {} & {} & {:.3f} & {} \\\\".format(
                _tex_escape(row["stress"]),
                _tex_escape(row["method"]),
                _fmt_pct(row["strict_success_rate"]).replace("%", r"\%"),
                _fmt_pct(row["near_collision_rate"]).replace("%", r"\%"),
                row["avg_min_clearance"],
                _fmt_pct(row.get("success_but_unsafe_rate", 0.0)).replace("%", r"\%"),
            )
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(tex_lines))
    print(f"[write] {tex_path}")


def _key_slice_filter(row: Dict) -> bool:
    stress = row["stress"]
    method = row["method"]
    if stress == "nominal":
        return method in {"apf_gap", "fsm_full"}
    if stress in {"yaw60", "extreme_yaw60", "lat020"}:
        return method in {"apf_gap", "fsm_full", "fsm_no_heading_alignment"}
    return False


KEY_SLICE_ORDER = {
    "nominal": 0,
    "yaw60": 1,
    "extreme_yaw60": 2,
    "lat020": 3,
}

KEY_SLICE_METHOD_ORDER = {
    "apf_gap": 0,
    "fsm_full": 1,
    "fsm_no_heading_alignment": 2,
}


def write_key_slices_table(summary: List[Dict], md_path: Path, tex_path: Path) -> None:
    rows = sorted(
        [row for row in summary if _key_slice_filter(row)],
        key=lambda row: (
            KEY_SLICE_ORDER.get(row["stress"], 99),
            KEY_SLICE_METHOD_ORDER.get(row["method"], 99),
        ),
    )
    note = (
        "Key stress slices for manuscript discussion. Clearance columns are diagnostic "
        "body-margin proxies, not calibrated physical contact measurements."
    )
    md_lines = [
        "# Table: Habitat Stress Key Slices",
        "",
        note,
        "",
        "| Stress | Method | Episodes | Success rate | Collision | Timeout | Avg steps | Strict success (diagnostic) | Near collision (diagnostic) | Avg min clearance (diagnostic) |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {stress} | {method} | {episodes} | {sr} | {collision} | {timeout} | {steps} | {strict} | {near} | {clearance:.3f} |".format(
                stress=row["stress"],
                method=row["method"],
                episodes=row["episodes"],
                sr=_fmt_pct(row["success_rate"]),
                collision=_fmt_pct(row["collision_rate"]),
                timeout=_fmt_pct(row["timeout_rate"]),
                steps=_fmt_num(row["avg_steps"], 1),
                strict=_fmt_pct(row["strict_success_rate"]),
                near=_fmt_pct(row["near_collision_rate"]),
                clearance=row["avg_min_clearance"],
            )
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[write] {md_path}")

    tex_lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Stress & Method & N & Success & Collision & Timeout & Steps & Strict diag. & Near diag. & Min clearance diag. \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            "{} & {} & {} & {} & {} & {} & {:.1f} & {} & {} & {:.3f} \\\\".format(
                _tex_escape(row["stress"]),
                _tex_escape(row["method"]),
                row["episodes"],
                _fmt_pct(row["success_rate"]).replace("%", r"\%"),
                _fmt_pct(row["collision_rate"]).replace("%", r"\%"),
                _fmt_pct(row["timeout_rate"]).replace("%", r"\%"),
                row["avg_steps"],
                _fmt_pct(row["strict_success_rate"]).replace("%", r"\%"),
                _fmt_pct(row["near_collision_rate"]).replace("%", r"\%"),
                row["avg_min_clearance"],
            )
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(tex_lines))
    print(f"[write] {tex_path}")


def write_split_stress_tables(summary: List[Dict]) -> None:
    write_stress_nominal_table(summary, STRESS_NOMINAL_MD, STRESS_NOMINAL_TEX)
    write_clearance_diagnostic_table(summary, CLEARANCE_DIAG_MD, CLEARANCE_DIAG_TEX)
    write_key_slices_table(summary, KEY_SLICES_MD, KEY_SLICES_TEX)


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
    copy_raw_stress(all_rows)
    summary = summarize(all_rows)
    write_markdown(summary, args.summary_md)
    write_latex(summary, args.summary_tex)
    write_split_stress_tables(summary)


if __name__ == "__main__":
    main()
