#!/usr/bin/env python3
"""Geometry-FSM evaluator for NarrowPassageNav-v0 on real HM3D scenes.

Ports the geometry + failure-memory FSM from the synthetic procedural env to the
Habitat task.  No training required — pure geometry-based decision-making.

Four FSM modes
  COMMIT  — safe geometry, move forward with soft alignment
  EXPLORE — medium risk, slow forward with stronger alignment
  RECOVER — collision / very high risk, back up and rotate toward more space
  REJECT  — memory gate: this geometry type failed too many times, don't enter

The observation vector (narrow_passage_features, 19-dim) has the same index layout
as ProceduralNarrowPassageEnv, so GeometryRiskEstimator and PassageFailureMemory
work directly — with one adaptation: clearance_left/right from Habitat's depth
sensor are wall distances (>0.2 m even in tight passages), while the risk
estimator was calibrated against body_margin (wall-distance minus robot radius).
_to_risk_obs() remaps obs[6:8] to body_margin before calling the estimator.

Usage
-----
# Geometry-FSM only (no memory), all 37 val episodes:
  python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py

# With failure-memory gating (paper's full method):
  python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py --use-memory 1

# Write per-episode CSV + print summary row for results_rl_summary.csv:
  python examples/narrow_passage_rl/eval_habitat_geometry_fsm.py \
      --use-memory 0 --output-csv results/narrow_passage_rl/habitat_fsm_episodes.csv
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from failure_memory import FailureMemoryConfig, PassageFailureMemory
from risk_estimator import RiskConfig  # noqa: F401  (kept for future use)

import habitat  # noqa: E402  (import after sys.path setup)


# ---------------------------------------------------------------------------
# Action space: narrow_passage.yaml specifies lin_vel_range=[-0.15, 0.35] m/s
# and ang_vel_range=[-45, 45] deg/s.  VelocityAction.step() converts the
# normalized [-1, 1] inputs via  physical = min + (x+1)/2 * (max-min).
# ---------------------------------------------------------------------------
_LIN_MIN, _LIN_MAX = -0.15, 0.35   # m/s
_ANG_MIN, _ANG_MAX = -45.0, 45.0   # deg/s


def _norm_lin(vx_ms: float) -> float:
    return max(-1.0, min(1.0, (vx_ms - _LIN_MIN) / (_LIN_MAX - _LIN_MIN) * 2.0 - 1.0))


def _norm_ang(wz_rads: float) -> float:
    wz_deg = wz_rads * 180.0 / math.pi
    return max(-1.0, min(1.0, (wz_deg - _ANG_MIN) / (_ANG_MAX - _ANG_MIN) * 2.0 - 1.0))


def _make_action(lin_norm: float, ang_norm: float) -> dict:
    return {
        "action": "velocity_control",
        "action_args": {
            "linear_velocity": float(lin_norm),
            "angular_velocity": float(ang_norm),
        },
    }


# ---------------------------------------------------------------------------
# Per-mode controllers (purely geometric, no trained models)
#
# Sign conventions verified for Habitat (diag_sign.py empirical tests):
#   heading_error from _compute_heading_error():
#     > 0 → goal is to the LEFT  of forward direction → positive wz (CCW) reduces it
#     < 0 → goal is to the RIGHT of forward direction → negative wz (CW)  reduces it
#     so:  wz_correction = +K * heading_error
#
#   lateral_offset = features[11] = 2D cross product (rel × line):
#     > 0 → agent is to the LEFT  of start→goal line → need right turn (wz < 0)
#     < 0 → agent is to the RIGHT of start→goal line → need left turn  (wz > 0)
#     so:  wz_correction = −K * lateral_offset
# ---------------------------------------------------------------------------

def _dist_to_goal(env) -> float:
    """2-D (x-z plane) Euclidean distance to goal, computed directly from sim state."""
    agent_pos = np.array(env.sim.get_agent_state().position, dtype=np.float32)
    goal_pos = np.array(env.current_episode.goals[0].position, dtype=np.float32)
    return float(np.linalg.norm((goal_pos - agent_pos)[[0, 2]]))


def _compute_heading_error(env) -> float:
    """Compute heading error using the CORRECT formula: atan2(-delta_x, -delta_z).

    The task's _heading_error() uses atan2(delta_x, -delta_z) which places the
    robot's facing direction at (-delta_x, 0, delta_z) when error=0 — wrong in X.
    The correct formula ensures forward=(-sin(yaw), 0, -cos(yaw)) ∝ (delta_x, 0, delta_z)
    when error=0, i.e., the robot faces toward the goal.

    Sign convention (same as task formula):
      > 0 → goal is to the LEFT  → positive wz (CCW) reduces error
      < 0 → goal is to the RIGHT → negative wz (CW)  reduces error
    """
    agent_state = env.sim.get_agent_state()
    agent_pos = np.array(agent_state.position, dtype=np.float32)
    goal_pos = np.array(env.current_episode.goals[0].position, dtype=np.float32)
    delta = goal_pos - agent_pos
    goal_yaw = math.atan2(-float(delta[0]), -float(delta[2]))  # CORRECT sign on delta_x
    rot = agent_state.rotation
    yaw = math.atan2(
        2.0 * (rot.real * rot.y + rot.x * rot.z),
        1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z),
    )
    err = goal_yaw - yaw
    return float((err + math.pi) % (2.0 * math.pi) - math.pi)


def _alignment_wz(heading_error: float, lateral_offset: float, gain: float = 1.0) -> float:
    """Alignment angular velocity (rad/s) from heading error + lateral offset.

    heading_error: from _compute_heading_error() — correct angle to goal
    lateral_offset: from features[11] — perpendicular distance from start→goal line
    """
    # heading_error > 0 → goal is to the LEFT → positive ang_vel (CCW/left) reduces it
    # lateral_offset > 0 → agent is LEFT of centerline → negative ang_vel (CW/right) corrects it
    # (verified empirically: positive Habitat angular_velocity = left/CCW turn)
    wz = +0.9 * heading_error - 1.2 * lateral_offset
    return float(np.clip(wz * gain, -1.2, 1.2))


def _align_action(heading_error: float) -> dict:
    """Rotate in place to reduce large heading error (> ~40°).
    Uses ONLY heading_error (not lateral_offset) so wz is always non-zero in ALIGN
    mode — combining heading + lateral can cancel to wz≈0 → is_stop_called = True.
    Physical velocity = 0: _norm_lin(0.0) = -0.4 → ((-0.4+1)/2)*0.5 - 0.15 = 0 m/s.
    """
    wz = float(np.clip(+0.9 * heading_error * 2.0, -1.2, 1.2))
    return _make_action(_norm_lin(0.0), _norm_ang(wz))


def _commit_action(heading_error: float, lateral_offset: float) -> dict:
    return _make_action(_norm_lin(0.20), _norm_ang(_alignment_wz(heading_error, lateral_offset, 1.0)))


def _explore_action(heading_error: float, lateral_offset: float) -> dict:
    return _make_action(_norm_lin(0.08), _norm_ang(_alignment_wz(heading_error, lateral_offset, 1.6)))


def _recover_action(heading_error: float) -> dict:
    """Reverse and reorient toward goal using heading_error direction.
    Depth-based clearance is not used for recovery rotation — unreliable in-passage.
    """
    wz = float(np.clip(+0.4 * heading_error, -0.8, 0.8))
    return _make_action(_norm_lin(-0.12), _norm_ang(wz))


def _is_recovered(heading_error: float, features: np.ndarray) -> bool:
    return (
        abs(heading_error) < 0.35          # heading roughly toward goal
        and abs(float(features[11])) < 0.20  # near centerline
        and float(features[15]) < 0.30       # not stuck
    )


# ---------------------------------------------------------------------------
# Risk observation remapping (used only for PassageFailureMemory key)
# The memory key uses passage_width (obs[8]) + heading/lateral/side — these are
# reliable.  body_margin (obs[9]) is substituted for clearance_left/right so the
# memory key discriminates passages by actual body clearance, not raw depth.
# ---------------------------------------------------------------------------

def _to_risk_obs(features: np.ndarray) -> np.ndarray:
    a = features.copy()
    bm = max(0.0, float(features[9]))  # clamp negative body_margin to 0 for memory key
    a[6] = bm
    a[7] = bm
    return a


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geometry-FSM (± failure memory) eval on Habitat NarrowPassageNav-v0"
    )
    parser.add_argument("--data-path",
                        default="data/datasets/narrow_passage/{split}/{split}.json.gz")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-episodes", type=int, default=37)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-recover-steps", type=int, default=30)
    parser.add_argument("--use-memory", type=int, default=0,
                        help="1 = enable cross-episode failure-memory gating")
    # Mode thresholds: task-state based (depth-based clearance not used for mode decisions)
    parser.add_argument("--high-risk-threshold", type=float, default=0.80,
                        help="stuck_score above this → RECOVER")
    parser.add_argument("--medium-risk-threshold", type=float, default=0.45,
                        help="abs(heading_error) rad above this → EXPLORE (or lateral > 0.25m)")
    parser.add_argument("--reject-after-failures", type=int, default=3)
    # Memory config
    parser.add_argument("--memory-trigger-count", type=int, default=1)
    parser.add_argument("--memory-reject-count", type=int, default=3)
    # Output
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Write per-episode CSV to this path")
    parser.add_argument("--allow-sliding", type=int, default=0,
                        help="0 = allow_sliding=False (matches PPO training); 1 = True")
    args = parser.parse_args()

    use_memory = bool(args.use_memory)
    label = "habitat_ours_memory" if use_memory else "habitat_geometry_fsm"

    data_path = args.data_path.format(split=args.split)
    allow_sliding_str = "True" if args.allow_sliding else "False"
    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "habitat/task=narrow_passage",
            f"habitat.dataset.data_path={data_path}",
            f"habitat.dataset.split={args.split}",
            "habitat.dataset.type=PointNav-v1",
            f"habitat.simulator.habitat_sim_v0.allow_sliding={allow_sliding_str}",
        ],
    )

    mem_cfg = FailureMemoryConfig(
        trigger_count=args.memory_trigger_count,
        reject_count=args.memory_reject_count,
    )

    # Cross-episode memory persists across all episodes to generalise failure avoidance.
    cross_memory = PassageFailureMemory(mem_cfg) if use_memory else None

    all_stats = []

    with habitat.Env(config=config) as env:
        total = env.number_of_episodes or args.num_episodes
        num_episodes = total if args.num_episodes <= 0 else min(args.num_episodes, total)
        print(f"[{label}] evaluating {num_episodes} episodes "
              f"(split={args.split}, use_memory={use_memory})")

        for ep_idx in range(num_episodes):
            observations = env.reset()
            episode = env.current_episode
            features = np.array(
                observations.get("narrow_passage_features", np.zeros(19, dtype=np.float32)),
                dtype=np.float32,
            )

            memory = cross_memory  # None when use_memory=False

            steps = 0
            rejected = False
            recover_triggers = 0
            memory_writes = 0
            any_collision = False
            min_bm = float("inf")   # min body_margin seen this episode
            near_collision = False

            done = False
            while not done and steps < args.max_steps:
                # ---- track clearance and near-collision ----
                bm = float(features[9])
                if bm < min_bm:
                    min_bm = bm
                if bm < 0.05:
                    near_collision = True

                # ---- collision flag from this step ----
                step_collision = float(features[16]) > 0.5
                if step_collision:
                    any_collision = True

                # ---- compute correct heading error directly from sim state ----
                # features[10] uses atan2(delta_x, -delta_z) which gives the wrong
                # facing direction (negates delta_x).  Correct formula below ensures
                # that heading_error=0 means the robot faces toward the goal.
                heading_error = _compute_heading_error(env)
                dist_now = _dist_to_goal(env)
                lateral_offset = float(features[11])
                stuck_score = float(features[15])

                risk_obs = _to_risk_obs(features)
                if dist_now < 0.25:
                    # Within success radius: send near-zero velocity to trigger is_stop_called.
                    # VelocityAction sets is_stop_called = True when |lin| < 0.01 m/s AND
                    # |ang| < 1 deg/s → episode terminates and success is scored.
                    mode = "STOP"
                elif memory is not None and memory.should_reject(risk_obs):
                    mode = "REJECT"
                elif step_collision or stuck_score > args.high_risk_threshold:
                    mode = "RECOVER"
                elif abs(heading_error) > 0.7:
                    # Misalignment > ~40°: stop and rotate first.
                    # Moving forward while > 40° misaligned causes lateral drift that
                    # cancels the heading correction → robot spirals outward from goal.
                    mode = "ALIGN"
                elif abs(heading_error) > args.medium_risk_threshold or abs(lateral_offset) > 0.25:
                    mode = "EXPLORE"
                else:
                    mode = "COMMIT"

                # ---- execute mode ----
                if mode == "STOP":
                    # Send zero velocity: is_stop_called → True → episode ends with success.
                    observations = env.step(_make_action(_norm_lin(0.0), _norm_ang(0.0)))
                    steps += 1
                    features = np.array(
                        observations.get("narrow_passage_features", features),
                        dtype=np.float32,
                    )
                    if env.episode_over:
                        done = True
                    continue

                if mode == "REJECT":
                    rejected = True
                    done = True
                    continue

                if mode == "RECOVER":
                    recover_triggers += 1
                    if step_collision and memory is not None:
                        memory.add_failure(risk_obs)
                        memory_writes += 1

                    for _ in range(args.max_recover_steps):
                        he_r = _compute_heading_error(env)
                        observations = env.step(_recover_action(he_r))
                        steps += 1
                        features = np.array(
                            observations.get("narrow_passage_features", features),
                            dtype=np.float32,
                        )
                        bm_r = float(features[9])
                        if bm_r < min_bm:
                            min_bm = bm_r
                        if float(features[16]) > 0.5:
                            any_collision = True
                            if memory is not None:
                                memory.add_failure(_to_risk_obs(features))
                                memory_writes += 1

                        if env.episode_over or steps >= args.max_steps:
                            done = True
                            break
                        he_r = _compute_heading_error(env)
                        if _is_recovered(he_r, features):
                            break
                    if env.episode_over or steps >= args.max_steps:
                        done = True
                    continue

                # ALIGN, COMMIT or EXPLORE
                if mode == "ALIGN":
                    action = _align_action(heading_error)
                elif mode == "COMMIT":
                    action = _commit_action(heading_error, lateral_offset)
                else:
                    action = _explore_action(heading_error, lateral_offset)
                observations = env.step(action)
                steps += 1
                features = np.array(
                    observations.get("narrow_passage_features", features),
                    dtype=np.float32,
                )
                if env.episode_over:
                    done = True

            # ---- episode metrics ----
            metrics = env.get_metrics()
            success = float(metrics.get("narrow_passage_success", 0.0))
            collision = float(any_collision)
            stuck = float(metrics.get("narrow_passage_stuck", 0.0))

            # Write cross-episode memory on episode failure (not on reject or success)
            if use_memory and cross_memory is not None and not rejected and success < 0.5:
                if any_collision or stuck > 0.5:
                    cross_memory.add_failure(_to_risk_obs(features))
                    memory_writes += 1

            if not np.isfinite(min_bm):
                min_bm = float(features[9])

            stat = {
                "episode_id": str(episode.episode_id),
                "scene_id": str(episode.scene_id).split("/")[-2],
                "steps": steps,
                "success": success,
                "collision": collision,
                "stuck": stuck,
                "rejected": float(rejected),
                "near_collision": float(near_collision),
                "min_clearance": float(min_bm),
                "recover_triggers": recover_triggers,
                "memory_writes": memory_writes,
            }
            all_stats.append(stat)

            print(
                f"  ep {ep_idx:3d}  success={success:.0f}  collision={collision:.0f}"
                f"  stuck={stuck:.0f}  rejected={int(rejected)}"
                f"  min_bm={min_bm:.3f}  steps={steps:3d}"
                f"  recover={recover_triggers}  mem_writes={memory_writes}"
            )

    # ---- summary ----
    n = len(all_stats)
    success_rate = np.mean([s["success"] for s in all_stats])
    collision_rate = np.mean([s["collision"] for s in all_stats])
    stuck_rate = np.mean([s["stuck"] for s in all_stats])
    reject_rate = np.mean([s["rejected"] for s in all_stats])
    near_collision_rate = np.mean([s["near_collision"] for s in all_stats])
    avg_min_clearance = np.mean([s["min_clearance"] for s in all_stats])
    avg_recover_triggers = np.mean([s["recover_triggers"] for s in all_stats])
    avg_memory_writes = np.mean([s["memory_writes"] for s in all_stats])

    print()
    print(f"=== {label}  |  {n} episodes  |  split={args.split} ===")
    print(f"  success_rate:        {success_rate:.3f}")
    print(f"  collision_rate:      {collision_rate:.3f}")
    print(f"  stuck_rate:          {stuck_rate:.3f}")
    print(f"  reject_rate:         {reject_rate:.3f}")
    print(f"  near_collision_rate: {near_collision_rate:.3f}")
    print(f"  avg_min_clearance:   {avg_min_clearance:.3f}")
    print(f"  avg_recover_triggers:{avg_recover_triggers:.3f}")
    print(f"  avg_memory_writes:   {avg_memory_writes:.3f}")

    # Print a ready-to-paste row for results_rl_summary.csv
    print()
    print("# Copy into results/narrow_passage_rl/results_rl_summary.csv:")
    print(
        f"{label},{success_rate:.4f},{collision_rate:.4f},"
        f"{collision_rate:.4f},{near_collision_rate:.4f},{avg_min_clearance:.4f}"
        f",,,,{avg_memory_writes:.4f},{reject_rate:.4f}"
    )

    if args.output_csv is not None:
        import csv
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_stats[0].keys()))
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"[write] {args.output_csv}")


if __name__ == "__main__":
    main()
