#!/usr/bin/env python3
"""Evaluate methods on the harder procedural benchmark (v2).

Runs six ablation variants across seven corridor types and prints a table
suitable for paper_table_harder.{md,tex}.

Methods
-------
  rule_baseline       — forward + align-to-goal (APF-style, no memory)
  geometry_fsm        — geometry-only FSM (ALIGN/COMMIT/EXPLORE/RECOVER)
  fsm_no_recovery     — FSM without RECOVER mode
  fsm_no_alignment    — FSM with only heading correction, no lateral mode
  fsm_local_memory    — FSM + episode-local PassageFailureMemory (v1 memory)
  fsm_cross_memory    — FSM + CrossEpisodeMemory (proposed contribution)

Corridor types (each run separately, then averaged):
  straight, l_shaped, s_shaped, narrow_exit, narrow_entry, asymmetric, false_feasible

Usage
-----
    python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500
    python examples/narrow_passage_rl/eval_harder_benchmark.py \
        --methods geometry_fsm fsm_cross_memory \
        --corridor-types l_shaped false_feasible \
        --episodes 200
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import CorridorType, HarderNarrowPassageEnv
from failure_memory import FailureMemoryConfig, PassageFailureMemory
from cross_episode_memory import (
    CrossEpisodeMemory,
    EpisodeOutcome,
    MemoryConfig,
    FSMMode,
)
from narrow_passage.models.belief_state import BeliefState, BeliefStateConfig
from narrow_passage.models.dynamic_feasibility import (
    DynamicFeasibilityConfig,
    DynamicFeasibilityEstimator,
)
from narrow_passage.models.feasibility_ablation import (
    ABLATIONS as STRICT_FEASIBILITY_ABLATIONS,
    AblationMode,
    ModeSelection,
    SelectorThresholds,
    SharedDecisionState,
    interval_bounds,
    canonical_ablation,
    select_ablation_mode,
    uses_fixed_uncertainty,
    uses_memory,
    uses_yaw_prior,
)
from evaluation.logging_schema import fieldnames_for_rows, finalize_episode_row

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"


@dataclass(frozen=True)
class RepairedControllerConfig:
    """Explicit active-sensing and bounded-recovery limits."""

    # The generated S route can require 6.3 m including approach/exit.  With a
    # 200-step, 0.25-s budget and 30 steps reserved for two 90-degree turns,
    # any shared translational mode below 0.15 m/s is physically incapable of
    # completing the longest labelled-feasible route.  These method-invariant
    # defaults satisfy that protocol-level kinematic lower bound.
    approach_speed: float = 0.15
    commit_speed: float = 0.20
    explore_speed: float = 0.15
    corner_turn_speed: float = 0.08
    explore_scan_rate: float = 0.24
    explore_half_scan_steps: int = 3
    information_gain_epsilon: float = 1e-5
    max_probe_no_information: int = 10
    max_recover_steps: int = 10
    max_recover_cycles: int = 2
    recover_observe_steps: int = 2
    recover_speed: float = -0.08
    max_goal_regression: float = 0.30
    recover_cooldown_steps: int = 5
    swept_obb_horizon_steps: int = 4
    swept_obb_minimum_clearance: float = 0.005
    readiness_yaw_threshold: float = 0.18
    readiness_lateral_threshold: float = 0.08
    readiness_clearance_threshold: float = 0.03
    max_align_steps: int = 35
    max_enter_steps: int = 45
    max_traverse_steps: int = 140
    max_exit_steps: int = 45

    def __post_init__(self) -> None:
        if not 0.0 < self.approach_speed <= 0.35:
            raise ValueError("approach_speed outside environment bounds")
        if not 0.0 < self.commit_speed <= 0.35:
            raise ValueError("commit_speed outside environment bounds")
        if not 0.0 < self.explore_speed <= 0.35:
            raise ValueError("explore_speed outside environment bounds")
        if not 0.0 < self.corner_turn_speed <= 0.35:
            raise ValueError("corner_turn_speed outside environment bounds")
        if not 0.0 < self.explore_scan_rate <= 0.8:
            raise ValueError("explore_scan_rate outside environment bounds")
        if self.max_recover_steps < 1 or self.max_recover_cycles < 1:
            raise ValueError("recover limits must be positive")


class PassageControlState(str, Enum):
    APPROACH = "APPROACH"
    ALIGN = "ALIGN"
    ENTER = "ENTER"
    TRAVERSE = "TRAVERSE"
    EXIT = "EXIT"
    RECOVER = "RECOVER"

CORE_VARIANTS = [
    "full",
    "no_alignment",
    "no_recovery",
    "deterministic_margin",
    "no_yaw_prior",
    "feasibility_reject",
]

LEGACY_METHOD_TO_VARIANT = {
    "geometry_fsm": "full",
    "fsm_no_recovery": "no_recovery",
    "fsm_no_alignment": "no_alignment",
    "fsm_local_memory": "full",
    "fsm_cross_memory": "full",
    "full": "full",
    "no_alignment": "no_alignment",
    "no_recovery": "no_recovery",
    "deterministic_margin": "deterministic_margin",
    "no_yaw_prior": "no_yaw_prior",
    "feasibility_reject": "feasibility_reject",
}

PROB_FEAS_THRESHOLD = 0.70
RISK_THRESHOLD = 0.35
TAU_MARGIN = 0.05
REJECT_P_FEAS_THRESHOLD = 0.08
REJECT_MARGIN_THRESHOLD = -0.08
REJECT_FRONT_DEPTH_THRESHOLD = 0.18
REJECT_STUCK_THRESHOLD = 0.65

FALSE_FEASIBLE_REQUIRED_COLUMNS = [
    "episode_id",
    "seed",
    "method",
    "corridor_type",
    "is_false_feasible",
    "passable_label",
    "success",
    "reject",
    "correct_reject",
    "false_reject",
    "collision",
    "near_collision",
    "timeout",
    "stuck",
    "wasted_attempt",
    "min_clearance",
    "body_margin_min",
    "final_mode",
    "mode_commit_count",
    "mode_explore_count",
    "mode_recover_count",
    "mode_reject_count",
]

MARGIN_PHASE_REQUIRED_COLUMNS = [
    "method",
    "scenario_id",
    "episode_id",
    "seed",
    "corridor_type",
    "passage_width",
    "entry_yaw",
    "lateral_offset",
    "passable_label",
    "success",
    "collision",
    "near_collision",
    "reject",
    "correct_reject",
    "false_reject",
    "timeout",
    "stuck",
    "d_hat",
    "w_req_prior",
    "w_req_cons",
    "delta_mean",
    "delta_var",
    "p_feas",
    "risk",
    "body_margin",
    "min_clearance",
    "margin_snapshot_step",
    "margin_snapshot_source",
    "belief_used_by_policy",
    "mode_commit_count",
    "mode_explore_count",
    "mode_recover_count",
    "mode_reject_count",
]

CORE_ABLATION_REQUIRED_COLUMNS = [
    "variant",
    "method",
    "episode_id",
    "seed",
    "corridor_type",
    "success",
    "collision",
    "near_collision",
    "reject",
    "correct_reject",
    "false_reject",
    "timeout",
    "stuck",
    "d_hat",
    "w_req_cons",
    "delta_mean",
    "delta_var",
    "p_feas",
    "mode_commit_count",
    "mode_explore_count",
    "mode_recover_count",
    "mode_reject_count",
]

METHOD_LABELS = {
    "rule_baseline": "Reactive rule baseline",
    "geometry_fsm": "DEGNAV-Rule / Geometry-FSM",
    "fsm_no_recovery": "DEGNAV-Rule w/o recovery",
    "fsm_no_alignment": "DEGNAV-Rule w/o alignment",
    "fsm_local_memory": "DEGNAV-Rule + local memory",
    "fsm_cross_memory": "DEGNAV-Rule + cross-episode memory",
    "full": "DEGNAV-Rule / Geometry-FSM",
    "no_alignment": "DEGNAV-Rule w/o alignment",
    "no_recovery": "DEGNAV-Rule w/o recovery",
    "deterministic_margin": "DEGNAV-Rule deterministic margin",
    "no_yaw_prior": "DEGNAV-Rule w/o yaw prior",
    "feasibility_reject": "DEGNAV-Rule conservative reject gate",
}


# ── Action helpers ────────────────────────────────────────────────────────────

def _act(vx, wz):
    return np.array([vx, wz], dtype=np.float32)


def _mode_bucket(mode: str) -> str:
    """Map implementation mode strings into paper-facing mode buckets."""

    m = str(mode or "").upper()
    if "REJECT" in m:
        return "reject"
    if "RECOVER" in m:
        return "recover"
    if (
        "EXPLORE" in m or "ALIGN" in m or "FOLLOW" in m
        or "CORRIDOR" in m or "APPROACH" in m
    ):
        return "explore"
    return "commit"


def _variant_for_method(method: str) -> str:
    return LEGACY_METHOD_TO_VARIANT.get(method, "full")


def _belief_cfg_for_variant(variant: str) -> BeliefStateConfig:
    return BeliefStateConfig(use_yaw_prior=(variant != "no_yaw_prior"))


def _belief_metrics(
    obs: np.ndarray,
    memory_risk: float = 0.0,
    variant: str = "full",
) -> dict:
    """Return common feasibility-belief diagnostics for CSV visualizations."""

    belief_cfg = _belief_cfg_for_variant(variant)
    belief = BeliefState.from_obs(
        obs,
        memory_risk=memory_risk,
        cfg=belief_cfg,
    )
    risk = float(np.clip(1.0 - belief.p_feas + belief.memory_risk, 0.0, 1.0))
    return {
        "mu_D": belief.d_hat,
        "var_D": float(belief_cfg.sigma_d) ** 2,
        "mu_W": belief.w_req_cons,
        "var_W": float(belief_cfg.sigma_w) ** 2,
        "mu_delta": belief.delta_mean,
        "var_delta": belief.delta_var,
        "d_hat": belief.d_hat,
        "w_req_prior": belief.w_req_prior,
        "w_req_cons": belief.w_req_cons,
        "delta_mean": belief.delta_mean,
        "delta_var": belief.delta_var,
        "p_feas": belief.p_feas,
        "risk": risk,
        "memory_risk": belief.memory_risk,
    }


def _finite_belief_metrics(obs: np.ndarray, variant: str = "full") -> Optional[dict]:
    """Return shared belief diagnostics when finite, otherwise ``None``."""

    try:
        row = _belief_metrics(obs, variant=variant)
    except ValueError:
        return None
    values = [row.get(k, float("nan")) for k in (
        "d_hat", "w_req_prior", "w_req_cons", "delta_mean", "delta_var", "p_feas"
    )]
    if not all(np.isfinite(float(v)) for v in values):
        return None
    return row


def _canonical_margin_snapshot(
    env: HarderNarrowPassageEnv,
    reset_obs: np.ndarray,
    variant: str = "full",
) -> tuple[dict, int, str, np.ndarray]:
    """Compute the pre-behavior margin snapshot shared by paired methods.

    The margin-phase x-axis should describe the passage before the controller
    choices diverge.  The v2 environment starts the robot behind the corridor,
    where the observation reports open space.  We therefore project the reset
    pose laterally onto the corridor entry, keep the reset yaw, and query the
    same 19-D geometry observation without executing an action.
    """

    fallback = _finite_belief_metrics(reset_obs, variant=variant)
    if fallback is None:
        fallback = {
            "d_hat": float("nan"),
            "w_req_prior": float("nan"),
            "w_req_cons": float("nan"),
            "delta_mean": float("nan"),
            "delta_var": float("nan"),
            "p_feas": float("nan"),
            "risk": float("nan"),
            "memory_risk": float("nan"),
        }

    if len(getattr(env, "_pts", [])) < 2:
        return fallback, 0, "reset_finite_fallback", reset_obs

    old_pose = env.pose.copy()
    old_prev = env.prev_action.copy()
    old_collision = bool(env.collision)
    old_step_count = int(env.step_count)
    old_stuck = int(env.stuck_steps)
    rng_state = getattr(env.rng.bit_generator, "state", None)

    try:
        first_tang = env._pts[1] - env._pts[0]
        first_tang = first_tang / (np.linalg.norm(first_tang) + 1e-9)
        left = np.array([-first_tang[1], first_tang[0]])
        lateral_n = float(np.dot(old_pose[:2] - env._pts[0], left))
        env.pose[:2] = env._pts[0] + left * lateral_n
        env.pose[2] = old_pose[2]
        env.prev_action[:] = 0.0
        env.collision = False
        obs = env._obs()
        row = _finite_belief_metrics(obs, variant=variant)
        if row is not None and float(obs[9]) <= 0.5:
            return row, 0, "entry_projection", obs.copy()
        if row is not None:
            return row, 0, "entry_projection_finite_fallback", obs.copy()
    finally:
        env.pose[:] = old_pose
        env.prev_action[:] = old_prev
        env.collision = old_collision
        env.step_count = old_step_count
        env.stuck_steps = old_stuck
        if rng_state is not None:
            env.rng.bit_generator.state = rng_state

    return fallback, 0, "reset_finite_fallback", reset_obs


def _feasibility_acceptance(obs: np.ndarray, variant: str,
                            memory_risk: float = 0.0) -> tuple[bool, dict, str]:
    """Return whether the passage should be accepted under the variant gate.

    ``full`` uses the probabilistic feasibility belief.  ``deterministic_margin``
    uses only the hard margin threshold and intentionally ignores ``p_feas`` and
    ``delta_var`` for mode decisions.  ``no_yaw_prior`` keeps the probabilistic
    machinery but builds the belief with a fixed frontal envelope.
    """

    belief_row = _belief_metrics(obs, memory_risk=memory_risk, variant=variant)
    if variant == "deterministic_margin":
        return bool(belief_row["delta_mean"] > TAU_MARGIN), belief_row, "deterministic_margin"
    accept = (
        float(belief_row["p_feas"]) >= PROB_FEAS_THRESHOLD
        and float(belief_row["risk"]) <= RISK_THRESHOLD
    )
    return bool(accept), belief_row, "probabilistic_belief"


def _conservative_reject_gate(obs: np.ndarray, belief_row: dict) -> bool:
    """Return True for explicit abstention from clearly unsafe entry.

    This gate is intentionally conservative and uses only observable geometry:
    a very low probabilistic margin, a clearly negative width margin, or a
    near-front blockage combined with stuckness.  It is *not* the default
    DEGNAV-Rule policy and never reads the environment passability label.
    """

    p_feas = float(belief_row.get("p_feas", 1.0))
    delta = float(belief_row.get("delta_mean", 1.0))
    d_center = float(obs[1])
    stuck = float(obs[15])
    collision_flag = float(obs[16]) > 0.5
    if collision_flag:
        return True
    if p_feas <= REJECT_P_FEAS_THRESHOLD:
        return True
    if delta <= REJECT_MARGIN_THRESHOLD:
        return True
    if d_center <= REJECT_FRONT_DEPTH_THRESHOLD and stuck >= REJECT_STUCK_THRESHOLD:
        return True
    return False


# ── Controllers ───────────────────────────────────────────────────────────────

def rule_baseline_action(obs):
    """APF-style: always forward + proportional heading correction."""
    he = float(obs[10])
    return _act(0.25, float(np.clip(he * 1.5, -0.8, 0.8)))


def _wz(obs, gain=1.0):
    he = float(obs[10])
    lat = float(obs[11])
    return float(np.clip((0.9 * he - 1.2 * lat) * gain, -0.8, 0.8))


def _follow_space_mode(obs):
    """Detect corner zone from sector depths; return (mode, action) or None.

    FOLLOW_SPACE activates when one near sector is very short while the other
    is open — a signature of an approaching L/S corridor turn.  It suppresses
    ALIGN (which would spin in place) and instead:
      - advances forward while d_cn is still open (pre-turn phase)
      - begins turning toward the inner corner direction once d_cn closes (turn phase)

    The inner corner being short signals the NEW path direction: turn toward it.
    """
    d_ln, d_cn, d_rn = float(obs[0]), float(obs[1]), float(obs[2])
    body_margin = float(obs[9])

    # Invalid (zero/NaN) rays cannot establish a corner.  Max-range/no-return
    # is positive free-space evidence and is exactly what identifies the open
    # arm of an L/S junction.
    near = np.asarray([d_ln, d_cn, d_rn], dtype=float)
    if not bool(np.all(np.isfinite(near) & (near > 0.0))):
        return None

    # Only activate inside the corridor (body_margin at default=4.82 means approach zone).
    if body_margin > 0.5:
        return None

    # Sector asymmetry test: one side tight (corner), other side open
    min_side = min(d_ln, d_rn)
    max_side = max(d_ln, d_rn)
    asymmetry = max_side - min_side

    if (
        d_cn > 0.80
        or asymmetry < 0.25
        or max_side < 0.50
        or min_side > 0.45
    ):
        return None   # symmetric or no tight side — not a corner

    # At an L-turn corner, the forward ray (d_cn) hits the outer wall at roughly
    # the same depth as the tight side (d_cn ≈ min_side).  A false positive from
    # an oblique approach to a straight wall has d_cn >> min_side (forward is
    # open, only one lateral sensor is blocked).  Suppress FOLLOW_SPACE when
    # forward is clearly more open than the tight side.
    if d_cn > 2.6 * min_side + 0.05:
        return None

    # Turn toward the locally observed open side.  The prior implementation
    # turned toward the short/hit ray and could steer directly into the wall.
    turn_dir = -1.0 if d_ln >= d_rn else 1.0

    # Progressive turn: zero while forward is open, ramps up as d_cn drops
    turn_blend = float(np.clip((0.80 - d_cn) / 0.60, 0.30, 1.0))
    wz = turn_dir * 0.65 * turn_blend

    return "FOLLOW_SPACE", _act(0.10, wz)


def _local_centerline_action(
    obs,
    *,
    commit_speed: float,
    explore_speed: float,
    committed: bool,
    lateral_lookahead: float = 0.30,
) -> np.ndarray:
    """Local geometry tracking for straight and curved passages.

    Near/far left-center-right rays vote for a short lookahead heading.  Invalid
    rays have zero weight; max-range no-return rays remain free-space evidence.
    The controller combines that heading with observed path-heading and lateral
    error and the directly observed left/right body clearances, and slows before
    a close front boundary.  The clearance term is necessary for the explicitly
    asymmetric scene: its feasible centerline is displaced around a one-sided
    protrusion even though the nominal polyline remains straight.
    """

    rays = np.asarray(obs[:6], dtype=float)
    valid = np.isfinite(rays) & (rays > 0.0)
    he = float(obs[10])
    lat = float(obs[11])
    d_center_candidates = [rays[index] for index in (1, 4) if valid[index]]
    d_center = min(d_center_candidates) if d_center_candidates else 0.0
    lookahead_heading = 0.0
    if d_center < 0.80:
        left_values = [rays[index] for index in (0, 3) if valid[index]]
        right_values = [rays[index] for index in (2, 5) if valid[index]]
        left_open = max(left_values) if left_values else 0.0
        right_open = max(right_values) if right_values else 0.0
        lookahead_heading = float(np.clip(
            0.80 * (right_open - left_open)
            / max(right_open + left_open, 0.10),
            -0.65,
            0.65,
        ))
    clearance_left = float(obs[6])
    clearance_right = float(obs[7])
    clearance_heading = 0.0
    if math.isfinite(clearance_left) and math.isfinite(clearance_right):
        # In a symmetric corridor ``clearance_right-clearance_left ==
        # -2*lateral_error``.  Use only the unexplained residual so lateral
        # centering is not counted twice; a one-sided protrusion contributes a
        # genuine non-zero residual.
        asymmetric_clearance_residual = (
            clearance_right - clearance_left + 2.0 * lat
        )
        clearance_heading = float(np.clip(
            0.55 * asymmetric_clearance_residual
            / max(abs(clearance_left) + abs(clearance_right), 0.10),
            -0.55,
            0.55,
        ))
    nominal_v = commit_speed if committed else explore_speed
    if d_center < 0.30:
        vx = min(nominal_v, 0.05)
    elif d_center < 0.70:
        vx = min(nominal_v, 0.08)
    else:
        vx = nominal_v
    wz = float(np.clip(
        1.35 * he - 1.8 * lat + 1.8 * lookahead_heading
        + clearance_heading,
        -0.80,
        0.80,
    ))
    return _act(vx, wz)


class TurnCommitFSM:
    """Stateful FSM that commits to L/S-turn once FOLLOW_SPACE triggers.

    Problem: FOLLOW_SPACE turns the robot ~13° then exits because depth-sensor
    asymmetry collapses (both rays see the junction's outer wall).  The centering
    correction then fights the turn, causing oscillation near the junction.

    Fix: once FOLLOW_SPACE fires, lock the turn direction and apply a constant
    (vx=0.08, wz=±0.60) until the robot enters the new corridor segment, detected
    by `|he|` growing past 45° then falling back below 0.35 rad (path-frame
    assignment switches to the new segment).  Timeout at 60 steps as a safety net.
    """

    def __init__(self, variant="full",
                 local_mem: Optional[PassageFailureMemory] = None,
                 cross_mem: Optional[CrossEpisodeMemory] = None,
                 strict_ablation: Optional[str] = None,
                 selector_cfg: Optional[SelectorThresholds] = None,
                 feasibility_cfg: Optional[DynamicFeasibilityConfig] = None,
                 corridor_type: str = "unknown",
                 controller_cfg: Optional[RepairedControllerConfig] = None,
                 env: Optional[HarderNarrowPassageEnv] = None):
        self.variant = variant
        self.local_mem = local_mem
        self.cross_mem = cross_mem
        self.strict_ablation = strict_ablation
        self.selector_cfg = selector_cfg or SelectorThresholds()
        self.feasibility_estimator = DynamicFeasibilityEstimator(feasibility_cfg)
        self.corridor_type = corridor_type
        self.controller_cfg = controller_cfg or RepairedControllerConfig()
        self.env = env
        self.last_audit: dict[str, Any] = {}
        self._previous_feasibility_mode: Optional[AblationMode] = None
        self._reset_control_state()
        self._reset_turn()

    def reset(self):
        self._reset_turn()
        self.feasibility_estimator.reset()
        self.last_audit = {}
        self._previous_feasibility_mode = None
        self._reset_control_state()

    def _reset_control_state(self) -> None:
        self._controller_state: dict[str, Any] = {
            "active_probe_step": 0,
            "probe_no_information_counter": 0,
            "corner_direction_evidence": 0.0,
        }
        self._previous_posterior_variance: float | None = None
        self._previous_controller_mode = ""
        self._recover_steps = 0
        self._recover_cycle_count = 0
        self._recover_start_step = -1
        self._recover_start_goal_distance = float("nan")
        self._recover_end_goal_distance = float("nan")
        self._recover_cooldown = 0
        self._recover_active = False
        self._controller_step_count = 0
        self._passage_state = PassageControlState.APPROACH
        self._passage_state_steps = 0
        self._has_entered_passage = False
        self._previous_entry_readiness_error = float("inf")
        self._previous_goal_distance = float("nan")
        self._last_goal_progress = 0.0
        self._forced_recover_steps = 0
        self._safety_stop_count = 0

    def _reset_turn(self, cooldown: int = 0):
        self._in_turn = False
        self._turn_dir = 0.0
        self._steps_in_turn = 0
        self._peak_abs_he = 0.0
        self._min_bm = float("inf")  # min body_margin seen during this turn
        self._cooldown = cooldown  # steps to suppress FOLLOW_SPACE after a turn
        if hasattr(self, "_controller_state"):
            self._controller_state["corner_direction_evidence"] = 0.0

    def _update_corner_direction_evidence(self, obs) -> None:
        """Accumulate a bounded multi-frame left/right free-space vote.

        Positive evidence means the locally observed open arm is to robot-right;
        negative means robot-left. Invalid rays contribute no evidence, so one
        noisy frame cannot reverse an established corner direction.
        """

        rays = np.asarray(obs[:3], dtype=float)
        score = float(self._controller_state["corner_direction_evidence"])
        if (
            float(obs[9]) <= 0.30
            and bool(np.all(np.isfinite(rays) & (rays > 0.0)))
            and float(rays[1]) <= 1.50
        ):
            vote = float(np.clip(
                (float(rays[2]) - float(rays[0]))
                / max(float(rays[2]) + float(rays[0]), 0.10),
                -1.0,
                1.0,
            ))
            if abs(vote) >= 0.08:
                score = 0.80 * score + 0.20 * vote
        else:
            score *= 0.95
        self._controller_state["corner_direction_evidence"] = float(
            np.clip(score, -1.0, 1.0)
        )

    def _strict_step_inputs(self, obs) -> tuple[Optional[dict], Optional[ModeSelection], dict]:
        if self.strict_ablation is None:
            return None, None, {}

        metrics = self.feasibility_estimator.update(
            obs,
            use_yaw_prior=uses_yaw_prior(self.strict_ablation),
            fixed_uncertainty=uses_fixed_uncertainty(self.strict_ablation),
            # Geometry memory is fused explicitly below.  Passing a binary risk
            # here would silently recreate the old hard-override path.
            memory_risk=0.0,
        )
        struct_lcb, struct_ucb = interval_bounds(
            float(metrics["mu_delta_struct"]),
            float(metrics["sigma_delta_struct"]),
            self.selector_cfg.kappa,
        )
        pose_lcb, pose_ucb = interval_bounds(
            float(metrics["mu_delta_pose"]),
            float(metrics["sigma_delta_pose"]),
            self.selector_cfg.kappa,
        )
        metrics.update({
            "LCB_struct": struct_lcb,
            "UCB_struct": struct_ucb,
            "LCB_pose": pose_lcb,
            "UCB_pose": pose_ucb,
            # Compatibility aliases remain pose-conditioned.
            "LCB": pose_lcb,
            "UCB": pose_ucb,
        })
        posterior_variance = float(metrics["posterior_var_delta_struct"])
        information_gain = (
            0.0
            if self._previous_posterior_variance is None
            else float(self._previous_posterior_variance - posterior_variance)
        )
        if self._previous_controller_mode == "EXPLORE":
            if information_gain <= self.controller_cfg.information_gain_epsilon:
                self._controller_state["probe_no_information_counter"] += 1
            else:
                self._controller_state["probe_no_information_counter"] = 0
        elif self._previous_controller_mode not in {"ALIGN", "APPROACH"}:
            self._controller_state["probe_no_information_counter"] = 0
        self._previous_posterior_variance = posterior_variance
        metrics.update({
            "information_gain": information_gain,
            "probe_no_information_counter": int(
                self._controller_state["probe_no_information_counter"]
            ),
        })
        prior_failed = bool(
            self._previous_feasibility_mode is AblationMode.COMMIT
            and (
                float(obs[16]) > 0.5
                or float(obs[15]) > min(0.40, self.selector_cfg.recover_stuck_score)
            )
        )
        near_rays = np.asarray(obs[:3], dtype=float)
        valid_near_hits = (
            np.isfinite(near_rays)
            & (near_rays > 0.0)
            & (near_rays < self.feasibility_estimator.cfg.max_depth * 0.995)
        )
        explicit_blocker = bool(
            bool(np.all(valid_near_hits))
            and float(obs[9]) <= 0.30
            and float(near_rays[1]) <= 0.10
            and math.isfinite(float(obs[4]))
            and 0.0 < float(obs[4]) <= 0.10
        )
        shared = SharedDecisionState(
            collision=float(obs[16]) > 0.5,
            stuck_score=float(obs[15]),
            memory_risk=0.0,
            prior_commitment_failed=prior_failed,
            explicit_blocker=explicit_blocker,
        )
        base_selection = select_ablation_mode(
            self.strict_ablation, metrics, shared, self.selector_cfg
        )
        if (
            not bool(metrics["passage_aperture_observable"])
            and base_selection.mode in {
                AblationMode.COMMIT,
                AblationMode.EXPLORE,
                AblationMode.REJECT,
            }
            and not shared.explicit_blocker
        ):
            base_selection = ModeSelection(
                AblationMode.EXPLORE,
                "passage aperture is not observable; approach without width-CI reject",
                "width_observability_gate",
            )

        memory_audit: dict[str, Any] = {
            "base_p_feas": float(metrics["p_feas_struct"]),
            "base_logit": float("nan"),
            "memory_delta": 0.0,
            "corrected_logit": float("nan"),
            "corrected_p_feas": float(metrics["p_feas_struct"]),
            "positive_geometry_support": 0,
            "negative_geometry_support": 0,
            "positive_geometry_weight": 0.0,
            "negative_geometry_weight": 0.0,
            "memory_confidence": 0.0,
            "evidence_conflict": False,
            "memory_terminal_reject": False,
            "memory_cautious": False,
            "similarity_weight_sum": 0.0,
            "min_similarity_distance": float("nan"),
            "mean_similarity_distance": float("nan"),
            "similarity_mode": "disabled",
        }
        if (
            self.cross_mem is not None
            and uses_memory(self.strict_ablation)
            and hasattr(self.cross_mem, "correct_feasibility")
        ):
            memory_audit.update(self.cross_mem.correct_feasibility(
                np.asarray(obs),
                base_p_feas=float(metrics["p_feas_struct"]),
                base_structural_margin=float(metrics["mu_delta_struct"]),
                corridor_type=self.corridor_type,
                morphology=self.feasibility_estimator.cfg.morphology,
            ))

        selection = base_selection
        if bool(memory_audit["memory_terminal_reject"]):
            selection = ModeSelection(
                AblationMode.REJECT,
                "uncontested repeated negative geometry memory with negative base margin",
                "bounded_memory_terminal_support",
            )
        elif (
            base_selection.mode is AblationMode.COMMIT
            and float(memory_audit["corrected_p_feas"])
            < float(getattr(self.cross_mem, "cfg", MemoryConfig()).memory_explore_probability)
            and not bool(memory_audit["evidence_conflict"])
        ):
            selection = ModeSelection(
                AblationMode.EXPLORE,
                "bounded negative memory evidence requests more observation",
                "bounded_memory_soft_correction",
            )
        memory_changed_decision = selection.mode is not base_selection.mode

        counterfactual = dict(metrics)
        counterfactual["mu_delta_pose"] = float(
            metrics["mu_delta_without_yaw"]
            if uses_yaw_prior(self.strict_ablation)
            else metrics["mu_delta_with_yaw"]
        )
        cf_lcb, cf_ucb = interval_bounds(
            float(counterfactual["mu_delta_pose"]),
            float(counterfactual["sigma_delta_pose"]),
            self.selector_cfg.kappa,
        )
        counterfactual["LCB_pose"] = cf_lcb
        counterfactual["UCB_pose"] = cf_ucb
        counterfactual_selection = select_ablation_mode(
            self.strict_ablation, counterfactual, shared, self.selector_cfg
        )
        audit = {
            **metrics,
            **memory_audit,
            # Conditional active-sensing branches update these values in place.
            # Declaring them on every step keeps the formal streaming CSV schema
            # fixed even when the first episode does not enter active sensing.
            "active_sensing_phase": "",
            "active_sensing_exit_reason": "",
            "base_selector_mode": base_selection.mode.value,
            "base_selector_reason": base_selection.reason,
            "memory_changed_final_mode": bool(memory_changed_decision),
            "feasibility_mode": selection.mode.value,
            "feasibility_reason": selection.reason,
            "decision_rule": selection.decision_rule,
            "uncertainty_triggered": bool(selection.uncertainty_triggered),
            "yaw_prior_counterfactual_mode": counterfactual_selection.mode.value,
            "yaw_prior_changed_decision": bool(
                selection.mode is not counterfactual_selection.mode
            ),
            "prior_commitment_failed": prior_failed,
            "explicit_blocker": shared.explicit_blocker,
        }
        self._previous_feasibility_mode = selection.mode
        return metrics, selection, audit

    def _update_passage_state(
        self, obs, controller_mode: str, audit: dict
    ) -> tuple[bool, str, bool]:
        """Advance the explicit APPROACH→ALIGN→ENTER→TRAVERSE→EXIT FSM."""

        cfg = self.controller_cfg
        he = abs(float(obs[10]))
        lat = abs(float(obs[11]))
        clearance = min(float(obs[6]), float(obs[7]))
        aperture_observable = bool(
            audit.get("passage_aperture_observable", True)
        )
        ready_to_enter = bool(
            aperture_observable
            and he < cfg.readiness_yaw_threshold
            and lat < cfg.readiness_lateral_threshold
            and clearance > cfg.readiness_clearance_threshold
        )
        readiness_error = float(
            max(0.0, he - cfg.readiness_yaw_threshold)
            + max(0.0, lat - cfg.readiness_lateral_threshold)
            + max(0.0, cfg.readiness_clearance_threshold - clearance)
        )
        goal_distance = float(obs[12])
        goal_progress = (
            0.0
            if not math.isfinite(self._previous_goal_distance)
            else self._previous_goal_distance - goal_distance
        )
        self._last_goal_progress = float(goal_progress)
        mode = str(controller_mode).upper()
        body_margin = float(obs[9])

        if mode == "RECOVER":
            next_state = PassageControlState.RECOVER
            reason = "collision/stuck/prior-commitment recovery"
        elif body_margin <= 0.30:
            self._has_entered_passage = True
            next_state = PassageControlState.TRAVERSE
            reason = "OBB is inside passage; follow local boundaries"
        elif self._has_entered_passage and goal_distance < 1.0:
            next_state = PassageControlState.EXIT
            reason = "left constrained passage after traversal"
        elif self._has_entered_passage:
            # L/S junctions briefly leave an individual rectangular segment;
            # they are not the route exit while the local goal is still far.
            next_state = PassageControlState.TRAVERSE
            reason = "open junction between locally observed passage segments"
        elif not aperture_observable:
            next_state = PassageControlState.APPROACH
            reason = "aperture not observable"
        elif he >= 0.45:
            next_state = PassageControlState.ALIGN
            reason = "large yaw error requires in-place alignment"
        elif (
            lat >= cfg.readiness_lateral_threshold
            or clearance <= cfg.readiness_clearance_threshold
        ):
            next_state = PassageControlState.APPROACH
            reason = "entry-waypoint lateral/clearance approach"
        elif he >= cfg.readiness_yaw_threshold:
            next_state = PassageControlState.ALIGN
            reason = "final yaw readiness not satisfied"
        elif ready_to_enter and mode in {"COMMIT", "CORRIDOR_FOLLOW"}:
            next_state = PassageControlState.ENTER
            reason = "yaw/lateral/clearance readiness satisfied"
        else:
            next_state = PassageControlState.APPROACH
            reason = "active observation before entry commitment"

        if next_state is self._passage_state:
            if (
                next_state is PassageControlState.ALIGN
                and readiness_error
                < self._previous_entry_readiness_error - 1e-3
            ):
                self._passage_state_steps = max(1, self._passage_state_steps - 1)
            elif (
                next_state in {
                    PassageControlState.ENTER,
                    PassageControlState.TRAVERSE,
                    PassageControlState.EXIT,
                }
                and goal_progress > 1e-3
            ):
                self._passage_state_steps = max(1, self._passage_state_steps - 1)
            else:
                self._passage_state_steps += 1
        else:
            self._passage_state = next_state
            self._passage_state_steps = 1
        limits = {
            PassageControlState.ALIGN: cfg.max_align_steps,
            PassageControlState.ENTER: cfg.max_enter_steps,
            PassageControlState.TRAVERSE: cfg.max_traverse_steps,
            PassageControlState.EXIT: cfg.max_exit_steps,
        }
        state_timeout = bool(
            next_state in limits and self._passage_state_steps > limits[next_state]
        )
        self._previous_entry_readiness_error = readiness_error
        self._previous_goal_distance = goal_distance
        return ready_to_enter, reason, state_timeout

    def _record_step(
        self, obs, action, mode: str, audit: dict
    ) -> tuple[np.ndarray, str]:
        cfg = self.controller_cfg
        raw_mode = str(mode)
        if self._forced_recover_steps > 0 and raw_mode != "REJECT":
            raw_mode = "RECOVER"
            mode = "RECOVER"
            self._forced_recover_steps -= 1
            if self._passage_state is not PassageControlState.RECOVER:
                self._passage_state = PassageControlState.RECOVER
                self._passage_state_steps = 1
        ready_to_enter, passage_state_reason, passage_state_timeout = (
            self._update_passage_state(obs, raw_mode, audit)
        )
        if passage_state_timeout and raw_mode not in {"REJECT", "RECOVER"}:
            if self._passage_state is PassageControlState.ALIGN:
                # Alignment cannot correct lateral offset in place.  Execute a
                # bounded low-speed S-curve approach instead of reverse Recover.
                raw_mode = "APPROACH"
                mode = "APPROACH"
                action = _act(
                    min(cfg.approach_speed, 0.08),
                    float(np.clip(
                        1.1 * float(obs[10]) - 2.4 * float(obs[11]),
                        -0.55,
                        0.55,
                    )),
                )
                self._passage_state_steps = 0
            else:
                raw_mode = "RECOVER"
                mode = "RECOVER"
                self._passage_state_steps = 0

        # The explicit passage FSM owns the motion command.  Feasibility mode
        # remains logged separately and can still terminate with REJECT or
        # request bounded RECOVER.
        if raw_mode not in {"REJECT", "RECOVER"}:
            he = float(obs[10])
            lat = float(obs[11])
            if self._passage_state is PassageControlState.ALIGN:
                mode = "ALIGN"
                action = _act(0.0, float(np.clip(1.8 * he, -0.8, 0.8)))
            elif self._passage_state is PassageControlState.APPROACH:
                lateral_lookahead = (
                    self.env.morphology.half_length
                    if self.env is not None else 0.30
                )
                target_heading_error = math.atan2(
                    lat, max(float(lateral_lookahead), 0.10)
                )
                mode = "APPROACH"
                action = _act(
                    cfg.approach_speed,
                    float(np.clip(
                        1.2 * (he - target_heading_error), -0.55, 0.55
                    )),
                )
            elif self._passage_state is PassageControlState.ENTER:
                mode = "ENTER"
                action = _act(
                    cfg.approach_speed,
                    float(np.clip(1.4 * he - 1.8 * lat, -0.50, 0.50)),
                )
            elif self._passage_state is PassageControlState.TRAVERSE:
                if raw_mode == "FOLLOW_SPACE":
                    mode = "TRAVERSE"
                    # Preserve the locally observed corner-turn action.
                else:
                    feasibility_commit = str(
                        audit.get("feasibility_mode", "EXPLORE")
                    ) == "COMMIT"
                    mode = "TRAVERSE"
                    action = _local_centerline_action(
                        obs,
                        commit_speed=cfg.commit_speed,
                        explore_speed=cfg.explore_speed,
                        committed=feasibility_commit,
                        lateral_lookahead=(
                            self.env.morphology.half_length
                            if self.env is not None else 0.30
                        ),
                    )
            elif self._passage_state is PassageControlState.EXIT:
                mode = "EXIT"
                action = _act(
                    0.16,
                    float(np.clip(1.1 * he - 1.2 * lat, -0.50, 0.50)),
                )
        recover_exit_reason = ""
        goal_distance = float(obs[12])

        if self._recover_cooldown > 0:
            self._recover_cooldown -= 1
        if raw_mode == "RECOVER" and self._recover_cooldown > 0:
            mode = "ALIGN"
            action = _act(0.0, float(np.clip(float(obs[10]) * 1.4, -0.5, 0.5)))
            recover_exit_reason = "recover_cooldown_realign"
            self._forced_recover_steps = 0
        elif raw_mode == "RECOVER":
            if not self._recover_active:
                self._recover_active = True
                self._recover_cycle_count += 1
                self._recover_steps = 0
                self._recover_start_step = self._controller_step_count
                self._recover_start_goal_distance = goal_distance
            self._recover_steps += 1
            regression = goal_distance - self._recover_start_goal_distance
            if self._recover_cycle_count >= cfg.max_recover_cycles:
                recover_exit_reason = "max_recover_cycles"
            elif self._recover_steps >= cfg.max_recover_steps:
                recover_exit_reason = "max_recover_steps"
            elif regression > cfg.max_goal_regression:
                recover_exit_reason = "max_goal_regression"

            if recover_exit_reason:
                self._recover_end_goal_distance = goal_distance
                self._recover_cooldown = cfg.recover_cooldown_steps
                self._recover_active = False
                self._forced_recover_steps = 0
                mode = "ALIGN"
                action = _act(
                    0.0, float(np.clip(float(obs[10]) * 1.4, -0.5, 0.5))
                )
                self._recover_steps = 0
            elif self._recover_steps <= cfg.recover_observe_steps:
                # STOP -> small rotation restores observation before rollback.
                scan_sign = (
                    self._turn_dir
                    if self._in_turn and abs(self._turn_dir) > 0.0
                    else (1.0 if self._recover_cycle_count % 2 else -1.0)
                )
                action = _act(0.0, 0.28 * scan_sign)
            else:
                reverse_lookahead = (
                    self.env.morphology.half_length
                    if self.env is not None else 0.30
                )
                reverse_lateral_correction = math.atan2(
                    float(obs[11]), max(float(reverse_lookahead), 0.10)
                )
                action = _act(
                    cfg.recover_speed,
                    float(np.clip(
                        float(obs[10]) * 0.35
                        + 1.2 * reverse_lateral_correction,
                        -0.45,
                        0.45,
                    )),
                )
        elif self._previous_controller_mode == "RECOVER":
            self._recover_end_goal_distance = goal_distance
            recover_exit_reason = "recovery_signal_cleared"
            self._recover_steps = 0
            self._recover_active = False

        requested_linear_action = float(action[0])
        requested_angular_action = float(action[1])
        safety_horizon_steps = (
            1
            if self._in_turn
            else cfg.swept_obb_horizon_steps
        )
        safety_projected = False
        safe_action_reason = "not_checked"
        if (
            self.env is not None
            and (
                abs(float(action[0])) > 1e-9
                or abs(float(action[1])) > 1e-9
            )
            and not self.env.action_is_safe(
                action,
                horizon_steps=safety_horizon_steps,
                minimum_clearance=cfg.swept_obb_minimum_clearance,
            )
        ):
            action, safe_action_reason = self.env.project_to_safe_action(
                action,
                horizon_steps=safety_horizon_steps,
                minimum_clearance=cfg.swept_obb_minimum_clearance,
            )
            safety_projected = True
            if safe_action_reason == "safe_backtrack_recover":
                mode = "RECOVER"
                if self._recover_cycle_count < cfg.max_recover_cycles:
                    self._forced_recover_steps = max(
                        self._forced_recover_steps, cfg.max_recover_steps
                    )
            if (
                abs(float(action[0])) < 1e-9
                and str(mode) in {
                    "EXPLORE", "COMMIT", "FOLLOW_SPACE", "APPROACH"
                }
            ):
                mode = "ALIGN"
            if (
                safe_action_reason in {"stop", "no_safe_candidate"}
                and self._passage_state in {
                    PassageControlState.ALIGN,
                    PassageControlState.APPROACH,
                    PassageControlState.ENTER,
                    PassageControlState.TRAVERSE,
                }
            ):
                self._safety_stop_count += 1
                # A safety STOP is censored control evidence, not proof of
                # geometric infeasibility.  Permit only a bounded number of
                # rollback cycles; after that remain safely stopped/replanning
                # instead of re-arming an endless Recover loop.
                if self._recover_cycle_count < cfg.max_recover_cycles:
                    self._forced_recover_steps = max(
                        self._forced_recover_steps,
                        cfg.max_recover_steps,
                    )
            elif safe_action_reason != "not_checked":
                self._safety_stop_count = 0

        out = dict(audit)
        out["selected_mode"] = _mode_bucket(str(mode)).upper()
        out["controller_mode"] = str(mode)
        out["requested_linear_action"] = requested_linear_action
        out["requested_angular_action"] = requested_angular_action
        out["linear_action"] = float(action[0])
        out["angular_action"] = float(action[1])
        out["translation_stuck_counter"] = int(
            getattr(self.env, "translation_stuck_steps", 0)
        )
        out["rotation_stuck_counter"] = int(
            getattr(self.env, "rotation_stuck_steps", 0)
        )
        out["probe_no_information_counter"] = int(
            self._controller_state["probe_no_information_counter"]
        )
        out["recover_start_step"] = int(self._recover_start_step)
        out["recover_steps"] = int(self._recover_steps)
        out["recover_cycle_count"] = int(self._recover_cycle_count)
        out["recover_start_goal_distance"] = float(
            self._recover_start_goal_distance
        )
        out["recover_end_goal_distance"] = float(
            self._recover_end_goal_distance
        )
        out["recover_exit_reason"] = recover_exit_reason
        out["safety_stop_count"] = int(self._safety_stop_count)
        out["corner_turn_locked"] = bool(self._in_turn)
        out["corner_turn_direction"] = float(self._turn_dir)
        out["corner_turn_steps"] = int(self._steps_in_turn)
        out["corner_direction_evidence"] = float(
            self._controller_state["corner_direction_evidence"]
        )
        out["swept_obb_safety_projected"] = bool(safety_projected)
        out["swept_obb_horizon_steps"] = int(safety_horizon_steps)
        out["safe_action_projection_reason"] = safe_action_reason
        out["passage_control_state"] = self._passage_state.value
        out["passage_state_steps"] = int(self._passage_state_steps)
        out["passage_state_reason"] = passage_state_reason
        out["passage_state_timeout"] = bool(passage_state_timeout)
        out["ready_to_enter"] = bool(ready_to_enter)
        out["predicted_min_clearance"] = float(
            min(float(obs[6]), float(obs[7]))
        )
        out["progress"] = float(self._last_goal_progress)
        if self.env is not None:
            out["swept_obb_clearance"] = float(
                self.env.action_minimum_clearance(
                    action, horizon_steps=safety_horizon_steps
                )
            )
        else:
            out["swept_obb_clearance"] = float("nan")
        self.last_audit = out
        self._previous_controller_mode = str(mode)
        self._controller_step_count += 1
        return action, mode

    def step(self, obs):
        """Return ``(action, mode)`` and expose the strict audit in ``last_audit``."""
        strict_metrics, strict_selection, audit = self._strict_step_inputs(obs)
        if not self._in_turn:
            self._update_corner_direction_evidence(obs)
        if self._in_turn:
            action, mode = self._continue_turn(
                obs, strict_metrics=strict_metrics,
                strict_selection=strict_selection,
                decision_audit=audit,
            )
            return self._record_step(obs, action, mode, audit)

        # Cooldown: don't allow FOLLOW_SPACE to retrigger right after a committed
        # turn completes.  The robot needs a few steps to stabilise heading in the
        # new arm before we trust the asymmetry sensor again.
        if self._cooldown > 0:
            self._cooldown -= 1
            action, mode = fsm_action(
                obs,
                self.variant,
                self.local_mem,
                self.cross_mem,
                strict_ablation=self.strict_ablation,
                selector_cfg=self.selector_cfg,
                corridor_type=self.corridor_type,
                strict_metrics=strict_metrics,
                strict_selection=strict_selection,
                decision_audit=audit,
                controller_state=self._controller_state,
                controller_cfg=self.controller_cfg,
                _env=self.env,
            )
            if mode == "FOLLOW_SPACE":
                # Override: apply centering correction instead of committing a new turn.
                he = float(obs[10])
                lat = float(obs[11])
                wz = float(np.clip(0.9 * he - 1.2 * lat, -0.80, 0.80))
                action, mode = _act(0.12, wz), "COMMIT"
            return self._record_step(obs, action, mode, audit)

        action, mode = fsm_action(
            obs,
            self.variant,
            self.local_mem,
            self.cross_mem,
            strict_ablation=self.strict_ablation,
            selector_cfg=self.selector_cfg,
            corridor_type=self.corridor_type,
            strict_metrics=strict_metrics,
            strict_selection=strict_selection,
            decision_audit=audit,
            controller_state=self._controller_state,
            controller_cfg=self.controller_cfg,
            _env=self.env,
        )
        if mode == "FOLLOW_SPACE":
            wz = float(action[1])
            direction_evidence = float(
                self._controller_state["corner_direction_evidence"]
            )
            if abs(direction_evidence) >= 0.05:
                wz = math.copysign(max(abs(wz), 0.20), direction_evidence)
                action = _act(float(action[0]), wz)
            self._in_turn = True
            self._turn_dir = 1.0 if wz >= 0 else -1.0
            self._steps_in_turn = 0
            self._peak_abs_he = abs(float(obs[10]))
        return self._record_step(obs, action, mode, audit)

    def _continue_turn(self, obs, *, strict_metrics=None, strict_selection=None,
                       decision_audit=None):
        he = float(obs[10])
        body_margin = float(obs[9])
        self._steps_in_turn += 1
        self._peak_abs_he = max(self._peak_abs_he, abs(he))
        self._min_bm = min(self._min_bm, body_margin)

        # Exit: peaked > 30° AND heading is small again → entered the new segment.
        # Threshold π/6 (not π/4) so seeds where peak is ~44° don't spin a full
        # circle before the exit condition fires.
        # Natural exit always gets cooldown=8: the robot just entered a new arm and
        # needs time to stabilise before FOLLOW_SPACE fires again.
        # For asymmetric/narrow_entry false positives, the path frame never switches
        # (straight corridor), so abs(he) never drops below 0.35 — those exits go
        # through the bm>0.5 or timeout paths, which keep cooldown=0.
        if self._peak_abs_he > math.pi / 6 and abs(he) < 0.35:
            self._reset_turn(cooldown=8)
            return fsm_action(
                obs,
                self.variant,
                self.local_mem,
                self.cross_mem,
                strict_ablation=self.strict_ablation,
                selector_cfg=self.selector_cfg,
                corridor_type=self.corridor_type,
                strict_metrics=strict_metrics,
                strict_selection=strict_selection,
                decision_audit=decision_audit,
                controller_state=self._controller_state,
                controller_cfg=self.controller_cfg,
                _env=self.env,
            )

        # Exit: open area (past the junction turn region).
        if body_margin > 0.5:
            self._reset_turn(cooldown=0)
            return fsm_action(
                obs,
                self.variant,
                self.local_mem,
                self.cross_mem,
                strict_ablation=self.strict_ablation,
                selector_cfg=self.selector_cfg,
                corridor_type=self.corridor_type,
                strict_metrics=strict_metrics,
                strict_selection=strict_selection,
                decision_audit=decision_audit,
                controller_state=self._controller_state,
                controller_cfg=self.controller_cfg,
                _env=self.env,
            )

        # Exit: timeout.
        if self._steps_in_turn > 60:
            self._reset_turn(cooldown=0)
            return fsm_action(
                obs,
                self.variant,
                self.local_mem,
                self.cross_mem,
                strict_ablation=self.strict_ablation,
                selector_cfg=self.selector_cfg,
                corridor_type=self.corridor_type,
                strict_metrics=strict_metrics,
                strict_selection=strict_selection,
                decision_audit=decision_audit,
                controller_state=self._controller_state,
                controller_cfg=self.controller_cfg,
                _env=self.env,
            )

        # Bounded forward + committed turn. Ramp angular velocity across several
        # closed-loop observations instead of extrapolating an immediate hard
        # turn from the still-narrow approach arm.  The swept-OBB projector owns
        # any further slowdown required by the actual geometry.
        vx = self.controller_cfg.corner_turn_speed
        if self._steps_in_turn <= 3:
            wz = self._turn_dir * 0.32
        elif self._steps_in_turn <= 7:
            wz = self._turn_dir * 0.45
        else:
            wz = self._turn_dir * 0.60
        return _act(vx, wz), "FOLLOW_SPACE"


def fsm_action(obs, variant="full", local_mem: Optional[PassageFailureMemory] = None,
               cross_mem: Optional[CrossEpisodeMemory] = None,
               max_recover_steps=25, _env=None,
               strict_ablation: Optional[str] = None,
               selector_cfg: Optional[SelectorThresholds] = None,
               corridor_type: str = "unknown",
               strict_metrics: Optional[dict] = None,
               strict_selection: Optional[ModeSelection] = None,
               decision_audit: Optional[dict] = None,
               controller_state: Optional[dict[str, Any]] = None,
               controller_cfg: Optional[RepairedControllerConfig] = None):
    """Geometry-FSM step.  Returns (action, mode_str)."""
    controller_state = controller_state if controller_state is not None else {}
    controller_cfg = controller_cfg or RepairedControllerConfig(
        max_recover_steps=int(max_recover_steps)
    )
    he = float(obs[10])
    lat = float(obs[11])
    stuck = float(obs[15])
    collision_flag = float(obs[16]) > 0.5

    dist = float(obs[12])
    if dist < 0.30:
        return _act(0.0, 0.0), "STOP"

    # FSM mode from cross-episode memory
    if cross_mem is not None and strict_ablation is None:
        # Legacy non-strict experiments retain their historical interface.
        # Strict evaluation fuses bounded memory evidence in TurnCommitFSM.
        cm_mode = cross_mem.fsm_mode(obs, corridor_type)
        if cm_mode == FSMMode.REJECT:
            return _act(-0.10, 0.0), "REJECT"
        cautious = cm_mode == FSMMode.CAUTIOUS
    elif strict_ablation is not None and decision_audit is not None:
        cautious = bool(decision_audit.get("memory_cautious", False))
    else:
        cautious = False

    speed_fwd = 0.15 if cautious else 0.20
    speed_explore = 0.05 if cautious else 0.08

    # Cautious memory is informative but is not the terminal REJECT state.  The
    # strict selector therefore receives a sub-threshold risk; explicit memory
    # rejection has already returned above.  The legacy path keeps its original
    # 0/1 risk behavior exactly.
    memory_risk = (0.5 if cautious else 0.0) if strict_ablation else (1.0 if cautious else 0.0)
    if strict_ablation is None:
        accept_feasible, _belief_row, _decision_rule = _feasibility_acceptance(
            obs,
            variant,
            memory_risk=memory_risk,
        )
    else:
        selector_cfg = selector_cfg or SelectorThresholds()
        if strict_metrics is None:
            estimator = DynamicFeasibilityEstimator()
            strict_metrics = estimator.update(
                obs,
                use_yaw_prior=uses_yaw_prior(strict_ablation),
                fixed_uncertainty=uses_fixed_uncertainty(strict_ablation),
                memory_risk=memory_risk,
            )
            strict_metrics["LCB_struct"], strict_metrics["UCB_struct"] = interval_bounds(
                float(strict_metrics["mu_delta_struct"]),
                float(strict_metrics["sigma_delta_struct"]),
                selector_cfg.kappa,
            )
            strict_metrics["LCB_pose"], strict_metrics["UCB_pose"] = interval_bounds(
                float(strict_metrics["mu_delta_pose"]),
                float(strict_metrics["sigma_delta_pose"]),
                selector_cfg.kappa,
            )
            strict_metrics["LCB"] = strict_metrics["LCB_pose"]
            strict_metrics["UCB"] = strict_metrics["UCB_pose"]
        _belief_row = strict_metrics
        accept_feasible = float(strict_metrics["mu_delta"]) > selector_cfg.tau_commit
        _decision_rule = "strict_ablation"
        if strict_selection is None:
            strict_selection = select_ablation_mode(
                strict_ablation,
                strict_metrics,
                SharedDecisionState(
                    collision=collision_flag,
                    stuck_score=stuck,
                    memory_risk=memory_risk,
                    explicit_blocker=bool(
                        float(obs[1]) <= 0.30 and max(float(obs[0]), float(obs[2])) <= 0.60
                    ),
                ),
                selector_cfg,
            )
        if decision_audit is not None:
            decision_audit.update({
                "feasibility_mode": strict_selection.mode.value,
                "feasibility_reason": strict_selection.reason,
                "decision_rule": strict_selection.decision_rule,
                "uncertainty_triggered": bool(
                    strict_selection.uncertainty_triggered
                ),
            })

    # FOLLOW_SPACE: sector-depth corner detection takes priority over ALIGN.
    # Only active for variants that use alignment (i.e. not no_alignment).
    fs = None if variant == "no_alignment" else _follow_space_mode(obs)
    if fs is not None:
        return fs[1], fs[0]

    if variant == "feasibility_reject" and _conservative_reject_gate(obs, _belief_row):
        return _act(0.0, 0.0), "REJECT"

    # Local memory overrides
    pose_alignment_required = bool(
        strict_selection is not None
        and strict_selection.mode is AblationMode.EXPLORE
        and strict_metrics is not None
        and uses_yaw_prior(strict_ablation)
        and float(strict_metrics["mu_delta_struct"]) > 0.0
        and float(strict_metrics["mu_delta_pose"]) <= selector_cfg.tau_commit
        and abs(he) > 1e-3
    )
    structural_probe_required = bool(
        strict_selection is not None
        and strict_selection.mode is AblationMode.EXPLORE
        and strict_metrics is not None
        and float(strict_metrics["LCB_struct"]) <= selector_cfg.tau_commit
        and float(obs[9]) > 0.30
    )
    aperture_observable = bool(
        strict_metrics is None
        or strict_metrics.get("passage_aperture_observable", True)
    )
    if local_mem is not None and local_mem.should_recover(obs):
        mode = "RECOVER"
    elif collision_flag or stuck > 0.80:
        mode = "RECOVER" if variant not in ("no_recovery",) else "COMMIT"
    elif variant == "no_alignment":
        mode = "COMMIT_NOALIGN" if accept_feasible else "EXPLORE_NOALIGN"
    elif pose_alignment_required:
        mode = "ALIGN"
    elif abs(he) > 0.7 and float(obs[9]) > 0.3:
        # Large heading error AND outside/wide corridor → full ALIGN (spin in place).
        mode = "ALIGN"
    elif abs(he) > 0.7:
        # Large heading error inside a NARROW corridor: spinning toward the final goal
        # would drive the robot into the wall (straight-line goal ≠ local path direction).
        # Instead use corridor-following: go slow and correct laterally only.
        mode = "CORRIDOR_FOLLOW"
    elif strict_selection is not None and not aperture_observable:
        mode = "APPROACH"
    elif strict_selection is not None:
        mode = strict_selection.mode.value
    elif not accept_feasible:
        mode = "EXPLORE"
    elif abs(he) > 0.45 or abs(lat) > 0.25:
        mode = "EXPLORE"
    else:
        mode = "COMMIT"

    # Inside a narrow corridor, replace goal-heading steering with local corridor
    # navigation (final goal heading ≠ local corridor direction for L/S corridors).
    in_narrow = float(obs[9]) < 0.30   # body_margin gate
    if in_narrow:
        cl_v, cr_v = float(obs[6]), float(obs[7])
        d_ln_v, d_cn_v, d_rn_v = float(obs[0]), float(obs[1]), float(obs[2])

        # (A) Lateral centering: balance left/right wall clearances.
        wz_center = float(np.clip(
            -1.6 * (cl_v - cr_v) / max(cl_v + cr_v, 0.05), -0.8, 0.8
        ))

        # (B) Gap-following: when d_cn is below the sector average the robot is
        # heading sideways.  Steer toward the more open sector direction.
        # FOLLOW_SPACE takes priority at turns, so this runs during straight segments.
        if d_cn_v < 0.5 * (d_ln_v + d_rn_v):
            wz_gap = float(np.clip(
                1.2 * (d_rn_v - d_ln_v) / max(d_rn_v + d_ln_v, 0.1), -0.8, 0.8
            ))
        else:
            wz_gap = 0.0

        wz_narrow = float(np.clip(wz_center + wz_gap, -0.8, 0.8))
    else:
        wz_narrow = None

    # Forward depth gate: if d_cn is very short the robot is heading into a wall.
    # Back up so the heading correction can rotate to a better direction first.
    # Threshold 0.20 avoids oscillation with the ~0.24-0.28 typical near-wall readings.
    d_cn_val = float(obs[1])
    if in_narrow and d_cn_val < 0.20:
        vx_in_narrow = -0.06   # back up
    else:
        vx_in_narrow = None   # use mode default

    if mode == "ALIGN":
        action = _act(0.0, float(np.clip(he * 1.8, -0.8, 0.8)))
    elif mode == "CORRIDOR_FOLLOW":
        # Inside narrow corridor with misleading final-goal heading error:
        # use combined centering + gap-following instead of spinning toward goal.
        wz_cf = wz_narrow if in_narrow else float(np.clip(-2.0 * lat, -0.8, 0.8))
        vx_cf = vx_in_narrow if vx_in_narrow is not None else 0.08
        action = _act(vx_cf, wz_cf)
    elif mode == "COMMIT":
        wz = wz_narrow if in_narrow else _wz(obs, 1.0 if not cautious else 1.5)
        vx = vx_in_narrow if vx_in_narrow is not None else speed_fwd
        action = _act(vx, wz)
    elif mode == "EXPLORE":
        wz = wz_narrow if in_narrow else _wz(obs, 1.6 if not cautious else 2.0)
        if structural_probe_required:
            no_info = int(controller_state.get("probe_no_information_counter", 0))
            if no_info >= controller_cfg.max_probe_no_information:
                # Finite active sensing: leave the scan and re-align instead of
                # waiting forever on an unchanged observation.
                mode = "ALIGN"
                action = _act(0.0, float(np.clip(he * 1.8, -0.8, 0.8)))
                if decision_audit is not None:
                    decision_audit["active_sensing_exit_reason"] = "no_information_gain"
                return action, mode
            probe_step = int(controller_state.get("active_probe_step", 0))
            half = max(1, int(controller_cfg.explore_half_scan_steps))
            phase = probe_step % (2 * half)
            scan_sign = 1.0 if phase < half else -1.0
            controller_state["active_probe_step"] = probe_step + 1
            vx = float(controller_cfg.explore_speed)
            wz = float(scan_sign * controller_cfg.explore_scan_rate)
            if decision_audit is not None:
                decision_audit["active_sensing_phase"] = (
                    "scan_left" if scan_sign > 0.0 else "scan_right"
                )
        else:
            vx = vx_in_narrow if vx_in_narrow is not None else speed_explore
        action = _act(vx, wz)
    elif mode == "APPROACH":
        action = _act(
            float(controller_cfg.approach_speed),
            float(np.clip(1.2 * he - 1.0 * lat, -0.45, 0.45)),
        )
    elif mode == "COMMIT_NOALIGN":
        vx = vx_in_narrow if vx_in_narrow is not None else speed_fwd
        action = _act(vx, 0.0)
    elif mode == "EXPLORE_NOALIGN":
        vx = vx_in_narrow if vx_in_narrow is not None else speed_explore
        action = _act(vx, 0.0)
    elif mode == "RECOVER":
        action = _act(
            float(controller_cfg.recover_speed),
            float(np.clip(he * 0.4, -0.8, 0.8)),
        )
    else:
        action = _act(0.0, 0.0)

    return action, mode


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env: HarderNarrowPassageEnv, method: str,
                local_mem: Optional[PassageFailureMemory],
                cross_mem: Optional[CrossEpisodeMemory],
                max_steps: int,
                entry_obs_ref: list,
                variant: str = "full",
                agent: Optional["TurnCommitFSM"] = None,
                entry_jitter_sigma: float = 0.0,
                episode_seed: Optional[int] = None,
                reset_options: Optional[dict] = None,
                observation_transform: Optional[Any] = None,
                step_trace_out: Optional[Any] = None,
                step_trace_context: Optional[dict] = None) -> dict:
    obs, _ = env.reset(seed=episode_seed, options=reset_options)
    if observation_transform is not None:
        obs = np.asarray(observation_transform(obs.copy(), 0), dtype=np.float32)

    # Lateral entry jitter: perturb start position perpendicular to heading.
    # Exposes RECOVER mode by forcing the robot off-center at episode start.
    if entry_jitter_sigma > 0.0:
        yaw = float(env.pose[2])
        jitter_rng = np.random.default_rng(episode_seed)
        lateral = float(jitter_rng.normal(0.0, entry_jitter_sigma))
        env.pose[0] += lateral * (-math.cos(yaw))
        env.pose[1] += lateral * math.sin(yaw)
        obs = env._obs()

    entry_obs = obs.copy()
    entry_obs_ref.clear()
    entry_obs_ref.append(entry_obs)
    entry_yaw = float(env.pose[2])
    entry_heading_error = float(entry_obs[10])
    entry_lateral_offset = float(entry_obs[11])
    margin_row, margin_step, margin_source, margin_obs = _canonical_margin_snapshot(
        env,
        entry_obs,
        variant=variant,
    )
    if agent is not None and agent.strict_ablation is not None:
        # The legacy memory stored the open-space reset observation.  Strict
        # feasibility runs store the canonical entry geometry so the supposedly
        # geometry-indexed memory is actually indexed by passage geometry.
        entry_obs_ref[0] = margin_obs.copy()

    if agent is not None:
        agent.reset()

    if agent is not None and agent.strict_ablation is not None:
        snapshot_estimator = DynamicFeasibilityEstimator(
            agent.feasibility_estimator.cfg
        )
        margin_row = snapshot_estimator.update(
            margin_obs,
            use_yaw_prior=uses_yaw_prior(agent.strict_ablation),
            fixed_uncertainty=uses_fixed_uncertainty(agent.strict_ablation),
        )
        margin_row["LCB_struct"], margin_row["UCB_struct"] = interval_bounds(
            float(margin_row["mu_delta_struct"]),
            float(margin_row["sigma_delta_struct"]),
            agent.selector_cfg.kappa,
        )
        margin_row["LCB_pose"], margin_row["UCB_pose"] = interval_bounds(
            float(margin_row["mu_delta_pose"]),
            float(margin_row["sigma_delta_pose"]),
            agent.selector_cfg.kappa,
        )
        margin_row["LCB"] = margin_row["LCB_pose"]
        margin_row["UCB"] = margin_row["UCB_pose"]

    done = False
    steps = 0
    env_step_count = 0
    any_collision = False
    min_bm = float("inf")
    tight_passage = False   # bm < 0.05 m (near-risk, not a physical collision)
    body_overlap_steps = 0  # steps where bm < 0 (Habitat sliding through wall)
    mode_counts: Counter[str] = Counter()
    last_mode = ""
    previous_mode_bucket: Optional[str] = None
    oscillation_count = 0
    alignment_steps = 0
    recovery_attempts = 0
    first_reject_step: Optional[int] = None
    wasted_commitment = False
    path_length = 0.0
    belief_samples: List[dict] = [margin_row]
    # Buffer one episode so the categorical termination reason can be written
    # into the final step even when the caller streams a multi-million-row full
    # evaluation to disk.
    episode_trace_rows: list[dict[str, Any]] = []

    if cross_mem is not None:
        cross_mem.reset_local()

    info: dict[str, Any] = {}
    while not done and steps < max_steps:
        bm = float(obs[9])
        if bm < min_bm:
            min_bm = bm
        if bm < 0.05:
            tight_passage = True
        if bm < 0.0:
            body_overlap_steps += 1
        if float(obs[16]) > 0.5:
            any_collision = True

        if agent is None or agent.strict_ablation is None:
            step_belief = _finite_belief_metrics(obs, variant=variant)
            if step_belief is not None:
                belief_samples.append(step_belief)

        if method == "rule_baseline":
            action = rule_baseline_action(obs)
            mode = "COMMIT"
        elif agent is not None:
            action, mode = agent.step(obs)
        else:
            action, mode = fsm_action(obs, variant, local_mem, cross_mem)

        if agent is not None and agent.strict_ablation is not None:
            audit_row = dict(agent.last_audit)
            belief_samples.append(audit_row)
            if step_trace_out is not None:
                navigation = env.navigation_diagnostics()
                candidate_action = [
                    float(audit_row.get("requested_linear_action", action[0])),
                    float(audit_row.get("requested_angular_action", action[1])),
                ]
                executed_action = [float(action[0]), float(action[1])]
                episode_trace_rows.append({
                    **dict(step_trace_context or {}),
                    "step": int(steps),
                    "method": str(agent.strict_ablation),
                    "robot_pose": json.dumps(
                        [float(value) for value in env.pose], separators=(",", ":")
                    ),
                    "local_goal": json.dumps(
                        navigation["local_goal"], separators=(",", ":")
                    ),
                    "active_corridor_arm": int(
                        navigation["active_corridor_arm"]
                    ),
                    "path_progress": float(navigation["path_progress"]),
                    "observation": json.dumps(
                        [float(value) for value in np.asarray(obs).reshape(-1)],
                        separators=(",", ":"),
                    ),
                    "D_hat": float(audit_row.get("d_hat", float("nan"))),
                    "W_req_mean": float(
                        audit_row.get("mu_W", audit_row.get("w_req_prior", float("nan")))
                    ),
                    "W_req_cons": float(
                        audit_row.get("w_req_cons", float("nan"))
                    ),
                    "margin_cons": float(
                        audit_row.get("LCB_pose", float("nan"))
                    ),
                    "uncertainty": float(
                        audit_row.get("sigma_delta", float("nan"))
                    ),
                    "candidate_action": json.dumps(
                        candidate_action, separators=(",", ":")
                    ),
                    "executed_action": json.dumps(
                        executed_action, separators=(",", ":")
                    ),
                    "robot_pose_after": json.dumps(
                        [float(value) for value in env.pose], separators=(",", ":")
                    ),
                    "termination_reason": "",
                    **audit_row,
                })

        last_mode = str(mode)
        current_mode_bucket = _mode_bucket(last_mode)
        mode_counts[current_mode_bucket] += 1
        if str(mode).upper() == "ALIGN":
            alignment_steps += 1
        if current_mode_bucket == "recover" and previous_mode_bucket != "recover":
            recovery_attempts += 1
        if current_mode_bucket == "commit" and not env.is_passable:
            wasted_commitment = True
        if (
            previous_mode_bucket is not None
            and current_mode_bucket != previous_mode_bucket
        ):
            oscillation_count += 1
        previous_mode_bucket = current_mode_bucket

        if current_mode_bucket == "reject":
            if first_reject_step is None:
                first_reject_step = steps
            steps += 1
            info = {
                "success": 0.0,
                "collision": 0.0,
                "stuck": float(obs[15] >= 1.0),
                "passable": env.is_passable,
                "corridor_type": env.corridor_type.value if env.corridor_type else "unknown",
                "passage_width": getattr(env, "_episode_W", float(obs[8])),
                "structural_margin_gt": (
                    getattr(env, "_episode_W", float(obs[8]))
                    - env.morphology.structural_required_width
                ),
                "body_margin": float(obs[9]),
                "min_body_margin": min_bm,
            }
            break

        previous_position = env.pose[:2].copy()
        obs, _, term, trunc, info = env.step(action)
        env_step_count += 1
        if observation_transform is not None:
            obs = np.asarray(
                observation_transform(obs.copy(), steps + 1), dtype=np.float32
            )
        path_length += float(np.linalg.norm(env.pose[:2] - previous_position))
        post_bm = float(obs[9])
        if post_bm < min_bm:
            min_bm = post_bm
        if post_bm < 0.05:
            tight_passage = True
        if post_bm < 0.0:
            body_overlap_steps += 1
        if float(obs[16]) > 0.5 or float(info.get("collision", 0.0)) > 0.5:
            any_collision = True
        if episode_trace_rows:
            episode_trace_rows[-1]["collision_flag"] = float(
                float(obs[16]) > 0.5
                or float(info.get("collision", 0.0)) > 0.5
            )
            episode_trace_rows[-1]["robot_pose_after"] = json.dumps(
                [float(value) for value in env.pose], separators=(",", ":")
            )
        if agent is None or agent.strict_ablation is None:
            post_belief = _finite_belief_metrics(obs, variant=variant)
            if post_belief is not None:
                belief_samples.append(post_belief)
        done = term or trunc

        if local_mem is not None and any_collision:
            local_mem.add_failure(obs)

        steps += 1

    if not np.isfinite(min_bm):
        min_bm = float(obs[9])

    success = float(info.get("success", 0.0))
    passable_label = bool(info.get("passable", True))
    is_false_feasible = (
        str(info.get("corridor_type", "")).lower() == "false_feasible"
        or not passable_label
    )
    timeout = bool(steps >= max_steps and success <= 0.5 and not any_collision)
    stuck = bool(info.get("stuck", 0.0))
    reject = bool(mode_counts.get("reject", 0) > 0)
    correct_reject = bool(reject and not passable_label)
    false_reject = bool(reject and passable_label)
    wasted_attempt = bool(is_false_feasible and not correct_reject)
    # Collision-free success: reached goal AND never penetrated a wall (bm >= 0 throughout).
    # The base success criterion only checks dist/heading/lateral — no collision guard.
    # body_overlap_steps > 0 means the robot was inside a wall (Habitat allow_sliding).
    collision_free_success = float(
        success > 0.5 and not any_collision and body_overlap_steps == 0
    )
    final_belief = belief_samples[-1] if belief_samples else margin_row
    finite_deltas = [
        float(sample["delta_mean"])
        for sample in belief_samples
        if np.isfinite(float(sample.get("delta_mean", float("nan"))))
    ]
    if correct_reject:
        final_outcome = "correct_reject"
    elif false_reject:
        final_outcome = "false_reject"
    elif any_collision:
        final_outcome = "collision"
    elif success > 0.5:
        final_outcome = "success"
    elif stuck:
        final_outcome = "stuck"
    elif timeout:
        final_outcome = "timeout"
    else:
        final_outcome = "controller_failure"

    # Canonical outcome flags are mutually exclusive and complete.  Preserve
    # the simulator's raw success below for audit, but never let simultaneous
    # goal/contact or stuck-at-budget events count in two outcome columns.
    simulator_success = float(success)
    success = float(final_outcome == "success")
    correct_reject = bool(final_outcome == "correct_reject")
    false_reject = bool(final_outcome == "false_reject")
    any_collision = bool(final_outcome == "collision")
    stuck = bool(final_outcome == "stuck")
    timeout = bool(final_outcome == "timeout")

    if episode_trace_rows:
        episode_trace_rows[-1]["termination_reason"] = final_outcome
    if step_trace_out is not None:
        for trace_row in episode_trace_rows:
            step_trace_out.append(trace_row)

    attempted = env_step_count > 0
    explicit_blocker_observed = bool(
        agent is not None and agent.last_audit.get("explicit_blocker", False)
    )
    if success > 0.5:
        episode_outcome = EpisodeOutcome.SUCCESS
    elif reject and not attempted:
        episode_outcome = EpisodeOutcome.REJECT_CENSORED
    elif any_collision:
        # The current observation does not identify whether contact was caused
        # by physical infeasibility or a poor command.  Keep it out of negative
        # geometry memory rather than using the evaluator's passable label.
        episode_outcome = EpisodeOutcome.COLLISION_UNKNOWN
    elif timeout:
        episode_outcome = EpisodeOutcome.TIMEOUT_CONTROL
    elif stuck:
        episode_outcome = EpisodeOutcome.STUCK_CONTROL
    elif reject and explicit_blocker_observed:
        episode_outcome = EpisodeOutcome.GEOMETRIC_INFEASIBLE
    else:
        episode_outcome = EpisodeOutcome.TIMEOUT_CONTROL

    return {
        "variant": variant,
        "success": success,
        "simulator_success": simulator_success,
        "collision_free_success": collision_free_success,
        "collision": float(any_collision),
        # tight_passage_rate: bm < 0.05 m (near-risk; NOT a physical collision)
        "tight_passage_rate": float(tight_passage),
        # body_overlap_rate: fraction of steps with bm < 0 (wall penetration via allow_sliding)
        "body_overlap_rate": body_overlap_steps / max(steps, 1),
        "min_body_margin": min_bm,
        "steps": steps,
        "env_step_count": int(env_step_count),
        "attempted": bool(attempted),
        "path_length": float(path_length),
        "completion_time": float(steps * env.dt),
        "alignment_time": float(alignment_steps * env.dt),
        "oscillation_count": int(oscillation_count),
        "recovery_attempts": int(recovery_attempts),
        "time_to_reject": (
            float(first_reject_step * env.dt)
            if first_reject_step is not None
            else float("nan")
        ),
        "wasted_commitment": float(wasted_commitment),
        "passable": passable_label,
        "passable_label": float(passable_label),
        "is_false_feasible": float(is_false_feasible),
        "corridor_type": info.get("corridor_type", "unknown"),
        "passage_width": info.get("passage_width", float(obs[8])),
        "available_width": info.get("passage_width", float(obs[8])),
        "structural_margin_gt": info.get(
            "structural_margin_gt",
            float(info.get("passage_width", float(obs[8])))
            - env.morphology.structural_required_width,
        ),
        "entry_yaw": entry_yaw,
        "initial_yaw": entry_heading_error,
        "entry_heading_error": entry_heading_error,
        "initial_lateral_offset": entry_lateral_offset,
        "lateral_offset": entry_lateral_offset,
        "strict_success": collision_free_success,
        "near_collision": float(tight_passage),
        "min_clearance": min_bm,
        "body_margin_min": min_bm,
        "body_margin": float(margin_obs[9]) if np.isfinite(float(margin_obs[9])) else min_bm,
        "reject": float(reject),
        "correct_reject": float(correct_reject),
        "false_reject": float(false_reject),
        "timeout": float(timeout),
        "stuck": float(stuck),
        "wasted_attempt": float(wasted_attempt),
        "final_outcome": final_outcome,
        "episode_outcome": episode_outcome.value,
        "geometry_caused": float(
            episode_outcome in {
                EpisodeOutcome.GEOMETRIC_INFEASIBLE,
                EpisodeOutcome.COLLISION_GEOMETRY,
            }
        ),
        "geometry_memory_write": 0.0,
        "memory_write_type": "not_written",
        "memory_write_reason": "runner has not applied episode-memory policy",
        "selected_mode": last_mode,
        "final_heading_error": float(obs[10]),
        "final_lateral_error": float(obs[11]),
        "margin_snapshot_step": margin_step,
        "margin_snapshot_source": margin_source,
        "final_delta_mean": float(final_belief.get("delta_mean", float("nan"))),
        "min_delta_mean": float(np.min(finite_deltas)) if finite_deltas else float("nan"),
        "_belief_row": margin_row,
        "_belief_samples": belief_samples,
        "_final_belief_row": final_belief,
        "_mode_counts": dict(mode_counts),
        "_last_mode": last_mode,
    }


# ── Per-method runner ─────────────────────────────────────────────────────────

def eval_method(method: str, ctypes: List[str], n_episodes: int,
                seed: int = 42, entry_jitter_sigma: float = 0.0,
                log_belief_diagnostics: bool = False,
                width_range: tuple[float, float] = (0.45, 0.90),
                max_steps: Optional[int] = None,
                strict_ablation: Optional[str] = None,
                cross_mem_override: Optional[CrossEpisodeMemory] = None,
                episode_seeds: Optional[List[int]] = None,
                episode_ids: Optional[List[str]] = None,
                progress: bool = True,
                yaw_noise: Optional[float] = None,
                step_trace_out: Optional[Any] = None) -> List[dict]:
    env_cfg = {
        "corridor_types": ctypes,
        "seed": seed,
        "width_range": width_range,
    }
    if max_steps is not None:
        env_cfg["max_steps"] = max_steps
    if yaw_noise is not None:
        env_cfg["yaw_noise"] = float(yaw_noise)
    env = HarderNarrowPassageEnv(env_cfg)
    if strict_ablation is not None and strict_ablation not in STRICT_FEASIBILITY_ABLATIONS:
        raise ValueError(
            f"Unknown strict ablation {strict_ablation!r}; "
            f"expected {STRICT_FEASIBILITY_ABLATIONS}"
        )
    variant = _variant_for_method(method)

    local_mem = None
    cross_mem = cross_mem_override
    agent = None

    if method == "fsm_local_memory":
        local_mem = PassageFailureMemory()
    elif method == "fsm_cross_memory" and cross_mem is None:
        cross_mem = CrossEpisodeMemory()
    if strict_ablation is not None and not uses_memory(strict_ablation):
        cross_mem = None

    # Use TurnCommitFSM for all FSM variants (handles L/S turn commitment)
    if method != "rule_baseline":
        agent = TurnCommitFSM(variant=variant, local_mem=local_mem,
                              cross_mem=cross_mem,
                              strict_ablation=strict_ablation,
                              feasibility_cfg=DynamicFeasibilityConfig(
                                  morphology=env.morphology
                              ),
                              corridor_type=(ctypes[0] if len(ctypes) == 1 else "unknown"),
                              env=env)

    all_stats = []
    entry_obs_ref = []
    for ep_idx in range(n_episodes):
        if method == "fsm_local_memory" and local_mem is not None:
            local_mem.reset()

        episode_seed = (
            int(episode_seeds[ep_idx])
            if episode_seeds is not None
            else None
        )
        scenario_id = (
            str(episode_ids[ep_idx])
            if episode_ids is not None
            else f"seed{seed}_ep{ep_idx:06d}"
        )
        stat = run_episode(env, method, local_mem, cross_mem,
                           max_steps=env.max_steps, entry_obs_ref=entry_obs_ref,
                           variant=variant,
                           agent=agent, entry_jitter_sigma=entry_jitter_sigma,
                           episode_seed=episode_seed,
                           step_trace_out=step_trace_out,
                           step_trace_context={
                               "episode_id": scenario_id,
                               "scenario_id": scenario_id,
                               "seed": seed,
                               "episode_seed": (
                                   episode_seed if episode_seed is not None else ""
                               ),
                               "corridor_type": (
                                   ctypes[0] if len(ctypes) == 1 else "unknown"
                               ),
                               "method": strict_ablation or method,
                           })
        belief_row = stat.pop("_belief_row", {})
        belief_samples = stat.pop("_belief_samples", [])
        final_belief_row = stat.pop("_final_belief_row", {})
        mode_counts = stat.pop("_mode_counts", {})
        last_mode = stat.pop("_last_mode", "unavailable")
        stat["episode_idx"] = ep_idx
        stat["scenario_id"] = scenario_id
        stat["episode_id"] = scenario_id if strict_ablation else f"{method}_{seed}_{ep_idx:06d}"
        stat["episode_seed"] = episode_seed if episode_seed is not None else ""
        stat["scene_id"] = "procedural_v2"
        stat["domain"] = "procedural_v2"
        stat["split"] = "procedural_v2"
        stat["seed"] = seed
        stat["method"] = method
        stat["variant"] = strict_ablation or variant
        stat["paper_method"] = METHOD_LABELS.get(method, method)
        has_interface = method != "rule_baseline"
        # The reactive rule baseline receives direct geometry observations but
        # does not implement the DEGNAV feasibility-belief interface.  Keep its
        # belief columns as NaN so margin-phase plots cannot mistake it for a
        # belief-aware method.
        stat = finalize_episode_row(
            stat,
            belief=belief_row if has_interface else None,
            belief_samples=belief_samples if has_interface else [],
            belief_available=has_interface,
            mode_counts=mode_counts if has_interface else None,
            mode_interface_available=has_interface,
            final_mode=last_mode if has_interface else "direct_velocity",
        )
        stat["belief_used_by_policy"] = int(has_interface)
        stat["mode_semantics_available"] = int(has_interface)
        if log_belief_diagnostics:
            # For direct/reactive baselines, these fields are diagnostic margins
            # computed from the shared environment geometry.  They are logged for
            # common x-axis binning in margin-phase plots, but
            # belief_available=False still records that the controller did not
            # consume or maintain a feasibility-belief state.
            for key, value in belief_row.items():
                stat[key] = value
            for key, value in final_belief_row.items():
                stat[f"final_{key}"] = value
            stat["diagnostic_margin_available"] = True
        # Explicit aliases for false-feasible outcome decomposition.  These use
        # the controller's actual episode modes, not inferred failure outcomes.
        if has_interface:
            stat["mode_commit_count"] = int(mode_counts.get("commit", 0))
            stat["mode_explore_count"] = int(mode_counts.get("explore", 0))
            stat["mode_recover_count"] = int(mode_counts.get("recover", 0))
            stat["mode_reject_count"] = int(mode_counts.get("reject", 0))
        else:
            stat["mode_commit_count"] = float("nan")
            stat["mode_explore_count"] = float("nan")
            stat["mode_recover_count"] = float("nan")
            stat["mode_reject_count"] = float("nan")
        all_stats.append(stat)

        # Update cross-episode memory
        if cross_mem is not None and entry_obs_ref:
            memory_audit = cross_mem.record_episode(
                entry_obs_ref[0],
                bool(stat["success"] > 0.5),
                stat["passage_width"],
                steps=stat["steps"],
                corridor_type=stat["corridor_type"],
                outcome=stat["episode_outcome"],
                attempted=bool(stat["attempted"]),
                env_step_count=int(stat["env_step_count"]),
                geometry_caused=bool(stat["geometry_caused"]),
                structural_margin=float(stat["structural_margin_gt"]),
                yaw_error=float(stat["entry_heading_error"]),
                morphology=env.morphology,
            )
            stat.update(memory_audit)

        if progress and (ep_idx % 100 == 0 or ep_idx < 5):
            sr = np.mean([s["success"] for s in all_stats])
            print(f"  [{method}] ep {ep_idx:3d} ctype={stat['corridor_type']:14s} "
                  f"success={stat['success']:.0f}  rolling_sr={sr:.3f}", flush=True)

    return all_stats


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_stats: List[dict], methods: List[str], ctypes: List[str],
                  multi_seed_std: Optional[Dict[str, float]] = None):
    print(f"\n{'Method':22s}  {'SR':>7}  {'CF-SR':>7}  {'Col':>6}  "
          f"{'Tight%':>7}  {'Overlap%':>8}  {'MinBM':>6}  ", end="")
    for ct in ctypes:
        short = ct[:6]
        print(f"  {short:>6}", end="")
    print()
    print("-" * (22 + 7 + 7 + 6 + 7 + 8 + 6 + 5 + len(ctypes) * 8))

    for method in methods:
        rows = [s for s in all_stats if s["method"] == method]
        if not rows:
            continue
        sr   = np.mean([s["success"] for s in rows])
        cfsr = np.mean([s["collision_free_success"] for s in rows])
        cr   = np.mean([s["collision"] for s in rows])
        tp   = np.mean([s["tight_passage_rate"] for s in rows])
        bo   = np.mean([s["body_overlap_rate"] for s in rows])
        bm   = np.mean([s["min_body_margin"] for s in rows])
        sr_str   = f"{sr:.3f}"
        cfsr_str = f"{cfsr:.3f}"
        if multi_seed_std and method in multi_seed_std:
            sr_str   += f"±{multi_seed_std[method]:.3f}"
        print(f"{method:22s}  {sr_str:>7}  {cfsr_str:>7}  {cr:.3f}  "
              f"{tp:.4f}  {bo:.5f}  {bm:+.3f}  ", end="")
        for ct in ctypes:
            grp = [s for s in rows if s["corridor_type"] == ct]
            dsr = np.mean([s["success"] for s in grp]) if grp else float("nan")
            print(f"  {dsr:.3f}", end="")
        print()


def write_csv(all_stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not all_stats:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_for_rows(all_stats), lineterminator="\n")
        w.writeheader()
        w.writerows(all_stats)
    print(f"[write] {path}")


def write_summary_rows(all_stats, methods, ctypes, path):
    """Write one CSV row per method for results_rl_summary.csv compatibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in methods:
        stats = [s for s in all_stats if s["method"] == method]
        if not stats:
            continue
        row = {
            "case": f"v2_{method}",
            "success_rate": round(np.mean([s["success"] for s in stats]), 4),
            # collision_free_success: primary metric — reached goal AND no wall penetration
            "collision_free_success_rate": round(
                np.mean([s["collision_free_success"] for s in stats]), 4),
            "collision_rate": round(np.mean([s["collision"] for s in stats]), 4),
            # tight_passage_rate: bm < 0.05 m episodes (near-risk, NOT physical collision)
            "tight_passage_rate": round(np.mean([s["tight_passage_rate"] for s in stats]), 4),
            # body_overlap_rate: steps with bm < 0 / total steps (allow_sliding wall penetration)
            "body_overlap_rate": round(np.mean([s["body_overlap_rate"] for s in stats]), 5),
            "avg_min_clearance": round(np.mean([s["min_body_margin"] for s in stats]), 4),
        }
        for ct in ctypes:
            grp = [s for s in stats if s["corridor_type"] == ct]
            row[f"sr_{ct}"] = round(np.mean([s["success"] for s in grp]), 4) if grp else ""
        rows.append(row)
    if rows:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"[write] {path}")


def _rate(rows: List[dict], key: str) -> float:
    if not rows:
        return float("nan")
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, 0.0)))
        except (TypeError, ValueError):
            vals.append(0.0)
    return float(np.mean(vals)) if vals else float("nan")


def _format_rate(value: float) -> str:
    if not np.isfinite(value):
        return "not run"
    return f"{100.0 * value:.1f}%"


def _format_latex_rate(value: float) -> str:
    return _format_rate(value).replace("%", r"\%")


def _outcome_value(row: dict, key: str) -> float:
    if key == "timeout_stuck":
        return float(
            _outcome_value(row, "timeout") > 0.5
            or _outcome_value(row, "stuck") > 0.5
        )
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _outcome_summary(rows: List[dict], key: str) -> dict:
    if not rows:
        return {"mean": float("nan"), "std": float("nan"), "count": 0, "total": 0}
    seeds = sorted({row.get("seed", "") for row in rows})
    seed_rates = []
    count = 0
    total = 0
    for seed in seeds:
        seed_rows = [row for row in rows if row.get("seed", "") == seed]
        values = [float(_outcome_value(row, key) > 0.5) for row in seed_rows]
        if not values:
            continue
        seed_rates.append(float(np.mean(values)))
        count += int(np.sum(values))
        total += len(values)
    return {
        "mean": float(np.mean(seed_rates)) if seed_rates else float("nan"),
        "std": float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else 0.0,
        "count": count,
        "total": total,
    }


def _format_summary(summary: dict) -> str:
    mean = float(summary.get("mean", float("nan")))
    if not np.isfinite(mean):
        return "not run"
    std = float(summary.get("std", 0.0) or 0.0)
    return (
        f"{100.0 * mean:.1f}±{100.0 * std:.1f}% "
        f"({summary.get('count', 0)}/{summary.get('total', 0)})"
    )


def _format_latex_summary(summary: dict) -> str:
    return _format_summary(summary).replace("%", r"\%").replace("±", r"$\pm$")


def _false_feasible_table_rows(all_stats: List[dict], methods: List[str]) -> List[dict]:
    rows = []
    for method in methods:
        subset = [
            row for row in all_stats
            if row.get("method") == method
            and str(row.get("corridor_type", "")).lower() == "false_feasible"
        ]
        if not subset:
            continue
        rows.append({
            "method": METHOD_LABELS.get(method, method),
            "episodes": len(subset),
            "success": _outcome_summary(subset, "success"),
            "reject": _outcome_summary(subset, "reject"),
            "correct_reject": _outcome_summary(subset, "correct_reject"),
            "collision": _outcome_summary(subset, "collision"),
            "near_collision": _outcome_summary(subset, "near_collision"),
            "timeout_stuck": _outcome_summary(subset, "timeout_stuck"),
            "wasted_attempt": _outcome_summary(subset, "wasted_attempt"),
        })
    return rows


def _false_feasible_markdown(rows: List[dict]) -> str:
    lines = [
        "# Table: False-Feasible Outcome Decomposition",
        "",
        "One-shot false-feasible traversal success does not establish correct rejection. This table separates explicit abstention from collision and non-collision execution failure.",
        "",
        "| Method | Episodes | Traversal success | Explicit reject | Correct reject | Collision | Near collision | Timeout/stuck | Wasted attempts |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {episodes} | {success} | {reject} | {correct_reject} | "
            "{collision} | {near_collision} | {timeout_stuck} | {wasted_attempt} |".format(
                method=row["method"],
                episodes=row["episodes"],
                success=_format_summary(row["success"]),
                reject=_format_summary(row["reject"]),
                correct_reject=_format_summary(row["correct_reject"]),
                collision=_format_summary(row["collision"]),
                near_collision=_format_summary(row["near_collision"]),
                timeout_stuck=_format_summary(row["timeout_stuck"]),
                wasted_attempt=_format_summary(row["wasted_attempt"]),
            )
        )
    lines.extend([
        "",
        "Definitions:",
        "- `explicit reject` means the controller selected Reject.",
        "- `correct_reject = explicit reject and passable_label == false`.",
        "- Timeout/stuck remains an execution failure category and is not labeled as safe rejection.",
        "- `wasted_attempt = attempted traversal on a false-feasible passage without correct rejection`.",
        "- `success == false` is never converted into correct rejection.",
        "- Collision and near-collision are logged independently and may overlap.",
    ])
    return "\n".join(lines)


def _false_feasible_latex(rows: List[dict]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{One-shot false-feasible traversal success does not establish correct rejection. This table separates explicit abstention from collision and non-collision execution failure.}",
        r"\label{tab:false-feasible-outcomes}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & Episodes & Traversal success & Explicit reject & Correct reject & Collision & Near collision & Timeout/stuck & Wasted attempts \\",
        r"\midrule",
    ]
    for row in rows:
        method = str(row["method"]).replace("_", r"\_")
        lines.append(
            f"{method} & {row['episodes']} & {_format_latex_summary(row['success'])} "
            f"& {_format_latex_summary(row['reject'])} "
            f"& {_format_latex_summary(row['correct_reject'])} "
            f"& {_format_latex_summary(row['collision'])} "
            f"& {_format_latex_summary(row['near_collision'])} "
            f"& {_format_latex_summary(row['timeout_stuck'])} "
            f"& {_format_latex_summary(row['wasted_attempt'])} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.25em}",
        r"\begin{minipage}{0.98\linewidth}",
        r"\footnotesize Explicit reject means the controller selected Reject; correct reject is explicit rejection on a benchmark-labeled infeasible passage. Timeout/stuck is not safe rejection. Collision and near-collision are logged independently and may overlap.",
        r"\end{minipage}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def write_false_feasible_outcome_tables(all_stats: List[dict], methods: List[str]):
    rows = _false_feasible_table_rows(all_stats, methods)
    if not rows:
        print("[warn] no false_feasible rows available for outcome table")
        return
    out_dir = RESULTS / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "paper_table_false_feasible_outcomes.md"
    tex_path = out_dir / "paper_table_false_feasible_outcomes.tex"
    md_path.write_text(_false_feasible_markdown(rows) + "\n")
    tex_path.write_text(_false_feasible_latex(rows) + "\n")
    print(f"[write] {md_path}")
    print(f"[write] {tex_path}")


def _path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _assert_columns(rows: List[dict], required: List[str], label: str) -> None:
    missing = [col for col in required if all(col not in row for row in rows)]
    if missing:
        raise RuntimeError(f"{label} columns missing: {missing}")


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _validate_paired_margin_rows(rows: List[dict], methods: List[str]) -> None:
    """Validate paired scenario semantics for margin-phase exports."""

    if not rows:
        raise RuntimeError("margin-phase validation received no rows")

    required_methods = set(methods)
    present_methods = {str(row.get("method", "")) for row in rows}
    missing_methods = sorted(required_methods - present_methods)
    if missing_methods:
        raise RuntimeError(f"missing methods in margin-phase rows: {missing_methods}")

    scenario_sets = {
        method: {
            str(row.get("scenario_id"))
            for row in rows
            if str(row.get("method", "")) == method
        }
        for method in methods
    }
    reference_method = methods[0]
    reference_set = scenario_sets[reference_method]
    for method, scenarios in scenario_sets.items():
        if scenarios != reference_set:
            missing = sorted(reference_set - scenarios)[:5]
            extra = sorted(scenarios - reference_set)[:5]
            raise RuntimeError(
                f"scenario_id mismatch for {method}: missing={missing}, extra={extra}"
            )

    rows_by_scenario: Dict[str, List[dict]] = {}
    for row in rows:
        rows_by_scenario.setdefault(str(row.get("scenario_id")), []).append(row)
    for scenario_id, scenario_rows in rows_by_scenario.items():
        methods_here = {str(row.get("method", "")) for row in scenario_rows}
        if not required_methods.issubset(methods_here):
            raise RuntimeError(
                f"scenario {scenario_id} does not contain all methods: {methods_here}"
            )
        corridor_types = {str(row.get("corridor_type", "")) for row in scenario_rows}
        if len(corridor_types) != 1:
            raise RuntimeError(
                f"scenario {scenario_id} has mismatched corridor_type values: {corridor_types}"
            )
        for key in ("passage_width", "entry_yaw", "lateral_offset"):
            values = [
                _float_or_nan(row.get(key))
                for row in scenario_rows
                if str(row.get("method", "")) in required_methods
            ]
            finite = [value for value in values if np.isfinite(value)]
            if len(finite) != len(required_methods):
                raise RuntimeError(f"scenario {scenario_id} has missing {key}: {values}")
            if max(finite) - min(finite) > 1e-6:
                raise RuntimeError(
                    f"scenario {scenario_id} has mismatched {key}: {finite}"
                )

    seen = set()
    for row in rows:
        key = (row.get("method"), row.get("seed"), row.get("episode_id"))
        if key in seen:
            raise RuntimeError(f"duplicate method+seed+episode_id row: {key}")
        seen.add(key)

    finite_delta_count = 0
    finite_margin_count = 0
    for row in rows:
        d_hat = _float_or_nan(row.get("d_hat"))
        w_req_cons = _float_or_nan(row.get("w_req_cons"))
        delta = _float_or_nan(row.get("delta_mean"))
        p_feas = _float_or_nan(row.get("p_feas"))
        if np.isfinite(delta):
            finite_delta_count += 1
        if np.isfinite(d_hat) and np.isfinite(w_req_cons) and np.isfinite(delta):
            finite_margin_count += 1
            if abs(delta - (d_hat - w_req_cons)) > 1e-6:
                raise RuntimeError(
                    "delta_mean does not match d_hat - w_req_cons for "
                    f"scenario={row.get('scenario_id')} method={row.get('method')}"
                )
        if np.isfinite(p_feas) and not (0.0 <= p_feas <= 1.0):
            raise RuntimeError(
                f"p_feas out of [0,1]: {p_feas} for row {row.get('episode_id')}"
            )

    min_expected = max(1, int(0.8 * len(rows)))
    if finite_delta_count < min_expected or finite_margin_count < min_expected:
        raise RuntimeError(
            "too few finite paired margin diagnostics: "
            f"delta={finite_delta_count}/{len(rows)}, "
            f"margin={finite_margin_count}/{len(rows)}"
        )


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def write_metadata(
    csv_path: Path,
    *,
    args: argparse.Namespace,
    methods: List[str],
    seed_values: List[int],
    rows: List[dict],
    ctypes: List[str],
) -> None:
    scenario_count = len({str(row.get("scenario_id", "")) for row in rows})
    belief_cfg = BeliefStateConfig()
    meta = {
        "git_commit": _git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(["python", *sys.argv]),
        "episode_semantics": "--episodes is per method per seed.",
        "methods": methods,
        "seeds": seed_values,
        "episodes_per_seed": int(args.episodes),
        "number_of_rows": len(rows),
        "scenario_count": scenario_count,
        "corridor_types": ctypes,
        "morphology_config": {
            "environment": "HarderNarrowPassageEnv",
            "width_range": list(args.width_range),
            "robot_radius": 0.18,
            "max_steps": int(args.max_steps) if args.max_steps is not None else 400,
            "entry_jitter_sigma": float(args.entry_jitter),
        },
        "belief_config": {
            "feature_dim": belief_cfg.feature_dim,
            "default_w_req_prior": belief_cfg.default_w_req_prior,
            "conservative_margin": belief_cfg.conservative_margin,
            "sigma_d": belief_cfg.sigma_d,
            "sigma_w": belief_cfg.sigma_w,
            "use_yaw_prior": belief_cfg.use_yaw_prior,
        },
        "margin_snapshot_definition": (
            "Primary delta_mean is computed at the canonical pre-behavior "
            "entry projection: the reset lateral offset is projected onto the "
            "corridor entry, reset yaw is preserved, and the first finite "
            "d_hat / w_req_cons diagnostic is used. If unavailable, the reset "
            "finite diagnostic is used as fallback."
        ),
    }
    meta_path = Path(csv_path).with_suffix(".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[write] {meta_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["rule_baseline", "geometry_fsm", "fsm_no_recovery",
                             "fsm_no_alignment", "fsm_local_memory", "fsm_cross_memory"])
    ap.add_argument("--variants", nargs="+", choices=CORE_VARIANTS, default=None,
                    help="Run core DEGNAV-Rule procedural ablations. When set, "
                         "this overrides --methods and uses the variant names as "
                         "the CSV method identifiers.")
    ap.add_argument("--corridor-types", "--ctypes", nargs="+", dest="corridor_types",
                    default=["straight", "l_shaped", "s_shaped",
                             "narrow_exit", "narrow_entry", "asymmetric", "false_feasible"])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="explicit seed list, e.g. --seeds 0 1 2. For backward "
                         "compatibility, a single value >1 is treated as a seed count.")
    ap.add_argument("--num-seeds", type=int, default=None,
                    help="number of independent seeds starting at --seed")
    ap.add_argument("--entry-jitter", type=float, default=0.0,
                    help="std dev (m) of lateral start-position noise — use 0.20–0.30 "
                         "to force RECOVER activations for ablation robustness testing")
    ap.add_argument("--width-range", nargs=2, type=float, default=(0.45, 0.90),
                    metavar=("MIN", "MAX"),
                    help="corridor width sampling range in meters. Use e.g. "
                         "--width-range 0.30 0.55 for an infeasible-side margin probe")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override HarderNarrowPassageEnv max_steps")
    ap.add_argument("--output-csv",
                    type=Path, default=RESULTS / "harder_benchmark_episodes.csv")
    ap.add_argument("--output-summary",
                    type=Path, default=None)
    ap.add_argument("--log-outcome-decomposition", action="store_true",
                    help="write false-feasible reject/collision/timeout/wasted-attempt tables")
    ap.add_argument("--log-belief-diagnostics", action="store_true",
                    help="validate and retain margin-phase belief diagnostic columns")
    args = ap.parse_args()

    ctypes = args.corridor_types
    methods = list(args.variants) if args.variants is not None else list(args.methods)
    if args.seeds is None:
        count = args.num_seeds or 1
        seed_values = [args.seed + i for i in range(count)]
    elif len(args.seeds) == 1 and args.num_seeds is None and args.seeds[0] > 1:
        # Previous versions used --seeds as a count.  Keep that command working:
        #   --seed 42 --seeds 3  ->  [42, 43, 44]
        seed_values = [args.seed + i for i in range(args.seeds[0])]
    else:
        seed_values = list(args.seeds)
    n_seeds = len(seed_values)
    jitter = args.entry_jitter
    output_summary = args.output_summary
    if output_summary is None:
        default_csv = RESULTS / "harder_benchmark_episodes.csv"
        if Path(args.output_csv) == default_csv:
            output_summary = RESULTS / "harder_benchmark_summary.csv"
        else:
            output_summary = Path(args.output_csv).with_name(
                f"{Path(args.output_csv).stem}_summary.csv"
            )

    # Collect per-seed stats for all methods
    per_seed_stats: Dict[str, List[List[dict]]] = {m: [] for m in methods}

    for seed_idx, run_seed in enumerate(seed_values):
        print(f"\n{'='*60}")
        print(f"Seed {run_seed}  ({seed_idx+1}/{n_seeds})"
              + (f"  entry_jitter={jitter:.2f}m" if jitter > 0 else ""))
        print(f"{'='*60}")

        for method in methods:
            print(f"\n--- {method} ---")
            stats = eval_method(method, ctypes, args.episodes, run_seed,
                                entry_jitter_sigma=jitter,
                                log_belief_diagnostics=args.log_belief_diagnostics,
                                width_range=tuple(args.width_range),
                                max_steps=args.max_steps)
            per_seed_stats[method].append(stats)
            for ct in ctypes:
                grp = [s for s in stats if s["corridor_type"] == ct]
                if grp:
                    dsr = np.mean([s["success"] for s in grp])
                    print(f"  {ct:14s}: SR={dsr:.3f}  n={len(grp)}")

    # Aggregate across seeds
    all_stats = []
    for method in methods:
        for seed_stats in per_seed_stats[method]:
            all_stats.extend(seed_stats)

    # Compute per-method SR std across seeds for the summary header
    multi_seed_std: Optional[Dict[str, float]] = None
    if n_seeds > 1:
        multi_seed_std = {}
        for method in methods:
            seed_srs = [
                np.mean([s["success"] for s in seed_stats])
                for seed_stats in per_seed_stats[method]
            ]
            multi_seed_std[method] = float(np.std(seed_srs))
        print(f"\n[multi-seed] SR std across {n_seeds} seeds:")
        for method, std in multi_seed_std.items():
            mean_sr = np.mean([s["success"]
                               for s in all_stats if s["method"] == method])
            print(f"  {method:22s}: {mean_sr:.3f} ± {std:.3f}")

    print_summary(all_stats, methods, ctypes, multi_seed_std)
    write_csv(all_stats, args.output_csv)
    write_summary_rows(all_stats, methods, ctypes, output_summary)
    if args.variants is not None:
        _assert_columns(all_stats, CORE_ABLATION_REQUIRED_COLUMNS, "core procedural ablation")
    if args.log_belief_diagnostics:
        _assert_columns(all_stats, MARGIN_PHASE_REQUIRED_COLUMNS, "margin-phase diagnostic")
        _validate_paired_margin_rows(all_stats, methods)
    if args.log_outcome_decomposition:
        _assert_columns(all_stats, FALSE_FEASIBLE_REQUIRED_COLUMNS, "false-feasible outcome")
        if set(str(ct).lower() for ct in ctypes) == {"false_feasible"}:
            if _path_is_under(Path(args.output_csv), RESULTS):
                write_false_feasible_outcome_tables(all_stats, methods)
            else:
                print("[info] output CSV is outside results/narrow_passage_rl; "
                      "skipping canonical false-feasible paper table update")
        else:
            print("[info] mixed corridor run: outcome decomposition columns were logged, "
                  "but false-feasible-only paper tables were not overwritten")
    write_metadata(
        Path(args.output_csv),
        args=args,
        methods=methods,
        seed_values=seed_values,
        rows=all_stats,
        ctypes=ctypes,
    )


if __name__ == "__main__":
    main()
