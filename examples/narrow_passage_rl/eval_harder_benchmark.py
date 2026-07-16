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

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import CorridorType, HarderNarrowPassageEnv
from failure_memory import FailureMemoryConfig, PassageFailureMemory
from cross_episode_memory import CrossEpisodeMemory, MemoryConfig, FSMMode
from narrow_passage.models.belief_state import BeliefState, BeliefStateConfig
from evaluation.logging_schema import fieldnames_for_rows, finalize_episode_row

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"

CORE_VARIANTS = [
    "full",
    "no_alignment",
    "no_recovery",
    "deterministic_margin",
    "no_yaw_prior",
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
}

PROB_FEAS_THRESHOLD = 0.70
RISK_THRESHOLD = 0.35
TAU_MARGIN = 0.05

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
    if "EXPLORE" in m or "ALIGN" in m or "FOLLOW" in m or "CORRIDOR" in m:
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

    belief = BeliefState.from_obs(
        obs,
        memory_risk=memory_risk,
        cfg=_belief_cfg_for_variant(variant),
    )
    risk = float(np.clip(1.0 - belief.p_feas + belief.memory_risk, 0.0, 1.0))
    return {
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

    # Only activate inside the corridor (body_margin at default=4.82 means approach zone).
    if body_margin > 0.5:
        return None

    # Sector asymmetry test: one side tight (corner), other side open
    min_side = min(d_ln, d_rn)
    max_side = max(d_ln, d_rn)
    asymmetry = max_side - min_side

    if asymmetry < 2.0 or min_side > 0.30:
        return None   # symmetric or no tight side — not a corner

    # At an L-turn corner, the forward ray (d_cn) hits the outer wall at roughly
    # the same depth as the tight side (d_cn ≈ min_side).  A false positive from
    # an oblique approach to a straight wall has d_cn >> min_side (forward is
    # open, only one lateral sensor is blocked).  Suppress FOLLOW_SPACE when
    # forward is clearly more open than the tight side.
    if d_cn > 1.5 * min_side:
        return None

    # Turn direction = toward the inner (short) side
    turn_dir = -1.0 if d_ln <= d_rn else 1.0

    # Progressive turn: zero while forward is open, ramps up as d_cn drops
    turn_blend = float(np.clip((0.65 - d_cn) / 0.65, 0.0, 1.0))
    wz = turn_dir * 0.65 * turn_blend

    return "FOLLOW_SPACE", _act(0.10, wz)


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
                 cross_mem: Optional[CrossEpisodeMemory] = None):
        self.variant = variant
        self.local_mem = local_mem
        self.cross_mem = cross_mem
        self._reset_turn()

    def reset(self):
        self._reset_turn()

    def _reset_turn(self, cooldown: int = 0):
        self._in_turn = False
        self._turn_dir = 0.0
        self._steps_in_turn = 0
        self._peak_abs_he = 0.0
        self._min_bm = float("inf")  # min body_margin seen during this turn
        self._cooldown = cooldown  # steps to suppress FOLLOW_SPACE after a turn

    def step(self, obs):
        """Returns (action, mode_str)."""
        if self._in_turn:
            return self._continue_turn(obs)

        # Cooldown: don't allow FOLLOW_SPACE to retrigger right after a committed
        # turn completes.  The robot needs a few steps to stabilise heading in the
        # new arm before we trust the asymmetry sensor again.
        if self._cooldown > 0:
            self._cooldown -= 1
            action, mode = fsm_action(obs, self.variant, self.local_mem, self.cross_mem)
            if mode == "FOLLOW_SPACE":
                # Override: apply centering correction instead of committing a new turn.
                he = float(obs[10])
                lat = float(obs[11])
                wz = float(np.clip(0.9 * he - 1.2 * lat, -0.80, 0.80))
                return _act(0.12, wz), "COMMIT"
            return action, mode

        action, mode = fsm_action(obs, self.variant, self.local_mem, self.cross_mem)
        if mode == "FOLLOW_SPACE":
            wz = float(action[1])
            self._in_turn = True
            self._turn_dir = 1.0 if wz >= 0 else -1.0
            self._steps_in_turn = 0
            self._peak_abs_he = abs(float(obs[10]))
        return action, mode

    def _continue_turn(self, obs):
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
            return fsm_action(obs, self.variant, self.local_mem, self.cross_mem)

        # Exit: open area (past the junction turn region).
        if body_margin > 0.5:
            self._reset_turn(cooldown=0)
            return fsm_action(obs, self.variant, self.local_mem, self.cross_mem)

        # Exit: timeout.
        if self._steps_in_turn > 60:
            self._reset_turn(cooldown=0)
            return fsm_action(obs, self.variant, self.local_mem, self.cross_mem)

        # Slow forward + committed turn.  Reduce top-tier vx from 0.08→0.06 to
        # avoid clipping the outer L-junction wall on wide-arm approaches.
        if body_margin < 0.10:
            vx = 0.04
        else:
            vx = 0.06
        wz = self._turn_dir * 0.60
        return _act(vx, wz), "FOLLOW_SPACE"


def fsm_action(obs, variant="full", local_mem: Optional[PassageFailureMemory] = None,
               cross_mem: Optional[CrossEpisodeMemory] = None,
               max_recover_steps=25, _env=None):
    """Geometry-FSM step.  Returns (action, mode_str)."""
    he = float(obs[10])
    lat = float(obs[11])
    stuck = float(obs[15])
    collision_flag = float(obs[16]) > 0.5

    dist = float(obs[12])
    if dist < 0.30:
        return _act(0.0, 0.0), "STOP"

    # FSM mode from cross-episode memory
    if cross_mem is not None:
        cm_mode = cross_mem.fsm_mode(obs)
        if cm_mode == FSMMode.REJECT:
            return _act(-0.10, 0.0), "REJECT"
        cautious = cm_mode == FSMMode.CAUTIOUS
    else:
        cautious = False

    speed_fwd = 0.15 if cautious else 0.20
    speed_explore = 0.05 if cautious else 0.08

    memory_risk = 1.0 if cautious else 0.0
    accept_feasible, _belief_row, _decision_rule = _feasibility_acceptance(
        obs,
        variant,
        memory_risk=memory_risk,
    )

    # FOLLOW_SPACE: sector-depth corner detection takes priority over ALIGN.
    # Only active for variants that use alignment (i.e. not no_alignment).
    fs = None if variant == "no_alignment" else _follow_space_mode(obs)
    if fs is not None:
        return fs[1], fs[0]

    # Local memory overrides
    if local_mem is not None and local_mem.should_recover(obs):
        mode = "RECOVER"
    elif collision_flag or stuck > 0.80:
        mode = "RECOVER" if variant not in ("no_recovery",) else "COMMIT"
    elif variant == "no_alignment":
        mode = "COMMIT_NOALIGN" if accept_feasible else "EXPLORE_NOALIGN"
    elif abs(he) > 0.7 and float(obs[9]) > 0.3:
        # Large heading error AND outside/wide corridor → full ALIGN (spin in place).
        mode = "ALIGN"
    elif abs(he) > 0.7:
        # Large heading error inside a NARROW corridor: spinning toward the final goal
        # would drive the robot into the wall (straight-line goal ≠ local path direction).
        # Instead use corridor-following: go slow and correct laterally only.
        mode = "CORRIDOR_FOLLOW"
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
        vx = vx_in_narrow if vx_in_narrow is not None else speed_explore
        action = _act(vx, wz)
    elif mode == "COMMIT_NOALIGN":
        vx = vx_in_narrow if vx_in_narrow is not None else speed_fwd
        action = _act(vx, 0.0)
    elif mode == "EXPLORE_NOALIGN":
        vx = vx_in_narrow if vx_in_narrow is not None else speed_explore
        action = _act(vx, 0.0)
    elif mode == "RECOVER":
        action = _act(-0.12, float(np.clip(he * 0.4, -0.8, 0.8)))
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
                entry_jitter_sigma: float = 0.0) -> dict:
    obs, _ = env.reset()

    # Lateral entry jitter: perturb start position perpendicular to heading.
    # Exposes RECOVER mode by forcing the robot off-center at episode start.
    if entry_jitter_sigma > 0.0:
        yaw = float(env.pose[2])
        lateral = float(np.random.normal(0.0, entry_jitter_sigma))
        env.pose[0] += lateral * (-math.cos(yaw))
        env.pose[1] += lateral * math.sin(yaw)
        obs = env._obs()

    entry_obs = obs.copy()
    entry_obs_ref.clear()
    entry_obs_ref.append(entry_obs)
    entry_yaw = float(env.pose[2])
    entry_lateral_offset = float(entry_obs[11])
    margin_row, margin_step, margin_source, margin_obs = _canonical_margin_snapshot(
        env,
        entry_obs,
        variant=variant,
    )

    if agent is not None:
        agent.reset()

    done = False
    steps = 0
    any_collision = False
    min_bm = float("inf")
    tight_passage = False   # bm < 0.05 m (near-risk, not a physical collision)
    body_overlap_steps = 0  # steps where bm < 0 (Habitat sliding through wall)
    mode_counts: Counter[str] = Counter()
    last_mode = ""
    belief_samples: List[dict] = [margin_row]

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

        last_mode = str(mode)
        mode_counts[_mode_bucket(last_mode)] += 1

        obs, _, term, trunc, info = env.step(action)
        post_bm = float(obs[9])
        if post_bm < min_bm:
            min_bm = post_bm
        if post_bm < 0.05:
            tight_passage = True
        if post_bm < 0.0:
            body_overlap_steps += 1
        if float(obs[16]) > 0.5 or float(info.get("collision", 0.0)) > 0.5:
            any_collision = True
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
    collision_free_success = float(success > 0.5 and body_overlap_steps == 0)
    final_belief = belief_samples[-1] if belief_samples else margin_row
    finite_deltas = [
        float(sample["delta_mean"])
        for sample in belief_samples
        if np.isfinite(float(sample.get("delta_mean", float("nan"))))
    ]

    return {
        "variant": variant,
        "success": success,
        "collision_free_success": collision_free_success,
        "collision": float(any_collision),
        # tight_passage_rate: bm < 0.05 m (near-risk; NOT a physical collision)
        "tight_passage_rate": float(tight_passage),
        # body_overlap_rate: fraction of steps with bm < 0 (wall penetration via allow_sliding)
        "body_overlap_rate": body_overlap_steps / max(steps, 1),
        "min_body_margin": min_bm,
        "steps": steps,
        "passable": passable_label,
        "passable_label": float(passable_label),
        "is_false_feasible": float(is_false_feasible),
        "corridor_type": info.get("corridor_type", "unknown"),
        "passage_width": info.get("passage_width", float(obs[8])),
        "entry_yaw": entry_yaw,
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
                log_belief_diagnostics: bool = False) -> List[dict]:
    env_cfg = {"corridor_types": ctypes, "seed": seed}
    env = HarderNarrowPassageEnv(env_cfg)
    variant = _variant_for_method(method)

    local_mem = None
    cross_mem = None
    agent = None

    if method == "fsm_local_memory":
        local_mem = PassageFailureMemory()
    elif method == "fsm_cross_memory":
        cross_mem = CrossEpisodeMemory()

    # Use TurnCommitFSM for all FSM variants (handles L/S turn commitment)
    if method != "rule_baseline":
        agent = TurnCommitFSM(variant=variant, local_mem=local_mem,
                              cross_mem=cross_mem)

    all_stats = []
    entry_obs_ref = []
    for ep_idx in range(n_episodes):
        if method == "fsm_local_memory" and local_mem is not None:
            local_mem.reset()

        stat = run_episode(env, method, local_mem, cross_mem,
                           max_steps=env.max_steps, entry_obs_ref=entry_obs_ref,
                           variant=variant,
                           agent=agent, entry_jitter_sigma=entry_jitter_sigma)
        belief_row = stat.pop("_belief_row", {})
        belief_samples = stat.pop("_belief_samples", [])
        final_belief_row = stat.pop("_final_belief_row", {})
        mode_counts = stat.pop("_mode_counts", {})
        last_mode = stat.pop("_last_mode", "unavailable")
        stat["episode_idx"] = ep_idx
        stat["scenario_id"] = f"seed{seed}_ep{ep_idx:06d}"
        stat["episode_id"] = f"{method}_{seed}_{ep_idx:06d}"
        stat["scene_id"] = "procedural_v2"
        stat["domain"] = "procedural_v2"
        stat["split"] = "procedural_v2"
        stat["seed"] = seed
        stat["method"] = method
        stat["variant"] = variant
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
            cross_mem.record_episode(
                entry_obs_ref[0],
                bool(stat["success"] > 0.5),
                stat["passage_width"],
                steps=stat["steps"],
                corridor_type=stat["corridor_type"],
            )

        if ep_idx % 100 == 0 or ep_idx < 5:
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
        timeout_stuck = float(np.mean([
            float(row.get("timeout", 0.0)) > 0.5 or float(row.get("stuck", 0.0)) > 0.5
            for row in subset
        ]))
        rows.append({
            "method": METHOD_LABELS.get(method, method),
            "episodes": len(subset),
            "success": _rate(subset, "success"),
            "correct_reject": _rate(subset, "correct_reject"),
            "false_reject": _rate(subset, "false_reject"),
            "collision": _rate(subset, "collision"),
            "near_collision": _rate(subset, "near_collision"),
            "timeout_stuck": timeout_stuck,
            "wasted_attempt": _rate(subset, "wasted_attempt"),
        })
    return rows


def _false_feasible_markdown(rows: List[dict]) -> str:
    lines = [
        "# Table: False-Feasible Outcome Decomposition",
        "",
        "0% traversal success is not equivalent to correct rejection; this table decomposes abstention and execution failure.",
        "",
        "| Method | Episodes | Traversal success | Correct reject | False reject | Collision | Near collision | Timeout/stuck | Wasted attempts |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {episodes} | {success} | {correct_reject} | {false_reject} | "
            "{collision} | {near_collision} | {timeout_stuck} | {wasted_attempt} |".format(
                method=row["method"],
                episodes=row["episodes"],
                success=_format_rate(row["success"]),
                correct_reject=_format_rate(row["correct_reject"]),
                false_reject=_format_rate(row["false_reject"]),
                collision=_format_rate(row["collision"]),
                near_collision=_format_rate(row["near_collision"]),
                timeout_stuck=_format_rate(row["timeout_stuck"]),
                wasted_attempt=_format_rate(row["wasted_attempt"]),
            )
        )
    lines.extend([
        "",
        "Definitions:",
        "- `correct_reject = reject and passable_label == false`.",
        "- `false_reject = reject and passable_label == true`.",
        "- `wasted_attempt = attempted execution on a false-feasible passage without correct rejection`.",
        "- `success == false` is never converted into correct rejection.",
    ])
    return "\n".join(lines)


def _false_feasible_latex(rows: List[dict]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & Episodes & Traversal success & Correct reject & False reject & Collision & Near collision & Timeout/stuck & Wasted attempts \\",
        r"\midrule",
    ]
    for row in rows:
        method = str(row["method"]).replace("_", r"\_")
        lines.append(
            f"{method} & {row['episodes']} & {_format_latex_rate(row['success'])} "
            f"& {_format_latex_rate(row['correct_reject'])} "
            f"& {_format_latex_rate(row['false_reject'])} "
            f"& {_format_latex_rate(row['collision'])} "
            f"& {_format_latex_rate(row['near_collision'])} "
            f"& {_format_latex_rate(row['timeout_stuck'])} "
            f"& {_format_latex_rate(row['wasted_attempt'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
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
            "width_range": [0.45, 0.90],
            "robot_radius": 0.18,
            "max_steps": 400,
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
                                log_belief_diagnostics=args.log_belief_diagnostics)
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
            write_false_feasible_outcome_tables(all_stats, methods)
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
