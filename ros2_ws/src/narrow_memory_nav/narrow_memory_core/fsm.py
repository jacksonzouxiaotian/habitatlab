"""Geometry-FSM and TurnCommitFSM — pure Python, no Habitat dependency.

Ported from examples/narrow_passage_rl/eval_harder_benchmark.py.
Observation vector layout (19-dim):
  [0-2]  d_ln, d_cn, d_rn  — near left/center/right depth (m)
  [3-5]  d_lf, d_cf, d_rf  — far  left/center/right depth (m)
  [6-7]  cl, cr             — clearance left/right beyond robot half-width (m)
  [8]    passage_width      — cl + cr (m)
  [9]    body_margin        — min(cl, cr) (m)
  [10]   heading_error      — signed angle to goal (rad, + = goal left of robot)
  [11]   lateral_offset     — cross-track error from start→goal line (m, + = left)
  [12]   dist_to_goal       — Euclidean distance (m)
  [13]   vx                 — current forward velocity (m/s)
  [14]   wz                 — current angular velocity (rad/s)
  [15]   stuck_score        — 0-1, 1 = commanded forward but not moving
  [16]   collision_flag     — 0 or 1
  [17]   prev_vx            — previous commanded vx
  [18]   prev_wz            — previous commanded wz
"""

import math
from typing import Optional, Tuple

import numpy as np


def _act(vx: float, wz: float) -> np.ndarray:
    return np.array([vx, wz], dtype=np.float32)


def _wz(obs: np.ndarray, gain: float = 1.0) -> float:
    he = float(obs[10])
    lat = float(obs[11])
    return float(np.clip((0.9 * he - 1.2 * lat) * gain, -0.8, 0.8))


def _follow_space_mode(obs: np.ndarray) -> Optional[Tuple[str, np.ndarray]]:
    """Detect L/S-shaped junction; return (mode, action) or None.

    Trigger condition: strong near-ray asymmetry while inside a narrow corridor.
    The tight side indicates the outer corner wall; the turn goes toward it.
    """
    d_ln, d_cn, d_rn = float(obs[0]), float(obs[1]), float(obs[2])
    body_margin = float(obs[9])

    if body_margin > 0.5:
        return None

    min_side = min(d_ln, d_rn)
    max_side = max(d_ln, d_rn)
    if max_side - min_side < 2.0 or min_side > 0.30:
        return None

    if d_cn > 1.5 * min_side:
        return None

    turn_dir = -1.0 if d_ln <= d_rn else 1.0
    turn_blend = float(np.clip((0.65 - d_cn) / 0.65, 0.0, 1.0))
    return "FOLLOW_SPACE", _act(0.10, turn_dir * 0.65 * turn_blend)


def fsm_action(
    obs: np.ndarray,
    variant: str = "full",
) -> Tuple[np.ndarray, str]:
    """Single Geometry-FSM step. Returns (action, mode_str)."""
    he = float(obs[10])
    lat = float(obs[11])
    stuck = float(obs[15])
    collision_flag = float(obs[16]) > 0.5
    dist = float(obs[12])

    if dist < 0.30:
        return _act(0.0, 0.0), "STOP"

    if variant == "no_alignment":
        return _act(0.20, float(np.clip(he * 0.9, -0.8, 0.8))), "COMMIT_NOALIGN"

    fs = _follow_space_mode(obs)
    if fs is not None:
        return fs[1], fs[0]

    if abs(he) > 0.7 and float(obs[9]) > 0.3:
        mode = "ALIGN"
    elif abs(he) > 0.7:
        mode = "CORRIDOR_FOLLOW"
    elif collision_flag or stuck > 0.80:
        mode = "RECOVER" if variant != "no_recovery" else "COMMIT"
    elif abs(he) > 0.45 or abs(lat) > 0.25:
        mode = "EXPLORE"
    else:
        mode = "COMMIT"

    in_narrow = float(obs[9]) < 0.30
    if in_narrow:
        cl_v, cr_v = float(obs[6]), float(obs[7])
        d_ln_v, d_cn_v, d_rn_v = float(obs[0]), float(obs[1]), float(obs[2])
        wz_center = float(np.clip(
            -1.6 * (cl_v - cr_v) / max(cl_v + cr_v, 0.05), -0.8, 0.8
        ))
        if d_cn_v < 0.5 * (d_ln_v + d_rn_v):
            wz_gap = float(np.clip(
                1.2 * (d_rn_v - d_ln_v) / max(d_rn_v + d_ln_v, 0.1), -0.8, 0.8
            ))
        else:
            wz_gap = 0.0
        wz_narrow = float(np.clip(wz_center + wz_gap, -0.8, 0.8))
    else:
        wz_narrow = None

    vx_in_narrow = -0.06 if (in_narrow and float(obs[1]) < 0.20) else None

    if mode == "ALIGN":
        action = _act(0.0, float(np.clip(he * 1.8, -0.8, 0.8)))
    elif mode == "CORRIDOR_FOLLOW":
        wz_cf = wz_narrow if in_narrow else float(np.clip(-2.0 * lat, -0.8, 0.8))
        vx_cf = vx_in_narrow if vx_in_narrow is not None else 0.08
        action = _act(vx_cf, wz_cf)
    elif mode == "COMMIT":
        wz = wz_narrow if in_narrow else _wz(obs, 1.0)
        vx = vx_in_narrow if vx_in_narrow is not None else 0.20
        action = _act(vx, wz)
    elif mode == "EXPLORE":
        wz = wz_narrow if in_narrow else _wz(obs, 1.6)
        vx = vx_in_narrow if vx_in_narrow is not None else 0.08
        action = _act(vx, wz)
    elif mode == "RECOVER":
        action = _act(-0.12, float(np.clip(he * 0.4, -0.8, 0.8)))
    else:
        action = _act(0.0, 0.0)

    return action, mode


class TurnCommitFSM:
    """Stateful FSM that commits to a junction turn until the new corridor is entered.

    Wraps fsm_action with a state machine that locks the turn direction once
    FOLLOW_SPACE fires, preventing oscillation at L/S-shaped junctions.
    """

    def __init__(self, variant: str = "full"):
        self.variant = variant
        self._reset_turn()

    def reset(self) -> None:
        self._reset_turn()

    def _reset_turn(self, cooldown: int = 0) -> None:
        self._in_turn = False
        self._turn_dir = 0.0
        self._steps_in_turn = 0
        self._peak_abs_he = 0.0
        self._cooldown = cooldown

    def step(self, obs: np.ndarray) -> Tuple[np.ndarray, str]:
        """Returns (action, mode_str)."""
        if self._in_turn:
            return self._continue_turn(obs)

        if self._cooldown > 0:
            self._cooldown -= 1
            action, mode = fsm_action(obs, self.variant)
            if mode == "FOLLOW_SPACE":
                he = float(obs[10])
                lat = float(obs[11])
                wz = float(np.clip(0.9 * he - 1.2 * lat, -0.80, 0.80))
                return _act(0.12, wz), "COMMIT"
            return action, mode

        action, mode = fsm_action(obs, self.variant)
        if mode == "FOLLOW_SPACE":
            self._in_turn = True
            self._turn_dir = 1.0 if float(action[1]) >= 0 else -1.0
            self._steps_in_turn = 0
            self._peak_abs_he = abs(float(obs[10]))
        return action, mode

    def _continue_turn(self, obs: np.ndarray) -> Tuple[np.ndarray, str]:
        he = float(obs[10])
        body_margin = float(obs[9])
        self._steps_in_turn += 1
        self._peak_abs_he = max(self._peak_abs_he, abs(he))

        if self._peak_abs_he > math.pi / 6 and abs(he) < 0.35:
            self._reset_turn(cooldown=8)
            return fsm_action(obs, self.variant)

        if body_margin > 0.5:
            self._reset_turn(cooldown=0)
            return fsm_action(obs, self.variant)

        if self._steps_in_turn > 60:
            self._reset_turn(cooldown=0)
            return fsm_action(obs, self.variant)

        vx = 0.04 if body_margin < 0.10 else 0.06
        return _act(vx, self._turn_dir * 0.60), "FOLLOW_SPACE"
