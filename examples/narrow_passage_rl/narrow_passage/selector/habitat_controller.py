"""Shared Habitat controller for learned or teacher-selected DEGNAV modes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contract import Mode


_LIN_MIN, _LIN_MAX = -0.15, 0.35
_ANG_MIN, _ANG_MAX = -45.0, 45.0


def distance_to_goal(env: Any) -> float:
    agent = np.asarray(env.sim.get_agent_state().position, dtype=np.float32)
    goal = np.asarray(env.current_episode.goals[0].position, dtype=np.float32)
    return float(np.linalg.norm((goal - agent)[[0, 2]]))


def heading_error(env: Any) -> float:
    state = env.sim.get_agent_state()
    agent = np.asarray(state.position, dtype=np.float32)
    goal = np.asarray(env.current_episode.goals[0].position, dtype=np.float32)
    delta = goal - agent
    goal_yaw = math.atan2(-float(delta[0]), -float(delta[2]))
    rotation = state.rotation
    yaw = math.atan2(
        2.0 * (rotation.real * rotation.y + rotation.x * rotation.z),
        1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
    )
    error = goal_yaw - yaw
    return float((error + math.pi) % (2.0 * math.pi) - math.pi)


def _normalize_linear(value_mps: float) -> float:
    value = (value_mps - _LIN_MIN) / (_LIN_MAX - _LIN_MIN) * 2.0 - 1.0
    return float(np.clip(value, -1.0, 1.0))


def _normalize_angular(value_radps: float) -> float:
    value_degps = value_radps * 180.0 / math.pi
    value = (value_degps - _ANG_MIN) / (_ANG_MAX - _ANG_MIN) * 2.0 - 1.0
    return float(np.clip(value, -1.0, 1.0))


def _action(linear_mps: float, angular_radps: float) -> dict[str, Any]:
    return {
        "action": "velocity_control",
        "action_args": {
            "linear_velocity": _normalize_linear(linear_mps),
            "angular_velocity": _normalize_angular(angular_radps),
        },
    }


def stop_action() -> dict[str, Any]:
    return _action(0.0, 0.0)


def mode_action(
    mode: Mode | int,
    current_heading_error: float,
    lateral_offset: float,
) -> dict[str, Any] | None:
    """Map a high-level mode to the fixed low-level Habitat controller.

    ``None`` is returned for REJECT because rejection is a selector-level
    terminal decision and must not move the simulator.
    """

    mode = Mode(int(mode))
    if mode is Mode.REJECT:
        return None
    if mode is Mode.RECOVER:
        angular = float(np.clip(0.4 * current_heading_error, -0.8, 0.8))
        return _action(-0.12, angular)
    if mode is Mode.EXPLORE and abs(current_heading_error) > 0.7:
        angular = float(np.clip(1.8 * current_heading_error, -1.2, 1.2))
        return _action(0.0, angular)

    gain = 1.6 if mode is Mode.EXPLORE else 1.0
    angular = float(
        np.clip(
            gain * (0.9 * current_heading_error - 1.2 * lateral_offset),
            -1.2,
            1.2,
        )
    )
    linear = 0.08 if mode is Mode.EXPLORE else 0.20
    return _action(linear, angular)


def privileged_teacher_mode(
    features_19d: np.ndarray,
    current_heading_error: float,
    *,
    morphology_infeasible: bool = False,
    high_risk_threshold: float = 0.80,
    medium_risk_threshold: float = 0.45,
    explore_body_margin_threshold: float = 0.30,
) -> Mode:
    """Training-only teacher; its 19-D input must never reach the E2E actor.

    Besides pose error, a small body margin triggers EXPLORE as a cautious
    probing action.  The 0.30 m gate is shared with the existing narrow-area
    controller and prevents aligned MP3D anchors from collapsing supervision
    to three modes.
    """

    collision = float(features_19d[16]) > 0.5
    stuck = float(features_19d[15]) > high_risk_threshold
    if morphology_infeasible and (collision or stuck):
        return Mode.REJECT
    if collision or stuck:
        return Mode.RECOVER
    lateral = abs(float(features_19d[11]))
    body_margin = float(features_19d[9])
    if (
        abs(current_heading_error) > medium_risk_threshold
        or lateral > 0.25
        or body_margin < explore_body_margin_threshold
    ):
        return Mode.EXPLORE
    return Mode.COMMIT
