"""Decision modes and mode-conditioned control interface."""

from dataclasses import dataclass
from enum import Enum


class DecisionMode(str, Enum):
    COMMIT = "commit"
    EXPLORE = "explore"
    RECOVER = "recover"
    REJECT = "reject"


@dataclass
class ModeDecision:
    mode: DecisionMode
    p_pass: float
    risk: float
    reason: str


def choose_mode(p_pass: float, risk: float, stuck_score: float, memory_risk: float) -> ModeDecision:
    if memory_risk > 0.80:
        return ModeDecision(DecisionMode.REJECT, p_pass, risk, "high memory failure risk")
    if stuck_score > 0.80:
        return ModeDecision(DecisionMode.RECOVER, p_pass, risk, "stuck or oscillating")
    if risk < 0.30 and p_pass > 0.70:
        return ModeDecision(DecisionMode.COMMIT, p_pass, risk, "low risk")
    return ModeDecision(DecisionMode.EXPLORE, p_pass, risk, "uncertain but feasible")


def controller(mode: DecisionMode, local_goal_angle: float, clearance_margin: float) -> tuple[float, float]:
    """Return normalized `(linear_velocity, angular_velocity)` command."""
    if mode == DecisionMode.REJECT:
        return 0.0, 0.0
    if mode == DecisionMode.RECOVER:
        return -0.4, max(-1.0, min(1.0, local_goal_angle))
    if mode == DecisionMode.EXPLORE:
        return 0.25, max(-0.8, min(0.8, local_goal_angle + 0.5 * clearance_margin))
    return 0.7, max(-0.6, min(0.6, local_goal_angle))

