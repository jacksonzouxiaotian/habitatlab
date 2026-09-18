"""Deterministic episode-level failure attribution rules.

The labels are diagnostic summaries, not learned causal estimates.  Priority is
assigned from terminal errors to increasingly indirect symptoms so that every
failed episode has one reproducible primary label and can retain other observed
problems as secondary labels.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def classify_episode_failure(
    *,
    success: bool,
    stop_called: bool,
    false_stop_count: int,
    steps: int,
    max_steps: int,
    direction_mae_deg: float,
    collision_count: int,
    blocked_step_count: int,
    planner_recovery_count: int,
    oscillation_count: int,
    target_switch_count: int,
    lost_target_step_count: int,
    direction_failure_threshold_deg: float = 45.0,
    blocked_step_threshold: int = 10,
    oscillation_threshold: int = 8,
) -> Dict[str, Any]:
    if success:
        return {
            "failure_mode": "none",
            "primary_failure_mode": "none",
            "secondary_failure_modes": [],
        }

    observed: list[str] = []
    if false_stop_count > 0:
        observed.append("false_stop")
    if planner_recovery_count >= blocked_step_threshold:
        observed.append("planner_no_feasible_heading")
    if blocked_step_count >= blocked_step_threshold:
        observed.append("blocked_path")
    if np.isfinite(direction_mae_deg) and direction_mae_deg >= direction_failure_threshold_deg:
        observed.append("direction_error")
    if oscillation_count >= oscillation_threshold:
        observed.append("oscillation")
    if target_switch_count > 0:
        observed.append("target_switch")
    if lost_target_step_count > 0:
        observed.append("lost_target")
    if collision_count > 0:
        observed.append("collision")
    if steps >= max_steps and not stop_called:
        observed.append("timeout")

    # Priority is causal/terminal first: a false STOP ends the episode; repeated
    # no-feasible-heading and blocking directly prevent progress; only then do
    # we attribute failure to direction quality or behavioral symptoms.
    priority = (
        "false_stop",
        "planner_no_feasible_heading",
        "blocked_path",
        "direction_error",
        "oscillation",
        "target_switch",
        "lost_target",
        "collision",
        "timeout",
    )
    primary = next((label for label in priority if label in observed), "timeout")
    secondary = [label for label in observed if label != primary]
    return {
        "failure_mode": primary,
        "primary_failure_mode": primary,
        "secondary_failure_modes": secondary,
    }
