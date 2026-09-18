"""Paper-style fixed-step controller for Habitat discrete navigation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eagor_repro.spherical.spherical_grid import wrap_to_pi


@dataclass(frozen=True)
class ControllerDecision:
    action: str
    reason: str
    yaw_error_rad: float


class FixedStepController:
    def __init__(
        self,
        turn_threshold_deg: float = 15.0,
        confidence_move_threshold: float = 0.25,
        low_confidence_action: str = "turn_left",
        stop_likelihood_area_threshold: float = 0.08,
        exploration_turn_steps: int = 6,
    ) -> None:
        if low_confidence_action not in (
            "turn_left",
            "follow_entropy",
            "explore_forward",
        ):
            raise ValueError(
                "low_confidence_action must be turn_left, follow_entropy, "
                "or explore_forward"
            )
        if exploration_turn_steps < 1:
            raise ValueError("exploration_turn_steps must be positive")
        self.turn_threshold_rad = np.deg2rad(turn_threshold_deg)
        self.confidence_move_threshold = float(confidence_move_threshold)
        self.low_confidence_action = low_confidence_action
        self.stop_likelihood_area_threshold = float(stop_likelihood_area_threshold)
        self.exploration_turn_steps = int(exploration_turn_steps)
        self.search_steps = 0
        self.collision_events = 0
        self.recovery_turns_remaining = 0
        self.recovery_action = "turn_right"

    def reset(self) -> None:
        self.search_steps = 0
        self.collision_events = 0
        self.recovery_turns_remaining = 0
        self.recovery_action = "turn_right"

    def notify_collision(self, collided: bool) -> None:
        """Schedule a deterministic turn after a blocked forward action.

        This is used only by the optional target-independent exploration mode.
        Alternating turn direction avoids repeatedly tracing the same short arc.
        """

        if not collided or self.low_confidence_action != "explore_forward":
            return
        self.collision_events += 1
        self.recovery_turns_remaining = self.exploration_turn_steps
        self.recovery_action = (
            "turn_right" if self.collision_events % 2 else "turn_left"
        )

    def act(
        self,
        predicted_azimuth: float,
        predicted_elevation: float,
        confidence: float,
        target_visible: bool,
        target_area_fraction: float = 0.0,
    ) -> ControllerDecision:
        yaw_error = float(wrap_to_pi(predicted_azimuth))
        if (
            target_visible
            and confidence >= self.confidence_move_threshold
            and target_area_fraction >= self.stop_likelihood_area_threshold
        ):
            return ControllerDecision("stop", "perception_area_stop", yaw_error)
        if self.recovery_turns_remaining > 0:
            self.recovery_turns_remaining -= 1
            return ControllerDecision(
                self.recovery_action, "collision_recovery", yaw_error
            )
        if confidence < self.confidence_move_threshold:
            self.search_steps += 1
            if self.low_confidence_action == "explore_forward":
                action = "move_forward"
            elif self.low_confidence_action == "follow_entropy":
                # The current direction is the least committal useful cue.  A
                # fixed left scan is used only when it is effectively forward.
                action = "turn_left" if abs(yaw_error) < 1e-6 else (
                    "turn_left" if yaw_error > 0 else "turn_right"
                )
            else:
                action = "turn_left"
            reason = (
                "target_independent_exploration"
                if action == "move_forward"
                else "low_confidence_search"
            )
            return ControllerDecision(action, reason, yaw_error)
        if yaw_error > self.turn_threshold_rad:
            return ControllerDecision("turn_left", "align_left", yaw_error)
        if yaw_error < -self.turn_threshold_rad:
            return ControllerDecision("turn_right", "align_right", yaw_error)
        return ControllerDecision("move_forward", "aligned", yaw_error)
