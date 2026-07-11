#!/usr/bin/env python3

import math
from typing import Any

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.narrow_passage.geometry import features_to_dict


@registry.register_measure
class NarrowPassageReward(Measure):
    cls_uuid: str = "narrow_passage_reward"

    def __init__(self, *args: Any, sim, config, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        self._prev_distance = None
        self._prev_action_vx = 0.0
        self._prev_action_wz = 0.0
        self._reward_terms = {}
        super().__init__(*args, sim=sim, config=config, **kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return NarrowPassageReward.cls_uuid

    def reset_metric(self, *args: Any, episode, task, observations, **kwargs):
        f = self._features(observations)
        self._prev_distance = f.get("distance_to_local_goal", None)
        self._prev_action_vx = f.get("previous_action_vx", 0.0)
        self._prev_action_wz = f.get("previous_action_wz", 0.0)
        self._reward_terms = {}
        self._metric = 0.0

    def _features(self, observations):
        if observations is None or "narrow_passage_features" not in observations:
            return {}
        return features_to_dict(observations["narrow_passage_features"])

    def update_metric(self, *args: Any, episode, task, observations, **kwargs):
        f = self._features(observations)
        distance = f.get("distance_to_local_goal", 0.0)
        progress = 0.0
        if self._prev_distance is not None:
            progress = self._prev_distance - distance

        terms = self._dense_terms(progress, f, success=0.0)

        success = 0.0
        task.measurements.check_measure_dependencies(
            self.uuid, ["narrow_passage_success"]
        )
        if "narrow_passage_success" in task.measurements.measures:
            success = task.measurements.measures[
                "narrow_passage_success"
            ].get_metric()
        terms["reward/success"] = float(
            getattr(self._config, "success_reward", 100.0)
        ) * success
        reward = float(sum(terms.values()))

        self._prev_distance = distance
        self._prev_action_vx = f.get("previous_action_vx", 0.0)
        self._prev_action_wz = f.get("previous_action_wz", 0.0)
        self._reward_terms = terms
        setattr(task, "narrow_passage_reward_terms", dict(terms))
        self._metric = reward

    def _dense_terms(self, progress, f, success):
        left = f.get("clearance_left", 0.0)
        right = f.get("clearance_right", 0.0)
        passage_width = f.get("passage_width", 0.0)
        body_margin = f.get("body_margin", 0.0)
        heading_error = abs(f.get("heading_error", 0.0))
        current_wz = f.get("current_wz", 0.0)
        action_delta = abs(f.get("current_vx", 0.0) - self._prev_action_vx)
        action_delta += abs(current_wz - self._prev_action_wz)
        oscillating = (
            abs(current_wz) > 0.05
            and abs(self._prev_action_wz) > 0.05
            and math.copysign(1.0, current_wz)
            != math.copysign(1.0, self._prev_action_wz)
        )
        clearance_balance = min(left, right) - 0.5 * abs(left - right)
        safe_body_margin = float(getattr(self._config, "safe_body_margin", 0.12))
        safe_width = float(getattr(self._config, "safe_passage_width", 0.48))
        unsafe_deficit = max(0.0, safe_body_margin - body_margin)
        unsafe_deficit += max(0.0, safe_width - passage_width)

        return {
            "reward/progress": float(getattr(self._config, "progress_weight", 4.0))
            * progress,
            "reward/heading_alignment": float(
                getattr(self._config, "alignment_weight", 0.5)
            )
            * math.cos(heading_error),
            "reward/lateral_centering": -float(
                getattr(self._config, "center_weight", 0.15)
            )
            * abs(f.get("lateral_offset", 0.0)),
            "reward/clearance": float(getattr(self._config, "clearance_weight", 0.1))
            * clearance_balance,
            "reward/body_margin": float(
                getattr(self._config, "body_margin_weight", 0.5)
            )
            * min(body_margin, safe_body_margin),
            "reward/collision": -float(
                getattr(self._config, "collision_penalty", 3.0)
            )
            * f.get("collision_flag", 0.0),
            "reward/stuck": -float(getattr(self._config, "stuck_penalty", 1.5))
            * f.get("stuck_score", 0.0),
            "reward/oscillation": -float(
                getattr(self._config, "oscillation_weight", 0.0)
            )
            * float(oscillating),
            "reward/action_smoothness": -float(
                getattr(self._config, "action_smoothness_weight", 0.05)
            )
            * action_delta,
            "reward/unsafe_margin": -float(
                getattr(self._config, "unsafe_margin_penalty", 2.0)
            )
            * unsafe_deficit,
            "reward/success": float(getattr(self._config, "success_reward", 100.0))
            * success,
            "reward/slack": -float(getattr(self._config, "slack_penalty", 0.003)),
        }

    def get_reward_terms(self):
        return dict(self._reward_terms)
