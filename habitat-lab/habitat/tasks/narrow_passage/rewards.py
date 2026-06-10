#!/usr/bin/env python3

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
        super().__init__(*args, sim=sim, config=config, **kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return NarrowPassageReward.cls_uuid

    def reset_metric(self, *args: Any, episode, task, observations, **kwargs):
        f = self._features(observations)
        self._prev_distance = f.get("distance_to_local_goal", None)
        self._prev_action_vx = f.get("previous_action_vx", 0.0)
        self._prev_action_wz = f.get("previous_action_wz", 0.0)
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

        min_clearance = min(
            f.get("clearance_left", 0.0), f.get("clearance_right", 0.0)
        )
        action_delta = abs(f.get("previous_action_vx", 0.0) - self._prev_action_vx)
        action_delta += abs(f.get("previous_action_wz", 0.0) - self._prev_action_wz)

        reward = 0.0
        reward += float(getattr(self._config, "progress_weight", 2.0)) * progress
        reward -= float(getattr(self._config, "center_weight", 0.5)) * abs(
            f.get("lateral_offset", 0.0)
        )
        reward -= float(getattr(self._config, "alignment_weight", 0.3)) * abs(
            f.get("heading_error", 0.0)
        )
        reward += float(getattr(self._config, "clearance_weight", 0.2)) * min_clearance
        reward -= float(getattr(self._config, "stuck_penalty", 5.0)) * f.get(
            "stuck_score", 0.0
        )
        reward -= float(getattr(self._config, "collision_penalty", 10.0)) * f.get(
            "collision_flag", 0.0
        )
        reward -= float(getattr(self._config, "oscillation_weight", 0.1)) * action_delta
        reward -= float(getattr(self._config, "slack_penalty", 0.01))

        success = 0.0
        task.measurements.check_measure_dependencies(
            self.uuid, ["narrow_passage_success"]
        )
        if "narrow_passage_success" in task.measurements.measures:
            success = task.measurements.measures[
                "narrow_passage_success"
            ].get_metric()
        reward += float(getattr(self._config, "success_reward", 10.0)) * success

        self._prev_distance = distance
        self._prev_action_vx = f.get("previous_action_vx", 0.0)
        self._prev_action_wz = f.get("previous_action_wz", 0.0)
        self._metric = reward
