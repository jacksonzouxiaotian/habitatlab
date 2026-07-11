#!/usr/bin/env python3

from typing import Any

from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.narrow_passage.geometry import features_to_dict


class _NarrowPassageMeasure(Measure):
    def __init__(self, *args: Any, sim, config, **kwargs: Any) -> None:
        self._sim = sim
        self._config = config
        super().__init__(*args, sim=sim, config=config, **kwargs)

    def _features(self, observations):
        if observations is None or "narrow_passage_features" not in observations:
            return {}
        return features_to_dict(observations["narrow_passage_features"])


@registry.register_measure
class NarrowPassageSuccess(_NarrowPassageMeasure):
    cls_uuid: str = "narrow_passage_success"

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return NarrowPassageSuccess.cls_uuid

    def reset_metric(self, *args: Any, episode, task, observations, **kwargs):
        self.update_metric(
            *args, episode=episode, task=task, observations=observations, **kwargs
        )

    def update_metric(self, *args: Any, episode, task, observations, **kwargs):
        f = self._features(observations)
        threshold = float(getattr(self._config, "success_distance", 0.35))
        aligned = abs(f.get("heading_error", 0.0)) < float(
            getattr(self._config, "heading_threshold", 0.35)
        )
        centered = abs(f.get("lateral_offset", 0.0)) < float(
            getattr(self._config, "lateral_threshold", 0.25)
        )
        require_stop = bool(getattr(self._config, "require_stop", False))
        stop_ok = (not require_stop) or bool(getattr(task, "is_stop_called", False))
        self._metric = float(
            f.get("distance_to_local_goal", 1e9) <= threshold
            and aligned
            and centered
            and stop_ok
        )


@registry.register_measure
class NarrowPassageCollision(_NarrowPassageMeasure):
    cls_uuid: str = "narrow_passage_collision"

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return NarrowPassageCollision.cls_uuid

    def reset_metric(self, *args: Any, episode, task, observations, **kwargs):
        self.update_metric(
            *args, episode=episode, task=task, observations=observations, **kwargs
        )

    def update_metric(self, *args: Any, episode, task, observations, **kwargs):
        f = self._features(observations)
        collided = float(f.get("collision_flag", 0.0))
        if observations is not None and "collided" in observations:
            collided = max(collided, float(observations["collided"]))
        self._metric = collided


@registry.register_measure
class NarrowPassageStuck(_NarrowPassageMeasure):
    cls_uuid: str = "narrow_passage_stuck"

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return NarrowPassageStuck.cls_uuid

    def reset_metric(self, *args: Any, episode, task, observations, **kwargs):
        self.update_metric(
            *args, episode=episode, task=task, observations=observations, **kwargs
        )

    def update_metric(self, *args: Any, episode, task, observations, **kwargs):
        f = self._features(observations)
        threshold = float(getattr(self._config, "stuck_threshold", 0.7))
        self._metric = float(f.get("stuck_score", 0.0) >= threshold)
