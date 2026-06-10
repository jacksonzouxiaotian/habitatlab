#!/usr/bin/env python3

from typing import Any, Optional

import numpy as np
from gym import spaces

from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.tasks.narrow_passage.geometry import (
    FEATURE_DIM,
    MEMORY_FEATURE_DIM,
    NarrowPassageMemoryState,
    NarrowPassageState,
    depth_to_passage_features,
)


@registry.register_sensor
class NarrowPassageGeometrySensor(Sensor):
    cls_uuid: str = "narrow_passage_features"

    def __init__(
        self, sim: Simulator, config, *args: Any, task=None, **kwargs: Any
    ) -> None:
        self._sim = sim
        self._task = task
        self._max_depth = float(getattr(config, "max_depth", 5.0))
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.finfo(np.float32).max,
            high=np.finfo(np.float32).max,
            shape=(FEATURE_DIM,),
            dtype=np.float32,
        )

    def _state_from_task(self) -> NarrowPassageState:
        if self._task is not None and hasattr(
            self._task, "get_narrow_passage_state"
        ):
            return self._task.get_narrow_passage_state()
        return NarrowPassageState()

    def get_observation(self, observations, *args: Any, **kwargs: Any):
        depth: Optional[np.ndarray] = None
        if observations is not None:
            depth = observations.get("depth", None)
        return depth_to_passage_features(
            depth, state=self._state_from_task(), max_depth=self._max_depth
        )


@registry.register_sensor
class NarrowPassageMemorySensor(Sensor):
    cls_uuid: str = "narrow_passage_memory"

    def __init__(
        self, sim: Simulator, config, *args: Any, task=None, **kwargs: Any
    ) -> None:
        self._sim = sim
        self._task = task
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(MEMORY_FEATURE_DIM,),
            dtype=np.float32,
        )

    def get_observation(self, observations, *args: Any, **kwargs: Any):
        if self._task is not None and hasattr(
            self._task, "get_narrow_passage_memory_state"
        ):
            return self._task.get_narrow_passage_memory_state().as_array()
        return NarrowPassageMemoryState().as_array()
