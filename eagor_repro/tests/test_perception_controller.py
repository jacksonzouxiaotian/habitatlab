from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eagor_repro.controllers.fixed_step_controller import FixedStepController
from eagor_repro.perception.oracle_semantic_backend import (
    OracleSemanticBackend,
    SemanticIdOracleBackend,
)
from eagor_repro.perception.qwen_grounding_backend import QwenGroundingBackend
from eagor_repro.perception.recorded_likelihood_backend import RecordedLikelihoodBackend
from eagor_repro.sensors.panorama_sensor import normalize_native_equirectangular


@dataclass
class _Category:
    value: str

    def name(self) -> str:
        return self.value


@dataclass
class _Object:
    semantic_id: int
    category: _Category
    id: str = "object_7"


@dataclass
class _Scene:
    objects: list[_Object]


@dataclass
class _Simulator:
    semantic_scene: _Scene


def test_native_erp_horizontal_normalization() -> None:
    image = np.arange(8).reshape(2, 4)
    normalized = normalize_native_equirectangular(image)
    assert np.array_equal(normalized, image[:, ::-1])


def test_oracle_semantic_keeps_seam_circular() -> None:
    semantic = np.zeros((16, 32), np.int32)
    semantic[7:9, 0] = 7
    backend = OracleSemanticBackend(
        sigma_azimuth_deg=10.0, sigma_elevation_deg=5.0
    )
    result = backend.infer(
        np.zeros((16, 32, 3), np.uint8),
        "chair",
        observations={"equirect_semantic_sensor": semantic},
        simulator_state=_Simulator(_Scene([_Object(7, _Category("chair"))])),
    )
    assert result.target_visible
    assert result.metadata["debug_oracle"]
    assert result.likelihood[:, 0].max() > 0.1
    assert result.likelihood[:, -1].max() > 0.1


def test_semantic_id_oracle_applies_reliable_pixel_threshold() -> None:
    semantic = np.zeros((8, 16), np.int32)
    semantic[3, 0] = 82
    backend = SemanticIdOracleBackend(
        {"television": [82, 87]},
        min_visible_pixels=2,
        sigma_azimuth_deg=10.0,
    )
    too_small = backend.infer(
        np.zeros((8, 16, 3), np.uint8),
        "television",
        observations={"equirect_semantic_sensor": semantic},
    )
    assert not too_small.target_visible
    assert np.allclose(too_small.likelihood, 1e-6)

    semantic[3, -1] = 87
    visible = backend.infer(
        np.zeros((8, 16, 3), np.uint8),
        "television",
        observations={"equirect_semantic_sensor": semantic},
    )
    assert visible.target_visible
    assert visible.raw_detections[0]["pixel_count"] == 2
    assert visible.metadata["explicit_dataset_semantic_ids"]
    assert visible.likelihood[:, 0].max() > 0.1
    assert visible.likelihood[:, -1].max() > 0.1


def test_recorded_likelihood_sequence() -> None:
    sequence = np.stack(
        [np.zeros((4, 8), np.float32), np.ones((4, 8), np.float32)]
    )
    backend = RecordedLikelihoodBackend(sequence)
    first = backend.infer(np.zeros((4, 8, 3)), "chair")
    second = backend.infer(np.zeros((4, 8, 3)), "chair")
    assert not first.target_visible
    assert second.target_visible


def test_qwen_seam_gaussian_is_circular_and_uncalibrated() -> None:
    backend = QwenGroundingBackend(
        inference_callable=lambda panorama, query: '[[0, 8]]',
        sigma_azimuth_deg=12,
    )
    result = backend.infer(np.zeros((16, 32, 3), np.uint8), "chair")
    assert result.target_visible
    assert not result.confidence_calibrated
    assert result.likelihood[8, 0] > 0.9
    assert result.likelihood[8, -1] > 0.5


def test_fixed_step_controller_actions() -> None:
    controller = FixedStepController(
        turn_threshold_deg=15, confidence_move_threshold=0.25
    )
    assert controller.act(0, 0, 0.1, False).action == "turn_left"
    assert controller.act(np.deg2rad(30), 0, 0.8, True).action == "turn_left"
    assert controller.act(np.deg2rad(-30), 0, 0.8, True).action == "turn_right"
    assert controller.act(0, 0, 0.8, True).action == "move_forward"
    assert controller.act(0, 0, 0.8, True, 0.2).action == "stop"


def test_fixed_step_controller_collision_recovery_exploration() -> None:
    controller = FixedStepController(
        confidence_move_threshold=0.25,
        low_confidence_action="explore_forward",
        exploration_turn_steps=2,
    )
    first = controller.act(0, 0, 0.0, False)
    assert first.action == "move_forward"
    assert first.reason == "target_independent_exploration"
    controller.notify_collision(True)
    assert controller.act(0, 0, 0.0, False).action == "turn_right"
    assert controller.act(0, 0, 0.0, False).action == "turn_right"
    assert controller.act(0, 0, 0.0, False).action == "move_forward"
    controller.notify_collision(True)
    assert controller.act(0, 0, 0.0, False).action == "turn_left"
