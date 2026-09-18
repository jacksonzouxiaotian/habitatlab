from __future__ import annotations

import numpy as np

from eagor_repro.controllers.stop_criteria import StopCriterion, robust_target_depth
from eagor_repro.evaluation.failure_attribution import classify_episode_failure
from eagor_repro.planning.candidate_heading_planner import CandidateHeadingPlanner
from eagor_repro.policies.oracle_direction_policy import OracleDirectionPolicy
from eagor_repro.spherical.spherical_grid import SphericalGrid, wrap_to_pi


def _oracle() -> OracleDirectionPolicy:
    return OracleDirectionPolicy(SphericalGrid(16, 32))


def test_oracle_direction_front_is_zero() -> None:
    prediction, distance, index = _oracle().predict_nearest(
        [0, 0, 0], np.eye(3), [[2, 0, 0], [5, 0, 0]]
    )
    assert np.isclose(prediction.azimuth, 0.0)
    assert np.isclose(distance, 2.0)
    assert index == 0
    assert prediction.confidence == 1.0


def test_oracle_direction_left_is_positive() -> None:
    prediction, _, _ = _oracle().predict_nearest(
        [0, 0, 0], np.eye(3), [[0, 2, 0]]
    )
    assert np.isclose(prediction.azimuth, np.pi / 2)


def test_oracle_direction_right_is_negative() -> None:
    prediction, _, _ = _oracle().predict_nearest(
        [0, 0, 0], np.eye(3), [[0, -2, 0]]
    )
    assert np.isclose(prediction.azimuth, -np.pi / 2)


def test_oracle_direction_changes_with_agent_yaw() -> None:
    # Body front points world-left after a +90 degree left yaw.  A world-front
    # target is therefore on the current body-right.
    rotation_world_from_body = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    prediction, _, _ = _oracle().predict_nearest(
        [0, 0, 0], rotation_world_from_body, [[2, 0, 0]]
    )
    assert np.isclose(prediction.azimuth, -np.pi / 2)


def _depth_stop() -> StopCriterion:
    return StopCriterion(
        mode="area_depth",
        confidence_threshold=0.25,
        stop_distance_m=1.0,
        min_target_pixels=2,
        min_valid_depth_pixels=2,
    )


def test_depth_stop_far_target_does_not_stop() -> None:
    assessment = _depth_stop().assess(
        target_visible=True,
        confidence=1.0,
        target_area_fraction=0.5,
        target_mask=np.ones((2, 2), bool),
        depth=np.full((2, 2), 2.0),
    )
    assert not assessment.should_stop
    assert np.isclose(assessment.target_depth_m, 2.0)


def test_depth_stop_near_target_stops() -> None:
    assessment = _depth_stop().assess(
        target_visible=True,
        confidence=1.0,
        target_area_fraction=0.01,
        target_mask=np.ones((2, 2), bool),
        depth=np.full((2, 2), 0.8),
    )
    assert assessment.should_stop
    assert assessment.reason == "perception_depth_stop"


def test_depth_stop_empty_mask_does_not_stop() -> None:
    assessment = _depth_stop().assess(
        target_visible=False,
        confidence=1.0,
        target_area_fraction=0.0,
        target_mask=np.zeros((2, 2), bool),
        depth=np.full((2, 2), 0.5),
    )
    assert not assessment.should_stop
    assert assessment.target_depth_m is None


def test_depth_stop_invalid_depth_does_not_stop() -> None:
    assessment = _depth_stop().assess(
        target_visible=True,
        confidence=1.0,
        target_area_fraction=0.5,
        target_mask=np.ones((2, 2), bool),
        depth=np.asarray([[np.nan, np.inf], [0.0, -1.0]]),
    )
    assert not assessment.should_stop
    assert assessment.target_depth_m is None


def test_depth_stop_partial_invalid_uses_robust_median() -> None:
    depth = np.asarray([[np.nan, 0.0], [0.6, 0.8]])
    value, count = robust_target_depth(depth, np.ones((2, 2), bool), 2)
    assert count == 2
    assert np.isclose(value, 0.7)
    assessment = _depth_stop().assess(
        target_visible=True,
        confidence=1.0,
        target_area_fraction=0.5,
        target_mask=np.ones((2, 2), bool),
        depth=depth,
    )
    assert assessment.should_stop


def test_oracle_stop_reuses_strict_habitat_success_distance() -> None:
    criterion = StopCriterion(mode="oracle_distance", success_distance_m=0.1)
    assert criterion.assess(
        target_visible=False,
        confidence=0.0,
        target_area_fraction=0.0,
        oracle_distance_m=0.09,
    ).should_stop
    assert not criterion.assess(
        target_visible=True,
        confidence=1.0,
        target_area_fraction=1.0,
        oracle_distance_m=0.1,
    ).should_stop


def _set_free_sector(
    depth: np.ndarray, heading_deg: float, half_width_deg: float = 10.0
) -> None:
    width = depth.shape[1]
    angles = (np.arange(width) + 0.5) / width * 360.0 - 180.0
    delta = (angles - heading_deg + 180.0) % 360.0 - 180.0
    depth[:, np.abs(delta) <= half_width_deg] = 2.0


def _planner() -> CandidateHeadingPlanner:
    return CandidateHeadingPlanner(
        mode="depth_traversability",
        sector_width_deg=5.0,
        vertical_band_deg=90.0,
        obstacle_distance_m=0.5,
        safety_distance_m=0.7,
    )


def test_traversability_target_direction_free() -> None:
    result = _planner().plan(0.0, np.full((16, 72), 2.0))
    assert np.isclose(result.selected_heading, 0.0)
    assert result.planner_status == "target_free"


def test_traversability_target_blocked_left_free() -> None:
    depth = np.full((16, 72), 0.3)
    _set_free_sector(depth, 45.0)
    result = _planner().plan(0.0, depth)
    assert result.selected_heading > 0.0
    assert result.planner_status == "detour"


def test_traversability_target_blocked_right_free() -> None:
    depth = np.full((16, 72), 0.3)
    _set_free_sector(depth, -45.0)
    result = _planner().plan(0.0, depth)
    assert result.selected_heading < 0.0
    assert result.planner_status == "detour"


def test_traversability_all_blocked_triggers_recovery() -> None:
    result = _planner().plan(0.0, np.full((16, 72), 0.3))
    assert result.recovery_triggered
    assert result.planner_status == "all_blocked_recovery"
    assert abs(result.selected_heading) >= np.deg2rad(45.0)


def test_traversability_wraps_around_pi() -> None:
    target = np.deg2rad(179.0)
    result = _planner().plan(target, np.full((16, 72), 2.0))
    assert -np.pi <= result.selected_heading < np.pi
    assert np.isclose(result.selected_heading, wrap_to_pi(target))


def test_oracle_nav_requires_and_uses_explicit_shortest_path_heading() -> None:
    planner = CandidateHeadingPlanner(mode="oracle_nav")
    try:
        planner.plan(0.0)
    except ValueError as error:
        assert "shortest-path heading" in str(error)
    else:
        raise AssertionError("oracle_nav accepted a missing GT path heading")
    result = planner.plan(0.3, oracle_heading=np.pi + 0.2)
    assert result.planner_status == "oracle_nav_shortest_path"
    assert np.isclose(result.selected_heading, wrap_to_pi(np.pi + 0.2))


def test_failure_attribution_prioritizes_false_stop() -> None:
    result = classify_episode_failure(
        success=False,
        stop_called=True,
        false_stop_count=1,
        steps=20,
        max_steps=300,
        direction_mae_deg=60.0,
        collision_count=3,
        blocked_step_count=3,
        planner_recovery_count=0,
        oscillation_count=0,
        target_switch_count=0,
        lost_target_step_count=0,
    )
    assert result["primary_failure_mode"] == "false_stop"
    assert "direction_error" in result["secondary_failure_modes"]
