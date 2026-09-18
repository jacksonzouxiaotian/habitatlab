from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from eagor_repro.evaluation.habitat_evaluator import (
    _episode_output_key,
    _goal_direction_and_distance,
    _habitat_action_name,
    _select_unique_scene_episodes,
)
from eagor_repro.evaluation.video_renderer import render_frame


def test_unique_scene_episode_selection() -> None:
    episodes = [
        SimpleNamespace(scene_id="scene_a", episode_id="0"),
        SimpleNamespace(scene_id="scene_a", episode_id="1"),
        SimpleNamespace(scene_id="scene_b", episode_id="2"),
        SimpleNamespace(scene_id="scene_c", episode_id="3"),
    ]
    selected = _select_unique_scene_episodes(episodes, 2)
    assert [episode.episode_id for episode in selected] == ["0", "2"]


def test_episode_output_key_includes_scene_name() -> None:
    first = _episode_output_key("data/scene_a/scene_a.glb", "0")
    second = _episode_output_key("data/scene_b/scene_b.glb", "0")
    assert first == "scene_a_0"
    assert second == "scene_b_0"
    assert first != second


def test_habitat_oracle_action_ids_map_to_env_actions() -> None:
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    assert _habitat_action_name(HabitatSimActions.stop) == "stop"
    assert _habitat_action_name(HabitatSimActions.move_forward) == "move_forward"
    assert _habitat_action_name(HabitatSimActions.turn_left) == "turn_left"
    assert _habitat_action_name(HabitatSimActions.turn_right) == "turn_right"


def test_objectnav_direction_gt_accepts_any_valid_instance() -> None:
    goals = [
        SimpleNamespace(position=[4.0, 0.0, 0.0]),
        SimpleNamespace(position=[0.0, 4.0, 0.0]),
    ]
    env = SimpleNamespace(
        sim=SimpleNamespace(
            get_agent_state=lambda: SimpleNamespace(position=[0.0, 0.0, 0.0])
        ),
        current_episode=SimpleNamespace(goals=goals),
    )
    direction, distance = _goal_direction_and_distance(
        env,
        np.eye(3),
        predicted_direction=np.asarray([0.0, 1.0, 0.0]),
    )
    assert np.allclose(direction, [0.0, 1.0, 0.0])
    assert np.isclose(distance, 4.0)


def test_objectnav_video_dashboard_shape() -> None:
    frame = render_frame(
        panorama=np.zeros((32, 64, 3), np.uint8),
        likelihood=np.zeros((32, 64), np.float32),
        belief=np.zeros((32, 64), np.float32),
        predicted_direction=np.asarray([1.0, 0.0, 0.0]),
        ground_truth_direction=np.asarray([1.0, 0.0, 0.0]),
        action="move_forward",
        confidence=0.5,
        angular_error=0.0,
        perspective=np.zeros((32, 64, 3), np.uint8),
    )
    assert frame.shape == (720, 1280, 3)
    assert frame.dtype == np.uint8
