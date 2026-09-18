"""Habitat-Lab 0.3.3 ObjectNav closed-loop evaluator."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eagor_repro.config import (
    make_backend,
    make_controller,
    make_grid,
    make_planner,
    make_policy,
    make_stop_criterion,
)
from eagor_repro.controllers.fixed_step_controller import ControllerDecision
from eagor_repro.evaluation.evaluator import ResultWriter
from eagor_repro.evaluation.failure_attribution import classify_episode_failure
from eagor_repro.evaluation.metrics import (
    DirectionMetricsAccumulator,
    angular_error_deg,
    navigation_summary,
)
from eagor_repro.evaluation.video_renderer import VideoStreamWriter, render_frame
from eagor_repro.sensors.panorama_sensor import (
    DEPTH_UUID,
    PERSPECTIVE_UUID,
    SEMANTIC_UUID,
    configure_equirectangular_sensors,
    extract_depth_panorama,
    extract_panorama,
    normalize_native_equirectangular,
)
from eagor_repro.spherical.rotation import (
    habitat_rotation_world_from_eagor_body,
    relative_view_rotation,
)
from eagor_repro.spherical.spherical_grid import direction_to_angles
from eagor_repro.policies.oracle_direction_policy import OracleDirectionPolicy


def _dataset_path(config: Any) -> Path:
    return Path(str(config.habitat.dataset_path).format(split=config.habitat.split))


def _scene_dataset_name(config: Any) -> str:
    explicit = str(config.habitat.get("dataset_name", "")).strip().lower()
    if explicit:
        return explicit
    descriptor = " ".join(
        (
            str(config.habitat.dataset_path),
            str(config.habitat.benchmark_config),
        )
    ).lower()
    for candidate in ("hm3d", "mp3d"):
        if candidate in descriptor:
            return candidate
    return "generic"


def _scene_glob(dataset_name: str) -> str:
    return {
        "hm3d": "hm3d/**/*.basis.glb",
        "mp3d": "mp3d/*/*.glb",
    }.get(dataset_name, "**/*.glb")


def _select_unique_scene_episodes(
    episodes: List[Any], limit: int
) -> List[Any]:
    selected: List[Any] = []
    seen_scenes = set()
    for episode in episodes:
        scene_id = str(getattr(episode, "scene_id", ""))
        if scene_id in seen_scenes:
            continue
        seen_scenes.add(scene_id)
        selected.append(episode)
        if len(selected) >= limit:
            break
    return selected


def _episode_output_key(scene_id: str, episode_id: str) -> str:
    """Create a filesystem key unique across content-scene episode files."""

    scene_name = Path(str(scene_id)).stem or "scene"
    raw_key = f"{scene_name}_{episode_id}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_key)


def habitat_preflight(config: Any) -> Dict[str, Any]:
    """Read-only check for runtime, ObjectNav episodes, and scene assets."""

    dataset_name = _scene_dataset_name(config)
    scene_glob = _scene_glob(dataset_name)
    report: Dict[str, Any] = {
        "dataset_path": str(_dataset_path(config)),
        "scenes_dir": str(config.habitat.scenes_dir),
        "scene_dataset": dataset_name,
        "scene_glob": scene_glob,
    }
    try:
        import habitat
        import habitat_sim

        report.update(
            {
                "habitat_version": habitat.__version__,
                "habitat_sim_version": habitat_sim.__version__,
                "native_equirectangular": hasattr(
                    habitat_sim, "EquirectangularSensorSpec"
                ),
            }
        )
    except ImportError as exc:
        report["error"] = f"Habitat runtime unavailable: {exc}"
        report["ready"] = False
        return report
    dataset = _dataset_path(config)
    scenes = Path(str(config.habitat.scenes_dir))
    report["dataset_exists"] = dataset.is_file()
    report["scenes_dir_exists"] = scenes.is_dir()
    report["scene_count"] = (
        sum(1 for _ in scenes.glob(scene_glob)) if scenes.is_dir() else 0
    )
    report[f"{dataset_name}_scene_count"] = report["scene_count"]
    report["ready"] = bool(
        report["native_equirectangular"]
        and report["dataset_exists"]
        and report["scene_count"] > 0
    )
    return report


def build_habitat_config(config: Any) -> Any:
    import habitat
    from habitat.config import read_write

    habitat_config = habitat.get_config(
        config_path=str(config.habitat.benchmark_config),
        overrides=[
            f'habitat.dataset.data_path="{str(config.habitat.dataset_path)}"',
            f'habitat.dataset.scenes_dir="{str(config.habitat.scenes_dir)}"',
            f"habitat.dataset.split={str(config.habitat.split)}",
            f"habitat.environment.max_episode_steps={int(config.evaluation.max_episode_steps)}",
            f"habitat.simulator.habitat_sim_v0.gpu_device_id={int(config.habitat.gpu_device_id)}",
        ],
    )
    with read_write(habitat_config):
        habitat_config.habitat.seed = int(config.seed)
        habitat_config.habitat.simulator.forward_step_size = float(
            config.controller.forward_step_m
        )
        habitat_config.habitat.simulator.turn_angle = int(
            round(float(config.controller.turn_step_deg))
        )
        habitat_config.habitat.task.measurements.success.success_distance = float(
            config.evaluation.success_distance_m
        )
        configure_equirectangular_sensors(
            habitat_config.habitat,
            height=int(config.panorama.height),
            width=int(config.panorama.width),
            sensor_height=float(config.panorama.sensor_height),
            include_semantic=True,
            include_depth=True,
            include_perspective=True,
        )
    return habitat_config


def _target_query(episode: Any) -> str:
    for attribute in ("object_category", "object_name", "goal_object_name"):
        value = getattr(episode, attribute, None)
        if value:
            return str(value)
    raise RuntimeError("Current ObjectNav episode has no target category")


def _goal_direction_and_distance(
    env: Any,
    rotation_world_from_body: np.ndarray,
    predicted_direction: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Return a valid ObjectNav-instance direction and nearest goal distance.

    ObjectNav accepts any instance of the requested category.  For directional
    evaluation, select the valid goal with minimum angular error to the current
    prediction instead of arbitrarily penalizing a visible instance because a
    different instance happens to be nearest in Euclidean distance.  This is an
    evaluation-only choice and is never passed to the policy or controller.
    """

    position = np.asarray(env.sim.get_agent_state().position, np.float64)
    goals = getattr(env.current_episode, "goals", ())
    if not goals:
        raise RuntimeError("ObjectNav episode contains no evaluation goals")
    deltas = [np.asarray(goal.position, np.float64) - position for goal in goals]
    distances = [float(np.linalg.norm(delta)) for delta in deltas]
    directions_body = []
    for delta, distance in zip(deltas, distances):
        direction_world = delta / max(distance, 1e-12)
        direction_body = rotation_world_from_body.T @ direction_world
        directions_body.append(
            direction_body / max(float(np.linalg.norm(direction_body)), 1e-12)
        )
    if predicted_direction is None:
        selected_index = int(np.argmin(distances))
    else:
        selected_index = int(
            np.argmin(
                [
                    angular_error_deg(predicted_direction, direction)
                    for direction in directions_body
                ]
            )
        )
    return directions_body[selected_index], min(distances)


def _goal_candidates(
    env: Any, rotation_world_from_body: np.ndarray
) -> tuple[List[np.ndarray], List[float]]:
    position = np.asarray(env.sim.get_agent_state().position, np.float64)
    goals = getattr(env.current_episode, "goals", ())
    if not goals:
        raise RuntimeError("ObjectNav episode contains no evaluation goals")
    directions: List[np.ndarray] = []
    distances: List[float] = []
    for goal in goals:
        delta = np.asarray(goal.position, np.float64) - position
        distance = float(np.linalg.norm(delta))
        direction_world = delta / max(distance, 1e-12)
        direction_body = rotation_world_from_body.T @ direction_world
        direction_body /= max(float(np.linalg.norm(direction_body)), 1e-12)
        directions.append(direction_body)
        distances.append(distance)
    return directions, distances


def _target_mask_from_observations(
    likelihood_result: Any, observations: Dict[str, Any]
) -> Optional[np.ndarray]:
    """Recover the Oracle semantic target mask for depth-aware diagnostics."""

    if SEMANTIC_UUID not in observations:
        return None
    semantic_ids: List[int] = []
    for detection in getattr(likelihood_result, "raw_detections", ()) or ():
        semantic_ids.extend(int(value) for value in detection.get("semantic_ids", ()))
    if not semantic_ids:
        return None
    semantic = np.asarray(
        normalize_native_equirectangular(observations[SEMANTIC_UUID])
    ).squeeze()
    return np.isin(semantic, semantic_ids)


def _task_distance_to_goal(env: Any, fallback: float) -> float:
    """Read Habitat's own geodesic/view-point distance measure."""

    value = env.get_metrics().get("distance_to_goal", fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _oracle_shortest_path_plan(
    env: Any, rotation_world_from_body: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return the next heading and selected endpoint of a navmesh path.

    The endpoint is also consumed by Habitat's discrete shortest-path follower.
    Merely steering toward the next polyline corner is not a valid action-space
    oracle: a fixed forward step can overshoot a corner and collide even though
    the polyline itself lies on the navmesh.
    """

    import habitat_sim

    destinations = [
        view_point.agent_state.position
        for goal in getattr(env.current_episode, "goals", ())
        for view_point in (getattr(goal, "view_points", ()) or ())
    ]
    if not destinations:
        destinations = [
            goal.position for goal in getattr(env.current_episode, "goals", ())
        ]
    if not destinations:
        raise RuntimeError("oracle_nav found no ObjectNav destinations")
    position = np.asarray(env.sim.get_agent_state().position, np.float64)
    path = habitat_sim.MultiGoalShortestPath()
    path.requested_start = position.astype(np.float32)
    path.requested_ends = np.asarray(destinations, np.float32)
    found = bool(env.sim.pathfinder.find_path(path))
    points = [np.asarray(point, np.float64) for point in path.points]
    if not found or not points:
        raise RuntimeError("Habitat navmesh found no path to any valid goal viewpoint")
    next_point = points[-1]
    for point in points[1:]:
        if float(np.linalg.norm(point - position)) > 0.05:
            next_point = point
            break
    delta_world = next_point - position
    direction_body = rotation_world_from_body.T @ delta_world
    horizontal = np.asarray([direction_body[0], direction_body[1], 0.0])
    if float(np.linalg.norm(horizontal)) <= 1e-12:
        heading = 0.0
    else:
        heading = direction_to_angles(horizontal)[0]
    return heading, np.asarray(points[-1], np.float64)


def _oracle_shortest_path_heading(
    env: Any, rotation_world_from_body: np.ndarray
) -> float:
    """Compatibility wrapper returning only the navmesh path heading."""

    return _oracle_shortest_path_plan(env, rotation_world_from_body)[0]


def _habitat_action_name(action: int) -> str:
    """Convert a Habitat-Sim discrete action id to the Env string action."""

    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    names = {
        HabitatSimActions.stop: "stop",
        HabitatSimActions.move_forward: "move_forward",
        HabitatSimActions.turn_left: "turn_left",
        HabitatSimActions.turn_right: "turn_right",
    }
    try:
        return names[int(action)]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Unsupported oracle follower action: {action!r}") from exc


def _agent_yaw(rotation_world_from_body: np.ndarray) -> float:
    forward = rotation_world_from_body[:, 0]
    return float(np.arctan2(-forward[0], -forward[2]))


def evaluate_objectnav(config: Any, policy_method: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run closed-loop ObjectNav with independently switchable attribution modules."""

    if str(config.get("planning", {}).get("mode", "direct")) == "online_frontier":
        from eagor_repro.evaluation.online_evaluator import evaluate_online

        return evaluate_online(config, policy_method or str(config.evaluation.policy_method))

    report = habitat_preflight(config)
    if not report.get("ready", False):
        raise RuntimeError(f"Habitat ObjectNav preflight failed: {report}")

    import habitat

    method = policy_method or str(config.evaluation.policy_method)
    grid = make_grid(config)
    policy = make_policy(method, config, grid)
    backend = make_backend(config)
    controller = make_controller(config)
    stop_criterion = make_stop_criterion(config)
    planner = make_planner(config)
    experiment_id = str(config.evaluation.get("experiment_id", "")).strip()
    output_root = Path(str(config.output.root))
    if experiment_id:
        output_root = output_root / experiment_id
    writer = ResultWriter(output_root / method)
    habitat_config = build_habitat_config(config)
    summaries: List[Dict[str, Any]] = []

    oracle_semantic = getattr(backend, "backend_name", "") in {
        "oracle_semantic",
        "semantic_id_oracle",
    }
    oracle_direction = isinstance(policy, OracleDirectionPolicy)
    oracle_upper_bound = bool(
        oracle_semantic
        or oracle_direction
        or stop_criterion.oracle_upper_bound
        or planner.oracle_upper_bound
    )

    dataset = habitat.make_dataset(
        id_dataset=habitat_config.habitat.dataset.type,
        config=habitat_config.habitat.dataset,
    )
    if bool(config.evaluation.get("unique_scenes", False)):
        dataset.episodes = _select_unique_scene_episodes(
            list(dataset.episodes), int(config.evaluation.num_episodes)
        )

    with habitat.Env(config=habitat_config, dataset=dataset) as env:
        oracle_follower = None
        if planner.mode == "oracle_nav":
            from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

            oracle_follower = ShortestPathFollower(
                env.sim,
                goal_radius=float(config.evaluation.success_distance_m),
                return_one_hot=False,
                stop_on_error=False,
            )
        count = min(
            int(config.evaluation.num_episodes),
            int(env.number_of_episodes or config.evaluation.num_episodes),
        )
        for episode_index in range(count):
            observations = env.reset()
            episode = env.current_episode
            episode_id = str(getattr(episode, "episode_id", episode_index))
            scene_id = str(getattr(episode, "scene_id", ""))
            episode_output_key = _episode_output_key(scene_id, episode_id)
            target_query = _target_query(episode)
            target_positions = [goal.position for goal in getattr(episode, "goals", ())]
            policy.reset()
            controller.reset()
            planner.reset()
            if hasattr(backend, "reset"):
                backend.reset()
            metrics = DirectionMetricsAccumulator(
                float(config.evaluation.seam_threshold_deg)
            )
            rows: List[Dict[str, Any]] = []
            video_writer = (
                VideoStreamWriter(
                    writer.video_dir / f"episode_{episode_output_key}.mp4",
                    int(config.evaluation.video_fps),
                )
                if bool(config.evaluation.save_video)
                else None
            )
            trajectory: List[np.ndarray] = []
            latency: List[float] = []
            path_length = 0.0
            collisions = 0
            blocked_step_count = 0
            planner_recovery_count = 0
            oscillation_count = 0
            target_switch_count = 0
            lost_target_step_count = 0
            target_seen = False
            entered_success_radius = False
            previous_rotation: Optional[np.ndarray] = None
            previous_goal_index: Optional[int] = None
            previous_turn_action: Optional[str] = None
            stop_called = False
            num_stop_attempts = 0
            distance_at_stop: Optional[float] = None
            target_visible_at_stop: Optional[bool] = None
            target_depth_at_stop: Optional[float] = None
            gt_distance = float("nan")

            for step in range(int(config.evaluation.max_episode_steps)):
                state = env.sim.get_agent_state()
                position = np.asarray(state.position, np.float64)
                rotation = habitat_rotation_world_from_eagor_body(state.rotation)
                relative_rotation = (
                    None
                    if previous_rotation is None
                    else relative_view_rotation(previous_rotation, rotation)
                )
                trajectory.append(position.copy())

                panorama = extract_panorama(observations)
                depth = (
                    extract_depth_panorama(observations)
                    if DEPTH_UUID in observations
                    else None
                )
                likelihood_result = backend.infer(
                    panorama,
                    target_query,
                    observations=observations,
                    simulator_state=env.sim,
                )
                target_mask = _target_mask_from_observations(
                    likelihood_result, observations
                )
                update_start = time.perf_counter()
                goal_directions, goal_distances = _goal_candidates(env, rotation)
                if oracle_direction:
                    prediction, gt_distance, selected_goal_index = policy.predict_nearest(
                        position, rotation, target_positions
                    )
                    gt_direction = prediction.direction_xyz
                else:
                    prediction = policy.update(
                        likelihood_result.likelihood,
                        likelihood_result.target_visible,
                        relative_rotation,
                    )
                    selected_goal_index = int(
                        np.argmin(
                            [
                                angular_error_deg(prediction.direction_xyz, direction)
                                for direction in goal_directions
                            ]
                        )
                    )
                    gt_direction = goal_directions[selected_goal_index]
                    gt_distance = goal_distances[selected_goal_index]

                if (
                    previous_goal_index is not None
                    and selected_goal_index != previous_goal_index
                ):
                    target_switch_count += 1
                previous_goal_index = selected_goal_index
                gt_azimuth, gt_elevation = direction_to_angles(gt_direction)
                step_metrics = metrics.add(
                    prediction.direction_xyz,
                    gt_direction,
                    prediction.confidence,
                    likelihood_result.target_visible,
                )
                target_area = float(np.mean(likelihood_result.likelihood >= 0.5))
                task_distance = _task_distance_to_goal(env, min(goal_distances))
                entered_success_radius = bool(
                    entered_success_radius
                    or task_distance < float(config.evaluation.success_distance_m)
                )
                stop_assessment = stop_criterion.assess(
                    target_visible=likelihood_result.target_visible,
                    confidence=prediction.confidence,
                    target_area_fraction=target_area,
                    target_mask=target_mask,
                    depth=depth,
                    oracle_distance_m=task_distance,
                )
                oracle_nav_destination = None
                if planner.mode == "oracle_nav":
                    oracle_nav_heading, oracle_nav_destination = (
                        _oracle_shortest_path_plan(env, rotation)
                    )
                else:
                    oracle_nav_heading = None
                local_heading = planner.plan(
                    prediction.azimuth, depth, oracle_heading=oracle_nav_heading
                )
                if local_heading.recovery_triggered:
                    planner_recovery_count += 1
                    blocked_step_count += 1

                if stop_assessment.should_stop:
                    decision = ControllerDecision(
                        "stop", stop_assessment.reason, local_heading.selected_heading
                    )
                    num_stop_attempts += 1
                    distance_at_stop = task_distance
                    target_visible_at_stop = bool(likelihood_result.target_visible)
                    target_depth_at_stop = stop_assessment.target_depth_m
                elif planner.mode == "oracle_nav":
                    if oracle_follower is None or oracle_nav_destination is None:
                        raise RuntimeError("oracle_nav follower was not initialized")
                    oracle_action = _habitat_action_name(
                        oracle_follower.get_next_action(oracle_nav_destination)
                    )
                    decision = ControllerDecision(
                        oracle_action,
                        "oracle_navmesh_discrete_follower",
                        local_heading.selected_heading,
                    )
                    if oracle_action == "stop":
                        num_stop_attempts += 1
                        distance_at_stop = task_distance
                        target_visible_at_stop = bool(
                            likelihood_result.target_visible
                        )
                        target_depth_at_stop = stop_assessment.target_depth_m
                else:
                    # Stop is handled by StopCriterion.  Passing target_visible=False
                    # disables only FixedStepController's legacy internal area stop.
                    decision = controller.act(
                        local_heading.selected_heading,
                        prediction.elevation,
                        prediction.confidence,
                        False,
                        0.0,
                    )

                update_ms = (time.perf_counter() - update_start) * 1000.0
                total_latency = float(likelihood_result.latency_ms + update_ms)
                latency.append(total_latency)
                target_seen = target_seen or bool(likelihood_result.target_visible)
                if target_seen and not likelihood_result.target_visible:
                    lost_target_step_count += 1
                if decision.action in ("turn_left", "turn_right"):
                    if previous_turn_action is not None and decision.action != previous_turn_action:
                        oscillation_count += 1
                    previous_turn_action = decision.action
                else:
                    previous_turn_action = None

                collision = False
                rows.append(
                    {
                        "episode_id": episode_id,
                        "scene_id": scene_id,
                        "step": step,
                        "target_query": target_query,
                        "agent_position_x": position[0],
                        "agent_position_y": position[1],
                        "agent_position_z": position[2],
                        "agent_yaw": _agent_yaw(rotation),
                        "gt_target_azimuth": gt_azimuth,
                        "gt_target_elevation": gt_elevation,
                        "pred_target_azimuth": prediction.azimuth,
                        "pred_target_elevation": prediction.elevation,
                        "angular_error_deg": step_metrics["angular_error_deg"],
                        "confidence": prediction.confidence,
                        "target_visible": likelihood_result.target_visible,
                        "target_area_fraction": target_area,
                        "target_depth_m": stop_assessment.target_depth_m,
                        "valid_target_depth_pixels": stop_assessment.valid_target_depth_pixels,
                        "distance_to_goal_m": task_distance,
                        "stop_mode": stop_criterion.mode,
                        "stop_requested": stop_assessment.should_stop,
                        "action": decision.action,
                        "action_reason": decision.reason,
                        "planning_mode": planner.mode,
                        "target_heading": local_heading.target_heading,
                        "selected_heading": local_heading.selected_heading,
                        "planner_status": local_heading.planner_status,
                        "planner_clearance_m": local_heading.clearance_m,
                        "blocked_candidates": local_heading.blocked_candidates,
                        "recovery_triggered": local_heading.recovery_triggered,
                        "collision": collision,
                        "likelihood_backend": likelihood_result.backend_name,
                        "policy_method": method,
                        "update_mode": str(config.sh_belief.update_mode),
                        "decode_mode": str(config.sh_belief.decode_mode),
                        "oracle_upper_bound": oracle_upper_bound,
                        "latency_ms": total_latency,
                    }
                )
                if video_writer is not None:
                    online_failure_signal = (
                        "stop_attempt"
                        if stop_assessment.should_stop
                        else (
                            "planner_no_feasible_heading"
                            if local_heading.recovery_triggered
                            else "none_so_far"
                        )
                    )
                    video_writer.append(
                        render_frame(
                            panorama,
                            likelihood_result.likelihood,
                            prediction.belief_heatmap,
                            prediction.direction_xyz,
                            gt_direction,
                            decision.action,
                            prediction.confidence,
                            step_metrics["angular_error_deg"],
                            trajectory,
                            perspective=observations.get(PERSPECTIVE_UUID),
                            planned_heading=local_heading.selected_heading,
                            metadata={
                                "scene": Path(scene_id).stem,
                                "episode": episode_id,
                                "step": step,
                                "target": target_query,
                                "policy": method,
                                "target_visible": likelihood_result.target_visible,
                                "action_reason": decision.reason,
                                "planning_mode": planner.mode,
                                "planner_status": local_heading.planner_status,
                                "target_heading_deg": np.rad2deg(local_heading.target_heading),
                                "selected_heading_deg": np.rad2deg(local_heading.selected_heading),
                                "clearance_m": local_heading.clearance_m,
                                "target_depth_m": stop_assessment.target_depth_m,
                                "stop_mode": stop_criterion.mode,
                                "failure_mode": online_failure_signal,
                                "oracle_upper_bound": oracle_upper_bound,
                            },
                        )
                    )

                previous_rotation = rotation
                observations = env.step(decision.action)
                stop_called = decision.action == "stop"
                new_position = np.asarray(env.sim.get_agent_state().position, np.float64)
                displacement = float(np.linalg.norm(new_position - position))
                path_length += displacement
                if decision.action == "move_forward":
                    collision = displacement < 1e-4
                    rows[-1]["collision"] = collision
                    collisions += int(collision)
                    if collision and not local_heading.recovery_triggered:
                        blocked_step_count += 1
                    if (
                        planner.mode != "oracle_nav"
                        and hasattr(controller, "notify_collision")
                    ):
                        controller.notify_collision(collision)
                if env.episode_over:
                    break

            if video_writer is not None:
                video_writer.close()
            habitat_metrics = env.get_metrics()
            success = bool(float(habitat_metrics.get("success", 0.0)))
            distance = float(habitat_metrics.get("distance_to_goal", gt_distance))
            entered_success_radius = bool(
                entered_success_radius
                or distance < float(config.evaluation.success_distance_m)
            )
            correct_stop = int(stop_called and success)
            false_stop = int(stop_called and not success)
            missed_stop = int(not stop_called and entered_success_radius)
            direction_summary = metrics.summary()
            attribution = classify_episode_failure(
                success=success,
                stop_called=stop_called,
                false_stop_count=false_stop,
                steps=len(rows),
                max_steps=int(config.evaluation.max_episode_steps),
                direction_mae_deg=float(direction_summary["mae_deg"]),
                collision_count=collisions,
                blocked_step_count=blocked_step_count,
                planner_recovery_count=planner_recovery_count,
                oscillation_count=oscillation_count,
                target_switch_count=target_switch_count,
                lost_target_step_count=lost_target_step_count,
                direction_failure_threshold_deg=float(
                    config.evaluation.get("direction_failure_threshold_deg", 45.0)
                ),
                blocked_step_threshold=int(
                    config.evaluation.get("blocked_step_threshold", 10)
                ),
                oscillation_threshold=int(
                    config.evaluation.get("oscillation_threshold", 8)
                ),
            )
            for row in rows:
                row["failure_mode"] = attribution["primary_failure_mode"]
            summary = {
                "episode_id": episode_id,
                "episode_output_key": episode_output_key,
                "experiment_id": experiment_id,
                "scene_id": scene_id,
                "target_query": target_query,
                "scene_dataset": _scene_dataset_name(config),
                "direction_gt_selection": (
                    "nearest_euclidean_valid_instance_for_oracle_control"
                    if oracle_direction
                    else "minimum_angular_error_over_valid_instances_for_evaluation_only"
                ),
                "policy_method": method,
                "planning_mode": planner.mode,
                "stop_mode": stop_criterion.mode,
                "likelihood_backend": getattr(backend, "backend_name", "unknown"),
                "oracle_semantic": oracle_semantic,
                "oracle_direction": oracle_direction,
                "oracle_stop": stop_criterion.oracle_upper_bound,
                "oracle_planner": planner.oracle_upper_bound,
                "oracle_upper_bound": oracle_upper_bound,
                "num_stop_attempts": num_stop_attempts,
                "correct_stop": correct_stop,
                "false_stop": false_stop,
                "missed_stop": missed_stop,
                "distance_at_stop_m": distance_at_stop,
                "target_visible_at_stop": target_visible_at_stop,
                "target_depth_at_stop_m": target_depth_at_stop,
                "blocked_step_count": blocked_step_count,
                "planner_recovery_count": planner_recovery_count,
                "oscillation_count": oscillation_count,
                "target_switch_count": target_switch_count,
                "lost_target_step_count": lost_target_step_count,
                **attribution,
                **direction_summary,
                **navigation_summary(
                    success=success,
                    spl=float(habitat_metrics.get("spl", 0.0)),
                    steps=len(rows),
                    path_length=path_length,
                    distance_to_success=distance,
                    collision_count=collisions,
                    stop_called=stop_called,
                    stop_correct=bool(correct_stop),
                    seam_crossing_success=success
                    and any(bool(item["is_seam"]) for item in metrics.rows),
                    latency_ms=latency,
                ),
            }
            writer.write_episode_csv(episode_output_key, rows)
            writer.write_summary(episode_output_key, summary)
            summaries.append(summary)
    writer.write_batch_summary(f"objectnav_{method}", summaries)
    return summaries
