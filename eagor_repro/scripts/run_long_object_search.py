#!/usr/bin/env python3
"""Render a long-horizon semantic object-search episode in ReplicaCAD.

The online loop has two explicit phases:

1. target-independent coverage patrol while the requested object is absent;
2. EAGOR SH-belief direction tracking and fixed-step approach after detection.

ReplicaCAD semantic IDs provide Oracle perception, so this is a debugging upper
bound and must not be reported as zero-shot ObjectNav.  Neither exploration nor
stopping reads an object coordinate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from eagor_repro.controllers.fixed_step_controller import FixedStepController
from eagor_repro.evaluation.video_renderer import (
    VideoStreamWriter,
    render_object_search_frame,
)
from eagor_repro.perception.oracle_semantic_backend import SemanticIdOracleBackend
from eagor_repro.policies.eagor_policy import EAGORPolicy
from eagor_repro.sensors.panorama_sensor import normalize_native_equirectangular
from eagor_repro.spherical.belief_filter import SphericalHarmonicBeliefFilter
from eagor_repro.spherical.rotation import (
    habitat_rotation_world_from_eagor_body,
    relative_view_rotation,
)
from eagor_repro.spherical.spherical_grid import SphericalGrid


REPLICA_CAD_ROOT = Path("data_old/versioned_data/replica_cad_dataset")
TARGET_QUERY = "television"
TARGET_SEMANTIC_IDS = (82, 87)  # TV body + screen in ReplicaCAD metadata.

# This fixed coverage route is intentionally independent of the target pose.
# It covers the north living area, west corridor, south room, and finally the
# upper level.  Habitat snaps every point to the navmesh.
START_POSITION = (-2.0, 0.0, 4.0)
PATROL_POSITIONS = (
    (0.45, 0.0, 7.5),
    (-2.0, 0.0, 0.0),
    (-2.0, 0.0, -4.0),
    (1.5, 1.7, -4.0),
    (4.0, 0.0, 7.5),
)


def _shortest_path(
    habitat_sim: Any,
    pathfinder: Any,
    start: np.ndarray,
    end: np.ndarray,
) -> Tuple[float, List[np.ndarray]]:
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, np.float32)
    path.requested_end = np.asarray(end, np.float32)
    if not pathfinder.find_path(path):
        return float("inf"), []
    return float(path.geodesic_distance), [
        np.asarray(point, np.float64) for point in path.points
    ]


def _body_azimuth_to_point(
    position: np.ndarray,
    rotation_world_from_body: np.ndarray,
    point: np.ndarray,
) -> float:
    delta = np.asarray(point, np.float64) - np.asarray(position, np.float64)
    direction_body = rotation_world_from_body.T @ delta
    return float(np.arctan2(direction_body[1], direction_body[0]))


def _patrol_action(
    habitat_sim: Any,
    simulator: Any,
    position: np.ndarray,
    rotation_world_from_body: np.ndarray,
    waypoints: Sequence[np.ndarray],
    waypoint_index: int,
    arrival_radius: float,
    turn_threshold_deg: float,
) -> Tuple[str, int, Optional[np.ndarray], float]:
    """Follow navmesh shortest-path corners using only the patrol waypoint."""

    while waypoint_index < len(waypoints):
        distance, points = _shortest_path(
            habitat_sim,
            simulator.pathfinder,
            position,
            waypoints[waypoint_index],
        )
        if distance <= arrival_radius:
            waypoint_index += 1
            continue
        if not points:
            return "turn_left", waypoint_index, waypoints[waypoint_index], distance
        local_goal = points[-1]
        for point in points[1:]:
            if np.linalg.norm((point - position)[[0, 2]]) > arrival_radius:
                local_goal = point
                break
        azimuth = _body_azimuth_to_point(
            position, rotation_world_from_body, local_goal
        )
        threshold = np.deg2rad(turn_threshold_deg)
        if azimuth > threshold:
            action = "turn_left"
        elif azimuth < -threshold:
            action = "turn_right"
        else:
            action = "move_forward"
        return action, waypoint_index, waypoints[waypoint_index], distance
    # Coverage exhausted without a detection: keep a deterministic panoramic
    # scan instead of steering toward any evaluation-only target coordinate.
    return "turn_left", waypoint_index, None, 0.0


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_episode(
    output_root: Path,
    scene: str = "apt_0",
    max_steps: int = 450,
    width: int = 512,
    height: int = 256,
    fps: int = 10,
    move_amount: float = 0.18,
    turn_amount_deg: float = 10.0,
    min_visible_pixels: int = 2000,
    stop_area_fraction: float = 0.035,
) -> Dict[str, Any]:
    import habitat_sim
    import quaternion
    from habitat_sim.utils.settings import default_sim_settings, make_cfg

    dataset_config = REPLICA_CAD_ROOT / "replicaCAD.scene_dataset_config.json"
    scene_instance = REPLICA_CAD_ROOT / "configs/scenes" / f"{scene}.scene_instance.json"
    if not dataset_config.is_file() or not scene_instance.is_file():
        raise FileNotFoundError(
            "ReplicaCAD assets are missing; expected "
            f"{dataset_config} and {scene_instance}"
        )

    settings = default_sim_settings.copy()
    settings.update(
        {
            "scene": scene,
            "scene_dataset_config_file": str(dataset_config),
            "width": int(width),
            "height": int(height),
            "color_sensor": True,
            "semantic_sensor": True,
            "equirect_rgba_sensor": True,
            "equirect_semantic_sensor": True,
            "sensor_height": 0.88,
            "agent_radius": 0.20,
            "enable_physics": False,
        }
    )
    sim_config = make_cfg(settings)
    for action_name in ("turn_left", "turn_right"):
        sim_config.agents[0].action_space[action_name].actuation.amount = float(
            turn_amount_deg
        )
    sim_config.agents[0].action_space["move_forward"].actuation.amount = float(
        move_amount
    )

    output_root.mkdir(parents=True, exist_ok=True)
    video_path = output_root / f"{scene}_{TARGET_QUERY}_long_search.mp4"
    csv_path = output_root / f"{scene}_{TARGET_QUERY}_steps.csv"
    json_path = output_root / f"{scene}_{TARGET_QUERY}_summary.json"
    simulator = habitat_sim.Simulator(sim_config)
    writer: Optional[VideoStreamWriter] = None
    try:
        if not simulator.pathfinder.is_loaded:
            raise RuntimeError(f"ReplicaCAD navmesh did not load for scene {scene}")
        # The bundled ReplicaCAD navmesh is stage-centric.  Rebuild it with
        # rigid static furniture so the camera cannot cut through a sofa, TV,
        # or cabinet during the approach video.
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_radius = 0.20
        navmesh_settings.agent_height = 1.50
        navmesh_settings.include_static_objects = True
        if not simulator.recompute_navmesh(
            simulator.pathfinder, navmesh_settings
        ):
            raise RuntimeError("Failed to rebuild furniture-aware ReplicaCAD navmesh")
        agent = simulator.get_agent(0)
        start = np.asarray(
            simulator.pathfinder.snap_point(np.asarray(START_POSITION, np.float32)),
            np.float64,
        )
        waypoints = [
            np.asarray(
                simulator.pathfinder.snap_point(np.asarray(point, np.float32)),
                np.float64,
            )
            for point in PATROL_POSITIONS
        ]
        if not np.isfinite(start).all() or any(
            not np.isfinite(point).all() for point in waypoints
        ):
            raise RuntimeError("One or more curated patrol points cannot be snapped")
        state = agent.get_state()
        state.position = start
        # Face away from the first route corner to include a genuine
        # initial reorientation in the discrete-action sequence.
        state.rotation = quaternion.from_rotation_vector(
            [0.0, np.deg2rad(145.0), 0.0]
        )
        agent.set_state(state)

        grid = SphericalGrid(height, width)
        policy = EAGORPolicy(
            SphericalHarmonicBeliefFilter(
                grid,
                bandlimit=7,
                update_mode="paper",
                decode_mode="probability",
                missing_observation_mode="propagate_only",
            )
        )
        approach_controller = FixedStepController(
            turn_threshold_deg=turn_amount_deg,
            confidence_move_threshold=0.20,
            stop_likelihood_area_threshold=stop_area_fraction,
        )
        backend = SemanticIdOracleBackend(
            {TARGET_QUERY: TARGET_SEMANTIC_IDS},
            sigma_azimuth_deg=5.0,
            sigma_elevation_deg=5.0,
            min_visible_pixels=min_visible_pixels,
        )

        bounds_raw = simulator.pathfinder.get_bounds()
        world_bounds = (
            np.asarray(bounds_raw[0], np.float64),
            np.asarray(bounds_raw[1], np.float64),
        )
        route_distances = []
        route_start = start
        for waypoint in waypoints:
            distance, _ = _shortest_path(
                habitat_sim, simulator.pathfinder, route_start, waypoint
            )
            route_distances.append(distance)
            route_start = waypoint

        writer = VideoStreamWriter(video_path, fps=fps)
        observations = simulator.get_sensor_observations()
        previous_rotation: Optional[np.ndarray] = None
        previous_position: Optional[np.ndarray] = None
        phase = "explore"
        first_detection_step: Optional[int] = None
        waypoint_index = 0
        trajectory: List[np.ndarray] = []
        rows: List[Dict[str, Any]] = []
        action_counts: Counter[str] = Counter()
        path_length = 0.0
        collision_count = 0
        previous_action: Optional[str] = None
        success = False

        for step in range(max_steps):
            state = agent.get_state()
            position = np.asarray(state.position, np.float64)
            rotation = habitat_rotation_world_from_eagor_body(state.rotation)
            if previous_position is not None:
                displacement = float(np.linalg.norm(position - previous_position))
                path_length += displacement
                if previous_action == "move_forward" and displacement < 0.01:
                    collision_count += 1
            relative_rotation = (
                None
                if previous_rotation is None
                else relative_view_rotation(previous_rotation, rotation)
            )

            panorama = normalize_native_equirectangular(
                observations["equirect_rgba_sensor"]
            )
            semantic = np.asarray(
                normalize_native_equirectangular(
                    observations["equirect_semantic_sensor"]
                )
            ).squeeze()
            target_mask = np.isin(semantic, TARGET_SEMANTIC_IDS)
            likelihood_result = backend.infer(
                panorama,
                TARGET_QUERY,
                observations=observations,
                simulator_state=simulator,
            )
            target_pixels = int(target_mask.sum())
            target_area = target_pixels / float(target_mask.size)
            if likelihood_result.target_visible and phase == "explore":
                phase = "track"
                first_detection_step = step

            prediction = policy.update(
                likelihood_result.likelihood
                if likelihood_result.target_visible
                else None,
                likelihood_result.target_visible,
                relative_rotation,
            )
            active_waypoint: Optional[np.ndarray]
            patrol_distance = 0.0
            if phase == "explore":
                action, waypoint_index, active_waypoint, patrol_distance = _patrol_action(
                    habitat_sim,
                    simulator,
                    position,
                    rotation,
                    waypoints,
                    waypoint_index,
                    arrival_radius=max(move_amount * 1.25, 0.24),
                    turn_threshold_deg=turn_amount_deg,
                )
                reason = "target_independent_coverage_patrol"
            else:
                active_waypoint = None
                aligned_for_stop = abs(prediction.azimuth) <= np.deg2rad(
                    turn_amount_deg
                )
                controller_area = target_area if aligned_for_stop else 0.0
                decision = approach_controller.act(
                    prediction.azimuth,
                    prediction.elevation,
                    prediction.confidence,
                    likelihood_result.target_visible,
                    controller_area,
                )
                action, reason = decision.action, decision.reason

            trajectory.append(position.copy())
            row: Dict[str, Any] = {
                "step": step,
                "phase": phase,
                "action": action,
                "reason": reason,
                "target_visible": bool(likelihood_result.target_visible),
                "target_pixels": target_pixels,
                "target_area_fraction": float(target_area),
                "confidence": float(prediction.confidence),
                "predicted_azimuth_deg": float(np.rad2deg(prediction.azimuth)),
                "predicted_elevation_deg": float(np.rad2deg(prediction.elevation)),
                "path_length_m": float(path_length),
                "waypoint_index": int(waypoint_index),
                "patrol_geodesic_remaining_m": float(patrol_distance),
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            }
            rows.append(row)
            writer.append(
                render_object_search_frame(
                    panorama=panorama,
                    perspective=observations["color_sensor"],
                    target_mask=target_mask,
                    likelihood=(
                        likelihood_result.likelihood
                        if likelihood_result.target_visible
                        else np.zeros(grid.shape, np.float32)
                    ),
                    belief=(
                        prediction.belief_heatmap
                        if prediction.valid
                        else np.zeros(grid.shape, np.float32)
                    ),
                    predicted_direction=(
                        prediction.direction_xyz if prediction.valid else None
                    ),
                    ground_truth_direction=None,
                    trajectory=trajectory,
                    world_bounds=world_bounds,
                    patrol_waypoints=waypoints,
                    active_waypoint=active_waypoint,
                    target_positions=(),
                    telemetry={
                        "target": TARGET_QUERY,
                        "phase": phase,
                        "step": step,
                        "action": action,
                        "visible": likelihood_result.target_visible,
                        "target_pixels": target_pixels,
                        "confidence": prediction.confidence,
                        "azimuth_deg": np.rad2deg(prediction.azimuth),
                        "path_length_m": path_length,
                    },
                )
            )
            action_counts[action] += 1
            if step % 25 == 0:
                print(
                    f"step={step:03d} phase={phase} action={action} "
                    f"pixels={target_pixels} path={path_length:.1f}m",
                    flush=True,
                )
            if action == "stop":
                success = bool(
                    likelihood_result.target_visible
                    and target_area >= stop_area_fraction
                )
                break

            previous_position = position.copy()
            previous_rotation = rotation
            previous_action = action
            observations = simulator.step(action)

        writer.close()
        writer = None
        _write_csv(csv_path, rows)
        discovery_path_length = None
        if first_detection_step is not None:
            discovery_path_length = rows[first_detection_step]["path_length_m"]
        result: Dict[str, Any] = {
            "status": "pass" if success else "incomplete",
            "success": success,
            "fixture": "ReplicaCAD apt_0 native RGB+semantic ERP",
            "target_query": TARGET_QUERY,
            "semantic_ids": list(TARGET_SEMANTIC_IDS),
            "perception_backend": "semantic_id_oracle",
            "oracle_debug_upper_bound": True,
            "not_zero_shot_objectnav": True,
            "controller_uses_target_coordinates": False,
            "target_coordinates_usage": "none",
            "exploration": "fixed target-independent coverage patrol",
            "navmesh": "runtime rebuild including static furniture, radius=0.20m",
            "steps": len(rows),
            "duration_seconds": len(rows) / float(fps),
            "first_detection_step": first_detection_step,
            "first_detection_path_length_m": discovery_path_length,
            "path_length_m": float(path_length),
            "collision_count": collision_count,
            "action_counts": dict(action_counts),
            "curated_route_geodesic_distances_m": route_distances,
            "curated_route_total_m": float(sum(route_distances)),
            "video": str(video_path),
            "step_csv": str(csv_path),
            "rows": rows,
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        return result
    finally:
        if writer is not None:
            writer.close()
        simulator.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a long-horizon ReplicaCAD object-search video"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("eagor_outputs/long_object_search"),
    )
    parser.add_argument("--scene", default="apt_0")
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--move-amount", type=float, default=0.18)
    parser.add_argument("--turn-amount-deg", type=float, default=10.0)
    parser.add_argument("--min-visible-pixels", type=int, default=2000)
    parser.add_argument("--stop-area-fraction", type=float, default=0.035)
    args = parser.parse_args()
    result = run_episode(
        output_root=args.output_root,
        scene=args.scene,
        max_steps=args.max_steps,
        width=args.width,
        height=args.height,
        fps=args.fps,
        move_amount=args.move_amount,
        turn_amount_deg=args.turn_amount_deg,
        min_visible_pixels=args.min_visible_pixels,
        stop_area_fraction=args.stop_area_fraction,
    )
    # Keep stdout compact; the full per-step trace is in JSON/CSV.
    compact = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
