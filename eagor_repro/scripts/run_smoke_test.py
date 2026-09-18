#!/usr/bin/env python3
"""Run deterministic synthetic tests and a bundled-scene Habitat-Sim loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from eagor_repro.controllers.fixed_step_controller import FixedStepController
from eagor_repro.evaluation.metrics import angular_error_deg
from eagor_repro.evaluation.video_renderer import render_frame, save_video
from eagor_repro.policies.centroid_policy import CentroidPolicy
from eagor_repro.policies.circular_centroid_policy import CircularCentroidPolicy
from eagor_repro.policies.eagor_policy import EAGORPolicy
from eagor_repro.policies.grid_belief_policy import GridBeliefPolicy
from eagor_repro.sensors.panorama_sensor import normalize_native_equirectangular
from eagor_repro.spherical.belief_filter import SphericalHarmonicBeliefFilter
from eagor_repro.spherical.rotation import (
    habitat_rotation_world_from_eagor_body,
    relative_view_rotation,
    yaw_rotation,
)
from eagor_repro.spherical.spherical_grid import SphericalGrid


def _methods(grid: SphericalGrid) -> Dict[str, Any]:
    return {
        "centroid": CentroidPolicy(grid),
        "circular_centroid": CircularCentroidPolicy(grid),
        "grid": GridBeliefPolicy(grid),
        "eagor": EAGORPolicy(
            SphericalHarmonicBeliefFilter(
                grid, bandlimit=7, update_mode="paper", decode_mode="probability"
            )
        ),
    }


def synthetic_smoke(output_root: Path) -> Dict[str, Any]:
    grid = SphericalGrid(48, 96)
    policies = _methods(grid)
    rows: List[Dict[str, Any]] = []

    def add_case(
        name: str,
        azimuth_deg: float,
        visible: bool,
        rotation: np.ndarray,
        reset: bool = False,
        record: bool = True,
    ) -> None:
        if reset:
            for policy in policies.values():
                policy.reset()
        target = np.asarray(
            [
                np.cos(np.deg2rad(azimuth_deg)),
                np.sin(np.deg2rad(azimuth_deg)),
                0.0,
            ]
        )
        likelihood = grid.gaussian_likelihood(target, 10.0) if visible else None
        for method, policy in policies.items():
            prediction = policy.update(likelihood, visible, rotation)
            if record:
                rows.append(
                    {
                        "scenario": name,
                        "method": method,
                        "target_visible": visible,
                        "angular_error_deg": angular_error_deg(
                            prediction.direction_xyz, target
                        ),
                        "confidence": prediction.confidence,
                    }
                )

    # Independent direction fixtures must not share a recursive posterior.
    add_case("single_peak", 40.0, True, np.eye(3), reset=True)
    add_case("seam_positive", 179.0, True, np.eye(3), reset=True)
    add_case("seam_negative", -179.0, True, np.eye(3), reset=True)
    # One actual temporal sequence: front observation, left turn/missing frame,
    # then reacquisition.  A left turn maps previous front to current right.
    add_case("occlusion_init", 0.0, True, np.eye(3), reset=True, record=False)
    add_case(
        "occluded_left_turn",
        -90.0,
        False,
        yaw_rotation(np.deg2rad(-90.0)),
    )
    add_case("visible_again", -90.0, True, np.eye(3))
    seam_rows = [row for row in rows if row["scenario"].startswith("seam")]
    summary = {
        "status": "pass",
        "num_cases": len(rows),
        "mean_error_deg_by_method": {
            method: float(
                np.mean(
                    [row["angular_error_deg"] for row in rows if row["method"] == method]
                )
            )
            for method in policies
        },
        "mean_seam_error_deg_by_method": {
            method: float(
                np.mean(
                    [
                        row["angular_error_deg"]
                        for row in seam_rows
                        if row["method"] == method
                    ]
                )
            )
            for method in policies
        },
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "synthetic_smoke.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def habitat_sim_smoke(
    output_root: Path,
    max_steps: int = 40,
    initial_heading_deg: float = 90.0,
    artifact_stem: str = "habitat_sim_smoke",
) -> Dict[str, Any]:
    """Exercise native ERP rendering, egomotion, SH-BF and discrete actions.

    The target here is a synthetic waypoint and the likelihood is an explicit
    waypoint-oracle fixture, not an ObjectNav/semantic or zero-shot result.
    """

    import habitat_sim
    import quaternion
    from habitat_sim.utils.settings import default_sim_settings, make_cfg

    scene = Path("data_old/versioned_data/habitat_test_scenes/van-gogh-room.glb")
    if not scene.is_file():
        raise FileNotFoundError(f"Bundled Habitat smoke scene is missing: {scene}")
    settings = default_sim_settings.copy()
    settings.update(
        {
            "scene": str(scene),
            "width": 128,
            "height": 64,
            "color_sensor": True,
            "equirect_rgba_sensor": True,
            "sensor_height": 0.88,
            "enable_physics": False,
        }
    )
    sim_config = make_cfg(settings)
    for action_name in ("turn_left", "turn_right"):
        sim_config.agents[0].action_space[action_name].actuation.amount = 15.0
    sim_config.agents[0].action_space["move_forward"].actuation.amount = 0.25
    simulator = habitat_sim.Simulator(sim_config)
    try:
        agent = simulator.get_agent(0)
        state = agent.get_state()
        start = np.asarray(state.position, np.float64)
        desired = start + np.asarray([0.0, 0.0, -1.5])
        target = np.asarray(simulator.pathfinder.try_step(start, desired), np.float64)
        if np.linalg.norm(target - start) < 0.75:
            target = np.asarray(
                simulator.pathfinder.get_random_navigable_point_near(start, 1.5),
                np.float64,
            )
        # Rotate away from the target to exercise direction/control propagation.
        state.rotation = quaternion.from_rotation_vector(
            [0.0, np.deg2rad(initial_heading_deg), 0.0]
        )
        agent.set_state(state)

        grid = SphericalGrid(64, 128)
        policy = EAGORPolicy(
            SphericalHarmonicBeliefFilter(
                grid, bandlimit=7, update_mode="paper", decode_mode="probability"
            )
        )
        controller = FixedStepController(
            turn_threshold_deg=15.0,
            confidence_move_threshold=0.25,
            stop_likelihood_area_threshold=0.02,
        )
        observations = simulator.get_sensor_observations()
        previous_rotation = None
        trajectory: List[np.ndarray] = []
        frames: List[np.ndarray] = []
        rows: List[Dict[str, Any]] = []
        success = False
        for step in range(max_steps):
            state = agent.get_state()
            position = np.asarray(state.position, np.float64)
            rotation = habitat_rotation_world_from_eagor_body(state.rotation)
            relative = (
                None
                if previous_rotation is None
                else relative_view_rotation(previous_rotation, rotation)
            )
            delta_world = target - position
            distance = float(np.linalg.norm(delta_world))
            gt_direction = rotation.T @ (delta_world / max(distance, 1e-12))
            sigma_deg = min(45.0, 10.0 / max(distance, 0.2))
            likelihood = grid.gaussian_likelihood(gt_direction, sigma_deg)
            # Every fifth frame deliberately misses the observation.
            visible = step % 5 != 4
            prediction = policy.update(
                likelihood if visible else None, visible, relative
            )
            target_area = float(np.mean(likelihood >= 0.5))
            decision = controller.act(
                prediction.azimuth,
                prediction.elevation,
                prediction.confidence,
                visible,
                target_area,
            )
            error = angular_error_deg(prediction.direction_xyz, gt_direction)
            trajectory.append(position.copy())
            panorama = normalize_native_equirectangular(
                observations["equirect_rgba_sensor"]
            )
            frames.append(
                render_frame(
                    panorama,
                    likelihood if visible else np.zeros(grid.shape),
                    prediction.belief_heatmap,
                    prediction.direction_xyz,
                    gt_direction,
                    decision.action,
                    prediction.confidence,
                    error,
                    trajectory,
                    perspective=observations.get("color_sensor"),
                )
            )
            rows.append(
                {
                    "step": step,
                    "distance_m": distance,
                    "angular_error_deg": error,
                    "confidence": prediction.confidence,
                    "target_visible": visible,
                    "action": decision.action,
                }
            )
            if decision.action == "stop":
                success = distance < 0.5
                break
            observations = simulator.step(decision.action)
            previous_rotation = rotation

        output_root.mkdir(parents=True, exist_ok=True)
        video_path = save_video(frames, output_root / f"{artifact_stem}.mp4", fps=8)
        result = {
            "status": "pass" if success else "incomplete",
            "fixture": "bundled van-gogh-room + synthetic waypoint-oracle likelihood",
            "not_an_objectnav_result": True,
            "initial_heading_deg": float(initial_heading_deg),
            "success": success,
            "steps": len(rows),
            "final_distance_m": rows[-1]["distance_m"],
            "mean_angular_error_deg": float(
                np.mean([row["angular_error_deg"] for row in rows])
            ),
            "video": str(video_path),
            "rows": rows,
        }
        with (output_root / f"{artifact_stem}.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(result, handle, indent=2)
        return result
    finally:
        simulator.close()


def habitat_lab_sensor_smoke() -> Dict[str, Any]:
    """Verify Habitat-Lab Env reset/step with the native ERP adapter."""

    import habitat
    from habitat.config import read_write

    from eagor_repro.sensors.panorama_sensor import (
        configure_equirectangular_sensors,
        extract_panorama,
    )

    dataset_path = (
        "data_old/versioned_data/habitat_test_pointnav_dataset_1.0/"
        "v1/val/val.json.gz"
    )
    config = habitat.get_config(
        "benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            f'habitat.dataset.data_path="{dataset_path}"',
            "habitat.dataset.split=val",
        ],
    )
    dataset = habitat.make_dataset(
        config.habitat.dataset.type, config=config.habitat.dataset
    )
    scene_root = Path("data_old/versioned_data/habitat_test_scenes").resolve()
    for episode in dataset.episodes:
        episode.scene_id = str(scene_root / Path(episode.scene_id).name)
    with read_write(config):
        configure_equirectangular_sensors(
            config.habitat, 32, 64, include_semantic=False
        )
    with habitat.Env(config=config, dataset=dataset) as env:
        observations = env.reset()
        reset_shape = list(extract_panorama(observations).shape)
        observations = env.step("turn_left")
        step_shape = list(extract_panorama(observations).shape)
        keys = sorted(observations.keys())
    return {
        "status": "pass",
        "fixture": "bundled Habitat PointNav dataset; API integration only",
        "not_an_objectnav_result": True,
        "reset_erp_shape": reset_shape,
        "step_erp_shape": step_shape,
        "observation_keys": keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("data/eagor_results/smoke"))
    parser.add_argument("--skip-habitat", action="store_true")
    parser.add_argument("--max-steps", type=int, default=40)
    args = parser.parse_args()
    report = {"synthetic": synthetic_smoke(args.output_root)}
    if not args.skip_habitat:
        report["habitat_sim"] = habitat_sim_smoke(args.output_root, args.max_steps)
        report["habitat_lab_sensor"] = habitat_lab_sensor_smoke()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
