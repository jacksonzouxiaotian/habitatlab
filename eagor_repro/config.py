"""Configuration loading and component factories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from eagor_repro.controllers.fixed_step_controller import FixedStepController
from eagor_repro.controllers.stop_criteria import StopCriterion
from eagor_repro.perception.oracle_semantic_backend import OracleSemanticBackend
from eagor_repro.perception.qwen_grounding_backend import QwenGroundingBackend
from eagor_repro.perception.recorded_likelihood_backend import RecordedLikelihoodBackend
from eagor_repro.policies.centroid_policy import CentroidPolicy
from eagor_repro.policies.circular_centroid_policy import CircularCentroidPolicy
from eagor_repro.policies.eagor_policy import EAGORPolicy
from eagor_repro.policies.grid_belief_policy import GridBeliefPolicy
from eagor_repro.policies.oracle_direction_policy import OracleDirectionPolicy
from eagor_repro.planning.candidate_heading_planner import CandidateHeadingPlanner
from eagor_repro.spherical.belief_filter import SphericalHarmonicBeliefFilter
from eagor_repro.spherical.spherical_grid import SphericalGrid


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    config = OmegaConf.load(Path(path))
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(config)
    return config


def make_grid(config: DictConfig) -> SphericalGrid:
    return SphericalGrid(config.panorama.height, config.panorama.width)


def make_policy(method: str, config: DictConfig, grid: SphericalGrid) -> Any:
    normalized = method.lower().replace("-", "_")
    if normalized == "centroid":
        return CentroidPolicy(grid)
    if normalized in ("circular_centroid", "centroid_circ", "cent_circ"):
        return CircularCentroidPolicy(grid)
    if normalized == "grid":
        return GridBeliefPolicy(
            grid,
            decay=float(config.grid_baseline.decay),
            interpolation_order=int(config.grid_baseline.interpolation_order),
        )
    if normalized in ("oracle", "oracle_direction", "gt_direction"):
        return OracleDirectionPolicy(grid)
    if normalized in ("eagor", "eagor_sh_bf", "sh_bf"):
        section = config.sh_belief
        belief_filter = SphericalHarmonicBeliefFilter(
            grid=grid,
            bandlimit=int(section.bandlimit),
            epsilon=float(config.likelihood.epsilon),
            update_mode=str(section.update_mode),
            belief_decay=float(section.belief_decay),
            observation_weight=float(section.observation_weight),
            missing_observation_mode=str(section.missing_observation_mode),
            decode_mode=str(section.decode_mode),
            max_belief_age=int(section.max_belief_age),
        )
        return EAGORPolicy(belief_filter)
    raise ValueError(f"Unknown policy method: {method}")


def make_backend(config: DictConfig) -> Any:
    section = config.likelihood
    backend = str(section.backend)
    if backend == "oracle_semantic":
        return OracleSemanticBackend(
            sigma_azimuth_deg=float(section.gaussian_sigma_azimuth_deg),
            sigma_elevation_deg=float(section.gaussian_sigma_elevation_deg),
            weak_observation=float(section.epsilon),
        )
    if backend == "recorded":
        return RecordedLikelihoodBackend(section.recorded_sources)
    if backend == "qwen_grounding":
        return QwenGroundingBackend(
            model_name_or_path=section.qwen.model_name_or_path,
            sigma_azimuth_deg=float(section.gaussian_sigma_azimuth_deg),
            sigma_elevation_deg=float(section.gaussian_sigma_elevation_deg),
            absence_floor=float(section.epsilon),
            timeout_s=float(section.qwen.timeout_s),
            max_new_tokens=int(section.qwen.max_new_tokens),
            fail_open=bool(section.qwen.fail_open),
        )
    raise ValueError(f"Unknown likelihood backend: {backend}")


def make_controller(config: DictConfig) -> FixedStepController:
    section = config.controller
    return FixedStepController(
        turn_threshold_deg=float(section.turn_threshold_deg),
        confidence_move_threshold=float(section.confidence_move_threshold),
        low_confidence_action=str(section.low_confidence_action),
        stop_likelihood_area_threshold=float(section.stop_likelihood_area_threshold),
        exploration_turn_steps=int(section.get("exploration_turn_steps", 6)),
    )


def make_stop_criterion(config: DictConfig) -> StopCriterion:
    section = config.controller
    return StopCriterion(
        mode=str(section.get("stop_mode", "area")),
        confidence_threshold=float(section.confidence_move_threshold),
        area_fraction_threshold=float(section.stop_likelihood_area_threshold),
        stop_distance_m=float(section.get("stop_distance_m", 1.0)),
        success_distance_m=float(config.evaluation.success_distance_m),
        min_target_pixels=int(section.get("min_target_pixels", 20)),
        min_valid_depth_pixels=int(section.get("min_valid_depth_pixels", 5)),
    )


def make_planner(config: DictConfig) -> CandidateHeadingPlanner:
    section = config.get("planning", {})
    return CandidateHeadingPlanner(
        mode=str(section.get("mode", "direct")),
        candidate_offsets_deg=list(
            section.get(
                "candidate_offsets_deg",
                [0, -15, 15, -30, 30, -45, 45, -60, 60],
            )
        ),
        sector_width_deg=float(section.get("sector_width_deg", 10.0)),
        clearance_percentile=float(section.get("clearance_percentile", 25.0)),
        vertical_band_deg=float(section.get("vertical_band_deg", 35.0)),
        obstacle_distance_m=float(section.get("obstacle_distance_m", 0.5)),
        safety_distance_m=float(section.get("safety_distance_m", 0.7)),
        goal_weight=float(section.get("goal_weight", 1.0)),
        clearance_weight=float(section.get("clearance_weight", 0.7)),
        obstacle_weight=float(section.get("obstacle_weight", 2.0)),
        turn_weight=float(section.get("turn_weight", 0.15)),
    )
