import math
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_harder_benchmark import RepairedControllerConfig, TurnCommitFSM
from procedural_env_v2 import (
    CorridorType,
    HarderNarrowPassageEnv,
    _build_corridor,
    ground_truth_passable,
    _path_arcs,
    structural_margin_ground_truth,
    _width_at,
)
from narrow_passage.models.dynamic_feasibility import DynamicFeasibilityConfig
from narrow_passage.models.feasibility_ablation import (
    AblationMode,
    SharedDecisionState,
    select_ablation_mode,
)
from narrow_passage.models.robot_morphology import RobotMorphology


@pytest.mark.parametrize("yaw_deg", [0, 30, 45, 60])
def test_obb_wall_collision_boundary_at_yaw(yaw_deg):
    morphology = RobotMorphology(width=0.36, length=0.60, safety_margin=0.03)
    env = HarderNarrowPassageEnv({
        "corridor_types": ["straight"],
        "morphology": morphology,
    })
    env.reset(seed=7, options={"corridor_type": "straight", "passage_width": 1.20})
    yaw = math.radians(yaw_deg)
    half_projection = 0.5 * morphology.projected_width(yaw)
    boundary_offset = 1.20 / 2.0 - half_projection
    mid_y = env._params.total_arc / 2.0

    # Path normal points toward negative world x for the straight corridor.
    env.pose[:] = np.asarray([-boundary_offset + 1e-5, mid_y, yaw])
    assert not env._is_collision()
    env.pose[:] = np.asarray([-boundary_offset - 1e-5, mid_y, yaw])
    assert env._is_collision()


@pytest.mark.parametrize(
    "morphology",
    [
        RobotMorphology(width=0.28, length=0.55, safety_margin=0.03),
        RobotMorphology(width=0.45, length=0.65, safety_margin=0.03),
        RobotMorphology(width=0.36, length=0.80, safety_margin=0.03),
    ],
)
@pytest.mark.parametrize("yaw_deg", [0, 30, 45, 60])
def test_obb_wall_collision_boundary_generalizes_across_morphology(
    morphology, yaw_deg
):
    env = HarderNarrowPassageEnv({
        "corridor_types": ["straight"],
        "morphology": morphology,
    })
    env.reset(seed=11, options={"corridor_type": "straight", "passage_width": 1.40})
    yaw = math.radians(yaw_deg)
    boundary_offset = 1.40 / 2.0 - 0.5 * morphology.projected_width(yaw)
    mid_y = env._params.total_arc / 2.0
    env.pose[:] = np.asarray([-boundary_offset + 1e-5, mid_y, yaw])
    assert not env._is_collision()
    env.pose[:] = np.asarray([-boundary_offset - 1e-5, mid_y, yaw])
    assert env._is_collision()


def test_ground_truth_label_and_margin_share_obb_morphology():
    morphology = RobotMorphology(width=0.36, length=0.60, safety_margin=0.03)
    rng = np.random.default_rng(3)
    narrow, _ = _build_corridor(
        CorridorType.STRAIGHT, rng, morphology=morphology, forced_width=0.419
    )
    feasible, _ = _build_corridor(
        CorridorType.STRAIGHT, rng, morphology=morphology, forced_width=0.421
    )
    assert structural_margin_ground_truth(narrow) == pytest.approx(-0.001)
    assert structural_margin_ground_truth(feasible) == pytest.approx(0.001)
    assert not ground_truth_passable(narrow)
    assert ground_truth_passable(feasible)


def test_environment_and_controller_share_same_morphology_object():
    morphology = RobotMorphology()
    env = HarderNarrowPassageEnv({"morphology": morphology})
    agent = TurnCommitFSM(
        strict_ablation="full_dynamic_uncertainty",
        feasibility_cfg=DynamicFeasibilityConfig(morphology=env.morphology),
    )
    assert env.morphology is morphology
    assert agent.feasibility_estimator.cfg.morphology is env.morphology


def test_shared_controller_speed_is_kinematically_compatible_with_budget():
    cfg = RepairedControllerConfig()
    longest_s_route = 0.9 + 1.5 + 1.5 + 2.0 + 0.4
    translational_steps_after_turn_reserve = 200 - 30
    available_distance = (
        cfg.explore_speed * 0.25 * translational_steps_after_turn_reserve
    )
    assert available_distance >= longest_s_route


@pytest.mark.parametrize("corridor_type", [CorridorType.L_SHAPED, CorridorType.S_SHAPED])
def test_turning_corridors_have_obb_consistent_corner_chambers(corridor_type):
    morphology = RobotMorphology()
    params, width = _build_corridor(
        corridor_type,
        np.random.default_rng(19),
        morphology=morphology,
        forced_width=morphology.structural_required_width + 0.05,
    )
    assert min(value for _, value in params.width_profile) == pytest.approx(width)
    assert max(value for _, value in params.width_profile) >= (
        morphology.turning_required_width
    )
    arcs, _ = _path_arcs(params.path)
    half_plateau = morphology.half_length + morphology.safety_margin
    for corner_s in arcs[1:-1]:
        # The conservative turning width must exist over physical arc length,
        # not only at the zero-measure vertex coordinate.
        for probe_s in (corner_s - half_plateau, corner_s, corner_s + half_plateau):
            assert _width_at(probe_s, params.width_profile) >= (
                morphology.turning_required_width
            )


@pytest.mark.parametrize("corridor_type", [CorridorType.L_SHAPED, CorridorType.S_SHAPED])
@pytest.mark.parametrize("yaw_deg", [0, 30, 45, 60, 90])
def test_widened_corner_uses_union_of_adjacent_corridor_arms(
    corridor_type, yaw_deg
):
    """The OBB must not collide merely because the nearest arm changes."""

    morphology = RobotMorphology()
    env = HarderNarrowPassageEnv({
        "corridor_types": [corridor_type.value],
        "morphology": morphology,
    })
    env.reset(
        seed=19,
        options={
            "corridor_type": corridor_type.value,
            "passage_width": morphology.structural_required_width + 0.075,
        },
    )
    corner = np.asarray(env._params.path[1], dtype=float)
    env.pose[:] = np.asarray([
        float(corner[0]),
        float(corner[1]),
        math.radians(yaw_deg),
    ])
    assert not env._is_collision()
    assert env.action_is_safe(
        np.asarray([0.0, 0.0], dtype=np.float32),
        horizon_steps=4,
        minimum_clearance=0.005,
    )


def test_fallback_action_box_preserves_negative_angular_bound_and_entry_width():
    env = HarderNarrowPassageEnv({"morphology": RobotMorphology()})
    obs, _ = env.reset(
        seed=5,
        options={"corridor_type": "straight", "passage_width": 0.50, "yaw_deg": 30},
    )
    assert env.action_space.low[1] < 0.0
    assert obs[8] == pytest.approx(0.50)
    initial_yaw = float(env.pose[2])
    env.step(np.asarray([0.0, -0.4], dtype=np.float32))
    assert float(env.pose[2]) < initial_yaw


def test_safe_action_projection_does_not_prefer_stop_over_safe_rotation():
    env = HarderNarrowPassageEnv({"morphology": RobotMorphology()})
    env.reset(
        seed=5,
        options={"corridor_type": "straight", "passage_width": 0.50},
    )
    env.pose[:] = np.asarray([0.02, env._params.total_arc / 2.0, 0.0])
    desired = np.asarray([0.0, 0.8], dtype=np.float32)
    assert not env.action_is_safe(
        desired, horizon_steps=4, minimum_clearance=0.005
    )
    projected, reason = env.project_to_safe_action(
        desired, horizon_steps=4, minimum_clearance=0.005
    )
    assert reason != "stop"
    assert abs(float(projected[1])) > 0.0


def test_safe_projection_preserves_small_continuous_rotation_candidate():
    """A small heading correction must not be quantised into a two-cycle."""

    env = HarderNarrowPassageEnv({"morphology": RobotMorphology()})
    env.reset(
        seed=675890850,
        options={
            "corridor_type": "s_shaped",
            "passage_width": 0.445,
            "yaw_deg": 30,
            "lateral_offset": 0.20,
        },
    )
    # Deterministic near-entry boundary state from the preregistered smoke
    # scenario.  Translation is unsafe, while the requested 0.03-rad/s pure
    # correction is safe.
    env.pose[:] = np.asarray([0.043, -0.079, -0.086])
    desired = np.asarray([0.10, 0.03], dtype=np.float32)
    assert not env.action_is_safe(
        desired, horizon_steps=4, minimum_clearance=0.005
    )
    assert env.action_is_safe(
        np.asarray([0.0, 0.03], dtype=np.float32),
        horizon_steps=4,
        minimum_clearance=0.005,
    )
    projected, reason = env.project_to_safe_action(
        desired, horizon_steps=4, minimum_clearance=0.005
    )
    assert reason.startswith("safe_progress_v")
    assert float(projected[0]) > 0.0
    assert abs(float(projected[1]) - 0.03) < 0.10


def test_safe_projection_uses_nonzero_progress_when_only_slow_motion_is_safe():
    env = HarderNarrowPassageEnv({"morphology": RobotMorphology()})
    env.reset(
        seed=675890850,
        options={
            "corridor_type": "s_shaped",
            "passage_width": 0.445,
            "yaw_deg": 30,
            "lateral_offset": 0.20,
        },
    )
    env.pose[:] = np.asarray([-0.032511588, -0.080047809, 0.059604581])
    desired = np.asarray([0.10, 0.0], dtype=np.float32)
    assert not env.action_is_safe(
        desired, horizon_steps=4, minimum_clearance=0.005
    )
    assert env.action_is_safe(
        np.asarray([0.02, 0.0], dtype=np.float32),
        horizon_steps=4,
        minimum_clearance=0.005,
    )
    projected, reason = env.project_to_safe_action(
        desired, horizon_steps=4, minimum_clearance=0.005
    )
    assert reason.startswith("safe_progress_v")
    assert float(projected[0]) > 0.0
    assert abs(float(projected[1])) <= 0.10


def test_safe_projection_uses_emergency_creep_instead_of_stop():
    env = HarderNarrowPassageEnv({"morphology": RobotMorphology()})
    env.reset(
        seed=675890850,
        options={
            "corridor_type": "s_shaped",
            "passage_width": 0.445,
            "yaw_deg": 30,
            "lateral_offset": 0.20,
        },
    )
    env.pose[:] = np.asarray([-0.033820767, -0.057255376, 0.06201049])
    desired = np.asarray([0.10, 0.0], dtype=np.float32)
    assert not env.action_is_safe(
        np.asarray([0.01, 0.0], dtype=np.float32),
        horizon_steps=4,
        minimum_clearance=0.005,
    )
    assert env.action_is_safe(
        np.asarray([0.005, -0.05], dtype=np.float32),
        horizon_steps=4,
        minimum_clearance=0.005,
    )
    projected, reason = env.project_to_safe_action(
        desired, horizon_steps=4, minimum_clearance=0.005
    )
    assert reason.startswith("safe_progress_v")
    assert float(projected[0]) > 0.0


def test_safe_projection_uses_fine_rotation_at_wall_boundary():
    morphology = RobotMorphology()
    env = HarderNarrowPassageEnv({"morphology": morphology})
    env.reset(
        seed=675890850,
        options={
            "corridor_type": "s_shaped",
            "passage_width": 0.445,
            "yaw_deg": 30,
            "lateral_offset": 0.20,
        },
    )
    env.pose[:] = np.asarray([-0.465391904, 1.120507836, -1.577637076])
    desired = np.asarray([0.15, -0.5754], dtype=np.float32)
    projected, reason = env.project_to_safe_action(
        desired, horizon_steps=4, minimum_clearance=0.005
    )
    assert reason != "stop"
    assert abs(float(projected[1])) > 0.0


def test_safe_projection_backtracks_when_forward_and_rotation_are_unsafe():
    env = HarderNarrowPassageEnv({"morphology": RobotMorphology()})
    env.reset(
        seed=675890850,
        options={
            "corridor_type": "s_shaped",
            "passage_width": 0.445,
            "yaw_deg": 30,
            "lateral_offset": 0.20,
        },
    )
    env.pose[:] = np.asarray([-0.474669963, 1.111023664, -1.572547436])
    desired = np.asarray([0.15, -0.5368], dtype=np.float32)
    projected, reason = env.project_to_safe_action(
        desired, horizon_steps=4, minimum_clearance=0.005
    )
    assert reason == "safe_backtrack_recover"
    assert float(projected[0]) < 0.0


def test_pose_infeasibility_cannot_directly_reject_structurally_feasible_passage():
    metrics = {
        "mu_delta_struct": 0.10,
        "sigma_delta_struct": 0.001,
        "mu_delta_pose": -0.10,
        "sigma_delta_pose": 0.001,
    }
    decision = select_ablation_mode(
        "full_dynamic_uncertainty", metrics, SharedDecisionState()
    )
    assert decision.mode is AblationMode.EXPLORE
    assert "pose-conditioned" in decision.reason


def _controller_obs(width: float, yaw_deg: float) -> np.ndarray:
    obs = np.zeros(19, dtype=np.float32)
    obs[:6] = np.asarray([0.72, 0.78, 0.74, 0.96, 1.02, 0.98])
    obs[6:8] = width / 2.0 - 0.18
    obs[8] = width
    obs[9] = min(obs[6], obs[7])
    obs[10] = math.radians(yaw_deg)
    obs[12] = 2.0
    return obs


def test_large_yaw_feasible_passage_aligns_then_commits():
    agent = TurnCommitFSM(strict_ablation="full_dynamic_uncertainty")
    _, first = agent.step(_controller_obs(1.20, 60))
    _, second = agent.step(_controller_obs(1.20, 0))
    assert first == "ALIGN"
    assert second == "ENTER"


def test_truly_narrow_passage_rejects_from_structural_evidence():
    agent = TurnCommitFSM(strict_ablation="full_dynamic_uncertainty")
    _, mode = agent.step(_controller_obs(0.20, 0))
    assert mode == "REJECT"
    assert "structural" in agent.last_audit["feasibility_reason"]
