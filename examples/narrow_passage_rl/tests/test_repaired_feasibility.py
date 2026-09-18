import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cross_episode_memory import (
    CrossEpisodeMemory,
    EpisodeOutcome,
    MemoryWriteType,
)
from eval_harder_benchmark import TurnCommitFSM, _follow_space_mode, fsm_action
from eval_structural_feasibility import _ordered_scenarios, _scenario_rows
from procedural_env_v2 import HarderNarrowPassageEnv
from narrow_passage.models.dynamic_feasibility import (
    DynamicFeasibilityConfig,
    DynamicFeasibilityEstimator,
)
from narrow_passage.models.feasibility_ablation import AblationMode, ModeSelection
from narrow_passage.models.robot_morphology import RobotMorphology


def _obs(width: float = 0.60, yaw: float = 0.0) -> np.ndarray:
    obs = np.zeros(19, dtype=np.float32)
    obs[:6] = 1.0
    obs[6:8] = width / 2.0
    obs[8] = width
    obs[10] = yaw
    obs[12] = 2.0
    return obs


def _record(
    memory: CrossEpisodeMemory,
    outcome: EpisodeOutcome,
    *,
    morphology: RobotMorphology | None = None,
    width: float = 0.60,
    env_steps: int = 5,
    geometry_caused: bool = False,
):
    morphology = morphology or RobotMorphology()
    return memory.record_episode(
        _obs(width),
        outcome is EpisodeOutcome.SUCCESS,
        width,
        steps=env_steps,
        corridor_type="straight",
        outcome=outcome,
        attempted=env_steps > 0,
        env_step_count=env_steps,
        geometry_caused=geometry_caused,
        structural_margin=width - morphology.structural_required_width,
        morphology=morphology,
    )


def test_censored_reject_never_writes_negative_geometry_memory():
    memory = CrossEpisodeMemory()
    audit = _record(memory, EpisodeOutcome.REJECT_CENSORED, env_steps=0)
    assert audit["attempted"] is False
    assert audit["geometry_memory_write"] is False
    assert audit["memory_write_type"] == MemoryWriteType.AUDIT_ONLY.value
    assert memory._calibrator.n_updates == 0


def test_control_failures_and_unknown_collision_are_audit_only():
    for outcome in (
        EpisodeOutcome.TIMEOUT_CONTROL,
        EpisodeOutcome.STUCK_CONTROL,
        EpisodeOutcome.COLLISION_CONTROL,
        EpisodeOutcome.COLLISION_UNKNOWN,
    ):
        memory = CrossEpisodeMemory()
        audit = _record(memory, outcome)
        assert audit["geometry_memory_write"] is False
        assert memory._calibrator.n_updates == 0


def test_confirmed_geometry_and_success_write_signed_evidence():
    memory = CrossEpisodeMemory()
    negative = _record(memory, EpisodeOutcome.GEOMETRIC_INFEASIBLE)
    positive = _record(memory, EpisodeOutcome.SUCCESS, width=0.70)
    assert negative["memory_write_type"] == MemoryWriteType.NEGATIVE_GEOMETRY.value
    assert positive["memory_write_type"] == MemoryWriteType.POSITIVE_GEOMETRY.value
    assert memory._calibrator.n_updates == 2


def test_one_similar_failure_cannot_hard_reject_clear_feasible_base():
    memory = CrossEpisodeMemory()
    _record(memory, EpisodeOutcome.GEOMETRIC_INFEASIBLE)
    corrected = memory.correct_feasibility(
        _obs(0.60),
        base_p_feas=0.95,
        base_structural_margin=0.18,
        corridor_type="straight",
        morphology=RobotMorphology(),
    )
    assert corrected["negative_geometry_support"] == 1
    assert corrected["corrected_p_feas"] > 0.5
    assert corrected["memory_terminal_reject"] is False


def test_conflicting_geometry_evidence_cannot_terminal_reject():
    memory = CrossEpisodeMemory()
    for _ in range(3):
        _record(memory, EpisodeOutcome.GEOMETRIC_INFEASIBLE, width=0.38)
    _record(memory, EpisodeOutcome.SUCCESS, width=0.38)
    corrected = memory.correct_feasibility(
        _obs(0.38),
        base_p_feas=0.10,
        base_structural_margin=-0.04,
        corridor_type="straight",
        morphology=RobotMorphology(),
    )
    assert corrected["evidence_conflict"] is True
    assert corrected["memory_terminal_reject"] is False


def test_different_morphology_does_not_share_geometry_failure_by_default():
    memory = CrossEpisodeMemory()
    _record(
        memory,
        EpisodeOutcome.GEOMETRIC_INFEASIBLE,
        morphology=RobotMorphology(width=0.45, length=0.65),
        width=0.49,
    )
    stats = memory.retrieval_stats(
        _obs(0.49),
        "straight",
        morphology=RobotMorphology(width=0.36, length=0.60),
        structural_margin=0.07,
    )
    assert stats["n_similar"] == 0
    assert stats["negative_geometry_support"] == 0


def test_seeded_order_is_deterministic_shared_and_not_legacy_bin_order():
    rows = _scenario_rows("smoke", [0], "balanced")
    first = _ordered_scenarios(rows, "seeded_shuffle")
    second = _ordered_scenarios(rows, "seeded_shuffle")
    ids_first = [row["scenario_id"] for row in first]
    ids_second = [row["scenario_id"] for row in second]
    assert ids_first == ids_second
    assert ids_first != [row["scenario_id"] for row in rows]
    # The runner iterates this list outside its method loop, so this exact order
    # is consumed by every paired method.
    assert len(ids_first) == len(set(ids_first))


def test_memory_reset_clears_cross_episode_state_at_seed_boundary():
    memory = CrossEpisodeMemory()
    _record(memory, EpisodeOutcome.SUCCESS)
    assert memory.summary()["n_episodes"] == 1
    memory.reset()
    assert memory.summary()["n_episodes"] == 0
    assert memory._calibrator.n_updates == 0


def test_max_range_is_valid_no_return_not_dropout():
    cfg = DynamicFeasibilityConfig()
    obs = _obs(width=10.0)
    obs[:6] = cfg.max_depth
    obs[6:8] = cfg.max_depth
    obs[9] = 4.82
    metrics = DynamicFeasibilityEstimator(cfg).update(obs)
    assert metrics["valid_hit_ratio"] == 0.0
    assert metrics["valid_no_return_ratio"] == 1.0
    assert metrics["invalid_ratio"] == 0.0
    assert metrics["dropout_ratio"] == 0.0
    assert metrics["passage_aperture_observable"] is False


def test_only_invalid_depth_encoding_increases_dropout():
    cfg = DynamicFeasibilityConfig()
    obs = _obs(width=0.60)
    obs[:6] = cfg.max_depth
    obs[0] = 0.0
    metrics = DynamicFeasibilityEstimator(cfg).update(obs)
    assert metrics["invalid_ratio"] == 1.0 / 6.0
    assert metrics["dropout_ratio"] == metrics["invalid_ratio"]
    assert metrics["valid_no_return_ratio"] == 5.0 / 6.0


def test_unobservable_open_space_uses_positive_velocity_approach():
    obs = _obs(width=10.0)
    obs[:6] = 5.0
    obs[6:8] = 5.0
    obs[9] = 4.82
    agent = TurnCommitFSM(strict_ablation="full_dynamic_uncertainty")
    action, mode = agent.step(obs)
    assert mode == "APPROACH"
    assert float(action[0]) > 0.0
    assert agent.last_audit["structural_width_ci_gate_enabled"] is False


def test_structural_explore_is_active_sensing_not_zero_speed():
    obs = _obs(width=0.55)
    obs[9] = 0.50
    metrics = {
        "LCB_struct": 0.0,
        "mu_delta": 0.10,
        "mu_delta_struct": 0.10,
        "mu_delta_pose": 0.10,
        "passage_aperture_observable": True,
    }
    action, mode = fsm_action(
        obs,
        strict_ablation="full_dynamic_uncertainty",
        strict_metrics=metrics,
        strict_selection=ModeSelection(
            AblationMode.EXPLORE, "test overlap", "test"
        ),
        controller_state={"probe_no_information_counter": 0},
    )
    assert mode == "EXPLORE"
    assert float(action[0]) > 0.0
    assert abs(float(action[1])) > 0.0


def test_intentional_rotation_does_not_increment_translation_stuck():
    env = HarderNarrowPassageEnv({"corridor_types": ["straight"]})
    env.reset(
        seed=2,
        options={"corridor_type": "straight", "passage_width": 0.80},
    )
    for _ in range(30):
        env.step(np.asarray([0.0, 0.25], dtype=np.float32))
    assert env.translation_stuck_steps == 0
    assert env.stuck_steps == 0


def test_recover_is_bounded_and_exits_to_realign():
    obs = _obs(width=0.80)
    obs[15] = 1.0
    agent = TurnCommitFSM(strict_ablation="full_dynamic_uncertainty")
    modes = [agent.step(obs.copy())[1] for _ in range(30)]
    longest = 0
    current = 0
    for mode in modes:
        current = current + 1 if mode == "RECOVER" else 0
        longest = max(longest, current)
    assert longest < agent.controller_cfg.max_recover_steps
    assert "ALIGN" in modes
    assert agent._recover_cycle_count >= 1


def test_swept_obb_check_rejects_action_that_crosses_wall():
    env = HarderNarrowPassageEnv({"corridor_types": ["straight"]})
    env.reset(
        seed=3,
        options={"corridor_type": "straight", "passage_width": 0.50},
    )
    # Put the OBB just inside the wall boundary and steer farther outward.
    env.pose[:] = np.asarray([-0.069, 1.0, 0.0], dtype=np.float32)
    action = np.asarray([0.20, -0.8], dtype=np.float32)
    assert not env.action_is_safe(action, horizon_steps=4)
    projected, _ = env.project_to_safe_action(action, horizon_steps=4)
    assert env.action_is_safe(projected, horizon_steps=4)
    assert np.all(projected >= env.action_space.low)
    assert np.all(projected <= env.action_space.high)


def test_invalid_depth_cannot_trigger_follow_space_corner_mode():
    obs = _obs(width=0.55)
    obs[:3] = 0.0
    obs[9] = 0.05
    assert _follow_space_mode(obs) is None


def test_remote_perpendicular_arm_does_not_create_entrance_ray_crosstalk():
    env = HarderNarrowPassageEnv({"corridor_types": ["l_shaped"]})
    obs, _ = env.reset(
        seed=19,
        options={
            "corridor_type": "l_shaped",
            "passage_width": 0.55,
            "yaw_deg": 45,
            "lateral_offset": 0.15,
        },
    )
    # The robot starts 0.9 m before the entry.  A future perpendicular arm must
    # not be reported as an obstacle one ray sample (0.02 m) away.
    positive = np.asarray(obs[:3], dtype=float)
    assert np.all(positive > 0.20)


def test_depth_corner_signature_starts_nonzero_local_turn():
    obs = _obs(width=0.80)
    obs[:3] = np.asarray([0.60, 0.60, 0.24], dtype=np.float32)
    obs[9] = 0.12
    detected = _follow_space_mode(obs)
    assert detected is not None
    mode, action = detected
    assert mode == "FOLLOW_SPACE"
    assert float(action[0]) > 0.0
    assert float(action[1]) < 0.0


def test_explicit_passage_fsm_aligns_enters_traverses_and_exits():
    agent = TurnCommitFSM(strict_ablation="mean_only")
    outside = _obs(width=1.20, yaw=0.0)
    outside[9] = 4.82
    _, mode = agent.step(outside)
    assert mode == "ENTER"
    assert agent.last_audit["passage_control_state"] == "ENTER"
    assert agent.last_audit["ready_to_enter"] is True

    inside = outside.copy()
    inside[9] = 0.20
    inside[6:8] = 0.20
    agent.step(inside)
    assert agent.last_audit["passage_control_state"] == "TRAVERSE"

    exited = outside.copy()
    exited[12] = 0.50
    agent.step(exited)
    assert agent.last_audit["passage_control_state"] == "EXIT"
