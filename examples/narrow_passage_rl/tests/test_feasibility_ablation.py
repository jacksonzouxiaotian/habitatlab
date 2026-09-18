import math
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_feasibility_stress import make_observation, mechanism_subset
from eval_harder_benchmark import TurnCommitFSM, fsm_action
from cross_episode_memory import FSMMode
from narrow_passage.models.dynamic_feasibility import (
    DynamicFeasibilityConfig,
    DynamicFeasibilityEstimator,
    GaussianMarginBelief,
    fuse_margin_observation,
    propagate_margin_belief,
    required_width_with_yaw,
    required_width_without_yaw,
)
from narrow_passage.models.feasibility_ablation import (
    AblationMode,
    SelectorThresholds,
    SharedDecisionState,
    interval_bounds,
    select_ablation_mode,
)
from narrow_passage.models.robot_morphology import RobotMorphology


class AccessAuditMapping(Mapping):
    def __init__(self, values):
        self.values = dict(values)
        self.accessed = []

    def __getitem__(self, key):
        self.accessed.append(key)
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


def _obs(*, width=0.65, heading=0.1, lateral=0.0, body_margin=0.12):
    obs = np.zeros(19, dtype=np.float32)
    obs[:6] = np.asarray([0.72, 0.78, 0.74, 0.96, 1.02, 0.98])
    obs[6] = width / 2.0
    obs[7] = width / 2.0
    obs[8] = width
    obs[9] = body_margin
    obs[10] = heading
    obs[11] = lateral
    obs[12] = 2.0
    return obs


def test_point_estimate_selector_reads_only_mean_margin():
    metrics = AccessAuditMapping(
        {
            "mu_delta_struct": 0.10, "mu_delta_pose": 0.10,
            "sigma_delta_struct": 999.0, "sigma_delta_pose": 999.0,
        }
    )
    decision = select_ablation_mode(
        "point_estimate", metrics, SharedDecisionState()
    )
    assert decision.mode is AblationMode.COMMIT
    assert metrics.accessed == ["mu_delta_struct", "mu_delta_pose"]


def test_no_uncertainty_logs_upstream_but_selector_reads_only_mean():
    metrics = AccessAuditMapping(
        {
            "mu_delta_struct": 0.10, "mu_delta_pose": 0.10,
            "sigma_delta_struct": 999.0, "sigma_delta_pose": 999.0,
        }
    )
    decision = select_ablation_mode(
        "no_uncertainty", metrics, SharedDecisionState()
    )
    assert decision.mode is AblationMode.COMMIT
    assert metrics.accessed == ["mu_delta_struct", "mu_delta_pose"]
    assert decision.decision_rule == "compatibility_alias_behaviorally_equivalent_to_mean_only"


def test_full_and_point_differ_on_high_uncertainty_boundary_case():
    metrics = {
        "mu_delta_struct": 0.05, "sigma_delta_struct": 0.06,
        "mu_delta_pose": 0.05, "sigma_delta_pose": 0.06,
    }
    full = select_ablation_mode("full", metrics, SharedDecisionState())
    point = select_ablation_mode("point_estimate", metrics, SharedDecisionState())
    assert full.mode is AblationMode.EXPLORE
    assert full.uncertainty_triggered
    assert point.mode is AblationMode.COMMIT


def test_interval_equations_and_exact_boundaries():
    cfg = SelectorThresholds(kappa=2.0, tau_commit=0.02, tau_reject=0.03)
    assert interval_bounds(0.10, 0.03, cfg.kappa) == (0.04000000000000001, 0.16)
    commit = select_ablation_mode(
        "full", {
            "mu_delta_struct": 0.10, "sigma_delta_struct": 0.03,
            "mu_delta_pose": 0.10, "sigma_delta_pose": 0.03,
        },
        SharedDecisionState(), cfg,
    )
    reject = select_ablation_mode(
        "full", {
            "mu_delta_struct": -0.10, "sigma_delta_struct": 0.02,
            "mu_delta_pose": -0.10, "sigma_delta_pose": 0.02,
        },
        SharedDecisionState(), cfg,
    )
    assert commit.mode is AblationMode.COMMIT
    assert reject.mode is AblationMode.REJECT


def test_scene_dependent_sigma_is_not_constant():
    cfg = DynamicFeasibilityConfig()
    estimator = DynamicFeasibilityEstimator(cfg)
    observations = [
        make_observation((0.05, 0, "clean", 0.0), 0, cfg)[0],
        make_observation((0.05, 30, "mild", 0.10), 1, cfg)[0],
        make_observation((0.05, 60, "severe", 0.20), 2, cfg)[0],
    ]
    sigmas = [float(estimator.update(obs)["sigma_delta"]) for obs in observations]
    assert len({round(value, 8) for value in sigmas}) > 1


def test_dynamic_estimator_logs_all_five_uncertainty_sources():
    cfg = DynamicFeasibilityConfig()
    metrics = DynamicFeasibilityEstimator(cfg).update(
        make_observation((0.05, 30, "severe", 0.10), 0, cfg)[0]
    )
    for field in (
        "ray_dispersion",
        "valid_depth_ratio",
        "dropout_ratio",
        "boundary_fitting_residual",
        "temporal_width_variation",
        "yaw_pose_sigma",
    ):
        assert field in metrics
        assert math.isfinite(float(metrics[field]))


def test_recursive_propagate_and_fuse_match_declared_scalar_formulas():
    prior = GaussianMarginBelief(mu=0.10, variance=0.04)
    propagated = propagate_margin_belief(
        prior,
        required_width_delta=0.03,
        process_variance=0.01,
    )
    assert propagated.mu == 0.07
    assert propagated.variance == 0.05
    posterior = fuse_margin_observation(
        propagated,
        observation_mu=-0.01,
        observation_variance=0.02,
        gain=0.25,
    )
    assert posterior.mu == pytest.approx(0.05)
    assert posterior.variance == pytest.approx(0.75**2 * 0.05 + 0.25**2 * 0.02)


def test_recursive_default_mean_is_equivalent_to_historical_ewma_and_yaw_projection():
    cfg = DynamicFeasibilityConfig()
    estimator = DynamicFeasibilityEstimator(cfg)
    sequence = [
        make_observation((0.05, 0, "clean", 0.0), 0, cfg)[0],
        make_observation((0.05, 30, "mild", 0.10), 1, cfg)[0],
        make_observation((0.10, 60, "severe", 0.20), 2, cfg)[0],
    ]
    for obs in sequence:
        metrics = estimator.update(obs)
        assert float(metrics["belief_mean_equivalence_error_struct"]) == pytest.approx(
            0.0, abs=1e-12
        )
        assert float(metrics["belief_mean_equivalence_error_pose"]) == pytest.approx(
            0.0, abs=1e-12
        )
        assert float(metrics["posterior_mu_delta_struct"]) == pytest.approx(
            float(metrics["mu_delta_struct"]), abs=1e-12
        )
        assert float(metrics["posterior_mu_delta_pose"]) == pytest.approx(
            float(metrics["mu_delta_pose"]), abs=1e-12
        )


def test_recursive_propagate_and_fuse_can_be_ablated_independently():
    cfg = DynamicFeasibilityConfig()
    first = make_observation((0.05, 0, "clean", 0.0), 0, cfg)[0]
    second = make_observation((0.10, 60, "severe", 0.20), 1, cfg)[0]
    estimators = {
        "both": DynamicFeasibilityEstimator(cfg),
        "no_propagate": DynamicFeasibilityEstimator(cfg),
        "no_fuse": DynamicFeasibilityEstimator(cfg),
    }
    for estimator in estimators.values():
        estimator.update(first)
    both = estimators["both"].update(second)
    no_propagate = estimators["no_propagate"].update(
        second, enable_belief_propagation=False
    )
    no_fuse = estimators["no_fuse"].update(second, enable_belief_fusion=False)

    assert both["belief_propagation_enabled"] is True
    assert both["belief_fusion_enabled"] is True
    assert no_propagate["belief_propagation_enabled"] is False
    assert no_propagate["belief_fusion_enabled"] is True
    assert no_fuse["belief_propagation_enabled"] is True
    assert no_fuse["belief_fusion_enabled"] is False
    assert float(both["posterior_mu_delta_pose"]) != pytest.approx(
        float(no_propagate["posterior_mu_delta_pose"])
    )
    assert float(both["posterior_mu_delta_pose"]) != pytest.approx(
        float(no_fuse["posterior_mu_delta_pose"])
    )
    # Ablating the diagnostic belief components cannot leak into established
    # selector-facing engineering fields.
    for field in (
        "mu_delta_struct", "sigma_delta_struct", "mu_delta_pose", "sigma_delta_pose"
    ):
        assert float(both[field]) == pytest.approx(float(no_propagate[field]))
        assert float(both[field]) == pytest.approx(float(no_fuse[field]))


def test_posterior_confidence_is_parallel_to_engineering_sigma():
    cfg = DynamicFeasibilityConfig()
    estimator = DynamicFeasibilityEstimator(cfg)
    rows = [
        estimator.update(make_observation((0.05, 0, "clean", 0.0), 0, cfg)[0]),
        estimator.update(make_observation((0.05, 30, "mild", 0.10), 1, cfg)[0]),
        estimator.update(make_observation((0.05, 60, "severe", 0.20), 2, cfg)[0]),
    ]
    assert len({round(float(row["posterior_var_delta_pose"]), 10) for row in rows}) > 1
    assert any(
        not math.isclose(
            float(row["posterior_sigma_delta_pose"]),
            float(row["sigma_delta_pose"]),
        )
        for row in rows
    )
    for row in rows:
        assert 0.0 < float(row["posterior_concentration_pose"]) <= 1.0


def test_open_space_width_sentinel_is_not_mixed_into_passage_temporal_variance():
    cfg = DynamicFeasibilityConfig()
    estimator = DynamicFeasibilityEstimator(cfg)
    open_obs = _obs(width=10.0, heading=0.0, body_margin=4.82)
    open_obs[:6] = cfg.max_depth
    estimator.update(open_obs)
    estimator.update(open_obs)
    passage_obs = _obs(width=0.50, heading=0.0, body_margin=0.07)
    metrics = estimator.update(passage_obs)
    assert metrics["in_passage_regime"] is True
    assert float(metrics["temporal_width_variation"]) == 0.0
    assert float(metrics["sigma_delta"]) < 0.20


def test_fixed_uncertainty_uses_global_sigma_but_keeps_dynamic_diagnostic():
    cfg = DynamicFeasibilityConfig(fixed_sigma_delta=0.05)
    observations = [
        make_observation((0.05, 0, "clean", 0.0), 0, cfg)[0],
        make_observation((0.05, 60, "severe", 0.20), 1, cfg)[0],
    ]
    metrics = [
        DynamicFeasibilityEstimator(cfg).update(obs, fixed_uncertainty=True)
        for obs in observations
    ]
    assert {float(item["sigma_delta"]) for item in metrics} == {0.05}
    assert len({round(float(item["dynamic_sigma_delta"]), 8) for item in metrics}) > 1


def test_rectangle_yaw_projection_uses_plus_terms_and_safety_addition():
    cfg = DynamicFeasibilityConfig(
        morphology=RobotMorphology(width=0.36, length=0.60, safety_margin=0.03)
    )
    yaw = math.radians(45)
    expected = 0.36 * abs(math.cos(yaw)) + 0.60 * abs(math.sin(yaw)) + 0.06
    assert math.isclose(required_width_with_yaw(yaw, cfg), expected)
    assert required_width_with_yaw(yaw, cfg) > required_width_without_yaw(cfg)

    proposed_subtractive_formula = (
        0.36 * abs(math.cos(yaw)) - 0.60 * abs(math.sin(yaw)) - 0.06
    )
    assert proposed_subtractive_formula < 0.0


def test_yaw_prior_changes_width_and_decision_in_large_yaw_critical_case():
    cfg = DynamicFeasibilityConfig()
    obs = make_observation((0.10, 30, "clean", 0.0), 0, cfg)[0]
    full = TurnCommitFSM(
        strict_ablation="full", feasibility_cfg=cfg, corridor_type="straight_stress"
    )
    no_yaw = TurnCommitFSM(
        variant="no_yaw_prior", strict_ablation="no_yaw_prior",
        feasibility_cfg=cfg, corridor_type="straight_stress",
    )
    full.step(obs.copy())
    no_yaw.step(obs.copy())
    assert full.last_audit["W_required_with_yaw"] > full.last_audit["W_required_without_yaw"]
    assert full.last_audit["feasibility_mode"] != no_yaw.last_audit["feasibility_mode"]


def test_recover_when_prior_commitment_fails():
    cfg = DynamicFeasibilityConfig()
    agent = TurnCommitFSM(strict_ablation="point_estimate", feasibility_cfg=cfg)
    obs = _obs(width=0.60, heading=0.0)
    _, first_mode = agent.step(obs)
    assert first_mode == "TRAVERSE"
    failed = obs.copy()
    failed[15] = 0.60
    _, second_mode = agent.step(failed)
    assert second_mode == "RECOVER"
    assert agent.last_audit["prior_commitment_failed"]


def test_paired_stress_observation_and_random_draws_are_method_independent():
    cfg = DynamicFeasibilityConfig()
    case = mechanism_subset(45)[7]
    first_obs, first_signature = make_observation(case, 2, cfg)
    second_obs, second_signature = make_observation(case, 2, cfg)
    np.testing.assert_array_equal(first_obs, second_obs)
    assert first_signature == second_signature


def test_no_yaw_prior_changes_only_selected_required_width_path():
    cfg = DynamicFeasibilityConfig()
    obs = make_observation((0.05, 30, "severe", 0.10), 0, cfg)[0]
    full = DynamicFeasibilityEstimator(cfg).update(obs, use_yaw_prior=True)
    no_yaw = DynamicFeasibilityEstimator(cfg).update(obs, use_yaw_prior=False)
    invariant_fields = (
        "mu_D",
        "dynamic_sigma_delta",
        "ray_dispersion",
        "valid_depth_ratio",
        "dropout_ratio",
        "boundary_fitting_residual",
        "temporal_width_variation",
        "W_required_with_yaw",
        "W_required_without_yaw",
    )
    for field in invariant_fields:
        assert math.isclose(float(full[field]), float(no_yaw[field]))
    assert full["mu_W"] != no_yaw["mu_W"]


def test_no_yaw_prior_keeps_heading_alignment_controller_enabled():
    action, mode = fsm_action(
        _obs(heading=0.9, body_margin=0.5),
        variant="no_yaw_prior",
        strict_ablation="no_yaw_prior",
    )
    assert mode == "ALIGN"
    assert action[0] == 0.0
    assert abs(float(action[1])) > 0.0


def test_strict_path_does_not_accept_legacy_memory_hard_reject_override():
    class RejectMemory:
        def fsm_mode(self, obs, corridor_type):
            return FSMMode.REJECT

    obs = _obs(width=0.80, heading=0.0, body_margin=0.22)
    full = TurnCommitFSM(
        strict_ablation="full_dynamic_uncertainty", cross_mem=RejectMemory()
    )
    no_memory = TurnCommitFSM(
        strict_ablation="no_memory", cross_mem=RejectMemory()
    )
    # Repaired strict memory must enter through bounded correct_feasibility(),
    # not through the legacy terminal fsm_mode() hook.
    assert full.step(obs.copy())[1] != "REJECT"
    assert no_memory.step(obs.copy())[1] != "REJECT"


def test_legacy_full_path_does_not_opt_into_strict_selector():
    obs = _obs(width=0.65, heading=0.1, body_margin=0.12)
    action, mode = fsm_action(obs, variant="full")
    assert mode == "COMMIT"
    np.testing.assert_allclose(action, np.array([0.20, 0.0], dtype=np.float32))
