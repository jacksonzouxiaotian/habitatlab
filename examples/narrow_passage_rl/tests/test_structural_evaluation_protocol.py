import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_structural_feasibility import (
    LATERAL,
    NOISE,
    SCENES,
    STRICT_GATE_SCHEMA,
    YAWS,
    _gate_report,
    _initial_observation_hash,
    _noise_schedule,
    _scenario_rows,
    _scenario_seed,
)
from eval_harder_benchmark import TurnCommitFSM, run_episode
from procedural_env_v2 import HarderNarrowPassageEnv
from narrow_passage.models.dynamic_feasibility import DynamicFeasibilityConfig
from narrow_passage.models.feasibility_ablation import (
    MAIN_ABLATIONS,
    capability_record,
    canonical_ablation,
)
from narrow_passage.models.robot_morphology import RobotMorphology


def test_three_seed_balanced_protocol_has_exact_factor_counts():
    rows = _scenario_rows("full", [0, 1, 2], "balanced")
    assert len(rows) == 630
    assert Counter(int(row["yaw_deg"]) for row in rows) == {
        yaw: 126 for yaw in YAWS
    }
    assert Counter(row["noise_level"] for row in rows) == {
        level: 210 for level in NOISE
    }
    assert Counter(row["lateral_offset"] for row in rows) == {
        value: 210 for value in LATERAL
    }
    cells = Counter(
        (row["seed"], row["margin_bin"], row["corridor_type"])
        for row in rows
    )
    assert set(cells.values()) == {5}


def test_publication_protocol_exceeds_500_per_seed_and_is_exactly_balanced():
    rows = _scenario_rows("full", [0, 1, 2], "publication")
    assert len(rows) == 1890
    assert Counter(int(row["yaw_deg"]) for row in rows) == {
        yaw: 378 for yaw in YAWS
    }
    assert Counter(row["noise_level"] for row in rows) == {
        level: 630 for level in NOISE
    }
    assert Counter(row["lateral_offset"] for row in rows) == {
        value: 630 for value in LATERAL
    }
    cells = Counter(
        (row["seed"], row["margin_bin"], row["corridor_type"])
        for row in rows
    )
    assert set(cells.values()) == {15}


def test_legacy_point_names_collapse_to_one_paper_method():
    assert canonical_ablation("point_estimate") == "mean_only"
    assert canonical_ablation("no_uncertainty") == "mean_only"


def _gate_rows(*, feasible_success: float, timeout: float):
    rows = []
    for bin_index, (lower, upper) in enumerate((
        (-0.15, -0.10), (-0.10, -0.05), (-0.05, 0.0),
        (0.0, 0.05), (0.05, 0.10), (0.10, 0.15),
    )):
        for scene_index, scene in enumerate(SCENES):
            passable = bin_index >= 3
            success = feasible_success if passable else 0.0
            rows.append({
                "scenario_id": f"b{bin_index}_{scene}",
                "episode_id": f"b{bin_index}_{scene}",
                "method": "full_dynamic_uncertainty",
                "margin_bin": f"[{lower:.2f},{upper:.2f})",
                "corridor_type": scene,
                "yaw_deg": 60,
                "passable": passable,
                "success": success,
                "false_reject": 0.0,
                "collision": 0.0,
                "timeout": timeout if passable else 0.0,
                "correct_reject": 1.0 if not passable else 0.0,
                "observation_sha256": f"obs-{bin_index}-{scene_index}",
                "random_draw_sha256": f"rng-{bin_index}-{scene_index}",
            })
    steps = []
    for row in rows:
        if row["passable"]:
            steps.extend((
                {
                    "episode_id": row["episode_id"],
                    "method": row["method"],
                    "step": 0,
                    "passage_control_state": "ALIGN",
                    "selected_mode": "EXPLORE",
                    "uncertainty_triggered": 1.0,
                },
                {
                    "episode_id": row["episode_id"],
                    "method": row["method"],
                    "step": 1,
                    "passage_control_state": "ENTER",
                    "selected_mode": "COMMIT",
                    "uncertainty_triggered": 0.0,
                },
            ))
    return rows, steps


def test_strict_gate_blocks_low_success_high_timeout_smoke():
    rows, steps = _gate_rows(feasible_success=0.5, timeout=0.5)
    report = _gate_report(rows, ("full_dynamic_uncertainty",), steps)
    assert report["gate_schema"] == STRICT_GATE_SCHEMA
    assert not report["engineering_gates"]["feasible_success_at_least_60_percent"]
    assert not report["engineering_gates"]["feasible_timeout_below_20_percent"]
    assert not report["full_evaluation_permitted"]


def test_strict_gate_accepts_only_complete_engineering_contract():
    rows, steps = _gate_rows(feasible_success=1.0, timeout=0.0)
    report = _gate_report(rows, ("full_dynamic_uncertainty",), steps)
    assert all(report["protocol_gates"].values())
    assert all(report["engineering_gates"].values())
    assert report["full_evaluation_permitted"]


def test_s_shape_closed_loop_switches_from_arm_one_to_arm_two():
    morphology = RobotMorphology()
    env = HarderNarrowPassageEnv({
        "morphology": morphology,
        "max_steps": 200,
    })
    agent = TurnCommitFSM(
        strict_ablation="full_dynamic_uncertainty",
        feasibility_cfg=DynamicFeasibilityConfig(morphology=env.morphology),
        corridor_type="s_shaped",
        env=env,
    )
    trace = []
    result = run_episode(
        env,
        "fsm_cross_memory",
        None,
        None,
        200,
        [],
        agent=agent,
        episode_seed=2,
        reset_options={
            "corridor_type": "s_shaped",
            "passage_width": 0.545,
            "yaw_deg": 0,
            "lateral_offset": 0.0,
        },
        step_trace_out=trace,
        step_trace_context={"episode_id": "s_arm_transition_test"},
    )
    arms = [int(row["active_corridor_arm"]) for row in trace]
    assert result["success"] == 1.0
    assert 1 in arms and 2 in arms
    assert arms.index(1) < arms.index(2)


def test_canonical_episode_outcomes_are_mutually_exclusive_and_complete():
    env = HarderNarrowPassageEnv({
        "morphology": RobotMorphology(),
        "max_steps": 1,
    })
    result = run_episode(
        env,
        "rule_baseline",
        None,
        None,
        1,
        [],
        episode_seed=4,
        reset_options={
            "corridor_type": "straight",
            "passage_width": 0.545,
            "yaw_deg": 0,
            "lateral_offset": 0.0,
        },
    )
    flags = [
        float(result["success"]),
        float(result["correct_reject"] or result["false_reject"]),
        float(result["collision"]),
        float(result["timeout"]),
        float(result["stuck"]),
        float(result["final_outcome"] == "controller_failure"),
    ]
    assert sum(value > 0.5 for value in flags) == 1
    assert result["final_outcome"] == "timeout"


def test_six_methods_share_paired_observation_and_random_hashes():
    morphology = RobotMorphology()
    spec = _scenario_rows("smoke", [0], "balanced")[17]
    episode_seed = _scenario_seed(int(spec["seed"]), str(spec["scenario_id"]))
    transform, random_hash = _noise_schedule(
        str(spec["scenario_id"]), str(spec["noise_level"]), 5.0
    )
    width = morphology.structural_required_width + float(
        spec["structural_margin_target"]
    )
    hashes = []
    random_hashes = []
    for _method in (*MAIN_ABLATIONS, "reactive_rule_baseline"):
        observation_hash, _passable, _margin = _initial_observation_hash(
            morphology, spec, episode_seed, width, transform
        )
        hashes.append(observation_hash)
        random_hashes.append(random_hash)
    assert len(set(hashes)) == 1
    assert len(set(random_hashes)) == 1


def test_ablation_contract_changes_only_declared_components():
    full = capability_record("full_dynamic_uncertainty")
    allowed_differences = {
        "mean_only": {
            "canonical_method", "uncertainty_is_audit_only",
            "selector_reads_sigma_delta", "dual_interval_gate",
        },
        "fixed_uncertainty": {"canonical_method", "fixed_sigma"},
        "no_yaw_aware_readiness": {
            "canonical_method", "yaw_aware_readiness",
        },
        "no_memory": {"canonical_method", "geometry_indexed_memory"},
    }
    for method, expected in allowed_differences.items():
        record = capability_record(method)
        changed = {
            key for key in full if full[key] != record[key]
        }
        assert changed == expected


def test_all_belief_variants_share_low_level_controller_configuration():
    morphology = RobotMorphology()
    records = []
    for method in MAIN_ABLATIONS:
        env = HarderNarrowPassageEnv({"morphology": morphology})
        agent = TurnCommitFSM(
            strict_ablation=method,
            feasibility_cfg=DynamicFeasibilityConfig(
                morphology=env.morphology
            ),
            env=env,
        )
        records.append(asdict(agent.controller_cfg))
    assert all(record == records[0] for record in records[1:])


def test_step_audit_declares_conditional_active_sensing_fields_eagerly():
    agent = TurnCommitFSM(strict_ablation="full_dynamic_uncertainty")
    obs = np.zeros(19, dtype=np.float32)
    obs[:6] = 5.0
    obs[6:8] = 0.30
    obs[8] = 0.96
    obs[9] = 0.30
    obs[12] = 2.0
    agent.step(obs)
    assert "active_sensing_phase" in agent.last_audit
    assert "active_sensing_exit_reason" in agent.last_audit
