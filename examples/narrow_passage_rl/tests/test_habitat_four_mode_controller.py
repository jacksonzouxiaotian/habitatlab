import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narrow_passage.selector.contract import Mode  # noqa: E402
from narrow_passage.selector.habitat_controller import (  # noqa: E402
    mode_action,
    privileged_teacher_mode,
)
from collect_habitat_e2e_teacher import actor_observation  # noqa: E402


def features():
    output = np.zeros(19, dtype=np.float32)
    output[8] = 0.7
    output[9] = 0.50
    return output


def test_teacher_exposes_all_modes():
    value = features()
    assert privileged_teacher_mode(value, 0.0) is Mode.COMMIT
    assert privileged_teacher_mode(value, 0.6) is Mode.EXPLORE
    value[9] = 0.29
    assert privileged_teacher_mode(value, 0.0) is Mode.EXPLORE
    value[9] = 0.50
    value[16] = 1.0
    assert privileged_teacher_mode(value, 0.0) is Mode.RECOVER
    assert privileged_teacher_mode(value, 0.0, morphology_infeasible=True) is Mode.REJECT
    value[16] = 0.0
    assert privileged_teacher_mode(value, 0.0, morphology_infeasible=True) is Mode.COMMIT


def test_controller_motion_signs_and_reject_terminal():
    commit = mode_action(Mode.COMMIT, 0.0, 0.0)
    explore = mode_action(Mode.EXPLORE, 0.0, 0.0)
    recover = mode_action(Mode.RECOVER, 0.0, 0.0)
    assert commit["action_args"]["linear_velocity"] > explore["action_args"]["linear_velocity"]
    assert recover["action_args"]["linear_velocity"] < -0.4
    assert mode_action(Mode.REJECT, 0.0, 0.0) is None


def test_actor_observation_snapshots_mutable_action_history():
    outcome = np.zeros(7, dtype=np.float32)
    observations = {
        "depth": np.ones((8, 8, 1), dtype=np.float32),
        "pointgoal_with_gps_compass": np.asarray([1.0, 0.0], dtype=np.float32),
    }
    snapshot = actor_observation(
        observations,
        outcome,
        8,
        "depth",
        include_privileged=False,
    )
    outcome[5] = 1.0
    assert snapshot["action_outcome"][5] == 0.0
    assert not np.shares_memory(snapshot["action_outcome"], outcome)
