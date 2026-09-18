import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from narrow_passage.envs.four_mode_selector_env import (
    REQUIRED_STEP_LOG_FIELDS,
    FourModeSelectorEnv,
)
from narrow_passage.selector.contract import MODE_NAMES, Mode, assert_selector_contract
from narrow_passage.selector.expert import training_rule_mode
from narrow_passage.selector.model import FourModeGRUPolicy


def observation(*, heading=0.0, lateral=0.0, width=0.7, stuck=0.0, collision=0.0):
    obs = np.zeros(19, dtype=np.float32)
    obs[:6] = 1.0
    obs[6:8] = width / 2
    obs[8] = width
    obs[9] = width / 2 - 0.18
    obs[10] = heading
    obs[11] = lateral
    obs[12] = 1.0
    obs[15] = stuck
    obs[16] = collision
    return obs


class FakeContinuousEnv(gym.Env):
    def __init__(self, passable=True, terminal="running"):
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(19,), dtype=np.float32)
        self.action_space = spaces.Box(
            np.asarray([-0.15, -0.8], dtype=np.float32),
            np.asarray([0.35, 0.8], dtype=np.float32),
        )
        self.is_passable = passable
        self.terminal = terminal
        self.max_steps = 1
        self.step_count = 0
        self.last_action = None
        self.obs = observation(heading=0.3)

    def reset(self, **kwargs):
        self.step_count = 0
        self.last_action = None
        self.obs = observation(heading=0.3)
        return self.obs.copy(), {}

    def step(self, action):
        self.step_count += 1
        self.last_action = np.asarray(action, dtype=np.float32)
        self.obs[12] -= 0.1
        info = {"passable": self.is_passable, "success": 0.0, "collision": 0.0, "stuck": 0.0}
        terminated = False
        if self.terminal == "collision":
            self.obs[16] = 1.0
            info["collision"] = 1.0
            terminated = True
        elif self.terminal == "timeout":
            terminated = True
        elif self.terminal == "success":
            info["success"] = 1.0
            terminated = True
        return self.obs.copy(), 0.0, terminated, False, info


@pytest.mark.parametrize(
    "mode, expected_sign",
    [(Mode.COMMIT, 1), (Mode.EXPLORE, 1), (Mode.RECOVER, -1)],
)
def test_three_motion_modes_reach_distinct_controller_actions(mode, expected_sign):
    base = FakeContinuousEnv()
    env = FourModeSelectorEnv(base)
    obs, _ = env.reset()
    assert_selector_contract(obs, env.action_space)
    env.set_policy_metadata(probability=0.7)
    _obs, _reward, _terminated, _truncated, info = env.step(int(mode))
    assert np.sign(base.last_action[0]) == expected_sign
    if mode is Mode.EXPLORE:
        assert base.last_action[0] == pytest.approx(0.10)
    assert info["raw_policy_mode"] == mode.name
    assert info["executed_mode"] == mode.name
    assert info["mode_probability"] == pytest.approx(0.7)
    assert all(field in info for field in REQUIRED_STEP_LOG_FIELDS)


def test_reject_is_action_three_and_terminates_without_environment_step():
    base = FakeContinuousEnv(passable=False)
    env = FourModeSelectorEnv(base)
    obs, _ = env.reset()
    assert env.action_space.n == 4
    next_obs, reward, terminated, truncated, info = env.step(3)
    assert np.array_equal(next_obs, obs)
    assert terminated and not truncated
    assert base.step_count == 0
    assert reward > 0
    assert info["termination_reason"] == "correct_reject"
    assert info["timeout"] == 0.0


def test_false_reject_has_negative_reward_and_distinct_label():
    env = FourModeSelectorEnv(FakeContinuousEnv(passable=True))
    env.reset()
    _, reward, terminated, _, info = env.step(Mode.REJECT)
    assert terminated
    assert reward < 0
    assert info["false_reject"] == 1.0
    assert info["termination_reason"] == "false_reject"


def test_collision_timeout_success_and_reject_are_distinct():
    reasons = []
    for terminal in ("collision", "timeout", "success"):
        env = FourModeSelectorEnv(FakeContinuousEnv(terminal=terminal))
        env.reset()
        reasons.append(env.step(Mode.COMMIT)[4]["termination_reason"])
    reject_env = FourModeSelectorEnv(FakeContinuousEnv(passable=False))
    reject_env.reset()
    reasons.append(reject_env.step(Mode.REJECT)[4]["termination_reason"])
    assert reasons == ["collision", "timeout", "success", "correct_reject"]


def test_startup_assertions_reject_bad_observations():
    env = FourModeSelectorEnv(FakeContinuousEnv())
    with pytest.raises(AssertionError):
        assert_selector_contract(np.zeros(18), env.action_space)
    bad = np.zeros(19)
    bad[2] = np.nan
    with pytest.raises(AssertionError):
        assert_selector_contract(bad, env.action_space)


def test_training_teacher_can_emit_all_four_modes():
    memory = np.zeros(4, dtype=np.float32)
    assert training_rule_mode(observation(), memory, feasible_label=True) is Mode.COMMIT
    assert training_rule_mode(observation(width=0.45), memory, feasible_label=True) is Mode.EXPLORE
    assert training_rule_mode(observation(heading=0.8), memory, feasible_label=True) is Mode.RECOVER
    assert training_rule_mode(observation(), memory, feasible_label=False) is Mode.REJECT


def test_gru_actor_shape_and_memory_is_separate():
    policy = FourModeGRUPolicy(hidden_size=32, memory_dim=4)
    obs = torch.zeros(5, 2, 19)
    memory = torch.zeros(5, 2, 4)
    starts = torch.zeros(5, 2)
    starts[0] = 1
    logits, values, hidden = policy.forward_sequence(
        obs, policy.initial_hidden(2), memory, starts
    )
    assert logits.shape == (5, 2, len(MODE_NAMES))
    assert values.shape == (5, 2)
    assert hidden.shape == (1, 2, 32)
    assert policy.obs_dim == 19
    assert policy.actor.in_features == 36
