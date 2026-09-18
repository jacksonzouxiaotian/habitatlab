"""Strict 19-D Gym interface for a learned four-mode selector.

This wrapper deliberately coexists with :mod:`belief_mode_env`.  The latter is
kept for provenance of historical 14-D MLP-PPO results; new experiments use
this wrapper and therefore cannot silently change the meaning of "19-D".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

from ..models.policy import DecisionMode, controller
from ..selector.contract import (
    MODE_NAMES,
    Mode,
    assert_selector_contract,
    coerce_memory_context,
    coerce_observation,
)


REQUIRED_STEP_LOG_FIELDS = (
    "raw_policy_mode",
    "executed_mode",
    "mode_probability",
    "mode_override_reason",
    "ground_truth_feasible_for_training_only",
    "estimated_margin",
    "alignment_ready",
    "memory_support",
    "collision",
    "stuck",
    "timeout",
    "success",
)


@dataclass(frozen=True)
class SelectorRewardConfig:
    progress_scale: float = 1.0
    success: float = 10.0
    collision: float = -5.0
    stuck: float = -2.0
    oscillation: float = -0.05
    step_cost: float = -0.01
    correct_reject_infeasible: float = 3.0
    false_reject_feasible: float = -5.0
    recover_when_misaligned_or_stuck: float = 1.0
    unnecessary_recover: float = -0.5
    repeated_commit_after_supported_failure: float = -1.0
    alignment_heading_threshold: float = 0.20
    alignment_lateral_threshold: float = 0.10
    required_width: float = 0.42
    # EXPLORE must remain a probing mode, not a slower substitute for COMMIT.
    # This cap is in the wrapped procedural environment's physical m/s units.
    explore_speed_cap: float = 0.10


class FourModeSelectorEnv(gym.Wrapper):
    """Expose exactly 19 actor features and `Discrete(4)` high-level modes."""

    def __init__(
        self,
        env,
        reward_cfg: SelectorRewardConfig | None = None,
        memory_context: np.ndarray | None = None,
    ) -> None:
        super().__init__(env)
        self.reward_cfg = reward_cfg or SelectorRewardConfig()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
        )
        self._memory_context = coerce_memory_context(memory_context)
        self._last_obs: np.ndarray | None = None
        self._last_info: dict[str, Any] = {}
        self._previous_mode: Mode | None = None
        self._policy_probability = float("nan")
        self._policy_override_reason = "none"
        self._episode_return = 0.0
        self._selector_steps = 0

    @property
    def memory_context(self) -> np.ndarray:
        return self._memory_context.copy()

    @property
    def training_feasible_label(self) -> bool | None:
        """Simulator label for reward/supervision; never part of observation."""

        return self._ground_truth_feasible()

    def set_memory_context(self, memory_context) -> None:
        self._memory_context = coerce_memory_context(memory_context)

    def set_policy_metadata(
        self, *, probability: float | None = None, override_reason: str = "none"
    ) -> None:
        """Attach policy-side diagnostics to the next environment step."""

        self._policy_probability = (
            float("nan") if probability is None else float(probability)
        )
        self._policy_override_reason = str(override_reason or "none")

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            raw_obs, info = result
        else:  # pragma: no cover
            raw_obs, info = result, {}
        obs = coerce_observation(raw_obs)
        assert_selector_contract(obs, self.action_space)
        self._last_obs = obs
        self._last_info = dict(info or {})
        self._previous_mode = None
        self._episode_return = 0.0
        self._selector_steps = 0
        return obs, self._reset_info(self._last_info)

    def step(self, mode_action):
        if self._last_obs is None:
            raise RuntimeError("FourModeSelectorEnv.step() called before reset()")
        mode = self._coerce_mode(mode_action)
        previous_obs = self._last_obs
        feasible = self._ground_truth_feasible()
        alignment_ready = self._alignment_ready(previous_obs)
        memory_support = float(self._memory_context[1] + self._memory_context[2])

        if mode is Mode.REJECT:
            correct_reject = feasible is False
            false_reject = feasible is True
            reward = self.reward_cfg.step_cost
            if correct_reject:
                reward += self.reward_cfg.correct_reject_infeasible
            elif false_reject:
                reward += self.reward_cfg.false_reject_feasible
            info = self._base_diagnostics(
                dict(self._last_info), mode, feasible, previous_obs
            )
            info.update(
                {
                    "reject": 1.0,
                    "correct_reject": float(correct_reject),
                    "false_reject": float(false_reject),
                    "terminated_by_reject": 1.0,
                    "termination_reason": "correct_reject"
                    if correct_reject
                    else "false_reject"
                    if false_reject
                    else "reject_unknown",
                    "collision": 0.0,
                    "stuck": 0.0,
                    "timeout": 0.0,
                    "success": 0.0,
                    "alignment_ready": float(alignment_ready),
                    "memory_support": memory_support,
                    "reward_progress": 0.0,
                    "reward_total": float(reward),
                }
            )
            self._episode_return += float(reward)
            info["episode_return"] = self._episode_return
            self._selector_steps += 1
            info["selector_step"] = self._selector_steps
            self._previous_mode = mode
            self._assert_log_contract(info)
            self._clear_policy_metadata()
            return previous_obs.copy(), float(reward), True, False, info

        continuous_action = self._continuous_action(mode, previous_obs)
        result = self.env.step(continuous_action)
        if len(result) == 5:
            raw_obs, _base_reward, terminated, truncated, info = result
        else:  # pragma: no cover
            raw_obs, _base_reward, done, info = result
            terminated, truncated = bool(done), False
        obs = coerce_observation(raw_obs)
        info = dict(info or {})
        self._last_obs = obs
        self._last_info = info

        previous_distance = float(previous_obs[12])
        current_distance = float(obs[12])
        progress = previous_distance - current_distance
        success = bool(float(info.get("success", 0.0)) > 0.5)
        collision = bool(
            float(info.get("collision", obs[16])) > 0.5 or obs[16] > 0.5
        )
        stuck = bool(float(info.get("stuck", 0.0)) > 0.5 or obs[15] >= 0.7)
        timeout = bool(
            truncated
            or (
                bool(terminated)
                and not success
                and not collision
                and getattr(self.env, "step_count", 0)
                >= getattr(self.env, "max_steps", float("inf"))
            )
        )
        reward = self.reward_cfg.step_cost + self.reward_cfg.progress_scale * progress
        reward += self.reward_cfg.success * float(success)
        reward += self.reward_cfg.collision * float(collision)
        reward += self.reward_cfg.stuck * float(stuck)
        if self._previous_mode is not None and mode is not self._previous_mode:
            reward += self.reward_cfg.oscillation
        if mode is Mode.RECOVER:
            needs_recovery = (
                not alignment_ready
                or float(previous_obs[15]) >= 0.7
                or float(previous_obs[16]) > 0.5
            )
            reward += (
                self.reward_cfg.recover_when_misaligned_or_stuck
                if needs_recovery
                else self.reward_cfg.unnecessary_recover
            )
        if mode is Mode.COMMIT and float(self._memory_context[1]) > 0.0:
            reward += self.reward_cfg.repeated_commit_after_supported_failure

        diagnostics = self._base_diagnostics(info, mode, feasible, previous_obs)
        diagnostics.update(
            {
                "reject": 0.0,
                "correct_reject": 0.0,
                "false_reject": 0.0,
                "terminated_by_reject": 0.0,
                "termination_reason": self._termination_reason(
                    success, collision, stuck, timeout, bool(terminated), bool(truncated)
                ),
                "collision": float(collision),
                "stuck": float(stuck),
                "timeout": float(timeout),
                "success": float(success),
                "alignment_ready": float(alignment_ready),
                "memory_support": memory_support,
                "reward_progress": float(
                    self.reward_cfg.progress_scale * progress
                ),
                "reward_total": float(reward),
                "continuous_action_vx": float(continuous_action[0]),
                "continuous_action_wz": float(continuous_action[1]),
            }
        )
        self._episode_return += float(reward)
        self._selector_steps += 1
        diagnostics["episode_return"] = self._episode_return
        diagnostics["selector_step"] = self._selector_steps
        self._previous_mode = mode
        self._assert_log_contract(diagnostics)
        self._clear_policy_metadata()
        return obs, float(reward), bool(terminated), bool(truncated), diagnostics

    def _reset_info(self, info: Mapping[str, Any]) -> dict[str, Any]:
        output = dict(info)
        feasible = self._ground_truth_feasible()
        output.update(
            {
                "ground_truth_feasible_for_training_only": self._label_value(feasible),
                "estimated_margin": float(self._last_obs[8] - self.reward_cfg.required_width),
                "alignment_ready": float(self._alignment_ready(self._last_obs)),
                "memory_support": float(self._memory_context[1] + self._memory_context[2]),
            }
        )
        return output

    def _base_diagnostics(
        self,
        info: dict[str, Any],
        mode: Mode,
        feasible: bool | None,
        decision_observation: np.ndarray,
    ) -> dict[str, Any]:
        info.update(
            {
                "raw_policy_mode": mode.name,
                "raw_policy_mode_index": int(mode),
                "executed_mode": mode.name,
                "executed_mode_index": int(mode),
                "mode_name": mode.name,
                "mode_probability": self._policy_probability,
                "mode_override_reason": self._policy_override_reason,
                "ground_truth_feasible_for_training_only": self._label_value(feasible),
                "estimated_margin": float(
                    decision_observation[8] - self.reward_cfg.required_width
                ),
            }
        )
        return info

    def _continuous_action(self, mode: Mode, obs: np.ndarray) -> np.ndarray:
        decision_mode = DecisionMode[mode.name]
        linear, angular = controller(
            decision_mode,
            local_goal_angle=float(obs[10]),
            clearance_margin=float(obs[7] - obs[6]),
        )
        action = np.asarray([linear, angular], dtype=np.float32)
        if mode is Mode.EXPLORE:
            action[0] = min(float(action[0]), self.reward_cfg.explore_speed_cap)
        low = getattr(self.env.action_space, "low", None)
        high = getattr(self.env.action_space, "high", None)
        if low is not None and high is not None:
            action = np.clip(action, low, high).astype(np.float32)
        return action

    def _ground_truth_feasible(self) -> bool | None:
        if "passable" in self._last_info:
            return bool(self._last_info["passable"])
        if hasattr(self.env, "is_passable"):
            return bool(getattr(self.env, "is_passable"))
        return None

    def _alignment_ready(self, obs: np.ndarray) -> bool:
        return bool(
            abs(float(obs[10])) <= self.reward_cfg.alignment_heading_threshold
            and abs(float(obs[11])) <= self.reward_cfg.alignment_lateral_threshold
        )

    @staticmethod
    def _coerce_mode(action) -> Mode:
        try:
            return Mode(int(np.asarray(action).reshape(-1)[0]))
        except (ValueError, TypeError, IndexError) as exc:
            raise ValueError(f"mode action must be an integer in [0,3], got {action!r}") from exc

    @staticmethod
    def _label_value(value: bool | None) -> int:
        return -1 if value is None else int(value)

    @staticmethod
    def _termination_reason(
        success: bool,
        collision: bool,
        stuck: bool,
        timeout: bool,
        terminated: bool,
        truncated: bool,
    ) -> str:
        if success:
            return "success"
        if collision:
            return "collision"
        if timeout or truncated:
            return "timeout"
        if stuck and terminated:
            return "stuck"
        if terminated:
            return "environment_terminal"
        return "running"

    @staticmethod
    def _assert_log_contract(info: Mapping[str, Any]) -> None:
        missing = [name for name in REQUIRED_STEP_LOG_FIELDS if name not in info]
        assert not missing, f"selector step log is missing fields: {missing}"
        assert info["raw_policy_mode"] in MODE_NAMES
        assert info["executed_mode"] in MODE_NAMES

    def _clear_policy_metadata(self) -> None:
        self._policy_probability = float("nan")
        self._policy_override_reason = "none"


def make_procedural_selector_env(config: Mapping[str, Any] | None = None):
    """Late import keeps the package usable without the procedural benchmark."""

    from procedural_env_v2 import HarderNarrowPassageEnv

    return FourModeSelectorEnv(HarderNarrowPassageEnv(dict(config or {})))
