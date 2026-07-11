"""Belief-state mode-selection wrapper for DEGNAV-RL.

The wrapped environment still executes continuous velocity commands.  The RL
interface is intentionally higher level: the agent observes the compact
feasibility belief state and chooses one of four decision modes:

    0 = COMMIT, 1 = EXPLORE, 2 = RECOVER, 3 = REJECT

This keeps PPO/SAC/TD3 direct-control baselines separate from DEGNAV-RL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - legacy gym fallback
    import gym
    from gym import spaces

from ..models.belief_state import BeliefState, BeliefStateConfig
from ..models.policy import DecisionMode, controller


MODE_ORDER = (
    DecisionMode.COMMIT,
    DecisionMode.EXPLORE,
    DecisionMode.RECOVER,
    DecisionMode.REJECT,
)

VALID_BELIEF_ABLATIONS = (
    "full",
    "no_p_feas",
    "no_delta_var",
    "no_memory",
    "no_alignment",
    "geometry_only",
)

GEOMETRY_ONLY_FEATURE_NAMES = (
    "d_hat",
    "body_margin",
    "clearance_left",
    "clearance_right",
    "heading_error",
    "lateral_error",
)


def belief_observation_feature_names(ablation: str = "full") -> list[str]:
    """Return the policy observation names for a DEGNAV-RL ablation."""

    _validate_ablation(ablation)
    if ablation == "geometry_only":
        return list(GEOMETRY_ONLY_FEATURE_NAMES)
    return BeliefState.feature_names()


def _validate_ablation(ablation: str) -> None:
    if ablation not in VALID_BELIEF_ABLATIONS:
        raise ValueError(
            f"Unknown belief ablation {ablation!r}; "
            f"expected one of {', '.join(VALID_BELIEF_ABLATIONS)}"
        )


@dataclass(frozen=True)
class BeliefModeRewardConfig:
    strict_success_reward: float = 60.0
    correct_reject_reward: float = 25.0
    collision_penalty: float = 35.0
    near_collision_penalty: float = 3.0
    false_reject_penalty: float = 35.0
    timeout_penalty: float = 20.0
    no_progress_penalty: float = 1.0
    caution_penalty: float = 0.5
    step_penalty: float = 0.03
    progress_weight: float = 2.0
    oscillation_penalty: float = 0.5
    oscillation_switch_threshold: int = 2
    near_collision_threshold: float = 0.05
    strict_clearance_threshold: float = 0.05
    min_progress: float = 1e-3


class BeliefModeEnv(gym.Wrapper):
    """Gym/Gymnasium wrapper for DEGNAV-RL high-level mode selection."""

    def __init__(
        self,
        env=None,
        env_config: Mapping[str, Any] | None = None,
        belief_cfg: BeliefStateConfig | None = None,
        reward_cfg: BeliefModeRewardConfig | None = None,
        memory_risk: float = 0.0,
        ablation: str = "full",
    ) -> None:
        if env is None:
            env = _make_harder_env(env_config)
        super().__init__(env)

        _validate_ablation(ablation)
        self.belief_cfg = belief_cfg or BeliefStateConfig()
        self.reward_cfg = reward_cfg or BeliefModeRewardConfig()
        self.default_memory_risk = float(memory_risk)
        self.ablation = ablation
        self.observation_feature_names = belief_observation_feature_names(ablation)

        self.action_space = spaces.Discrete(len(MODE_ORDER))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.observation_feature_names),),
            dtype=np.float32,
        )

        self._last_raw_obs = None
        self._last_belief: BeliefState | None = None
        self._last_info: dict[str, Any] = {}
        self._prev_distance = None
        self._min_clearance = float("inf")
        self._prev_mode: DecisionMode | None = None
        self._mode_switch_count = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            raw_obs, info = result
        else:  # pragma: no cover - old gym compatibility
            raw_obs, info = result, {}

        info = dict(info or {})
        info["ablation"] = self.ablation
        belief = self._belief_from_raw_obs(raw_obs, info)
        self._last_raw_obs = raw_obs
        self._last_belief = belief
        self._last_info = info
        self._prev_distance = self._distance_from_obs(raw_obs)
        self._min_clearance = belief.body_margin
        self._prev_mode = None
        self._mode_switch_count = 0
        return self._policy_obs_from_belief(belief), info

    def step(self, mode_action):
        mode_index = self._mode_index(mode_action)
        mode = MODE_ORDER[mode_index]
        prev_belief = self._require_last_belief()
        passable = self._passability_label(self._last_info)

        if mode == DecisionMode.REJECT and passable is not None:
            correct_reject = float(not passable)
            false_reject = float(passable)
            reward = (
                self.reward_cfg.correct_reject_reward * correct_reject
                - self.reward_cfg.false_reject_penalty * false_reject
                - self.reward_cfg.step_penalty
            )
            info = self._augment_info(
                dict(self._last_info),
                mode_index,
                mode,
                prev_belief,
                reject=True,
                correct_reject=bool(correct_reject),
                false_reject=bool(false_reject),
                strict_success=False,
                near_collision=prev_belief.body_margin
                < self.reward_cfg.near_collision_threshold,
                terminated_by_reject=True,
            )
            return self._policy_obs_from_belief(prev_belief), float(reward), True, False, info

        action = self._continuous_action(mode, prev_belief)
        raw_result = self.env.step(action)
        if len(raw_result) == 5:
            raw_obs, _base_reward, terminated, truncated, info = raw_result
        else:  # pragma: no cover - old gym compatibility
            raw_obs, _base_reward, done, info = raw_result
            terminated, truncated = bool(done), False
        info = dict(info or {})

        belief = self._belief_from_raw_obs(raw_obs, info)
        self._last_raw_obs = raw_obs
        self._last_belief = belief
        self._last_info = info
        self._min_clearance = min(self._min_clearance, belief.body_margin)

        reward, strict_success, near_collision = self._mode_reward(
            mode=mode,
            prev_belief=prev_belief,
            belief=belief,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
            unknown_reject=(mode == DecisionMode.REJECT and passable is None),
        )
        self._update_mode_switches(mode)
        info = self._augment_info(
            info,
            mode_index,
            mode,
            belief,
            reject=mode == DecisionMode.REJECT,
            correct_reject=False,
            false_reject=False,
            strict_success=strict_success,
            near_collision=near_collision,
            terminated_by_reject=False,
        )
        return self._policy_obs_from_belief(belief), float(reward), bool(terminated), bool(truncated), info

    def _belief_from_raw_obs(self, raw_obs, info: Mapping[str, Any]) -> BeliefState:
        memory_risk = float(info.get("memory_risk", self.default_memory_risk))
        return BeliefState.from_obs(
            raw_obs,
            memory_risk=memory_risk,
            cfg=self.belief_cfg,
        )

    def _policy_obs_from_belief(self, belief: BeliefState) -> np.ndarray:
        """Apply the requested belief-state ablation to the policy observation.

        The stored ``BeliefState`` remains unmodified for rewards, logging, and
        the shared mode-conditioned controller.  The ablation tests only what
        information the learned high-level mode selector receives.
        """

        values = {name: float(value) for name, value in zip(BeliefState.feature_names(), belief.as_array())}
        if self.ablation == "no_p_feas":
            values["p_feas"] = 0.5
        elif self.ablation == "no_delta_var":
            values["delta_var"] = max(
                float(self.belief_cfg.min_variance),
                float(self.belief_cfg.sigma_d) ** 2 + float(self.belief_cfg.sigma_w) ** 2,
            )
        elif self.ablation == "no_memory":
            values["memory_risk"] = 0.0
        elif self.ablation == "no_alignment":
            values["heading_error"] = 0.0
            values["lateral_error"] = 0.0

        obs = np.asarray(
            [values[name] for name in self.observation_feature_names],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(obs)):
            raise ValueError(f"BeliefModeEnv produced non-finite observation for {self.ablation}")
        return obs

    def _continuous_action(
        self,
        mode: DecisionMode,
        belief: BeliefState,
    ) -> np.ndarray:
        clearance_margin = belief.clearance_right - belief.clearance_left
        lin, ang = controller(
            mode,
            local_goal_angle=belief.heading_error,
            clearance_margin=clearance_margin,
        )
        action = np.asarray([lin, ang], dtype=np.float32)

        # HarderNarrowPassageEnv uses physical velocity limits.  Habitat velocity
        # control may use normalized limits.  Clipping lets the same
        # mode-conditioned controller feed either low-level realization.
        low = getattr(self.env.action_space, "low", None)
        high = getattr(self.env.action_space, "high", None)
        if low is not None and high is not None:
            action = np.clip(action, low, high).astype(np.float32)
        return action

    def _mode_reward(
        self,
        mode: DecisionMode,
        prev_belief: BeliefState,
        belief: BeliefState,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        unknown_reject: bool,
    ) -> tuple[float, bool, bool]:
        cfg = self.reward_cfg
        success = bool(float(info.get("success", 0.0)) > 0.5)
        collision = bool(
            float(info.get("collision", belief.collision_flag)) > 0.5
            or belief.collision_flag > 0.5
        )
        near_collision = belief.body_margin < cfg.near_collision_threshold
        strict_success = bool(
            success
            and not collision
            and self._min_clearance >= cfg.strict_clearance_threshold
        )

        prev_dist = self._prev_distance
        dist = self._distance_from_obs(self._last_raw_obs)
        progress = 0.0 if prev_dist is None or dist is None else prev_dist - dist
        self._prev_distance = dist

        reward = -cfg.step_penalty
        reward += cfg.progress_weight * float(progress)
        reward += cfg.strict_success_reward * float(strict_success)
        reward -= cfg.collision_penalty * float(collision)

        if near_collision:
            deficit = cfg.near_collision_threshold - belief.body_margin
            scale = max(1.0, cfg.near_collision_threshold)
            reward -= cfg.near_collision_penalty * float(deficit / scale)

        if unknown_reject:
            reward -= cfg.caution_penalty
        if truncated:
            reward -= cfg.timeout_penalty
        elif not strict_success and not collision and progress < cfg.min_progress:
            reward -= cfg.no_progress_penalty

        if (
            self._prev_mode is not None
            and mode != self._prev_mode
            and self._mode_switch_count >= cfg.oscillation_switch_threshold
        ):
            reward -= cfg.oscillation_penalty

        return float(reward), strict_success, near_collision

    def _augment_info(
        self,
        info: dict[str, Any],
        mode_index: int,
        mode: DecisionMode,
        belief: BeliefState,
        reject: bool,
        correct_reject: bool,
        false_reject: bool,
        strict_success: bool,
        near_collision: bool,
        terminated_by_reject: bool,
    ) -> dict[str, Any]:
        risk = float(np.clip(1.0 - belief.p_feas + belief.memory_risk, 0.0, 1.0))
        info.update(
            {
                "mode": int(mode_index),
                "mode_name": mode.name,
                "ablation": self.ablation,
                "p_feas": float(belief.p_feas),
                "delta_mean": float(belief.delta_mean),
                "delta_var": float(belief.delta_var),
                "d_hat": float(belief.d_hat),
                "w_req_cons": float(belief.w_req_cons),
                "memory_risk": float(belief.memory_risk),
                "risk": risk,
                "reject": float(reject),
                "correct_reject": float(correct_reject),
                "false_reject": float(false_reject),
                "strict_success": float(strict_success),
                "near_collision": float(near_collision),
                "min_clearance": float(self._min_clearance),
                "body_margin": float(belief.body_margin),
                "terminated_by_reject": float(terminated_by_reject),
            }
        )
        return info

    def _passability_label(self, info: Mapping[str, Any]) -> bool | None:
        if "passable" in info:
            return bool(info["passable"])
        if hasattr(self.env, "is_passable"):
            return bool(getattr(self.env, "is_passable"))
        ctype = getattr(self.env, "corridor_type", None)
        if ctype is not None:
            ctype_name = getattr(ctype, "value", str(ctype)).lower()
            if "false_feasible" in ctype_name:
                return False
        return None

    def _distance_from_obs(self, raw_obs) -> float | None:
        try:
            arr = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.shape[0] <= 12 or not np.isfinite(arr[12]):
            return None
        return float(arr[12])

    def _mode_index(self, mode_action) -> int:
        try:
            mode_index = int(np.asarray(mode_action).reshape(-1)[0])
        except Exception as exc:
            raise ValueError(f"Invalid mode action {mode_action!r}") from exc
        if mode_index < 0 or mode_index >= len(MODE_ORDER):
            raise ValueError(
                f"Mode action must be in [0, {len(MODE_ORDER) - 1}], "
                f"got {mode_index}"
            )
        return mode_index

    def _require_last_belief(self) -> BeliefState:
        if self._last_belief is None:
            raise RuntimeError("BeliefModeEnv.step() called before reset()")
        return self._last_belief

    def _update_mode_switches(self, mode: DecisionMode) -> None:
        if self._prev_mode is not None and mode != self._prev_mode:
            self._mode_switch_count += 1
        self._prev_mode = mode


def _make_harder_env(env_config: Mapping[str, Any] | None):
    try:
        from procedural_env_v2 import HarderNarrowPassageEnv
    except ImportError as exc:  # pragma: no cover - import path guidance
        raise ImportError(
            "Could not import HarderNarrowPassageEnv. Run with "
            "examples/narrow_passage_rl on PYTHONPATH or pass an env instance."
        ) from exc
    return HarderNarrowPassageEnv(dict(env_config or {}))
