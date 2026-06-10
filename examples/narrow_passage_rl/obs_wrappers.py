#!/usr/bin/env python3

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        gym = None


ABLATION_MASKS = {
    "full": (),
    "no_failure": (15, 16, 17, 18),
    "no_clearance": (6, 7, 9),
}


class ObservationMaskWrapper(gym.ObservationWrapper):
    """Zero selected features while keeping the policy input dimension fixed."""

    def __init__(self, env, mask_indices=()):
        super().__init__(env)
        self.mask_indices = tuple(mask_indices)

    def observation(self, observation):
        obs = np.array(observation, dtype=np.float32, copy=True)
        if self.mask_indices:
            obs[list(self.mask_indices)] = 0.0
        return obs


def apply_ablation(env, ablation: str):
    if ablation not in ABLATION_MASKS:
        raise ValueError(
            f"Unknown ablation {ablation}. Expected one of {sorted(ABLATION_MASKS)}"
        )
    if ablation == "full":
        return env
    return ObservationMaskWrapper(env, ABLATION_MASKS[ablation])
