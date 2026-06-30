"""Failure memory bank interface."""

from dataclasses import dataclass

import numpy as np

from examples.narrow_passage_rl.cross_episode_memory import CrossEpisodeMemory, MemoryConfig
from examples.narrow_passage_rl.failure_memory import FailureMemoryConfig, PassageFailureMemory


@dataclass
class MemoryItem:
    geometry_embedding: np.ndarray
    scene_id: str
    passage_width: float
    width_body_ratio: float
    action_mode: str
    outcome: str
    min_clearance: float
    oscillation_count: int
    final_pose_error: float


def geometry_similarity(
    current: MemoryItem,
    memory: MemoryItem,
    w_cosine: float = 1.0,
    w_width_ratio: float = 0.5,
    w_entrance_angle: float = 0.25,
    w_clearance_asymmetry: float = 0.25,
) -> float:
    zc = current.geometry_embedding
    zm = memory.geometry_embedding
    denom = max(float(np.linalg.norm(zc) * np.linalg.norm(zm)), 1e-6)
    cosine = float(np.dot(zc, zm) / denom)
    return (
        w_cosine * cosine
        - w_width_ratio * abs(current.width_body_ratio - memory.width_body_ratio)
        - w_entrance_angle * abs(float(zc[6]) - float(zm[6]))
        - w_clearance_asymmetry * abs(float(zc[7]) - float(zm[7]))
    )


__all__ = [
    "CrossEpisodeMemory",
    "FailureMemoryConfig",
    "MemoryConfig",
    "MemoryItem",
    "PassageFailureMemory",
    "geometry_similarity",
]

