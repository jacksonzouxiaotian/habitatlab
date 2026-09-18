"""Deterministic balanced procedural scenarios for selector training only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import MEMORY_DIM


SAMPLING_RATIOS = {
    "feasible": 0.35,
    "infeasible": 0.25,
    "near_boundary": 0.20,
    "recoverable_misalignment": 0.10,
    "transient_or_memory_cases": 0.10,
}


@dataclass(frozen=True)
class ScenarioSpec:
    category: str
    reset_options: dict[str, float | str]
    memory_context: np.ndarray


class BalancedScenarioSampler:
    """Sample the requested five strata without leaking labels to the actor."""

    _categories = tuple(SAMPLING_RATIOS)
    _probabilities = np.asarray(tuple(SAMPLING_RATIOS.values()), dtype=np.float64)

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample(self) -> ScenarioSpec:
        category = str(self.rng.choice(self._categories, p=self._probabilities))
        return self.sample_category(category)

    def sample_category(self, category: str) -> ScenarioSpec:
        rng = self.rng
        memory = np.zeros(MEMORY_DIM, dtype=np.float32)
        if category == "feasible":
            ctype = str(rng.choice(["straight", "l_shaped", "s_shaped", "narrow_entry", "narrow_exit", "asymmetric"]))
            options = {
                "corridor_type": ctype,
                "passage_width": float(rng.uniform(0.55, 0.85)),
                "yaw_deg": float(rng.uniform(-12.0, 12.0)),
                "lateral_offset": float(rng.uniform(-0.06, 0.06)),
            }
        elif category == "infeasible":
            options = {
                "corridor_type": "false_feasible",
                "passage_width": float(rng.uniform(0.55, 0.80)),
                "yaw_deg": float(rng.uniform(-15.0, 15.0)),
                "lateral_offset": float(rng.uniform(-0.08, 0.08)),
            }
        elif category == "near_boundary":
            options = {
                "corridor_type": "straight",
                "passage_width": float(rng.uniform(0.38, 0.47)),
                "yaw_deg": float(rng.uniform(-20.0, 20.0)),
                "lateral_offset": float(rng.uniform(-0.08, 0.08)),
            }
        elif category == "recoverable_misalignment":
            options = {
                "corridor_type": str(rng.choice(["straight", "narrow_entry", "l_shaped"])),
                "passage_width": float(rng.uniform(0.65, 0.90)),
                "yaw_deg": float(rng.choice([-1.0, 1.0]) * rng.uniform(35.0, 70.0)),
                "lateral_offset": float(rng.choice([-1.0, 1.0]) * rng.uniform(0.14, 0.24)),
            }
        elif category == "transient_or_memory_cases":
            infeasible_history = bool(rng.random() < 0.5)
            options = {
                "corridor_type": "false_feasible" if infeasible_history else "asymmetric",
                "passage_width": float(rng.uniform(0.52, 0.72)),
                "yaw_deg": float(rng.uniform(-25.0, 25.0)),
                "lateral_offset": float(rng.uniform(-0.12, 0.12)),
            }
            failures = int(rng.integers(1, 6))
            successes = int(rng.integers(0, 2 if infeasible_history else 5))
            total = max(1, failures + successes)
            memory = np.asarray(
                [
                    float(rng.uniform(0.75, 1.0)),
                    float(failures),
                    float(successes),
                    float(failures / total),
                ],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"unknown sampling category: {category}")
        return ScenarioSpec(category, options, memory)


def deterministic_category(episode_index: int) -> str:
    """Exact 20-episode cycle matching the configured proportions."""

    cycle = (
        ["feasible"] * 7
        + ["infeasible"] * 5
        + ["near_boundary"] * 4
        + ["recoverable_misalignment"] * 2
        + ["transient_or_memory_cases"] * 2
    )
    return cycle[int(episode_index) % len(cycle)]
