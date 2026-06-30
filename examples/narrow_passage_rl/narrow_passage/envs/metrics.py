"""Metric definitions for narrow-passage navigation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NarrowPassageMetricSpec:
    name: str
    description: str
    higher_is_better: bool


METRICS = [
    NarrowPassageMetricSpec("success_rate", "Goal reached and stop called", True),
    NarrowPassageMetricSpec("collision_rate", "Any collision or body intersection", False),
    NarrowPassageMetricSpec("near_collision_rate", "Clearance below safety threshold", False),
    NarrowPassageMetricSpec("oscillation_count", "Alternating steering near entrance", False),
    NarrowPassageMetricSpec("stuck_rate", "Timeout or low progress", False),
    NarrowPassageMetricSpec("recovery_success", "Successful recovery after Recover mode", True),
    NarrowPassageMetricSpec("min_clearance", "Minimum body clearance", True),
    NarrowPassageMetricSpec("time_to_goal", "Steps or seconds until success", False),
    NarrowPassageMetricSpec("spl", "Success weighted by path efficiency", True),
    NarrowPassageMetricSpec("repeated_failure_rate", "Re-entry into historical failed passage", False),
]


def metric_names() -> list[str]:
    return [m.name for m in METRICS]

