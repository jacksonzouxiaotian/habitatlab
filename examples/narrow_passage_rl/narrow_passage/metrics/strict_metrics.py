"""Clearance-aware strict success metrics.

The defaults preserve the current Habitat SB3 diagnostic semantics:

- ``strict_clearance_threshold = -0.02`` allows small negative HM3D depth
  artifacts at valid navmesh positions.
- ``near_collision_threshold = 0.05`` marks near-boundary traversal for safety
  reporting.

Callers may override these thresholds explicitly, but should report the chosen
values with their table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrictMetricConfig:
    strict_clearance_threshold: float = -0.02
    near_collision_threshold: float = 0.05
    success_key: str = "success"
    collision_key: str = "collision"
    stuck_key: str = "stuck"


def compute_strict_metrics(
    success: float,
    collision: float,
    stuck: float,
    min_clearance: float,
    cfg: StrictMetricConfig | None = None,
) -> dict[str, float]:
    """Compute clearance-aware safety metrics from episode-level outcomes."""

    cfg = cfg or StrictMetricConfig()
    success_bool = float(success) > 0.5
    collision_bool = float(collision) > 0.5
    stuck_bool = float(stuck) > 0.5
    clearance_safe = float(min_clearance) >= cfg.strict_clearance_threshold
    strict_success = (
        success_bool
        and not collision_bool
        and not stuck_bool
        and clearance_safe
    )
    return {
        "strict_success": float(strict_success),
        "clearance_safe": float(clearance_safe),
        "success_but_unsafe": float(success_bool and not strict_success),
        "near_collision": float(
            float(min_clearance) < cfg.near_collision_threshold
        ),
    }
