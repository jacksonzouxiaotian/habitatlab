"""Single source of truth for strict DEGNav robot geometry.

The procedural-v2 simulator and strict feasibility controller intentionally
share the same immutable object.  Physical collision uses the bare oriented
rectangle; decision margins add ``safety_margin`` on both corridor walls.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class RobotMorphology:
    """Rectangular footprint and decision clearance, in metres."""

    width: float = 0.36
    length: float = 0.60
    safety_margin: float = 0.03

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.length <= 0.0:
            raise ValueError("Robot width and length must be positive")
        if self.safety_margin < 0.0:
            raise ValueError("Robot safety_margin must be non-negative")

    @property
    def half_width(self) -> float:
        return 0.5 * self.width

    @property
    def half_length(self) -> float:
        return 0.5 * self.length

    @property
    def structural_required_width(self) -> float:
        return self.width + 2.0 * self.safety_margin

    @property
    def turning_required_width(self) -> float:
        """Conservative square-junction width for arbitrary in-place yaw."""

        return math.hypot(self.width, self.length) + 2.0 * self.safety_margin

    def projected_width(self, yaw_error: float) -> float:
        """Exact lateral projection of the OBB at a relative yaw angle."""

        yaw = abs(float(yaw_error))
        return (
            self.width * abs(math.cos(yaw))
            + self.length * abs(math.sin(yaw))
        )

    def pose_required_width(self, yaw_error: float) -> float:
        return self.projected_width(yaw_error) + 2.0 * self.safety_margin

    def record(self) -> dict[str, float]:
        return asdict(self)


def morphology_from_config(config: object | None) -> RobotMorphology:
    """Normalize an env/controller config value to one immutable object."""

    if config is None:
        return RobotMorphology()
    if isinstance(config, RobotMorphology):
        return config
    if isinstance(config, Mapping):
        return RobotMorphology(
            width=float(config.get("width", 0.36)),
            length=float(config.get("length", 0.60)),
            safety_margin=float(config.get("safety_margin", 0.03)),
        )
    raise TypeError(
        "morphology must be RobotMorphology, mapping, or None; "
        f"got {type(config).__name__}"
    )
