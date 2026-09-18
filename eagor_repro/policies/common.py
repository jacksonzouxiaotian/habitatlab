from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DirectionPrediction:
    direction_xyz: np.ndarray
    azimuth: float
    elevation: float
    confidence: float
    belief_heatmap: np.ndarray
    valid: bool
    method: str

