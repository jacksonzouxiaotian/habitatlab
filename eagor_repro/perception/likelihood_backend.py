"""Common interface for semantic and VLM likelihood providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LikelihoodResult:
    likelihood: np.ndarray
    target_visible: bool
    raw_detections: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    backend_name: str = "unknown"
    confidence_calibrated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.likelihood = np.nan_to_num(
            np.asarray(self.likelihood, dtype=np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        self.likelihood = np.clip(self.likelihood, 0.0, 1.0)


class LikelihoodBackend(ABC):
    @abstractmethod
    def infer(
        self,
        panorama: np.ndarray,
        target_query: str,
        observations: Optional[Dict[str, Any]] = None,
        simulator_state: Any = None,
    ) -> LikelihoodResult:
        """Infer an ``H x W`` target-direction likelihood in ``[0,1]``."""

