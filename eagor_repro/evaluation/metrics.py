"""Direction and navigation metrics for EAGOR and matched baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from eagor_repro.spherical.spherical_grid import direction_to_angles


def angular_error_deg(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    predicted = np.asarray(predicted, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    denominator = float(np.linalg.norm(predicted) * np.linalg.norm(ground_truth))
    if denominator <= 1e-12:
        return float("nan")
    cosine = np.clip(float(np.dot(predicted, ground_truth)) / denominator, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


@dataclass
class DirectionMetricsAccumulator:
    """Accumulate per-step direction diagnostics.

    Temporal consistency is defined here as the absolute error between the
    predicted and GT inter-frame angular displacement, in degrees per step:
    ``abs(angle(pred_t,pred_t-1)-angle(gt_t,gt_t-1))``.
    """

    seam_threshold_deg: float = 30.0
    rows: List[Dict[str, Any]] = field(default_factory=list)
    previous_prediction: Optional[np.ndarray] = None
    previous_ground_truth: Optional[np.ndarray] = None

    def add(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray,
        confidence: float,
        target_visible: bool,
        success: bool = False,
    ) -> Dict[str, Any]:
        error = angular_error_deg(predicted, ground_truth)
        gt_azimuth, _ = direction_to_angles(ground_truth)
        distance_to_seam = 180.0 - abs(float(np.rad2deg(gt_azimuth)))
        is_seam = distance_to_seam <= self.seam_threshold_deg
        temporal = float("nan")
        if self.previous_prediction is not None and self.previous_ground_truth is not None:
            predicted_delta = angular_error_deg(predicted, self.previous_prediction)
            ground_truth_delta = angular_error_deg(ground_truth, self.previous_ground_truth)
            temporal = abs(predicted_delta - ground_truth_delta)
        row = {
            "angular_error_deg": error,
            "confidence": float(confidence),
            "target_visible": bool(target_visible),
            "success": bool(success),
            "is_seam": bool(is_seam),
            "temporal_consistency_deg_per_step": temporal,
        }
        self.rows.append(row)
        self.previous_prediction = np.asarray(predicted, dtype=np.float64).copy()
        self.previous_ground_truth = np.asarray(ground_truth, dtype=np.float64).copy()
        return row

    def summary(self) -> Dict[str, float]:
        if not self.rows:
            return {
                "mae_deg": float("nan"),
                "temporal_consistency_deg_per_step": float("nan"),
                "seam_error_deg": float("nan"),
                "occlusion_error_deg": float("nan"),
            }

        def mean_for(key: str, predicate=lambda row: True) -> float:
            values = [
                float(row[key])
                for row in self.rows
                if predicate(row) and np.isfinite(float(row[key]))
            ]
            return float(np.mean(values)) if values else float("nan")

        return {
            "mae_deg": mean_for("angular_error_deg"),
            "temporal_consistency_deg_per_step": mean_for(
                "temporal_consistency_deg_per_step"
            ),
            "seam_error_deg": mean_for(
                "angular_error_deg", lambda row: bool(row["is_seam"])
            ),
            "occlusion_error_deg": mean_for(
                "angular_error_deg", lambda row: not bool(row["target_visible"])
            ),
            "mean_confidence": mean_for("confidence"),
        }


def navigation_summary(
    success: bool,
    spl: float,
    steps: int,
    path_length: float,
    distance_to_success: float,
    collision_count: int,
    stop_called: bool,
    stop_correct: bool,
    seam_crossing_success: bool,
    latency_ms: List[float],
) -> Dict[str, Any]:
    return {
        "success": bool(success),
        "success_rate": float(bool(success)),
        "spl": float(spl),
        "steps": int(steps),
        "path_length_m": float(path_length),
        "distance_to_success_m": float(distance_to_success),
        "collision_count": int(collision_count),
        "stop_called": bool(stop_called),
        "stop_correct": bool(stop_correct),
        "seam_crossing_success": bool(seam_crossing_success),
        "average_inference_update_latency_ms": (
            float(np.mean(latency_ms)) if latency_ms else float("nan")
        ),
    }

