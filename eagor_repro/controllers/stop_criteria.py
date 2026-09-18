"""Independently switchable stop criteria for navigation attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class StopAssessment:
    should_stop: bool
    mode: str
    reason: str
    target_depth_m: Optional[float]
    target_pixels: int
    valid_target_depth_pixels: int


def robust_target_depth(
    depth: np.ndarray | None,
    target_mask: np.ndarray | None,
    min_valid_depth_pixels: int = 5,
) -> tuple[Optional[float], int]:
    """Return median metric depth over valid target pixels."""

    if depth is None or target_mask is None:
        return None, 0
    values = np.asarray(depth, dtype=np.float64).squeeze()
    mask = np.asarray(target_mask, dtype=bool).squeeze()
    if values.shape != mask.shape:
        raise ValueError(
            f"Depth/mask shape mismatch: depth={values.shape}, mask={mask.shape}"
        )
    valid = mask & np.isfinite(values) & (values > 0.0)
    count = int(valid.sum())
    if count < int(min_valid_depth_pixels):
        return None, count
    return float(np.median(values[valid])), count


class StopCriterion:
    """Area, area+depth, or GT-distance stopping behind one interface."""

    VALID_MODES = {"area", "depth", "area_depth", "oracle_distance"}

    def __init__(
        self,
        mode: str = "area",
        confidence_threshold: float = 0.25,
        area_fraction_threshold: float = 0.08,
        stop_distance_m: float = 1.0,
        success_distance_m: float = 0.1,
        min_target_pixels: int = 20,
        min_valid_depth_pixels: int = 5,
    ) -> None:
        normalized = str(mode).lower()
        if normalized not in self.VALID_MODES:
            raise ValueError(f"Unknown stop mode {mode!r}; expected {self.VALID_MODES}")
        self.mode = normalized
        self.confidence_threshold = float(confidence_threshold)
        self.area_fraction_threshold = float(area_fraction_threshold)
        self.stop_distance_m = float(stop_distance_m)
        self.success_distance_m = float(success_distance_m)
        self.min_target_pixels = int(min_target_pixels)
        self.min_valid_depth_pixels = int(min_valid_depth_pixels)

    @property
    def oracle_upper_bound(self) -> bool:
        return self.mode == "oracle_distance"

    def assess(
        self,
        *,
        target_visible: bool,
        confidence: float,
        target_area_fraction: float,
        target_mask: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        oracle_distance_m: float | None = None,
    ) -> StopAssessment:
        target_pixels = int(np.asarray(target_mask, dtype=bool).sum()) if target_mask is not None else 0
        target_depth, valid_depth_pixels = robust_target_depth(
            depth, target_mask, self.min_valid_depth_pixels
        )

        if self.mode == "oracle_distance":
            should_stop = bool(
                oracle_distance_m is not None
                and np.isfinite(float(oracle_distance_m))
                # Habitat's Success measure uses a strict inequality.
                and float(oracle_distance_m) < self.success_distance_m
            )
            return StopAssessment(
                should_stop,
                self.mode,
                "oracle_distance_stop" if should_stop else "oracle_distance_continue",
                target_depth,
                target_pixels,
                valid_depth_pixels,
            )

        common_ready = bool(
            target_visible and float(confidence) >= self.confidence_threshold
        )
        if self.mode == "area":
            should_stop = bool(
                common_ready
                and float(target_area_fraction) >= self.area_fraction_threshold
            )
            reason = "perception_area_stop" if should_stop else "area_continue"
        else:
            should_stop = bool(
                common_ready
                and target_pixels >= self.min_target_pixels
                and target_depth is not None
                and target_depth <= self.stop_distance_m
            )
            reason = "perception_depth_stop" if should_stop else "depth_continue"
        return StopAssessment(
            should_stop,
            self.mode,
            reason,
            target_depth,
            target_pixels,
            valid_depth_pixels,
        )
