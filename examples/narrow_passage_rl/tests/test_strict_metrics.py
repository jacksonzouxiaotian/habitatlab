import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from narrow_passage.metrics.strict_metrics import (
    StrictMetricConfig,
    compute_strict_metrics,
)


def test_successful_safe_episode_is_strict_success():
    metrics = compute_strict_metrics(
        success=1.0,
        collision=0.0,
        stuck=0.0,
        min_clearance=0.10,
        cfg=StrictMetricConfig(strict_clearance_threshold=0.05),
    )
    assert metrics["strict_success"] == 1.0
    assert metrics["clearance_safe"] == 1.0
    assert metrics["success_but_unsafe"] == 0.0


def test_successful_unsafe_episode_is_success_but_unsafe():
    metrics = compute_strict_metrics(
        success=1.0,
        collision=0.0,
        stuck=0.0,
        min_clearance=0.01,
        cfg=StrictMetricConfig(strict_clearance_threshold=0.05),
    )
    assert metrics["strict_success"] == 0.0
    assert metrics["clearance_safe"] == 0.0
    assert metrics["success_but_unsafe"] == 1.0


def test_collision_episode_is_not_strict_success():
    metrics = compute_strict_metrics(
        success=1.0,
        collision=1.0,
        stuck=0.0,
        min_clearance=0.10,
        cfg=StrictMetricConfig(strict_clearance_threshold=0.05),
    )
    assert metrics["strict_success"] == 0.0


def test_min_clearance_below_near_threshold_is_near_collision():
    metrics = compute_strict_metrics(
        success=0.0,
        collision=0.0,
        stuck=0.0,
        min_clearance=0.04,
        cfg=StrictMetricConfig(near_collision_threshold=0.05),
    )
    assert metrics["near_collision"] == 1.0
