import math
from typing import Iterable, Sequence

from sensor_msgs.msg import LaserScan

from narrow_memory_core.features import PassageFeatures


def _valid_ranges(scan: LaserScan, values: Iterable[float]) -> list[float]:
    return [
        float(v)
        for v in values
        if math.isfinite(float(v)) and scan.range_min < float(v) < scan.range_max
    ]


def _sector(scan: LaserScan, start_deg: float, end_deg: float) -> Sequence[float]:
    start = math.radians(start_deg)
    end = math.radians(end_deg)
    if start > end:
        start, end = end, start
    ranges = []
    for idx, value in enumerate(scan.ranges):
        angle = scan.angle_min + idx * scan.angle_increment
        if start <= angle <= end:
            ranges.append(value)
    return ranges


def _percentile_min(scan: LaserScan, values: Iterable[float], default: float) -> float:
    valid = sorted(_valid_ranges(scan, values))
    if not valid:
        return default
    idx = max(0, min(len(valid) - 1, int(0.10 * (len(valid) - 1))))
    return valid[idx]


def features_from_scan(
    scan: LaserScan,
    current_vx: float,
    current_wz: float,
    previous_action_vx: float,
    previous_action_wz: float,
    stuck_score: float,
    lateral_offset: float = 0.0,
    heading_error: float = 0.0,
) -> PassageFeatures:
    default = min(5.0, scan.range_max if math.isfinite(scan.range_max) else 5.0)
    left = _percentile_min(scan, _sector(scan, 55.0, 110.0), default)
    right = _percentile_min(scan, _sector(scan, -110.0, -55.0), default)
    front = _percentile_min(scan, _sector(scan, -15.0, 15.0), default)
    clearance_left = min(left, front)
    clearance_right = min(right, front)
    return PassageFeatures(
        clearance_left=clearance_left,
        clearance_right=clearance_right,
        passage_width=left + right,
        heading_error=heading_error,
        lateral_offset=lateral_offset,
        current_vx=current_vx,
        current_wz=current_wz,
        stuck_score=stuck_score,
        previous_action_vx=previous_action_vx,
        previous_action_wz=previous_action_wz,
    )
