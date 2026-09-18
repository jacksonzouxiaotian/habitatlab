"""Robust local clearance estimates from a metric-depth ERP."""

from __future__ import annotations

import numpy as np

from eagor_repro.spherical.spherical_grid import wrap_to_pi


def sector_clearance(
    depth_erp: np.ndarray,
    heading_rad: float,
    sector_width_deg: float = 10.0,
    percentile: float = 25.0,
    vertical_band_deg: float = 35.0,
) -> tuple[float, int]:
    """Estimate obstacle clearance in a circular ERP angular sector.

    The statistic uses a horizontal sector and a band around the horizon,
    rejecting NaN, infinity, and non-positive depth.  A low percentile is more
    robust than a single minimum while remaining sensitive to partial obstacles.
    """

    depth = np.asarray(depth_erp, dtype=np.float64).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW ERP depth, got {depth.shape}")
    height, width = depth.shape
    heading = float(wrap_to_pi(heading_rad))
    center_u = ((heading + np.pi) / (2.0 * np.pi) * width) % width
    half_columns = max(1, int(np.ceil(width * sector_width_deg / 360.0)))
    center_column = int(np.floor(center_u)) % width
    columns = (center_column + np.arange(-half_columns, half_columns + 1)) % width

    half_rows = max(1, int(np.ceil(height * vertical_band_deg / 180.0)))
    center_row = height // 2
    row_start = max(0, center_row - half_rows)
    row_stop = min(height, center_row + half_rows + 1)
    values = depth[row_start:row_stop][:, columns].reshape(-1)
    valid = values[np.isfinite(values) & (values > 0.0)]
    if valid.size == 0:
        return 0.0, 0
    return float(np.percentile(valid, percentile)), int(valid.size)
