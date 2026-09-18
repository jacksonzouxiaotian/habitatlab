"""Compact debug-video renderer with ERP, likelihood, belief and trajectory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np

from eagor_repro.spherical.spherical_grid import sphere_to_erp


class VideoStreamWriter:
    """Append debug frames without retaining a full 500-step episode in RAM."""

    def __init__(self, path: str | Path, fps: int = 10) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(
            self.path, fps=fps, codec="libx264", quality=7
        )

    def append(self, frame: np.ndarray) -> None:
        self._writer.append_data(_as_rgb(frame))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _as_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.nan_to_num(array, nan=0.0)
        if float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _heatmap(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, np.float32))
    values -= float(values.min(initial=0.0))
    values /= max(float(values.max(initial=0.0)), 1e-6)
    bgr = cv2.applyColorMap((values * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _telemetry_number(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}f}" if np.isfinite(number) else "-"


def _mark_direction(
    image: np.ndarray,
    direction: Optional[np.ndarray],
    color: tuple[int, int, int],
) -> None:
    if direction is None:
        return
    height, width = image.shape[:2]
    u, v = sphere_to_erp(direction, width, height)
    point = (int(round(float(u))) % width, int(np.clip(round(float(v)), 0, height - 1)))
    cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, 18, 2)


def _trajectory_panel(
    trajectory: Sequence[Sequence[float]], height: int, width: int
) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, np.uint8)
    if not trajectory:
        return canvas
    points = np.asarray(trajectory, np.float64)[:, [0, 2]]
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high - low, 0.5)
    normalized = (points - low) / span
    pixels = np.column_stack(
        (20 + normalized[:, 0] * (width - 40), height - 20 - normalized[:, 1] * (height - 40))
    ).astype(np.int32)
    if len(pixels) > 1:
        cv2.polylines(canvas, [pixels], False, (40, 40, 40), 2)
    cv2.circle(canvas, tuple(pixels[0]), 5, (0, 180, 0), -1)
    cv2.circle(canvas, tuple(pixels[-1]), 5, (220, 40, 40), -1)
    return canvas


def render_frame(
    panorama: np.ndarray,
    likelihood: np.ndarray,
    belief: np.ndarray,
    predicted_direction: Optional[np.ndarray],
    ground_truth_direction: Optional[np.ndarray],
    action: str,
    confidence: float,
    angular_error: float,
    trajectory: Sequence[Sequence[float]] = (),
    perspective: Optional[np.ndarray] = None,
    planned_heading: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    panorama = _as_rgb(panorama).copy()
    _mark_direction(panorama, predicted_direction, (255, 40, 40))
    _mark_direction(panorama, ground_truth_direction, (40, 255, 40))
    if planned_heading is not None:
        planned_direction = np.asarray(
            [np.cos(planned_heading), np.sin(planned_heading), 0.0], np.float64
        )
        _mark_direction(panorama, planned_direction, (40, 170, 255))
    canvas = np.full((720, 1280, 3), 14, np.uint8)
    forward = panorama if perspective is None else _as_rgb(perspective)
    panels = (
        ("FORWARD RGB", _fit_panel(forward, 600, 280), 20, 48),
        (
            "360 ERP | red=belief green=GT blue=planner",
            _fit_panel(panorama, 620, 280),
            640,
            48,
        ),
        ("CURRENT LIKELIHOOD", _fit_panel(_heatmap(likelihood), 300, 240), 20, 382),
        ("POLICY BELIEF", _fit_panel(_heatmap(belief), 300, 240), 340, 382),
        (
            "X-Z TRAJECTORY",
            _trajectory_panel(trajectory, 240, 300),
            660,
            382,
        ),
    )
    for title, panel, x, y in panels:
        cv2.putText(
            canvas,
            title,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel

    metadata = metadata or {}
    telemetry = [
        "TELEMETRY",
        f"scene: {metadata.get('scene', '-')}",
        f"episode: {metadata.get('episode', '-')}",
        f"step: {metadata.get('step', '-')}",
        f"target: {metadata.get('target', '-')}",
        f"policy: {metadata.get('policy', '-')}",
        f"visible: {metadata.get('target_visible', '-')}",
        f"action: {action}",
        f"reason: {metadata.get('action_reason', '-')}",
        f"planner: {metadata.get('planning_mode', '-')}",
        f"status: {metadata.get('planner_status', '-')}",
        f"goal/selected: {_telemetry_number(metadata.get('target_heading_deg'))}/{_telemetry_number(metadata.get('selected_heading_deg'))} deg",
        f"clearance: {_telemetry_number(metadata.get('clearance_m'))} m",
        f"target depth: {_telemetry_number(metadata.get('target_depth_m'))} m",
        f"stop mode: {metadata.get('stop_mode', '-')}",
        f"failure: {metadata.get('failure_mode', '-')}",
        f"confidence: {confidence:.3f}",
        f"angular error: {angular_error:.2f} deg",
    ]
    cv2.rectangle(canvas, (980, 382), (1260, 702), (42, 42, 42), -1)
    for index, line in enumerate(telemetry):
        cv2.putText(
            canvas,
            line,
            (990, 401 + 18 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    oracle_message = (
        "ORACLE UPPER BOUND: diagnostic GT may enter perception/direction/stop."
        if bool(metadata.get("oracle_upper_bound", False))
        else "GT marker is evaluation-only and never enters control."
    )
    cv2.putText(
        canvas,
        oracle_message,
        (20, 690),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_video(frames: Iterable[np.ndarray], path: str | Path, fps: int = 10) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoStreamWriter(path, fps)
    try:
        for frame in frames:
            writer.append(frame)
    finally:
        writer.close()
    return path


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Aspect-preserving resize onto a dark fixed-size panel."""

    source = _as_rgb(image)
    scale = min(width / source.shape[1], height / source.shape[0])
    resized = cv2.resize(
        source,
        (
            max(1, int(round(source.shape[1] * scale))),
            max(1, int(round(source.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    panel = np.full((height, width, 3), 18, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _search_trajectory_panel(
    trajectory: Sequence[Sequence[float]],
    width: int,
    height: int,
    world_bounds: tuple[Sequence[float], Sequence[float]],
    patrol_waypoints: Sequence[Sequence[float]] = (),
    active_waypoint: Optional[Sequence[float]] = None,
    target_positions: Sequence[Sequence[float]] = (),
) -> np.ndarray:
    """Draw a stable x-z map whose scale does not jump during an episode."""

    canvas = np.full((height, width, 3), 238, np.uint8)
    low = np.asarray(world_bounds[0], np.float64)[[0, 2]]
    high = np.asarray(world_bounds[1], np.float64)[[0, 2]]
    span = np.maximum(high - low, 0.5)

    def pixels(points: Sequence[Sequence[float]]) -> np.ndarray:
        values = np.asarray(points, np.float64).reshape(-1, 3)[:, [0, 2]]
        normalized = (values - low) / span
        return np.column_stack(
            (
                18 + normalized[:, 0] * (width - 36),
                height - 18 - normalized[:, 1] * (height - 36),
            )
        ).astype(np.int32)

    cv2.rectangle(canvas, (8, 8), (width - 9, height - 9), (170, 170, 170), 1)
    if patrol_waypoints:
        waypoint_pixels = pixels(patrol_waypoints)
        if len(waypoint_pixels) > 1:
            cv2.polylines(canvas, [waypoint_pixels], False, (130, 130, 130), 1)
        for index, point in enumerate(waypoint_pixels):
            cv2.circle(canvas, tuple(point), 4, (75, 105, 220), -1)
            cv2.putText(
                canvas,
                str(index + 1),
                (int(point[0] + 5), int(point[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                (50, 50, 80),
                1,
                cv2.LINE_AA,
            )
    if target_positions:
        for point in pixels(target_positions):
            cv2.drawMarker(
                canvas, tuple(point), (40, 170, 40), cv2.MARKER_STAR, 14, 2
            )
    if active_waypoint is not None:
        point = pixels([active_waypoint])[0]
        cv2.circle(canvas, tuple(point), 9, (255, 145, 30), 2)
    if trajectory:
        path_pixels = pixels(trajectory)
        if len(path_pixels) > 1:
            cv2.polylines(canvas, [path_pixels], False, (45, 45, 45), 2)
        cv2.circle(canvas, tuple(path_pixels[0]), 5, (40, 180, 40), -1)
        cv2.circle(canvas, tuple(path_pixels[-1]), 6, (220, 45, 45), -1)
    return canvas


def render_object_search_frame(
    panorama: np.ndarray,
    perspective: np.ndarray,
    target_mask: np.ndarray,
    likelihood: np.ndarray,
    belief: np.ndarray,
    predicted_direction: Optional[np.ndarray],
    ground_truth_direction: Optional[np.ndarray],
    trajectory: Sequence[Sequence[float]],
    world_bounds: tuple[Sequence[float], Sequence[float]],
    patrol_waypoints: Sequence[Sequence[float]],
    active_waypoint: Optional[Sequence[float]],
    target_positions: Sequence[Sequence[float]],
    telemetry: Dict[str, Any],
) -> np.ndarray:
    """Render a readable 1280x720 long-horizon object-search dashboard."""

    canvas = np.full((720, 1280, 3), 12, np.uint8)
    panorama_rgb = _as_rgb(panorama).copy()
    mask = np.asarray(target_mask, bool)
    if mask.shape == panorama_rgb.shape[:2] and mask.any():
        tint = np.asarray((35, 255, 75), np.float32)
        panorama_rgb[mask] = np.clip(
            0.4 * panorama_rgb[mask].astype(np.float32) + 0.6 * tint,
            0,
            255,
        ).astype(np.uint8)
    _mark_direction(panorama_rgb, predicted_direction, (255, 45, 45))
    _mark_direction(panorama_rgb, ground_truth_direction, (45, 255, 45))

    erp_title = "360 ERP | green=target red-cross=SH direction"
    if ground_truth_direction is not None:
        erp_title += " green-cross=GT(eval)"
    trajectory_title = "X-Z TRAJECTORY | blue=coverage route red=agent"
    if target_positions:
        trajectory_title += " star=GT(eval)"
    panels = (
        ("FORWARD RGB", _fit_panel(perspective, 600, 300), 20, 70),
        (erp_title, _fit_panel(panorama_rgb, 620, 300), 640, 70),
        ("CURRENT LIKELIHOOD", _fit_panel(_heatmap(likelihood), 390, 220), 20, 420),
        ("SH BELIEF FIELD", _fit_panel(_heatmap(belief), 390, 220), 435, 420),
        (
            trajectory_title,
            _search_trajectory_panel(
                trajectory,
                410,
                220,
                world_bounds,
                patrol_waypoints,
                active_waypoint,
                target_positions,
            ),
            850,
            420,
        ),
    )
    for title, panel, x, y in panels:
        canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
        cv2.putText(
            canvas,
            title,
            (x, y - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (205, 215, 225),
            1,
            cv2.LINE_AA,
        )

    phase = str(telemetry.get("phase", "unknown")).upper()
    target = str(telemetry.get("target", "object"))
    step = int(telemetry.get("step", 0))
    action = str(telemetry.get("action", ""))
    phase_color = (255, 185, 45) if phase == "EXPLORE" else (65, 235, 105)
    cv2.putText(
        canvas,
        f"EAGOR LONG-HORIZON OBJECT SEARCH  |  target: {target}",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (235, 240, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{phase}  step {step:03d}  action={action}",
        (790, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        phase_color,
        2,
        cv2.LINE_AA,
    )
    footer_parts = [
        f"visible={bool(telemetry.get('visible', False))}  "
        f"pixels={int(telemetry.get('target_pixels', 0))}  "
        f"conf={float(telemetry.get('confidence', 0.0)):.3f}  "
        f"az={float(telemetry.get('azimuth_deg', 0.0)):+.1f} deg  "
        f"path={float(telemetry.get('path_length_m', 0.0)):.1f} m"
    ]
    if "angular_error_deg" in telemetry:
        footer_parts.append(
            f"  error={float(telemetry['angular_error_deg']):.1f} deg (eval)"
        )
    if "distance_m" in telemetry:
        footer_parts.append(
            f"  distance={float(telemetry['distance_m']):.1f} m (eval)"
        )
    footer = "".join(footer_parts)
    cv2.putText(
        canvas,
        footer,
        (20, 686),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.53,
        (220, 225, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Oracle semantic perception: debugging upper bound, NOT zero-shot ObjectNav",
        (20, 710),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (120, 175, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
