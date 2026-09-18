"""Output isolation and per-episode structured logging."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


CSV_FIELDS = [
    "episode_id",
    "scene_id",
    "step",
    "target_query",
    "agent_position_x",
    "agent_position_y",
    "agent_position_z",
    "agent_yaw",
    "gt_target_azimuth",
    "gt_target_elevation",
    "pred_target_azimuth",
    "pred_target_elevation",
    "angular_error_deg",
    "confidence",
    "target_visible",
    "target_area_fraction",
    "target_depth_m",
    "valid_target_depth_pixels",
    "distance_to_goal_m",
    "stop_mode",
    "stop_requested",
    "action",
    "action_reason",
    "planning_mode",
    "target_heading",
    "selected_heading",
    "planner_status",
    "planner_clearance_m",
    "blocked_candidates",
    "recovery_triggered",
    "collision",
    "likelihood_backend",
    "policy_method",
    "update_mode",
    "decode_mode",
    "oracle_upper_bound",
    "failure_mode",
    "latency_ms",
]


class ResultWriter:
    def __init__(self, output_root: str | Path) -> None:
        self.root = Path(output_root)
        self.raw_dir = self.root / "raw"
        self.summary_dir = self.root / "summaries"
        self.video_dir = self.root / "videos"
        self.plot_dir = self.root / "plots"
        self.log_dir = self.root / "logs"
        for directory in (
            self.raw_dir,
            self.summary_dir,
            self.video_dir,
            self.plot_dir,
            self.log_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def write_episode_csv(
        self, episode_id: str, rows: Iterable[Dict[str, Any]]
    ) -> Path:
        path = self.raw_dir / f"episode_{episode_id}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        return path

    def write_summary(self, episode_id: str, summary: Dict[str, Any]) -> Path:
        path = self.summary_dir / f"episode_{episode_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=True)
        return path

    def write_batch_summary(self, name: str, summaries: List[Dict[str, Any]]) -> Path:
        path = self.summary_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2, sort_keys=True, allow_nan=True)
        return path
