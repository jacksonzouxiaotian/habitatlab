#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np


def make_episode_trace():
    return {
        "min_clearance": float("inf"),
        "min_body_margin": float("inf"),
        "max_heading_error": 0.0,
        "max_lateral_offset": 0.0,
        "high_risk_steps": 0,
        "steps": 0,
    }


def update_episode_trace(trace, obs, risk_thresholds=None):
    risk_thresholds = risk_thresholds or {}
    min_clearance = min(float(obs[6]), float(obs[7]))
    body_margin = float(obs[9])
    heading_error = abs(float(obs[10]))
    lateral_offset = abs(float(obs[11]))
    trace["min_clearance"] = min(trace["min_clearance"], min_clearance)
    trace["min_body_margin"] = min(trace["min_body_margin"], body_margin)
    trace["max_heading_error"] = max(trace["max_heading_error"], heading_error)
    trace["max_lateral_offset"] = max(trace["max_lateral_offset"], lateral_offset)
    trace["steps"] += 1

    clearance_threshold = float(risk_thresholds.get("clearance", 0.10))
    lateral_threshold = float(risk_thresholds.get("lateral", 0.30))
    heading_threshold = float(risk_thresholds.get("heading", 0.65))
    if (
        min_clearance < clearance_threshold
        or lateral_offset > lateral_threshold
        or heading_error > heading_threshold
    ):
        trace["high_risk_steps"] += 1


def finalize_episode_info(info, trace, near_collision_threshold=0.05):
    info = dict(info)
    steps = max(1, int(trace.get("steps", 0)))
    min_clearance = trace.get("min_clearance", float("inf"))
    if not np.isfinite(min_clearance):
        min_clearance = float(info.get("min_clearance", 0.0))
    info["min_clearance"] = min(float(info.get("min_clearance", min_clearance)), min_clearance)
    info["trajectory_min_clearance"] = min_clearance
    info["min_body_margin"] = trace.get("min_body_margin", min_clearance)
    info["max_heading_error"] = trace.get("max_heading_error", 0.0)
    info["max_lateral_offset"] = trace.get("max_lateral_offset", 0.0)
    info["high_risk_steps"] = int(trace.get("high_risk_steps", 0))
    info["high_risk_fraction"] = float(info["high_risk_steps"]) / float(steps)
    info["near_collision"] = float(min_clearance < near_collision_threshold)
    info["narrow_passage"] = float(float(info.get("passage_width", 1.0)) <= 0.65)
    return info


def _mean(stats, key):
    values = [float(s.get(key, 0.0)) for s in stats]
    return float(np.mean(values)) if values else 0.0


def _add_group_summary(summary, stats, name, prefix=""):
    if not stats:
        return
    summary.update(
        {
            f"{prefix}{name}_episodes": len(stats),
            f"{prefix}{name}_success_rate": round(_mean(stats, "success"), 4),
            f"{prefix}{name}_collision_rate": round(_mean(stats, "collision"), 4),
            f"{prefix}{name}_near_collision_rate": round(
                _mean(stats, "near_collision"), 4
            ),
            f"{prefix}{name}_avg_min_clearance": round(
                _mean(stats, "min_clearance"), 4
            ),
        }
    )


def summarize_episode_stats(stats, prefix=""):
    summary = {
        f"{prefix}episodes": len(stats),
        f"{prefix}success_rate": round(_mean(stats, "success"), 4),
        f"{prefix}collision_rate": round(_mean(stats, "collision"), 4),
        f"{prefix}stuck_rate": round(_mean(stats, "stuck"), 4),
        f"{prefix}avg_min_clearance": round(_mean(stats, "min_clearance"), 4),
        f"{prefix}near_collision_rate": round(_mean(stats, "near_collision"), 4),
        f"{prefix}avg_high_risk_fraction": round(
            _mean(stats, "high_risk_fraction"), 4
        ),
    }
    false_stats = [s for s in stats if float(s.get("false_feasible", 0.0)) > 0.5]
    normal_stats = [s for s in stats if float(s.get("false_feasible", 0.0)) <= 0.5]
    narrow_stats = [s for s in stats if float(s.get("narrow_passage", 0.0)) > 0.5]
    wide_stats = [s for s in stats if float(s.get("narrow_passage", 0.0)) <= 0.5]
    high_risk_stats = [
        s for s in stats if float(s.get("high_risk_fraction", 0.0)) > 0.25
    ]
    near_collision_stats = [
        s for s in stats if float(s.get("near_collision", 0.0)) > 0.5
    ]
    _add_group_summary(summary, false_stats, "false_feasible", prefix)
    _add_group_summary(summary, normal_stats, "normal", prefix)
    _add_group_summary(summary, narrow_stats, "narrow", prefix)
    _add_group_summary(summary, wide_stats, "wide", prefix)
    _add_group_summary(summary, high_risk_stats, "high_risk", prefix)
    _add_group_summary(summary, near_collision_stats, "near_collision_group", prefix)
    return summary


def print_summary(summary):
    for key, value in summary.items():
        print(f"{key}: {value}")


def write_summary_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
