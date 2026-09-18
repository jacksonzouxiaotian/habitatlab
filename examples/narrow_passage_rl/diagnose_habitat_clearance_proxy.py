#!/usr/bin/env python3
"""Recompute historical Habitat clearance diagnostics in physical metres."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "narrow_passage_rl"
DEFAULT_INPUT = RESULTS / "habitat_stress_validation.csv"
METHODS = {"apf_gap", "fsm_full", "fsm_no_heading_alignment"}


def as_bool(row: dict[str, str], key: str) -> bool:
    return float(row.get(key, 0.0)) > 0.5


def corrected_margin(old_margin: float, *, depth_min: float, depth_max: float, proxy_radius: float) -> float:
    """Undo ``old_margin=d_norm-proxy_radius`` and return metric margin."""

    normalized_depth = old_margin + proxy_radius
    metric_depth = normalized_depth * (depth_max - depth_min) + depth_min
    return metric_depth - proxy_radius


def summarize(rows: list[dict[str, str]], args) -> list[dict]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("method") in METHODS and row.get("stress") == "nominal":
            groups[(row["stress"], row["method"])].append(row)
    output = []
    for (stress, method), values in sorted(groups.items()):
        old = [float(row["min_clearance"]) for row in values]
        new = [
            corrected_margin(
                value,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                proxy_radius=args.proxy_radius,
            )
            for value in old
        ]
        sim_aligned = [value + (args.proxy_radius - args.sim_radius) for value in new]
        old_strict = mean(float(as_bool(row, "strict_success")) for row in values)
        new_strict = mean(
            float(
                as_bool(row, "success")
                and not as_bool(row, "collision")
                and not as_bool(row, "stuck")
                and margin >= args.strict_threshold
            )
            for row, margin in zip(values, new)
        )
        output.append(
            {
                "stress": stress,
                "method": method,
                "episodes": len(values),
                "success_rate": mean(float(as_bool(row, "success")) for row in values),
                "historical_strict_success_rate": old_strict,
                "corrected_strict_success_rate": new_strict,
                "historical_avg_min_body_margin_mixed_units": mean(old),
                "corrected_avg_min_body_margin_m": mean(new),
                "sim_radius_aligned_avg_min_margin_m": mean(sim_aligned),
                "historical_near_collision_rate": mean(float(as_bool(row, "near_collision")) for row in values),
                "corrected_near_collision_rate": mean(float(value < args.near_threshold) for value in new),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--depth-min", type=float, default=0.0)
    parser.add_argument("--depth-max", type=float, default=10.0)
    parser.add_argument("--proxy-radius", type=float, default=0.18)
    parser.add_argument("--sim-radius", type=float, default=0.10)
    parser.add_argument("--strict-threshold", type=float, default=-0.02)
    parser.add_argument("--near-threshold", type=float, default=0.05)
    args = parser.parse_args()
    if args.depth_max <= args.depth_min:
        raise ValueError("depth-max must be greater than depth-min")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or RESULTS / "audits" / f"habitat_clearance_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    with args.input_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    summary = summarize(rows, args)
    write_csv(out_dir / "clearance_recalculation.csv", summary)

    md = [
        "# Supplementary S4 — Habitat clearance proxy calibration audit",
        "",
        "HabitatSimDepthSensor 在当前 PointNav 配置中输出归一化深度 `d_norm=(d_m-0)/(10-0)`，而历史 narrow-passage geometry 直接把 `d_norm` 当作米并计算 `b_old=d_norm-0.18`。离线修正式为：",
        "",
        "`b_metric = (b_old + 0.18) × 10 - 0.18 = 10 b_old + 1.62 m`。",
        "",
        "此换算对 nominal 条件是精确的代数单位修复；它没有重放轨迹，也不改变 Success/Collision/Stuck。",
        "",
        "| Method | N | Success | Strict SR (old) | Strict SR (metric) | Avg min margin (old) | Avg min margin (metric) | Near collision (old) | Near collision (metric) |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        md.append(
            "| {method} | {episodes} | {success} | {old_strict} | {new_strict} | {old_margin:.3f} | {new_margin:.3f} | {old_near} | {new_near} |".format(
                method=row["method"],
                episodes=row["episodes"],
                success=pct(row["success_rate"]),
                old_strict=pct(row["historical_strict_success_rate"]),
                new_strict=pct(row["corrected_strict_success_rate"]),
                old_margin=row["historical_avg_min_body_margin_mixed_units"],
                new_margin=row["corrected_avg_min_body_margin_m"],
                old_near=pct(row["historical_near_collision_rate"]),
                new_near=pct(row["corrected_near_collision_rate"]),
            )
        )
    md.extend(
        [
            "",
            "## 剩余几何不一致",
            "",
            "当前 PointNav `AgentConfig.radius=0.10 m`，历史 clearance proxy 使用 `0.18 m`，挖掘标签使用 `robot_radius=0.21 m`（0.18 m body + 0.03 m safety）。因此 proxy 相对 simulator 碰撞体保守 0.08 m，挖掘 body-margin 相对 simulator 碰撞体保守 0.11 m。当前 strict 阈值 `-0.02 m` 在 0.18 m proxy 下等价于中心到障碍至少 0.16 m，比 0.10 m simulator 接触边界多 0.06 m。单位错误已经修复，但在统一 simulator morphology 并重跑前，不能宣称 clearance proxy 与物理碰撞完全校准。",
            "",
            "`sim_radius_aligned_avg_min_margin_m` 仅用于展示减去 0.10 m 半径时的 0.08 m 平移，不应替代统一碰撞几何后的正式重跑。",
        ]
    )
    (out_dir / "supplementary_s4.md").write_text("\n".join(md) + "\n")
    manifest = {
        "input_csv": str(args.input_csv),
        "historical_files_overwritten": False,
        "depth_normalization": {"min_m": args.depth_min, "max_m": args.depth_max},
        "historical_proxy_radius_m": args.proxy_radius,
        "current_pointnav_sim_radius_m": args.sim_radius,
        "radius_mismatch_m": args.proxy_radius - args.sim_radius,
        "strict_threshold_m": args.strict_threshold,
        "recalculation_scope": "nominal historical episode rows; no trajectory replay",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out_dir)


if __name__ == "__main__":
    main()
