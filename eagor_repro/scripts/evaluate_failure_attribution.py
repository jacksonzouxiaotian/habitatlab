#!/usr/bin/env python3
"""Run the Phase 1-4 MP3D navigation failure-attribution matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eagor_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from eagor_repro.config import load_config
from eagor_repro.evaluation.habitat_evaluator import evaluate_objectnav


EXPERIMENTS = (
    {
        "id": "A",
        "name": "eagor_direct_area",
        "direction": "eagor",
        "policy": "eagor",
        "planner": "direct",
        "stop": "area",
    },
    {
        "id": "B",
        "name": "grid_direct_area",
        "direction": "grid",
        "policy": "grid",
        "planner": "direct",
        "stop": "area",
    },
    {
        "id": "C",
        "name": "oracle_direct_area",
        "direction": "oracle",
        "policy": "oracle_direction",
        "planner": "direct",
        "stop": "area",
    },
    {
        "id": "D",
        "name": "oracle_direct_oracle_stop",
        "direction": "oracle",
        "policy": "oracle_direction",
        "planner": "direct",
        "stop": "oracle_distance",
    },
    {
        "id": "E",
        "name": "oracle_depth_planner_oracle_stop",
        "direction": "oracle",
        "policy": "oracle_direction",
        "planner": "depth_traversability",
        "stop": "oracle_distance",
    },
    {
        "id": "F",
        "name": "oracle_depth_planner_depth_stop",
        "direction": "oracle",
        "policy": "oracle_direction",
        "planner": "depth_traversability",
        "stop": "area_depth",
    },
    {
        "id": "G",
        "name": "eagor_depth_planner_depth_stop",
        "direction": "eagor",
        "policy": "eagor",
        "planner": "depth_traversability",
        "stop": "area_depth",
    },
    {
        "id": "H",
        "name": "grid_depth_planner_depth_stop",
        "direction": "grid",
        "policy": "grid",
        "planner": "depth_traversability",
        "stop": "area_depth",
    },
    {
        "id": "I",
        "name": "oracle_navmesh_oracle_stop",
        "direction": "oracle",
        "policy": "oracle_direction",
        "planner": "oracle_nav",
        "stop": "oracle_distance",
    },
)


METRICS = (
    "success_rate",
    "spl",
    "steps",
    "path_length_m",
    "collision_count",
    "false_stop",
    "correct_stop",
    "missed_stop",
    "mae_deg",
    "planner_recovery_count",
    "blocked_step_count",
    "average_inference_update_latency_ms",
)


def _mean(episodes: Iterable[Dict[str, Any]], key: str) -> float:
    values = [
        float(item[key])
        for item in episodes
        if key in item and item[key] is not None and np.isfinite(float(item[key]))
    ]
    return float(np.mean(values)) if values else float("nan")


def _aggregate(spec: Dict[str, str], episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "id": spec["id"],
        "direction": spec["direction"],
        "planner": spec["planner"],
        "stop": spec["stop"],
        "episodes": len(episodes),
        "oracle_upper_bound": any(bool(item["oracle_upper_bound"]) for item in episodes),
    }
    row.update({metric: _mean(episodes, metric) for metric in METRICS})
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if not np.isfinite(value) else f"{value:.4f}"
    return str(value)


def _markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "id",
        "direction",
        "planner",
        "stop",
        "success_rate",
        "spl",
        "steps",
        "path_length_m",
        "collision_count",
        "false_stop",
        "mae_deg",
        "planner_recovery_count",
        "average_inference_update_latency_ms",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row[key]) for key in headers) + " |")
    return "\n".join(lines)


def _stop_comparison(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {row["id"]: row for row in rows}
    definitions = (
        ("Area", "A"),
        ("Area", "B"),
        ("Area", "C"),
        ("Area + Depth", "F"),
        ("Oracle Distance", "D"),
    )
    return [
        {
            "stop_method": label,
            "matrix_id": matrix_id,
            "direction": by_id[matrix_id]["direction"],
            "planner": by_id[matrix_id]["planner"],
            "false_stop": by_id[matrix_id]["false_stop"],
            "correct_stop": by_id[matrix_id]["correct_stop"],
            "success_rate": by_id[matrix_id]["success_rate"],
        }
        for label, matrix_id in definitions
        if matrix_id in by_id
    ]


def _generic_markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers = list(rows[0])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row[key]) for key in headers) + " |")
    return "\n".join(lines)


def _diagnosis(rows: List[Dict[str, Any]]) -> str:
    by_id = {row["id"]: row for row in rows}
    required = set("ABCDEFGH")
    if not required.issubset(by_id):
        missing = ", ".join(sorted(required - set(by_id)))
        return (
            "## Automatic failure-attribution diagnosis\n\n"
            f"Partial matrix only; missing experiments: {missing}. "
            "No dominant-bottleneck conclusion is generated."
        )
    sr = {key: float(value["success_rate"]) for key, value in by_id.items()}
    lines = ["## Automatic failure-attribution diagnosis", ""]
    if sr["C"] <= max(sr["A"], sr["B"]) + 0.05:
        lines.append(
            f"- **Direction upper bound:** C (GT direction + direct + area) SR={sr['C']:.3f}, "
            f"while A/B are {sr['A']:.3f}/{sr['B']:.3f}. Perfect bearing alone does not "
            "remove the failure, so direction estimation is not the only dominant bottleneck."
        )
    else:
        lines.append(
            f"- **Direction upper bound:** C improves SR to {sr['C']:.3f} from "
            f"A/B={sr['A']:.3f}/{sr['B']:.3f}; direction quality contributes materially."
        )
    if sr["D"] > sr["C"] + 0.05:
        lines.append(
            f"- **Stop criterion:** replacing area stop by Oracle distance raises SR from "
            f"{sr['C']:.3f} to {sr['D']:.3f}; stop is a material bottleneck."
        )
    else:
        lines.append(
            f"- **Stop criterion:** C→D changes SR {sr['C']:.3f}→{sr['D']:.3f}; "
            "Oracle stop alone does not provide a material gain under direct control."
        )
    if sr["E"] > sr["D"] + 0.05:
        lines.append(
            f"- **Local traversability:** adding the depth planner under the same Oracle "
            f"direction/stop raises SR {sr['D']:.3f}→{sr['E']:.3f}; planning is a material bottleneck."
        )
    else:
        lines.append(
            f"- **Local traversability:** D→E changes SR {sr['D']:.3f}→{sr['E']:.3f}. "
            "The first-order local planner is not sufficient to close the navigation gap."
        )
    if sr["F"] > sr["G"] + 0.05:
        lines.append(
            f"- **Belief after matched planner/stop:** F (Oracle direction) SR={sr['F']:.3f} "
            f"exceeds G (EAGOR) SR={sr['G']:.3f}; belief quality becomes a measurable bottleneck."
        )
    else:
        lines.append(
            f"- **Belief after matched planner/stop:** F/G SR={sr['F']:.3f}/{sr['G']:.3f}; "
            "this matrix does not yet show a clear EAGOR-vs-Oracle direction gap."
        )
    if "I" in sr and sr["I"] > sr["E"] + 0.05:
        lines.append(
            f"- **Navmesh upper bound:** I raises SR from local-depth E={sr['E']:.3f} "
            f"to oracle-navmesh I={sr['I']:.3f}. The missing capability is global/navmesh "
            "reachability rather than another spherical-belief update."
        )
    elif "I" in sr:
        lines.append(
            f"- **Navmesh upper bound:** I SR={sr['I']:.3f}; even the true path heading "
            "does not close the gap, so controller/goal-interface behavior remains limiting."
        )
    if max(sr.values()) == 0.0:
        dominant = (
            "**Current conclusion: E — multiple navigation factors jointly dominate.** "
            "Even the GT-direction + depth-local-planner + Oracle-stop upper bound has SR=0, "
            "so SH-BF modification is not justified yet; a local heading heuristic does not "
            "replace global/topological path planning and robust control."
        )
    elif "I" in sr and sr["I"] > sr["E"] + 0.05:
        dominant = (
            "**Current conclusion: C — global/path reachability is the strongest isolated "
            "bottleneck.** The fixed-step controller succeeds when driven by the real "
            "navmesh path, while direct and single-frame local-depth headings fail."
        )
    elif sr["E"] > sr["D"] + 0.05 and sr["D"] > sr["C"] + 0.05:
        dominant = "**Current conclusion: E — path planning and stop criterion jointly dominate.**"
    elif sr["E"] > sr["D"] + 0.05:
        dominant = "**Current conclusion: C — path planning is the strongest isolated bottleneck.**"
    elif sr["D"] > sr["C"] + 0.05:
        dominant = "**Current conclusion: D — stop criterion is the strongest isolated bottleneck.**"
    elif sr["C"] > max(sr["A"], sr["B"]) + 0.05:
        dominant = "**Current conclusion: A — direction belief quality is the strongest isolated bottleneck.**"
    else:
        dominant = (
            "**Current conclusion: E — the tested controller/planner/stop stack has interacting "
            "limitations; no single module is isolated as sufficient.**"
        )
    lines.extend(["", dominant])
    return "\n".join(lines)


def _plot(rows: List[Dict[str, Any]], path: Path) -> None:
    labels = [row["id"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specs = (
        ("success_rate", "Success rate", "tab:green"),
        ("collision_count", "Mean collisions", "tab:red"),
        ("false_stop", "Mean false stops", "tab:orange"),
        ("planner_recovery_count", "Mean planner recoveries", "tab:blue"),
    )
    for axis, (key, title, color) in zip(axes.flat, specs):
        values = [float(row[key]) for row in rows]
        bars = axis.bar(labels, values, color=color, alpha=0.8)
        axis.set_title(title)
        axis.set_xlabel("Experiment ID")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("MP3D Phase 1-4 Failure Attribution (diagnostic Oracle semantics)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eagor/base.yaml"))
    parser.add_argument(
        "--overlay",
        type=Path,
        default=Path("configs/eagor/mp3d_failure_attribution.yaml"),
    )
    parser.add_argument("--experiments", nargs="+", default=[item["id"] for item in EXPERIMENTS])
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config = OmegaConf.merge(config, OmegaConf.load(args.overlay))
    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    if args.no_video:
        config.evaluation.save_video = False
    OmegaConf.resolve(config)
    requested = {value.upper() for value in args.experiments}
    specs = [item for item in EXPERIMENTS if item["id"] in requested]
    unknown = requested - {item["id"] for item in EXPERIMENTS}
    if unknown:
        raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")

    output_root = Path(str(config.output.root))
    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_episodes: List[Dict[str, Any]] = []
    aggregate: List[Dict[str, Any]] = []
    for spec in specs:
        case_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        experiment_id = f"{spec['id']}_{spec['name']}"
        case_config.evaluation.experiment_id = experiment_id
        case_config.evaluation.policy_method = spec["policy"]
        case_config.planning.mode = spec["planner"]
        case_config.controller.stop_mode = spec["stop"]
        batch_path = (
            output_root
            / experiment_id
            / spec["policy"]
            / "summaries"
            / f"objectnav_{spec['policy']}.json"
        )
        if args.resume and batch_path.is_file():
            episodes = json.loads(batch_path.read_text(encoding="utf-8"))
            print(f"RESUME {spec['id']} from {batch_path}", flush=True)
        else:
            print(
                f"RUN {spec['id']}: direction={spec['direction']} "
                f"planner={spec['planner']} stop={spec['stop']}",
                flush=True,
            )
            episodes = evaluate_objectnav(case_config, spec["policy"])
        for episode in episodes:
            episode["matrix_id"] = spec["id"]
            episode["matrix_direction"] = spec["direction"]
        all_episodes.extend(episodes)
        aggregate.append(_aggregate(spec, episodes))

    attribution_csv = summary_dir / "failure_attribution.csv"
    cases_csv = summary_dir / "failure_cases.csv"
    attribution_md = summary_dir / "failure_attribution.md"
    diagnosis_md = summary_dir / "failure_attribution_summary.md"
    plot_path = summary_dir / "failure_attribution.png"
    stop_csv = summary_dir / "stop_comparison.csv"
    _write_csv(attribution_csv, aggregate)
    case_fields = [
        "matrix_id",
        "episode_output_key",
        "scene_id",
        "target_query",
        "success",
        "primary_failure_mode",
        "secondary_failure_modes",
        "mae_deg",
        "collision_count",
        "false_stop",
        "blocked_step_count",
        "planner_recovery_count",
    ]
    _write_csv(cases_csv, [{key: item.get(key) for key in case_fields} for item in all_episodes])
    table = _markdown_table(aggregate)
    stop_rows = _stop_comparison(aggregate)
    if stop_rows:
        _write_csv(stop_csv, stop_rows)
    stop_table = _generic_markdown_table(stop_rows) if stop_rows else "Not available."
    attribution_md.write_text(
        "# MP3D failure-attribution matrix\n\n" + table + "\n",
        encoding="utf-8",
    )
    diagnosis_md.write_text(
        "# MP3D Phase 1-4 failure attribution\n\n"
        "> All rows use Oracle semantic masks to isolate navigation. Rows C-F use "
        "GT direction and/or GT stop; row I additionally uses the Habitat navmesh "
        "shortest path. These are diagnostic upper bounds.\n\n"
        + table
        + "\n\n## Stop-method comparison\n\n"
        + stop_table
        + "\n\nArea + Depth is row F and therefore also includes the depth planner; "
        "C versus D is the clean direct-controller area/oracle-stop isolation.\n\n"
        + _diagnosis(aggregate)
        + "\n",
        encoding="utf-8",
    )
    _plot(aggregate, plot_path)
    print(
        json.dumps(
            {
                "failure_attribution_csv": str(attribution_csv),
                "failure_cases_csv": str(cases_csv),
                "failure_attribution_markdown": str(attribution_md),
                "failure_attribution_summary": str(diagnosis_md),
                "failure_attribution_plot": str(plot_path),
                "stop_comparison_csv": str(stop_csv),
                "aggregate": aggregate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
