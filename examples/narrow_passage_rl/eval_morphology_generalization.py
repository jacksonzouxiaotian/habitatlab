#!/usr/bin/env python3
"""Run and summarize strict feasibility across paired robot morphologies."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RESULT_PARENT = SCRIPT_DIR / "results" / "ablation_feasibility"
BASE_RESULT = RESULT_PARENT / "structural_full_20260818_194833_final"
METHODS = (
    "full_dynamic_uncertainty",
    "mean_only",
    "fixed_uncertainty",
    # Necessary causal controls for the two requested generalization claims.
    "no_yaw_aware_readiness",
    "no_memory",
)
MORPHOLOGIES = {
    "narrow_028x055": (0.28, 0.55, 0.03),
    "wide_045x065": (0.45, 0.65, 0.03),
    "long_036x080": (0.36, 0.80, 0.03),
}


def _default_output_dir() -> Path:
    return RESULT_PARENT / datetime.now().strftime(
        "morphology_generalization_%Y%m%d_%H%M%S"
    )


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _run_morphology(name: str, dims: tuple[float, float, float], root: Path) -> Path:
    width, length, margin = dims
    target = root / name
    target.mkdir(parents=True, exist_ok=False)
    evaluator = SCRIPT_DIR / "eval_structural_feasibility.py"
    shared = [
        "--body-width", str(width),
        "--body-length", str(length),
        "--safety-margin", str(margin),
        "--max-steps", "200",
        "--methods", *METHODS,
    ]
    smoke = target / "smoke"
    full = target / "full"
    log_path = target / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        smoke_cmd = [
            sys.executable, str(evaluator), "--phase", "smoke", "--seeds", "0",
            *shared, "--output-dir", str(smoke),
        ]
        subprocess.run(
            smoke_cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
            check=True,
        )
        gate = json.loads((smoke / "gate_report.json").read_text(encoding="utf-8"))
        if not gate.get("full_evaluation_permitted", False):
            raise RuntimeError(f"{name} smoke gate failed; see {log_path}")
        full_cmd = [
            sys.executable, str(evaluator), "--phase", "full",
            "--seeds", "0", "1", "2", "--factor-design", "balanced",
            *shared,
            "--gate-report", str(smoke / "gate_report.json"),
            "--output-dir", str(full),
        ]
        subprocess.run(
            full_cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
            check=True,
        )
    return full


def _paired_ci(
    rows: list[dict[str, str]], left: str, right: str, field: str
) -> tuple[float, float, float]:
    paired: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["method"] in {left, right}:
            paired.setdefault(row["episode_id"], {})[row["method"]] = float(row[field])
    differences = np.asarray([
        values[left] - values[right]
        for values in paired.values() if left in values and right in values
    ])
    if len(differences) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(20260820)
    draws = np.mean(
        rng.choice(differences, size=(5000, len(differences)), replace=True), axis=1
    )
    return (
        100.0 * float(np.mean(differences)),
        100.0 * float(np.quantile(draws, 0.025)),
        100.0 * float(np.quantile(draws, 0.975)),
    )


def _summarize(result_dirs: dict[str, Path], output_dir: Path) -> None:
    table: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    for morphology, result_dir in result_dirs.items():
        rows = _read(result_dir / "episodes.csv")
        feasible = [row for row in rows if float(row["passable_label"]) > 0.5]
        for method in METHODS:
            selected = [row for row in feasible if row["method"] == method]
            table.append({
                "morphology": morphology,
                "method": method,
                "feasible_episodes": len(selected),
                "success_pct": 100.0 * float(np.mean([float(r["success"]) for r in selected])),
                "collision_pct": 100.0 * float(np.mean([float(r["collision"]) for r in selected])),
                "false_reject_pct": 100.0 * float(np.mean([float(r["false_reject"]) for r in selected])),
                "result_dir": str(result_dir),
            })
        memory = _paired_ci(
            feasible, "full_dynamic_uncertainty", "no_memory", "false_reject"
        )
        yaw = _paired_ci(
            feasible, "full_dynamic_uncertainty", "no_yaw_aware_readiness", "success"
        )
        claim_rows.extend([
            {
                "morphology": morphology,
                "claim": "memory False Reject delta (full - no_memory)",
                "difference_pp": memory[0], "ci95_low_pp": memory[1],
                "ci95_high_pp": memory[2],
                "supports_claim": memory[1] > 0.0,
            },
            {
                "morphology": morphology,
                "claim": "yaw-aware Success delta (full - no_yaw)",
                "difference_pp": yaw[0], "ci95_low_pp": yaw[1],
                "ci95_high_pp": yaw[2],
                "supports_claim": yaw[1] <= 0.0 <= yaw[2],
            },
        ])
    _write_csv(table, output_dir / "morphology_metrics.csv")
    _write_csv(claim_rows, output_dir / "morphology_claims.csv")

    memory_claims = [row for row in claim_rows if row["claim"].startswith("memory")]
    yaw_claims = [row for row in claim_rows if row["claim"].startswith("yaw")]
    lines = [
        "# Cross-morphology strict feasibility generalization",
        "",
        "主比较按要求报告 full / mean-only / fixed；为检验两个 headline，额外运行",
        "`no_memory` 和 `no_yaw_aware_readiness`。所有方法共享同一 morphology 内的",
        "630 个 paired scenarios、观测噪声与随机哈希。",
        "",
        "| Morphology | Method | Feasible N | Success | Collision | False Reject |",
        "|:---|:---|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['morphology']} | {row['method']} | {row['feasible_episodes']} | "
            f"{float(row['success_pct']):.2f}% | {float(row['collision_pct']):.2f}% | "
            f"{float(row['false_reject_pct']):.2f}% |"
        )
    lines.extend([
        "",
        "| Morphology | Paired contrast | Δ (pp) | 95% paired bootstrap CI | Verdict |",
        "|:---|:---|---:|:---:|:---|",
    ])
    for row in claim_rows:
        lines.append(
            f"| {row['morphology']} | {row['claim']} | "
            f"{float(row['difference_pp']):+.2f} | "
            f"[{float(row['ci95_low_pp']):+.2f}, {float(row['ci95_high_pp']):+.2f}] | "
            f"{'supported' if row['supports_claim'] else 'not supported'} |"
        )
    memory_all = all(bool(row["supports_claim"]) for row in memory_claims)
    yaw_all = all(bool(row["supports_claim"]) for row in yaw_claims)
    max_abs_yaw_delta = max(abs(float(row["difference_pp"])) for row in yaw_claims)
    lines.extend([
        "",
        f"结论：(a) memory 导致 False Reject 显著上升在所有 morphology 上"
        f"{'成立' if memory_all else '并非全部成立'}；(b) yaw-aware Success 差异的 95% CI "
        f"在所有 morphology 上{'均包含 0' if yaw_all else '并非都包含 0'}，"
        f"观测到的最大绝对差为 {max_abs_yaw_delta:.2f} pp。",
        "",
    ])
    (output_dir / "morphology_generalization.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run(output_dir: Path, base_result: Path, workers: int, execute: bool) -> None:
    if execute and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dirs = {"base_036x060": base_result.resolve()}
    if execute:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_morphology, name, dims, output_dir): name
                for name, dims in MORPHOLOGIES.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                result_dirs[name] = future.result()
                print(f"[complete] {name}: {result_dirs[name]}", flush=True)
    else:
        for name in MORPHOLOGIES:
            result_dirs[name] = output_dir / name / "full"
    missing = [str(path) for path in result_dirs.values() if not (path / "episodes.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing morphology result(s): {missing}")
    _summarize(result_dirs, output_dir)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_result_reused_without_modification": str(base_result.resolve()),
        "new_morphologies": MORPHOLOGIES,
        "methods": METHODS,
        "formal_protocol": "6 margin bins x 7 scenes x 3 seeds x 5 balanced yaw cases",
        "paired_scenarios_per_morphology": 630,
        "result_dirs": {key: str(value) for key, value in result_dirs.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-result", type=Path, default=BASE_RESULT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output_dir = args.output_dir or _default_output_dir()
    run(output_dir, args.base_result, args.workers, args.execute)
    print(f"[write] {output_dir / 'morphology_generalization.md'}")


if __name__ == "__main__":
    main()
