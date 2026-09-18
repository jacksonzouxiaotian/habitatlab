#!/usr/bin/env python3
"""Paired smoke sweep of kappa with fixed tau_commit=tau_reject=0.02 m."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PARENT = SCRIPT_DIR / "results" / "ablation_feasibility"
KAPPAS = (0.5, 1.0, 1.28, 1.645, 2.0, 2.5)
PAPER_KAPPA = 1.645
TAU = 0.020
METHOD = "full_dynamic_uncertainty"


def _default_output_dir() -> Path:
    return DEFAULT_PARENT / datetime.now().strftime("kappa_sweep_%Y%m%d_%H%M%S")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rate(rows: list[dict[str, str]], field: str) -> float:
    return 100.0 * float(np.mean([float(row[field]) for row in rows]))


def run(output_dir: Path) -> list[dict[str, Any]]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = SCRIPT_DIR / "eval_structural_feasibility.py"
    sweep_rows: list[dict[str, Any]] = []
    cross_kappa_hashes: dict[str, dict[str, set[str]]] = {}
    for kappa in KAPPAS:
        child = output_dir / f"kappa_{kappa:g}"
        command = [
            sys.executable,
            str(evaluator),
            "--phase", "smoke",
            "--seeds", "0",
            "--max-steps", "200",
            "--kappa", str(kappa),
            "--tau-commit", str(TAU),
            "--tau-reject", str(TAU),
            "--output-dir", str(child),
        ]
        print("[run] " + " ".join(command), flush=True)
        subprocess.run(command, cwd=SCRIPT_DIR.parent.parent, check=True)
        full_rows = [
            row for row in _read_csv(child / "episodes.csv")
            if row["method"] == METHOD
        ]
        for row in full_rows:
            audit = cross_kappa_hashes.setdefault(
                row["episode_id"], {"observation": set(), "random": set()}
            )
            audit["observation"].add(row["observation_sha256"])
            audit["random"].add(row["random_draw_sha256"])
        episodes = [row for row in full_rows if float(row["passable_label"]) > 0.5]
        if len(episodes) != 18:
            raise RuntimeError(
                f"Expected 18 feasible cases from 42 paired scenarios, got {len(episodes)}"
            )
        sweep_rows.append({
            "kappa": kappa,
            "tau_commit_m": TAU,
            "tau_reject_m": TAU,
            "method": METHOD,
            "scenario_count": 42,
            "feasible_episode_count": len(episodes),
            "denominator": "ground-truth feasible episodes",
            "false_reject_pct": _rate(episodes, "false_reject"),
            "collision_pct": _rate(episodes, "collision"),
            "success_pct": _rate(episodes, "success"),
            "paper_kappa": kappa == PAPER_KAPPA,
            "result_dir": str(child),
        })

    pairing_ok = len(cross_kappa_hashes) == 42 and all(
        len(values["observation"]) == 1 and len(values["random"]) == 1
        for values in cross_kappa_hashes.values()
    )
    if not pairing_ok:
        raise RuntimeError("Observation/random hashes changed across kappa values")
    _write_csv(sweep_rows, output_dir / "sweep.csv")
    fig, axis = plt.subplots(figsize=(5.4, 4.2))
    x = [float(row["kappa"]) for row in sweep_rows]
    specs = (
        ("false_reject_pct", "False Reject", "#ff7f0e"),
        ("collision_pct", "Collision", "#d62728"),
        ("success_pct", "Success", "#2ca02c"),
    )
    for field, label, color in specs:
        y = [float(row[field]) for row in sweep_rows]
        axis.plot(x, y, marker="o", linewidth=1.7, color=color, label=label)
        paper_index = x.index(PAPER_KAPPA)
        axis.scatter(
            [PAPER_KAPPA], [y[paper_index]], s=58, facecolor="white",
            edgecolor=color, linewidth=1.7, zorder=5,
        )
    axis.axvline(PAPER_KAPPA, color="#4d4d4d", linestyle="--", linewidth=1.0)
    axis.annotate(
        "paper claim: κ=1.645", xy=(PAPER_KAPPA, 0.98),
        xycoords=("data", "axes fraction"), xytext=(5, -2),
        textcoords="offset points", fontsize=8, ha="left", va="top",
    )
    axis.set_xlabel("κ")
    axis.set_ylabel("Feasible-subset outcome rate (%)")
    axis.set_xticks(x)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.text(
        0.5, -0.01,
        "τ_commit = τ_reject = 0.02 m; open markers denote the current paper κ.",
        ha="center", fontsize=8,
    )
    for suffix, kwargs in ((".png", {"dpi": 220}), (".pdf", {})):
        path = output_dir / f"kappa_tradeoff{suffix}"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(fig)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kappas": KAPPAS,
        "tau_commit_m": TAU,
        "tau_reject_m": TAU,
        "method": METHOD,
        "protocol": "eval_structural_feasibility.py --phase smoke --seeds 0 --max-steps 200",
        "metric_denominator": "ground-truth feasible episodes (18 per kappa)",
        "paper_kappa": PAPER_KAPPA,
        "cross_kappa_observation_and_random_hashes_identical": pairing_ok,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return sweep_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or _default_output_dir()
    rows = run(output_dir)
    print(f"[write] {output_dir / 'sweep.csv'}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
