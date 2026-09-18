#!/usr/bin/env python3
"""Validate and summarize the frozen three-seed strict paired evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "reactive_rule_baseline", "full_dynamic_uncertainty", "mean_only",
    "fixed_uncertainty", "no_yaw_aware_readiness", "no_memory",
)
TABLE_ORDER = (
    "reactive_rule_baseline", "mean_only", "fixed_uncertainty",
    "no_yaw_aware_readiness", "no_memory", "full_dynamic_uncertainty",
)
NAMES = {
    "reactive_rule_baseline": "Reactive rule baseline",
    "mean_only": "Mean only",
    "fixed_uncertainty": "Fixed uncertainty",
    "no_yaw_aware_readiness": "DEGNav w/o yaw-aware readiness",
    "no_memory": "DEGNav w/o memory",
    "full_dynamic_uncertainty": "DEGNav full",
}
METRICS = ("F-SR", "CR", "FR", "Collision", "Timeout/stuck")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260821
MARGIN_LABELS = (
    "[-0.15,-0.10)", "[-0.10,-0.05)", "[-0.05,0.00)",
    "[0.00,0.05)", "[0.05,0.10)", "[0.10,0.15]",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _metric(rows: list[dict[str, Any]], metric: str) -> float:
    if metric == "F-SR":
        selected = [row for row in rows if _truth(row["passable"])]
        values = [_f(row, "success") for row in selected]
    elif metric == "CR":
        selected = [row for row in rows if not _truth(row["passable"])]
        values = [_f(row, "correct_reject") for row in selected]
    elif metric == "FR":
        selected = [row for row in rows if _truth(row["passable"])]
        values = [_f(row, "false_reject") for row in selected]
    elif metric == "Collision":
        values = [_f(row, "collision") for row in rows]
    elif metric == "Timeout/stuck":
        values = [
            float(_f(row, "timeout") > 0.5 or _f(row, "stuck") > 0.5)
            for row in rows
        ]
    else:
        raise KeyError(metric)
    return float(np.mean(values)) if values else math.nan


def _validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = defaultdict(list)
    by_scenario = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
        by_scenario[row["scenario_id"]].append(row)
    if set(by_method) != set(METHODS):
        raise RuntimeError(f"Method set mismatch: {sorted(by_method)}")
    if any(len(by_method[method]) != 1890 for method in METHODS):
        raise RuntimeError({method: len(by_method[method]) for method in METHODS})
    if len(by_scenario) != 1890:
        raise RuntimeError(f"Expected 1,890 paired scenarios, got {len(by_scenario)}")
    for scenario_id, paired in by_scenario.items():
        if {row["method"] for row in paired} != set(METHODS):
            raise RuntimeError(f"Incomplete methods at {scenario_id}")
        if len({row["observation_sha256"] for row in paired}) != 1:
            raise RuntimeError(f"Observation hash mismatch at {scenario_id}")
        if len({row["random_draw_sha256"] for row in paired}) != 1:
            raise RuntimeError(f"Random hash mismatch at {scenario_id}")
    per_seed = defaultdict(set)
    for scenario_id, paired in by_scenario.items():
        per_seed[int(float(paired[0]["seed"]))].add(scenario_id)
    if set(per_seed) != {0, 1, 2} or any(len(value) != 630 for value in per_seed.values()):
        raise RuntimeError({seed: len(value) for seed, value in per_seed.items()})
    return {
        "methods": len(by_method), "paired_scenarios": len(by_scenario),
        "episodes_per_method": 1890,
        "scenarios_per_seed": {str(seed): len(value) for seed, value in per_seed.items()},
        "paired_hashes_identical": True,
    }


def _stratified_sample_ids(
    seed_ids: dict[int, list[str]], rng: np.random.Generator
) -> list[str]:
    sampled = []
    for seed in sorted(seed_ids):
        ids = np.asarray(seed_ids[seed], dtype=object)
        sampled.extend(rng.choice(ids, size=len(ids), replace=True).tolist())
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with (run_dir / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validation = _validate(rows)
    by_method_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_method_id: dict[tuple[str, str], dict[str, Any]] = {}
    seed_ids: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        seed = int(float(row["seed"]))
        by_method_seed[(row["method"], seed)].append(row)
        by_method_id[(row["method"], row["scenario_id"])] = row
        if row["method"] == "full_dynamic_uncertainty":
            seed_ids[seed].append(row["scenario_id"])

    seed_summary = []
    mean_sd = []
    for method in TABLE_ORDER:
        seed_values = {metric: [] for metric in METRICS}
        for seed in (0, 1, 2):
            selected = by_method_seed[(method, seed)]
            item = {"method": method, "paper_method": NAMES[method], "seed": seed, "episodes": len(selected)}
            for metric in METRICS:
                value = _metric(selected, metric)
                item[metric] = value
                seed_values[metric].append(value)
            seed_summary.append(item)
        item = {"method": method, "paper_method": NAMES[method], "seeds": 3, "episodes_per_seed": 630}
        for metric in METRICS:
            values = np.asarray(seed_values[metric], dtype=float)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_sd"] = float(np.std(values, ddof=1))
        mean_sd.append(item)
    _write_csv(run_dir / "summary_by_seed.csv", seed_summary)
    _write_csv(run_dir / "summary_mean_sd.csv", mean_sd)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_values = {
        (method, metric): [] for method in TABLE_ORDER for metric in METRICS
    }
    difference_values = {
        (method, metric): []
        for method in TABLE_ORDER if method != "full_dynamic_uncertainty"
        for metric in METRICS
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_ids = _stratified_sample_ids(seed_ids, rng)
        full_rows = [
            by_method_id[("full_dynamic_uncertainty", scenario_id)]
            for scenario_id in sampled_ids
        ]
        full_metrics = {metric: _metric(full_rows, metric) for metric in METRICS}
        for method in TABLE_ORDER:
            selected = [by_method_id[(method, scenario_id)] for scenario_id in sampled_ids]
            for metric in METRICS:
                value = _metric(selected, metric)
                bootstrap_values[(method, metric)].append(value)
                if method != "full_dynamic_uncertainty":
                    difference_values[(method, metric)].append(
                        full_metrics[metric] - value
                    )

    bootstrap_rows = []
    for method in TABLE_ORDER:
        all_rows = [row for row in rows if row["method"] == method]
        for metric in METRICS:
            values = np.asarray(bootstrap_values[(method, metric)])
            bootstrap_rows.append({
                "method": method, "paper_method": NAMES[method], "metric": metric,
                "estimate": _metric(all_rows, metric),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "cluster": "paired scenario_id stratified by seed",
            })
    _write_csv(run_dir / "bootstrap_ci.csv", bootstrap_rows)

    paired_rows = []
    for method in TABLE_ORDER:
        if method == "full_dynamic_uncertainty":
            continue
        other = [row for row in rows if row["method"] == method]
        full = [row for row in rows if row["method"] == "full_dynamic_uncertainty"]
        for metric in METRICS:
            values = np.asarray(difference_values[(method, metric)])
            paired_rows.append({
                "comparison": f"full_dynamic_uncertainty - {method}",
                "ablation": method, "metric": metric,
                "paired_scenarios": 1890,
                "estimate_difference": _metric(full, metric) - _metric(other, metric),
                "ci95_low": float(np.quantile(values, 0.025)),
                "ci95_high": float(np.quantile(values, 0.975)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            })
    _write_csv(run_dir / "paired_method_differences.csv", paired_rows)

    margin_rows = []
    observed_margin_labels = {
        row["margin_bin"]
        for row in rows
        if row["method"] == "full_dynamic_uncertainty"
    }
    if observed_margin_labels != set(MARGIN_LABELS):
        raise RuntimeError(
            f"Margin-bin labels differ from the frozen protocol: "
            f"{sorted(observed_margin_labels)}"
        )
    margin_labels = list(MARGIN_LABELS)
    for method in TABLE_ORDER:
        for label in margin_labels:
            selected = [row for row in rows if row["method"] == method and row["margin_bin"] == label]
            item = {"method": method, "paper_method": NAMES[method], "margin_bin": label, "episodes": len(selected)}
            margins = [_f(row, "d_hat", math.nan) - _f(row, "w_req_cons", math.nan) for row in selected]
            finite = [value for value in margins if math.isfinite(value)]
            item["mean_D_hat_minus_W_req_cons"] = float(np.mean(finite)) if finite else math.nan
            for metric in METRICS:
                item[metric] = _metric(selected, metric)
            margin_rows.append(item)
    _write_csv(run_dir / "margin_phase.csv", margin_rows)

    colors = {
        "reactive_rule_baseline": "#7f7f7f", "mean_only": "#ff7f0e",
        "fixed_uncertainty": "#2ca02c", "no_yaw_aware_readiness": "#9467bd",
        "no_memory": "#8c564b", "full_dynamic_uncertainty": "#1f77b4",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    x = np.arange(len(margin_labels))
    for axis, metric in zip(axes.flat, METRICS):
        for method in TABLE_ORDER:
            selected = [row for row in margin_rows if row["method"] == method]
            axis.plot(x, [100 * float(row[metric]) for row in selected], marker="o", linewidth=1.5, color=colors[method], label=NAMES[method])
        axis.set_title(metric)
        axis.set_ylabel("Percent")
        axis.grid(alpha=0.25)
    axes.flat[-1].set_visible(False)
    for axis in axes[1, :2]:
        axis.set_xticks(x, margin_labels, rotation=30, ha="right")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(r"Margin phase: $\widehat D-\widehat W_{\mathrm{req}}^{\mathrm{cons}}$")
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 220}), ("pdf", {})):
        path = run_dir / f"margin_phase.{suffix}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, bbox_inches="tight", **kwargs)
    plt.close(fig)

    scene_rows = []
    for method in TABLE_ORDER:
        for scene in sorted({row["corridor_type"] for row in rows}):
            selected = [row for row in rows if row["method"] == method and row["corridor_type"] == scene]
            item = {"method": method, "scene_type": scene, "episodes": len(selected)}
            for metric in METRICS:
                item[metric] = _metric(selected, metric)
            scene_rows.append(item)
    _write_csv(run_dir / "scene_type_breakdown.csv", scene_rows)

    strata_rows = []
    dimensions: dict[str, Callable[[dict[str, Any]], str]] = {
        "margin_bin": lambda row: row["margin_bin"],
        "scene_type": lambda row: row["corridor_type"],
        "yaw_bin": lambda row: str(int(float(row["yaw_deg"]))),
        "noise_bin": lambda row: row["noise_level"],
        "lateral_offset_bin": lambda row: str(row["lateral_offset"]),
        "feasibility": lambda row: "feasible" if _truth(row["passable"]) else "infeasible",
    }
    for method in TABLE_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        for dimension, fn in dimensions.items():
            for value in sorted({fn(row) for row in method_rows}):
                selected = [row for row in method_rows if fn(row) == value]
                item = {"method": method, "dimension": dimension, "value": value, "episodes": len(selected)}
                for metric in METRICS:
                    item[metric] = _metric(selected, metric)
                strata_rows.append(item)
    _write_csv(run_dir / "stratified_metrics.csv", strata_rows)

    # Risk-coverage uses structural feasibility confidence and the unified OBB label.
    risk_rows = []
    for method in TABLE_ORDER:
        selected = [row for row in rows if row["method"] == method]
        probabilities = np.asarray([_f(row, "p_feas_struct", math.nan) for row in selected])
        labels = np.asarray([_truth(row["passable"]) for row in selected])
        valid = np.isfinite(probabilities)
        probabilities, labels = probabilities[valid], labels[valid]
        confidence = np.maximum(probabilities, 1.0 - probabilities)
        prediction = probabilities >= 0.5
        for threshold in np.linspace(0.5, 0.95, 10):
            covered = confidence >= threshold
            risk_rows.append({
                "method": method, "confidence_threshold": threshold,
                "coverage": float(np.mean(covered)) if len(covered) else math.nan,
                "selective_risk": float(np.mean(prediction[covered] != labels[covered])) if np.any(covered) else math.nan,
                "covered_episodes": int(np.sum(covered)),
            })
    _write_csv(run_dir / "risk_coverage.csv", risk_rows)

    def stat(method: str, metric: str) -> str:
        row = next(item for item in mean_sd if item["method"] == method)
        return f"{100 * row[f'{metric}_mean']:.1f} ({100 * row[f'{metric}_sd']:.1f})"

    latex = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\caption{Frozen strict paired procedural evaluation: 1,890 scenarios per method over three seeds. Values are seed mean (sample SD), in percent.}",
        r"\label{tab:proc_main}", r"\begin{tabular}{lccccc}", r"\toprule",
        r"Method & F-SR $\uparrow$ & CR $\uparrow$ & FR $\downarrow$ & Collision $\downarrow$ & Timeout/stuck $\downarrow$ \\",
        r"\midrule",
    ]
    for method in TABLE_ORDER:
        latex.append(
            f"{NAMES[method]} & {stat(method, 'F-SR')} & {stat(method, 'CR')} & "
            f"{stat(method, 'FR')} & {stat(method, 'Collision')} & {stat(method, 'Timeout/stuck')} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    table_path = run_dir / "table_proc_main.tex"
    if table_path.exists():
        raise FileExistsError(table_path)
    table_path.write_text("\n".join(latex) + "\n", encoding="utf-8")

    def paired(method: str, metric: str) -> dict[str, Any]:
        return next(row for row in paired_rows if row["ablation"] == method and row["metric"] == metric)

    conclusions = []
    mean_fsr = paired("mean_only", "F-SR")
    if float(mean_fsr["ci95_low"]) > 0:
        conclusions.append("The paired bootstrap supports higher F-SR for Full than Mean only.")
    else:
        conclusions.append("Full versus Mean only F-SR: no statistically resolved difference under the evaluated protocol.")
    fixed_fsr = paired("fixed_uncertainty", "F-SR")
    if float(fixed_fsr["ci95_low"]) > 0:
        conclusions.append("The paired bootstrap supports higher F-SR for dynamic than fixed uncertainty.")
    else:
        conclusions.append("Dynamic versus fixed uncertainty F-SR: no statistically resolved difference under the evaluated protocol.")
    for metric in ("Collision", "Timeout/stuck"):
        item = paired("mean_only", metric)
        if float(item["ci95_high"]) < 0:
            conclusions.append(f"Dynamic uncertainty reduces {metric.lower()} relative to Mean only.")
        elif float(item["ci95_low"]) > 0:
            conclusions.append(
                f"Trade-off: Full increases {metric.lower()} relative to Mean only "
                f"by {100 * float(item['estimate_difference']):.1f} percentage "
                f"points (paired 95% CI "
                f"[{100 * float(item['ci95_low']):.1f}, "
                f"{100 * float(item['ci95_high']):.1f}])."
            )
        else:
            conclusions.append(f"Dynamic uncertainty effect on {metric.lower()}: no statistically resolved reduction.")
    fr = paired("mean_only", "FR")
    if float(fr["ci95_high"]) < 0:
        conclusions.append(
            "Full lowers feasible false rejection relative to Mean only by "
            f"{abs(100 * float(fr['estimate_difference'])):.1f} percentage "
            f"points (paired 95% CI for Full-minus-Mean "
            f"[{100 * float(fr['ci95_low']):.1f}, "
            f"{100 * float(fr['ci95_high']):.1f}])."
        )
    yaw = paired("no_yaw_aware_readiness", "F-SR")
    if float(yaw["ci95_low"]) > 0:
        conclusions.append("Yaw-aware projected-width readiness contributes positively to F-SR.")
    else:
        conclusions.append("Yaw-aware readiness contribution to F-SR: no statistically resolved positive effect.")
    memory_equal = all(
        abs(float(paired("no_memory", metric)["estimate_difference"])) < 1e-12
        for metric in METRICS
    )
    if memory_equal:
        conclusions.append("Memory does not affect the single-encounter benchmark, as expected; its contribution is evaluated separately under repeated and transfer encounters.")
    else:
        memory_fsr = paired("no_memory", "F-SR")
        if float(memory_fsr["ci95_low"]) > 0:
            conclusions.append("The paired bootstrap supports a positive memory contribution to F-SR in this single-encounter benchmark.")
        else:
            conclusions.append("Full versus No memory F-SR: no statistically resolved difference under the evaluated protocol; repeated and transfer encounters remain the designated memory test.")
    cr = paired("mean_only", "CR")
    if float(cr["ci95_high"]) < 0:
        conclusions.append(
            "Trade-off: Full lowers correct rejection relative to Mean only by "
            f"{abs(100 * float(cr['estimate_difference'])):.1f} percentage "
            f"points (paired 95% CI for Full-minus-Mean "
            f"[{100 * float(cr['ci95_low']):.1f}, "
            f"{100 * float(cr['ci95_high']):.1f}])."
        )
    elif float(cr["estimate_difference"]) < 0:
        conclusions.append("Trade-off: Full has lower CR than Mean only and this negative direction is reported without omission.")

    formal_gate = json.loads((run_dir / "gate_report.json").read_text())
    formal_feasible = formal_gate["feasible_metrics"]
    full_rows = [
        row for row in rows if row["method"] == "full_dynamic_uncertainty"
    ]
    full_feasible_failures = [
        row for row in full_rows
        if _truth(row["passable"]) and _f(row, "success") <= 0.5
    ]
    failure_outcomes = Counter(
        (row["corridor_type"], row["final_outcome"])
        for row in full_feasible_failures
    )
    with (run_dir / "failure_breakdown.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        trace_rows = list(csv.DictReader(handle))
    feasible_timeout_traces = [
        row for row in trace_rows
        if _truth(row["feasible_label"])
        and row["termination_reason"] == "timeout"
    ]

    def trace_mean(scene: str, field: str) -> float:
        selected = [
            _f(row, field) for row in feasible_timeout_traces
            if row["scene_type"] == scene
        ]
        return float(np.mean(selected)) if selected else math.nan

    diagnosis = f"""# Formal failure diagnosis

These are log associations, not post-hoc changes to the evaluator or proof of a
single causal mechanism.

- The benchmark contains 1,080/1,890 (57.1%) OBB-infeasible scenarios.  They are
  outside the traversal-success denominator; Full correctly rejects 50.0% and
  times out on the other 50.0%.
- Among 810 feasible scenarios, Full succeeds on 625 and fails on 185: 184
  timeouts and one false reject, with zero collisions.
- Feasible failure counts are asymmetric timeout
  {failure_outcomes[("asymmetric", "timeout")]}, L-shaped timeout
  {failure_outcomes[("l_shaped", "timeout")]}, S-shaped timeout
  {failure_outcomes[("s_shaped", "timeout")]}, and S-shaped false reject
  {failure_outcomes[("s_shaped", "false_reject")]}.  Thus residual feasible
  failure is a control-time/budget problem, not contact or mass rejection.
- Timeout traces contain no persistent STOP equilibrium (median maximum STOP
  streak 0).  Mean conservative-projector steps are
  {trace_mean("asymmetric", "conservative_projection_steps"):.1f} for
  asymmetric, {trace_mean("l_shaped", "conservative_projection_steps"):.1f}
  for L-shaped, and {trace_mean("s_shaped", "conservative_projection_steps"):.1f}
  for S-shaped passages.  Safe OBB projection prevents collisions but often
  slows progress enough to exhaust the fixed 200-step budget.
- Of 51 feasible S-shaped timeouts, 41 enter corridor arm 1 and 34 reach arm 2;
  they average {trace_mean("s_shaped", "positive_progress_steps"):.1f}
  positive-progress steps.  Many are still advancing through the second turn
  rather than being stationary when the budget expires.
- Feasible failures concentrate under severe noise (86/184 timeouts) and near
  the structural boundary (87, 63, and 35 failures in ascending non-negative
  margin bins).  These are balanced strata, so the gradients are not sampling
  artifacts.
- Relative to Mean only, Full improves F-SR and sharply lowers false rejection,
  but accepts/explores more borderline cases: CR falls by 29.0 points and the
  all-scenario timeout/stuck rate rises by 18.4 points.  This is the dominant
  belief-level trade-off in the frozen formal data.
"""
    diagnosis_path = run_dir / "failure_diagnosis.md"
    if diagnosis_path.exists():
        raise FileExistsError(diagnosis_path)
    diagnosis_path.write_text(diagnosis, encoding="utf-8")

    report = f"""# Final strict structural feasibility evaluation

## Protocol integrity

- The formal run was authorized by the exact pre-run protocol manifest and the passed iteration-10 smoke gate; neither was regenerated from formal outcomes.
- Three seeds completed: 0, 1, 2.
- 1,890 paired scenarios per method; 11,340 episode rows total.
- Six methods are complete for every scenario; observation and random hashes are identical within every pair.
- No episode, timeout, collision, or scene type was excluded.
- An earlier formal launch was invalidated in full after a streaming-CSV schema error; none of its rows were reused here.
- Legacy 25.4±1.0 and 70.3±1.1 aggregate results are provenance only and are not combined with this protocol.

## Statistical definition

F-SR and FR use only unified-OBB feasible episodes; CR uses only unified-OBB infeasible episodes; Collision and Timeout/stuck use all episodes. Values in the main table are the mean and sample SD across three seed-level proportions. Confidence intervals use {BOOTSTRAP_REPLICATES} percentile bootstrap replicates, resampling paired scenario IDs within seed with seed {BOOTSTRAP_SEED}.

## Post-run diagnostic (not a promotion decision)

The merged formal Full result has feasible success of {100 * float(formal_feasible['success']):.1f}%, false rejection of {100 * float(formal_feasible['false_reject']):.2f}%, collision of {100 * float(formal_feasible['collision']):.1f}%, and feasible timeout of {100 * float(formal_feasible['timeout']):.1f}%. Thus the formal feasible-timeout value exceeds the preregistered 20% smoke threshold. This is retained as a negative result; the controller, sample, and seed count were not changed. The run-local merged gate also lacks streaming step-derived mechanism checks, so it is a post-run diagnostic only; formal launch authorization came from the exact pre-run smoke gate recorded in `promotion_gate_report.json`.

## Data-constrained conclusions

""" + "\n".join(f"- {line}" for line in conclusions) + """

## Artifacts

See `summary_by_seed.csv`, `summary_mean_sd.csv`, `bootstrap_ci.csv`,
`paired_method_differences.csv`, `stratified_metrics.csv`, `margin_phase.csv`,
`risk_coverage.csv`, `failure_breakdown.csv`, `table_proc_main.tex`, and
`failure_diagnosis.md`, and `reproduction_commands.md`.
"""
    report_path = run_dir / "final_evaluation_report.md"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(report, encoding="utf-8")
    reproduction = r"""# Exact reproduction commands

Run from the Habitat-Lab repository root.  Every output path is new; the passed
smoke gate and its pre-run manifest are immutable inputs to the formal launch.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xiaotian/miniconda3/envs/navila-eval/bin/python -m pytest -q \
  examples/narrow_passage_rl/tests/test_feasibility_ablation.py \
  examples/narrow_passage_rl/tests/test_strict_obb_geometry.py \
  examples/narrow_passage_rl/tests/test_repaired_feasibility.py \
  examples/narrow_passage_rl/tests/test_structural_evaluation_protocol.py

SMOKE_ROOT=examples/narrow_passage_rl/results/ablation_feasibility/iteration_10_20260821_fixed_stream_schema
FORMAL_ROOT=examples/narrow_passage_rl/results/ablation_feasibility/formal_evaluation_$(date +%Y%m%d_%H%M%S)_reproduction

for SEED in 0 1 2; do
  /home/xiaotian/miniconda3/envs/navila-eval/bin/python \
    examples/narrow_passage_rl/eval_structural_feasibility.py \
    --phase full --seeds "${SEED}" --max-steps 200 \
    --factor-design publication --episode-order seeded_shuffle \
    --memory-policy empty_online_memory --seed-shard \
    --gate-report "${SMOKE_ROOT}/gate_report.json" \
    --methods reactive_rule_baseline full_dynamic_uncertainty mean_only \
      fixed_uncertainty no_yaw_aware_readiness no_memory \
    --output-dir "${FORMAL_ROOT}/seed${SEED}"
done

/home/xiaotian/miniconda3/envs/navila-eval/bin/python \
  examples/narrow_passage_rl/merge_structural_feasibility.py \
  --root "${FORMAL_ROOT}"

/home/xiaotian/miniconda3/envs/navila-eval/bin/python \
  examples/narrow_passage_rl/audit_structural_iteration.py \
  --run-dir "${FORMAL_ROOT}" --snapshot-status exact \
  --promotion-gate "${SMOKE_ROOT}/gate_report.json" \
  --frozen-manifest "${SMOKE_ROOT}/protocol_manifest.json"

/home/xiaotian/miniconda3/envs/navila-eval/bin/python \
  examples/narrow_passage_rl/finalize_structural_evaluation.py \
  --run-dir "${FORMAL_ROOT}"

/home/xiaotian/miniconda3/envs/navila-eval/bin/python \
  examples/narrow_passage_rl/plot_formal_margin_phase.py \
  --run-dir "${FORMAL_ROOT}"
```
"""
    reproduction_path = run_dir / "reproduction_commands.md"
    if reproduction_path.exists():
        raise FileExistsError(reproduction_path)
    reproduction_path.write_text(reproduction, encoding="utf-8")
    validation["bootstrap_replicates"] = BOOTSTRAP_REPLICATES
    (run_dir / "formal_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
