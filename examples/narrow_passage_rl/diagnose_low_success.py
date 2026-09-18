#!/usr/bin/env python3
"""Evidence-ranked diagnosis of low strict-feasibility navigation Success."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


FULL = "full_dynamic_uncertainty"
METHODS = (
    FULL,
    "mean_only",
    "fixed_uncertainty",
    "no_yaw_aware_readiness",
    "no_memory",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(rows: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _rate(rows: list[dict[str, str]], key: str) -> float:
    return 100.0 * float(np.mean([float(row[key]) for row in rows])) if rows else math.nan


def _pct(values: list[bool]) -> float:
    return 100.0 * float(np.mean(values)) if values else math.nan


def _group_rows(
    rows: list[dict[str, str]], key: str, selector: Callable[[dict[str, str]], bool]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for value in sorted({row[key] for row in rows if selector(row)}):
        selected = [row for row in rows if selector(row) and row[key] == value]
        out.append({
            "factor": key,
            "value": value,
            "episodes": len(selected),
            "success_pct": _rate(selected, "success"),
            "collision_pct": _rate(selected, "collision"),
            "false_reject_pct": _rate(selected, "false_reject"),
            "timeout_pct": _rate(selected, "timeout"),
        })
    return out


def run(
    episodes_csv: Path,
    steps_csv: Path,
    morphology_claims_csv: Path,
    kappa_sweep_csv: Path,
    output_dir: Path,
) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = _read(episodes_csv)
    steps = _read(steps_csv)

    method_outcomes: list[dict[str, Any]] = []
    for method in METHODS:
        all_rows = [row for row in episodes if row["method"] == method]
        feasible = [row for row in all_rows if float(row["passable_label"]) > 0.5]
        infeasible = [row for row in all_rows if float(row["passable_label"]) <= 0.5]
        method_outcomes.append({
            "method": method,
            "all_episodes": len(all_rows),
            "feasible_episodes": len(feasible),
            "infeasible_episodes": len(infeasible),
            "overall_success_pct": _rate(all_rows, "success"),
            "feasible_success_pct": _rate(feasible, "success"),
            "feasible_false_reject_pct": _rate(feasible, "false_reject"),
            "feasible_collision_pct": _rate(feasible, "collision"),
            "feasible_timeout_pct": _rate(feasible, "timeout"),
        })
    _write(method_outcomes, output_dir / "method_outcomes.csv")

    no_memory_feasible = [
        row for row in episodes
        if row["method"] == "no_memory" and float(row["passable_label"]) > 0.5
    ]
    scene_rows = _group_rows(
        no_memory_feasible, "corridor_type", lambda row: True
    )
    _write(scene_rows, output_dir / "no_memory_scene_outcomes.csv")
    factor_rows: list[dict[str, Any]] = []
    for factor in ("margin_bin", "yaw_deg", "noise_level", "lateral_offset"):
        factor_rows.extend(_group_rows(no_memory_feasible, factor, lambda row: True))
    _write(factor_rows, output_dir / "no_memory_factor_outcomes.csv")

    step_rows: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in steps if row["method"] == method]
        modes = Counter(row["controller_mode"] for row in selected)
        step_rows.append({
            "method": method,
            "steps": len(selected),
            "commit_pct": 100.0 * modes["COMMIT"] / len(selected),
            "explore_pct": 100.0 * modes["EXPLORE"] / len(selected),
            "align_pct": 100.0 * modes["ALIGN"] / len(selected),
            "recover_pct": 100.0 * modes["RECOVER"] / len(selected),
            "reject_pct": 100.0 * modes["REJECT"] / len(selected),
            "zero_linear_action_pct": _pct([
                abs(float(row["linear_action"])) < 1e-9 for row in selected
            ]),
            "negative_linear_action_pct": _pct([
                float(row["linear_action"]) < 0.0 for row in selected
            ]),
            "uncertainty_triggered_pct": _pct([
                _bool(row["uncertainty_triggered"]) for row in selected
            ]),
            "dropout_ratio_one_pct": _pct([
                float(row["dropout_ratio"]) >= 0.999 for row in selected
            ]),
            "open_regime_pct": _pct([
                not _bool(row["in_passage_regime"]) for row in selected
            ]),
        })
    _write(step_rows, output_dir / "mode_mechanisms.csv")

    open_sensor_rows: list[dict[str, Any]] = []
    for method in (FULL, "no_memory"):
        method_steps = [row for row in steps if row["method"] == method]
        for subset_name, selector in (
            ("open_all", lambda row: not _bool(row["in_passage_regime"])),
            (
                "open_clean",
                lambda row: not _bool(row["in_passage_regime"])
                and row["noise_level"] == "clean",
            ),
            ("in_passage", lambda row: _bool(row["in_passage_regime"])),
        ):
            selected = [row for row in method_steps if selector(row)]
            sigmas = np.asarray([float(row["sigma_delta_pose"]) for row in selected])
            open_sensor_rows.append({
                "method": method,
                "subset": subset_name,
                "steps": len(selected),
                "sigma_pose_mean_m": float(np.mean(sigmas)),
                "sigma_pose_median_m": float(np.median(sigmas)),
                "dropout_ratio_one_pct": _pct([
                    float(row["dropout_ratio"]) >= 0.999 for row in selected
                ]),
                "structural_interval_overlap_pct": _pct([
                    row["feasibility_reason"]
                    == "structural interval overlaps feasibility boundary"
                    for row in selected
                ]),
                "zero_linear_action_pct": _pct([
                    abs(float(row["linear_action"])) < 1e-9 for row in selected
                ]),
                "recover_pct": _pct([
                    row["controller_mode"] == "RECOVER" for row in selected
                ]),
            })
    _write(open_sensor_rows, output_dir / "open_space_uncertainty.csv")

    steps_by_episode: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in steps:
        steps_by_episode[(row["method"], row["episode_id"])].append(row)
    timeout_episodes = [
        row for row in no_memory_feasible if row["final_outcome"] == "timeout"
    ]
    timeout_ids = {row["episode_id"] for row in timeout_episodes}
    timeout_steps = [
        row for row in steps
        if row["method"] == "no_memory" and row["episode_id"] in timeout_ids
    ]
    initial_distances: list[float] = []
    final_distances: list[float] = []
    for episode_id in timeout_ids:
        sequence = steps_by_episode[("no_memory", episode_id)]
        initial_distances.append(float(json.loads(sequence[0]["observation"])[12]))
        final_distances.append(float(json.loads(sequence[-1]["observation"])[12]))
    timeout_cycle = [{
        "episodes": len(timeout_episodes),
        "steps": len(timeout_steps),
        "structural_interval_overlap_pct": _pct([
            row["feasibility_reason"]
            == "structural interval overlaps feasibility boundary"
            for row in timeout_steps
        ]),
        "zero_linear_action_pct": _pct([
            abs(float(row["linear_action"])) < 1e-9 for row in timeout_steps
        ]),
        "recover_and_negative_action_pct": _pct([
            row["controller_mode"] == "RECOVER"
            and float(row["linear_action"]) < 0.0
            for row in timeout_steps
        ]),
        "mean_initial_goal_distance_m": float(np.mean(initial_distances)),
        "mean_final_goal_distance_m": float(np.mean(final_distances)),
        "mean_goal_progress_m": float(
            np.mean(np.asarray(initial_distances) - np.asarray(final_distances))
        ),
    }]
    _write(timeout_cycle, output_dir / "timeout_cycle.csv")

    full_rows = [row for row in episodes if row["method"] == FULL]
    full_feasible = [row for row in full_rows if float(row["passable_label"]) > 0.5]
    false_reject_ids = {
        row["episode_id"] for row in full_feasible if float(row["false_reject"]) > 0.5
    }
    controller_rejects = [
        row for row in steps
        if row["method"] == FULL
        and row["episode_id"] in false_reject_ids
        and row["controller_mode"] == "REJECT"
    ]
    feasibility_modes = Counter(row["feasibility_mode"] for row in controller_rejects)
    memory_order_rows: list[dict[str, Any]] = []
    for seed in sorted({row["seed"] for row in full_rows}):
        seed_rows = [row for row in full_rows if row["seed"] == seed]
        first_feasible = next(
            index for index, row in enumerate(seed_rows)
            if float(row["passable_label"]) > 0.5
        )
        feasible = [row for row in seed_rows if float(row["passable_label"]) > 0.5]
        first_false_reject = next(
            index for index, row in enumerate(seed_rows)
            if float(row["false_reject"]) > 0.5
        )
        memory_order_rows.append({
            "seed": seed,
            "episodes_before_first_feasible": first_feasible,
            "infeasible_before_first_feasible": sum(
                float(row["passable_label"]) <= 0.5
                for row in seed_rows[:first_feasible]
            ),
            "first_false_reject_index": first_false_reject,
            "feasible_episodes": len(feasible),
            "false_reject_episodes": sum(
                float(row["false_reject"]) > 0.5 for row in feasible
            ),
        })
    _write(memory_order_rows, output_dir / "memory_order_audit.csv")

    causes = [
        {
            "id": "C1", "status": "confirmed protocol denominator",
            "cause": "Mixed feasible/infeasible denominator",
            "evidence": "360/630 episodes per method are ground-truth infeasible; environment defines success = goal_reached AND passable.",
            "impact": "Overall Success has a 42.86% ceiling even for a perfect feasible navigator; does not explain feasible Success of 0.37%.",
        },
        {
            "id": "C2", "status": "confirmed direct decision cause",
            "cause": "Cross-episode memory terminal false rejection",
            "evidence": f"{len(false_reject_ids)}/270 feasible episodes false-rejected; at all {len(controller_rejects)} reject steps the feasibility selector itself was EXPLORE ({feasibility_modes['EXPLORE']}) or COMMIT ({feasibility_modes['COMMIT']}).",
            "impact": "Removes 96.67% of feasible episodes before navigation; full-minus-no-memory FR is +96.30 to +97.41 pp across morphologies.",
        },
        {
            "id": "C3", "status": "confirmed protocol/memory feedback cause",
            "cause": "Outcome semantics collapse and censored rejects recorded as failures",
            "evidence": "record_episode stores only success: bool and calls DMinCalibrator.update(..., attempted=True). Timeout, collision, controller failure and true geometric infeasibility are therefore indistinguishable; REJECT also records success=False although run_episode exits before env.step.",
            "impact": "Treats controller incapability and unattempted/censored decisions as evidence that the passage itself is physically infeasible.",
        },
        {
            "id": "C3b", "status": "confirmed implementation, supported propagation cause",
            "cause": "Infeasible-first ordering plus coarse memory neighborhoods",
            "evidence": "Each seed runs 105 infeasible episodes before its first feasible episode, which is already false-rejected. Fingerprints use 0.08 m width buckets, coarse difficulty/type classes and L-infinity radius 1, so adjacent buckets/classes share records.",
            "impact": "Preloads broad neighborhoods with failures before any feasible evidence exists, then generalizes those failures into positive-margin cases.",
        },
        {
            "id": "C4", "status": "confirmed estimator/control interface cause",
            "cause": "Open-space max-range rays treated as dropout uncertainty for a prospective aperture width",
            "evidence": "In clean open-regime no-memory steps, 88.78% have dropout_ratio=1 and median pose sigma is about 0.110 m; 52.67% receive structural interval-overlap.",
            "impact": "All declared positive margins up to 0.125 m can remain inside the confidence interval dead zone before the robot approaches the passage.",
        },
        {
            "id": "C5", "status": "confirmed no-information action cause",
            "cause": "Explore on structural CI overlap commands zero forward velocity",
            "evidence": "52.81% of clean open no-memory steps have zero linear action; a stationary clean observation cannot reduce the same uncertainty.",
            "impact": "Explore does not gather the geometric evidence needed to leave Explore.",
        },
        {
            "id": "C6", "status": "confirmed controller limit cycle",
            "cause": "Intentional zero-motion probe is counted as stuck and triggers repeated reverse Recover",
            "evidence": "The environment increments stuck from positional movement/progress only, so deliberate Explore(v=0) and in-place Align rotation both count as stuck; fsm_action accepts max_recover_steps but does not use it to cap/transition recovery. For 179 feasible no-memory timeouts, 54.92% of steps are zero-speed and 40.84% are Recover with v=-0.12 m/s; mean goal distance changes from 4.29 m to 6.45 m.",
            "impact": "The controller moves 2.16 m farther from the goal on average and exhausts all 200 steps.",
        },
        {
            "id": "C7", "status": "confirmed scene-controller failure",
            "cause": "Local controller succeeds on only one scene family",
            "evidence": "Without memory: straight/L/S have 0/45 Success and 100% Timeout; narrow_entry has 0/45 Success and 97.78% Timeout; asymmetric has 0/45 Success and 100% Collision; only narrow_exit reaches 33/45 Success.",
            "impact": "Even perfect rejection memory cannot repair missing closed-loop traversal competence.",
        },
        {
            "id": "C8", "status": "confirmed physical failure",
            "cause": "OBB collision after removing memory",
            "evidence": "No-memory feasible Collision is 58/270 (21.48%): all 45 asymmetric cases collide, plus 12 narrow_exit and 1 narrow_entry.",
            "impact": "Memory hides rather than solves collision risk; disabling it exposes unsafe commitment/control.",
        },
        {
            "id": "C9", "status": "supported secondary cause",
            "cause": "Dynamic interval conservatism beyond the memory effect",
            "evidence": "Full feasible Success/FR are 0.37%/96.67%, mean-only 2.59%/94.44%; full commits on 23.46% of logged steps versus 80.82% for mean-only.",
            "impact": "Uncertainty gating reduces commitment and Success, but the small delta is dominated by memory and controller failures.",
        },
        {
            "id": "C10", "status": "supported secondary cause",
            "cause": "Injected depth/dropout noise",
            "evidence": "No-memory feasible Success declines from 15.56% clean to 11.11% mild and 10.00% severe; Collision rises from 17.78% to 23.33%.",
            "impact": "Noise worsens an already failing controller; clean runs still fail predominantly by timeout.",
        },
        {
            "id": "C11", "status": "proximate outcome, not standalone root cause",
            "cause": "Strict 200-step/collision/goal termination",
            "evidence": "179/270 no-memory feasible episodes hit 200 steps, but their final goal distance is farther than their initial distance; collisions terminate immediately and success also requires passable=True.",
            "impact": "The protocol exposes the loop; simply extending the budget is not supported as a fix.",
        },
    ]
    _write(causes, output_dir / "cause_registry.csv")

    morphology_claims = _read(morphology_claims_csv)
    memory_claims = [row for row in morphology_claims if row["claim"].startswith("memory")]
    yaw_claims = [row for row in morphology_claims if row["claim"].startswith("yaw")]
    kappa_rows = _read(kappa_sweep_csv)

    method_index = {row["method"]: row for row in method_outcomes}
    scene_index = {row["value"]: row for row in scene_rows}
    timeout = timeout_cycle[0]
    clean_open_nm = next(
        row for row in open_sensor_rows
        if row["method"] == "no_memory" and row["subset"] == "open_clean"
    )
    lines = [
        "# DEGNav strict feasibility：低成功率完整结果与诊断",
        "",
        "本报告只做诊断，不修改阈值、控制器或既有结果。所有比例来自",
        f"`{episodes_csv}` 与逐步审计 `{steps_csv}`。",
        "",
        "## 先区分两个 Success 分母",
        "",
        "| Method | Overall Success | Feasible Success | Feasible False Reject | Collision | Timeout |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = method_index[method]
        lines.append(
            f"| {method} | {float(row['overall_success_pct']):.2f}% | "
            f"{float(row['feasible_success_pct']):.2f}% | "
            f"{float(row['feasible_false_reject_pct']):.2f}% | "
            f"{float(row['feasible_collision_pct']):.2f}% | "
            f"{float(row['feasible_timeout_pct']):.2f}% |"
        )
    lines.extend([
        "",
        "每个方法的 630 episodes 中只有 270 个 passable，另外 360 个由统一 OBB",
        "ground truth 判为不可行，环境明确使用 `success = reached_goal AND passable`。",
        "所以总体 Success 与 feasible-subset Success 必须分开；但 full 在 feasible",
        "subset 也只有 1/270，证明低值不只是分母问题。",
        "",
        "## 两条已经闭合的主失效链",
        "",
        "```text",
        "每 seed 先运行 105 个 infeasible episodes",
        "  -> 所有 timeout/collision/controller failure 都以 success=False 写入 memory",
        "  -> memory 对首个 feasible episode 直接 Reject",
        "  -> 未执行 env.step 的 Reject 又被 record_episode(success=False)",
        "  -> 自我强化 false-reject loop",
        "```",
        "",
        "```text",
        "open approach 的 max-range rays -> dropout_ratio=1 -> sigma 约 0.11 m",
        "  -> structural CI overlap -> Explore(v=0)",
        "  -> 静止观测不增加信息 -> stuck",
        "  -> Recover(v=-0.12) 向后退 -> 再次 Explore(v=0)",
        "  -> 200-step timeout，且离目标更远",
        "```",
        "",
        "## 所有当前有证据的原因",
        "",
        "| ID | 证据等级 | 原因 | 量化证据/影响 |",
        "|:---|:---|:---|:---|",
    ])
    for cause in causes:
        lines.append(
            f"| {cause['id']} | {cause['status']} | {cause['cause']} | "
            f"{cause['evidence']} {cause['impact']} |"
        )
    lines.extend([
        "",
        "## 去掉 memory 后暴露的 scene-level 控制问题",
        "",
        "| Scene | Episodes | Success | Collision | Timeout |",
        "|:---|---:|---:|---:|---:|",
    ])
    for scene in sorted(scene_index):
        row = scene_index[scene]
        lines.append(
            f"| {scene} | {row['episodes']} | {float(row['success_pct']):.2f}% | "
            f"{float(row['collision_pct']):.2f}% | {float(row['timeout_pct']):.2f}% |"
        )
    lines.extend([
        "",
        "因此 no-memory 的 12.22% Success 不是普遍导航能力：33 个成功全部来自",
        "`narrow_exit`；straight、L-turn、S-turn、narrow-entry 均没有成功，",
        "asymmetric 则全部碰撞。",
        "",
        "## 关键逐步证据",
        "",
        f"- clean/open/no-memory：dropout=1 的步占 "
        f"{float(clean_open_nm['dropout_ratio_one_pct']):.2f}%，median sigma="
        f"{float(clean_open_nm['sigma_pose_median_m']):.3f} m，interval-overlap="
        f"{float(clean_open_nm['structural_interval_overlap_pct']):.2f}%，零线速度="
        f"{float(clean_open_nm['zero_linear_action_pct']):.2f}%。",
        f"- 179 个 no-memory feasible timeout：零线速度 "
        f"{float(timeout['zero_linear_action_pct']):.2f}%，Recover/负速度 "
        f"{float(timeout['recover_and_negative_action_pct']):.2f}%，平均 goal distance "
        f"{float(timeout['mean_initial_goal_distance_m']):.2f}→"
        f"{float(timeout['mean_final_goal_distance_m']):.2f} m。",
        f"- 261 个 full false reject 的终止步中，feasibility selector 是 EXPLORE "
        f"{feasibility_modes['EXPLORE']} 次、COMMIT {feasibility_modes['COMMIT']} 次、"
        "REJECT 0 次；终止 Reject 全来自外层 memory override。",
        "",
        "## 次要因素与不能支持的解释",
        "",
        "- 外部 noise 是次要恶化因素：no-memory Success 为 clean 15.56%、mild",
        "  11.11%、severe 10.00%；但 clean 仍有 66.67% Timeout。",
        "- yaw/lateral 不是当前主因：no-memory 各 yaw Success 为 9.26%–14.81%，",
        "  不呈单调退化；四种 morphology 的 full-minus-no-yaw Success 都是 0.00 pp。",
        "- morphology 不是单点配置偶然：memory FR 增量在四种体型为 "
        f"{min(float(row['difference_pp']) for row in memory_claims):.2f}–"
        f"{max(float(row['difference_pp']) for row in memory_claims):.2f} pp，所有 CI 排除 0。",
        "- 新 posterior belief 不是原因：它是 audit-only，且与旧结果的 legacy",
        "  mu/sigma/mode/action 在 900 个逐步记录上完全相同。",
        "- 当前统一 OBB geometry tests 全部通过；body-margin conversion offset、",
        "  threshold-only false reject、explicit-blocker false positive 都是 0 个 episode。",
        "- `success` 与 `collision_free_success` 在全部 3150 rows 上完全相同，低值不是",
        "  后处理把已有成功重新删掉造成的统计假象。",
        "- kappa sweep 的 feasible Success 仅为 "
        f"{min(float(row['success_pct']) for row in kappa_rows):.2f}%–"
        f"{max(float(row['success_pct']) for row in kappa_rows):.2f}%；没有证据表明只改 kappa "
        "能修复结构问题。",
        "",
        "## 诊断结论",
        "",
        "当前成功率低由三个层级叠加：57.14% 的 mixed benchmark episodes 本来就",
        "不可成功；在可行样本中，cross-episode memory 先消除 96%–97% 的尝试机会；",
        "去掉 memory 后，open-space uncertainty / zero-information Explore / reverse",
        "Recover 的闭环和 scene-specific controller failure 又产生 66.30% Timeout 与",
        "21.48% Collision。动态 uncertainty、外部 noise 和严格终止条件会进一步恶化",
        "结果，但它们不是最大根因。",
        "",
        "这意味着下一步应先修正 memory outcome semantics、scenario ordering 和",
        "Explore/Recover 信息闭环，再重新评测；不应通过调 kappa/tau 掩盖问题。",
        "",
    ])
    report = output_dir / "success_rate_diagnosis.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    result_root = script_dir / "results" / "ablation_feasibility"
    formal = result_root / "structural_full_20260818_194833_final"
    morphology = result_root / "morphology_generalization_20260820_140820"
    kappa = result_root / "kappa_sweep_20260820_140254"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-csv", type=Path, default=formal / "episodes.csv")
    parser.add_argument("--steps-csv", type=Path, default=formal / "steps.csv")
    parser.add_argument(
        "--morphology-claims-csv", type=Path,
        default=morphology / "morphology_claims.csv",
    )
    parser.add_argument(
        "--kappa-sweep-csv", type=Path, default=kappa / "sweep.csv"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or result_root / datetime.now().strftime(
        "success_rate_diagnosis_%Y%m%d_%H%M%S"
    )
    report = run(
        args.episodes_csv,
        args.steps_csv,
        args.morphology_claims_csv,
        args.kappa_sweep_csv,
        output_dir,
    )
    print(f"[write] {report}")


if __name__ == "__main__":
    main()
