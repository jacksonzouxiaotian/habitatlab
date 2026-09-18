#!/usr/bin/env python3
"""Audit the saved Compact Habitat Stress Evaluation protocol.

This audit is deliberately read-only with respect to historical result files.
It separates facts present in the CSV from settings inferred from the current
runner/config source, because the historical stress CSV has no run manifest or
observation/random hashes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "narrow_passage_rl"
DEFAULT_STRESS = RESULTS / "habitat_stress_validation.csv"
DEFAULT_SET_A = RESULTS / "habitat_fsm_v2_episodes.csv"
DEFAULT_DATASET = Path(
    "/media/xiaotian/ACD525D1B7A093D9/habitat_data/"
    "datasets/narrow_passage/val/val.json.gz"
)

METHODS = {
    "apf_gap": "APF+Gap",
    "fsm_full": "DEGNav full",
    "fsm_no_heading_alignment": "DEGNav w/o heading alignment",
}
REFERENCE = "fsm_full"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def id_hash(ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt") as f:
        return list(json.load(f).get("episodes", []))


def source_contract(repo_root: Path) -> dict[str, object]:
    runner = repo_root / "examples/narrow_passage_rl/eval_habitat_stress_validation.py"
    task_cfg = repo_root / "habitat-lab/habitat/config/habitat/task/narrow_passage.yaml"
    benchmark_cfg = (
        repo_root
        / "habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_habitat_test.yaml"
    )
    runner_text = runner.read_text()
    task_text = task_cfg.read_text()
    benchmark_text = benchmark_cfg.read_text()
    required = {
        "method_specific_rng": "_stable_int(method, stress.name)" in runner_text,
        "apf_raw_depth": "_depth_from_obs(obs, stress, rng)" in runner_text,
        "fsm_features": "_features_from_obs(env, obs, stress, rng)" in runner_text,
        "max_steps_500": 'default=500' in runner_text and "max_episode_steps: 500" in benchmark_text,
        "success_distance_050": "success_distance: 0.5" in task_text,
        "heading_threshold_052": "heading_threshold: 0.52" in task_text,
        "lateral_threshold_040": "lateral_threshold: 0.4" in task_text,
        "require_stop_false": "require_stop: False" in task_text,
        "extreme_filter": "body_margin < args.extreme_margin" in runner_text,
        "extreme_margin_005": '"--extreme-margin", type=float, default=0.05' in runner_text,
    }
    if not all(required.values()):
        missing = sorted(k for k, value in required.items() if not value)
        raise RuntimeError(f"Current source contract changed; audit rules need review: {missing}")
    return {
        "runner": str(runner.relative_to(repo_root)),
        "task_config": str(task_cfg.relative_to(repo_root)),
        "benchmark_config": str(benchmark_cfg.relative_to(repo_root)),
        "max_steps": 500,
        "success_distance_m": 0.5,
        "heading_threshold_rad": 0.52,
        "lateral_threshold_m": 0.4,
        "require_stop": False,
        "checks": required,
    }


def audit_sets_and_protocol(rows: list[dict[str, str]], contract: dict) -> tuple[list[dict], list[str]]:
    by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("method") in METHODS:
            by_key[(row["stress"], row["method"])].append(row)

    conditions = sorted({stress for stress, _ in by_key})
    report: list[dict] = []
    differences: list[str] = []
    random_conditions = {"drop010", "drop030", "noise002", "noise005"}
    for stress in conditions:
        ref = by_key.get((stress, REFERENCE), [])
        ref_ids = {row["episode_id"] for row in ref}
        ref_params = {
            (row["yaw_deg"], row["lateral_m"], row["depth_dropout"], row["feature_noise"])
            for row in ref
        }
        for method, paper_name in METHODS.items():
            values = by_key.get((stress, method), [])
            ids = {row["episode_id"] for row in values}
            params = {
                (row["yaw_deg"], row["lateral_m"], row["depth_dropout"], row["feature_noise"])
                for row in values
            }
            same_ids = ids == ref_ids
            same_params = params == ref_params
            is_apf = method == "apf_gap"
            randomized = stress in random_conditions
            report.append(
                {
                    "condition": stress,
                    "method": paper_name,
                    "csv_method": method,
                    "episodes": len(values),
                    "unique_episode_ids": len(ids),
                    "episode_id_sha256": id_hash(ids),
                    "same_episode_ids_as_full": bool_text(same_ids),
                    "same_stress_parameters_as_full": bool_text(same_params),
                    "shared_habitat_sensor_config": "yes (source-default evidence)",
                    "controller_input": "raw depth + PointGoal" if is_apf else "19-D narrow_passage_features",
                    "identical_controller_input_across_methods": "no",
                    "perturbation_domain": (
                        "depth pixels/thirds" if is_apf else "geometry features/six sectors"
                    ),
                    "random_draws_paired_across_methods": (
                        "not applicable (deterministic condition)"
                        if not randomized
                        else "no; RNG seed includes method name"
                    ),
                    "observation_hash_logged": "no",
                    "random_hash_logged": "no",
                    "action_budget": contract["max_steps"],
                    "same_action_budget": "yes (source-default evidence)",
                    "success_definition": "distance<=0.50m, |heading|<0.52rad, |lateral|<0.40m, no STOP required",
                    "same_success_definition": "yes (shared Habitat measure; source-default evidence)",
                    "max_observed_steps": max((int(float(v["steps"])) for v in values), default=0),
                }
            )
            if not same_ids:
                differences.append(f"{stress}: {paper_name} uses a different episode-id set")
            if not same_params:
                differences.append(f"{stress}: {paper_name} has different saved stress parameters")

    differences.extend(
        [
            "APF+Gap consumes perturbed raw depth + PointGoal; DEGNav variants consume perturbed 19-D geometry features.",
            "For dropout/noise conditions, the RNG seed includes the method name, so perturbation draws are not paired across methods.",
            "Historical CSV rows contain neither observation hashes nor random-draw hashes; realized-observation equality is not auditable.",
            "The 500-step budget and success thresholds are inferred from current source defaults because the historical CSV has no run manifest.",
        ]
    )
    return report, differences


def audit_157_to_151(set_a: list[dict[str, str]], stress_rows: list[dict[str, str]]) -> tuple[list[dict], dict]:
    set_b_rows = [
        row for row in stress_rows if row.get("stress") == "nominal" and row.get("method") == REFERENCE
    ]
    a_by_id = {row["episode_id"]: row for row in set_a}
    b_by_id = {row["episode_id"]: row for row in set_b_rows}
    label_only = sorted(set(a_by_id) - set(b_by_id))
    same_metadata = 0
    for episode_id in set(a_by_id) & set(b_by_id):
        a, b = a_by_id[episode_id], b_by_id[episode_id]
        same_scene = a.get("scene_id") == b.get("scene_id")
        try:
            same_margin = abs(float(a.get("body_margin", "nan")) - float(b.get("body_margin", "nan"))) < 1e-6
        except ValueError:
            same_margin = False
        same_metadata += int(same_scene and same_margin)

    rows = [
        {
            "episode_id_label_in_set_A_not_set_B": episode_id,
            "status": "identifier_only_not_proven_exclusion",
            "exclusion_reason": "not available; no explicit 157-to-151 exclusion rule exists",
            "provenance_note": "episode IDs are reassigned after each independent mining run",
        }
        for episode_id in label_only
    ]
    facts = {
        "set_a_rows": len(set_a),
        "set_a_unique_ids": len(a_by_id),
        "set_b_nominal_rows": len(set_b_rows),
        "set_b_unique_ids": len(b_by_id),
        "id_label_difference": label_only,
        "matching_id_labels_with_identical_scene_and_body_margin": same_metadata,
        "set_b_difficulty_counts": Counter(row.get("difficulty", "?") for row in set_b_rows),
        "conclusion": (
            "未找到显式排除规则，不能给出六个真实被排除 episode 的逐条原因；"
            "现有代码与计数只证明独立挖掘的 Val set B 窄通道配额少 6 条，候选级原因未被日志保留。"
        ),
    }
    return rows, facts


def audit_extreme(stress_rows: list[dict[str, str]], dataset: list[dict]) -> tuple[list[dict], dict]:
    nominal = [
        row for row in stress_rows if row.get("stress") == "nominal" and row.get("method") == REFERENCE
    ]
    extreme = [
        row for row in stress_rows if row.get("stress") == "extreme_yaw60" and row.get("method") == REFERENCE
    ]
    expected = {row["episode_id"] for row in nominal if float(row["body_margin"]) < 0.05}
    actual = {row["episode_id"] for row in extreme}
    dataset_expected = {
        str(ep.get("episode_id"))
        for ep in dataset
        if float((ep.get("info") or {}).get("body_margin", "inf")) < 0.05
    }
    detail = [
        {
            "episode_id": row["episode_id"],
            "scene_id": row["scene_id"],
            "body_margin_m": row["body_margin"],
            "selection_rule": "unperturbed episode.info.body_margin < 0.05 m",
            "yaw_applied_after_selection_deg": 60,
        }
        for row in sorted(extreme, key=lambda item: item["episode_id"])
    ]
    facts = {
        "nominal_count": len(nominal),
        "nominal_body_margin_lt_005": len(expected),
        "extreme_yaw60_count": len(extreme),
        "actual_equals_nominal_filter": actual == expected,
        "dataset_count": len(dataset),
        "dataset_body_margin_lt_005": len(dataset_expected) if dataset else None,
        "actual_equals_dataset_filter": actual == dataset_expected if dataset else None,
        "selection_occurs_before_yaw": True,
        "automatic_post_perturbation_geometry_filter": False,
    }
    return detail, facts


def write_markdown(out_dir: Path, exclusion: dict, extreme: dict, differences: list[str]) -> None:
    label_ids = exclusion["id_label_difference"]
    difficulty = exclusion["set_b_difficulty_counts"]
    lines = [
        "# Supplementary S3 — Habitat protocol provenance audit",
        "",
        "## 157 与 151 条记录",
        "",
        (
            "Val set A（157 条）与 Val set B（151 条）是两次独立挖掘输出。"
            "在生成扰动表的执行路径中，未找到把 157 条输入过滤为 151 条的规则；"
            "`generate_habitat_episodes.py` 对已接受 anchor 逐条写入，"
            "`eval_habitat_stress_validation.py` 对 nominal 条件逐条评测当前数据集。"
        ),
        "",
        (
            f"Val set B 的组成是 narrow={difficulty.get('narrow', 0)}、"
            f"normal={difficulty.get('normal', 0)}、wide={difficulty.get('wide', 0)}。"
            "相对挖掘器声明的 79/55/23 配额，只有 narrow 少 6 条；挖掘器在场景与尝试次数耗尽时会写出已找到的全部样本，"
            "并且只在候选被接受后分配 episode id，因此失败候选没有 episode id，也没有逐候选拒绝日志。"
        ),
        "",
        (
            "按字符串集合比较，Set A 中而 Set B 中没有的标签为 "
            + ", ".join(f"`{x}`" for x in label_ids)
            + "。这些标签不能解释为被排除的六个真实 episode：ID 在每次独立挖掘完成后按行号重新生成，"
            f"且两个结果中同名 ID 只有 {exclusion['matching_id_labels_with_identical_scene_and_body_margin']} 条具有相同 scene 与 body-margin。"
        ),
        "",
        "> 未找到显式排除规则，不能给出六个真实被排除 episode 的逐条原因；差异的候选级原因不明。",
        "",
        "## Extreme + yaw 60° 子集",
        "",
        (
            f"Extreme 子集由 151 条 nominal episode 的未扰动元数据按 "
            f"`episode.info.body_margin < 0.05 m` 筛选，得到 {extreme['nominal_body_margin_lt_005']} 条；"
            "随后才施加 60° 起始 yaw，得到 Extreme + yaw 60° 的 "
            f"{extreme['extreme_yaw60_count']} 条记录。代码没有在扰动后按可执行性再次剔除 episode。"
            "因此它属于预先定义的最窄 margin 子采样（选项 a），不是扰动后自动剔除（选项 b）。"
        ),
        "",
        "## 跨方法协议结论",
        "",
        "保存结果证明三个方法在每个条件下使用相同 episode-id 集合与相同扰动参数。当前共享 runner/config 规定相同的 500-step 预算和相同 Habitat success measure。以下差异阻止论文写成“同一实际 observation/同一随机扰动”：",
        "",
    ]
    lines.extend(f"- {item}" for item in differences)
    (out_dir / "supplementary_s3.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-csv", type=Path, default=DEFAULT_STRESS)
    parser.add_argument("--set-a-csv", type=Path, default=DEFAULT_SET_A)
    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    repo_root = HERE.parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or RESULTS / "audits" / f"habitat_protocol_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    stress_rows = read_csv(args.stress_csv)
    set_a = read_csv(args.set_a_csv)
    dataset = load_dataset(args.dataset_json)
    contract = source_contract(repo_root)
    protocol_rows, differences = audit_sets_and_protocol(stress_rows, contract)
    exclusions, exclusion_facts = audit_157_to_151(set_a, stress_rows)
    extreme_rows, extreme_facts = audit_extreme(stress_rows, dataset)

    write_csv(out_dir / "protocol_audit.csv", protocol_rows)
    write_csv(out_dir / "157_to_151_identifier_difference.csv", exclusions)
    write_csv(out_dir / "extreme_yaw60_subset.csv", extreme_rows)
    write_markdown(out_dir, exclusion_facts, extreme_facts, differences)
    manifest = {
        "stress_csv": str(args.stress_csv),
        "set_a_csv": str(args.set_a_csv),
        "dataset_json": str(args.dataset_json),
        "dataset_available": args.dataset_json.exists(),
        "source_contract": contract,
        "exclusion_audit": {**exclusion_facts, "set_b_difficulty_counts": dict(exclusion_facts["set_b_difficulty_counts"])},
        "extreme_audit": extreme_facts,
        "differences": differences,
    }
    (out_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(out_dir)


if __name__ == "__main__":
    main()
