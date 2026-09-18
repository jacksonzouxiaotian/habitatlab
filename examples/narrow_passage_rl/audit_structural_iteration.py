#!/usr/bin/env python3
"""Create immutable audit artifacts for a strict structural evaluation run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
RUNTIME_FILES = (
    SCRIPT_DIR / "procedural_env_v2.py",
    SCRIPT_DIR / "eval_harder_benchmark.py",
    SCRIPT_DIR / "eval_structural_feasibility.py",
    SCRIPT_DIR / "cross_episode_memory.py",
    SCRIPT_DIR / "narrow_passage/models/dynamic_feasibility.py",
    SCRIPT_DIR / "narrow_passage/models/feasibility_ablation.py",
    SCRIPT_DIR / "narrow_passage/models/robot_morphology.py",
)
TESTS = (
    SCRIPT_DIR / "tests/test_feasibility_ablation.py",
    SCRIPT_DIR / "tests/test_strict_obb_geometry.py",
    SCRIPT_DIR / "tests/test_repaired_feasibility.py",
    SCRIPT_DIR / "tests/test_structural_evaluation_protocol.py",
)
ANALYSIS_FILES = (
    SCRIPT_DIR / "audit_structural_iteration.py",
    SCRIPT_DIR / "finalize_structural_evaluation.py",
    SCRIPT_DIR / "merge_structural_feasibility.py",
)
SNAPSHOT_FILES = (*RUNTIME_FILES, *TESTS, *ANALYSIS_FILES)
TIMELINE_FIELDS = (
    "scenario_id", "corridor_type", "margin_bin", "passable_label",
    "yaw_deg", "lateral_offset", "noise_level", "step", "robot_pose",
    "local_goal", "active_corridor_arm", "D_hat", "W_req_mean",
    "W_req_cons", "margin_cons", "p_feas", "uncertainty",
    "selected_mode", "candidate_action", "executed_action",
    "swept_obb_clearance", "collision_flag", "stuck_score", "progress",
    "termination_reason", "controller_mode", "passage_control_state",
    "safe_action_projection_reason", "corner_turn_locked",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_patch() -> str:
    tracked = subprocess.run(
        ["git", "diff", "--binary", "--", *(str(path.relative_to(REPO)) for path in SNAPSHOT_FILES)],
        cwd=REPO, text=True, capture_output=True, check=False,
    ).stdout
    pieces = [tracked]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *(str(path.relative_to(REPO)) for path in SNAPSHOT_FILES)],
        cwd=REPO, text=True, capture_output=True, check=False,
    ).stdout.splitlines()
    for line in status:
        if not line.startswith("?? "):
            continue
        path = REPO / line[3:]
        result = subprocess.run(
            ["git", "diff", "--no-index", "--binary", "/dev/null", str(path)],
            cwd=REPO, text=True, capture_output=True, check=False,
        )
        pieces.append(result.stdout)
    return "".join(pieces)


def _semantics() -> str:
    return """# Frozen ablation semantics

All belief variants use identical scenarios, observations, random draws, morphology,
FSM, swept-OBB projector, velocity limits, success definition, and 200-step budget.

| Method | Changed component | Components retained |
|---|---|---|
| `reactive_rule_baseline` | No feasibility belief, memory, or interval selector | Unified OBB environment and outcome definitions |
| `full_dynamic_uncertainty` | Reference: dynamic mean/sigma and dual structural/pose interval gate | All components |
| `mean_only` | Selector reads structural and pose means only; dynamic uncertainty is computed and logged but cannot affect mode | Geometry, yaw projection, alignment, memory, low-level control |
| `fixed_uncertainty` | Replaces scene-dependent sigma with one fixed global sigma | Means, yaw projection, interval gate, memory, low-level control |
| `no_yaw_aware_readiness` | In the current implementation this removes yaw projection from pose-required width (`use_yaw_prior=False`); it does not remove the shared heading-alignment FSM | Dynamic sigma, interval gate, memory, low-level control |
| `no_memory` | Removes recurrence-risk input and memory-triggered correction/rejection | Belief, yaw projection, alignment, low-level control |

`point_estimate` and `no_uncertainty` are compatibility aliases of `mean_only` and
must not be reported as independent rows.  The historical name
`no_yaw_aware_readiness` is retained because it was preregistered in the smoke
protocol; paper text must state its actual implementation above and must not call it
"w/o yaw-aware prior" without this qualification.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--snapshot-status", choices=("exact", "posthoc"), default="exact"
    )
    parser.add_argument(
        "--promotion-gate",
        type=Path,
        help=(
            "Frozen smoke gate that authorized the formal run.  When omitted, "
            "the run-local gate_report.json is used."
        ),
    )
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        help=(
            "Pre-run protocol_manifest.json.  Its runtime hashes are verified "
            "and it is preserved verbatim in a formal result directory; the "
            "post-run audit is then written separately."
        ),
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    episodes = _read_csv(run_dir / "episodes.csv")
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    formal_gate_path = run_dir / "gate_report.json"
    promotion_gate_path = (
        args.promotion_gate.resolve() if args.promotion_gate else formal_gate_path
    )
    formal_gate = json.loads(formal_gate_path.read_text())
    promotion_gate = json.loads(promotion_gate_path.read_text())
    methods = list(metadata["methods"])

    paired: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in episodes:
        paired[row["scenario_id"]].append(row)
    paired_audit = []
    for scenario_id, rows in sorted(paired.items()):
        observed = sorted({row["method"] for row in rows})
        obs = {row["observation_sha256"] for row in rows}
        random = {row["random_draw_sha256"] for row in rows}
        paired_audit.append({
            "scenario_id": scenario_id,
            "method_count": len(observed),
            "methods": ";".join(observed),
            "expected_methods": ";".join(sorted(methods)),
            "observation_hash_count": len(obs),
            "random_hash_count": len(random),
            "paired_methods_complete": observed == sorted(methods),
            "observation_hash_identical": len(obs) == 1,
            "random_hash_identical": len(random) == 1,
            "paired_audit_pass": (
                observed == sorted(methods) and len(obs) == 1 and len(random) == 1
            ),
        })
    _write_csv(run_dir / "paired_hash_audit.csv", paired_audit)

    failures = {
        row["episode_id"]: row for row in episodes
        if row["method"] == "full_dynamic_uncertainty"
        and float(row.get("success", 0.0)) <= 0.5
    }
    s_timeouts = {
        episode_id for episode_id, row in failures.items()
        if row.get("corridor_type") == "s_shaped"
        and row.get("final_outcome") == "timeout"
    }
    stats = defaultdict(lambda: {
        "arms": [], "stop_streak": 0, "max_stop_streak": 0,
        "align_steps": 0, "recover_steps": 0, "positive_progress_steps": 0,
        "conservative_projection_steps": 0, "rows": 0,
    })
    timelines: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (run_dir / "steps.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("method") != "full_dynamic_uncertainty":
                continue
            episode_id = row.get("episode_id", "")
            if episode_id not in failures:
                continue
            item = stats[episode_id]
            arm = int(float(row.get("active_corridor_arm", -1)))
            item["arms"].append(arm)
            is_stop = (
                abs(float(row.get("linear_action", 0.0))) < 1e-9
                and abs(float(row.get("angular_action", 0.0))) < 1e-9
            )
            item["stop_streak"] = item["stop_streak"] + 1 if is_stop else 0
            item["max_stop_streak"] = max(
                item["max_stop_streak"], item["stop_streak"]
            )
            item["align_steps"] += str(row.get("controller_mode", "")).upper() == "ALIGN"
            item["recover_steps"] += str(row.get("controller_mode", "")).upper() == "RECOVER"
            item["positive_progress_steps"] += float(row.get("progress", 0.0)) > 1e-3
            item["conservative_projection_steps"] += str(
                row.get("safe_action_projection_reason", "")
            ) not in {"", "not_checked", "desired_safe"}
            item["rows"] += 1
            if episode_id in s_timeouts:
                timelines[episode_id].append({
                    key: row.get(key, "") for key in TIMELINE_FIELDS
                })

    failure_rows = []
    for episode_id, row in sorted(failures.items()):
        item = stats[episode_id]
        arms = item["arms"]
        failure_rows.append({
            "episode_id": episode_id,
            "scenario_id": row["scenario_id"],
            "scene_type": row["corridor_type"],
            "margin_bin": row["margin_bin"],
            "feasible_label": row["passable"],
            "entry_yaw": row["entry_yaw"],
            "lateral_offset": row["initial_lateral_offset"],
            "noise_level": row["noise_level"],
            "termination_reason": row["final_outcome"],
            "steps": row["steps"],
            "path_length": row["path_length"],
            "entered_first_turn": 1 in arms,
            "completed_first_turn": 2 in arms,
            "switched_to_second_corridor_arm": 2 in arms,
            "first_arm_1_trace_index": (
                arms.index(1) if 1 in arms else ""
            ),
            "first_arm_2_trace_index": (
                arms.index(2) if 2 in arms else ""
            ),
            "active_arms": ";".join(map(str, sorted(set(arms)))),
            "max_stop_streak": item["max_stop_streak"],
            "align_steps": item["align_steps"],
            "recover_steps": item["recover_steps"],
            "positive_progress_steps": item["positive_progress_steps"],
            "conservative_projection_steps": item["conservative_projection_steps"],
            "collision": row["collision"],
            "false_reject": row["false_reject"],
        })
    _write_csv(run_dir / "failure_breakdown.csv", failure_rows)
    timeline_dir = run_dir / "s_shape_timelines"
    timeline_dir.mkdir(exist_ok=False)
    for episode_id, rows in timelines.items():
        _write_csv(timeline_dir / f"{episode_id}.csv", rows)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    (run_dir / "code_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (run_dir / "git_diff.patch").write_text(_git_patch(), encoding="utf-8")
    test_env = dict(os.environ)
    test_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *map(str, TESTS)],
        cwd=REPO, env=test_env, text=True, capture_output=True, check=False,
    )
    test_report = (
        f"returncode={completed.returncode}\ncommand={sys.executable} -m pytest -q "
        + " ".join(map(str, TESTS)) + "\n\nSTDOUT\n" + completed.stdout
        + "\nSTDERR\n" + completed.stderr
    )
    (run_dir / "regression_test_report.txt").write_text(
        test_report, encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError("Regression tests failed; refusing to freeze manifest")

    audit_manifest = {
        "schema": "degnav_strict_structural_protocol_v2",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_status": args.snapshot_status,
        "git_commit": commit,
        "working_tree_dirty": bool(_git_patch()),
        "runtime_source_sha256": {
            str(path.relative_to(REPO)): _sha256(path) for path in RUNTIME_FILES
        },
        "analysis_and_test_source_sha256": {
            str(path.relative_to(REPO)): _sha256(path)
            for path in (*TESTS, *ANALYSIS_FILES)
        },
        "run_metadata": metadata,
        "formal_gate_report_sha256": _sha256(formal_gate_path),
        "promotion_gate_report": str(promotion_gate_path),
        "promotion_gate_report_sha256": _sha256(promotion_gate_path),
        "all_runtime_gates_passed": bool(
            promotion_gate.get("full_evaluation_permitted")
        ),
        "all_regression_tests_passed": True,
        "outcome_definitions": {
            "F-SR": "success / OBB-labelled feasible episodes",
            "CR": "correct_reject / OBB-labelled infeasible episodes",
            "FR": "false_reject / OBB-labelled feasible episodes",
            "Collision": "collision / all episodes",
            "Timeout/stuck": "(timeout OR stuck) / all episodes",
        },
        "bootstrap": {
            "replicates": 2000,
            "seed": 20260821,
            "cluster": "paired scenario_id, stratified by seed",
            "interval": "percentile 95%",
        },
        "formal_contract": {
            "seeds": [0, 1, 2], "max_steps": 200,
            "factor_design": "publication", "scenarios_per_seed": 630,
            "scenarios_per_method": 1890, "selective_reruns": False,
            "protected_legacy_result": "paper_dynamic_20260818_170356",
        },
    }
    if args.frozen_manifest:
        frozen_manifest_path = args.frozen_manifest.resolve()
        frozen_text = frozen_manifest_path.read_text(encoding="utf-8")
        frozen = json.loads(frozen_text)
        if not frozen.get("all_runtime_gates_passed"):
            raise RuntimeError("Pre-run frozen manifest did not pass all gates")
        if frozen.get("snapshot_status") != "exact":
            raise RuntimeError("Pre-run manifest is not an exact snapshot")
        current_hashes = audit_manifest["runtime_source_sha256"]
        if frozen.get("runtime_source_sha256") != current_hashes:
            raise RuntimeError(
                "Runtime source changed after protocol freeze; formal run is invalid"
            )
        protocol_out = run_dir / "protocol_manifest.json"
        if protocol_out.exists():
            raise FileExistsError(f"Refusing to overwrite {protocol_out}")
        protocol_out.write_text(frozen_text, encoding="utf-8")
        promotion_gate_out = run_dir / "promotion_gate_report.json"
        if promotion_gate_out.exists():
            raise FileExistsError(f"Refusing to overwrite {promotion_gate_out}")
        promotion_gate_out.write_text(
            promotion_gate_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        audit_manifest["pre_run_protocol_manifest"] = str(frozen_manifest_path)
        audit_manifest["pre_run_protocol_manifest_sha256"] = _sha256(
            frozen_manifest_path
        )
        audit_manifest["portable_promotion_gate_report"] = (
            "promotion_gate_report.json"
        )
        audit_out = run_dir / "formal_audit_manifest.json"
    else:
        audit_out = run_dir / "protocol_manifest.json"
    if audit_out.exists():
        raise FileExistsError(f"Refusing to overwrite {audit_out}")
    audit_out.write_text(
        json.dumps(audit_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "ablation_semantics.md").write_text(
        _semantics(), encoding="utf-8"
    )
    change_log = f"""# Engineering change log

- Snapshot status: `{args.snapshot_status}`.
- Fixed zero-measure L/S turning chambers by generating finite OBB turning plateaus.
- Made the shared swept-OBB projection continuous in low-speed translation and rotation; STOP is used only when forward, rotation, and bounded backtrack are all unsafe.
- Corrected the shared action-budget inconsistency: Approach/Enter now use the declared 0.15 m/s configuration; the original 0.08 m/s corner speed is retained.
- Added entry-only pure-pursuit lateral correction and removed duplicate symmetric clearance feedback; only one-sided clearance residual affects the asymmetric correction.
- Added bounded reverse lateral correction for a failed wall-adjacent commitment.
- Canonical outcome flags are mutually exclusive and complete; all required per-step diagnostic fields are logged.
- Every change applies to all belief variants and reads neither method identity nor feasibility label.
- Regression tests: `{completed.stdout.strip()}`.
- The formal run was authorized only by the frozen smoke gate at `{promotion_gate_path}`; it reports `full_evaluation_permitted={promotion_gate.get('full_evaluation_permitted')}`.  The merged formal `gate_report.json` is retained as a result-set diagnostic and is not retroactively used to authorize the run.
"""
    (run_dir / "engineering_change_log.md").write_text(
        change_log, encoding="utf-8"
    )
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], text=True,
        capture_output=True, check=False,
    ).stdout
    (run_dir / "package_versions.txt").write_text(
        f"python={sys.version}\nexecutable={sys.executable}\n\n{packages}",
        encoding="utf-8",
    )
    print(json.dumps({
        "run_dir": str(run_dir), "paired_scenarios": len(paired),
        "failures": len(failures), "s_timeouts": len(s_timeouts),
        "tests": completed.returncode,
        "promotion_full_evaluation_permitted": promotion_gate.get(
            "full_evaluation_permitted"
        ),
        "formal_diagnostic_full_evaluation_permitted": formal_gate.get(
            "full_evaluation_permitted"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
