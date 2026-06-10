#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

from rl_eval_utils import write_summary_csv


ROOT = Path(__file__).resolve().parent


def parse_output(output):
    row = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        try:
            if "." in value:
                row[key] = float(value)
            else:
                row[key] = int(value)
        except ValueError:
            row[key] = value
    return row


def run_case(name, script, args, required_paths=None, strict=False):
    required_paths = required_paths or []
    missing = [str(path) for path in required_paths if path and not Path(path).exists()]
    if missing:
        print(f"[skip] {name}: missing {', '.join(missing)}")
        return None

    cmd = [sys.executable, str(ROOT / script), *map(str, args)]
    print(f"[run] {name}")
    completed = subprocess.run(
        cmd,
        cwd=ROOT.parents[1],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        print(f"[fail] {name}: return code {completed.returncode}")
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr)
        if strict:
            completed.check_returncode()
        return None
    print(completed.stdout)
    row = parse_output(completed.stdout)
    row["case"] = name
    row["script"] = script
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="hard")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/narrow_passage_rl"))
    parser.add_argument(
        "--passage-model",
        type=Path,
        default=Path("data/narrow_passage_sb3_hard/ppo_narrow_passage.zip"),
    )
    parser.add_argument(
        "--recovery-model",
        type=Path,
        default=Path("data/narrow_passage_recovery/ppo_recovery.zip"),
    )
    parser.add_argument(
        "--risk-recovery-model",
        type=Path,
        default=Path("data/narrow_passage_risk_recovery/ppo_risk_recovery.zip"),
    )
    parser.add_argument(
        "--no-failure-model",
        type=Path,
        default=Path("data/narrow_passage_ablation_no_failure/ppo_narrow_passage.zip"),
    )
    parser.add_argument(
        "--no-clearance-model",
        type=Path,
        default=Path("data/narrow_passage_ablation_no_clearance/ppo_narrow_passage.zip"),
    )
    parser.add_argument(
        "--no-curriculum-model",
        type=Path,
        default=Path("data/narrow_passage_no_curriculum/ppo_narrow_passage.zip"),
    )
    args = parser.parse_args()

    rows = []
    cases = [
        (
            "rule_baseline",
            "eval_rule_baseline.py",
            ["--difficulty", args.difficulty, "--episodes", args.episodes],
            [],
        ),
        (
            "passage_ppo",
            "eval_sb3.py",
            [
                "--model",
                args.passage_model,
                "--difficulty",
                args.difficulty,
                "--episodes",
                args.episodes,
            ],
            [args.passage_model],
        ),
        (
            "ablation_no_failure",
            "eval_sb3.py",
            [
                "--model",
                args.no_failure_model,
                "--difficulty",
                args.difficulty,
                "--ablation",
                "no_failure",
                "--episodes",
                args.episodes,
            ],
            [args.no_failure_model],
        ),
        (
            "ablation_no_clearance",
            "eval_sb3.py",
            [
                "--model",
                args.no_clearance_model,
                "--difficulty",
                args.difficulty,
                "--ablation",
                "no_clearance",
                "--episodes",
                args.episodes,
            ],
            [args.no_clearance_model],
        ),
        (
            "no_curriculum",
            "eval_sb3.py",
            [
                "--model",
                args.no_curriculum_model,
                "--difficulty",
                args.difficulty,
                "--episodes",
                args.episodes,
            ],
            [args.no_curriculum_model],
        ),
        (
            "recovery_standalone",
            "eval_recovery_sb3.py",
            [
                "--model",
                args.recovery_model,
                "--difficulty",
                args.difficulty,
                "--episodes",
                args.episodes,
            ],
            [args.recovery_model],
        ),
        (
            "risk_recovery_standalone",
            "eval_risk_recovery_sb3.py",
            [
                "--model",
                args.risk_recovery_model,
                "--difficulty",
                args.difficulty,
                "--episodes",
                args.episodes,
            ],
            [args.risk_recovery_model],
        ),
        (
            "passage_plus_collision_recovery",
            "eval_passage_with_recovery.py",
            [
                "--passage-model",
                args.passage_model,
                "--recovery-model",
                args.recovery_model,
                "--difficulty",
                args.difficulty,
                "--episodes",
                args.episodes,
            ],
            [args.passage_model, args.recovery_model],
        ),
        (
            "passage_plus_risk_recovery",
            "eval_passage_with_recovery.py",
            [
                "--passage-model",
                args.passage_model,
                "--recovery-model",
                args.recovery_model,
                "--risk-recovery-model",
                args.risk_recovery_model,
                "--enable-risk-recovery",
                "--difficulty",
                args.difficulty,
                "--episodes",
                args.episodes,
            ],
            [args.passage_model, args.recovery_model, args.risk_recovery_model],
        ),
        (
            "geometry_fsm",
            "eval_mode_fsm.py",
            [
                "--passage-model",
                args.passage_model,
                "--risk-recovery-model",
                args.risk_recovery_model,
                "--difficulty",
                args.difficulty,
                "--episodes",
                args.episodes,
            ],
            [args.passage_model, args.risk_recovery_model],
        ),
        (
            "memory_gated_fsm",
            "eval_memory_fsm.py",
            [
                "--passage-model",
                args.passage_model,
                "--risk-recovery-model",
                args.risk_recovery_model,
                "--difficulty",
                args.difficulty,
                "--episodes",
                args.episodes,
            ],
            [args.passage_model, args.risk_recovery_model],
        ),
    ]

    for name, script, case_args, required in cases:
        row = run_case(name, script, case_args, required, args.strict)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No RL experiment cases were run.")

    output_path = args.output_dir / "results_rl_summary.csv"
    write_summary_csv(output_path, rows)
    print(f"[write] {output_path}")


if __name__ == "__main__":
    main()
