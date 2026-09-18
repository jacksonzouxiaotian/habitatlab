#!/usr/bin/env python3
"""Combine unit, mechanism, and closed-loop gates before a full evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TESTS = (
    SCRIPT_DIR / "tests" / "test_feasibility_ablation.py",
    SCRIPT_DIR / "tests" / "test_strict_obb_geometry.py",
    SCRIPT_DIR / "tests" / "test_structural_evaluation_protocol.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism-report", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *(str(path) for path in TESTS)],
        cwd=SCRIPT_DIR.parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    mechanism = json.loads(args.mechanism_report.read_text(encoding="utf-8"))
    smoke = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    gates = {
        "unit_collision_geometry_and_label_tests": completed.returncode == 0,
        "large_yaw_align_then_commit": completed.returncode == 0,
        "truly_narrow_reject": completed.returncode == 0,
        "mechanism_validation": bool(mechanism.get("full_evaluation_permitted")),
        "closed_loop_smoke": bool(smoke.get("full_evaluation_permitted")),
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "pytest_command": [sys.executable, "-m", "pytest", "-q", *map(str, TESTS)],
        "pytest_returncode": completed.returncode,
        "pytest_stdout": completed.stdout.strip(),
        "pytest_stderr": completed.stderr.strip(),
        "mechanism_report": str(args.mechanism_report.resolve()),
        "smoke_report": str(args.smoke_report.resolve()),
        "gates": gates,
        "full_evaluation_permitted": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["full_evaluation_permitted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
