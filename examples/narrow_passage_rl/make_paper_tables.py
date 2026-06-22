#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


MAIN_CASES = [
    "rule_baseline",
    "passage_ppo",
    "passage_plus_collision_recovery",
    "passage_plus_risk_recovery",
    "geometry_fsm",
    "memory_gated_fsm",
]

ABLATION_CASES = [
    "passage_ppo",
    "ablation_no_failure",
    "ablation_no_clearance",
    "no_curriculum",
]

# Habitat sim generalization results (real HM3D scenes, NarrowPassageNav-v0 task)
HABITAT_CASES = [
    "habitat_ppo_baseline",
    "habitat_geometry_fsm",
    "habitat_ours_memory",
]

COLUMNS = [
    ("case", "Method"),
    ("success_rate", "Success"),
    ("collision_rate", "Collision"),
    ("final_collision_rate", "Final collision"),
    ("near_collision_rate", "Near collision"),
    ("avg_min_clearance", "Min clearance"),
    ("false_feasible_collision_rate", "False feasible collision"),
    ("narrow_collision_rate", "Narrow collision"),
    ("avg_recover_triggers", "Recover triggers"),
    ("avg_memory_writes", "Memory writes"),
    ("reject_rate", "Reject"),
]

# Columns for the Habitat generalization table (subset — memory/risk metrics not applicable)
HABITAT_COLUMNS = [
    ("case", "Method"),
    ("success_rate", "Success"),
    ("collision_rate", "Collision"),
    ("near_collision_rate", "Near collision"),
    ("avg_min_clearance", "Min clearance"),
]


def read_rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def value(row, key):
    raw = row.get(key, "")
    if raw == "" and key == "collision_rate":
        raw = row.get("final_collision_rate", "")
    if raw == "" and key == "final_collision_rate":
        raw = row.get("collision_rate", "")
    if raw == "":
        return "-"
    try:
        return f"{float(raw):.3f}"
    except ValueError:
        return raw


def select_rows(rows, cases):
    by_case = {row.get("case", ""): row for row in rows}
    return [by_case[case] for case in cases if case in by_case]


def markdown_table(rows, columns):
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(value(row, key) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def latex_table(rows, columns):
    col_spec = "l" + "c" * (len(columns) - 1)
    lines = [f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule"]
    lines.append(" & ".join(label for _, label in columns) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(value(row, key) for key, _ in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/narrow_passage_rl/results_rl_summary.csv"),
    )
    parser.add_argument(
        "--habitat-input",
        type=Path,
        default=None,
        help="Separate CSV for Habitat results (defaults to same as --input)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/narrow_passage_rl")
    )
    args = parser.parse_args()

    rows = read_rows(args.input)
    main_rows = select_rows(rows, MAIN_CASES)
    ablation_rows = select_rows(rows, ABLATION_CASES)

    habitat_csv = args.habitat_input if args.habitat_input is not None else args.input
    habitat_rows = select_rows(read_rows(habitat_csv), HABITAT_CASES)

    if main_rows:
        write_text(args.output_dir / "paper_table_main.md", markdown_table(main_rows, COLUMNS))
        write_text(args.output_dir / "paper_table_main.tex", latex_table(main_rows, COLUMNS))
        print("[write] paper_table_main.md")
        print("[write] paper_table_main.tex")

    if ablation_rows:
        write_text(args.output_dir / "paper_table_ablation.md", markdown_table(ablation_rows, COLUMNS[:8]))
        write_text(args.output_dir / "paper_table_ablation.tex", latex_table(ablation_rows, COLUMNS[:8]))
        print("[write] paper_table_ablation.md")
        print("[write] paper_table_ablation.tex")

    if habitat_rows:
        write_text(args.output_dir / "paper_table_habitat.md", markdown_table(habitat_rows, HABITAT_COLUMNS))
        write_text(args.output_dir / "paper_table_habitat.tex", latex_table(habitat_rows, HABITAT_COLUMNS))
        print("[write] paper_table_habitat.md")
        print("[write] paper_table_habitat.tex")


if __name__ == "__main__":
    main()
