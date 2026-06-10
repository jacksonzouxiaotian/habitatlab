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
        "--output-dir", type=Path, default=Path("results/narrow_passage_rl")
    )
    args = parser.parse_args()

    rows = read_rows(args.input)
    main_rows = select_rows(rows, MAIN_CASES)
    ablation_rows = select_rows(rows, ABLATION_CASES)

    main_md = markdown_table(main_rows, COLUMNS)
    ablation_md = markdown_table(ablation_rows, COLUMNS[:8])
    main_tex = latex_table(main_rows, COLUMNS)
    ablation_tex = latex_table(ablation_rows, COLUMNS[:8])

    write_text(args.output_dir / "paper_table_main.md", main_md)
    write_text(args.output_dir / "paper_table_ablation.md", ablation_md)
    write_text(args.output_dir / "paper_table_main.tex", main_tex)
    write_text(args.output_dir / "paper_table_ablation.tex", ablation_tex)

    print("[write] paper_table_main.md")
    print("[write] paper_table_ablation.md")
    print("[write] paper_table_main.tex")
    print("[write] paper_table_ablation.tex")


if __name__ == "__main__":
    main()
