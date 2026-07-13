#!/usr/bin/env python3

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from narrow_passage.metrics.strict_metrics import compute_strict_metrics


RESULTS_DIR = Path(__file__).parent / "results" / "narrow_passage_rl"

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
# Order: strongest baselines first, ours last
HABITAT_CASES = [
    "habitat_ppo_baseline",
    "habitat_ppo_policy",
    "habitat_apf_gap",
    "habitat_geometry_fsm",
    "habitat_ours_memory",
]

# FSM ablation on HM3D val set
HABITAT_ABLATION_CASES = [
    "habitat_geometry_fsm",
    "habitat_fsm_no_recovery",
    "habitat_fsm_no_alignment",
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
    ("success_rate", "Success ↑"),
    ("narrow_sr", "Narrow ↑"),
    ("normal_sr", "Normal ↑"),
    ("wide_sr", "Wide ↑"),
    ("collision_rate", "Collision ↓"),
    ("avg_min_clearance", "Min clearance"),
]

BELIEF_MODE_COLUMNS = [
    ("method", "Method"),
    ("input", "Input"),
    ("decision_type", "Decision type"),
    ("overall_sr", "Overall SR"),
    ("strict_sr", "Strict SR"),
    ("collision", "Collision"),
    ("near_collision", "Near collision"),
    ("reject", "Reject"),
    ("correct_reject", "Correct reject"),
    ("false_reject", "False reject"),
]

BELIEF_MODE_ABLATION_COLUMNS = [
    ("ablation", "Ablation"),
    ("input", "Policy input"),
    ("overall_sr", "Overall SR"),
    ("strict_sr", "Strict SR"),
    ("collision", "Collision"),
    ("near_collision", "Near collision"),
    ("reject", "Reject"),
    ("correct_reject", "Correct reject"),
    ("false_reject", "False reject"),
]

BELIEF_MODE_ABLATIONS = [
    "full",
    "no_p_feas",
    "no_delta_var",
    "no_memory",
    "no_alignment",
    "geometry_only",
]

BELIEF_MODE_ABLATION_INPUTS = {
    "full": "All belief features",
    "no_p_feas": "p_feas replaced by 0.5",
    "no_delta_var": "delta_var replaced by constant",
    "no_memory": "memory_risk replaced by 0",
    "no_alignment": "heading/lateral errors replaced by 0",
    "geometry_only": "d_hat, body margin, clearances, heading/lateral",
}

BELIEF_MODE_METHODS = [
    {
        "method": "PPO direct velocity (single-run mined-val)",
        "input": "19-D geometry observation; single-run provenance",
        "decision_type": "Direct velocity",
        "path": "habitat_ppo_v2_mined_val.csv",
    },
    {
        "method": "SAC direct velocity",
        "input": "19-D geometry observation",
        "decision_type": "Direct velocity",
        "path": "habitat_sac_v2_mined_val.csv",
    },
    {
        "method": "TD3 direct velocity",
        "input": "19-D geometry observation",
        "decision_type": "Direct velocity",
        "path": "habitat_td3_v2_mined_val.csv",
    },
    {
        "method": "APF+Gap",
        "input": "Depth + local goal",
        "decision_type": "Classical reactive planner",
        "path": "habitat_apf_gap_episodes.csv",
    },
    {
        "method": "DEGNAV-Rule / Geometry-FSM",
        "input": "19-D geometry observation + rule belief",
        "decision_type": "Rule mode selector",
        "path": "habitat_fsm_mined_val.csv",
    },
]


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
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


def _display_path(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path


def _mean(rows, key):
    vals = []
    for row in rows:
        raw = row.get(key, "")
        if raw == "":
            continue
        try:
            vals.append(float(raw))
        except ValueError:
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def _fmt_metric(value):
    if value is None:
        return "not run"
    return f"{100.0 * float(value):.1f}%"


def _fmt_latex(value):
    text = _fmt_metric(value)
    return text.replace("%", r"\%")


def _strict_values(rows):
    if not rows:
        return None
    if "strict_success" in rows[0]:
        return _mean(rows, "strict_success")
    values = []
    for row in rows:
        if "success" not in row or "min_clearance" not in row:
            continue
        try:
            metrics = compute_strict_metrics(
                success=float(row.get("success", 0.0)),
                collision=float(row.get("collision", 0.0)),
                stuck=float(row.get("stuck", 0.0)),
                min_clearance=float(row.get("min_clearance", 0.0)),
            )
            values.append(metrics["strict_success"])
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _near_collision_value(rows):
    if not rows:
        return None
    if "near_collision" in rows[0]:
        return _mean(rows, "near_collision")
    values = []
    for row in rows:
        if "min_clearance" not in row:
            continue
        try:
            metrics = compute_strict_metrics(
                success=float(row.get("success", 0.0)),
                collision=float(row.get("collision", 0.0)),
                stuck=float(row.get("stuck", 0.0)),
                min_clearance=float(row.get("min_clearance", 0.0)),
            )
            values.append(metrics["near_collision"])
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _summary_from_episode_csv(path):
    rows = read_rows(path)
    return _summary_from_episode_rows(rows)


def _summary_from_episode_rows(rows):
    if not rows:
        return {}
    return {
        "overall_sr": _mean(rows, "success"),
        "strict_sr": _strict_values(rows),
        "collision": _mean(rows, "collision"),
        "near_collision": _near_collision_value(rows),
        "reject": _mean(rows, "reject") if "reject" in rows[0] else _mean(rows, "rejected"),
        "correct_reject": _mean(rows, "correct_reject"),
        "false_reject": _mean(rows, "false_reject"),
    }


def _belief_mode_paths(output_dir, explicit_path=None):
    candidates = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            output_dir / "eval_belief_mode_ppo.csv",
            output_dir / "degnav_rl_belief_mode_ppo_eval.csv",
            output_dir / "belief_mode_ppo_eval.csv",
            output_dir / "belief_mode_ppo_quick_eval.csv",
        ]
    )
    seen = set()
    unique = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _belief_mode_main_paths(output_dir, explicit_path=None, explicit_paths=None):
    candidates = []
    for path in explicit_paths or []:
        candidates.append(Path(path))
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    candidates.extend(sorted(output_dir.glob("belief_mode_full_seed*_eval.csv")))
    candidates.extend(_belief_mode_paths(output_dir))

    seen = set()
    unique = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _belief_mode_ablation_paths(output_dir, explicit_paths=None):
    candidates = []
    for path in explicit_paths or []:
        candidates.append(Path(path))
    patterns = [
        "belief_mode_*_seed*_eval.csv",
        "*belief*mode*ppo*.csv",
        "*degnav*belief*ppo*.csv",
        "eval_belief_mode_ppo*.csv",
    ]
    for pattern in patterns:
        candidates.extend(sorted(output_dir.glob(pattern)))
    seen = set()
    unique = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _belief_mode_table_rows(output_dir, belief_mode_input=None, belief_mode_inputs=None):
    rows = []
    notes = []

    for spec in BELIEF_MODE_METHODS:
        path = output_dir / spec["path"]
        metrics = _summary_from_episode_csv(path)
        if not metrics:
            notes.append(f"{spec['method']}: no CSV found at `{path}`.")
        row = {
            "method": spec["method"],
            "input": spec["input"],
            "decision_type": spec["decision_type"],
            "overall_sr": metrics.get("overall_sr"),
            "strict_sr": metrics.get("strict_sr"),
            "collision": metrics.get("collision"),
            "near_collision": metrics.get("near_collision"),
            "reject": metrics.get("reject"),
            "correct_reject": metrics.get("correct_reject"),
            "false_reject": metrics.get("false_reject"),
        }
        rows.append(row)

    belief_paths = [
        path
        for path in _belief_mode_main_paths(
            output_dir, belief_mode_input, belief_mode_inputs
        )
        if path.exists()
    ]
    if not belief_paths:
        notes.append(
            "DEGNAV-RL: no evaluation CSV found. Expected one of "
            "`belief_mode_full_seed*_eval.csv`, `eval_belief_mode_ppo.csv`, "
            "`degnav_rl_belief_mode_ppo_eval.csv`, `belief_mode_ppo_eval.csv`, "
            "or `belief_mode_ppo_quick_eval.csv`."
        )
        metrics = {}
    else:
        belief_rows = []
        for path in belief_paths:
            belief_rows.extend(read_rows(path))
        metrics = _summary_from_episode_rows(belief_rows)
        joined = ", ".join(f"`{_display_path(p)}`" for p in belief_paths)
        notes.append(f"DEGNAV-RL row loaded from {joined}.")

    rows.append(
        {
            "method": "DEGNAV-RL",
            "input": "Explicit feasibility belief state",
            "decision_type": "Learned high-level mode selector",
            "overall_sr": metrics.get("overall_sr"),
            "strict_sr": metrics.get("strict_sr"),
            "collision": metrics.get("collision"),
            "near_collision": metrics.get("near_collision"),
            "reject": metrics.get("reject"),
            "correct_reject": metrics.get("correct_reject"),
            "false_reject": metrics.get("false_reject"),
        }
    )
    return rows, notes


def _infer_belief_ablation(path):
    name = Path(path).stem.lower()
    for ablation in BELIEF_MODE_ABLATIONS:
        if ablation in name:
            return ablation
    return "full"


def _belief_mode_ablation_table_rows(output_dir, explicit_paths=None):
    grouped = {ablation: [] for ablation in BELIEF_MODE_ABLATIONS}
    loaded_paths = []
    for path in _belief_mode_ablation_paths(output_dir, explicit_paths):
        rows = read_rows(path)
        if not rows:
            continue
        loaded_paths.append(path)
        inferred = _infer_belief_ablation(path)
        for row in rows:
            ablation = row.get("ablation", "").strip() or inferred
            if ablation not in grouped:
                continue
            grouped[ablation].append(row)

    table_rows = []
    for ablation in BELIEF_MODE_ABLATIONS:
        metrics = _summary_from_episode_rows(grouped[ablation])
        table_rows.append(
            {
                "ablation": ablation,
                "input": BELIEF_MODE_ABLATION_INPUTS[ablation],
                "overall_sr": metrics.get("overall_sr"),
                "strict_sr": metrics.get("strict_sr"),
                "collision": metrics.get("collision"),
                "near_collision": metrics.get("near_collision"),
                "reject": metrics.get("reject"),
                "correct_reject": metrics.get("correct_reject"),
                "false_reject": metrics.get("false_reject"),
            }
        )

    notes = []
    if loaded_paths:
        joined = ", ".join(f"`{_display_path(p)}`" for p in loaded_paths)
        notes.append(f"Loaded DEGNAV-RL ablation CSVs from {joined}.")
    else:
        notes.append("No DEGNAV-RL ablation CSVs found; rows are marked as not run.")
    return table_rows, notes


def belief_mode_markdown(rows, notes):
    header = "| " + " | ".join(label for _, label in BELIEF_MODE_COLUMNS) + " |"
    sep = "| " + " | ".join(":---" if i < 3 else "---:" for i, _ in enumerate(BELIEF_MODE_COLUMNS)) + " |"
    body = []
    for row in rows:
        cells = []
        for key, _label in BELIEF_MODE_COLUMNS:
            if key in {"method", "input", "decision_type"}:
                cells.append(str(row[key]))
            else:
                cells.append(_fmt_metric(row.get(key)))
        body.append("| " + " | ".join(cells) + " |")

    note_lines = [
        "",
        "Notes:",
        "- DEGNAV-RL learns only the high-level mode selector pi(m_t | b_t). In the current setup, it remains unsafe and does not reliably use Recover or Reject. We therefore report DEGNAV-RL as a diagnostic learning variant rather than as the main method.",
        "- The main method remains DEGNAV-Rule / Geometry-FSM.",
        "- Learning-only PPO/SAC/TD3 baselines output direct velocity actions from geometry observations.",
        "- The DEGNAV-RL row is a procedural v2 mode-selection result; Habitat DEGNAV-RL evaluation is not included yet.",
        "- Do not claim DEGNAV-RL improves over DEGNAV-Rule or solves the task.",
        "- The PPO direct-velocity row here is a single-run mined-val provenance row when loaded from `habitat_ppo_v2_mined_val.csv`; the formal main PPO result remains 2.1% +/- 2.6% in `paper_table_formal_baselines.md`.",
    ]
    for note in notes:
        note_lines.append(f"- {note}")
    intro = [
        "# Table: DEGNAV-RL Diagnostic Belief-Mode Comparison",
        "",
        "DEGNAV-RL learns only the high-level mode selector pi(m_t | b_t). In the current setup, it remains unsafe and does not reliably use Recover or Reject. We therefore report DEGNAV-RL as a diagnostic learning variant rather than as the main method.",
        "",
    ]
    return "\n".join([*intro, header, sep, *body, *note_lines])


def belief_mode_latex(rows, notes):
    del notes  # Notes are written in Markdown; LaTeX contains the table only.
    lines = [
        r"\begin{tabular}{lllrrrrrrr}",
        r"\toprule",
        "Method & Input & Decision type & Overall SR & Strict SR & Collision & Near collision & Reject & Correct reject & False reject \\\\",
        r"\midrule",
    ]
    for row in rows:
        cells = []
        for key, _label in BELIEF_MODE_COLUMNS:
            if key in {"method", "input", "decision_type"}:
                cells.append(str(row[key]).replace("_", r"\_"))
            else:
                cells.append(_fmt_latex(row.get(key)))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def belief_mode_ablation_markdown(rows, notes):
    header = "| " + " | ".join(label for _, label in BELIEF_MODE_ABLATION_COLUMNS) + " |"
    sep = "| " + " | ".join(":---" if i < 2 else "---:" for i, _ in enumerate(BELIEF_MODE_ABLATION_COLUMNS)) + " |"
    body = []
    for row in rows:
        cells = []
        for key, _label in BELIEF_MODE_ABLATION_COLUMNS:
            if key in {"ablation", "input"}:
                cells.append(str(row[key]))
            else:
                cells.append(_fmt_metric(row.get(key)))
        body.append("| " + " | ".join(cells) + " |")
    note_lines = [
        "",
        "Notes:",
        "- These ablations test the learned high-level mode selector input, not the low-level mode-conditioned controller.",
        "- DEGNAV-RL learns only the high-level mode selector pi(m_t | b_t). In the current setup, it remains unsafe and does not reliably use Recover or Reject. We therefore report DEGNAV-RL as a diagnostic learning variant rather than as the main method.",
        "- The full belief state is not better than `geometry_only` in this run, so this table should not be used to claim that belief-guided PPO solves the task.",
    ]
    for note in notes:
        note_lines.append(f"- {note}")
    return "\n".join(["# Table: DEGNAV-RL Diagnostic Belief-State Ablation", "", header, sep, *body, *note_lines])


def belief_mode_ablation_latex(rows, notes):
    del notes
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        "Ablation & Policy input & Overall SR & Strict SR & Collision & Near collision & Reject & Correct reject & False reject \\\\",
        r"\midrule",
    ]
    for row in rows:
        cells = []
        for key, _label in BELIEF_MODE_ABLATION_COLUMNS:
            if key in {"ablation", "input"}:
                cells.append(str(row[key]).replace("_", r"\_"))
            else:
                cells.append(_fmt_latex(row.get(key)))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


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
        "--output-dir", type=Path, default=RESULTS_DIR
    )
    parser.add_argument(
        "--belief-mode-input",
        type=Path,
        default=None,
        help="Optional single DEGNAV-RL eval CSV. Prefer --belief-mode-inputs for multi-seed tables.",
    )
    parser.add_argument(
        "--belief-mode-inputs",
        nargs="*",
        type=Path,
        default=None,
        help="Optional DEGNAV-RL eval CSVs for the main belief-mode row.",
    )
    parser.add_argument(
        "--belief-mode-ablation-inputs",
        nargs="*",
        type=Path,
        default=None,
        help="Optional DEGNAV-RL ablation eval CSVs. If omitted, common result names are searched.",
    )
    args = parser.parse_args()

    rows = read_rows(args.input)
    main_rows = select_rows(rows, MAIN_CASES)
    ablation_rows = select_rows(rows, ABLATION_CASES)

    habitat_csv = args.habitat_input if args.habitat_input is not None else args.input
    habitat_rows = select_rows(read_rows(habitat_csv), HABITAT_CASES)
    habitat_ablation_rows = select_rows(read_rows(habitat_csv), HABITAT_ABLATION_CASES)

    if main_rows:
        legacy_dir = args.output_dir / "legacy"
        write_text(
            legacy_dir / "legacy_paper_table_main_old_procedural.md",
            markdown_table(main_rows, COLUMNS),
        )
        write_text(
            legacy_dir / "legacy_paper_table_main_old_procedural.tex",
            latex_table(main_rows, COLUMNS),
        )
        print("[write] legacy/legacy_paper_table_main_old_procedural.md")
        print("[write] legacy/legacy_paper_table_main_old_procedural.tex")

    if ablation_rows:
        legacy_dir = args.output_dir / "legacy"
        write_text(
            legacy_dir / "legacy_paper_table_ablation_old_cases.md",
            markdown_table(ablation_rows, COLUMNS[:8]),
        )
        write_text(
            legacy_dir / "legacy_paper_table_ablation_old_cases.tex",
            latex_table(ablation_rows, COLUMNS[:8]),
        )
        print("[write] legacy/legacy_paper_table_ablation_old_cases.md")
        print("[write] legacy/legacy_paper_table_ablation_old_cases.tex")

    if habitat_rows:
        write_text(args.output_dir / "paper_table_habitat.md", markdown_table(habitat_rows, HABITAT_COLUMNS))
        write_text(args.output_dir / "paper_table_habitat.tex", latex_table(habitat_rows, HABITAT_COLUMNS))
        print("[write] paper_table_habitat.md")
        print("[write] paper_table_habitat.tex")

    if habitat_ablation_rows:
        write_text(args.output_dir / "paper_table_habitat_ablation.md",
                   markdown_table(habitat_ablation_rows, HABITAT_COLUMNS))
        write_text(args.output_dir / "paper_table_habitat_ablation.tex",
                   latex_table(habitat_ablation_rows, HABITAT_COLUMNS))
        print("[write] paper_table_habitat_ablation.md")
        print("[write] paper_table_habitat_ablation.tex")

    belief_rows, belief_notes = _belief_mode_table_rows(
        args.output_dir, args.belief_mode_input, args.belief_mode_inputs
    )
    write_text(
        args.output_dir / "paper_table_belief_mode_rl.md",
        belief_mode_markdown(belief_rows, belief_notes),
    )
    write_text(
        args.output_dir / "paper_table_degnav_rl_diagnostic.md",
        belief_mode_markdown(belief_rows, belief_notes),
    )
    write_text(
        args.output_dir / "paper_table_belief_mode_rl.tex",
        belief_mode_latex(belief_rows, belief_notes),
    )
    print("[write] paper_table_belief_mode_rl.md")
    print("[write] paper_table_degnav_rl_diagnostic.md")
    print("[write] paper_table_belief_mode_rl.tex")

    ablation_rows, ablation_notes = _belief_mode_ablation_table_rows(
        args.output_dir, args.belief_mode_ablation_inputs
    )
    write_text(
        args.output_dir / "paper_table_belief_mode_ablation.md",
        belief_mode_ablation_markdown(ablation_rows, ablation_notes),
    )
    write_text(
        args.output_dir / "paper_table_belief_mode_ablation.tex",
        belief_mode_ablation_latex(ablation_rows, ablation_notes),
    )
    print("[write] paper_table_belief_mode_ablation.md")
    print("[write] paper_table_belief_mode_ablation.tex")


if __name__ == "__main__":
    main()
