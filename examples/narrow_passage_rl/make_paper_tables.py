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

FALSE_FEASIBLE_OUTCOME_COLUMNS = [
    ("method", "Method"),
    ("episodes", "Episodes"),
    ("success", "Traversal success"),
    ("reject", "Explicit reject"),
    ("correct_reject", "Correct reject"),
    ("collision", "Collision"),
    ("near_collision", "Near collision"),
    ("timeout_stuck", "Timeout/stuck"),
    ("wasted_attempt", "Wasted attempts"),
]

CALIBRATION_EXTENDED_COLUMNS = [
    ("prior_width", "Prior W_hat"),
    ("interpretation", "Type"),
    ("final_posterior_mean", "Final posterior mean"),
    ("final_posterior_q95", "Final q95"),
    ("mean_p_feas", "Mean p_feas"),
    ("empirical_feasible_rate", "Empirical feasible"),
    ("reject_rate", "Reject"),
    ("unsafe_attempt_rate", "Unsafe attempt"),
    ("brier", "Brier"),
    ("ece", "ECE"),
]

CALIBRATION_EXTENDED_COMMAND = (
    "python examples/narrow_passage_rl/eval_dmin_calibration.py "
    "--priors 0.26 0.31 0.36 0.46 0.56 --true-width 0.36 "
    "--episodes 300 --seeds 0 1 2 "
    "--output-dir examples/narrow_passage_rl/results/narrow_passage_rl"
)

HABITAT_STRESS_COMMAND = (
    "python examples/narrow_passage_rl/eval_habitat_stress_validation.py "
    "--preset paper --split val --num-episodes -1"
)

HABITAT_RQ1_SUMMARY = (
    RESULTS_DIR
    / "audits"
    / "habitat_rq1_20260821_110829"
    / "summary.csv"
)
HABITAT_RQ1_COMMAND = (
    "conda run -n habitat python "
    "examples/narrow_passage_rl/eval_habitat_feasibility_ablations.py --verbose"
)

FALSE_FEASIBLE_METHOD_LABELS = {
    "rule_baseline": "Reactive rule baseline",
    "geometry_fsm": "DEGNAV-Rule",
    "fsm_no_recovery": "DEGNAV-Rule w/o recovery",
    "fsm_no_alignment": "DEGNAV-Rule w/o alignment",
    "fsm_local_memory": "DEGNAV-Rule + local memory",
    "fsm_cross_memory": "DEGNAV-Rule + cross-episode memory",
}

PROCEDURAL_MAIN_METHODS = [
    ("rule_baseline", "Reactive rule baseline"),
    ("geometry_fsm", "DEGNAV-Rule"),
]

PROCEDURAL_MAIN_COLUMNS = [
    ("method", "Method"),
    ("overall", "Overall"),
    ("straight", "Straight"),
    ("l_shaped", "L-shaped"),
    ("s_shaped", "S-shaped"),
    ("narrow_exit", "Narrow exit"),
    ("narrow_entry", "Narrow entry"),
    ("asymmetric", "Asymmetric"),
    ("false_feasible_success", "False-feasible traversal success"),
]

PROCEDURAL_MAIN_CORRIDORS = [
    ("straight", "straight"),
    ("l_shaped", "l_shaped"),
    ("s_shaped", "s_shaped"),
    ("narrow_exit", "narrow_exit"),
    ("narrow_entry", "narrow_entry"),
    ("asymmetric", "asymmetric"),
]

PROCEDURAL_MAIN_COMMAND = (
    "python examples/narrow_passage_rl/eval_harder_benchmark.py "
    "--methods rule_baseline geometry_fsm --episodes 500 --seeds 42 43 44"
)

PROCEDURAL_CORE_ABLATION_COLUMNS = [
    ("variant", "Variant"),
    ("overall", "Overall"),
    ("delta_overall", "Δ Overall vs full"),
    ("straight", "Straight"),
    ("l_shaped", "L-shaped"),
    ("s_shaped", "S-shaped"),
    ("narrow_entry", "Narrow entry"),
    ("asymmetric", "Asymmetric"),
    ("collision", "Collision"),
    ("timeout_stuck", "Timeout/stuck"),
]

PROCEDURAL_CORE_ABLATION_ORDER = [
    "full",
    "no_alignment",
    "no_recovery",
    "deterministic_margin",
    "no_yaw_prior",
]

PROCEDURAL_CORE_ABLATION_LABELS = {
    "full": "DEGNAV full",
    "no_alignment": "w/o alignment",
    "no_recovery": "w/o recovery",
    "deterministic_margin": "deterministic margin",
    "no_yaw_prior": "w/o yaw prior",
}

PROCEDURAL_CORE_ABLATION_OPTIONAL_ORDER = [
    "no_uncertainty",
    "no_free_space_continuation",
]

PROCEDURAL_CORE_ABLATION_OPTIONAL_LABELS = {
    "no_uncertainty": "w/o uncertainty",
    "no_free_space_continuation": "w/o free-space continuation",
}

PROCEDURAL_CORE_CORRIDORS = [
    ("straight", "straight"),
    ("l_shaped", "l_shaped"),
    ("s_shaped", "s_shaped"),
    ("narrow_exit", "narrow_exit"),
    ("narrow_entry", "narrow_entry"),
    ("asymmetric", "asymmetric"),
]

PROCEDURAL_CORE_COMMAND = (
    "python examples/narrow_passage_rl/eval_harder_benchmark.py "
    "--variants full no_alignment no_recovery deterministic_margin no_yaw_prior "
    "--episodes 500 --seeds 0 1 2 --log-belief-diagnostics "
    "--log-outcome-decomposition --output-csv "
    "examples/narrow_passage_rl/results/narrow_passage_rl/raw/procedural_ablation_core.csv"
)


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


def _fmt_percent_for_table(value):
    if value is None:
        return "not run"
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "not run"


def _fmt_percent_latex_for_table(value):
    return _fmt_percent_for_table(value).replace("%", r"\%")


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
        "- DEGNAV-RL is included as a diagnostic policy rather than a competitive final method. Under the current reward and action interface, the learned policy collapses to Commit and Explore and does not demonstrate meaningful Recover or Reject behavior.",
        "- The main method remains DEGNAV-Rule / Geometry-FSM.",
        "- Learning-only PPO/SAC/TD3 baselines output direct velocity actions from geometry observations.",
        "- The DEGNAV-RL row is a procedural v2 mode-selection result; Habitat DEGNAV-RL evaluation is not included yet.",
        "- DEGNAV-RL full-belief mode usage is Commit: 31.0%, Explore: 69.0%, Recover: 0.0%, Reject: 0.0%; it should not be presented as learned recovery or learned rejection.",
        "- Do not claim DEGNAV-RL improves over DEGNAV-Rule or solves the task.",
        "- The PPO direct-velocity row here is a single-run mined-val provenance row when loaded from `habitat_ppo_v2_mined_val.csv`; the 3-seed PPO Val set A result is listed in `paper_table_formal_baselines.md`, which is a mixed-split Habitat diagnostic comparison rather than a fair main ranking.",
    ]
    for note in notes:
        note_lines.append(f"- {note}")
    intro = [
        "# Table: DEGNAV-RL Diagnostic Belief-Mode Comparison",
        "",
        "DEGNAV-RL is included as a diagnostic policy rather than a competitive final method. Under the current reward and action interface, the learned policy collapses to Commit and Explore and does not demonstrate meaningful Recover or Reject behavior.",
        "",
        "This table is a diagnostic provenance comparison, not a main-paper leaderboard: the DEGNAV-RL row is procedural v2, while the direct-control rows are Habitat diagnostics or mixed-provenance baselines.",
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
        "- DEGNAV-RL is included as a diagnostic policy rather than a competitive final method. Under the current reward and action interface, the learned policy collapses to Commit and Explore and does not demonstrate meaningful Recover or Reject behavior.",
        "- The full belief state is not better than `geometry_only` in this run, so this table should not be used to claim that belief-guided PPO solves the task.",
        "- Full-belief mode usage over 1500 evaluation episodes is Commit: 31.0%, Explore: 69.0%, Recover: 0.0%, Reject: 0.0%.",
    ]
    for note in notes:
        note_lines.append(f"- {note}")
    return "\n".join([
        "# Table: DEGNAV-RL Diagnostic Belief-State Ablation",
        "",
        "DEGNAV-RL is included as a diagnostic policy rather than a competitive final method. Under the current reward and action interface, the learned policy collapses to Commit and Explore and does not demonstrate meaningful Recover or Reject behavior.",
        "",
        header,
        sep,
        *body,
        *note_lines,
    ])


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


def _ff_bool_value(row, key):
    if key == "timeout_stuck":
        return float(
            _ff_bool_value(row, "timeout") > 0.5
            or _ff_bool_value(row, "stuck") > 0.5
        )
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _false_feasible_metric_summary(rows, key):
    if not rows:
        return {"mean": None, "std": None, "count": 0, "total": 0, "seeds": 0}
    seeds = sorted({row.get("seed", "") for row in rows})
    seed_rates = []
    total_count = 0
    total_rows = 0
    for seed in seeds:
        seed_rows = [row for row in rows if row.get("seed", "") == seed]
        if not seed_rows:
            continue
        values = [float(_ff_bool_value(row, key) > 0.5) for row in seed_rows]
        seed_rates.append(sum(values) / len(values))
        total_count += int(sum(values))
        total_rows += len(values)
    mean, std = _mean_std(seed_rates)
    return {
        "mean": mean,
        "std": std,
        "count": total_count,
        "total": total_rows,
        "seeds": len(seed_rates),
    }


def _fmt_rate_with_count(metric):
    if not isinstance(metric, dict) or metric.get("mean") is None:
        return "not run"
    return (
        f"{100.0 * metric['mean']:.1f}±{100.0 * metric.get('std', 0.0):.1f}% "
        f"({metric.get('count', 0)}/{metric.get('total', 0)})"
    )


def _fmt_rate_with_count_latex(metric):
    return _fmt_rate_with_count(metric).replace("%", r"\%").replace("±", r"$\pm$")


def _false_feasible_outcome_rows(path):
    rows = read_rows(path)
    if not rows:
        return [], [], []
    filtered = [
        row for row in rows
        if str(row.get("corridor_type", "")).lower() == "false_feasible"
        or str(row.get("is_false_feasible", "")).lower() in {"1", "1.0", "true"}
    ]
    if not filtered:
        return [], [], [f"No false-feasible rows found in `{_display_path(path)}`."]

    available_methods = []
    for method in ["rule_baseline", "geometry_fsm"]:
        if any(row.get("method", "") == method for row in filtered):
            available_methods.append(method)

    table_rows = []
    summary_rows = []
    for method in available_methods:
        subset = [row for row in filtered if row.get("method", "") == method]
        row_out = {
            "method": FALSE_FEASIBLE_METHOD_LABELS.get(method, method),
            "method_key": method,
            "episodes": len(subset),
            "seeds": len({row.get("seed", "") for row in subset}),
        }
        for key, _label in FALSE_FEASIBLE_OUTCOME_COLUMNS:
            if key in {"method", "episodes"}:
                continue
            metric = _false_feasible_metric_summary(subset, key)
            row_out[key] = metric
            summary_rows.append({
                "method": method,
                "label": row_out["method"],
                "metric": key,
                "mean": "" if metric["mean"] is None else f"{metric['mean']:.6f}",
                "std": "" if metric["std"] is None else f"{metric['std']:.6f}",
                "count": metric["count"],
                "total": metric["total"],
                "seeds": metric["seeds"],
                "episodes": len(subset),
                "protocol": "one_shot_false_feasible",
                "raw_csv": str(_display_path(path)),
            })
        table_rows.append(row_out)

    excluded = sorted({
        row.get("method", "")
        for row in filtered
        if row.get("method", "") not in set(available_methods)
    })
    notes = [
        f"Loaded raw outcome rows from `{_display_path(path)}`.",
        "Protocol: one-shot benchmark-labeled false-feasible passages.",
        "Methods included: Reactive rule baseline and DEGNAV-Rule.",
        "Each listed method uses 3 seeds and 500 false-feasible episodes per seed.",
    ]
    if excluded:
        notes.append(
            "Excluded methods not in the one-shot main row group: "
            + ", ".join(excluded)
            + "."
        )
    return table_rows, summary_rows, notes


def false_feasible_outcome_markdown(rows, notes):
    header = "| " + " | ".join(label for _, label in FALSE_FEASIBLE_OUTCOME_COLUMNS) + " |"
    sep = "| " + " | ".join(":---" if i == 0 else "---:" for i, _ in enumerate(FALSE_FEASIBLE_OUTCOME_COLUMNS)) + " |"
    body = []
    for row in rows:
        cells = []
        for key, _label in FALSE_FEASIBLE_OUTCOME_COLUMNS:
            if key == "method":
                cells.append(str(row[key]))
            elif key == "episodes":
                cells.append(f"{row[key]} ({row.get('seeds', 0)} seeds)")
            else:
                cells.append(_fmt_rate_with_count(row.get(key)))
        body.append("| " + " | ".join(cells) + " |")

    note_lines = [
        "",
        "Notes:",
        "- One-shot false-feasible traversal success does not establish correct rejection. This table separates explicit abstention from collision and non-collision execution failure.",
        "- DEGNAV-Rule reduces hard collision relative to the reactive baseline, but one-shot correct rejection remains weak; recurrence handling is evaluated separately through memory.",
        "- `explicit reject` means the controller selected Reject.",
        "- `correct_reject = explicit reject and passable_label == false`.",
        "- Timeout/stuck remains an execution failure category and is not labeled as safe rejection.",
        "- `wasted_attempt = attempted traversal on a false-feasible passage without correct rejection`.",
        "- `success == false` is never converted into correct rejection.",
        "- Collision and near-collision are logged independently and may overlap.",
    ]
    for note in notes:
        note_lines.append(f"- {note}")
    return "\n".join([
        "# Table: False-Feasible Outcome Decomposition",
        "",
        *([header, sep, *body] if body else ["No false-feasible outcome data found."]),
        *note_lines,
    ])


def false_feasible_outcome_latex(rows):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{One-shot false-feasible traversal success does not establish correct rejection. This table separates explicit abstention from collision and non-collision execution failure.}",
        r"\label{tab:false-feasible-outcomes}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & Episodes & Traversal success & Explicit reject & Correct reject & Collision & Near collision & Timeout/stuck & Wasted attempts \\",
        r"\midrule",
    ]
    for row in rows:
        method = str(row["method"]).replace("_", r"\_")
        lines.append(
            f"{method} & {row['episodes']} & {_fmt_rate_with_count_latex(row.get('success'))} "
            f"& {_fmt_rate_with_count_latex(row.get('reject'))} "
            f"& {_fmt_rate_with_count_latex(row.get('correct_reject'))} "
            f"& {_fmt_rate_with_count_latex(row.get('collision'))} "
            f"& {_fmt_rate_with_count_latex(row.get('near_collision'))} "
            f"& {_fmt_rate_with_count_latex(row.get('timeout_stuck'))} "
            f"& {_fmt_rate_with_count_latex(row.get('wasted_attempt'))} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.25em}",
        r"\begin{minipage}{0.98\linewidth}",
        r"\footnotesize Explicit reject means the controller selected Reject; correct reject is explicit rejection on a benchmark-labeled infeasible passage. Timeout/stuck is not safe rejection. Collision and near-collision are logged independently and may overlap. Recurrence handling is evaluated separately through memory.",
        r"\end{minipage}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def _calibration_extended_rows(path):
    rows = read_rows(path)
    if not rows:
        return [], [f"No calibration prior-sweep CSV found at `{_display_path(path)}`."]
    return rows, [f"Loaded calibration prior sweep from `{_display_path(path)}`."]


def _calibration_percent(row, key):
    raw = row.get(key, "")
    if raw == "":
        return "not run"
    try:
        return _fmt_percent_for_table(float(raw))
    except (TypeError, ValueError):
        return "not run"


def _latex_percent_text(text):
    return str(text).replace("%", r"\%")


def calibration_extended_markdown(rows, notes):
    header = "| " + " | ".join(label for _, label in CALIBRATION_EXTENDED_COLUMNS) + " |"
    sep = "| " + " | ".join(":---" if i == 1 else "---:" for i, _ in enumerate(CALIBRATION_EXTENDED_COLUMNS)) + " |"
    body = []
    for row in rows:
        mean = row.get("final_posterior_mean", "not run")
        mean_std = row.get("final_posterior_mean_std", "not run")
        q95 = row.get("final_posterior_q95", "not run")
        q95_std = row.get("final_posterior_q95_std", "not run")
        cells = [
            f"{float(row.get('prior_width', 0.0)):.2f}",
            row.get("interpretation", ""),
            f"{mean}±{mean_std}",
            f"{q95}±{q95_std}",
            row.get("mean_p_feas", "not run"),
            _calibration_percent(row, "empirical_feasible_rate"),
            _calibration_percent(row, "reject_rate"),
            _calibration_percent(row, "unsafe_attempt_rate"),
            row.get("brier", "not run"),
            row.get("ece", "not run"),
        ]
        body.append("| " + " | ".join(cells) + " |")
    note_lines = [
        "",
        "Notes:",
        "- `outcome_success` in the raw episode CSV is the oracle physical feasibility label under `true_width`, used for p_feas reliability analysis.",
        "- `unsafe_attempt` is an attempted passage with `passage_width < true_width`, exposing under-conservative priors.",
        "- Reliability claims should be made from `calibration_episode_predictions.csv` and `calibration_reliability_bins.csv`, not from the summary row alone.",
        f"- Exact command: `{CALIBRATION_EXTENDED_COMMAND}`.",
    ]
    for note in notes:
        note_lines.append(f"- {note}")
    return "\n".join([
        "# Table: Extended Required-Width Calibration",
        "",
        *([header, sep, *body] if body else ["No calibration data found."]),
        *note_lines,
    ])


def calibration_extended_latex(rows):
    lines = [
        r"\begin{tabular}{rlrrrrrrrr}",
        r"\toprule",
        r"$\hat{W}$ & Type & Final mean & Final q95 & Mean $p_{feas}$ & Emp. feasible & Reject & Unsafe attempt & Brier & ECE \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{float(row.get('prior_width', 0.0)):.2f} & {row.get('interpretation', '')} "
            f"& {row.get('final_posterior_mean', 'not run')} $\\pm$ {row.get('final_posterior_mean_std', 'not run')} "
            f"& {row.get('final_posterior_q95', 'not run')} $\\pm$ {row.get('final_posterior_q95_std', 'not run')} "
            f"& {row.get('mean_p_feas', 'not run')} "
            f"& {_latex_percent_text(_calibration_percent(row, 'empirical_feasible_rate'))} "
            f"& {_latex_percent_text(_calibration_percent(row, 'reject_rate'))} "
            f"& {_latex_percent_text(_calibration_percent(row, 'unsafe_attempt_rate'))} "
            f"& {row.get('brier', 'not run')} & {row.get('ece', 'not run')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_calibration_extended_tables(output_dir, csv_path):
    rows, notes = _calibration_extended_rows(csv_path)
    if not rows:
        print(f"[warn] {notes[0] if notes else 'no calibration rows found'}")
        return False
    table_dir = Path(output_dir) / "tables"
    write_text(
        table_dir / "paper_table_calibration_extended.md",
        calibration_extended_markdown(rows, notes),
    )
    write_text(
        table_dir / "paper_table_calibration_extended.tex",
        calibration_extended_latex(rows),
    )
    print("[write] tables/paper_table_calibration_extended.md")
    print("[write] tables/paper_table_calibration_extended.tex")
    return True


def _stress_float(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _stress_fmt_pct(value):
    return _fmt_percent_for_table(value)


def _stress_fmt_pct_latex(value):
    return _fmt_percent_latex_for_table(value)


def _habitat_stress_rows(path):
    rows = read_rows(path)
    if rows:
        return rows, [
            f"Loaded raw Habitat stress rows from `{_display_path(path)}`.",
            f"Regenerate with: `{HABITAT_STRESS_COMMAND}`.",
        ]
    return [], [
        f"No raw Habitat stress CSV found at `{_display_path(path)}`.",
        f"Regenerate with: `{HABITAT_STRESS_COMMAND}`.",
    ]


def _summarize_habitat_stress(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row.get("stress", ""), row.get("method", "")), []).append(row)
    summary = []
    for (stress, method), vals in sorted(groups.items()):
        n = len(vals)
        if not n:
            continue
        summary.append({
            "stress": stress,
            "method": method,
            "episodes": n,
            "success_rate": _mean(vals, "success"),
            "strict_success_rate": _mean(vals, "strict_success"),
            "collision_rate": _mean(vals, "collision"),
            "near_collision_rate": _mean(vals, "near_collision"),
            "avg_min_clearance": _mean(vals, "min_clearance"),
            "avg_steps": _mean(vals, "steps"),
            "timeout_rate": _mean(vals, "timeout"),
            "success_but_unsafe_rate": sum(
                float(_stress_float(v, "success") > 0.5 and _stress_float(v, "strict_success") < 0.5)
                for v in vals
            ) / n,
        })
    return summary


def _stress_tex(text):
    return str(text).replace("_", r"\_").replace("%", r"\%")


def _write_raw_habitat_stress_copy(output_dir, rows):
    if not rows:
        return
    raw_path = Path(output_dir) / "raw" / "habitat_stress_all.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("[write] raw/habitat_stress_all.csv")


def _habitat_rq1_summary(path):
    rows = read_rows(path)
    if rows:
        return rows, [
            f"Loaded new belief-gated Habitat RQ1 rows from `{_display_path(path)}`.",
            f"Regenerate with: `{HABITAT_RQ1_COMMAND}`.",
        ]
    return [], [
        f"No Habitat RQ1 summary found at `{_display_path(path)}`; the stress table is unchanged.",
        f"Regenerate with: `{HABITAT_RQ1_COMMAND}`.",
    ]


def habitat_stress_nominal_markdown(summary, notes, rq1_summary=None, rq1_notes=None):
    rq1_summary = rq1_summary or []
    rq1_notes = rq1_notes or []
    lines = [
        "# Table: Habitat Stress Validation - Nominal Metrics",
        "",
        "Nominal task metrics under controlled Habitat perturbations. Clearance-aware diagnostics are reported separately in `paper_table_habitat_clearance_diagnostic.md`.",
        "",
        "| Stress | Method | Episodes | Success rate | Collision | Timeout | Avg steps |",
        "|:---|:---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {stress} | {method} | {episodes} | {sr} | {collision} | {timeout} | {steps:.1f} |".format(
                stress=row["stress"],
                method=row["method"],
                episodes=row["episodes"],
                sr=_stress_fmt_pct(row["success_rate"]),
                collision=_stress_fmt_pct(row["collision_rate"]),
                timeout=_stress_fmt_pct(row["timeout_rate"]),
                steps=row["avg_steps"],
            )
        )
    if rq1_summary:
        lines.extend([
            "",
            "## Nominal RQ1 belief-feasibility ablation (new belief-gated run)",
            "",
            "These rows use the same 151 nominal episode IDs, 500-step budget, and Habitat success definition. They are kept in a separate block because the historical stress rows used the legacy FSM controller; the new rows route `DynamicFeasibilityEstimator` into the mode/action path and use metric-depth clearance.",
            "",
            "| Method | Episodes | Success rate | Strict SR (metric depth) | Collision | False Reject | Timeout | Avg steps |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rq1_summary:
            method = row["method"]
            if method in {"point_estimate", "no_uncertainty"}:
                method += "†"
            lines.append(
                "| {method} | {episodes} | {success} | {strict} | {collision} | {reject} | {timeout} | {steps:.1f} |".format(
                    method=method,
                    episodes=int(float(row["episodes"])),
                    success=_stress_fmt_pct(float(row["success_rate"])),
                    strict=_stress_fmt_pct(float(row["strict_success_rate"])),
                    collision=_stress_fmt_pct(float(row["collision_rate"])),
                    reject=_stress_fmt_pct(float(row["false_reject_rate"])),
                    timeout=_stress_fmt_pct(float(row["timeout_rate"])),
                    steps=float(row["avg_steps"]),
                )
            )
        lines.extend([
            "",
            "† `point_estimate` and `no_uncertainty` have identical episode-level behavior hashes and are compatibility aliases, not independent paper methods.",
            "",
            "The four RQ1 variants have identical aggregate Success (99.3%). Full vs point-estimate differs on two aligned mode decisions in one episode, without changing the outcome; full vs no-yaw-prior has zero mode disagreements on these nominal aligned anchors.",
        ])
    lines.extend(["", "Notes:"])
    for note in notes:
        lines.append(f"- {note}")
    for note in rq1_notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def habitat_stress_nominal_latex(summary, rq1_summary=None):
    rq1_summary = rq1_summary or []
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Stress & Method & N & Success & Collision & Timeout & Avg steps \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {:.1f} \\\\".format(
                _stress_tex(row["stress"]),
                _stress_tex(row["method"]),
                row["episodes"],
                _stress_fmt_pct_latex(row["success_rate"]),
                _stress_fmt_pct_latex(row["collision_rate"]),
                _stress_fmt_pct_latex(row["timeout_rate"]),
                row["avg_steps"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if rq1_summary:
        lines.extend([
            "",
            r"\par\medskip",
            r"\noindent\textit{Nominal RQ1 belief-feasibility ablation (new belief-gated run).}",
            "",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r"Method & N & Success & Strict SR & Collision & False Reject & Timeout & Avg steps \\",
            r"\midrule",
        ])
        for row in rq1_summary:
            method = row["method"]
            if method in {"point_estimate", "no_uncertainty"}:
                method += r"$^{\dagger}$"
            lines.append(
                "{} & {} & {} & {} & {} & {} & {} & {:.1f} \\\\".format(
                    _stress_tex(method),
                    int(float(row["episodes"])),
                    _stress_fmt_pct_latex(float(row["success_rate"])),
                    _stress_fmt_pct_latex(float(row["strict_success_rate"])),
                    _stress_fmt_pct_latex(float(row["collision_rate"])),
                    _stress_fmt_pct_latex(float(row["false_reject_rate"])),
                    _stress_fmt_pct_latex(float(row["timeout_rate"])),
                    float(row["avg_steps"]),
                )
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\par\footnotesize $^{\dagger}$Behaviorally equivalent compatibility aliases; do not count as independent paper methods.\normalsize",
        ])
    return "\n".join(lines)


def habitat_clearance_diagnostic_markdown(summary, notes):
    diagnostic_note = (
        "The strict clearance metric is a depth-derived body-margin proxy. "
        "Negative values and high near-collision rates may reflect scanned-scene, "
        "navmesh, or depth-proxy artifacts. We report it as a diagnostic metric, "
        "not as a direct physical contact measurement."
    )
    lines = [
        "# Table: Habitat Clearance Diagnostic Metrics",
        "",
        diagnostic_note,
        "",
        "| Stress | Method | Strict success | Near collision | Avg min clearance | Success-but-unsafe |",
        "|:---|:---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {stress} | {method} | {strict} | {near} | {clearance:.3f} | {sbu} |".format(
                stress=row["stress"],
                method=row["method"],
                strict=_stress_fmt_pct(row["strict_success_rate"]),
                near=_stress_fmt_pct(row["near_collision_rate"]),
                clearance=row["avg_min_clearance"],
                sbu=_stress_fmt_pct(row["success_but_unsafe_rate"]),
            )
        )
    lines.extend(["", "Notes:"])
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def habitat_clearance_diagnostic_latex(summary):
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Stress & Method & Strict success & Near collision & Min clearance & Success-but-unsafe \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            "{} & {} & {} & {} & {:.3f} & {} \\\\".format(
                _stress_tex(row["stress"]),
                _stress_tex(row["method"]),
                _stress_fmt_pct_latex(row["strict_success_rate"]),
                _stress_fmt_pct_latex(row["near_collision_rate"]),
                row["avg_min_clearance"],
                _stress_fmt_pct_latex(row["success_but_unsafe_rate"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _habitat_key_slice(row):
    stress = row["stress"]
    method = row["method"]
    if stress == "nominal":
        return method in {"apf_gap", "fsm_full"}
    if stress in {"yaw60", "extreme_yaw60", "lat020"}:
        return method in {"apf_gap", "fsm_full", "fsm_no_heading_alignment"}
    return False


HABITAT_KEY_SLICE_ORDER = {
    "nominal": 0,
    "yaw60": 1,
    "extreme_yaw60": 2,
    "lat020": 3,
}

HABITAT_KEY_SLICE_METHOD_ORDER = {
    "apf_gap": 0,
    "fsm_full": 1,
    "fsm_no_heading_alignment": 2,
}


def habitat_key_slices_markdown(summary, notes):
    rows = sorted(
        [row for row in summary if _habitat_key_slice(row)],
        key=lambda row: (
            HABITAT_KEY_SLICE_ORDER.get(row["stress"], 99),
            HABITAT_KEY_SLICE_METHOD_ORDER.get(row["method"], 99),
        ),
    )
    lines = [
        "# Table: Habitat Stress Key Slices",
        "",
        "Key slices for manuscript discussion. Clearance columns are diagnostic body-margin proxies, not calibrated physical contact measurements.",
        "",
        "| Stress | Method | Episodes | Success rate | Collision | Timeout | Avg steps | Strict success (diagnostic) | Near collision (diagnostic) | Avg min clearance (diagnostic) |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {stress} | {method} | {episodes} | {sr} | {collision} | {timeout} | {steps:.1f} | {strict} | {near} | {clearance:.3f} |".format(
                stress=row["stress"],
                method=row["method"],
                episodes=row["episodes"],
                sr=_stress_fmt_pct(row["success_rate"]),
                collision=_stress_fmt_pct(row["collision_rate"]),
                timeout=_stress_fmt_pct(row["timeout_rate"]),
                steps=row["avg_steps"],
                strict=_stress_fmt_pct(row["strict_success_rate"]),
                near=_stress_fmt_pct(row["near_collision_rate"]),
                clearance=row["avg_min_clearance"],
            )
        )
    lines.extend(["", "Notes:"])
    for note in notes:
        lines.append(f"- {note}")
    return "\n".join(lines)


def habitat_key_slices_latex(summary):
    rows = sorted(
        [row for row in summary if _habitat_key_slice(row)],
        key=lambda row: (
            HABITAT_KEY_SLICE_ORDER.get(row["stress"], 99),
            HABITAT_KEY_SLICE_METHOD_ORDER.get(row["method"], 99),
        ),
    )
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Stress & Method & N & Success & Collision & Timeout & Steps & Strict diag. & Near diag. & Min clearance diag. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {:.1f} & {} & {} & {:.3f} \\\\".format(
                _stress_tex(row["stress"]),
                _stress_tex(row["method"]),
                row["episodes"],
                _stress_fmt_pct_latex(row["success_rate"]),
                _stress_fmt_pct_latex(row["collision_rate"]),
                _stress_fmt_pct_latex(row["timeout_rate"]),
                row["avg_steps"],
                _stress_fmt_pct_latex(row["strict_success_rate"]),
                _stress_fmt_pct_latex(row["near_collision_rate"]),
                row["avg_min_clearance"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_habitat_stress_split_tables(output_dir, csv_path, rq1_path=None):
    rows, notes = _habitat_stress_rows(csv_path)
    if not rows:
        for note in notes:
            print(f"[warn] {note}")
        return False
    summary = _summarize_habitat_stress(rows)
    rq1_summary, rq1_notes = _habitat_rq1_summary(
        rq1_path or HABITAT_RQ1_SUMMARY
    )
    table_dir = Path(output_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    _write_raw_habitat_stress_copy(output_dir, rows)
    write_text(
        table_dir / "paper_table_habitat_stress_nominal.md",
        habitat_stress_nominal_markdown(summary, notes, rq1_summary, rq1_notes),
    )
    write_text(
        table_dir / "paper_table_habitat_stress_nominal.tex",
        habitat_stress_nominal_latex(summary, rq1_summary),
    )
    write_text(
        table_dir / "paper_table_habitat_clearance_diagnostic.md",
        habitat_clearance_diagnostic_markdown(summary, notes),
    )
    write_text(
        table_dir / "paper_table_habitat_clearance_diagnostic.tex",
        habitat_clearance_diagnostic_latex(summary),
    )
    write_text(
        table_dir / "paper_table_habitat_stress_key_slices.md",
        habitat_key_slices_markdown(summary, notes),
    )
    write_text(
        table_dir / "paper_table_habitat_stress_key_slices.tex",
        habitat_key_slices_latex(summary),
    )
    print("[write] tables/paper_table_habitat_stress_nominal.md")
    print("[write] tables/paper_table_habitat_stress_nominal.tex")
    print("[write] tables/paper_table_habitat_clearance_diagnostic.md")
    print("[write] tables/paper_table_habitat_clearance_diagnostic.tex")
    print("[write] tables/paper_table_habitat_stress_key_slices.md")
    print("[write] tables/paper_table_habitat_stress_key_slices.tex")
    return True


def _as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return float(default)


def _seed_metric(rows, metric):
    if not rows:
        return None
    if metric == "timeout_stuck":
        values = [
            float(_as_float(row, "timeout") > 0.5 or _as_float(row, "stuck") > 0.5)
            for row in rows
        ]
    else:
        values = [_as_float(row, metric) for row in rows]
    return sum(values) / len(values) if values else None


def _mean_std(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) <= 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, var ** 0.5


def _fmt_mean_std(mean, std):
    if mean is None:
        return "not run"
    if std is None:
        return f"{100.0 * mean:.1f}%"
    return f"{100.0 * mean:.1f}±{100.0 * std:.1f}%"


def _fmt_mean_std_latex(mean, std):
    return _fmt_mean_std(mean, std).replace("%", r"\%").replace("±", r"$\pm$")


def _fmt_delta_pp(mean, std):
    if mean is None:
        return "not run"
    if std is None:
        return f"{100.0 * mean:+.1f} pp"
    return f"{100.0 * mean:+.1f}±{100.0 * std:.1f} pp"


def _fmt_delta_pp_latex(mean, std):
    return _fmt_delta_pp(mean, std).replace("±", r"$\pm$")


def _procedural_main_rows(path):
    rows = read_rows(path)
    if not rows:
        notes = [f"No procedural v2 benchmark CSV found at `{_display_path(path)}`."]
        return [], [], notes

    by_method_seed = {}
    for row in rows:
        method = row.get("method", "").strip()
        seed = row.get("seed", "")
        by_method_seed.setdefault((method, seed), []).append(row)

    table_rows = []
    summary_rows = []
    for method, label in PROCEDURAL_MAIN_METHODS:
        method_rows = [row for row in rows if row.get("method", "").strip() == method]
        seeds = sorted({row.get("seed", "") for row in method_rows})
        if not method_rows:
            table_rows.append({
                "method": label,
                "method_key": method,
                "episodes": 0,
                "seeds": 0,
            })
            continue

        table_row = {
            "method": label,
            "method_key": method,
            "episodes": len(method_rows),
            "seeds": len(seeds),
        }
        metric_specs = [
            ("overall", "success", None),
            *[(out_key, "success", corridor) for out_key, corridor in PROCEDURAL_MAIN_CORRIDORS],
            ("false_feasible_success", "success", "false_feasible"),
        ]
        for out_key, metric, corridor in metric_specs:
            seed_values = []
            episode_count = 0
            for seed in seeds:
                seed_rows = by_method_seed.get((method, seed), [])
                if corridor is not None:
                    seed_rows = [
                        row for row in seed_rows
                        if str(row.get("corridor_type", "")).lower() == corridor
                    ]
                episode_count += len(seed_rows)
                seed_values.append(_seed_metric(seed_rows, metric))
            mean, std = _mean_std(seed_values)
            table_row[out_key] = {"mean": mean, "std": std}
            summary_rows.append({
                "method": method,
                "label": label,
                "metric": out_key,
                "corridor_type": corridor or "all",
                "mean": "" if mean is None else f"{mean:.6f}",
                "std": "" if std is None else f"{std:.6f}",
                "seeds": len([v for v in seed_values if v is not None]),
                "episodes": episode_count,
                "benchmark_version": "procedural_v2_harder",
                "raw_csv": str(_display_path(path)),
            })
        table_rows.append(table_row)

    seeds_seen = sorted({
        row.get("seed", "")
        for row in rows
        if row.get("method", "").strip() in {m for m, _ in PROCEDURAL_MAIN_METHODS}
    })
    notes = [
        f"Loaded raw episode rows from `{_display_path(path)}`.",
        "Benchmark version: procedural v2 / HarderNarrowPassageEnv.",
        f"Seeds: {', '.join(seeds_seen) if seeds_seen else 'not available'}.",
        "Episode budget: 500 episodes per seed per method for the overall metric.",
        f"Regeneration command: `{PROCEDURAL_MAIN_COMMAND}`.",
    ]
    return table_rows, summary_rows, notes


def _best_feasible_metric_keys(rows):
    best = {}
    feasible_keys = [
        key for key, _label in PROCEDURAL_MAIN_COLUMNS[1:]
        if key != "false_feasible_success"
    ]
    for key in feasible_keys:
        values = [
            row.get(key, {}).get("mean")
            for row in rows
            if isinstance(row.get(key), dict) and row.get(key, {}).get("mean") is not None
        ]
        if values:
            best[key] = max(values)
    return best


def _bold_if_best(text, row, key, best):
    metric = row.get(key, {})
    mean = metric.get("mean") if isinstance(metric, dict) else None
    if mean is not None and key in best and abs(mean - best[key]) < 1e-12:
        return f"**{text}**"
    return text


def _bold_if_best_latex(text, row, key, best):
    metric = row.get(key, {})
    mean = metric.get("mean") if isinstance(metric, dict) else None
    if mean is not None and key in best and abs(mean - best[key]) < 1e-12:
        return rf"\textbf{{{text}}}"
    return text


def procedural_main_markdown(rows, notes):
    best = _best_feasible_metric_keys(rows)
    header = "| " + " | ".join(label for _, label in PROCEDURAL_MAIN_COLUMNS) + " |"
    sep = "| " + " | ".join(
        ":---" if i == 0 else "---:" for i, _ in enumerate(PROCEDURAL_MAIN_COLUMNS)
    ) + " |"
    body = []
    for row in rows:
        cells = []
        for key, _label in PROCEDURAL_MAIN_COLUMNS:
            if key == "method":
                cells.append(str(row[key]))
                continue
            metric = row.get(key, {})
            text = _fmt_mean_std(metric.get("mean"), metric.get("std"))
            cells.append(_bold_if_best(text, row, key, best))
        body.append("| " + " | ".join(cells) + " |")

    note_lines = [
        "",
        "Notes:",
        "- Values are mean±std across seeds; rates are computed per seed first, then averaged.",
        "- The main table contains only the primary reactive baseline and the main paper method.",
        "- Local-memory and cross-episode memory rows are intentionally omitted here because their single-encounter procedural v2 results match DEGNAV-Rule; memory is evaluated separately under recurrence and transfer.",
        "- False-feasible traversal success alone does not distinguish correct rejection, collision, or timeout/stuck, so it is not bolded as a success criterion.",
        "- Use the false-feasible outcome decomposition table for explicit rejection, wasted attempts, collision, and timeout/stuck analysis.",
    ]
    for note in notes:
        note_lines.append(f"- {note}")
    return "\n".join([
        "# Table: Procedural v2 Main Benchmark",
        "",
        *([header, sep, *body] if body else ["No procedural v2 benchmark data found."]),
        *note_lines,
    ])


def procedural_main_latex(rows, notes):
    best = _best_feasible_metric_keys(rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Procedural v2 main benchmark. Values are mean$\pm$std success rates across seeds. False-feasible traversal success alone does not distinguish correct rejection, collision, or timeout/stuck.}",
        r"\label{tab:procedural-v2-main}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & Overall & Straight & L-shaped & S-shaped & Narrow exit & Narrow entry & Asymmetric & False-feas. traversal \\",
        r"\midrule",
    ]
    for row in rows:
        method = str(row["method"]).replace("_", r"\_")
        cells = [method]
        for key, _label in PROCEDURAL_MAIN_COLUMNS[1:]:
            metric = row.get(key, {})
            text = _fmt_mean_std_latex(metric.get("mean"), metric.get("std"))
            cells.append(_bold_if_best_latex(text, row, key, best))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.25em}",
        r"\begin{minipage}{0.98\linewidth}",
        r"\footnotesize Memory variants are omitted from this single-encounter table because they match DEGNAV-Rule here; recurrence and transfer are evaluated separately. False-feasible traversal success is not interpreted as correct rejection.",
        r"\end{minipage}",
        r"\end{table*}",
    ])
    if notes:
        lines.append("% Provenance:")
        for note in notes:
            lines.append("% " + note.replace("%", r"\%"))
    return "\n".join(lines)


def write_procedural_main_tables(output_dir, csv_path):
    rows, summary_rows, notes = _procedural_main_rows(csv_path)
    if not rows:
        print(f"[warn] {notes[0] if notes else 'no procedural v2 benchmark rows found'}")
        return False

    table_dir = Path(output_dir) / "tables"
    md = procedural_main_markdown(rows, notes)
    tex = procedural_main_latex(rows, notes)
    write_text(table_dir / "paper_table_procedural_v2_main.md", md)
    write_text(table_dir / "paper_table_procedural_v2_main.tex", tex)
    write_text(Path(output_dir) / "paper_table_procedural_v2_main.md", md)
    write_text(Path(output_dir) / "paper_table_procedural_v2_main.tex", tex)

    if summary_rows:
        summary_path = table_dir / "procedural_v2_main_summary.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "method",
                    "label",
                    "metric",
                    "corridor_type",
                    "mean",
                    "std",
                    "seeds",
                    "episodes",
                    "benchmark_version",
                    "raw_csv",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print("[write] tables/procedural_v2_main_summary.csv")
    print("[write] tables/paper_table_procedural_v2_main.md")
    print("[write] tables/paper_table_procedural_v2_main.tex")
    print("[write] paper_table_procedural_v2_main.md")
    print("[write] paper_table_procedural_v2_main.tex")
    return True


def _variant_order_for_rows(rows):
    present = {
        (row.get("variant", "").strip() or row.get("method", "").strip())
        for row in rows
    }
    order = list(PROCEDURAL_CORE_ABLATION_ORDER)
    order.extend(
        variant for variant in PROCEDURAL_CORE_ABLATION_OPTIONAL_ORDER
        if variant in present
    )
    return order


def _variant_label(variant):
    if variant in PROCEDURAL_CORE_ABLATION_LABELS:
        return PROCEDURAL_CORE_ABLATION_LABELS[variant]
    if variant in PROCEDURAL_CORE_ABLATION_OPTIONAL_LABELS:
        return PROCEDURAL_CORE_ABLATION_OPTIONAL_LABELS[variant]
    return variant


def _procedural_core_ablation_rows(path):
    rows = read_rows(path)
    if not rows:
        notes = [f"No procedural core ablation CSV found at `{_display_path(path)}`."]
        return [], [], notes

    by_variant_seed = {}
    for row in rows:
        variant = row.get("variant", "").strip() or row.get("method", "").strip()
        seed = row.get("seed", "")
        by_variant_seed.setdefault((variant, seed), []).append(row)

    table_rows = []
    summary_rows = []
    full_seed_overall = {
        seed: _seed_metric(by_variant_seed.get(("full", seed), []), "success")
        for seed in sorted({
            seed for variant, seed in by_variant_seed
            if variant == "full"
        })
    }
    for variant in _variant_order_for_rows(rows):
        label = _variant_label(variant)
        variant_rows = [row for row in rows if (row.get("variant", "").strip() or row.get("method", "").strip()) == variant]
        seeds = sorted({row.get("seed", "") for row in variant_rows})
        table_row = {
            "variant": label,
            "variant_key": variant,
            "episodes": len(variant_rows),
            "seeds": len(seeds),
        }
        metric_specs = [
            ("overall", "success", None),
            ("straight", "success", "straight"),
            ("l_shaped", "success", "l_shaped"),
            ("s_shaped", "success", "s_shaped"),
            ("narrow_entry", "success", "narrow_entry"),
            ("asymmetric", "success", "asymmetric"),
            ("collision", "collision", None),
            ("timeout_stuck", "timeout_stuck", None),
        ]
        for out_key, metric, corridor in metric_specs:
            seed_values = []
            episode_count = 0
            for seed in seeds:
                seed_rows = by_variant_seed.get((variant, seed), [])
                if corridor is not None:
                    seed_rows = [
                        row for row in seed_rows
                        if str(row.get("corridor_type", "")).lower() == corridor
                    ]
                episode_count += len(seed_rows)
                seed_values.append(_seed_metric(seed_rows, metric))
            mean, std = _mean_std(seed_values)
            table_row[out_key] = {"mean": mean, "std": std}
            summary_rows.append({
                "variant": variant,
                "label": label,
                "metric": out_key,
                "mean": "" if mean is None else f"{mean:.6f}",
                "std": "" if std is None else f"{std:.6f}",
                "seeds": len([v for v in seed_values if v is not None]),
                "episodes": episode_count,
                "corridor_type": corridor or "all",
                "benchmark_version": "procedural_v2_core_ablation",
                "raw_csv": str(_display_path(path)),
            })
        delta_values = []
        for seed in seeds:
            variant_mean = _seed_metric(by_variant_seed.get((variant, seed), []), "success")
            full_mean = full_seed_overall.get(seed)
            if variant_mean is not None and full_mean is not None:
                delta_values.append(variant_mean - full_mean)
        delta_mean, delta_std = _mean_std(delta_values)
        table_row["delta_overall"] = {"mean": delta_mean, "std": delta_std}
        summary_rows.append({
            "variant": variant,
            "label": label,
            "metric": "delta_overall",
            "mean": "" if delta_mean is None else f"{delta_mean:.6f}",
            "std": "" if delta_std is None else f"{delta_std:.6f}",
            "seeds": len(delta_values),
            "episodes": len(variant_rows),
            "corridor_type": "all",
            "benchmark_version": "procedural_v2_core_ablation",
            "raw_csv": str(_display_path(path)),
        })
        table_rows.append(table_row)

    counts_by_variant = {
        variant: len([
            row for row in rows
            if (row.get("variant", "").strip() or row.get("method", "").strip()) == variant
        ])
        for variant in _variant_order_for_rows(rows)
    }
    notes = [
        f"Loaded raw episode rows from `{_display_path(path)}`.",
        "Benchmark version: procedural v2 / HarderNarrowPassageEnv.",
        "Common configuration: 500 episodes per seed, seeds 0, 1, 2 for every listed variant.",
        "Rows per variant: " + ", ".join(
            f"{_variant_label(variant)}={count}"
            for variant, count in counts_by_variant.items()
        ) + ".",
        f"Exact evaluation command: `{PROCEDURAL_CORE_COMMAND}`.",
    ]
    return table_rows, summary_rows, notes


def procedural_core_ablation_markdown(rows, notes):
    header = "| " + " | ".join(label for _, label in PROCEDURAL_CORE_ABLATION_COLUMNS) + " |"
    sep = "| " + " | ".join(":---" if i == 0 else "---:" for i, _ in enumerate(PROCEDURAL_CORE_ABLATION_COLUMNS)) + " |"
    body = []
    for row in rows:
        cells = []
        for key, _label in PROCEDURAL_CORE_ABLATION_COLUMNS:
            if key == "variant":
                cells.append(str(row[key]))
            elif key == "delta_overall":
                metric = row.get(key, {})
                cells.append(_fmt_delta_pp(metric.get("mean"), metric.get("std")))
            else:
                metric = row.get(key, {})
                cells.append(_fmt_mean_std(metric.get("mean"), metric.get("std")))
        body.append("| " + " | ".join(cells) + " |")

    note_lines = [
        "",
        "Notes:",
        "- Values are mean±std across seeds. Rates are computed per seed first, then averaged.",
        "- Δ Overall is computed as a paired per-seed difference from DEGNAV full, then averaged.",
        "- The current benchmark identifies alignment as the dominant measured component. Deterministic-margin and no-yaw-prior variants remain close to full DEGNAV, so the benchmark does not fully isolate the probabilistic belief and yaw-prior contributions.",
        "- `deterministic margin` replaces probabilistic feasibility gating with `d_hat - w_req_cons > tau_margin` while preserving the yaw-aware width prior and alignment controller.",
        "- `w/o yaw prior` keeps probabilistic uncertainty and alignment but uses a fixed frontal required-width prior.",
        "- Recovery should be interpreted conservatively here: w/o recovery matches full DEGNAV on this unperturbed benchmark, so recovery benefit should be assessed in stress or stuck-specific settings.",
        "- No statistical significance is claimed without a separate statistical test.",
    ]
    for note in notes:
        note_lines.append(f"- {note}")
    return "\n".join([
        "# Table: Procedural v2 Core Ablations",
        "",
        *([header, sep, *body] if body else ["No procedural core ablation data found."]),
        *note_lines,
    ])


def procedural_core_ablation_latex(rows, notes=None):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Procedural v2 core ablations. The current benchmark identifies alignment as the dominant measured component. Deterministic-margin and no-yaw-prior variants remain close to full DEGNAV, so this benchmark does not fully isolate the probabilistic belief and yaw-prior contributions.}",
        r"\label{tab:procedural-v2-core-ablation}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Variant & Overall & $\Delta$ Overall & Straight & L-shaped & S-shaped & Narrow entry & Asymmetric & Collision & Timeout/stuck \\",
        r"\midrule",
    ]
    for row in rows:
        variant = str(row["variant"]).replace("_", r"\_")
        cells = [variant]
        for key, _label in PROCEDURAL_CORE_ABLATION_COLUMNS[1:]:
            metric = row.get(key, {})
            if key == "delta_overall":
                cells.append(_fmt_delta_pp_latex(metric.get("mean"), metric.get("std")))
            else:
                cells.append(_fmt_mean_std_latex(metric.get("mean"), metric.get("std")))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.25em}",
        r"\begin{minipage}{0.98\linewidth}",
        r"\footnotesize Values are mean$\pm$std across seeds. $\Delta$ Overall is paired by seed against DEGNAV full. No statistical significance is claimed without a separate test.",
        r"\end{minipage}",
        r"\end{table*}",
    ])
    if notes:
        lines.append("% Provenance:")
        for note in notes:
            lines.append("% " + note.replace("%", r"\%"))
    return "\n".join(lines)


def write_procedural_core_ablation_tables(output_dir, csv_path):
    rows, summary_rows, notes = _procedural_core_ablation_rows(csv_path)
    if not rows:
        print(f"[warn] {notes[0] if notes else 'no procedural core ablation rows found'}")
        return False
    table_dir = Path(output_dir) / "tables"
    write_text(
        table_dir / "paper_table_procedural_v2_ablation_core.md",
        procedural_core_ablation_markdown(rows, notes),
    )
    write_text(
        table_dir / "paper_table_procedural_v2_ablation_core.tex",
        procedural_core_ablation_latex(rows, notes),
    )
    if summary_rows:
        summary_path = table_dir / "procedural_v2_ablation_core_summary.csv"
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "variant",
                    "label",
                    "metric",
                    "mean",
                    "std",
                    "seeds",
                    "episodes",
                    "corridor_type",
                    "benchmark_version",
                    "raw_csv",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        with (table_dir / "procedural_ablation_core_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "variant",
                    "label",
                    "metric",
                    "mean",
                    "std",
                    "seeds",
                    "episodes",
                    "corridor_type",
                    "benchmark_version",
                    "raw_csv",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print("[write] tables/procedural_v2_ablation_core_summary.csv")
        print("[write] tables/procedural_ablation_core_summary.csv")
    # Keep the previous filenames as compatibility aliases for older README links.
    write_text(
        table_dir / "paper_table_procedural_core_ablation.md",
        procedural_core_ablation_markdown(rows, notes),
    )
    write_text(
        table_dir / "paper_table_procedural_core_ablation.tex",
        procedural_core_ablation_latex(rows, notes),
    )
    print("[write] tables/paper_table_procedural_v2_ablation_core.md")
    print("[write] tables/paper_table_procedural_v2_ablation_core.tex")
    return True


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
    parser.add_argument(
        "--false-feasible-outcomes",
        type=Path,
        default=None,
        help="Optional raw false-feasible outcome decomposition CSV.",
    )
    parser.add_argument(
        "--only-false-feasible-outcomes",
        action="store_true",
        help="Only generate the false-feasible outcome decomposition table.",
    )
    parser.add_argument(
        "--procedural-core-ablation",
        type=Path,
        default=None,
        help="Optional raw procedural core ablation CSV.",
    )
    parser.add_argument(
        "--procedural-main",
        type=Path,
        default=None,
        help="Optional raw procedural v2 main benchmark episode CSV.",
    )
    parser.add_argument(
        "--calibration-prior-sweep",
        type=Path,
        default=None,
        help="Optional calibration prior-sweep summary CSV.",
    )
    parser.add_argument(
        "--habitat-stress-input",
        type=Path,
        default=None,
        help="Optional raw Habitat stress episode CSV.",
    )
    parser.add_argument(
        "--habitat-rq1-input",
        type=Path,
        default=None,
        help="Optional summary.csv from the new belief-gated Habitat RQ1 run.",
    )
    parser.add_argument(
        "--only-procedural-core-ablation",
        action="store_true",
        help="Only generate the procedural core ablation table.",
    )
    parser.add_argument(
        "--only-procedural-main",
        action="store_true",
        help="Only generate the compact procedural v2 main benchmark table.",
    )
    parser.add_argument(
        "--only-calibration-extended",
        action="store_true",
        help="Only generate the extended calibration table.",
    )
    parser.add_argument(
        "--only-habitat-stress-split",
        action="store_true",
        help="Only generate split Habitat stress/clearance diagnostic tables.",
    )
    args = parser.parse_args()

    procedural_core_path = args.procedural_core_ablation or (
        args.output_dir / "raw" / "procedural_ablation_core.csv"
    )
    procedural_main_path = args.procedural_main or (
        args.output_dir / "harder_benchmark_episodes.csv"
    )
    calibration_path = args.calibration_prior_sweep or (
        args.output_dir / "raw" / "calibration_prior_sweep.csv"
    )
    habitat_stress_path = args.habitat_stress_input or (
        args.output_dir / "habitat_stress_validation.csv"
    )
    ff_path = args.false_feasible_outcomes or (
        args.output_dir / "raw" / "false_feasible_outcomes.csv"
    )
    if args.only_false_feasible_outcomes:
        ff_rows, ff_summary_rows, ff_notes = _false_feasible_outcome_rows(ff_path)
        if ff_rows:
            table_dir = args.output_dir / "tables"
            write_text(
                table_dir / "paper_table_false_feasible_outcomes.md",
                false_feasible_outcome_markdown(ff_rows, ff_notes),
            )
            write_text(
                table_dir / "paper_table_false_feasible_outcomes.tex",
                false_feasible_outcome_latex(ff_rows),
            )
            if ff_summary_rows:
                summary_path = table_dir / "false_feasible_outcomes_summary.csv"
                with summary_path.open("w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "method",
                            "label",
                            "metric",
                            "mean",
                            "std",
                            "count",
                            "total",
                            "seeds",
                            "episodes",
                            "protocol",
                            "raw_csv",
                        ],
                        lineterminator="\n",
                    )
                    writer.writeheader()
                    writer.writerows(ff_summary_rows)
                print("[write] tables/false_feasible_outcomes_summary.csv")
            print("[write] tables/paper_table_false_feasible_outcomes.md")
            print("[write] tables/paper_table_false_feasible_outcomes.tex")
        else:
            for note in ff_notes:
                print(f"[warn] {note}")
        return
    if args.only_procedural_main:
        write_procedural_main_tables(args.output_dir, procedural_main_path)
        return
    if args.only_procedural_core_ablation:
        write_procedural_core_ablation_tables(args.output_dir, procedural_core_path)
        return
    if args.only_calibration_extended:
        write_calibration_extended_tables(args.output_dir, calibration_path)
        return
    if args.only_habitat_stress_split:
        write_habitat_stress_split_tables(
            args.output_dir,
            habitat_stress_path,
            args.habitat_rq1_input,
        )
        return

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

    ff_rows, ff_summary_rows, ff_notes = _false_feasible_outcome_rows(ff_path)
    if ff_rows:
        table_dir = args.output_dir / "tables"
        write_text(
            table_dir / "paper_table_false_feasible_outcomes.md",
            false_feasible_outcome_markdown(ff_rows, ff_notes),
        )
        write_text(
            table_dir / "paper_table_false_feasible_outcomes.tex",
            false_feasible_outcome_latex(ff_rows),
        )
        if ff_summary_rows:
            summary_path = table_dir / "false_feasible_outcomes_summary.csv"
            with summary_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "method",
                        "label",
                        "metric",
                        "mean",
                        "std",
                        "count",
                        "total",
                        "seeds",
                        "episodes",
                        "protocol",
                        "raw_csv",
                    ],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(ff_summary_rows)
            print("[write] tables/false_feasible_outcomes_summary.csv")
        print("[write] tables/paper_table_false_feasible_outcomes.md")
        print("[write] tables/paper_table_false_feasible_outcomes.tex")

    if procedural_main_path.exists():
        write_procedural_main_tables(args.output_dir, procedural_main_path)

    if procedural_core_path.exists():
        write_procedural_core_ablation_tables(args.output_dir, procedural_core_path)

    if calibration_path.exists():
        write_calibration_extended_tables(args.output_dir, calibration_path)

    if habitat_stress_path.exists():
        write_habitat_stress_split_tables(
            args.output_dir,
            habitat_stress_path,
            args.habitat_rq1_input,
        )


if __name__ == "__main__":
    main()
