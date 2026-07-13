#!/usr/bin/env python3
"""Generate harder benchmark Markdown/LaTeX tables from episode CSV."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "narrow_passage_rl"

CORRIDORS = [
    ("straight", "Straight"),
    ("l_shaped", "L-shaped"),
    ("s_shaped", "S-shaped"),
    ("narrow_exit", "Narrow exit"),
    ("narrow_entry", "Narrow entry"),
    ("asymmetric", "Asymmetric"),
    ("false_feasible", "False-feas."),
]

MAIN_METHODS = [
    ("rule_baseline", "Rule baseline"),
    ("geometry_fsm", "**Geometry-FSM (ours)**"),
    ("fsm_local_memory", "FSM + local memory"),
    ("fsm_cross_memory", "FSM + cross memory"),
]

ABLATION_METHODS = [
    ("geometry_fsm", "**Geometry-FSM (full)**"),
    ("fsm_no_recovery", "FSM w/o recovery"),
    ("fsm_no_alignment", "FSM w/o alignment"),
]


def _float(value: object, default: float = float("nan")) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.mean(vals) if vals else float("nan")


def _std(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _seed_values(rows: list[dict[str, str]], method: str, corridor: str | None = None) -> list[float]:
    by_seed: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("method") != method:
            continue
        if corridor is not None and row.get("corridor_type") != corridor:
            continue
        by_seed[str(row.get("seed", ""))].append(_float(row.get("success")))
    return [_mean(vals) for _seed, vals in sorted(by_seed.items())]


def _fmt_pct(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{100.0 * value:.1f}"


def _fmt_overall(seed_values: list[float], bold: bool = False) -> str:
    mean = _mean(seed_values)
    std = _std(seed_values)
    text = f"{_fmt_pct(mean)} ({_fmt_pct(std)})"
    return f"**{text}**" if bold else text


def _markdown_table(rows: list[dict[str, str]], methods: list[tuple[str, str]], title: str, subtitle: str) -> str:
    lines = [
        f"# {title}",
        f"# {subtitle}",
        "# Metrics: success rate mean (std across seeds)",
        "",
        "| Method | Overall | " + " | ".join(label for _key, label in CORRIDORS) + " |",
        "|:---|:---:|" + "|".join(":---:" for _ in CORRIDORS) + "|",
    ]
    for method, label in methods:
        seed_values = _seed_values(rows, method)
        bold = label.startswith("**")
        cells = [label, _fmt_overall(seed_values, bold=bold)]
        for corridor, _corridor_label in CORRIDORS:
            values = _seed_values(rows, method, corridor)
            text = _fmt_pct(_mean(values))
            cells.append(f"**{text}**" if bold else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _tex_escape(text: str) -> str:
    return text.replace("**", "").replace("_", r"\_").replace("%", r"\%")


def _latex_table(rows: list[dict[str, str]], methods: list[tuple[str, str]]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & Overall & Straight & L-shaped & S-shaped & Narrow exit & Narrow entry & Asymmetric & False-feas. \\",
        r"\midrule",
    ]
    for method, label in methods:
        seed_values = _seed_values(rows, method)
        cells = [_tex_escape(label), _fmt_overall(seed_values, bold=False)]
        for corridor, _corridor_label in CORRIDORS:
            cells.append(_fmt_pct(_mean(_seed_values(rows, method, corridor))))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=RESULTS / "harder_benchmark_episodes.csv")
    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=None,
        help=(
            "Optional ablation episode CSV.  If omitted, the script does not "
            "touch paper_table_harder_ablation.* because the main benchmark CSV "
            "uses no entry jitter while the ablation protocol may differ."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main_md = _markdown_table(
        rows,
        MAIN_METHODS,
        "Table: Harder Synthetic Benchmark v2 - Main Comparison",
        "500 episodes x 3 seeds (seeds 42-44), no entry jitter",
    )

    (args.output_dir / "paper_table_procedural_v2_main.md").write_text(main_md)
    (args.output_dir / "paper_table_harder_main.tex").write_text(_latex_table(rows, MAIN_METHODS))

    print(f"[write] {args.output_dir / 'paper_table_procedural_v2_main.md'}")
    print(f"[write] {args.output_dir / 'paper_table_harder_main.tex'}")

    if args.ablation_csv is not None:
        ablation_rows = read_rows(args.ablation_csv)
        ablation_md = _markdown_table(
            ablation_rows,
            ABLATION_METHODS,
            "Table: Harder Synthetic Benchmark v2 - FSM Ablation",
            "Ablation protocol from the supplied ablation CSV",
        )
        (args.output_dir / "paper_table_harder_ablation.md").write_text(ablation_md)
        (args.output_dir / "paper_table_harder_ablation.tex").write_text(
            _latex_table(ablation_rows, ABLATION_METHODS)
        )
        print(f"[write] {args.output_dir / 'paper_table_harder_ablation.md'}")
        print(f"[write] {args.output_dir / 'paper_table_harder_ablation.tex'}")


if __name__ == "__main__":
    main()
