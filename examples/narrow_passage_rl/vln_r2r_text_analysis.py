#!/usr/bin/env python3
"""Analyze R2R/VLN-CE instruction text without MP3D scene assets.

This script is intentionally dataset-only: it reads R2R annotation JSON files
and optional saved NaVILA diagnostic predictions, then produces corpus figures
and tables.  It does not run Habitat, does not compute SR/SPL/NE/nDTW, and does
not require MP3D `.glb` or `.navmesh` files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import matplotlib.pyplot as plt


DEFAULT_STORE = Path(
    "/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln"
)
DEFAULT_DATASET_ROOT = (
    DEFAULT_STORE / "data/datasets/R2R_VLNCE_v1-3_preprocessed"
)
DEFAULT_PREDICTIONS = Path(
    "examples/narrow_passage_rl/results/vln_batch_diagnostic/predictions.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "examples/narrow_passage_rl/results/vln_r2r_text_analysis"
)


TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")

CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "forward_motion": (
        "walk",
        "go",
        "head",
        "continue",
        "proceed",
        "move",
        "straight",
        "forward",
        "pass",
        "enter",
        "exit",
    ),
    "left_turn": ("left",),
    "right_turn": ("right",),
    "stop_goal": (
        "stop",
        "end",
        "finish",
        "near",
        "beside",
        "at",
        "by",
        "next",
    ),
    "rooms_places": (
        "room",
        "bedroom",
        "bathroom",
        "kitchen",
        "hall",
        "hallway",
        "stairs",
        "staircase",
        "living",
        "dining",
        "office",
        "closet",
        "door",
        "doorway",
    ),
    "objects_landmarks": (
        "table",
        "chair",
        "couch",
        "sofa",
        "bed",
        "rug",
        "sink",
        "cabinet",
        "dresser",
        "plant",
        "lamp",
        "counter",
        "tv",
        "fireplace",
        "picture",
    ),
    "spatial_relation": (
        "around",
        "through",
        "toward",
        "towards",
        "across",
        "between",
        "past",
        "behind",
        "front",
        "opposite",
        "corner",
        "opening",
    ),
}


@dataclass(frozen=True)
class EpisodeText:
    split: str
    episode_id: str
    scene_id: str
    scene_name: str
    trajectory_id: str
    instruction: str
    tokens: list[str]
    word_count: int
    char_count: int
    categories: dict[str, bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate R2R instruction-corpus diagnostics without MP3D."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--splits",
        default="train,val_seen,val_unseen",
        help="Comma-separated R2R_VLNCE_v1-3_preprocessed splits.",
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def scene_name(scene_id: str) -> str:
    parts = str(scene_id).split("/")
    if len(parts) >= 2 and parts[0] == "mp3d":
        return parts[1]
    if scene_id.endswith(".glb"):
        return Path(scene_id).stem
    return str(scene_id)


def instruction_text(episode: dict) -> str:
    instruction = episode.get("instruction", "")
    if isinstance(instruction, dict):
        text = instruction.get("instruction_text", "")
    else:
        text = instruction or episode.get("instruction_text", "")
    return str(text).strip()


def load_split(dataset_root: Path, split: str) -> list[EpisodeText]:
    path = dataset_root / split / f"{split}.json.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes", [])
    rows: list[EpisodeText] = []
    for index, episode in enumerate(episodes):
        text = instruction_text(episode)
        tokens = tokenize(text)
        token_set = set(tokens)
        categories = {
            category: any(word in token_set for word in words)
            for category, words in CATEGORY_PATTERNS.items()
        }
        sid = str(episode.get("scene_id", ""))
        rows.append(
            EpisodeText(
                split=split,
                episode_id=str(episode.get("episode_id", index)),
                scene_id=sid,
                scene_name=scene_name(sid),
                trajectory_id=str(episode.get("trajectory_id", "")),
                instruction=text,
                tokens=tokens,
                word_count=len(tokens),
                char_count=len(text),
                categories=categories,
            )
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def split_summary(rows: list[EpisodeText]) -> list[dict]:
    by_split: dict[str, list[EpisodeText]] = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)
    output: list[dict] = []
    for split in sorted(by_split):
        items = by_split[split]
        lengths = [item.word_count for item in items]
        scenes = {item.scene_name for item in items}
        summary = {
            "split": split,
            "episodes": len(items),
            "unique_scenes": len(scenes),
            "mean_words": mean(lengths) if lengths else math.nan,
            "median_words": median(lengths) if lengths else math.nan,
            "min_words": min(lengths) if lengths else math.nan,
            "max_words": max(lengths) if lengths else math.nan,
        }
        for category in CATEGORY_PATTERNS:
            summary[f"{category}_rate"] = (
                sum(item.categories[category] for item in items) / len(items)
                if items
                else math.nan
            )
        output.append(summary)
    return output


def scene_counts(rows: list[EpisodeText]) -> list[dict]:
    counts: Counter[tuple[str, str]] = Counter(
        (row.split, row.scene_name) for row in rows
    )
    return [
        {"split": split, "scene_name": scene, "episodes": count}
        for (split, scene), count in sorted(counts.items())
    ]


def token_counts(rows: list[EpisodeText], top_k: int = 80) -> list[dict]:
    counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts.update(row.tokens)
        split_counts[row.split].update(row.tokens)
    result: list[dict] = []
    for token, count in counts.most_common(top_k):
        item = {"token": token, "count": count}
        for split in sorted(split_counts):
            item[f"{split}_count"] = split_counts[split][token]
        result.append(item)
    return result


def category_rows(rows: list[EpisodeText]) -> list[dict]:
    result: list[dict] = []
    by_split: dict[str, list[EpisodeText]] = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)
    for split in sorted(by_split):
        items = by_split[split]
        for category in CATEGORY_PATTERNS:
            count = sum(item.categories[category] for item in items)
            result.append(
                {
                    "split": split,
                    "category": category,
                    "episodes": len(items),
                    "count": count,
                    "rate": count / len(items) if items else math.nan,
                }
            )
    return result


def load_predictions(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def prediction_text_rows(predictions: list[dict]) -> list[dict]:
    rows = [
        row
        for row in predictions
        if row.get("suite") == "r2r_language_probe"
        and row.get("subgroup") == "original_instruction"
    ]
    output: list[dict] = []
    for row in rows:
        tokens = tokenize(row.get("instruction", ""))
        token_set = set(tokens)
        item = {
            "dataset_episode_id": row.get("dataset_episode_id", ""),
            "scene_id": row.get("scene_id", ""),
            "parsed_action": row.get("parsed_action", ""),
            "word_count": len(tokens),
        }
        for category, words in CATEGORY_PATTERNS.items():
            item[category] = any(word in token_set for word in words)
        output.append(item)
    return output


def action_by_category(rows: list[dict]) -> list[dict]:
    result: list[dict] = []
    if not rows:
        return result
    actions = ("move_forward", "turn_left", "turn_right", "stop", "invalid")
    for category in CATEGORY_PATTERNS:
        subset = [row for row in rows if row[category]]
        for action in actions:
            if action == "invalid":
                count = sum(row["parsed_action"] not in actions[:-1] for row in subset)
            else:
                count = sum(row["parsed_action"] == action for row in subset)
            result.append(
                {
                    "category": category,
                    "action": action,
                    "episodes": len(subset),
                    "count": count,
                    "rate": count / len(subset) if subset else math.nan,
                }
            )
    return result


def save_figures(
    output_dir: Path,
    summaries: list[dict],
    categories: list[dict],
    top_tokens: list[dict],
    action_category: list[dict],
) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    splits = [row["split"] for row in summaries]
    means = [row["mean_words"] for row in summaries]
    medians = [row["median_words"] for row in summaries]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = range(len(splits))
    ax.bar([i - 0.18 for i in x], means, width=0.36, label="mean")
    ax.bar([i + 0.18 for i in x], medians, width=0.36, label="median")
    ax.set_xticks(list(x), splits)
    ax.set_ylabel("Instruction length (words)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "r2r_instruction_length.png", dpi=240)
    fig.savefig(fig_dir / "r2r_instruction_length.pdf")
    plt.close(fig)

    val_unseen = [row for row in categories if row["split"] == "val_unseen"]
    if val_unseen:
        labels = [row["category"].replace("_", " ") for row in val_unseen]
        values = [100.0 * row["rate"] for row in val_unseen]
        fig, ax = plt.subplots(figsize=(6.2, 3.0))
        ax.bar(labels, values)
        ax.set_ylabel("Episodes containing cue (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(fig_dir / "r2r_val_unseen_language_cues.png", dpi=240)
        fig.savefig(fig_dir / "r2r_val_unseen_language_cues.pdf")
        plt.close(fig)

    token_rows = top_tokens[:25]
    if token_rows:
        labels = [row["token"] for row in reversed(token_rows)]
        values = [row["count"] for row in reversed(token_rows)]
        fig, ax = plt.subplots(figsize=(5.0, 5.4))
        ax.barh(labels, values)
        ax.set_xlabel("Token count")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "r2r_top_tokens.png", dpi=240)
        fig.savefig(fig_dir / "r2r_top_tokens.pdf")
        plt.close(fig)

    if action_category:
        categories_order = list(CATEGORY_PATTERNS)
        action_order = ["move_forward", "turn_left", "turn_right", "stop"]
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharey=True)
        for ax, action in zip(axes.flat, action_order):
            values = []
            for category in categories_order:
                matches = [
                    row
                    for row in action_category
                    if row["category"] == category and row["action"] == action
                ]
                values.append(100.0 * matches[0]["rate"] if matches else math.nan)
            ax.bar([c.replace("_", "\n") for c in categories_order], values)
            ax.set_title(action)
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", labelsize=8)
        axes[0, 0].set_ylabel("NaVILA action share (%)")
        axes[1, 0].set_ylabel("NaVILA action share (%)")
        fig.tight_layout()
        fig.savefig(fig_dir / "navila_action_prior_by_language_cue.png", dpi=240)
        fig.savefig(fig_dir / "navila_action_prior_by_language_cue.pdf")
        plt.close(fig)


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(":---" if i == 0 else "---:" for i in range(len(columns))) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                if column.endswith("_rate") or column == "rate":
                    value = pct(value)
                else:
                    value = f"{value:.2f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(
    output_dir: Path,
    dataset_root: Path,
    splits: list[str],
    summaries: list[dict],
    categories: list[dict],
    pred_rows: list[dict],
) -> None:
    report = output_dir / "report.md"
    total_episodes = sum(row["episodes"] for row in summaries)
    val_unseen_categories = [
        row for row in categories if row["split"] == "val_unseen"
    ]
    lines = [
        "# R2R Text-Only Corpus Diagnostic",
        "",
        "This diagnostic uses R2R/VLN-CE annotation text only. It does not use",
        "MP3D scene assets and does not report SR, SPL, NE, nDTW, or simulator",
        "navigation performance.",
        "",
        "## Data",
        "",
        f"- Dataset root: `{dataset_root}`",
        f"- Splits: `{', '.join(splits)}`",
        f"- Total episodes analyzed: `{total_episodes}`",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Split Summary",
        "",
        *markdown_table(
            summaries,
            [
                "split",
                "episodes",
                "unique_scenes",
                "mean_words",
                "median_words",
                "min_words",
                "max_words",
            ],
        ),
        "",
        "## Val-Unseen Language Cue Rates",
        "",
        *markdown_table(
            val_unseen_categories,
            ["category", "episodes", "count", "rate"],
        ),
        "",
    ]
    if pred_rows:
        action_counts = Counter(row["parsed_action"] for row in pred_rows)
        total = sum(action_counts.values())
        action_table = [
            {
                "action": action,
                "count": count,
                "rate": count / total if total else math.nan,
            }
            for action, count in sorted(action_counts.items())
        ]
        lines.extend(
            [
                "## NaVILA Text-Probe Action Prior",
                "",
                (
                    "This section uses existing `vln_batch_diagnostic` saved "
                    "predictions. R2R text is paired with unmatched demo frames, "
                    "so it diagnoses language/action priors only."
                ),
                "",
                *markdown_table(action_table, ["action", "count", "rate"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Figures",
            "",
            "- `figures/r2r_instruction_length.png`",
            "- `figures/r2r_val_unseen_language_cues.png`",
            "- `figures/r2r_top_tokens.png`",
            "- `figures/navila_action_prior_by_language_cue.png` if saved predictions exist.",
            "",
            "## Paper-Safe Interpretation",
            "",
            (
                "These results support dataset and interface analysis: R2R "
                "instructions contain dense directional, spatial, room, and "
                "object cues, while current NaVILA text probes expose action "
                "prior bias under unmatched visuals. They should be used to "
                "motivate hierarchical VLN-to-DEGNAV interfaces, not as formal "
                "VLN benchmark evidence."
            ),
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    rows: list[EpisodeText] = []
    for split in splits:
        rows.extend(load_split(args.dataset_root, split))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = split_summary(rows)
    categories = category_rows(rows)
    top_tokens = token_counts(rows)
    scenes = scene_counts(rows)
    episode_rows = [
        {
            "split": row.split,
            "episode_id": row.episode_id,
            "scene_id": row.scene_id,
            "scene_name": row.scene_name,
            "trajectory_id": row.trajectory_id,
            "word_count": row.word_count,
            "char_count": row.char_count,
            **{category: row.categories[category] for category in CATEGORY_PATTERNS},
            "instruction": row.instruction,
        }
        for row in rows
    ]

    predictions = load_predictions(args.predictions)
    pred_rows = prediction_text_rows(predictions)
    action_category = action_by_category(pred_rows)

    write_csv(output_dir / "r2r_instruction_episodes.csv", episode_rows)
    write_csv(output_dir / "split_summary.csv", summaries)
    write_csv(output_dir / "scene_counts.csv", scenes)
    write_csv(output_dir / "top_tokens.csv", top_tokens)
    write_csv(output_dir / "language_cue_rates.csv", categories)
    write_csv(output_dir / "navila_action_by_language_cue.csv", action_category)

    metadata = {
        "dataset_root": str(args.dataset_root.resolve()),
        "splits": splits,
        "predictions": str(args.predictions.resolve())
        if args.predictions.exists()
        else "",
        "episode_count": len(rows),
        "prediction_text_probe_count": len(pred_rows),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Text-only R2R analysis. No MP3D scene assets, no Habitat rollout, "
            "and no formal VLN metrics are used."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    save_figures(output_dir, summaries, categories, top_tokens, action_category)
    write_report(output_dir, args.dataset_root, splits, summaries, categories, pred_rows)

    print(f"[write] {output_dir / 'report.md'}")
    print(f"[write] {output_dir / 'figures'}")
    print(f"[summary] episodes={len(rows)} text_probe_predictions={len(pred_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
