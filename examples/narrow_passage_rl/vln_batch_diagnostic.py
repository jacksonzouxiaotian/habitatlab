#!/usr/bin/env python3
"""Run a controlled large-batch diagnostic on a downloaded NaVILA model.

This is not a simulator benchmark. It combines explicit local-command probes,
perturbed frames from NaVILA's repository demo, and text instructions from the
R2R val-unseen split. The R2R instructions are deliberately paired with
unmatched demo frames, so they are used only to measure language sensitivity
and stop-override compliance. SR, SPL, NE, and nDTW are not computed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import platform
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from vln_dataset_free_smoke import (
    Case as SmokeCase,
    git_commit,
    load_demo_window,
    package_version,
    patch_transformers_4bit_compat,
    run_prediction,
    synthetic_corridor_frames,
    validate_model_artifacts,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


ACTIONS = ("move_forward", "turn_left", "turn_right", "stop")
PERTURBATIONS = (
    "clean",
    "brightness_low",
    "brightness_high",
    "blur",
    "gaussian_noise",
    "center_occlusion",
    "frame_dropout",
    "temporal_reverse",
)


@dataclass(frozen=True)
class BatchCase:
    case_id: str
    suite: str
    subgroup: str
    source: str
    instruction: str
    images: list[Image.Image]
    visual_id: str
    expected_action: str = ""
    label_scope: str = ""
    pair_id: str = ""
    perturbation: str = ""
    dataset_episode_id: str = ""
    scene_id: str = ""


def parse_args() -> argparse.Namespace:
    store = Path(
        os.environ.get(
            "VLN_STORE",
            "/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln",
        )
    )
    parser = argparse.ArgumentParser(
        description="Run a controlled, dataset-light NaVILA batch diagnostic."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=store / "models/navila-llama3-8b-8f",
    )
    parser.add_argument(
        "--navila-root",
        type=Path,
        default=Path("/home/xiaotian/vla/NaVILA"),
    )
    parser.add_argument(
        "--sample-gif",
        type=Path,
        default=Path("/home/xiaotian/vla/NaVILA/assets/sample.gif"),
    )
    parser.add_argument(
        "--r2r-json",
        type=Path,
        default=(
            store
            / "data/datasets/R2R_VLNCE_v1-3_preprocessed"
            / "val_unseen/val_unseen.json.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "examples/narrow_passage_rl/results/vln_batch_diagnostic"
        ),
    )
    parser.add_argument("--r2r-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional case limit for debugging; 0 runs the complete suite.",
    )
    return parser.parse_args()


def command_prompts() -> dict[str, tuple[str, ...]]:
    return {
        "move_forward": (
            "Move forward 50 centimeters.",
            "Proceed straight ahead through the passage.",
            "Continue forward toward the doorway.",
            "Advance straight without turning.",
        ),
        "turn_left": (
            "Turn left 45 degrees.",
            "Rotate 45 degrees to your left.",
            "Take the opening on the left.",
            "Face the left-hand corridor.",
        ),
        "turn_right": (
            "Turn right 45 degrees.",
            "Rotate 45 degrees to your right.",
            "Take the opening on the right.",
            "Face the right-hand corridor.",
        ),
        "stop": (
            "Stop now.",
            "The navigation task is complete. Stop.",
            "Do not move any farther.",
            "Remain at the current position.",
        ),
    }


def build_command_cases(frame_count: int) -> list[BatchCase]:
    contexts = {
        "open": synthetic_corridor_frames(frame_count, blocked=False),
        "blocked": synthetic_corridor_frames(frame_count, blocked=True),
        "left_branch": synthetic_corridor_frames(
            frame_count, blocked=False, turn="left"
        ),
        "right_branch": synthetic_corridor_frames(
            frame_count, blocked=False, turn="right"
        ),
    }
    cases: list[BatchCase] = []
    for context_name, images in contexts.items():
        for action, prompts in command_prompts().items():
            for prompt_index, instruction in enumerate(prompts):
                cases.append(
                    BatchCase(
                        case_id=(
                            f"command_{context_name}_{action}_{prompt_index:02d}"
                        ),
                        suite="command_grounding",
                        subgroup=context_name,
                        source="controlled_synthetic",
                        instruction=instruction,
                        images=[image.copy() for image in images],
                        visual_id=f"synthetic_{context_name}",
                        expected_action=action,
                        label_scope="explicit_local_command",
                    )
                )
    return cases


def demo_windows(sample_gif: Path, frame_count: int) -> dict[str, list[Image.Image]]:
    width = 0.24
    starts = np.linspace(0.0, 1.0 - width, 8)
    return {
        f"window_{index:02d}": load_demo_window(
            sample_gif,
            float(start),
            float(start + width),
            frame_count,
        )
        for index, start in enumerate(starts)
    }


def perturb_images(
    images: list[Image.Image],
    perturbation: str,
    seed: int,
) -> list[Image.Image]:
    if perturbation == "clean":
        return [image.copy() for image in images]
    if perturbation == "brightness_low":
        return [ImageEnhance.Brightness(image).enhance(0.55) for image in images]
    if perturbation == "brightness_high":
        return [ImageEnhance.Brightness(image).enhance(1.45) for image in images]
    if perturbation == "blur":
        return [image.filter(ImageFilter.GaussianBlur(radius=2.0)) for image in images]
    if perturbation == "gaussian_noise":
        rng = np.random.default_rng(seed)
        result = []
        for image in images:
            array = np.asarray(image, dtype=np.float32)
            noise = rng.normal(0.0, 20.0, size=array.shape)
            result.append(
                Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))
            )
        return result
    if perturbation == "center_occlusion":
        result = []
        for image in images:
            item = image.copy()
            draw = ImageDraw.Draw(item)
            width, height = item.size
            draw.rectangle(
                (
                    int(0.32 * width),
                    int(0.28 * height),
                    int(0.68 * width),
                    int(0.72 * height),
                ),
                fill=(32, 32, 32),
            )
            result.append(item)
        return result
    if perturbation == "frame_dropout":
        result = [images[0].copy()]
        for index in range(1, len(images)):
            source = result[-1] if index % 2 else images[index]
            result.append(source.copy())
        return result
    if perturbation == "temporal_reverse":
        return [image.copy() for image in reversed(images)]
    raise ValueError(f"Unknown perturbation: {perturbation}")


def build_visual_robustness_cases(
    sample_gif: Path,
    frame_count: int,
    seed: int,
) -> tuple[list[BatchCase], dict[str, list[Image.Image]]]:
    instruction = (
        "Walk through the small room with a lounge chair and dresser. Turn the "
        "corner and walk past the sink. Walk into the kitchen through the door "
        "on the other end. Stop near the long wooden dining table."
    )
    windows = demo_windows(sample_gif, frame_count)
    cases: list[BatchCase] = []
    for window_index, (window_id, images) in enumerate(windows.items()):
        for perturbation_index, perturbation in enumerate(PERTURBATIONS):
            cases.append(
                BatchCase(
                    case_id=f"robust_{window_id}_{perturbation}",
                    suite="visual_robustness",
                    subgroup=perturbation,
                    source="navila_repository_demo",
                    instruction=instruction,
                    images=perturb_images(
                        images,
                        perturbation,
                        seed + 100 * window_index + perturbation_index,
                    ),
                    visual_id=f"{window_id}_{perturbation}",
                    pair_id=window_id,
                    perturbation=perturbation,
                )
            )
    return cases, windows


def load_r2r_episodes(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"No episodes found in {path}")
    return episodes


def instruction_text(episode: dict[str, Any]) -> str:
    instruction = episode.get("instruction")
    if isinstance(instruction, dict):
        text = instruction.get("instruction_text", "")
    else:
        text = instruction or episode.get("instruction_text", "")
    text = str(text).strip()
    if not text:
        raise ValueError(
            f"Episode {episode.get('episode_id')} has no instruction text."
        )
    return text


def build_r2r_probe_cases(
    r2r_json: Path,
    windows: dict[str, list[Image.Image]],
    sample_count: int,
    seed: int,
) -> tuple[list[BatchCase], list[str]]:
    episodes = load_r2r_episodes(r2r_json)
    if sample_count > len(episodes):
        raise ValueError(
            f"Requested {sample_count} R2R probes, but only {len(episodes)} exist."
        )
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(episodes), size=sample_count, replace=False)
    window_items = list(windows.items())
    cases: list[BatchCase] = []
    selected_ids: list[str] = []
    for sample_index, episode_index in enumerate(selected_indices):
        episode = episodes[int(episode_index)]
        episode_id = str(episode.get("episode_id", episode_index))
        selected_ids.append(episode_id)
        pair_id = f"r2r_{sample_index:03d}_{episode_id}"
        window_id, images = window_items[sample_index % len(window_items)]
        original_instruction = instruction_text(episode)
        common = {
            "source": "r2r_val_unseen_text_with_unmatched_demo_frames",
            "images": [image.copy() for image in images],
            "visual_id": window_id,
            "pair_id": pair_id,
            "dataset_episode_id": episode_id,
            "scene_id": str(episode.get("scene_id", "")),
        }
        cases.append(
            BatchCase(
                case_id=f"{pair_id}_original",
                suite="r2r_language_probe",
                subgroup="original_instruction",
                instruction=original_instruction,
                **common,
            )
        )
        cases.append(
            BatchCase(
                case_id=f"{pair_id}_stop_override",
                suite="r2r_language_probe",
                subgroup="stop_override",
                instruction=(
                    f"{original_instruction} Instruction update: the navigation "
                    "task is complete; stop immediately."
                ),
                expected_action="stop",
                label_scope="explicit_stop_override_only",
                **common,
            )
        )
    return cases, selected_ids


def save_representative_fixtures(
    command_cases: list[BatchCase],
    robustness_cases: list[BatchCase],
    output_dir: Path,
) -> None:
    fixture_dir = output_dir / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for case in [*command_cases, *robustness_cases]:
        key = case.visual_id
        if key in seen:
            continue
        seen.add(key)
        case.images[-1].save(fixture_dir / f"{key}.png")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: csv_value(row.get(key)) for key in fields} for row in rows
        )


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - half, center + half


def rate_summary(values: Iterable[bool]) -> dict[str, Any]:
    items = [bool(value) for value in values]
    successes = sum(items)
    low, high = wilson_interval(successes, len(items))
    return {
        "count": len(items),
        "successes": successes,
        "rate": successes / len(items) if items else math.nan,
        "wilson95_low": low,
        "wilson95_high": high,
    }


def normalized_action_entropy(actions: list[str]) -> float:
    counts = Counter(action for action in actions if action in ACTIONS)
    total = sum(counts.values())
    if total == 0:
        return math.nan
    probabilities = [count / total for count in counts.values() if count]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return entropy / math.log(len(ACTIONS))


def prediction_row(case: BatchCase, prediction: Any) -> dict[str, Any]:
    exact_match: Optional[bool] = None
    if case.expected_action:
        exact_match = prediction.parsed_action == case.expected_action
    return {
        "case_id": case.case_id,
        "suite": case.suite,
        "subgroup": case.subgroup,
        "source": case.source,
        "dataset_episode_id": case.dataset_episode_id,
        "scene_id": case.scene_id,
        "pair_id": case.pair_id,
        "visual_id": case.visual_id,
        "perturbation": case.perturbation,
        "label_scope": case.label_scope,
        "instruction": case.instruction,
        "expected_action": case.expected_action,
        "response": prediction.response,
        "parsed_action": prediction.parsed_action,
        "action_value": prediction.action_value,
        "valid_action": prediction.valid_action,
        "exact_match": exact_match,
        "latency_seconds": prediction.latency_seconds,
        "input_tokens": prediction.input_tokens,
        "output_tokens": prediction.output_tokens,
        "peak_gpu_memory_mb": prediction.peak_gpu_memory_mb,
    }


def command_analysis(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    command_rows = [row for row in rows if row["suite"] == "command_grounding"]
    summaries: list[dict[str, Any]] = []
    for expected in ACTIONS:
        subset = [row for row in command_rows if row["expected_action"] == expected]
        stats = rate_summary(row["exact_match"] for row in subset)
        summaries.append(
            {
                "expected_action": expected,
                **stats,
                "predicted_move_forward": sum(
                    row["parsed_action"] == "move_forward" for row in subset
                ),
                "predicted_turn_left": sum(
                    row["parsed_action"] == "turn_left" for row in subset
                ),
                "predicted_turn_right": sum(
                    row["parsed_action"] == "turn_right" for row in subset
                ),
                "predicted_stop": sum(
                    row["parsed_action"] == "stop" for row in subset
                ),
            }
        )

    confusion: list[dict[str, Any]] = []
    for expected in ACTIONS:
        subset = [row for row in command_rows if row["expected_action"] == expected]
        for predicted in ACTIONS:
            count = sum(row["parsed_action"] == predicted for row in subset)
            confusion.append(
                {
                    "expected_action": expected,
                    "predicted_action": predicted,
                    "count": count,
                    "row_rate": count / len(subset) if subset else math.nan,
                }
            )
    return summaries, confusion


def visual_analysis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visual_rows = [row for row in rows if row["suite"] == "visual_robustness"]
    clean_by_pair = {
        row["pair_id"]: row
        for row in visual_rows
        if row["perturbation"] == "clean"
    }
    for row in visual_rows:
        clean = clean_by_pair[row["pair_id"]]
        row["action_matches_clean"] = row["parsed_action"] == clean["parsed_action"]
        row["response_matches_clean"] = row["response"] == clean["response"]

    summaries: list[dict[str, Any]] = []
    for perturbation in PERTURBATIONS:
        subset = [
            row for row in visual_rows if row["perturbation"] == perturbation
        ]
        action_stats = rate_summary(row["action_matches_clean"] for row in subset)
        response_stats = rate_summary(row["response_matches_clean"] for row in subset)
        summaries.append(
            {
                "perturbation": perturbation,
                "count": len(subset),
                "action_consistency": action_stats["rate"],
                "action_wilson95_low": action_stats["wilson95_low"],
                "action_wilson95_high": action_stats["wilson95_high"],
                "exact_response_consistency": response_stats["rate"],
            }
        )
    return summaries


def r2r_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    r2r_rows = [row for row in rows if row["suite"] == "r2r_language_probe"]
    originals = [row for row in r2r_rows if row["subgroup"] == "original_instruction"]
    overrides = [row for row in r2r_rows if row["subgroup"] == "stop_override"]
    originals_by_pair = {row["pair_id"]: row for row in originals}
    for row in overrides:
        row["action_changed_from_original"] = (
            row["parsed_action"]
            != originals_by_pair[row["pair_id"]]["parsed_action"]
        )
    original_counts = Counter(row["parsed_action"] for row in originals)
    override_counts = Counter(row["parsed_action"] for row in overrides)
    stop_stats = rate_summary(row["parsed_action"] == "stop" for row in overrides)
    change_stats = rate_summary(row["action_changed_from_original"] for row in overrides)
    return {
        "original_count": len(originals),
        "stop_override_count": len(overrides),
        "original_action_distribution": dict(sorted(original_counts.items())),
        "stop_override_action_distribution": dict(sorted(override_counts.items())),
        "original_normalized_action_entropy": normalized_action_entropy(
            [row["parsed_action"] for row in originals]
        ),
        "stop_override_compliance": stop_stats,
        "stop_override_action_change": change_stats,
        "original_unique_response_count": len(
            {row["response"] for row in originals}
        ),
    }


def severity(score: float) -> str:
    if score >= 0.50:
        return "high"
    if score >= 0.20:
        return "medium"
    return "low"


def build_issues(
    rows: list[dict[str, Any]],
    command_summary: list[dict[str, Any]],
    visual_summary: list[dict[str, Any]],
    r2r_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    command_rate = {
        row["expected_action"]: row["rate"] for row in command_summary
    }
    explicit_stop_rows = [
        row
        for row in rows
        if row["expected_action"] == "stop"
        and row["label_scope"]
        in {"explicit_local_command", "explicit_stop_override_only"}
    ]
    explicit_stop_rate = mean(
        row["parsed_action"] == "stop" for row in explicit_stop_rows
    )
    blocked_forward_rows = [
        row
        for row in rows
        if row["suite"] == "command_grounding"
        and row["subgroup"] == "blocked"
        and row["expected_action"] == "move_forward"
    ]
    blocked_forward_rate = mean(
        row["parsed_action"] == "move_forward" for row in blocked_forward_rows
    )
    turn_rate = mean([command_rate["turn_left"], command_rate["turn_right"]])
    original_distribution = r2r_summary["original_action_distribution"]
    original_total = sum(original_distribution.values())
    dominant_share = (
        max(original_distribution.values()) / original_total
        if original_total
        else math.nan
    )
    photometric_names = {
        "brightness_low",
        "brightness_high",
        "blur",
        "gaussian_noise",
        "center_occlusion",
    }
    photometric_consistency = mean(
        row["action_consistency"]
        for row in visual_summary
        if row["perturbation"] in photometric_names
    )
    temporal_rows = {
        row["perturbation"]: row["action_consistency"]
        for row in visual_summary
    }

    issues = [
        {
            "issue": "Explicit stop commands are ignored",
            "evidence_metric": "explicit_stop_noncompliance_rate",
            "observed_value": 1.0 - explicit_stop_rate,
            "problem_score": 1.0 - explicit_stop_rate,
            "interpretation": (
                "The policy continues moving after an explicit completion/stop cue."
            ),
            "recommended_improvement": (
                "Add stop/termination supervision, counterfactual stop examples, "
                "and a deterministic completion safety gate."
            ),
        },
        {
            "issue": "Blocked views do not reliably suppress forward motion",
            "evidence_metric": "blocked_forward_output_rate",
            "observed_value": blocked_forward_rate,
            "problem_score": blocked_forward_rate,
            "interpretation": (
                "A forward command can dominate an obvious synthetic obstruction."
            ),
            "recommended_improvement": (
                "Fuse depth-derived clearance and collision risk below the VLN "
                "waypoint policy; mask unsafe forward actions."
            ),
        },
        {
            "issue": "Left/right local command grounding is weak",
            "evidence_metric": "mean_turn_command_failure_rate",
            "observed_value": 1.0 - turn_rate,
            "problem_score": 1.0 - turn_rate,
            "interpretation": (
                "Explicit left/right instructions are not consistently reflected "
                "in the next action."
            ),
            "recommended_improvement": (
                "Balance action labels and add direction-contrastive instruction "
                "pairs with matched visual branches."
            ),
        },
        {
            "issue": "Action outputs are dominated by one action prior",
            "evidence_metric": "dominant_action_share_on_r2r_text_probe",
            "observed_value": dominant_share,
            "problem_score": max(0.0, (dominant_share - 0.25) / 0.75),
            "interpretation": (
                "With unmatched observations, the model exposes a strong learned "
                "action prior rather than broad instruction-conditioned variation."
            ),
            "recommended_improvement": (
                "Use action-balanced training, hard counterfactuals, and report "
                "per-action calibration/confusion."
            ),
        },
        {
            "issue": "Photometric and occlusion sensitivity",
            "evidence_metric": "photometric_action_flip_rate",
            "observed_value": 1.0 - photometric_consistency,
            "problem_score": 1.0 - photometric_consistency,
            "interpretation": (
                "Mild appearance changes can alter the discrete navigation action."
            ),
            "recommended_improvement": (
                "Train with lighting, blur, noise, dropout, and occlusion "
                "augmentation; add depth/geometry features."
            ),
        },
        {
            "issue": "Temporal history perturbations change decisions",
            "evidence_metric": "temporal_reverse_action_flip_rate",
            "observed_value": 1.0
            - temporal_rows.get("temporal_reverse", math.nan),
            "problem_score": 1.0
            - temporal_rows.get("temporal_reverse", math.nan),
            "interpretation": (
                "The output is sensitive to history order; this is diagnostic and "
                "not necessarily an error because reversal changes motion context."
            ),
            "recommended_improvement": (
                "Evaluate with matched trajectory labels and train temporal-order "
                "and frame-dropout augmentations."
            ),
        },
    ]
    for issue in issues:
        issue["severity"] = severity(float(issue["problem_score"]))
    return sorted(
        issues,
        key=lambda item: (-float(item["problem_score"]), item["issue"]),
    )


def aggregate_summary(rows: list[dict[str, Any]], model_info: dict[str, float]) -> dict[str, Any]:
    action_counts = Counter(row["parsed_action"] for row in rows)
    suite_counts = Counter(row["suite"] for row in rows)
    expected_rows = [row for row in rows if row["expected_action"]]
    return {
        "status": "pass",
        "case_count": len(rows),
        "suite_counts": dict(sorted(suite_counts.items())),
        "valid_action_rate": mean(bool(row["valid_action"]) for row in rows),
        "labeled_case_count": len(expected_rows),
        "labeled_exact_match_rate": mean(
            bool(row["exact_match"]) for row in expected_rows
        ),
        "action_distribution": dict(sorted(action_counts.items())),
        "mean_latency_seconds": mean(row["latency_seconds"] for row in rows),
        "median_latency_seconds": median(row["latency_seconds"] for row in rows),
        "p95_latency_seconds": float(
            np.percentile(
                np.asarray([row["latency_seconds"] for row in rows]), 95
            )
        ),
        "mean_output_tokens": mean(row["output_tokens"] for row in rows),
        "peak_gpu_memory_mb": max(row["peak_gpu_memory_mb"] for row in rows),
        **model_info,
    }


def plot_results(
    output_dir: Path,
    command_summary: list[dict[str, Any]],
    visual_summary: list[dict[str, Any]],
    r2r_summary: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    labels = [row["expected_action"].replace("_", " ") for row in command_summary]
    rates = np.asarray([row["rate"] for row in command_summary])
    lower = rates - np.asarray([row["wilson95_low"] for row in command_summary])
    upper = np.asarray([row["wilson95_high"] for row in command_summary]) - rates
    fig, axis = plt.subplots(figsize=(6.4, 3.5))
    axis.bar(np.arange(len(labels)), rates, yerr=[lower, upper], capsize=4)
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Exact command compliance")
    axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(figure_dir / "command_compliance.png", dpi=240)
    fig.savefig(figure_dir / "command_compliance.pdf")
    plt.close(fig)

    visual_labels = [
        row["perturbation"].replace("_", " ") for row in visual_summary
    ]
    visual_rates = [row["action_consistency"] for row in visual_summary]
    fig, axis = plt.subplots(figsize=(8.2, 3.8))
    axis.bar(np.arange(len(visual_labels)), visual_rates)
    axis.set_xticks(
        np.arange(len(visual_labels)),
        visual_labels,
        rotation=28,
        ha="right",
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Action consistency vs clean")
    axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(figure_dir / "visual_robustness.png", dpi=240)
    fig.savefig(figure_dir / "visual_robustness.pdf")
    plt.close(fig)

    original = r2r_summary["original_action_distribution"]
    override = r2r_summary["stop_override_action_distribution"]
    x = np.arange(len(ACTIONS))
    original_total = max(sum(original.values()), 1)
    override_total = max(sum(override.values()), 1)
    fig, axis = plt.subplots(figsize=(6.8, 3.6))
    axis.bar(
        x - 0.18,
        [original.get(action, 0) / original_total for action in ACTIONS],
        width=0.36,
        label="Original R2R text",
    )
    axis.bar(
        x + 0.18,
        [override.get(action, 0) / override_total for action in ACTIONS],
        width=0.36,
        label="Explicit stop override",
    )
    axis.set_xticks(x, [action.replace("_", " ") for action in ACTIONS])
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Output frequency")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(figure_dir / "r2r_stop_override.png", dpi=240)
    fig.savefig(figure_dir / "r2r_stop_override.pdf")
    plt.close(fig)


def percent(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def markdown_report(
    metadata: dict[str, Any],
    summary: dict[str, Any],
    command_summary: list[dict[str, Any]],
    visual_summary: list[dict[str, Any]],
    r2r_summary: dict[str, Any],
    issues: list[dict[str, Any]],
) -> str:
    lines = [
        "# NaVILA Large-Batch Controlled Diagnostic",
        "",
        "## Scope",
        "",
        (
            "This run diagnoses a downloaded NaVILA checkpoint without MP3D "
            "scene assets. It is not an R2R navigation benchmark: R2R text "
            "instructions are paired with unmatched repository-demo frames only "
            "to probe language sensitivity and explicit stop overrides. SR, SPL, "
            "NE, and nDTW are therefore not reported."
        ),
        "",
        "## Test Composition",
        "",
        "| Suite | Cases | Valid interpretation |",
        "| :--- | ---: | :--- |",
        (
            f"| Controlled local-command grounding | "
            f"{summary['suite_counts'].get('command_grounding', 0)} | "
            "Exact action compliance on synthetic fixtures |"
        ),
        (
            f"| Repository-demo visual robustness | "
            f"{summary['suite_counts'].get('visual_robustness', 0)} | "
            "Action consistency relative to clean frames |"
        ),
        (
            f"| R2R val-unseen language probe | "
            f"{summary['suite_counts'].get('r2r_language_probe', 0)} | "
            "Action prior and stop-override response; not navigation accuracy |"
        ),
        f"| **Total** | **{summary['case_count']}** | |",
        "",
        "## Runtime",
        "",
        "| Metric | Result |",
        "| :--- | ---: |",
        f"| Completed inference cases | {summary['case_count']} |",
        f"| Parseable action rate | {percent(summary['valid_action_rate'])} |",
        f"| Mean latency | {summary['mean_latency_seconds']:.3f} s |",
        f"| P95 latency | {summary['p95_latency_seconds']:.3f} s |",
        f"| Model load time | {summary['model_load_seconds']:.3f} s |",
        f"| GPU memory after load | {summary['model_memory_after_load_mb']:.1f} MiB |",
        f"| Peak GPU memory | {summary['peak_gpu_memory_mb']:.1f} MiB |",
        "",
        "## Explicit Command Grounding",
        "",
        "| Expected action | Cases | Exact match | Wilson 95% CI |",
        "| :--- | ---: | ---: | :--- |",
    ]
    for row in command_summary:
        lines.append(
            f"| {row['expected_action'].replace('_', ' ')} | {row['count']} | "
            f"{percent(row['rate'])} | "
            f"[{percent(row['wilson95_low'])}, {percent(row['wilson95_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Visual Robustness",
            "",
            "| Perturbation | Windows | Action consistency | Exact-response consistency |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for row in visual_summary:
        lines.append(
            f"| {row['perturbation'].replace('_', ' ')} | {row['count']} | "
            f"{percent(row['action_consistency'])} | "
            f"{percent(row['exact_response_consistency'])} |"
        )
    stop_stats = r2r_summary["stop_override_compliance"]
    change_stats = r2r_summary["stop_override_action_change"]
    lines.extend(
        [
            "",
            "## R2R Language Probe",
            "",
            (
                f"- Original instructions: {r2r_summary['original_count']}; "
                f"stop-overridden paired instructions: "
                f"{r2r_summary['stop_override_count']}."
            ),
            (
                "- Original action distribution: "
                f"`{r2r_summary['original_action_distribution']}`."
            ),
            (
                "- Stop-override action distribution: "
                f"`{r2r_summary['stop_override_action_distribution']}`."
            ),
            (
                f"- Explicit stop-override compliance: {percent(stop_stats['rate'])} "
                f"(Wilson 95% CI "
                f"[{percent(stop_stats['wilson95_low'])}, "
                f"{percent(stop_stats['wilson95_high'])}])."
            ),
            (
                f"- Action changed after the stop override: "
                f"{percent(change_stats['rate'])}."
            ),
            (
                f"- Normalized action entropy on original instructions: "
                f"{r2r_summary['original_normalized_action_entropy']:.3f}."
            ),
            "",
            "## Ranked Problems and Improvements",
            "",
            "| Rank | Severity | Observed issue | Evidence | Improvement direction |",
            "| ---: | :--- | :--- | ---: | :--- |",
        ]
    )
    for rank, issue in enumerate(issues, start=1):
        lines.append(
            f"| {rank} | {issue['severity']} | {issue['issue']} | "
            f"{percent(float(issue['observed_value']))} | "
            f"{issue['recommended_improvement']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence-Aligned Conclusions",
            "",
            (
                "- This run can support conclusions about model loading, action "
                "format, explicit command compliance, controlled perturbation "
                "sensitivity, and action-prior behavior."
            ),
            (
                "- It cannot support claims about navigation success, scene "
                "generalization, trajectory efficiency, or sim-to-real transfer."
            ),
            (
                "- The next formal evaluation requires MP3D scene assets and "
                "paired R2R episodes, followed by SR/SPL/NE/nDTW evaluation."
            ),
            "",
            "## Provenance",
            "",
            f"- Model: `{metadata['model_path']}`",
            f"- NaVILA commit: `{metadata['navila_commit']}`",
            f"- R2R text source: `{metadata['r2r_json']}`",
            f"- R2R source episode count: {metadata['r2r_source_episode_count']}",
            f"- Selected R2R text probes: {metadata['r2r_selected_count']}",
            f"- Seed: {metadata['seed']}",
            f"- Quantization: `{metadata['quantization']}`",
            f"- GPU: `{metadata['runtime']['gpu_name']}`",
            f"- Generated: `{metadata['generated_at']}`",
            "",
        ]
    )
    return "\n".join(lines)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this NaVILA batch diagnostic.")
    if not args.sample_gif.is_file():
        raise FileNotFoundError(args.sample_gif)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    artifact_validation = validate_model_artifacts(args.model)
    compat_patch_applied = patch_transformers_4bit_compat()

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    tokenizer, model, image_processor, context_length = load_pretrained_model(
        str(args.model),
        get_model_name_from_path(str(args.model)),
        load_4bit=True,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    model_memory_mb = torch.cuda.memory_allocated() / (1024**2)
    frame_count = int(getattr(model.config, "num_video_frames", 8))

    command_cases = build_command_cases(frame_count)
    robustness_cases, windows = build_visual_robustness_cases(
        args.sample_gif, frame_count, args.seed
    )
    r2r_cases, selected_r2r_ids = build_r2r_probe_cases(
        args.r2r_json,
        windows,
        args.r2r_samples,
        args.seed,
    )
    cases = [*command_cases, *robustness_cases, *r2r_cases]
    if args.limit > 0:
        cases = cases[: args.limit]
    save_representative_fixtures(
        command_cases, robustness_cases, args.output_dir
    )

    rows: list[dict[str, Any]] = []
    raw_path = args.output_dir / "predictions.csv"
    for index, case in enumerate(cases, start=1):
        prediction = run_prediction(
            SmokeCase(
                case_id=case.case_id,
                source=case.source,
                instruction=case.instruction,
                images=case.images,
                heuristic_expected_action=case.expected_action,
            ),
            tokenizer,
            model,
            image_processor,
            args.max_new_tokens,
        )
        rows.append(prediction_row(case, prediction))
        if index % 16 == 0 or index == len(cases):
            write_rows(rows, raw_path)
            print(
                f"[progress] {index}/{len(cases)} "
                f"last={prediction.parsed_action}",
                flush=True,
            )

    command_summary, confusion = command_analysis(rows)
    visual_summary = visual_analysis(rows)
    r2r_summary = r2r_analysis(rows)
    model_info = {
        "model_load_seconds": load_seconds,
        "model_memory_after_load_mb": model_memory_mb,
    }
    summary = aggregate_summary(rows, model_info)
    issues = build_issues(
        rows, command_summary, visual_summary, r2r_summary
    )

    write_rows(command_summary, args.output_dir / "command_compliance.csv")
    write_rows(confusion, args.output_dir / "command_confusion.csv")
    write_rows(visual_summary, args.output_dir / "visual_robustness.csv")
    write_rows(issues, args.output_dir / "issues_and_improvements.csv")

    suite_rows = []
    for suite, count in sorted(Counter(row["suite"] for row in rows).items()):
        subset = [row for row in rows if row["suite"] == suite]
        suite_rows.append(
            {
                "suite": suite,
                "cases": count,
                "valid_action_rate": mean(
                    bool(row["valid_action"]) for row in subset
                ),
                "mean_latency_seconds": mean(
                    row["latency_seconds"] for row in subset
                ),
                "move_forward_rate": mean(
                    row["parsed_action"] == "move_forward" for row in subset
                ),
                "turn_left_rate": mean(
                    row["parsed_action"] == "turn_left" for row in subset
                ),
                "turn_right_rate": mean(
                    row["parsed_action"] == "turn_right" for row in subset
                ),
                "stop_rate": mean(
                    row["parsed_action"] == "stop" for row in subset
                ),
            }
        )
    write_rows(suite_rows, args.output_dir / "summary_by_suite.csv")

    source_episode_count = len(load_r2r_episodes(args.r2r_json))
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "scope": "controlled large-batch VLN diagnostic without MP3D scenes",
        "model_path": str(args.model.resolve()),
        "navila_root": str(args.navila_root.resolve()),
        "navila_commit": git_commit(args.navila_root),
        "sample_gif": str(args.sample_gif.resolve()),
        "r2r_json": str(args.r2r_json.resolve()),
        "r2r_source_episode_count": source_episode_count,
        "r2r_selected_count": len(selected_r2r_ids),
        "r2r_selected_episode_ids": selected_r2r_ids,
        "r2r_pairing_warning": (
            "R2R instructions are paired with unmatched NaVILA repository-demo "
            "frames. They are language probes, not navigation episodes."
        ),
        "seed": args.seed,
        "frame_count": frame_count,
        "max_new_tokens": args.max_new_tokens,
        "context_length": context_length,
        "case_generation": {
            "command_grounding": len(command_cases),
            "visual_robustness": len(robustness_cases),
            "r2r_language_probe": len(r2r_cases),
            "executed_total": len(cases),
            "complete_requested_total": (
                len(command_cases) + len(robustness_cases) + len(r2r_cases)
            ),
        },
        "quantization": "bitsandbytes NF4 4-bit",
        "artifact_validation": artifact_validation,
        "model_load": {
            "status": "pass",
            "seconds": load_seconds,
            "gpu_memory_mb": model_memory_mb,
            "transformers_4bit_compat_patch_applied": compat_patch_applied,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_mb": (
                torch.cuda.get_device_properties(0).total_memory / (1024**2)
            ),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "bitsandbytes": package_version("bitsandbytes"),
            "flash_attn": package_version("flash-attn"),
            "vila": package_version("vila"),
        },
        "summary": summary,
        "command_analysis": command_summary,
        "visual_analysis": visual_summary,
        "r2r_analysis": r2r_summary,
        "ranked_issues": issues,
        "limitations": [
            "No MP3D scene observations or simulator trajectories are used.",
            "R2R instruction probes use unmatched visual observations.",
            "Synthetic fixtures are controlled interface tests, not a benchmark.",
            "SR, SPL, NE, nDTW, scene generalization, and sim-to-real performance "
            "cannot be inferred.",
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(json_safe(metadata), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(
            json_safe(
                {
                    "metadata": metadata,
                    "predictions": rows,
                }
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        markdown_report(
            metadata,
            summary,
            command_summary,
            visual_summary,
            r2r_summary,
            issues,
        ),
        encoding="utf-8",
    )
    plot_results(
        args.output_dir,
        command_summary,
        visual_summary,
        r2r_summary,
    )

    print(json.dumps(json_safe(summary), indent=2))
    print(f"[write] {raw_path}")
    print(f"[write] {args.output_dir / 'report.md'}")
    print(f"[write] {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
