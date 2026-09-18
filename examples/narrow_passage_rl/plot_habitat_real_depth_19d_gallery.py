#!/usr/bin/env python3
"""Collect and plot 12 real Habitat-Sim depth observations from HM3D.

The default dataset contains 24 extreme-narrow episodes from 12 HM3D scenes.
One episode is selected per scene, the Geometry-FSM controller is executed, and
the valid trajectory frame with the smallest observed body margin is retained.
Raw normalized depth, metric depth, RGB, the 19-D feature vector, and provenance
metadata are saved alongside paper-ready PNG/PDF figures.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from record_habitat_video import (  # noqa: E402
    GeometryFSMController,
    _features,
    _reset_to_episode,
    make_env,
)
from habitat.tasks.narrow_passage.geometry import (  # noqa: E402
    FEATURE_NAMES,
    depth_to_metric_units,
)


DEFAULT_DATASET = Path(
    "data/datasets/narrow_passage/extreme_narrow/extreme_narrow.json.gz"
)
DEFAULT_SCENE_ROOT = Path("data/versioned_data/hm3d-0.2/hm3d")
DEFAULT_OUTPUT_DIR = Path(
    "examples/narrow_passage_rl/results/narrow_passage_rl/figures/"
    "habitat_real_depth_12"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_dataset(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _scene_name(scene_id: str) -> str:
    path = Path(scene_id)
    if path.parent.name:
        return path.parent.name.split("-", maxsplit=1)[-1]
    return path.stem


def _resolve_scene(scene_id: str, scene_root: Path) -> Path:
    """Map dataset-relative HM3D scene IDs to the local HM3D asset tree."""
    parts = Path(scene_id).parts
    try:
        hm3d_index = parts.index("hm3d")
    except ValueError as exc:
        raise ValueError(f"Scene ID does not contain an hm3d component: {scene_id}") from exc
    resolved = (scene_root / Path(*parts[hm3d_index + 1 :])).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"HM3D scene asset is missing: {resolved}")
    return resolved


def _select_one_per_scene(data: dict[str, Any], count: int) -> list[tuple[int, dict[str, Any]]]:
    """Prefer the episode with the smallest annotated body margin per scene."""
    best: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, episode in enumerate(data.get("episodes", [])):
        scene_id = str(episode.get("scene_id", ""))
        margin = float(episode.get("info", {}).get("body_margin", float("inf")))
        previous = best.get(scene_id)
        if previous is None:
            best[scene_id] = (index, episode)
            continue
        previous_margin = float(
            previous[1].get("info", {}).get("body_margin", float("inf"))
        )
        if margin < previous_margin:
            best[scene_id] = (index, episode)
    selected = sorted(best.values(), key=lambda item: _scene_name(item[1]["scene_id"]))
    if len(selected) < count:
        raise ValueError(
            f"Requested {count} distinct scenes, but dataset only contains {len(selected)}"
        )
    return selected[:count]


def _patched_dataset(
    source: dict[str, Any], selected: list[tuple[int, dict[str, Any]]], scene_root: Path
) -> tuple[dict[str, Any], list[int]]:
    patched = copy.deepcopy(source)
    patched["episodes"] = []
    original_indices: list[int] = []
    for original_index, episode in selected:
        item = copy.deepcopy(episode)
        item["scene_id"] = str(_resolve_scene(str(item["scene_id"]), scene_root))
        patched["episodes"].append(item)
        original_indices.append(original_index)
    return patched, original_indices


def _metric_depth(depth: np.ndarray) -> np.ndarray:
    converted = depth_to_metric_units(
        depth,
        normalized=True,
        min_depth=0.0,
        max_depth=10.0,
    )
    if converted is None:
        raise ValueError("Habitat observation did not contain depth")
    arr = np.asarray(converted, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _snapshot(
    obs: dict[str, Any], env: Any, *, step: int, mode: str
) -> dict[str, Any]:
    depth_normalized = np.asarray(obs["depth"], dtype=np.float32)
    if depth_normalized.ndim == 3:
        depth_normalized = depth_normalized[..., 0]
    rgb = np.asarray(obs.get("rgb", np.empty((0, 0, 3), dtype=np.uint8)))
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    features = _features(obs).copy()
    agent_state = env.sim.get_agent_state()
    return {
        "step": int(step),
        "mode": str(mode),
        "depth_normalized": depth_normalized.copy(),
        "depth_m": _metric_depth(depth_normalized),
        "rgb": rgb.astype(np.uint8, copy=True),
        "features": features,
        "agent_position": np.asarray(agent_state.position, dtype=np.float32),
    }


def _valid_frame(sample: dict[str, Any]) -> bool:
    depth = sample["depth_m"]
    valid_fraction = float(np.mean(np.isfinite(depth) & (depth > 1.0e-4)))
    # Some HM3D views legitimately contain large open/invalid regions (windows,
    # stair voids, or missing ceiling geometry).  The six task ROIs only require
    # valid local returns, so do not reject an otherwise genuine frame merely
    # because less than half of the full image has a return.
    valid_sectors = int(np.count_nonzero(sample["features"][:6] > 1.0e-3))
    margin = float(sample["features"][9])
    return (
        valid_fraction >= 0.1
        and valid_sectors >= 5
        and margin <= 0.8
        and np.isfinite(sample["features"]).all()
    )


def _frame_score(sample: dict[str, Any]) -> tuple[float, float, int]:
    """Prefer readable passage geometry while staying near a tight margin."""
    features = sample["features"]
    margin = float(features[9])
    valid = sample["depth_m"]
    valid = valid[np.isfinite(valid) & (valid > 1.0e-4)]
    clipped = np.clip(valid, 0.0, 5.0)
    image_contrast = float(np.percentile(clipped, 90) - np.percentile(clipped, 10))
    sector_contrast = float(np.std(features[:6]))
    valid_fraction = float(valid.size / sample["depth_m"].size)
    # A pure nearest-margin criterion often selects a nearly uniform close wall.
    # Contrast makes the six partitions legible; the margin term keeps the view
    # tied to narrow-passage behavior rather than a distant open room.
    quality = (
        image_contrast
        + 1.2 * sector_contrast
        + 0.4 * min(valid_fraction, 0.8)
        - 0.35 * abs(margin - 0.15)
    )
    return -quality, abs(margin - 0.15), -int(sample["step"])


def _collect_episode(env: Any, patched_index: int, data_path: str, max_steps: int) -> dict[str, Any]:
    obs = _reset_to_episode(env, data_path, patched_index)
    controller = GeometryFSMController()
    candidates: list[dict[str, Any]] = []
    mode = "INIT"
    step = 0
    while step <= max_steps:
        sample = _snapshot(obs, env, step=step, mode=mode)
        if _valid_frame(sample):
            candidates.append(sample)
        if step == max_steps or env.episode_over:
            break
        action, mode = controller.act(env, obs, sample["features"], step)
        obs = env.step(action)
        step += 1
    if not candidates:
        raise RuntimeError(f"No valid depth frames collected for patched episode {patched_index}")
    best = min(candidates, key=_frame_score)
    best["trajectory_steps"] = int(step)
    return best


def _roi_geometry(height: int, width: int) -> dict[str, tuple[int, int, int, int]]:
    y_far_0, y_far_1 = int(0.25 * height), int(0.55 * height)
    y_near_0, y_near_1 = int(0.55 * height), int(0.90 * height)
    x0, x1, x2, x3 = 0, int(width / 3), int(2 * width / 3), width
    return {
        "F-L": (x0, y_far_0, x1, y_far_1),
        "F-C": (x1, y_far_0, x2, y_far_1),
        "F-R": (x2, y_far_0, x3, y_far_1),
        "N-L": (x0, y_near_0, x1, y_near_1),
        "N-C": (x1, y_near_0, x2, y_near_1),
        "N-R": (x2, y_near_0, x3, y_near_1),
    }


def _draw_rois(ax: Any, sample: dict[str, Any]) -> None:
    height, width = sample["depth_m"].shape
    values = sample["features"]
    value_map = {
        "N-L": values[0],
        "N-C": values[1],
        "N-R": values[2],
        "F-L": values[3],
        "F-C": values[4],
        "F-R": values[5],
    }
    for label, (x0, y0, x1, y1) in _roi_geometry(height, width).items():
        color = "#00E5FF" if label.startswith("F") else "#FFB000"
        ax.add_patch(
            patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor=color, linewidth=1.25,
            )
        )
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            f"{label}\n{float(value_map[label]):.2f}m",
            ha="center",
            va="center",
            fontsize=5.7,
            color="white",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.16", "fc": "#111111B8", "ec": "none"},
        )


def _plot_gallery(samples: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 180})
    fig, axes = plt.subplots(3, 4, figsize=(13.4, 10.0))
    fig.subplots_adjust(left=0.025, right=0.99, top=0.91, bottom=0.13, wspace=0.18, hspace=0.31)
    image = None
    letters = "abcdefghijkl"
    for number, (ax, sample) in enumerate(zip(axes.flat, samples)):
        image = ax.imshow(sample["depth_m"], cmap="turbo_r", vmin=0.0, vmax=5.0)
        _draw_rois(ax, sample)
        feature = sample["features"]
        ax.set_title(
            f"({letters[number]}) {sample['scene']}  |  ep {sample['episode_id']}\n"
            f"step {sample['step']}/{sample['trajectory_steps']}  "
            f"est.W={feature[8]:.2f} m  est.margin={feature[9]:.2f} m",
            fontsize=7.8,
            pad=4,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    if image is None:
        raise ValueError("No samples supplied")
    colorbar_axis = fig.add_axes([0.18, 0.061, 0.64, 0.017])
    cbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    cbar.set_label("Metric depth (m), clipped at 5 m", fontsize=9)
    fig.suptitle(
        "Twelve Real Habitat-Sim Depth Observations from HM3D Narrow Passages",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.018,
        "Cyan: far band (25%-55% H); orange: near band (55%-90% H). "
        "Each cell reports the valid-pixel 10th percentile used in x[0:6].",
        ha="center",
        fontsize=8,
        color="#333333",
    )
    png_path = output_dir / "habitat_real_depth_12_gallery.png"
    pdf_path = output_dir / "habitat_real_depth_12_gallery.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _plot_rgb_gallery(samples: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    """Plot the unmodified Habitat RGB observations in depth-gallery order."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 180})
    fig, axes = plt.subplots(3, 4, figsize=(13.4, 9.2))
    fig.subplots_adjust(left=0.025, right=0.99, top=0.91, bottom=0.045, wspace=0.18, hspace=0.31)
    letters = "abcdefghijkl"
    for number, (ax, sample) in enumerate(zip(axes.flat, samples)):
        rgb = sample["rgb"]
        if not rgb.size:
            raise ValueError(f"Sample {number + 1} does not contain an RGB observation")
        ax.imshow(rgb)
        ax.set_title(
            f"({letters[number]}) {sample['scene']}  |  ep {sample['episode_id']}\n"
            f"step {sample['step']}/{sample['trajectory_steps']}",
            fontsize=8.2,
            pad=4,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Original Habitat-Sim RGB Observations Corresponding to the 12 Depth Frames",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Panels (a)-(l) use exactly the same scene, episode, step, and order as the depth gallery.",
        ha="center",
        fontsize=8.5,
        color="#333333",
    )
    png_path = output_dir / "habitat_real_rgb_12_gallery.png"
    pdf_path = output_dir / "habitat_real_rgb_12_gallery.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _plot_feature_matrix(samples: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    matrix = np.stack([sample["features"] for sample in samples])
    lo = np.nanmin(matrix, axis=0)
    hi = np.nanmax(matrix, axis=0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    normalized = (matrix - lo) / scale

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 180})
    fig, ax = plt.subplots(figsize=(16.5, 6.4), constrained_layout=True)
    ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if normalized[row, col] > 0.58 else "#111111"
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=5.8, color=color)
    ax.set_xticks(np.arange(len(FEATURE_NAMES)))
    ax.set_xticklabels(
        [f"x[{i}]\n{name}" for i, name in enumerate(FEATURE_NAMES)],
        rotation=55,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(np.arange(len(samples)))
    ax.set_yticklabels(
        [f"{i + 1:02d} {sample['scene']} / t={sample['step']}" for i, sample in enumerate(samples)],
        fontsize=7.5,
    )
    ax.set_title(
        "Exact 19-D Observations for the Twelve Habitat-Sim Depth Frames",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Column colors are normalized independently for readability; annotations are raw values.")
    for boundary in (5.5, 9.5, 12.5):
        ax.axvline(boundary, color="#D94801", linewidth=1.4)
    png_path = output_dir / "habitat_real_depth_12_feature_matrix.png"
    pdf_path = output_dir / "habitat_real_depth_12_feature_matrix.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _save_samples(samples: list[dict[str, Any]], output_dir: Path) -> None:
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    # This directory is owned by this deterministic generator.  Remove only its
    # previous sample artifacts so reruns still contain exactly twelve records.
    for stale in sample_dir.glob("sample_*"):
        if stale.is_file() and stale.suffix.lower() in {".npz", ".png"}:
            stale.unlink()
    rows: list[dict[str, Any]] = []
    for number, sample in enumerate(samples, start=1):
        stem = f"sample_{number:02d}_{sample['scene']}_ep_{sample['episode_id']}_step_{sample['step']}"
        npz_path = sample_dir / f"{stem}.npz"
        np.savez_compressed(
            npz_path,
            depth_normalized=sample["depth_normalized"],
            depth_m=sample["depth_m"],
            rgb=sample["rgb"],
            features=sample["features"],
            feature_names=np.asarray(FEATURE_NAMES),
            agent_position=sample["agent_position"],
        )
        if sample["rgb"].size:
            Image.fromarray(sample["rgb"]).save(sample_dir / f"{stem}_rgb.png")
        plt.imsave(
            sample_dir / f"{stem}_depth.png",
            sample["depth_m"],
            cmap="turbo_r",
            vmin=0.0,
            vmax=5.0,
        )
        row = {
            "sample": number,
            "scene": sample["scene"],
            "scene_id": sample["scene_id"],
            "episode_id": sample["episode_id"],
            "source_episode_index": sample["source_episode_index"],
            "step": sample["step"],
            "trajectory_steps": sample["trajectory_steps"],
            "controller_mode": sample["mode"],
            "source_dataset": sample["source_dataset"],
            "npz": str(npz_path),
        }
        row.update({name: float(value) for name, value in zip(FEATURE_NAMES, sample["features"])})
        rows.append(row)
    with (output_dir / "habitat_real_depth_12_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_and_plot(args: argparse.Namespace) -> list[Path]:
    dataset_path = args.dataset.resolve()
    scene_root = args.scene_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = _load_dataset(dataset_path)
    selected = _select_one_per_scene(source, args.num_samples)
    patched, source_indices = _patched_dataset(source, selected, scene_root)
    samples: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="habitat_depth_gallery_") as temp_dir:
        patched_path = Path(temp_dir) / "episodes.json.gz"
        with gzip.open(patched_path, "wt", encoding="utf-8") as handle:
            json.dump(patched, handle)
        with make_env(str(patched_path), args.split) as env:
            for patched_index, source_index in enumerate(source_indices):
                episode = source["episodes"][source_index]
                sample = _collect_episode(
                    env, patched_index, str(patched_path), args.max_steps
                )
                sample.update(
                    {
                        "scene": _scene_name(str(episode["scene_id"])),
                        "scene_id": str(episode["scene_id"]),
                        "episode_id": str(episode["episode_id"]),
                        "source_episode_index": int(source_index),
                        "source_dataset": str(dataset_path),
                    }
                )
                samples.append(sample)
                print(
                    f"[{len(samples):02d}/{args.num_samples:02d}] "
                    f"scene={sample['scene']} ep={sample['episode_id']} "
                    f"step={sample['step']} margin={sample['features'][9]:.3f}m"
                )

    _save_samples(samples, output_dir)
    gallery_paths = _plot_gallery(samples, output_dir)
    rgb_gallery_paths = _plot_rgb_gallery(samples, output_dir)
    matrix_paths = _plot_feature_matrix(samples, output_dir)
    manifest = {
        "kind": "real_habitat_sim_depth_gallery",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "scene_root": str(scene_root),
        "split": args.split,
        "controller": "GeometryFSMController",
        "selection": (
            "one minimum-annotated-margin episode per HM3D scene; quality-aware "
            "trajectory frame with >=5 valid ROIs and estimated margin <=0.8 m"
        ),
        "max_steps": args.max_steps,
        "num_samples": len(samples),
        "depth_sensor": {
            "observation_encoding": "Habitat normalized depth",
            "metric_conversion_m": "depth_normalized * (10.0 - 0.0) + 0.0",
            "feature_clip_m": 5.0,
            "roi_statistic": "10th percentile of valid pixels",
        },
        "feature_names": list(FEATURE_NAMES),
        "outputs": [
            str(path)
            for path in (*gallery_paths, *rgb_gallery_paths, *matrix_paths)
        ],
    }
    manifest_path = output_dir / "habitat_real_depth_12_provenance.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return [*gallery_paths, *rgb_gallery_paths, *matrix_paths, manifest_path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.num_samples != 12:
        raise ValueError("This paper figure is fixed to exactly 12 samples")
    for output in collect_and_plot(args):
        print(output)


if __name__ == "__main__":
    main()
