#!/usr/bin/env python3
"""Dataset-free NaVILA runtime and qualitative navigation diagnostics.

This script is intentionally not a replacement for R2R/RxR evaluation. It
checks model artifacts, loads NaVILA in 4-bit mode, runs deterministic
multimodal inference on repository demo frames and synthetic fixtures, and
writes reproducible latency/action-format diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


ACTION_PATTERNS = (
    ("stop", re.compile(r"\bstop\b", re.IGNORECASE)),
    ("move_forward", re.compile(r"\b(?:is\s+)?move forward\b", re.IGNORECASE)),
    ("turn_left", re.compile(r"\b(?:is\s+)?turn left\b", re.IGNORECASE)),
    ("turn_right", re.compile(r"\b(?:is\s+)?turn right\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class Case:
    case_id: str
    source: str
    instruction: str
    images: list[Image.Image]
    heuristic_expected_action: str = ""
    repeat_of: str = ""


@dataclass
class Prediction:
    case_id: str
    source: str
    instruction: str
    frame_count: int
    response: str
    parsed_action: str
    action_value: float
    valid_action: bool
    heuristic_expected_action: str
    heuristic_match: Optional[bool]
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    peak_gpu_memory_mb: float
    repeat_of: str


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def sha256_small(path: Path, max_bytes: int = 2 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = max_bytes
        while remaining > 0:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def validate_model_artifacts(model_path: Path) -> dict[str, Any]:
    expected = [
        "config.json",
        "llm/config.json",
        "llm/model.safetensors.index.json",
        "llm/model-00001-of-00004.safetensors",
        "llm/model-00002-of-00004.safetensors",
        "llm/model-00003-of-00004.safetensors",
        "llm/model-00004-of-00004.safetensors",
        "vision_tower/model.safetensors",
        "mm_projector/model.safetensors",
    ]
    missing = [name for name in expected if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")

    index_path = model_path / "llm/model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    tensor_counts: dict[str, int] = {}
    for rel_path in [
        *(Path("llm") / shard for shard in shards),
        Path("vision_tower/model.safetensors"),
        Path("mm_projector/model.safetensors"),
    ]:
        with safe_open(str(model_path / rel_path), framework="pt", device="cpu") as handle:
            tensor_counts[str(rel_path)] = len(handle.keys())

    return {
        "status": "pass",
        "model_size_bytes": sum(
            path.stat().st_size for path in model_path.rglob("*") if path.is_file()
        ),
        "index_tensor_count": len(index["weight_map"]),
        "llm_shards": shards,
        "tensor_counts": tensor_counts,
        "config_prefix_sha256": sha256_small(model_path / "config.json"),
    }


def patch_transformers_4bit_compat() -> bool:
    """Bridge NaVILA's replacement modeling_utils to Transformers 4.37."""
    import inspect
    import transformers.integrations as integrations

    original = integrations.set_module_quantized_tensor_to_device
    if "fp16_statistics" in inspect.signature(original).parameters:
        return False

    def compatible(
        module: torch.nn.Module,
        tensor_name: str,
        device: Any,
        value: Any = None,
        fp16_statistics: Any = None,
        quantized_stats: Any = None,
        **_: Any,
    ) -> Any:
        stats = quantized_stats if quantized_stats is not None else fp16_statistics
        return original(
            module,
            tensor_name,
            device,
            value=value,
            quantized_stats=stats,
        )

    integrations.set_module_quantized_tensor_to_device = compatible
    return True


def parse_action(response: str) -> tuple[str, float]:
    for action, pattern in ACTION_PATTERNS:
        if not pattern.search(response):
            continue
        if action == "move_forward":
            match = re.search(r"move forward\s+(\d+(?:\.\d+)?)\s*cm", response, re.I)
        elif action in {"turn_left", "turn_right"}:
            match = re.search(
                rf"{action.replace('_', ' ')}\s+(\d+(?:\.\d+)?)\s*degree",
                response,
                re.I,
            )
        else:
            match = None
        return action, float(match.group(1)) if match else math.nan
    return "unparsed", math.nan


def sample_indices(start: int, stop: int, count: int) -> list[int]:
    if stop <= start:
        return [start] * count
    return [int(round(value)) for value in np.linspace(start, stop, count)]


def load_demo_window(
    gif_path: Path,
    start_fraction: float,
    stop_fraction: float,
    frame_count: int,
) -> list[Image.Image]:
    gif = Image.open(gif_path)
    total = getattr(gif, "n_frames", 1)
    start = int(round((total - 1) * start_fraction))
    stop = int(round((total - 1) * stop_fraction))
    images: list[Image.Image] = []
    for index in sample_indices(start, stop, frame_count):
        gif.seek(index)
        frame = gif.convert("RGB")
        side = min(frame.height, frame.width // 4)
        images.append(frame.crop((0, 0, side, side)))
    return images


def synthetic_corridor_frames(
    frame_count: int,
    blocked: bool,
    turn: str = "",
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    width = height = 384
    for index in range(frame_count):
        image = Image.new("RGB", (width, height), (205, 213, 218))
        draw = ImageDraw.Draw(image)
        horizon = 125
        center = width // 2 + (index - frame_count // 2) * 2
        draw.polygon(
            [(0, height), (center - 58, horizon), (center + 58, horizon), (width, height)],
            fill=(150, 139, 121),
        )
        draw.polygon(
            [(0, 0), (center - 58, horizon), (0, height)],
            fill=(224, 220, 207),
        )
        draw.polygon(
            [(width, 0), (center + 58, horizon), (width, height)],
            fill=(196, 202, 206),
        )
        draw.rectangle(
            (center - 50, horizon - 75, center + 50, horizon + 45),
            outline=(58, 65, 68),
            width=8,
        )
        if blocked:
            draw.rectangle(
                (center - 68, horizon + 26, center + 68, horizon + 115),
                fill=(120, 55, 42),
                outline=(70, 35, 28),
                width=5,
            )
        if turn == "left":
            draw.polygon(
                [(0, 150), (center - 55, horizon), (center - 70, 260), (0, 300)],
                fill=(106, 119, 128),
            )
        elif turn == "right":
            draw.polygon(
                [(width, 150), (center + 55, horizon), (center + 70, 260), (width, 300)],
                fill=(106, 119, 128),
            )
        frames.append(image)
    return frames


def build_cases(sample_gif: Path, frame_count: int) -> list[Case]:
    demo_instruction = (
        "Walk through the small room with a lounge chair and dresser. Turn the "
        "corner and walk past the sink. Walk into the kitchen through the door "
        "on the other end. Stop near the long wooden dining table."
    )
    middle = load_demo_window(sample_gif, 0.32, 0.68, frame_count)
    return [
        Case(
            "demo_early",
            "repository_demo",
            demo_instruction,
            load_demo_window(sample_gif, 0.0, 0.34, frame_count),
        ),
        Case(
            "demo_middle",
            "repository_demo",
            demo_instruction,
            middle,
        ),
        Case(
            "demo_late",
            "repository_demo",
            demo_instruction,
            load_demo_window(sample_gif, 0.66, 1.0, frame_count),
        ),
        Case(
            "demo_middle_repeat",
            "repository_demo",
            demo_instruction,
            [image.copy() for image in middle],
            repeat_of="demo_middle",
        ),
        Case(
            "counterfactual_stop",
            "repository_demo_counterfactual",
            "The navigation task is complete. Stop now.",
            [image.copy() for image in middle],
            heuristic_expected_action="stop",
        ),
        Case(
            "synthetic_open_corridor",
            "synthetic_fixture",
            "Move forward through the open doorway.",
            synthetic_corridor_frames(frame_count, blocked=False),
            heuristic_expected_action="move_forward",
        ),
        Case(
            "synthetic_blocked_corridor",
            "synthetic_fixture",
            "Approach the doorway, but do not move through the blocking obstacle.",
            synthetic_corridor_frames(frame_count, blocked=True),
        ),
    ]


def build_prompt(instruction: str, frame_count: int) -> tuple[str, str]:
    from llava.conversation import SeparatorStyle, conv_templates

    historical = "<image>\n" * (frame_count - 1)
    question = (
        "Imagine you are a robot programmed for navigation tasks. You have "
        f"been given a video of historical observations {historical}, and "
        f'current observation <image>\n. Your assigned task is: "{instruction}" '
        "Analyze this series of images to decide your next action, which could "
        "be turning left or right by a specific degree, moving forward a "
        "certain distance, or stop if the task is completed."
    )
    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    stop_string = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    return conv.get_prompt(), stop_string


def run_prediction(
    case: Case,
    tokenizer: Any,
    model: Any,
    image_processor: Any,
    max_new_tokens: int,
) -> Prediction:
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        process_images,
        tokenizer_image_token,
    )

    prompt, stop_string = build_prompt(case.instruction, len(case.images))
    images_tensor = process_images(case.images, image_processor, model.config).to(
        model.device, dtype=torch.float16
    )
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).to(model.device)
    stopping = KeywordsStoppingCriteria([stop_string], tokenizer, input_ids)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping],
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if (
        output_ids.shape[1] >= input_ids.shape[1]
        and torch.equal(output_ids[:, : input_ids.shape[1]], input_ids)
    ):
        decoded_ids = output_ids[:, input_ids.shape[1] :]
    else:
        decoded_ids = output_ids
    response = tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)[0].strip()
    if response.endswith(stop_string):
        response = response[: -len(stop_string)].strip()
    action, value = parse_action(response)
    expected = case.heuristic_expected_action
    return Prediction(
        case_id=case.case_id,
        source=case.source,
        instruction=case.instruction,
        frame_count=len(case.images),
        response=response,
        parsed_action=action,
        action_value=value,
        valid_action=action != "unparsed",
        heuristic_expected_action=expected,
        heuristic_match=(action == expected) if expected else None,
        latency_seconds=elapsed,
        input_tokens=int(input_ids.numel()),
        output_tokens=int(decoded_ids.numel()),
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / (1024**2),
        repeat_of=case.repeat_of,
    )


def save_fixtures(cases: Iterable[Case], output_dir: Path) -> None:
    fixture_root = output_dir / "fixtures"
    for case in cases:
        case_dir = fixture_root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(case.images):
            image.save(case_dir / f"frame_{index:02d}.png")


def write_csv(predictions: list[Prediction], path: Path) -> None:
    rows = [asdict(prediction) for prediction in predictions]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile_95(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def make_summary(
    predictions: list[Prediction],
    load_seconds: float,
    model_memory_mb: float,
) -> dict[str, Any]:
    latencies = [prediction.latency_seconds for prediction in predictions]
    action_counts = Counter(prediction.parsed_action for prediction in predictions)
    repeat_matches = []
    by_id = {prediction.case_id: prediction for prediction in predictions}
    for prediction in predictions:
        if prediction.repeat_of:
            repeat_matches.append(
                prediction.response == by_id[prediction.repeat_of].response
            )
    heuristic_rows = [
        prediction
        for prediction in predictions
        if prediction.heuristic_expected_action
    ]
    return {
        "status": "pass",
        "case_count": len(predictions),
        "successful_inference_count": sum(bool(item.response) for item in predictions),
        "nonempty_response_rate": mean(bool(item.response) for item in predictions),
        "action_parse_rate": mean(item.valid_action for item in predictions),
        "heuristic_prompt_match_rate": (
            mean(bool(item.heuristic_match) for item in heuristic_rows)
            if heuristic_rows
            else None
        ),
        "deterministic_repeat_match_rate": (
            mean(repeat_matches) if repeat_matches else None
        ),
        "mean_latency_seconds": mean(latencies),
        "median_latency_seconds": median(latencies),
        "p95_latency_seconds": percentile_95(latencies),
        "mean_output_tokens": mean(item.output_tokens for item in predictions),
        "max_peak_gpu_memory_mb": max(
            item.peak_gpu_memory_mb for item in predictions
        ),
        "model_load_seconds": load_seconds,
        "model_memory_after_load_mb": model_memory_mb,
        "action_distribution": dict(sorted(action_counts.items())),
    }


def markdown_report(
    metadata: dict[str, Any],
    predictions: list[Prediction],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Dataset-Free NaVILA VLN Diagnostic",
        "",
        "## Scope",
        "",
        (
            "This is a model/runtime and qualitative action-interface diagnostic. "
            "It does not use R2R, RxR, or MP3D episodes and therefore does not "
            "report SR, SPL, NE, nDTW, or benchmark generalization."
        ),
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "| :--- | ---: |",
        f"| Model artifact validation | {metadata['artifact_validation']['status'].upper()} |",
        f"| 4-bit model load | {metadata['model_load']['status'].upper()} |",
        f"| Inference cases | {summary['case_count']} |",
        f"| Non-empty response rate | {100 * summary['nonempty_response_rate']:.1f}% |",
        f"| Action parse rate | {100 * summary['action_parse_rate']:.1f}% |",
        (
            "| Deterministic repeat match | "
            f"{100 * summary['deterministic_repeat_match_rate']:.1f}% |"
        ),
        f"| Mean latency | {summary['mean_latency_seconds']:.3f} s |",
        f"| P95 latency | {summary['p95_latency_seconds']:.3f} s |",
        f"| Model load time | {summary['model_load_seconds']:.3f} s |",
        f"| Model GPU memory after load | {summary['model_memory_after_load_mb']:.1f} MiB |",
        f"| Peak GPU memory | {summary['max_peak_gpu_memory_mb']:.1f} MiB |",
        "",
        "## Predictions",
        "",
        "| Case | Source | Parsed action | Value | Latency (s) | Response |",
        "| :--- | :--- | :--- | ---: | ---: | :--- |",
    ]
    for item in predictions:
        value = "" if math.isnan(item.action_value) else f"{item.action_value:.1f}"
        response = item.response.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item.case_id} | {item.source} | {item.parsed_action} | "
            f"{value} | {item.latency_seconds:.3f} | {response} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- PASS means the downloaded checkpoint is structurally complete, "
                "loads on the available GPU with 4-bit quantization, accepts an "
                "eight-frame visual history and language instruction, and returns "
                "machine-parseable navigation actions."
            ),
            (
                "- Synthetic prompt matches are interface sanity checks only; they "
                "are not navigation accuracy measurements."
            ),
            (
                "- Formal VLN conclusions require paired simulator episodes and "
                "standard R2R/RxR metrics after MP3D is available."
            ),
            "",
            "## Provenance",
            "",
            f"- Model: `{metadata['model_path']}`",
            f"- NaVILA commit: `{metadata['navila_commit']}`",
            f"- Quantization: `{metadata['quantization']}`",
            f"- GPU: `{metadata['runtime']['gpu_name']}`",
            f"- Generated: `{metadata['generated_at']}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    default_store = Path(
        os.environ.get(
            "VLN_STORE",
            "/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln",
        )
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=default_store / "models/navila-llama3-8b-8f",
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
        "--output-dir",
        type=Path,
        default=Path(
            "examples/narrow_passage_rl/results/vln_dataset_free"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NaVILA diagnostic.")
    if not args.sample_gif.is_file():
        raise FileNotFoundError(args.sample_gif)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    cases = build_cases(args.sample_gif, frame_count)
    save_fixtures(cases, args.output_dir)
    predictions = [
        run_prediction(
            case,
            tokenizer,
            model,
            image_processor,
            args.max_new_tokens,
        )
        for case in cases
    ]
    summary = make_summary(predictions, load_seconds, model_memory_mb)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "scope": "dataset-free qualitative VLN diagnostic",
        "model_path": str(args.model.resolve()),
        "sample_gif": str(args.sample_gif.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "navila_commit": git_commit(args.navila_root),
        "quantization": "bitsandbytes NF4 4-bit",
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "frame_count": frame_count,
        "context_length": context_length,
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
            "gpu_total_memory_mb": torch.cuda.get_device_properties(0).total_memory
            / (1024**2),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "bitsandbytes": package_version("bitsandbytes"),
            "flash_attn": package_version("flash-attn"),
            "vila": package_version("vila"),
        },
        "summary": summary,
        "limitations": [
            "No simulator episodes or navigation trajectories are evaluated.",
            "Repository demo frames and synthetic fixtures have no benchmark labels.",
            "SR, SPL, NE, nDTW, and generalization cannot be inferred.",
        ],
    }

    write_csv(predictions, args.output_dir / "predictions.csv")
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "predictions": [asdict(item) for item in predictions],
            },
            indent=2,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        markdown_report(metadata, predictions, summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"[write] {args.output_dir / 'predictions.csv'}")
    print(f"[write] {args.output_dir / 'results.json'}")
    print(f"[write] {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
