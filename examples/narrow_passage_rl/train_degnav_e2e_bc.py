#!/usr/bin/env python3
"""Train the visual DEGNAV selector from Habitat teacher trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from narrow_passage.selector.contract import MODE_NAMES
from narrow_passage.selector.visual_model import VisualFourModeGRUPolicy


DEFAULT_DATA_ROOT = Path(
    "/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/pointnav_assets/"
    "mp3d_narrow_v1/e2e_teacher"
)
DEFAULT_CHECKPOINT_ROOT = Path(
    "/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/pointnav_assets/checkpoints"
)


class EpisodeChunkDataset(Dataset):
    def __init__(
        self,
        directory: Path,
        input_type: str,
        sequence_length: int,
        stride: int,
        with_memory: bool,
    ) -> None:
        self.directory = directory
        self.input_type = input_type
        self.sequence_length = int(sequence_length)
        self.with_memory = bool(with_memory)
        self.samples: list[tuple[Path, int, int]] = []
        self.mode_counts: Counter[int] = Counter()
        for path in sorted(directory.glob("*.npz")):
            with np.load(path, allow_pickle=False) as episode:
                modes = np.asarray(episode["mode"], dtype=np.int64)
                if input_type == "rgbd" and "rgb" not in episode:
                    raise ValueError(f"RGB missing from {path}")
            self.mode_counts.update(modes.tolist())
            for start in range(0, len(modes), int(stride)):
                end = min(start + self.sequence_length, len(modes))
                self.samples.append((path, start, end))
                if end == len(modes):
                    break
        if not self.samples:
            raise ValueError(f"no .npz demonstrations found in {directory}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, start, end = self.samples[index]
        length = end - start
        with np.load(path, allow_pickle=False) as episode:
            depth = np.asarray(episode["depth"][start:end], dtype=np.float32)
            pointgoal = np.asarray(episode["pointgoal"][start:end], dtype=np.float32)
            action_outcome = np.asarray(
                episode["action_outcome"][start:end], dtype=np.float32
            )
            memory = (
                np.asarray(episode["memory_context"][start:end], dtype=np.float32)
                if self.with_memory
                else None
            )
            modes = np.asarray(episode["mode"][start:end], dtype=np.int64)
            rgb = (
                np.asarray(episode["rgb"][start:end], dtype=np.uint8)
                if self.input_type == "rgbd"
                else None
            )

        def pad(value: np.ndarray, shape: tuple[int, ...], fill=0):
            output = np.full(shape, fill, dtype=value.dtype)
            output[:length] = value
            return output

        size = depth.shape[-1]
        output = {
            "depth": torch.from_numpy(
                pad(depth, (self.sequence_length, size, size))
            ).unsqueeze(-1)
            / 65535.0,
            "pointgoal": torch.from_numpy(
                pad(pointgoal, (self.sequence_length, 2))
            ),
            "action_outcome": torch.from_numpy(
                pad(action_outcome, (self.sequence_length, action_outcome.shape[-1]))
            ),
            "mode": torch.from_numpy(
                pad(modes, (self.sequence_length,), fill=-100)
            ),
            "valid": torch.arange(self.sequence_length) < length,
            "episode_starts": torch.zeros(self.sequence_length),
        }
        if memory is not None:
            output["memory_context"] = torch.from_numpy(
                pad(memory, (self.sequence_length, memory.shape[-1]))
            )
        output["episode_starts"][0] = 1.0
        if rgb is not None:
            output["rgb"] = torch.from_numpy(
                pad(rgb, (self.sequence_length, size, size, 3))
            )
        return output


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_released_visual_backbone(
    policy: VisualFourModeGRUPolicy, checkpoint: Path
) -> int:
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    prefix = "actor_critic.net.visual_encoder.backbone."
    state = {
        key[len(prefix) :]: value
        for key, value in data["state_dict"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise ValueError(f"no released visual backbone found in {checkpoint}")
    policy.visual_backbone.load_state_dict(state, strict=True)
    return len(state)


def class_weights(
    counts: Counter[int], device: torch.device, power: float
) -> torch.Tensor:
    values = np.asarray([max(1, counts[index]) for index in range(4)], dtype=np.float64)
    weights = (values.sum() / values) ** float(power)
    weights /= weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    policy: VisualFourModeGRUPolicy,
    loader: DataLoader,
    device: torch.device,
    weights: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    with_memory: bool,
) -> dict[str, Any]:
    training = optimizer is not None
    policy.train(training)
    confusion = torch.zeros(4, 4, dtype=torch.long)
    total_loss = 0.0
    batches = 0
    for batch in loader:
        # DataLoader is [B,T,...]; recurrent policy contract is [T,B,...].
        move = lambda value: value.transpose(0, 1).to(device, non_blocking=True)
        depth = move(batch["depth"])
        pointgoal = move(batch["pointgoal"])
        action_outcome = move(batch["action_outcome"])
        memory = move(batch["memory_context"]) if with_memory else None
        starts = move(batch["episode_starts"])
        targets = move(batch["mode"])
        rgb = move(batch["rgb"]) if "rgb" in batch else None
        hidden = policy.initial_hidden(depth.shape[1], device)
        with torch.set_grad_enabled(training):
            logits, _values, _hidden = policy.forward_sequence(
                depth,
                pointgoal,
                action_outcome,
                hidden,
                rgb=rgb,
                memory_context=memory,
                episode_starts=starts,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, 4),
                targets.reshape(-1),
                weight=weights,
                ignore_index=-100,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.item())
        batches += 1
        valid = targets >= 0
        predictions = logits.argmax(dim=-1)
        for target, prediction in zip(
            targets[valid].detach().cpu(), predictions[valid].detach().cpu()
        ):
            confusion[int(target), int(prediction)] += 1

    support = confusion.sum(dim=1)
    recalls = confusion.diag().float() / support.clamp_min(1)
    accuracy = float(confusion.diag().sum() / confusion.sum().clamp_min(1))
    return {
        "loss": total_loss / max(1, batches),
        "accuracy": accuracy,
        "macro_recall": float(recalls.mean()),
        "recall_by_mode": {
            name: float(recalls[index]) for index, name in enumerate(MODE_NAMES)
        },
        "support_by_mode": {
            name: int(support[index]) for index, name in enumerate(MODE_NAMES)
        },
        "confusion": confusion.tolist(),
    }


def save_checkpoint(
    path: Path,
    policy: VisualFourModeGRUPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "model_config": {
                "input_type": policy.input_type,
                "hidden_size": policy.hidden_size,
                "memory_dim": policy.memory_dim,
                "privileged_critic": bool(policy.privileged_dim),
                "baseplanes": config["baseplanes"],
            },
            "experiment_config": config,
            "validation": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--input-type", choices=("depth", "rgbd"), default="depth")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--baseplanes", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--class-balance-power",
        type=float,
        default=1.0,
        help=(
            "Exponent applied to inverse class frequency. 1.0 gives equal "
            "total loss mass to each mode; 0.5 is the weaker sqrt heuristic."
        ),
    )
    parser.add_argument("--with-memory", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.init_checkpoint is None:
        checkpoint_name = (
            "mp3d-depth-best.pth"
            if args.input_type == "depth"
            else "mp3d-rgbd-best.pth"
        )
        args.init_checkpoint = DEFAULT_CHECKPOINT_ROOT / checkpoint_name
    if not args.init_checkpoint.is_file():
        parser.error(f"missing PointNav initialization: {args.init_checkpoint}")

    train_data = EpisodeChunkDataset(
        args.data_root / args.input_type / "train",
        args.input_type,
        args.sequence_length,
        args.stride,
        args.with_memory,
    )
    val_data = EpisodeChunkDataset(
        args.data_root / args.input_type / "val",
        args.input_type,
        args.sequence_length,
        args.sequence_length,
        args.with_memory,
    )
    for split_name, dataset in (("train", train_data), ("val", val_data)):
        missing = [
            MODE_NAMES[index]
            for index in range(4)
            if dataset.mode_counts[index] <= 0
        ]
        if missing:
            parser.error(
                f"{split_name} demonstrations contain no labels for {missing}; "
                "refusing to train an incomplete four-mode policy"
            )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    policy = VisualFourModeGRUPolicy(
        input_type=args.input_type,
        hidden_size=args.hidden_size,
        memory_dim=4 if args.with_memory else 0,
        baseplanes=args.baseplanes,
    )
    loaded_tensors = load_released_visual_backbone(policy, args.init_checkpoint)
    policy.to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if not 0.0 <= args.class_balance_power <= 1.0:
        parser.error("--class-balance-power must be in [0, 1]")
    weights = class_weights(
        train_data.mode_counts, device, args.class_balance_power
    )
    config = {
        **vars(args),
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "init_checkpoint": str(args.init_checkpoint),
        "baseplanes": args.baseplanes,
        "released_backbone_tensors_loaded": loaded_tensors,
        "train_mode_counts": dict(train_data.mode_counts),
        "val_mode_counts": dict(val_data.mode_counts),
        "class_weights": weights.detach().cpu().tolist(),
        "actor_uses_19d": False,
        "actor_uses_handdesigned_task_memory": bool(args.with_memory),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            policy, train_loader, device, weights, optimizer, args.with_memory
        )
        with torch.inference_mode():
            val_metrics = run_epoch(
                policy, val_loader, device, weights, None, args.with_memory
            )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_recall": train_metrics["macro_recall"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_recall": val_metrics["macro_recall"],
            **{
                f"val_recall_{name.lower()}": val_metrics["recall_by_mode"][name]
                for name in MODE_NAMES
            },
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        save_checkpoint(
            args.output_dir / "last.pt", policy, optimizer, epoch, config, val_metrics
        )
        if val_metrics["macro_recall"] > best_score:
            best_score = val_metrics["macro_recall"]
            save_checkpoint(
                args.output_dir / "best.pt",
                policy,
                optimizer,
                epoch,
                config,
                val_metrics,
            )

    with (args.output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (args.output_dir / "validation_best.json").write_text(
        json.dumps(
            torch.load(
                args.output_dir / "best.pt", map_location="cpu", weights_only=False
            )["validation"],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[write] {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
