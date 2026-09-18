#!/usr/bin/env python3
"""Class-balanced behavior cloning for the 19-D four-mode GRU selector."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))

from narrow_passage.selector.contract import MODE_NAMES
from narrow_passage.selector.model import FourModeGRUPolicy


class ModeSequenceDataset(Dataset):
    def __init__(self, data, split: str, history: int) -> None:
        self.obs = data["obs_19d"].astype(np.float32)
        self.memory = data["memory_context"].astype(np.float32)
        self.labels = data["rule_mode"].astype(np.int64)
        self.feasible = data["feasibility_label"].astype(np.int8)
        self.episode = data["episode_index"].astype(np.int64)
        self.indices = np.flatnonzero(data["split"].astype(str) == split)
        self.history = int(history)
        self._episode_ranges: dict[int, tuple[int, int]] = {}
        for episode_id in np.unique(self.episode):
            members = np.flatnonzero(self.episode == episode_id)
            self._episode_ranges[int(episode_id)] = (int(members[0]), int(members[-1]) + 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        episode_start, _ = self._episode_ranges[int(self.episode[index])]
        start = max(episode_start, index - self.history + 1)
        obs = self.obs[start : index + 1]
        memory = self.memory[start : index + 1]
        padding = self.history - len(obs)
        if padding:
            obs = np.concatenate([np.repeat(obs[:1], padding, axis=0), obs], axis=0)
            memory = np.concatenate(
                [np.repeat(memory[:1], padding, axis=0), memory], axis=0
            )
        episode_starts = np.zeros(self.history, dtype=np.float32)
        episode_starts[0] = 1.0
        return (
            torch.from_numpy(obs),
            torch.from_numpy(memory),
            torch.from_numpy(episode_starts),
            torch.tensor(self.labels[index], dtype=torch.long),
            torch.tensor(self.feasible[index], dtype=torch.int8),
        )


def confusion_metrics(labels: np.ndarray, predictions: np.ndarray, feasible: np.ndarray) -> dict:
    matrix = np.zeros((4, 4), dtype=np.int64)
    for target, predicted in zip(labels, predictions):
        matrix[int(target), int(predicted)] += 1
    per_mode = {}
    for index, name in enumerate(MODE_NAMES):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_mode[name] = {
            "support": support,
            "predicted": predicted_count,
            "true_positive": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    feasible_mask = feasible == 1
    infeasible_mask = feasible == 0
    reject_predictions = predictions == 3
    correct = int((labels == predictions).sum())
    return {
        "samples": int(len(labels)),
        "correct": correct,
        "accuracy": correct / len(labels) if len(labels) else 0.0,
        "confusion_matrix_rows_target_cols_prediction": matrix.tolist(),
        "per_mode": per_mode,
        "reject_prediction_on_feasible": {
            "numerator": int((reject_predictions & feasible_mask).sum()),
            "denominator": int(feasible_mask.sum()),
            "rate": float(reject_predictions[feasible_mask].mean()) if feasible_mask.any() else 0.0,
        },
        "reject_prediction_on_infeasible": {
            "numerator": int((reject_predictions & infeasible_mask).sum()),
            "denominator": int(infeasible_mask.sum()),
            "rate": float(reject_predictions[infeasible_mask].mean()) if infeasible_mask.any() else 0.0,
        },
    }


@torch.no_grad()
def evaluate(policy, loader, device, with_memory: bool) -> tuple[float, dict]:
    policy.eval()
    losses = []
    all_labels, all_predictions, all_feasible = [], [], []
    for obs, memory, starts, labels, feasible in loader:
        obs = obs.transpose(0, 1).to(device)
        memory = memory.transpose(0, 1).to(device)
        starts = starts.transpose(0, 1).to(device)
        hidden = policy.initial_hidden(obs.shape[1], device)
        logits, _, _ = policy.forward_sequence(
            obs,
            hidden,
            memory if with_memory else None,
            starts,
        )
        final_logits = logits[-1]
        losses.append(float(F.cross_entropy(final_logits, labels.to(device)).item()))
        all_labels.append(labels.numpy())
        all_predictions.append(final_logits.argmax(-1).cpu().numpy())
        all_feasible.append(feasible.numpy())
    labels_np = np.concatenate(all_labels)
    predictions_np = np.concatenate(all_predictions)
    feasible_np = np.concatenate(all_feasible)
    return float(np.mean(losses)), confusion_metrics(labels_np, predictions_np, feasible_np)


def write_metrics_markdown(metrics: dict, path: Path) -> None:
    lines = [
        "# Four-mode BC validation",
        "",
        f"Accuracy: {metrics['correct']}/{metrics['samples']} = {metrics['accuracy']:.4f}",
        "",
        "| Mode | Support | Predicted | Precision | Recall | F1 |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for name in MODE_NAMES:
        row = metrics["per_mode"][name]
        lines.append(
            f"| {name} | {row['support']} | {row['predicted']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    for title, key in (
        ("False Reject on feasible labels", "reject_prediction_on_feasible"),
        ("Reject on infeasible labels", "reject_prediction_on_infeasible"),
    ):
        row = metrics[key]
        lines.extend(["", f"{title}: {row['numerator']}/{row['denominator']} = {row['rate']:.4f}"])
    lines.extend(["", "Confusion matrix (rows=target, columns=prediction):", "", "| target | COMMIT | EXPLORE | RECOVER | REJECT |", "|:---|---:|---:|---:|---:|"])
    for name, values in zip(MODE_NAMES, metrics["confusion_matrix_rows_target_cols_prediction"]):
        lines.append(f"| {name} | " + " | ".join(str(v) for v in values) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--with-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    data = np.load(args.dataset, allow_pickle=False)
    train = ModeSequenceDataset(data, "train", args.history)
    validation = ModeSequenceDataset(data, "validation", args.history)
    if not len(train) or not len(validation):
        raise ValueError("dataset must contain non-empty train and validation episode splits")
    counts = np.bincount(train.labels[train.indices], minlength=4)
    if np.any(counts == 0):
        raise ValueError(f"training split lacks one or more modes: {dict(zip(MODE_NAMES, counts.tolist()))}")
    sample_weights = 1.0 / counts[train.labels[train.indices]]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False)
    policy = FourModeGRUPolicy(
        hidden_size=args.hidden_size,
        memory_dim=4 if args.with_memory else 0,
    ).to(device)
    train_obs = torch.from_numpy(data["obs_19d"][train.indices]).to(device)
    policy.update_normalization(train_obs)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_rows = []
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = args.output_dir / "bc_best.pt"

    for epoch in range(1, args.epochs + 1):
        policy.train()
        train_losses = []
        for obs, memory, starts, labels, _feasible in train_loader:
            obs = obs.transpose(0, 1).to(device)
            memory = memory.transpose(0, 1).to(device)
            starts = starts.transpose(0, 1).to(device)
            hidden = policy.initial_hidden(obs.shape[1], device)
            logits, _, _ = policy.forward_sequence(
                obs, hidden, memory if args.with_memory else None, starts
            )
            loss = F.cross_entropy(logits[-1], labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            train_losses.append(float(loss.item()))
        validation_loss, metrics = evaluate(policy, validation_loader, device, args.with_memory)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
            "validation_accuracy": metrics["accuracy"],
        }
        history_rows.append(row)
        print(json.dumps(row, sort_keys=True))
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": policy.state_dict(),
                    "model_config": {
                        "hidden_size": args.hidden_size,
                        "memory_dim": 4 if args.with_memory else 0,
                        "privileged_dim": 0,
                    },
                    "training_config": vars(args),
                    "best_epoch": best_epoch,
                    "validation_loss": best_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["model"])
    _, metrics = evaluate(policy, validation_loader, device, args.with_memory)
    metrics.update(
        {
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "training_mode_counts": dict(zip(MODE_NAMES, counts.tolist())),
            "dataset": str(args.dataset),
            "split_unit": "whole episode",
        }
    )
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    write_metrics_markdown(metrics, args.output_dir / "validation_metrics.md")
    with (args.output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)
    with (args.output_dir / "confusion_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target\\prediction", *MODE_NAMES])
        for name, values in zip(MODE_NAMES, metrics["confusion_matrix_rows_target_cols_prediction"]):
            writer.writerow([name, *values])
    print(f"[saved] {checkpoint_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
