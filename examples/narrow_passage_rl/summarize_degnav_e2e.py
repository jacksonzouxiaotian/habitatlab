#!/usr/bin/env python3
"""Aggregate multi-seed DEGNAV-E2E results without strict-success metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np


METRICS = (
    "success_rate_all",
    "success_rate_feasible",
    "infeasible_traversal_success_rate",
    "decision_accuracy",
    "collision_rate_feasible",
    "false_reject_rate_feasible",
    "correct_reject_rate_infeasible",
    "timeout_rate_all",
    "mean_steps",
)
EXPECTED_SEEDS = (1701, 1702, 1703)
EXPECTED_EPISODES = 80
EXPECTED_FEASIBLE = 64
EXPECTED_INFEASIBLE = 16


def episode_keys(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            (str(row["scene_id"]), str(row["episode_id"]))
            for row in csv.DictReader(handle)
        ]


def mean_std(rows, key):
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values), statistics.pstdev(values)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def teacher_audit(root: Path) -> dict:
    audit = {}
    for split, expected in (("train", 400), ("val", 80)):
        directory = root / "depth" / split
        metadata_path = directory / "metadata.json"
        episodes_path = directory / "episodes.csv"
        if not metadata_path.is_file() or not episodes_path.is_file():
            raise SystemExit(f"incomplete E2E teacher split: {directory}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            episodes = list(csv.DictReader(handle))
        archives = sorted(directory.glob("*.npz"))
        if (
            len(episodes) != expected
            or len(archives) != expected
            or int(metadata.get("episodes", -1)) != expected
            or int(metadata.get("history_storage_version", -1)) != 2
            or any(
                int(metadata.get("mode_counts", {}).get(name, 0)) <= 0
                for name in ("COMMIT", "EXPLORE", "RECOVER", "REJECT")
            )
        ):
            raise SystemExit(f"invalid E2E teacher split: {directory}")
        reject_frames = 0
        reject_without_observed_failure = 0
        history_alignment_errors = 0
        for path in archives:
            with np.load(path, allow_pickle=False) as episode:
                modes = np.asarray(episode["mode"])
                outcomes = np.asarray(episode["action_outcome"])
            if not np.allclose(outcomes[0], 0.0):
                history_alignment_errors += 1
            if len(modes) > 1:
                expected_previous_mode = modes[:-1]
                previous_mode_one_hot = outcomes[1:, :4]
                history_alignment_errors += int(
                    (
                        previous_mode_one_hot.argmax(axis=1)
                        != expected_previous_mode
                    ).sum()
                )
                history_alignment_errors += int(
                    (~np.isclose(previous_mode_one_hot.sum(axis=1), 1.0)).sum()
                )
            reject = modes == 3
            observed_failure = (outcomes[:, 4] > 0.5) | (outcomes[:, 5] > 0.80)
            reject_frames += int(reject.sum())
            reject_without_observed_failure += int((reject & ~observed_failure).sum())
        if reject_without_observed_failure:
            raise SystemExit(
                f"{split} teacher contains {reject_without_observed_failure} "
                "REJECT frames without an actor-observable failure"
            )
        if history_alignment_errors:
            raise SystemExit(
                f"{split} teacher contains {history_alignment_errors} "
                "previous-action/history alignment errors"
            )
        dataset_path = Path(metadata["dataset"])
        audit[split] = {
            "episodes": expected,
            "frames": int(metadata["frames"]),
            "mode_counts": metadata["mode_counts"],
            "reject_frames": reject_frames,
            "reject_without_actor_observable_failure": 0,
            "previous_action_history_alignment_errors": 0,
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
        }

    with (root / "depth" / "val" / "episodes.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        val_rows = list(csv.DictReader(handle))
    feasible = [row for row in val_rows if int(row["morphology_infeasible"]) == 0]
    infeasible = [row for row in val_rows if int(row["morphology_infeasible"]) == 1]
    audit["privileged_teacher_val_reference"] = {
        "feasible_episodes": len(feasible),
        "infeasible_episodes": len(infeasible),
        "feasible_success_rate": statistics.fmean(
            float(row["success"]) for row in feasible
        ),
        "feasible_false_reject_rate": statistics.fmean(
            float(row["rejected"]) for row in feasible
        ),
        "infeasible_correct_reject_rate": statistics.fmean(
            float(row["rejected"]) for row in infeasible
        ),
        "deployable": False,
        "reason": "teacher uses privileged 19-D geometry and feasibility labels",
    }
    return audit


def fmt(pair, percent=False):
    mean, std = pair
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return f"{mean * scale:.2f}±{std * scale:.2f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    args = parser.parse_args()
    audited_teacher = teacher_audit(args.teacher_root)
    rows = []
    episode_sets = {}
    for path in sorted(args.input_dir.glob("seed*/eval/summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["seed"] = int(path.parents[1].name.removeprefix("seed"))
        checkpoint_path = Path(row["checkpoint"])
        if not checkpoint_path.is_file():
            raise SystemExit(f"missing evaluated checkpoint: {checkpoint_path}")
        row["checkpoint_sha256"] = sha256_file(checkpoint_path)
        validation_path = path.parents[1] / "train" / "validation_best.json"
        if not validation_path.is_file():
            raise SystemExit(
                f"missing best-checkpoint validation metrics: {validation_path}"
            )
        row["offline_validation"] = json.loads(
            validation_path.read_text(encoding="utf-8")
        )
        rows.append(row)
        episodes_path = path.parent / "episodes.csv"
        if not episodes_path.is_file():
            raise SystemExit(f"missing episode-level result: {episodes_path}")
        row["episodes_sha256"] = sha256_file(episodes_path)
        keys = episode_keys(episodes_path)
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            episode_rows = list(csv.DictReader(handle))
        feasible_rows = [
            item for item in episode_rows if int(item["morphology_infeasible"]) == 0
        ]
        infeasible_rows = [
            item for item in episode_rows if int(item["morphology_infeasible"]) == 1
        ]
        row["success_rate_all"] = statistics.fmean(
            float(item["success"]) for item in episode_rows
        )
        row["infeasible_traversal_success_rate"] = statistics.fmean(
            float(item["success"]) for item in infeasible_rows
        )
        row["decision_accuracy"] = (
            sum(float(item["success"]) for item in feasible_rows)
            + sum(float(item["correct_reject"]) for item in infeasible_rows)
        ) / len(episode_rows)
        row["timeout_rate_all"] = statistics.fmean(
            item.get("termination_reason") == "timeout" for item in episode_rows
        )
        row["termination_counts"] = {
            name: sum(item.get("termination_reason") == name for item in episode_rows)
            for name in ("success", "reject", "timeout")
        }
        row["mode_step_totals"] = {
            name: sum(int(item[f"{name.lower()}_steps"]) for item in episode_rows)
            for name in ("COMMIT", "EXPLORE", "RECOVER", "REJECT")
        }
        row["episodes_using_mode"] = {
            name: sum(
                int(item[f"{name.lower()}_steps"]) > 0 for item in episode_rows
            )
            for name in ("COMMIT", "EXPLORE", "RECOVER", "REJECT")
        }
        if len(keys) != EXPECTED_EPISODES or len(set(keys)) != EXPECTED_EPISODES:
            raise SystemExit(
                f"seed {row['seed']} does not have {EXPECTED_EPISODES} unique episodes"
            )
        episode_sets[row["seed"]] = set(keys)
    if not rows:
        raise SystemExit(f"no completed seed results below {args.input_dir}")
    observed_seeds = tuple(sorted(int(row["seed"]) for row in rows))
    if observed_seeds != EXPECTED_SEEDS:
        raise SystemExit(
            f"expected completed seeds {EXPECTED_SEEDS}, got {observed_seeds}"
        )
    for row in rows:
        if (
            int(row["episodes"]) != EXPECTED_EPISODES
            or int(row["feasible_episodes"]) != EXPECTED_FEASIBLE
            or int(row["infeasible_episodes"]) != EXPECTED_INFEASIBLE
            or bool(row.get("actor_uses_19d"))
            or bool(row.get("actor_uses_handdesigned_task_memory"))
            or bool(row.get("sample_actions"))
            or row.get("evaluator_uses_19d_for_task_measures") is not True
        ):
            raise SystemExit(f"invalid deployment contract for seed {row['seed']}: {row}")
    canonical_set = episode_sets[EXPECTED_SEEDS[0]]
    if not all(value == canonical_set for value in episode_sets.values()):
        raise SystemExit("E2E seeds were not evaluated on the same episode set")
    aggregate = {key: mean_std(rows, key) for key in METRICS}
    offline_validation = {
        "accuracy": mean_std(
            [{"value": row["offline_validation"]["accuracy"]} for row in rows],
            "value",
        ),
        "macro_recall": mean_std(
            [
                {"value": row["offline_validation"]["macro_recall"]}
                for row in rows
            ],
            "value",
        ),
        "recall_by_mode": {
            name: mean_std(
                [
                    {
                        "value": row["offline_validation"]["recall_by_mode"][
                            name
                        ]
                    }
                    for row in rows
                ],
                "value",
            )
            for name in ("COMMIT", "EXPLORE", "RECOVER", "REJECT")
        },
    }
    total_mode_steps = {
        name: sum(row["mode_step_totals"][name] for row in rows)
        for name in ("COMMIT", "EXPLORE", "RECOVER", "REJECT")
    }
    episodes_using_mode = {
        name: sum(row["episodes_using_mode"][name] for row in rows)
        for name in ("COMMIT", "EXPLORE", "RECOVER", "REJECT")
    }
    summary_rows = [
        {
            "seed": row["seed"],
            **{key: row[key] for key in METRICS},
            "episodes": row["episodes"],
        }
        for row in rows
    ]
    with (args.input_dir / "summary_by_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# DEGNAV-E2E — MP3D-derived held-out narrow passages",
        "",
        "The actor uses raw Depth + PointGoal + action/outcome history + GRU; "
        "neither the 19-D geometry vector nor hand-designed task memory is an "
        "actor input. No strict-success "
        "column is reported.",
        "",
        "| Method | Train seeds | Val episodes/seed | All Success | Feasible Success | Feasible Collision | Feasible False Reject | Candidate traversal | Decision accuracy | Candidate Correct Reject | Timeout | Steps |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| DEGNAV-E2E (Depth, offline BC) | {seeds} | {episodes} | {all_success} | {success} | {collision} | {false_reject} | {infeasible_success} | {decision_accuracy} | {correct_reject} | {timeout} | {steps} |".format(
            seeds=len(rows),
            episodes=rows[0]["episodes"],
            all_success=fmt(aggregate["success_rate_all"], percent=True),
            success=fmt(aggregate["success_rate_feasible"], percent=True),
            collision=fmt(aggregate["collision_rate_feasible"], percent=True),
            false_reject=fmt(
                aggregate["false_reject_rate_feasible"], percent=True
            ),
            infeasible_success=fmt(
                aggregate["infeasible_traversal_success_rate"], percent=True
            ),
            decision_accuracy=fmt(aggregate["decision_accuracy"], percent=True),
            correct_reject=fmt(aggregate["correct_reject_rate_infeasible"], percent=True),
            timeout=fmt(aggregate["timeout_rate_all"], percent=True),
            steps=fmt(aggregate["mean_steps"]),
        ),
        "",
        "Protocol: 400 training and 80 validation episodes mined from official "
        "MP3D scans using the official PointNav train/val scene partition; "
        "0.36 m body, sliding disabled, scene-disjoint evaluation.",
        "",
        "The privileged training-only teacher reaches "
        f"{audited_teacher['privileged_teacher_val_reference']['feasible_success_rate'] * 100:.2f}% "
        "feasible Success and "
        f"{audited_teacher['privileged_teacher_val_reference']['infeasible_correct_reject_rate'] * 100:.2f}% "
        "infeasible Correct Reject on these validation episodes. It is not a "
        "deployable comparator because it consumes 19-D geometry and feasibility labels.",
        "",
        "The candidate-infeasible subset is not a collision-certified proof of "
        "absolute impossibility. Under the evaluated policy, its traversal rate is "
        f"{fmt(aggregate['infeasible_traversal_success_rate'], percent=True)} and "
        "its explicit Correct Reject rate is "
        f"{fmt(aggregate['correct_reject_rate_infeasible'], percent=True)}; both "
        "outcomes are reported instead of relabeling the subset after evaluation.",
        "",
        "Closed-loop mode usage across all seeds (mode steps / episodes with at "
        "least one selection): "
        + ", ".join(
            f"{name} {total_mode_steps[name]}/{episodes_using_mode[name]}"
            for name in ("COMMIT", "EXPLORE", "RECOVER", "REJECT")
        )
        + ".",
        "",
    ]
    (args.input_dir / "paper_table_degnav_e2e.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (args.input_dir / "summary.json").write_text(
        json.dumps(
            {
                "seeds": [row["seed"] for row in rows],
                "aggregate_mean_std": {
                    key: {"mean": value[0], "std": value[1]}
                    for key, value in aggregate.items()
                },
                "actor_uses_19d": False,
                "actor_uses_handdesigned_task_memory": False,
                "strict_success_reported": False,
                "training_method": "offline behavior cloning / policy distillation",
                "offline_validation_mean_std": {
                    "accuracy": {
                        "mean": offline_validation["accuracy"][0],
                        "std": offline_validation["accuracy"][1],
                    },
                    "macro_recall": {
                        "mean": offline_validation["macro_recall"][0],
                        "std": offline_validation["macro_recall"][1],
                    },
                    "recall_by_mode": {
                        name: {"mean": value[0], "std": value[1]}
                        for name, value in offline_validation[
                            "recall_by_mode"
                        ].items()
                    },
                },
                "teacher_audit": audited_teacher,
                "deployment_trace_audit": {
                    "by_seed": {
                        str(row["seed"]): {
                            "termination_counts": row["termination_counts"],
                            "mode_step_totals": row["mode_step_totals"],
                            "episodes_using_mode": row["episodes_using_mode"],
                        }
                        for row in rows
                    },
                    "all_seed_mode_step_totals": total_mode_steps,
                    "all_seed_episodes_using_mode": episodes_using_mode,
                },
                "completion_audit": {
                    "expected_seeds": list(EXPECTED_SEEDS),
                    "episodes_per_seed": EXPECTED_EPISODES,
                    "feasible_episodes_per_seed": EXPECTED_FEASIBLE,
                    "infeasible_episodes_per_seed": EXPECTED_INFEASIBLE,
                    "unique_episode_key": ["scene_id", "episode_id"],
                    "all_seeds_match_episode_set": True,
                    "checkpoint_sha256_by_seed": {
                        str(row["seed"]): row["checkpoint_sha256"] for row in rows
                    },
                    "episode_csv_sha256_by_seed": {
                        str(row["seed"]): row["episodes_sha256"] for row in rows
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[write] {args.input_dir / 'paper_table_degnav_e2e.md'}")


if __name__ == "__main__":
    main()
