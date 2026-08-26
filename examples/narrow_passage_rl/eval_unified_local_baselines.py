#!/usr/bin/env python3
"""Paired Procedural-v2 evaluation for the four local-navigation methods.

The runner evaluates every method on exactly the same episode seeds and keeps
training-seed provenance separate from evaluation-scene seeds.  It intentionally
creates new artifacts instead of overwriting the existing paper tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from cross_episode_memory import CrossEpisodeMemory
from eval_harder_benchmark import eval_method, run_episode
from procedural_env_v2 import HarderNarrowPassageEnv


DEFAULT_CTYPES = [
    "straight",
    "l_shaped",
    "s_shaped",
    "narrow_exit",
    "narrow_entry",
    "asymmetric",
    "false_feasible",
]

METHOD_ORDER = [
    "geometry_rule",
    "direct_control_ppo",
    "recurrent_ppo",
    "recurrent_ppo_direct_init",
    "degnav_memory",
]

METHOD_LABELS = {
    "geometry_rule": "Geometry rule",
    "direct_control_ppo": "Direct-control PPO",
    "recurrent_ppo": "Recurrent PPO",
    "recurrent_ppo_direct_init": "Recurrent PPO (Direct-init + PPO fine-tune)",
    "degnav_memory": "DEGNAV + geometry-guided memory",
}


class _DirectPPOAgent:
    strict_ablation = None

    def __init__(self, model: Any) -> None:
        self.model = model
        self.last_audit: Dict[str, Any] = {}

    def reset(self) -> None:
        self.last_audit = {}

    def step(self, obs: np.ndarray):
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32), "DIRECT_VELOCITY"


class _RecurrentPPOAgent:
    strict_ablation = None

    def __init__(self, model: Any) -> None:
        self.model = model
        self.last_audit: Dict[str, Any] = {}
        self._state = None
        self._episode_start = np.ones((1,), dtype=bool)

    def reset(self) -> None:
        self.last_audit = {}
        self._state = None
        self._episode_start[:] = True

    def step(self, obs: np.ndarray):
        action, self._state = self.model.predict(
            obs,
            state=self._state,
            episode_start=self._episode_start,
            deterministic=True,
        )
        self._episode_start[:] = False
        return np.asarray(action, dtype=np.float32), "RECURRENT_DIRECT_VELOCITY"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(args: List[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def _episode_schedule(eval_seed: int, episodes: int, start_index: int = 0):
    indices = range(start_index, start_index + episodes)
    episode_seeds = [int(eval_seed) * 1_000_000 + idx for idx in indices]
    episode_ids = [f"eval{eval_seed}_ep{idx:06d}" for idx in indices]
    return episode_seeds, episode_ids


def _clean_internal_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _eval_rule_or_degnav(
    method: str,
    output_method: str,
    ctypes: List[str],
    episodes: int,
    eval_seed: int,
    width_range: tuple[float, float],
    max_steps: int,
    episode_start_index: int,
) -> List[Dict[str, Any]]:
    episode_seeds, episode_ids = _episode_schedule(
        eval_seed, episodes, episode_start_index
    )
    cross_memory = CrossEpisodeMemory() if method == "fsm_cross_memory" else None
    rows = eval_method(
        method,
        ctypes,
        episodes,
        seed=eval_seed,
        width_range=width_range,
        max_steps=max_steps,
        cross_mem_override=cross_memory,
        episode_seeds=episode_seeds,
        episode_ids=episode_ids,
        progress=False,
    )
    for row in rows:
        row["source_method"] = method
        row["method"] = output_method
        row["paper_method"] = METHOD_LABELS[output_method]
        row["eval_seed"] = eval_seed
        row["training_seed"] = "deterministic"
        row["training_steps"] = 0
    return rows


def _eval_rl(
    agent: Any,
    method: str,
    ctypes: List[str],
    episodes: int,
    eval_seed: int,
    width_range: tuple[float, float],
    max_steps: int,
    training_seed: int,
    training_steps: int,
    episode_start_index: int,
) -> List[Dict[str, Any]]:
    env = HarderNarrowPassageEnv(
        {
            "corridor_types": ctypes,
            "seed": eval_seed,
            "width_range": width_range,
            "max_steps": max_steps,
        }
    )
    episode_seeds, episode_ids = _episode_schedule(
        eval_seed, episodes, episode_start_index
    )
    rows: List[Dict[str, Any]] = []
    for episode_idx, (episode_seed, scenario_id) in enumerate(
        zip(episode_seeds, episode_ids)
    ):
        stat = run_episode(
            env,
            method="external_rl",
            local_mem=None,
            cross_mem=None,
            max_steps=max_steps,
            entry_obs_ref=[],
            agent=agent,
            episode_seed=episode_seed,
        )
        stat = _clean_internal_fields(stat)
        stat.update(
            {
                "episode_idx": episode_start_index + episode_idx,
                "scenario_id": scenario_id,
                "episode_id": f"{method}_{scenario_id}",
                "episode_seed": episode_seed,
                "scene_id": "procedural_v2",
                "domain": "procedural_v2",
                "split": "paired_held_out",
                "seed": eval_seed,
                "eval_seed": eval_seed,
                "method": method,
                "source_method": "sb3_model",
                "paper_method": METHOD_LABELS[method],
                "training_seed": training_seed,
                "training_steps": training_steps,
                "belief_used_by_policy": 0,
                "mode_semantics_available": 0,
            }
        )
        rows.append(stat)
        if episode_idx < 3 or (episode_idx + 1) % 100 == 0:
            rolling = float(np.mean([float(row["success"]) for row in rows]))
            print(
                f"  [{method}] eval_seed={eval_seed} ep={episode_idx + 1}/{episodes} "
                f"rolling_sr={rolling:.3f}",
                flush=True,
            )
    return rows


def _number(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _seed_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    infeasible_rows = [
        row for row in rows if str(row.get("corridor_type", "")).lower() == "false_feasible"
    ]
    feasible_rows = [
        row for row in rows if str(row.get("corridor_type", "")).lower() != "false_feasible"
    ]
    return {
        "episodes": len(rows),
        "success_rate": float(np.mean([_number(row, "success") for row in rows])),
        "strict_success_rate": float(
            np.mean([_number(row, "collision_free_success") for row in rows])
        ),
        "collision_rate": float(
            np.mean([_number(row, "collision") for row in rows])
        ),
        "near_collision_rate": float(
            np.mean([_number(row, "near_collision") for row in rows])
        ),
        "timeout_or_stuck_rate": float(
            np.mean(
                [
                    float(
                        _number(row, "timeout") > 0.5
                        or _number(row, "stuck") > 0.5
                    )
                    for row in rows
                ]
            )
        ),
        "avg_steps": float(np.mean([_number(row, "steps") for row in rows])),
        "avg_min_body_margin": float(
            np.mean([_number(row, "min_body_margin") for row in rows])
        ),
        "correct_reject_rate": float(
            np.mean([_number(row, "correct_reject") for row in infeasible_rows])
            if infeasible_rows
            else 0.0
        ),
        "false_reject_rate": float(
            np.mean([_number(row, "false_reject") for row in feasible_rows])
            if feasible_rows
            else 0.0
        ),
    }


def _aggregate(
    rows: List[Dict[str, Any]],
    eval_seeds: List[int],
    corridor_types: List[str],
    methods: List[str],
):
    by_seed_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    metric_keys = [
        "success_rate",
        "strict_success_rate",
        "collision_rate",
        "near_collision_rate",
        "timeout_or_stuck_rate",
        "avg_steps",
        "avg_min_body_margin",
        "correct_reject_rate",
        "false_reject_rate",
    ]
    for method in methods:
        method_seed_summaries = []
        for eval_seed in eval_seeds:
            selected = [
                row
                for row in rows
                if row["method"] == method and int(row["eval_seed"]) == eval_seed
            ]
            if not selected:
                raise ValueError(
                    f"no rows for method={method!r}, eval_seed={eval_seed}"
                )
            summary = _seed_summary(selected)
            summary.update(
                {
                    "method": method,
                    "paper_method": METHOD_LABELS[method],
                    "eval_seed": eval_seed,
                }
            )
            method_seed_summaries.append(summary)
            by_seed_rows.append(summary)

        aggregate: Dict[str, Any] = {
            "method": method,
            "paper_method": METHOD_LABELS[method],
            "eval_seeds": len(eval_seeds),
            "episodes_per_eval_seed": int(method_seed_summaries[0]["episodes"]),
        }
        for key in metric_keys:
            values = np.asarray([float(item[key]) for item in method_seed_summaries])
            aggregate[f"{key}_mean"] = float(np.mean(values))
            aggregate[f"{key}_std"] = float(np.std(values, ddof=0))
        for corridor_type in corridor_types:
            seed_rates = []
            for eval_seed in eval_seeds:
                selected = [
                    row
                    for row in rows
                    if row["method"] == method
                    and int(row["eval_seed"]) == eval_seed
                    and str(row.get("corridor_type", "")).lower() == corridor_type
                ]
                seed_rates.append(
                    float(np.mean([_number(row, "success") for row in selected]))
                    if selected
                    else float("nan")
                )
            finite_rates = [value for value in seed_rates if math.isfinite(value)]
            aggregate[f"sr_{corridor_type}_mean"] = (
                float(np.mean(finite_rates)) if finite_rates else float("nan")
            )
            aggregate[f"sr_{corridor_type}_std"] = (
                float(np.std(finite_rates, ddof=0)) if finite_rates else float("nan")
            )
        aggregate_rows.append(aggregate)
    return by_seed_rows, aggregate_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "method",
        "paper_method",
        "training_seed",
        "training_steps",
        "eval_seed",
        "episode_idx",
        "episode_id",
        "scenario_id",
        "episode_seed",
        "corridor_type",
        "passable",
        "success",
        "collision_free_success",
        "collision",
        "near_collision",
        "timeout",
        "stuck",
        "steps",
        "min_body_margin",
        "correct_reject",
        "false_reject",
        "final_outcome",
    ]
    all_fields = set().union(*(row.keys() for row in rows))
    fieldnames = [field for field in preferred if field in all_fields]
    fieldnames.extend(sorted(all_fields - set(fieldnames)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {path}")


def _read_episode_csvs(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    return rows


def _rate_cell(row: Dict[str, Any], key: str) -> str:
    return f"{100.0 * row[key + '_mean']:.1f}±{100.0 * row[key + '_std']:.1f}%"


def _write_markdown(
    path: Path,
    aggregate_rows: List[Dict[str, Any]],
    training_steps_by_method: Dict[str, int],
    episodes: int,
    eval_seeds: List[int],
    recurrent_training_note: str,
) -> None:
    lines = [
        "# Paired Procedural-v2 Local-Baseline Rerun",
        "",
        "| Method | Success ↑ | Collision ↓ | Near collision ↓ | Timeout/stuck ↓ | Correct reject ↑ | False reject ↓ | Avg steps ↓ |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        has_reject_interface = row["method"] == "degnav_memory"
        lines.append(
            "| {label} | {sr} | {collision} | {near} | {timeout} | {correct_reject} | {false_reject} | {steps:.1f}±{steps_std:.1f} |".format(
                label=row["paper_method"],
                sr=_rate_cell(row, "success_rate"),
                collision=_rate_cell(row, "collision_rate"),
                near=_rate_cell(row, "near_collision_rate"),
                timeout=_rate_cell(row, "timeout_or_stuck_rate"),
                correct_reject=(
                    _rate_cell(row, "correct_reject_rate")
                    if has_reject_interface
                    else "—"
                ),
                false_reject=(
                    _rate_cell(row, "false_reject_rate")
                    if has_reject_interface
                    else "—"
                ),
                steps=row["avg_steps_mean"],
                steps_std=row["avg_steps_std"],
            )
        )
    lines.extend(
        [
            "",
            "## Success by corridor type",
            "",
            "| Method | Straight | L-shaped | S-shaped | Narrow exit | Narrow entry | Asymmetric | False feasible |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {label} | {straight} | {l_shaped} | {s_shaped} | {narrow_exit} | {narrow_entry} | {asymmetric} | {false_feasible} |".format(
                label=row["paper_method"],
                straight=_rate_cell(row, "sr_straight"),
                l_shaped=_rate_cell(row, "sr_l_shaped"),
                s_shaped=_rate_cell(row, "sr_s_shaped"),
                narrow_exit=_rate_cell(row, "sr_narrow_exit"),
                narrow_entry=_rate_cell(row, "sr_narrow_entry"),
                asymmetric=_rate_cell(row, "sr_asymmetric"),
                false_feasible=_rate_cell(row, "sr_false_feasible"),
            )
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Environment: current `HarderNarrowPassageEnv` / Procedural v2.",
            f"- Paired evaluation seeds: `{eval_seeds}`; `{episodes}` episodes per evaluation seed and method.",
            "- All methods receive the same ordered episode seeds, width range, maximum step budget, and seven corridor types.",
            "- Values are mean±population-std across evaluation-scene seeds, not across independent RL training seeds.",
            "- `Strict success` means goal success without simulator collision or body overlap; near collision is the procedural body-margin diagnostic.",
            "- `Correct reject` is conditioned on `false_feasible` episodes; `False reject` is conditioned on all other corridor types.",
            "- Direct and recurrent policies have no explicit Commit/Explore/Recover/Reject interface; their mode-specific fields are unavailable.",
            "- This is a new paired rerun and does not overwrite the existing canonical tables.",
        ]
    )
    for method in (
        "direct_control_ppo",
        "recurrent_ppo",
        "recurrent_ppo_direct_init",
    ):
        if method in training_steps_by_method:
            lines.append(
                f"- {METHOD_LABELS[method]} checkpoint steps: "
                f"`{training_steps_by_method[method]}`; one training seed."
            )
    if recurrent_training_note:
        lines.append(f"- Recurrent training note: {recurrent_training_note}")
    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-model", type=Path, required=True)
    parser.add_argument("--recurrent-model", type=Path, required=True)
    parser.add_argument(
        "--legacy-recurrent-model",
        type=Path,
        default=None,
        help="optional provenance checkpoint when a merged table retains an older recurrent row",
    )
    parser.add_argument("--direct-training-seed", type=int, default=0)
    parser.add_argument("--recurrent-training-seed", type=int, default=0)
    parser.add_argument("--training-num-envs", type=int, default=8)
    parser.add_argument("--training-n-steps", type=int, default=1024)
    parser.add_argument("--training-batch-size", type=int, default=256)
    parser.add_argument("--training-n-epochs", type=int, default=3)
    parser.add_argument("--training-reward-mode", default="fair")
    parser.add_argument("--direct-training-device", default="cpu")
    parser.add_argument("--recurrent-training-device", default="cuda")
    parser.add_argument(
        "--recurrent-training-note",
        default="",
        help="provenance note written to the paper table and metadata",
    )
    parser.add_argument("--inference-device", default="cpu")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument(
        "--episode-start-index",
        type=int,
        default=0,
        help="offset into the deterministic episode schedule (for parallel chunks)",
    )
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--corridor-types", nargs="+", default=DEFAULT_CTYPES)
    parser.add_argument("--width-range", nargs=2, type=float, default=[0.45, 0.90])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument(
        "--methods", nargs="+", choices=METHOD_ORDER, default=METHOD_ORDER
    )
    parser.add_argument(
        "--episode-inputs",
        nargs="+",
        type=Path,
        default=None,
        help="merge existing per-seed episodes.csv files instead of rerunning episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent
        / "results"
        / "narrow_passage_rl"
        / "unified_local_baselines_20260824",
    )
    args = parser.parse_args()

    if not args.direct_model.is_file():
        raise FileNotFoundError(args.direct_model)
    if not args.recurrent_model.is_file():
        raise FileNotFoundError(args.recurrent_model)
    if args.legacy_recurrent_model and not args.legacy_recurrent_model.is_file():
        raise FileNotFoundError(args.legacy_recurrent_model)

    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    direct_model = PPO.load(str(args.direct_model), device=args.inference_device)
    recurrent_model = RecurrentPPO.load(
        str(args.recurrent_model), device=args.inference_device
    )
    direct_steps = int(getattr(direct_model, "num_timesteps", 0))
    recurrent_steps = int(getattr(recurrent_model, "num_timesteps", 0))
    ctypes = [str(value).lower() for value in args.corridor_types]
    width_range = (float(args.width_range[0]), float(args.width_range[1]))

    if args.episode_inputs:
        all_rows = _read_episode_csvs(args.episode_inputs)
        print(f"[merge] loaded {len(all_rows)} episode rows")
    else:
        all_rows: List[Dict[str, Any]] = []
        for eval_seed in args.eval_seeds:
            print(f"[eval] paired evaluation seed {eval_seed}")
            if "geometry_rule" in args.methods:
                all_rows.extend(
                    _eval_rule_or_degnav(
                        "rule_baseline",
                        "geometry_rule",
                        ctypes,
                        args.episodes,
                        eval_seed,
                        width_range,
                        args.max_steps,
                        args.episode_start_index,
                    )
                )
            if "direct_control_ppo" in args.methods:
                all_rows.extend(
                    _eval_rl(
                        _DirectPPOAgent(direct_model),
                        "direct_control_ppo",
                        ctypes,
                        args.episodes,
                        eval_seed,
                        width_range,
                        args.max_steps,
                        args.direct_training_seed,
                        direct_steps,
                        args.episode_start_index,
                    )
                )
            if "recurrent_ppo" in args.methods:
                all_rows.extend(
                    _eval_rl(
                        _RecurrentPPOAgent(recurrent_model),
                        "recurrent_ppo",
                        ctypes,
                        args.episodes,
                        eval_seed,
                        width_range,
                        args.max_steps,
                        args.recurrent_training_seed,
                        recurrent_steps,
                        args.episode_start_index,
                    )
                )
            if "recurrent_ppo_direct_init" in args.methods:
                all_rows.extend(
                    _eval_rl(
                        _RecurrentPPOAgent(recurrent_model),
                        "recurrent_ppo_direct_init",
                        ctypes,
                        args.episodes,
                        eval_seed,
                        width_range,
                        args.max_steps,
                        args.recurrent_training_seed,
                        recurrent_steps,
                        args.episode_start_index,
                    )
                )
            if "degnav_memory" in args.methods:
                all_rows.extend(
                    _eval_rule_or_degnav(
                        "fsm_cross_memory",
                        "degnav_memory",
                        ctypes,
                        args.episodes,
                        eval_seed,
                        width_range,
                        args.max_steps,
                        args.episode_start_index,
                    )
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = args.output_dir / "episodes.csv"
    by_seed_path = args.output_dir / "summary_by_eval_seed.csv"
    aggregate_path = args.output_dir / "summary.csv"
    table_path = args.output_dir / "paper_table_unified_local_baselines.md"
    metadata_path = args.output_dir / "metadata.json"

    by_seed_rows, aggregate_rows = _aggregate(
        all_rows, args.eval_seeds, ctypes, args.methods
    )
    training_steps_by_method: Dict[str, int] = {}
    for method in args.methods:
        values = {
            int(float(row["training_steps"]))
            for row in all_rows
            if row["method"] == method and str(row.get("training_steps", ""))
        }
        if values:
            if len(values) != 1:
                raise ValueError(
                    f"multiple training-step values for method={method}: {sorted(values)}"
                )
            training_steps_by_method[method] = values.pop()
    _write_csv(episodes_path, all_rows)
    _write_csv(by_seed_path, by_seed_rows)
    _write_csv(aggregate_path, aggregate_rows)
    _write_markdown(
        table_path,
        aggregate_rows,
        training_steps_by_method,
        args.episodes,
        args.eval_seeds,
        args.recurrent_training_note,
    )

    repo_root = Path(__file__).resolve().parents[2]
    metadata = {
        "protocol": "paired_procedural_v2_local_baselines",
        "methods": args.methods,
        "corridor_types": ctypes,
        "width_range": list(width_range),
        "max_steps": args.max_steps,
        "inference_device": args.inference_device,
        "episodes_per_eval_seed": args.episodes,
        "episode_start_index": args.episode_start_index,
        "eval_seeds": args.eval_seeds,
        "merged_episode_inputs": [str(path) for path in args.episode_inputs or []],
        "rl_training_seeds": {
            "direct_control_ppo": args.direct_training_seed,
            "recurrent_ppo": args.recurrent_training_seed,
        },
        "rl_training_protocol": {
            "num_envs": args.training_num_envs,
            "n_steps": args.training_n_steps,
            "batch_size": args.training_batch_size,
            "n_epochs": args.training_n_epochs,
            "reward_mode": args.training_reward_mode,
            "corridor_types": "full",
            "devices": {
                "direct_control_ppo": args.direct_training_device,
                "recurrent_ppo": args.recurrent_training_device,
            },
            "recurrent_training_note": args.recurrent_training_note,
        },
        "checkpoints": {
            "direct_control_ppo": {
                "path": str(args.direct_model.resolve()),
                "sha256": _sha256(args.direct_model),
                "num_timesteps": direct_steps,
            },
            (
                "recurrent_ppo_direct_init"
                if "recurrent_ppo_direct_init" in args.methods
                else "recurrent_ppo"
            ): {
                "path": str(args.recurrent_model.resolve()),
                "sha256": _sha256(args.recurrent_model),
                "num_timesteps": recurrent_steps,
            },
        },
        "git_commit": _git_value(["rev-parse", "HEAD"], repo_root),
        "git_status": _git_value(["status", "--short"], repo_root),
        "artifacts": {
            "episodes": str(episodes_path),
            "summary_by_eval_seed": str(by_seed_path),
            "summary": str(aggregate_path),
            "table": str(table_path),
        },
    }
    if args.legacy_recurrent_model:
        legacy_recurrent = RecurrentPPO.load(
            str(args.legacy_recurrent_model), device=args.inference_device
        )
        metadata["checkpoints"]["recurrent_ppo"] = {
            "path": str(args.legacy_recurrent_model.resolve()),
            "sha256": _sha256(args.legacy_recurrent_model),
            "num_timesteps": int(getattr(legacy_recurrent, "num_timesteps", 0)),
        }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[write] {metadata_path}")


if __name__ == "__main__":
    main()
