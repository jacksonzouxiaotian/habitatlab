#!/usr/bin/env python3
"""Preflight checks for NaVILA R2R/VLN-CE paired evaluation.

The real R2R/VLN-CE evaluation requires licensed Matterport3D scene assets.
This script does not download or fabricate those assets.  It verifies the
local model, annotations, scene files, and NaVILA evaluation entry points, then
writes a machine-readable report explaining whether the paired evaluation can
run.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_VLN_STORE = Path(
    "/media/xiaotian/ACD525D1B7A093D9/robot_nav_data/vln"
)
DEFAULT_NAVILA_ROOT = Path("/home/xiaotian/vla/NaVILA")
DEFAULT_MODEL = DEFAULT_VLN_STORE / "models/navila-llama3-8b-8f"
DEFAULT_DATA_ROOT = DEFAULT_VLN_STORE / "data"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "results/vln_r2r_eval_preflight"
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _as_rel(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _git_commit(path: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _load_episodes(dataset_file: Path) -> list[dict[str, Any]]:
    with gzip.open(dataset_file, "rt", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{dataset_file} does not contain an episodes list")
    return episodes


def _scene_name_from_episode(ep: dict[str, Any]) -> str:
    scene_id = str(ep.get("scene_id", ""))
    # Expected VLN-CE form: mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb
    parts = scene_id.split("/")
    if len(parts) >= 2 and parts[0] == "mp3d":
        return parts[1]
    if scene_id.endswith(".glb"):
        return Path(scene_id).stem
    return scene_id


def _check_import(module: str) -> Check:
    try:
        imported = importlib.import_module(module)
        location = getattr(imported, "__file__", "no __file__")
        return Check(f"python import {module}", True, str(location))
    except Exception as exc:  # pragma: no cover - environment dependent
        return Check(f"python import {module}", False, repr(exc))


def _ensure_link(link: Path, target: Path, checks: list[Check]) -> None:
    if link.exists() or link.is_symlink():
        resolved = link.resolve() if link.exists() or link.is_symlink() else None
        ok = resolved == target.resolve()
        checks.append(
            Check(
                f"symlink {link}",
                ok,
                f"points to {resolved}; expected {target.resolve()}",
            )
        )
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    checks.append(Check(f"symlink {link}", True, f"created -> {target}"))


def _scene_status(scene_root: Path, scene_names: list[str]) -> dict[str, Any]:
    missing_glb: list[str] = []
    missing_navmesh: list[str] = []
    present_glb = 0
    present_navmesh = 0
    present_house = 0
    for scene in scene_names:
        scene_dir = scene_root / scene
        glb = scene_dir / f"{scene}.glb"
        navmesh = scene_dir / f"{scene}.navmesh"
        house = scene_dir / f"{scene}.house"
        if glb.exists():
            present_glb += 1
        else:
            missing_glb.append(scene)
        if navmesh.exists():
            present_navmesh += 1
        else:
            missing_navmesh.append(scene)
        if house.exists():
            present_house += 1
    return {
        "scene_root": _as_rel(scene_root),
        "expected_scene_count": len(scene_names),
        "present_glb_count": present_glb,
        "present_navmesh_count": present_navmesh,
        "present_house_count": present_house,
        "missing_glb": missing_glb,
        "missing_navmesh": missing_navmesh,
    }


def _write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "r2r_vlnce_preflight.json"
    md_path = output_dir / "r2r_vlnce_preflight.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    checks = report["checks"]
    blockers = report["blockers"]
    scene = report["scene_status"]
    lines = [
        "# R2R/VLN-CE Paired Evaluation Preflight",
        "",
        f"Status: **{report['status']}**",
        f"Generated: `{report['generated_at']}`",
        f"Git commit: `{report['git_commit']}`",
        "",
        "## Summary",
        "",
        f"- NaVILA root: `{report['navila_root']}`",
        f"- Model path: `{report['model_path']}`",
        f"- Dataset split: `{report['split']}`",
        f"- Dataset file: `{report['dataset_file']}`",
        f"- Episodes: `{report['episode_count']}`",
        f"- Unique MP3D scenes in split: `{scene['expected_scene_count']}`",
        f"- MP3D `.glb` present: `{scene['present_glb_count']}`",
        f"- MP3D `.navmesh` present: `{scene['present_navmesh_count']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| :--- | :---: | :--- |",
    ]
    for check in checks:
        status = "PASS" if check["ok"] else "FAIL"
        detail = str(check["detail"]).replace("\n", " ")
        lines.append(f"| {check['name']} | {status} | `{detail}` |")
    lines.extend(["", "## Blockers", ""])
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Missing Scene Examples",
            "",
            "First missing `.glb` scene IDs:",
            "",
            "```text",
            "\n".join(scene["missing_glb"][:20]) or "none",
            "```",
            "",
            "## Reproduction Command",
            "",
            "Run this after MP3D assets are installed:",
            "",
            "```bash",
            report["run_command"],
            "```",
            "",
            "Aggregate NaVILA JSON outputs with:",
            "",
            "```bash",
            report["aggregate_command"],
            "```",
            "",
            "## Data Note",
            "",
            (
                "MP3D scene reconstructions are license-gated assets. This "
                "preflight only checks local availability; it does not download "
                "or redistribute Matterport3D data."
            ),
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate local inputs for NaVILA R2R/VLN-CE evaluation."
    )
    parser.add_argument("--navila-root", type=Path, default=DEFAULT_NAVILA_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-start-idx", type=int, default=0)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--ensure-symlinks",
        action="store_true",
        help="Create NaVILA evaluation/data symlinks if they are missing.",
    )
    parser.add_argument(
        "--skip-import-check",
        action="store_true",
        help="Skip importing habitat/habitat_sim/habitat_baselines.",
    )
    args = parser.parse_args()

    navila_root = args.navila_root.resolve()
    eval_root = navila_root / "evaluation"
    data_root = args.data_root.resolve()
    dataset_file = (
        data_root
        / "datasets/R2R_VLNCE_v1-3_preprocessed"
        / args.split
        / f"{args.split}.json.gz"
    )
    gt_file = (
        data_root
        / "datasets/R2R_VLNCE_v1-3_preprocessed"
        / args.split
        / f"{args.split}_gt.json.gz"
    )
    scene_root = data_root / "scene_datasets/mp3d"
    ddppo = data_root / "ddppo-models/gibson-2plus-resnet50.pth"
    model = args.model.resolve()

    checks: list[Check] = [
        Check("NaVILA root", navila_root.exists(), _as_rel(navila_root)),
        Check("NaVILA evaluation/run.py", (eval_root / "run.py").exists(), _as_rel(eval_root / "run.py")),
        Check(
            "NaVILA r2r.sh",
            (eval_root / "scripts/eval/r2r.sh").exists(),
            _as_rel(eval_root / "scripts/eval/r2r.sh"),
        ),
        Check(
            "NaVILA R2R config",
            (eval_root / "vlnce_baselines/config/r2r_baselines/navila.yaml").exists(),
            _as_rel(eval_root / "vlnce_baselines/config/r2r_baselines/navila.yaml"),
        ),
        Check("NaVILA model directory", model.exists(), _as_rel(model)),
        Check("R2R split file", dataset_file.exists(), _as_rel(dataset_file)),
        Check("R2R GT file", gt_file.exists(), _as_rel(gt_file)),
        Check("DDPPO depth encoder", ddppo.exists(), _as_rel(ddppo)),
        Check("MP3D scene root", scene_root.exists(), _as_rel(scene_root)),
    ]

    if args.ensure_symlinks:
        _ensure_link(eval_root / "data/datasets", data_root / "datasets", checks)
        _ensure_link(eval_root / "data/scene_datasets", data_root / "scene_datasets", checks)
        _ensure_link(eval_root / "data/ddppo-models", data_root / "ddppo-models", checks)
    else:
        for name in ("datasets", "scene_datasets", "ddppo-models"):
            link = eval_root / "data" / name
            expected = data_root / name
            ok = link.exists() and link.resolve() == expected.resolve()
            checks.append(
                Check(
                    f"NaVILA data symlink {name}",
                    ok,
                    f"{_as_rel(link)} -> {_as_rel(link.resolve()) if link.exists() else 'missing'}; expected {_as_rel(expected)}",
                )
            )

    if not args.skip_import_check:
        for module in ("habitat", "habitat_sim", "habitat_baselines"):
            checks.append(_check_import(module))

    episodes: list[dict[str, Any]] = []
    scene_names: list[str] = []
    dataset_error = ""
    if dataset_file.exists():
        try:
            episodes = _load_episodes(dataset_file)
            scene_names = sorted({_scene_name_from_episode(ep) for ep in episodes})
            checks.append(
                Check(
                    "R2R episode parse",
                    True,
                    f"{len(episodes)} episodes, {len(scene_names)} unique scenes",
                )
            )
        except Exception as exc:
            dataset_error = repr(exc)
            checks.append(Check("R2R episode parse", False, dataset_error))

    scene = _scene_status(scene_root, scene_names) if scene_names else {
        "scene_root": _as_rel(scene_root),
        "expected_scene_count": 0,
        "present_glb_count": 0,
        "present_navmesh_count": 0,
        "present_house_count": 0,
        "missing_glb": [],
        "missing_navmesh": [],
    }

    if scene_names:
        checks.append(
            Check(
                "MP3D .glb coverage for split",
                scene["present_glb_count"] == scene["expected_scene_count"],
                f"{scene['present_glb_count']}/{scene['expected_scene_count']} present",
            )
        )
        checks.append(
            Check(
                "MP3D .navmesh coverage for split",
                scene["present_navmesh_count"] == scene["expected_scene_count"],
                f"{scene['present_navmesh_count']}/{scene['expected_scene_count']} present",
            )
        )

    disk_path = data_root if data_root.exists() else data_root.parent
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    usage = shutil.disk_usage(disk_path)
    checks.append(
        Check(
            "disk free",
            usage.free > 50 * 1024**3,
            f"{usage.free / 1024**3:.1f} GiB free at {disk_path}",
        )
    )

    blockers: list[str] = []
    for check in checks:
        if not check.ok:
            blockers.append(f"{check.name}: {check.detail}")
    if scene_names and scene["missing_glb"]:
        blockers.append(
            (
                "Install licensed Matterport3D Habitat scene assets under "
                f"{scene_root}/{{scene}}/{{scene}}.glb. "
                "VLN-CE expects 90 MP3D scenes in total; this split currently "
                f"needs {scene['expected_scene_count']} scenes."
            )
        )
    if scene_names and scene["missing_navmesh"]:
        blockers.append(
            (
                "Install or generate matching `.navmesh` files under "
                f"{scene_root}/{{scene}}/{{scene}}.navmesh."
            )
        )

    ckpt_name = model.name
    run_command = (
        f"cd {eval_root} && "
        f"bash scripts/eval/r2r.sh {model} {args.num_chunks} "
        f"{args.chunk_start_idx} \"{args.gpu_ids}\""
    )
    aggregate_command = (
        f"cd {eval_root} && "
        f"python scripts/eval_jsons.py ./eval_out/{ckpt_name}/VLN-CE-v1/"
        f"{args.split} {args.num_chunks}"
    )

    report = {
        "status": "PASS" if not blockers else "BLOCKED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "git_commit": _git_commit(Path.cwd()),
        "navila_git_commit": _git_commit(navila_root),
        "navila_root": _as_rel(navila_root),
        "eval_root": _as_rel(eval_root),
        "model_path": _as_rel(model),
        "data_root": _as_rel(data_root),
        "split": args.split,
        "dataset_file": _as_rel(dataset_file),
        "gt_file": _as_rel(gt_file),
        "episode_count": len(episodes),
        "unique_scene_count": len(scene_names),
        "checks": [asdict(c) for c in checks],
        "blockers": blockers,
        "scene_status": scene,
        "run_command": run_command,
        "aggregate_command": aggregate_command,
    }
    json_path, md_path = _write_report(report, args.output_dir)
    print(f"[preflight] {report['status']}")
    print(f"[write] {json_path}")
    print(f"[write] {md_path}")
    if blockers:
        print("[blockers]")
        for blocker in blockers:
            print(f"  - {blocker}")
        return 2
    print("[run]", run_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
