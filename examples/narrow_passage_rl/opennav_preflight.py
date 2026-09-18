#!/usr/bin/env python3
"""Preflight checks for the local Open-Nav installation.

Open-Nav depends on Habitat 0.1.7, R2R-CE annotations, MP3D scenes, a waypoint
predictor checkpoint, a DDPPO depth encoder, RAM, SpatialBot3B, and an LLM
backend/API key.  This script reports which pieces are installed without
pretending that missing licensed/gated assets are available.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OPENNAV_ROOT = Path("/home/xiaotian/vla/Open-Nav")
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "results/opennav_preflight"
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def import_check(name: str) -> Check:
    try:
        module = importlib.import_module(name)
        return Check(f"python import {name}", True, str(getattr(module, "__file__", "")))
    except Exception as exc:
        return Check(f"python import {name}", False, repr(exc))


def file_check(name: str, path: Path) -> Check:
    return Check(name, path.exists(), str(path.resolve() if path.exists() else path))


def torch_checkpoint_check(name: str, path: Path) -> Check:
    if not path.is_file():
        return Check(name, False, f"missing: {path}")
    try:
        import torch

        obj = torch.load(path, map_location="cpu")
        detail = f"readable torch checkpoint; type={type(obj).__name__}"
        if isinstance(obj, dict):
            detail += f"; keys={list(obj)[:8]}"
        return Check(name, True, detail)
    except Exception as exc:
        return Check(name, False, f"{path}: {repr(exc)}")


def load_episodes(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{path} does not contain episodes")
    return episodes


def scene_name(scene_id: str) -> str:
    parts = scene_id.split("/")
    if len(parts) >= 2 and parts[0] == "mp3d":
        return parts[1]
    if scene_id.endswith(".glb"):
        return Path(scene_id).stem
    return scene_id


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "opennav_preflight.json"
    md_path = output_dir / "opennav_preflight.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Open-Nav Installation Preflight",
        "",
        f"Status: **{report['status']}**",
        f"Generated: `{report['generated_at']}`",
        f"Open-Nav root: `{report['opennav_root']}`",
        f"Open-Nav commit: `{report['opennav_commit']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "| :--- | :---: | :--- |",
    ]
    for check in report["checks"]:
        status = "PASS" if check["ok"] else "FAIL"
        detail = str(check["detail"]).replace("\n", " ")
        lines.append(f"| {check['name']} | {status} | `{detail}` |")

    lines.extend(["", "## Dataset", ""])
    dataset = report["dataset"]
    for key, value in dataset.items():
        if key != "missing_scenes":
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "Missing MP3D scene examples:", "", "```text"])
    lines.append("\n".join(dataset.get("missing_scenes", [])[:30]) or "none")
    lines.extend(["```", "", "## Blockers", ""])
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append(f"- {blocker}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Run Command After PASS",
            "",
            "```bash",
            "cd /home/xiaotian/vla/Open-Nav",
            (
                "PYTHONPATH=/home/xiaotian/vla/Open-Nav:"
                "/home/xiaotian/vla/habitat-lab-v0.1.7 "
                "conda run -n navila-eval bash run_OpenNav.bash"
            ),
            "```",
            "",
            "## Notes",
            "",
            "- This report does not download gated SpatialBot3B or licensed MP3D assets.",
            "- Open-Nav official SR/SPL requires MP3D simulator rollouts; any unresolved prerequisite keeps this diagnostic-only.",
            "- `OPENAI_API_KEY` or a configured local OpenAI-compatible LLM endpoint is required for actual inference.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local Open-Nav install.")
    parser.add_argument("--opennav-root", type=Path, default=DEFAULT_OPENNAV_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    root = args.opennav_root.resolve()
    dataset = root / "data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/OpenNav_R2R-CE_100_bertidx.json.gz"
    gt = root / "data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/val_unseen_gt.json.gz"
    scene_root = root / "data/scene_datasets/mp3d"
    ddppo = root / "data/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth"
    waypoint = root / "waypoint_prediction/checkpoints/check_val_best_avg_wayscore"
    ram_ckpt = root / "recognize_anything/pretrained/ram_swin_large_14m.pth"
    spatialbot = root / "SpatialBot3B"

    checks: list[Check] = [
        file_check("Open-Nav root", root),
        file_check("run.py", root / "run.py"),
        file_check("run_OpenNav.yaml", root / "run_OpenNav.yaml"),
        file_check("run_OpenNav.bash", root / "run_OpenNav.bash"),
        file_check("OpenNav_R2R-CE_100 dataset", dataset),
        file_check("OpenNav val_unseen GT", gt),
        file_check("DDPPO depth encoder", ddppo),
        torch_checkpoint_check("waypoint predictor checkpoint", waypoint),
        file_check("RAM source tree", root / "recognize_anything/ram"),
        torch_checkpoint_check("RAM checkpoint", ram_ckpt),
        file_check("SpatialBot3B directory", spatialbot),
        file_check("SpatialBot3B config", spatialbot / "config.json"),
        file_check("SpatialBot3B code: configuration_bunny_phi.py", spatialbot / "configuration_bunny_phi.py"),
        file_check("SpatialBot3B code: modeling_bunny_phi.py", spatialbot / "modeling_bunny_phi.py"),
        file_check("SpatialBot3B weights index", spatialbot / "model.safetensors.index.json"),
        file_check("SpatialBot3B shard 1", spatialbot / "model-00001-of-00002.safetensors"),
        file_check("SpatialBot3B shard 2", spatialbot / "model-00002-of-00002.safetensors"),
        Check("OPENAI_API_KEY", bool(os.environ.get("OPENAI_API_KEY")), "set" if os.environ.get("OPENAI_API_KEY") else "missing"),
    ]

    for module in (
        "habitat",
        "habitat_sim",
        "habitat_baselines",
        "torch",
        "transformers",
        "openai",
        "tenacity",
        "jsonlines",
        "pytorch_transformers",
        "recognize_anything.ram",
    ):
        checks.append(import_check(module))

    dataset_info: dict[str, Any] = {
        "dataset_file": str(dataset),
        "episode_count": 0,
        "unique_scene_count": 0,
        "mp3d_glb_present": 0,
        "mp3d_navmesh_present": 0,
        "missing_scenes": [],
    }
    if dataset.exists():
        try:
            episodes = load_episodes(dataset)
            scenes = sorted({scene_name(str(ep.get("scene_id", ""))) for ep in episodes})
            missing = []
            glb_present = 0
            navmesh_present = 0
            for scene in scenes:
                if (scene_root / scene / f"{scene}.glb").exists():
                    glb_present += 1
                else:
                    missing.append(scene)
                if (scene_root / scene / f"{scene}.navmesh").exists():
                    navmesh_present += 1
            dataset_info.update(
                {
                    "episode_count": len(episodes),
                    "unique_scene_count": len(scenes),
                    "mp3d_glb_present": f"{glb_present}/{len(scenes)}",
                    "mp3d_navmesh_present": f"{navmesh_present}/{len(scenes)}",
                    "missing_scenes": missing,
                }
            )
            checks.append(Check("OpenNav dataset parse", True, f"{len(episodes)} episodes, {len(scenes)} scenes"))
            checks.append(Check("OpenNav MP3D .glb coverage", glb_present == len(scenes), f"{glb_present}/{len(scenes)}"))
            checks.append(Check("OpenNav MP3D .navmesh coverage", navmesh_present == len(scenes), f"{navmesh_present}/{len(scenes)}"))
        except Exception as exc:
            checks.append(Check("OpenNav dataset parse", False, repr(exc)))

    blockers = [f"{check.name}: {check.detail}" for check in checks if not check.ok]
    report = {
        "status": "PASS" if not blockers else "BLOCKED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "opennav_root": str(root),
        "opennav_commit": git_commit(root),
        "checks": [asdict(check) for check in checks],
        "dataset": dataset_info,
        "blockers": blockers,
    }
    json_path, md_path = write_report(report, args.output_dir)
    print(f"[preflight] {report['status']}")
    print(f"[write] {json_path}")
    print(f"[write] {md_path}")
    if blockers:
        for blocker in blockers:
            print(f"[blocker] {blocker}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
