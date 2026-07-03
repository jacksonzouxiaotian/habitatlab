#!/usr/bin/env python3
"""Record Habitat NarrowPassageNav-v0 episodes as MP4 videos.

The script reuses the existing Habitat evaluators' controller/model logic and
adds paper-friendly overlays for success, safety, and narrow-passage state.

Examples:
    conda run -n habitat python examples/narrow_passage_rl/record_habitat_video.py \
        --method geometry_fsm --split val --episode-index 0

    conda run -n habitat python examples/narrow_passage_rl/record_habitat_video.py \
        --method ppo_sb3 \
        --model data/narrow_passage_sb3_v2_ppo/ppo_narrow_passage_v2.zip \
        --split val --episode-index 0 --save-keyframes
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

FEATURE_DIM = 19
HIDDEN_SIZE = 256


def make_env(data_path: str, split: str):
    import habitat

    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_habitat_test.yaml",
        overrides=[
            "habitat/task=narrow_passage",
            f"habitat.dataset.data_path={data_path}",
            f"habitat.dataset.split={split}",
            "habitat.dataset.type=PointNav-v1",
            "habitat.simulator.habitat_sim_v0.allow_sliding=False",
        ],
    )
    return habitat.Env(config=config)


def _episode_id_for_json_index(data_path: str, episode_index: int) -> str:
    with gzip.open(data_path, "rt") as f:
        data = json.load(f)
    episodes = data.get("episodes", [])
    if episode_index < 0 or episode_index >= len(episodes):
        raise IndexError(f"episode-index {episode_index} out of range for {len(episodes)} JSON episodes")
    return str(episodes[episode_index].get("episode_id", episode_index))


def _reset_to_episode(env, data_path: str, episode_index: int) -> Dict[str, Any]:
    """Reset to the requested dataset episode index.

    Habitat's default episode iterator can shuffle or group episodes by scene,
    so repeatedly calling reset() is not guaranteed to map CLI index N to
    dataset episode N.  Setting current_episode first uses Env's force-changed
    path and preserves the requested episode on reset().
    """
    if episode_index < 0:
        raise ValueError("--episode-index must be >= 0")
    total = env.number_of_episodes
    if total and episode_index >= total:
        raise IndexError(f"episode-index {episode_index} out of range for {total} episodes")

    target_id = _episode_id_for_json_index(data_path, episode_index)
    matching = [ep for ep in env.episodes if str(ep.episode_id) == target_id]
    if not matching:
        raise KeyError(f"Could not find episode_id={target_id} in Habitat env episodes")

    env.current_episode = matching[0]
    return env.reset()


def _episode_meta(env) -> Dict[str, Any]:
    episode = env.current_episode
    info = episode.info if hasattr(episode, "info") and episode.info else {}
    return {
        "episode_id": str(episode.episode_id),
        "scene_id": str(episode.scene_id).split("/")[-2],
        "difficulty": str(info.get("difficulty", "?")),
        "body_margin": float(info.get("body_margin", float("nan"))),
    }


def _features(obs: Dict[str, Any]) -> np.ndarray:
    return np.array(
        obs.get("narrow_passage_features", np.zeros(FEATURE_DIM, dtype=np.float32)),
        dtype=np.float32,
    )


def _metrics(env, min_clearance: float, any_collision: bool) -> Dict[str, float]:
    m = env.get_metrics()
    return {
        "success": float(m.get("narrow_passage_success", 0.0)),
        "collision": float(any_collision),
        "stuck": float(m.get("narrow_passage_stuck", 0.0)),
        "min_clearance": float(min_clearance) if np.isfinite(min_clearance) else 0.0,
    }


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    valid = arr[arr > 1e-4]
    vmax = float(np.percentile(valid, 95)) if valid.size else 1.0
    vmax = max(vmax, 1e-3)
    x = np.clip(arr / vmax, 0.0, 1.0)

    # Lightweight blue -> cyan -> yellow heatmap, no matplotlib dependency.
    r = np.clip(2.0 * x - 0.25, 0.0, 1.0)
    g = np.clip(1.6 * x, 0.0, 1.0)
    b = np.clip(1.2 - 1.4 * x, 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _rgb_frame(obs: Dict[str, Any]) -> np.ndarray:
    if "rgb" not in obs:
        raise KeyError("Observation has no 'rgb' frame")
    rgb = np.asarray(obs["rgb"])
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return np.ascontiguousarray(rgb.astype(np.uint8))


def _obs_to_frame(obs: Dict[str, Any], view: str) -> np.ndarray:
    if view in ("auto", "rgb") and "rgb" in obs:
        return _rgb_frame(obs)
    if view == "rgb" and "rgb" not in obs:
        raise KeyError("Observation has no 'rgb' frame")
    if view == "depth":
        if "depth" not in obs:
            raise KeyError("Observation has no 'depth' frame")
        return _depth_to_rgb(np.asarray(obs["depth"]))
    if view == "rgb_depth":
        if "rgb" not in obs or "depth" not in obs:
            raise KeyError("rgb_depth view requires both 'rgb' and 'depth'")
        rgb = _rgb_frame(obs)
        depth = _depth_to_rgb(np.asarray(obs["depth"]))
        if depth.shape[:2] != rgb.shape[:2]:
            depth = np.asarray(Image.fromarray(depth).resize((rgb.shape[1], rgb.shape[0])))
        divider = np.full((rgb.shape[0], 4, 3), 255, dtype=np.uint8)
        return np.concatenate([rgb, divider, depth], axis=1)
    if "depth" in obs:
        return _depth_to_rgb(np.asarray(obs["depth"]))
    raise KeyError("Observation has neither 'rgb' nor 'depth'; cannot render video frame")


def _fmt(value: float, ndigits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{ndigits}f}"


def _overlay(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    line_h = 13
    pad = 6
    box_w = min(image.width, 238)
    box_h = pad * 2 + line_h * len(lines)
    draw.rectangle((0, 0, box_w, box_h), fill=(0, 0, 0, 170))
    y = pad
    for line in lines:
        draw.text((pad, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h
    return np.asarray(image, dtype=np.uint8)


def _annotate(
    obs: Dict[str, Any],
    args,
    meta: Dict[str, Any],
    step: int,
    mode: str,
    features: np.ndarray,
    metrics: Dict[str, float],
) -> np.ndarray:
    heading_error = float(features[10]) if features.size > 10 else 0.0
    lateral_offset = float(features[11]) if features.size > 11 else 0.0
    body_margin = float(features[9]) if features.size > 9 else meta["body_margin"]
    lines = [
        f"method: {args.method}",
        f"episode: {meta['episode_id']}",
        f"scene: {meta['scene_id']}",
        f"difficulty: {meta['difficulty']}",
        f"step: {step}  mode: {mode}",
        f"body_margin: {_fmt(body_margin)}",
        f"min_clearance: {_fmt(metrics['min_clearance'])}",
        f"heading_error: {_fmt(heading_error)}",
        f"lateral_offset: {_fmt(lateral_offset)}",
        (
            f"success: {int(metrics['success'])}  "
            f"collision: {int(metrics['collision'])}  stuck: {int(metrics['stuck'])}"
        ),
    ]
    if (
        abs(args.heading_perturb_deg) > 1e-6
        or abs(args.lateral_perturb_m) > 1e-6
        or abs(args.start_distance_shift_m) > 1e-6
    ):
        lines.append(
            "stress: "
            f"yaw={args.heading_perturb_deg:.0f}deg "
            f"lat={args.lateral_perturb_m:.2f}m "
            f"dist={args.start_distance_shift_m:.2f}m"
        )
    return _overlay(_obs_to_frame(obs, args.view), lines)


def _safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text)


def _load_episode_candidates(data_path: str) -> List[Dict[str, Any]]:
    with gzip.open(data_path, "rt") as f:
        data = json.load(f)

    candidates = []
    for idx, episode in enumerate(data.get("episodes", [])):
        start = episode.get("start_position", [0.0, 0.0, 0.0])
        goals = episode.get("goals", [])
        goal = goals[0].get("position", [0.0, 0.0, 0.0]) if goals else [0.0, 0.0, 0.0]
        distance = float(np.linalg.norm(np.asarray(goal)[[0, 2]] - np.asarray(start)[[0, 2]]))
        info = episode.get("info", {})
        candidates.append({
            "index": idx,
            "episode_id": str(episode.get("episode_id", idx)),
            "scene_id": str(episode.get("scene_id", "")).split("/")[-2],
            "difficulty": str(info.get("difficulty", "?")),
            "body_margin": float(info.get("body_margin", float("nan"))),
            "distance": distance,
        })
    return candidates


def _filter_candidates(candidates: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    rows = candidates
    if args.difficulty:
        rows = [r for r in rows if r["difficulty"] == args.difficulty]
    if args.min_start_goal_distance > 0.0:
        rows = [r for r in rows if r["distance"] >= args.min_start_goal_distance]
    return rows


def _print_candidates(candidates: List[Dict[str, Any]], limit: int) -> None:
    print("index distance episode_id difficulty body_margin scene")
    for row in sorted(candidates, key=lambda r: r["distance"], reverse=True)[:limit]:
        print(
            f"{row['index']:4d}  {row['distance']:7.2f}  {row['episode_id']:20s}  "
            f"{row['difficulty']:6s}  {row['body_margin']:10.4f}  {row['scene_id']}"
        )


def select_episode_indices(data_path: str, args) -> List[int]:
    candidates = _load_episode_candidates(data_path)
    rows = _filter_candidates(candidates, args)
    if args.list_candidates:
        _print_candidates(rows, args.list_candidates)
        return []

    if args.select_longest:
        rows = sorted(rows, key=lambda r: r["distance"], reverse=True)
        if not rows:
            raise ValueError("No episodes match the requested distance/difficulty filters")
        return [int(r["index"]) for r in rows[:args.num_episodes]]

    return list(range(args.episode_index, args.episode_index + args.num_episodes))


def _make_velocity_action(lin_norm: float, ang_norm: float) -> dict:
    return {
        "action": "velocity_control",
        "action_args": {
            "linear_velocity": float(np.clip(lin_norm, -1.0, 1.0)),
            "angular_velocity": float(np.clip(ang_norm, -1.0, 1.0)),
        },
    }


def _yaw_from_rotation(rot) -> float:
    return math.atan2(
        2.0 * (rot.real * rot.y + rot.x * rot.z),
        1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z),
    )


def _yaw_to_quat(yaw: float) -> list:
    half = 0.5 * yaw
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _apply_heading_perturb(env, degrees: float) -> None:
    if abs(degrees) < 1e-6:
        return
    state = env.sim.get_agent_state()
    yaw = _yaw_from_rotation(state.rotation) + math.radians(degrees)
    env.sim.set_agent_state(state.position, _yaw_to_quat(yaw), reset_sensors=False)


def _apply_lateral_perturb(env, meters: float) -> None:
    if abs(meters) < 1e-6:
        return
    state = env.sim.get_agent_state()
    yaw = _yaw_from_rotation(state.rotation)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float32)
    pos = np.array(state.position, dtype=np.float32) + meters * right
    env.sim.set_agent_state(pos, state.rotation, reset_sensors=False)


def _apply_start_distance_shift(env, meters: float) -> None:
    """Move start along start-goal line; negative moves farther from goal."""
    if abs(meters) < 1e-6:
        return
    state = env.sim.get_agent_state()
    ap = np.array(state.position, dtype=np.float32)
    gp = np.array(env.current_episode.goals[0].position, dtype=np.float32)
    direction = gp - ap
    direction[1] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return
    pos = ap + meters * direction / norm
    env.sim.set_agent_state(pos, state.rotation, reset_sensors=False)


def _apply_demo_perturbations(env, args) -> None:
    _apply_heading_perturb(env, args.heading_perturb_deg)
    _apply_lateral_perturb(env, args.lateral_perturb_m)
    _apply_start_distance_shift(env, args.start_distance_shift_m)


def _refresh_sim_frames(env, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh visual sensors after manually perturbing the agent pose."""
    try:
        sim_obs = env.sim.get_sensor_observations()
    except Exception:
        return obs
    out = dict(obs)
    for key in ("rgb", "depth"):
        if key in sim_obs:
            out[key] = sim_obs[key]
    return out


class Controller:
    def act(self, env, obs: Dict[str, Any], features: np.ndarray, step: int) -> Tuple[dict, str]:
        raise NotImplementedError


class GeometryFSMController(Controller):
    def __init__(self):
        from eval_habitat_geometry_fsm import (
            _align_action,
            _commit_action,
            _compute_heading_error,
            _dist_to_goal,
            _explore_action,
            _recover_action,
        )

        self._align_action = _align_action
        self._commit_action = _commit_action
        self._compute_heading_error = _compute_heading_error
        self._dist_to_goal = _dist_to_goal
        self._explore_action = _explore_action
        self._recover_action = _recover_action

    def act(self, env, obs: Dict[str, Any], features: np.ndarray, step: int) -> Tuple[dict, str]:
        heading_error = self._compute_heading_error(env)
        dist_now = self._dist_to_goal(env)
        lateral_offset = float(features[11])
        stuck_score = float(features[15])
        collision_flag = float(features[16]) > 0.5

        if dist_now < 0.25:
            from eval_habitat_geometry_fsm import _make_action, _norm_ang, _norm_lin

            return _make_action(_norm_lin(0.0), _norm_ang(0.0)), "STOP"
        if collision_flag or stuck_score > 0.80:
            return self._recover_action(heading_error), "RECOVER"
        if abs(heading_error) > 0.7:
            return self._align_action(heading_error), "ALIGN"
        if abs(heading_error) > 0.45 or abs(lateral_offset) > 0.25:
            return self._explore_action(heading_error, lateral_offset), "EXPLORE"
        return self._commit_action(heading_error, lateral_offset), "COMMIT"


class APFGapController(Controller):
    def __init__(self):
        from eval_habitat_apf_gap import STOP_DIST, _make_action, apf_gap_act

        self._make_action = _make_action
        self._apf_gap_act = apf_gap_act
        self._stop_dist = STOP_DIST

    def act(self, env, obs: Dict[str, Any], features: np.ndarray, step: int) -> Tuple[dict, str]:
        depth = np.array(obs["depth"], dtype=np.float32)
        gps = np.array(obs["pointgoal_with_gps_compass"], dtype=np.float32)
        lin_n, ang_n = self._apf_gap_act(depth, gps)
        mode = "STOP" if float(gps[0]) < self._stop_dist else "APF_GAP"
        return self._make_action(lin_n, ang_n), mode


class SB3Controller(Controller):
    def __init__(self, model_path: Path, device: str):
        from stable_baselines3 import PPO, SAC, TD3

        name = str(model_path).lower()
        if "sac" in name:
            cls = SAC
        elif "td3" in name:
            cls = TD3
        else:
            cls = PPO
        self.model = cls.load(str(model_path), device=device)

    def act(self, env, obs: Dict[str, Any], features: np.ndarray, step: int) -> Tuple[dict, str]:
        action, _ = self.model.predict(features, deterministic=True)
        return _make_velocity_action(float(action[0]), float(action[1])), "PPO"


class HabitatPolicyController(Controller):
    def __init__(self, ckpt_path: Path, device: str):
        import torch
        from eval_habitat_ppo_policy import make_action_space, make_obs_space, obs_to_tensors
        from habitat_baselines.rl.ppo.narrow_passage_policy import NarrowPassagePolicy

        self.torch = torch
        self.device = torch.device(device)
        self.obs_to_tensors = obs_to_tensors
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.actor_critic = NarrowPassagePolicy.from_config(
            ckpt["config"],
            observation_space=make_obs_space(),
            action_space=make_action_space(),
        ).to(self.device)
        self.actor_critic.eval()
        self.actor_critic.load_state_dict(ckpt["state_dict"])
        self.rnn_hidden = torch.zeros(1, 1, HIDDEN_SIZE, device=self.device)
        self.masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        self.prev_actions = torch.zeros(1, 2, device=self.device)

    def act(self, env, obs: Dict[str, Any], features: np.ndarray, step: int) -> Tuple[dict, str]:
        with self.torch.no_grad():
            obs_t = self.obs_to_tensors(obs, self.device)
            action_data = self.actor_critic.act(
                obs_t, self.rnn_hidden, self.prev_actions, self.masks, deterministic=True
            )
            self.rnn_hidden = action_data.rnn_hidden_states
            self.masks = self.torch.ones(1, 1, dtype=self.torch.bool, device=self.device)
            act = action_data.actions[0]
            self.prev_actions = action_data.actions
            lin_norm = float(self.torch.clamp(act[0], -1.0, 1.0))
            ang_norm = float(self.torch.clamp(act[1], -1.0, 1.0))
        return _make_velocity_action(lin_norm, ang_norm), "PPO"


def make_controller(args) -> Controller:
    if args.method == "geometry_fsm":
        return GeometryFSMController()
    if args.method == "apf_gap":
        return APFGapController()
    if args.method == "ppo_sb3":
        return SB3Controller(args.model, args.device)
    if args.method == "ppo_policy":
        return HabitatPolicyController(args.ckpt, args.device)
    raise ValueError(f"Unknown method: {args.method}")


def write_video(frames: List[np.ndarray], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(str(output_path), fps=fps, quality=8, macro_block_size=1)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return
    except Exception as exc:
        try:
            from habitat.utils.visualizations.utils import images_to_video

            images_to_video(
                frames,
                str(output_path.parent),
                output_path.stem,
                fps=fps,
                quality=8,
                verbose=False,
            )
            return
        except Exception as fallback_exc:
            raise RuntimeError(
                "Could not write MP4. Install imageio[ffmpeg] in the habitat env."
            ) from fallback_exc


def save_keyframes(
    frames: List[np.ndarray],
    records: List[Dict[str, float]],
    keyframe_dir: Path,
) -> None:
    if not frames:
        return
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)

    before_idx = next(
        (
            i
            for i, r in enumerate(records)
            if r["step"] > 0 and (r["body_margin"] < 0.12 or r["distance"] < 0.75 * records[0]["distance"])
        ),
        max(0, int(0.25 * (n - 1))),
    )
    inside_idx = next(
        (
            i
            for i, r in enumerate(records)
            if r["body_margin"] < 0.05 or r["collision"] > 0.5 or r["stuck"] > 0.5
        ),
        max(0, int(0.60 * (n - 1))),
    )
    keyframes = [
        ("00_start.png", 0),
        ("01_before_entry.png", before_idx),
        ("02_inside_or_failure.png", inside_idx),
        ("03_final.png", n - 1),
    ]
    for name, idx in keyframes:
        Image.fromarray(frames[idx]).save(keyframe_dir / name)


def _distance_feature(features: np.ndarray) -> float:
    if features.size > 12 and np.isfinite(features[12]):
        return float(features[12])
    return 0.0


def record_one(env, args, episode_index: int) -> Path:
    data_path = args.data_path.format(split=args.split)
    obs = _reset_to_episode(env, data_path, episode_index)
    _apply_demo_perturbations(env, args)
    obs = _refresh_sim_frames(env, obs)
    controller = make_controller(args)
    meta = _episode_meta(env)

    frames: List[np.ndarray] = []
    records: List[Dict[str, float]] = []
    min_clearance = float("inf")
    any_collision = False
    done = False
    step = 0
    mode = "INIT"

    while not done and step < args.max_steps:
        features = _features(obs)
        bm = float(features[9]) if features.size > 9 else float("inf")
        min_clearance = min(min_clearance, bm)
        any_collision = any_collision or (features.size > 16 and float(features[16]) > 0.5)
        current_metrics = _metrics(env, min_clearance, any_collision)
        frames.append(_annotate(obs, args, meta, step, mode, features, current_metrics))
        records.append({
            "step": float(step),
            "body_margin": bm,
            "distance": _distance_feature(features),
            "collision": current_metrics["collision"],
            "stuck": current_metrics["stuck"],
        })

        action, mode = controller.act(env, obs, features, step)
        obs = env.step(action)
        step += 1
        if env.episode_over:
            done = True

    features = _features(obs)
    bm = float(features[9]) if features.size > 9 else float("inf")
    min_clearance = min(min_clearance, bm)
    any_collision = any_collision or (features.size > 16 and float(features[16]) > 0.5)
    final_metrics = _metrics(env, min_clearance, any_collision)
    final_mode = "SUCCESS" if final_metrics["success"] > 0.5 else "TIMEOUT_OR_FAILURE"
    frames.append(_annotate(obs, args, meta, step, final_mode, features, final_metrics))
    records.append({
        "step": float(step),
        "body_margin": bm,
        "distance": _distance_feature(features),
        "collision": final_metrics["collision"],
        "stuck": final_metrics["stuck"],
    })

    video_name = f"{args.method}_ep{episode_index:03d}_{_safe_name(meta['episode_id'])}.mp4"
    output_path = args.output_dir / video_name
    write_video(frames, output_path, args.fps)

    if args.save_keyframes:
        keyframe_dir = (
            Path("results/narrow_passage_rl/keyframes")
            / f"{args.method}_ep{episode_index:03d}"
        )
        save_keyframes(frames, records, keyframe_dir)
        print(f"[keyframes] {keyframe_dir}")

    print(
        f"[video] {output_path}  steps={step}  success={final_metrics['success']:.0f}  "
        f"collision={final_metrics['collision']:.0f}  stuck={final_metrics['stuck']:.0f}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["geometry_fsm", "ppo_sb3", "apf_gap", "ppo_policy"],
        default="geometry_fsm",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--data-path", default="data/datasets/narrow_passage/{split}/{split}.json.gz")
    parser.add_argument("--select-longest", action="store_true",
                        help="Record the longest start-goal episodes matching the filters")
    parser.add_argument("--min-start-goal-distance", type=float, default=0.0,
                        help="Filter candidate episodes by 2-D start-goal distance in metres")
    parser.add_argument("--difficulty", choices=["narrow", "normal", "wide"], default=None,
                        help="Optional candidate filter used with --select-longest/--list-candidates")
    parser.add_argument("--list-candidates", type=int, default=0,
                        help="Print the top-N longest candidate episodes and exit")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/narrow_passage_sb3_v2_ppo/ppo_narrow_passage_v2.zip"),
    )
    parser.add_argument("--ckpt", type=Path, default=Path("data/narrow_passage_checkpoints/latest.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("video_dir/narrow_passage_habitat"))
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--view", choices=["auto", "rgb", "depth", "rgb_depth"], default="auto",
                        help="Frame source; rgb_depth is best for paper demos of narrowness")
    parser.add_argument("--heading-perturb-deg", type=float, default=0.0,
                        help="Visualization stress: rotate start pose after reset")
    parser.add_argument("--lateral-perturb-m", type=float, default=0.0,
                        help="Visualization stress: shift start pose sideways in agent frame")
    parser.add_argument("--start-distance-shift-m", type=float, default=0.0,
                        help="Visualization stress: move start along start-goal line; negative = farther")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-keyframes", action="store_true")
    args = parser.parse_args()

    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be >= 1 for video recording")

    data_path = args.data_path.format(split=args.split)
    episode_indices = select_episode_indices(data_path, args)
    if not episode_indices:
        return

    for ep_idx in episode_indices:
        with make_env(data_path, args.split) as env:
            total = env.number_of_episodes
            if total and ep_idx >= total:
                raise IndexError(
                    f"requested episode-index {ep_idx}, but split has {total} episodes"
                )
            record_one(env, args, ep_idx)


if __name__ == "__main__":
    main()
