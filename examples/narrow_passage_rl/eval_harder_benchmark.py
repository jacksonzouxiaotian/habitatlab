#!/usr/bin/env python3
"""Evaluate methods on the harder procedural benchmark (v2).

Runs six ablation variants across seven corridor types and prints a table
suitable for paper_table_harder.{md,tex}.

Methods
-------
  rule_baseline       — forward + align-to-goal (APF-style, no memory)
  geometry_fsm        — geometry-only FSM (ALIGN/COMMIT/EXPLORE/RECOVER)
  fsm_no_recovery     — FSM without RECOVER mode
  fsm_no_alignment    — FSM with only heading correction, no lateral mode
  fsm_local_memory    — FSM + episode-local PassageFailureMemory (v1 memory)
  fsm_cross_memory    — FSM + CrossEpisodeMemory (proposed contribution)

Corridor types (each run separately, then averaged):
  straight, l_shaped, s_shaped, narrow_exit, narrow_entry, asymmetric, false_feasible

Usage
-----
    python examples/narrow_passage_rl/eval_harder_benchmark.py --episodes 500
    python examples/narrow_passage_rl/eval_harder_benchmark.py \
        --methods geometry_fsm fsm_cross_memory \
        --corridor-types l_shaped false_feasible \
        --episodes 200
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import CorridorType, HarderNarrowPassageEnv
from failure_memory import FailureMemoryConfig, PassageFailureMemory
from cross_episode_memory import CrossEpisodeMemory, MemoryConfig, FSMMode

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"


# ── Action helpers ────────────────────────────────────────────────────────────

def _act(vx, wz):
    return np.array([vx, wz], dtype=np.float32)


# ── Controllers ───────────────────────────────────────────────────────────────

def rule_baseline_action(obs):
    """APF-style: always forward + proportional heading correction."""
    he = float(obs[10])
    return _act(0.25, float(np.clip(he * 1.5, -0.8, 0.8)))


def _wz(obs, gain=1.0):
    he = float(obs[10])
    lat = float(obs[11])
    return float(np.clip((0.9 * he - 1.2 * lat) * gain, -0.8, 0.8))


def _follow_space_mode(obs):
    """Detect corner zone from sector depths; return (mode, action) or None.

    FOLLOW_SPACE activates when one near sector is very short while the other
    is open — a signature of an approaching L/S corridor turn.  It suppresses
    ALIGN (which would spin in place) and instead:
      - advances forward while d_cn is still open (pre-turn phase)
      - begins turning toward the inner corner direction once d_cn closes (turn phase)

    The inner corner being short signals the NEW path direction: turn toward it.
    """
    d_ln, d_cn, d_rn = float(obs[0]), float(obs[1]), float(obs[2])
    body_margin = float(obs[9])

    # Only activate inside the corridor (body_margin at default=4.82 means approach zone).
    if body_margin > 0.5:
        return None

    # Sector asymmetry test: one side tight (corner), other side open
    min_side = min(d_ln, d_rn)
    max_side = max(d_ln, d_rn)
    asymmetry = max_side - min_side

    if asymmetry < 2.0 or min_side > 0.30:
        return None   # symmetric or no tight side — not a corner

    # At an L-turn corner, the forward ray (d_cn) hits the outer wall at roughly
    # the same depth as the tight side (d_cn ≈ min_side).  A false positive from
    # an oblique approach to a straight wall has d_cn >> min_side (forward is
    # open, only one lateral sensor is blocked).  Suppress FOLLOW_SPACE when
    # forward is clearly more open than the tight side.
    if d_cn > 1.5 * min_side:
        return None

    # Turn direction = toward the inner (short) side
    turn_dir = -1.0 if d_ln <= d_rn else 1.0

    # Progressive turn: zero while forward is open, ramps up as d_cn drops
    turn_blend = float(np.clip((0.65 - d_cn) / 0.65, 0.0, 1.0))
    wz = turn_dir * 0.65 * turn_blend

    return "FOLLOW_SPACE", _act(0.10, wz)


class TurnCommitFSM:
    """Stateful FSM that commits to L/S-turn once FOLLOW_SPACE triggers.

    Problem: FOLLOW_SPACE turns the robot ~13° then exits because depth-sensor
    asymmetry collapses (both rays see the junction's outer wall).  The centering
    correction then fights the turn, causing oscillation near the junction.

    Fix: once FOLLOW_SPACE fires, lock the turn direction and apply a constant
    (vx=0.08, wz=±0.60) until the robot enters the new corridor segment, detected
    by `|he|` growing past 45° then falling back below 0.35 rad (path-frame
    assignment switches to the new segment).  Timeout at 60 steps as a safety net.
    """

    def __init__(self, variant="full",
                 local_mem: Optional[PassageFailureMemory] = None,
                 cross_mem: Optional[CrossEpisodeMemory] = None):
        self.variant = variant
        self.local_mem = local_mem
        self.cross_mem = cross_mem
        self._reset_turn()

    def reset(self):
        self._reset_turn()

    def _reset_turn(self, cooldown: int = 0):
        self._in_turn = False
        self._turn_dir = 0.0
        self._steps_in_turn = 0
        self._peak_abs_he = 0.0
        self._min_bm = float("inf")  # min body_margin seen during this turn
        self._cooldown = cooldown  # steps to suppress FOLLOW_SPACE after a turn

    def step(self, obs):
        """Returns (action, mode_str)."""
        if self._in_turn:
            return self._continue_turn(obs)

        # Cooldown: don't allow FOLLOW_SPACE to retrigger right after a committed
        # turn completes.  The robot needs a few steps to stabilise heading in the
        # new arm before we trust the asymmetry sensor again.
        if self._cooldown > 0:
            self._cooldown -= 1
            action, mode = fsm_action(obs, self.variant, self.local_mem, self.cross_mem)
            if mode == "FOLLOW_SPACE":
                # Override: apply centering correction instead of committing a new turn.
                he = float(obs[10])
                lat = float(obs[11])
                wz = float(np.clip(0.9 * he - 1.2 * lat, -0.80, 0.80))
                return _act(0.12, wz), "COMMIT"
            return action, mode

        action, mode = fsm_action(obs, self.variant, self.local_mem, self.cross_mem)
        if mode == "FOLLOW_SPACE":
            wz = float(action[1])
            self._in_turn = True
            self._turn_dir = 1.0 if wz >= 0 else -1.0
            self._steps_in_turn = 0
            self._peak_abs_he = abs(float(obs[10]))
        return action, mode

    def _continue_turn(self, obs):
        he = float(obs[10])
        body_margin = float(obs[9])
        self._steps_in_turn += 1
        self._peak_abs_he = max(self._peak_abs_he, abs(he))
        self._min_bm = min(self._min_bm, body_margin)

        # Exit: peaked > 30° AND heading is small again → entered the new segment.
        # Threshold π/6 (not π/4) so seeds where peak is ~44° don't spin a full
        # circle before the exit condition fires.
        # Natural exit always gets cooldown=8: the robot just entered a new arm and
        # needs time to stabilise before FOLLOW_SPACE fires again.
        # For asymmetric/narrow_entry false positives, the path frame never switches
        # (straight corridor), so abs(he) never drops below 0.35 — those exits go
        # through the bm>0.5 or timeout paths, which keep cooldown=0.
        if self._peak_abs_he > math.pi / 6 and abs(he) < 0.35:
            self._reset_turn(cooldown=8)
            return fsm_action(obs, self.variant, self.local_mem, self.cross_mem)

        # Exit: open area (past the junction turn region).
        if body_margin > 0.5:
            self._reset_turn(cooldown=0)
            return fsm_action(obs, self.variant, self.local_mem, self.cross_mem)

        # Exit: timeout.
        if self._steps_in_turn > 60:
            self._reset_turn(cooldown=0)
            return fsm_action(obs, self.variant, self.local_mem, self.cross_mem)

        # Slow forward + committed turn.  Reduce top-tier vx from 0.08→0.06 to
        # avoid clipping the outer L-junction wall on wide-arm approaches.
        if body_margin < 0.10:
            vx = 0.04
        else:
            vx = 0.06
        wz = self._turn_dir * 0.60
        return _act(vx, wz), "FOLLOW_SPACE"


def fsm_action(obs, variant="full", local_mem: Optional[PassageFailureMemory] = None,
               cross_mem: Optional[CrossEpisodeMemory] = None,
               max_recover_steps=25, _env=None):
    """Geometry-FSM step.  Returns (action, mode_str)."""
    he = float(obs[10])
    lat = float(obs[11])
    stuck = float(obs[15])
    collision_flag = float(obs[16]) > 0.5

    dist = float(obs[12])
    if dist < 0.30:
        return _act(0.0, 0.0), "STOP"

    # FSM mode from cross-episode memory
    if cross_mem is not None:
        cm_mode = cross_mem.fsm_mode(obs)
        if cm_mode == FSMMode.REJECT:
            return _act(-0.10, 0.0), "REJECT"
        cautious = cm_mode == FSMMode.CAUTIOUS
    else:
        cautious = False

    speed_fwd = 0.15 if cautious else 0.20
    speed_explore = 0.05 if cautious else 0.08

    if variant == "no_alignment":
        return _act(speed_fwd, float(np.clip(he * 0.9, -0.8, 0.8))), "COMMIT_NOALIGN"

    # FOLLOW_SPACE: sector-depth corner detection takes priority over ALIGN.
    # Only active for variants that use alignment (i.e. not no_alignment).
    fs = _follow_space_mode(obs)
    if fs is not None:
        return fs[1], fs[0]

    # Local memory overrides
    if local_mem is not None and local_mem.should_recover(obs):
        mode = "RECOVER"
    elif abs(he) > 0.7 and float(obs[9]) > 0.3:
        # Large heading error AND outside/wide corridor → full ALIGN (spin in place).
        mode = "ALIGN"
    elif abs(he) > 0.7:
        # Large heading error inside a NARROW corridor: spinning toward the final goal
        # would drive the robot into the wall (straight-line goal ≠ local path direction).
        # Instead use corridor-following: go slow and correct laterally only.
        mode = "CORRIDOR_FOLLOW"
    elif collision_flag or stuck > 0.80:
        mode = "RECOVER" if variant not in ("no_recovery",) else "COMMIT"
    elif abs(he) > 0.45 or abs(lat) > 0.25:
        mode = "EXPLORE"
    else:
        mode = "COMMIT"

    # Inside a narrow corridor, replace goal-heading steering with local corridor
    # navigation (final goal heading ≠ local corridor direction for L/S corridors).
    in_narrow = float(obs[9]) < 0.30   # body_margin gate
    if in_narrow:
        cl_v, cr_v = float(obs[6]), float(obs[7])
        d_ln_v, d_cn_v, d_rn_v = float(obs[0]), float(obs[1]), float(obs[2])

        # (A) Lateral centering: balance left/right wall clearances.
        wz_center = float(np.clip(
            -1.6 * (cl_v - cr_v) / max(cl_v + cr_v, 0.05), -0.8, 0.8
        ))

        # (B) Gap-following: when d_cn is below the sector average the robot is
        # heading sideways.  Steer toward the more open sector direction.
        # FOLLOW_SPACE takes priority at turns, so this runs during straight segments.
        if d_cn_v < 0.5 * (d_ln_v + d_rn_v):
            wz_gap = float(np.clip(
                1.2 * (d_rn_v - d_ln_v) / max(d_rn_v + d_ln_v, 0.1), -0.8, 0.8
            ))
        else:
            wz_gap = 0.0

        wz_narrow = float(np.clip(wz_center + wz_gap, -0.8, 0.8))
    else:
        wz_narrow = None

    # Forward depth gate: if d_cn is very short the robot is heading into a wall.
    # Back up so the heading correction can rotate to a better direction first.
    # Threshold 0.20 avoids oscillation with the ~0.24-0.28 typical near-wall readings.
    d_cn_val = float(obs[1])
    if in_narrow and d_cn_val < 0.20:
        vx_in_narrow = -0.06   # back up
    else:
        vx_in_narrow = None   # use mode default

    if mode == "ALIGN":
        action = _act(0.0, float(np.clip(he * 1.8, -0.8, 0.8)))
    elif mode == "CORRIDOR_FOLLOW":
        # Inside narrow corridor with misleading final-goal heading error:
        # use combined centering + gap-following instead of spinning toward goal.
        wz_cf = wz_narrow if in_narrow else float(np.clip(-2.0 * lat, -0.8, 0.8))
        vx_cf = vx_in_narrow if vx_in_narrow is not None else 0.08
        action = _act(vx_cf, wz_cf)
    elif mode == "COMMIT":
        wz = wz_narrow if in_narrow else _wz(obs, 1.0 if not cautious else 1.5)
        vx = vx_in_narrow if vx_in_narrow is not None else speed_fwd
        action = _act(vx, wz)
    elif mode == "EXPLORE":
        wz = wz_narrow if in_narrow else _wz(obs, 1.6 if not cautious else 2.0)
        vx = vx_in_narrow if vx_in_narrow is not None else speed_explore
        action = _act(vx, wz)
    elif mode == "RECOVER":
        action = _act(-0.12, float(np.clip(he * 0.4, -0.8, 0.8)))
    else:
        action = _act(0.0, 0.0)

    return action, mode


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env: HarderNarrowPassageEnv, method: str,
                local_mem: Optional[PassageFailureMemory],
                cross_mem: Optional[CrossEpisodeMemory],
                max_steps: int,
                entry_obs_ref: list,
                agent: Optional["TurnCommitFSM"] = None) -> dict:
    obs, _ = env.reset()
    entry_obs = obs.copy()
    entry_obs_ref.clear()
    entry_obs_ref.append(entry_obs)

    if agent is not None:
        agent.reset()

    done = False
    steps = 0
    any_collision = False
    min_bm = float("inf")
    near_collision = False

    if cross_mem is not None:
        cross_mem.reset_local()

    while not done and steps < max_steps:
        bm = float(obs[9])
        if bm < min_bm:
            min_bm = bm
        if bm < 0.05:
            near_collision = True
        if float(obs[16]) > 0.5:
            any_collision = True

        if method == "rule_baseline":
            action = rule_baseline_action(obs)
        elif agent is not None:
            action, _ = agent.step(obs)
        else:
            variant = {
                "geometry_fsm": "full",
                "fsm_no_recovery": "no_recovery",
                "fsm_no_alignment": "no_alignment",
                "fsm_local_memory": "full",
                "fsm_cross_memory": "full",
            }.get(method, "full")
            action, _ = fsm_action(obs, variant, local_mem, cross_mem)

        obs, _, term, trunc, info = env.step(action)
        done = term or trunc

        if local_mem is not None and any_collision:
            local_mem.add_failure(obs)

        steps += 1

    if not np.isfinite(min_bm):
        min_bm = float(obs[9])

    return {
        "success": float(info.get("success", 0.0)),
        "collision": float(any_collision),
        "near_collision": float(near_collision),
        "min_body_margin": min_bm,
        "steps": steps,
        "passable": info.get("passable", True),
        "corridor_type": info.get("corridor_type", "unknown"),
        "passage_width": info.get("passage_width", float(obs[8])),
    }


# ── Per-method runner ─────────────────────────────────────────────────────────

def eval_method(method: str, ctypes: List[str], n_episodes: int,
                seed: int = 42) -> List[dict]:
    env_cfg = {"corridor_types": ctypes, "seed": seed}
    env = HarderNarrowPassageEnv(env_cfg)

    local_mem = None
    cross_mem = None
    agent = None

    if method == "fsm_local_memory":
        local_mem = PassageFailureMemory()
    elif method == "fsm_cross_memory":
        cross_mem = CrossEpisodeMemory()

    # Use TurnCommitFSM for all FSM variants (handles L/S turn commitment)
    if method != "rule_baseline":
        variant = {
            "geometry_fsm": "full",
            "fsm_no_recovery": "no_recovery",
            "fsm_no_alignment": "no_alignment",
            "fsm_local_memory": "full",
            "fsm_cross_memory": "full",
        }.get(method, "full")
        agent = TurnCommitFSM(variant=variant, local_mem=local_mem,
                              cross_mem=cross_mem)

    all_stats = []
    entry_obs_ref = []
    for ep_idx in range(n_episodes):
        if method == "fsm_local_memory" and local_mem is not None:
            local_mem.reset()

        stat = run_episode(env, method, local_mem, cross_mem,
                           max_steps=env.max_steps, entry_obs_ref=entry_obs_ref,
                           agent=agent)
        stat["episode_idx"] = ep_idx
        stat["method"] = method
        all_stats.append(stat)

        # Update cross-episode memory
        if cross_mem is not None and entry_obs_ref:
            cross_mem.record_episode(
                entry_obs_ref[0],
                bool(stat["success"] > 0.5),
                stat["passage_width"],
                steps=stat["steps"],
                corridor_type=stat["corridor_type"],
            )

        if ep_idx % 100 == 0 or ep_idx < 5:
            sr = np.mean([s["success"] for s in all_stats])
            print(f"  [{method}] ep {ep_idx:3d} ctype={stat['corridor_type']:14s} "
                  f"success={stat['success']:.0f}  rolling_sr={sr:.3f}", flush=True)

    return all_stats


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_stats: List[dict], methods: List[str], ctypes: List[str]):
    print(f"\n{'Method':22s}  {'SR':>6}  {'Col':>6}  {'NarCol':>7}  "
          f"{'MinBM':>6}  ", end="")
    for ct in ctypes:
        short = ct[:6]
        print(f"  {short:>6}", end="")
    print()
    print("-" * (22 + 6 + 6 + 7 + 6 + 4 + len(ctypes) * 8))

    for method in methods:
        rows = [s for s in all_stats if s["method"] == method]
        if not rows:
            continue
        sr = np.mean([s["success"] for s in rows])
        cr = np.mean([s["collision"] for s in rows])
        nc = np.mean([s["near_collision"] for s in rows])
        bm = np.mean([s["min_body_margin"] for s in rows])
        print(f"{method:22s}  {sr:.3f}  {cr:.3f}  {nc:.4f}  {bm:+.3f}  ", end="")
        for ct in ctypes:
            grp = [s for s in rows if s["corridor_type"] == ct]
            dsr = np.mean([s["success"] for s in grp]) if grp else float("nan")
            print(f"  {dsr:.3f}", end="")
        print()


def write_csv(all_stats, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not all_stats:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_stats[0].keys()))
        w.writeheader()
        w.writerows(all_stats)
    print(f"[write] {path}")


def write_summary_rows(all_stats, methods, ctypes, path):
    """Write one CSV row per method for results_rl_summary.csv compatibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in methods:
        stats = [s for s in all_stats if s["method"] == method]
        if not stats:
            continue
        row = {
            "case": f"v2_{method}",
            "success_rate": round(np.mean([s["success"] for s in stats]), 4),
            "collision_rate": round(np.mean([s["collision"] for s in stats]), 4),
            "near_collision_rate": round(np.mean([s["near_collision"] for s in stats]), 4),
            "avg_min_clearance": round(np.mean([s["min_body_margin"] for s in stats]), 4),
        }
        for ct in ctypes:
            grp = [s for s in stats if s["corridor_type"] == ct]
            row[f"sr_{ct}"] = round(np.mean([s["success"] for s in grp]), 4) if grp else ""
        rows.append(row)
    if rows:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[write] {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["rule_baseline", "geometry_fsm", "fsm_no_recovery",
                             "fsm_no_alignment", "fsm_local_memory", "fsm_cross_memory"])
    ap.add_argument("--corridor-types", nargs="+",
                    default=["straight", "l_shaped", "s_shaped",
                             "narrow_exit", "narrow_entry", "asymmetric", "false_feasible"])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-csv",
                    type=Path, default=RESULTS / "harder_benchmark_episodes.csv")
    ap.add_argument("--output-summary",
                    type=Path, default=RESULTS / "harder_benchmark_summary.csv")
    args = ap.parse_args()

    ctypes = args.corridor_types
    all_stats = []

    for method in args.methods:
        print(f"\n=== {method} ===")
        stats = eval_method(method, ctypes, args.episodes, args.seed)
        all_stats.extend(stats)
        # Quick per-type breakdown
        for ct in ctypes:
            grp = [s for s in stats if s["corridor_type"] == ct]
            if grp:
                dsr = np.mean([s["success"] for s in grp])
                print(f"  {ct:14s}: SR={dsr:.3f}  n={len(grp)}")

    print_summary(all_stats, args.methods, ctypes)
    write_csv(all_stats, args.output_csv)
    write_summary_rows(all_stats, args.methods, ctypes, args.output_summary)


if __name__ == "__main__":
    main()
