#!/usr/bin/env python3
"""Train BC or DAgger baselines from Geometry-FSM expert actions."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from eval_harder_benchmark import TurnCommitFSM
from procedural_env_v2 import HarderNarrowPassageEnv

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"


class ImitationPolicy(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_low, action_high):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        self.register_buffer("center", (self.action_low + self.action_high) * 0.5)
        self.register_buffer("scale", (self.action_high - self.action_low) * 0.5)

    def forward(self, obs):
        return self.center + self.scale * self.net(obs)


def load_dataset(path: Path):
    data = np.load(path, allow_pickle=True)
    return data["obs"].astype(np.float32), data["actions"].astype(np.float32)


def train_policy(policy, obs, actions, args):
    device = torch.device("cpu")
    policy.to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    act_t = torch.as_tensor(actions, dtype=torch.float32)
    n = len(obs)
    idxs = np.arange(n)
    for epoch in range(args.epochs):
        np.random.shuffle(idxs)
        losses = []
        for start in range(0, n, args.batch_size):
            mb = idxs[start : start + args.batch_size]
            pred = policy(obs_t[mb])
            loss = nn.functional.mse_loss(pred, act_t[mb])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        print(f"[train] epoch={epoch + 1}/{args.epochs} mse={np.mean(losses):.6f}")


def collect_dagger(policy, episodes: int, seed: int, max_steps: int):
    env = HarderNarrowPassageEnv({"seed": seed})
    new_obs = []
    new_actions = []
    ep_rows = []
    policy.eval()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert = TurnCommitFSM(variant="full")
        done = False
        steps = 0
        info = {}
        min_bm = float("inf")
        while not done and steps < max_steps:
            min_bm = min(min_bm, float(obs[9]))
            expert_action, _ = expert.step(obs)
            with torch.no_grad():
                policy_action = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            new_obs.append(obs.astype(np.float32))
            new_actions.append(expert_action.astype(np.float32))
            obs, _, term, trunc, info = env.step(policy_action.squeeze(0).numpy())
            done = term or trunc
            steps += 1
        ep_rows.append(
            {
                "episode": ep,
                "corridor_type": info.get("corridor_type", "unknown"),
                "success": float(info.get("success", 0.0)),
                "collision": float(info.get("collision", 0.0)),
                "steps": steps,
                "min_body_margin": min_bm,
            }
        )
    print(
        f"[dagger collect] episodes={episodes} transitions={len(new_obs)} "
        f"policy_SR={np.mean([r['success'] for r in ep_rows]):.3f}"
    )
    return np.asarray(new_obs, dtype=np.float32), np.asarray(new_actions, dtype=np.float32), ep_rows


def evaluate(policy, episodes: int, seed: int, output_csv: Path, method: str):
    env = HarderNarrowPassageEnv({"seed": seed})
    rows = []
    policy.eval()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        steps = 0
        min_bm = float("inf")
        info = {}
        while not done and steps < env.max_steps:
            min_bm = min(min_bm, float(obs[9]))
            with torch.no_grad():
                action = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            obs, _, term, trunc, info = env.step(action.squeeze(0).numpy())
            done = term or trunc
            steps += 1
        rows.append(
            {
                "episode": ep,
                "method": method,
                "corridor_type": info.get("corridor_type", "unknown"),
                "success": float(info.get("success", 0.0)),
                "collision": float(info.get("collision", 0.0)),
                "steps": steps,
                "min_body_margin": min_bm,
            }
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write] {output_csv}")
    print(
        f"[eval] {method} episodes={episodes} "
        f"SR={np.mean([r['success'] for r in rows]):.3f} "
        f"collision={np.mean([r['collision'] for r in rows]):.3f}"
    )
    return rows


def write_summary(rows, path: Path, method: str, train_transitions: int):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["corridor_type"], []).append(row)
    fields = [
        "case",
        "train_transitions",
        "success_rate",
        "collision_rate",
        "avg_min_clearance",
    ] + [f"sr_{k}" for k in sorted(grouped)]
    out = {
        "case": f"v2_{method}",
        "train_transitions": train_transitions,
        "success_rate": round(float(np.mean([r["success"] for r in rows])), 4),
        "collision_rate": round(float(np.mean([r["collision"] for r in rows])), 4),
        "avg_min_clearance": round(float(np.mean([r["min_body_margin"] for r in rows])), 4),
    }
    for ctype, vals in sorted(grouped.items()):
        out[f"sr_{ctype}"] = round(float(np.mean([r["success"] for r in vals])), 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(out)
    print(f"[write] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["bc", "dagger"], default="bc")
    ap.add_argument("--dataset", type=Path, default=RESULTS / "expert_fsm_v2.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dagger-iters", type=int, default=3)
    ap.add_argument("--dagger-episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--save-dir", type=Path, default=RESULTS / "checkpoints" / "bc_dagger_v2")
    ap.add_argument("--output-csv", type=Path, default=None)
    ap.add_argument("--output-summary", type=Path, default=None)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    obs, actions = load_dataset(args.dataset)
    env = HarderNarrowPassageEnv({"seed": args.seed})
    policy = ImitationPolicy(
        obs_dim=obs.shape[1],
        hidden_dim=args.hidden_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )

    train_policy(policy, obs, actions, args)
    if args.algo == "dagger":
        for it in range(args.dagger_iters):
            add_obs, add_actions, _ = collect_dagger(
                policy,
                args.dagger_episodes,
                seed=args.seed + 1000 * (it + 1),
                max_steps=args.max_steps,
            )
            obs = np.concatenate([obs, add_obs], axis=0)
            actions = np.concatenate([actions, add_actions], axis=0)
            print(f"[dagger] iteration={it + 1}/{args.dagger_iters} dataset={len(obs)}")
            train_policy(policy, obs, actions, args)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.save_dir / f"{args.algo}_policy.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "obs_dim": obs.shape[1],
            "hidden_dim": args.hidden_dim,
            "action_low": env.action_space.low,
            "action_high": env.action_space.high,
        },
        ckpt,
    )
    print(f"[saved] {ckpt}")

    output_csv = args.output_csv or (RESULTS / f"{args.algo}_v2_eval.csv")
    output_summary = args.output_summary or (RESULTS / f"{args.algo}_v2_summary.csv")
    rows = evaluate(policy, args.eval_episodes, 10000 + args.seed, output_csv, args.algo)
    write_summary(rows, output_summary, args.algo, len(obs))


if __name__ == "__main__":
    main()
