#!/usr/bin/env python3
"""Lightweight GRU-PPO baseline for HarderNarrowPassageEnv.

The preferred paper implementation is SB3-Contrib RecurrentPPO when available.
The current local environment does not provide stable_baselines3/sb3_contrib, so
this script supplies a small PyTorch recurrent PPO runner that uses the same
19-D v2 geometry observation and continuous action space.  It is meant to make
the GRU/Recurrent PPO baseline runnable in this repository without hiding the
dependency gap.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

sys.path.insert(0, str(Path(__file__).parent))

from procedural_env_v2 import HarderNarrowPassageEnv

RESULTS = Path(__file__).parent / "results" / "narrow_passage_rl"


class GRUPPOPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_low, action_high):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, 2)
        self.value = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.7))
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
        center = (self.action_high + self.action_low) * 0.5
        scale = (self.action_high - self.action_low) * 0.5
        self.register_buffer("action_center", center)
        self.register_buffer("action_scale", scale)

    def forward(self, obs, hidden):
        x = self.encoder(obs)
        h = self.gru(x, hidden)
        mean_unit = torch.tanh(self.mean(h))
        mean = self.action_center + self.action_scale * mean_unit
        value = self.value(h).squeeze(-1)
        return mean, value, h

    def dist_value(self, obs, hidden):
        mean, value, next_hidden = self(obs, hidden)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), value, next_hidden


def _tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def collect_rollout(env, policy, rollout_steps, hidden_dim, device):
    obs_np, _ = env.reset()
    hidden = torch.zeros(1, hidden_dim, device=device)
    rows = []

    for _ in range(rollout_steps):
        obs = _tensor(obs_np).unsqueeze(0).to(device)
        hidden_in = hidden.detach()
        with torch.no_grad():
            dist, value, hidden_next = policy.dist_value(obs, hidden_in)
            raw_action = dist.sample()
            logp = dist.log_prob(raw_action).sum(-1)
            action = torch.max(torch.min(raw_action, policy.action_high), policy.action_low)

        next_obs, reward, done, _, info = env.step(action.squeeze(0).cpu().numpy())
        rows.append(
            {
                "obs": obs.squeeze(0).cpu().numpy(),
                "hidden": hidden_in.squeeze(0).cpu().numpy(),
                "action": raw_action.squeeze(0).cpu().numpy(),
                "logp": float(logp.item()),
                "reward": float(reward),
                "done": bool(done),
                "value": float(value.item()),
                "info": info,
            }
        )

        if done:
            obs_np, _ = env.reset()
            hidden = torch.zeros(1, hidden_dim, device=device)
        else:
            obs_np = next_obs
            hidden = hidden_next.detach()

    with torch.no_grad():
        obs = _tensor(obs_np).unsqueeze(0).to(device)
        _, last_value, _ = policy.dist_value(obs, hidden)
    return rows, float(last_value.item())


def compute_gae(rows, last_value, gamma, gae_lambda):
    advantages = np.zeros(len(rows), dtype=np.float32)
    returns = np.zeros(len(rows), dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rows))):
        next_nonterminal = 0.0 if rows[t]["done"] else 1.0
        next_value = last_value if t == len(rows) - 1 else rows[t + 1]["value"]
        delta = rows[t]["reward"] + gamma * next_value * next_nonterminal - rows[t]["value"]
        gae = delta + gamma * gae_lambda * next_nonterminal * gae
        advantages[t] = gae
        returns[t] = advantages[t] + rows[t]["value"]
    adv_std = float(advantages.std())
    advantages = (advantages - float(advantages.mean())) / max(adv_std, 1e-6)
    return advantages, returns


def ppo_update(policy, optimizer, rows, advantages, returns, args, device):
    obs = _tensor(np.stack([r["obs"] for r in rows])).to(device)
    hidden = _tensor(np.stack([r["hidden"] for r in rows])).to(device)
    actions = _tensor(np.stack([r["action"] for r in rows])).to(device)
    old_logp = _tensor([r["logp"] for r in rows]).to(device)
    adv = _tensor(advantages).to(device)
    ret = _tensor(returns).to(device)
    idxs = np.arange(len(rows))

    for _ in range(args.epochs):
        np.random.shuffle(idxs)
        for start in range(0, len(rows), args.batch_size):
            mb = idxs[start : start + args.batch_size]
            dist, value, _ = policy.dist_value(obs[mb], hidden[mb])
            logp = dist.log_prob(actions[mb]).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            ratio = torch.exp(logp - old_logp[mb])
            unclipped = ratio * adv[mb]
            clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * adv[mb]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (ret[mb] - value).pow(2).mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()


def evaluate(policy, episodes, seed, hidden_dim, device, output_csv=None):
    env = HarderNarrowPassageEnv({"seed": seed})
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        hidden = torch.zeros(1, hidden_dim, device=device)
        done = False
        steps = 0
        min_bm = float("inf")
        info = {}
        while not done and steps < env.max_steps:
            min_bm = min(min_bm, float(obs[9]))
            with torch.no_grad():
                obs_t = _tensor(obs).unsqueeze(0).to(device)
                dist, _, hidden = policy.dist_value(obs_t, hidden)
                action = dist.mean
                action = torch.max(torch.min(action, policy.action_high), policy.action_low)
            obs, _, done, _, info = env.step(action.squeeze(0).cpu().numpy())
            steps += 1
        rows.append(
            {
                "episode": ep,
                "method": "gru_ppo_lightweight",
                "corridor_type": info.get("corridor_type", "unknown"),
                "success": float(info.get("success", 0.0)),
                "collision": float(info.get("collision", 0.0)),
                "steps": steps,
                "min_body_margin": min_bm,
            }
        )
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[write] {output_csv}")
    sr = float(np.mean([r["success"] for r in rows]))
    col = float(np.mean([r["collision"] for r in rows]))
    print(f"[eval] episodes={episodes} SR={sr:.3f} collision={col:.3f}")
    return rows


def write_summary(rows, train_steps, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["corridor_type"], []).append(row)
    with path.open("w", newline="") as f:
        fields = [
            "case",
            "train_steps",
            "success_rate",
            "collision_rate",
            "avg_min_clearance",
            "notes",
        ]
        for ctype in sorted(grouped):
            fields.append(f"sr_{ctype}")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        out = {
            "case": "v2_gru_ppo_lightweight",
            "train_steps": train_steps,
            "success_rate": round(float(np.mean([r["success"] for r in rows])), 4),
            "collision_rate": round(float(np.mean([r["collision"] for r in rows])), 4),
            "avg_min_clearance": round(float(np.mean([r["min_body_margin"] for r in rows])), 4),
            "notes": "PyTorch lightweight GRU-PPO; SB3-Contrib unavailable in current env",
        }
        for ctype, vals in sorted(grouped.items()):
            out[f"sr_{ctype}"] = round(float(np.mean([r["success"] for r in vals])), 4)
        writer.writerow(out)
    print(f"[write] {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=20000)
    ap.add_argument("--rollout-steps", type=int, default=1024)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--save-dir", type=Path, default=Path("data/narrow_passage_gru_ppo_v2"))
    ap.add_argument("--output-csv", type=Path, default=RESULTS / "gru_ppo_v2_eval.csv")
    ap.add_argument("--output-summary", type=Path, default=RESULTS / "gru_ppo_v2_summary.csv")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    env = HarderNarrowPassageEnv({"seed": args.seed})
    policy = GRUPPOPolicy(
        obs_dim=env.observation_space.shape[0],
        hidden_dim=args.hidden_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    completed = 0
    while completed < args.total_steps:
        steps = min(args.rollout_steps, args.total_steps - completed)
        rows, last_value = collect_rollout(env, policy, steps, args.hidden_dim, device)
        adv, ret = compute_gae(rows, last_value, args.gamma, args.gae_lambda)
        ppo_update(policy, optimizer, rows, adv, ret, args, device)
        completed += steps
        recent_success = np.mean([r["info"].get("success", 0.0) for r in rows if r["done"]])
        if math.isnan(float(recent_success)):
            recent_success = 0.0
        print(f"[train] steps={completed}/{args.total_steps} done_success={recent_success:.3f}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.save_dir / "gru_ppo_lightweight.pt"
    torch.save({"model": policy.state_dict(), "args": vars(args)}, ckpt)
    print(f"[saved] {ckpt}")

    eval_rows = evaluate(
        policy,
        episodes=args.eval_episodes,
        seed=10000 + args.seed,
        hidden_dim=args.hidden_dim,
        device=device,
        output_csv=args.output_csv,
    )
    write_summary(eval_rows, args.total_steps, args.output_summary)


if __name__ == "__main__":
    main()
