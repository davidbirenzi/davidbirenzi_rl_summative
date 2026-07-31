"""
Policy-gradient training: REINFORCE (custom), PPO, and A2C (Stable-Baselines3).
Each algorithm has 10 hyperparameter configurations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import EVDispatchEnv
from training.hyperparams import A2C_CONFIGS, PPO_CONFIGS, REINFORCE_CONFIGS

ROOT = Path(__file__).resolve().parents[1]
PG_DIR = ROOT / "models" / "pg"
LOG_ROOT = ROOT / "logs"


class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self.entropies: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        ent = self.model.logger.name_to_value.get("train/entropy_loss")
        if ent is not None:
            # SB3 logs entropy loss as negative entropy; store magnitude
            self.entropies.append(float(abs(ent)))
        return True


def evaluate_policy_np(predict_fn, n_episodes: int = 20, seed: int = 123) -> float:
    env = EVDispatchEnv()
    rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        total = 0.0
        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += reward
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


# ---------------------------------------------------------------------------
# REINFORCE (custom — not included in Stable-Baselines3)
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def train_reinforce(cfg: dict, seed: int = 0) -> dict:
    out_dir = PG_DIR / "reinforce"
    log_dir = LOG_ROOT / "reinforce"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = EVDispatchEnv()
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n, cfg["hidden"])
    optimizer = optim.Adam(policy.parameters(), lr=cfg["learning_rate"])
    gamma = cfg["gamma"]
    entropy_coef = cfg["entropy_coef"]
    episode_rewards: list[float] = []
    entropy_log: list[float] = []

    for ep in range(cfg["max_episodes"]):
        obs, _ = env.reset(seed=seed + ep)
        log_probs = []
        rewards = []
        entropies = []
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            logits = policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            rewards.append(reward)
            done = terminated or truncated

        # Discounted returns
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = []
        for lp, Gt, ent in zip(log_probs, returns_t, entropies):
            loss.append(-lp * Gt - entropy_coef * ent)
        loss_t = torch.stack(loss).sum()
        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        ep_ret = float(sum(rewards))
        episode_rewards.append(ep_ret)
        entropy_log.append(float(torch.stack(entropies).mean().item()))
        if (ep + 1) % 50 == 0:
            print(f"[REINFORCE] {cfg['name']} ep={ep+1} return={ep_ret:.1f}")

    def predict(obs):
        with torch.no_grad():
            logits = policy(torch.as_tensor(obs, dtype=torch.float32))
            return int(torch.argmax(logits).item())

    mean_reward = evaluate_policy_np(predict)
    model_path = out_dir / f"{cfg['name']}.pt"
    torch.save(
        {"state_dict": policy.state_dict(), "cfg": cfg, "obs_dim": env.observation_space.shape[0]},
        model_path,
    )
    np.save(log_dir / f"{cfg['name']}_episode_rewards.npy", np.array(episode_rewards))
    np.save(log_dir / f"{cfg['name']}_entropy.npy", np.array(entropy_log))
    env.close()
    result = {
        **cfg,
        "mean_reward": round(mean_reward, 3),
        "model_path": str(model_path),
    }
    print(f"[REINFORCE] {cfg['name']} mean_reward={mean_reward:.2f}")
    return result


# ---------------------------------------------------------------------------
# PPO / A2C via Stable-Baselines3
# ---------------------------------------------------------------------------
def train_sb3(algo: str, cfg: dict, total_timesteps: int, seed: int = 0) -> dict:
    assert algo in {"ppo", "a2c"}
    out_dir = PG_DIR / algo
    log_dir = LOG_ROOT / algo
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(EVDispatchEnv())
    callback = RewardCallback()
    common = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        n_steps=cfg["n_steps"],
        ent_coef=cfg["ent_coef"],
        verbose=1,
        seed=seed,
        tensorboard_log=str(log_dir / cfg["name"]),
    )
    if algo == "ppo":
        model = PPO(
            **common,
            clip_range=cfg["clip_range"],
            gae_lambda=cfg["gae_lambda"],
        )
    else:
        model = A2C(**common, vf_coef=cfg["vf_coef"])

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    def predict(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    mean_reward = evaluate_policy_np(predict)
    model_path = out_dir / cfg["name"]
    model.save(str(model_path))
    np.save(log_dir / f"{cfg['name']}_episode_rewards.npy", np.array(callback.episode_rewards))
    np.save(log_dir / f"{cfg['name']}_entropy.npy", np.array(callback.entropies))
    env.close()
    result = {
        **cfg,
        "mean_reward": round(mean_reward, 3),
        "model_path": str(model_path) + ".zip",
    }
    print(f"[{algo.upper()}] {cfg['name']} mean_reward={mean_reward:.2f}")
    return result


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_reinforce_sweep(only: Optional[str] = None):
    configs = REINFORCE_CONFIGS
    if only:
        configs = [c for c in configs if c["name"] == only]
    results = [train_reinforce(c) for c in configs]
    _write_csv(
        LOG_ROOT / "reinforce" / "reinforce_results.csv",
        results,
        ["name", "learning_rate", "gamma", "hidden", "entropy_coef", "max_episodes", "mean_reward", "model_path"],
    )
    best = max(results, key=lambda r: r["mean_reward"])
    (LOG_ROOT / "reinforce" / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return results


def run_ppo_sweep(total_timesteps: int = 80_000, only: Optional[str] = None):
    configs = PPO_CONFIGS
    if only:
        configs = [c for c in configs if c["name"] == only]
    results = [train_sb3("ppo", c, total_timesteps) for c in configs]
    _write_csv(
        LOG_ROOT / "ppo" / "ppo_results.csv",
        results,
        ["name", "learning_rate", "gamma", "n_steps", "ent_coef", "clip_range", "gae_lambda", "mean_reward", "model_path"],
    )
    best = max(results, key=lambda r: r["mean_reward"])
    (LOG_ROOT / "ppo" / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    from shutil import copyfile

    copyfile(best["model_path"], str(PG_DIR / "ppo" / "best_ppo.zip"))
    return results


def run_a2c_sweep(total_timesteps: int = 80_000, only: Optional[str] = None):
    configs = A2C_CONFIGS
    if only:
        configs = [c for c in configs if c["name"] == only]
    results = [train_sb3("a2c", c, total_timesteps) for c in configs]
    _write_csv(
        LOG_ROOT / "a2c" / "a2c_results.csv",
        results,
        ["name", "learning_rate", "gamma", "n_steps", "ent_coef", "vf_coef", "mean_reward", "model_path"],
    )
    best = max(results, key=lambda r: r["mean_reward"])
    (LOG_ROOT / "a2c" / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    from shutil import copyfile

    copyfile(best["model_path"], str(PG_DIR / "a2c" / "best_a2c.zip"))
    return results


def main():
    parser = argparse.ArgumentParser(description="Train PG methods on EV Dispatch")
    parser.add_argument("--algo", choices=["reinforce", "ppo", "a2c", "all"], default="all")
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.algo in ("reinforce", "all"):
        run_reinforce_sweep(only=args.config if args.algo == "reinforce" else None)
    if args.algo in ("ppo", "all"):
        run_ppo_sweep(total_timesteps=args.timesteps, only=args.config if args.algo == "ppo" else None)
    if args.algo in ("a2c", "all"):
        run_a2c_sweep(total_timesteps=args.timesteps, only=args.config if args.algo == "a2c" else None)


if __name__ == "__main__":
    main()
