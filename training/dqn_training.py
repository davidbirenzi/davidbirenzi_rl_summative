"""DQN training with 10 hyperparameter configurations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import EVDispatchEnv
from training.hyperparams import DQN_CONFIGS

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "dqn"
LOG_DIR = ROOT / "logs" / "dqn"


class EpisodeRewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self.losses: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        # Track DQN objective if available
        if "loss" in self.model.logger.name_to_value:
            self.losses.append(float(self.model.logger.name_to_value["loss"]))
        return True


def make_env(seed: int = 0):
    env = EVDispatchEnv()
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def evaluate_mean_reward(model, n_episodes: int = 20, seed: int = 123) -> float:
    env = EVDispatchEnv()
    rewards = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total += reward
            done = terminated or truncated
        rewards.append(total)
    env.close()
    return float(np.mean(rewards))


def train_one(cfg: dict, total_timesteps: int, seed: int = 0) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = make_env(seed)
    callback = EpisodeRewardCallback()
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        exploration_fraction=cfg["exploration_fraction"],
        exploration_final_eps=cfg["exploration_final_eps"],
        learning_starts=1_000,
        verbose=1,
        seed=seed,
        tensorboard_log=str(LOG_DIR / cfg["name"]),
    )
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    mean_reward = evaluate_mean_reward(model)
    out_path = MODEL_DIR / cfg["name"]
    model.save(str(out_path))

    # Save episode rewards for plots
    np.save(LOG_DIR / f"{cfg['name']}_episode_rewards.npy", np.array(callback.episode_rewards))
    result = {
        **cfg,
        "mean_reward": round(mean_reward, 3),
        "n_episodes_logged": len(callback.episode_rewards),
        "model_path": str(out_path) + ".zip",
        "exploration_strategy": (
            f"eps-greedy frac={cfg['exploration_fraction']} "
            f"final={cfg['exploration_final_eps']}"
        ),
    }
    env.close()
    print(f"[DQN] {cfg['name']} mean_reward={mean_reward:.2f}")
    return result


def run_sweep(total_timesteps: int = 80_000, only: str | None = None):
    results = []
    configs = DQN_CONFIGS
    if only:
        configs = [c for c in configs if c["name"] == only]
        if not configs:
            raise SystemExit(f"Unknown config: {only}")

    for cfg in configs:
        results.append(train_one(cfg, total_timesteps=total_timesteps))

    csv_path = LOG_DIR / "dqn_results.csv"
    fieldnames = [
        "name",
        "learning_rate",
        "gamma",
        "buffer_size",
        "batch_size",
        "exploration_strategy",
        "mean_reward",
        "model_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    best = max(results, key=lambda r: r["mean_reward"])
    (LOG_DIR / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    # Copy-friendly best pointer
    best_link = MODEL_DIR / "best_dqn"
    from shutil import copyfile

    copyfile(best["model_path"], str(best_link) + ".zip")
    print(f"Best DQN: {best['name']} ({best['mean_reward']})")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train DQN on EV Dispatch env")
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--config", type=str, default=None, help="Run a single named config")
    args = parser.parse_args()
    run_sweep(total_timesteps=args.timesteps, only=args.config)


if __name__ == "__main__":
    main()
