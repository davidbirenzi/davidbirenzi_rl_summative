"""
Play the best trained agent with GUI + verbose terminal output.

Usage:
  uv run play.py
  uv run play.py --algo ppo
  uv run play.py --algo dqn --episodes 3
  uv run play.py --algo reinforce --model models/pg/reinforce/rf_01_baseline.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import A2C, DQN, PPO

from environment.custom_env import EVDispatchEnv
from training.pg_training import PolicyNet

ROOT = Path(__file__).resolve().parent


def load_sb3(algo: str, model_path: Path):
    cls = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algo]
    return cls.load(str(model_path))


def load_reinforce(model_path: Path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    policy = PolicyNet(ckpt["obs_dim"], EVDispatchEnv.N_ACTIONS, cfg.get("hidden", 64))
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    def predict(obs):
        with torch.no_grad():
            logits = policy(torch.as_tensor(obs, dtype=torch.float32))
            return int(torch.argmax(logits).item())

    return predict


def resolve_model(algo: str, model: str | None) -> Path:
    if model:
        path = Path(model)
        if not path.is_absolute():
            path = ROOT / path
        return path

    defaults = {
        "dqn": ROOT / "models" / "dqn" / "best_dqn.zip",
        "ppo": ROOT / "models" / "pg" / "ppo" / "best_ppo.zip",
        "a2c": ROOT / "models" / "pg" / "a2c" / "best_a2c.zip",
        "reinforce": None,
    }
    if algo == "reinforce":
        # Pick best from best.json if present, else first .pt
        best_json = ROOT / "logs" / "reinforce" / "best.json"
        if best_json.exists():
            data = json.loads(best_json.read_text(encoding="utf-8"))
            return Path(data["model_path"])
        pts = sorted((ROOT / "models" / "pg" / "reinforce").glob("*.pt"))
        if not pts:
            raise FileNotFoundError("No REINFORCE model found. Train first.")
        return pts[0]

    path = defaults[algo]
    # Fallbacks: any zip in folder
    if path and path.exists():
        return path
    search = {
        "dqn": ROOT / "models" / "dqn",
        "ppo": ROOT / "models" / "pg" / "ppo",
        "a2c": ROOT / "models" / "pg" / "a2c",
    }[algo]
    zips = sorted(search.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(
            f"No {algo.upper()} model found under {search}. "
            f"Train with: uv run main.py --mode train --algo {algo}"
        )
    return zips[0]


def run(algo: str, model: str | None, episodes: int, delay: float, seed: int):
    model_path = resolve_model(algo, model)
    print(f"Loading {algo.upper()} model from: {model_path}")

    if algo == "reinforce":
        predict = load_reinforce(model_path)
        sb3_model = None
    else:
        sb3_model = load_sb3(algo, model_path)
        predict = None

    env = EVDispatchEnv(render_mode="human")
    meanings = EVDispatchEnv.action_meanings()
    summary = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        print("\n" + "=" * 72)
        print(f"EPISODE {ep + 1} | start={info['zone_name']} SOC={info['soc']:.2f}")
        print("Objective: complete rides without killing the battery; charge smartly.")
        print("=" * 72)
        done = False
        total = 0.0
        while not done:
            if sb3_model is not None:
                action, _ = sb3_model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            print(
                f"[t={info['step']:02d}] act={action} ({meanings[action]}) | "
                f"zone={info['zone_name']:<12} SOC={info['soc']*100:5.1f}% | "
                f"trip={info['on_trip']} charge={info['is_charging']} | "
                f"R={reward:7.2f} event={info.get('event')} | "
                f"rides={info['rides_completed']} missed={info['rides_missed']}"
            )
            time.sleep(delay)
            done = terminated or truncated

        ep_stats = {
            "episode": ep + 1,
            "return": round(total, 2),
            "rides_completed": info["rides_completed"],
            "rides_missed": info["rides_missed"],
            "final_soc": info["soc"],
            "revenue": info["revenue"],
            "energy_cost": info["energy_cost"],
        }
        summary.append(ep_stats)
        print(f"\nEpisode summary: {ep_stats}")

    env.close()
    out = {
        "algorithm": algo,
        "model_path": str(model_path),
        "episodes": summary,
        "mean_return": round(float(np.mean([s["return"] for s in summary])), 2),
    }
    out_path = ROOT / ".last_run_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved API-ready summary -> {out_path}")
    print(json.dumps(out, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Play best EV Dispatch agent")
    parser.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c"], default="ppo")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.algo, args.model, args.episodes, args.delay, args.seed)


if __name__ == "__main__":
    main()
