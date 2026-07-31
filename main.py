"""
Entry point for the EV Dispatch RL summative.

Examples:
  uv run main.py
  uv run main.py --mode demo
  uv run main.py --mode train --algo dqn --timesteps 50000
  uv run main.py --mode train --algo all
  uv run main.py --mode json
"""

from __future__ import annotations

import argparse
import time

from environment.custom_env import EVDispatchEnv


def demo_random(episodes: int = 2, delay: float = 0.15):
    env = EVDispatchEnv(render_mode="human")
    for ep in range(episodes):
        obs, info = env.reset()
        print(f"\n=== Episode {ep + 1} | start zone={info['zone_name']} SOC={info['soc']:.2f} ===")
        done = False
        total = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            print(
                f"step={info['step']:02d} action={action} "
                f"zone={info['zone_name']:<12} soc={info['soc']:.2f} "
                f"reward={reward:6.2f} event={info.get('event')}"
            )
            time.sleep(delay)
            done = terminated or truncated
        print(f"Episode return={total:.2f} rides={info['rides_completed']} missed={info['rides_missed']}")
    env.close()


def dump_json():
    env = EVDispatchEnv()
    env.reset()
    print(env.to_json())
    env.close()


def train(algo: str, timesteps: int, config: str | None):
    if algo == "dqn":
        from training.dqn_training import run_sweep

        run_sweep(total_timesteps=timesteps, only=config)
    elif algo in {"reinforce", "ppo", "a2c", "all"}:
        from training import pg_training

        if algo == "all":
            from training.dqn_training import run_sweep

            run_sweep(total_timesteps=timesteps, only=config)
            pg_training.run_reinforce_sweep(only=None)
            pg_training.run_ppo_sweep(total_timesteps=timesteps, only=None)
            pg_training.run_a2c_sweep(total_timesteps=timesteps, only=None)
        elif algo == "reinforce":
            pg_training.run_reinforce_sweep(only=config)
        elif algo == "ppo":
            pg_training.run_ppo_sweep(total_timesteps=timesteps, only=config)
        else:
            pg_training.run_a2c_sweep(total_timesteps=timesteps, only=config)
    else:
        raise SystemExit(f"Unknown algo: {algo}")


def main():
    parser = argparse.ArgumentParser(description="EV Dispatch RL summative")
    parser.add_argument(
        "--mode",
        choices=["demo", "train", "json"],
        default="demo",
        help="demo=random agent GUI, train=run sweeps, json=API payload sample",
    )
    parser.add_argument("--algo", default="dqn", choices=["dqn", "reinforce", "ppo", "a2c", "all"])
    parser.add_argument("--timesteps", type=int, default=80_000)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "demo":
        demo_random(episodes=args.episodes)
    elif args.mode == "json":
        dump_json()
    else:
        train(args.algo, args.timesteps, args.config)


if __name__ == "__main__":
    main()
