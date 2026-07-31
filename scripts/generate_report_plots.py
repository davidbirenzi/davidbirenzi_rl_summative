"""Evaluate generalization + generate report plots from training logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, DQN, PPO

from environment.custom_env import EVDispatchEnv
from training.pg_training import PolicyNet
import torch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
LOGS = ROOT / "logs"


def load_predict(algo: str):
    if algo == "dqn":
        path = ROOT / "models" / "dqn" / "best_dqn.zip"
        if not path.exists():
            zips = list((ROOT / "models" / "dqn").glob("*.zip"))
            path = zips[0] if zips else None
        if path is None:
            return None
        model = DQN.load(str(path))
        return lambda obs: int(model.predict(obs, deterministic=True)[0])
    if algo in {"ppo", "a2c"}:
        folder = ROOT / "models" / "pg" / algo
        best = folder / f"best_{algo}.zip"
        path = best if best.exists() else next(iter(folder.glob("*.zip")), None)
        if path is None:
            return None
        cls = PPO if algo == "ppo" else A2C
        model = cls.load(str(path))
        return lambda obs: int(model.predict(obs, deterministic=True)[0])
    # reinforce
    best_json = LOGS / "reinforce" / "best.json"
    if best_json.exists():
        path = Path(json.loads(best_json.read_text())["model_path"])
    else:
        path = next(iter((ROOT / "models" / "pg" / "reinforce").glob("*.pt")), None)
    if path is None:
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    policy = PolicyNet(ckpt["obs_dim"], EVDispatchEnv.N_ACTIONS, ckpt["cfg"].get("hidden", 64))
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    def predict(obs):
        with torch.no_grad():
            return int(torch.argmax(policy(torch.as_tensor(obs, dtype=torch.float32))).item())

    return predict


def run_episodes(predict, n=30, seed=1000, unseen: bool = False):
    env = EVDispatchEnv()
    returns = []
    for i in range(n):
        options = None
        if unseen:
            # Unseen initial states: low SOC + fixed uncommon starts
            options = {
                "start_zone": i % 4,
                "start_soc": 0.25 + 0.05 * (i % 5),
            }
        obs, _ = env.reset(seed=seed + i, options=options)
        done = False
        total = 0.0
        while not done:
            action = predict(obs)
            obs, reward, term, trunc, _ = env.step(action)
            total += reward
            done = term or trunc
        returns.append(total)
    env.close()
    return np.array(returns)


def _load_tb_scalars(run_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (steps, values) for a TensorBoard scalar tag from an SB3 run folder."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # Prefer the latest nested DQN_* / PPO_* folder if present
    candidates = [run_dir]
    nested = sorted(
        [p for p in run_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    candidates = nested + candidates if nested else candidates

    for folder in reversed(candidates):
        events = list(folder.glob("events*"))
        if not events and folder != run_dir:
            continue
        target = folder if events else run_dir
        if not list(target.glob("events*")):
            continue
        ea = EventAccumulator(str(target))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if tag not in tags:
            continue
        scalars = ea.Scalars(tag)
        steps = np.array([s.step for s in scalars], dtype=np.float64)
        values = np.array([s.value for s in scalars], dtype=np.float64)
        return steps, values
    return np.array([]), np.array([])


def plot_dqn_objective():
    """DQN training objective (TD loss) from TensorBoard logs — report requirement."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    best_json = LOGS / "dqn" / "best.json"
    preferred = None
    if best_json.exists():
        preferred = json.loads(best_json.read_text(encoding="utf-8"))["name"]

    # Plot best run prominently + a few comparison configs
    run_dirs = sorted([p for p in (LOGS / "dqn").iterdir() if p.is_dir()])
    if preferred:
        run_dirs = sorted(run_dirs, key=lambda p: (p.name != preferred, p.name))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: best DQN loss curve
    ax = axes[0]
    plotted = False
    for run_dir in run_dirs:
        steps, values = _load_tb_scalars(run_dir, "train/loss")
        if len(values) == 0:
            continue
        # Smooth for readability
        window = max(3, len(values) // 25)
        if len(values) >= window:
            kernel = np.ones(window) / window
            smooth = np.convolve(values, kernel, mode="valid")
            x = steps[window - 1 :]
            ax.plot(x, smooth, color="#e76f51", label=f"{run_dir.name} (smoothed)")
        else:
            ax.plot(steps, values, color="#e76f51", label=run_dir.name)
        plotted = True
        break

    if plotted:
        ax.set_title("DQN objective (TD loss) — best model")
        ax.set_xlabel("Training timestep")
        ax.set_ylabel("train/loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title("DQN objective (no TensorBoard loss found)")
        ax.text(0.5, 0.5, "Retrain DQN with tensorboard_log", ha="center", va="center")

    # Right: compare a few DQN configs' final loss trajectory
    ax = axes[1]
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    n_plotted = 0
    for run_dir, color in zip(run_dirs[:5], colors):
        steps, values = _load_tb_scalars(run_dir, "train/loss")
        if len(values) == 0:
            continue
        window = max(3, len(values) // 20)
        if len(values) >= window:
            kernel = np.ones(window) / window
            smooth = np.convolve(values, kernel, mode="valid")
            ax.plot(steps[window - 1 :], smooth, color=color, label=run_dir.name, alpha=0.9)
        else:
            ax.plot(steps, values, color=color, label=run_dir.name, alpha=0.9)
        n_plotted += 1

    ax.set_title("DQN objective — hyperparameter comparison")
    ax.set_xlabel("Training timestep")
    ax.set_ylabel("train/loss (smoothed)")
    if n_plotted:
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = ASSETS / "dqn_objective_curve.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def save_environment_screenshot():
    """Capture GUI frame (rgb_array) for the report Environment section."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    import os

    # Headless-friendly on some setups
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    env = EVDispatchEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=42, options={"start_zone": 1, "start_soc": 0.72})
    # Take a few meaningful actions so the frame shows a request / movement
    for action in (4, 3, 5, 4, 2, 0, 1):
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(seed=43)
    frame = env.render()
    env.close()

    if frame is None:
        print("Could not capture environment frame (render returned None)")
        return

    out = ASSETS / "environment_gui.png"
    plt.figure(figsize=(10, 6))
    plt.imshow(frame)
    plt.axis("off")
    plt.title(
        f"EV Dispatch Environment — zone={info.get('zone_name')} | "
        f"SOC={info.get('soc', 0):.0%} | rides={info.get('rides_completed', 0)}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")


def plot_cumulative_rewards():
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    mapping = {
        "DQN": LOGS / "dqn",
        "REINFORCE": LOGS / "reinforce",
        "PPO": LOGS / "ppo",
        "A2C": LOGS / "a2c",
    }
    for ax, (name, folder) in zip(axes.ravel(), mapping.items()):
        files = sorted(folder.glob("*_episode_rewards.npy"))
        if not files:
            ax.set_title(f"{name} (no data)")
            ax.text(0.5, 0.5, "Train first", ha="center", va="center")
            continue
        # Prefer best-named file if best.json exists
        best_json = folder / "best.json"
        chosen = files[-1]
        if best_json.exists():
            bname = json.loads(best_json.read_text())["name"]
            candidate = folder / f"{bname}_episode_rewards.npy"
            if candidate.exists():
                chosen = candidate
        rewards = np.load(chosen)
        # Cumulative mean
        cum = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
        ax.plot(cum, color="#2a9d8f")
        ax.set_title(f"{name} cumulative mean reward")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean reward")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "cumulative_rewards.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_entropy_and_convergence():
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, algo in zip(axes, ["reinforce", "ppo", "a2c"]):
        files = sorted((LOGS / algo).glob("*_entropy.npy"))
        if not files:
            ax.set_title(f"{algo} entropy (no data)")
            continue
        data = np.load(files[-1])
        if len(data) == 0:
            ax.set_title(f"{algo} entropy (empty)")
            continue
        ax.plot(data, color="#e9c46a")
        ax.set_title(f"{algo.upper()} policy entropy")
        ax.set_xlabel("Update / episode")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "pg_entropy_curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")

    # Convergence subplot: rolling mean of rewards
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (title, folder) in zip(
        axes.ravel(),
        [
            ("DQN", LOGS / "dqn"),
            ("REINFORCE", LOGS / "reinforce"),
            ("PPO", LOGS / "ppo"),
            ("A2C", LOGS / "a2c"),
        ],
    ):
        files = sorted(folder.glob("*_episode_rewards.npy"))
        if not files:
            ax.set_title(f"{title} convergence (no data)")
            continue
        r = np.load(files[-1])
        window = max(5, len(r) // 20)
        if len(r) >= window:
            kernel = np.ones(window) / window
            smooth = np.convolve(r, kernel, mode="valid")
            ax.plot(smooth, color="#264653")
        else:
            ax.plot(r, color="#264653")
        ax.set_title(f"{title} episodes to stable performance")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward (smoothed)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "convergence_plots.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out}")


def generalization_report():
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = []
    for algo in ["dqn", "reinforce", "ppo", "a2c"]:
        predict = load_predict(algo)
        if predict is None:
            print(f"Skip {algo}: no model")
            continue
        seen = run_episodes(predict, n=20, seed=200, unseen=False)
        unseen = run_episodes(predict, n=20, seed=9000, unseen=True)
        rows.append(
            {
                "algo": algo,
                "seen_mean": float(seen.mean()),
                "unseen_mean": float(unseen.mean()),
            }
        )
        print(
            f"{algo}: seen={seen.mean():.2f} ± {seen.std():.2f} | "
            f"unseen={unseen.mean():.2f} ± {unseen.std():.2f}"
        )

    if not rows:
        print("No models available for generalization test.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [r["algo"].upper() for r in rows]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, [r["seen_mean"] for r in rows], w, label="Training-like starts")
    ax.bar(x + w / 2, [r["unseen_mean"] for r in rows], w, label="Unseen starts")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean episode return")
    ax.set_title("Generalization: seen vs unseen initial states")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "generalization.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    (ASSETS / "generalization.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--generalization-only", action="store_true")
    parser.add_argument("--dqn-only", action="store_true")
    parser.add_argument("--screenshot-only", action="store_true")
    args = parser.parse_args()
    if args.generalization_only:
        generalization_report()
        return
    if args.dqn_only:
        plot_dqn_objective()
        return
    if args.screenshot_only:
        save_environment_screenshot()
        return
    plot_cumulative_rewards()
    plot_entropy_and_convergence()
    plot_dqn_objective()
    save_environment_screenshot()
    if not args.plots_only:
        generalization_report()


if __name__ == "__main__":
    main()
