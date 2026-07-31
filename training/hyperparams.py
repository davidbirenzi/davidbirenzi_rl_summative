"""Shared hyperparameter grids (10 configs per algorithm)."""

DQN_CONFIGS = [
    {"name": "dqn_01_baseline", "learning_rate": 1e-3, "gamma": 0.99, "buffer_size": 50_000, "batch_size": 64, "exploration_fraction": 0.3, "exploration_final_eps": 0.05},
    {"name": "dqn_02_high_lr", "learning_rate": 5e-3, "gamma": 0.99, "buffer_size": 50_000, "batch_size": 64, "exploration_fraction": 0.3, "exploration_final_eps": 0.05},
    {"name": "dqn_03_low_lr", "learning_rate": 1e-4, "gamma": 0.99, "buffer_size": 50_000, "batch_size": 64, "exploration_fraction": 0.3, "exploration_final_eps": 0.05},
    {"name": "dqn_04_low_gamma", "learning_rate": 1e-3, "gamma": 0.90, "buffer_size": 50_000, "batch_size": 64, "exploration_fraction": 0.3, "exploration_final_eps": 0.05},
    {"name": "dqn_05_large_buffer", "learning_rate": 1e-3, "gamma": 0.99, "buffer_size": 200_000, "batch_size": 64, "exploration_fraction": 0.3, "exploration_final_eps": 0.05},
    {"name": "dqn_06_large_batch", "learning_rate": 1e-3, "gamma": 0.99, "buffer_size": 50_000, "batch_size": 256, "exploration_fraction": 0.3, "exploration_final_eps": 0.05},
    {"name": "dqn_07_high_explore", "learning_rate": 1e-3, "gamma": 0.99, "buffer_size": 50_000, "batch_size": 64, "exploration_fraction": 0.5, "exploration_final_eps": 0.10},
    {"name": "dqn_08_low_explore", "learning_rate": 1e-3, "gamma": 0.99, "buffer_size": 50_000, "batch_size": 64, "exploration_fraction": 0.15, "exploration_final_eps": 0.02},
    {"name": "dqn_09_mid_gamma_batch", "learning_rate": 3e-4, "gamma": 0.95, "buffer_size": 100_000, "batch_size": 128, "exploration_fraction": 0.35, "exploration_final_eps": 0.05},
    {"name": "dqn_10_tuned", "learning_rate": 5e-4, "gamma": 0.99, "buffer_size": 100_000, "batch_size": 128, "exploration_fraction": 0.4, "exploration_final_eps": 0.05},
]

REINFORCE_CONFIGS = [
    {"name": "rf_01_baseline", "learning_rate": 1e-3, "gamma": 0.99, "hidden": 64, "entropy_coef": 0.01, "max_episodes": 400},
    {"name": "rf_02_high_lr", "learning_rate": 5e-3, "gamma": 0.99, "hidden": 64, "entropy_coef": 0.01, "max_episodes": 400},
    {"name": "rf_03_low_lr", "learning_rate": 1e-4, "gamma": 0.99, "hidden": 64, "entropy_coef": 0.01, "max_episodes": 400},
    {"name": "rf_04_low_gamma", "learning_rate": 1e-3, "gamma": 0.90, "hidden": 64, "entropy_coef": 0.01, "max_episodes": 400},
    {"name": "rf_05_wide", "learning_rate": 1e-3, "gamma": 0.99, "hidden": 128, "entropy_coef": 0.01, "max_episodes": 400},
    {"name": "rf_06_high_entropy", "learning_rate": 1e-3, "gamma": 0.99, "hidden": 64, "entropy_coef": 0.05, "max_episodes": 400},
    {"name": "rf_07_no_entropy", "learning_rate": 1e-3, "gamma": 0.99, "hidden": 64, "entropy_coef": 0.0, "max_episodes": 400},
    {"name": "rf_08_mid", "learning_rate": 5e-4, "gamma": 0.95, "hidden": 64, "entropy_coef": 0.02, "max_episodes": 400},
    {"name": "rf_09_long", "learning_rate": 3e-4, "gamma": 0.99, "hidden": 128, "entropy_coef": 0.01, "max_episodes": 500},
    {"name": "rf_10_tuned", "learning_rate": 7e-4, "gamma": 0.97, "hidden": 96, "entropy_coef": 0.015, "max_episodes": 450},
]

PPO_CONFIGS = [
    {"name": "ppo_01_baseline", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.01, "clip_range": 0.2, "gae_lambda": 0.95},
    {"name": "ppo_02_high_lr", "learning_rate": 1e-3, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.01, "clip_range": 0.2, "gae_lambda": 0.95},
    {"name": "ppo_03_low_lr", "learning_rate": 1e-4, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.01, "clip_range": 0.2, "gae_lambda": 0.95},
    {"name": "ppo_04_small_steps", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 512, "ent_coef": 0.01, "clip_range": 0.2, "gae_lambda": 0.95},
    {"name": "ppo_05_large_steps", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 4096, "ent_coef": 0.01, "clip_range": 0.2, "gae_lambda": 0.95},
    {"name": "ppo_06_high_entropy", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.05, "clip_range": 0.2, "gae_lambda": 0.95},
    {"name": "ppo_07_no_entropy", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.0, "clip_range": 0.2, "gae_lambda": 0.95},
    {"name": "ppo_08_wide_clip", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.01, "clip_range": 0.3, "gae_lambda": 0.95},
    {"name": "ppo_09_low_gae", "learning_rate": 3e-4, "gamma": 0.95, "n_steps": 2048, "ent_coef": 0.01, "clip_range": 0.2, "gae_lambda": 0.90},
    {"name": "ppo_10_tuned", "learning_rate": 2e-4, "gamma": 0.99, "n_steps": 2048, "ent_coef": 0.005, "clip_range": 0.2, "gae_lambda": 0.98},
]

A2C_CONFIGS = [
    {"name": "a2c_01_baseline", "learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01, "vf_coef": 0.5},
    {"name": "a2c_02_high_lr", "learning_rate": 2e-3, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01, "vf_coef": 0.5},
    {"name": "a2c_03_low_lr", "learning_rate": 1e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01, "vf_coef": 0.5},
    {"name": "a2c_04_more_steps", "learning_rate": 7e-4, "gamma": 0.99, "n_steps": 20, "ent_coef": 0.01, "vf_coef": 0.5},
    {"name": "a2c_05_high_entropy", "learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.05, "vf_coef": 0.5},
    {"name": "a2c_06_no_entropy", "learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.0, "vf_coef": 0.5},
    {"name": "a2c_07_high_vf", "learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01, "vf_coef": 1.0},
    {"name": "a2c_08_low_gamma", "learning_rate": 7e-4, "gamma": 0.90, "n_steps": 5, "ent_coef": 0.01, "vf_coef": 0.5},
    {"name": "a2c_09_mid", "learning_rate": 3e-4, "gamma": 0.95, "n_steps": 10, "ent_coef": 0.02, "vf_coef": 0.5},
    {"name": "a2c_10_tuned", "learning_rate": 5e-4, "gamma": 0.99, "n_steps": 8, "ent_coef": 0.01, "vf_coef": 0.25},
]
