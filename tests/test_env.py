"""Smoke tests for EV Dispatch environment."""

import json

import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.custom_env import EVDispatchEnv


def test_env_spaces_and_reset():
    env = EVDispatchEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "soc" in info
    assert obs.shape == env.observation_space.shape
    env.close()


def test_env_step_runs():
    env = EVDispatchEnv()
    obs, _ = env.reset(seed=1)
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()


def test_json_serialization():
    env = EVDispatchEnv()
    env.reset(seed=2)
    payload = json.loads(env.to_json())
    assert "observation" in payload
    assert "action_meanings" in payload
    assert len(payload["observation"]) == env.obs_dim
    env.close()


def test_gymnasium_api():
    env = EVDispatchEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_unseen_initial_state_options():
    env = EVDispatchEnv()
    obs, info = env.reset(seed=3, options={"start_zone": 3, "start_soc": 0.3})
    assert info["zone"] == 3
    assert abs(info["soc"] - 0.3) < 1e-6
    env.close()
