"""
EV Fleet Charging & Dispatch Environment
----------------------------------------
A mission-based Gymnasium environment where an RL agent controls one electric
vehicle during a city shift: accept rides across zones, manage battery state
of charge (SOC), and decide when/where to charge under dynamic electricity
prices and charger congestion.

Designed for fair comparison of DQN, REINFORCE, PPO, and A2C.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Zone layout (adjacency graph for travel):
#   0 (Residential) -- 1 (Downtown)
#        |                  |
#   2 (Industrial)  -- 3 (Airport)
ZONE_NAMES = ("Residential", "Downtown", "Industrial", "Airport")
N_ZONES = 4
ADJACENCY = {
    0: (1, 2),
    1: (0, 3),
    2: (0, 3),
    3: (1, 2),
}
# Charging hubs are located in Downtown and Airport
CHARGER_ZONES = (1, 3)
TRAVEL_ENERGY = 0.06  # SOC drain per zone hop
IDLE_ENERGY = 0.005
CHARGE_RATE = 0.12  # SOC gained per charge action
RIDE_TIMEOUT = 8  # steps before a pending request expires
MAX_STEPS = 80  # one city shift


class EVDispatchEnv(gym.Env):
    """Single-EV dispatch + charging MDP."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 8}

    # Discrete actions
    # 0-3: move toward / into zone i (no-op if already there / not adjacent)
    # 4: accept pending ride (if any)
    # 5: start / continue charging (must be at charger zone)
    # 6: stop charging / unplug
    # 7: idle (wait one step)
    N_ACTIONS = 8

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = MAX_STEPS,
        seed: Optional[int] = None,
        demand_seed_offset: int = 0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.demand_seed_offset = demand_seed_offset

        # Observation: continuous features for production-like telemetry
        # [soc, zone_oh(4), charger_q0, charger_q1, ride_origin_oh(4),
        #  ride_dest_oh(4), has_pending, on_trip, trip_progress,
        #  time_norm, price_norm, is_charging, battery_critical]
        self.obs_dim = 1 + 4 + 2 + 4 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.N_ACTIONS)

        self.np_random = np.random.default_rng(seed)
        self._renderer = None

        # Episode state (set in reset)
        self.soc = 1.0
        self.zone = 0
        self.step_count = 0
        self.is_charging = False
        self.on_trip = False
        self.trip_origin = -1
        self.trip_dest = -1
        self.trip_phase = 0  # 0=to pickup, 1=to dropoff
        self.pending_origin = -1
        self.pending_dest = -1
        self.pending_age = 0
        self.charger_queues = {1: 0, 3: 0}
        self.rides_completed = 0
        self.rides_missed = 0
        self.total_revenue = 0.0
        self.total_energy_cost = 0.0
        self.last_info: dict[str, Any] = {}
        self.last_action = 7
        self.last_reward = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        options = options or {}
        # Unseen initial states for generalization tests
        start_zone = options.get("start_zone", int(self.np_random.integers(0, N_ZONES)))
        start_soc = options.get(
            "start_soc", float(self.np_random.uniform(0.45, 0.95))
        )

        self.soc = float(np.clip(start_soc, 0.05, 1.0))
        self.zone = int(start_zone)
        self.step_count = 0
        self.is_charging = False
        self.on_trip = False
        self.trip_origin = -1
        self.trip_dest = -1
        self.trip_phase = 0
        self.pending_origin = -1
        self.pending_dest = -1
        self.pending_age = 0
        self.charger_queues = {
            1: int(self.np_random.integers(0, 3)),
            3: int(self.np_random.integers(0, 3)),
        }
        self.rides_completed = 0
        self.rides_missed = 0
        self.total_revenue = 0.0
        self.total_energy_cost = 0.0
        self.last_action = 7
        self.last_reward = 0.0

        self._maybe_spawn_ride()
        obs = self._get_obs()
        info = self._get_info()
        self.last_info = info
        return obs, info

    def step(self, action: int):
        action = int(action)
        self.last_action = action
        reward = 0.0
        terminated = False
        truncated = False
        event = "none"

        self.step_count += 1
        price = self._electricity_price()

        # Age pending ride / miss handling
        if self.pending_origin >= 0 and not self.on_trip:
            self.pending_age += 1
            if self.pending_age > RIDE_TIMEOUT:
                self.rides_missed += 1
                reward -= 12.0
                event = "ride_missed"
                self.pending_origin = -1
                self.pending_dest = -1
                self.pending_age = 0

        # Dynamic charger queues (stochastic arrivals/departures)
        for z in CHARGER_ZONES:
            delta = int(self.np_random.integers(-1, 2))
            self.charger_queues[z] = int(
                np.clip(self.charger_queues[z] + delta, 0, 5)
            )

        # Execute action
        if action in (0, 1, 2, 3):
            reward += self._act_move(action)
            event = "move"
        elif action == 4:
            reward += self._act_accept_ride()
            event = "accept" if self.on_trip else "accept_failed"
        elif action == 5:
            r, event = self._act_charge(price)
            reward += r
        elif action == 6:
            self.is_charging = False
            event = "unplug"
            reward -= 0.2
        else:  # idle
            self.is_charging = False
            self.soc = max(0.0, self.soc - IDLE_ENERGY)
            reward -= 0.3
            event = "idle"

        # Trip progress if en route
        if self.on_trip:
            reward += self._progress_trip()

        # Battery health shaping
        if self.soc < 0.15:
            reward -= 2.0
        if self.soc <= 0.0:
            reward -= 50.0
            terminated = True
            event = "battery_dead"

        # Small living cost
        reward -= 0.15

        # Spawn new demand if free
        if not self.on_trip and self.pending_origin < 0:
            self._maybe_spawn_ride()

        if self.step_count >= self.max_steps:
            truncated = True
            # End-of-shift bonus for healthy SOC and completed work
            reward += 5.0 * self.rides_completed
            reward += 3.0 * self.soc

        obs = self._get_obs()
        info = self._get_info()
        info["event"] = event
        info["electricity_price"] = price
        self.last_reward = float(reward)
        self.last_info = info

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _act_move(self, target_zone: int) -> float:
        self.is_charging = False
        if self.zone == target_zone:
            self.soc = max(0.0, self.soc - IDLE_ENERGY)
            return -0.5  # pointless move
        if target_zone not in ADJACENCY[self.zone]:
            # Illegal hop: stay, small penalty (agent must learn graph)
            self.soc = max(0.0, self.soc - IDLE_ENERGY)
            return -1.5
        self.zone = target_zone
        drain = TRAVEL_ENERGY * (1.0 + 0.3 * self._traffic_factor())
        self.soc = max(0.0, self.soc - drain)
        return -0.4

    def _act_accept_ride(self) -> float:
        self.is_charging = False
        if self.on_trip or self.pending_origin < 0:
            return -1.0
        # Must not accept if SOC is critically low for a round trip estimate
        est = self._estimate_trip_energy(self.pending_origin, self.pending_dest)
        if self.soc < est + 0.08:
            return -2.0
        self.on_trip = True
        self.trip_origin = self.pending_origin
        self.trip_dest = self.pending_dest
        self.trip_phase = 0
        self.pending_origin = -1
        self.pending_dest = -1
        self.pending_age = 0
        return 2.0

    def _act_charge(self, price: float) -> tuple[float, str]:
        if self.zone not in CHARGER_ZONES:
            return -2.0, "charge_wrong_zone"
        if self.on_trip:
            return -2.0, "charge_while_trip"
        queue = self.charger_queues[self.zone]
        # Congestion: if queue busy, waiting costs time/energy and little charge
        if not self.is_charging:
            if queue >= 4:
                self.soc = max(0.0, self.soc - IDLE_ENERGY)
                return -1.5, "charger_full"
            self.is_charging = True
            # Joining queue slightly increases perceived congestion
            self.charger_queues[self.zone] = min(5, queue + 1)

        gained = CHARGE_RATE * (0.55 if queue >= 3 else 1.0)
        gained = min(gained, 1.0 - self.soc)
        self.soc += gained
        cost = gained * price * 8.0  # scale into reward units
        self.total_energy_cost += cost
        # Prefer charging when price is low
        reward = 1.5 * gained - cost
        if self.soc >= 0.98:
            reward += 1.0
        return reward, "charging"

    def _progress_trip(self) -> float:
        reward = 0.0
        target = self.trip_origin if self.trip_phase == 0 else self.trip_dest
        # Distance shaping
        dist = self._zone_distance(self.zone, target)
        reward += 0.8 * (1.0 / (1.0 + dist))

        if self.zone == target:
            if self.trip_phase == 0:
                self.trip_phase = 1
                reward += 5.0  # pickup
            else:
                # Dropoff complete
                fare = 18.0 + 6.0 * self._zone_distance(
                    self.trip_origin, self.trip_dest
                )
                self.rides_completed += 1
                self.total_revenue += fare
                reward += fare
                self.on_trip = False
                self.trip_origin = -1
                self.trip_dest = -1
                self.trip_phase = 0
        return reward

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------
    def _maybe_spawn_ride(self):
        # Demand depends on time-of-day (rush hours) and zone popularity
        t = self.step_count / max(1, self.max_steps)
        rush = 1.0 if (0.2 <= t <= 0.35 or 0.65 <= t <= 0.85) else 0.45
        if self.np_random.random() < 0.55 * rush:
            o = int(self.np_random.integers(0, N_ZONES))
            d = int(self.np_random.integers(0, N_ZONES))
            while d == o:
                d = int(self.np_random.integers(0, N_ZONES))
            self.pending_origin = o
            self.pending_dest = d
            self.pending_age = 0

    def _electricity_price(self) -> float:
        """Normalized price in [0.3, 1.0]; higher in evening peak."""
        t = self.step_count / max(1, self.max_steps)
        base = 0.35 + 0.55 * abs(np.sin(np.pi * t))
        noise = float(self.np_random.uniform(-0.05, 0.05))
        return float(np.clip(base + noise, 0.3, 1.0))

    def _traffic_factor(self) -> float:
        t = self.step_count / max(1, self.max_steps)
        return 0.8 if (0.2 <= t <= 0.35 or 0.65 <= t <= 0.85) else 0.2

    def _zone_distance(self, a: int, b: int) -> int:
        if a == b:
            return 0
        if b in ADJACENCY[a]:
            return 1
        return 2

    def _estimate_trip_energy(self, origin: int, dest: int) -> float:
        hops = self._zone_distance(self.zone, origin) + self._zone_distance(
            origin, dest
        )
        return hops * TRAVEL_ENERGY * 1.25

    # ------------------------------------------------------------------
    # Observations / info / serialization
    # ------------------------------------------------------------------
    def _one_hot(self, idx: int, n: int) -> np.ndarray:
        v = np.zeros(n, dtype=np.float32)
        if 0 <= idx < n:
            v[idx] = 1.0
        return v

    def _get_obs(self) -> np.ndarray:
        parts = [
            np.array([self.soc], dtype=np.float32),
            self._one_hot(self.zone, N_ZONES),
            np.array(
                [
                    self.charger_queues[1] / 5.0,
                    self.charger_queues[3] / 5.0,
                ],
                dtype=np.float32,
            ),
            self._one_hot(self.pending_origin, N_ZONES),
            self._one_hot(self.pending_dest, N_ZONES),
            np.array(
                [
                    1.0 if self.pending_origin >= 0 else 0.0,
                    1.0 if self.on_trip else 0.0,
                    (
                        0.0
                        if not self.on_trip
                        else (0.5 if self.trip_phase == 0 else 1.0)
                    ),
                    self.step_count / max(1, self.max_steps),
                    self._electricity_price(),
                    1.0 if self.is_charging else 0.0,
                    1.0 if self.soc < 0.2 else 0.0,
                ],
                dtype=np.float32,
            ),
        ]
        return np.concatenate(parts).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        return {
            "soc": round(self.soc, 4),
            "zone": self.zone,
            "zone_name": ZONE_NAMES[self.zone],
            "is_charging": self.is_charging,
            "on_trip": self.on_trip,
            "pending_origin": self.pending_origin,
            "pending_dest": self.pending_dest,
            "rides_completed": self.rides_completed,
            "rides_missed": self.rides_missed,
            "step": self.step_count,
            "revenue": round(self.total_revenue, 2),
            "energy_cost": round(self.total_energy_cost, 2),
            "charger_queues": dict(self.charger_queues),
            "last_action": self.last_action,
            "last_reward": self.last_reward,
        }

    def to_json(self) -> str:
        """Serialize environment state for web/mobile API frontends."""
        payload = {
            "observation": self._get_obs().tolist(),
            "info": self._get_info(),
            "action_meanings": self.action_meanings(),
            "zones": list(ZONE_NAMES),
            "charger_zones": list(CHARGER_ZONES),
            "metadata": {
                "mission": "EV Fleet Charging & Dispatch",
                "max_steps": self.max_steps,
                "obs_dim": self.obs_dim,
                "n_actions": self.N_ACTIONS,
            },
        }
        return json.dumps(payload)

    @staticmethod
    def action_meanings() -> dict[int, str]:
        return {
            0: "Move to Residential",
            1: "Move to Downtown",
            2: "Move to Industrial",
            3: "Move to Airport",
            4: "Accept ride request",
            5: "Charge at station",
            6: "Stop charging / unplug",
            7: "Idle / wait",
        }

    def render(self):
        if self.render_mode is None:
            return None
        from environment.rendering import EVRenderer

        if self._renderer is None:
            self._renderer = EVRenderer(self)
        return self._renderer.render(mode=self.render_mode)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
