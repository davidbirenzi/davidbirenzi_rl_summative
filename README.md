# EV Dispatch RL Summative

Mission-based reinforcement learning project: train an agent to **dispatch an electric vehicle**, complete ride requests across a city, and **decide when/where to charge** under dynamic electricity prices and charger congestion.

Algorithms compared on the **same** custom Gymnasium environment:
- **DQN** (value-based)
- **REINFORCE** (custom PyTorch policy gradient)
- **PPO** and **A2C** (Stable-Baselines3)

> Rename this repository to `yourname_rl_summative` before submission (assignment naming rule).

## Why this mission?

EV adoption is growing across African cities. A practical operator problem is: *take more paid trips without stranding the vehicle with an empty battery, while avoiding expensive peak charging and congested stations.* This environment turns that into a learnable MDP that can later plug into a dispatcher API / mobile dashboard via JSON state export.

## Project layout

```
project_root/
├── pyproject.toml
├── uv.lock
├── README.md
├── main.py
├── play.py
├── environment/
│   ├── custom_env.py
│   └── rendering.py
├── training/
│   ├── dqn_training.py
│   ├── pg_training.py
│   └── hyperparams.py
├── models/
├── logs/
├── assets/
├── scripts/
│   └── generate_report_plots.py
└── tests/
```

## Setup (uv only)

Install [uv](https://docs.astral.sh/uv/) if needed, then:

```bash
uv sync
```

Optional dev/tests:

```bash
uv sync --extra dev
uv run pytest -q
```

## Run

### Environment demo (random agent + GUI)

```bash
uv run main.py
# or
uv run main.py --mode demo --episodes 2
```

### JSON API sample (frontend-ready state)

```bash
uv run main.py --mode json
```

### Train

```bash
# Single algorithm (10 hyperparameter runs)
uv run main.py --mode train --algo dqn --timesteps 80000
uv run main.py --mode train --algo reinforce
uv run main.py --mode train --algo ppo --timesteps 80000
uv run main.py --mode train --algo a2c --timesteps 80000

# Everything
uv run main.py --mode train --algo all --timesteps 80000
```

Direct module entry points also work:

```bash
uv run python -m training.dqn_training --timesteps 80000
uv run python -m training.pg_training --algo ppo --timesteps 80000
```

### Play best agent (video demo)

```bash
uv run play.py --algo ppo --episodes 2
uv run play.py --algo dqn --episodes 2
```

Shows **GUI + verbose terminal** (action, SOC, zone, rides, rewards).

### Report plots + generalization

```bash
uv run python scripts/generate_report_plots.py
```

## Environment summary

| Item | Detail |
|------|--------|
| Agent | One EV during a city shift |
| Zones | Residential, Downtown, Industrial, Airport |
| Chargers | Downtown & Airport (queue congestion) |
| Actions (8) | Move to zone 0–3, accept ride, charge, unplug, idle |
| Observations | SOC, zone, charger queues, ride OD, trip flags, time, price, charging flag |
| Rewards | Fare for completed trips; penalties for missed rides, illegal moves, dead battery; price-aware charging cost |
| Terminal | Battery depleted, or shift timeout (`max_steps=80`) |

Real-world observation mapping (for the report table): SOC ← BMS; zone ← GPS; queues ← charger network API; ride requests ← dispatch API; price ← utility/TOU tariff API.

## Notes for markers

1. `uv sync` then `uv run main.py` should work with no manual venv setup.
2. REINFORCE is implemented in PyTorch because Stable-Baselines3 does not ship REINFORCE.
3. `env.to_json()` serializes state for a web/mobile frontend / API.
4. Hyperparameter grids: 10 configs each in `training/hyperparams.py`.
