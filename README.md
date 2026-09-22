# Multi-Agent Cab Dispatch using Reinforcement Learning

A research-grade prototype for decentralized multi-agent cab dispatch on a 5×5 simulated city grid with 8 cabs.

## What this project contains

- 5×5 grid-city simulator
- 8 cab agents
- Synthetic demand with normal, morning peak, evening peak and airport hotspot behavior
- Nearest-cab and zone-balancing baselines
- Parameter-shared DQN agent with replay buffer and target network
- Local observations and neighbour information
- Hybrid individual/shared reward design
- Evaluation metrics: wait time, service rate, cancellation rate, empty driving, utilisation and earnings
- Interactive browser dashboard for simulation and results
- Python training/evaluation implementation for reproducible experiments

## Architecture

Long-running RL training is intentionally kept in Python rather than inside Vercel request handlers. The browser dashboard is deployed as a static application and visualizes the simulator and measured experiment outputs.

```text
Demand generator → Multi-agent environment → Baselines / DQN → Metrics → Results
                                      ↓
                              Interactive dashboard
```

## Run the dashboard locally

Open `frontend/index.html` in a browser or serve the repository with any static server.

## Run the Python simulator

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m backend.demo
```

## Train DQN

```bash
python -m backend.train --episodes 1000
```

For a quick smoke test:

```bash
python -m backend.train --episodes 10
```

## Evaluate baselines and DQN

```bash
python -m backend.evaluate --episodes 100
```

## Research formulation

Each cab observes a local state containing its position, time, status, nearby requests and local/neighboring idle-cab information. The conceptual action set is accept, wait, or reposition. The implementation expands reposition into valid directional moves on the grid.

The project is intentionally staged:

1. Nearest-cab baseline
2. Zone-balancing baseline
3. Parameter-shared DQN
4. Neighbour-aware/shared-reward DQN
5. Optional VDN/QMIX extension

No performance claim is made until the evaluation scripts produce measured results.

## Repository structure

```text
backend/
  environment/   city, demand, cabs and simulation
  agents/        DQN and replay buffer
  baselines/     deterministic benchmark policies
  metrics/       experiment metrics
  demo.py        simulator smoke demo
  train.py       DQN training entry point
  evaluate.py    benchmark entry point
frontend/
  index.html     deployed research dashboard
  styles.css     dashboard styling
  app.js         interactive simulation and charts
configs/         experiment configuration
results/         generated experiment outputs
models/          generated model checkpoints
```

## Deployment

The frontend is Vercel-friendly because it has no build-time dependency on a Python runtime. Training remains reproducible in Python and can be moved to a GPU/compute service later if larger experiments are required.
