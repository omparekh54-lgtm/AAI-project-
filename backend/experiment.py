from __future__ import annotations

import argparse
import csv
import json
import os
from statistics import mean, pstdev

import torch

from .agents.dqn import SharedDQNAgent
from .baselines.nearest_cab import nearest_cab_policy
from .baselines.zone_balancing import zone_balancing_policy
from .environment.simulator import CabDispatchEnv


def valid_actions(env: CabDispatchEnv, cab) -> list[int]:
    actions = [env.ACTION_WAIT, env.ACTION_ACCEPT]
    x, y = cab.position
    if x > 0:
        actions.append(env.ACTION_NORTH)
    if x < 4:
        actions.append(env.ACTION_SOUTH)
    if y > 0:
        actions.append(env.ACTION_WEST)
    if y < 4:
        actions.append(env.ACTION_EAST)
    return actions


def metrics_from_env(env: CabDispatchEnv) -> dict:
    return {
        "minutes": env.day_minutes,
        "requests": len(env.requests),
        "completed": env.completed,
        "cancelled": env.cancelled,
        "service_rate": env.completed / max(1, env.completed + env.cancelled),
        "avg_wait": env.total_wait / max(1, env.completed),
        "empty_distance": env.total_empty_distance,
        "empty_driving_ratio": env.total_empty_distance / max(
            1, env.total_empty_distance + sum(c.busy_minutes for c in env.cabs)
        ),
        "utilisation": sum(c.busy_minutes for c in env.cabs)
        / max(1, env.fleet_size * env.day_minutes),
        "earnings": env.total_fare,
    }


def run_dqn_episode(agent: SharedDQNAgent, seed: int, fleet_size: int = 8, day_minutes: int = 180) -> dict:
    env = CabDispatchEnv(seed=seed, fleet_size=fleet_size, day_minutes=day_minutes)
    env.reset()
    while env.minute < env.day_minutes:
        states = env.observations()
        actions = {}
        for cab in env.cabs:
            if cab.status == "idle":
                actions[cab.cab_id] = agent.act(
                    states[cab.cab_id], valid_actions(env, cab), explore=False
                )
            else:
                actions[cab.cab_id] = env.ACTION_WAIT
        env.step(actions)
    return metrics_from_env(env)


def load_agent(path: str) -> SharedDQNAgent:
    agent = SharedDQNAgent(gamma=0.95, seed=42)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    agent.policy.load_state_dict(state)
    agent.target.load_state_dict(agent.policy.state_dict())
    agent.epsilon = 0.0
    return agent


def aggregate(rows: list[dict]) -> dict:
    metrics = [
        "service_rate",
        "avg_wait",
        "empty_distance",
        "empty_driving_ratio",
        "utilisation",
        "earnings",
    ]
    out = {"episodes": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        out[f"{metric}_mean"] = mean(values) if values else 0.0
        out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return out


def evaluate_policy(name: str, policy, seeds: list[int], fleet_size: int, day_minutes: int) -> dict:
    rows = []
    for seed in seeds:
        env = CabDispatchEnv(seed=seed, fleet_size=fleet_size, day_minutes=day_minutes)
        rows.append(env.run(policy))
    return {"method": name, **aggregate(rows)}


def main(episodes: int = 20, fleet_size: int = 8, day_minutes: int = 180, model: str = "models/dqn_latest.pt") -> None:
    seeds = list(range(1000, 1000 + episodes))
    results = [
        evaluate_policy("Nearest Cab", nearest_cab_policy, seeds, fleet_size, day_minutes),
        evaluate_policy("Zone Balancing", zone_balancing_policy, seeds, fleet_size, day_minutes),
    ]

    if os.path.exists(model):
        agent = load_agent(model)
        rows = [run_dqn_episode(agent, seed, fleet_size, day_minutes) for seed in seeds]
        results.append({"method": "Parameter-shared DQN", **aggregate(rows)})

    os.makedirs("results", exist_ok=True)
    with open("results/experiment_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    fieldnames = sorted({key for row in results for key in row})
    with open("results/experiment_results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reproducible cab-dispatch experiments.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--fleet-size", type=int, default=8)
    parser.add_argument("--day-minutes", type=int, default=180)
    parser.add_argument("--model", default="models/dqn_latest.pt")
    args = parser.parse_args()
    main(args.episodes, args.fleet_size, args.day_minutes, args.model)
