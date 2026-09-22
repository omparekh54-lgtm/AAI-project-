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


def metrics_from_env(env):
    return env.run(lambda _env: {c.cab_id: env.ACTION_WAIT for c in env.cabs}) if False else {
        "minutes": env.day_minutes,
        "requests": env.total_requests,
        "completed": env.completed,
        "cancelled": env.cancelled,
        "service_rate": env.completed / max(1, env.completed + env.cancelled),
        "avg_wait": env.total_wait / max(1, env.completed),
        "avg_pickup_distance": env.total_pickup_distance / max(1, env.completed),
        "empty_distance": env.total_empty_distance,
        "empty_driving_ratio": env.total_empty_distance / max(1, env.total_empty_distance + sum(c.busy_minutes for c in env.cabs)),
        "utilisation": sum(c.busy_minutes for c in env.cabs) / max(1, env.fleet_size * env.day_minutes),
        "earnings": sum(c.earnings for c in env.cabs),
    }


def run_dqn_episode(agent, seed, fleet_size=8, day_minutes=1440):
    env = CabDispatchEnv(seed=seed, fleet_size=fleet_size, day_minutes=day_minutes)
    states = env.reset()
    while env.minute < env.day_minutes:
        actions = {}
        for cab in env.cabs:
            if cab.status == "idle":
                actions[cab.cab_id] = agent.act(states[cab.cab_id], env.valid_actions(cab), explore=False)
            else:
                actions[cab.cab_id] = env.ACTION_WAIT
        env.step(actions)
        states = env.observations()
    return env.run(lambda _env: {c.cab_id: _env.ACTION_WAIT for c in _env.cabs}) if False else {
        "minutes": env.day_minutes,
        "requests": env.total_requests,
        "completed": env.completed,
        "cancelled": env.cancelled,
        "service_rate": env.completed / max(1, env.completed + env.cancelled),
        "avg_wait": env.total_wait / max(1, env.completed),
        "avg_pickup_distance": env.total_pickup_distance / max(1, env.completed),
        "empty_distance": env.total_empty_distance,
        "empty_driving_ratio": env.total_empty_distance / max(1, env.total_empty_distance + sum(c.busy_minutes for c in env.cabs)),
        "utilisation": sum(c.busy_minutes for c in env.cabs) / max(1, env.fleet_size * env.day_minutes),
        "earnings": sum(10.0 + 3.0 * r.distance for r in env.requests if r.status == "completed"),
        "earnings_fairness": _fairness(env),
    }


def _fairness(env):
    earnings = [sum(10.0 + 3.0 * r.distance for r in env.requests if r.status == "completed" and r.assigned_cab == c.cab_id) for c in env.cabs]
    m = sum(earnings) / max(1, len(earnings))
    if m <= 0:
        return 0.0
    return max(0.0, 1.0 - (((sum((x - m) ** 2 for x in earnings) / max(1, len(earnings))) ** 0.5) / m))


def evaluate_policy(name, policy, seeds, fleet_size, day_minutes):
    rows = []
    for seed in seeds:
        env = CabDispatchEnv(seed=seed, fleet_size=fleet_size, day_minutes=day_minutes)
        env.reset()
        while env.minute < env.day_minutes:
            env.step(policy(env))
        rows.append({
            "minutes": env.day_minutes, "requests": env.total_requests, "completed": env.completed,
            "cancelled": env.cancelled, "service_rate": env.completed / max(1, env.completed + env.cancelled),
            "avg_wait": env.total_wait / max(1, env.completed),
            "avg_pickup_distance": env.total_pickup_distance / max(1, env.completed),
            "empty_distance": env.total_empty_distance,
            "empty_driving_ratio": env.total_empty_distance / max(1, env.total_empty_distance + sum(c.busy_minutes for c in env.cabs)),
            "utilisation": sum(c.busy_minutes for c in env.cabs) / max(1, fleet_size * day_minutes),
            "earnings": sum(10.0 + 3.0 * r.distance for r in env.requests if r.status == "completed"),
            "earnings_fairness": _fairness(env),
        })
    return {"method": name, **aggregate(rows)}


def aggregate(rows):
    metrics = ["service_rate", "avg_wait", "avg_pickup_distance", "empty_distance", "empty_driving_ratio", "utilisation", "earnings", "earnings_fairness"]
    out = {"episodes": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        out[f"{metric}_mean"] = mean(values) if values else 0.0
        out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return out


def main(evaluation_episodes=20, fleet_sizes=(4, 8, 12, 16), day_minutes=1440, model="models/dqn_latest.pt"):
    seeds = list(range(1000, 1000 + evaluation_episodes))
    methods = [("Nearest Cab", nearest_cab_policy), ("Zone Balancing", zone_balancing_policy)]
    all_results = []
    for fleet in fleet_sizes:
        for name, policy in methods:
            result = evaluate_policy(name, policy, seeds, fleet, day_minutes)
            result["fleet_size"] = fleet
            all_results.append(result)

    if os.path.exists(model):
        agent = SharedDQNAgent(gamma=0.95, seed=42)
        checkpoint = torch.load(model, map_location="cpu", weights_only=False)
        agent.policy.load_state_dict(checkpoint["model"])
        agent.target.load_state_dict(agent.policy.state_dict())
        agent.epsilon = 0.0
        for fleet in fleet_sizes:
            rows = [run_dqn_episode(agent, seed, fleet, day_minutes) for seed in seeds]
            result = {"method": "Parameter-shared DQN", **aggregate(rows), "fleet_size": fleet}
            all_results.append(result)

    # One 8-cab split by demand period for the requested peak/off-peak analysis.
    period_results = []
    for fleet in [8]:
        for name, policy in methods:
            rows = []
            for seed in seeds:
                env = CabDispatchEnv(seed=seed, fleet_size=fleet, day_minutes=day_minutes)
                env.reset()
                while env.minute < env.day_minutes:
                    env.step(policy(env))
                for period, lo, hi in (("off_peak", 0, 7), ("morning_peak", 7, 10), ("midday", 11, 16), ("evening_peak", 17, 21)):
                    reqs = [r for r in env.requests if lo <= (r.created_at % 1440) / 60 < hi]
                    done = [r for r in reqs if r.status == "completed"]
                    cancelled = [r for r in reqs if r.status == "cancelled"]
                    rows.append({"method": name, "period": period, "service_rate": len(done) / max(1, len(done) + len(cancelled)),
                                 "avg_wait": sum(r.wait_minutes for r in done) / max(1, len(done))})
            for period in ("off_peak", "morning_peak", "midday", "evening_peak"):
                p = [r for r in rows if r["period"] == period]
                period_results.append({"method": name, "period": period, "service_rate_mean": mean(r["service_rate"] for r in p),
                                       "avg_wait_mean": mean(r["avg_wait"] for r in p)})

    os.makedirs("results", exist_ok=True)
    payload = {"metadata": {"training_episodes": 360, "evaluation_episodes": evaluation_episodes,
                             "day_minutes": day_minutes, "fleet_sizes": list(fleet_sizes),
                             "note": "Measured simulator outputs; synthetic demand profile is a calibration surface."},
               "results": all_results, "period_results": period_results}
    with open("results/experiment_results.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open("results/experiment_results.csv", "w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in all_results for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(all_results)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--fleet-sizes", default="4,8,12,16")
    parser.add_argument("--day-minutes", type=int, default=1440)
    parser.add_argument("--model", default="models/dqn_latest.pt")
    args = parser.parse_args()
    main(args.episodes, tuple(int(x) for x in args.fleet_sizes.split(",")), args.day_minutes, args.model)
