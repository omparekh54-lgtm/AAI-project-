from __future__ import annotations
import json
import os
from .environment.simulator import CabDispatchEnv
from .baselines.nearest_cab import nearest_cab_policy
from .baselines.zone_balancing import zone_balancing_policy
from .metrics.metrics import summarise

POLICIES = {"Nearest Cab": nearest_cab_policy, "Zone Balancing": zone_balancing_policy}

def main(episodes=10):
    all_results = {}
    for name, policy in POLICIES.items():
        rows = []
        for seed in range(42, 42 + episodes):
            env = CabDispatchEnv(seed=seed, fleet_size=8, day_minutes=180)
            rows.append(env.run(policy))
        all_results[name] = summarise(rows)
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    main(parser.parse_args().episodes)
