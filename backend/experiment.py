from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from statistics import mean, pstdev
import torch
from .agents.dqn import SharedDQNAgent
from .baselines.nearest_cab import nearest_cab_policy
from .baselines.zone_balancing import zone_balancing_policy
from .environment.simulator import CabDispatchEnv
from .coordination.predictive import DemandForecast, PredictiveDispatchPolicy, coordinated_matching

METRICS = ('service_rate', 'cancellation_rate', 'avg_wait', 'avg_pickup_distance',
           'empty_distance', 'empty_driving_ratio', 'utilisation', 'earnings', 'earnings_fairness')


def aggregate(rows, metrics=METRICS):
    out = {'episodes': len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        out[f'{metric}_mean'] = mean(values)
        out[f'{metric}_std'] = pstdev(values)
    return out


def dqn_policy(agent):
    def policy(env):
        states = env.observations()
        return {c.cab_id: agent.act(states[c.cab_id], env.valid_actions(c), explore=False) for c in env.cabs}
    return policy


def run_episode(policy, seed, fleet_size, day_minutes):
    env = CabDispatchEnv(seed=seed, fleet_size=fleet_size, day_minutes=day_minutes)
    env.run(policy)
    return env


def run_dqn_episode(agent, seed, fleet_size=8, day_minutes=1440):
    return run_episode(dqn_policy(agent), seed, fleet_size, day_minutes).metrics()


def evaluate_policy(name, policy, seeds, fleet_size, day_minutes):
    rows = [run_episode(policy, seed, fleet_size, day_minutes).metrics() for seed in seeds]
    return {'method': name, **aggregate(rows)}


def period_metrics(env, peak):
    def is_peak(r):
        hour = (r.created_at % 1440) / 60
        return 7 <= hour < 10 or 17 <= hour < 21
    requests = [r for r in env.requests if is_peak(r) == peak]
    picked = [r for r in requests if r.pickup_time is not None]
    return {'service_rate': sum(r.status == 'completed' for r in requests) / max(1, len(requests)),
            'cancellation_rate': sum(r.status == 'cancelled' for r in requests) / max(1, len(requests)),
            'avg_wait': sum(r.wait_minutes for r in picked) / max(1, len(picked))}


def main(evaluation_episodes=20, fleet_sizes=(4, 8, 12, 16), day_minutes=1440, model='models/dqn_latest.pt', forecast='models/demand_forecast.json', evaluation_seed_start=300000):
    if evaluation_episodes < 1:
        raise ValueError('At least one evaluation seed is required')
    # Kept disjoint from the configured training seed range and recorded explicitly.
    seeds = list(range(evaluation_seed_start, evaluation_seed_start + evaluation_episodes))
    methods = [('Nearest Cab', nearest_cab_policy), ('Zone Balancing', zone_balancing_policy)]
    if os.path.exists(forecast):
        demand_forecast = DemandForecast.load(forecast)
        if set(seeds) & set(demand_forecast.training_seeds):
            raise ValueError('Forecast training and evaluation seeds overlap')
        methods += [('Coordinated Matching', coordinated_matching),
                    ('Predictive Coordinated Dispatch', PredictiveDispatchPolicy(demand_forecast, movement_cost=0.05))]
    checkpoint = None
    if os.path.exists(model):
        checkpoint = torch.load(model, map_location='cpu', weights_only=True)
        if checkpoint.get('simulator_version') != 2:
            raise ValueError('Retrain the checkpoint with the corrected simulator before evaluating')
        train_seeds = range(checkpoint['seed'] + 1, checkpoint['seed'] + checkpoint['episodes'] + 1)
        if any(seed in train_seeds for seed in seeds):
            raise ValueError('Training and evaluation seeds overlap')
        agent = SharedDQNAgent(seed=42)
        agent.policy.load_state_dict(checkpoint['model'])
        methods.append(('Parameter-shared DQN', dqn_policy(agent)))
    results, periods, raw = [], [], []
    for fleet in fleet_sizes:
        for name, policy in methods:
            rows, period_rows = [], {True: [], False: []}
            for seed in seeds:
                env = run_episode(policy, seed, fleet, day_minutes)
                row = env.metrics()
                rows.append(row)
                raw.append({'method': name, 'fleet_size': fleet, 'seed': seed, **row})
                if fleet == 8:
                    for peak in period_rows:
                        period_rows[peak].append(period_metrics(env, peak))
            results.append({'method': name, 'fleet_size': fleet, **aggregate(rows)})
            if fleet == 8:
                for peak, values in period_rows.items():
                    periods.append({'method': name, 'period': 'peak' if peak else 'off_peak',
                                    **aggregate(values, ('service_rate', 'cancellation_rate', 'avg_wait'))})
    payload = {'metadata': {'simulator_version': 2,
                           'training_episodes': checkpoint['episodes'] if checkpoint else 0,
                           'forecast_training_seeds': demand_forecast.training_seeds if os.path.exists(forecast) else [],
                           'validation_seeds': list(range(200000, 200005)),
                           'movement_cost': 0.05,
                           'forecast_sha256': hashlib.sha256(open(forecast, 'rb').read()).hexdigest() if os.path.exists(forecast) else None,
                           'training_day_minutes': checkpoint['day_minutes'] if checkpoint else None,
                           'model_sha256': hashlib.sha256(open(model, 'rb').read()).hexdigest() if checkpoint else None,
                           'evaluation_episodes': evaluation_episodes, 'evaluation_seeds': seeds,
                           'day_minutes': day_minutes, 'fleet_sizes': list(fleet_sizes),
                           'note': 'Measured synthetic simulator outputs. Pending end-of-day trips remain unserved. Fares are synthetic units.'},
               'results': results, 'period_results': periods, 'raw_results': raw}
    os.makedirs('results', exist_ok=True)
    with open('results/experiment_results.json', 'w') as handle:
        json.dump(payload, handle, indent=2)
    with open('results/experiment_results.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--fleet-sizes', default='4,8,12,16')
    parser.add_argument('--day-minutes', type=int, default=1440)
    parser.add_argument('--model', default='models/dqn_latest.pt')
    parser.add_argument('--forecast', default='models/demand_forecast.json')
    parser.add_argument('--seed-start', type=int, default=300000)
    args = parser.parse_args()
    main(args.episodes, tuple(int(x) for x in args.fleet_sizes.split(',')), args.day_minutes, args.model, args.forecast, args.seed_start)
