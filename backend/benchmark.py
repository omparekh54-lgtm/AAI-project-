"""Held-out benchmark: rule baselines, central planners and learned (RL) agents.

Every method sees identical demand days. Learned agents are pooled over all
available training seeds so the spread reflects training restarts as well as
demand days. Output: results/experiment_results.{json,csv} and results/summary.md.
"""
from __future__ import annotations
import argparse
import csv
import glob
import hashlib
import json
import os
import re
from statistics import mean, pstdev
import torch
from .agents.dqn import SharedDQNAgent
from .agents.vdn import VDNAgent
from .agents.policies import dqn_policy, HybridMatchingPolicy, VDNPolicy
from .baselines.nearest_cab import nearest_cab_policy
from .baselines.zone_balancing import zone_balancing_policy
from .compare_improvements import paired_interval
from .coordination.predictive import (DemandForecast, PredictiveDispatchPolicy,
                                      IncomingAwareDispatchPolicy, coordinated_matching)
from .environment.simulator import CabDispatchEnv
from .experiment import aggregate, period_metrics, METRICS
from .train import STATE_SIZE

DQN_NAME = 'DQN (decentralised, shared weights)'
HYBRID_NAME = 'DQN + Matching (RL repositioning)'
VDN_NAME = 'VDN (value decomposition)'
CENTRAL_REFERENCE = 'Incoming-aware Coordinated Dispatch'


def load_rl(models_dir='models'):
    """Returns {method name: [(train_seed, policy_factory, checkpoint_path)]}"""
    found = {DQN_NAME: [], HYBRID_NAME: [], VDN_NAME: []}
    for path in sorted(glob.glob(f'{models_dir}/dqn_full_s*.pt') + glob.glob(f'{models_dir}/dqn_hybrid_s*.pt')
                       + glob.glob(f'{models_dir}/vdn_s*.pt')):
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        if ckpt.get('simulator_version') != 3:
            raise ValueError(f'{path}: retrain with the current simulator')
        seed = ckpt['seed']
        if 'vdn' in path:
            agent = VDNAgent(state_size=STATE_SIZE)
            agent.policy.load_state_dict(ckpt['model'])
            found[VDN_NAME].append((seed, VDNPolicy(agent), path, ckpt))
        else:
            agent = SharedDQNAgent(state_size=STATE_SIZE)
            agent.policy.load_state_dict(ckpt['model'])
            if ckpt['mode'] == 'hybrid':
                found[HYBRID_NAME].append((seed, HybridMatchingPolicy(agent), path, ckpt))
            else:
                found[DQN_NAME].append((seed, dqn_policy(agent), path, ckpt))
    return {k: v for k, v in found.items() if v}


def central_methods(forecast_path='models/demand_forecast.json'):
    methods = [('Nearest Cab', nearest_cab_policy), ('Zone Balancing', zone_balancing_policy)]
    forecast = DemandForecast.load(forecast_path)
    methods += [('Coordinated Matching', coordinated_matching),
                ('Predictive Coordinated Dispatch', PredictiveDispatchPolicy(forecast, movement_cost=0.05))]
    config = json.load(open('models/incoming_config.json'))
    methods.append((CENTRAL_REFERENCE, IncomingAwareDispatchPolicy(forecast, **config)))
    return methods, forecast


def run(policy, seed, fleet, reach, day_minutes):
    env = CabDispatchEnv(seed=seed, fleet_size=fleet, day_minutes=day_minutes, dispatch_reach=reach)
    env.run(policy)
    return env


def main(episodes=10, fleet_sizes=(4, 8, 12, 16), day_minutes=1440, seed_start=900000):
    torch.set_num_threads(1)
    seeds = list(range(seed_start, seed_start + episodes))
    central, forecast = central_methods()
    if set(seeds) & set(forecast.training_seeds):
        raise ValueError('forecast and evaluation seeds overlap')
    rl = load_rl()
    for entries in rl.values():
        for train_seed, _, _, ckpt in entries:
            if any(s in range(train_seed + 1, train_seed + ckpt['episodes'] + 1) for s in seeds):
                raise ValueError('training and evaluation seeds overlap')
    results, periods, raw = [], [], []
    spread = {}
    for fleet in fleet_sizes:
        for name, policy in central + [(n, None) for n in rl]:
            entries = [(None, policy, False)] if policy else [(e[0], e[1], True) for e in rl[name]]
            rows, period_rows = [], {True: [], False: []}
            per_train = {}
            for train_seed, pol, reach in entries:
                for seed in seeds:
                    env = run(pol, seed, fleet, reach, day_minutes)
                    row = env.metrics()
                    rows.append(row)
                    per_train.setdefault(train_seed, []).append(row)
                    raw.append({'method': name, 'fleet_size': fleet, 'seed': seed, 'train_seed': train_seed, **row})
                    if fleet == 8:
                        for peak in period_rows:
                            period_rows[peak].append(period_metrics(env, peak))
            results.append({'method': name, 'fleet_size': fleet, **aggregate(rows)})
            if len(per_train) > 1:
                spread[f'{name}|{fleet}'] = {m: pstdev([mean(r[m] for r in v) for v in per_train.values()])
                                              for m in ('service_rate', 'avg_wait', 'empty_distance')}
            if fleet == 8:
                for peak, values in period_rows.items():
                    periods.append({'method': name, 'period': 'peak' if peak else 'off_peak',
                                    **aggregate(values, ('service_rate', 'cancellation_rate', 'avg_wait'))})
            print(f'fleet {fleet:2d} | {name[:40]:40s} | served {100*results[-1]["service_rate_mean"]:5.1f}% '
                  f'| wait {results[-1]["avg_wait_mean"]:.2f} | empty {results[-1]["empty_distance_mean"]:.0f}', flush=True)

    # Paired comparison isolating learned repositioning: same matcher, with/without RL.
    paired = []
    if HYBRID_NAME in rl:
        for fleet in fleet_sizes:
            for baseline in ('Coordinated Matching', 'Zone Balancing', 'Nearest Cab'):
                entry = {'fleet_size': fleet, 'method': HYBRID_NAME, 'versus': baseline}
                base_rows = [r for r in raw if r['method'] == baseline and r['fleet_size'] == fleet]
                hyb_rows = [r for r in raw if r['method'] == HYBRID_NAME and r['fleet_size'] == fleet]
                for metric in ('avg_wait', 'service_rate', 'empty_distance'):
                    deltas = [mean(h[metric] for h in hyb_rows if h['seed'] == b['seed']) - b[metric] for b in base_rows]
                    entry[metric] = paired_interval(deltas)
                paired.append(entry)

    metadata = {'simulator_version': 3, 'day_minutes': day_minutes, 'fleet_sizes': list(fleet_sizes),
                'evaluation_episodes': episodes, 'evaluation_seeds': seeds,
                'forecast_training_seeds': forecast.training_seeds,
                'incoming_config': json.load(open('models/incoming_config.json')),
                'rl_checkpoints': {n: [{'train_seed': e[0], 'path': e[2], 'episodes': e[3]['episodes'],
                                        'sha256': hashlib.sha256(open(e[2], 'rb').read()).hexdigest()} for e in v]
                                   for n, v in rl.items()},
                'training_episodes': max((e[3]['episodes'] for v in rl.values() for e in v), default=0),
                'training_seed_spread': spread,
                'note': ('Measured synthetic simulator outputs. Learned agents are pooled over training seeds; '
                         'their std covers demand days and training restarts. Fares are synthetic units. '
                         'Pending end-of-day trips remain unserved.')}
    payload = {'metadata': metadata, 'results': results, 'period_results': periods,
               'paired_results': paired, 'raw_results': raw}
    os.makedirs('results', exist_ok=True)
    json.dump(payload, open('results/experiment_results.json', 'w'), indent=1)
    with open('results/experiment_results.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(results)
    write_summary(payload)
    return payload


def write_summary(payload):
    meta = payload['metadata']
    lines = ['# Measured results', '',
             f"{meta['evaluation_episodes']} held-out demand days per fleet (seeds {meta['evaluation_seeds'][0]}–{meta['evaluation_seeds'][-1]}), "
             'synthetic 5×5 city, 24-hour days. Learned agents pooled over training seeds.', '']
    for fleet in meta['fleet_sizes']:
        lines += [f'## {fleet} cabs', '', '| Method | Served | Wait (min) | Empty zones/day | Fairness |', '| --- | ---: | ---: | ---: | ---: |']
        for r in payload['results']:
            if r['fleet_size'] == fleet:
                lines.append(f"| {r['method']} | {100*r['service_rate_mean']:.1f}% | {r['avg_wait_mean']:.2f} | "
                             f"{r['empty_distance_mean']:.0f} | {r['earnings_fairness_mean']:.2f} |")
        lines.append('')
    lines += ['## Learned repositioning vs the same matcher without learning (paired, 95% t interval)', '',
              '| Fleet | Versus | Wait Δ min | Served Δ pp | Empty-distance Δ |', '| --- | --- | ---: | ---: | ---: |']
    for p in payload['paired_results']:
        w, s, e = p['avg_wait'], p['service_rate'], p['empty_distance']
        lines.append(f"| {p['fleet_size']} | {p['versus']} | {w['mean_delta']:.3f} [{w['ci95_low']:.3f}, {w['ci95_high']:.3f}] | "
                     f"{100*s['mean_delta']:.2f} | {e['mean_delta']:.0f} [{e['ci95_low']:.0f}, {e['ci95_high']:.0f}] |")
    lines += ['', 'Negative wait and empty-distance deltas favour the learned method. Intervals cover demand-day variability.']
    open('results/summary.md', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--fleet-sizes', default='4,8,12,16')
    parser.add_argument('--seed-start', type=int, default=900000)
    a = parser.parse_args()
    main(a.episodes, tuple(int(x) for x in a.fleet_sizes.split(',')), 1440, a.seed_start)
