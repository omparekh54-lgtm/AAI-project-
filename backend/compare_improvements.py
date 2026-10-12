"""Fresh paired evaluation and confidence intervals for the coordination extension."""
from __future__ import annotations
import json
from pathlib import Path
from statistics import mean
import numpy as np
from scipy.stats import t
from .experiment import main, METRICS, run_episode, dqn_policy
from .coordination.predictive import DemandForecast, PredictiveDispatchPolicy, coordinated_matching
from .baselines.nearest_cab import nearest_cab_policy
from .baselines.zone_balancing import zone_balancing_policy
from .environment.simulator import CabDispatchEnv


def paired_interval(values):
    values = np.asarray(values, dtype=float)
    center = float(values.mean())
    if len(values) < 2:
        return {'mean_delta': center, 'ci95_low': None, 'ci95_high': None}
    margin = float(t.ppf(0.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {'mean_delta': center, 'ci95_low': center-margin, 'ci95_high': center+margin}


def compare(raw):
    comparisons = []
    proposed = {(r['fleet_size'], r['seed']): r for r in raw if r['method'] == 'Predictive Coordinated Dispatch'}
    for fleet in sorted({r['fleet_size'] for r in raw}):
        for baseline in ['Nearest Cab', 'Zone Balancing', 'Coordinated Matching', 'Parameter-shared DQN']:
            base = [r for r in raw if r['fleet_size'] == fleet and r['method'] == baseline]
            if not base:
                continue
            item = {'fleet_size': fleet, 'baseline': baseline, 'paired_seeds': len(base)}
            for metric in METRICS:
                differences = [proposed[fleet, r['seed']][metric] - r[metric] for r in base]
                item[metric] = paired_interval(differences)
            comparisons.append(item)
    return comparisons


def stress_benchmark(forecast, episodes=20):
    # Sensitivity test: each arrival probability is increased by 50%. Each
    # method sees identical seeds and transitions. Model is not refitted.
    methods = [('Nearest Cab', nearest_cab_policy), ('Zone Balancing', zone_balancing_policy),
               ('Predictive Coordinated Dispatch', PredictiveDispatchPolicy(forecast))]
    rows = []
    for name, policy in methods:
        for seed in range(400000, 400000 + episodes):
            env = CabDispatchEnv(seed=seed)
            original = env.demand._rate
            env.demand._rate = lambda minute, rate=original: rate(minute) * 1.5
            # Initial minute-zero requests have already been generated uniformly
            # for every method. All later arrival probabilities use 1.5x demand.
            while env.minute < env.day_minutes:
                env.step(policy(env))
            rows.append({'method': name, 'fleet_size': 8, 'seed': seed, **env.metrics()})
    return {'scenario': '1.5x arrival probabilities after minute zero; no retraining',
            'raw_results': rows, 'paired_comparisons': compare(rows)}


def write_summary(payload, review):
    lines = ['# Measured improvement review', '',
             'Synthetic 5×5 grid, 24-hour days, one fixed trained forecast and one DQN training seed.',
             f"{payload['metadata']['evaluation_episodes']} fresh paired demand seeds per fleet. No tuning on the reported test seeds.", '',
             '## Eight-cab pilot', '',
             '| Method | Service rate | Avg wait (min) | Empty zones/day | Earnings fairness |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in payload['results']:
        if row['fleet_size'] == 8:
            lines.append(f"| {row['method']} | {100*row['service_rate_mean']:.2f}% | {row['avg_wait_mean']:.3f} | {row['empty_distance_mean']:.1f} | {row['earnings_fairness_mean']:.3f} |")
    lines += ['', '## Paired differences from zone balancing', '',
              'Differences are proposed controller minus baseline. Lower wait and empty distance are better.', '',
              '| Fleet | Service difference (pp) | Wait difference (min) | Empty distance difference |',
              '| --- | ---: | ---: | ---: |']
    for row in review['normal_scenario_paired_comparisons']:
        if row['baseline'] == 'Zone Balancing':
            lines.append(f"| {row['fleet_size']} | {100*row['service_rate']['mean_delta']:.3f} | {row['avg_wait']['mean_delta']:.3f} | {row['empty_distance']['mean_delta']:.1f} |")
    pilot = next(row for row in review['normal_scenario_paired_comparisons'] if row['baseline']=='Zone Balancing' and row['fleet_size']==8)
    lines += ['', 'Eight-cab paired 95% t confidence intervals:', '']
    for metric in ('avg_wait', 'empty_distance', 'service_rate'):
        value = pilot[metric]
        lines.append(f"- {metric}: [{value['ci95_low']:.5f}, {value['ci95_high']:.5f}]")
    lines += ['', '## Stress test', '',
              f"Arrival probabilities increase by 50%, using {len(review['stress_test']['raw_results']) // 3} further fresh seeds and no retraining.", '',
              '| Method | Service rate | Avg wait (min) | Empty zones/day |',
              '| --- | ---: | ---: | ---: |']
    stress = review['stress_test']['raw_results']
    for name in sorted({row['method'] for row in stress}):
        rows = [row for row in stress if row['method']==name]
        lines.append(f"| {name} | {100*mean(row['service_rate'] for row in rows):.2f}% | {mean(row['avg_wait'] for row in rows):.3f} | {mean(row['empty_distance'] for row in rows):.1f} |")
    lines += ['', '## Interpretation and limits', '',
              'The forecast-and-optimisation controller improves the pilot wait/empty-driving trade-off against static zone balancing. Service gain versus zone balancing is not statistically established in the normal pilot.',
              'Compared with nearest-cab dispatch, the controller serves more requests and reduces wait, but increases empty driving. At 16 cabs, zone balancing has lower wait, while the controller uses much less empty driving.',
              'Plain DQN still underperforms. Do not attribute the controller gains to DQN or claim universal dominance. The extension uses central coordination, beyond the original strictly local execution design.',
              'Results concern synthetic demand. Confidence intervals cover demand-day variability, not training restarts or real-city outcomes. Overnight off-shift cancellations limit all service rates.', '']
    Path('results/improvement_summary.md').write_text('\n'.join(lines))


def main_review(episodes=20):
    payload = main(evaluation_episodes=episodes, evaluation_seed_start=300000)
    review = {'normal_scenario_paired_comparisons': compare(payload['raw_results']),
              'stress_test': stress_benchmark(DemandForecast.load('models/demand_forecast.json'), episodes),
              'interpretation': 'Paired 95% t intervals over independent demand seeds. Conditional on one trained forecast and one DQN training seed; not across training restarts. No tuning on these evaluation seeds.'}
    Path('results/improvement_review.json').write_text(json.dumps(review, indent=2))
    write_summary(payload, review)
    print('Saved paired improvement and stress tests', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=20)
    main_review(parser.parse_args().episodes)
