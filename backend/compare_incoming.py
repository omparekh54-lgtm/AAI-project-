"""Fresh evaluation of the incoming-aware extension against prior controller."""
import json
from pathlib import Path
from statistics import mean
from .experiment import main, METRICS
from .compare_improvements import paired_interval
from .coordination.predictive import DemandForecast, IncomingAwareDispatchPolicy, PredictiveDispatchPolicy
from .baselines.nearest_cab import nearest_cab_policy
from .baselines.zone_balancing import zone_balancing_policy
from .environment.simulator import CabDispatchEnv

NAME = 'Incoming-aware Coordinated Dispatch'


def comparisons(raw):
    proposed = {(row['fleet_size'],row['seed']):row for row in raw if row['method']==NAME}
    out=[]
    for fleet in sorted({row['fleet_size'] for row in raw}):
        for method in sorted({row['method'] for row in raw if row['method']!=NAME}):
            rows=[row for row in raw if row['fleet_size']==fleet and row['method']==method]
            out.append({'fleet_size':fleet,'baseline':method,'paired_seeds':len(rows),
                        **{metric:paired_interval([proposed[fleet,row['seed']][metric]-row[metric] for row in rows]) for metric in METRICS}})
    return out


def write_summary(payload,review):
    n=payload['metadata']['evaluation_episodes']
    lines=['# Incoming-aware dispatch: measured results','',
           f'{n} new paired demand seeds per fleet, 24-hour days, synthetic 5×5 city. No test-seed tuning.', '',
           '| Eight-cab method | Served | Wait (min) | Empty zones/day | Fairness |',
           '| --- | ---: | ---: | ---: | ---: |']
    for row in payload['results']:
        if row['fleet_size']==8:
            lines.append(f"| {row['method']} | {100*row['service_rate_mean']:.2f}% | {row['avg_wait_mean']:.3f} | {row['empty_distance_mean']:.1f} | {row['earnings_fairness_mean']:.3f} |")
    lines+=['','## Paired change from the previous controller','',
            'Negative waiting/empty-driving changes are improvements. Intervals are paired 95% t intervals.', '',
            '| Fleet | Wait delta [95% CI], min | Empty-distance delta [95% CI] | Service delta, pp |',
            '| --- | ---: | ---: | ---: |']
    for row in review['normal_comparisons']:
        if row['baseline']=='Predictive Coordinated Dispatch':
            w,e=row['avg_wait'],row['empty_distance']
            lines.append(f"| {row['fleet_size']} | {w['mean_delta']:.3f} [{w['ci95_low']:.3f}, {w['ci95_high']:.3f}] | {e['mean_delta']:.1f} [{e['ci95_low']:.1f}, {e['ci95_high']:.1f}] | {100*row['service_rate']['mean_delta']:.3f} |")
    lines+=['','## Higher-demand stress test','',
            '| Eight-cab method | Served | Wait (min) | Empty zones/day |',
            '| --- | ---: | ---: | ---: |']
    raw=review['stress_raw_results']
    for method in sorted({row['method'] for row in raw}):
        rows=[row for row in raw if row['method']==method]
        lines.append(f"| {method} | {100*mean(row['service_rate'] for row in rows):.2f}% | {mean(row['avg_wait'] for row in rows):.3f} | {mean(row['empty_distance'] for row in rows):.1f} |")
    lines+=['','## Scope and reproducibility','',
            'The new controller considers known trip destinations/remaining travel and current unmatched requests. Future arrivals come only from the training forecast.',
            'Its parameters were selected on validation seeds 500000–500004. Normal test seeds start at 600000, and stress seeds at 700000. The stress test raises arrival probabilities by 50% after minute zero without refitting.',
            'This remains a forecast/optimisation extension with central coordination. It does not establish plain DQN superiority or real-city performance. Each interval covers demand-seed variability, conditional on a fixed forecast and DQN training run.',
            'Judge the recorded service, wait, earnings and empty-driving results together. No universal dominance is claimed. Overnight shift constraints affect all service rates.', '']
    Path('results/incoming_summary.md').write_text('\n'.join(lines))


def review(episodes=20):
    payload=main(evaluation_episodes=episodes,evaluation_seed_start=600000)
    forecast=DemandForecast.load('models/demand_forecast.json')
    config=json.load(open('models/incoming_config.json'))
    methods=[('Nearest Cab',nearest_cab_policy),('Zone Balancing',zone_balancing_policy),
             ('Predictive Coordinated Dispatch',PredictiveDispatchPolicy(forecast)),
             (NAME,IncomingAwareDispatchPolicy(forecast,**config))]
    rows=[]
    for name,policy in methods:
        for seed in range(700000,700000+episodes):
            env=CabDispatchEnv(seed=seed)
            original=env.demand._rate
            env.demand._rate=lambda minute, rate=original:rate(minute)*1.5
            while env.minute<env.day_minutes:
                env.step(policy(env))
            rows.append({'method':name,'seed':seed,'fleet_size':8,**env.metrics()})
    result={'normal_comparisons':comparisons(payload['raw_results']),
            'stress_raw_results':rows,'stress_comparisons':comparisons(rows),
            'selected_config':config,'validation_seeds':list(range(500000,500005))}
    Path('results/incoming_review.json').write_text(json.dumps(result,indent=2))
    write_summary(payload,result)
    print('Saved fresh incoming-aware benchmark',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=20)
    args=parser.parse_args()
    if args.episodes<2:
        parser.error('At least two seeds are needed for paired intervals')
    review(args.episodes)
