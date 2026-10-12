"""Reproduce movement-cost selection using validation demand, never test demand."""
import json
from pathlib import Path
from statistics import mean
from .coordination.predictive import DemandForecast, PredictiveDispatchPolicy
from .environment.simulator import CabDispatchEnv


def main():
    forecast = DemandForecast.load('models/demand_forecast.json')
    seeds = list(range(200000, 200005))
    candidates = []
    for movement_cost in [0.02, 0.03, 0.05, 0.08]:
        policy = PredictiveDispatchPolicy(forecast, movement_cost)
        rows = [CabDispatchEnv(seed=seed).run(policy) for seed in seeds]
        metrics = {key: mean(row[key] for row in rows) for key in rows[0]}
        # Explicit trade-off for this synthetic pilot, not a real fuel price:
        # one average waiting minute is traded against 2,000 daily empty zones.
        score = metrics['avg_wait'] + metrics['empty_distance'] / 2000
        candidates.append({'movement_cost': movement_cost, 'score': score, 'metrics': metrics})
    payload = {'seeds': seeds, 'objective': 'avg_wait + empty_distance / 2000',
               'selected_movement_cost': min(candidates, key=lambda row: row['score'])['movement_cost'],
               'candidates': candidates}
    Path('results/coordination_validation.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
