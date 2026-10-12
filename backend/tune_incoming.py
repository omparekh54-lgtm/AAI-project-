"""Select the incoming-aware controller on a new validation split only."""
import json
from pathlib import Path
from statistics import mean
from .coordination.predictive import DemandForecast, IncomingAwareDispatchPolicy, PredictiveDispatchPolicy
from .environment.simulator import CabDispatchEnv

SEEDS = list(range(500000, 500005))

def main():
    forecast = DemandForecast.load('models/demand_forecast.json')
    choices = []
    for availability_cost in (0.15, 0.3, 0.5):
        for movement_cost in (0.02, 0.05, 0.08):
            policy = IncomingAwareDispatchPolicy(forecast, movement_cost=movement_cost, availability_cost=availability_cost)
            rows = [CabDispatchEnv(seed=seed).run(policy) for seed in SEEDS]
            metrics = {key: mean(row[key] for row in rows) for key in rows[0]}
            score = metrics['avg_wait'] + metrics['empty_distance'] / 2000
            choices.append({'availability_cost': availability_cost, 'movement_cost': movement_cost,
                            'queue_weight': 0.1, 'score': score, 'metrics': metrics})
            print(availability_cost,movement_cost, {key:round(metrics[key],4) for key in ('service_rate','avg_wait','empty_distance')},flush=True)
    old_rows = [CabDispatchEnv(seed=seed).run(PredictiveDispatchPolicy(forecast)) for seed in SEEDS]
    old = {key:mean(row[key] for row in old_rows) for key in old_rows[0]}
    # Require no validation service loss before selecting the wait/driving trade-off.
    feasible = [choice for choice in choices if choice['metrics']['service_rate'] >= old['service_rate']]
    best = min(feasible or choices, key=lambda choice: choice['score'])
    result = {'validation_seeds':SEEDS, 'objective':'avg_wait + empty_distance / 2000; prefer no service loss',
              'previous_controller':old, 'candidates':choices, 'selected':best}
    Path('results/incoming_validation.json').write_text(json.dumps(result,indent=2))
    Path('models/incoming_config.json').write_text(json.dumps({key:best[key] for key in ('availability_cost','movement_cost','queue_weight')},indent=2))
    print('selected', best, flush=True)

if __name__=='__main__':
    main()
