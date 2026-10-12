from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from backend.environment.city import GRID_SIZE, manhattan
from backend.environment.demand import DemandGenerator

ZONES = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
DISTANCES = np.array([[manhattan(a, b) for b in ZONES] for a in ZONES])


class DemandForecast:
    """Smoothed hourly origin rates, fitted only to training demand days."""
    def __init__(self, rates, training_seeds):
        self.rates = np.asarray(rates, dtype=float)
        self.training_seeds = list(training_seeds)
        if self.rates.shape != (24, GRID_SIZE * GRID_SIZE):
            raise ValueError('Expected 24 hourly rates over 25 zones')

    @classmethod
    def fit(cls, seeds=range(42, 172)):
        seeds = list(seeds)
        counts = np.zeros((24, len(ZONES)))
        for seed in seeds:
            generator = DemandGenerator(seed + 100)
            for minute in range(1440):
                for request in generator.requests_for_minute(minute):
                    counts[minute // 60, request.origin[0] * GRID_SIZE + request.origin[1]] += 1
        # Blend hourly spatial counts with the pooled spatial prior to reduce variance.
        pooled = (counts.sum(axis=0) + 1) / (counts.sum() + len(ZONES))
        hourly_total = counts.sum(axis=1, keepdims=True)
        spatial = (counts + 120 * pooled) / (hourly_total + 120)
        return cls(spatial * hourly_total / (len(seeds) * 60), seeds)

    def weights(self, minute, horizon=15):
        return np.mean([self.rates[((minute + offset) % 1440) // 60] for offset in range(horizon)], axis=0)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({'rates': self.rates.tolist(), 'training_seeds': self.training_seeds}, indent=2))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(data['rates'], data['training_seeds'])


def coordinated_matching(env):
    """Maximise feasible matches, then minimise pickup distance.

    Dummy columns allow waiting. A cardinality bonus larger than the total
    possible distance makes serving one more rider the primary objective.
    """
    actions = {c.cab_id: env.ACTION_WAIT for c in env.cabs}
    idle = [c for c in env.cabs if c.status == 'idle' and env.is_shift_active]
    # Rotate deterministic ties, rather than always preferring the first cab.
    idle.sort(key=lambda c: (c.cab_id - env.minute) % env.fleet_size)
    requests = [r for r in env.requests if r.status == 'open' and env.minute - r.created_at < env.cancellation_timeout]
    if not idle or not requests:
        return actions
    bonus = (len(idle) + 1) * (env.max_pickup_radius + env.cancellation_timeout + 1)
    costs = np.zeros((len(idle), len(requests) + len(idle)))
    costs[:, :len(requests)] = bonus
    for i, cab in enumerate(idle):
        for j, request in enumerate(requests):
            distance = manhattan(cab.position, request.origin)
            if distance <= env.max_pickup_radius:
                costs[i, j] = -bonus + distance - 0.1 * (env.minute - request.created_at)
    rows, columns = linear_sum_assignment(costs)
    for i, j in zip(rows, columns):
        if j < len(requests) and costs[i, j] < 0:
            actions[idle[i].cab_id] = (env.ACTION_ACCEPT, requests[j].request_id)
    return actions


class PredictiveDispatchPolicy:
    """Central matching and demand-aware, coordinated cab repositioning.

    This is a forecast/optimisation extension, not a renamed DQN. All methods
    share the same travel, timeout and fleet constraints. No future requests
    from the evaluated day are visible to the policy.
    """
    def __init__(self, forecast, movement_cost=0.05, horizon=15):
        self.forecast = forecast
        self.movement_cost = movement_cost
        self.horizon = horizon

    def coverage_floor(self, env, actions):
        return np.full(len(ZONES), 8.0)

    def demand_weights(self, env, actions):
        return self.forecast.weights(env.minute, self.horizon)

    def __call__(self, env):
        actions = coordinated_matching(env)
        if not env.is_shift_active:
            return actions
        weights = self.demand_weights(env, actions)
        incoming = self.coverage_floor(env, actions)
        idle = [c for c in env.cabs if c.status == 'idle' and actions[c.cab_id] == env.ACTION_WAIT]
        if not idle:
            return actions
        planned = {c.cab_id: c.position for c in idle}
        # Reservations update sequentially so cabs do not all chase the same gap.
        idle.sort(key=lambda c: (c.cab_id - env.minute) % env.fleet_size)
        for cab in idle:
            others = [p[0] * GRID_SIZE + p[1] for i, p in planned.items() if i != cab.cab_id]
            distances = DISTANCES[:, others].min(axis=1) if others else np.full(len(ZONES), 8)
            distances = np.minimum(distances, incoming)
            x, y = cab.position
            candidates = [(env.ACTION_WAIT, (x, y))]
            for action, position in [(env.ACTION_NORTH, (x-1,y)), (env.ACTION_SOUTH, (x+1,y)),
                                     (env.ACTION_WEST, (x,y-1)), (env.ACTION_EAST, (x,y+1))]:
                if 0 <= position[0] < GRID_SIZE and 0 <= position[1] < GRID_SIZE:
                    candidates.append((action, position))
            def cost(candidate):
                action, position = candidate
                nearest = np.minimum(distances, DISTANCES[:, position[0] * GRID_SIZE + position[1]])
                # Outside pickup coverage carries a further unserved-request cost.
                coverage_cost = nearest + 2 * np.maximum(0, nearest - env.max_pickup_radius)
                return float(weights @ coverage_cost) + (self.movement_cost if action != env.ACTION_WAIT else 0)
            action, position = min(candidates, key=lambda candidate: (cost(candidate), candidate[0]))
            actions[cab.cab_id] = action
            planned[cab.cab_id] = position
        return actions


class IncomingAwareDispatchPolicy(PredictiveDispatchPolicy):
    """Account for cabs about to finish a known trip and current unmatched riders.

    Only accepted trip destinations and current requests are used. Future
    arrivals still come from the training forecast, never the evaluated day.
    """
    def __init__(self, forecast, movement_cost=0.05, horizon=15, availability_cost=0.3, queue_weight=0.1):
        super().__init__(forecast, movement_cost, horizon)
        self.availability_cost = availability_cost
        self.queue_weight = queue_weight

    def coverage_floor(self, env, actions):
        incoming = np.full(len(ZONES), 8.0)
        rides = {r.request_id: r for r in env.requests}
        for cab in env.cabs:
            action = actions.get(cab.cab_id, env.ACTION_WAIT)
            if isinstance(action, tuple):
                ride = rides[action[1]]
                eta = manhattan(cab.position, ride.origin) + ride.distance
                destination = ride.destination
            elif cab.status != 'idle':
                ride = rides[cab.ride_id]
                eta = cab.remaining + (ride.distance if cab.status == 'en_route' else 0)
                destination = ride.destination
            else:
                continue
            if eta < self.horizon:
                index = destination[0] * GRID_SIZE + destination[1]
                incoming = np.minimum(incoming, DISTANCES[:, index] + eta * self.availability_cost)
        return incoming

    def demand_weights(self, env, actions):
        weights = super().demand_weights(env, actions).copy()
        assigned_ids = {action[1] for action in actions.values() if isinstance(action, tuple)}
        for request in env.requests:
            age = env.minute - request.created_at
            if request.status == 'open' and request.request_id not in assigned_ids and age < env.cancellation_timeout:
                urgency = 1 + age / max(1, env.cancellation_timeout)
                weights[request.origin[0] * GRID_SIZE + request.origin[1]] += self.queue_weight * urgency
        return weights


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-start', type=int, default=42)
    parser.add_argument('--days', type=int, default=130)
    parser.add_argument('--output', default='models/demand_forecast.json')
    args = parser.parse_args()
    DemandForecast.fit(range(args.seed_start, args.seed_start + args.days)).save(args.output)
