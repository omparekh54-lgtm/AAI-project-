import numpy as np
from backend.coordination.predictive import DemandForecast, PredictiveDispatchPolicy, coordinated_matching
from backend.environment.simulator import CabDispatchEnv
from backend.environment.demand import RideRequest
from backend.baselines.nearest_cab import nearest_cab_policy


def test_joint_matching_avoids_greedy_feasibility_trap():
    env = CabDispatchEnv(fleet_size=2, shift_start=0)
    env.cabs[0].position = (0, 0)
    env.cabs[1].position = (0, 3)
    # Greedy gives cab 0 the first request, then cannot serve the second.
    env.requests = [RideRequest(1, (0,1), (1,1), 0), RideRequest(2, (2,0), (3,0), 1)]
    env.minute = 1
    greedy = nearest_cab_policy(env)
    joint = coordinated_matching(env)
    assert sum(isinstance(a, tuple) for a in greedy.values()) == 1
    assert sum(isinstance(a, tuple) for a in joint.values()) == 2
    assert joint[0] == (env.ACTION_ACCEPT, 2)
    assert joint[1] == (env.ACTION_ACCEPT, 1)


def test_forecast_serialisation_preserves_training_provenance(tmp_path):
    forecast = DemandForecast.fit([1, 2])
    path = tmp_path / 'forecast.json'
    forecast.save(path)
    loaded = DemandForecast.load(path)
    assert loaded.training_seeds == [1, 2]
    assert np.array_equal(forecast.rates, loaded.rates)
    assert (loaded.weights(1439) >= 0).all()


def test_controller_does_not_advance_or_read_future_demand():
    forecast = DemandForecast(np.full((24,25), 0.02), [1])
    policy = PredictiveDispatchPolicy(forecast)
    env = CabDispatchEnv(seed=15, shift_start=0)
    before = [(r.request_id, r.status) for r in env.requests]
    def forbidden(minute):
        raise AssertionError('Policy must not request future demand')
    env.demand.requests_for_minute = forbidden
    actions = policy(env)
    assert env.minute == 0
    assert before == [(r.request_id, r.status) for r in env.requests]
    assigned = [a[1] for a in actions.values() if isinstance(a, tuple)]
    assert len(assigned) == len(set(assigned))


def test_predictive_policy_respects_shift_and_grid():
    forecast = DemandForecast(np.full((24,25), 0.02), [1])
    policy = PredictiveDispatchPolicy(forecast)
    env = CabDispatchEnv()
    assert all(action == env.ACTION_WAIT for action in policy(env).values())
    env.minute = 360
    for cab in env.cabs:
        action = policy(env)[cab.cab_id]
        if not isinstance(action, tuple):
            assert action in env.valid_actions(cab)


def test_exploration_is_not_decayed_by_overnight_forced_wait_training(tmp_path, monkeypatch):
    import json
    from backend.train import train
    monkeypatch.chdir(tmp_path)
    train(episodes=1, day_minutes=300)
    curve = json.loads((tmp_path / 'results/training_curve.json').read_text())
    assert curve['losses'] == [0.0]
    assert curve['epsilons'] == [1.0]


def test_incoming_coverage_uses_known_dropoff_and_remaining_time():
    from backend.coordination.predictive import IncomingAwareDispatchPolicy
    forecast = DemandForecast(np.full((24,25), 0.02), [1])
    policy = IncomingAwareDispatchPolicy(forecast, availability_cost=0.5)
    env = CabDispatchEnv(fleet_size=1, shift_start=0)
    ride = RideRequest(1, (0,0), (0,4), 0, status='assigned', assigned_cab=0)
    env.requests = [ride]
    cab = env.cabs[0]
    cab.status, cab.ride_id, cab.destination, cab.remaining = 'occupied', 1, (0,4), 2
    floor = policy.coverage_floor(env, {0:env.ACTION_WAIT})
    assert floor[4] == 1  # two remaining minutes, availability cost 0.5
    assert floor[0] == 5  # four zones farther from known drop-off


def test_queue_priority_does_not_double_count_matched_requests():
    from backend.coordination.predictive import IncomingAwareDispatchPolicy
    forecast = DemandForecast(np.full((24,25), 0.02), [1])
    policy = IncomingAwareDispatchPolicy(forecast)
    env = CabDispatchEnv(fleet_size=1, shift_start=0)
    env.requests = [RideRequest(1, (0,0), (1,0), 0)]
    env.minute = 2
    unmatched = policy.demand_weights(env, {0:env.ACTION_WAIT})
    matched = policy.demand_weights(env, {0:(env.ACTION_ACCEPT,1)})
    assert matched[0] == forecast.weights(env.minute)[0]
    assert unmatched[0] > matched[0]


def test_incoming_controller_does_not_query_future_demand():
    from backend.coordination.predictive import IncomingAwareDispatchPolicy
    forecast = DemandForecast(np.full((24,25), 0.02), [1])
    policy = IncomingAwareDispatchPolicy(forecast)
    env = CabDispatchEnv(seed=15, shift_start=0)
    def forbidden(minute):
        raise AssertionError('Future demand must not be queried')
    env.demand.requests_for_minute = forbidden
    before = [(c.position,c.status) for c in env.cabs]
    policy(env)
    assert before == [(c.position,c.status) for c in env.cabs]
    assert env.minute == 0
