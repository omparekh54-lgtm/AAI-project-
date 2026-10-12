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
