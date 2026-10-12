import torch
from backend.environment.simulator import CabDispatchEnv
from backend.environment.demand import RideRequest
from backend.baselines.nearest_cab import nearest_cab_policy
from backend.agents.dqn import SharedDQNAgent
from backend.agents.vdn import VDNAgent
from backend.experiment import period_metrics


def fixture_env():
    env = CabDispatchEnv(fleet_size=1, day_minutes=20, shift_start=0, shift_end=1440)
    env.requests = []
    env.total_requests = 0
    env.demand.requests_for_minute = lambda minute: []
    return env


def test_pickup_and_dropoff_take_physical_travel_time():
    env = fixture_env()
    request = RideRequest(1, (0, 2), (0, 3), 0)
    env.requests = [request]
    env.total_requests = 1
    env.step({0: (env.ACTION_ACCEPT, 1)})
    assert env.cabs[0].status == 'en_route'
    assert env.cabs[0].position == (0, 1)
    assert env.completed == 0
    env.step({})
    assert env.cabs[0].status == 'occupied'
    assert request.pickup_time == 2
    env.step({})
    assert request.completed_at == 3
    assert env.completed == 1
    assert env.metrics()['avg_pickup_distance'] == 2
    assert env.metrics()['empty_driving_ratio'] == 2 / 3
    assert env.cabs[0].earnings == 13


def test_local_actions_do_not_reveal_distant_requests():
    env = fixture_env()
    env.requests = [RideRequest(1, (0, 3), (1, 3), 0)]
    assert env.ACTION_ACCEPT not in env.valid_actions(env.cabs[0])
    assert env.observations()[0][6] == 0


def test_observation_direction_slots_do_not_shift_at_edges():
    env = fixture_env()
    env.requests = [RideRequest(1, (1, 0), (1, 1), 0)]
    obs = env.observations()[0]
    assert obs[8] == 0  # north (outside grid)
    assert obs[10] == 0.2  # south
    assert obs[12] == 0  # west


def test_nearest_dispatcher_reserves_the_closest_cab_once():
    env = CabDispatchEnv(fleet_size=2, shift_start=0)
    env.cabs[0].position = (0, 0)
    env.cabs[1].position = (0, 2)
    env.requests = [RideRequest(1, (0, 2), (1, 2), 0)]
    actions = nearest_cab_policy(env)
    assert actions[0] == env.ACTION_WAIT
    assert actions[1] == (env.ACTION_ACCEPT, 1)


def test_pending_requests_do_not_inflate_service_or_cancellation_rates():
    env = fixture_env()
    env.requests = [RideRequest(1, (0, 0), (1, 0), 0, status='completed'),
                    RideRequest(2, (0, 0), (1, 0), 0)]
    env.total_requests = 2
    env.completed = 1
    assert env.metrics()['service_rate'] == 0.5
    assert env.metrics()['cancellation_rate'] == 0
    assert env.metrics()['pending'] == 1


def test_timeout_and_shift_are_enforced():
    env = fixture_env()
    env.requests = [RideRequest(1, (0, 0), (1, 0), 0)]
    env.minute = 8
    env.step({0: env.ACTION_ACCEPT})
    assert env.cancelled == 1
    env.shift_start = 360
    assert env.valid_actions(env.cabs[0]) == [env.ACTION_WAIT]


def constant_network(network, biases):
    for parameter in network.parameters():
        parameter.data.zero_()
    network.net[-1].bias.data.copy_(torch.tensor(biases, dtype=torch.float32))


def test_dqn_bootstrap_ignores_invalid_high_q_actions():
    agent = SharedDQNAgent(batch_size=1)
    constant_network(agent.policy, [0, 0, 0, 0, 0, 0])
    constant_network(agent.target, [1, 100, 100, 100, 100, 100])
    agent.remember([0]*16, 0, 0, [0]*16, False, [0])
    assert abs(agent.learn() - 0.5 * 0.95**2) < 1e-5


def test_vdn_bootstrap_ignores_invalid_high_q_actions():
    agent = VDNAgent(n_agents=1, batch_size=1)
    constant_network(agent.policy, [0, 0, 0, 0, 0, 0])
    constant_network(agent.target, [1, 100, 100, 100, 100, 100])
    agent.remember([[0]*16], [0], [0], [[0]*16], [[0]], False)
    assert abs(agent.learn() - 0.5 * 0.95**2) < 1e-5


def test_peak_and_offpeak_partition_every_request():
    env = fixture_env()
    env.requests = [RideRequest(i, (0,0), (1,0), minute, status='completed')
                    for i, minute in enumerate([0, 420, 600, 660, 960, 1020, 1260, 1439])]
    assert period_metrics(env, True)['service_rate'] == 1
    assert period_metrics(env, False)['service_rate'] == 1
