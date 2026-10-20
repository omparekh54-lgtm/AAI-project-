"""Tests for the fair-reach learning agents, semi-MDP credit and the hybrid policy."""
import torch
from backend.agents.dqn import SharedDQNAgent
from backend.agents.policies import HybridMatchingPolicy, reposition_actions
from backend.environment.demand import RideRequest
from backend.environment.simulator import CabDispatchEnv
from backend.train import STATE_SIZE, train


def reach_env(fleet=1):
    env = CabDispatchEnv(fleet_size=fleet, day_minutes=30, shift_start=0, shift_end=1440, dispatch_reach=True)
    env.requests = []
    env.total_requests = 0
    env.demand.requests_for_minute = lambda minute: []
    return env


def test_learning_agents_can_accept_within_the_pickup_radius_but_not_beyond():
    env = reach_env()
    env.requests = [RideRequest(1, (0, 3), (1, 3), 0)]
    assert env.ACTION_ACCEPT in env.valid_actions(env.cabs[0])        # distance 3 = max pickup radius
    env.requests = [RideRequest(1, (3, 3), (1, 3), 0)]                  # distance 6
    assert env.ACTION_ACCEPT not in env.valid_actions(env.cabs[0])


def test_default_environment_keeps_the_original_radius_one_sensing():
    env = CabDispatchEnv(fleet_size=1, shift_start=0)
    env.requests = [RideRequest(1, (0, 3), (1, 3), 0)]
    assert env.ACTION_ACCEPT not in env.valid_actions(env.cabs[0])
    assert len(env.observations()[0]) == 16


def test_reach_observation_has_expected_size_and_direction_features():
    env = reach_env()
    env.requests = [RideRequest(1, (0, 2), (1, 2), 0)]
    state = env.observations()[0]
    assert len(state) == STATE_SIZE == 22
    assert state[16] > 0                      # open-request count in reach
    assert abs(state[17] - (1 - 2 / 4)) < 1e-9  # nearest request at distance 2
    assert state[21] > 0 and state[18] == 0    # request lies east, none north


def constant_network(network, biases):
    for parameter in network.parameters():
        parameter.data.zero_()
    network.net[-1].bias.data.copy_(torch.tensor(biases, dtype=torch.float32))


def test_multi_minute_transition_uses_its_own_discount():
    agent = SharedDQNAgent(state_size=4, batch_size=1)
    constant_network(agent.policy, [0] * 6)
    constant_network(agent.target, [1, 100, 100, 100, 100, 100])
    agent.remember([0] * 4, 0, 0.0, [0] * 4, False, [0], 0.5)   # gamma**k = 0.5
    assert abs(agent.learn() - 0.5 * 0.5 ** 2) < 1e-6


def test_hybrid_policy_assigns_riders_and_never_accepts_for_the_dqn():
    env = reach_env(fleet=2)
    env.cabs[0].position = (0, 0)
    env.cabs[1].position = (4, 4)
    env.requests = [RideRequest(1, (0, 1), (1, 1), 0)]
    agent = SharedDQNAgent(state_size=STATE_SIZE)
    actions = HybridMatchingPolicy(agent)(env)
    assert actions[0] == (env.ACTION_ACCEPT, 1)             # matched by the dispatcher
    assert actions[1] != env.ACTION_ACCEPT and not isinstance(actions[1], tuple)
    assert env.ACTION_ACCEPT not in reposition_actions(env, env.cabs[1])


def test_short_training_runs_for_both_modes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for mode in ('full', 'hybrid'):
        path = train(episodes=1, day_minutes=400, mode=mode, fleets=(4,), out_dir=str(tmp_path), quiet=True)
        checkpoint = torch.load(path, weights_only=True)
        assert checkpoint['simulator_version'] == 3 and checkpoint['state_size'] == STATE_SIZE
