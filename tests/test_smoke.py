from backend.agents.dqn import SharedDQNAgent
from backend.environment.simulator import CabDispatchEnv


def test_environment_advances_and_returns_observations():
    env = CabDispatchEnv(seed=7, fleet_size=8, day_minutes=20)
    observations = env.reset()
    assert len(observations) == 8
    assert all(len(state) == 8 for state in observations.values())

    env.step({cab_id: env.ACTION_WAIT for cab_id in observations})
    assert env.minute == 1


def test_dqn_can_select_a_valid_action():
    env = CabDispatchEnv(seed=11, fleet_size=8, day_minutes=20)
    observations = env.reset()
    agent = SharedDQNAgent(seed=11)
    cab = env.cabs[0]
    action = agent.act(observations[cab.cab_id], [env.ACTION_WAIT, env.ACTION_ACCEPT], explore=False)
    assert action in {env.ACTION_WAIT, env.ACTION_ACCEPT}


def test_simulation_produces_research_metrics():
    env = CabDispatchEnv(seed=13, fleet_size=8, day_minutes=20)
    result = env.run(lambda _: {cab.cab_id: env.ACTION_WAIT for cab in env.cabs})
    assert 0.0 <= result["service_rate"] <= 1.0
    assert 0.0 <= result["utilisation"] <= 1.0
    assert result["earnings"] >= 0.0
