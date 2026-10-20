"""Check replay provenance, simulator agreement and the deployable data contract."""
import gzip
import json
from pathlib import Path
import torch
from backend.agents.policies import dqn_policy, HybridMatchingPolicy
from backend.environment.simulator import CabDispatchEnv
from backend.baselines.nearest_cab import nearest_cab_policy
from backend.baselines.zone_balancing import zone_balancing_policy
from backend.export_replay import load_agent, CHECKPOINTS

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('dqn', 'hybrid', 'nearest', 'zone')


def test_static_bundle_is_inside_render_publish_directory():
    research = json.loads((ROOT / 'frontend/data/experiment_results.json').read_text())
    original = json.loads((ROOT / 'results/experiment_results.json').read_text())
    assert research['results'] == original['results']
    assert research['metadata'] == original['metadata']
    assert json.loads((ROOT / 'frontend/data/training_curve.json').read_text())['curves']
    for fleet in (4, 8, 12, 16):
        for method in METHODS:
            data = json.loads(gzip.decompress((ROOT / f'frontend/data/traces/{method}-{fleet}.json.gz').read_bytes()))
            assert data['seed'] == 810000
            assert len(data['frames']) == 1441
            assert len(data['frames'][420]['c']) == fleet
            assert data['frames'][-1]['m']['minutes'] == 1440
            assert data['frames'][-1]['m']['completed'] > 0


def test_replay_decisions_match_python_policy_and_motion():
    torch.set_num_threads(1)
    arrival_counts = []
    for method in METHODS:
        data = json.loads(gzip.decompress((ROOT / f'frontend/data/traces/{method}-8.json.gz').read_bytes()))
        learned = method in CHECKPOINTS
        env = CabDispatchEnv(seed=data['seed'], fleet_size=8, dispatch_reach=learned)
        if method == 'dqn':
            agent = load_agent(ROOT / CHECKPOINTS[method])
            policy = dqn_policy(agent)
        elif method == 'hybrid':
            agent = load_agent(ROOT / CHECKPOINTS[method])
            policy = HybridMatchingPolicy(agent)
        else:
            policy = nearest_cab_policy if method == 'nearest' else zone_balancing_policy
        counts = []
        # Check every minute through the first active hour, including off-shift
        # cancellations, first dispatches, pickups and completed rides.
        for frame in data['frames'][:421]:
            assert frame['t'] == env.minute
            assert frame['m'] == env.metrics()
            states = env.observations()
            actions = policy(env)
            for recorded, cab in zip(frame['c'], env.cabs):
                assert recorded[1:3] == list(cab.position)
                action = actions[cab.cab_id]
                assert recorded[4] == (action[0] if isinstance(action, tuple) else action)
                if learned:
                    with torch.no_grad():
                        values = agent.policy(torch.tensor(states[cab.cab_id], dtype=torch.float32)).tolist()
                    assert all(abs(a - b) < .001 for a, b in zip(frame['q'][cab.cab_id], values))
            counts.append(frame['m']['requests'])
            env.step(actions)
        arrival_counts.append(counts)
    assert all(c == arrival_counts[0] for c in arrival_counts)
