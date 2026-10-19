"""Check replay provenance, simulator agreement and the deployable data contract."""
import gzip
import json
from pathlib import Path
import torch
from backend.agents.dqn import SharedDQNAgent
from backend.environment.simulator import CabDispatchEnv
from backend.baselines.nearest_cab import nearest_cab_policy
from backend.baselines.zone_balancing import zone_balancing_policy

ROOT = Path(__file__).resolve().parents[1]

def test_static_bundle_is_inside_render_publish_directory():
    research = json.loads((ROOT / 'frontend/data/experiment_results.json').read_text())
    original = json.loads((ROOT / 'results/experiment_results.json').read_text())
    assert research['results'] == original['results']
    assert research['metadata'] == original['metadata']
    assert json.loads((ROOT / 'frontend/data/training_curve.json').read_text())['rewards']
    for fleet in (4, 8, 12, 16):
        for method in ('dqn', 'nearest', 'zone'):
            data = json.loads(gzip.decompress((ROOT / f'frontend/data/traces/{method}-{fleet}.json.gz').read_bytes()))
            assert data['seed'] == 810000
            assert len(data['frames']) == 1441
            assert len(data['frames'][420]['c']) == fleet
            assert data['frames'][-1]['m']['minutes'] == 1440
            assert data['frames'][-1]['m']['completed'] > 0

def test_replay_decisions_match_python_policy_and_motion():
    torch.set_num_threads(1)
    agent = SharedDQNAgent()
    checkpoint = torch.load(ROOT / 'models/dqn_latest.pt', map_location='cpu', weights_only=True)
    agent.policy.load_state_dict(checkpoint['model'])
    agent.policy.eval()
    arrival_counts = []
    for method in ('dqn', 'nearest', 'zone'):
        data = json.loads(gzip.decompress((ROOT / f'frontend/data/traces/{method}-8.json.gz').read_bytes()))
        env = CabDispatchEnv(seed=data['seed'], fleet_size=8)
        counts = []
        # Check every minute through the first active hour, including off-shift
        # cancellations, first dispatches, pickups and completed rides.
        for frame in data['frames'][:421]:
            assert frame['t'] == env.minute
            assert frame['m'] == env.metrics()
            states = env.observations()
            if method == 'dqn':
                actions = {c.cab_id: agent.act(states[c.cab_id], env.valid_actions(c), explore=False) for c in env.cabs}
            else:
                actions = (nearest_cab_policy if method == 'nearest' else zone_balancing_policy)(env)
            for recorded, cab in zip(frame['c'], env.cabs):
                assert recorded[1:3] == list(cab.position)
                action = actions[cab.cab_id]
                assert recorded[4] == (action[0] if isinstance(action, tuple) else action)
                if method == 'dqn':
                    with torch.no_grad():
                        values = agent.policy(torch.tensor(states[cab.cab_id], dtype=torch.float32)).tolist()
                    assert all(abs(a-b) < .001 for a,b in zip(frame['q'][cab.cab_id], values))
            counts.append(frame['m']['requests'])
            env.step(actions)
        arrival_counts.append(counts)
    assert arrival_counts[0] == arrival_counts[1] == arrival_counts[2]
