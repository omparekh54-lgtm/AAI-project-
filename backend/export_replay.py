"""Export actual simulator trajectories, with trained Q-values, for static playback.

No random browser animation or invented decisions. All policies share seed 810000.
Run from repository root: python -m backend.export_replay
"""
import gzip
import hashlib
import json
from pathlib import Path
import torch
from .agents.dqn import SharedDQNAgent
from .agents.policies import dqn_policy, HybridMatchingPolicy
from .baselines.nearest_cab import nearest_cab_policy
from .baselines.zone_balancing import zone_balancing_policy
from .environment.simulator import CabDispatchEnv
from .train import STATE_SIZE

SEED = 810000
STATUSES = {'idle': 0, 'en_route': 1, 'occupied': 2}
CHECKPOINTS = {'dqn': 'models/dqn_full_s42.pt', 'hybrid': 'models/dqn_hybrid_s42.pt'}


def load_agent(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('simulator_version') != 3:
        raise ValueError('Replay requires a checkpoint from the current simulator')
    agent = SharedDQNAgent(state_size=STATE_SIZE)
    agent.policy.load_state_dict(checkpoint['model'])
    agent.policy.eval()
    return agent


def export(method, fleet, agent=None):
    learned = method in CHECKPOINTS
    env = CabDispatchEnv(seed=SEED, fleet_size=fleet, dispatch_reach=learned)
    if method == 'dqn':
        policy = dqn_policy(agent)
    elif method == 'hybrid':
        policy = HybridMatchingPolicy(agent)
    else:
        policy = nearest_cab_policy if method == 'nearest' else zone_balancing_policy
    frames = []
    while True:
        states = env.observations()
        qvalues = None
        if learned:
            with torch.no_grad():
                qvalues = agent.policy(torch.tensor([states[c.cab_id] for c in env.cabs], dtype=torch.float32)).tolist()
        actions = policy(env)
        # Every frame is the pre-action state. Its decisions take effect on next step.
        requests = [r for r in env.requests if r.status in ('open', 'assigned')]
        frame = {
            't': env.minute,
            'c': [[c.cab_id, *c.position, STATUSES[c.status],
                   (actions[c.cab_id][0] if isinstance(actions[c.cab_id], tuple) else actions[c.cab_id]),
                   round(env.last_rewards[c.cab_id], 3), c.earnings, c.ride_id, c.remaining] for c in env.cabs],
            'r': [[r.request_id, *r.origin, *r.destination, r.created_at, r.assigned_cab, r.pickup_time] for r in requests],
            'm': env.metrics(),
        }
        if qvalues is not None:
            frame['q'] = [[round(v, 4) for v in row] for row in qvalues]
        frames.append(frame)
        if env.minute == env.day_minutes:
            break
        env.step(actions)
    return {'method': method, 'fleet': fleet, 'seed': SEED, 'schema': 2,
            'model_sha256': hashlib.sha256(Path(CHECKPOINTS[method]).read_bytes()).hexdigest() if learned else None,
            'sensing_radius': env.max_pickup_radius if learned else 1,
            'note': 'Recorded Python policy execution. No browser training. Synthetic demand. All methods use identical arrivals.',
            'frames': frames}


def main():
    torch.set_num_threads(1)
    agents = {m: load_agent(p) for m, p in CHECKPOINTS.items()}
    output = Path('frontend/data/traces')
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob('*.json.gz'):
        old.unlink()
    for fleet in (4, 8, 12, 16):
        for method in ('dqn', 'hybrid', 'nearest', 'zone'):
            payload = export(method, fleet, agents.get(method))
            path = output / f'{method}-{fleet}.json.gz'
            path.write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))
            print(path, len(payload['frames']), 'frames', flush=True)


if __name__ == '__main__':
    main()
