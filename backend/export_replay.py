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
from .baselines.nearest_cab import nearest_cab_policy
from .baselines.zone_balancing import zone_balancing_policy
from .environment.simulator import CabDispatchEnv

SEED = 810000
STATUSES = {'idle': 0, 'en_route': 1, 'occupied': 2}

def export(method, fleet, agent=None):
    env = CabDispatchEnv(seed=SEED, fleet_size=fleet)
    frames = []
    while True:
        states = env.observations()
        qvalues = None
        if method == 'dqn':
            with torch.no_grad():
                qvalues = agent.policy(torch.tensor(list(states.values()), dtype=torch.float32)).tolist()
            actions = {c.cab_id: max(env.valid_actions(c), key=lambda a: qvalues[c.cab_id][a]) for c in env.cabs}
        else:
            actions = (nearest_cab_policy if method == 'nearest' else zone_balancing_policy)(env)
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
    return {'method': method, 'fleet': fleet, 'seed': SEED, 'schema': 1,
            'model_sha256': hashlib.sha256(Path('models/dqn_latest.pt').read_bytes()).hexdigest() if agent else None,
            'note': 'Recorded Python policy execution. No browser training. Synthetic demand. All methods use identical arrivals.',
            'frames': frames}

def main():
    torch.set_num_threads(1)
    checkpoint = torch.load('models/dqn_latest.pt', map_location='cpu', weights_only=True)
    if checkpoint.get('simulator_version') != 2:
        raise ValueError('Replay requires corrected simulator checkpoint')
    agent = SharedDQNAgent()
    agent.policy.load_state_dict(checkpoint['model'])
    agent.policy.eval()
    output = Path('frontend/data/traces')
    output.mkdir(parents=True, exist_ok=True)
    for fleet in (4, 8, 12, 16):
        for method in ('dqn', 'nearest', 'zone'):
            payload = export(method, fleet, agent if method == 'dqn' else None)
            path = output / f'{method}-{fleet}.json.gz'
            path.write_bytes(gzip.compress(json.dumps(payload, separators=(',', ':')).encode(), mtime=0))
            print(path, len(payload['frames']), 'frames', flush=True)

if __name__ == '__main__':
    main()
