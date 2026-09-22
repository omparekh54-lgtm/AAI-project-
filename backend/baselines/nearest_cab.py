from __future__ import annotations
from backend.environment.city import manhattan

def nearest_cab_policy(env):
    actions = {}
    open_requests = [r for r in env.requests if r.status == "open"]
    for cab in env.cabs:
        if cab.status != "idle":
            continue
        if not open_requests:
            actions[cab.cab_id] = env.ACTION_WAIT
            continue
        nearest = min(open_requests, key=lambda r: manhattan(cab.position, r.origin))
        distance = manhattan(cab.position, nearest.origin)
        actions[cab.cab_id] = env.ACTION_ACCEPT if distance <= 3 else env.ACTION_WAIT
    return actions
