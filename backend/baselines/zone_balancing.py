from __future__ import annotations
from collections import Counter
from backend.environment.city import neighbours, manhattan

def zone_balancing_policy(env):
    actions = {}
    demand = Counter(r.origin for r in env.requests if r.status == "open")
    for cab in env.cabs:
        if cab.status != "idle":
            continue
        requests_here = demand.get(cab.position, 0)
        if requests_here:
            actions[cab.cab_id] = env.ACTION_ACCEPT
            continue
        candidates = list(demand.items())
        if not candidates:
            actions[cab.cab_id] = env.ACTION_WAIT
            continue
        target, _ = max(candidates, key=lambda item: (item[1], -manhattan(cab.position, item[0])))
        if manhattan(cab.position, target) <= 1:
            actions[cab.cab_id] = env.ACTION_ACCEPT
            continue
        moves = {env.ACTION_NORTH:(-1,0), env.ACTION_SOUTH:(1,0), env.ACTION_WEST:(0,-1), env.ACTION_EAST:(0,1)}
        valid = []
        for action,(dx,dy) in moves.items():
            p = (cab.position[0]+dx, cab.position[1]+dy)
            if 0 <= p[0] < 5 and 0 <= p[1] < 5:
                valid.append((manhattan(p,target), action))
        actions[cab.cab_id] = min(valid)[1] if valid else env.ACTION_WAIT
    return actions
