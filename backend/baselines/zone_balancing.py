from backend.environment.city import AIRPORT_ZONE, GRID_SIZE, manhattan
from .nearest_cab import nearest_cab_policy


def zone_balancing_policy(env):
    """Greedy matching plus static historical-prior fleet coverage.

    Synthetic prior matches the generator's airport probability. No future or
    held-out requests are used to construct the target zones.
    """
    actions = nearest_cab_policy(env)
    targets = [AIRPORT_ZONE] * max(1, round(env.fleet_size * env.demand.airport_hotspot))
    zones = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if (x, y) != AIRPORT_ZONE]
    remaining = env.fleet_size - len(targets)
    targets += [zones[(i * len(zones)) // max(1, remaining)] for i in range(remaining)]
    for cab in env.cabs:
        if cab.status != 'idle' or not env.is_shift_active or actions[cab.cab_id] != env.ACTION_WAIT:
            continue
        target = targets[cab.cab_id]
        x, y = cab.position
        tx, ty = target
        if x != tx:
            actions[cab.cab_id] = env.ACTION_SOUTH if tx > x else env.ACTION_NORTH
        elif y != ty:
            actions[cab.cab_id] = env.ACTION_EAST if ty > y else env.ACTION_WEST
    return actions
