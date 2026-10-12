from backend.environment.city import manhattan


def nearest_cab_policy(env):
    """Central dispatcher: oldest request gets the closest available cab."""
    actions = {c.cab_id: env.ACTION_WAIT for c in env.cabs}
    idle = [c for c in env.cabs if c.status == "idle" and env.is_shift_active]
    for request in sorted((r for r in env.requests if r.status == "open"), key=lambda r: r.created_at):
        eligible = [c for c in idle if manhattan(c.position, request.origin) <= env.max_pickup_radius]
        if eligible:
            cab = min(eligible, key=lambda c: (manhattan(c.position, request.origin), c.cab_id))
            actions[cab.cab_id] = (env.ACTION_ACCEPT, request.request_id)
            idle.remove(cab)
    return actions
