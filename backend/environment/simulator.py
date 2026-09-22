from __future__ import annotations
from dataclasses import dataclass
import random
from .city import GRID_SIZE, manhattan, neighbours
from .cab import Cab
from .demand import DemandGenerator, RideRequest

@dataclass
class StepStats:
    minute: int
    open_requests: int
    completed: int
    cancelled: int
    avg_wait: float
    shared_wait_cost: float
    unserved_cost: float


class CabDispatchEnv:
    """5x5, 8-cab decentralised dispatch simulator.

    Local observations, neighbour demand/supply information, maximum pickup radius,
    cancellation timeout, fixed fleet and configurable driver shifts are modelled.
    A default episode is a complete 24-hour simulated day.
    """
    ACTION_WAIT = 0
    ACTION_ACCEPT = 1
    ACTION_NORTH = 2
    ACTION_SOUTH = 3
    ACTION_WEST = 4
    ACTION_EAST = 5
    ACTION_NAMES = {0: "WAIT", 1: "ACCEPT", 2: "NORTH", 3: "SOUTH", 4: "WEST", 5: "EAST"}

    def __init__(self, seed=42, fleet_size=8, day_minutes=1440, max_pickup_radius=3,
                 cancellation_timeout=8, shift_start=6 * 60, shift_end=22 * 60,
                 shared_reward_weight=0.25):
        self.seed = seed
        self.rng = random.Random(seed)
        self.fleet_size = fleet_size
        self.day_minutes = day_minutes
        self.max_pickup_radius = max_pickup_radius
        self.cancellation_timeout = cancellation_timeout
        self.shift_start = shift_start
        self.shift_end = shift_end
        self.shared_reward_weight = shared_reward_weight
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)
        self.minute = 0
        self.cabs = [Cab(i, (i // GRID_SIZE, i % GRID_SIZE)) for i in range(self.fleet_size)]
        self.demand = DemandGenerator(self.seed + 100)
        self.requests = []
        self.completed = 0
        self.cancelled = 0
        self.total_wait = 0
        self.total_fare = 0.0
        self.total_empty_distance = 0
        self.total_pickup_distance = 0
        self.total_requests = 0
        self.history = []
        self.last_rewards = {i: 0.0 for i in range(self.fleet_size)}
        self.last_actions = {i: self.ACTION_WAIT for i in range(self.fleet_size)}
        return self.observations()

    @property
    def is_shift_active(self):
        hour_minute = self.minute % 1440
        return self.shift_start <= hour_minute < self.shift_end

    def valid_actions(self, cab):
        actions = [self.ACTION_WAIT]
        if cab.status == "idle" and self.is_shift_active:
            if self.open_requests_for(cab):
                actions.append(self.ACTION_ACCEPT)
            x, y = cab.position
            if x > 0: actions.append(self.ACTION_NORTH)
            if x < GRID_SIZE - 1: actions.append(self.ACTION_SOUTH)
            if y > 0: actions.append(self.ACTION_WEST)
            if y < GRID_SIZE - 1: actions.append(self.ACTION_EAST)
        return actions

    def open_requests_for(self, cab):
        return [r for r in self.requests if r.status == "open" and manhattan(cab.position, r.origin) <= self.max_pickup_radius]

    def _move(self, cab, action):
        if not self.is_shift_active:
            return False
        dxdy = {self.ACTION_NORTH: (-1, 0), self.ACTION_SOUTH: (1, 0),
                self.ACTION_WEST: (0, -1), self.ACTION_EAST: (0, 1)}
        if action not in dxdy:
            return False
        dx, dy = dxdy[action]
        nx, ny = cab.position[0] + dx, cab.position[1] + dy
        if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
            return False
        cab.position = (nx, ny)
        cab.empty_distance += 1
        self.total_empty_distance += 1
        return True

    def _assign(self, cab, request):
        pickup_distance = manhattan(cab.position, request.origin)
        if pickup_distance > self.max_pickup_radius or not self.is_shift_active:
            return False
        cab.empty_distance += pickup_distance
        self.total_empty_distance += pickup_distance
        self.total_pickup_distance += pickup_distance
        request.status = "assigned"
        request.assigned_cab = cab.cab_id
        request.pickup_time = self.minute + pickup_distance
        cab.status = "occupied"
        cab.ride_id = request.request_id
        cab.destination = request.destination
        cab.remaining = max(1, request.distance)
        return True

    def _complete_rides(self):
        by_id = {r.request_id: r for r in self.requests}
        completed_fares = {i: 0.0 for i in range(self.fleet_size)}
        for cab in self.cabs:
            if cab.status != "occupied":
                continue
            cab.tick_busy()
            cab.remaining -= 1
            if cab.remaining <= 0:
                ride = by_id.get(cab.ride_id)
                if ride:
                    ride.status = "completed"
                    ride.completed_at = self.minute
                    self.completed += 1
                    self.total_wait += ride.wait_minutes
                    fare = 10.0 + 3.0 * ride.distance
                    self.total_fare += fare
                    completed_fares[cab.cab_id] += fare
                cab.position = cab.destination or cab.position
                cab.status, cab.ride_id, cab.destination = "idle", None, None
        return completed_fares

    def _cancel_expired(self):
        cancelled_now = 0
        for r in self.requests:
            if r.status == "open" and self.minute - r.created_at >= self.cancellation_timeout:
                r.status = "cancelled"
                self.cancelled += 1
                cancelled_now += 1
        return cancelled_now

    def step(self, actions):
        self.requests.extend(self.demand.requests_for_minute(self.minute))
        self.total_requests = len(self.requests)
        before_empty = {c.cab_id: c.empty_distance for c in self.cabs}
        completed_fares = self._complete_rides()
        cancelled_now = self._cancel_expired()

        for cab in self.cabs:
            if cab.status != "idle" or not self.is_shift_active:
                continue
            action = actions.get(cab.cab_id, self.ACTION_WAIT)
            self.last_actions[cab.cab_id] = action
            if action == self.ACTION_ACCEPT:
                choices = self.open_requests_for(cab)
                if choices:
                    request = min(choices, key=lambda r: (manhattan(cab.position, r.origin), r.created_at))
                    self._assign(cab, request)
            elif action in (self.ACTION_NORTH, self.ACTION_SOUTH, self.ACTION_WEST, self.ACTION_EAST):
                self._move(cab, action)

        open_count = sum(r.status == "open" for r in self.requests)
        wait_cost = sum(max(0, self.minute - r.created_at) for r in self.requests if r.status == "open")
        unserved_cost = float(open_count + cancelled_now)
        avg_wait = self.total_wait / max(1, self.completed)

        for cab in self.cabs:
            own_empty = cab.empty_distance - before_empty[cab.cab_id]
            individual = completed_fares[cab.cab_id] - 0.20 * own_empty
            shared = -0.05 * wait_cost - 1.0 * unserved_cost
            self.last_rewards[cab.cab_id] = individual + self.shared_reward_weight * shared

        stats = StepStats(self.minute, open_count, self.completed, self.cancelled, avg_wait,
                          float(wait_cost), unserved_cost)
        self.history.append(stats)
        self.minute += 1
        return stats

    def observations(self):
        obs = {}
        for cab in self.cabs:
            nearby_zones = neighbours(*cab.position)
            nearby_requests = [r for r in self.requests if r.status == "open" and manhattan(cab.position, r.origin) <= 2]
            directional = []
            for zone in nearby_zones:
                directional.append(sum(r.status == "open" and r.origin == zone for r in self.requests))
                directional.append(sum(c.status == "idle" and c.position == zone for c in self.cabs))
            while len(directional) < 8:
                directional.extend([0, 0])
            zone_open = sum(r.status == "open" and r.origin == cab.position for r in self.requests)
            local_idle = sum(c.status == "idle" and c.position == cab.position for c in self.cabs)
            hour = (self.minute % 1440) / 60.0
            obs[cab.cab_id] = [
                cab.position[0] / max(1, GRID_SIZE - 1),
                cab.position[1] / max(1, GRID_SIZE - 1),
                hour / 24.0,
                1.0 if cab.status == "idle" else 0.0,
                1.0 if self.is_shift_active else 0.0,
                min(zone_open, 5) / 5,
                min(len(nearby_requests), 5) / 5,
                min(local_idle, self.fleet_size) / max(1, self.fleet_size),
            ] + [min(v, 5) / 5 if i % 2 == 0 else min(v, self.fleet_size) / max(1, self.fleet_size)
                  for i, v in enumerate(directional[:8])]
        return obs

    def run(self, policy):
        self.reset()
        while self.minute < self.day_minutes:
            self.step(policy(self))
        busy = sum(c.busy_minutes for c in self.cabs)
        for cab in self.cabs:
            cab.earnings = sum(10.0 + 3.0 * r.distance for r in self.requests
                               if r.status == "completed" and r.assigned_cab == cab.cab_id)
        earnings = [c.earnings for c in self.cabs]
        mean_earn = sum(earnings) / max(1, len(earnings))
        fairness = 1.0 - (((sum((x - mean_earn) ** 2 for x in earnings) / max(1, len(earnings))) ** 0.5)
                          / max(1e-9, mean_earn))
        return {
            "minutes": self.day_minutes,
            "requests": self.total_requests,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "service_rate": self.completed / max(1, self.completed + self.cancelled),
            "avg_wait": self.total_wait / max(1, self.completed),
            "avg_pickup_distance": self.total_pickup_distance / max(1, self.completed),
            "empty_distance": self.total_empty_distance,
            "empty_driving_ratio": self.total_empty_distance / max(1, self.total_empty_distance + busy),
            "utilisation": busy / max(1, self.fleet_size * self.day_minutes),
            "earnings": sum(earnings),
            "earnings_fairness": max(0.0, fairness),
        }
