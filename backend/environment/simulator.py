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

class CabDispatchEnv:
    """Small deterministic-friendly simulator used by baselines and DQN."""
    ACTION_WAIT = 0
    ACTION_ACCEPT = 1
    ACTION_NORTH = 2
    ACTION_SOUTH = 3
    ACTION_WEST = 4
    ACTION_EAST = 5
    ACTION_NAMES = {0:"WAIT",1:"ACCEPT",2:"NORTH",3:"SOUTH",4:"WEST",5:"EAST"}

    def __init__(self, seed: int = 42, fleet_size: int = 8, day_minutes: int = 180):
        self.seed = seed
        self.rng = random.Random(seed)
        self.fleet_size = fleet_size
        self.day_minutes = day_minutes
        self.reset()

    def reset(self) -> dict:
        self.rng = random.Random(self.seed)
        self.minute = 0
        self.next_cab_positions = [(i // GRID_SIZE, i % GRID_SIZE) for i in range(self.fleet_size)]
        self.cabs = [Cab(i, self.next_cab_positions[i]) for i in range(self.fleet_size)]
        self.demand = DemandGenerator(self.seed + 100)
        self.requests: list[RideRequest] = []
        self.completed = 0
        self.cancelled = 0
        self.total_wait = 0
        self.total_fare = 0.0
        self.total_empty_distance = 0
        self.history: list[StepStats] = []
        return self.observations()

    def _move(self, cab: Cab, action: int) -> bool:
        dxdy = {self.ACTION_NORTH:(-1,0), self.ACTION_SOUTH:(1,0), self.ACTION_WEST:(0,-1), self.ACTION_EAST:(0,1)}
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

    def _assign(self, cab: Cab, request: RideRequest) -> None:
        pickup_distance = manhattan(cab.position, request.origin)
        cab.empty_distance += pickup_distance
        self.total_empty_distance += pickup_distance
        request.status = "assigned"
        request.assigned_cab = cab.cab_id
        request.pickup_time = self.minute + pickup_distance
        cab.position = request.origin
        trip_distance = max(1, request.distance)
        cab.status = "occupied"
        cab.ride_id = request.request_id
        cab.destination = request.destination
        cab.remaining = trip_distance
        cab.earnings += 10.0 + 3.0 * trip_distance

    def _complete_rides(self) -> None:
        by_id = {r.request_id:r for r in self.requests}
        for cab in self.cabs:
            if cab.status == "occupied":
                cab.tick_busy()
                cab.remaining -= 1
                if cab.remaining <= 0:
                    ride = by_id.get(cab.ride_id)
                    if ride:
                        ride.status = "completed"
                        ride.completed_at = self.minute
                        self.completed += 1
                        self.total_wait += ride.wait_minutes
                        self.total_fare += 10.0 + 3.0 * ride.distance
                    cab.position = cab.destination or cab.position
                    cab.status, cab.ride_id, cab.destination = "idle", None, None

    def _cancel_expired(self) -> None:
        for r in self.requests:
            if r.status == "open" and self.minute - r.created_at >= 8:
                r.status = "cancelled"
                self.cancelled += 1

    def step(self, actions: dict[int,int]) -> StepStats:
        new_requests = self.demand.requests_for_minute(self.minute)
        self.requests.extend(new_requests)
        self._complete_rides()
        self._cancel_expired()

        for cab in self.cabs:
            if cab.status != "idle":
                continue
            action = actions.get(cab.cab_id, self.ACTION_WAIT)
            if action == self.ACTION_ACCEPT:
                open_requests = [r for r in self.requests if r.status == "open"]
                if open_requests:
                    request = min(open_requests, key=lambda r: manhattan(cab.position, r.origin))
                    self._assign(cab, request)
            elif action in (self.ACTION_NORTH,self.ACTION_SOUTH,self.ACTION_WEST,self.ACTION_EAST):
                self._move(cab, action)

        open_count = sum(r.status == "open" for r in self.requests)
        avg_wait = self.total_wait / max(1, self.completed)
        stats = StepStats(self.minute, open_count, self.completed, self.cancelled, avg_wait)
        self.history.append(stats)
        self.minute += 1
        return stats

    def observations(self) -> dict[int, list[float]]:
        obs = {}
        for cab in self.cabs:
            nearby = [r for r in self.requests if r.status == "open" and manhattan(cab.position, r.origin) <= 2]
            n = neighbours(*cab.position)
            local_idle = sum(c.status == "idle" and c.position == cab.position for c in self.cabs)
            neighbour_idle = sum(c.status == "idle" and c.position in n for c in self.cabs)
            zone_open = sum(r.status == "open" and r.origin == cab.position for r in self.requests)
            obs[cab.cab_id] = [cab.position[0]/4, cab.position[1]/4, self.minute/self.day_minutes, 1.0 if cab.status == "idle" else 0.0,
                               min(zone_open,5)/5, min(len(nearby),5)/5, min(local_idle,8)/8, min(neighbour_idle,8)/8]
        return obs

    def run(self, policy) -> dict:
        self.reset()
        while self.minute < self.day_minutes:
            actions = policy(self)
            self.step(actions)
        utilisation = sum(c.busy_minutes for c in self.cabs) / max(1, self.fleet_size * self.day_minutes)
        return {
            "minutes": self.day_minutes,
            "requests": len([r for r in self.requests if r.status in {"completed","cancelled","assigned"}]),
            "completed": self.completed,
            "cancelled": self.cancelled,
            "service_rate": self.completed / max(1, self.completed + self.cancelled),
            "avg_wait": self.total_wait / max(1, self.completed),
            "empty_distance": self.total_empty_distance,
            "empty_driving_ratio": self.total_empty_distance / max(1, self.total_empty_distance + sum(c.busy_minutes for c in self.cabs)),
            "utilisation": utilisation,
            "earnings": self.total_fare,
        }
