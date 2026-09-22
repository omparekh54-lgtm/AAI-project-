from __future__ import annotations
from dataclasses import dataclass
import random
from .city import AIRPORT_ZONE, GRID_SIZE

@dataclass
class RideRequest:
    request_id: int
    origin: tuple[int,int]
    destination: tuple[int,int]
    created_at: int
    status: str = "open"
    assigned_cab: int | None = None
    pickup_time: int | None = None
    completed_at: int | None = None

    @property
    def wait_minutes(self) -> int:
        if self.pickup_time is None:
            return 0
        return max(0, self.pickup_time - self.created_at)

    @property
    def distance(self) -> int:
        return abs(self.origin[0]-self.destination[0]) + abs(self.origin[1]-self.destination[1])

class DemandGenerator:
    def __init__(self, seed: int = 42, airport_hotspot: float = 0.22):
        self.rng = random.Random(seed)
        self.airport_hotspot = airport_hotspot
        self.next_id = 1

    def requests_for_minute(self, minute: int) -> list[RideRequest]:
        hour = (6 * 60 + minute) / 60.0
        if 8 <= hour < 10:
            rate = 0.72
        elif 17 <= hour < 20:
            rate = 0.86
        elif 11 <= hour < 16:
            rate = 0.48
        else:
            rate = 0.28
        count = 1 if self.rng.random() < rate else 0
        if self.rng.random() < rate * 0.25:
            count += 1
        out = []
        for _ in range(count):
            if self.rng.random() < self.airport_hotspot:
                origin = AIRPORT_ZONE
            else:
                origin = (self.rng.randrange(GRID_SIZE), self.rng.randrange(GRID_SIZE))
            destination = (self.rng.randrange(GRID_SIZE), self.rng.randrange(GRID_SIZE))
            if destination == origin:
                destination = ((destination[0] + 1) % GRID_SIZE, destination[1])
            out.append(RideRequest(self.next_id, origin, destination, minute))
            self.next_id += 1
        return out
