from __future__ import annotations
from dataclasses import dataclass
import random
from .city import AIRPORT_ZONE, GRID_SIZE

@dataclass
class RideRequest:
    request_id: int
    origin: tuple[int, int]
    destination: tuple[int, int]
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
        return abs(self.origin[0] - self.destination[0]) + abs(self.origin[1] - self.destination[1])


class DemandGenerator:
    """Stochastic 24-hour demand profile with peaks and an airport hotspot.

    The rates are configurable synthetic priors intended as a calibration surface
    for later fitting to public taxi-trip statistics. They are not claimed to be
    measurements from a particular city.
    """
    def __init__(self, seed: int = 42, airport_hotspot: float = 0.22):
        self.rng = random.Random(seed)
        self.airport_hotspot = airport_hotspot
        self.next_id = 1

    def _rate(self, minute: int) -> float:
        hour = (minute % 1440) / 60.0
        if 7 <= hour < 10:
            return 0.55
        if 17 <= hour < 21:
            return 0.65
        if 11 <= hour < 16:
            return 0.38
        if 0 <= hour < 6:
            return 0.12
        return 0.24

    def requests_for_minute(self, minute: int) -> list[RideRequest]:
        rate = self._rate(minute)
        count = int(self.rng.random() < rate)
        count += int(self.rng.random() < rate * 0.35)
        count += int(self.rng.random() < rate * 0.10)
        out = []
        for _ in range(count):
            origin = AIRPORT_ZONE if self.rng.random() < self.airport_hotspot else (
                self.rng.randrange(GRID_SIZE), self.rng.randrange(GRID_SIZE)
            )
            destination = (self.rng.randrange(GRID_SIZE), self.rng.randrange(GRID_SIZE))
            if destination == origin:
                destination = ((destination[0] + 1) % GRID_SIZE, destination[1])
            out.append(RideRequest(self.next_id, origin, destination, minute))
            self.next_id += 1
        return out
