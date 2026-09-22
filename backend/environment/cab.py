from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Cab:
    cab_id: int
    position: tuple[int,int]
    status: str = "idle"
    ride_id: int | None = None
    destination: tuple[int,int] | None = None
    remaining: int = 0
    earnings: float = 0.0
    empty_distance: int = 0
    busy_minutes: int = 0

    def tick_busy(self) -> None:
        if self.status != "idle":
            self.busy_minutes += 1
