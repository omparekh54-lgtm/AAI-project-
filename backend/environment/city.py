from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

GRID_SIZE = 5
AIRPORT_ZONE = (0, 4)

@dataclass(frozen=True)
class Zone:
    x: int
    y: int
    @property
    def id(self) -> int:
        return self.x * GRID_SIZE + self.y


def neighbours(x: int, y: int) -> list[tuple[int, int]]:
    candidates = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
    return [(a,b) for a,b in candidates if 0 <= a < GRID_SIZE and 0 <= b < GRID_SIZE]


def manhattan(a: tuple[int,int], b: tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def all_zones() -> Iterable[Zone]:
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            yield Zone(x, y)
