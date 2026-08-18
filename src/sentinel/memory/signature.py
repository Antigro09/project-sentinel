"""What a world looks like, before you know how it works.

A signature is a fingerprint built only from things a frame actually shows:
how big the board is, how much of it is wall, how many targets there are,
whether switches or gates or hazards appear at all. It deliberately contains
**no mechanics** -- not step distance, not the hidden counter, not whether
order matters. Those are the things being inferred, and a retrieval key that
already encoded them would be looking up the answer.

The purpose is retrieval: worlds that look alike often behave alike, so a
verified rule set from a similar world is a good place to start searching.
That is a prior, not a conclusion, and everything downstream re-verifies it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentinel.env.types import Observation
from sentinel.gen.grid import AGENT, GATE_CLOSED, GATE_OPEN, HAZARD, SWITCH, TARGET, WALL


@dataclass(frozen=True, slots=True)
class Signature:
    """Observable structure of one world."""

    field_size: int
    n_walls: int
    n_targets: int
    n_hazards: int
    n_switches: int
    n_gates: int
    wall_density: float

    @classmethod
    def from_frame(cls, frame: Observation, field_size: int) -> Signature:
        counts = {WALL: 0, TARGET: 0, HAZARD: 0, SWITCH: 0, GATE_OPEN: 0, GATE_CLOSED: 0, AGENT: 0}
        for y in range(field_size):
            for x in range(field_size):
                v = frame.grid[y][x]
                if v in counts:
                    counts[v] += 1
        area = max(1, field_size * field_size)
        return cls(
            field_size=field_size,
            n_walls=counts[WALL],
            n_targets=counts[TARGET],
            n_hazards=counts[HAZARD],
            n_switches=counts[SWITCH],
            n_gates=counts[GATE_OPEN] + counts[GATE_CLOSED],
            wall_density=counts[WALL] / area,
        )

    def distance(self, other: Signature) -> float:
        """Scale-aware dissimilarity. 0.0 means indistinguishable.

        Presence/absence of a feature counts for more than an exact count:
        a world with hazards is categorically unlike one without, while ten
        walls versus twelve is barely a difference at all.
        """
        d = 0.0
        d += abs(self.field_size - other.field_size) / 32.0
        d += abs(self.wall_density - other.wall_density) * 2.0
        d += abs(self.n_targets - other.n_targets) / 8.0
        for a, b in (
            (self.n_hazards, other.n_hazards),
            (self.n_switches, other.n_switches),
            (self.n_gates, other.n_gates),
        ):
            d += 1.0 if (a > 0) != (b > 0) else 0.0
        return d

    def to_json(self) -> dict[str, Any]:
        return {
            "field_size": self.field_size,
            "n_walls": self.n_walls,
            "n_targets": self.n_targets,
            "n_hazards": self.n_hazards,
            "n_switches": self.n_switches,
            "n_gates": self.n_gates,
            "wall_density": self.wall_density,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Signature:
        return cls(
            field_size=int(d["field_size"]),
            n_walls=int(d["n_walls"]),
            n_targets=int(d["n_targets"]),
            n_hazards=int(d["n_hazards"]),
            n_switches=int(d["n_switches"]),
            n_gates=int(d["n_gates"]),
            wall_density=float(d["wall_density"]),
        )
