"""The space of environments the generator can produce.

A `WorldSpec` fully determines an environment: same spec, same world,
always. That matters twice over — it keeps the corpus reproducible, and it
means a generated world can be stored as a few hundred bytes of
configuration rather than a serialised grid.

The mechanics here were chosen to span the axes ARC-AGI-3 actually tests,
not to be an exhaustive game engine:

- **hidden state** (`charge_period`) — the defining difficulty. Identical
  visible grids with different successors, so a model must posit structure
  it cannot see.
- **switches and gates** — state that persists and changes what later
  actions do, which is where "the same action means different things at
  different times" comes from.
- **hazards** — irreversible failure, so a planner must prune rather than
  explore freely.
- **ordered objectives** — goals that are not commutative, which defeats
  greedy strategies and forces real search.

A world is only emitted if it is *solvable*, verified by search against
its own exact model. An unsolvable world in the corpus would teach the
teacher that giving up is sometimes correct, which is the last lesson we
want it to learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Coord = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Mechanics:
    """Which rules are active. The axes of variation."""

    step_distance: int = 1
    """Cells travelled per move under normal conditions."""

    charge_period: int | None = None
    """If set, every Nth move travels one extra cell. Invisible in the grid.

    This is the hidden-state mechanic. A model that tracks only what it can
    see will be wrong exactly 1 move in N, which is enough to make every
    multi-step plan unreliable while leaving cell-level accuracy near
    perfect — the trap Phase 1 measured.
    """

    wrap_edges: bool = False
    """Walking off one edge re-enters from the opposite side."""

    has_hazards: bool = False
    has_switches: bool = False
    """Switch cells toggle every gate cell between solid and passable."""

    ordered_targets: bool = False
    """Targets must be collected in a fixed order; wrong order does nothing."""

    def summary(self) -> str:
        bits = [f"step={self.step_distance}"]
        if self.charge_period:
            bits.append(f"charge={self.charge_period}")
        if self.wrap_edges:
            bits.append("wrap")
        if self.has_hazards:
            bits.append("hazards")
        if self.has_switches:
            bits.append("switches")
        if self.ordered_targets:
            bits.append("ordered")
        return " ".join(bits)

    def to_json(self) -> dict[str, Any]:
        return {
            "step_distance": self.step_distance,
            "charge_period": self.charge_period,
            "wrap_edges": self.wrap_edges,
            "has_hazards": self.has_hazards,
            "has_switches": self.has_switches,
            "ordered_targets": self.ordered_targets,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Mechanics:
        return cls(
            step_distance=int(d.get("step_distance", 1)),
            charge_period=d.get("charge_period"),
            wrap_edges=bool(d.get("wrap_edges", False)),
            has_hazards=bool(d.get("has_hazards", False)),
            has_switches=bool(d.get("has_switches", False)),
            ordered_targets=bool(d.get("ordered_targets", False)),
        )


@dataclass(frozen=True, slots=True)
class LevelSpec:
    """One static layout."""

    start: Coord
    walls: frozenset[Coord] = field(default_factory=frozenset)
    hazards: frozenset[Coord] = field(default_factory=frozenset)
    targets: tuple[Coord, ...] = ()
    """Ordered. Under `ordered_targets` the sequence is binding; otherwise
    it is just a stable listing."""
    switches: frozenset[Coord] = field(default_factory=frozenset)
    gates: frozenset[Coord] = field(default_factory=frozenset)

    def to_json(self) -> dict[str, Any]:
        return {
            "start": list(self.start),
            "walls": sorted(list(c) for c in self.walls),
            "hazards": sorted(list(c) for c in self.hazards),
            "targets": [list(c) for c in self.targets],
            "switches": sorted(list(c) for c in self.switches),
            "gates": sorted(list(c) for c in self.gates),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> LevelSpec:
        return cls(
            start=(int(d["start"][0]), int(d["start"][1])),
            walls=frozenset((int(a), int(b)) for a, b in d.get("walls", [])),
            hazards=frozenset((int(a), int(b)) for a, b in d.get("hazards", [])),
            targets=tuple((int(a), int(b)) for a, b in d.get("targets", [])),
            switches=frozenset((int(a), int(b)) for a, b in d.get("switches", [])),
            gates=frozenset((int(a), int(b)) for a, b in d.get("gates", [])),
        )


@dataclass(frozen=True, slots=True)
class WorldSpec:
    """A complete, reproducible environment."""

    world_id: str
    seed: int
    field_size: int
    mechanics: Mechanics
    levels: tuple[LevelSpec, ...]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def summary(self) -> str:
        return (
            f"{self.world_id} [{self.field_size}x{self.field_size}, "
            f"{self.num_levels} levels, {self.mechanics.summary()}]"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "seed": self.seed,
            "field_size": self.field_size,
            "mechanics": self.mechanics.to_json(),
            "levels": [lv.to_json() for lv in self.levels],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> WorldSpec:
        return cls(
            world_id=str(d["world_id"]),
            seed=int(d["seed"]),
            field_size=int(d["field_size"]),
            mechanics=Mechanics.from_json(d["mechanics"]),
            levels=tuple(LevelSpec.from_json(lv) for lv in d["levels"]),
        )
