"""The scaffold's own parameters, as something that can be searched.

**Scope, stated honestly.** This is self-modification of the scaffold's
*configuration*, not of its source code. SICA-style code rewriting needs a
model that writes code, and the whole point of Phase 3 onward is to remove
the LLM from the loop -- so what remains is a search over the knobs the
scaffold exposes. That is a real and useful form of self-improvement, and
it is deliberately not called more than it is.

**Every knob that buys performance with resources is bounded.** Exploration
costs real environment actions, which is the currency the benchmark
charges; a genome allowed to explore for a thousand steps would "improve"
by spending what it is supposed to conserve. The bounds here are the first
line of defence against a system optimising its own metric, and `fitness`
in `search.py` charges for actions as the second.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ScaffoldGenome:
    """One configuration of the perceive-infer-plan-act loop."""

    explore_steps: int = 60
    """Environment actions spent gathering evidence before inferring."""

    simplicity_weight: int = 1
    """Tie-break strength toward simpler rule sets when fitness ties."""

    order_search_cap: int = 4
    """Max targets whose orderings are enumerated for ordered hypotheses."""

    library_k: int = 8
    """Neighbours consulted when building a retrieval prior."""

    library_strength: float = 1.0
    """How hard the library is allowed to pull. 0.0 disables memory."""

    planner_nodes: int = 120_000
    stall_limit: int = 4

    BOUNDS: dict[str, tuple[float, float]] = None  # type: ignore[assignment]

    def clamped(self) -> ScaffoldGenome:
        return replace(
            self,
            explore_steps=int(np.clip(self.explore_steps, 8, 120)),
            simplicity_weight=int(np.clip(self.simplicity_weight, 0, 4)),
            order_search_cap=int(np.clip(self.order_search_cap, 2, 5)),
            library_k=int(np.clip(self.library_k, 1, 32)),
            library_strength=float(np.clip(self.library_strength, 0.0, 4.0)),
            planner_nodes=int(np.clip(self.planner_nodes, 10_000, 300_000)),
            stall_limit=int(np.clip(self.stall_limit, 1, 8)),
        )

    def mutate(self, rng: np.random.Generator, rate: float = 0.4) -> ScaffoldGenome:
        """Perturb a few knobs. Always returns a genome inside the bounds."""
        g = self
        if rng.random() < rate:
            g = replace(g, explore_steps=int(g.explore_steps + rng.integers(-20, 21)))
        if rng.random() < rate:
            g = replace(g, simplicity_weight=int(g.simplicity_weight + rng.integers(-1, 2)))
        if rng.random() < rate:
            g = replace(g, order_search_cap=int(g.order_search_cap + rng.integers(-1, 2)))
        if rng.random() < rate:
            g = replace(g, library_k=int(g.library_k + rng.integers(-4, 5)))
        if rng.random() < rate:
            g = replace(g, library_strength=float(g.library_strength + rng.normal(0, 0.5)))
        if rng.random() < rate:
            g = replace(g, stall_limit=int(g.stall_limit + rng.integers(-1, 2)))
        return g.clamped()

    def to_json(self) -> dict[str, Any]:
        return {
            "explore_steps": self.explore_steps,
            "simplicity_weight": self.simplicity_weight,
            "order_search_cap": self.order_search_cap,
            "library_k": self.library_k,
            "library_strength": self.library_strength,
            "planner_nodes": self.planner_nodes,
            "stall_limit": self.stall_limit,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ScaffoldGenome:
        return cls(
            explore_steps=int(d.get("explore_steps", 60)),
            simplicity_weight=int(d.get("simplicity_weight", 1)),
            order_search_cap=int(d.get("order_search_cap", 4)),
            library_k=int(d.get("library_k", 8)),
            library_strength=float(d.get("library_strength", 1.0)),
            planner_nodes=int(d.get("planner_nodes", 120_000)),
            stall_limit=int(d.get("stall_limit", 4)),
        )

    def summary(self) -> str:
        return (
            f"explore={self.explore_steps} simp={self.simplicity_weight} "
            f"order={self.order_search_cap} k={self.library_k} "
            f"str={self.library_strength:.2f} stall={self.stall_limit}"
        )
