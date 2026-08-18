"""The Phase 5 question: does environment N+1 get cheaper as N grows?

This is the commercially decisive gate in the plan and the one the field is
worst at. It is also the easiest to fake, so the measurement is built to be
hard to fool:

- **Cost is counted, not judged.** Verifier replays to find the rules, and
  environment actions spent exploring. Both are objective counts.
- **The no-memory arm runs on the same worlds in the same order.** Any
  difference is the library, not the sample.
- **Accuracy is reported alongside cost.** A library that made search
  cheaper by making it wrong would show up immediately as a drop here, and
  cost improvements are only meaningful while accuracy holds.

Worlds are processed in sequence, and each is added to the library only
*after* it has been solved -- so every measurement is made with the library
as it stood before that world was seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sentinel.adapt.hypothesis import classes_from_mechanics, scorable_segment
from sentinel.adapt.search import ALL_HYPOTHESES, exhaustive_search
from sentinel.core.agent import read_layout
from sentinel.env.types import Action
from sentinel.gen.grid import GridWorld
from sentinel.gen.spec import WorldSpec

from .library import Entry, SkillLibrary
from .signature import Signature


@dataclass
class WorldCost:
    world_id: str
    replays: int
    correct: bool
    library_size: int


@dataclass
class CurveResult:
    label: str
    costs: list[WorldCost] = field(default_factory=list)

    @property
    def mean_replays(self) -> float:
        return float(np.mean([c.replays for c in self.costs])) if self.costs else 0.0

    @property
    def accuracy(self) -> float:
        return float(np.mean([c.correct for c in self.costs])) if self.costs else 0.0

    def window_means(self, windows: int = 4) -> list[float]:
        """Mean replays per equal slice of the sequence.

        A downward-sloping list is the Phase 5 exit condition. A flat one is
        the kill criterion -- and personal AGI is what dies there, so this
        number is not one to interpret generously.
        """
        if not self.costs:
            return []
        chunks = np.array_split(np.array([c.replays for c in self.costs]), windows)
        return [float(c.mean()) for c in chunks if len(c)]

    def summary(self) -> str:
        w = self.window_means()
        trail = "  ".join(f"{v:.1f}" for v in w)
        return (
            f"{self.label:16} mean {self.mean_replays:5.1f} replays  "
            f"acc {self.accuracy:5.1%}  by quarter: {trail}"
        )


def explore(spec: WorldSpec, steps: int = 60, seed: int = 0):
    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if world.done:
            break
        world.step(Action(int(rng.integers(1, 6))))
    return world.history


def run_sequence(
    specs: list[WorldSpec],
    library: SkillLibrary | None,
    explore_steps: int = 60,
    seed: int = 0,
) -> CurveResult:
    """Solve worlds in order, optionally learning from each as it goes.

    Passing `library=None` is the no-memory control: identical work, no
    ordering, nothing retained.
    """
    label = "with memory" if library is not None else "no memory"
    result = CurveResult(label=label)

    for spec in specs:
        history = explore(spec, explore_steps, seed)
        segment = scorable_segment(history)
        if len(segment.steps) < 5:
            continue
        observed = read_layout(segment.initial.grid, spec.field_size)
        signature = Signature.from_frame(segment.initial, spec.field_size)

        order = (
            library.rank(signature, ALL_HYPOTHESES) if library is not None else ALL_HYPOTHESES
        )
        found = exhaustive_search(history, observed, spec.field_size, order=order)

        truth = classes_from_mechanics(spec.mechanics)
        result.costs.append(
            WorldCost(
                world_id=spec.world_id,
                replays=found.replays,
                correct=found.best.classes == truth,
                library_size=len(library) if library is not None else 0,
            )
        )
        if library is not None:
            library.add(
                Entry(
                    world_id=spec.world_id,
                    signature=signature,
                    classes=found.best.classes,
                    fitness=found.best.fitness,
                )
            )

    return result
