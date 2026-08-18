"""Evolving the scaffold, with an overseer that assumes it will be cheated.

The loop is ordinary: mutate the incumbent, evaluate, keep what wins. What
matters is the three guards around it, each answering a specific way this
kind of search goes wrong.

**Fitness charges for actions.** Solve rate alone is trivially improved by
exploring longer, and exploration spends exactly the resource the benchmark
measures. A genome that solves more worlds by spending twice the actions
has not improved anything, so actions are priced into the score.

**Promotion is decided on worlds the search never saw.** A candidate that
beats the incumbent on the tuning set is re-evaluated on a held-out guard
set and promoted only if it holds up there. This is the direct defence
against fitting the tuning worlds, and it is why `Archive.best` ranks on
the guard score and never on the tuning score.

**Every version is archived, promoted or not.** Rejected candidates are the
record of what the search tried to get away with.

The plan's warning is worth repeating: a system rewriting itself against a
metric will find holes in that metric. These guards close the ones that are
known. They cannot close the ones that are not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel.adapt.hypothesis import scorable_segment
from sentinel.adapt.search import ALL_HYPOTHESES, exhaustive_search
from sentinel.core.agent import read_layout, run_episode
from sentinel.env.types import Action
from sentinel.gen.grid import GridWorld
from sentinel.gen.spec import Mechanics, WorldSpec
from sentinel.memory.library import Entry, SkillLibrary
from sentinel.memory.signature import Signature

from .archive import Archive, Version
from .genome import ScaffoldGenome

ACTION_PRICE = 0.001
"""Score charged per environment action.

Set so that ~100 extra actions costs about as much as 10% of solve rate --
enough that spending actions must actually buy solutions, not so much that
a genuinely better but slightly slower configuration is rejected."""


@dataclass(frozen=True, slots=True)
class Evaluation:
    solve_rate: float
    mean_actions: float
    mechanics_exact: float

    @property
    def score(self) -> float:
        """Solve rate, priced for the actions it consumed."""
        return self.solve_rate - ACTION_PRICE * self.mean_actions

    def summary(self) -> str:
        return (
            f"solve {self.solve_rate:5.1%} actions {self.mean_actions:5.1f} "
            f"exact {self.mechanics_exact:5.1%} score {self.score:.3f}"
        )


def evaluate_genome(
    genome: ScaffoldGenome,
    specs: list[WorldSpec],
    library: SkillLibrary | None = None,
    seed: int = 0,
) -> Evaluation:
    """Run the full loop under one configuration and price the result."""
    solved: list[bool] = []
    actions: list[float] = []
    exact: list[bool] = []

    for spec in specs:
        world = GridWorld(spec)
        world.reset()
        rng = np.random.default_rng(seed)
        used = 0
        for _ in range(genome.explore_steps):
            if world.done:
                break
            world.step(Action(int(rng.integers(1, 6))))
            used += 1

        history = world.history
        segment = scorable_segment(history)
        if len(segment.steps) < 5:
            mechanics = Mechanics(step_distance=1, charge_period=None)
        else:
            observed = read_layout(segment.initial.grid, spec.field_size)
            order = ALL_HYPOTHESES
            if library is not None and genome.library_strength > 0:
                signature = Signature.from_frame(segment.initial, spec.field_size)
                order = library.rank(signature, ALL_HYPOTHESES, k=genome.library_k)
            found = exhaustive_search(history, observed, spec.field_size, order=order)
            mechanics = found.mechanics
            if library is not None:
                library.add(
                    Entry(
                        world_id=spec.world_id,
                        signature=Signature.from_frame(segment.initial, spec.field_size),
                        classes=found.best.classes,
                        fitness=found.best.fitness,
                    )
                )

        outcome = run_episode(spec, mechanics, seed=seed)
        solved.append(outcome.solved)
        actions.append(used + outcome.actions_used)
        exact.append(mechanics.summary() == spec.mechanics.summary())

    n = max(1, len(solved))
    return Evaluation(
        solve_rate=sum(solved) / n,
        mean_actions=float(np.mean(actions)) if actions else 0.0,
        mechanics_exact=sum(exact) / n,
    )


def evolve(
    train: list[WorldSpec],
    guard: list[WorldSpec],
    generations: int = 6,
    population: int = 4,
    seed: int = 0,
    verbose: bool = True,
) -> Archive:
    """Search scaffold configurations. Returns the full archive.

    `guard` must be worlds the search is never scored on during selection.
    Passing the same list for both defeats the entire point and is the one
    misuse worth being loud about.
    """
    if guard and train and {s.world_id for s in guard} & {s.world_id for s in train}:
        raise ValueError(
            "guard worlds overlap the tuning set; promotion would be measuring "
            "the thing it is meant to check"
        )

    rng = np.random.default_rng(seed)
    archive = Archive()

    incumbent = ScaffoldGenome()
    base_train = evaluate_genome(incumbent, train, SkillLibrary(), seed)
    base_guard = evaluate_genome(incumbent, guard, SkillLibrary(), seed)
    archive.record(
        Version(0, incumbent, base_train.score, base_guard.score,
                base_train.mean_actions, True, "baseline")
    )
    if verbose:
        print(f"gen 0 baseline  train {base_train.summary()}", flush=True)
    best_guard = base_guard.score

    for generation in range(1, generations + 1):
        for _ in range(population):
            candidate = incumbent.mutate(rng)
            train_eval = evaluate_genome(candidate, train, SkillLibrary(), seed)

            if train_eval.score <= base_train.score:
                archive.record(
                    Version(generation, candidate, train_eval.score, None,
                            train_eval.mean_actions, False, "no gain on tuning set")
                )
                continue

            # Only now does it cost guard evaluations.
            guard_eval = evaluate_genome(candidate, guard, SkillLibrary(), seed)
            held = guard_eval.score >= best_guard
            archive.record(
                Version(generation, candidate, train_eval.score, guard_eval.score,
                        train_eval.mean_actions, held,
                        "" if held else "gain did not survive the guard set")
            )
            if verbose:
                print(
                    f"gen {generation}  cand {candidate.summary()}  "
                    f"train {train_eval.score:.3f} guard {guard_eval.score:.3f} "
                    f"{'PROMOTED' if held else 'rejected'}",
                    flush=True,
                )
            if held:
                incumbent, base_train, best_guard = candidate, train_eval, guard_eval.score

    return archive
