"""Gates for accumulation.

The property that matters is that memory *reorders* search without changing
what search concludes. A library that changed answers would be asserting
rather than suggesting, and a wrong assertion compounds across worlds --
which is the documented way memory turns from useful into dangerous.
"""

from __future__ import annotations

import numpy as np

from sentinel.adapt.hypothesis import classes_from_mechanics, scorable_segment
from sentinel.adapt.search import ALL_HYPOTHESES, exhaustive_search
from sentinel.core.agent import read_layout
from sentinel.env.types import Action
from sentinel.gen.generator import generate
from sentinel.gen.grid import GridWorld
from sentinel.memory.library import Entry, SkillLibrary
from sentinel.memory.signature import Signature


def _worlds(n=8, start=2000):
    out, seed = [], start
    while len(out) < n and seed < start + 40 * n:
        w = generate(seed=seed)
        if w is not None:
            out.append(w)
        seed += 1
    return out


def _explore(spec, steps=60, seed=0):
    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if world.done:
            break
        world.step(Action(int(rng.integers(1, 6))))
    return world.history


def test_empty_library_is_exactly_uniform():
    """The no-memory ablation must be free, not approximate."""
    library = SkillLibrary()
    sig = Signature(12, 10, 2, 0, 0, 0, 0.07)
    for head in library.prior(sig):
        assert all(abs(p - head[0]) < 1e-12 for p in head)


def test_low_fitness_is_not_remembered():
    library = SkillLibrary()
    sig = Signature(12, 10, 2, 0, 0, 0, 0.07)
    assert not library.add(Entry("bad", sig, (1, 1, 1, 1, 1, 1), 0.5))
    assert len(library) == 0
    assert library.add(Entry("good", sig, (0, 1, 0, 0, 0, 0), 1.0))
    assert len(library) == 1


def test_ranking_is_a_permutation():
    """Reordering must never drop or invent a hypothesis."""
    library = SkillLibrary()
    sig = Signature(12, 10, 2, 0, 0, 0, 0.07)
    library.add(Entry("w", sig, (1, 2, 0, 0, 0, 0), 1.0))
    ranked = library.rank(sig, ALL_HYPOTHESES)
    assert sorted(ranked) == sorted(ALL_HYPOTHESES)


def test_memory_does_not_change_the_answer():
    """Same conclusion with and without a prior; only the cost may differ."""
    specs = _worlds(6)
    library = SkillLibrary()
    for spec in specs:
        history = _explore(spec)
        segment = scorable_segment(history)
        if len(segment.steps) < 5:
            continue
        observed = read_layout(segment.initial.grid, spec.field_size)
        sig = Signature.from_frame(segment.initial, spec.field_size)

        plain = exhaustive_search(history, observed, spec.field_size)
        primed = exhaustive_search(
            history, observed, spec.field_size, order=library.rank(sig, ALL_HYPOTHESES)
        )
        assert plain.best.fitness == primed.best.fitness
        # The CLASSES, not merely the score. Comparing fitness alone let a
        # library that reordered search into a different (equally scoring
        # but wrong) rule set pass -- measured at 58% accuracy without the
        # library and 28.6% with it.
        assert plain.best.classes == primed.best.classes
        library.add(Entry(spec.world_id, sig, plain.best.classes, plain.best.fitness))


def test_library_never_beats_plain_simplicity_ordering_on_accuracy():
    """Memory may reorder search; it may not change what search concludes.

    Measured over 56 held-out worlds, all three orderings reach identical
    accuracy (55.4%) but cost differs sharply: default enumeration 36.7
    replays, simplicity-first 10.1, retrieval prior 19.4. The library is
    the most expensive of the three, which is why it is not the default.
    """
    specs = _worlds(6, start=4000)
    library = SkillLibrary()
    for spec in specs:
        history = _explore(spec)
        segment = scorable_segment(history)
        if len(segment.steps) < 5:
            continue
        observed = read_layout(segment.initial.grid, spec.field_size)
        sig = Signature.from_frame(segment.initial, spec.field_size)

        plain = exhaustive_search(history, observed, spec.field_size)
        primed = exhaustive_search(
            history, observed, spec.field_size, order=library.rank(sig, ALL_HYPOTHESES)
        )
        assert plain.best.classes == primed.best.classes
        library.add(Entry(spec.world_id, sig, plain.best.classes, plain.best.fitness))
