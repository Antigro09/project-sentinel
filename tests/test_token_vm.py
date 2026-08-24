"""The token VM's fast path must not be coarser than its interpreter.

X48 swapped the grid for a token stream. Two things there are genuinely new
-- emission composes by ADDING counts, and halting is a state rather than an
absorbing sentinel -- and the second one was wrong. Every halted state
collapsed to one value, so `SEQ(ADV, HALT)` and bare `HALT` had identical
tables while the interpreter told them apart. The search returned
`(IF ','@+1 HALT ADV)` for "advance, then halt on a comma": halting WITHOUT
advancing, matching on every situation of every search tape.

A fast path coarser than the trusted path does not just lose speed. It
invents equivalences that are not there, and the search will happily spend
its budget inside them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x48_token_vm as T  # noqa: E402

TAPES = ["ab,12 c", "d3 0,a1", " ,bc23d"]


@pytest.fixture(scope="module")
def space():
    return T.TokenSpace(TAPES)


def test_halting_preserves_the_head_position(space):
    """The regression test. These two must NOT be the same behaviour."""
    halt_only = space.table("HALT")
    adv_then_halt = space.table(("SEQ", "ADV", "HALT"))
    assert not np.array_equal(halt_only, adv_then_halt)
    for e in ("HALT", ("SEQ", "ADV", "HALT")):
        assert np.array_equal(space.table(e), space.interpret(e))


def test_fast_path_equals_interpreter_on_random_programs(space):
    preds = T.atom_preds((0, 1))
    rng = np.random.default_rng(9)
    for _ in range(60):
        p, q = (preds[int(rng.integers(0, len(preds)))] for _ in range(2))
        a, b, c = (T.ACTIONS[int(rng.integers(0, len(T.ACTIONS)))]
                   for _ in range(3))
        e = ("SEQ", ("LOOP", ("IF", p, a, ("SEQ", b, c))), ("IF", q, c, a))
        assert np.array_equal(space.table(e), space.interpret(e)), T.render(e)


def test_fast_path_equals_interpreter_on_nested_loops(space):
    """Fuel must not bind, or the table describes a machine nobody runs."""
    preds = T.atom_preds((0,))
    rng = np.random.default_rng(21)
    for _ in range(20):
        p = preds[int(rng.integers(0, len(preds)))]
        e = ("LOOP", ("LOOP", ("IF", p, "ADV", ("SEQ", "EMIT", "ADV"))))
        assert np.array_equal(space.table(e), space.interpret(e))


def test_fast_path_agrees_on_every_truth(space):
    for name, truth in T.TRUTHS.items():
        assert np.array_equal(space.table(truth), space.interpret(truth)), name


def test_atoms_at_one_offset_partition_the_situations(space):
    """The premise the whole derivation rests on.

    If character tests at one offset partition the situations, then for any
    event set b the union of atoms meeting b is the minimal expressible
    superset of b -- computable in one pass, no lattice. It also means the
    OR-closure of that family IS its powerset, which is why the token lattice
    cannot be enumerated the way the grid's could.
    """
    for off in (0, 1):
        masks = [space.pred(a) for a in T.atom_preds((off,))]
        stacked = np.stack(masks)
        assert (stacked.sum(0) == 1).all(), f"offset {off} is not a partition"


def test_derived_is_digit_is_exact_and_correct(space):
    truth = T.TRUTHS["copy digits"]
    families = {off: [(a, space.pred(a)) for a in T.atom_preds((off,))]
                for off in (0, 1)}
    derived = T.derive_predicates(space, space.interpret(truth), families)
    exact = {name: term for name, term, ok in derived if ok}
    assert "emitted-here@+0" in exact
    got = space.pred(exact["emitted-here@+0"])
    want = space.pred(T.or_chain([("EQ", 0, c) for c in "0123"]))
    assert np.array_equal(got, want)


def test_the_truths_are_behaviourally_distinct(space):
    seen = {}
    for name, truth in T.TRUTHS.items():
        key = space.interpret(truth).tobytes()
        assert key not in seen, f"{name} is indistinguishable from {seen[key]}"
        seen[key] = name


def test_disagreement_synthesis_finds_a_splitting_input():
    """Pillar 4: the query is constructed, not chosen from a menu."""
    truth = T.TRUTHS["halt on comma"]
    impostor = ("IF", ("EQ", 1, ","), "HALT", "ADV")   # halts without advancing
    tape, _, tries = T.discriminating_tape([truth, impostor], truth,
                                           np.random.default_rng(0))
    assert tape is not None, "no input separated two genuinely different programs"
    assert T.space_free_output(truth, tape) != T.space_free_output(impostor, tape)
    assert tries < 100
