"""The unified substrate must not drift, and its accelerator must not lie.

X46 put motion and hazards in one program space; X47 added a table-based
fast path that composes behaviours by array indexing instead of re-running
the interpreter. That fast path is a second implementation of the same
semantics, and a second implementation is a second chance to be wrong.

It WAS wrong. `LOOP` ran the body one extra time, because starting the
composition at `out = a` is already the first pass. Nothing detected it on
programs that settle -- a fixed point is a fixed point however many times
you reach it -- but a body that OSCILLATES, STEP followed by UNDO, lands
somewhere different on an odd pass than on an even one. 401 of 5,000
situations disagreed. These tests pin that down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x46_unified_substrate as X  # noqa: E402
import x47_priced_vocabulary as W  # noqa: E402


@pytest.fixture(scope="module")
def space():
    boards = [X.make_board(11, 5), X.make_board(12, 5)]
    X.PROBE_BOARDS = boards
    return W.Space(boards)


def test_step_moves_without_knowing_about_walls(space):
    """STEP must not do the collision test the program is meant to express.

    X46's first version had STEP refuse to enter a wall, which recovered
    `step 1` as bare STEP -- the primitive quietly doing the work.
    """
    board = space.boards[0]
    inside = [(x, y) for y in range(board.size) for x in range(board.size)]
    walls = [c for c in inside if c in board.walls]
    assert walls, "probe board has no walls to step into"
    for wall in walls:
        for aid, (dx, dy) in X.DIRS.items():
            src = (wall[0] - dx, wall[1] - dy)
            if src not in inside:
                continue
            st = X.State(pos=src, began=src, dir=(dx, dy))
            out, _ = X.run("STEP", board, st)
            assert out.pos == wall, "STEP must enter walls; the guard is the program's job"


def test_step_stops_at_the_board_edge(space):
    board = space.boards[0]
    for aid, (dx, dy) in X.DIRS.items():
        for i in range(board.size):
            pos = {(0, -1): (i, 0), (0, 1): (i, board.size - 1),
                   (-1, 0): (0, i), (1, 0): (board.size - 1, i)}[(dx, dy)]
            st = X.State(pos=pos, began=pos, dir=(dx, dy))
            out, _ = X.run("STEP", board, st)
            assert out.pos == pos


def test_fast_path_equals_interpreter_on_oscillating_loops(space):
    """The regression test for the off-by-one: bodies that oscillate."""
    preds = X.enumerate_preds(3)
    rng = np.random.default_rng(11)
    for _ in range(40):
        p = preds[int(rng.integers(0, len(preds)))]
        e = ("LOOP", ("IF", p, "STEP", "UNDO"))
        assert np.array_equal(space.table(e), space.interpret(e))


def test_fast_path_equals_interpreter_on_random_programs(space):
    preds = X.enumerate_preds(4)
    rng = np.random.default_rng(5)
    for _ in range(80):
        p, q = (preds[int(rng.integers(0, len(preds)))] for _ in range(2))
        a, b, c = (X.ACTIONS[int(rng.integers(0, len(X.ACTIONS)))] for _ in range(3))
        e = ("SEQ", ("IF", p, a, ("LOOP", ("IF", q, b, c))), ("IF", q, c, a))
        assert np.array_equal(space.table(e), space.interpret(e))


def test_fast_path_agrees_on_the_six_truths(space):
    for name, truth in X.TRUTHS.items():
        assert np.array_equal(space.table(truth), space.interpret(truth)), name


def test_predicate_lattice_closes_under_or(space):
    """`near` is not deep -- it is one element of a finite lattice.

    This is what X47 turned on: the wall was the PRICE of a predicate, not
    the difficulty of building one.
    """
    sizes = [len(X.enumerate_preds(n)) for n in (5, 6, 7)]
    assert sizes[1] == sizes[2], f"lattice not closed: {sizes}"
    assert sizes[0] < sizes[1]
    tabs = {space.pred(p).tobytes() for p in X.enumerate_preds(6)}
    assert space.pred(X.NEAR).tobytes() in tabs


def test_the_six_truths_are_behaviourally_distinct(space):
    """If two targets share a behaviour, recovering one scores the other."""
    seen = {}
    for name, truth in X.TRUTHS.items():
        key = space.interpret(truth).tobytes()
        assert key not in seen, f"{name} is indistinguishable from {seen[key]}"
        seen[key] = name


def test_dead_is_absorbing(space):
    """Once dead, nothing else runs -- in both implementations."""
    for tail in X.ACTIONS:
        e = ("SEQ", "DIE", tail)
        tab = space.table(e)
        assert (tab == W.DEAD).all()
        assert np.array_equal(tab, space.interpret(e))


def test_training_filter_actually_fires(space):
    """The leak guard must be shown to work, not merely to exist.

    In the real run it discards 0 tasks out of 500 -- which is plausible in
    a space this large, and also means the guard is untested by that run. So
    test it here: given a forbidden set it certainly collides with, it must
    drop tasks rather than pass them through.
    """
    preds = X.enumerate_preds(2)
    ptabs = [space.pred(p) for p in preds]
    pool = list(W.depth1(space, preds, ptabs))
    rng = np.random.default_rng(2)
    forbidden = [t for _, t in pool[:400]]
    _, _, dropped = W.training_set(space, pool, preds, ptabs, forbidden, 60, rng)
    assert dropped > 0
    _, _, none_dropped = W.training_set(space, pool, preds, ptabs, [], 60,
                                        np.random.default_rng(2))
    assert none_dropped == 0
