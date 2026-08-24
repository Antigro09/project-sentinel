"""Nested structure needs memory, and Occam has to be applied at the end.

Two findings from X49 that are easy to regress. The first is that the wall is
real: a window-only program cannot copy what lies inside brackets, and the
certificate for that is computable rather than arguable. The second is that
greedy rule construction will happily recover a 60-node ENUMERATION of the
(character, depth) pairs it happened to observe where a one-node DEEP
belonged -- exact on the evidence, wrong on the next tape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import x49_nested_structure as X  # noqa: E402


@pytest.fixture(scope="module")
def space():
    return X.Space(X.TAPES)


def test_window_certificate_exists_for_nested_brackets():
    """Same window, opposite decisions: the impossibility, computed."""
    win, byd = X.window_certificate(X.TAPES, X.TRUTHS["copy inside []"])
    assert win is not None
    assert set(byd) == {True, False}


def test_no_window_certificate_for_the_flat_tasks():
    """The certificate must not fire on everything, or it means nothing."""
    for name in ("copy digits", "strip brackets"):
        win, _ = X.window_certificate(X.TAPES, X.TRUTHS[name])
        assert win is None, f"{name} unexpectedly has colliding windows"


def test_fast_path_equals_interpreter(space):
    for name, truth in X.TRUTHS.items():
        assert np.array_equal(space.table(truth), space.interpret(truth)), name
    rng = np.random.default_rng(3)
    preds = [("AT", 0, c, None) for c in "[]a0"] + [("DEEP",)]
    for _ in range(30):
        p = preds[int(rng.integers(0, len(preds)))]
        a, b = (X.ACTIONS[int(rng.integers(0, len(X.ACTIONS)))] for _ in range(2))
        e = ("LOOP", ("IF", p, ("SEQ", a, "ADV"), ("SEQ", b, "ADV")))
        assert np.array_equal(space.table(e), space.interpret(e))


def test_char_depth_product_is_a_partition(space):
    """The premise the conjunction-derivation rests on."""
    chars = sorted({c for t in X.TAPES for c in t})
    fam = [space.pred(("AT", 0, c, d))
           for c in chars + ["$"] for d in range(space.depths)]
    assert (np.stack(fam).sum(0) == 1).all()


def test_counter_is_load_bearing(space):
    """DEEP cannot be dropped from the recovered program without breaking it."""
    truth = X.TRUTHS["copy inside []"]
    crippled = ("LOOP", ("IF", X.OPEN, ("SEQ", "ADV", "INC"),
                         ("IF", X.CLOSE, ("SEQ", "ADV", "DEC"),
                          ("SEQ", "EMIT", "ADV"))))
    assert any(X.output(crippled, t) != X.output(truth, t) for t in X.EVAL)


def test_simplification_replaces_an_enumeration_with_deep(space):
    """The regression test for the pass that made the program generalise."""
    truth = X.TRUTHS["copy inside []"]
    target = space.interpret(truth)
    chars = sorted({c for t in X.TAPES for c in t})
    fams = X.families(space, chars, (0,))
    derived = [d[1] for d in X.derive(space, target, fams) if d[0].endswith("xdepth")]
    assert derived, "no product-partition predicate was derived"
    enumerated = max(derived, key=X.size_of)
    assert X.size_of(enumerated) > 10

    bloated = ("IF", X.OPEN, ("SEQ", "ADV", "INC"),
               ("IF", X.CLOSE, ("SEQ", "ADV", "DEC"),
                ("IF", enumerated, ("SEQ", "EMIT", "ADV"), "ADV")))
    wrap = lambda t: space.loop(t, 14)
    assert np.array_equal(wrap(space.table(bloated)), target), "premise broken"

    lean = X.simplify_rules(space, bloated, [("DEEP",), X.OPEN, X.CLOSE],
                            target, wrap)
    assert X.size_of(lean) < X.size_of(bloated)
    assert np.array_equal(wrap(space.table(lean)), target)
    # and the simplified one must survive inputs the bloated one fails
    assert any(X.output(("LOOP", bloated), t) != X.output(truth, t) for t in X.EVAL)
    assert all(X.output(("LOOP", lean), t) == X.output(truth, t) for t in X.EVAL)


def test_probe_uses_bytes_the_evidence_never_showed():
    """You cannot discover an enumeration by testing what you already saw."""
    chars = sorted({c for t in X.TAPES for c in t})
    assert set(X.UNSEEN).isdisjoint(chars)
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(40):
        tape = "".join(rng.choice(list(chars) + X.UNSEEN, 12))
        seen |= set(tape)
    assert seen & set(X.UNSEEN), "probe never reaches an unobserved byte"
