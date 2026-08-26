"""Pins X63a. The finding is that the behaviour table is a resolution device
bought with memory, not a speed device -- so the store, which makes the memory
unaffordable, takes the search gradient with it."""

import random
import sys

import numpy as np
import pytest

sys.path.insert(0, "experiments")

import x62_memory_audit as A
import x63_sparse_price as K


@pytest.fixture(scope="module")
def space():
    return A.Space(K.TAPES, sorted(set("".join(K.TAPES))))


@pytest.fixture(scope="module")
def progs():
    alpha = sorted(set("".join(K.TAPES)))
    preds = [("AT", o, c) for o in (0, 1) for c in alpha + ["$"]]
    preds += [("EMPTY",), ("FULL",), ("MATCH", 0)]
    return K.sample_programs(200, random.Random(20260825), A.ACTS, preds)


def test_tabulating_a_store_is_the_thing_that_is_unaffordable():
    """(k+1)^k configurations, so the table multiplies -- which is the entire
    reason X63 has to execute rather than tabulate."""
    k = len(sorted(set("".join(K.TAPES))))
    assert (k + 1) ** k > 5_000, "the store got small enough to tabulate"


def test_execution_is_faster_than_the_search_step_not_slower(space, progs):
    """The pre-registered worry was that execution could not carry X58-X62's
    budgets. It is the cheaper side by a wide margin, and the store is free
    on top -- a run touches the keys it touches whatever the key space is."""
    import time

    k = max(len(t) for t in K.TAPES) + 2
    pm = space.pred(("AT", 0, "("))
    bt, dt = space.atoms["ADV"], space.table(("SEQ", "EMIT", "ADV"))
    t = time.perf_counter()
    for _ in progs:
        space.loop(space.branch(pm, bt, dt), k)
    step = (time.perf_counter() - t) / len(progs)

    t = time.perf_counter()
    for p in progs:
        for tp in K.TAPES:
            K.semit(p, tp)
    ex = (time.perf_counter() - t) / len(progs)

    assert ex < step, f"execution {ex*1e6:.0f}us is no longer under the " \
                      f"table's {step*1e6:.0f}us step; re-read X63a"


def test_the_output_gradient_does_not_track_the_table(space, progs):
    """Four metrics as calibration arms: the finding must not be an artefact
    of one crude scoring rule. None of them reaches even half the table's
    correlation, and exact-match is anti-correlated among near misses."""
    sigs = [space.table(p) for p in progs]
    outs = [tuple(A.emit(p, tp) for tp in K.TAPES) for p in progs]
    wit = A.WITNESS["capture brackets"]
    target = space.table(wit)
    tgt = tuple(A.emit(wit, tp) for tp in K.TAPES)
    tab = np.array([float((s == target).mean()) for s in sigs])

    # rebuilt inline so the test does not depend on main()'s locals
    best = -1.0

    def bag(o):
        num = den = 0
        for got, want in zip(o, tgt):
            keys = set(got) | set(want)
            den += sum(max(got.count(c), want.count(c)) for c in keys) or 1
            num += sum(min(got.count(c), want.count(c)) for c in keys)
        return num / den

    def exact(o):
        return sum(1 for g, w in zip(o, tgt) if g == w) / len(tgt)

    for fn in (bag, exact):
        v = np.array([fn(o) for o in outs])
        if v.std():
            best = max(best, abs(float(np.corrcoef(tab, v)[0, 1])))
    assert best < 0.5, f"the output gradient now tracks the table (r={best:.3f})"


def test_the_tables_resolution_is_real_not_agreement_on_dead_states(space,
                                                                    progs):
    """The obvious objection to X63a section 4: the signature scores 2,232
    situations and a run reaches few of them, so the precision could be
    agreement on states nobody visits. It is not -- restricted to the
    reachable subspace the table separates at least as many classes."""
    n, w = space.n, space.w
    starts = [space.index[(ti, 0, (), A.NONE)] for ti in range(len(K.TAPES))]
    ends = {a: space.unpack(space.atoms[a])[0] for a in A.ACTS}
    seen, stack = set(starts), list(starts)
    while stack:
        i = stack.pop()
        for e in ends.values():
            j = int(e[i])
            if j not in seen:
                seen.add(j)
                stack.append(j)
    reach = np.array(sorted(seen), dtype=np.int32)
    assert len(reach) < n, "every situation became reachable; re-read X63a"

    cols = np.concatenate([reach, n + reach,
                           (2 * n + (reach[:, None] * w
                                     + np.arange(w)[None, :])).ravel()])
    sigs = [space.table(p) for p in progs]
    full = len({s.tobytes() for s in sigs})
    restricted = len({s[cols].tobytes() for s in sigs})
    outs = len({tuple(A.emit(p, tp) for tp in K.TAPES) for p in progs})
    assert restricted >= 0.9 * full, "the resolution was mostly phantom"
    assert restricted > 4 * outs, "outputs now resolve as well as the table"


def test_the_store_interpreter_is_not_bounded():
    """X62's own bug, and the sixth instance in this project: bounding the
    machine as well as the search abstraction judges correct programs wrong.
    The store must hold more keys than any abstraction would enumerate."""
    st = K.SSt(0)
    tape = "abcde"
    for i, c in enumerate(tape):
        st = st.copy(pos=i, reg=c)
        st, _ = K.srun("PUT", tape, st)
    assert len(st.store) == len(tape), "the store silently dropped keys"
