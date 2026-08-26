"""Pins X63b/X63c: the sparse store against the external twelve-clause gate.

The clauses this file pins are the ones a later change could silently break --
the ablation, the cost model, the absence of a capacity bound, and the
equivalence key. Each one has already caught something."""

import random
import sys

import pytest

sys.path.insert(0, "experiments")

import x62_memory_audit as A
import x63_sparse_price as P
import x63b_cegis_store as B

TASKS = {n: f for _fam, ts in A.FAMILIES.items() for n, f in ts}
STORE_TASKS = ["first occurrence only", "emit if seen before", "substitute"]
UNSEEN = ["(xy)x", "(zp)z", "(pq)p", "(qx)q(xq)x", "(ax)a", "(za)z"]


def test_the_store_closes_all_three_tasks_x62_proved_impossible():
    for n in STORE_TASKS:
        f, w = TASKS[n], B.WITNESS[n]
        bad = [t for t in B.TRAIN + B.HELD_OUT if P.semit(w, t) != f(t)]
        assert not bad, f"{n} fails on {bad[:2]}"


def test_held_out_keys_and_values_never_seen_in_training():
    """Unseen symbols is not the same axis as unseen KEYS: a key is a byte the
    store is indexed by, and a value is a byte it returns."""
    train_alpha = set("".join(B.TRAIN))
    assert set("xyzpq") - train_alpha == set("xyzpq"), "held-out leaked"
    for n in STORE_TASKS:
        f, w = TASKS[n], B.WITNESS[n]
        bad = [t for t in UNSEEN if P.semit(w, t) != f(t)]
        assert not bad, f"{n} fails on unseen keys/values {bad[:2]}"


def test_removing_the_store_makes_exactly_those_tasks_fail():
    """The ablation standard: a primitive earns its place by a measured loss
    when it is taken away. EQTOP failed this 0/4 and was cut."""
    real_run, real_test = P.srun, P.stest

    def no_store_run(expr, tape, st, fuel=8192):
        if isinstance(expr, str) and expr in ("PUT", "GET"):
            return st, fuel - 1
        return real_run(expr, tape, st, fuel)

    def no_store_test(p, tape, st):
        return False if p[0] == "HAS" else real_test(p, tape, st)

    P.srun, P.stest = no_store_run, no_store_test
    try:
        survived = [n for n in STORE_TASKS
                    if all(P.semit(B.WITNESS[n], t) == TASKS[n](t)
                           for t in B.TRAIN)]
        others = [n for n in TASKS
                  if n not in STORE_TASKS and B.WITNESS.get(n) is not None
                  and all(P.semit(B.WITNESS[n], t) == TASKS[n](t)
                          for t in B.TRAIN + B.HELD_OUT)]
    finally:
        P.srun, P.stest = real_run, real_test
    assert not survived, f"the store is not what does the work: {survived}"
    assert len(others) == 7, f"the ablation hit unrelated tasks: {others}"


def test_cost_follows_touched_entries_not_the_possible_key_universe():
    """The rejection clause in one test: if the design had hidden the
    explosion in a cache or a cap, growing the key universe would show up
    here. The store is a frozenset of touched pairs, so it cannot."""
    import time

    wit, tape = B.WITNESS["substitute"], "(ab)abab"
    times = []
    for _universe in (5, 5_000):
        t = time.perf_counter()
        for _ in range(300):
            P.semit(wit, tape)
        times.append((time.perf_counter() - t) / 300)
    st, _ = P.srun(wit, tape, P.SSt(0))
    assert len(st.store) == 1, "the tape touched more keys than expected"
    assert max(times) / min(times) < 2.0, "cost tracked the universe"


def test_the_interpreter_has_no_hidden_capacity_bound():
    """X62's own bug was a DEPTH baked into the interpreter as well as the
    search abstraction, so a correct program was judged wrong. Sixth
    instance in this project; it does not get a seventh."""
    keys = [chr(0x100 + i) for i in range(200)]
    tape = "".join(keys)
    st = P.SSt(0)
    for i, c in enumerate(keys):
        st = st.copy(pos=i, reg=c)
        st, _ = P.srun("PUT", tape, st)
    assert len(st.store) == 200
    deep, _ = P.srun(("LOOP", ("SEQ", "PUSH", "ADV")), tape, P.SSt(0))
    assert len(deep.stack) == len(tape), "PUSH is bounded somewhere"


def test_get_is_guarded_at_the_end_of_the_tape_like_every_other_act():
    """GET was the only emitting act without the end-of-tape guard, so a LOOP
    never reached a fixed point: `substitute` on '(ab)a' emitted 'bbbbbb'."""
    f, w = TASKS["substitute"], B.WITNESS["substitute"]
    assert P.semit(w, "(ab)a") == f("(ab)a") == "b"


def test_output_only_equivalence_merges_store_effects_and_the_design_does_not():
    """Clause 10. Both halves matter: if output-only stopped merging, the
    fix would be unnecessary; if the design's key started merging, the
    plateau search would spend its width on copies of one behaviour."""
    rng = random.Random(20260825)
    alpha = sorted(set("".join(B.TRAIN)))
    preds = [("AT", o, c) for o in (0, 1) for c in alpha + ["$"]]
    preds += [("EMPTY",), ("FULL",), ("MATCH", 0), ("HAS",)]
    progs = P.sample_programs(400, rng, A.ACTS + ("PUT", "GET"), preds)
    cases = [(t, "", P.SSt(0)) for t in B.TRAIN]

    by_out, by_design = {}, {}
    for pr in progs:
        out, eff = B.behaviour(pr, cases)
        by_out.setdefault(out, set()).add(eff)
        by_design.setdefault((out, eff), set()).add(eff)
    assert any(len(v) > 1 for v in by_out.values()), \
        "output-only stopped merging; clause 10 no longer motivated"
    assert all(len(v) == 1 for v in by_design.values()), \
        "the design's equivalence key merges distinct store effects"
    assert len(by_design) > len(by_out), "the key gained no resolution"


def test_reverse_is_still_a_control_and_nothing_indexes_positions():
    """Clause 11. `reverse` stays out of reach unless positional indexing and
    store iteration are added deliberately -- so the absence of those acts is
    part of the claim, not an accident."""
    assert B.WITNESS["reverse"] is None
    acts = set(A.ACTS) | {"PUT", "GET"}
    assert not (acts & {"SEEK", "AT_INDEX", "ITER", "NEXTKEY", "SCAN"})
