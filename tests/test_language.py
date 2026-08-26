"""Pins X64B-2. The lexicon is authored, so the properties worth pinning are
the ones that could silently rot: that words carry constraints rather than
task identities, that the semantics composes, and that language never
excludes the target it is meant to describe."""

import random
import sys

import pytest

sys.path.insert(0, "experiments")

import x63_sparse_price as P
import x64a_identify as X
import x64b1_openworld as O
import x64b2_language as L


@pytest.fixture(scope="module")
def pool():
    return O.build(3)


def test_language_never_excludes_a_target_that_is_in_the_pool(pool):
    """The failure that makes everything else meaningless. Two real lexicon
    bugs were caught by exactly this check."""
    bad = []
    for n, (canon, paras) in L.INTENTS.items():
        f = L.TASKS[n]
        tb = tuple(f(t) for t in L.UNIVERSE)
        if tb not in pool:
            continue
        for instr in [canon] + paras:
            if tb not in L.narrow(pool, L.meaning(instr)[0]):
                bad.append((n, instr))
    assert not bad, f"language excludes its own target: {bad[:3]}"


def test_no_word_encodes_a_task_identity(pool):
    """If a single word narrowed the pool to one behaviour, the lexicon
    would be a lookup table with extra steps."""
    for w, preds in L.LEXICON.items():
        if not preds:
            continue
        n = len(L.narrow(pool, preds))
        assert n > 1, f"the word {w!r} alone selects a single behaviour"


def test_the_semantics_composes_over_words_not_phrases():
    """Meaning is a union over words, so an unseen word order and unseen
    combination still has one. Word order must not matter."""
    a, _u = L.meaning("remove repeats in a row")
    b, _u2 = L.meaning("in a row remove repeats")
    assert a == b
    one, _ = L.meaning("remove repeats")
    more, _ = L.meaning("remove repeats in a row")
    assert one < more, "adding a word did not add a constraint"


def test_adding_a_word_narrows_and_the_ambiguity_is_real(pool):
    """`remove repeats` is the ambiguity class the review named: it could be
    adjacent deduplication or first-occurrence filtering, and both targets
    have to survive it."""
    vague = L.narrow(pool, L.meaning("remove repeats")[0])
    sharp = L.narrow(pool, L.meaning("remove repeats in a row")[0])
    assert len(sharp) < len(vague)
    for n in ("dedupe adjacent", "first occurrence only"):
        f = L.TASKS[n]
        assert tuple(f(t) for t in L.UNIVERSE) in vague, \
            f"{n} does not survive the ambiguous instruction"


def test_a_mis_mapped_word_makes_the_paraphrase_gate_fail(pool):
    """The calibration for L3. A paraphrase check that cannot fail measures
    nothing, so break one entry and require that it does."""
    f = L.TASKS["dedupe adjacent"]
    demos = {t: f(t) for t in L.FEW}
    base = L.solve_lang(L.INTENTS["dedupe adjacent"][0], f, random.Random(5),
                        demos=demos)
    saved = L.LEXICON["adjacent"]
    L.LEXICON["adjacent"] = {"a prefix"}
    try:
        broke = L.solve_lang("drop adjacent duplicates", f, random.Random(5),
                             demos=demos)
    finally:
        L.LEXICON["adjacent"] = saved
    bb = X.behaviour(base["rep"]) if base["rep"] is not None else None
    rb = X.behaviour(broke["rep"]) if broke["rep"] is not None else None
    assert bb != rb, "a mis-mapped word changes nothing; L3 is vacuous"


def test_conflicting_instruction_and_demonstrations_are_never_forced():
    """Contradiction surfaces two ways and both are acceptable: as an empty
    intersection, reported directly as `conflict`; or as an intersection that
    is non-empty but wrong, caught later by confirmation and ending in
    none-of-the-above. What must never happen is a confident answer."""
    names = list(L.TASKS)
    verdicts, forced = [], []
    for i, n in enumerate(names):
        f = L.TASKS[n]
        other = L.TASKS[names[(i + 1) % len(names)]]
        r = L.solve_lang(L.INTENTS[n][0], f, random.Random(5),
                         demos={t: other(t) for t in L.EVIDENCE0})
        verdicts.append(r["verdict"])
        if r["verdict"] == "answered":
            forced.append(n)
    assert not forced, f"a wrong program was forced for {forced}"
    assert "conflict" in verdicts, "the direct conflict path never fired"


def test_the_solver_only_reaches_the_target_as_a_callable():
    """L8, measured. Every input the target is asked about must be one the
    solver was entitled to ask about, and never a held-out tape."""
    legal = set(L.UNIVERSE) | set(L.CHALLENGE)
    calls = []
    f = L.TASKS["capture brackets"]

    def watched(t):
        calls.append(t)
        return f(t)

    L.solve_lang(L.INTENTS["capture brackets"][0], watched, random.Random(5),
                 demos={t: watched(t) for t in L.FEW})
    assert set(calls) <= legal
    assert not (set(calls) & set(L.HELD_OUT))
