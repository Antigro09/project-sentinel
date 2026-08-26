"""Pins X64D. The properties worth pinning are the structural guarantees --
that language cannot delete a candidate and cannot commit without evidence
-- plus the split discipline that makes the numbers mean anything."""

import random
import sys

import pytest

sys.path.insert(0, "experiments")

import x64a_identify as X
import x64d_senses as D


@pytest.fixture(scope="module")
def senses():
    return D.induce(D.group(D.DEV_PAIRS), variants=(0, 1))


def test_the_splits_share_no_composition():
    """A test item must be an unseen COMBINATION, and every scope and filter
    must still appear in development or the test measures vocabulary
    coverage instead of composition."""
    assert not (D.DEV_PAIRS & D.TEST_PAIRS)
    assert not (D.VAL_PAIRS & D.TEST_PAIRS)
    dev_s = {s for s, _f in D.DEV_PAIRS}
    dev_f = {f for _s, f in D.DEV_PAIRS}
    assert {s for s, _f in D.TEST_PAIRS} <= dev_s
    assert {f for _s, f in D.TEST_PAIRS} <= dev_f


def test_variant_two_words_never_appear_in_development(senses):
    """The language holdout is only a holdout if its surface words are
    genuinely absent from induction."""
    unseen = {t for c, _f, _b in D.group(D.TEST_PAIRS)
              for t in D.realise(c, 2) if t not in senses}
    assert len(unseen) >= 10, f"only {len(unseen)} unseen tokens"


def test_language_can_never_delete_the_last_candidate(senses):
    """D5, as a property of the definition rather than an observation. The
    empty reading is always available, so the target survives whatever the
    senses say."""
    test = D.group(D.TEST_PAIRS)[:8]
    for c, _f, b in test:
        for v in (0, 1, 2):
            r = D.solve(c, v, senses, mode="joint", budget=0)
            assert r["retained"], f"{c} v{v}: language removed the target"


def test_a_wrong_sense_costs_specificity_not_the_target(senses):
    """The same claim, attacked directly: corrupt every sense and the target
    must still be retained."""
    bad = {t: frozenset({frozenset(D.PI_NAMES[:4])}) for t in senses}
    for c, _f, _b in D.group(D.TEST_PAIRS)[:6]:
        r = D.solve(c, 0, bad, mode="joint", budget=0)
        assert r["retained"], f"{c}: a corrupt sense removed the target"


def test_no_answer_is_given_while_rivals_remain(senses):
    """Committing on the language-preferred tier produced four confident
    errors on the test split. Answering waits for the evidence."""
    wrong = 0
    for c, f, _b in D.group(D.TEST_PAIRS):
        r = D.solve(c, 0, senses, mode="joint")
        if r["verdict"] == "answered" and D.held(r, f) != 10:
            wrong += 1
    assert wrong == 0, f"{wrong} confident errors"


def test_the_hard_filter_still_loses_targets(senses):
    """X64C's design, kept as the baseline. If this ever stops failing, the
    comparison that motivates the whole experiment has gone away."""
    lost = 0
    for c, _f, _b in D.group(D.TEST_PAIRS):
        for v in (0, 1, 2):
            if not D.solve(c, v, senses, mode="hard", budget=0)["retained"]:
                lost += 1
    assert lost > 0, "the hard filter no longer excludes any target"


def test_polysemy_is_induced_not_authored(senses):
    """Nothing in the code assigns a sense to a word. Different roles get
    different senses because the examples differ."""
    poly = {}
    for (w, r), ss in senses.items():
        poly.setdefault(w, {})[r] = ss
    differ = [w for w, rs in poly.items()
              if len(rs) > 1 and len({frozenset(v) for v in rs.values()}) > 1]
    assert len(differ) >= 2, f"only {differ} differ by role"


def test_conflict_detection_is_not_claimed(senses):
    """D7's negative result, pinned. Precision at chance is the finding; if
    a future change makes this pass, the claim can be upgraded -- but it
    must not be asserted while the measurement says otherwise."""
    test = D.group(D.TEST_PAIRS)
    tp = sum(1 for i, (c, _f, _b) in enumerate(test)
             if (D.conflict_gap(c, test[(i + 1) % len(test)][1], senses)
                 or 0) >= 1)
    fp = sum(1 for c, f, _b in test
             if (D.conflict_gap(c, f, senses) or 0) >= 1)
    prec = tp / max(1, tp + fp)
    assert prec < 0.7, f"conflict precision reached {prec:.2f}; re-read D7"
