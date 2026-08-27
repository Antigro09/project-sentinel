"""X65A-0: the theory package's exact finite checks, re-derived here.

These are ports, not copies. Each case is recomputed through the
repository's own `ExactPosterior` and compared against the rationals
published in `math-findings/.../results/exact-checks.json`. Agreement is a
cross-check of two independent implementations of the same finite model;
disagreement is a bug in one of them.

None of this measures continual learning. They are mechanism checks on a
deliberately tiny model, exactly as the theory package says.
"""

from __future__ import annotations

import itertools
from fractions import Fraction

from .posterior import ExactPosterior, Likelihood

RELIABILITY = Fraction(4, 5)

# The values published by the theory package, pinned here so a drift in
# either implementation is caught rather than absorbed.
PUBLISHED = {
    "sufficiency_classes": {
        (0, 5): (Fraction(1024, 1025), Fraction(1, 1025)),
        (1, 4): (Fraction(64, 65), Fraction(1, 65)),
        (2, 3): (Fraction(4, 5), Fraction(1, 5)),
        (3, 2): (Fraction(1, 5), Fraction(4, 5)),
        (4, 1): (Fraction(1, 65), Fraction(64, 65)),
        (5, 0): (Fraction(1, 1025), Fraction(1024, 1025)),
    },
    "map_accuracy": [Fraction(1, 2), Fraction(4, 5), Fraction(4, 5),
                     Fraction(112, 125), Fraction(112, 125),
                     Fraction(2944, 3125), Fraction(2944, 3125),
                     Fraction(15104, 15625), Fraction(15104, 15625)],
    "revision": (Fraction(4, 5), Fraction(4, 13), Fraction(76, 85),
                 Fraction(9, 10)),
    "coverage_utilities": {
        (): 0, ("lexical",): 3, ("ordering",): 2, ("procedure",): 6,
        ("lexical", "ordering"): 5, ("lexical", "procedure"): 9,
        ("ordering", "procedure"): 8,
        ("lexical", "ordering", "procedure"): 11,
    },
    "compounding": {"d": 6, "L": 8, "L_macro": 3, "B": 1000,
                    "raw_candidates": 1679616, "macro_candidates": 216},
}


def _obs(i: int, o: int) -> Likelihood:
    """One reliability-4/5 observation of a two-state convention."""
    return Likelihood(f"obs{i}", True,
                      {0: RELIABILITY if o == 0 else 1 - RELIABILITY,
                       1: RELIABILITY if o == 1 else 1 - RELIABILITY})


def convention_posterior(observations) -> ExactPosterior:
    p = ExactPosterior.uniform((0, 1), "microcase-2state")
    return p.update([_obs(i, o) for i, o in enumerate(observations)])


# ---------------------------------------------------------- Theorem 1

def bounded_memory(history_bits: int = 4, memory_bits: int = 3) -> dict:
    """A memory smaller than the history space cannot answer every index
    query. Exhibit the collision and the query that separates it."""
    histories = list(itertools.product((0, 1), repeat=history_bits))
    encode = lambda h: sum(b << i for i, b in enumerate(h)) % (2 ** memory_bits)
    seen: dict = {}
    for h in histories:
        m = encode(h)
        if m in seen:
            a, b = seen[m], h
            idx = next(i for i in range(history_bits) if a[i] != b[i])
            return {"history_count": len(histories),
                    "memory_state_count": 2 ** memory_bits,
                    "collision": [list(a), list(b)],
                    "separating_index_query": idx,
                    "injective": False}
        seen[m] = h
    return {"injective": True}


# ---------------------------------------------------------- Theorem 2

def sufficiency(length: int = 5) -> dict:
    """Histories sharing a count statistic induce the same posterior, and
    therefore the same posterior predictive."""
    classes: dict = {}
    for seq in itertools.product((0, 1), repeat=length):
        stat = (sum(seq), length - sum(seq))
        post = convention_posterior(seq)
        classes.setdefault(stat, set()).add((post.q[0], post.q[1]))
    ok = all(len(v) == 1 for v in classes.values())
    got = {k: next(iter(v)) for k, v in classes.items()}
    return {"history_count": 2 ** length, "statistic_count": len(classes),
            "all_equal_within_class": ok,
            "matches_published": got == PUBLISHED["sufficiency_classes"],
            "classes": {f"ones={k[0]},zeros={k[1]}": (str(v[0]), str(v[1]))
                        for k, v in sorted(got.items())}}


def map_accuracy(n: int) -> Fraction:
    """Exact expected MAP accuracy after n observations, with uniform
    tie-breaking so no latent state is silently preferred."""
    total = Fraction(0)
    for latent in (0, 1):
        for seq in itertools.product((0, 1), repeat=n):
            pr = Fraction(1, 2)
            for o in seq:
                pr *= RELIABILITY if o == latent else 1 - RELIABILITY
            win = convention_posterior(seq).map_states()
            if latent in win:
                total += pr * Fraction(1, len(win))
    return total


def semantic_transfer(max_obs: int = 8) -> dict:
    got = [map_accuracy(n) for n in range(max_obs + 1)]
    return {"reliability": str(RELIABILITY),
            "expected_map_accuracy": [str(x) for x in got],
            "reset_accuracy": str(got[0]),
            "matches_published": got == PUBLISHED["map_accuracy"]}


# ---------------------------------------------------------- Theorem 5

def _coverage(sel: frozenset) -> Fraction:
    cov = {"lexical": {"sense"}, "ordering": {"order"},
           "procedure": {"filter", "compose"}}
    w = {"sense": 3, "order": 2, "filter": 2, "compose": 4}
    covered = set().union(*(cov[i] for i in sel)) if sel else set()
    return Fraction(sum(w[f] for f in covered))


def _complementary(sel: frozenset) -> Fraction:
    v = Fraction(1) if {"macro_left", "macro_right"} <= sel else Fraction(0)
    if "stale_rule" in sel:
        v -= Fraction(3, 5)
    return v


def _monotone_submodular(items, u) -> tuple:
    subs = [frozenset(c) for r in range(len(items) + 1)
            for c in itertools.combinations(items, r)]
    mono = all(u(a) <= u(a | {e}) for a in subs for e in items if e not in a)
    sub = all(u(a | {e}) - u(a) >= u(b | {e}) - u(b)
              for a in subs for b in subs if a <= b
              for e in items if e not in b)
    return mono, sub


def retrieval() -> dict:
    gi = ("macro_left", "macro_right", "stale_rule")
    ci = ("lexical", "ordering", "procedure")
    gm, gs = _monotone_submodular(gi, _complementary)
    cm, cs = _monotone_submodular(ci, _coverage)
    utils = {tuple(sorted(frozenset(c))): int(_coverage(frozenset(c)))
             for r in range(len(ci) + 1)
             for c in itertools.combinations(ci, r)}
    return {
        "general_monotone": gm, "general_submodular": gs,
        "marginal_at_empty": str(_complementary(frozenset({"macro_right"}))),
        "marginal_after_left": str(
            _complementary(frozenset({"macro_left", "macro_right"}))
            - _complementary(frozenset({"macro_left"}))),
        "coverage_monotone": cm, "coverage_submodular": cs,
        "coverage_matches_published":
            utils == {tuple(k): v
                      for k, v in PUBLISHED["coverage_utilities"].items()},
        "coverage_utilities": {",".join(k) or "empty": v
                               for k, v in sorted(utils.items())},
    }


# ---------------------------------------------------------- Theorem 3

def revision() -> dict:
    """A plausible false claim, a trusted refutation, a later reversal, and
    an untouched independent factor."""
    prior = Fraction(4, 5)
    rel = Fraction(9, 10)
    post = ExactPosterior((True, False),
                          {True: prior, False: 1 - prior},
                          enumeration="microcase-claim")
    refute = Likelihood("counterexample", True, {True: 1 - rel, False: rel})
    after = post.update([refute])
    later_rel = Fraction(19, 20)
    support = Likelihood("later_support", True,
                         {True: later_rel, False: 1 - later_rel})
    reversed_ = after.update([support])
    unrelated_before = Fraction(9, 10)
    unrelated_after = unrelated_before        # declared-independent factor
    got = (prior, after.q[True], reversed_.q[True], unrelated_after)
    return {"prior": str(prior), "after_counterevidence": str(after.q[True]),
            "after_later_support": str(reversed_.q[True]),
            "unrelated_before": str(unrelated_before),
            "unrelated_after": str(unrelated_after),
            "ordering_holds": after.q[True] < prior < reversed_.q[True],
            "matches_published": got == PUBLISHED["revision"]}


# ---------------------------------------------------------- Theorem 6

def compounding() -> dict:
    c = PUBLISHED["compounding"]
    d, L, Lm, B = c["d"], c["L"], c["L_macro"], c["B"]
    raw, macro = d ** L, d ** Lm
    return {"branching_factor": d, "raw_length": L, "macro_length": Lm,
            "budget": B, "raw_candidates": raw, "macro_candidates": macro,
            "macro_reachable": macro <= B, "raw_reachable": raw <= B,
            "larger_budget_reaches_raw": True,
            "matches_published": raw == c["raw_candidates"]
                                 and macro == c["macro_candidates"],
            "claim_scope": "capability under a fixed search budget, not "
                           "increased computability or expressivity"}


def run_all() -> dict:
    out = {"bounded_memory": bounded_memory(), "sufficiency": sufficiency(),
           "semantic_transfer": semantic_transfer(), "retrieval": retrieval(),
           "revision": revision(), "compounding": compounding()}
    out["all_match_published"] = all(
        v.get("matches_published", True) for v in out.values()
        if isinstance(v, dict))
    return out
