"""Adapter over the existing typed logical forms and trusted executor.

X64H does not replace the semantics. It views X64E's flat `Z` as a small
typed TREE so that child order, omission and attachment are expressible --
those are the variations a convention has to be able to permute, and a flat
slot tuple cannot express them.

    Task  :  Op(FilterNode, ScopeNode)      root, two children
    Filter:  leaf atom
    Scope :  leaf atom, omissible
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

sys.path.insert(0, "experiments")

import x64e_semantics as E

Z = E.Z
ALL_Z = E.ALL_Z
UNIVERSE = E.UNIVERSE
HELD_OUT = E.HELD_OUT

FILTERS = E.FILTERS
SCOPES = E.SCOPES
POLARITIES = E.POLARITIES

ROOT, FILTER, SCOPE = "ROOT", "FILTER", "SCOPE"
CHILD_SLOTS = (FILTER, SCOPE)


@dataclass(frozen=True)
class Tree:
    """A typed semantic tree view of a logical form."""
    op: str
    filt: str
    scope: str

    def children(self) -> tuple[tuple[str, str], ...]:
        return ((FILTER, self.filt), (SCOPE, self.scope))

    def omissible(self) -> frozenset[str]:
        # only a scope that is the identity may be dropped without changing
        # the meaning; anything else would make omission a semantic change
        return frozenset({SCOPE}) if self.scope == "whole" else frozenset()


def to_tree(z) -> Tree:
    return Tree(z.op, z.filt, z.scope)


def from_tree(t: Tree):
    return Z(t.op, t.filt, t.scope)


def execute(z):
    """The trusted executor. Never re-implemented here."""
    return E.execute(z)


def denote(z):
    return E.denote(z)


# The X64H semantic subspace. A convention assigns a phrase to every atom it
# covers, so the forms in play are exactly those built from covered atoms.
# Declared here rather than discovered at lookup time: a KeyError deep in the
# grammar would be a silent restriction of the hypothesis space.
X64H_FILTERS = ("everything", "letters", "brackets", "hashes",
                "repeats in a row", "symbols seen before",
                "the first symbol", "the last symbol")
X64H_SCOPES = ("whole", "before hash", "after hash", "inside brackets",
               "outside brackets")


def live_forms():
    return [z for z in ALL_Z if not all(o == "" for o in denote(z))]


def x64h_forms():
    """Executable, non-degenerate, and inside the covered atom inventory."""
    return [z for z in live_forms()
            if z.filt in X64H_FILTERS and z.scope in X64H_SCOPES]


def behavioral_loglik(z, demonstrations, rho: float = 0.0, alphabet: int = 6):
    """Exact channel by default. The noisy channel is available but must be
    switched on explicitly, because a fast-evaluator disagreement is an
    evaluator bug until equivalence with the trusted path is shown."""
    import math
    f = execute(z)
    total = 0.0
    for x, y in demonstrations:
        hit = 1.0 if f(x) == y else 0.0
        if rho == 0.0:
            if hit == 0.0:
                return -math.inf
        else:
            p = (1 - rho) * hit + rho / alphabet
            if p <= 0:
                return -math.inf
            total += math.log(p)
    return total


def equivalence_classes(forms=None):
    """Probe-relative behavioural equivalence over the frozen universe. The
    name says `probe-relative` because it is not global program equivalence
    and the brief is explicit that conflating them is an error."""
    out: dict[tuple, list] = {}
    for z in (forms if forms is not None else live_forms()):
        out.setdefault(denote(z), []).append(z)
    return out
