"""X65A-0: the exact finite posterior, in rationals, with evidence
deduplication as a structural property rather than a convention.

Everything is `Fraction`. No logs, no floats, no tolerance: two posteriors
are equal or they are not. The pilot is small enough that this is the
cheaper choice as well as the honest one.

EVIDENCE MAY BE COUNTED ONCE. Only base external observations carry a
likelihood factor. Semantic claims, procedures, negative entries and
consolidated summaries reference the evidence they came from and contribute
nothing further, so retrieving five descendants of one episode cannot raise
that episode to the fifth power. This is enforced by construction --
`Likelihood.base` and the absorbed-id set -- not by remembering to be
careful at each call site.

A zero normalizer is an OPEN-WORLD STATE, never a silent renormalization
onto whichever in-class candidate happened to survive.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from .types import OpenWorld, TaintError, content_id


@dataclass(frozen=True)
class Likelihood:
    """One factor. `base` marks an external observation; anything derived
    from one carries the same evidence_id and contributes nothing."""
    evidence_id: str
    base: bool
    values: dict            # state -> Fraction
    label: str = ""

    def __post_init__(self):
        for v in self.values.values():
            if not isinstance(v, Fraction):
                raise TaintError("likelihoods must be exact Fractions")
            if v < 0:
                raise TaintError("negative likelihood")


def deterministic_summary(of: Likelihood, label: str = "summary") -> Likelihood:
    """A summary of evidence is not evidence. Same id, base=False."""
    return Likelihood(of.evidence_id, False, dict(of.values), label)


@dataclass(frozen=True)
class ExactPosterior:
    states: tuple
    q: dict                 # state -> Fraction, sums to exactly 1
    absorbed: frozenset = frozenset()
    status: OpenWorld = OpenWorld.RELEVANT
    enumeration: str = ""

    @staticmethod
    def uniform(states, enumeration: str = "") -> "ExactPosterior":
        states = tuple(states)
        n = len(states)
        if n == 0:
            raise TaintError("empty latent enumeration")
        return ExactPosterior(states, {s: Fraction(1, n) for s in states},
                              enumeration=enumeration
                              or content_id("enum", [str(s) for s in states]))

    def check(self) -> None:
        if sum(self.q.values(), Fraction(0)) != 1:
            raise TaintError("posterior is not exactly normalized")

    def update(self, likelihoods: Iterable[Likelihood]) -> "ExactPosterior":
        w = dict(self.q)
        absorbed = set(self.absorbed)
        used = []
        for f in likelihoods:
            if not f.base:
                continue                      # derived: references, never counts
            if f.evidence_id in absorbed:
                continue                      # already absorbed: counted once
            absorbed.add(f.evidence_id)
            used.append(f.evidence_id)
            for s in self.states:
                w[s] = w[s] * f.values.get(s, Fraction(0))
        z = sum(w.values(), Fraction(0))
        if z == 0:
            # No in-class candidate survives. Report it; do not renormalize.
            return ExactPosterior(self.states, dict(self.q), self.absorbed,
                                  OpenWorld.MISSING_REPRESENTATION,
                                  self.enumeration)
        out = ExactPosterior(self.states, {s: w[s] / z for s in self.states},
                             frozenset(absorbed), OpenWorld.RELEVANT,
                             self.enumeration)
        out.check()
        return out

    def predictive(self, channel: dict) -> dict:
        """p(tau | H) = sum over lambda of p(tau | lambda) Q(lambda).
        Theorem 2: this depends on the history only through Q."""
        out: dict = {}
        for s in self.states:
            for outcome, p in channel[s].items():
                out[outcome] = out.get(outcome, Fraction(0)) + self.q[s] * p
        return out

    def map_states(self) -> list:
        best = max(self.q.values())
        return [s for s in self.states if self.q[s] == best]

    def canon(self):
        return {"states": [str(s) for s in self.states],
                "q": {str(s): self.q[s] for s in self.states},
                "absorbed": sorted(self.absorbed),
                "status": self.status.value, "enumeration": self.enumeration}


# ----------------------------------------------- the three dedup invariants

def invariant_summary_is_free(post: ExactPosterior, ev: Likelihood) -> bool:
    """posterior(E) == posterior(E + deterministic_summary(E))"""
    a = post.update([ev])
    b = post.update([ev, deterministic_summary(ev)])
    return a.q == b.q and a.absorbed == b.absorbed


def invariant_descendants_do_not_multiply(post: ExactPosterior,
                                          ev: Likelihood, n: int = 5) -> bool:
    """Retrieving many descendants of one evidence item does not multiply
    its contribution."""
    a = post.update([ev])
    kids = [deterministic_summary(ev, f"child{i}") for i in range(n)]
    b = post.update([ev] + kids)
    return a.q == b.q


def invariant_consolidation_preserves_predictive(post: ExactPosterior,
                                                 evs: list, channel: dict
                                                 ) -> bool:
    """Consolidating evidence into a semantic/procedural node preserves the
    posterior predictive exactly."""
    a = post.update(evs)
    consolidated = [evs[0]] + [deterministic_summary(e, "consolidated")
                               for e in evs]
    b = post.update(consolidated + evs[1:])
    return a.predictive(channel) == b.predictive(channel)
