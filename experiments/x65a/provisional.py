"""X65A-S2: confirmed state, provisional branches, and active criticism.

X65A-S1 failed S1.7 because the quarantine rule was

    quarantine only if the event contradicts EVERY surviving convention

which admits an alien observation whenever it happens to agree with a
surviving but false convention. That is not a tuning problem. Section 0
below shows that no policy acting on (posterior, event) alone can be correct
in both worlds, so the fix cannot be a better one-shot rule -- it has to be
an extra observation.

    ConfirmedState        what the system is willing to act on with no
                          questions asked
    ProvisionalBranch     evidence that would change ConfirmedState, held
                          apart until a challenge resolves it

A provisional branch may choose which question to ask. It may not write to
ConfirmedState and it may not authorise a zero-query action.

THE CHALLENGE. A legal question names a meaning and receives the utterance
the partner would use for it. That is a semantic question in X64H's sense,
introduced here rather than inherited: X64H's behavioural probes ask about
task outputs and cannot separate two conventions that agree on the current
task.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product

import numpy as np

from .semantic_mem import GroundedObservation, surviving_mask
from .types import Status

# Frozen before validation. Rational, like every other posterior in this
# line: the cause model is exact and the canonical writer refuses floats.
PRIOR_OUT = Fraction(1, 10)
THETA_PROMOTE = Fraction(9, 10)
THETA_REJECT = Fraction(9, 10)
CHALLENGE_BUDGET = 3

PROMOTE = "PROMOTE_IN_FAMILY"
REJECT = "REJECT_OUT_OF_FAMILY"
UNRESOLVED = "UNRESOLVED"
MISSING = "MISSING_REPRESENTATION"


# ---------------------------------------------------- 0. the impossibility

def impossibility_microcase(fam) -> dict:
    """Two worlds the learner cannot tell apart from (q, e) alone.

    World A: the partner is phi_A, the event is out-of-family, and it is
             inconsistent with phi_A but consistent with phi_B.
    World B: the partner is phi_B and the same event is valid evidence.

    The learner sees the same confirmed posterior q -- uniform over
    {phi_A, phi_B} -- and the same event in both. Correct behaviour differs.
    """
    found = None
    for a in range(0, fam.n, 7):
        for b in range(a + 1, min(a + 60, fam.n)):
            z = next((z for z in range(fam.m)
                      if fam.u3[a, z] != fam.u3[b, z]), None)
            if z is None:
                continue
            # the event: the utterance phi_B would use for meaning z
            e = (z, int(fam.u3[b, z]))
            # a confirmed record that leaves exactly {a, b} alive
            g = _record_pinning(fam, a, b)
            if g is None:
                continue
            m = surviving_mask(fam, g)
            if int(m.sum()) != 2:
                continue
            found = {"phi_A": a, "phi_B": b, "meaning": z, "event": e,
                     "grounded": g}
            break
        if found:
            break
    if not found:
        return {"constructed": False}

    # both worlds present the learner with the identical pair (q, e)
    q = surviving_mask(fam, found["grounded"]).astype(float)
    q /= q.sum()
    obs = {"posterior_support": sorted(np.where(q > 0)[0].tolist()),
           "event": list(found["event"])}
    correct = {"world_A": REJECT, "world_B": PROMOTE}

    # deterministic policies on (q, e): a policy is one action, applied to
    # both worlds, so it is wrong in exactly one of them
    det = {act: sum(1 for w in correct.values() if w != act)
           for act in (PROMOTE, REJECT, UNRESOLVED)}
    # randomised: accept with probability p; errors are p and 1 - p
    rand = [{"p_accept": Fraction(k, 4),
             "error_world_A": Fraction(k, 4),
             "error_world_B": Fraction(4 - k, 4),
             "total": Fraction(k, 4) + Fraction(4 - k, 4)}
            for k in range(5)]
    return {
        "constructed": True, **{k: v for k, v in found.items()
                                if k != "grounded"},
        "identical_observation": obs, "correct_action": correct,
        "deterministic_policy_errors": det,
        "min_deterministic_errors": min(det.values()),
        "randomised_total_error_always_one": all(r["total"] == 1
                                                 for r in rand),
        "randomised_grid": [{k: str(v) for k, v in r.items()} for r in rand],
        "conclusion": "no policy on (q, e) alone is correct in both worlds; "
                      "resolution requires a further observation",
    }


def _record_pinning(fam, a: int, b: int, tries: int = 24):
    """Grounded observations true of BOTH a and b, leaving exactly them."""
    zs = [z for z in range(fam.m) if fam.u3[a, z] == fam.u3[b, z]]
    g = tuple(GroundedObservation(z, int(fam.u3[a, z]), f"pin{z}")
              for z in zs[:tries])
    if not g:
        return None
    m = surviving_mask(fam, g)
    return g if int(m.sum()) >= 2 else None


# ------------------------------------------------------- 1. the two tiers

@dataclass(frozen=True)
class ConfirmedState:
    identity: str
    grounded: tuple = ()
    version: int = 1
    evidence_ids: frozenset = frozenset()
    provenance: tuple = ()

    def canon(self):
        return {"identity": self.identity,
                "grounded": [g.canon() for g in self.grounded],
                "version": self.version,
                "evidence_ids": sorted(self.evidence_ids)}


@dataclass
class ProvisionalBranch:
    trigger: GroundedObservation
    provisional_grounded: tuple
    cause_posterior: dict
    affected_confirmed_mass: Fraction
    status: str = "OPEN"
    unresolved_questions: tuple = ()
    created_at: int = 0
    queries_used: int = 0
    budget: int = CHALLENGE_BUDGET
    answers: tuple = ()

    def canon(self):
        return {"trigger": self.trigger.canon(),
                "provisional_grounded": [g.canon()
                                         for g in self.provisional_grounded],
                "cause_posterior": dict(self.cause_posterior),
                "affected_confirmed_mass": self.affected_confirmed_mass,
                "status": self.status, "created_at": self.created_at,
                "queries_used": self.queries_used, "budget": self.budget,
                "answers": [list(a) for a in self.answers]}


# ------------------------------------------------------ 2. the cause model

def other_likelihood(fam, event) -> Fraction:
    """p(e | phi, OUT_OF_FAMILY): the marginal probability that an unknown
    speaker drawn from the frozen family produces this utterance for this
    meaning. Independent of the partner's phi, fixed by the family and not
    by the event, so it cannot be tuned into a sink for hard observations."""
    z, u = event
    return Fraction(int((fam.u3[:, z] == u).sum()), fam.n)


def cause_posterior(fam, confirmed_mask, event, answers=(),
                    prior_out: Fraction = PRIOR_OUT,
                    with_other: bool = True) -> dict:
    """p(C | e, A, H) with phi marginalised exactly over the confirmed
    posterior. Answers are folded in: they constrain phi under BOTH causes,
    because the partner answers truthfully whoever produced the event."""
    m = confirmed_mask.copy()
    for zq, a in answers:
        m &= (fam.u3[:, zq] == a)
    n_answer = int(m.sum())
    zero, one = Fraction(0), Fraction(1)
    if n_answer == 0:
        return {"IN_FAMILY": zero, "OUT_OF_FAMILY": zero,
                "MISSING_REPRESENTATION": one}
    z, u = event
    n_both = int((m & (fam.u3[:, z] == u)).sum())
    w_in = (one - prior_out) * Fraction(n_both, n_answer)
    w_out = (prior_out * other_likelihood(fam, event)) if with_other else zero
    tot = w_in + w_out
    if tot <= 0:
        return {"IN_FAMILY": zero, "OUT_OF_FAMILY": one,
                "MISSING_REPRESENTATION": zero}
    return {"IN_FAMILY": w_in / tot, "OUT_OF_FAMILY": w_out / tot,
            "MISSING_REPRESENTATION": zero}


# ------------------------------------------------- 3. model-criticism query

def _entropy(counts) -> float:
    tot = sum(counts)
    if tot <= 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counts if c > 0)


def challenge_gain(fam, confirmed_mask, event, zq, answers=(),
                   prior_out: Fraction = PRIOR_OUT) -> float:
    """I((C, Phi); A_q | H, e). The answer channel is deterministic given
    phi, so the mutual information is the entropy of the answer under the
    joint over causes -- which is exactly what a question that separates the
    provisional branch from the rest of the confirmed set maximises."""
    m = confirmed_mask.copy()
    for z0, a0 in answers:
        m &= (fam.u3[:, z0] == a0)
    if not m.any():
        return 0.0
    z, u = event
    cons = m & (fam.u3[:, z] == u)
    cp = cause_posterior(fam, confirmed_mask, event, answers, prior_out)
    w = np.zeros(fam.n)
    if cons.any():
        w[cons] += float(cp["IN_FAMILY"]) / int(cons.sum())
    if m.any():
        w[m] += float(cp["OUT_OF_FAMILY"]) / int(m.sum())
    if w.sum() <= 0:
        return 0.0
    w /= w.sum()
    buckets: dict = {}
    codes = fam.u3[:, zq]
    for i in np.where(w > 0)[0]:
        buckets[int(codes[i])] = buckets.get(int(codes[i]), 0.0) + w[i]
    return _entropy(list(buckets.values()))


def choose_challenge(fam, confirmed_mask, event, policy, legal, answers,
                     rng) -> int | None:
    asked = {z for z, _a in answers}
    cand = [z for z in legal if z not in asked]
    if not cand:
        return None
    if policy == "random":
        return cand[rng.randrange(len(cand))]
    if policy == "in_family_disagreement":
        # maximum disagreement among in-family conventions only: ignores the
        # OUT hypothesis entirely, which is the comparison that matters
        m = confirmed_mask.copy()
        for z0, a0 in answers:
            m &= (fam.u3[:, z0] == a0)
        z, u = event
        m &= (fam.u3[:, z] == u)
        if not m.any():
            return cand[rng.randrange(len(cand))]
        best, score = None, -1.0
        for zq in cand:
            h = _entropy(list(np.bincount(fam.u3[m, zq]).tolist()))
            if h > score:
                best, score = zq, h
        return best
    best, score = None, -1.0
    for zq in cand:
        g = challenge_gain(fam, confirmed_mask, event, zq, answers)
        if g > score:
            best, score = zq, g
    return best


# ------------------------------------------------------ 4. promotion rule

def resolve(fam, confirmed: ConfirmedState, event, phi_true: int, arm: str,
            legal, rng, budget: int = CHALLENGE_BUDGET,
            prior_out: Fraction = PRIOR_OUT,
            oracle_cause: str | None = None):
    """Returns (outcome, new_confirmed, branch, queries_used)."""
    mask = surviving_mask(fam, confirmed.grounded)
    z, u = event
    cons = mask & (fam.u3[:, z] == u)
    affected = Fraction(1) - Fraction(int(cons.sum()),
                                      max(1, int(mask.sum())))
    obs = GroundedObservation(z, u, f"ev:{z}:{u}")

    def accept():
        return ConfirmedState(confirmed.identity,
                              confirmed.grounded + (obs,),
                              confirmed.version + 1,
                              confirmed.evidence_ids | {obs.base_evidence},
                              confirmed.provenance)

    # ---- rules that do not use a provisional branch at all
    if arm == "always_accept":
        return PROMOTE, accept(), None, 0
    if arm == "always_quarantine":
        return REJECT, confirmed, None, 0
    if arm == "old_quarantine":
        return ((REJECT, confirmed, None, 0) if not cons.any()
                else (PROMOTE, accept(), None, 0))
    if arm == "survivor_majority":
        ok = int(cons.sum()) * 2 > int(mask.sum())
        return ((PROMOTE, accept(), None, 0) if ok
                else (REJECT, confirmed, None, 0))
    if arm == "map_protection":
        keep = int(np.where(mask)[0][0])
        ok = bool(cons[keep])
        return ((PROMOTE, accept(), None, 0) if ok
                else (REJECT, confirmed, None, 0))
    if arm == "oracle_cause":
        good = (oracle_cause == "IN_FAMILY")
        return ((PROMOTE, accept(), None, 0) if good
                else (REJECT, confirmed, None, 0))

    cp = cause_posterior(fam, mask, event, (), prior_out,
                         with_other=(arm != "no_other"))
    branch = ProvisionalBranch(obs, confirmed.grounded + (obs,), cp, affected)

    if arm == "cause_mixture_no_query":
        if cp["MISSING_REPRESENTATION"] == 1:
            branch.status = MISSING
            return MISSING, confirmed, branch, 0
        if cp["IN_FAMILY"] >= THETA_PROMOTE:
            branch.status = PROMOTE
            return PROMOTE, accept(), branch, 0
        if cp["OUT_OF_FAMILY"] >= THETA_REJECT:
            branch.status = REJECT
            return REJECT, confirmed, branch, 0
        branch.status = UNRESOLVED
        return UNRESOLVED, confirmed, branch, 0
    if arm == "no_other":
        if cp["MISSING_REPRESENTATION"] == 1:
            branch.status = MISSING
            return MISSING, confirmed, branch, 0
        branch.status = PROMOTE
        return PROMOTE, accept(), branch, 0
    if arm == "confirmation_bypass":
        branch.status = PROMOTE
        return PROMOTE, accept(), branch, 0

    policy = {"provisional_random": "random",
              "provisional_disagreement": "in_family_disagreement",
              "main": "info_gain"}[arm]
    answers: list = []
    used = 0
    while used < budget:
        zq = choose_challenge(fam, mask, event, policy, legal, answers, rng)
        if zq is None:
            break
        a = int(fam.u3[phi_true, zq])          # the partner answers truthfully
        answers.append((zq, a))
        used += 1
        cp = cause_posterior(fam, mask, event, tuple(answers), prior_out)
        if cp["MISSING_REPRESENTATION"] == 1:
            branch.cause_posterior = cp
            branch.answers = tuple(answers)
            branch.queries_used = used
            branch.status = MISSING
            return MISSING, confirmed, branch, used
        if cp["OUT_OF_FAMILY"] >= THETA_REJECT:
            break
        if cp["IN_FAMILY"] >= THETA_PROMOTE and int(
                (surviving_mask(fam, confirmed.grounded)
                 & (fam.u3[:, z] == u)
                 & _answer_mask(fam, answers)).sum()) == int(
                    (surviving_mask(fam, confirmed.grounded)
                     & _answer_mask(fam, answers)).sum()):
            break
    branch.cause_posterior = cp
    branch.answers = tuple(answers)
    branch.queries_used = used
    if cp["MISSING_REPRESENTATION"] == 1:
        branch.status = MISSING
        return MISSING, confirmed, branch, used
    if cp["OUT_OF_FAMILY"] >= THETA_REJECT:
        branch.status = REJECT
        return REJECT, confirmed, branch, used
    if cp["IN_FAMILY"] >= THETA_PROMOTE:
        branch.status = PROMOTE
        new = ConfirmedState(
            confirmed.identity,
            confirmed.grounded + (obs,) + tuple(
                GroundedObservation(zq, a, f"ans:{zq}:{a}")
                for zq, a in answers),
            confirmed.version + 1,
            confirmed.evidence_ids | {obs.base_evidence},
            confirmed.provenance)
        return PROMOTE, new, branch, used
    branch.status = UNRESOLVED
    return UNRESOLVED, confirmed, branch, used


def _answer_mask(fam, answers):
    m = np.ones(fam.n, dtype=bool)
    for zq, a in answers:
        m &= (fam.u3[:, zq] == a)
    return m


ARMS = ("always_accept", "old_quarantine", "survivor_majority",
        "map_protection", "always_quarantine", "cause_mixture_no_query",
        "provisional_random", "provisional_disagreement", "main",
        "oracle_cause", "no_other", "confirmation_bypass")
