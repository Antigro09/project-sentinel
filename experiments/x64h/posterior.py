"""Joint, conflict, behaviour and OTHER posteriors, all in log space.

Four quantities are computed and never conflated, because X64D lost an
experiment to conflating two of them:

    ambiguity   spread of the in-class posterior over behaviours
    conflict    the utterance and the demonstrations describe different
                meanings -- a Bayes factor, not a set emptiness
    OTHER       the episode is outside the model: unknown realisation,
                unknown meaning, or unknown program
    behaviour   the in-class posterior pushed through denotation

The in-class posterior is normalised only inside (IN, M=0). OTHER mass is
never discarded and the in-class posterior is renormalised before any
decision is taken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import grammar as G
from . import semantic as S


def _log(x: float) -> float:
    """log(0) is -inf here, not an exception: the no-open-world ablation
    legitimately sets three of the four mixture priors to zero."""
    return -math.inf if x <= 0.0 else math.log(x)


def logsumexp(xs) -> float:
    xs = [x for x in xs if x > -math.inf]
    if not xs:
        return -math.inf
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


@dataclass(frozen=True)
class OtherModel:
    """Frozen base likelihoods for the three ways of being outside the
    model. These are part of the freeze digest; tuning them after seeing a
    final seed would invalidate the run."""
    prior_in: float = 0.94
    prior_unknown_realization: float = 0.02
    prior_unknown_meaning: float = 0.02
    prior_unknown_program: float = 0.02
    vocab: int = 20
    out_alphabet: int = 8

    def log_realization(self, u) -> float:
        # a string nothing in the grammar generates: uniform over the vocab
        return -len(u) * math.log(self.vocab)

    def log_meaning(self, demos) -> float:
        return -sum(max(1, len(y)) for _x, y in demos) * math.log(
            self.out_alphabet)

    def log_program(self, demos) -> float:
        return self.log_meaning(demos)


@dataclass(frozen=True)
class Config:
    other: OtherModel = OtherModel()
    prior_conflict: float = 0.10
    rho: float = 0.0
    exact: bool = True


@dataclass(frozen=True)
class Posterior:
    log_joint: dict
    behaviour: dict
    p_conflict: float
    p_other: dict
    ambiguity: float
    incomplete_candidates: bool
    log_evidence_in: float

    def top_behaviour(self):
        if not self.behaviour:
            return None, 0.0
        b = max(self.behaviour, key=self.behaviour.get)
        return b, self.behaviour[b]


def joint(evidence, phis, log_p_phi, forms, cfg: Config,
          log_p_z=None, incomplete=False) -> Posterior:
    u = evidence.utterance
    demos = evidence.demonstrations
    n = len(forms)
    lpz = log_p_z or {z: -math.log(n) for z in forms}

    beh_ll = {z: S.behavioral_loglik(z, demos, cfg.rho) for z in forms}
    lang = {}
    for i, phi in enumerate(phis):
        tab = G.loglik_table(phi, u, forms)
        for z in forms:
            lang[(i, z)] = tab[z]

    log_joint, per_phi_u, = {}, []
    for i, phi in enumerate(phis):
        terms = []
        for z in forms:
            v = log_p_phi[i] + lpz[z] + lang[(i, z)] + beh_ll[z]
            if v > -math.inf:
                log_joint[(i, z)] = v
            terms.append(log_p_phi[i] + lpz[z] + lang[(i, z)])
        per_phi_u.append(logsumexp(terms))

    L0 = logsumexp(list(log_joint.values()))
    # M = 1: the utterance and the demonstrations are generated independently
    lD = logsumexp([lpz[z] + beh_ll[z] for z in forms])
    L1 = logsumexp([per_phi_u[i] + lD for i in range(len(phis))]) \
        - logsumexp(list(log_p_phi))

    o = cfg.other
    # The four components must explain the SAME observation -- the utterance
    # and the demonstrations both. An earlier version scored UNKNOWN_MEANING
    # on the demonstrations alone, so with no demonstrations its likelihood
    # was 1 and it dominated every episode.
    u_marg = logsumexp(per_phi_u) - logsumexp(list(log_p_phi))
    l_in = _log(o.prior_in) + L0
    l_ur = _log(o.prior_unknown_realization) + o.log_realization(u) + lD
    l_um = (_log(o.prior_unknown_meaning) + o.log_realization(u)
            + o.log_meaning(demos))
    l_up = _log(o.prior_unknown_program) + u_marg + o.log_program(demos)
    tot = logsumexp([l_in, l_ur, l_um, l_up])
    p_other = {
        "IN": math.exp(l_in - tot) if tot > -math.inf else 0.0,
        "UNKNOWN_REALIZATION": math.exp(l_ur - tot) if tot > -math.inf else 0.0,
        "UNKNOWN_MEANING": math.exp(l_um - tot) if tot > -math.inf else 0.0,
        "UNKNOWN_PROGRAM": math.exp(l_up - tot) if tot > -math.inf else 0.0,
    }

    a0 = _log(1 - cfg.prior_conflict) + L0
    a1 = _log(cfg.prior_conflict) + L1
    den = logsumexp([a0, a1])
    p_conflict = math.exp(a1 - den) if den > -math.inf else 0.0

    beh: dict = {}
    if L0 > -math.inf:
        for (i, z), v in log_joint.items():
            b = S.denote(z)
            beh[b] = beh.get(b, 0.0) + math.exp(v - L0)
    amb = 0.0
    for p in beh.values():
        if p > 0:
            amb -= p * math.log2(p)
    return Posterior(log_joint, beh, p_conflict, p_other, amb,
                     incomplete or not cfg.exact, L0)


def convention_posterior(log_joint, n_phi):
    out = [-math.inf] * n_phi
    for (i, _z), v in log_joint.items():
        out[i] = logsumexp([out[i], v])
    return out
