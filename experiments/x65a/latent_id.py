"""X65A-L: which stored record applies, when no identity is supplied.

THE SKETCH. A record's likelihood depends on it only through its surviving
convention set, and that set is determined by the grounded (z, u) pairs. So
the pairs WITHOUT provenance are an exact sufficient sketch:

    likelihood_from_sketch(e) == likelihood_from_full_record(e)

exactly, not approximately. That is option A of the retrieval specification,
and it is why a 512-byte retrieval budget is workable at all: a full record
is ~330 bytes and two of them already blow the budget, while a sketch is
~45.

THE LATENT VARIABLE.

    J in {1..N, NEW_IDENTITY, OUT_OF_FAMILY}
    L_j(e)   = sum_phi q_j(phi) sum_z p(e, z | phi, J=j)
    L_new(e) = sum_phi p_family(phi) sum_z p(e, z | phi, NEW)
    L_out(e) = the frozen OTHER likelihood

Identity is NOT committed when several records hold observationally
equivalent conventions; convention-equivalence recovery is the
capability-relevant score and literal identity is reported beside it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from x64h import episode as EP


from .provisional import (MISSING, PRIOR_OUT, THETA_PROMOTE, other_likelihood)
from .semantic_mem import GroundedObservation, surviving_mask
from .types import byte_cost

NEW_IDENTITY = "NEW_IDENTITY"
OUT_OF_FAMILY = "OUT_OF_FAMILY"

ASSIGN_EXISTING = "ASSIGN_EXISTING"
CREATE_NEW = "CREATE_NEW_IDENTITY"
QUARANTINE_OUT = "OUT_OF_FAMILY"
UNRESOLVED_IDENTITY = "UNRESOLVED_IDENTITY"

P_NEW = Fraction(1, 10)
P_OUT = PRIOR_OUT
SHORTLIST = 4
RETRIEVAL_BYTES = 512
ACTIVE_BYTES = 4096
GROUNDING_FOR_NEW = 2          # one ambiguous utterance is not an identity


@dataclass(frozen=True)
class IdentitySketch:
    """Exactly sufficient for the identity likelihood, and nothing else:
    no label, no convention, no provenance."""
    pairs: tuple                      # ((z, u), ...)

    def canon(self):
        return {"p": [[z, u] for z, u in self.pairs]}

    def bytes(self) -> int:
        return byte_cost(self.canon())

    def mask(self, fam) -> np.ndarray:
        return surviving_mask(fam, tuple(
            GroundedObservation(z, u, "") for z, u in self.pairs))


def sketch_of(record) -> IdentitySketch:
    return IdentitySketch(tuple((g.z, g.u) for g in record.grounded))


@dataclass
class ConfirmedIdentityRecord:
    key: int                          # internal slot; never shown to an arm
    grounded: tuple
    verification_domain: tuple        # the legal query universe it was
    challenge_digest: str             # verified against
    validity_context: str = "ctx0"
    version: int = 1
    status: str = "CONFIRMED"
    scope: str = "empirical"          # empirical | global_in_finite_model

    def canon(self):
        return {"grounded": [g.canon() for g in self.grounded],
                "verification_domain": list(self.verification_domain),
                "challenge_digest": self.challenge_digest,
                "validity_context": self.validity_context,
                "version": self.version, "status": self.status,
                "scope": self.scope}


@dataclass
class ProvisionalIdentityBranch:
    identity_posterior: dict
    convention_posterior_support: int
    cause_posterior: dict
    evidence_ids: tuple
    asked: tuple = ()
    query_universe: tuple = ()
    status: str = "OPEN"
    update_budget: int = 3

    def canon(self):
        return {"identity_posterior": {str(k): v for k, v in
                                       sorted(self.identity_posterior.items(),
                                              key=lambda kv: str(kv[0]))},
                "convention_posterior_support":
                    self.convention_posterior_support,
                "cause_posterior": dict(self.cause_posterior),
                "evidence_ids": list(self.evidence_ids),
                "asked": [list(a) for a in self.asked],
                "query_universe": list(self.query_universe),
                "status": self.status, "update_budget": self.update_budget}


# ------------------------------------------------------------ likelihoods

def task_weights(fam, task):
    """p(u | phi, z, selected) for every convention and every live meaning,
    from the frozen X64H selection-aware model. Cached by the audit module,
    and identical for every arm."""
    from x64h import audit0c as A0
    return A0.selection_weights(fam, list(task.live), task.u, task.pool)


def record_likelihood(fam, mask, W) -> float:
    """L_j(e) = sum_phi q_j(phi) sum_z p(e, z | phi)."""
    k = int(mask.sum())
    if k == 0:
        return 0.0
    return float(W[mask].sum()) / (k * W.shape[1])


def identity_posterior(fam, sketch_masks, task, W=None, p_new=P_NEW,
                       p_out=P_OUT, with_new=True, with_out=True) -> dict:
    if W is None:
        W = task_weights(fam, task)
    n = len(sketch_masks)
    p_rec = (Fraction(1) - (p_new if with_new else 0)
             - (p_out if with_out else 0)) / max(1, n)
    w: dict = {}
    for j, m in enumerate(sketch_masks):
        w[j] = float(p_rec) * record_likelihood(fam, m, W)
    if with_new:
        allm = np.ones(fam.n, dtype=bool)
        w[NEW_IDENTITY] = float(p_new) * record_likelihood(fam, allm, W)
    if with_out:
        # the frozen OTHER likelihood for a two-token utterance: an unknown
        # speaker drawn from outside the family emits uniformly
        w[OUT_OF_FAMILY] = float(p_out) * (1.0 / (fam.A ** 2))
    tot = sum(w.values())
    if tot <= 0:
        return {k: 0.0 for k in w}
    return {k: v / tot for k, v in w.items()}


def meaning_posterior(fam, sketch_masks, task, ident, W=None) -> np.ndarray:
    """p(z | e, M) = sum_J p(J|e,M) sum_phi p(z | e, phi) q_J(phi)."""
    if W is None:
        W = task_weights(fam, task)
    live = list(task.live)
    out = np.zeros(fam.m)
    for j, p in ident.items():
        if p <= 0:
            continue
        if j == OUT_OF_FAMILY:
            continue
        m = (np.ones(fam.n, dtype=bool) if j == NEW_IDENTITY
             else sketch_masks[j])
        if not m.any():
            continue
        col = W[m].sum(axis=0)
        s = col.sum()
        if s > 0:
            out[live] += p * (col / s)
    t = out.sum()
    return out / t if t > 0 else out


def predict(fam, sketch_masks, task, ident, W=None):
    b = meaning_posterior(fam, sketch_masks, task, ident, W)
    best, bs = None, -1.0
    for j in task.tie:
        if b[j] > bs + 1e-12:
            best, bs = j, b[j]
    return best, b


# ------------------------------------------------------ retrieval, budgeted

def retrieve(fam, sketches, task, k: int = SHORTLIST, W=None):
    """Scan sketches, shortlist the k best by their EXACT likelihood. The
    sketch is sufficient, so shortlisting loses nothing that scoring would
    have recovered -- but the shortlist itself is a cap, and truncation is
    reported rather than hidden."""
    if W is None:
        W = task_weights(fam, task)
    masks = [s.mask(fam) for s in sketches]
    scored = sorted(((record_likelihood(fam, m, W), j)
                     for j, m in enumerate(masks)), reverse=True)
    keep = [j for _s, j in scored[:k]]
    scanned = sum(s.bytes() for s in sketches)
    retrieved = sum(sketches[j].bytes() for j in keep)
    return keep, masks, {"bytes_scanned": scanned,
                         "bytes_retrieved": retrieved,
                         "nodes_retrieved": len(keep),
                         "incomplete_retrieval": len(sketches) > k,
                         "within_512": retrieved <= RETRIEVAL_BYTES}


# --------------------------------------------------------- query selection

def _entropy(ps) -> float:
    return -sum(p * math.log2(p) for p in ps if p > 0)


def joint_gain(fam, sketch_masks, task, ident, zq, mode="joint",
               W=None) -> float:
    """I((J, Phi, C, Z); A_q | e, M). The answer is deterministic given phi,
    so the information is the entropy of the answer under the joint."""
    if W is None:
        W = task_weights(fam, task)
    buckets: dict = {}
    for j, p in ident.items():
        if p <= 0 or j == OUT_OF_FAMILY:
            continue
        m = (np.ones(fam.n, dtype=bool) if j == NEW_IDENTITY
             else sketch_masks[j])
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        if mode == "identity_only":
            buckets[j] = buckets.get(j, 0.0) + p
            continue
        codes = fam.u3[idx, zq]
        for c in np.unique(codes):
            share = float((codes == c).sum()) / len(idx)
            key = int(c) if mode != "identity_only" else (j, int(c))
            buckets[key] = buckets.get(key, 0.0) + p * share
    tot = sum(buckets.values())
    if tot <= 0:
        return 0.0
    return _entropy([v / tot for v in buckets.values()])


def choose_query(fam, sketch_masks, task, ident, policy, legal, asked, rng,
                 W=None):
    cand = [z for z in legal if z not in {a for a, _b in asked}]
    if not cand:
        return None
    if policy == "random":
        return cand[rng.randrange(len(cand))]
    mode = {"joint": "joint", "identity": "identity_only",
            "convention": "convention"}.get(policy, "joint")
    best, score = None, -1.0
    for zq in cand:
        g = joint_gain(fam, sketch_masks, task, ident, zq, mode, W)
        if g > score:
            best, score = zq, g
    return best


# ------------------------------------------------------- the decision rule

def resolve_identity(fam, sketches, task, phi_true, arm, legal, rng,
                     budget: int = 3, known_true: int | None = None):
    """Returns (outcome, branch, prediction, stats)."""
    W = task_weights(fam, task)
    if arm in ("unlimited_retrieval", "exact_all_record"):
        keep = list(range(len(sketches)))
        masks = [s.mask(fam) for s in sketches]
        rstat = {"bytes_scanned": sum(s.bytes() for s in sketches),
                 "bytes_retrieved": sum(s.bytes() for s in sketches),
                 "nodes_retrieved": len(sketches),
                 "incomplete_retrieval": False,
                 "within_512": sum(s.bytes() for s in sketches)
                 <= RETRIEVAL_BYTES}
    else:
        keep, masks, rstat = retrieve(fam, sketches, task, SHORTLIST, W)

    if arm == "random_record":
        keep = [rng.randrange(len(sketches))]
    elif arm == "most_recent":
        keep = [len(sketches) - 1]
    elif arm == "surface_nearest":
        keep = [max(range(len(sketches)),
                    key=lambda j: sum(1 for z, u in sketches[j].pairs
                                      if u == task.u))]
    elif arm == "shuffled":
        keep = [rng.randrange(len(sketches))]
    elif arm == "oracle_identity" and known_true is not None:
        keep = [known_true]

    sub = [masks[j] for j in keep]
    with_new = arm not in ("no_new_identity", "no_new_forced")
    with_out = arm not in ("no_out_of_family",)
    ident = identity_posterior(fam, sub, task, W, with_new=with_new,
                               with_out=with_out)
    asked: list = []
    policy = {"main": "joint", "random_clarification": "random",
              "joint_infogain": "joint", "identity_query": "identity",
              "convention_query": "convention"}.get(arm, "joint")
    used = 0
    if arm in ("main", "random_clarification", "joint_infogain",
               "identity_query", "convention_query", "exact_all_record"):
        while used < budget:
            top = max(ident.values()) if ident else 0.0
            if top >= float(THETA_PROMOTE):
                break
            zq = choose_query(fam, sub, task, ident, policy, legal, asked,
                              rng, W)
            if zq is None:
                break
            a = int(fam.u3[phi_true, zq])
            asked.append((zq, a))
            used += 1
            sub = [m & (fam.u3[:, zq] == a) for m in sub]
            ident = identity_posterior(fam, sub, task, W, with_new=with_new,
                                       with_out=with_out)

    best, _b = predict(fam, sub, task, ident, W)
    top_key = max(ident, key=ident.get) if ident else OUT_OF_FAMILY
    top_p = ident.get(top_key, 0.0)
    if not any(m.any() for m in sub) and top_key != NEW_IDENTITY:
        outcome = MISSING
    elif top_key == OUT_OF_FAMILY:
        outcome = QUARANTINE_OUT
    elif top_key == NEW_IDENTITY:
        outcome = (CREATE_NEW if used >= GROUNDING_FOR_NEW
                   else UNRESOLVED_IDENTITY)
    elif arm == "no_new_forced":
        # THE CALIBRATION ARM FOR L7. Removing NEW_IDENTITY alone does not
        # cause forced assimilation -- it pushes cases to UNRESOLVED, so the
        # arm never fires and "prevents assimilation" stays untested. This
        # arm removes NEW *and* forces a decision, which is what forced
        # assimilation actually means.
        outcome = ASSIGN_EXISTING
    elif top_p < float(THETA_PROMOTE):
        outcome = UNRESOLVED_IDENTITY
    else:
        outcome = ASSIGN_EXISTING
    branch = ProvisionalIdentityBranch(
        {str(k): round(v, 12) for k, v in ident.items()},
        int(sum(int(m.sum()) for m in sub)),
        {"IN_FAMILY": Fraction(0), "OUT_OF_FAMILY": Fraction(0),
         "MISSING_REPRESENTATION": Fraction(0)},
        tuple(f"ev:{task.z}:{task.u}"), tuple(asked), tuple(legal[:8]),
        outcome, budget)
    stats = {**rstat, "queries": used,
             "identity_top": str(top_key), "identity_top_p": top_p,
             "assigned": keep[max(range(len(keep)),
                                  key=lambda i: ident.get(i, 0.0))]
                         if keep else None,
             "shortlist": list(keep)}
    return outcome, branch, best, stats


ARMS = ("no_memory", "stable_id_oracle", "exact_all_record", "main",
        "random_record", "most_recent", "surface_nearest", "shuffled",
        "wrong_similar", "map_destructive", "no_provisional",
        "no_new_identity", "no_new_forced", "no_out_of_family",
        "no_confirmation",
        "random_clarification", "joint_infogain", "unlimited_retrieval",
        "oracle_identity", "oracle_convention", "bigger_query_memoryless")
