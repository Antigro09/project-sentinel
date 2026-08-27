"""X64H-0C: the pre-freeze audit.

Three things X64H-0B got wrong or left implicit.

1. IT CONFLATED SUPPORT WITH INFORMATION. "Every utterance leaves all 13824
   conventions live, 0.0 bits leaked" is a statement about the SUPPORT of
   p(phi | u), not about its shape. Non-zero support everywhere is perfectly
   compatible with a sharply non-uniform posterior. The correct quantities
   are I(Z; U) and I(Phi; U), and they are different numbers: the family's
   closure under role-wise relabelling forces I(Z; U) = 0 exactly, and it
   forces nothing at all about I(Phi; U).

2. IT USED THE WRONG LIKELIHOOD. The teacher chooses the exposure pattern so
   that the true convention identifies the task, and rejects candidates
   where no pattern does. The arms modelled `p(u | phi, z) = uniform over
   the pool`, which is not the generator. The correctly specified likelihood
   conditions on the selection and on acceptance:

       p(u | phi, z, D, accepted) = [u in good(phi, z, D)] / |good(phi, z, D)|
       good(phi, z, D) = the pool patterns whose utterance leaves exactly z

   X64H-0B claimed misspecification "can only cost the treatment arms". That
   claim is withdrawn: a misspecified likelihood can help or hurt depending
   on how the error correlates with the truth, and only measurement decides.

3. IT REPORTED ONE DISTRIBUTION. Accuracy on ACCEPTED tasks is conditional
   on the acceptance event. The unconditional distribution is a different
   quantity and measures coverage, not mechanism.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from . import episode as EP
from . import family as FAM


# --------------------------------------------------------- B. information

def information_audit(fam, pool=FAM.P2) -> dict:
    """Exact enumeration over phi ~ U(family), z ~ U(forms), pattern ~
    U(pool). No sampling anywhere."""
    n, m = fam.n, fam.m
    us = sorted({int(x) for p in pool for x in np.unique(fam.codes(p))})
    HPhi, HZ = math.log2(n), math.log2(m)
    pu, HPhi_u, HZ_u, supp, acc_phi, acc_z = [], [], [], [], [], []
    for u in us:
        C = fam.counts(u, pool)                    # (n, m) pattern counts
        tot = C.sum()
        if tot <= 0:
            continue
        p_u = tot / (n * m * len(pool))
        pu.append(p_u)
        row = C.sum(axis=1); row = row / row.sum()
        col = C.sum(axis=0); col = col / col.sum()
        HPhi_u.append(_H(row)); HZ_u.append(_H(col))
        supp.append(int((C.sum(axis=1) > 0).sum()))
        acc_phi.append(float(row.max())); acc_z.append(float(col.max()))
    pu = np.array(pu); pu = pu / pu.sum()
    out = {
        "H_Phi": HPhi,
        "H_Phi_given_U": float((pu * np.array(HPhi_u)).sum()),
        "H_Z": HZ,
        "H_Z_given_U": float((pu * np.array(HZ_u)).sum()),
        "support_over_Phi_after_U_min": min(supp),
        "support_over_Phi_after_U_max": max(supp),
        "one_utterance_convention_accuracy": float((pu * np.array(acc_phi)).sum()),
        "one_utterance_task_meaning_accuracy": float((pu * np.array(acc_z)).sum()),
        "chance_convention_accuracy": 1.0 / n,
        "chance_task_meaning_accuracy": 1.0 / m,
        "distinct_utterances": len(pu),
    }
    out["I_Phi_U"] = out["H_Phi"] - out["H_Phi_given_U"]
    out["I_Z_U"] = out["H_Z"] - out["H_Z_given_U"]
    return out


def _H(p) -> float:
    q = np.asarray(p, dtype=float)
    q = q[q > 0]
    return float(-(q * np.log2(q)).sum())


def conditional_information(fam, beh, tasks, pool=FAM.P2) -> dict:
    """I(Z; U | D) on the tasks as generated, both ways: ignoring the
    acceptance event (the unconditional generative model) and conditioning
    on it (what the learner actually faces)."""
    naive_H, aware_H, base_H = [], [], []
    for t in tasks:
        L = list(t.live)
        base_H.append(math.log2(len(L)))
        C = fam.counts(t.u, pool)[:, L]
        col = C.sum(axis=0)
        naive_H.append(_H(col / col.sum()) if col.sum() > 0 else 0.0)
        w = selection_weights(fam, L, t.u, pool)
        s = w.sum(axis=0)
        aware_H.append(_H(s / s.sum()) if s.sum() > 0 else 0.0)
    return {
        "H_Z_given_D": float(np.mean(base_H)),
        "H_Z_given_U_D_unconditional": float(np.mean(naive_H)),
        "I_Z_U_given_D_unconditional":
            float(np.mean(base_H) - np.mean(naive_H)),
        "H_Z_given_U_D_accepted": float(np.mean(aware_H)),
        "I_Z_U_given_D_accepted": float(np.mean(base_H) - np.mean(aware_H)),
        "tasks": len(tasks),
    }


# ------------------------------------------------- C. selection likelihood

_WCACHE: dict = {}


def selection_weights(fam, live, u_obs: int, pool) -> np.ndarray:
    """p(u_obs | phi, z, D, accepted) for every convention and every
    candidate meaning in D, exactly as the generator draws it:

        good(phi, z, D) = the pool patterns whose utterance leaves exactly z
        p(u | .)        = #{p in good realising u_obs} / |good|

    Returns an (n_conventions, |live|) array. Zero rows are conventions
    under which the task could not have been generated at all -- which is
    itself evidence, and the naive likelihood throws it away.
    """
    L = list(live)
    key = (id(fam), tuple(L), int(u_obs), tuple(pool))
    hit = _WCACHE.get(key)
    if hit is not None:
        return hit
    M = [fam.codes(p)[:, L] for p in pool]          # each (n, |L|)
    n, k = fam.n, len(L)
    num = np.zeros((n, k)); den = np.zeros((n, k))
    for a in range(k):
        for pi in range(len(pool)):
            cand = M[pi][:, a]                       # (n,)
            hits = np.zeros((n, k), dtype=bool)
            for q in range(len(pool)):
                hits |= (M[q] == cand[:, None])
            qual = hits.sum(axis=1) == 1             # only one meaning left
            qual &= hits[:, a]                       # and it is this one
            den[:, a] += qual
            num[:, a] += qual & (cand == u_obs)
    out = np.zeros((n, k))
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    if len(_WCACHE) < 20000:
        _WCACHE[key] = out
    return out


def infer_selection_aware(fam, p_phi, u, pool, live, tie):
    """The same exact joint as `episode.infer`, with the generator's own
    likelihood in place of the uniform-pool one."""
    L = list(live)
    W = selection_weights(fam, L, u, pool)
    joint = p_phi[:, None] * W
    tot = joint.sum()
    if tot <= 0:
        return EP.infer(fam, p_phi, u, pool, live, tie)
    b = np.zeros(fam.m)
    b[L] = joint.sum(axis=0) / tot
    conv = joint.sum(axis=1) / tot
    best, bs = None, -1.0
    for j in tie:
        if b[j] > bs + 1e-12:
            best, bs = j, b[j]
    return b, conv, best


def infer_selection_only(fam, u, pool, live, tie):
    """Diagnostic: no convention learning at all, and no proper
    normalisation -- score each meaning by HOW MANY conventions would have
    made this utterance select it. Isolates what the selection rule alone
    is worth."""
    L = list(live)
    W = selection_weights(fam, L, u, pool)
    b = np.zeros(fam.m)
    s = (W > 0).sum(axis=0).astype(float)
    if s.sum() <= 0:
        s = np.ones(len(L))
    b[L] = s / s.sum()
    best, bs = None, -1.0
    for j in tie:
        if b[j] > bs + 1e-12:
            best, bs = j, b[j]
    return b, np.full(fam.n, 1.0 / fam.n), best


# ------------------------------------------------- conflict, D and E and F

def conflict_curves(fam, beh, ep, cfg, seed, likelihood="aware") -> dict:
    """X64E's conflict statistic on this generator. MATCHED pairs are the
    tasks as generated; CONTRADICTORY pairs keep the utterance and swap in
    another task's demonstrations, so language and evidence disagree."""
    r = EP.run_arm(fam, beh, ep, "persist", cfg, seed, likelihood=likelihood)
    p_end = _final_prior(fam, beh, ep, cfg, seed, likelihood)
    matched, contra = [], []
    tr = [ep.tasks[i] for i in ep.tr_idx]
    for i, t in enumerate(tr):
        matched.append(EP.conflict_of(fam, p_end, t.u, t.pool, t.live, t.tie))
        other = tr[(i + 1) % len(tr)]
        if set(other.live) != set(t.live):
            contra.append(EP.conflict_of(fam, p_end, t.u, t.pool,
                                         other.live, t.tie))
    return {"matched": matched, "contradictory": contra,
            "arm_conflict": r["conflict"]}


def _final_prior(fam, beh, ep, cfg, seed, likelihood):
    p = np.full(fam.n, 1.0 / fam.n)
    for pos, t in enumerate(ep.tasks):
        _b, conv, _best = EP._infer_by(likelihood, fam, p, t.u, t.pool,
                                       t.live, t.tie)
        s = conv.sum()
        if s > 0:
            p = conv / s
    return p


def auroc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    xs = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    xs.sort()
    ranks, i = {}, 0
    order = []
    while i < len(xs):
        j = i
        while j < len(xs) and xs[j][0] == xs[i][0]:
            j += 1
        r = (i + j + 1) / 2.0
        for k in range(i, j):
            order.append((r, xs[k][1]))
        i = j
    s = sum(r for r, lab in order if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def auprc(pos, neg) -> float:
    if not pos or not neg:
        return float("nan")
    xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg], reverse=True)
    tp = fp = 0
    prev_r, area = 0.0, 0.0
    for v, lab in xs:
        if lab:
            tp += 1
        else:
            fp += 1
        rec = tp / len(pos)
        prec = tp / (tp + fp)
        area += (rec - prev_r) * prec
        prev_r = rec
    return area


def change_diagnostic(fam, beh, cfg, seed, likelihood="aware") -> dict:
    """E. What the posterior actually does when the convention changes at
    the declared boundary. No change DETECTOR is claimed: `contradiction` is
    the measured state in which the observation carries zero mass under
    every surviving convention. It is the only state here that identifies a
    change, and it is reported whether or not it fires."""
    ep = EP.change_episode(fam, beh, cfg, seed)
    cls = fam.class_of()
    old_c, new_c = int(cls[ep.phi]), int(cls[ep.phi_alt])
    p = np.full(fam.n, 1.0 / fam.n)
    rows, declared = [], None
    for pos, t in enumerate(ep.tasks):
        changed = pos >= ep.boundary
        W = (selection_weights(fam, list(t.live), t.u, t.pool)
             if likelihood == "aware"
             else fam.counts(t.u, t.pool)[:, list(t.live)] / len(t.pool))
        contradiction = bool((p[:, None] * W).sum() <= 0)
        if contradiction and declared is None and changed:
            declared = pos
        b, conv, best = EP._infer_by(likelihood, fam, p, t.u, t.pool, t.live,
                                     t.tie)
        s = conv.sum()
        if s > 0:
            p = conv / s
        if t.kind == "transfer":
            rows.append({
                "pos": pos, "after_change": changed,
                "correct": bool(best == t.z),
                "mass_old_class": float(p[cls == old_c].sum()),
                "mass_new_class": float(p[cls == new_c].sum()),
                "H": EP.entropy_bits(p), "contradiction": contradiction,
            })
    return {"rows": rows, "boundary": ep.boundary, "declared_at": declared,
            "unselectable_after_change": ep.unselectable_after_change}


def paired_bootstrap(a, b, reps: int = 10000, seed: int = 20260827) -> dict:
    """Resample EPISODES, not transfer tasks. `a` and `b` are per-episode
    means for the two arms on the same episodes."""
    rng = random.Random(seed)
    n = len(a)
    d = [a[i] - b[i] for i in range(n)]
    obs = sum(d) / n
    boot = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        boot.append(sum(d[i] for i in idx) / n)
    boot.sort()
    lo = boot[int(0.025 * reps)]
    hi = boot[min(reps - 1, int(0.975 * reps))]
    return {"delta": obs, "lo": lo, "hi": hi, "episodes": n,
            "excludes_zero": (lo > 0) or (hi < 0)}
