"""X65A-S2: the balanced record/event suite and the detectability audit.

The population that matters is UNDER-DETERMINED confirmed records. On a
determined record the surviving set is a single convention, so any alien
event contradicts it and even the old zero-survivor rule catches it. Every
S1.7 corruption happened on an under-determined record, so the suite is
built to contain them in quantity rather than by accident.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .provisional import ConfirmedState
from .semantic_mem import GroundedObservation, surviving_mask


@dataclass(frozen=True)
class Case:
    kind: str                 # legit | alien
    record_class: str         # determined | underdetermined
    confirmed: ConfirmedState
    phi_true: int
    event: tuple
    detect_class: str = ""    # one_query | multi_query | indistinguishable
    survivors_before: int = 0


def make_record(fam, phi, rng, target: int) -> ConfirmedState | None:
    """Ground observations until the surviving set reaches `target`
    (1 = determined, >1 = under-determined)."""
    zs = list(range(fam.m))
    rng.shuffle(zs)
    g: list = []
    for z in zs:
        trial = g + [GroundedObservation(z, int(fam.u3[phi, z]), f"g{z}")]
        k = int(surviving_mask(fam, tuple(trial)).sum())
        if k < target:
            continue                     # would overshoot; skip this meaning
        g = trial
        if k == target:
            break
    if not g:
        return None
    if int(surviving_mask(fam, tuple(g)).sum()) != target:
        return None
    return ConfirmedState("id:x", tuple(g))


def detect_class(fam, confirmed, phi_true, event, legal, budget=3) -> str:
    """A: some legal question's true answer disagrees with every surviving
    provisional hypothesis. B: no single one does but a bounded sequence
    does. C: some provisional hypothesis matches the partner on every legal
    question, so nothing in the query set can separate them."""
    mask = surviving_mask(fam, confirmed.grounded)
    z, u = event
    prov = mask & (fam.u3[:, z] == u)
    if not prov.any():
        return "zero_survivor"
    idx = np.where(prov)[0]
    for zq in legal:
        a = int(fam.u3[phi_true, zq])
        if not (fam.u3[idx, zq] == a).any():
            return "one_query"
    # any hypothesis agreeing on the whole legal set is indistinguishable
    agree = np.ones(len(idx), dtype=bool)
    for zq in legal:
        agree &= (fam.u3[idx, zq] == int(fam.u3[phi_true, zq]))
    if agree.any():
        return "indistinguishable"
    live = idx
    for _ in range(budget):
        best, keep = None, None
        for zq in legal:
            a = int(fam.u3[phi_true, zq])
            nxt = live[fam.u3[live, zq] == a]
            if best is None or len(nxt) < best:
                best, keep = len(nxt), nxt
        live = keep
        if len(live) == 0:
            return "multi_query"
    return "multi_query_over_budget"


def build_suite(fam, seed: int, n_determined: int = 100,
                n_underdetermined: int = 100, legal=None,
                budget: int = 3) -> list:
    rng = random.Random(seed)
    legal = list(range(fam.m)) if legal is None else list(legal)
    cases: list = []
    for cls, target_pool in (("determined", (1,)),
                             ("underdetermined", (2, 3, 4))):
        want = n_determined if cls == "determined" else n_underdetermined
        made = 0
        guard = 0
        while made < want and guard < want * 40:
            guard += 1
            phi = rng.randrange(fam.n)
            tgt = target_pool[rng.randrange(len(target_pool))]
            rec = make_record(fam, phi, rng, tgt)
            if rec is None:
                continue
            mask = surviving_mask(fam, rec.grounded)
            nsurv = int(mask.sum())
            grounded_z = {g.z for g in rec.grounded}
            free = [z for z in range(fam.m) if z not in grounded_z]
            if not free:
                continue
            # a legitimate in-family disambiguating event
            zl = free[rng.randrange(len(free))]
            cases.append(Case("legit", cls, rec, phi,
                              (zl, int(fam.u3[phi, zl])),
                              "n/a", nsurv))
            # an alien event: consistent with a surviving but FALSE
            # convention where one exists, which is the harmful case
            alt = [i for i in np.where(mask)[0] if i != phi]
            placed = False
            rng.shuffle(free)
            for za in free:
                for other in alt:
                    ua = int(fam.u3[other, za])
                    if ua != int(fam.u3[phi, za]):
                        cases.append(Case(
                            "alien", cls, rec, phi, (za, ua),
                            detect_class(fam, rec, phi, (za, ua), legal,
                                         budget), nsurv))
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                za = free[0]
                ua = (int(fam.u3[phi, za]) + 1) % (fam.A ** 3)
                cases.append(Case("alien", cls, rec, phi, (za, ua),
                                  detect_class(fam, rec, phi, (za, ua),
                                               legal, budget), nsurv))
            made += 1

    # A CONFIRMED RECORD THAT IS ALREADY WRONG. Without this class
    # MISSING_REPRESENTATION is unreachable -- if the confirmed set always
    # contains the partner, truthful answers can never eliminate all of it,
    # and a gate on MISSING would be testing nothing.
    made = 0
    guard = 0
    while made < max(20, n_underdetermined // 5) and guard < 2000:
        guard += 1
        phi = rng.randrange(fam.n)
        rec = make_record(fam, phi, rng, target_pool[rng.randrange(3)])
        if rec is None:
            continue
        mask = surviving_mask(fam, rec.grounded)
        alt = [i for i in np.where(mask)[0] if i != phi]
        gz = {g.z for g in rec.grounded}
        free = [z for z in range(fam.m) if z not in gz]
        if not alt or len(free) < 2:
            continue
        other = alt[rng.randrange(len(alt))]
        za = free[0]
        if int(fam.u3[other, za]) == int(fam.u3[phi, za]):
            continue
        poisoned = ConfirmedState(
            rec.identity,
            rec.grounded + (GroundedObservation(za, int(fam.u3[other, za]),
                                                "poison"),))
        pm = surviving_mask(fam, poisoned.grounded)
        if not pm.any() or bool(pm[phi]):
            continue                    # must actually exclude the partner
        zl = free[1]
        cases.append(Case("legit_after_corruption", "corrupted", poisoned,
                          phi, (zl, int(fam.u3[phi, zl])), "n/a",
                          int(pm.sum())))
        made += 1
    return cases


def audit(cases) -> dict:
    out: dict = {}
    for c in cases:
        key = (c.kind, c.record_class, c.detect_class)
        out[key] = out.get(key, 0) + 1
    return {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in sorted(out.items())}
