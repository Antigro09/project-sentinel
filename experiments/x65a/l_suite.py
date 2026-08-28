"""X65A-L: the latent-identity evaluation suite.

The identity set is CONSTRUCTED to contain the relations the phase is about,
rather than sampled and hoped for:

    slots 0,1   the SAME convention          -> literal identity is not
                                                recoverable; convention
                                                equivalence is
    slots 2,3   differ by one lexical mapping
    slots 4,5   differ only by the order bit
    slots 6,7   independent

plus a partner that was never grounded (NEW_IDENTITY), an out-of-family
speaker, a stale record whose convention has moved on, and evidence that
initially favours the wrong record.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from x64h import episode as EP
from x64h import family as FAM

from . import latent_id as LI
from .semantic_mem import GroundedObservation, surviving_mask


@dataclass
class Identity:
    slot: int
    phi: int
    relation: str
    grounded: tuple
    survivors: int


@dataclass
class Probe:
    kind: str              # returning | new | out_of_family | stale |
                           # ambiguous | misleading
    slot: int              # the true identity slot, or -1
    task: object
    phi_true: int
    equivalence: tuple     # slots holding an observationally equal record


def _one_mapping_apart(fam, phi, rng):
    for _ in range(400):
        i = rng.randrange(fam.n)
        d = (int((fam.PO[i] != fam.PO[phi]).any())
             + int((fam.PF[i] != fam.PF[phi]).any())
             + int((fam.PS[i] != fam.PS[phi]).any())
             + int(fam.ORD[i] != fam.ORD[phi]))
        if d == 1 and fam.ORD[i] == fam.ORD[phi]:
            return i
    return (phi + 1) % fam.n


def _order_apart(fam, phi):
    cand = np.where((fam.PO == fam.PO[phi]).all(axis=1)
                    & (fam.PF == fam.PF[phi]).all(axis=1)
                    & (fam.PS == fam.PS[phi]).all(axis=1)
                    & (fam.ORD != fam.ORD[phi]))[0]
    return int(cand[0]) if len(cand) else (phi + 1) % fam.n


def build_identities(fam, seed: int, n: int = 8) -> list:
    rng = random.Random(seed)
    base = rng.randrange(fam.n)
    # the order pair needs its OWN base: order_apart is an involution, so
    # order_apart(order_apart(base)) == base and slot 5 silently collided
    # with slot 0 in the first version of this construction
    other = rng.randrange(fam.n)
    while other == base or fam.ORD[other] != fam.ORD[base]:
        other = rng.randrange(fam.n)
    phis = [base, base,
            _one_mapping_apart(fam, base, rng), 0,
            other, _order_apart(fam, other)]
    phis[3] = _one_mapping_apart(fam, phis[2], rng)
    while len(phis) < n:
        phis.append(rng.randrange(fam.n))
    rel = ["same_convention", "same_convention", "one_mapping_apart",
           "one_mapping_apart", "order_apart", "order_apart"] + \
          ["independent"] * (n - 6)
    out = []
    for slot in range(n):
        phi = phis[slot]
        target = 1 if slot % 3 else rng.choice([2, 3])
        zs = list(range(fam.m))
        rng.shuffle(zs)
        g: list = []
        for z in zs:
            trial = g + [GroundedObservation(z, int(fam.u3[phi, z]), f"g{z}")]
            k = int(surviving_mask(fam, tuple(trial)).sum())
            if k < target:
                continue
            g = trial
            if k == target:
                break
        if not g:
            continue
        out.append(Identity(slot, phi, rel[slot], tuple(g),
                            int(surviving_mask(fam, tuple(g)).sum())))
    return out


def equivalence_of(ids, slot) -> tuple:
    """Slots whose stored records are observationally identical. Literal
    identity is not recoverable inside such a group and must not be scored
    as a retrieval failure."""
    me = ids[slot]
    return tuple(i.slot for i in ids
                 if {(g.z, g.u) for g in i.grounded}
                 == {(g.z, g.u) for g in me.grounded}
                 or i.phi == me.phi)


def build_probes(fam, beh, cfg, ids, seed: int, n_per: int = 3) -> list:
    rng = random.Random(seed + 77)
    probes: list = []
    used: dict = {}
    for ident in ids:
        gz = {g.z for g in ident.grounded}
        free = [z for z in range(fam.m) if z not in gz]
        rng.shuffle(free)
        for z in free[:n_per]:
            t = _transfer_task(fam, beh, cfg, ident.phi, z, rng)
            if t is None:
                continue
            used.setdefault(ident.slot, set()).add(z)
            probes.append(Probe("returning", ident.slot, t, ident.phi,
                                equivalence_of(ids, ident.slot)))
    # a partner never grounded: NEW_IDENTITY
    known = {i.phi for i in ids}
    for _ in range(n_per * 2):
        phi = rng.randrange(fam.n)
        if phi in known:
            continue
        z = rng.randrange(fam.m)
        t = _transfer_task(fam, beh, cfg, phi, z, rng)
        if t is not None:
            probes.append(Probe("new", -1, t, phi, ()))
    # an out-of-family speaker: an utterance no convention realises for z
    for _ in range(n_per):
        z = rng.randrange(fam.m)
        t = _transfer_task(fam, beh, cfg, rng.randrange(fam.n), z, rng)
        if t is None:
            continue
        # an utterance NO convention realises for any live meaning under
        # the pool. Searched over the finite code space with a hard bound:
        # the first version looped forever whenever every code happened to
        # be realisable, which is the common case.
        alien_u = None
        for cand in range(fam.A ** 2):
            if fam.counts(cand, t.pool)[:, list(t.live)].sum() == 0:
                alien_u = cand
                break
        if alien_u is None:
            # In the shared 4-word alphabet EVERY two-token code has an
            # in-family explanation for some live meaning, so no single
            # transfer utterance can carry an out-of-family signature. That
            # is a property of the alphabet, reported rather than patched.
            continue
        probes.append(Probe("out_of_family", -1,
                            EP.Task(t.kind, t.z, t.demos, t.live, alien_u,
                                    t.pool, t.tie), -1, ()))
    # an UNKNOWN_MEANING probe: demonstrations no in-family form explains,
    # so the live set is empty and no identity can be assigned
    for _ in range(n_per):
        t = _transfer_task(fam, beh, cfg, ids[0].phi,
                           rng.randrange(fam.m), rng)
        if t is None:
            continue
        probes.append(Probe("unknown_meaning", -1,
                            EP.Task(t.kind, t.z, t.demos, (), t.u, t.pool,
                                    t.tie), ids[0].phi, ()))

    # evidence consistent with several stored identities, and evidence that
    # initially favours the wrong one
    for ident in ids[:2]:
        gz = {g.z for g in ident.grounded}
        free = [z for z in range(fam.m) if z not in gz]
        if free:
            t = _transfer_task(fam, beh, cfg, ident.phi, free[0], rng)
            if t is not None:
                probes.append(Probe("ambiguous", ident.slot, t, ident.phi,
                                    equivalence_of(ids, ident.slot)))
    for ident in ids[2:4]:
        gz = {g.z for g in ident.grounded}
        free = [z for z in range(fam.m) if z not in gz]
        if free:
            t = _transfer_task(fam, beh, cfg, ident.phi, free[-1], rng)
            if t is not None:
                probes.append(Probe("misleading", ident.slot, t, ident.phi,
                                    equivalence_of(ids, ident.slot)))
    return probes


def _transfer_task(fam, beh, cfg, phi, z, rng):
    lo, hi = cfg.ambiguity
    for _ in range(cfg.retries):
        target = rng.randint(lo, hi)
        d, live = EP.pick_demos(beh, fam.m, z, target, cfg.demos_transfer_cap,
                                rng)
        if not (lo <= len(live) <= hi) or z not in live:
            continue
        good = [fam.realise(phi, z, p) for p in FAM.P2
                if [k for k in live
                    if any(fam.realise(phi, k, q) == fam.realise(phi, z, p)
                           for q in FAM.P2)] == [z]]
        if not good:
            continue
        return EP.Task("transfer", z, d, live,
                       good[rng.randrange(len(good))], FAM.P2,
                       tuple(rng.sample(range(fam.m), fam.m)))
    return None
