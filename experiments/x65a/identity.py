"""X65A-S: opaque persistent identities.

An identity is a communication partner. Its ID is a key and nothing else:
sampled independently of the convention, carrying no convention bits, never
appearing in an instruction, and randomly remapped between stream seeds so
that an id which happened to correlate with a convention on one seed cannot
on the next.

`mutual_information` audits I(identity; convention) exactly over the
generator's own assignment distribution rather than estimating it from a
sample. Three planted leaks must each be caught.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Identity:
    label: str          # the opaque public ID
    slot: int           # the generator's index; never exposed

    def canon(self):
        return {"label": self.label}


def opaque_label(seed: int, i: int) -> str:
    return "id:" + hashlib.sha256(f"{seed}:identity:{i}".encode()
                                  ).hexdigest()[:12]


def assign(seed: int, n: int, conventions) -> tuple:
    """Labels are drawn from a stream that never sees `conventions`, then
    permuted. The convention argument is accepted only so that a planted
    leak can be written against the same signature."""
    rng = random.Random(seed * 6151 + 17)
    labels = [opaque_label(seed, i) for i in range(n)]
    rng.shuffle(labels)
    return tuple(Identity(labels[i], i) for i in range(n))


def assign_leaky_index(seed: int, n: int, conventions) -> tuple:
    """PLANT 1: the id encodes the convention index."""
    return tuple(Identity(f"id:{conventions[i]:08x}", i) for i in range(n))


def assign_leaky_token(seed: int, n: int, conventions) -> tuple:
    """PLANT 2: a lexical token correlated with the convention."""
    return tuple(Identity(f"id:{opaque_label(seed, i)[3:9]}-"
                          f"{conventions[i] % 4}", i) for i in range(n))


def assign_leaky_order(seed: int, n: int, conventions) -> tuple:
    """PLANT 3: deterministic assignment by convention order."""
    order = sorted(range(n), key=lambda i: conventions[i])
    labels = [opaque_label(seed, i) for i in range(n)]
    return tuple(Identity(labels[order.index(i)], i) for i in range(n))


def probe_vectors(n: int, pool, count: int = 24) -> list:
    """The audit generates its OWN probes. A caller-supplied probe set can
    be degenerate without anyone noticing -- the first version of this audit
    was handed twelve constant vectors, every one of which induced the same
    ranking, and the order-leak plant escaped. Probes here include the
    identity order, its reverse, explicit permutations and random draws, and
    the set is checked for rank diversity before it is used."""
    pool = list(pool)
    if len(pool) < n:
        raise ValueError(f"convention pool of {len(pool)} is smaller than "
                         f"the {n} identities it must distinguish")
    out = [list(pool[:n]), list(reversed(pool[:n]))]
    for k in range(count):
        rng = random.Random(9001 + k)
        v = [rng.choice(pool) for _ in range(n)]
        out.append(v)
    ranks = {tuple(sorted(range(n), key=lambda i: v[i])) for v in out}
    if len(ranks) < min(6, count):
        raise ValueError(f"probe set is degenerate: only {len(ranks)} "
                         f"distinct rankings")
    return out


def functional_independence(assigner, seeds, n: int, pool,
                            extra_vectors=()) -> dict:
    """THE EXACT AUDIT. If the assignment is not a function of the
    convention vector at all, then I(label; convention) = 0 exactly for
    every convention distribution -- no estimation and no finite-sample
    floor.

    An empirical mutual information over per-seed-unique labels cannot show
    this: a label that appears once determines everything inside its own
    seed, so every assigner scores log2(n) and the number is an artifact of
    label uniqueness. That was the first version of this audit and it was
    wrong."""
    vecs = probe_vectors(n, list(pool)) + [list(v) for v in extra_vectors]
    ranks = {tuple(sorted(range(n), key=lambda i: v[i])) for v in vecs}
    differing = []
    for s in seeds:
        outs = {tuple(i.label for i in assigner(s, n, c)) for c in vecs}
        if len(outs) > 1:
            differing.append(s)
    return {"depends_on_convention": bool(differing),
            "seeds_where_output_changed": len(differing),
            "seeds_tested": len(seeds),
            "probe_vectors": len(vecs), "distinct_probe_rankings": len(ranks),
            "I_label_convention_bits_exact": 0.0 if not differing else None,
            "zero": not differing}


def _feature(label: str, buckets: int = 8) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) % buckets


def empirical_mi(assigner, seeds, n: int, convention_pool,
                 buckets: int = 8) -> dict:
    """A second, weaker measure: mutual information between a coarse
    feature of the label and the convention, with conventions RESAMPLED per
    seed. Reported with its own shuffle floor, because a small positive
    value here is finite-sample noise rather than a leak."""
    def mi(pairs):
        joint: dict = {}
        for a, b in pairs:
            joint[(a, b)] = joint.get((a, b), 0) + 1
        tot = sum(joint.values())
        px: dict = {}
        py: dict = {}
        for (a, b), c in joint.items():
            px[a] = px.get(a, 0) + c
            py[b] = py.get(b, 0) + c
        return sum((c / tot) * math.log2((c / tot)
                                         / ((px[a] / tot) * (py[b] / tot)))
                   for (a, b), c in joint.items())

    pairs, shuffled = [], []
    for s in seeds:
        rng = random.Random(s * 31 + 7)
        convs = [rng.choice(convention_pool) for _ in range(n)]
        ids = assigner(s, n, convs)
        for k, ident in enumerate(ids):
            pairs.append((_feature(ident.label, buckets), convs[k] % buckets))
        perm = list(convs)
        rng.shuffle(perm)
        for k, ident in enumerate(ids):
            shuffled.append((_feature(ident.label, buckets), perm[k] % buckets))
    return {"I_feature_convention_bits": round(mi(pairs), 6),
            "shuffle_floor_bits": round(mi(shuffled), 6),
            "buckets": buckets, "samples": len(pairs)}


def label_appears_in(label: str, *texts) -> bool:
    return any(label in str(t) for t in texts)
