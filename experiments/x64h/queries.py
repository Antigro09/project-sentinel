"""One question interface, two question kinds, and honest information gain.

A behavioural question names an input and asks for its output. A semantic
question asks which reading is meant -- a sense, an attachment, an omitted
argument, or a paraphrase choice. Both go through the same enumeration of
the answer distribution under (Z, Phi, O, M), so the information-gain and
random-disagreement policies share a pool, a budget and a stopping rule.

No adaptive-submodularity guarantee is claimed. Greedy optimality was a
measured property of one finite instance in the theory package, not a
theorem, and the runner records posterior entropy, expected answer entropy,
expected risk reduction, the realised answer and the posterior change for
every question so that the claim can be checked rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import posterior as PO
from . import semantic as S


@dataclass(frozen=True)
class Question:
    kind: str            # "behavioral" | "semantic"
    payload: object      # an input tape, or a (slot, value) sense probe
    cost: float


def behavioral_pool(universe, asked, cost):
    return [Question("behavioral", t, cost) for t in universe if t not in asked]


def semantic_pool(forms, asked, cost):
    out = []
    for slot in ("op", "filt", "scope"):
        for v in sorted({getattr(z, slot) for z in forms}):
            if ("semantic", slot, v) in asked:
                continue
            out.append(Question("semantic", (slot, v), cost))
    return out


def answer_distribution(q, log_joint, forms, phis):
    """P(answer) and the posterior mass behind each answer, exactly."""
    tot = PO.logsumexp(list(log_joint.values()))
    if tot == -math.inf:
        return {}
    buckets: dict[object, float] = {}
    for (i, z), v in log_joint.items():
        w = math.exp(v - tot)
        if q.kind == "behavioral":
            a = S.execute(z)(q.payload)
        else:
            slot, val = q.payload
            a = (getattr(z, slot) == val)
        buckets[a] = buckets.get(a, 0.0) + w
    return buckets


def mutual_information(q, log_joint, forms, phis) -> float:
    """H(state) - E_answer[H(state | answer)] = H(answer) for a
    deterministic answer channel, which both kinds are here."""
    d = answer_distribution(q, log_joint, forms, phis)
    return -sum(p * math.log2(p) for p in d.values() if p > 0)


def restrict(q, answer, log_joint, forms):
    out = {}
    for (i, z), v in log_joint.items():
        if q.kind == "behavioral":
            a = S.execute(z)(q.payload)
        else:
            slot, val = q.payload
            a = (getattr(z, slot) == val)
        if a == answer:
            out[(i, z)] = v
    return out


def choose(policy, pool, log_joint, forms, phis, rng):
    """`infogain` maximises mutual information minus the frozen query cost;
    `random` picks uniformly among questions that split the posterior at
    all. Identical pool, identical stopping rule, identical answer access."""
    live = [q for q in pool if len(answer_distribution(q, log_joint, forms,
                                                       phis)) > 1]
    if not live:
        return None
    if policy == "random":
        return live[rng.randrange(len(live))]
    best, score = None, -math.inf
    for q in live:
        v = mutual_information(q, log_joint, forms, phis) - q.cost
        if v > score:
            best, score = q, v
    return best
