"""FT-SPCFG: a finite typed synchronous grammar, with EXACT inside.

Given a convention phi and a semantic tree z, the utterance is produced by
an ordered chain of emitters:

    B0  E1  B1  E2  B2  ...  En  Bn

Each Ei realises one semantic node (the operator marker, the filter atom,
the scope atom), each Bi is an optional-function-word slot, the child order
and the operator's attachment come from phi, and an omissible child may be
dropped. Every emitter carries a normalised distribution over token
sequences including epsilon, so the chain is a proper distribution over
strings.

    p(u | phi, z) = sum over drop patterns and over all segmentations of u
                    into the chain, of the product of emitter probabilities

The sum over segmentations is the inside algorithm -- a linear-chain dynamic
program, O(chain x |u|^2). It is exact: no beam, no truncation, no top-k.
`brute_force_likelihood` enumerates derivations explicitly and the two are
pinned equal in the tests.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

from . import convention as C
from . import semantic as S

EPS: tuple[str, ...] = ()


def _boundary_dist(phi: C.ConventionSpec) -> tuple[tuple[tuple[str, ...], float], ...]:
    """One optional function word, or nothing. Normalised by construction."""
    out, mass = [], 0.0
    n = max(1, len(phi.optional()))
    for w, p in phi.optional():
        q = p / n
        if q > 0:
            out.append(((w,), q))
            mass += q
    out.append((EPS, 1.0 - mass))
    return tuple(out)


def chain(phi: C.ConventionSpec, tree: S.Tree):
    """The ordered emitter chain, as a list of normalised distributions."""
    lex = phi.lex()
    ctx = tree.op
    order = phi.order(ctx)
    omissible = tree.omissible()
    bd = _boundary_dist(phi)

    nodes: list[tuple[tuple[tuple[str, ...], float], ...]] = []
    op_emit = ((lex[(S.ROOT, tree.op)], 1.0),)
    child_emits = []
    for slot in order:
        atom = tree.filt if slot == S.FILTER else tree.scope
        ph = lex[(slot, atom)]
        d = phi.drop_prob(slot) if slot in omissible else 0.0
        child_emits.append(((ph, 1.0 - d), (EPS, d)) if d > 0
                           else ((ph, 1.0),))
    if phi.attachment(ctx) == "pre":
        nodes = [op_emit] + child_emits
    else:
        nodes = child_emits + [op_emit]

    out = [bd]
    for n in nodes:
        out.append(n)
        out.append(bd)
    return out


def inside(phi: C.ConventionSpec, z, u: tuple[str, ...]) -> float:
    """Exact p(u | phi, z)."""
    ch = chain(phi, S.to_tree(z))
    n = len(u)
    cur = [0.0] * (n + 1)
    cur[0] = 1.0
    for dist in ch:
        nxt = [0.0] * (n + 1)
        for j in range(n + 1):
            if cur[j] == 0.0:
                continue
            for toks, p in dist:
                if p == 0.0:
                    continue
                k = len(toks)
                if j + k <= n and tuple(u[j:j + k]) == toks:
                    nxt[j + k] += cur[j] * p
        cur = nxt
    return cur[n]


def brute_force_likelihood(phi: C.ConventionSpec, z, u: tuple[str, ...]) -> float:
    """Enumerate every derivation and concatenate. Only for microcases."""
    ch = chain(phi, S.to_tree(z))
    total = 0.0
    for combo in itertools.product(*ch):
        toks = tuple(t for seq, _p in combo for t in seq)
        if toks != u:
            continue
        p = 1.0
        for _seq, q in combo:
            p *= q
        total += p
    return total


def support(phi: C.ConventionSpec, z) -> dict[tuple[str, ...], float]:
    """Every string the chain can produce, with its total probability. Used
    to check that each per-(phi, z) likelihood is normalised."""
    ch = chain(phi, S.to_tree(z))
    acc: dict[tuple[str, ...], float] = {(): 1.0}
    for dist in ch:
        nxt: dict[tuple[str, ...], float] = {}
        for pre, pp in acc.items():
            for toks, p in dist:
                if p == 0.0:
                    continue
                key = pre + toks
                nxt[key] = nxt.get(key, 0.0) + pp * p
        acc = nxt
    return acc


def generate(phi: C.ConventionSpec, z, rng) -> tuple[str, ...]:
    ch = chain(phi, S.to_tree(z))
    out: list[str] = []
    for dist in ch:
        r, acc = rng.random(), 0.0
        for toks, p in dist:
            acc += p
            if r <= acc:
                out.extend(toks)
                break
    return tuple(out)


@lru_cache(maxsize=None)
def _cached_forms():
    return tuple(S.x64h_forms())


def loglik_table(phi: C.ConventionSpec, u: tuple[str, ...], forms=None):
    """log p(u | phi, z) for every z, exactly."""
    import math
    out = {}
    for z in (forms if forms is not None else _cached_forms()):
        p = inside(phi, z, u)
        out[z] = math.log(p) if p > 0 else -math.inf
    return out
