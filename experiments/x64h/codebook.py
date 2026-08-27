"""X64H-0: overlapping hidden codebooks.

H1 failed because conventions drew phrases from DISJOINT pools, so the
vocabulary in an utterance identified the convention before any inference
happened. A static family-aware parser therefore matched the oracle at 1.00
for every family size.

The fix is a shared surface alphabet with hidden role-specific
permutations. Every convention uses the SAME codewords; only the mapping
differs. A codeword carries no information about which convention produced
it, so one utterance cannot identify phi, and the convention has to be
accumulated across tasks.

    roles          O (operator), F (filter), S (scope)
    values         |V_O| = 2, |V_F| = 4, |V_S| = 4
    codewords      W_O = 2 words, disjoint;  W = 4 words SHARED by F and S
    convention     phi = (pi_O, pi_F, pi_S, order_bit)
    exposure       every utterance shows exactly two roles, and which two is
                   NOT announced

A two-word utterance containing an operator codeword exposes (O, ?) and the
agent cannot tell whether the second word is a filter or a scope. One
without an operator codeword exposes (F, S) in an order the hidden bit
decides. Nothing on the surface distinguishes conventions.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from . import semantic as S

V_O = ("keep", "remove")
V_F = ("everything", "letters", "brackets", "hashes")
V_S = ("whole", "before hash", "after hash", "inside brackets")

W_O = ("o1", "o2")
W = ("c1", "c2", "c3", "c4")            # shared by the filter and scope roles

PATTERNS = (("O", "F"), ("O", "S"), ("F", "S"))


@dataclass(frozen=True)
class Codebook:
    pi_O: tuple[str, ...]      # V_O -> W_O
    pi_F: tuple[str, ...]      # V_F -> W
    pi_S: tuple[str, ...]      # V_S -> W
    order: int                 # 0 = declared order, 1 = reversed

    def word(self, role: str, value: str) -> str:
        if role == "O":
            return self.pi_O[V_O.index(value)]
        if role == "F":
            return self.pi_F[V_F.index(value)]
        return self.pi_S[V_S.index(value)]

    def key(self) -> tuple:
        return (self.pi_O, self.pi_F, self.pi_S, self.order)


def full_family(fix_op: bool = True) -> tuple[Codebook, ...]:
    """All codebooks. With the operator mapping fixed the family is
    24 x 24 x 2 = 1152; letting it vary doubles that."""
    ops = (W_O,) if fix_op else tuple(itertools.permutations(W_O))
    return tuple(Codebook(o, f, s, b)
                 for o in ops
                 for f in itertools.permutations(W)
                 for s in itertools.permutations(W)
                 for b in (0, 1))


def forms() -> tuple:
    """The X64H-0 semantic space: 2 x 4 x 4 = 32 typed forms."""
    return tuple(S.Z(o, f, sc) for o in V_O for f in V_F for sc in V_S)


def realise(phi: Codebook, z, pattern) -> tuple[str, ...]:
    vals = {"O": z.op, "F": z.filt, "S": z.scope}
    toks = [phi.word(r, vals[r]) for r in pattern]
    return tuple(reversed(toks)) if phi.order else tuple(toks)


def utterances(phi: Codebook, z) -> set[tuple[str, ...]]:
    return {realise(phi, z, p) for p in PATTERNS}


def consistent(phi: Codebook, z, u: tuple[str, ...]) -> bool:
    """Exposure is not announced, so the utterance is explained if ANY
    pattern produces it."""
    return any(realise(phi, z, p) == u for p in PATTERNS)


def leak_audit(family, fs) -> dict:
    """V9: does one utterance identify the convention through any surface
    feature? Every measurement here must come out at chance."""
    by_word: dict[str, set] = {}
    lengths: dict[int, set] = {}
    for phi in family:
        for z in fs:
            for p in PATTERNS:
                u = realise(phi, z, p)
                lengths.setdefault(len(u), set()).add(phi.key())
                for w in u:
                    by_word.setdefault(w, set()).add(phi.key())
    n = len(family)
    unique_words = [w for w, ks in by_word.items() if len(ks) < n]
    unique_lengths = [L for L, ks in lengths.items() if len(ks) < n]
    return {
        "distinct_words": len(by_word),
        "words_not_used_by_every_convention": len(unique_words),
        "utterance_lengths": sorted(lengths),
        "lengths_not_used_by_every_convention": len(unique_lengths),
        "leak_free": not unique_words and not unique_lengths,
    }


def observational_signature(phi: Codebook, fs) -> tuple:
    """Everything a convention can ever produce. Two codebooks with the same
    signature are indistinguishable by any legal observation and must be
    scored as one class."""
    return tuple(sorted((tuple(z), tuple(sorted(utterances(phi, z))))
                        for z in fs))


def equivalence_classes(family, fs) -> dict:
    out: dict[tuple, list[int]] = {}
    for i, phi in enumerate(family):
        out.setdefault(observational_signature(phi, fs), []).append(i)
    return out
