"""X64H-0B: the convention family, its accounting, and its separating sets.

X64H-0 reported 1152 codebooks. The correct raw count for a free
(pi_O, pi_F, pi_S, order) is

    |inj(V_O -> W_O)| * |perm(V_F -> W)| * |perm(V_S -> W)| * 2

which for the disjoint-operator alphabet is 2 * 24 * 24 * 2 = 2304. The
1152 came from `codebook.full_family(fix_op=True)`, which PINNED pi_O to
the identity. That is a design restriction, not an observational quotient,
and reporting it as a family size was an accounting error. `accounting()`
below reports raw assignments, unique executable conventions, observational
classes and canonical representatives separately, and `pin_no_operator_symmetry`
shows that the pinned half is observationally distinct from the half it
excluded -- so there is no two-to-one symmetry to quotient by.

Two alphabets are supported.

    shared        one 4-word alphabet for all three roles. pi_O is an
                  injection V_O -> W, so nothing in a token's identity or
                  position says which role produced it. 12*24*24*2 = 13824.
    disjoint_op   the operator draws from its own 2-word alphabet. This is
                  the X64H-0 family with pi_O freed: 2*24*24*2 = 2304. It
                  carries an ORDER ARTIFACT -- a two-token utterance holding
                  an operator word fixes the order bit by position -- which
                  the audit reports rather than hides.

Exposure is by role pattern. The calibration pool shows all three roles;
the transfer pool shows one or two. The pattern is never announced, so
`p(u | phi, z) = (patterns in the pool realising u) / |pool|`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from . import semantic as S

# Slot inventories. Chosen so that all 2 x 4 x 4 = 32 typed forms have
# DISTINCT behaviour over the frozen universe: with the X64H-0 inventory
# eight forms collapsed into one all-empty class, which caps any oracle
# below 0.98 for reasons that have nothing to do with the convention.
OPS = ("keep", "remove")
FILTERS_0B = ("letters", "symbols seen before",
              "the first symbol", "the last symbol")
SCOPES_0B = ("whole", "before hash", "after hash", "outside brackets")

ROLES = ("O", "F", "S")
CAL_POOL = (("O", "F", "S"),)
P2 = (("O", "F"), ("O", "S"), ("F", "S"))
P1 = (("O",), ("F",), ("S",))


def forms() -> list:
    return [S.Z(o, f, s) for o in OPS for f in FILTERS_0B for s in SCOPES_0B]


@dataclass(frozen=True)
class FamilySpec:
    overlap: str = "shared"          # shared | disjoint_op
    n_words: int = 4

    def alphabet(self) -> int:
        return self.n_words if self.overlap == "shared" else self.n_words + 2


class Family:
    """Every convention in the family, as int arrays over word ids."""

    def __init__(self, spec: FamilySpec = FamilySpec()):
        self.spec = spec
        n = spec.n_words
        shared = tuple(range(n))
        if spec.overlap == "shared":
            ops = list(itertools.permutations(shared, len(OPS)))
        elif spec.overlap == "disjoint_op":
            ops = list(itertools.permutations(range(n, n + 2)))
        else:
            raise ValueError(f"unknown overlap {spec.overlap!r}")
        fs_ = list(itertools.permutations(shared))
        self.n_op_maps, self.n_f_maps, self.n_s_maps = len(ops), len(fs_), len(fs_)
        rows = [(o, f, s, b)
                for o in ops for f in fs_ for s in fs_ for b in (0, 1)]
        self.n = len(rows)
        self.PO = np.array([r[0] for r in rows], dtype=np.int16)
        self.PF = np.array([r[1] for r in rows], dtype=np.int16)
        self.PS = np.array([r[2] for r in rows], dtype=np.int16)
        self.ORD = np.array([r[3] for r in rows], dtype=np.int16)

        self.forms = forms()
        self.op_i = np.array([OPS.index(z.op) for z in self.forms])
        self.f_i = np.array([FILTERS_0B.index(z.filt) for z in self.forms])
        self.s_i = np.array([SCOPES_0B.index(z.scope) for z in self.forms])
        self.m = len(self.forms)

        self.A = spec.alphabet()
        self.recode()
        self._classes = None

    def recode(self) -> None:
        """Rebuild every utterance code from the mappings and the alphabet
        width. Separated out so a PLANTED family can widen the alphabet and
        still encode unambiguously -- a plant whose private word aliases an
        ordinary pair would make the audit that catches it vacuous."""
        self.wo = self.PO[:, self.op_i]
        self.wf = self.PF[:, self.f_i]
        self.ws = self.PS[:, self.s_i]
        rev = (self.ORD == 1)[:, None]
        A = self.A
        self.u3 = np.where(rev,
                           self.ws * A * A + self.wf * A + self.wo,
                           self.wo * A * A + self.wf * A + self.ws)
        self.u2 = {
            ("O", "F"): np.where(rev, self.wf * A + self.wo,
                                 self.wo * A + self.wf),
            ("O", "S"): np.where(rev, self.ws * A + self.wo,
                                 self.wo * A + self.ws),
            ("F", "S"): np.where(rev, self.ws * A + self.wf,
                                 self.wf * A + self.ws),
        }
        self.u1 = {("O",): self.wo, ("F",): self.wf, ("S",): self.ws}

    # ------------------------------------------------------- observation

    def codes(self, pattern):
        if len(pattern) == 3:
            return self.u3
        if len(pattern) == 2:
            return self.u2[tuple(pattern)]
        return self.u1[tuple(pattern)]

    def counts(self, u, pool) -> np.ndarray:
        """(n_conventions, n_forms) count of pool patterns realising u."""
        acc = np.zeros((self.n, self.m), dtype=np.float64)
        for p in pool:
            acc += (self.codes(p) == u)
        return acc

    def realise(self, i: int, j: int, pattern) -> int:
        return int(self.codes(pattern)[i, j])

    # ------------------------------------------------------- accounting

    def signatures(self) -> np.ndarray:
        """Everything a convention can ever produce, as one row per
        convention: the 3-role code plus the SORTED multiset of 2-role and
        1-role codes for every form. Sorted because the pattern is not
        announced, so only the multiset is observable."""
        parts = [self.u3[:, :, None]]
        two = np.stack([self.u2[p] for p in P2], axis=2)
        parts.append(np.sort(two, axis=2))
        one = np.stack([self.u1[p] for p in P1], axis=2)
        parts.append(np.sort(one, axis=2))
        return np.concatenate(parts, axis=2).reshape(self.n, -1)

    def classes(self) -> dict:
        if self._classes is None:
            sig = self.signatures()
            _, inv = np.unique(sig, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            out: dict[int, list[int]] = {}
            for i, c in enumerate(inv):
                out.setdefault(int(c), []).append(i)
            self._classes = out
            self._inv = inv
        return self._classes

    def class_of(self) -> np.ndarray:
        self.classes()
        return self._inv

    def canonical(self) -> list[int]:
        return sorted(min(v) for v in self.classes().values())

    def accounting(self) -> dict:
        cls = self.classes()
        sizes: dict[int, int] = {}
        for v in cls.values():
            sizes[len(v)] = sizes.get(len(v), 0) + 1
        exec_rows = np.unique(
            np.concatenate([self.PO, self.PF, self.PS, self.ORD[:, None]],
                           axis=1), axis=0)
        return {
            "overlap": self.spec.overlap,
            "alphabet_size": self.A,
            "raw_parameter_assignments": self.n,
            "raw_factorisation":
                f"{self.n_op_maps} * {self.n_f_maps} * {self.n_s_maps} * 2",
            "unique_executable_conventions": int(exec_rows.shape[0]),
            "observational_equivalence_classes": len(cls),
            "class_size_histogram": sizes,
            "canonical_representatives": len(self.canonical()),
            "quotient": ("none: parameters -> observations is injective"
                         if len(cls) == self.n else "see class_size_histogram"),
        }

    # -------------------------------------------------- separating sets

    def separates(self, cal_forms) -> bool:
        """Does grounding these calibration meanings determine the
        convention? Observing (z, u3) for every z in the set leaves one
        convention iff the 3-role code rows are pairwise distinct."""
        sub = self.u3[:, list(cal_forms)]
        return int(np.unique(sub, axis=0).shape[0]) == self.n

    def residual_classes(self, cal_forms) -> int:
        sub = self.u3[:, list(cal_forms)]
        return int(np.unique(sub, axis=0).shape[0])

    def minimal_separating_size(self, kmax: int = 5) -> dict:
        """Exhaustive at each k until one is found. The structural lower
        bound is that k-1 grounded F values leave pi_F determined only by
        elimination, so k < 3 cannot separate a 4-value permutation."""
        for k in range(1, kmax + 1):
            found = None
            tried = 0
            for c in itertools.combinations(range(self.m), k):
                tried += 1
                if self.separates(c):
                    found = c
                    break
            if found is not None:
                return {"k": k, "example": [str(self.forms[j]) for j in found],
                        "example_idx": list(found), "combinations_tried": tried,
                        "exhaustive_below_k": True}
        return {"k": None, "exhaustive_below_k": True}

    def greedy_separating(self, order=None) -> dict:
        """Greedy: repeatedly add the meaning that most reduces the number
        of conventions still tied to the true one."""
        chosen: list[int] = []
        cand = list(order if order is not None else range(self.m))
        best_res = 1
        while True:
            res = self.residual_classes(chosen) if chosen else 1
            if chosen and res == self.n:
                break
            pick, score = None, -1
            for j in cand:
                if j in chosen:
                    continue
                r = self.residual_classes(chosen + [j])
                if r > score:
                    pick, score = j, r
            if pick is None:
                break
            chosen.append(pick)
            best_res = score
            if score == self.n:
                break
        return {"size": len(chosen), "idx": chosen,
                "separating": self.residual_classes(chosen) == self.n,
                "residual_conventions": self.residual_classes(chosen)}

    # ------------------------------------------------------ leak audits

    def one_utterance_audit(self, pool=P2) -> dict:
        """V8. For every utterance the pool can produce, how many
        conventions remain possible when the meaning is unknown?"""
        live = {}
        for p in pool:
            for u in np.unique(self.codes(p)):
                c = self.counts(int(u), pool)
                n_live = int((c.sum(axis=1) > 0).sum())
                live[int(u)] = max(live.get(int(u), 0), n_live)
        vals = list(live.values())
        by_word: dict[int, set] = {}
        lens: dict[int, set] = {}
        for p in pool:
            code = self.codes(p)
            for i in range(self.n):
                for j in range(self.m):
                    u = int(code[i, j])
                    lens.setdefault(len(p), set()).add(i)
                    for w in _tokens(u, len(p), self.A):
                        by_word.setdefault(w, set()).add(i)
        return {
            "utterances": len(vals),
            "min_conventions_left": min(vals),
            "max_conventions_left": max(vals),
            "mean_conventions_left": sum(vals) / len(vals),
            "fraction_of_family_left": min(vals) / self.n,
            "bits_leaked_worst_case":
                float(np.log2(self.n) - np.log2(min(vals))),
            "words_not_used_by_every_convention":
                sum(1 for s in by_word.values() if len(s) < self.n),
            "lengths_not_used_by_every_convention":
                sum(1 for s in lens.values() if len(s) < self.n),
            "identifies_convention": min(vals) == 1,
        }


def _tokens(u: int, k: int, A: int) -> list[int]:
    out = []
    for _ in range(k):
        out.append(u % A)
        u //= A
    return out[::-1]


def pin_no_operator_symmetry(fam: Family) -> dict:
    """The X64H-0 family pinned pi_O to the identity. If that were a
    symmetry the excluded half would duplicate the kept half; it does not."""
    ident = fam.PO[0]
    keep = np.where((fam.PO == ident[None, :]).all(axis=1))[0]
    drop = np.array([i for i in range(fam.n) if i not in set(keep.tolist())])
    sig = fam.signatures()
    kept = {sig[i].tobytes() for i in keep}
    collide = sum(1 for i in drop if sig[i].tobytes() in kept)
    return {"kept_by_pinning": int(len(keep)),
            "excluded_by_pinning": int(len(drop)),
            "excluded_that_duplicate_a_kept_convention": collide,
            "pinning_is_a_symmetry": collide == len(drop)}


def subset(fam: Family, idx) -> Family:
    """A view on part of the family, for calibration arms that need a
    DELIBERATELY BROKEN family the audits must catch."""
    import copy
    out = copy.copy(fam)
    idx = list(idx)
    out.n = len(idx)
    for a in ("PO", "PF", "PS", "ORD", "wo", "wf", "ws", "u3"):
        setattr(out, a, getattr(fam, a)[idx])
    out.u2 = {k: v[idx] for k, v in fam.u2.items()}
    out.u1 = {k: v[idx] for k, v in fam.u1.items()}
    out._classes = None
    return out


def symmetry_audit(fam: Family, pools=(P2, CAL_POOL, P1)) -> dict:
    """Under a UNIFORM prior over the family, how much does one utterance
    say about the meaning?

        sum over phi of p(u | phi, z)

    If that is constant in z then the utterance is EXACTLY uninformative
    without the convention, and every point of transfer advantage is
    attributable to the convention posterior rather than to the surface
    form. The family is closed under relabelling codewords role by role, so
    for any z, z' the map phi -> phi . sigma is a bijection carrying the
    conventions that realise u from z onto those that realise it from z'.
    This enumerates that argument instead of assuming it.
    """
    worst = 0.0
    per_pool = {}
    for pool in pools:
        w = 0.0
        for p in pool:
            for u in np.unique(fam.codes(p)):
                col = fam.counts(int(u), pool).sum(axis=0)
                w = max(w, float(col.max() - col.min()))
        per_pool[str(pool)] = w
        worst = max(worst, w)
    return {"max_spread_over_meanings": worst,
            "per_pool": per_pool,
            "utterance_is_uninformative_without_convention": worst == 0.0}


def plant_private_codeword(fam: Family) -> Family:
    """A generator bug in which one convention owns a codeword no other
    convention can produce. The alphabet is widened so the private word
    cannot alias an ordinary pair, which is what the audit must catch."""
    out = subset(fam, range(fam.n))
    out.A = fam.A + 1
    out.PF = out.PF.copy()
    out.PF[0, 0] = fam.A
    out.recode()
    out._classes = None
    return out
