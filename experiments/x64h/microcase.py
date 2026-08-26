"""A four-atom, two-constructor FT-SPCFG whose posteriors can be worked out
by hand.

Its purpose is to make the inference checkable rather than plausible. The
family is built so that some conventions are genuinely indistinguishable
from a given observation -- an automorphism, not an error -- and so the
query machinery has something to be uncertain about.

    atoms      a1 a2 (filter)   s1 s2 (scope)
    surface    two phrases, deliberately shared across roles
    conventions differ in the lexical map and in child order
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from . import posterior as PO


@dataclass(frozen=True)
class MicroConvention:
    lex: tuple[tuple[str, str], ...]      # atom -> word
    order: tuple[str, ...]                # ("F","S") or ("S","F")

    def word(self, atom: str) -> str:
        return dict(self.lex)[atom]


ATOMS_F = ("a1", "a2")
ATOMS_S = ("s1", "s2")
WORDS = ("w1", "w2")
MICRO_Z = tuple(itertools.product(ATOMS_F, ATOMS_S))


def micro_family_functional() -> tuple[MicroConvention, ...]:
    """Lexical maps drawn from all FUNCTIONS into a three-word surface, not
    only bijections. With more words than atoms an unobserved atom's word is
    genuinely free, so a non-separating observation set leaves a real
    automorphism -- which the bijection family cannot exhibit, because with
    two atoms and two words changing one changes both."""
    words3 = ("w1", "w2", "w3")
    out = []
    for fa in itertools.product(words3, repeat=len(ATOMS_F)):
        for sa in itertools.product(words3, repeat=len(ATOMS_S)):
            lex = (("a1", fa[0]), ("a2", fa[1]),
                   ("s1", sa[0]), ("s2", sa[1]))
            for order in (("F", "S"), ("S", "F")):
                out.append(MicroConvention(lex, order))
    return tuple(out)


def micro_family() -> tuple[MicroConvention, ...]:
    out = []
    for fmap in itertools.permutations(WORDS):
        for smap in itertools.permutations(WORDS):
            lex = (("a1", fmap[0]), ("a2", fmap[1]),
                   ("s1", smap[0]), ("s2", smap[1]))
            for order in (("F", "S"), ("S", "F")):
                out.append(MicroConvention(lex, order))
    return tuple(out)


def realise(phi: MicroConvention, z) -> tuple[str, ...]:
    f, s = z
    parts = {"F": phi.word(f), "S": phi.word(s)}
    return tuple(parts[k] for k in phi.order)


def likelihood(phi: MicroConvention, z, u) -> float:
    return 1.0 if realise(phi, z) == u else 0.0


def exact_posterior(u, family=None, log_p_phi=None, log_p_z=None):
    """Uniform priors unless given. Returns the normalised joint, the
    convention marginal and the semantic marginal."""
    fam = family or micro_family()
    n_phi, n_z = len(fam), len(MICRO_Z)
    lpp = log_p_phi or [-math.log(n_phi)] * n_phi
    lpz = log_p_z or {z: -math.log(n_z) for z in MICRO_Z}
    joint = {}
    for i, phi in enumerate(fam):
        for z in MICRO_Z:
            if likelihood(phi, z, u) > 0:
                joint[(i, z)] = lpp[i] + lpz[z]
    tot = PO.logsumexp(list(joint.values()))
    post = {k: math.exp(v - tot) for k, v in joint.items()}
    phi_marg = [0.0] * n_phi
    z_marg = {z: 0.0 for z in MICRO_Z}
    for (i, z), p in post.items():
        phi_marg[i] += p
        z_marg[z] += p
    return post, phi_marg, z_marg


def observational_class_given(phi: MicroConvention, observed, family=None):
    """The class under a RESTRICTED set of observed meanings. When the
    observations do not separate the atoms, distinct conventions become
    indistinguishable -- a genuine automorphism, and the evaluator must
    score the class rather than the draw."""
    fam = family or micro_family()
    mine = tuple(realise(phi, z) for z in observed)
    return tuple(i for i, other in enumerate(fam)
                 if tuple(realise(other, z) for z in observed) == mine)


def observational_class(phi: MicroConvention, family=None):
    """Conventions producing the same utterance for every meaning. Scoring
    the identity of one draw instead of this class would count an
    automorphism as an error."""
    fam = family or micro_family()
    mine = tuple(realise(phi, z) for z in MICRO_Z)
    return tuple(i for i, other in enumerate(fam)
                 if tuple(realise(other, z) for z in MICRO_Z) == mine)
