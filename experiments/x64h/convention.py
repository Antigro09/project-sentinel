"""The frozen finite convention family and its post-freeze sampler.

A ConventionSpec is the seven-tuple the specification names. Every field is
a persistent property of an EPISODE, not of a task, which is what makes the
convention posterior worth carrying across tasks.

Nothing here is shown to any arm except the oracle-convention arm. The
sampler is deliberately separated from the meta-grammar so that development
conventions can be drawn freely while final conventions require a released
seed (see protocol.release_final_seeds).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass

from . import semantic as S

FILTER_ATOMS = S.X64H_FILTERS
SCOPE_ATOMS = S.X64H_SCOPES
OP_ATOMS = ("keep", "remove")

# Surface inventory. Phrases are shared on purpose: the same phrase can be
# the realisation of different atoms in different roles, which is the
# contextual polysemy the convention has to encode.
WORDS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
         "theta", "iota", "kappa", "lambda", "mu", "nu", "xi")
FUNCTION_WORDS = ("uh", "so", "well")
PHRASE_POOL = tuple([(w,) for w in WORDS]
                    + [(a, b) for a, b in itertools.combinations(WORDS[:6], 2)])


@dataclass(frozen=True)
class ConventionSpec:
    lexical_map: tuple[tuple[tuple[str, str], tuple[str, ...]], ...]
    order_rules: tuple[tuple[str, tuple[str, ...]], ...]
    contextual_senses: tuple[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], ...]
    phrase_rules: tuple[tuple[str, tuple[str, ...]], ...]
    optional_word_rules: tuple[tuple[str, float], ...]
    argument_drop_rules: tuple[tuple[str, float], ...]
    attachment_rules: tuple[tuple[str, str], ...]

    def lex(self) -> dict[tuple[str, str], tuple[str, ...]]:
        return dict(self.lexical_map)

    def order(self, context: str) -> tuple[str, ...]:
        return dict(self.order_rules).get(context, S.CHILD_SLOTS)

    def drop_prob(self, slot: str) -> float:
        return dict(self.argument_drop_rules).get(slot, 0.0)

    def optional(self) -> tuple[tuple[str, float], ...]:
        return self.optional_word_rules

    def attachment(self, context: str) -> str:
        return dict(self.attachment_rules).get(context, "pre")

    def digest(self) -> str:
        payload = json.dumps({
            "lex": [[list(k), list(v)] for k, v in self.lexical_map],
            "order": [[k, list(v)] for k, v in self.order_rules],
            "senses": [[list(p), [list(x) for x in s]]
                       for p, s in self.contextual_senses],
            "phrase": [[k, list(v)] for k, v in self.phrase_rules],
            "optional": [[w, p] for w, p in self.optional_word_rules],
            "drop": [[k, p] for k, p in self.argument_drop_rules],
            "attach": list(map(list, self.attachment_rules)),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


def sample_convention(seed: int) -> ConventionSpec:
    """Draw one convention from the meta-grammar.

    Phrases are drawn WITHOUT replacement inside a role and WITH replacement
    across roles, so a phrase can be role-ambiguous while never being
    ambiguous within a role. That keeps the convention identifiable in
    principle while still requiring context to decode.
    """
    rg = random.Random(seed)
    pool = list(PHRASE_POOL)
    rg.shuffle(pool)

    lex: dict[tuple[str, str], tuple[str, ...]] = {}
    fpool = pool[: len(FILTER_ATOMS)]
    for a, ph in zip(FILTER_ATOMS, fpool):
        lex[(S.FILTER, a)] = ph
    spool = list(pool[: len(SCOPE_ATOMS)])       # deliberate reuse of phrases
    rg.shuffle(spool)
    for a, ph in zip(SCOPE_ATOMS, spool):
        lex[(S.SCOPE, a)] = ph
    opool = pool[len(FILTER_ATOMS): len(FILTER_ATOMS) + len(OP_ATOMS)]
    for a, ph in zip(OP_ATOMS, opool):
        lex[(S.ROOT, a)] = ph

    orders = []
    for ctx in ("keep", "remove"):
        perm = list(S.CHILD_SLOTS)
        if rg.random() < 0.5:
            perm.reverse()
        orders.append((ctx, tuple(perm)))

    senses: dict[tuple[str, ...], list[tuple[str, str]]] = {}
    for (slot, atom), ph in lex.items():
        senses.setdefault(ph, []).append((slot, atom))

    return ConventionSpec(
        lexical_map=tuple(sorted((k, v) for k, v in lex.items())),
        order_rules=tuple(orders),
        contextual_senses=tuple(sorted(
            (p, tuple(sorted(v))) for p, v in senses.items())),
        phrase_rules=tuple(sorted(
            (a, lex[(S.FILTER, a)]) for a in FILTER_ATOMS
            if len(lex[(S.FILTER, a)]) > 1)),
        optional_word_rules=tuple(
            (w, round(rg.choice([0.0, 0.25, 0.5]), 2))
            for w in FUNCTION_WORDS),
        argument_drop_rules=((S.SCOPE, round(rg.choice([0.0, 0.3]), 2)),),
        attachment_rules=tuple(
            (ctx, rg.choice(["pre", "post"])) for ctx in ("keep", "remove")),
    )


# ------------------------------------------------------------- audit

def lexical_incidence(phi: ConventionSpec) -> dict[tuple[str, ...], tuple]:
    """Which (slot, atom) pairs a phrase can realise. Two conventions with
    the same incidence signature are candidates for observational
    equivalence and must be scored as a class."""
    out: dict[tuple[str, ...], list] = {}
    for (slot, atom), ph in phi.lex().items():
        out.setdefault(ph, []).append((slot, atom))
    return {p: tuple(sorted(v)) for p, v in out.items()}


def structural_audit(phi: ConventionSpec) -> dict:
    """Frozen structural criteria only. A convention is rejected here for
    being malformed, never for being hard -- rejecting a hard one because an
    arm failed would select the test set on the result."""
    lex = phi.lex()
    per_role_collision = {}
    for slot in (S.FILTER, S.SCOPE, S.ROOT):
        seen: dict[tuple[str, ...], list[str]] = {}
        for (sl, atom), ph in lex.items():
            if sl == slot:
                seen.setdefault(ph, []).append(atom)
        per_role_collision[slot] = {p: v for p, v in seen.items()
                                    if len(v) > 1}
    inc = lexical_incidence(phi)
    cross_role = {p: v for p, v in inc.items() if len(v) > 1}
    malformed = any(per_role_collision.values())
    return {
        "digest": phi.digest(),
        "malformed": malformed,
        "reason": "phrase ambiguous WITHIN a role" if malformed else "",
        "cross_role_ambiguous_phrases": len(cross_role),
        "distinct_phrases": len(inc),
        "incidence": {" ".join(p): list(map(list, v)) for p, v in inc.items()},
    }


def equivalence_class(phi: ConventionSpec, family) -> tuple[str, ...]:
    """Conventions in `family` with the same lexical incidence signature.
    The evaluator scores posterior mass on this class, not on the identity
    of one draw -- an automorphism is not an error."""
    mine = tuple(sorted((tuple(p), v) for p, v in lexical_incidence(phi).items()))
    out = []
    for other in family:
        sig = tuple(sorted((tuple(p), v)
                           for p, v in lexical_incidence(other).items()))
        if sig == mine:
            out.append(other.digest())
    return tuple(sorted(out))
