"""X65A-L1: exact sketch sufficiency and honest retrieval accounting.

This module is deliberately separate from :mod:`x65a.latent_id`.  X65A-L
used a useful development implementation, but its exactness check compared a
mask with itself and its four-node claim did not charge the eight exact
summaries used to nominate that four.  L1 needs two narrower objects:

* an algebraic certificate for the grounded-pair statistic, corroborated by
  exact end-to-end differential checks; and
* two retrieval protocols whose physical reads are explicit.

The theorem is conditional on the authored X64H model.  Persistent evidence
is restricted to grounded three-role calibration pairs.  Such an observation
has indicator likelihood, so the stored convention posterior is uniform on

    S_g = {phi : for every (z, u) in g, u3[phi, z] == u}.

For current evidence e=(u,D,pool), let W_e(phi,z) be the X64H-0C
selection-aware likelihood.  Then

    q(phi | g) = 1[phi in S_g] / |S_g|
    L(e | g)   = sum_{phi in S_g,z in D} W_e(phi,z) / (|S_g| |D|).

Both expressions depend on a full record only through ``g``.  A clarification
answer ``(zq, a)`` replaces S_g by its intersection with
``{phi: u3[phi,zq] == a}``, so the same statement holds after every reachable
clarification path.  Importantly, W_e can be nonuniform inside S_g.  Query
utility below therefore uses ``q(phi | g,e) proportional to sum_z W_e``;
uniformly counting surviving conventions is not the claimed model.

No floating-point value is part of serializable state in this module.
Selection weights, posteriors, likelihoods, and answer distributions are exact
integers/Fractions.  Decimal logarithms are used only as an ephemeral ordering
of exact answer distributions when choosing a maximum-entropy question.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from x64h import audit0c as A0
from x64h import episode as EP
from x64h import family as F

from . import l_suite as LS
from .latent_id import (ACTIVE_BYTES, ASSIGN_EXISTING, CREATE_NEW,
                        GROUNDING_FOR_NEW, IdentitySketch, NEW_IDENTITY,
                        OUT_OF_FAMILY, P_NEW, QUARANTINE_OUT,
                        RETRIEVAL_BYTES, UNRESOLVED_IDENTITY, sketch_of)
from .provisional import MISSING, PRIOR_OUT, THETA_PROMOTE
from .semantic_mem import GroundedObservation, surviving_mask
from .types import TaintError, byte_cost, encode


PROOF_COVERAGE = (
    "stored_posterior",
    "selection_aware_likelihood",
    "task_posterior",
    "clarification",
    "query_utility",
    "decision",
    "new_identity",
    "out_of_family",
)

PROOF_DOCUMENT = Path(__file__).with_name("X65A-L1-SUFFICIENCY-PROOF.md")


@dataclass(frozen=True)
class FactorOverride:
    """Exact planted replacement for one persistent likelihood entry."""

    z: int
    u: int
    phi: int
    value: Fraction

    def canon(self):
        return {"z": self.z, "u": self.u, "phi": self.phi,
                "value": self.value}


@dataclass(frozen=True)
class HiddenPersistentWeight:
    name: str
    phi: int
    value: Fraction

    def canon(self):
        return {"name": self.name, "phi": self.phi, "value": self.value}


@dataclass(frozen=True)
class SufficiencyDomainSpec:
    """All premises on which the finite-algebra theorem is conditional.

    Overrides and hidden weights are empty in the authored model.  They are
    first-class inputs so the *same* validator can reject countermodels rather
    than trusting a caller-supplied label saying that evidence is indicator.
    """

    overlap: str
    sketch_fields: tuple[str, ...] = ("z", "u")
    current_likelihood_record_dependencies: tuple[str, ...] = ()
    factor_overrides: tuple[FactorOverride, ...] = ()
    hidden_persistent_weights: tuple[HiddenPersistentWeight, ...] = ()

    def canon(self):
        return {
            "overlap": self.overlap,
            "sketch_fields": list(self.sketch_fields),
            "current_likelihood_record_dependencies":
                list(self.current_likelihood_record_dependencies),
            "factor_overrides": [x.canon() for x in self.factor_overrides],
            "hidden_persistent_weights":
                [x.canon() for x in self.hidden_persistent_weights],
        }


@dataclass(frozen=True)
class DomainValidation:
    overlap: str
    checks: dict[str, bool]
    convention_count: int
    meaning_count: int
    legal_grounded_pairs: int
    persistent_factor_entries_checked: int
    override_entries_checked: int
    proof_document_sha256: str

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(name for name, ok in self.checks.items() if not ok)

    def canon(self):
        return {"overlap": self.overlap, "checks": self.checks,
                "convention_count": self.convention_count,
                "meaning_count": self.meaning_count,
                "legal_grounded_pairs": self.legal_grounded_pairs,
                "persistent_factor_entries_checked":
                    self.persistent_factor_entries_checked,
                "override_entries_checked": self.override_entries_checked,
                "proof_document_sha256": self.proof_document_sha256,
                "passed": self.passed,
                "failed_checks": list(self.failed_checks)}


def authored_sufficiency_domain(overlap: str) -> SufficiencyDomainSpec:
    return SufficiencyDomainSpec(overlap=overlap)


def _proof_document_digest() -> str:
    if not PROOF_DOCUMENT.is_file():
        return ""
    return hashlib.sha256(PROOF_DOCUMENT.read_bytes()).hexdigest()


@lru_cache(maxsize=16)
def validate_sufficiency_domain(spec: SufficiencyDomainSpec
                                ) -> DomainValidation:
    """Exhaustively check theorem premises over one finite X64H stratum.

    Every realizable grounded pair and every convention/factor entry is
    covered.  The check is a premise validator, not a proof assistant: the
    finite-algebra derivation using these premises lives in the tracked proof
    document.
    """

    fam = F.Family(F.FamilySpec(overlap=spec.overlap))
    overrides = {(x.z, x.u, x.phi): x.value for x in spec.factor_overrides}
    unique_override_keys = len(overrides) == len(spec.factor_overrides)
    override_targets_legal = True
    indicator = True
    support_reconstructed = True
    production_mask_matches = True
    sample_pair: tuple[int, int] | None = None
    legal_pairs = factor_entries = 0
    seen_overrides: set[tuple[int, int, int]] = set()

    for z in range(fam.m):
        codes = fam.u3[:, z]
        for u_raw in np.unique(codes):
            u = int(u_raw)
            if sample_pair is None:
                sample_pair = (z, u)
            legal_pairs += 1
            factor_entries += fam.n
            expected_support = codes == u
            # Default authored factors are exact 0/1 indicators.  Sparse
            # overrides let the same exhaustive loop validate planted models.
            actual_support = expected_support.copy()
            for (oz, ou, phi), value in overrides.items():
                if oz != z or ou != u:
                    continue
                if not (0 <= phi < fam.n):
                    override_targets_legal = False
                    continue
                seen_overrides.add((oz, ou, phi))
                if value not in (Fraction(0), Fraction(1)):
                    indicator = False
                actual_support[phi] = value > 0
            if not np.array_equal(actual_support, expected_support):
                support_reconstructed = False
            production = surviving_mask(
                fam, (GroundedObservation(z, u, "premise-check"),))
            if not np.array_equal(production, expected_support):
                production_mask_matches = False

    for key, value in overrides.items():
        z, u, phi = key
        if not (0 <= z < fam.m and 0 <= phi < fam.n
                and int(fam.u3[phi, z]) == u):
            override_targets_legal = False
        if value not in (Fraction(0), Fraction(1)):
            indicator = False

    uniform_mass = Fraction(1, fam.n)
    if sample_pair is None:
        raise TaintError("finite family exposed no legal grounded pair")
    sample_sketch = IdentitySketch((sample_pair,))
    sketch_schema_matches = (
        set(sample_sketch.canon()) == {"p"}
        and tuple(tuple(x) for x in sample_sketch.canon()["p"])
            == (sample_pair,))
    current_signature = tuple(inspect.signature(
        exact_selection_weights).parameters)
    checks = {
        "finite_nonempty_family": fam.n > 0 and fam.m > 0,
        "uniform_prior_exactly_normalized": uniform_mass * fam.n == 1,
        "all_persistent_factors_are_indicators": indicator,
        "grounded_pair_support_reconstructed_exactly": support_reconstructed,
        "production_surviving_mask_matches_indicator_table":
            production_mask_matches,
        "sketch_retains_exactly_grounded_pair_keys":
            spec.sketch_fields == ("z", "u") and sketch_schema_matches,
        "no_hidden_persistent_weights":
            len(spec.hidden_persistent_weights) == 0,
        "current_likelihood_is_record_independent":
            (len(spec.current_likelihood_record_dependencies) == 0
             and current_signature == ("fam", "task")),
        "override_keys_are_unique": unique_override_keys,
        "override_targets_are_legal_and_checked":
            override_targets_legal and len(seen_overrides) == len(overrides),
        "tracked_mathematical_proof_present":
            len(_proof_document_digest()) == 64,
    }
    return DomainValidation(
        spec.overlap, checks, fam.n, fam.m, legal_pairs, factor_entries,
        len(seen_overrides), _proof_document_digest())


def planted_nonindicator_hidden_weight_domain(
        overlap: str) -> SufficiencyDomainSpec:
    """Support-preserving countermodel the grounded-pair sketch cannot encode."""

    fam = F.Family(F.FamilySpec(overlap=overlap))
    for z in range(fam.m):
        codes = fam.u3[:, z]
        for u_raw in np.unique(codes):
            support = np.flatnonzero(codes == u_raw)
            if len(support) >= 2:
                phi = int(support[0])
                hidden = HiddenPersistentWeight("speaker_reliability", phi,
                                                Fraction(1, 2))
                override = FactorOverride(z, int(u_raw), phi, Fraction(1, 2))
                return SufficiencyDomainSpec(
                    overlap=overlap, factor_overrides=(override,),
                    hidden_persistent_weights=(hidden,))
    raise TaintError("could not construct non-indicator countermodel")


@dataclass(frozen=True)
class CountermodelWitness:
    overlap: str
    z: int
    u: int
    phi: int
    support_size: int
    full_mass: Fraction
    sketch_mass: Fraction
    support_preserved: bool

    @property
    def posterior_gap(self) -> bool:
        return self.full_mass != self.sketch_mass

    def canon(self):
        return {"overlap": self.overlap, "z": self.z, "u": self.u,
                "phi": self.phi, "support_size": self.support_size,
                "full_mass": self.full_mass, "sketch_mass": self.sketch_mass,
                "support_preserved": self.support_preserved,
                "posterior_gap": self.posterior_gap}


def countermodel_witness(spec: SufficiencyDomainSpec) -> CountermodelWitness:
    if len(spec.factor_overrides) != 1:
        raise TaintError("countermodel witness requires exactly one override")
    override = spec.factor_overrides[0]
    fam = F.Family(F.FamilySpec(overlap=spec.overlap))
    support = np.flatnonzero(fam.u3[:, override.z] == override.u)
    if override.phi not in support or override.value <= 0:
        raise TaintError("plant must preserve the grounded-pair support")
    normalizer = Fraction(len(support) - 1) + override.value
    return CountermodelWitness(
        spec.overlap, override.z, override.u, override.phi, len(support),
        override.value / normalizer, Fraction(1, len(support)), True)


@dataclass(frozen=True)
class ProofObligation:
    name: str
    equality: str
    reason: str

    def canon(self):
        return {"name": self.name, "equality": self.equality,
                "reason": self.reason}


@dataclass(frozen=True)
class AlgebraicSufficiencyCertificate:
    """Index to the proof and executable validations of all its premises.

    The derivation is a mathematical finite-algebra proof in the tracked
    document.  It was not checked by Lean or another proof assistant.  Unlike
    the former prose-label certificate, ``valid`` requires exhaustive premise
    validation in both authored strata.  Generated-path differential checks
    are explicitly corroboration and do not make this certificate valid.
    """

    theorem: str
    assumptions: tuple[str, ...]
    obligations: tuple[ProofObligation, ...]
    domain_validations: tuple[DomainValidation, ...]
    proof_document: str
    proof_document_sha256: str
    arithmetic: str = "exact integer counts and Fraction posteriors"
    proof_kind: str = "mathematical finite-algebra proof"
    proof_assistant_verified: bool = False
    differential_role: str = "corroboration only"

    def valid(self) -> bool:
        names = tuple(o.name for o in self.obligations)
        return (set(names) == set(PROOF_COVERAGE)
                and len(names) == len(set(names))
                and {d.overlap for d in self.domain_validations}
                    == {"shared", "disjoint_op"}
                and all(d.passed for d in self.domain_validations)
                and len(self.proof_document_sha256) == 64
                and all(d.proof_document_sha256 == self.proof_document_sha256
                        for d in self.domain_validations)
                and not self.proof_assistant_verified
                and self.differential_role == "corroboration only")

    def canon(self):
        return {"theorem": self.theorem,
                "assumptions": list(self.assumptions),
                "obligations": [o.canon() for o in self.obligations],
                "domain_validations": [d.canon()
                                       for d in self.domain_validations],
                "proof_document": self.proof_document,
                "proof_document_sha256": self.proof_document_sha256,
                "arithmetic": self.arithmetic,
                "proof_kind": self.proof_kind,
                "proof_assistant_verified": self.proof_assistant_verified,
                "differential_role": self.differential_role,
                "valid": self.valid()}


def sufficiency_certificate() -> AlgebraicSufficiencyCertificate:
    validations = tuple(validate_sufficiency_domain(
        authored_sufficiency_domain(overlap))
        for overlap in ("shared", "disjoint_op"))
    return AlgebraicSufficiencyCertificate(
        theorem=("grounded-pair sketch g determines S_g and is sufficient "
                 "for every legal current-evidence inference in X65A-L1"),
        assumptions=(
            "grounded calibration only",
            "indicator calibration likelihood",
            "uniform family prior before grounded evidence",
            "selection-aware current likelihood is record-independent",
            "NEW uses the family prior and OUT uses the frozen OTHER channel",
        ),
        obligations=(
            ProofObligation(
                "stored_posterior",
                "q_full(phi|g)=q_sketch(phi|g)=1[phi in S_g]/|S_g|",
                "all persistent factors are grounded indicator factors"),
            ProofObligation(
                "selection_aware_likelihood",
                "L_full(e)=L_sketch(e)=sum_{S_g,D}W_e/(|S_g||D|)",
                "W_e depends on current evidence and phi,z, not metadata"),
            ProofObligation(
                "task_posterior",
                "p_full(z|e,g)=p_sketch(z|e,g)",
                "both normalize identical column sums of W_e over S_g"),
            ProofObligation(
                "clarification",
                "S_(g+(zq,a))=S_g intersect {phi:u3[phi,zq]=a}",
                "a legal answer is another grounded indicator factor"),
            ProofObligation(
                "query_utility",
                "p_full(a|zq,e,g)=p_sketch(a|zq,e,g)",
                "answer mass uses q(phi|g,e), including W_e reweighting"),
            ProofObligation(
                "decision",
                "delta(full inference)=delta(sketch inference)",
                "thresholding and tie order receive identical posteriors"),
            ProofObligation(
                "new_identity",
                "L_NEW and queried S_NEW are independent of old metadata",
                "NEW starts at the family support and intersects answers"),
            ProofObligation(
                "out_of_family",
                "L_OUT=1/A^2 is independent of every stored record",
                "the frozen OTHER channel carries no record statistic"),
        ),
        domain_validations=validations,
        proof_document=str(PROOF_DOCUMENT.relative_to(PROOF_DOCUMENT.parents[2])),
        proof_document_sha256=_proof_document_digest())


@dataclass(frozen=True)
class ExactSelectionWeights:
    """Selection-aware W represented with one exact common denominator."""

    live: tuple[int, ...]
    scaled: np.ndarray
    denominator: int

    def weight(self, phi: int, column: int) -> Fraction:
        return Fraction(int(self.scaled[phi, column]), self.denominator)

    def row_scores(self, support: tuple[int, ...]) -> np.ndarray:
        if not support:
            return np.zeros(0, dtype=np.int64)
        return self.scaled[np.asarray(support, dtype=np.int64)].sum(axis=1)


_WEIGHT_CACHE: dict[tuple, ExactSelectionWeights] = {}


def family_signature(fam) -> tuple:
    """Content signature for cache isolation; never use recyclable ``id``.

    X64H's family is deterministic, but audit fixtures can keep constructing
    and discarding instances.  CPython may then reuse an object address across
    strata.  The frozen X64H cache keys on that address, which is unsafe for a
    long interleaved L1 audit.  L1 keys on the actual family content instead.
    """

    digest = hashlib.sha256()
    for array in (fam.PO, fam.PF, fam.PS, fam.ORD):
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes())
    return (fam.spec.overlap, int(fam.spec.n_words), int(fam.A), int(fam.n),
            int(fam.m), digest.hexdigest())


def exact_selection_weights(fam, task) -> ExactSelectionWeights:
    """Recompute X64H-0C's selection likelihood from integer counts.

    This mirrors ``x64h.audit0c.selection_weights`` but delays division.  Its
    result is also checked against that frozen implementation by the
    differential audit.
    """

    live = tuple(int(z) for z in task.live)
    key = (family_signature(fam), live, int(task.u), tuple(task.pool))
    hit = _WEIGHT_CACHE.get(key)
    if hit is not None:
        return hit
    matrices = [fam.codes(p)[:, list(live)] for p in task.pool]
    n, k = fam.n, len(live)
    num = np.zeros((n, k), dtype=np.int16)
    den = np.zeros((n, k), dtype=np.int16)
    for a in range(k):
        for pi in range(len(task.pool)):
            candidate = matrices[pi][:, a]
            hits = np.zeros((n, k), dtype=bool)
            for q in range(len(task.pool)):
                hits |= matrices[q] == candidate[:, None]
            qualifies = (hits.sum(axis=1) == 1) & hits[:, a]
            den[:, a] += qualifies
            num[:, a] += qualifies & (candidate == task.u)
    common = 1
    for d in np.unique(den[den > 0]):
        common = math.lcm(common, int(d))
    scaled = np.zeros((n, k), dtype=np.int64)
    ok = den > 0
    scaled[ok] = num[ok].astype(np.int64) * (common // den[ok])
    out = ExactSelectionWeights(live, scaled, common)
    if len(_WEIGHT_CACHE) < 5000:
        _WEIGHT_CACHE[key] = out
    return out


def _frozen_selection_weights_uncached(fam, task) -> np.ndarray:
    """Call the frozen model without trusting its recyclable-object-id cache.

    This removes only the current runtime entry.  No frozen source or result is
    modified.  It is used solely for the corroborating cross-implementation
    comparison; L1 inference itself uses ``exact_selection_weights``.
    """

    live = tuple(int(z) for z in task.live)
    frozen_key = (id(fam), live, int(task.u), tuple(task.pool))
    A0._WCACHE.pop(frozen_key, None)
    return A0.selection_weights(fam, list(live), task.u, task.pool)


def support_from_full_record(fam, grounded: Iterable) -> tuple[int, ...]:
    """Independent full-record path: apply each evidence object directly."""

    mask = np.ones(fam.n, dtype=bool)
    for observation in grounded:
        mask &= fam.u3[:, int(observation.z)] == int(observation.u)
    return tuple(int(x) for x in np.flatnonzero(mask))


def support_from_sketch(fam, sketch: IdentitySketch) -> tuple[int, ...]:
    """Sketch path: use only serialized ``(z,u)`` primitives."""

    mask = np.ones(fam.n, dtype=bool)
    for z, u in sketch.pairs:
        mask &= fam.u3[:, int(z)] == int(u)
    return tuple(int(x) for x in np.flatnonzero(mask))


def clarify_support(fam, support: tuple[int, ...], zq: int,
                    answer: int) -> tuple[int, ...]:
    if not support:
        return ()
    idx = np.asarray(support, dtype=np.int64)
    return tuple(int(x) for x in idx[fam.u3[idx, zq] == answer])


def stored_posterior(support: tuple[int, ...]) -> tuple[tuple[int, Fraction], ...]:
    if not support:
        return ()
    p = Fraction(1, len(support))
    return tuple((phi, p) for phi in support)


def exact_record_likelihood(weights: ExactSelectionWeights,
                            support: tuple[int, ...]) -> Fraction:
    if not support or not weights.live:
        return Fraction(0)
    total = int(weights.scaled[np.asarray(support, dtype=np.int64)].sum())
    return Fraction(total,
                    weights.denominator * len(support) * len(weights.live))


def _component_task_posterior(weights: ExactSelectionWeights,
                              support: tuple[int, ...]) -> dict[int, Fraction]:
    if not support:
        return {}
    cols = weights.scaled[np.asarray(support, dtype=np.int64)].sum(axis=0)
    total = int(cols.sum())
    if total <= 0:
        return {}
    return {z: Fraction(int(cols[i]), total)
            for i, z in enumerate(weights.live) if int(cols[i]) > 0}


def exact_identity_posterior(
        fam, supports: tuple[tuple[int, tuple[int, ...]], ...],
        weights: ExactSelectionWeights, new_support: tuple[int, ...],
        p_new: Fraction = P_NEW, p_out: Fraction = PRIOR_OUT,
        with_new: bool = True, with_out: bool = True) -> dict:
    n = len(supports)
    new_prior = p_new if with_new else Fraction(0)
    out_prior = p_out if with_out else Fraction(0)
    record_prior = (Fraction(1) - new_prior - out_prior) / max(1, n)
    unnormalised: dict = {
        key: record_prior * exact_record_likelihood(weights, support)
        for key, support in supports
    }
    if with_new:
        unnormalised[NEW_IDENTITY] = (
            new_prior * exact_record_likelihood(weights, new_support))
    if with_out:
        unnormalised[OUT_OF_FAMILY] = out_prior * Fraction(1, fam.A ** 2)
    total = sum(unnormalised.values(), Fraction(0))
    if total == 0:
        return {key: Fraction(0) for key in unnormalised}
    return {key: value / total for key, value in unnormalised.items()}


def exact_task_posterior(
        supports: tuple[tuple[int, tuple[int, ...]], ...],
        new_support: tuple[int, ...], weights: ExactSelectionWeights,
        identity: Mapping) -> dict[int, Fraction]:
    by_key = dict(supports)
    mixture: dict[int, Fraction] = {}
    for key, identity_mass in identity.items():
        if identity_mass <= 0 or key == OUT_OF_FAMILY:
            continue
        support = new_support if key == NEW_IDENTITY else by_key[key]
        component = _component_task_posterior(weights, support)
        for z, pz in component.items():
            mixture[z] = mixture.get(z, Fraction(0)) + identity_mass * pz
    total = sum(mixture.values(), Fraction(0))
    if total == 0:
        return {}
    return {z: p / total for z, p in mixture.items()}


def query_answer_distribution(
        fam, supports: tuple[tuple[int, tuple[int, ...]], ...],
        new_support: tuple[int, ...], weights: ExactSelectionWeights,
        identity: Mapping, zq: int, *, selection_weighted: bool = True
        ) -> dict[int, Fraction]:
    """Exact p(answer | query,current evidence), conditioned on queryable J.

    OUT is deliberately excluded: X65A-L defines a frozen current-utterance
    likelihood for OTHER but no semantic-answer channel for an alien speaker.
    This restriction is explicit in the certificate rather than silently
    inventing an OUT query model.
    """

    by_key = dict(supports)
    answer_mass: dict[int, Fraction] = {}
    for key, identity_mass in identity.items():
        if identity_mass <= 0 or key == OUT_OF_FAMILY:
            continue
        support = new_support if key == NEW_IDENTITY else by_key[key]
        if not support:
            continue
        idx = np.asarray(support, dtype=np.int64)
        if selection_weighted:
            row_mass = weights.scaled[idx].sum(axis=1).astype(np.int64)
        else:
            row_mass = np.ones(len(idx), dtype=np.int64)
        denominator = int(row_mass.sum())
        if denominator <= 0:
            continue
        codes = fam.u3[idx, zq]
        grouped = np.zeros(fam.A ** 3, dtype=np.int64)
        np.add.at(grouped, codes, row_mass)
        for answer in np.flatnonzero(grouped):
            p = Fraction(int(grouped[answer]), denominator)
            answer_mass[int(answer)] = (
                answer_mass.get(int(answer), Fraction(0)) + identity_mass * p)
    total = sum(answer_mass.values(), Fraction(0))
    if total == 0:
        return {}
    return {answer: mass / total for answer, mass in answer_mass.items()}


def _entropy(distribution: Mapping[int, Fraction]) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 60
        ln2 = Decimal(2).ln()
        out = Decimal(0)
        for p in distribution.values():
            if p <= 0:
                continue
            x = Decimal(p.numerator) / Decimal(p.denominator)
            out -= x * (x.ln() / ln2)
        return +out


def choose_query(
        fam, supports: tuple[tuple[int, tuple[int, ...]], ...],
        new_support: tuple[int, ...], weights: ExactSelectionWeights,
        identity: Mapping, legal: Iterable[int], asked: Iterable[int] = (),
        *, selection_weighted: bool = True) -> tuple[int | None, dict]:
    used = set(int(x) for x in asked)
    best_query = None
    best_distribution: dict = {}
    best_score: Decimal | None = None
    for zq in sorted(int(x) for x in legal if int(x) not in used):
        distribution = query_answer_distribution(
            fam, supports, new_support, weights, identity, zq,
            selection_weighted=selection_weighted)
        score = _entropy(distribution)
        if best_score is None or score > best_score:
            best_query, best_distribution, best_score = zq, distribution, score
    return best_query, best_distribution


def exact_decision(identity: Mapping, supports, queries_asked: int) -> str:
    if not identity:
        return MISSING
    ordered = sorted(identity, key=lambda k: (not isinstance(k, int), str(k)))
    top = max(ordered, key=lambda k: identity[k])
    if (not any(support for _key, support in supports)
            and top != NEW_IDENTITY):
        return MISSING
    if top == OUT_OF_FAMILY:
        return QUARANTINE_OUT
    if top == NEW_IDENTITY:
        return (CREATE_NEW if queries_asked >= GROUNDING_FOR_NEW
                else UNRESOLVED_IDENTITY)
    if identity[top] < THETA_PROMOTE:
        return UNRESOLVED_IDENTITY
    return ASSIGN_EXISTING


def exact_prediction(task, task_posterior: Mapping[int, Fraction]) -> int | None:
    best = None
    best_mass = Fraction(-1)
    for z in task.tie:
        mass = task_posterior.get(int(z), Fraction(0))
        if mass > best_mass:
            best, best_mass = int(z), mass
    return best


@dataclass(frozen=True)
class ExactInferenceSnapshot:
    identity_posterior: dict
    task_posterior: dict
    selected_query: int | None
    selected_query_distribution: dict
    decision: str
    prediction: int | None

    def canon(self):
        return {
            "identity_posterior": {str(k): v for k, v in
                                   sorted(self.identity_posterior.items(),
                                          key=lambda kv: str(kv[0]))},
            "task_posterior": self.task_posterior,
            "selected_query": self.selected_query,
            "selected_query_distribution": self.selected_query_distribution,
            "decision": self.decision,
            "prediction": self.prediction,
        }


def infer_snapshot(fam, task, supports, new_support, legal, asked=()
                   ) -> ExactInferenceSnapshot:
    weights = exact_selection_weights(fam, task)
    identity = exact_identity_posterior(
        fam, supports, weights, new_support)
    task_p = exact_task_posterior(supports, new_support, weights, identity)
    query, distribution = choose_query(
        fam, supports, new_support, weights, identity, legal, asked)
    return ExactInferenceSnapshot(
        identity, task_p, query, distribution,
        exact_decision(identity, supports, len(tuple(asked))),
        exact_prediction(task, task_p))


# ---------------------------------------------------------------- retrieval


@dataclass(frozen=True)
class ExactIndexEntry:
    record_key: int
    sketch: IdentitySketch

    def canon(self):
        # The ordered pair is the charged association between key and summary.
        return [self.record_key, self.sketch.canon()]


@dataclass(frozen=True)
class GlobalExactSketchIndex:
    entries: tuple[ExactIndexEntry, ...]

    def canon(self):
        return {"e": [entry.canon() for entry in self.entries],
                "v": "global_exact_v1"}

    def bytes(self) -> int:
        return byte_cost(self.canon())


def build_global_exact_index(records: Mapping[int, object]
                             ) -> GlobalExactSketchIndex:
    entries = tuple(ExactIndexEntry(int(key), sketch_of(record))
                    for key, record in sorted(records.items()))
    if len({entry.record_key for entry in entries}) != len(entries):
        raise TaintError("duplicate key in exact sketch index")
    return GlobalExactSketchIndex(entries)


@dataclass(frozen=True)
class CoarseIndexEntry:
    record_key: int
    bucket: int

    def canon(self):
        return [self.record_key, self.bucket]


@dataclass(frozen=True)
class CoarseNominationIndex:
    """One-bit stable projection: useful only to nominate, never sufficient."""

    entries: tuple[CoarseIndexEntry, ...]

    def canon(self):
        return {"e": [entry.canon() for entry in self.entries],
                "v": "coarse_1bit_v1"}

    def bytes(self) -> int:
        return byte_cost(self.canon())


def _coarse_bucket(sketch: IdentitySketch) -> int:
    return hashlib.sha256(encode(sketch.canon())).digest()[0] & 1


def build_coarse_index(exact: GlobalExactSketchIndex) -> CoarseNominationIndex:
    return CoarseNominationIndex(tuple(
        CoarseIndexEntry(entry.record_key, _coarse_bucket(entry.sketch))
        for entry in exact.entries))


@dataclass(frozen=True)
class CollisionWitness:
    bucket: int
    left_key: int
    right_key: int
    left_support_digest: str
    right_support_digest: str

    def canon(self):
        return {"bucket": self.bucket, "left_key": self.left_key,
                "right_key": self.right_key,
                "left_support_digest": self.left_support_digest,
                "right_support_digest": self.right_support_digest,
                "distinct_exact_supports":
                    self.left_support_digest != self.right_support_digest}


def _support_digest(support: tuple[int, ...]) -> str:
    return hashlib.sha256(encode(list(support))).hexdigest()


def coarse_collision_witness(index: CoarseNominationIndex,
                             store: Mapping[int, IdentitySketch], fam
                             ) -> CollisionWitness | None:
    for i, left in enumerate(index.entries):
        ls = support_from_sketch(fam, store[left.record_key])
        ld = _support_digest(ls)
        for right in index.entries[i + 1:]:
            if right.bucket != left.bucket:
                continue
            rs = support_from_sketch(fam, store[right.record_key])
            rd = _support_digest(rs)
            if ld != rd:
                return CollisionWitness(left.bucket, left.record_key,
                                        right.record_key, ld, rd)
    return None


@dataclass(frozen=True)
class RetrievalAccounting:
    protocol: str
    index_bytes_scanned: int
    identity_specific_summaries_inspected: int
    identity_likelihoods_evaluated: int
    shortlist_size: int
    full_records_loaded: int
    sketch_bytes_loaded: int
    total_retrieval_bytes: int
    total_retrieval_node_equivalents: int
    incomplete_retrieval: bool
    within_512: bool
    four_node_claim: bool

    def canon(self):
        return self.__dict__.copy()


@dataclass(frozen=True)
class RetrievalAccountingContract:
    """Pure expected physical-accounting facts for one protocol invocation."""

    protocol: str
    mode: str
    identity_count: int
    expected_index_bytes: int
    expected_sketch_bytes_loaded: int
    expected_summaries_inspected: int
    expected_shortlist_size: int
    expected_incomplete_retrieval: bool
    byte_budget: int = RETRIEVAL_BYTES
    node_limit: int = 4

    def canon(self):
        return self.__dict__.copy()


@dataclass(frozen=True)
class RetrievalAccountingValidation:
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(name for name, ok in self.checks.items() if not ok)

    def canon(self):
        return {"checks": self.checks, "passed": self.passed,
                "failed_checks": list(self.failed_checks)}


def validate_retrieval_accounting(
        row: RetrievalAccounting,
        contract: RetrievalAccountingContract
        ) -> RetrievalAccountingValidation:
    """Validate a row using only values supplied in row and pure contract."""

    checks = {
        "protocol_matches_contract": row.protocol == contract.protocol,
        "full_canonical_index_charged":
            row.index_bytes_scanned == contract.expected_index_bytes,
        "loaded_sketch_container_charged":
            row.sketch_bytes_loaded
            == contract.expected_sketch_bytes_loaded,
        "summary_inspection_count_exact":
            row.identity_specific_summaries_inspected
            == contract.expected_summaries_inspected,
        "likelihood_evaluations_cover_every_inspected_summary":
            row.identity_likelihoods_evaluated
            == row.identity_specific_summaries_inspected,
        "node_equivalents_equal_exact_summaries_inspected":
            row.total_retrieval_node_equivalents
            == row.identity_specific_summaries_inspected,
        "shortlist_not_larger_than_inspected_set":
            row.shortlist_size
            <= row.identity_specific_summaries_inspected,
        "shortlist_size_matches_frozen_protocol":
            row.shortlist_size == contract.expected_shortlist_size,
        "no_full_record_load_hidden": row.full_records_loaded == 0,
        "within_512_flag_is_derived_from_total":
            row.within_512
            == (row.total_retrieval_bytes <= contract.byte_budget),
    }
    if contract.mode == "A":
        checks.update({
            "global_exact_scan_inspects_all_identities":
                row.identity_specific_summaries_inspected
                == contract.identity_count,
            "global_exact_scan_charges_physical_index_once":
                row.total_retrieval_bytes == contract.expected_index_bytes,
            "global_exact_scan_makes_no_four_node_claim":
                not row.four_node_claim,
            "global_exact_scan_reports_shortlist_truncation":
                row.incomplete_retrieval
                == contract.expected_incomplete_retrieval,
        })
    elif contract.mode == "B":
        checks.update({
            "coarse_plus_loaded_bytes_sum_to_total":
                row.total_retrieval_bytes
                == (contract.expected_index_bytes
                    + contract.expected_sketch_bytes_loaded),
            "four_record_protocol_respects_node_limit":
                row.identity_specific_summaries_inspected
                <= contract.node_limit,
            "four_record_shortlist_respects_node_limit":
                row.shortlist_size <= contract.node_limit,
            "four_record_claim_is_explicit": row.four_node_claim,
            "truncation_is_reported":
                row.incomplete_retrieval
                == (contract.identity_count
                    > row.identity_specific_summaries_inspected),
        })
    else:
        checks["known_protocol_mode"] = False
    return RetrievalAccountingValidation(checks)


def protocol_a_accounting_contract(
        index: GlobalExactSketchIndex,
        shortlist_size: int = 4) -> RetrievalAccountingContract:
    return RetrievalAccountingContract(
        "A_GLOBAL_EXACT_SCAN", "A", len(index.entries), index.bytes(),
        sum(entry.sketch.bytes() for entry in index.entries),
        len(index.entries), shortlist_size,
        len(index.entries) > shortlist_size)


def protocol_b_accounting_contract(
        index: CoarseNominationIndex, identity_count: int,
        loaded_batch_bytes: int, loaded_count: int
        ) -> RetrievalAccountingContract:
    return RetrievalAccountingContract(
        "B_FOUR_RECORD_COARSE_NOMINATION", "B", identity_count,
        index.bytes(), loaded_batch_bytes, loaded_count, loaded_count,
        identity_count > loaded_count)


def planted_undercharged_exact_index_row(
        valid_protocol_a: RetrievalAccounting) -> RetrievalAccounting:
    """Known-bad row: hides association/container bytes and claims four nodes."""

    return replace(
        valid_protocol_a,
        index_bytes_scanned=valid_protocol_a.sketch_bytes_loaded,
        identity_specific_summaries_inspected=4,
        identity_likelihoods_evaluated=4,
        total_retrieval_bytes=valid_protocol_a.sketch_bytes_loaded,
        total_retrieval_node_equivalents=4,
        four_node_claim=True)


@dataclass(frozen=True)
class RetrievalSelection:
    selected_keys: tuple[int, ...]
    likelihoods: tuple[tuple[int, Fraction], ...]
    accounting: RetrievalAccounting
    collision_witness: CollisionWitness | None = None

    def canon(self):
        return {"selected_keys": list(self.selected_keys),
                "likelihoods": [[key, value]
                                for key, value in self.likelihoods],
                "accounting": self.accounting.canon(),
                "collision_witness": (self.collision_witness.canon()
                                      if self.collision_witness else None)}


def _rank_likelihoods(rows: Iterable[tuple[int, Fraction]], k: int
                      ) -> tuple[tuple[int, Fraction], ...]:
    return tuple(sorted(rows, key=lambda row: (-row[1], row[0]))[:k])


def _protocol_a_rank(index: GlobalExactSketchIndex,
                     likelihoods: tuple[tuple[int, Fraction], ...], task,
                     strategy: str, seed: int) -> tuple[int, ...]:
    """Return a complete deterministic ranking after the charged exact scan."""

    if strategy == "exact_likelihood":
        return tuple(key for key, _value in _rank_likelihoods(
            likelihoods, len(likelihoods)))
    if strategy == "random":
        task_digest = hashlib.sha256(encode({
            "live": list(task.live), "pool": [list(p) for p in task.pool],
            "u": int(task.u)})).hexdigest()
        return tuple(sorted(
            (entry.record_key for entry in index.entries),
            key=lambda key: (hashlib.sha256(encode({
                "seed": int(seed), "task": task_digest,
                "record_key": int(key)})).digest(), key)))
    if strategy == "recency":
        return tuple(sorted(
            (entry.record_key for entry in index.entries), reverse=True))
    if strategy == "surface_nearest":
        return tuple(entry.record_key for entry in sorted(
            index.entries,
            key=lambda entry: (-sum(
                int(u == int(task.u)) for _z, u in entry.sketch.pairs),
                entry.record_key)))
    if strategy == "all_records":
        return tuple(entry.record_key for entry in index.entries)
    raise ValueError(f"unknown Protocol A ranking strategy {strategy!r}")


def retrieve_protocol_a(index: GlobalExactSketchIndex, fam, task, k: int = 4,
                        *, strategy: str = "exact_likelihood", seed: int = 0
                        ) -> RetrievalSelection:
    """Charged global scan with an explicit, possibly truncated shortlist.

    Every strategy inspects all exact summaries and evaluates all identity
    likelihoods.  This lets the audit compare exact-likelihood, random,
    recency, and surface ranking at identical physical resource use.  A
    shortlist smaller than the identity set is reported as incomplete; the
    global scan is never described as a four-node retrieval.
    """

    weights = exact_selection_weights(fam, task)
    likelihoods = tuple(
        (entry.record_key,
         exact_record_likelihood(weights,
                                 support_from_sketch(fam, entry.sketch)))
        for entry in index.entries)
    if not 0 < k <= len(index.entries):
        raise TaintError("Protocol A shortlist must be nonempty and bounded")
    order = _protocol_a_rank(index, likelihoods, task, strategy, seed)
    by_key = dict(likelihoods)
    ranked = tuple((key, by_key[key]) for key in order[:k])
    physical_bytes = index.bytes()
    stats = RetrievalAccounting(
        protocol="A_GLOBAL_EXACT_SCAN",
        index_bytes_scanned=physical_bytes,
        identity_specific_summaries_inspected=len(index.entries),
        identity_likelihoods_evaluated=len(index.entries),
        shortlist_size=len(ranked),
        full_records_loaded=0,
        sketch_bytes_loaded=sum(entry.sketch.bytes()
                                for entry in index.entries),
        total_retrieval_bytes=physical_bytes,
        total_retrieval_node_equivalents=len(index.entries),
        incomplete_retrieval=len(index.entries) > len(ranked),
        within_512=physical_bytes <= RETRIEVAL_BYTES,
        four_node_claim=False)
    validation = validate_retrieval_accounting(
        stats, protocol_a_accounting_contract(index, len(ranked)))
    if not validation.passed:
        raise TaintError("Protocol A accounting invalid: "
                         + ", ".join(validation.failed_checks))
    # Retain every evaluated likelihood, not just the shortlist.  This makes
    # the reported evaluation count auditable and lets resource-matched
    # control rankings reuse the evaluator's scan while still charging that
    # shared scan to every arm.
    return RetrievalSelection(tuple(key for key, _value in ranked),
                              likelihoods, stats)


def rerank_protocol_a(index: GlobalExactSketchIndex, task,
                      evaluated: RetrievalSelection, k: int, *,
                      strategy: str, seed: int = 0) -> RetrievalSelection:
    """Apply another frozen ranking to one already evaluated global scan."""

    expected = tuple(entry.record_key for entry in index.entries)
    got = tuple(key for key, _value in evaluated.likelihoods)
    if set(got) != set(expected) or len(got) != len(expected):
        raise TaintError("Protocol A rerank lacks all evaluated likelihoods")
    if not 0 < k <= len(expected):
        raise TaintError("Protocol A shortlist must be nonempty and bounded")
    order = _protocol_a_rank(
        index, evaluated.likelihoods, task, strategy, seed)
    stats = replace(
        evaluated.accounting,
        shortlist_size=k,
        incomplete_retrieval=len(expected) > k)
    validation = validate_retrieval_accounting(
        stats, protocol_a_accounting_contract(index, k))
    if not validation.passed:
        raise TaintError("Protocol A rerank accounting invalid: "
                         + ", ".join(validation.failed_checks))
    return RetrievalSelection(tuple(order[:k]), evaluated.likelihoods, stats)


def _task_bucket(task) -> int:
    payload = {"live": list(task.live), "pool": [list(p) for p in task.pool],
               "u": int(task.u)}
    return hashlib.sha256(encode(payload)).digest()[0] & 1


def nominate_coarse(index: CoarseNominationIndex, task, k: int = 4
                    ) -> tuple[int, ...]:
    """Nominate using only the nonsufficient index and current public task."""

    target = _task_bucket(task)
    ranked = sorted(index.entries,
                    key=lambda entry: (entry.bucket != target,
                                       entry.record_key))
    return tuple(entry.record_key for entry in ranked[:k])


def _loaded_batch_bytes(rows: Iterable[tuple[int, IdentitySketch]]) -> int:
    # Charge a second ordered container: these are the summaries physically
    # loaded after the coarse scan, including their record associations.
    return byte_cost({"e": [[key, sketch.canon()] for key, sketch in rows],
                      "v": "loaded_sketches_v1"})


def retrieve_protocol_b(index: CoarseNominationIndex,
                        store: Mapping[int, IdentitySketch], fam, task,
                        collision_witness: CollisionWitness | None,
                        k: int = 4) -> RetrievalSelection:
    """Scan a nonsufficient index, then inspect at most four exact sketches."""

    if k > 4:
        raise TaintError("Protocol B may inspect at most four record sketches")
    # The collision is a validation-time calibration.  Re-discovering it here
    # would inspect exact summaries during every task and smuggle those reads
    # past Protocol B's four-record counter.
    witness = collision_witness
    if witness is None or witness.left_support_digest == witness.right_support_digest:
        raise TaintError("coarse index lacks a distinct-support collision; "
                         "nonsufficiency calibration did not fire")
    keys = nominate_coarse(index, task, k)
    loaded = tuple((key, store[key]) for key in keys)
    weights = exact_selection_weights(fam, task)
    likelihoods = tuple(
        (key, exact_record_likelihood(weights,
                                      support_from_sketch(fam, sketch)))
        for key, sketch in loaded)
    ranked = _rank_likelihoods(likelihoods, k)
    index_bytes = index.bytes()
    loaded_bytes = _loaded_batch_bytes(loaded)
    total = index_bytes + loaded_bytes
    stats = RetrievalAccounting(
        protocol="B_FOUR_RECORD_COARSE_NOMINATION",
        index_bytes_scanned=index_bytes,
        identity_specific_summaries_inspected=len(loaded),
        identity_likelihoods_evaluated=len(loaded),
        shortlist_size=len(ranked),
        full_records_loaded=0,
        sketch_bytes_loaded=loaded_bytes,
        total_retrieval_bytes=total,
        total_retrieval_node_equivalents=len(loaded),
        incomplete_retrieval=len(store) > len(loaded),
        within_512=total <= RETRIEVAL_BYTES,
        four_node_claim=True)
    validation = validate_retrieval_accounting(
        stats, protocol_b_accounting_contract(
            index, len(store), loaded_bytes, len(loaded)))
    if not validation.passed:
        raise TaintError("Protocol B accounting invalid: "
                         + ", ".join(validation.failed_checks))
    return RetrievalSelection(tuple(key for key, _value in ranked), ranked,
                              stats, witness)


# ------------------------------------------------ differential corroboration


@dataclass(frozen=True)
class DifferentialAudit:
    overlap: str
    seeds: tuple[int, ...]
    tasks: int
    reachable_states: int
    clarification_answers: int
    exact_comparisons: int
    mismatches: dict[str, int]
    selection_weight_nonuniform_states: int
    weighted_query_differs_from_uniform: int

    @property
    def passed(self) -> bool:
        return (sufficiency_certificate().valid()
                and all(value == 0 for value in self.mismatches.values()))

    def canon(self):
        return {"overlap": self.overlap, "seeds": list(self.seeds),
                "tasks": self.tasks,
                "reachable_states": self.reachable_states,
                "clarification_answers": self.clarification_answers,
                "exact_comparisons": self.exact_comparisons,
                "mismatches": self.mismatches,
                "selection_weight_nonuniform_states":
                    self.selection_weight_nonuniform_states,
                "weighted_query_differs_from_uniform":
                    self.weighted_query_differs_from_uniform,
                "passed": self.passed}


def _generated_probe_sample(probes, limit: int):
    # Stratify first, then fill in stream order.  This keeps the corroboration
    # small while covering returning, ambiguous/misleading, NEW, and OUT where
    # the stratum permits a nonvacuous OUT construction.
    selected = []
    for kind in ("returning", "ambiguous", "misleading", "new",
                 "out_of_family"):
        hit = next((p for p in probes if p.kind == kind and p.task.live), None)
        if hit is not None and hit not in selected:
            selected.append(hit)
    for probe in probes:
        if probe.task.live and probe not in selected:
            selected.append(probe)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _has_nonuniform_current_weights(fam, probe, supports) -> bool:
    weights = exact_selection_weights(fam, probe.task)
    for _key, support in supports:
        positive = {int(x) for x in weights.row_scores(support) if int(x) > 0}
        if len(positive) > 1:
            return True
    return False


def _calibrated_probe_sample(fam, probes, supports, limit: int):
    """Ensure the bounded differential includes a nonuniform-W state.

    A generic stratified prefix can by chance contain only masks on which the
    selection likelihood is constant.  That would make an unweighted query
    implementation pass the most important calibration.  Search the finite
    generated stream and place one nonuniform reachable probe in the bounded
    sample when it exists.
    """

    selected = list(_generated_probe_sample(probes, limit))
    if any(_has_nonuniform_current_weights(fam, p, supports)
           for p in selected):
        return selected
    witness = next((p for p in probes
                    if p.task.live
                    and _has_nonuniform_current_weights(fam, p, supports)),
                   None)
    if witness is None or limit <= 0:
        return selected
    if len(selected) < limit:
        selected.append(witness)
    elif witness not in selected:
        selected[-1] = witness
    return selected


def verify_generated_paths(overlap: str, seeds: tuple[int, ...] = (400,),
                           task_limit: int = 6, query_depth: int = 2
                           ) -> DifferentialAudit:
    """Exact full-record/sketch differential check on reachable query paths."""

    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    cfg = EP.Config(overlap=overlap)
    mismatches = {name: 0 for name in PROOF_COVERAGE}
    mismatches["selection_model"] = 0
    tasks = states = answers = comparisons = nonuniform = query_changed = 0
    for seed in seeds:
        identities = LS.build_identities(fam, seed)
        records = {identity.slot:
                   type("FullRecord", (), {"grounded": identity.grounded})()
                   for identity in identities}
        sketches = {key: sketch_of(record) for key, record in records.items()}
        base_full = tuple(
            (key, support_from_full_record(fam, record.grounded))
            for key, record in sorted(records.items()))
        base_sketch = tuple(
            (key, support_from_sketch(fam, sketches[key]))
            for key in sorted(sketches))
        probes = LS.build_probes(fam, beh, cfg, identities, seed)
        sample = _calibrated_probe_sample(fam, probes, base_full, task_limit)
        for probe in sample:
            task = probe.task
            tasks += 1
            weights = exact_selection_weights(fam, task)
            frozen = _frozen_selection_weights_uncached(fam, task)
            rebuilt = weights.scaled.astype(np.float64) / weights.denominator
            if not np.array_equal(frozen, rebuilt):
                mismatches["selection_model"] += 1
            full = base_full
            sketch = base_sketch
            new_full = tuple(range(fam.n))
            new_sketch = tuple(range(fam.n))
            asked: list[int] = []
            for depth in range(query_depth + 1):
                states += 1
                # Stored posterior and selection-aware likelihood per record.
                for (_kf, sf), (_ks, ss) in zip(full, sketch):
                    comparisons += 2
                    mismatches["stored_posterior"] += (
                        stored_posterior(sf) != stored_posterior(ss))
                    mismatches["selection_aware_likelihood"] += (
                        exact_record_likelihood(weights, sf)
                        != exact_record_likelihood(weights, ss))
                ident_f = exact_identity_posterior(
                    fam, full, weights, new_full)
                ident_s = exact_identity_posterior(
                    fam, sketch, weights, new_sketch)
                comparisons += 3
                mismatches["new_identity"] += (
                    ident_f.get(NEW_IDENTITY) != ident_s.get(NEW_IDENTITY))
                mismatches["out_of_family"] += (
                    ident_f.get(OUT_OF_FAMILY) != ident_s.get(OUT_OF_FAMILY))
                task_f = exact_task_posterior(full, new_full, weights, ident_f)
                task_s = exact_task_posterior(sketch, new_sketch, weights,
                                              ident_s)
                mismatches["task_posterior"] += task_f != task_s
                legal = tuple(z for z in range(fam.m) if z not in asked)
                qf, df = choose_query(fam, full, new_full, weights, ident_f,
                                      legal, asked)
                qs, ds = choose_query(fam, sketch, new_sketch, weights,
                                      ident_s, legal, asked)
                comparisons += 2
                mismatches["query_utility"] += (qf != qs or df != ds)
                decision_f = exact_decision(ident_f, full, len(asked))
                decision_s = exact_decision(ident_s, sketch, len(asked))
                mismatches["decision"] += decision_f != decision_s

                # Audit the fact that current evidence can make phi nonuniform.
                if any(len(set(int(x) for x in
                               weights.row_scores(support)
                               if int(x) > 0)) > 1
                       for _key, support in full):
                    nonuniform += 1
                qu, _du = choose_query(
                    fam, full, new_full, weights, ident_f, legal, asked,
                    selection_weighted=False)
                query_changed += qf != qu

                if depth == query_depth or qf is None or probe.phi_true < 0:
                    break
                answer = int(fam.u3[probe.phi_true, qf])
                full_next = tuple((key, clarify_support(fam, support, qf,
                                                        answer))
                                  for key, support in full)
                sketch_next = tuple((key, clarify_support(fam, support, qf,
                                                          answer))
                                    for key, support in sketch)
                new_full_next = clarify_support(fam, new_full, qf, answer)
                new_sketch_next = clarify_support(fam, new_sketch, qf, answer)
                comparisons += len(full) + 1
                mismatches["clarification"] += full_next != sketch_next
                mismatches["new_identity"] += new_full_next != new_sketch_next
                full, sketch = full_next, sketch_next
                new_full, new_sketch = new_full_next, new_sketch_next
                asked.append(qf)
                answers += 1
    return DifferentialAudit(overlap, tuple(seeds), tasks, states, answers,
                             comparisons, mismatches, nonuniform,
                             query_changed)


def audit_both_strata(seeds: tuple[int, ...] = (400,), task_limit: int = 6,
                      query_depth: int = 2) -> dict[str, DifferentialAudit]:
    return {overlap: verify_generated_paths(overlap, seeds, task_limit,
                                             query_depth)
            for overlap in ("shared", "disjoint_op")}


def active_bytes_with_indexes(records: Mapping[int, object],
                              exact: GlobalExactSketchIndex,
                              coarse: CoarseNominationIndex | None = None
                              ) -> int:
    """Canonical active-state accounting helper; archive is not included."""

    record_bytes = byte_cost({"e": [[key, record.canon()]
                                     for key, record in sorted(records.items())],
                              "v": "confirmed_records_v1"})
    return record_bytes + exact.bytes() + (coarse.bytes() if coarse else 0)


def active_budget_ok(records: Mapping[int, object],
                     exact: GlobalExactSketchIndex,
                     coarse: CoarseNominationIndex | None = None) -> bool:
    return active_bytes_with_indexes(records, exact, coarse) <= ACTIVE_BYTES
