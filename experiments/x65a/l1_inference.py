"""Exact, matched inference and clarification machinery for X65A-L1.

This module is deliberately independent of the X65A-L runner.  It provides a
single inference path for stable-identity, latent-identity, and memoryless
audits so that evidence, query answers, stopping rules, and denominators cannot
silently diverge between arms.

The represented joint law is

    p(J, phi, z, e, answers)
      = p(J) q_J(phi) p(z) p(e | phi, z, selected)
        1[phi agrees with semantic answers]
        1[z agrees with behavioural answers].

``q_J`` is uniform on the identity's *original* grounded support.  Answer
conditioning never renormalises identities independently; that was the old
L-arm bug.  Selection-aware likelihoods are retained as exact integer
numerator/denominator tables.  Fractions are used for every reported posterior
and risk.  Entropy necessarily contains logarithms, so it is used only as a
numeric query-ranking calculation and is serialized as decimal text.

Integration entry points
------------------------

``make_stable_state`` / ``make_latent_state``
    Construct matched states from grounded convention masks.

``matched_retrieval_audit``
    Run q=0, a shared q=1 replay, and a shared oracle-query replay.  Latent-arm
    actions are evaluated under the true stable conditional distribution,
    which is the taskwise Bayes-risk comparison licensed by knowing identity.

``memoryless_policy_curves``
    Run the seven L1.5 controls at budgets q=0..4, applying every truthful
    answer and returning complete query/entropy/candidate-class accounting.

``old_semantics_calibration``
    A red calibration: unmatched budgets/history and counted-but-unapplied
    answers must be rejected before a risk comparison is allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

import numpy as np


BEHAVIORAL = "behavioral"
SEMANTIC = "semantic"

RANDOM_LEGAL = "no_memory_random_legal"
BEHAVIORAL_DISAGREEMENT = "no_memory_behavioral_disagreement"
TASK_INFORMATION_GAIN = "no_memory_exact_task_information_gain"
JOINT_INFORMATION_GAIN = "no_memory_exact_convention_task_information_gain"
ORACLE_TASK_SEPARATING = "no_memory_oracle_task_separating"
FRESH_FAMILY_PRIOR = "fresh_x64h_family_prior"
STABLE_ID_FRESH = "stable_id_fresh_no_posterior"

MEMORYLESS_POLICIES = (
    RANDOM_LEGAL,
    BEHAVIORAL_DISAGREEMENT,
    TASK_INFORMATION_GAIN,
    JOINT_INFORMATION_GAIN,
    ORACLE_TASK_SEPARATING,
    FRESH_FAMILY_PRIOR,
    STABLE_ID_FRESH,
)

LATENT_QUANTITY = {
    RANDOM_LEGAL: "mixed_random",
    BEHAVIORAL_DISAGREEMENT: "task_meaning",
    TASK_INFORMATION_GAIN: "task_meaning",
    JOINT_INFORMATION_GAIN: "convention_and_task",
    ORACLE_TASK_SEPARATING: "task_meaning_oracle",
    # These reproduce X64H/S's fresh learner: behavioral clarification
    # narrows the current task meaning while the convention prior stays
    # fresh.  Knowing a stable label without a stored posterior adds no
    # information and must therefore trace identically.
    FRESH_FAMILY_PRIOR: "task_meaning",
    STABLE_ID_FRESH: "task_meaning",
}


def _plain(value):
    """A stable, float-free JSON value for seeds and digests."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in sorted(
            value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(v) for v in value]
    raise TypeError(f"unsupported stable value {type(value).__name__}")


def stable_seed(*parts) -> int:
    """Process-independent RNG seed derived with SHA256, never ``hash``."""
    blob = json.dumps(_plain(parts), sort_keys=True,
                      separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def _fraction_canon(x: Fraction) -> list[int]:
    return [x.numerator, x.denominator]


def _entropy_text(x: float) -> str:
    return format(x, ".12f")


def _normalise(raw: Mapping) -> dict:
    total = sum(raw.values(), Fraction(0))
    if total <= 0:
        return {k: Fraction(0) for k in raw}
    return {k: v / total for k, v in raw.items()}


def _entropy_probs(ps: Iterable[Fraction | float]) -> float:
    out = 0.0
    for p0 in ps:
        p = float(p0)
        if p > 0:
            out -= p * math.log2(p)
    return out


def _task_digest(task) -> str:
    # ``task.z`` is evaluator-only truth and is intentionally absent.
    payload = {
        "kind": str(task.kind),
        "demos": list(task.demos),
        "live": list(task.live),
        "u": int(task.u),
        "pool": [list(p) if isinstance(p, tuple) else p for p in task.pool],
        "tie": list(task.tie),
    }
    blob = json.dumps(_plain(payload), sort_keys=True,
                      separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class ExactSelectionWeights:
    """Exact generator-selection likelihoods for one current utterance.

    ``num[phi, z] / den[phi, z]`` is exactly the X64H-0C selection-aware
    likelihood.  ``scaled / scale`` is an equivalent common-denominator
    integer representation used for fast exact marginalisation.
    """

    live: tuple[int, ...]
    num: np.ndarray = field(repr=False, compare=False)
    den: np.ndarray = field(repr=False, compare=False)
    scaled: np.ndarray = field(repr=False, compare=False)
    scale: int
    digest: str

    def fraction(self, phi: int, z: int) -> Fraction:
        col = self.live.index(int(z))
        d = int(self.den[int(phi), col])
        return (Fraction(0) if d == 0
                else Fraction(int(self.num[int(phi), col]), d))

    def canon(self) -> dict:
        return {"live": list(self.live), "shape": list(self.num.shape),
                "scale": self.scale, "digest": self.digest,
                "representation": "integer_num_den"}


_SELECTION_CACHE: dict[tuple, ExactSelectionWeights] = {}


def exact_selection_weights(fam, live: Sequence[int], u_obs: int,
                            pool: Sequence) -> ExactSelectionWeights:
    """Reproduce ``audit0c.selection_weights`` without floating division."""
    live_t = tuple(int(z) for z in live)
    pool_t = tuple(tuple(p) if isinstance(p, (tuple, list)) else p
                   for p in pool)
    # Keep the family object alive while a cached likelihood refers to it.
    # Keying by ``id(fam)`` is unsafe because the global cache outlives
    # short-lived Family instances: CPython may recycle a shared-family id
    # for a later disjoint-family object and return a table with the wrong
    # convention dimension.
    key = (fam, live_t, int(u_obs), pool_t)
    hit = _SELECTION_CACHE.get(key)
    if hit is not None:
        return hit

    matrices = [fam.codes(p)[:, list(live_t)] for p in pool_t]
    n, k = fam.n, len(live_t)
    num = np.zeros((n, k), dtype=np.int16)
    den = np.zeros((n, k), dtype=np.int16)
    for a in range(k):
        for pi in range(len(pool_t)):
            cand = matrices[pi][:, a]
            hits = np.zeros((n, k), dtype=bool)
            for q in range(len(pool_t)):
                hits |= matrices[q] == cand[:, None]
            qual = (hits.sum(axis=1) == 1) & hits[:, a]
            den[:, a] += qual.astype(np.int16)
            num[:, a] += (qual & (cand == int(u_obs))).astype(np.int16)

    nonzero_den = sorted({int(x) for x in den.ravel() if int(x) > 0})
    scale = 1
    for d in nonzero_den:
        scale = math.lcm(scale, d)
    scaled = np.zeros_like(num, dtype=np.int32)
    ok = den > 0
    # Integer division is exact because ``scale`` is the LCM.
    scaled[ok] = num[ok].astype(np.int32) * (
        scale // den[ok].astype(np.int32))

    for arr in (num, den, scaled):
        arr.flags.writeable = False
    h = hashlib.sha256()
    h.update(json.dumps({"live": live_t, "u": int(u_obs),
                         "pool": _plain(pool_t), "scale": scale},
                        sort_keys=True).encode())
    h.update(num.tobytes())
    h.update(den.tobytes())
    out = ExactSelectionWeights(live_t, num, den, scaled, scale,
                                h.hexdigest())
    if len(_SELECTION_CACHE) < 20000:
        _SELECTION_CACHE[key] = out
    return out


@dataclass(frozen=True, order=True)
class Query:
    kind: str
    item: int

    def __post_init__(self):
        if self.kind not in (BEHAVIORAL, SEMANTIC):
            raise ValueError(f"unknown query kind {self.kind}")

    def canon(self) -> dict:
        return {"kind": self.kind, "item": self.item}


@dataclass(frozen=True)
class AnswerEvent:
    query: Query
    answer: object

    def canon(self) -> dict:
        return {"query": self.query.canon(), "answer": _plain(self.answer)}


@dataclass(frozen=True)
class IdentityComponent:
    """One identity prior and its original grounded convention support."""

    key: str
    prior: Fraction
    support: tuple[int, ...]

    def __post_init__(self):
        if self.prior < 0:
            raise ValueError("identity prior cannot be negative")
        if not self.support:
            raise ValueError("identity support cannot be empty")
        if tuple(sorted(set(self.support))) != self.support:
            raise ValueError("identity support must be sorted and unique")

    def canon(self) -> dict:
        return {"key": self.key, "prior": _fraction_canon(self.prior),
                "support_size": len(self.support),
                "support_digest": hashlib.sha256(
                    json.dumps(list(self.support)).encode()).hexdigest()}


def component_from_mask(key: str, mask, prior: Fraction) -> IdentityComponent:
    idx = tuple(int(x) for x in np.flatnonzero(np.asarray(mask, dtype=bool)))
    return IdentityComponent(str(key), prior, idx)


@dataclass(frozen=True)
class JointState:
    """Immutable exact evidence state shared by every L1 arm."""

    fam: object = field(repr=False, compare=False)
    beh: object = field(repr=False, compare=False)
    task: object = field(repr=False, compare=False)
    components: tuple[IdentityComponent, ...]
    weights: ExactSelectionWeights = field(repr=False, compare=False)
    history: tuple[AnswerEvent, ...] = ()

    def __post_init__(self):
        if tuple(self.task.live) != self.weights.live:
            raise ValueError("selection weights and task candidate pool differ")
        if not self.components:
            raise ValueError("joint state needs at least one identity component")
        if sum((c.prior for c in self.components), Fraction(0)) <= 0:
            raise ValueError("identity prior has zero total mass")

    @property
    def task_digest(self) -> str:
        return _task_digest(self.task)

    @property
    def semantic_answers(self) -> tuple[AnswerEvent, ...]:
        return tuple(e for e in self.history if e.query.kind == SEMANTIC)

    @property
    def behavioral_answers(self) -> tuple[AnswerEvent, ...]:
        return tuple(e for e in self.history if e.query.kind == BEHAVIORAL)

    def _component_indices(self, comp: IdentityComponent) -> np.ndarray:
        idx = np.fromiter(comp.support, dtype=np.int32)
        for event in self.semantic_answers:
            idx = idx[self.fam.u3[idx, event.query.item] == event.answer]
        return idx

    def allowed_meanings(self) -> tuple[int, ...]:
        out = list(self.weights.live)
        for event in self.behavioral_answers:
            out = [z for z in out
                   if self.beh[z][event.query.item] == event.answer]
        return tuple(out)

    def _allowed_cols(self) -> tuple[int, ...]:
        allowed = set(self.allowed_meanings())
        return tuple(i for i, z in enumerate(self.weights.live) if z in allowed)

    def _component_coeff(self, comp: IdentityComponent) -> Fraction:
        # The ORIGINAL support and meaning-prior denominators remain in place
        # after answers.  Conditioning happens only through indicators.
        return (comp.prior
                / (len(comp.support) * len(self.weights.live)
                   * self.weights.scale))

    def normalizer(self) -> Fraction:
        cols = self._allowed_cols()
        if not cols:
            return Fraction(0)
        total = Fraction(0)
        for comp in self.components:
            idx = self._component_indices(comp)
            if len(idx) == 0:
                continue
            mass = int(self.weights.scaled[np.ix_(idx, cols)].sum())
            total += self._component_coeff(comp) * mass
        return total

    def task_posterior(self) -> dict[int, Fraction]:
        cols = self._allowed_cols()
        raw = {z: Fraction(0) for z in self.weights.live}
        for comp in self.components:
            idx = self._component_indices(comp)
            if len(idx) == 0:
                continue
            coeff = self._component_coeff(comp)
            for col in cols:
                mass = int(self.weights.scaled[idx, col].sum())
                raw[self.weights.live[col]] += coeff * mass
        return _normalise(raw)

    def identity_posterior(self) -> dict[str, Fraction]:
        cols = self._allowed_cols()
        raw = {c.key: Fraction(0) for c in self.components}
        for comp in self.components:
            idx = self._component_indices(comp)
            if len(idx) == 0 or not cols:
                continue
            mass = int(self.weights.scaled[np.ix_(idx, cols)].sum())
            raw[comp.key] += self._component_coeff(comp) * mass
        return _normalise(raw)

    def convention_posterior(self) -> dict[int, Fraction]:
        cols = self._allowed_cols()
        raw: dict[int, Fraction] = {}
        for comp in self.components:
            idx = self._component_indices(comp)
            if len(idx) == 0 or not cols:
                continue
            sums = self.weights.scaled[np.ix_(idx, cols)].sum(axis=1)
            coeff = self._component_coeff(comp)
            for phi, mass0 in zip(idx, sums):
                mass = int(mass0)
                if mass:
                    p = int(phi)
                    raw[p] = raw.get(p, Fraction(0)) + coeff * mass
        return _normalise(raw)

    def _numeric_marginal(self, target: str) -> np.ndarray:
        cols = self._allowed_cols()
        if target == "task":
            return np.asarray([float(v) for v in
                               self.task_posterior().values()], dtype=float)
        if target == "identity":
            return np.asarray([float(v) for v in
                               self.identity_posterior().values()], dtype=float)
        if target == "convention":
            raw = np.zeros(self.fam.n, dtype=float)
            for comp in self.components:
                idx = self._component_indices(comp)
                if len(idx) == 0 or not cols:
                    continue
                sums = self.weights.scaled[np.ix_(idx, cols)].sum(axis=1)
                raw[idx] += float(comp.prior / len(comp.support)) * sums
            s = raw.sum()
            return raw / s if s > 0 else raw
        if target == "joint":
            pieces = []
            for comp in self.components:
                idx = self._component_indices(comp)
                if len(idx) == 0 or not cols:
                    continue
                vals = self.weights.scaled[np.ix_(idx, cols)].astype(float)
                vals *= float(comp.prior / len(comp.support))
                pieces.append(vals.ravel())
            if not pieces:
                return np.zeros(0)
            raw = np.concatenate(pieces)
            s = raw.sum()
            return raw / s if s > 0 else raw
        raise ValueError(f"unknown entropy target {target}")

    def entropy(self, target: str) -> float:
        p = self._numeric_marginal(target)
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum()) if len(p) else 0.0

    def condition(self, event: AnswerEvent) -> "JointState":
        if event.query in {e.query for e in self.history}:
            raise ValueError("a clarification question cannot be asked twice")
        if event.query.kind == SEMANTIC:
            if not (0 <= event.query.item < self.fam.m):
                raise ValueError("semantic query outside meaning universe")
        elif event.query.item < 0:
            raise ValueError("negative behavioural query")
        return replace(self, history=self.history + (event,))

    def truthful_event(self, query: Query, phi_true: int,
                       z_true: int) -> AnswerEvent:
        if query.kind == SEMANTIC:
            answer = int(self.fam.u3[int(phi_true), query.item])
        else:
            answer = self.beh[int(z_true)][query.item]
        return AnswerEvent(query, answer)

    def apply_truth(self, query: Query, phi_true: int,
                    z_true: int) -> "JointState":
        return self.condition(self.truthful_event(query, phi_true, z_true))

    def possible_answers(self, query: Query) -> tuple[int, ...]:
        if query.kind == BEHAVIORAL:
            vals = {self.beh[z][query.item]
                    for z in self.allowed_meanings()}
        else:
            vals = set()
            for comp in self.components:
                idx = self._component_indices(comp)
                vals.update(int(x) for x in
                            np.unique(self.fam.u3[idx, query.item]))
        possible = []
        for answer in sorted(vals):
            if self.condition(AnswerEvent(query, answer)).normalizer() > 0:
                possible.append(answer)
        return tuple(possible)

    def answer_distribution(self, query: Query) -> dict[object, Fraction]:
        total = self.normalizer()
        if total <= 0:
            return {}
        out = {}
        for answer in self.possible_answers(query):
            mass = self.condition(AnswerEvent(query, answer)).normalizer()
            if mass > 0:
                out[answer] = mass / total
        return out

    def information_gain(self, query: Query, target: str) -> float:
        before = self.entropy(target)
        after = 0.0
        for answer, p in self.answer_distribution(query).items():
            child = self.condition(AnswerEvent(query, answer))
            after += float(p) * child.entropy(target)
        return before - after

    def answer_entropy(self, query: Query) -> float:
        return _entropy_probs(self.answer_distribution(query).values())

    def candidate_class_count(self,
                              legal_behavioral: Sequence[int]) -> int:
        post = self.task_posterior()
        candidates = [z for z, p in post.items() if p > 0]
        if not candidates:
            return 0
        sigs = {tuple(self.beh[z][k] for k in legal_behavioral)
                for z in candidates}
        return len(sigs)

    def canon(self) -> dict:
        return {
            "task_digest": self.task_digest,
            "selection": self.weights.canon(),
            "components": [c.canon() for c in self.components],
            "history": [e.canon() for e in self.history],
            "normalizer": _fraction_canon(self.normalizer()),
        }


def _state(fam, beh, task, components: Sequence[IdentityComponent]) -> JointState:
    weights = exact_selection_weights(fam, task.live, task.u, task.pool)
    return JointState(fam, beh, task, tuple(components), weights)


def make_stable_state(fam, beh, task, true_mask,
                      key: str = "TRUE_IDENTITY") -> JointState:
    """State conditional on the known true identity, with no phi oracle."""
    return _state(fam, beh, task,
                  (component_from_mask(key, true_mask, Fraction(1)),))


def make_latent_state(fam, beh, task, masks: Sequence,
                      priors: Sequence[Fraction] | None = None,
                      include_fresh: bool = False,
                      fresh_prior: Fraction = Fraction(1, 10)) -> JointState:
    """Exact all-record latent state.

    The optional fresh component is an all-family NEW_IDENTITY hypothesis.
    OUT_OF_FAMILY is intentionally outside this comparison: it has no legal
    ``(phi, z)`` state and belongs in the separate L1.8 safety audit.
    """
    n = len(masks)
    if n == 0:
        raise ValueError("latent inference needs at least one record")
    if priors is None:
        record_mass = Fraction(1) - (fresh_prior if include_fresh else 0)
        priors = tuple(record_mass / n for _ in masks)
    if len(priors) != n:
        raise ValueError("one prior is required per record mask")
    comps = [component_from_mask(f"record:{j}", mask, Fraction(priors[j]))
             for j, mask in enumerate(masks)]
    if include_fresh:
        comps.append(IdentityComponent("NEW_IDENTITY", fresh_prior,
                                       tuple(range(fam.n))))
    return _state(fam, beh, task, comps)


def make_memoryless_state(fam, beh, task,
                          identity_key: str = "FAMILY_PRIOR") -> JointState:
    return _state(fam, beh, task,
                  (IdentityComponent(identity_key, Fraction(1),
                                     tuple(range(fam.n))),))


@dataclass(frozen=True)
class DecisionRule:
    """Bayes-optimal 0/1 decision with an exact abstention loss."""

    abstention_loss: Fraction = Fraction(1)

    def decide(self, state: JointState) -> int | None:
        post = state.task_posterior()
        best = None
        best_p = Fraction(-1)
        for z in state.task.tie:
            p = post.get(int(z), Fraction(0))
            if p > best_p:
                best, best_p = int(z), p
        if best is None or best_p <= 0:
            return None
        action_risk = Fraction(1) - best_p
        return None if self.abstention_loss < action_risk else best

    def risk(self, state: JointState, action: int | None) -> Fraction:
        if action is None:
            return self.abstention_loss
        return Fraction(1) - state.task_posterior().get(
            int(action), Fraction(0))

    def canon(self) -> dict:
        return {"loss": "zero_one_with_abstention",
                "abstention_loss": _fraction_canon(self.abstention_loss)}


@dataclass(frozen=True)
class ArmEvaluation:
    """Protocol metadata coupled to the evidence state it claims to score."""

    state: JointState = field(repr=False, compare=False)
    query_budget: int
    policy: str
    decision_rule: DecisionRule
    metric_denominator: str
    queries_offered: int
    asked: tuple[AnswerEvent, ...]

    def validation_errors(self) -> tuple[str, ...]:
        errors = []
        if self.query_budget < 0:
            errors.append("negative query budget")
        if len(self.asked) > self.query_budget:
            errors.append("asked queries exceed budget")
        if self.queries_offered < len(self.asked):
            errors.append("asked queries exceed offered queries")
        if self.asked != self.state.history:
            errors.append("counted answer was not applied to posterior")
        if len({e.query for e in self.asked}) != len(self.asked):
            errors.append("duplicate clarification query")
        return tuple(errors)

    def canon(self) -> dict:
        return {"query_budget": self.query_budget, "policy": self.policy,
                "decision_rule": self.decision_rule.canon(),
                "metric_denominator": self.metric_denominator,
                "queries_offered": self.queries_offered,
                "queries_asked": len(self.asked),
                "asked": [e.canon() for e in self.asked],
                "valid": not self.validation_errors(),
                "validation_errors": list(self.validation_errors())}


@dataclass(frozen=True)
class RiskComparison:
    matched: bool
    passed: bool
    stable_action: int | None
    latent_action: int | None
    stable_risk: Fraction | None
    latent_action_risk_under_stable: Fraction | None
    mismatches: tuple[str, ...]

    def canon(self) -> dict:
        return {
            "matched": self.matched, "passed": self.passed,
            "stable_action": self.stable_action,
            "latent_action": self.latent_action,
            "stable_risk": (None if self.stable_risk is None else
                            _fraction_canon(self.stable_risk)),
            "latent_action_risk_under_stable": (
                None if self.latent_action_risk_under_stable is None else
                _fraction_canon(self.latent_action_risk_under_stable)),
            "mismatches": list(self.mismatches),
        }


def compare_taskwise_bayes_risk(stable: ArmEvaluation,
                                latent: ArmEvaluation) -> RiskComparison:
    """Compare both actions under the true stable conditional distribution.

    This is the pointwise information-ordering check.  Realized correctness,
    or each arm's self-assessed risk, is not a Bayes-risk ordering.
    """
    mismatch = list(stable.validation_errors()) + list(latent.validation_errors())
    pairs = (
        (stable.state.task_digest, latent.state.task_digest, "current evidence"),
        (stable.state.weights.digest, latent.state.weights.digest,
         "selection-aware likelihood"),
        (stable.query_budget, latent.query_budget, "query budget"),
        (stable.policy, latent.policy, "clarification policy"),
        (stable.decision_rule, latent.decision_rule, "decision/abstention rule"),
        (stable.metric_denominator, latent.metric_denominator,
         "metric denominator"),
        (stable.asked, latent.asked, "applied clarification history"),
    )
    for left, right, name in pairs:
        if left != right:
            mismatch.append(f"unmatched {name}")
    if mismatch:
        return RiskComparison(False, False, None, None, None, None,
                              tuple(dict.fromkeys(mismatch)))

    sa = stable.decision_rule.decide(stable.state)
    la = latent.decision_rule.decide(latent.state)
    sr = stable.decision_rule.risk(stable.state, sa)
    lr = stable.decision_rule.risk(stable.state, la)
    return RiskComparison(True, sr <= lr, sa, la, sr, lr, ())


def legal_questions(state: JointState, legal_behavioral: Sequence[int],
                    legal_semantic: Sequence[int]) -> tuple[Query, ...]:
    asked = {e.query for e in state.history}
    demos = {int(x) for x in state.task.demos}
    out = [Query(BEHAVIORAL, int(k)) for k in legal_behavioral
           if int(k) not in demos]
    out += [Query(SEMANTIC, int(z)) for z in legal_semantic]
    return tuple(q for q in sorted(set(out)) if q not in asked)


def _best_by_score(questions: Sequence[Query], score,
                   maximise: bool = True) -> Query | None:
    if not questions:
        return None
    rows = [(float(score(q)), q) for q in questions]
    if maximise:
        value = max(v for v, _q in rows)
        return min(q for v, q in rows if abs(v - value) <= 1e-12)
    value = min(v for v, _q in rows)
    return min(q for v, q in rows if abs(v - value) <= 1e-12)


def select_policy_question(policy: str, state: JointState,
                           questions: Sequence[Query], rng: random.Random,
                           phi_true: int, z_true: int,
                           decision_rule: DecisionRule) -> Query | None:
    """Select one question; answer application is always done by the caller."""
    if not questions:
        return None
    if policy == RANDOM_LEGAL:
        return questions[rng.randrange(len(questions))]
    if policy == BEHAVIORAL_DISAGREEMENT:
        qs = [q for q in questions if q.kind == BEHAVIORAL]
        return _best_by_score(qs, state.answer_entropy)
    if policy == TASK_INFORMATION_GAIN:
        return _best_by_score(questions,
                              lambda q: state.information_gain(q, "task"))
    if policy == JOINT_INFORMATION_GAIN:
        return _best_by_score(questions,
                              lambda q: state.information_gain(q, "joint"))
    if policy in (FRESH_FAMILY_PRIOR, STABLE_ID_FRESH):
        qs = [q for q in questions if q.kind == BEHAVIORAL]
        return _best_by_score(qs,
                              lambda q: state.information_gain(q, "task"))
    if policy == ORACLE_TASK_SEPARATING:
        qs = [q for q in questions if q.kind == BEHAVIORAL]
        if not qs:
            return None

        def oracle_score(q):
            child = state.apply_truth(q, phi_true, z_true)
            remaining = sum(p > 0 for p in child.task_posterior().values())
            action = decision_rule.decide(child)
            risk = decision_rule.risk(child, action)
            # Exact risk dominates, then remaining candidates.
            return float(risk) + remaining * 1e-9

        return _best_by_score(qs, oracle_score, maximise=False)
    raise ValueError(f"unknown memoryless policy {policy}")


def oracle_legal_question(state: JointState, questions: Sequence[Query],
                          phi_true: int, z_true: int,
                          decision_rule: DecisionRule) -> Query | None:
    """Legal evaluator oracle: minimize true stable-conditional action risk."""
    def risk(q):
        child = state.apply_truth(q, phi_true, z_true)
        return float(decision_rule.risk(child, decision_rule.decide(child)))
    return _best_by_score(questions, risk, maximise=False)


def _evaluation(state: JointState, budget: int, policy: str,
                rule: DecisionRule, denominator: str,
                offered: int) -> ArmEvaluation:
    return ArmEvaluation(state, budget, policy, rule, denominator, offered,
                         state.history)


def matched_retrieval_audit(stable_state: JointState,
                            latent_state: JointState,
                            phi_true: int, z_true: int,
                            legal_behavioral: Sequence[int],
                            legal_semantic: Sequence[int],
                            decision_rule: DecisionRule = DecisionRule(),
                            metric_denominator: str = "all_matched_tasks") -> dict:
    """q=0, shared q=1, and shared oracle-query taskwise risk audits."""
    q0s = _evaluation(stable_state, 0, "matched_q0", decision_rule,
                     metric_denominator, 0)
    q0l = _evaluation(latent_state, 0, "matched_q0", decision_rule,
                     metric_denominator, 0)
    q0 = compare_taskwise_bayes_risk(q0s, q0l)

    offered = legal_questions(latent_state, legal_behavioral, legal_semantic)
    shared_q = _best_by_score(
        offered, lambda q: latent_state.information_gain(q, "task"))
    if shared_q is None:
        q1s_state, q1l_state, q1_offered = stable_state, latent_state, 0
    else:
        event = stable_state.truthful_event(shared_q, phi_true, z_true)
        q1s_state = stable_state.condition(event)
        q1l_state = latent_state.condition(event)
        q1_offered = len(offered)
    q1s = _evaluation(q1s_state, 1, "shared_task_ig_replay",
                     decision_rule, metric_denominator, q1_offered)
    q1l = _evaluation(q1l_state, 1, "shared_task_ig_replay",
                     decision_rule, metric_denominator, q1_offered)
    q1 = compare_taskwise_bayes_risk(q1s, q1l)

    oracle_offered = legal_questions(stable_state, legal_behavioral,
                                     legal_semantic)
    oq = oracle_legal_question(stable_state, oracle_offered, phi_true, z_true,
                               decision_rule)
    if oq is None:
        os_state, ol_state, o_count = stable_state, latent_state, 0
    else:
        event = stable_state.truthful_event(oq, phi_true, z_true)
        os_state = stable_state.condition(event)
        ol_state = latent_state.condition(event)
        o_count = len(oracle_offered)
    os = _evaluation(os_state, 1, "shared_oracle_query_replay",
                     decision_rule, metric_denominator, o_count)
    ol = _evaluation(ol_state, 1, "shared_oracle_query_replay",
                     decision_rule, metric_denominator, o_count)
    oracle = compare_taskwise_bayes_risk(os, ol)
    return {
        "q0": q0,
        "q1": q1,
        "oracle_query": oracle,
        "shared_q1_question": shared_q,
        "oracle_question": oq,
        "all_pass": q0.passed and q1.passed and oracle.passed,
    }


@dataclass(frozen=True)
class ResolutionEffect:
    """What one applied answer actually changed in the exact joint state.

    A policy name says what a question intended to resolve.  This record says
    what its truthful answer *did* resolve.  Support counts and change flags
    are exact/discrete; no entropy float is needed to audit the effect.
    """

    event: AnswerEvent
    identity_changed: bool
    convention_changed: bool
    task_changed: bool
    cause_changed: bool
    identity_support_before: int
    identity_support_after: int
    convention_support_before: int
    convention_support_after: int
    task_support_before: int
    task_support_after: int

    @property
    def resolved_quantities(self) -> tuple[str, ...]:
        flags = (
            ("identity", self.identity_changed),
            ("convention", self.convention_changed),
            ("task", self.task_changed),
            ("cause", self.cause_changed),
        )
        return tuple(name for name, changed in flags if changed)

    def canon(self) -> dict:
        return {
            "event": self.event.canon(),
            "changed": {
                "identity": self.identity_changed,
                "convention": self.convention_changed,
                "task": self.task_changed,
                "cause": self.cause_changed,
            },
            "support": {
                "identity": [self.identity_support_before,
                             self.identity_support_after],
                "convention": [self.convention_support_before,
                               self.convention_support_after],
                "task": [self.task_support_before,
                         self.task_support_after],
            },
            "resolved_quantities": list(self.resolved_quantities),
        }


def resolution_effect(before: JointState, after: JointState,
                      event: AnswerEvent) -> ResolutionEffect:
    """Compute the actual exact-posterior effect of one applied answer."""
    if after.history != before.history + (event,):
        raise ValueError("resolution effect requires exactly one applied answer")
    bi, ai = before.identity_posterior(), after.identity_posterior()
    bc, ac = before.convention_posterior(), after.convention_posterior()
    bt, at = before.task_posterior(), after.task_posterior()
    support = lambda p: sum(v > 0 for v in p.values())
    return ResolutionEffect(
        event,
        bi != ai,
        bc != ac,
        bt != at,
        False,                       # cause is outside the L1.5 state
        support(bi), support(ai),
        support(bc), support(ac),
        support(bt), support(at),
    )


@dataclass(frozen=True)
class PolicyRun:
    policy: str
    latent_quantity: str
    query_budget: int
    state: JointState = field(repr=False, compare=False)
    evaluation: ArmEvaluation = field(repr=False, compare=False)
    offered_each: tuple[int, ...]
    query_types: tuple[tuple[str, int], ...]
    resolution_effects: tuple[ResolutionEffect, ...]
    action: int | None
    correct: bool
    task_risk: Fraction
    task_entropy_before: str
    task_entropy_after: str
    convention_entropy_before: str
    convention_entropy_after: str
    joint_entropy_before: str
    joint_entropy_after: str
    candidate_classes_before: int
    candidate_classes_after: int

    @property
    def queries_asked(self) -> int:
        return len(self.state.history)

    @property
    def queries_offered(self) -> int:
        return sum(self.offered_each)

    @property
    def answers_applied(self) -> bool:
        return not self.evaluation.validation_errors()

    def canon(self) -> dict:
        return {
            "policy": self.policy, "latent_quantity": self.latent_quantity,
            "query_budget": self.query_budget,
            "queries_offered": self.queries_offered,
            "offered_each": list(self.offered_each),
            "queries_asked": self.queries_asked,
            "query_types": dict(self.query_types),
            "answers_applied": self.answers_applied,
            "resolution_effects": [e.canon()
                                   for e in self.resolution_effects],
            "action": self.action, "correct": self.correct,
            "task_risk": _fraction_canon(self.task_risk),
            "task_entropy": {"before": self.task_entropy_before,
                             "after": self.task_entropy_after},
            "convention_entropy": {"before": self.convention_entropy_before,
                                   "after": self.convention_entropy_after},
            "joint_entropy": {"before": self.joint_entropy_before,
                              "after": self.joint_entropy_after},
            "candidate_classes": {"before": self.candidate_classes_before,
                                  "after": self.candidate_classes_after},
            "evaluation": self.evaluation.canon(),
        }


def _policy_snapshot(policy: str, initial: JointState, current: JointState,
                     budget: int, offered_each: Sequence[int], z_true: int,
                     legal_behavioral: Sequence[int], rule: DecisionRule,
                     denominator: str,
                     effects: Sequence[ResolutionEffect]) -> PolicyRun:
    counts = {BEHAVIORAL: 0, SEMANTIC: 0}
    for event in current.history:
        counts[event.query.kind] += 1
    action = rule.decide(current)
    ev = ArmEvaluation(current, budget, policy, rule, denominator,
                       sum(offered_each), current.history)
    return PolicyRun(
        policy, LATENT_QUANTITY[policy], budget, current, ev,
        tuple(offered_each), tuple(sorted(counts.items())), tuple(effects),
        action,
        action == int(z_true), rule.risk(current, action),
        _entropy_text(initial.entropy("task")),
        _entropy_text(current.entropy("task")),
        _entropy_text(initial.entropy("convention")),
        _entropy_text(current.entropy("convention")),
        _entropy_text(initial.entropy("joint")),
        _entropy_text(current.entropy("joint")),
        initial.candidate_class_count(legal_behavioral),
        current.candidate_class_count(legal_behavioral),
    )


def memoryless_policy_curves(fam, beh, task, phi_true: int, z_true: int,
                             legal_behavioral: Sequence[int],
                             legal_semantic: Sequence[int],
                             budgets: Sequence[int] = (0, 1, 2, 3, 4),
                             seed: int = 0,
                             decision_rule: DecisionRule = DecisionRule(),
                             metric_denominator: str = "all_tasks") -> dict:
    """Run all seven L1.5 controls with truthful, applied answers.

    Curves are prefix-consistent: q=2 contains the same first answer as q=1.
    Random selection is paired by a SHA256 seed over public task evidence.
    """
    budgets_t = tuple(sorted(set(int(b) for b in budgets)))
    if not budgets_t or budgets_t[0] < 0:
        raise ValueError("query budgets must be nonnegative")
    max_budget = max(budgets_t)
    curves = {}
    for policy in MEMORYLESS_POLICIES:
        identity_key = ("STABLE_ID_FRESH" if policy == STABLE_ID_FRESH
                        else "FAMILY_PRIOR")
        initial = make_memoryless_state(fam, beh, task, identity_key)
        current = initial
        offered_each: list[int] = []
        effects: list[ResolutionEffect] = []
        rng = random.Random(stable_seed("X65A-L1", seed, policy,
                                       initial.task_digest))
        rows = {}
        if 0 in budgets_t:
            rows[0] = _policy_snapshot(
                policy, initial, current, 0, offered_each, z_true,
                legal_behavioral, decision_rule, metric_denominator, effects)
        for step in range(1, max_budget + 1):
            offered = legal_questions(current, legal_behavioral,
                                      legal_semantic)
            offered_each.append(len(offered))
            query = select_policy_question(policy, current, offered, rng,
                                           phi_true, z_true, decision_rule)
            if query is not None:
                before = current
                event = current.truthful_event(query, phi_true, z_true)
                current = current.condition(event)
                effects.append(resolution_effect(before, current, event))
            if step in budgets_t:
                rows[step] = _policy_snapshot(
                    policy, initial, current, step, offered_each, z_true,
                    legal_behavioral, decision_rule, metric_denominator,
                    effects)
        curves[policy] = rows
    return curves


def old_semantics_calibration(stable_state: JointState,
                              latent_state: JointState,
                              event: AnswerEvent,
                              decision_rule: DecisionRule = DecisionRule()) -> dict:
    """Plant the two old evaluator defects and require both to be rejected."""
    stable_q0 = ArmEvaluation(stable_state, 0, "old_unmatched", decision_rule,
                              "returning", 0, ())
    latent_q1_state = latent_state.condition(event)
    latent_q1 = ArmEvaluation(latent_q1_state, 1, "old_unmatched",
                              decision_rule, "returning", 1,
                              latent_q1_state.history)
    unmatched = compare_taskwise_bayes_risk(stable_q0, latent_q1)

    # This is the old larger-query memoryless semantics: count an answer but
    # leave the posterior untouched.
    unapplied = ArmEvaluation(latent_state, 1, "counted_not_applied",
                              decision_rule, "returning", 1, (event,))
    return {
        "old_unmatched_rejected": not unmatched.matched,
        "answer_not_applied_rejected": bool(unapplied.validation_errors()),
        "unmatched_reasons": unmatched.mismatches,
        "unapplied_reasons": unapplied.validation_errors(),
        "fires": (not unmatched.matched
                  and bool(unapplied.validation_errors())),
    }
