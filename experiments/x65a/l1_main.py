"""One exact open-world inference adapter for every X65A-L1 MAIN audit.

The first L1 evaluator accidentally replaced the production latent model with
an old-record-only mixture.  That made its matched-risk, central retrieval,
safety, and restart sections describe different algorithms.  This module is
the shared adapter.  Its latent state always carries

    J in {stored records, NEW_IDENTITY, OUT_OF_FAMILY}

with the frozen priors and exact selection-aware likelihoods implemented in
``l1_retrieval``.  A stable-ID state is the conditional model in which the
known record has mass one; it uses the same current evidence, convention
likelihood, task loss, tie order, clarification history, and denominator.

X65A-L defines no semantic-answer channel for an alien speaker.  Consequently
OUT participates exactly in the current-utterance identity posterior, while
task prediction and semantic-query utility condition on a queryable in-family
identity.  This is the production L semantics made explicit, not an invented
alien answer model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import l1_inference as INF
from . import l1_retrieval as RET
from .latent_id import NEW_IDENTITY, OUT_OF_FAMILY


INFORMATION_GAIN = "information_gain"
RANDOM = "random"
METRIC_DENOMINATOR = "all_returning_ambiguous_misleading_tasks"


def _support(mask) -> tuple[int, ...]:
    return tuple(int(x) for x in np.flatnonzero(
        np.asarray(mask, dtype=bool)))


def supports_from_masks(masks: Sequence) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple((key, _support(mask)) for key, mask in enumerate(masks))


@dataclass(frozen=True)
class OpenWorldState:
    fam: object = field(repr=False, compare=False)
    task: object = field(repr=False, compare=False)
    supports: tuple[tuple[int, tuple[int, ...]], ...]
    new_support: tuple[int, ...]
    with_new: bool
    with_out: bool
    history: tuple[tuple[int, int], ...] = ()
    weights: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.weights is None:
            object.__setattr__(
                self, "weights", RET.exact_selection_weights(self.fam,
                                                               self.task))
        keys = tuple(key for key, _support0 in self.supports)
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate record key in open-world state")
        if any(tuple(sorted(set(s))) != s for _key, s in self.supports):
            raise ValueError("record supports must be sorted and unique")

    @property
    def asked(self) -> tuple[int, ...]:
        return tuple(z for z, _answer in self.history)

    def identity_posterior(self) -> dict:
        return RET.exact_identity_posterior(
            self.fam, self.supports, self.weights, self.new_support,
            with_new=self.with_new, with_out=self.with_out)

    def task_posterior(self) -> dict[int, Fraction]:
        return RET.exact_task_posterior(
            self.supports, self.new_support, self.weights,
            self.identity_posterior())

    def prediction(self) -> int | None:
        return RET.exact_prediction(self.task, self.task_posterior())

    def task_risk(self, action: int | None) -> Fraction:
        if action is None:
            return Fraction(1)
        return Fraction(1) - self.task_posterior().get(
            int(action), Fraction(0))

    def identity_decision(self) -> str:
        return RET.exact_decision(
            self.identity_posterior(), self.supports, len(self.history))

    def query_distribution(self, zq: int) -> dict[int, Fraction]:
        return RET.query_answer_distribution(
            self.fam, self.supports, self.new_support, self.weights,
            self.identity_posterior(), int(zq))

    def choose_information_query(self, legal: Iterable[int]) -> int | None:
        query, _distribution = RET.choose_query(
            self.fam, self.supports, self.new_support, self.weights,
            self.identity_posterior(), legal, self.asked)
        return query

    def condition(self, zq: int, answer: int) -> "OpenWorldState":
        if int(zq) in self.asked:
            raise ValueError("clarification question repeated")
        supports = tuple(
            (key, RET.clarify_support(self.fam, support, int(zq), int(answer)))
            for key, support in self.supports)
        new_support = RET.clarify_support(
            self.fam, self.new_support, int(zq), int(answer))
        return OpenWorldState(
            self.fam, self.task, supports, new_support,
            self.with_new, self.with_out,
            self.history + ((int(zq), int(answer)),), self.weights)

    def apply_truth(self, zq: int, phi_true: int) -> "OpenWorldState":
        return self.condition(int(zq), int(self.fam.u3[int(phi_true), int(zq)]))

    def convention_posteriors(self) -> dict[int, dict[int, Fraction]]:
        """Current selection-aware posterior within each stored record."""
        out: dict[int, dict[int, Fraction]] = {}
        for key, support in self.supports:
            if not support:
                out[key] = {}
                continue
            rows = self.weights.row_scores(support)
            total = int(rows.sum())
            out[key] = ({phi: Fraction(int(mass), total)
                         for phi, mass in zip(support, rows) if int(mass) > 0}
                        if total > 0 else {})
        return out

    def canon(self) -> dict:
        return {
            "supports": [[key, list(support)]
                         for key, support in self.supports],
            "new_support_size": len(self.new_support),
            "with_new": self.with_new,
            "with_out": self.with_out,
            "history": [list(row) for row in self.history],
            "identity_posterior": {
                str(k): v for k, v in sorted(
                    self.identity_posterior().items(), key=lambda kv: str(kv[0]))},
            "task_posterior": self.task_posterior(),
            "identity_decision": self.identity_decision(),
            "prediction": self.prediction(),
        }


def latent_state(fam, task, masks: Sequence) -> OpenWorldState:
    return OpenWorldState(
        fam, task, supports_from_masks(masks), tuple(range(fam.n)), True, True)


def stable_state(fam, task, true_mask, key: int = 0) -> OpenWorldState:
    return OpenWorldState(
        fam, task, ((int(key), _support(true_mask)),), (), False, False)


def subset_state(fam, task, masks: Sequence, keys: Sequence[int], *,
                 with_new: bool = True, with_out: bool = True) -> OpenWorldState:
    supports = tuple((int(key), _support(masks[int(key)])) for key in keys)
    return OpenWorldState(
        fam, task, supports, tuple(range(fam.n)) if with_new else (),
        with_new, with_out)


@dataclass(frozen=True)
class PolicyRun:
    state: OpenWorldState = field(repr=False, compare=False)
    policy: str
    query_budget: int
    queries_offered: int
    queries_asked: int
    action: int | None
    correct: bool
    confidence: Fraction
    identity_decision: str

    def canon(self) -> dict:
        return {
            "policy": self.policy,
            "query_budget": self.query_budget,
            "queries_offered": self.queries_offered,
            "queries_asked": self.queries_asked,
            "history": [list(v) for v in self.state.history],
            "action": self.action,
            "correct": self.correct,
            "confidence": self.confidence,
            "identity_decision": self.identity_decision,
            "identity_posterior": {
                str(k): v for k, v in sorted(
                    self.state.identity_posterior().items(),
                    key=lambda kv: str(kv[0]))},
        }


def run_policy(initial: OpenWorldState, policy: str, budget: int,
               phi_true: int, z_true: int, legal: Sequence[int], seed: int,
               *, stop_when_identity_decisive: bool = False) -> PolicyRun:
    if budget < 0:
        raise ValueError("negative query budget")
    state = initial
    offered_total = 0
    rng = random.Random(INF.stable_seed(
        "X65A-L1-main", seed, policy, int(initial.task.u),
        tuple(initial.task.live)))
    for _step in range(budget):
        if stop_when_identity_decisive and state.identity_decision() not in (
                "UNRESOLVED_IDENTITY",):
            break
        offered = tuple(int(z) for z in legal if int(z) not in state.asked)
        offered_total += len(offered)
        if not offered:
            break
        if policy == INFORMATION_GAIN:
            query = state.choose_information_query(offered)
        elif policy == RANDOM:
            query = offered[rng.randrange(len(offered))]
        else:
            raise ValueError(f"unknown MAIN query policy {policy!r}")
        if query is None:
            break
        state = state.apply_truth(query, phi_true)
    action = state.prediction()
    post = state.task_posterior()
    confidence = post.get(action, Fraction(0)) if action is not None else Fraction(0)
    return PolicyRun(
        state, policy, budget, offered_total, len(state.history), action,
        action == int(z_true), confidence, state.identity_decision())


def _risk_row(stable: OpenWorldState, latent: OpenWorldState) -> dict:
    stable_action = stable.prediction()
    latent_action = latent.prediction()
    stable_risk = stable.task_risk(stable_action)
    latent_action_risk = stable.task_risk(latent_action)
    return {
        "stable_action": stable_action,
        "latent_action": latent_action,
        "stable_risk": stable_risk,
        "latent_action_risk_under_stable": latent_action_risk,
        "passed": stable_risk <= latent_action_risk,
        "matched_history": stable.history == latent.history,
        "stable_history": stable.history,
        "latent_history": latent.history,
        "latent_has_NEW": NEW_IDENTITY in latent.identity_posterior(),
        "latent_has_OUT": OUT_OF_FAMILY in latent.identity_posterior(),
    }


def matched_risk_audit(stable: OpenWorldState, latent: OpenWorldState,
                       phi_true: int, legal: Sequence[int]) -> dict:
    """Matched q=0, shared q=1, and shared stable-oracle-query audit."""
    q0 = _risk_row(stable, latent)

    query = latent.choose_information_query(legal)
    if query is None:
        stable_q1, latent_q1 = stable, latent
    else:
        stable_q1 = stable.apply_truth(query, phi_true)
        latent_q1 = latent.apply_truth(query, phi_true)
    q1 = _risk_row(stable_q1, latent_q1)

    candidates = tuple(int(z) for z in legal)
    if candidates:
        oracle = min(
            candidates,
            key=lambda z: (stable.apply_truth(z, phi_true).task_risk(
                stable.apply_truth(z, phi_true).prediction()), z))
        stable_o = stable.apply_truth(oracle, phi_true)
        latent_o = latent.apply_truth(oracle, phi_true)
    else:
        oracle, stable_o, latent_o = None, stable, latent
    oracle_row = _risk_row(stable_o, latent_o)
    return {
        "q0": q0,
        "q1": q1,
        "oracle_query": oracle_row,
        "shared_q1_question": query,
        "oracle_question": oracle,
        "all_pass": all(row["passed"] and row["matched_history"]
                        for row in (q0, q1, oracle_row)),
        "model": {
            "latent_components": (
                "stored_records", NEW_IDENTITY, OUT_OF_FAMILY),
            "out_query_semantics": (
                "OUT affects current identity posterior; task/query posterior "
                "conditions on a queryable in-family identity because X65A-L "
                "defines no alien clarification-answer channel"),
        },
    }


def result_vector(rows: Sequence[PolicyRun], field: str) -> tuple:
    return tuple(getattr(row, field) for row in rows)
