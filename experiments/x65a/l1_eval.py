"""Common task/stream accounting for the X65A-L1 audit."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping

from x64h import episode as EP

from . import l1_inference as I
from . import l_suite as LS


FROZEN_TARGET_ACCURACY = Fraction(19, 20)


def mean(values) -> Fraction:
    values = list(values)
    return (sum((Fraction(v) for v in values), Fraction(0)) / len(values)
            if values else Fraction(0))


def _top(post, order=()):
    if not post:
        return None
    best = max(post.values())
    preferred = [x for x in order if post.get(x) == best]
    if preferred:
        return preferred[0]
    return sorted((k for k, v in post.items() if v == best), key=str)[0]


@dataclass(frozen=True)
class TaskMetric:
    stream_seed: int
    condition: str
    policy: str
    query_budget: int
    correct: bool
    action: int | None
    confidence: Fraction
    false_confident: bool
    identity_top: str | None
    literal_identity: bool
    equivalence_retrieval: bool
    queries_offered: int
    queries_asked: int
    physical_query_types: tuple[tuple[str, int], ...]
    resolved_latent_quantities: tuple[tuple[str, int], ...]
    resolution_effects: tuple[object, ...]
    unresolved: bool
    provisional_branches: int
    established_record_corruption: int = 0
    retrieval: object | None = field(default=None, repr=False, compare=False)

    def canon(self):
        return {
            "stream_seed": self.stream_seed,
            "condition": self.condition,
            "policy": self.policy,
            "query_budget": self.query_budget,
            "correct": self.correct,
            "action": self.action,
            "confidence": self.confidence,
            "false_confident": self.false_confident,
            "identity_top": self.identity_top,
            "literal_identity": self.literal_identity,
            "equivalence_retrieval": self.equivalence_retrieval,
            "queries_offered": self.queries_offered,
            "queries_asked": self.queries_asked,
            "physical_query_types": dict(self.physical_query_types),
            "resolved_latent_quantities": dict(
                self.resolved_latent_quantities),
            "resolution_effects": [e.canon() for e in
                                   self.resolution_effects],
            "unresolved": self.unresolved,
            "provisional_branches": self.provisional_branches,
            "established_record_corruption":
                self.established_record_corruption,
            "retrieval": (self.retrieval.canon()
                          if hasattr(self.retrieval, "canon") else
                          self.retrieval),
        }


def run_policy(initial: I.JointState, policy: str, budget: int,
               phi_true: int, z_true: int,
               legal_behavioral, legal_semantic, seed: int,
               rule: I.DecisionRule = I.DecisionRule(Fraction(1))):
    state = initial
    offered_each = []
    events = []
    effects = []
    resolved = {"identity": 0, "convention": 0,
                "task": 0, "cause": 0}
    physical = {"semantic": 0, "task": 0}
    rng = random.Random(I.stable_seed("l1-policy", seed, policy,
                                     initial.task_digest))
    for _step in range(budget):
        offered = I.legal_questions(state, legal_behavioral, legal_semantic)
        offered_each.append(len(offered))
        q = I.select_policy_question(policy, state, offered, rng,
                                     phi_true, z_true, rule)
        if q is None:
            break
        event = state.truthful_event(q, phi_true, z_true)
        before = state
        state = state.condition(event)
        effect = I.resolution_effect(before, state, event)
        events.append(event)
        effects.append(effect)
        physical["semantic" if q.kind == I.SEMANTIC else "task"] += 1
        resolved["identity"] += int(effect.identity_changed)
        resolved["convention"] += int(effect.convention_changed)
        resolved["task"] += int(effect.task_changed)
    action = rule.decide(state)
    post = state.task_posterior()
    conf = post.get(action, Fraction(0)) if action is not None else Fraction(0)
    return {
        "state": state,
        "events": tuple(events),
        "resolution_effects": tuple(effects),
        "queries_offered": sum(offered_each),
        "offered_each": tuple(offered_each),
        "physical": physical,
        "resolved": resolved,
        "action": action,
        "confidence": conf,
        "correct": action == z_true,
        "false_confident": action != z_true and conf >= Fraction(19, 20),
        "unresolved": action is None,
    }


def task_metric(initial: I.JointState, probe, policy: str, budget: int,
                legal_behavioral, legal_semantic, seed: int,
                equivalence=(), retrieval=None,
                rule: I.DecisionRule = I.DecisionRule(Fraction(1))
                ) -> TaskMetric:
    run = run_policy(initial, policy, budget, probe.phi_true, probe.task.z,
                     legal_behavioral, legal_semantic, seed, rule)
    return _task_metric_from_run(run, probe, policy, budget, seed,
                                 equivalence, retrieval)


def _task_metric_from_run(run, probe, policy: str, budget: int, seed: int,
                          equivalence=(), retrieval=None) -> TaskMetric:
    ident = _top(run["state"].identity_posterior())
    literal = ident == f"record:{probe.slot}"
    eq_keys = {f"record:{j}" for j in equivalence}
    equivalent = ident in eq_keys if eq_keys else literal
    return TaskMetric(
        seed, probe.kind, policy, budget, run["correct"], run["action"],
        run["confidence"], run["false_confident"],
        None if ident is None else str(ident), literal, equivalent,
        run["queries_offered"], len(run["events"]),
        tuple(sorted(run["physical"].items())),
        tuple(sorted(run["resolved"].items())),
        tuple(run["resolution_effects"]), run["unresolved"],
        int(run["unresolved"]), 0, retrieval,
    )


def questions_to_correct(*_args, **_kwargs):
    """Removed: per-task truth-aware hitting time is not an operational metric.

    Use :func:`aggregate_accuracy_curve`, which finds the smallest *budget*
    whose aggregate accuracy reaches the validation-frozen target.
    """
    raise RuntimeError(
        "per-task questions-to-correct is forbidden; use the aggregate "
        "frozen-target accuracy curve")


@dataclass(frozen=True)
class EvaluationCase:
    """One task and initial state, retaining its complete-stream cluster."""

    initial: I.JointState = field(repr=False, compare=False)
    probe: object = field(repr=False, compare=False)
    stream_seed: int
    equivalence: tuple = ()
    retrieval: object | None = field(default=None, repr=False, compare=False)


def matched_scored_population(probes) -> tuple:
    """Every returning/ambiguous/misleading probe with a matched identity."""
    return tuple(p for p in probes if p.slot >= 0 and p.task.live
                 and p.kind in ("returning", "ambiguous", "misleading"))


def distinct_returning_population(probes) -> tuple:
    """One returning probe for every available distinct identity slot."""
    chosen = {}
    for probe in probes:
        if (probe.kind == "returning" and probe.slot >= 0 and probe.task.live
                and probe.slot not in chosen):
            chosen[probe.slot] = probe
    return tuple(chosen[k] for k in sorted(chosen))


def memoryless_population(probes, population: str = "all_matched_scored") -> tuple:
    """Preregistered L1.5 populations; never a positional first-two slice."""
    if population == "all_matched_scored":
        return matched_scored_population(probes)
    if population == "all_distinct_returning_slots":
        return distinct_returning_population(probes)
    raise ValueError(f"unknown memoryless population {population}")


def minimum_questions_to_target(budget_rows: Mapping[int, Mapping],
                                target: Fraction =
                                FROZEN_TARGET_ACCURACY) -> int | None:
    """Smallest aggregate query budget reaching an exact frozen accuracy."""
    target = Fraction(target)
    return next((int(q) for q in sorted(budget_rows)
                 if Fraction(budget_rows[q]["task_accuracy"]) >= target),
                None)


def aggregate_accuracy_curve(
        cases, policy: str, legal_behavioral, legal_semantic,
        budgets=(0, 1, 2, 3, 4), *,
        target: Fraction = FROZEN_TARGET_ACCURACY,
        rule: I.DecisionRule = I.DecisionRule(Fraction(1))) -> dict:
    """Prefix-consistent aggregate accuracy curve over complete task cases.

    Each budget reruns the same public-evidence-seeded policy prefix.  Truth is
    used only to score the aggregate accuracy after a declared budget, never
    to stop an individual task's query sequence.
    """
    cases = tuple(cases)
    budgets = tuple(sorted(set(int(q) for q in budgets)))
    if not cases or not budgets or budgets[0] < 0:
        raise ValueError("non-empty cases and nonnegative budgets required")
    by_budget = {q: [] for q in budgets}
    prefix_consistent = True
    for case in cases:
        previous = ()
        for budget in budgets:
            run = run_policy(
                case.initial, policy, budget, case.probe.phi_true,
                case.probe.task.z, legal_behavioral, legal_semantic,
                case.stream_seed, rule)
            events = run["events"]
            prefix_consistent &= events[:len(previous)] == previous
            previous = events
            by_budget[budget].append(_task_metric_from_run(
                run, case.probe, policy, budget, case.stream_seed,
                case.equivalence, case.retrieval))
    summaries = {q: summarise(rows) for q, rows in by_budget.items()}
    return {
        "policy": policy,
        "frozen_target_accuracy": Fraction(target),
        "population_tasks": len(cases),
        "complete_stream_seeds": tuple(sorted(
            {int(c.stream_seed) for c in cases})),
        "budgets": summaries,
        "prefix_consistent": bool(prefix_consistent),
        "minimum_questions_to_frozen_target":
            minimum_questions_to_target(summaries, Fraction(target)),
        "metric_definition": (
            "minimum declared query budget whose aggregate task accuracy "
            "reaches the validation-frozen target"),
    }


def summarise(rows) -> dict:
    rows = list(rows)
    return {
        "tasks": len(rows),
        "task_accuracy": mean(r.correct for r in rows),
        "equivalence_retrieval": mean(r.equivalence_retrieval for r in rows),
        "literal_identity": mean(r.literal_identity for r in rows),
        "queries_offered": sum(r.queries_offered for r in rows),
        "queries_asked": sum(r.queries_asked for r in rows),
        "mean_queries_all_tasks": mean(r.queries_asked for r in rows),
        "mean_queries_ambiguous_tasks": mean(
            r.queries_asked for r in rows
            if r.condition in ("ambiguous", "misleading")),
        "false_confident_answers": sum(r.false_confident for r in rows),
        "unresolved_outcomes": sum(r.unresolved for r in rows),
        "provisional_branches": sum(r.provisional_branches for r in rows),
        "established_record_corruption": sum(
            r.established_record_corruption for r in rows),
        "physical_query_types": {
            k: sum(dict(r.physical_query_types).get(k, 0) for r in rows)
            for k in ("semantic", "task")
        },
        "resolved_latent_quantities": {
            k: sum(dict(r.resolved_latent_quantities).get(k, 0)
                   for r in rows)
            for k in ("identity", "convention", "task", "cause")
        },
    }


def legacy_query_accounting(overlap="disjoint_op",
                            seeds=(400, 401, 402)) -> dict:
    """Reproduce and correctly label the published L q=2.46 table.

    This deliberately invokes the old scorer.  It is a diagnostic record,
    not the repaired L1 evaluation.
    """
    from x64h import family as F
    from x65a_l_latent import score as legacy_score

    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    cfg = EP.Config(overlap=overlap)
    ids = LS.build_identities(fam, seeds[0])
    probes_by_seed = [LS.build_probes(fam, beh, cfg, ids, s)
                      for s in seeds]
    probes = [p for block in probes_by_seed for p in block]
    legal = list(range(fam.m))
    out = {}
    for budget in (0, 1, 3):
        rows = legacy_score(fam, ids, probes, "main", seeds[0], legal,
                            budget=budget)
        ret = [r for r in rows if r["kind"] in
               ("returning", "ambiguous", "misleading")]
        amb = [r for r in rows if r["kind"] in
               ("ambiguous", "misleading")]
        total = sum(r["queries"] for r in rows)
        offered = sum(sum(len(legal) - i for i in range(r["queries"]))
                      for r in rows)
        # Preserve complete-stream totals using the original concatenation
        # boundaries.  No task is resampled here.
        stream_totals = []
        pos = 0
        for block in probes_by_seed:
            chunk = rows[pos:pos + len(block)]
            stream_totals.append(sum(r["queries"] for r in chunk))
            pos += len(block)
        out[budget] = {
            "query_budget": budget,
            "queries_offered": offered,
            "queries_actually_asked": total,
            "mean_over_all_tasks": mean(r["queries"] for r in rows),
            "mean_over_ambiguous_tasks": mean(
                r["queries"] for r in amb),
            "mean_over_scored_returning_tasks": mean(
                r["queries"] for r in ret),
            "total_per_stream": tuple(stream_totals),
            "task_accuracy": mean(r["correct"] for r in ret),
            "metric_denominator": "returning+ambiguous+misleading",
            "query_type": {
                "semantic": total,
                "identity": total,
                "convention": total,
                "task": 0,
                "cause": 0,
                "note": "counts overlap: semantic answers constrain both "
                        "identity and convention",
            },
        }
    curve_ok = (out[0]["task_accuracy"] == Fraction(53, 84)
                and out[1]["task_accuracy"] == Fraction(80, 84))
    label_consistent = all(
        out[b]["mean_over_all_tasks"]
        == Fraction(out[b]["queries_actually_asked"], len(probes))
        for b in out)
    return {
        "overlap": overlap,
        "seeds": tuple(seeds),
        "tasks": len(probes),
        "budgets": out,
        "published_curve_reproduced": curve_ok,
        "published_curve": "0.631 -> 0.952 at budget one",
        "q_2_46_definition": (
            "mean actual semantic questions over all 120 probe rows at "
            "budget three; it is neither a budget nor the returning-task "
            "mean"),
        "metrics_internally_consistent": label_consistent,
        "calibration": {
            "old_zero_query_label_rejected":
                out[3]["queries_actually_asked"] > 0,
            "fires": out[3]["queries_actually_asked"] > 0,
        },
    }
