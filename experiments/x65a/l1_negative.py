"""X65A-L1 negative-transfer audit with nonvacuous planted failures.

Eight authored conditions exercise the MAIN retrieval path:

* a correct returning record;
* a wrong but structurally similar record initially favoured;
* a stale record whose owner convention no longer describes the partner;
* two identities with the same convention;
* an out-of-family partner (with the shared transfer limitation explicit);
* a restricted-query indistinguishable pair;
* a genuinely new identity;
* current evidence consistent with multiple old records.

MAIN and no-memory receive the same current task, truthful answer channel,
query budget, 0-1 task loss, stopping rule, and all-task denominator.  MAIN
uses the exact open-world semantic-query adapter.  No-memory uses the
development-frozen fresh-family-prior exact task-information policy calibrated
in L1.5, including its legal behavioural questions.  The different legal
question channels are reported rather than hidden behind a false matched-
policy label.  Identity evidence remains provisional; this audit performs no
established-record write.

The bad arms are real state transitions, not evaluator booleans.  They append
incompatible grounded observations to established supports, force an action
from stale memory, and force a new partner into an old record.  A calibration
passes only when owner support is actually destroyed and the forced action is
actually false-confident.

Every reported probability and margin is a ``Fraction``.  Entropy is used only
inside query selection and never enters the canonical output.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Mapping, Sequence

import numpy as np

from x64h import episode as EP
from x64h import family as FAM

from . import l1_eval as EV
from . import l1_inference as INF
from . import l1_main as MAIN
from . import l1_retrieval as RET
from . import l1_safety as SAFE
from . import l_suite as LS
from . import semantic_mem as SM
from .latent_id import (ASSIGN_EXISTING, CREATE_NEW, NEW_IDENTITY,
                        OUT_OF_FAMILY as OUT_COMPONENT, QUARANTINE_OUT,
                        UNRESOLVED_IDENTITY)
from .provisional import MISSING
from .types import Status, encode


CORRECT_RETURN = "correct_returning_record"
WRONG_SIMILAR = "wrong_similar_initially_favored"
STALE = "stale_record"
EQUIVALENT = "two_equivalent_identities"
OUT_OF_FAMILY = "out_of_family_partner"
RESTRICTED = "restricted_query_ambiguity"
NEW = "new_identity"
MULTIPLE_OLD = "multiple_old_records_consistent"

CONDITIONS = (CORRECT_RETURN, WRONG_SIMILAR, STALE, EQUIVALENT,
              OUT_OF_FAMILY, RESTRICTED, NEW, MULTIPLE_OLD)
QUERY_BUDGET = 1
ACCURACY_MARGIN = Fraction(-1, 20)
IDENTITY_THRESHOLD = Fraction(9, 10)
LEGAL_SEMANTIC = tuple(range(8))
MAX_TASK_ATTEMPTS = 6
MAX_STALE_OWNER_ATTEMPTS = 64


class ConstructionError(RuntimeError):
    """A required stress population could not be constructed finitely."""


def _digest(value: Any) -> str:
    return hashlib.sha256(encode(value)).hexdigest()


def _mask_support(mask) -> tuple[int, ...]:
    return tuple(int(v) for v in np.flatnonzero(np.asarray(mask, dtype=bool)))


def _equal_priors(n: int, total: Fraction = Fraction(1)) -> tuple[Fraction, ...]:
    return tuple(total / n for _ in range(n))


def _convention_distance(fam, a: int, b: int) -> int:
    return (int((fam.PO[a] != fam.PO[b]).any())
            + int((fam.PF[a] != fam.PF[b]).any())
            + int((fam.PS[a] != fam.PS[b]).any())
            + int(fam.ORD[a] != fam.ORD[b]))


def _near_convention(fam, phi: int) -> int:
    """First one-component neighbour; the search is bounded by ``fam.n``."""
    for other in range(fam.n):
        if other != phi and _convention_distance(fam, phi, other) == 1:
            return other
    raise ConstructionError(f"no one-component neighbour for convention {phi}")


@dataclass(frozen=True)
class RecordView:
    key: str
    owner_phi: int
    support: tuple[int, ...]
    grounded: tuple[SM.GroundedObservation, ...] = field(compare=False)

    def mask(self, fam):
        out = np.zeros(fam.n, dtype=bool)
        out[list(self.support)] = True
        return out

    def canon(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "owner_phi": self.owner_phi,
            "support_size": len(self.support),
            "support_digest": _digest(list(self.support)),
            "grounded": [g.canon() for g in self.grounded],
        }


def _record_from_identity(fam, identity, key: str | None = None) -> RecordView:
    mask = SM.surviving_mask(fam, identity.grounded)
    return RecordView(str(key if key is not None else f"record:{identity.slot}"),
                      int(identity.phi), _mask_support(mask),
                      tuple(identity.grounded))


def _pin_record(fam, phi: int, key: str) -> RecordView:
    """Boundedly ground full-role observations until ``phi`` is singleton."""
    mask = np.ones(fam.n, dtype=bool)
    grounded: list[SM.GroundedObservation] = []
    for z in range(fam.m):
        u = int(fam.u3[phi, z])
        nxt = mask & (fam.u3[:, z] == u)
        if int(nxt.sum()) >= int(mask.sum()):
            continue
        grounded.append(SM.GroundedObservation(
            z, u, f"negative:{key}:pin:{len(grounded)}"))
        mask = nxt
        if int(mask.sum()) == 1:
            break
    support = _mask_support(mask)
    if support != (phi,):
        raise ConstructionError(f"failed to pin {phi}; support={len(support)}")
    return RecordView(key, phi, support, tuple(grounded))


def _broad_record(fam, phi: int, key: str) -> RecordView:
    """One genuine grounded observation leaving an underdetermined record."""
    for z in range(fam.m):
        u = int(fam.u3[phi, z])
        mask = fam.u3[:, z] == u
        if 1 < int(mask.sum()) < fam.n:
            obs = SM.GroundedObservation(z, u, f"negative:{key}:broad")
            return RecordView(key, phi, _mask_support(mask), (obs,))
    raise ConstructionError(f"could not make an underdetermined record for {phi}")


def _task(fam, beh, phi: int, z: int, seed: int, tag: str):
    cfg = EP.Config(overlap=fam.spec.overlap)
    for attempt in range(MAX_TASK_ATTEMPTS):
        rng = random.Random(INF.stable_seed(
            "x65a-l1-negative", fam.spec.overlap, seed, tag, phi, z, attempt))
        task = LS._transfer_task(fam, beh, cfg, phi, z, rng)
        if task is not None and task.live and task.z in task.live:
            return task
    return None


def _first_task(fam, beh, phi: int, seed: int, tag: str,
                forbidden: Sequence[int] = ()):
    blocked = set(int(v) for v in forbidden)
    for z in range(fam.m):
        if z in blocked:
            continue
        task = _task(fam, beh, phi, z, seed, tag)
        if task is not None:
            return task
    raise ConstructionError(f"no transfer task for {tag}")


@dataclass(frozen=True)
class ConditionCase:
    name: str
    task: object | None = field(repr=False, compare=False)
    true_phi: int | None
    true_z: int | None
    records: tuple[RecordView, ...]
    priors: tuple[Fraction, ...]
    legal_behavioral: tuple[int, ...]
    legal_semantic: tuple[int, ...]
    include_fresh: bool = False
    fresh_prior: Fraction = Fraction(1, 10)
    transfer_constructible: bool = True
    tested_via: str = "transfer_task"
    invariants: tuple[tuple[str, bool], ...] = ()
    calibration_target: int | None = None
    calibration_observations: tuple[tuple[int, int], ...] = ()

    def canon(self) -> dict[str, Any]:
        applied = (_equal_priors(len(self.records), Fraction(4, 5))
                   if self.records else ())
        return {
            "name": self.name,
            "task_digest": None if self.task is None else _digest({
                "kind": self.task.kind, "demos": tuple(self.task.demos),
                "live": tuple(self.task.live), "u": int(self.task.u),
                "pool": tuple(self.task.pool), "tie": tuple(self.task.tie)}),
            "true_phi": self.true_phi,
            "true_z": self.true_z,
            "records": [r.canon() for r in self.records],
            # Earlier evaluator drafts labelled these constructor hints as
            # applied priors even though l1_main used its frozen open-world
            # prior.  Keep them only as provenance and report the actual
            # model separately.
            "construction_design_priors_not_applied": list(self.priors),
            "applied_prior_contract": {
                "record_priors": list(applied),
                "NEW_IDENTITY": Fraction(1, 10),
                "OUT_OF_FAMILY": Fraction(1, 10),
                "source": "l1_main frozen open-world prior",
            },
            "legal_behavioral": list(self.legal_behavioral),
            "legal_semantic": list(self.legal_semantic),
            "include_fresh": self.include_fresh,
            "fresh_prior": self.fresh_prior,
            "transfer_constructible": self.transfer_constructible,
            "tested_via": self.tested_via,
            "invariants": dict(self.invariants),
        }


def _make_latent_state(fam, beh, case: ConditionCase):
    masks = [r.mask(fam) for r in case.records]
    records = {key: record for key, record in enumerate(case.records)}
    exact = RET.build_global_exact_index(records)
    selected = RET.retrieve_protocol_a(
        exact, fam, case.task, k=min(4, len(records)),
        strategy="exact_likelihood",
        seed=INF.stable_seed("negative-retrieval", fam.spec.overlap,
                             case.name))
    return MAIN.subset_state(
        fam, case.task, masks, selected.selected_keys,
        with_new=True, with_out=True)


def _case_retrieval(fam, case: ConditionCase):
    records = {key: record for key, record in enumerate(case.records)}
    exact = RET.build_global_exact_index(records)
    return RET.retrieve_protocol_a(
        exact, fam, case.task, k=min(4, len(records)),
        strategy="exact_likelihood",
        seed=INF.stable_seed("negative-retrieval", fam.spec.overlap,
                             case.name))


def _argmax_task(state) -> tuple[int | None, Fraction]:
    post = state.task_posterior()
    best = None
    best_p = Fraction(-1)
    for z in state.task.tie:
        p = post.get(int(z), Fraction(0))
        if p > best_p:
            best, best_p = int(z), p
    return (best if best_p > 0 else None,
            best_p if best_p > 0 else Fraction(0))


def _positive_identities(state) -> int:
    return sum(isinstance(key, int) and p > 0
               for key, p in state.identity_posterior().items())


def _common_records(fam, identities) -> tuple[RecordView, ...]:
    return tuple(_record_from_identity(fam, ident) for ident in identities)


def _correct_case(fam, beh, probes, identities) -> ConditionCase:
    probe = next((p for p in probes
                  if p.kind == "returning" and p.slot >= 0 and p.task.live), None)
    if probe is None:
        raise ConstructionError("no correct returning probe")
    record = _record_from_identity(fam, identities[probe.slot])
    return ConditionCase(
        CORRECT_RETURN, probe.task, probe.phi_true, probe.task.z, (record,),
        (Fraction(1),), tuple(range(8)), LEGAL_SEMANTIC,
        invariants=(("owner_is_true", record.owner_phi == probe.phi_true),
                    ("true_survives_record",
                     probe.phi_true in record.support)))


def _wrong_similar_case(fam, beh, seed: int) -> ConditionCase:
    true_phi = (seed * 17 + 3) % fam.n
    wrong_phi = _near_convention(fam, true_phi)
    true_record = _broad_record(fam, true_phi, "true-underdetermined")
    wrong_record = _pin_record(fam, wrong_phi, "wrong-similar")
    records = (true_record, wrong_record)

    chosen = None
    priors = (Fraction(1, 2), Fraction(1, 2))
    for z in range(fam.m):
        task = _task(fam, beh, true_phi, z, seed, WRONG_SIMILAR)
        if task is None:
            continue
        case = ConditionCase(
            WRONG_SIMILAR, task, true_phi, task.z, records, priors,
            tuple(range(8)), LEGAL_SEMANTIC)
        post = _make_latent_state(fam, beh, case).identity_posterior()
        if post.get(1, Fraction(0)) > post.get(0, Fraction(0)):
            chosen = task
            break
    if chosen is None:
        raise ConstructionError("no wrong-similar initially-favoured task")
    diff_z = next((z for z in range(fam.m)
                   if int(fam.u3[true_phi, z])
                   != int(fam.u3[wrong_phi, z])), None)
    if diff_z is None:
        raise ConstructionError("near conventions have no disagreement")
    final = ConditionCase(
        WRONG_SIMILAR, chosen, true_phi, chosen.z, records, priors,
        tuple(range(8)), LEGAL_SEMANTIC,
        invariants=(("one_component_apart",
                     _convention_distance(fam, true_phi, wrong_phi) == 1),
                    ("wrong_initially_favored", True),
                    ("wrong_record_excludes_true",
                     true_phi not in wrong_record.support)),
        calibration_target=1,
        calibration_observations=((diff_z, int(fam.u3[true_phi, diff_z])),))
    post = _make_latent_state(fam, beh, final).identity_posterior()
    if not post[1] > post[0]:
        raise ConstructionError("wrong-favoured assertion did not survive")
    return final


def _stale_case(fam, beh, seed: int) -> ConditionCase:
    start = (seed * 29 + 11) % fam.n
    chosen = None
    forced = None
    old_phi = current_phi = None
    stale = None
    # A stream seed selects a bounded candidate sequence, not a promise that
    # its first stale owner already makes the deliberately unsafe forced arm
    # fail.  Search a preregistered finite prefix and retain the first actual
    # false-action witness.  This keeps the condition nonvacuous without an
    # unbounded retry or truth-dependent threshold change.
    for owner_offset in range(min(MAX_STALE_OWNER_ATTEMPTS, fam.n)):
        candidate_old = (start + owner_offset) % fam.n
        candidate_current = _near_convention(fam, candidate_old)
        candidate_stale = _pin_record(
            fam, candidate_old, "stale-established")
        for z in range(fam.m):
            task = _task(fam, beh, candidate_current, z, seed, STALE)
            if task is None:
                continue
            case = ConditionCase(
                STALE, task, candidate_current, task.z,
                (candidate_stale,), (Fraction(1),), tuple(range(8)),
                LEGAL_SEMANTIC)
            state = _make_latent_state(fam, beh, case)
            action, confidence = _argmax_task(state)
            if (action is not None and action != task.z
                    and confidence >= Fraction(19, 20)):
                old_phi, current_phi, stale = (
                    candidate_old, candidate_current, candidate_stale)
                chosen, forced = task, action
                break
        if chosen is not None:
            break
    if chosen is None:
        raise ConstructionError(
            "no stale false-action witness in bounded owner search")
    assert old_phi is not None and current_phi is not None and stale is not None
    diff_z = next(z for z in range(fam.m)
                  if int(fam.u3[old_phi, z])
                  != int(fam.u3[current_phi, z]))
    return ConditionCase(
        STALE, chosen, current_phi, chosen.z, (stale,), (Fraction(1),),
        tuple(range(8)), LEGAL_SEMANTIC,
        invariants=(("stale_owner_differs", old_phi != current_phi),
                    ("current_phi_absent", current_phi not in stale.support),
                    ("forced_stale_action_is_wrong", forced != chosen.z)),
        calibration_target=0,
        calibration_observations=((diff_z,
                                   int(fam.u3[current_phi, diff_z])),))


def _equivalent_case(fam, beh, probes, identities) -> ConditionCase:
    if identities[0].phi != identities[1].phi:
        raise ConstructionError("planted equivalent identities disappeared")
    probe = next((p for p in probes
                  if p.kind == "returning" and p.slot == 0 and p.task.live), None)
    if probe is None:
        raise ConstructionError("no equivalent-identity task")
    records = (_record_from_identity(fam, identities[0]),
               _record_from_identity(fam, identities[1]))
    return ConditionCase(
        EQUIVALENT, probe.task, probe.phi_true, probe.task.z, records,
        _equal_priors(2), tuple(range(8)), LEGAL_SEMANTIC,
        invariants=(("same_convention",
                     records[0].owner_phi == records[1].owner_phi),
                    ("distinct_records", records[0].key != records[1].key)))


def _oof_case(fam, correct: ConditionCase) -> ConditionCase:
    grounded = SAFE.grounded_out_of_family_event(fam)
    if grounded is None:
        raise ConstructionError("no grounded out-of-family construction")
    gmask = np.ones(fam.n, dtype=bool)
    for z, u in grounded:
        gmask &= fam.u3[:, int(z)] == int(u)
    transfer = SAFE.transfer_out_of_family_utterance(fam)
    if transfer is None:
        return ConditionCase(
            OUT_OF_FAMILY, None, None, None, correct.records,
            correct.priors, tuple(range(8)), (),
            transfer_constructible=False,
            tested_via="grounded_event_only_transfer_unconstructible",
            invariants=(("grounded_event_zero_support", not gmask.any()),
                        ("transfer_explicitly_unconstructible", True)))
    base = correct.task
    task = EP.Task(base.kind, base.z, base.demos, base.live, int(transfer),
                   base.pool, base.tie)
    return ConditionCase(
        OUT_OF_FAMILY, task, None, task.z, correct.records, correct.priors,
        tuple(range(8)), (), transfer_constructible=True,
        tested_via="nonvacuous_transfer_and_grounded_event",
        invariants=(("grounded_event_zero_support", not gmask.any()),
                    ("transfer_has_zero_family_likelihood",
                     bool(fam.counts(int(transfer), task.pool)[:,
                                     list(task.live)].sum() == 0))))


def _calibration_task(fam, beh, phi_a: int, phi_b: int,
                      query_set: Sequence[int], seed: int):
    cfg = EP.Config(overlap=fam.spec.overlap)
    for z in query_set:
        if int(fam.u3[phi_a, z]) != int(fam.u3[phi_b, z]):
            continue
        for attempt in range(MAX_TASK_ATTEMPTS):
            rng = random.Random(INF.stable_seed(
                "negative-restricted", fam.spec.overlap, seed, z, attempt))
            target = min(4, max(2, cfg.ambiguity[0]))
            demos, live = EP.pick_demos(
                beh, fam.m, z, target, cfg.demos_cal_cap, rng)
            if z not in live or len(live) < 2:
                continue
            task = EP.Task("cal", z, demos, live,
                           int(fam.u3[phi_a, z]), FAM.CAL_POOL,
                           tuple(rng.sample(range(fam.m), fam.m)))
            a = _pin_record(fam, phi_a, "restricted-a")
            b = _pin_record(fam, phi_b, "restricted-b")
            case = ConditionCase(
                RESTRICTED, task, phi_a, task.z, (a, b), _equal_priors(2),
                tuple(range(8)), tuple(int(v) for v in query_set))
            if _positive_identities(_make_latent_state(fam, beh, case)) >= 2:
                return task, a, b
    raise ConstructionError("no restricted task leaves both identities live")


def _restricted_case(fam, beh, seed: int) -> ConditionCase:
    witness = SAFE.restricted_indistinguishable_case(fam)
    if witness is None:
        raise ConstructionError("restricted ambiguity is unconstructible")
    task, a, b = _calibration_task(
        fam, beh, witness["phi_a"], witness["phi_b"],
        witness["query_set"], seed)
    q = tuple(witness["query_set"])
    equal = all(int(fam.u3[a.owner_phi, z]) == int(fam.u3[b.owner_phi, z])
                for z in q)
    outside = int(witness["outside_witness"])
    return ConditionCase(
        RESTRICTED, task, a.owner_phi, task.z, (a, b), _equal_priors(2),
        tuple(range(8)), q,
        invariants=(("equal_on_restricted_queries", equal),
                    ("different_outside_restriction",
                     int(fam.u3[a.owner_phi, outside])
                     != int(fam.u3[b.owner_phi, outside])),
                    ("multiple_identities_initially_live", True)))


def _new_case(fam, beh, identities, seed: int) -> ConditionCase:
    plan = SAFE.find_new_identity_plan(fam, identities)
    if plan is None:
        raise ConstructionError("new identity plan unavailable")
    task = _first_task(fam, beh, int(plan["phi"]), seed, NEW,
                       forbidden=plan["questions"])
    records = _common_records(fam, identities)
    total_old = Fraction(9, 10)
    return ConditionCase(
        NEW, task, int(plan["phi"]), task.z, records,
        _equal_priors(len(records), total_old), tuple(range(8)),
        LEGAL_SEMANTIC, include_fresh=True,
        invariants=(("new_phi_absent_from_every_old_record",
                     all(int(plan["phi"]) not in r.support for r in records)),
                    ("two_grounding_questions", len(plan["questions"]) == 2),
                    ("answers_exclude_every_old_record",
                     not any(plan["old_compatible"]))),
        calibration_target=0,
        calibration_observations=tuple(
            (int(z), int(u)) for z, u in
            zip(plan["questions"], plan["answers"])))


def _multiple_case(fam, beh, probes, identities) -> ConditionCase:
    records = _common_records(fam, identities)
    priors = _equal_priors(len(records))
    candidates = [p for p in probes if p.task.live and p.slot >= 0]
    candidates.sort(key=lambda p: (p.kind != "ambiguous", p.slot, p.task.z))
    for probe in candidates:
        case = ConditionCase(
            MULTIPLE_OLD, probe.task, probe.phi_true, probe.task.z, records,
            priors, tuple(range(8)), LEGAL_SEMANTIC)
        count = _positive_identities(_make_latent_state(fam, beh, case))
        if count >= 2:
            return ConditionCase(
                MULTIPLE_OLD, probe.task, probe.phi_true, probe.task.z,
                records, priors, tuple(range(8)), LEGAL_SEMANTIC,
                invariants=(("multiple_old_records_have_positive_mass",
                             count >= 2),
                            ("true_record_present",
                             probe.phi_true in records[probe.slot].support)))
    raise ConstructionError("no task consistent with multiple old records")


def build_conditions(fam, beh, seed: int = 6400) -> tuple[ConditionCase, ...]:
    identities = LS.build_identities(fam, seed)
    if len(identities) != 8:
        raise ConstructionError("negative audit requires eight identities")
    probes = LS.build_probes(
        fam, beh, EP.Config(overlap=fam.spec.overlap), identities, seed)
    correct = _correct_case(fam, beh, probes, identities)
    cases = (
        correct,
        _wrong_similar_case(fam, beh, seed),
        _stale_case(fam, beh, seed),
        _equivalent_case(fam, beh, probes, identities),
        _oof_case(fam, correct),
        _restricted_case(fam, beh, seed),
        _new_case(fam, beh, identities, seed),
        _multiple_case(fam, beh, probes, identities),
    )
    if tuple(c.name for c in cases) != CONDITIONS:
        raise ConstructionError("condition order or membership changed")
    for case in cases:
        failed = [name for name, ok in case.invariants if not ok]
        if failed:
            raise ConstructionError(f"{case.name} failed invariants {failed}")
    return cases


def _best_question(state, legal_behavioral, legal_semantic):
    questions = INF.legal_questions(state, legal_behavioral, legal_semantic)
    best = None
    score = float("-inf")
    for query in questions:
        value = state.information_gain(query, "task")
        if value > score:
            best, score = query, value
    return best, len(questions)


@dataclass(frozen=True)
class ArmMetrics:
    task_accuracy: Fraction | None
    queries_offered: int
    queries_asked: int
    excess_questions: int
    false_confident_actions: int
    established_record_corruption: int
    provisional_branches: int
    unresolved_outcomes: int
    action: int | None
    query_policy: str
    identity_decision: str
    has_new_component: bool
    has_out_component: bool

    def canon(self) -> dict[str, Any]:
        return {
            "task_accuracy": self.task_accuracy,
            "queries_offered": self.queries_offered,
            "queries_asked": self.queries_asked,
            "excess_questions": self.excess_questions,
            "false_confident_actions": self.false_confident_actions,
            "established_record_corruption":
                self.established_record_corruption,
            "provisional_branches": self.provisional_branches,
            "unresolved_outcomes": self.unresolved_outcomes,
            "action": self.action,
            "query_policy": self.query_policy,
            "identity_decision": self.identity_decision,
            "has_new_component": self.has_new_component,
            "has_out_component": self.has_out_component,
        }


@dataclass(frozen=True)
class ConditionResult:
    condition: ConditionCase
    main: ArmMetrics
    no_memory: ArmMetrics
    accuracy_delta: Fraction | None
    noninferior: bool
    matched_protocol: bool
    protocol_contract: Mapping[str, Any]

    def canon(self) -> dict[str, Any]:
        return {
            "condition": self.condition.canon(),
            "main": self.main.canon(),
            "no_memory": self.no_memory.canon(),
            "accuracy_delta": self.accuracy_delta,
            "frozen_margin": ACCURACY_MARGIN,
            "noninferior": self.noninferior,
            "matched_protocol": self.matched_protocol,
            "protocol_contract": dict(self.protocol_contract),
        }


def _run_case(fam, beh, case: ConditionCase) -> ConditionResult:
    if case.task is None:
        # The shared alphabet has no nonvacuous alien two-token transfer.  The
        # separate grounded contradiction remains live, but task accuracy is
        # deliberately absent from the transfer denominator.
        null = ArmMetrics(
            None, 0, 0, 0, 0, 0, 0, 1, None, MAIN.INFORMATION_GAIN,
            MISSING, True, True)
        baseline = ArmMetrics(
            None, 0, 0, 0, 0, 0, 0, 1, None,
            INF.TASK_INFORMATION_GAIN, "NOT_APPLICABLE_MEMORYLESS",
            False, False)
        contract = {
            "same_current_task": True,
            "same_truthful_answer_channel": True,
            "same_query_budget": True,
            "same_zero_one_task_loss": True,
            "same_budget_exhaustion_stopping_rule": True,
            "same_metric_denominator": True,
            "main_legal_query_types": ("semantic",),
            "no_memory_legal_query_types": ("behavioral", "semantic"),
            "different_query_universes_explicit": True,
            "answers_applied_to_current_posterior": True,
            "both_within_budget": True,
        }
        return ConditionResult(
            case, null, baseline, None, True,
            all(value for key, value in contract.items()
                if key not in ("main_legal_query_types",
                               "no_memory_legal_query_types")), contract)

    main_state = _make_latent_state(fam, beh, case)
    main_retrieval = _case_retrieval(fam, case)
    memoryless = INF.make_memoryless_state(fam, beh, case.task)
    before_hashes = tuple(_digest(r.canon()) for r in case.records)
    run_seed = INF.stable_seed(
        "x65a-l1-negative-policy", fam.spec.overlap, case.name,
        int(case.task.u), tuple(case.task.live))
    phi_truth = int(case.true_phi) if case.true_phi is not None else 0
    main_run = MAIN.run_policy(
        main_state, MAIN.INFORMATION_GAIN, QUERY_BUDGET, phi_truth,
        int(case.true_z), case.legal_semantic, run_seed)
    memoryless_run = EV.run_policy(
        memoryless, INF.TASK_INFORMATION_GAIN, QUERY_BUDGET, phi_truth,
        int(case.true_z), case.legal_behavioral, case.legal_semantic,
        run_seed)

    def operational_action(run):
        # ``exact_prediction`` breaks ties deterministically.  Empty posterior
        # mass is not a prediction, so the operational adapter abstains without
        # inspecting the condition name or truth.
        return run.action if run.state.task_posterior() else None

    main_action = operational_action(main_run)
    fresh_action = memoryless_run["action"]

    task_truth = int(case.true_z)
    main_correct = int(main_action == task_truth)
    fresh_correct = int(fresh_action == task_truth)
    main_acc = Fraction(main_correct)
    fresh_acc = Fraction(fresh_correct)
    delta = main_acc - fresh_acc

    unresolved_decisions = {
        UNRESOLVED_IDENTITY, QUARANTINE_OUT, MISSING,
    }
    main_identity_unresolved = (
        main_run.identity_decision in unresolved_decisions)
    main_provisional = int(main_run.identity_decision == UNRESOLVED_IDENTITY)
    after_hashes = tuple(_digest(r.canon()) for r in case.records)
    corruption = int(before_hashes != after_hashes)
    if corruption:
        raise AssertionError("MAIN mutated an established record")

    main = ArmMetrics(
        main_acc, main_run.queries_offered, main_run.queries_asked,
        main_run.queries_asked - len(memoryless_run["events"]),
        int(main_action is not None and not main_correct
            and main_run.confidence >= Fraction(19, 20)), corruption,
        main_provisional,
        int(main_identity_unresolved or main_action is None), main_action,
        main_run.policy, main_run.identity_decision,
        NEW_IDENTITY in main_state.identity_posterior(),
        OUT_COMPONENT in main_state.identity_posterior())
    no_memory = ArmMetrics(
        fresh_acc, memoryless_run["queries_offered"],
        len(memoryless_run["events"]), 0,
        int(fresh_action is not None and not fresh_correct
            and memoryless_run["confidence"] >= Fraction(19, 20)), 0,
        0, int(fresh_action is None), fresh_action,
        INF.TASK_INFORMATION_GAIN, "NOT_APPLICABLE_MEMORYLESS",
        False, False)
    contract = {
        "same_current_task": memoryless.task is case.task,
        "same_truthful_answer_channel": True,
        "same_query_budget": main_run.query_budget == QUERY_BUDGET,
        "same_zero_one_task_loss": True,
        "same_budget_exhaustion_stopping_rule": True,
        "same_metric_denominator": True,
        "main_legal_query_types": ("semantic",),
        "no_memory_legal_query_types": ("behavioral", "semantic"),
        "different_query_universes_explicit": True,
        "answers_applied_to_current_posterior": (
            len(memoryless_run["state"].history)
            == len(memoryless_run["events"])),
        "both_within_budget": (
            main_run.queries_asked <= QUERY_BUDGET
            and len(memoryless_run["events"]) <= QUERY_BUDGET),
        "main_retrieval": main_retrieval.accounting.canon(),
        "main_retrieval_selected_keys": main_retrieval.selected_keys,
    }
    matched = all(
        value for key, value in contract.items()
        if key not in ("main_legal_query_types",
                       "no_memory_legal_query_types", "main_retrieval",
                       "main_retrieval_selected_keys"))
    return ConditionResult(case, main, no_memory, delta,
                           delta >= ACCURACY_MARGIN, matched, contract)


def _mutated_support(fam, record: RecordView,
                     observations: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    mask = record.mask(fam)
    for z, u in observations:
        mask &= fam.u3[:, int(z)] == int(u)
    return _mask_support(mask)


def _main_safety_predicate(row: Mapping[str, Any]) -> bool:
    """The same observable predicate used for MAIN and every red arm."""
    return bool(
        int(row.get("established_record_corruption", 0)) == 0
        and int(row.get("false_confident_actions", 0)) == 0
        and int(row.get("false_global_promotions", 0)) == 0)


def _calibrations(fam, beh, cases: Mapping[str, ConditionCase]) -> dict:
    wrong = cases[WRONG_SIMILAR]
    wrong_target = wrong.records[int(wrong.calibration_target)]
    wrong_after = _mutated_support(
        fam, wrong_target, wrong.calibration_observations)
    wrong_corrupted = (wrong_target.owner_phi in wrong_target.support
                       and wrong_target.owner_phi not in wrong_after)
    bad_record = SM.SemanticRecord(
        wrong_target.key, tuple(wrong_target.grounded),
        status=Status.CONFIRMED, surviving=len(wrong_target.support))
    bad_z, bad_u = wrong.calibration_observations[0]
    no_confirmation_after, no_confirmation_transition = SM.absorb(
        fam, bad_record,
        SM.GroundedObservation(
            int(bad_z), int(bad_u), "negative:no-confirmation:alien"),
        task_index=1, ledger=None, quarantine=False)
    no_confirmation_support = _mask_support(
        SM.surviving_mask(fam, no_confirmation_after.grounded))
    no_confirmation_corrupted = bool(
        wrong_target.owner_phi in wrong_target.support
        and wrong_target.owner_phi not in no_confirmation_support)

    stale = cases[STALE]
    stale_state = _make_latent_state(fam, beh, stale)
    stale_action, stale_confidence = _argmax_task(stale_state)
    stale_false = bool(
        stale_action is not None and stale_action != stale.true_z
        and stale_confidence >= Fraction(19, 20))

    new = cases[NEW]
    new_target = new.records[int(new.calibration_target)]
    new_after = _mutated_support(fam, new_target,
                                 new.calibration_observations)
    new_corrupted = (new_target.owner_phi in new_target.support
                     and new_target.owner_phi not in new_after)

    scope = SAFE.scope_audit(fam)
    false_global = int(scope.get("calibration_false_global_promotions", 0))
    out: dict[str, Any] = {
        "immediate_map_write": {
            "owner_survived_before":
                wrong_target.owner_phi in wrong_target.support,
            "owner_survived_after": wrong_target.owner_phi in wrong_after,
            "support_before": len(wrong_target.support),
            "support_after": len(wrong_after),
            "established_record_corruption": int(wrong_corrupted),
            "fires": bool(wrong_corrupted),
        },
        "forced_stale_action": {
            "forced_actions": int(stale_action is not None),
            "action": stale_action,
            "confidence": stale_confidence,
            "false_confident_actions": int(stale_false),
            "fires": bool(stale_false),
        },
        "forced_new_assimilation": {
            "owner_survived_before":
                new_target.owner_phi in new_target.support,
            "owner_survived_after": new_target.owner_phi in new_after,
            "support_before": len(new_target.support),
            "support_after": len(new_after),
            "forced_decisions": 1,
            "established_record_corruption": int(new_corrupted),
            "fires": bool(new_corrupted),
        },
        "no_confirmation_contamination": {
            "transition": no_confirmation_transition,
            "owner_survived_before":
                wrong_target.owner_phi in wrong_target.support,
            "owner_survived_after":
                wrong_target.owner_phi in no_confirmation_support,
            "support_before": len(wrong_target.support),
            "support_after": len(no_confirmation_support),
            "established_record_corruption":
                int(no_confirmation_corrupted),
            "fires": bool(no_confirmation_corrupted),
        },
        "false_global_promotion": {
            "false_global_promotions": false_global,
            "fires": false_global > 0,
        },
    }
    for row in out.values():
        row["rejected_by_main_safety_predicate"] = not _main_safety_predicate(
            row)
    out["all_fire"] = all(row["fires"] for row in out.values())
    out["same_predicate_rejections"] = all(
        row["rejected_by_main_safety_predicate"] for row in out.values()
        if isinstance(row, dict))
    return out


@dataclass(frozen=True)
class NegativeTransferAudit:
    overlap: str
    seed: int
    query_budget: int
    accuracy_margin: Fraction
    conditions: tuple[ConditionResult, ...]
    calibrations: Mapping[str, Any]
    gates: Mapping[str, bool]

    def canon(self) -> dict[str, Any]:
        return {
            "phase": "X65A-L1-negative-transfer",
            "overlap": self.overlap,
            "seed": self.seed,
            "query_budget": self.query_budget,
            "accuracy_margin": self.accuracy_margin,
            "metric_denominator": "all_tasks_per_constructible_condition",
            "conditions": [c.canon() for c in self.conditions],
            "calibrations": dict(self.calibrations),
            "gates": dict(self.gates),
        }


def audit_stratum(overlap: str, seed: int = 6400) -> NegativeTransferAudit:
    from x64h import family as F

    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    cases = build_conditions(fam, beh, seed)
    results = tuple(_run_case(fam, beh, case) for case in cases)
    by_name = {case.name: case for case in cases}
    calibrations = _calibrations(fam, beh, by_name)
    gates = {
        "all_eight_conditions_constructed":
            tuple(r.condition.name for r in results) == CONDITIONS,
        "all_construction_invariants_hold": all(
            all(ok for _name, ok in r.condition.invariants) for r in results),
        "matched_evidence_and_query_budget": all(
            r.matched_protocol for r in results),
        "main_noninferior_at_frozen_margin": all(
            r.noninferior for r in results),
        "main_never_corrupts_established_records": all(
            r.main.established_record_corruption == 0 for r in results),
        "main_safety_predicate_holds": all(
            _main_safety_predicate(r.main.canon()) for r in results),
        "bad_arms_fire": bool(
            calibrations["all_fire"]
            and calibrations["same_predicate_rejections"]),
        "shared_oof_transfer_scope_honest": (
            overlap != "shared"
            or not by_name[OUT_OF_FAMILY].transfer_constructible),
        "disjoint_oof_transfer_nonvacuous": (
            overlap != "disjoint_op"
            or by_name[OUT_OF_FAMILY].transfer_constructible),
    }
    return NegativeTransferAudit(
        overlap, seed, QUERY_BUDGET, ACCURACY_MARGIN, results,
        calibrations, gates)


def audit_both(seed: int = 6400) -> dict[str, NegativeTransferAudit]:
    return {overlap: audit_stratum(overlap, seed)
            for overlap in ("shared", "disjoint_op")}
