"""X65A-L1 safety constructions that the X65A-L evaluator did not run.

The functions here deliberately separate *constructibility* from a metric.
An absent population is reported as untestable; it is never converted into a
perfect score.  All searches are finite and bounded by the frozen families.
"""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from x64h import family as FAM
from x64h import episode as EP

from . import latent_id as LI
from . import l1_inference as INF
from . import l1_main as MAIN
from . import l1_retrieval as RET
from . import provisional as P
from . import semantic_mem as SM
from .types import Status, byte_cost, encode


def digest(label: str, values) -> str:
    payload = repr((label, tuple(values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class VerificationScope:
    """Scope attached to an actual promotion, not merely printed nearby."""

    challenge_universe_digest: str
    query_set_digest: str
    validity_scope: str
    status: str  # global_in_finite_model | empirical

    def canon(self):
        return {
            "challenge_universe_digest": self.challenge_universe_digest,
            "query_set_digest": self.query_set_digest,
            "validity_scope": self.validity_scope,
            "status": self.status,
        }


@dataclass(frozen=True)
class ScopedIdentityRecord:
    key: int
    grounded: tuple
    scope: VerificationScope
    status: str = "CONFIRMED"

    @property
    def identity(self) -> str:
        """Opaque store key used by :class:`SemanticStore`."""
        return str(self.key)

    def canon(self):
        return {
            "key": self.key,
            "grounded": [g.canon() for g in self.grounded],
            "scope": self.scope.canon(),
            "status": self.status,
        }

    def bytes(self) -> int:
        return byte_cost(self.canon())


def scope_for(challenge_universe, query_set, *, global_status: bool) -> VerificationScope:
    return VerificationScope(
        digest("challenge", challenge_universe),
        digest("queries", query_set),
        "controlled authored X64H semantic family",
        "global_in_finite_model" if global_status else "empirical",
    )


@dataclass(frozen=True)
class AuthoredOutOfFamilyConvention:
    """A complete role mapping outside the frozen injective/permutation family.

    It uses the stratum's existing alphabet.  The planted defect is a repeated
    codeword in one role map, not an impossible utterance selected after the
    fact.  Its complete calibration row is checked against every frozen family
    member and a grounded contradiction is then derived from that row.
    """

    operator_map: tuple[int, ...]
    filter_map: tuple[int, ...]
    scope_map: tuple[int, ...]
    reverse: int
    calibration_codes: tuple[int, ...]
    defect: str
    grounded_contradiction: tuple[tuple[int, int], ...]

    def canon(self):
        return {
            "operator_map": list(self.operator_map),
            "filter_map": list(self.filter_map),
            "scope_map": list(self.scope_map),
            "reverse": self.reverse,
            "calibration_codes": list(self.calibration_codes),
            "defect": self.defect,
            "grounded_contradiction": [list(v)
                                       for v in self.grounded_contradiction],
        }


def _codes_for_maps(fam, po, pf, ps, reverse: int) -> tuple[int, ...]:
    wo = np.asarray(po, dtype=np.int64)[fam.op_i]
    wf = np.asarray(pf, dtype=np.int64)[fam.f_i]
    ws = np.asarray(ps, dtype=np.int64)[fam.s_i]
    if int(reverse):
        values = ws * fam.A * fam.A + wf * fam.A + wo
    else:
        values = wo * fam.A * fam.A + wf * fam.A + ws
    return tuple(int(v) for v in values)


def _grounded_contradiction_for_codes(fam, codes) -> tuple[tuple[int, int], ...]:
    mask = np.ones(fam.n, dtype=bool)
    observations = []
    for z, u in enumerate(codes):
        nxt = mask & (fam.u3[:, z] == int(u))
        if int(nxt.sum()) < int(mask.sum()):
            observations.append(SM.GroundedObservation(
                z, int(u), f"authored-oof:{len(observations)}"))
            mask = nxt
        if not mask.any():
            break
    if mask.any():
        return ()
    minimal = SM.minimize(fam, tuple(observations))
    return tuple((int(g.z), int(g.u)) for g in minimal)


def authored_out_of_family_convention(fam):
    """Construct one bounded, complete alien convention in either stratum."""
    po0 = tuple(int(v) for v in fam.PO[0])
    pf0 = tuple(int(v) for v in fam.PF[0])
    ps0 = tuple(int(v) for v in fam.PS[0])
    reverse = int(fam.ORD[0])
    candidates = []
    for role, base in (("operator", po0), ("filter", pf0), ("scope", ps0)):
        if len(base) < 2:
            continue
        for target in range(1, len(base)):
            mutated = list(base)
            mutated[target] = mutated[0]
            maps = {"operator": po0, "filter": pf0, "scope": ps0}
            maps[role] = tuple(mutated)
            candidates.append((maps, f"noninjective_{role}_map"))
    for maps, defect in candidates:
        codes = _codes_for_maps(
            fam, maps["operator"], maps["filter"], maps["scope"], reverse)
        membership = int(np.all(
            fam.u3 == np.asarray(codes, dtype=fam.u3.dtype)[None, :],
            axis=1).sum())
        contradiction = _grounded_contradiction_for_codes(fam, codes)
        if membership == 0 and contradiction:
            return AuthoredOutOfFamilyConvention(
                maps["operator"], maps["filter"], maps["scope"], reverse,
                codes, defect, contradiction)
    return None


def grounded_out_of_family_event(fam):
    """Grounded contradiction produced by a separately authored convention."""
    alien = authored_out_of_family_convention(fam)
    return None if alien is None else alien.grounded_contradiction


def transfer_out_of_family_utterance(fam, live=None, pool=FAM.P2):
    """A two-token code with no reading for any supplied live meaning."""
    live = tuple(range(fam.m)) if live is None else tuple(live)
    for u in range(fam.A ** 2):
        if fam.counts(u, pool)[:, list(live)].sum() == 0:
            return u
    return None


def derive_live_from_demonstrations(beh, demonstrations):
    live = list(range(len(beh)))
    for k, y in demonstrations:
        live = [z for z in live if beh[z][k] == y]
    return tuple(live)


def unknown_meaning_demonstrations(beh):
    """Find jointly inconsistent public demonstrations.

    Each singleton observation has an in-family explanation; their
    conjunction has none.  This exercises the derivation rather than
    injecting ``live=()`` into a task fixture.
    """
    n_inputs = len(beh[0])
    outputs = [sorted({row[k] for row in beh}, key=repr)
               for k in range(n_inputs)]
    for k1 in range(n_inputs):
        for k2 in range(k1 + 1, n_inputs):
            for y1 in outputs[k1]:
                a = derive_live_from_demonstrations(beh, ((k1, y1),))
                if not a:
                    continue
                for y2 in outputs[k2]:
                    b = derive_live_from_demonstrations(beh, ((k2, y2),))
                    if not b:
                        continue
                    both = ((k1, y1), (k2, y2))
                    if not derive_live_from_demonstrations(beh, both):
                        return both
    return None


def restricted_indistinguishable_case(fam, query_set=(0, 1, 2, 3)):
    """Find conventions equal on the legal questions but unequal globally."""
    q = tuple(query_set)
    sig = fam.u3[:, list(q)]
    groups: dict[tuple, list[int]] = {}
    for phi, row in enumerate(sig):
        groups.setdefault(tuple(int(x) for x in row), []).append(phi)
    outside = tuple(z for z in range(fam.m) if z not in q)
    for members in groups.values():
        if len(members) < 2:
            continue
        for a, b in itertools.combinations(members[:32], 2):
            diff = [z for z in outside
                    if int(fam.u3[a, z]) != int(fam.u3[b, z])]
            if diff:
                return {
                    "phi_a": a,
                    "phi_b": b,
                    "query_set": q,
                    "outside_witness": diff[0],
                    "restricted_equal": True,
                    "globally_equal": False,
                }
    return None


def validate_scope_promotion(record: ScopedIdentityRecord, case,
                             challenge_universe, query_set) -> dict:
    """Validate a promoted record's scope through one shared predicate."""

    expected = scope_for(
        challenge_universe, query_set,
        global_status=bool(case["globally_equal"]))
    checks = {
        "challenge_universe_digest_matches":
            record.scope.challenge_universe_digest
            == expected.challenge_universe_digest,
        "query_set_digest_matches":
            record.scope.query_set_digest == expected.query_set_digest,
        "validity_scope_matches":
            record.scope.validity_scope == expected.validity_scope,
        "global_status_requires_global_equivalence": (
            record.scope.status != "global_in_finite_model"
            or bool(case["globally_equal"])),
        "empirical_status_required_for_restricted_witness": (
            bool(case["globally_equal"])
            or record.scope.status == "empirical"),
        "confirmed_record": record.status == "CONFIRMED",
        "grounded_universe_matches_query_set": (
            tuple(g.z for g in record.grounded) == tuple(query_set)),
    }
    return {"checks": checks, "passed": all(checks.values())}


def scope_audit(fam, query_set=(0, 1, 2, 3)) -> dict:
    case = restricted_indistinguishable_case(fam, query_set)
    if case is None:
        return {"constructible": False, "promotions": 0,
                "false_global_promotions": 0,
                "calibration_false_global_promotions": 0}
    obs = tuple(
        SM.GroundedObservation(z, int(fam.u3[case["phi_a"], z]), f"scope:{z}")
        for z in case["query_set"]
    )
    empirical = ScopedIdentityRecord(
        0, obs, scope_for(range(fam.m), case["query_set"], global_status=False)
    )
    planted = ScopedIdentityRecord(
        0, obs, scope_for(range(fam.m), case["query_set"], global_status=True)
    )
    valid = validate_scope_promotion(
        empirical, case, range(fam.m), case["query_set"])
    planted_validation = validate_scope_promotion(
        planted, case, range(fam.m), case["query_set"])
    false_main = int(not valid["passed"])
    false_plant = int(not planted_validation["passed"])
    return {
        "constructible": True,
        "case": case,
        "promotions": 1,
        "record": empirical.canon(),
        "validation": valid,
        "false_global_promotions": false_main,
        "calibration_false_global_promotions": false_plant,
        "calibration": {
            "plant": planted.canon(),
            "same_validator": planted_validation,
            "fires": valid["passed"] and not planted_validation["passed"],
        },
    }


def find_new_identity_plan(fam, identities):
    """Find a novel convention and two fixed onboarding questions.

    The questions are selected from the stored supports, before the partner
    is exposed.  The stream generator then chooses a new convention that the
    pair distinguishes from every old record.  This is an authored
    validation population, not an oracle query policy.
    """
    masks = [SM.surviving_mask(fam, i.grounded) for i in identities]
    # Rank query pairs by how many distinct record-answer signatures they
    # create.  Tie-breaking is lexical and independent of the future phi.
    best_pair = None
    best_classes = -1
    for z1, z2 in itertools.combinations(range(fam.m), 2):
        signatures = set()
        for mask in masks:
            vals = np.where(mask)[0]
            signatures.add(tuple(sorted({(int(fam.u3[p, z1]),
                                           int(fam.u3[p, z2]))
                                          for p in vals})))
        if len(signatures) > best_classes:
            best_classes = len(signatures)
            best_pair = (z1, z2)
    if best_pair is None:
        return None
    z1, z2 = best_pair
    for phi in range(fam.n):
        if any(mask[phi] for mask in masks):
            continue
        answers = (int(fam.u3[phi, z1]), int(fam.u3[phi, z2]))
        compatible = []
        for mask in masks:
            keep = mask & (fam.u3[:, z1] == answers[0])
            keep &= fam.u3[:, z2] == answers[1]
            compatible.append(bool(keep.any()))
        if not any(compatible):
            return {"phi": phi, "questions": best_pair,
                    "answers": answers, "old_compatible": compatible}
    # If the single best pair has no witness, search all pairs without any
    # unbounded retry loop.
    for z1, z2 in itertools.combinations(range(fam.m), 2):
        for phi in range(fam.n):
            if any(mask[phi] for mask in masks):
                continue
            a1, a2 = int(fam.u3[phi, z1]), int(fam.u3[phi, z2])
            if all(not (mask & (fam.u3[:, z1] == a1)
                            & (fam.u3[:, z2] == a2)).any()
                   for mask in masks):
                return {"phi": phi, "questions": (z1, z2),
                        "answers": (a1, a2),
                        "old_compatible": [False] * len(masks)}
    return None


def _compatible_records(fam, identities, questions, answers):
    survivors = []
    for i in identities:
        mask = SM.surviving_mask(fam, i.grounded)
        for z, u in zip(questions, answers):
            mask &= fam.u3[:, z] == u
        survivors.append(mask)
    return survivors, [j for j, mask in enumerate(survivors) if mask.any()]


def _classify_from_answers(fam, identities, questions, answers):
    _survivors, live = _compatible_records(
        fam, identities, questions, answers)
    if not live:
        return LI.CREATE_NEW, live
    if len(live) == 1:
        return LI.ASSIGN_EXISTING, live
    return LI.UNRESOLVED_IDENTITY, live


def _nearest_record(fam, identities, questions, answers) -> int:
    """Bounded surface-independent nearest record for forced bad arms."""
    ranked = []
    for j, identity in enumerate(identities):
        support = np.flatnonzero(SM.surviving_mask(fam, identity.grounded))
        best = 0
        for phi in support:
            best = max(best, sum(
                int(int(fam.u3[int(phi), int(z)]) == int(u))
                for z, u in zip(questions, answers)))
        ranked.append((-best, j))
    return min(ranked)[1]


def _identity_transition(fam, identities, questions, answers, policy: str, *,
                         is_new: bool, true_slot: int | None) -> dict:
    """One shared transition/evaluator for MAIN and all required baselines."""
    _survivors, compatible = _compatible_records(
        fam, identities, questions, answers)
    nearest = _nearest_record(fam, identities, questions, answers)
    assigned = None
    if policy == "main":
        if not compatible:
            outcome = LI.CREATE_NEW
        elif len(compatible) == 1:
            outcome, assigned = LI.ASSIGN_EXISTING, compatible[0]
        else:
            outcome = LI.UNRESOLVED_IDENTITY
    elif policy == "always_reuse_nearest":
        outcome, assigned = LI.ASSIGN_EXISTING, nearest
    elif policy == "no_new_unresolved":
        if len(compatible) == 1:
            outcome, assigned = LI.ASSIGN_EXISTING, compatible[0]
        else:
            outcome = LI.UNRESOLVED_IDENTITY
    elif policy == "no_new_forced":
        outcome = LI.ASSIGN_EXISTING
        assigned = compatible[0] if len(compatible) == 1 else nearest
    elif policy == "always_create_new":
        outcome = LI.CREATE_NEW
    elif policy == "oracle_new_returning_status":
        outcome = LI.CREATE_NEW if is_new else LI.ASSIGN_EXISTING
        assigned = None if is_new else int(true_slot)
    else:
        raise ValueError(f"unknown NEW policy {policy!r}")
    return {
        "policy": policy,
        "truth": "new" if is_new else "returning",
        "true_slot": true_slot,
        "outcome": outcome,
        "assigned_record": assigned,
        "compatible_records": tuple(compatible),
        "forced_assimilation": int(is_new and outcome == LI.ASSIGN_EXISTING),
        "false_new": int(not is_new and outcome == LI.CREATE_NEW),
        "correct_status": bool(
            (is_new and outcome == LI.CREATE_NEW)
            or (not is_new and outcome == LI.ASSIGN_EXISTING)),
    }


def _identity_metrics(trials) -> dict:
    trials = tuple(trials)
    new = [r for r in trials if r["truth"] == "new"]
    returning = [r for r in trials if r["truth"] == "returning"]
    tp = sum(r["outcome"] == LI.CREATE_NEW for r in new)
    fp = sum(r["outcome"] == LI.CREATE_NEW for r in returning)
    fn = len(new) - tp
    forced = sum(r["forced_assimilation"] for r in new)
    unresolved = sum(r["outcome"] == LI.UNRESOLVED_IDENTITY for r in new)
    predicted_new = tp + fp
    return {
        "new_trials": len(new),
        "returning_trials": len(returning),
        "true_new_created": tp,
        "returning_false_new": fp,
        "new_false_negative": fn,
        "new_forced_assimilation": forced,
        "new_unresolved": unresolved,
        "precision": (Fraction(tp, predicted_new)
                      if predicted_new else Fraction(0)),
        "recall": Fraction(tp, len(new)) if new else Fraction(0),
        "new_recall": Fraction(tp, len(new)) if new else Fraction(0),
        "false_new_rate_returning": (
            Fraction(fp, len(returning)) if returning else Fraction(0)),
        "forced_assimilation_rate": (
            Fraction(forced, len(new)) if new else Fraction(0)),
        "unresolved_new_rate": (
            Fraction(unresolved, len(new)) if new else Fraction(0)),
        "decisions": trials,
    }


def _classification_gate(metrics) -> bool:
    return bool(
        metrics["precision"] == 1
        and metrics["recall"] == 1
        and metrics["false_new_rate_returning"] == 0
        and metrics["forced_assimilation_rate"] == 0
        and metrics["unresolved_new_rate"] == 0)


def _promotion_validation(row) -> dict:
    """One validator for the audited promotion and its planted bad arms."""
    checks = {
        "classified_as_new": bool(row["classified_as_new"]),
        "candidate_absent_from_confirmed_store": bool(
            row["candidate_absent_from_confirmed_store"]),
        "provisional_branch_opened": bool(row["provisional_branch_opened"]),
        "confirmed_unchanged_while_provisional": bool(
            row["confirmed_unchanged_while_provisional"]),
        "main_outcome_promotes": row["outcome"] == P.PROMOTE,
        "branch_records_promotion": row["branch_status"] == P.PROMOTE,
        "confirmation_was_queried": int(row["queries_used"]) > 0,
        "confirmed_version_advanced": int(row["confirmed_version_after"])
            == int(row["confirmed_version_before"]) + 1,
        "truth_survives_promoted_record": bool(
            row["truth_survives_promoted_record"]),
        "trigger_retained_after_promotion": bool(
            row["trigger_retained_after_promotion"]),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _promotion_trace(fam, plan, key: int, seed: int, arm: str,
                     *, classified_as_new: bool,
                     candidate_absent_from_confirmed_store: bool) -> dict:
    """Execute the trusted S2 provisional/challenge/promotion transition."""
    questions = tuple(int(v) for v in plan["questions"])
    answers = tuple(int(v) for v in plan["answers"])
    trigger = (questions[0], answers[0])
    # The classification questions were already paid for.  The confirmation
    # mechanism may not ask either one again.
    legal = tuple(z for z in range(fam.m) if z not in set(questions))
    candidate = P.ConfirmedState(str(key))
    before_hash = hashlib.sha256(encode(candidate)).hexdigest()
    rng = random.Random(INF.stable_seed(
        "x65a-l1-new-promotion", fam.spec.overlap, seed, arm,
        int(plan["phi"]), trigger))
    outcome, confirmed, branch, used = P.resolve(
        fam, candidate, trigger, int(plan["phi"]), arm, legal, rng,
        budget=P.CHALLENGE_BUDGET)
    after_input_hash = hashlib.sha256(encode(candidate)).hexdigest()
    promoted_mask = SM.surviving_mask(fam, confirmed.grounded)
    row = {
        "arm": arm,
        "classified_as_new": classified_as_new,
        "candidate_absent_from_confirmed_store":
            candidate_absent_from_confirmed_store,
        "trigger": trigger,
        "legal_confirmation_questions": legal,
        "outcome": outcome,
        "queries_used": used,
        "provisional_branch_opened": branch is not None,
        "branch_status": None if branch is None else branch.status,
        "branch": None if branch is None else branch.canon(),
        "confirmed_before": candidate.canon(),
        "confirmed_after": confirmed.canon(),
        "confirmed_version_before": candidate.version,
        "confirmed_version_after": confirmed.version,
        "confirmed_unchanged_while_provisional": bool(
            branch is not None and before_hash == after_input_hash),
        "truth_survives_promoted_record": bool(
            promoted_mask[int(plan["phi"])]),
        "trigger_retained_after_promotion": any(
            int(g.z) == trigger[0] and int(g.u) == trigger[1]
            for g in confirmed.grounded),
        "confirmed_state": confirmed,
    }
    row["validation"] = _promotion_validation(row)
    return row


def _reuse_validation(row) -> dict:
    """One validator for genuine returning-identity reuse and red plants."""
    shortlist = tuple(row.get("shortlist") or ())
    checks = {
        "promotion_passed": bool(row.get("promotion_passed")),
        "promoted_key_in_shortlist": row.get("promoted_key") in shortlist,
        "promoted_key_is_identity_top": (
            row.get("identity_top") == row.get("promoted_key")),
        "assigns_existing_identity": (
            row.get("identity_decision") == LI.ASSIGN_EXISTING),
        "task_action_correct": int(row.get("task_accuracy") or 0) == 1,
        "no_unresolved_branch": int(
            row.get("unresolved_branches") or 0) == 0,
        "no_stable_id_oracle": not bool(row.get("used_stable_identity")),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _first_later_task(fam, beh, phi: int, forbidden, seed: int):
    """First seeded valid task; selection never conditions on audit success."""
    from . import l_suite as LS

    cfg = EP.Config(overlap=fam.spec.overlap)
    rng = random.Random(INF.stable_seed(
        "x65a-l1-new-reuse", fam.spec.overlap, seed, phi))
    for z in range(fam.m):
        if z in set(int(v) for v in forbidden):
            continue
        task = LS._transfer_task(fam, beh, cfg, phi, z, rng)
        if task is not None and task.live and task.z in task.live:
            return task
    return None


def new_identity_audit(fam, identities, beh=None, seed: int = 6400) -> dict:
    """Counted NEW classification, real promotion, and latent later reuse."""
    plan = find_new_identity_plan(fam, identities)
    if plan is None:
        return {"constructible": False}
    q = plan["questions"]
    a = plan["answers"]
    policies = (
        "main", "always_reuse_nearest", "no_new_unresolved",
        "no_new_forced", "always_create_new",
        "oracle_new_returning_status")
    by_policy = {policy: [] for policy in policies}
    for policy in policies:
        by_policy[policy].append(_identity_transition(
            fam, identities, q, a, policy, is_new=True, true_slot=None))
        for identity in identities:
            answers = tuple(int(fam.u3[identity.phi, z]) for z in q)
            by_policy[policy].append(_identity_transition(
                fam, identities, q, answers, policy, is_new=False,
                true_slot=identity.slot))
    metrics = {policy: _identity_metrics(rows)
               for policy, rows in by_policy.items()}
    main = metrics["main"]
    outcome = by_policy["main"][0]["outcome"]
    store = SM.SemanticStore()
    for identity in identities:
        store.put(SM.SemanticRecord(
            str(identity.slot), tuple(identity.grounded), status=Status.CONFIRMED,
            surviving=int(SM.surviving_mask(fam, identity.grounded).sum())))
    store_records_before_promotion = len(store.records)
    old_hashes = tuple(hashlib.sha256(encode(r)).hexdigest()
                       for _k, r in sorted(store.records.items()))

    promotion = _promotion_trace(
        fam, plan, len(identities), seed, "main",
        classified_as_new=outcome == LI.CREATE_NEW,
        candidate_absent_from_confirmed_store=(
            str(len(identities)) not in store.records))
    promoted_state = promotion["confirmed_state"]
    challenge_queries = tuple(
        int(z) for z, _answer in (
            () if promotion["branch"] is None
            else promotion["branch"]["answers"]))
    verification_queries = tuple(q) + challenge_queries
    rec = ScopedIdentityRecord(
        len(identities), tuple(promoted_state.grounded),
        scope_for(range(fam.m), verification_queries, global_status=False),
    )
    new_mask = SM.surviving_mask(fam, rec.grounded)
    promoted = bool(promotion["validation"]["passed"])
    if promoted:
        store.put(rec)
    after_old_hashes = tuple(hashlib.sha256(encode(store.records[str(i.slot)])).hexdigest()
                             for i in identities)

    reuse_support = bool(promoted and new_mask[plan["phi"]])
    reuse_accuracy = None
    reuse_task = None
    reuse_top = None
    reuse_accounting = None
    reuse_identity_decision = None
    reuse_identity_posterior = None
    reuse_task_posterior = None
    reuse_shortlist = None
    reuse_action = None
    reuse_query_history = None
    reuse_unresolved_branches = None
    reuse_validation = None
    reuse_validator_control = None
    if beh is not None and reuse_support:
        forbidden = tuple(int(g.z) for g in rec.grounded) + tuple(q)
        task = _first_later_task(
            fam, beh, int(plan["phi"]), forbidden, seed)
        if task is not None:
            records = {int(key): value for key, value in store.records.items()}
            index = RET.build_global_exact_index(records)
            retrieval = RET.retrieve_protocol_a(index, fam, task)
            masks = [RET.support_from_sketch(fam, entry.sketch)
                     for entry in index.entries]
            bool_masks = []
            for support in masks:
                mask = np.zeros(fam.n, dtype=bool)
                mask[list(support)] = True
                bool_masks.append(mask)
            initial = MAIN.latent_state(fam, task, bool_masks)
            run = MAIN.run_policy(
                initial, MAIN.INFORMATION_GAIN, 1, int(plan["phi"]),
                int(task.z), tuple(range(8)), seed)
            post = run.state.identity_posterior()
            ordered = sorted(post, key=lambda key: (
                not isinstance(key, int), str(key)))
            reuse_top = max(ordered, key=lambda key: post[key]) if ordered else None
            reuse_accuracy = int(run.correct)
            reuse_task = {"z": task.z, "u": task.u,
                          "live": tuple(task.live)}
            reuse_accounting = retrieval.accounting.canon()
            reuse_identity_decision = run.identity_decision
            reuse_identity_posterior = {
                str(key): value for key, value in sorted(
                    post.items(), key=lambda row: str(row[0]))}
            reuse_task_posterior = run.state.task_posterior()
            reuse_shortlist = retrieval.selected_keys
            reuse_action = run.action
            reuse_query_history = run.state.history
            reuse_unresolved_branches = int(
                run.identity_decision == LI.UNRESOLVED_IDENTITY)
            reuse_row = {
                "promotion_passed": promoted,
                "promoted_key": rec.key,
                "shortlist": retrieval.selected_keys,
                "identity_top": reuse_top,
                "identity_decision": run.identity_decision,
                "task_accuracy": reuse_accuracy,
                "unresolved_branches": reuse_unresolved_branches,
                "used_stable_identity": False,
            }
            reuse_validation = _reuse_validation(reuse_row)

            # Positive validator control: the same exact open-world adapter,
            # but a one-record fixture in which ASSIGN_EXISTING is genuinely
            # decisive.  This is not used for the audited eight-record result.
            one_record = MAIN.subset_state(
                fam, task, bool_masks, (rec.key,), with_new=True,
                with_out=True)
            control_run = MAIN.run_policy(
                one_record, MAIN.INFORMATION_GAIN, 0, int(plan["phi"]),
                int(task.z), tuple(range(8)), seed)
            control_row = {
                "promotion_passed": promoted,
                "promoted_key": rec.key,
                "shortlist": (rec.key,),
                "identity_top": rec.key,
                "identity_decision": control_run.identity_decision,
                "task_accuracy": int(control_run.correct),
                "unresolved_branches": int(
                    control_run.identity_decision == LI.UNRESOLVED_IDENTITY),
                "used_stable_identity": False,
            }
            reuse_validator_control = {
                "fixture": "one_record_open_world_q0_not_stable_identity",
                "row": control_row,
                "validation": _reuse_validation(control_row),
            }

    arms = {name: metrics[name] for name in policies if name != "main"}
    bad_names = ("always_reuse_nearest", "no_new_unresolved",
                 "no_new_forced", "always_create_new")
    quarantine_promotion = _promotion_trace(
        fam, plan, len(identities), seed, "always_quarantine",
        classified_as_new=outcome == LI.CREATE_NEW,
        candidate_absent_from_confirmed_store=True)
    bypass_promotion = _promotion_trace(
        fam, plan, len(identities), seed, "confirmation_bypass",
        classified_as_new=outcome == LI.CREATE_NEW,
        candidate_absent_from_confirmed_store=True)
    promotion_calibration = {
        "main_positive_control_accepted": promotion["validation"]["passed"],
        "always_quarantine_rejected": not quarantine_promotion[
            "validation"]["passed"],
        "confirmation_bypass_rejected": not bypass_promotion[
            "validation"]["passed"],
    }
    promotion_calibration["same_validator_rejections"] = all(
        promotion_calibration.values())

    control_row = (None if reuse_validator_control is None
                   else reuse_validator_control["row"])
    if control_row is None:
        reuse_calibration = {
            "positive_control_accepted": False,
            "drop_promoted_shortlist_rejected": False,
            "unresolved_branch_rejected": False,
            "wrong_action_rejected": False,
            "same_validator_rejections": False,
        }
    else:
        dropped = {**control_row, "shortlist": ()}
        unresolved_plant = {
            **control_row,
            "identity_decision": LI.UNRESOLVED_IDENTITY,
            "unresolved_branches": 1,
        }
        wrong_action = {**control_row, "task_accuracy": 0}
        reuse_calibration = {
            "positive_control_accepted": _reuse_validation(
                control_row)["passed"],
            "drop_promoted_shortlist_rejected": not _reuse_validation(
                dropped)["passed"],
            "unresolved_branch_rejected": not _reuse_validation(
                unresolved_plant)["passed"],
            "wrong_action_rejected": not _reuse_validation(
                wrong_action)["passed"],
        }
        reuse_calibration["same_validator_rejections"] = all(
            reuse_calibration.values())

    calibrations = {
        "always_reuse_forces_assimilation": (
            not _classification_gate(arms["always_reuse_nearest"])
            and arms["always_reuse_nearest"]["forced_assimilation_rate"] == 1),
        "always_create_false_new": (
            not _classification_gate(arms["always_create_new"])
            and arms["always_create_new"]["returning_false_new"]
            == len(identities)),
        "unresolved_does_not_count_as_recall": (
            not _classification_gate(arms["no_new_unresolved"])
            and arms["no_new_unresolved"]["recall"] == 0
            and arms["no_new_unresolved"]["new_false_negative"] == 1),
        "no_new_forced_is_rejected": (
            not _classification_gate(arms["no_new_forced"])
            and arms["no_new_forced"]["forced_assimilation_rate"] == 1),
        "oracle_control_passes_same_gate": _classification_gate(
            arms["oracle_new_returning_status"]),
        "promotion_same_validator_rejects_bad_arms":
            promotion_calibration["same_validator_rejections"],
        "reuse_same_validator_rejects_plants":
            reuse_calibration["same_validator_rejections"],
    }
    assert all(not _classification_gate(arms[name]) for name in bad_names)

    return {
        "constructible": True,
        "plan": plan,
        "confusion": {k: main[k] for k in (
            "new_trials", "returning_trials", "true_new_created",
            "returning_false_new", "new_false_negative",
            "new_forced_assimilation", "new_unresolved")},
        "precision": main["precision"],
        "recall": main["recall"],
        "false_new_rate_returning": main["false_new_rate_returning"],
        "forced_assimilation_rate": main["forced_assimilation_rate"],
        "unresolved_new_rate": main["unresolved_new_rate"],
        "questions_to_grounded_creation": len(q) + promotion["queries_used"],
        "successfully_promoted_new_records": int(promoted),
        "later_reuse_of_new_records": int(bool(
            reuse_validation and reuse_validation["passed"])),
        "later_reuse_task_accuracy": reuse_accuracy,
        "later_reuse_task": reuse_task,
        "later_reuse_identity_top": reuse_top,
        "later_reuse_identity_decision": reuse_identity_decision,
        "later_reuse_identity_posterior": reuse_identity_posterior,
        "later_reuse_task_posterior": reuse_task_posterior,
        "later_reuse_shortlist": reuse_shortlist,
        "later_reuse_action": reuse_action,
        "later_reuse_query_history": reuse_query_history,
        "later_reuse_unresolved_branches": reuse_unresolved_branches,
        "later_reuse_validation": reuse_validation,
        "later_reuse_retrieval_accounting": reuse_accounting,
        "later_reuse_used_stable_identity": False,
        "promoted_record_key": rec.key,
        "store_records_before_promotion": store_records_before_promotion,
        "store_records_after_creation": len(store.records),
        "contamination_during_creation": int(
            not new_mask[plan["phi"]] or old_hashes != after_old_hashes),
        "record_bytes_added": rec.bytes() if promoted else 0,
        "record": rec.canon(),
        "promotion": {key: value for key, value in promotion.items()
                      if key != "confirmed_state"},
        "promotion_calibration": promotion_calibration,
        "reuse_validator_control": reuse_validator_control,
        "reuse_calibration": reuse_calibration,
        "main_decisions": main["decisions"],
        "arms": arms,
        "main_classification_gate": _classification_gate(main),
        "calibration_fired": calibrations,
    }


def stratum_constructibility(fam, beh) -> dict:
    alien = authored_out_of_family_convention(fam)
    grounded = (None if alien is None else alien.grounded_contradiction)
    transfer = transfer_out_of_family_utterance(fam)
    unknown = unknown_meaning_demonstrations(beh)
    restricted = restricted_indistinguishable_case(fam)
    grounded_obs = tuple(
        SM.GroundedObservation(z, u, f"oof:{k}")
        for k, (z, u) in enumerate(grounded or ()))
    zero_survivors = bool(
        grounded_obs and not SM.surviving_mask(fam, grounded_obs).any())
    # Exercise the trusted S2 cause model.  Supplying the complete grounded
    # contradiction as observed answers must activate its exact MISSING mass;
    # this is an outcome, not a renamed constructibility boolean.
    missing_cause = (P.cause_posterior(
        fam, np.ones(fam.n, dtype=bool), grounded[0], answers=grounded)
        if grounded else {P.MISSING: Fraction(0)})
    missing_outcome = (P.MISSING
                       if missing_cause.get(P.MISSING) == 1 else P.UNRESOLVED)
    membership = (None if alien is None else int(np.all(
        fam.u3 == np.asarray(alien.calibration_codes,
                             dtype=fam.u3.dtype)[None, :], axis=1).sum()))
    return {
        "out_of_family_convention": {
            "constructible": alien is not None,
            "convention": None if alien is None else alien.canon(),
            "family_membership_count": membership,
            "tested_via": "authored noninjective role map, then grounded",
            "grounded_contradiction": {
                "events": grounded,
                "zero_survivors": zero_survivors,
            },
        },
        "out_of_family_transfer_utterance": {
            "constructible": transfer is not None,
            "utterance": transfer,
            "scope": ("untestable in frozen shared two-token alphabet"
                      if transfer is None and fam.spec.overlap == "shared"
                      else "nonvacuous frozen-alphabet transfer"),
            "zero_family_likelihood": bool(
                transfer is not None and
                fam.counts(transfer, FAM.P2).sum() == 0),
        },
        "out_of_family_grounded_event": {
            "constructible": grounded is not None,
            "events": grounded,
            "minimum_event_count": len(grounded) if grounded else None,
            "source": "authored_out_of_family_convention",
            "zero_survivors": zero_survivors,
        },
        "UNKNOWN_MEANING": {
            "constructible": unknown is not None,
            "demonstrations": unknown,
            "derived_live_count": len(derive_live_from_demonstrations(
                beh, unknown)) if unknown is not None else None,
        },
        "MISSING_REPRESENTATION": {
            "constructible": grounded is not None,
            "tested": grounded is not None,
            "outcome": missing_outcome,
            "cause_posterior": missing_cause,
        },
        "restricted_query_indistinguishable": {
            "constructible": restricted is not None,
            "case": restricted,
        },
    }
