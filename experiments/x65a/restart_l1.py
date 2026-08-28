"""Exact, resumable restart boundary for the X65A-L1 MAIN learner.

The checkpoint persists every input needed to reconstruct
``l1_main.OpenWorldState`` after a real clarification: public task evidence,
post-query record and NEW supports, exact priors, a seal of the recomputed
selection-aware weights, policy/history/budget, provisional state, and the
retrieval shortlist.  Evaluator truth is never checkpointed.

The parent validates and writes the checkpoint, computes an uninterrupted
MAIN clarification, and exits.  A scrubbed child reads only those bytes and a
public query/answer suffix, reconstructs the production state, selects the
same information-gain question, applies the answer, and must produce the same
exact final checkpoint hash.  Synthetic Bayes-factor transitions are not a
supported restart path.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .types import decode, encode


SCHEMA = 2
NEW_IDENTITY = "NEW_IDENTITY"
OUT_OF_FAMILY = "OUT_OF_FAMILY"
FORBIDDEN_ENV = "X65A_L1_FORBIDDEN_TARGET"
FORBIDDEN_VALUE = "future_latent_target_answer"
_RUNTIME_CACHE: dict[str, str] = {}

# Every reconstructing component has its own content seal and its own
# drop/mutation calibration through the production child loader.
HASHED_FIELDS = (
    "identity_posterior",
    "record_convention_posteriors",
    "new_mass",
    "out_mass",
    "confirmed_records",
    "provisional_branches",
    "retrieval_shortlist",
    "retrieval_accounting",
    "task_evidence",
    "post_query_record_supports",
    "post_query_new_support",
    "inference_priors",
    "selection_weights",
    "query_policy_state",
)
AUDITED_FIELDS = HASHED_FIELDS + ("serialized_hashes",)
STATE_FIELDS = frozenset(("schema", "overlap", "step") + AUDITED_FIELDS)


class RestartIntegrityError(ValueError):
    """The checkpoint is incomplete, non-exact, or not reconstructible."""


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _digest(obj: Any) -> str:
    return _sha(encode(obj))


def _require_keys(obj: Mapping[str, Any], required: set[str] | frozenset[str],
                  where: str) -> None:
    got = set(obj)
    missing = set(required) - got
    extra = got - set(required)
    if missing or extra:
        raise RestartIntegrityError(
            f"{where} keys mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}")


def _frac(value: Any, where: str) -> Fraction:
    if isinstance(value, bool):
        raise RestartIntegrityError(f"{where} must be an exact rational")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    raise RestartIntegrityError(
        f"{where} must be Fraction/int, got {type(value).__name__}")


def _posterior_from_obj(obj: Any, where: str
                        ) -> tuple[tuple[str, Fraction], ...]:
    if not isinstance(obj, Mapping):
        raise RestartIntegrityError(f"{where} must be a mapping")
    return tuple(sorted((str(k), _frac(v, f"{where}.{k}"))
                        for k, v in obj.items()))


def _posterior_dict(pairs: Sequence[tuple[str, Fraction]]) -> dict[str, Fraction]:
    return {str(k): _frac(v, str(k)) for k, v in pairs}


def _validate_posterior(pairs: Sequence[tuple[str, Fraction]], where: str,
                        required_keys: set[str] | None = None) -> None:
    if not pairs:
        raise RestartIntegrityError(f"{where} is empty")
    keys = [str(k) for k, _v in pairs]
    if len(keys) != len(set(keys)):
        raise RestartIntegrityError(f"{where} contains duplicate keys")
    values = [_frac(v, f"{where}.{k}") for k, v in pairs]
    if any(v < 0 for v in values):
        raise RestartIntegrityError(f"{where} contains negative mass")
    if sum(values, Fraction(0)) != 1:
        raise RestartIntegrityError(f"{where} does not sum exactly to one")
    if required_keys is not None and set(keys) != required_keys:
        raise RestartIntegrityError(f"{where} support mismatch")


def _support_map_from_obj(obj: Any, where: str
                          ) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if not isinstance(obj, Mapping):
        raise RestartIntegrityError(f"{where} must be a mapping")
    return tuple(sorted((str(key), tuple(int(x) for x in support))
                        for key, support in obj.items()))


def _validate_support(support: Sequence[int], fam, where: str,
                      *, allow_empty: bool = True) -> None:
    values = tuple(int(x) for x in support)
    if not allow_empty and not values:
        raise RestartIntegrityError(f"{where} is empty")
    if tuple(sorted(set(values))) != values:
        raise RestartIntegrityError(f"{where} is not sorted and unique")
    if any(phi < 0 or phi >= fam.n for phi in values):
        raise RestartIntegrityError(f"{where} contains an invalid convention")


def _scaled_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array, dtype=np.dtype("<i8"))
    return _sha(value.tobytes(order="C"))


@dataclass(frozen=True)
class GroundedPair:
    z: int
    u: int
    evidence_id: str

    def canon(self) -> dict[str, Any]:
        return {"z": self.z, "u": self.u, "evidence_id": self.evidence_id}

    @classmethod
    def from_obj(cls, obj: Any) -> "GroundedPair":
        if not isinstance(obj, Mapping):
            raise RestartIntegrityError("grounded pair must be a mapping")
        _require_keys(obj, {"z", "u", "evidence_id"}, "grounded pair")
        return cls(int(obj["z"]), int(obj["u"]), str(obj["evidence_id"]))


@dataclass(frozen=True)
class RecordConventionPosterior:
    record_key: str
    support: tuple[int, ...]
    posterior: tuple[tuple[int, Fraction], ...]

    def canon(self) -> dict[str, Any]:
        return {"record_key": self.record_key, "support": list(self.support),
                "posterior": {str(phi): p for phi, p in self.posterior}}

    @classmethod
    def from_obj(cls, obj: Any) -> "RecordConventionPosterior":
        if not isinstance(obj, Mapping):
            raise RestartIntegrityError("record posterior must be a mapping")
        _require_keys(obj, {"record_key", "support", "posterior"},
                      "record posterior")
        if not isinstance(obj["posterior"], Mapping):
            raise RestartIntegrityError("record posterior weights must map phi")
        return cls(
            str(obj["record_key"]), tuple(int(v) for v in obj["support"]),
            tuple(sorted((int(k), _frac(v, f"posterior.{k}"))
                         for k, v in obj["posterior"].items())))


@dataclass(frozen=True)
class ConfirmedRecordSnapshot:
    record_key: str
    grounded: tuple[GroundedPair, ...]
    verification_domain: tuple[int, ...]
    challenge_digest: str
    query_set_digest: str
    validity_scope: str
    status: str
    version: int
    evidence_ids: tuple[str, ...]

    def canon(self) -> dict[str, Any]:
        return {
            "record_key": self.record_key,
            "grounded": [g.canon() for g in self.grounded],
            "verification_domain": list(self.verification_domain),
            "challenge_digest": self.challenge_digest,
            "query_set_digest": self.query_set_digest,
            "validity_scope": self.validity_scope,
            "status": self.status,
            "version": self.version,
            "evidence_ids": list(self.evidence_ids),
        }

    @classmethod
    def from_obj(cls, obj: Any) -> "ConfirmedRecordSnapshot":
        if not isinstance(obj, Mapping):
            raise RestartIntegrityError("confirmed record must be a mapping")
        fields = {"record_key", "grounded", "verification_domain",
                  "challenge_digest", "query_set_digest", "validity_scope",
                  "status", "version", "evidence_ids"}
        _require_keys(obj, fields, "confirmed record")
        return cls(
            str(obj["record_key"]),
            tuple(GroundedPair.from_obj(g) for g in obj["grounded"]),
            tuple(int(v) for v in obj["verification_domain"]),
            str(obj["challenge_digest"]), str(obj["query_set_digest"]),
            str(obj["validity_scope"]), str(obj["status"]),
            int(obj["version"]), tuple(str(v) for v in obj["evidence_ids"]))


@dataclass(frozen=True)
class TaskEvidenceSnapshot:
    """Public current task only; evaluator truth ``z`` is absent."""

    kind: str
    demos: tuple[int, ...]
    live: tuple[int, ...]
    u: int
    pool: tuple[tuple[str, ...], ...]
    tie: tuple[int, ...]
    accepted: bool

    def canon(self) -> dict[str, Any]:
        return {"kind": self.kind, "demos": list(self.demos),
                "live": list(self.live), "u": self.u,
                "pool": [list(pattern) for pattern in self.pool],
                "tie": list(self.tie), "accepted": self.accepted}

    @classmethod
    def from_obj(cls, obj: Any) -> "TaskEvidenceSnapshot":
        if not isinstance(obj, Mapping):
            raise RestartIntegrityError("task evidence must be a mapping")
        _require_keys(obj, {"kind", "demos", "live", "u", "pool", "tie",
                            "accepted"}, "task evidence")
        if not isinstance(obj["accepted"], bool):
            raise RestartIntegrityError("task accepted flag must be boolean")
        return cls(str(obj["kind"]), tuple(int(v) for v in obj["demos"]),
                   tuple(int(v) for v in obj["live"]), int(obj["u"]),
                   tuple(tuple(str(role) for role in pattern)
                         for pattern in obj["pool"]),
                   tuple(int(v) for v in obj["tie"]), bool(obj["accepted"]))

    @classmethod
    def from_task(cls, task) -> "TaskEvidenceSnapshot":
        return cls(str(task.kind), tuple(int(v) for v in task.demos),
                   tuple(int(v) for v in task.live), int(task.u),
                   tuple(tuple(str(role) for role in pattern)
                         for pattern in task.pool),
                   tuple(int(v) for v in task.tie), bool(task.accepted))

    def task(self):
        from x64h import episode as EP
        # -1 is a non-truth sentinel; inference never consumes task.z.
        return EP.Task(self.kind, -1, self.demos, self.live, self.u,
                       self.pool, self.tie, self.accepted)


@dataclass(frozen=True)
class InferencePriorsSnapshot:
    p_new: Fraction
    p_out: Fraction
    record_prior: Fraction
    with_new: bool
    with_out: bool

    def canon(self) -> dict[str, Any]:
        return {"p_new": self.p_new, "p_out": self.p_out,
                "record_prior": self.record_prior,
                "with_new": self.with_new, "with_out": self.with_out}

    @classmethod
    def from_obj(cls, obj: Any) -> "InferencePriorsSnapshot":
        if not isinstance(obj, Mapping):
            raise RestartIntegrityError("inference priors must be a mapping")
        _require_keys(obj, {"p_new", "p_out", "record_prior", "with_new",
                            "with_out"}, "inference priors")
        if not isinstance(obj["with_new"], bool) or not isinstance(
                obj["with_out"], bool):
            raise RestartIntegrityError("prior enable flags must be boolean")
        return cls(_frac(obj["p_new"], "p_new"),
                   _frac(obj["p_out"], "p_out"),
                   _frac(obj["record_prior"], "record_prior"),
                   bool(obj["with_new"]), bool(obj["with_out"]))


@dataclass(frozen=True)
class SelectionWeightsSnapshot:
    live: tuple[int, ...]
    denominator: int
    shape: tuple[int, int]
    scaled_sha256: str
    task_input_sha256: str
    family_signature_sha256: str
    reconstruction: str = "recompute_exact_selection_weights_from_task"

    def canon(self) -> dict[str, Any]:
        return {"live": list(self.live), "denominator": self.denominator,
                "shape": list(self.shape),
                "scaled_sha256": self.scaled_sha256,
                "task_input_sha256": self.task_input_sha256,
                "family_signature_sha256": self.family_signature_sha256,
                "reconstruction": self.reconstruction}

    @classmethod
    def from_obj(cls, obj: Any) -> "SelectionWeightsSnapshot":
        if not isinstance(obj, Mapping):
            raise RestartIntegrityError("selection weights must be a mapping")
        fields = {"live", "denominator", "shape", "scaled_sha256",
                  "task_input_sha256", "family_signature_sha256",
                  "reconstruction"}
        _require_keys(obj, fields, "selection weights")
        shape = tuple(int(v) for v in obj["shape"])
        if len(shape) != 2:
            raise RestartIntegrityError("selection weight shape is not rank two")
        return cls(tuple(int(v) for v in obj["live"]),
                   int(obj["denominator"]), (shape[0], shape[1]),
                   str(obj["scaled_sha256"]),
                   str(obj["task_input_sha256"]),
                   str(obj["family_signature_sha256"]),
                   str(obj["reconstruction"]))


@dataclass(frozen=True)
class QueryPolicySnapshot:
    policy: str
    legal_queries: tuple[int, ...]
    history: tuple[tuple[int, int], ...]
    query_budget: int
    stop_when_identity_decisive: bool

    def canon(self) -> dict[str, Any]:
        return {"policy": self.policy, "legal_queries": list(self.legal_queries),
                "history": [list(v) for v in self.history],
                "query_budget": self.query_budget,
                "stop_when_identity_decisive": self.stop_when_identity_decisive}

    @classmethod
    def from_obj(cls, obj: Any) -> "QueryPolicySnapshot":
        if not isinstance(obj, Mapping):
            raise RestartIntegrityError("query policy state must be a mapping")
        fields = {"policy", "legal_queries", "history", "query_budget",
                  "stop_when_identity_decisive"}
        _require_keys(obj, fields, "query policy state")
        stop = obj["stop_when_identity_decisive"]
        if not isinstance(stop, bool):
            raise RestartIntegrityError("query stop flag must be boolean")
        return cls(str(obj["policy"]),
                   tuple(int(v) for v in obj["legal_queries"]),
                   tuple((int(v[0]), int(v[1])) for v in obj["history"]),
                   int(obj["query_budget"]), bool(stop))


@dataclass(frozen=True)
class RetrievalAccountingSnapshot:
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

    def canon(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_obj(cls, obj: Any) -> "RetrievalAccountingSnapshot":
        if not isinstance(obj, Mapping):
            raise RestartIntegrityError("retrieval accounting must be a mapping")
        fields = {"protocol", "index_bytes_scanned",
                  "identity_specific_summaries_inspected",
                  "identity_likelihoods_evaluated", "shortlist_size",
                  "full_records_loaded", "sketch_bytes_loaded",
                  "total_retrieval_bytes",
                  "total_retrieval_node_equivalents", "incomplete_retrieval",
                  "within_512", "four_node_claim"}
        _require_keys(obj, fields, "retrieval accounting")
        bool_fields = ("incomplete_retrieval", "within_512", "four_node_claim")
        if any(not isinstance(obj[name], bool) for name in bool_fields):
            raise RestartIntegrityError("retrieval accounting flags must be bool")
        return cls(str(obj["protocol"]), int(obj["index_bytes_scanned"]),
                   int(obj["identity_specific_summaries_inspected"]),
                   int(obj["identity_likelihoods_evaluated"]),
                   int(obj["shortlist_size"]), int(obj["full_records_loaded"]),
                   int(obj["sketch_bytes_loaded"]),
                   int(obj["total_retrieval_bytes"]),
                   int(obj["total_retrieval_node_equivalents"]),
                   bool(obj["incomplete_retrieval"]), bool(obj["within_512"]),
                   bool(obj["four_node_claim"]))

    @classmethod
    def from_accounting(cls, accounting) -> "RetrievalAccountingSnapshot":
        return cls(**accounting.canon())


@dataclass(frozen=True)
class ProvisionalBranchSnapshot:
    branch_id: str
    identity_posterior: tuple[tuple[str, Fraction], ...]
    record_support: tuple[tuple[str, tuple[int, ...]], ...]
    new_support: tuple[int, ...]
    new_mass: Fraction
    out_mass: Fraction
    evidence_ids: tuple[str, ...]
    asked: tuple[tuple[int, int], ...]
    query_universe: tuple[int, ...]
    policy: str
    status: str
    update_budget: int

    def canon(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "identity_posterior": _posterior_dict(self.identity_posterior),
            "record_support": {k: list(v) for k, v in self.record_support},
            "new_support": list(self.new_support),
            "new_mass": self.new_mass, "out_mass": self.out_mass,
            "evidence_ids": list(self.evidence_ids),
            "asked": [list(a) for a in self.asked],
            "query_universe": list(self.query_universe),
            "policy": self.policy, "status": self.status,
            "update_budget": self.update_budget,
        }

    @classmethod
    def from_obj(cls, obj: Any) -> "ProvisionalBranchSnapshot":
        if not isinstance(obj, Mapping):
            raise RestartIntegrityError("provisional branch must be a mapping")
        fields = {"branch_id", "identity_posterior", "record_support",
                  "new_support", "new_mass", "out_mass", "evidence_ids",
                  "asked", "query_universe", "policy", "status",
                  "update_budget"}
        _require_keys(obj, fields, "provisional branch")
        return cls(
            str(obj["branch_id"]),
            _posterior_from_obj(obj["identity_posterior"],
                                "branch.identity_posterior"),
            _support_map_from_obj(obj["record_support"],
                                  "branch.record_support"),
            tuple(int(v) for v in obj["new_support"]),
            _frac(obj["new_mass"], "branch.new_mass"),
            _frac(obj["out_mass"], "branch.out_mass"),
            tuple(str(v) for v in obj["evidence_ids"]),
            tuple((int(v[0]), int(v[1])) for v in obj["asked"]),
            tuple(int(v) for v in obj["query_universe"]),
            str(obj["policy"]), str(obj["status"]), int(obj["update_budget"]))


@dataclass(frozen=True)
class LatentRestartState:
    schema: int
    overlap: str
    step: int
    identity_posterior: tuple[tuple[str, Fraction], ...]
    record_convention_posteriors: tuple[RecordConventionPosterior, ...]
    new_mass: Fraction
    out_mass: Fraction
    confirmed_records: tuple[ConfirmedRecordSnapshot, ...]
    provisional_branches: tuple[ProvisionalBranchSnapshot, ...]
    retrieval_shortlist: tuple[str, ...]
    retrieval_accounting: RetrievalAccountingSnapshot
    task_evidence: TaskEvidenceSnapshot
    post_query_record_supports: tuple[tuple[str, tuple[int, ...]], ...]
    post_query_new_support: tuple[int, ...]
    inference_priors: InferencePriorsSnapshot
    selection_weights: SelectionWeightsSnapshot
    query_policy_state: QueryPolicySnapshot
    serialized_hashes: tuple[tuple[str, str], ...] = ()

    def canon(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "overlap": self.overlap, "step": self.step,
            "identity_posterior": _posterior_dict(self.identity_posterior),
            "record_convention_posteriors":
                [r.canon() for r in self.record_convention_posteriors],
            "new_mass": self.new_mass, "out_mass": self.out_mass,
            "confirmed_records": [r.canon() for r in self.confirmed_records],
            "provisional_branches":
                [b.canon() for b in self.provisional_branches],
            "retrieval_shortlist": list(self.retrieval_shortlist),
            "retrieval_accounting": self.retrieval_accounting.canon(),
            "task_evidence": self.task_evidence.canon(),
            "post_query_record_supports":
                {k: list(v) for k, v in self.post_query_record_supports},
            "post_query_new_support": list(self.post_query_new_support),
            "inference_priors": self.inference_priors.canon(),
            "selection_weights": self.selection_weights.canon(),
            "query_policy_state": self.query_policy_state.canon(),
            "serialized_hashes": dict(self.serialized_hashes),
        }


def _component_hashes(state: LatentRestartState
                      ) -> tuple[tuple[str, str], ...]:
    canon = state.canon()
    hashes = [(field, _digest(canon[field])) for field in HASHED_FIELDS]
    hashes.append(("metadata", _digest(
        {"schema": state.schema, "overlap": state.overlap,
         "step": state.step})))
    return tuple(sorted(hashes))


def seal_state(state: LatentRestartState) -> LatentRestartState:
    bare = replace(state, serialized_hashes=())
    return replace(bare, serialized_hashes=_component_hashes(bare))


def state_from_payload(obj: Any, verify_hashes: bool = True
                       ) -> LatentRestartState:
    if not isinstance(obj, Mapping):
        raise RestartIntegrityError("latent restart payload must be a mapping")
    _require_keys(obj, STATE_FIELDS, "latent restart payload")
    if not isinstance(obj["serialized_hashes"], Mapping):
        raise RestartIntegrityError("serialized_hashes must be a mapping")
    state = LatentRestartState(
        int(obj["schema"]), str(obj["overlap"]), int(obj["step"]),
        _posterior_from_obj(obj["identity_posterior"], "identity_posterior"),
        tuple(RecordConventionPosterior.from_obj(v)
              for v in obj["record_convention_posteriors"]),
        _frac(obj["new_mass"], "new_mass"),
        _frac(obj["out_mass"], "out_mass"),
        tuple(ConfirmedRecordSnapshot.from_obj(v)
              for v in obj["confirmed_records"]),
        tuple(ProvisionalBranchSnapshot.from_obj(v)
              for v in obj["provisional_branches"]),
        tuple(str(v) for v in obj["retrieval_shortlist"]),
        RetrievalAccountingSnapshot.from_obj(obj["retrieval_accounting"]),
        TaskEvidenceSnapshot.from_obj(obj["task_evidence"]),
        _support_map_from_obj(obj["post_query_record_supports"],
                              "post_query_record_supports"),
        tuple(int(v) for v in obj["post_query_new_support"]),
        InferencePriorsSnapshot.from_obj(obj["inference_priors"]),
        SelectionWeightsSnapshot.from_obj(obj["selection_weights"]),
        QueryPolicySnapshot.from_obj(obj["query_policy_state"]),
        tuple(sorted((str(k), str(v))
                     for k, v in obj["serialized_hashes"].items())))
    validate_state(state, verify_hashes=verify_hashes)
    return state


def _record_slot(key: str) -> int:
    if not key.startswith("record:"):
        raise RestartIntegrityError(f"non-MAIN record key {key!r}")
    try:
        value = int(key.split(":", 1)[1])
    except ValueError as exc:
        raise RestartIntegrityError(f"invalid MAIN record key {key!r}") from exc
    if f"record:{value}" != key:
        raise RestartIntegrityError(f"noncanonical MAIN record key {key!r}")
    return value


def _selection_snapshot(fam, task_snapshot: TaskEvidenceSnapshot, weights
                        ) -> SelectionWeightsSnapshot:
    from . import l1_retrieval as RET
    return SelectionWeightsSnapshot(
        tuple(int(v) for v in weights.live), int(weights.denominator),
        (int(weights.scaled.shape[0]), int(weights.scaled.shape[1])),
        _scaled_sha256(weights.scaled), _digest(task_snapshot.canon()),
        _digest(RET.family_signature(fam)))


def _initial_supports(fam, confirmed: Sequence[ConfirmedRecordSnapshot],
                      active_keys: Sequence[str]
                      ) -> tuple[tuple[int, tuple[int, ...]], ...]:
    by_key = {record.record_key: record for record in confirmed}
    if set(active_keys) - set(by_key):
        raise RestartIntegrityError("active support references unknown record")
    rows = []
    for key in active_keys:
        record = by_key[key]
        mask = np.ones(fam.n, dtype=bool)
        for obs in record.grounded:
            if not (0 <= obs.z < fam.m):
                raise RestartIntegrityError(
                    f"confirmed {record.record_key} has invalid meaning")
            mask &= fam.u3[:, obs.z] == obs.u
        support = tuple(int(v) for v in np.flatnonzero(mask))
        if not support:
            raise RestartIntegrityError(
                f"confirmed {record.record_key} reconstructs empty support")
        rows.append((_record_slot(record.record_key), support))
    return tuple(rows)


def _retrieval_selection(fam, task, confirmed, shortlist_size: int):
    from . import l1_retrieval as RET
    from . import semantic_mem as SM
    records = {
        _record_slot(record.record_key): SM.SemanticRecord(
            record.record_key, record.grounded)
        for record in confirmed
    }
    index = RET.build_global_exact_index(records)
    return RET.retrieve_protocol_a(
        index, fam, task, k=int(shortlist_size),
        strategy="exact_likelihood", seed=0)


def _mapped_identity(open_state) -> tuple[tuple[str, Fraction], ...]:
    mapped = {}
    for key, mass in open_state.identity_posterior().items():
        mapped[f"record:{key}" if isinstance(key, int) else str(key)] = \
            Fraction(mass)
    return tuple(sorted(mapped.items()))


def _mapped_conventions(open_state) -> tuple[RecordConventionPosterior, ...]:
    posteriors = open_state.convention_posteriors()
    return tuple(RecordConventionPosterior(
        f"record:{key}", tuple(int(v) for v in support),
        tuple(sorted((int(phi), Fraction(mass))
                     for phi, mass in posteriors[key].items())))
        for key, support in open_state.supports)


def _replay_from_confirmed(state: LatentRestartState):
    """Reconstruct and independently replay every persisted MAIN answer."""
    from x64h import episode as EP
    from x64h import family as F
    from . import l1_main as MAIN
    from . import l1_retrieval as RET
    from .latent_id import P_NEW
    from .provisional import PRIOR_OUT

    fam = F.Family(F.FamilySpec(overlap=state.overlap))
    task = state.task_evidence.task()
    if not task.live:
        raise RestartIntegrityError("restart task has no legal meanings")
    if tuple(sorted(set(task.live))) != tuple(task.live):
        raise RestartIntegrityError("restart task live set is not canonical")
    if tuple(sorted(task.tie)) != tuple(range(fam.m)):
        raise RestartIntegrityError("restart task tie order is not a permutation")
    if any(z < 0 or z >= len(EP.UNIVERSE) for z in task.demos):
        raise RestartIntegrityError("restart task demonstration is invalid")

    policy = state.query_policy_state
    if policy.policy != MAIN.INFORMATION_GAIN:
        raise RestartIntegrityError(
            "only exact information-gain MAIN policy is resumable")
    if policy.stop_when_identity_decisive:
        raise RestartIntegrityError(
            "identity-decisive early stopping is not represented after restart")
    if (not policy.legal_queries
            or tuple(sorted(set(policy.legal_queries)))
            != policy.legal_queries
            or any(z < 0 or z >= fam.m for z in policy.legal_queries)):
        raise RestartIntegrityError("query universe is invalid")
    if policy.query_budget < 1 or len(policy.history) > policy.query_budget:
        raise RestartIntegrityError("query history exceeds its exact budget")

    priors = state.inference_priors
    if not priors.with_new or not priors.with_out:
        raise RestartIntegrityError("L1 MAIN restart must retain NEW and OUT")
    n_records = len(state.confirmed_records)
    expected_record = (Fraction(1) - P_NEW - PRIOR_OUT) / n_records
    if (priors.p_new != P_NEW or priors.p_out != PRIOR_OUT
            or priors.record_prior != expected_record):
        raise RestartIntegrityError("checkpoint priors differ from frozen MAIN")

    weights = RET.exact_selection_weights(fam, task)
    if state.selection_weights != _selection_snapshot(
            fam, state.task_evidence, weights):
        raise RestartIntegrityError(
            "selection-aware weights do not reconstruct from task evidence")

    selected = _retrieval_selection(
        fam, task, state.confirmed_records, len(state.retrieval_shortlist))
    selected_keys = tuple(f"record:{int(key)}" for key in selected.selected_keys)
    if selected_keys != state.retrieval_shortlist:
        raise RestartIntegrityError(
            "retrieval shortlist does not reproduce Protocol-A ranking")
    if state.retrieval_accounting != RetrievalAccountingSnapshot.from_accounting(
            selected.accounting):
        raise RestartIntegrityError("retrieval accounting does not reproduce")
    accounting = state.retrieval_accounting
    if (accounting.protocol != "A_GLOBAL_EXACT_SCAN"
            or accounting.identity_specific_summaries_inspected
                != len(state.confirmed_records)
            or accounting.identity_likelihoods_evaluated
                != len(state.confirmed_records)
            or accounting.total_retrieval_node_equivalents
                != len(state.confirmed_records)
            or accounting.shortlist_size != len(state.retrieval_shortlist)
            or accounting.full_records_loaded != 0
            or not accounting.within_512
            or not accounting.incomplete_retrieval
            or accounting.four_node_claim):
        raise RestartIntegrityError(
            "checkpoint smuggles or undercharges Protocol-A retrieval")

    open_state = MAIN.OpenWorldState(
        fam, task, _initial_supports(
            fam, state.confirmed_records, state.retrieval_shortlist),
        tuple(range(fam.n)), priors.with_new, priors.with_out, (), weights)
    for index, (query, answer) in enumerate(policy.history):
        selected = open_state.choose_information_query(policy.legal_queries)
        if selected is None or int(selected) != int(query):
            raise RestartIntegrityError(
                f"stored clarification {index} was not chosen by MAIN policy")
        distribution = open_state.query_distribution(query)
        if distribution.get(int(answer), Fraction(0)) <= 0:
            raise RestartIntegrityError(
                f"stored clarification {index} has impossible answer")
        open_state = open_state.condition(query, answer)
    return open_state


def open_world_from_state(state: LatentRestartState):
    """Return the production state after validating all redundancies."""
    validate_state(state)
    return _replay_from_confirmed(state)


def validate_state(state: LatentRestartState, verify_hashes: bool = True) -> None:
    if state.schema != SCHEMA:
        raise RestartIntegrityError(f"unsupported schema {state.schema}")
    if state.overlap not in ("shared", "disjoint_op"):
        raise RestartIntegrityError(f"unknown overlap {state.overlap!r}")
    if state.step < 0:
        raise RestartIntegrityError("step must be nonnegative")

    confirmed_keys = [record.record_key for record in state.confirmed_records]
    if (len(confirmed_keys) != 8
            or len(confirmed_keys) != len(set(confirmed_keys))
            or any(_record_slot(key) < 0 for key in confirmed_keys)):
        raise RestartIntegrityError("confirmed MAIN record keys are invalid")
    for record in state.confirmed_records:
        if record.status != "CONFIRMED" or record.version < 1:
            raise RestartIntegrityError("confirmed record status/version invalid")
        if len(record.evidence_ids) != len(set(record.evidence_ids)):
            raise RestartIntegrityError("confirmed record duplicates evidence")
        if not {g.evidence_id for g in record.grounded}.issubset(
                set(record.evidence_ids)):
            raise RestartIntegrityError("confirmed record lost evidence identity")

    support_map = dict(state.post_query_record_supports)
    active_keys = tuple(state.retrieval_shortlist)
    if set(support_map) != set(active_keys):
        raise RestartIntegrityError("post-query support keys mismatch shortlist")
    if len(state.post_query_record_supports) != len(support_map):
        raise RestartIntegrityError("duplicate post-query record support")

    open_state = _replay_from_confirmed(state)
    fam = open_state.fam
    for key, support in state.post_query_record_supports:
        _validate_support(support, fam, f"post-query support {key}")
    _validate_support(state.post_query_new_support, fam, "post-query NEW support")
    replay_supports = {f"record:{key}": tuple(support)
                       for key, support in open_state.supports}
    if support_map != replay_supports:
        raise RestartIntegrityError(
            "post-query record supports do not replay from evidence/history")
    if state.post_query_new_support != open_state.new_support:
        raise RestartIntegrityError(
            "post-query NEW support does not replay from query history")

    expected_identity = _mapped_identity(open_state)
    required_identity = set(active_keys) | {NEW_IDENTITY, OUT_OF_FAMILY}
    _validate_posterior(state.identity_posterior, "identity_posterior",
                        required_identity)
    if state.identity_posterior != expected_identity:
        raise RestartIntegrityError(
            "identity posterior differs from reconstructed MAIN state")
    identity = _posterior_dict(state.identity_posterior)
    if identity[NEW_IDENTITY] != state.new_mass:
        raise RestartIntegrityError("new_mass disagrees with MAIN posterior")
    if identity[OUT_OF_FAMILY] != state.out_mass:
        raise RestartIntegrityError("out_mass disagrees with MAIN posterior")

    expected_conventions = _mapped_conventions(open_state)
    if state.record_convention_posteriors != expected_conventions:
        raise RestartIntegrityError(
            "convention posteriors differ from reconstructed MAIN state")
    for record in state.record_convention_posteriors:
        _validate_support(record.support, fam,
                          f"record support {record.record_key}")
        keys = tuple(phi for phi, _mass in record.posterior)
        if not set(keys).issubset(set(record.support)):
            raise RestartIntegrityError("convention posterior escaped support")
        if record.posterior:
            _validate_posterior(tuple((str(phi), mass)
                                      for phi, mass in record.posterior),
                                f"convention posterior {record.record_key}")

    policy = state.query_policy_state
    if not state.provisional_branches:
        raise RestartIntegrityError("checkpoint has no provisional branch")
    branch_ids = [branch.branch_id for branch in state.provisional_branches]
    if len(branch_ids) != len(set(branch_ids)):
        raise RestartIntegrityError("duplicate provisional branch ID")
    for branch in state.provisional_branches:
        _validate_posterior(branch.identity_posterior,
                            f"branch {branch.branch_id} posterior",
                            required_identity)
        if branch.identity_posterior != state.identity_posterior:
            raise RestartIntegrityError("provisional identity posterior stale")
        if branch.record_support != state.post_query_record_supports:
            raise RestartIntegrityError("provisional record support stale")
        if branch.new_support != state.post_query_new_support:
            raise RestartIntegrityError("provisional NEW support stale")
        if branch.new_mass != state.new_mass or branch.out_mass != state.out_mass:
            raise RestartIntegrityError("provisional NEW/OUT mass stale")
        if (branch.asked != policy.history
                or branch.query_universe != policy.legal_queries
                or branch.policy != policy.policy
                or branch.update_budget != policy.query_budget):
            raise RestartIntegrityError("provisional query policy/history stale")
        if len(branch.evidence_ids) != len(set(branch.evidence_ids)):
            raise RestartIntegrityError("provisional evidence IDs repeat")

    if (len(state.retrieval_shortlist) != 4
            or len(set(state.retrieval_shortlist))
            != len(state.retrieval_shortlist)
            or not set(state.retrieval_shortlist).issubset(set(confirmed_keys))):
        raise RestartIntegrityError("invalid retrieval shortlist")

    if verify_hashes:
        expected_hashes = _component_hashes(replace(
            state, serialized_hashes=()))
        if state.serialized_hashes != expected_hashes:
            raise RestartIntegrityError("serialized component hash mismatch")


def _state_from_open_world(previous: LatentRestartState, open_state,
                           evidence_id: str) -> LatentRestartState:
    identity = _mapped_identity(open_state)
    identity_map = _posterior_dict(identity)
    supports = tuple((f"record:{key}", tuple(int(v) for v in support))
                     for key, support in open_state.supports)
    history = tuple((int(z), int(answer))
                    for z, answer in open_state.history)
    query_state = replace(previous.query_policy_state, history=history)
    status = ("BUDGET_EXHAUSTED"
              if len(history) >= query_state.query_budget else "CHALLENGED")
    branches = tuple(replace(
        branch, identity_posterior=identity, record_support=supports,
        new_support=tuple(int(v) for v in open_state.new_support),
        new_mass=identity_map[NEW_IDENTITY],
        out_mass=identity_map[OUT_OF_FAMILY],
        evidence_ids=branch.evidence_ids + (str(evidence_id),),
        asked=history, status=status)
        for branch in previous.provisional_branches)
    state = LatentRestartState(
        SCHEMA, previous.overlap, previous.step + 1, identity,
        _mapped_conventions(open_state), identity_map[NEW_IDENTITY],
        identity_map[OUT_OF_FAMILY], previous.confirmed_records, branches,
        previous.retrieval_shortlist, previous.retrieval_accounting,
        previous.task_evidence, supports,
        tuple(int(v) for v in open_state.new_support),
        previous.inference_priors, previous.selection_weights, query_state, ())
    state = seal_state(state)
    validate_state(state)
    return state


def advance_state(state: LatentRestartState,
                  transition: Mapping[str, Any]) -> LatentRestartState:
    """Apply one real legal MAIN information-gain clarification."""
    validate_state(state)
    fields = {"type", "policy", "query", "answer", "evidence_id"}
    if not isinstance(transition, Mapping):
        raise RestartIntegrityError("MAIN clarification must be a mapping")
    _require_keys(transition, fields, "MAIN clarification")
    if transition["type"] != "main_clarification":
        raise RestartIntegrityError("synthetic restart transition is forbidden")
    policy = state.query_policy_state
    if str(transition["policy"]) != policy.policy:
        raise RestartIntegrityError("continuation policy changed at restart")
    if len(policy.history) >= policy.query_budget:
        raise RestartIntegrityError("no query budget remains after restart")

    open_state = open_world_from_state(state)
    selected = open_state.choose_information_query(policy.legal_queries)
    query = int(transition["query"])
    answer = int(transition["answer"])
    if selected is None or int(selected) != query:
        raise RestartIntegrityError(
            "suffix query is not the production MAIN information-gain choice")
    distribution = open_state.query_distribution(query)
    if distribution.get(answer, Fraction(0)) <= 0:
        raise RestartIntegrityError("suffix answer is impossible under MAIN")
    next_state = open_state.condition(query, answer)
    return _state_from_open_world(state, next_state,
                                  str(transition["evidence_id"]))


def continue_suffix(state: LatentRestartState,
                    suffix: Sequence[Mapping[str, Any]]) -> LatentRestartState:
    if not suffix:
        raise RestartIntegrityError("restart suffix contains no clarification")
    out = state
    for transition in suffix:
        out = advance_state(out, transition)
    return out


def state_from_main(overlap: str, seed: int, identities, main_state,
                    retrieval_shortlist: Sequence[int], *,
                    query_policy: str = "information_gain",
                    query_universe: Sequence[int] = tuple(range(8)),
                    query_budget: int | None = None,
                    stop_when_identity_decisive: bool = False
                    ) -> LatentRestartState:
    """Checkpoint an actual post-query MAIN state.

    ``query_budget`` is the total policy budget, not the number already used.
    A resumable checkpoint must have at least one query left; callers that do
    not supply it get the narrow one-step continuation contract.
    """
    from . import l1_main as MAIN
    from . import l1_retrieval as RET
    from .latent_id import P_NEW
    from .provisional import PRIOR_OUT

    if main_state.fam.spec.overlap != overlap:
        raise RestartIntegrityError("MAIN state alphabet mismatch")
    identities = tuple(identities)
    if len(identities) != 8:
        raise RestartIntegrityError(
            "L1 MAIN restart requires the frozen eight-record stream")
    if query_policy != MAIN.INFORMATION_GAIN:
        raise RestartIntegrityError(
            "random-policy RNG state is not represented; checkpoint incomplete")
    if not main_state.history:
        raise RestartIntegrityError("MAIN restart must be post-query")
    total_budget = (len(main_state.history) + 1
                    if query_budget is None else int(query_budget))
    if total_budget <= len(main_state.history):
        raise RestartIntegrityError("MAIN checkpoint has no continuation budget")

    confirmed_keys = tuple(
        f"record:{int(identity.slot)}" for identity in identities)
    if len(confirmed_keys) != len(set(confirmed_keys)):
        raise RestartIntegrityError("MAIN identities duplicate record slots")
    active_slots = tuple(int(key) for key, _support in main_state.supports)
    if (len(active_slots) != 4
            or not set(active_slots).issubset(
                {int(identity.slot) for identity in identities})):
        raise RestartIntegrityError(
            "MAIN state is not a nonempty at-most-four record subset")

    confirmed = []
    for identity, key in zip(identities, confirmed_keys):
        grounded = tuple(GroundedPair(
            int(obs.z), int(obs.u), str(obs.base_evidence))
            for obs in identity.grounded)
        evidence_ids = tuple(obs.evidence_id for obs in grounded)
        domain = tuple(range(main_state.fam.m))
        confirmed.append(ConfirmedRecordSnapshot(
            key, grounded, domain,
            _digest({"overlap": overlap, "seed": seed,
                     "record": key, "challenge": domain}),
            _digest({"overlap": overlap, "seed": seed,
                     "queries": tuple(int(v) for v in query_universe)}),
            "empirical", "CONFIRMED", 1, evidence_ids))

    task = TaskEvidenceSnapshot.from_task(main_state.task)
    recomputed_weights = RET.exact_selection_weights(
        main_state.fam, task.task())
    supplied_weights = main_state.weights
    if (tuple(supplied_weights.live) != tuple(recomputed_weights.live)
            or int(supplied_weights.denominator)
            != int(recomputed_weights.denominator)
            or not np.array_equal(supplied_weights.scaled,
                                  recomputed_weights.scaled)):
        raise RestartIntegrityError("MAIN supplied selection weights mismatch")
    weights_snapshot = _selection_snapshot(
        main_state.fam, task, recomputed_weights)

    retrieval = _retrieval_selection(
        main_state.fam, task.task(), tuple(confirmed), len(active_slots))
    expected_slots = tuple(int(v) for v in retrieval.selected_keys)
    supplied_shortlist = tuple(int(v) for v in retrieval_shortlist)
    if active_slots != expected_slots or supplied_shortlist != expected_slots:
        raise RestartIntegrityError(
            "checkpoint MAIN state is not the Protocol-A exact shortlist")
    accounting = RetrievalAccountingSnapshot.from_accounting(
        retrieval.accounting)

    supports = tuple((f"record:{key}", tuple(int(v) for v in support))
                     for key, support in main_state.supports)
    new_support = tuple(int(v) for v in main_state.new_support)
    identity = _mapped_identity(main_state)
    identity_map = _posterior_dict(identity)
    required = {f"record:{slot}" for slot in active_slots} | {
        NEW_IDENTITY, OUT_OF_FAMILY}
    _validate_posterior(identity, "MAIN identity posterior", required)
    priors = InferencePriorsSnapshot(
        P_NEW, PRIOR_OUT,
        (Fraction(1) - P_NEW - PRIOR_OUT) / len(identities),
        bool(main_state.with_new), bool(main_state.with_out))
    query_state = QueryPolicySnapshot(
        query_policy, tuple(sorted(set(int(v) for v in query_universe))),
        tuple((int(z), int(answer)) for z, answer in main_state.history),
        total_budget, bool(stop_when_identity_decisive))
    branch = ProvisionalBranchSnapshot(
        f"branch:{overlap}:{seed}", identity, supports, new_support,
        identity_map[NEW_IDENTITY], identity_map[OUT_OF_FAMILY],
        tuple([f"current-task:{overlap}:{seed}:{task.u}"] + [
            f"clarification:{overlap}:{seed}:{index}:{z}:{answer}"
            for index, (z, answer) in enumerate(query_state.history)]),
        query_state.history, query_state.legal_queries, query_state.policy,
        "OPEN", query_state.query_budget)
    shortlist = tuple(f"record:{int(key)}" for key in retrieval_shortlist)
    state = LatentRestartState(
        SCHEMA, overlap, 0, identity, _mapped_conventions(main_state),
        identity_map[NEW_IDENTITY], identity_map[OUT_OF_FAMILY],
        tuple(confirmed), (branch,), shortlist, accounting, task, supports,
        new_support,
        priors, weights_snapshot, query_state, ())
    state = seal_state(state)
    validate_state(state)
    return state


def fixture_state(overlap: str, seed: int = 400) -> LatentRestartState:
    """Build a real post-query MAIN checkpoint for restart unit tests."""
    from x64h import episode as EP
    from x64h import family as F
    from . import l1_main as MAIN
    from . import l1_retrieval as RET
    from . import l_suite as LS
    from . import semantic_mem as SM

    fam = F.Family(F.FamilySpec(overlap=overlap))
    identities = tuple(LS.build_identities(fam, seed))
    behaviour = EP.behaviour_table(fam.forms)
    probes = LS.build_probes(
        fam, behaviour, EP.Config(overlap=overlap), identities, seed)
    masks = [SM.surviving_mask(fam, identity.grounded)
             for identity in identities]
    chosen = None
    for probe in probes:
        if probe.slot < 0 or not probe.task.live:
            continue
        records = {identity.slot: SM.SemanticRecord(
            f"record:{identity.slot}", identity.grounded)
            for identity in identities}
        index = RET.build_global_exact_index(records)
        retrieval = RET.retrieve_protocol_a(
            index, fam, probe.task, k=4, strategy="exact_likelihood",
            seed=seed)
        initial = MAIN.subset_state(
            fam, probe.task, masks, retrieval.selected_keys)
        run = MAIN.run_policy(initial, MAIN.INFORMATION_GAIN, 1,
                              probe.phi_true, probe.task.z, tuple(range(8)),
                              seed)
        if run.state.history:
            chosen = probe, run.state, retrieval
            break
    if chosen is None:
        raise RestartIntegrityError("fixture has no legal MAIN clarification")
    _probe, main_state, retrieval = chosen
    return state_from_main(
        overlap, seed, identities, main_state, retrieval.selected_keys,
        query_budget=2)


def _clarification_suffix(state: LatentRestartState, answer: int,
                          query: int) -> tuple[dict[str, Any], ...]:
    return ({"type": "main_clarification",
             "policy": state.query_policy_state.policy,
             "query": int(query), "answer": int(answer),
             "evidence_id":
                 f"restart:{state.overlap}:{state.step}:{int(query)}:{int(answer)}"},)


def fixture_suffix(state: LatentRestartState) -> tuple[dict[str, Any], ...]:
    """One model-consistent legal clarification (not an evaluator oracle)."""
    open_state = open_world_from_state(state)
    query = open_state.choose_information_query(
        state.query_policy_state.legal_queries)
    if query is None:
        raise RestartIntegrityError("fixture checkpoint has no remaining query")
    distribution = open_state.query_distribution(query)
    if not distribution:
        raise RestartIntegrityError("fixture query has no legal answer")
    answer = min(answer for answer, mass in distribution.items() if mass > 0)
    return _clarification_suffix(state, answer, query)


def truthful_main_suffix(state: LatentRestartState, phi_true: int
                         ) -> tuple[dict[str, Any], ...]:
    """Create an environment answer from in-memory evaluator truth.

    The convention is not included in the checkpoint or serialized suffix.
    """
    open_state = open_world_from_state(state)
    query = open_state.choose_information_query(
        state.query_policy_state.legal_queries)
    if query is None:
        raise RestartIntegrityError("MAIN checkpoint has no remaining query")
    phi = int(phi_true)
    if phi < 0 or phi >= open_state.fam.n:
        raise RestartIntegrityError("truth convention is outside the family")
    answer = int(open_state.fam.u3[phi, int(query)])
    if open_state.query_distribution(query).get(answer, Fraction(0)) <= 0:
        raise RestartIntegrityError(
            "truth answer is impossible under checkpointed MAIN state")
    return _clarification_suffix(state, answer, query)


def run_parent(request_path: Path, state_path: Path) -> dict[str, Any]:
    _RUNTIME_CACHE["forbidden"] = os.environ.get(FORBIDDEN_ENV,
                                                  FORBIDDEN_VALUE)
    request = decode(request_path.read_bytes())
    if not isinstance(request, Mapping):
        raise RestartIntegrityError("parent request must be a mapping")
    _require_keys(request, {"initial_state", "suffix"}, "parent request")
    state = state_from_payload(request["initial_state"])
    suffix = tuple(request["suffix"])
    expected = continue_suffix(state, suffix)
    blob = encode(state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_bytes(blob)
    return {
        "pid": os.getpid(), "checkpoint_sha256": _sha(blob),
        "checkpoint_hashes": dict(state.serialized_hashes),
        "expected_final_sha256": _sha(encode(expected)),
        "expected_final_hashes": dict(expected.serialized_hashes),
        "expected_final_state_sha256": _digest(expected.canon()),
        "bytes": len(blob),
        "forbidden_in_checkpoint": FORBIDDEN_VALUE.encode() in blob,
    }


def run_child(state_path: Path, suffix_path: Path) -> dict[str, Any]:
    blob = state_path.read_bytes()
    state = state_from_payload(decode(blob))
    suffix_payload = decode(suffix_path.read_bytes())
    if not isinstance(suffix_payload, Mapping):
        raise RestartIntegrityError("suffix payload must be a mapping")
    _require_keys(suffix_payload, {"suffix"}, "suffix payload")
    suffix = tuple(suffix_payload["suffix"])
    final = continue_suffix(state, suffix)
    return {
        "pid": os.getpid(), "loaded_checkpoint_sha256": _sha(blob),
        "loaded_checkpoint_hashes": dict(state.serialized_hashes),
        "loaded_step": state.step, "final_step": final.step,
        "final_sha256": _sha(encode(final)),
        "final_hashes": dict(final.serialized_hashes),
        "final_state_sha256": _digest(final.canon()),
        "continuation_policy": state.query_policy_state.policy,
        "continuation_queries": [int(row["query"]) for row in suffix],
        "continuation_answers": [int(row["answer"]) for row in suffix],
        "real_main_continuation": all(
            row.get("type") == "main_clarification" for row in suffix),
        "env_size": len(os.environ),
        "forbidden_in_checkpoint": FORBIDDEN_VALUE.encode() in blob,
        "forbidden_in_env": any(FORBIDDEN_VALUE in value
                                for value in os.environ.values()),
        "forbidden_in_globals": FORBIDDEN_VALUE in json.dumps(_RUNTIME_CACHE),
    }


def _mutate_payload(payload: dict[str, Any], field: str, mode: str) -> None:
    if field not in AUDITED_FIELDS:
        raise ValueError(field)
    if mode == "drop":
        payload.pop(field)
        return
    if mode != "mutate":
        raise ValueError(mode)
    if field == "identity_posterior":
        payload[field][sorted(payload[field])[0]] += Fraction(1, 997)
    elif field == "record_convention_posteriors":
        record = next(
            (row for row in payload[field] if row["posterior"]), None)
        if record is not None:
            key = sorted(record["posterior"], key=int)[0]
            record["posterior"][key] += Fraction(1, 997)
        else:
            # Empty post-query supports are legal.  Reversing an empty or
            # singleton support is a no-op, so corrupt the association instead
            # and guarantee that every scheduled plant changes the payload.
            payload[field][0]["record_key"] += ":CALIBRATION_CORRUPTION"
    elif field in ("new_mass", "out_mass"):
        payload[field] += Fraction(1, 997)
    elif field == "confirmed_records":
        payload[field][0]["version"] += 1
    elif field == "provisional_branches":
        payload[field][0]["status"] = "CALIBRATION_CORRUPTION"
    elif field == "retrieval_shortlist":
        payload[field] = list(reversed(payload[field]))
        if len(payload[field]) == 1:
            payload[field] = []
    elif field == "retrieval_accounting":
        payload[field]["total_retrieval_node_equivalents"] = 4
        payload[field]["four_node_claim"] = True
    elif field == "task_evidence":
        payload[field]["u"] += 1
    elif field == "post_query_record_supports":
        key = sorted(payload[field])[0]
        payload[field][key] = list(reversed(payload[field][key]))
        if len(payload[field][key]) < 2:
            payload[field][key] = [999999]
    elif field == "post_query_new_support":
        payload[field] = payload[field][1:]
    elif field == "inference_priors":
        payload[field]["p_new"] += Fraction(1, 997)
    elif field == "selection_weights":
        payload[field]["scaled_sha256"] = "0" * 64
    elif field == "query_policy_state":
        payload[field]["policy"] = "random"
    elif field == "serialized_hashes":
        key = sorted(payload[field])[0]
        payload[field][key] = "0" * 64


def _json_result(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        return json.loads(proc.stdout)
    except Exception:
        return {"ok": False, "error": (proc.stderr or proc.stdout)[-800:]}


def _child_env(experiments_path: str) -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "PYTHONPATH": experiments_path,
            "PYTHONDONTWRITEBYTECODE": "1"}


def _run_calibrations(checkpoint_blob: bytes, suffix_path: Path,
                      directory: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    experiments_path = str(Path("experiments").resolve())
    for field in AUDITED_FIELDS:
        for mode in ("drop", "mutate"):
            payload = decode(checkpoint_blob)
            _mutate_payload(payload, field, mode)
            path = directory / f"{mode}-{field}.json"
            path.write_bytes(encode(payload))
            process = subprocess.run(
                [sys.executable, "-m", "x65a.restart_l1", "child",
                 str(path), str(suffix_path)], capture_output=True, text=True,
                env=_child_env(experiments_path))
            detail = _json_result(process)
            out[f"{mode}:{field}"] = {
                "rejected": process.returncode != 0,
                "returncode": process.returncode,
                "error": str(detail.get("error", ""))[:300],
            }
    return out


def cycle(path: Path, overlap: str = "shared", seed: int = 400,
          state: LatentRestartState | None = None,
          suffix: Sequence[Mapping[str, Any]] | None = None,
          run_calibrations: bool = True) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = fixture_state(overlap, seed) if state is None else state
    validate_state(state)
    if state.overlap != overlap:
        raise RestartIntegrityError("cycle overlap disagrees with checkpoint")
    suffix = fixture_suffix(state) if suffix is None else tuple(suffix)

    experiments_path = str(Path("experiments").resolve())
    with tempfile.TemporaryDirectory(prefix="restart-l1-",
                                     dir=path.parent) as tmp:
        work = Path(tmp)
        request_path = work / "request.json"
        suffix_path = work / "suffix.json"
        request_path.write_bytes(encode(
            {"initial_state": state.canon(), "suffix": tuple(suffix)}))
        suffix_path.write_bytes(encode({"suffix": tuple(suffix)}))

        parent_env = dict(os.environ)
        parent_env[FORBIDDEN_ENV] = FORBIDDEN_VALUE
        parent_env["PYTHONPATH"] = experiments_path
        parent_process = subprocess.run(
            [sys.executable, "-m", "x65a.restart_l1", "parent",
             str(request_path), str(path)], capture_output=True, text=True,
            env=parent_env)
        if parent_process.returncode != 0:
            return {"ok": False, "stage": "parent",
                    "error": _json_result(parent_process).get("error", "")}
        parent = _json_result(parent_process)

        alive = True
        try:
            os.kill(int(parent["pid"]), 0)
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True

        child_process = subprocess.run(
            [sys.executable, "-m", "x65a.restart_l1", "child",
             str(path), str(suffix_path)], capture_output=True, text=True,
            env=_child_env(experiments_path))
        if child_process.returncode != 0:
            return {"ok": False, "stage": "child", "parent": parent,
                    "error": _json_result(child_process).get("error", "")}
        child = _json_result(child_process)
        checkpoint_blob = path.read_bytes()
        calibrations = (_run_calibrations(checkpoint_blob, suffix_path, work)
                        if run_calibrations else {})

    same_checkpoint = (
        parent["checkpoint_sha256"] == child["loaded_checkpoint_sha256"]
        == _sha(path.read_bytes()))
    same_final = (
        parent["expected_final_sha256"] == child["final_sha256"]
        and parent["expected_final_hashes"] == child["final_hashes"]
        and parent["expected_final_state_sha256"]
            == child["final_state_sha256"])
    calibration_ok = all(row["rejected"] for row in calibrations.values())
    forbidden_closed = not any((
        parent["forbidden_in_checkpoint"], child["forbidden_in_checkpoint"],
        child["forbidden_in_env"], child["forbidden_in_globals"]))
    ok = (not alive and int(parent["pid"]) != int(child["pid"])
          and same_checkpoint and same_final and forbidden_closed
          and calibration_ok and child["real_main_continuation"])
    return {
        "ok": ok, "overlap": overlap,
        "parent_pid": parent["pid"], "child_pid": child["pid"],
        "parent_pid_gone": not alive,
        "checkpoint_sha256": parent["checkpoint_sha256"],
        "checkpoint_bytes": parent["bytes"],
        "checkpoint_hashes": parent["checkpoint_hashes"],
        "child_loaded_parent_state": same_checkpoint,
        "uninterrupted_final_sha256": parent["expected_final_sha256"],
        "restarted_final_sha256": child["final_sha256"],
        "final_hashes_identical": same_final,
        "final_state_identical":
            parent["expected_final_state_sha256"]
            == child["final_state_sha256"],
        "loaded_step": child["loaded_step"],
        "final_step": child["final_step"],
        "real_main_continuation": child["real_main_continuation"],
        "continuation_policy": child["continuation_policy"],
        "continuation_queries": child["continuation_queries"],
        "continuation_answers": child["continuation_answers"],
        "forbidden_channel_closed": forbidden_closed,
        "child_env_size": child["env_size"],
        "calibrations": calibrations,
        "all_calibrations_rejected": calibration_ok,
    }


def _main(argv: Sequence[str]) -> int:
    try:
        mode = argv[1]
        if mode == "parent":
            result = run_parent(Path(argv[2]), Path(argv[3]))
        elif mode == "child":
            result = run_child(Path(argv[2]), Path(argv[3]))
        else:
            raise ValueError(f"unknown mode {mode!r}")
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error":
                          f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
