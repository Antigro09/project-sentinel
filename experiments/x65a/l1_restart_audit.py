"""Stream-complete genuine restart replication for X65A-L1 MAIN states.

``restart_l1`` supplies the exact checkpoint schema and one genuine
parent-death/scrubbed-child cycle.  This module lifts that boundary to the
development/validation stream matrix:

* callers supply the actual post-query top-four ``OpenWorldState`` produced by
  a charged Protocol-A scan of all eight records for every required
  ``(alphabet, split, seed)``;
* the matrix contract rejects omitted, duplicate, pre-query, or non-eight-
  record cases;
* every case runs a real parent process, verifies that it died, and continues
  in a child whose environment contains only PATH, PYTHONPATH, and the
  bytecode switch;
* the report carries active identity/convention posteriors, NEW/OUT mass, all
  eight confirmed records, exact retrieval accounting, provisional supports,
  reconstructing task/policy inputs, shortlist, and component hashes;
* exact corrupt-checkpoint plants are distributed across the matrix and sent
  to the same child loader/``validate_state`` path as the real checkpoint.

The batch validator is pure with respect to the filesystem and subprocesses:
it consumes case reports plus a contract.  ``planted_accepted_corruption``
therefore provides a known-bad report that the same validator must reject.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from x64h import episode as EP
from x64h import family as F

from . import l1_main as MAIN
from . import l1_retrieval as RET
from . import l_suite as LS
from . import restart_l1 as R
from . import semantic_mem as SM
from .types import decode, encode


SCHEMA = "x65a-l1-main-restart-matrix-v2"
SPLITS = ("development", "validation")
OVERLAPS = ("shared", "disjoint_op")


@dataclass(frozen=True)
class RestartMatrixContract:
    development_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    overlaps: tuple[str, ...] = OVERLAPS
    identities_per_stream: int = 8
    required_corruption_fields: tuple[str, ...] = R.AUDITED_FIELDS
    corruption_mode: str = "mutate"

    def __post_init__(self):
        if not self.development_seeds or not self.validation_seeds:
            raise ValueError("restart matrix needs development and validation")
        if set(self.development_seeds) & set(self.validation_seeds):
            raise ValueError("development and validation seeds must be disjoint")
        if set(self.overlaps) != set(OVERLAPS):
            raise ValueError("restart matrix must cover both alphabet strata")
        if self.identities_per_stream != 8:
            raise ValueError("L1 restart contract requires exactly eight records")
        if not set(self.required_corruption_fields).issubset(R.AUDITED_FIELDS):
            raise ValueError("unknown corrupt-checkpoint field")
        if self.corruption_mode not in ("drop", "mutate"):
            raise ValueError("unknown corruption mode")

    def seeds_for(self, split: str) -> tuple[int, ...]:
        if split == "development":
            return self.development_seeds
        if split == "validation":
            return self.validation_seeds
        raise ValueError(split)

    def expected_keys(self) -> tuple[tuple[str, str, int], ...]:
        return tuple((overlap, split, int(seed))
                     for overlap in self.overlaps for split in SPLITS
                     for seed in self.seeds_for(split))

    def required_calibration_keys(self) -> tuple[str, ...]:
        return tuple(f"{self.corruption_mode}:{field}"
                     for field in self.required_corruption_fields)

    def canon(self):
        return {"development_seeds": list(self.development_seeds),
                "validation_seeds": list(self.validation_seeds),
                "overlaps": list(self.overlaps),
                "identities_per_stream": self.identities_per_stream,
                "required_corruption_fields":
                    list(self.required_corruption_fields),
                "corruption_mode": self.corruption_mode,
                "expected_streams": len(self.expected_keys())}


@dataclass(frozen=True)
class MainRestartCase:
    overlap: str
    split: str
    seed: int
    identities: tuple = field(repr=False, compare=False)
    main_state: object = field(repr=False, compare=False)
    retrieval_shortlist: tuple[int, ...]
    phi_true: int = field(repr=False, compare=False)
    total_query_budget: int = 2

    @property
    def key(self) -> tuple[str, str, int]:
        return self.overlap, self.split, int(self.seed)


def validate_main_case(case: MainRestartCase,
                       identities_per_stream: int = 8) -> None:
    if case.overlap not in OVERLAPS or case.split not in SPLITS:
        raise R.RestartIntegrityError("invalid MAIN restart stream key")
    if len(case.identities) != identities_per_stream:
        raise R.RestartIntegrityError(
            "MAIN restart input is not an eight-record stream")
    slots = tuple(int(identity.slot) for identity in case.identities)
    if len(set(slots)) != identities_per_stream:
        raise R.RestartIntegrityError("MAIN restart identities duplicate slots")
    if case.main_state.fam.spec.overlap != case.overlap:
        raise R.RestartIntegrityError("MAIN restart alphabet mismatch")
    active_keys = tuple(int(key) for key, _support in case.main_state.supports)
    if active_keys != case.retrieval_shortlist:
        raise R.RestartIntegrityError(
            "MAIN restart state is not the charged retrieval shortlist")
    if not case.main_state.history:
        raise R.RestartIntegrityError(
            "MAIN restart input must be an actual post-query state")
    if not case.retrieval_shortlist or len(case.retrieval_shortlist) > 4:
        raise R.RestartIntegrityError("MAIN restart shortlist must contain 1..4")
    if not set(case.retrieval_shortlist).issubset(set(slots)):
        raise R.RestartIntegrityError("MAIN restart shortlist references no record")
    if case.total_query_budget <= len(case.main_state.history):
        raise R.RestartIntegrityError("MAIN restart case has no residual budget")
    if not 0 <= int(case.phi_true) < int(case.main_state.fam.n):
        raise R.RestartIntegrityError("MAIN restart truth convention invalid")


def make_actual_main_case(overlap: str, split: str, seed: int,
                          query_budget: int = 1) -> MainRestartCase:
    """Build one real eight-record MAIN state for tests and audit callers."""

    if query_budget < 1:
        raise ValueError("post-query restart case needs a positive query budget")
    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    identities = tuple(LS.build_identities(fam, seed))
    probes = LS.build_probes(
        fam, beh, EP.Config(overlap=overlap), identities, seed, n_per=1)
    masks = [SM.surviving_mask(fam, identity.grounded)
             for identity in identities]
    chosen = None
    for probe in probes:
        if (probe.slot < 0 or not probe.task.live
                or probe.kind not in ("returning", "ambiguous", "misleading")):
            continue
        records = {identity.slot: SM.SemanticRecord(
            f"record:{identity.slot}", identity.grounded)
            for identity in identities}
        exact = RET.build_global_exact_index(records)
        retrieval = RET.retrieve_protocol_a(
            exact, fam, probe.task, k=4, strategy="exact_likelihood",
            seed=seed)
        initial = MAIN.subset_state(
            fam, probe.task, masks, retrieval.selected_keys)
        run = MAIN.run_policy(
            initial, MAIN.INFORMATION_GAIN, query_budget, probe.phi_true,
            probe.task.z, tuple(range(8)), seed)
        if run.state.history:
            chosen = probe, run, retrieval
            break
    if chosen is None:
        raise R.RestartIntegrityError(
            "stream has no reachable post-query MAIN state")
    probe, run, retrieval = chosen
    case = MainRestartCase(
        overlap, split, int(seed), identities, run.state,
        tuple(int(v) for v in retrieval.selected_keys), int(probe.phi_true),
        max(int(query_budget) + 1, len(run.state.history) + 1))
    validate_main_case(case)
    return case


@dataclass(frozen=True)
class MainRestartCaseReport:
    overlap: str
    split: str
    seed: int
    post_query_history: tuple[tuple[int, int], ...]
    checkpoint_state: R.LatentRestartState
    cycle_result: dict[str, Any]
    calibration_plants: tuple[tuple[str, str], ...]
    corrupt_checkpoint_calibrations: dict[str, dict[str, Any]]

    @property
    def key(self) -> tuple[str, str, int]:
        return self.overlap, self.split, self.seed

    def canon(self):
        state = self.checkpoint_state.canon()
        return {
            "overlap": self.overlap, "split": self.split, "seed": self.seed,
            "post_query_history": [list(v) for v in self.post_query_history],
            "state_schema": state["schema"], "state_step": state["step"],
            "identity_posterior": state["identity_posterior"],
            "record_convention_posteriors":
                state["record_convention_posteriors"],
            "new_mass": state["new_mass"], "out_mass": state["out_mass"],
            "confirmed_records": state["confirmed_records"],
            "provisional_branches": state["provisional_branches"],
            "retrieval_shortlist": state["retrieval_shortlist"],
            "retrieval_accounting": state["retrieval_accounting"],
            "task_evidence": state["task_evidence"],
            "post_query_record_supports":
                state["post_query_record_supports"],
            "post_query_new_support": state["post_query_new_support"],
            "inference_priors": state["inference_priors"],
            "selection_weights": state["selection_weights"],
            "query_policy_state": state["query_policy_state"],
            "serialized_hashes": state["serialized_hashes"],
            "cycle": self.cycle_result,
            "calibration_plants": [list(v) for v in self.calibration_plants],
            "corrupt_checkpoint_calibrations":
                self.corrupt_checkpoint_calibrations,
        }


@dataclass(frozen=True)
class RestartMatrixValidation:
    checks: dict[str, bool]
    case_checks: dict[str, dict[str, bool]]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (all(self.checks.values())
                and all(all(row.values()) for row in self.case_checks.values())
                and not self.errors)

    def canon(self):
        return {"checks": self.checks, "case_checks": self.case_checks,
                "errors": list(self.errors), "passed": self.passed}


@dataclass(frozen=True)
class MainRestartMatrixAudit:
    schema: str
    contract: RestartMatrixContract
    cases: tuple[MainRestartCaseReport, ...]
    validation: RestartMatrixValidation

    def canon(self):
        return {"schema": self.schema, "contract": self.contract.canon(),
                "cases": [case.canon() for case in self.cases],
                "validation": self.validation.canon()}


def _json_result(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        return json.loads(proc.stdout)
    except Exception:
        return {"ok": False, "error": (proc.stderr or proc.stdout)[-800:]}


def _run_corruption_plants(checkpoint_path: Path,
                           suffix: Sequence[Mapping[str, Any]],
                           plants: Sequence[tuple[str, str]], directory: Path
                           ) -> dict[str, dict[str, Any]]:
    """Send corrupted real checkpoint bytes through restart_l1's child path."""

    if not plants:
        return {}
    suffix_path = directory / "suffix.json"
    suffix_path.write_bytes(encode({"suffix": tuple(suffix)}))
    checkpoint_blob = checkpoint_path.read_bytes()
    experiments_path = str(Path("experiments").resolve())
    out = {}
    for field, mode in plants:
        payload = decode(checkpoint_blob)
        R._mutate_payload(payload, field, mode)
        corrupt_path = directory / f"{mode}-{field}.json"
        corrupt_path.write_bytes(encode(payload))
        proc = subprocess.run(
            [sys.executable, "-m", "x65a.restart_l1", "child",
             str(corrupt_path), str(suffix_path)], capture_output=True,
            text=True, env=R._child_env(experiments_path))
        detail = _json_result(proc)
        out[f"{mode}:{field}"] = {
            "rejected": proc.returncode != 0,
            "returncode": proc.returncode,
            "error": str(detail.get("error", ""))[:300],
            "same_child_validator": "x65a.restart_l1 child/state_from_payload",
        }
    return out


def _calibration_schedule(cases: Sequence[MainRestartCase],
                          contract: RestartMatrixContract
                          ) -> dict[tuple[str, str, int], tuple[tuple[str, str], ...]]:
    ordered = tuple(sorted((case.key for case in cases)))
    schedule: dict[tuple[str, str, int], list[tuple[str, str]]] = {
        key: [] for key in ordered}
    for index, field in enumerate(contract.required_corruption_fields):
        key = ordered[index % len(ordered)]
        schedule[key].append((field, contract.corruption_mode))
    return {key: tuple(value) for key, value in schedule.items()}


def _stream_suffix(case: MainRestartCase, state: R.LatentRestartState
                   ) -> tuple[dict[str, Any], ...]:
    """One production-selected MAIN query with its truthful legal answer."""

    return R.truthful_main_suffix(state, case.phi_true)


def _case_label(key: tuple[str, str, int]) -> str:
    return f"{key[0]}/{key[1]}/{key[2]}"


def validate_restart_matrix(
        reports: Sequence[MainRestartCaseReport],
        contract: RestartMatrixContract) -> RestartMatrixValidation:
    """Pure validator for both real reports and planted bad reports."""

    expected = set(contract.expected_keys())
    keys = [report.key for report in reports]
    actual = set(keys)
    errors: list[str] = []
    case_checks: dict[str, dict[str, bool]] = {}
    seen_calibrations: set[str] = set()
    for report in reports:
        label = _case_label(report.key)
        state = report.checkpoint_state
        cycle = report.cycle_result
        state_valid = True
        try:
            R.validate_state(state)
        except Exception as exc:
            state_valid = False
            errors.append(f"{label}: state invalid: {exc}")
        identity = dict(state.identity_posterior)
        branch_history_matches = (
            bool(state.provisional_branches)
            and state.provisional_branches[0].asked == report.post_query_history)
        calibration_keys = {
            f"{mode}:{field}" for field, mode in report.calibration_plants}
        seen_calibrations.update(report.corrupt_checkpoint_calibrations)
        expected_hashes = dict(state.serialized_hashes)
        row = {
            "state_valid": state_valid,
            "actual_post_query_history_present": bool(report.post_query_history),
            "identity_posterior_exactly_normalized":
                sum(identity.values(), Fraction(0)) == 1,
            "identity_posterior_has_shortlist_new_out":
                (len(identity) == len(state.retrieval_shortlist) + 2
                 and R.NEW_IDENTITY in identity
                 and R.OUT_OF_FAMILY in identity),
            "new_mass_matches": identity.get(R.NEW_IDENTITY) == state.new_mass,
            "out_mass_matches": identity.get(R.OUT_OF_FAMILY) == state.out_mass,
            "shortlisted_convention_posteriors":
                len(state.record_convention_posteriors)
                == len(state.retrieval_shortlist),
            "eight_confirmed_records":
                len(state.confirmed_records) == contract.identities_per_stream,
            "provisional_branch_survives": bool(state.provisional_branches),
            "provisional_history_matches_main": branch_history_matches,
            "shortlist_is_nonempty_at_most_four":
                0 < len(state.retrieval_shortlist) <= 4,
            "protocol_a_scan_charged_as_eight_nodes":
                (state.retrieval_accounting.protocol == "A_GLOBAL_EXACT_SCAN"
                 and state.retrieval_accounting.
                     identity_specific_summaries_inspected
                     == contract.identities_per_stream
                 and state.retrieval_accounting.identity_likelihoods_evaluated
                     == contract.identities_per_stream
                 and state.retrieval_accounting.
                     total_retrieval_node_equivalents
                     == contract.identities_per_stream
                 and state.retrieval_accounting.incomplete_retrieval
                 and not state.retrieval_accounting.four_node_claim),
            "serialized_hashes_complete":
                set(expected_hashes) == set(R.HASHED_FIELDS) | {"metadata"},
            "cycle_reports_success": bool(cycle.get("ok")),
            "parent_really_died": bool(cycle.get("parent_pid_gone")),
            "child_pid_differs": cycle.get("parent_pid") != cycle.get("child_pid"),
            "scrubbed_child_loaded_parent_bytes":
                bool(cycle.get("child_loaded_parent_state")),
            "checkpoint_sha_matches_actual_state":
                cycle.get("checkpoint_sha256") == R._sha(encode(state)),
            "checkpoint_component_hashes_match":
                cycle.get("checkpoint_hashes") == expected_hashes,
            "restart_matches_uninterrupted_hash":
                (cycle.get("uninterrupted_final_sha256")
                 == cycle.get("restarted_final_sha256")
                 and bool(cycle.get("final_hashes_identical"))),
            "restart_matches_uninterrupted_exact_state":
                bool(cycle.get("final_state_identical")),
            "child_continued_real_main_policy":
                (bool(cycle.get("real_main_continuation"))
                 and cycle.get("continuation_policy") == MAIN.INFORMATION_GAIN
                 and len(cycle.get("continuation_queries", ())) == 1
                 and len(cycle.get("continuation_answers", ())) == 1
                 and cycle.get("final_step") == state.step + 1),
            "forbidden_channel_closed":
                bool(cycle.get("forbidden_channel_closed")),
            "child_environment_scrubbed":
                int(cycle.get("child_env_size", 999)) <= 5,
            "scheduled_calibration_rows_present":
                set(report.corrupt_checkpoint_calibrations) == calibration_keys,
            "all_scheduled_corrupt_checkpoints_rejected":
                all(value.get("rejected") and value.get("returncode") != 0
                    for value in
                    report.corrupt_checkpoint_calibrations.values()),
        }
        case_checks[label] = row
        if not all(row.values()):
            errors.extend(f"{label}: {name}" for name, ok in row.items()
                          if not ok)
    checks = {
        "schema_stream_matrix_complete": actual == expected,
        "no_duplicate_stream_reports": len(keys) == len(set(keys)),
        "all_required_corruption_fields_calibrated":
            set(contract.required_calibration_keys()) == seen_calibrations,
        "all_case_rows_valid": all(all(row.values())
                                   for row in case_checks.values()),
    }
    if actual != expected:
        errors.append(f"stream matrix mismatch missing={sorted(expected-actual)} "
                      f"extra={sorted(actual-expected)}")
    return RestartMatrixValidation(checks, case_checks, tuple(errors))


def audit_main_restart_matrix(
        cases: Sequence[MainRestartCase], output_dir: Path,
        contract: RestartMatrixContract) -> MainRestartMatrixAudit:
    """Run genuine cycles for a complete caller-supplied MAIN stream matrix."""

    cases = tuple(cases)
    if not cases:
        raise R.RestartIntegrityError("empty MAIN restart matrix")
    for case in cases:
        validate_main_case(case, contract.identities_per_stream)
    if {case.key for case in cases} != set(contract.expected_keys()) \
            or len({case.key for case in cases}) != len(cases):
        raise R.RestartIntegrityError(
            "caller did not supply every development/validation stream exactly once")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule = _calibration_schedule(cases, contract)
    reports = []
    for case in sorted(cases, key=lambda value: value.key):
        state = R.state_from_main(
            case.overlap, case.seed, case.identities, case.main_state,
            case.retrieval_shortlist,
            query_budget=case.total_query_budget)
        suffix = _stream_suffix(case, state)
        checkpoint = output_dir / (
            f"{case.overlap}-{case.split}-{case.seed}.json")
        cycle = R.cycle(checkpoint, overlap=case.overlap, seed=case.seed,
                        state=state, suffix=suffix, run_calibrations=False)
        plants = schedule[case.key]
        if checkpoint.is_file():
            with tempfile.TemporaryDirectory(
                    prefix="l1-restart-calibration-", dir=output_dir) as tmp:
                calibrations = _run_corruption_plants(
                    checkpoint, suffix, plants, Path(tmp))
        else:
            calibrations = {
                f"{mode}:{field}": {
                    "rejected": False, "returncode": 0,
                    "error": "normal parent produced no checkpoint",
                    "same_child_validator":
                        "x65a.restart_l1 child/state_from_payload",
                }
                for field, mode in plants}
        reports.append(MainRestartCaseReport(
            case.overlap, case.split, case.seed,
            tuple((int(z), int(answer)) for z, answer in case.main_state.history),
            state, cycle, plants, calibrations))
    reports_tuple = tuple(reports)
    validation = validate_restart_matrix(reports_tuple, contract)
    return MainRestartMatrixAudit(SCHEMA, contract, reports_tuple, validation)


def planted_accepted_corruption(
        reports: Sequence[MainRestartCaseReport]
        ) -> tuple[MainRestartCaseReport, ...]:
    """Known-bad report: one corrupt child is falsely marked accepted."""

    out = list(reports)
    for index, report in enumerate(out):
        if not report.corrupt_checkpoint_calibrations:
            continue
        calibrations = decode(encode(report.corrupt_checkpoint_calibrations))
        key = sorted(calibrations)[0]
        calibrations[key]["rejected"] = False
        calibrations[key]["returncode"] = 0
        out[index] = replace(
            report, corrupt_checkpoint_calibrations=calibrations)
        return tuple(out)
    raise R.RestartIntegrityError("no corruption calibration available to plant")
