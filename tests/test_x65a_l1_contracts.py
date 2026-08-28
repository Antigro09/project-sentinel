"""Executable L1 artifact contracts and their planted red calibrations."""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, "experiments")

from x64h import episode as EP
from x64h import family as F
from x65a import l1_contracts as C
from x65a import l1_eval as E
from x65a import l1_inference as I
from x65a import l1_stats as ST
from x65a import l_suite as LS


def _provenance():
    dev, val = (6400, 6401), (7400, 7401)
    conditions = {
        ov: {
            "development": {s: {"returning": 8} for s in dev},
            "validation": {s: {"returning": 8} for s in val},
        }
        for ov in ("shared", "disjoint_op")
    }
    root = Path.cwd().resolve()
    names = ("runner", "authoritative_json", "readme", "inference",
             "evaluation", "contracts", "retrieval", "safety",
             "negative_transfer", "statistics", "restart")
    return {
        "x65a_l_commit_full_hash":
            "5205543b110ba6da2e3f6da30630809941f821c4",
        "current_HEAD": "a" * 40,
        "branch": "phase-1-verifier",
        "tracked_tree_clean": False,
        "tracked_status": (" M README.md",),
        "untracked_tree_clean": False,
        "untracked_count": 1,
        "untracked_status": ("new.py",),
        "development_stream_seeds": dev,
        "validation_stream_seeds": val,
        "streams_per_alphabet_stratum": {
            "development": len(dev), "validation": len(val)},
        "identities_per_stream": 8,
        "task_conditions_per_stream": conditions,
        "query_budgets": (0, 1, 2, 3, 4),
        "validation_frozen_margins": {
            "L3_retrieval_noninferiority": Fraction(1, 20),
            "L10_negative_transfer": Fraction(1, 20)},
        "experiment_runtime_ms": 123,
        "full_suite_runtime_ms": 456,
        "full_suite_evidence": {
            "command": ("uv", "run", "pytest", "-q"),
            "exit_code": 0,
            "passed": 640,
            "skipped": 1,
            "runtime_ms": 456,
            "result_line": "640 passed, 1 skipped in 456 ms",
        },
        "artifact_paths": {name: str(root / f"{name}.json")
                           for name in names},
        "final_manifest_written": False,
        "final_stream_seed_sampled": False,
    }


def _memoryless_run():
    fam = F.Family(F.FamilySpec(overlap="disjoint_op"))
    beh = EP.behaviour_table(fam.forms)
    ids = LS.build_identities(fam, 400)
    probe = next(p for p in LS.build_probes(
        fam, beh, EP.Config(overlap="disjoint_op"), ids, 400,
        n_per=1) if p.kind == "returning")
    curves = I.memoryless_policy_curves(
        fam, beh, probe.task, probe.phi_true, probe.task.z,
        range(8), range(8), budgets=(0, 1), seed=400)
    return curves[I.TASK_INFORMATION_GAIN][1]


def _active_bundle():
    zeros = [Fraction(0)] * 4
    intervals = {
        key: ST.paired_interval(zeros, zeros, reps=100, seed=i)
        for i, key in enumerate(sorted(C.REQUIRED_ACTIVE_METRICS))
    }
    return {
        "streams": {s: {} for s in (6400, 6401, 6402, 6403)},
        "intervals": intervals,
        "component_status": "not_measured_in_X65A-L",
        "all_operational_intervals_include_zero": True,
    }


def test_provenance_contract_and_its_planted_defects():
    value = _provenance()
    assert C.validate_provenance(value).ok
    calibration = C.calibrate_provenance(value)
    assert calibration["fires"]
    assert all(calibration["rejected"].values())


def test_q246_contract_uses_the_exact_total_and_denominator():
    value = E.legacy_query_accounting()
    assert C.validate_q246(value).ok
    calibration = C.calibrate_q246(value)
    assert calibration["fires"]
    assert all(calibration["rejected"].values())


def test_memoryless_contract_rejects_counted_or_untraced_answers():
    run = _memoryless_run()
    assert C.validate_memoryless_answer_application(run).ok
    calibration = C.calibrate_memoryless_answer_application(run)
    assert calibration["fires"]
    assert all(calibration["rejected"].values())


def test_active_interval_contract_and_its_planted_defects():
    value = _active_bundle()
    assert C.validate_active_intervals(value, 4).ok
    calibration = C.calibrate_active_intervals(value, 4)
    assert calibration["fires"]
    assert all(calibration["rejected"].values())


def test_active_interval_contract_rejects_floats_and_wrong_status():
    value = _active_bundle()
    value["intervals"]["task_accuracy"]["delta"] = 0.0
    value["component_status"] = "measured"
    got = C.validate_active_intervals(value, 4)
    assert not got.ok
    assert any("not exact" in error for error in got.errors)
