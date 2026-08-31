"""Scale-0 gate: the encoder preflight refuses to improvise.

The matrix names two frozen encoder families and forbids picking a third after
results exist. These tests hold the preflight to that: a blocked family stops
the matrix, a gated licence is a block rather than something to work around, and
the candidate list itself is checked against the document.

The suite stays hermetic. Network behaviour is exercised through synthesised
preflight records, and the one live path is called with `allow_network=False`.
"""

from __future__ import annotations

import pytest

from sentinel.wm.backbones import (
    FROZEN_CANDIDATES,
    BackbonePreflight,
    BlockReason,
    PreflightVerdict,
    detect_runtime,
    matrix_may_run,
    preflight_all,
)
from sentinel.wm.latent_contract import Precision


def record(candidate, verdict, reasons=()) -> BackbonePreflight:
    return BackbonePreflight(candidate=candidate, verdict=verdict, reasons=tuple(reasons))


def test_the_frozen_candidate_list_is_exactly_the_two_named_families():
    assert len(FROZEN_CANDIDATES) == 2
    by_id = {c.encoder_id: c for c in FROZEN_CANDIDATES}
    assert set(by_id) == {"qwen3_vl_4b", "gemma3_4b"}
    assert by_id["qwen3_vl_4b"].repository == "Qwen/Qwen3-VL-4B-Instruct"
    assert by_id["gemma3_4b"].repository == "google/gemma-3-4b-it"
    assert all(c.declared_precision is Precision.BF16 for c in FROZEN_CANDIDATES)


def test_the_matrix_runs_only_when_both_families_are_runnable():
    both_ok = tuple(record(c, PreflightVerdict.RUNNABLE) for c in FROZEN_CANDIDATES)
    ok, why = matrix_may_run(both_ok)
    assert ok and "runnable" in why


@pytest.mark.parametrize("blocked_index", [0, 1])
def test_one_blocked_family_stops_the_whole_matrix(blocked_index):
    records = []
    for i, candidate in enumerate(FROZEN_CANDIDATES):
        if i == blocked_index:
            records.append(
                record(candidate, PreflightVerdict.BLOCKED, [BlockReason.GATED_LICENCE])
            )
        else:
            records.append(record(candidate, PreflightVerdict.RUNNABLE))
    ok, why = matrix_may_run(tuple(records))
    assert not ok
    assert FROZEN_CANDIDATES[blocked_index].encoder_id in why


def test_an_unchecked_family_is_not_treated_as_runnable():
    records = (
        record(FROZEN_CANDIDATES[0], PreflightVerdict.RUNNABLE),
        record(FROZEN_CANDIDATES[1], PreflightVerdict.UNCHECKED),
    )
    ok, _ = matrix_may_run(records)
    assert not ok


def test_preflight_records_hash_their_whole_evidence():
    a = record(FROZEN_CANDIDATES[0], PreflightVerdict.RUNNABLE)
    b = record(FROZEN_CANDIDATES[0], PreflightVerdict.BLOCKED, [BlockReason.NO_CREDENTIAL])
    assert a.digest != b.digest


def test_offline_preflight_reports_unchecked_rather_than_guessing():
    results = preflight_all(allow_network=False)
    assert len(results) == 2
    assert all(r.verdict is PreflightVerdict.UNCHECKED for r in results)
    ok, _ = matrix_may_run(results)
    assert not ok


def test_runtime_detection_reports_what_is_actually_importable():
    runtime = detect_runtime()
    assert set(runtime) >= {"mlx", "torch", "transformers", "metal"}
    assert all(isinstance(v, bool) for v in runtime.values())
