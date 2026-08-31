"""Scale-0 gate: a run either matches the frozen matrix or it is not a matrix run.

`SCALE-0-RUN-MATRIX.md` is the contract; `sentinel.wm.matrix` is its executable
transcription. These tests check the transcription against the document's own
arithmetic and then check that the matching rule actually rejects violations --
including the ones that would be tempting to wave through, such as a dropped
failed seed or a 2% parameter drift.
"""

from __future__ import annotations

import pytest

from sentinel.wm import matrix as M
from sentinel.wm.latent_contract import ContractViolation, RepresentationKind
from sentinel.wm.versioning import digest_of


def claim(**overrides) -> M.RunClaim:
    base = dict(
        workload_id="qwen3_vl_4b.hybrid.50M.w512.s6600",
        seed=6600,
        trainable_parameters=50_000_000,
        target_parameters=50_000_000,
        transition_ids_digest=digest_of("transitions"),
        split_manifest_digest=digest_of("splits"),
        optimizer_updates=M.OPTIMIZER_UPDATES,
        sequence_length=M.SEQUENCE_LENGTH,
        sequences_per_batch=M.SEQUENCES_PER_BATCH,
        online_interactions=0,
        planner_invocations=len(M.PLANNER_HORIZONS) * M.PLANNER_INVOCATIONS_PER_HORIZON,
        planner_candidates=M.PLANNER_CANDIDATES_TOTAL,
        required_probe_digest=M.REQUIRED_PROBE_DIGEST,
    )
    base.update(overrides)
    return M.RunClaim(**base)


# ---- the document's arithmetic ----------------------------------------------


def test_matrix_arithmetic_reproduces_the_frozen_counts():
    assert M.matrix_arithmetic() == {
        "primary_cells": 12,
        "primary_workloads": 36,
        "dimension_control_cells": 4,
        "dimension_control_workloads": 12,
        "total_workloads": 48,
        "transition_positions_per_run": 204_800,
        "planner_candidates_per_run": 19_200,
    }


def test_frozen_factors_match_the_document():
    assert M.ENCODER_IDS == ("qwen3_vl_4b", "gemma3_4b")
    assert M.REPRESENTATION_ARMS == (
        RepresentationKind.CONTINUOUS,
        RepresentationKind.DISCRETE,
        RepresentationKind.HYBRID,
    )
    assert M.TRAINABLE_TARGETS == (50_000_000, 200_000_000)
    assert M.DEVELOPMENT_SEEDS == (6600, 6601, 6602)
    assert M.PRIMARY_WIDTH == 512
    assert M.DIMENSION_CONTROL_WIDTHS == (256, 1024)
    assert M.TOTAL_TRANSITIONS == 100_000
    assert sum(M.TRANSITIONS_PER_ENVIRONMENT.values()) == M.TOTAL_TRANSITIONS
    assert sum(M.MIXTURE_COUNTS.values()) == 50_000  # per environment
    assert M.ONLINE_INTERACTIONS_PER_RUN == 0
    assert M.LOSS_PLUMBING_WEIGHT == 1.0


def test_dimension_control_covers_both_encoders_at_the_hybrid_50m_arm():
    controls = M.dimension_control_cells()
    assert len(controls) == 4
    assert {c.encoder_id for c in controls} == set(M.ENCODER_IDS)
    assert {c.latent_width for c in controls} == {256, 1024}
    assert {c.representation for c in controls} == {RepresentationKind.HYBRID}
    assert {c.target_parameters for c in controls} == {50_000_000}


def test_every_workload_id_is_unique():
    ids = [w.workload_id for w in M.all_workloads()]
    assert len(ids) == len(set(ids)) == 48


def test_resource_ceilings_match_the_document():
    assert M.PEAK_MEMORY_LIMIT_BYTES == 112 * 1024**3
    assert M.ARTIFACT_LIMIT_BYTES == 200 * 1024**3
    assert M.RUN_TIMEOUT_SECONDS == 7200
    assert M.MATRIX_TIMEOUT_SECONDS == 259_200
    assert M.CACHE_BUILD_TIMEOUT_SECONDS == 28_800


# ---- the matching rule bites -------------------------------------------------


def test_a_conforming_run_matches():
    assert M.check_match([claim()]) == []


@pytest.mark.parametrize(
    "override,fragment",
    [
        ({"trainable_parameters": 51_000_000}, "outside +/-1%"),
        ({"trainable_parameters": 49_000_000}, "outside +/-1%"),
        ({"optimizer_updates": 199}, "optimizer updates"),
        ({"sequence_length": 16}, "batch shape"),
        ({"sequences_per_batch": 64}, "batch shape"),
        ({"online_interactions": 1}, "online interactions"),
        ({"planner_invocations": 299}, "planner invocations"),
        ({"planner_candidates": 19_199}, "planner candidates"),
        ({"required_probe_digest": digest_of(["reward"])}, "required probe set"),
        ({"optimizer": "SGD"}, "optimizer SGD"),
        ({"seed": 1234}, "non-development seed"),
    ],
)
def test_each_matching_rule_rejects_its_own_violation(override, fragment):
    failures = M.check_match([claim(**override)])
    assert failures, f"{override} was accepted"
    assert any(fragment in f for f in failures), failures


def test_parameter_tolerance_is_exactly_one_percent():
    assert M.parameters_within_tolerance(50_500_000, 50_000_000)
    assert M.parameters_within_tolerance(49_500_000, 50_000_000)
    assert not M.parameters_within_tolerance(50_500_001, 50_000_000)
    assert not M.parameters_within_tolerance(49_499_999, 50_000_000)


def test_arms_using_different_data_or_splits_are_not_matched():
    a = claim()
    b = claim(workload_id="other", transition_ids_digest=digest_of("other-transitions"))
    assert any("transition-ID sets" in f for f in M.check_match([a, b]))
    c = claim(workload_id="other", split_manifest_digest=digest_of("other-splits"))
    assert any("split manifests" in f for f in M.check_match([a, c]))


def test_a_dropped_failed_seed_is_a_matching_failure():
    runs = [claim(seed=6600), claim(seed=6601), claim(seed=6600, workload_id="dup")]
    failures = M.check_match(runs)
    assert any("6602" in f for f in failures), failures


def test_control_encoders_cannot_be_used_as_matrix_encoders():
    for good in M.ENCODER_IDS:
        assert M.assert_matrix_encoder(good) == good
    for bad in ("sentinel-control", "mlx-community/gemma-3-4b-it", "qwen3_vl_8b"):
        with pytest.raises(ContractViolation):
            M.assert_matrix_encoder(bad)


# ---- resource envelope -------------------------------------------------------


def test_resource_envelope_flags_each_ceiling():
    assert M.check_resource_envelope(1, 1, 1.0) == []
    assert any("peak process memory" in f for f in M.check_resource_envelope(113 * 1024**3, 1, 1.0))
    assert any("artifact storage" in f for f in M.check_resource_envelope(1, 201 * 1024**3, 1.0))
    assert any("per-run ceiling" in f for f in M.check_resource_envelope(1, 1, 7201))
    assert any("72 h" in f for f in M.check_resource_envelope(1, 1, 1.0, matrix_seconds=259_201))
    assert any("8 h" in f for f in M.check_resource_envelope(1, 1, 1.0, cache_build_seconds=28_801))
