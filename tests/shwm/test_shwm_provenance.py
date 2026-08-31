"""Scale-0 gate: freeze, taint, and the final-seed guard.

The rule these tests defend is the cheapest one to break and the most expensive
one to detect afterwards: a final seed that becomes visible before the freeze is
committed cannot be un-seen, and every result downstream of it is retracted.
"""

from __future__ import annotations

import pytest

from sentinel.wm.latent_contract import ContractViolation, Taint
from sentinel.wm.provenance import (
    FinalSeedAccessError,
    FinalSeedGuard,
    FreezeManifest,
    TaintLedger,
    environment_state,
    git_state,
)
from sentinel.wm.versioning import digest_file, digest_of


def manifest(**overrides) -> FreezeManifest:
    base = dict(
        phase="SHWM-SCALE-0",
        base_commit="5205543b110ba6da2e3f6da30630809941f821c4",
        implementation_commit="0" * 40,
        dirty_tracked=False,
        dependency_lock_sha256=digest_of("lock"),
        encoder_identities=(),
        environment_generator_sha256=digest_of("gen"),
        split_procedure_sha256=digest_of("split"),
        evaluator_sha256=digest_of("eval"),
        config_sha256=digest_of("config"),
        gate_document_sha256=digest_of("gates"),
    )
    base.update(overrides)
    return FreezeManifest(**base)


def test_scale0_manifest_names_no_final_seed_file():
    assert manifest().final_seed_file is None
    assert manifest().created_before_final_seed is True
    assert manifest().grants_final_seed_access is False


def test_manifest_cannot_name_a_seed_file_and_claim_to_predate_it():
    with pytest.raises(ContractViolation):
        manifest(final_seed_file=digest_of("seeds"), created_before_final_seed=True)


def test_manifest_digest_changes_with_every_field(tmp_path):
    base = manifest()
    for field, value in [
        ("base_commit", "1" * 40),
        ("implementation_commit", "2" * 40),
        ("dirty_tracked", True),
        ("dependency_lock_sha256", digest_of("other-lock")),
        ("evaluator_sha256", digest_of("other-eval")),
        ("gate_document_sha256", digest_of("other-gates")),
    ]:
        assert manifest(**{field: value}).digest != base.digest, field
    written = tmp_path / "freeze.json"
    assert base.write(written) == base.digest
    assert written.exists()


def test_final_seeds_cannot_be_loaded_without_a_manifest(tmp_path):
    seeds = tmp_path / "final-seeds.txt"
    seeds.write_text("1 2 3\n")
    with pytest.raises(FinalSeedAccessError):
        FinalSeedGuard().load_final_seeds(seeds)


def test_final_seeds_cannot_be_loaded_under_a_scale0_manifest(tmp_path):
    seeds = tmp_path / "final-seeds.txt"
    seeds.write_text("1 2 3\n")
    guard = FinalSeedGuard(manifest=manifest())
    with pytest.raises(FinalSeedAccessError):
        guard.load_final_seeds(seeds)


def test_final_seeds_load_only_after_a_post_freeze_manifest_names_them(tmp_path):
    seeds = tmp_path / "final-seeds.txt"
    seeds.write_text("7001 7002\n")
    granting = manifest(final_seed_file=digest_file(seeds), created_before_final_seed=False)
    assert FinalSeedGuard(manifest=granting).load_final_seeds(seeds) == (7001, 7002)


def test_a_tampered_seed_file_is_rejected_even_with_a_granting_manifest(tmp_path):
    seeds = tmp_path / "final-seeds.txt"
    seeds.write_text("7001 7002\n")
    granting = manifest(final_seed_file=digest_file(seeds), created_before_final_seed=False)
    seeds.write_text("9999\n")
    with pytest.raises(FinalSeedAccessError):
        FinalSeedGuard(manifest=granting).load_final_seeds(seeds)


def test_only_the_three_frozen_development_seeds_are_accepted():
    guard = FinalSeedGuard()
    for seed in (6600, 6601, 6602):
        assert guard.check_development_seed(seed) == seed
    for seed in (6603, 42, 0):
        with pytest.raises(FinalSeedAccessError):
            guard.check_development_seed(seed)


def test_taint_ledger_blocks_a_consumer_from_a_taint_it_may_not_see():
    ledger = TaintLedger(
        permitted={
            "trainer": frozenset({Taint.DEVELOPMENT}),
            "evaluator": frozenset({Taint.DEVELOPMENT, Taint.EVALUATOR_ONLY, Taint.ORACLE}),
        }
    )
    ledger.record("trainer", frozenset({Taint.DEVELOPMENT}))
    ledger.record("evaluator", frozenset({Taint.EVALUATOR_ONLY}))
    with pytest.raises(ContractViolation):
        ledger.record("trainer", frozenset({Taint.FINAL}))
    with pytest.raises(ContractViolation):
        ledger.record("undeclared-consumer", frozenset({Taint.DEVELOPMENT}))


def test_provenance_probes_report_the_running_environment(tmp_path):
    state = git_state(tmp_path)
    assert set(state) >= {"commit", "branch", "dirty_tracked", "untracked_entries"}
    env = environment_state()
    assert env["python"].startswith("3.")
    assert "mlx" in env
