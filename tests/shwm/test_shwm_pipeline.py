"""Scale-0 gate: the driver runs the frozen workload and cannot pretend otherwise.

Two things are checked that no other file covers. The configured cell list must
equal the frozen matrix, so a config edit cannot quietly add or drop a cell. And
a dry run must be incapable of passing the Scale-0 gate however well it goes --
otherwise the fake-model preflight becomes a way of declaring the matrix
complete without the backbones that define it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments" / "shwm"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from sentinel.wm import matrix as M  # noqa: E402
from sentinel.wm.dataset import CollectorPolicy  # noqa: E402
from sentinel.wm.latent_contract import RepresentationKind  # noqa: E402

import yaml  # noqa: E402

import dataset as dataset_module  # noqa: E402
import workload as workload_module  # noqa: E402

CONFIG_PATH = EXPERIMENTS / "configs" / "scale0.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_the_config_transcribes_the_frozen_matrix(config):
    matrix = config["matrix"]
    assert matrix["encoders"] == list(M.ENCODER_IDS)
    assert matrix["representations"] == [arm.value for arm in M.REPRESENTATION_ARMS]
    assert matrix["targets"] == list(M.TRAINABLE_TARGETS)
    assert matrix["primary_width"] == M.PRIMARY_WIDTH
    assert matrix["control_widths"] == list(M.DIMENSION_CONTROL_WIDTHS)
    assert matrix["seeds"] == list(M.DEVELOPMENT_SEEDS)


def test_the_config_transcribes_the_frozen_workload(config):
    optimisation = config["optimisation"]
    assert optimisation["sequence_length"] == M.SEQUENCE_LENGTH
    assert optimisation["sequences_per_batch"] == M.SEQUENCES_PER_BATCH
    assert optimisation["updates"] == M.OPTIMIZER_UPDATES
    assert optimisation["optimizer"] == M.OPTIMIZER
    assert optimisation["learning_rate"] == pytest.approx(M.LEARNING_RATE)
    assert tuple(optimisation["betas"]) == M.BETAS
    assert optimisation["weight_decay"] == M.WEIGHT_DECAY
    assert optimisation["gradient_clip_norm"] == M.GRADIENT_CLIP_NORM
    assert optimisation["compute_precision"] == M.COMPUTE_PRECISION.value
    assert optimisation["accumulator_precision"] == M.ACCUMULATOR_PRECISION.value
    assert optimisation["loss_weight"] == M.LOSS_PLUMBING_WEIGHT

    planner = config["planner"]
    assert tuple(planner["horizons"]) == M.PLANNER_HORIZONS
    assert planner["invocations_per_horizon"] == M.PLANNER_INVOCATIONS_PER_HORIZON
    assert planner["candidates_per_invocation"] == M.PLANNER_CANDIDATES_PER_INVOCATION
    assert planner["online_actions"] == M.ONLINE_INTERACTIONS_PER_RUN

    data = config["data"]
    assert data["total_transitions"] == M.TOTAL_TRANSITIONS
    assert dict(data["per_environment"]) == dict(M.TRANSITIONS_PER_ENVIRONMENT)
    assert sum(data["mixture"].values()) == pytest.approx(1.0)

    limits = config["limits"]
    assert limits["peak_memory_gib"] * 1024**3 == M.PEAK_MEMORY_LIMIT_BYTES
    assert limits["artifact_gib"] * 1024**3 == M.ARTIFACT_LIMIT_BYTES
    assert limits["run_timeout_seconds"] == M.RUN_TIMEOUT_SECONDS
    assert limits["matrix_timeout_seconds"] == M.MATRIX_TIMEOUT_SECONDS
    assert limits["cache_build_timeout_seconds"] == M.CACHE_BUILD_TIMEOUT_SECONDS


def test_the_configured_cells_equal_the_frozen_cells(config):
    from scale0_preflight import cells_from_config

    cells = cells_from_config(config)
    assert {c.cell_id for c in cells} == {c.cell_id for c in M.all_cells()}
    assert len(cells) == 16
    assert len(cells) * len(config["matrix"]["seeds"]) == 48


def test_a_config_that_drops_a_cell_is_refused(config):
    from scale0_preflight import cells_from_config

    tampered = {**config, "matrix": {**config["matrix"], "representations": ["continuous", "hybrid"]}}
    with pytest.raises(SystemExit, match="matrix amendment"):
        cells_from_config(tampered)


def test_the_mixture_plan_sums_to_the_declared_transition_count(config):
    plans = dataset_module.build_plans(config["data"])
    assert set(plans) == set(M.TRANSITIONS_PER_ENVIRONMENT)
    for environment, plan in plans.items():
        assert plan.transitions == M.TRANSITIONS_PER_ENVIRONMENT[environment]
        assert sum(plan.mixture.values()) == plan.transitions
        assert set(plan.mixture) == set(CollectorPolicy)


def test_the_planner_dry_run_spends_exactly_the_frozen_budget(config):
    account, invocations, candidates = workload_module.planner_dry_run(config)
    assert invocations == len(M.PLANNER_HORIZONS) * M.PLANNER_INVOCATIONS_PER_HORIZON
    assert candidates == M.PLANNER_CANDIDATES_TOTAL
    assert account["score_calls"] == M.PLANNER_CANDIDATES_TOTAL
    # A distinct root per invocation, or the reported rate measures a cache.
    assert account["distinct_plans"] > invocations // 2


def test_the_verifier_dry_run_is_scored_against_planted_mismatches(tmp_path):
    """A detection rate with no known-bad inputs measures nothing."""
    from sentinel.env.adapters.synthetic_control import SyntheticControlAdapter
    from sentinel.wm.cache import LatentCache
    from sentinel.wm.collect import CollectionPlan, collect
    from sentinel.wm.dataset import Split, SplitManifest
    from sentinel.wm.encoder import CachedEncoder, DeterministicControlEncoder
    from sentinel.wm.versioning import digest_of

    plan = CollectionPlan(
        environment="synthetic_control",
        transitions=200,
        mixture={
            CollectorPolicy.RANDOM: 60,
            CollectorPolicy.SCRIPTED_ORACLE: 50,
            CollectorPolicy.SENTINEL: 50,
            CollectorPolicy.UNCERTAINTY_SEEKING: 40,
        },
    )
    manifest = SplitManifest(salt="v", weights={Split.TRAIN: 1.0})
    encoder = CachedEncoder(
        DeterministicControlEncoder(feature_dimension=16),
        LatentCache(tmp_path),
        digest_of("projector"),
    )
    collected = collect(
        lambda gate: SyntheticControlAdapter(gate=gate), plan, manifest, encoder,
        family="synthetic_control",
    )
    report = workload_module.verifier_dry_run(collected.records, sample=100)
    assert report["planted_mismatches"] > 0
    assert report["detection_rate"] == 1.0, "the verifier missed a planted observable mismatch"
    assert report["authorised_actions"] + report["denied_actions"] == report["verifications"]
    assert report["bridge"]["verifications"] == report["verifications"]
    assert 0.0 < report["mean_coverage"] <= 1.0


def test_the_verifier_dry_run_consumes_no_environment_interactions(tmp_path):
    """Offline by construction: the matrix fixes online interactions at zero."""
    import inspect

    source = inspect.getsource(workload_module.verifier_dry_run)
    assert ".step(" not in source
    assert "record.probes_t1" in source


# ---- which encoder a run is allowed to use -------------------------------------


def test_a_dry_run_gets_the_control_encoder(config):
    from sentinel.wm.encoder import CONTROL_PROVIDER, DeterministicControlEncoder

    encoder = dataset_module.build_inner_encoder(
        "qwen3_vl_4b", "control-slot-a", {**config, "mode": "dry_run"}, 512
    )
    assert isinstance(encoder, DeterministicControlEncoder)
    assert encoder.identity.provider == CONTROL_PROVIDER


def test_a_matrix_run_refuses_a_slot_that_is_not_a_frozen_family(config):
    from sentinel.wm.latent_contract import ContractViolation

    with pytest.raises(ContractViolation, match="frozen matrix encoder"):
        dataset_module.build_inner_encoder(
            "sentinel-control", "x", {**config, "mode": "matrix"}, 512
        )


def test_a_matrix_run_refuses_to_fall_back_to_the_control_encoder(config, tmp_path):
    """Absent weights must stop the run, not quietly downgrade it.

    A silent fallback is the failure that would make a dry run indistinguishable
    from a matrix run in the artefact, which is the one confusion the two modes
    exist to prevent.
    """
    from sentinel.wm.latent_contract import ContractViolation

    tampered = {
        **config,
        "mode": "matrix",
        "encoder": {**config["encoder"], "weights_root": str(tmp_path / "absent")},
    }
    with pytest.raises(ContractViolation, match="cannot substitute the control encoder"):
        dataset_module.build_inner_encoder("gemma3_4b", "x", tampered, 512)


def test_the_config_pins_a_revision_and_licence_for_each_frozen_family(config):
    for encoder_id in M.ENCODER_IDS:
        revision = config["encoder"]["revisions"][encoder_id]
        assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
        assert config["encoder"]["licences"][encoder_id]
