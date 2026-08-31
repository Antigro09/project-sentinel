"""Scale-0 gate: an interrupted run continues as if it had never stopped.

The gate is not "a checkpoint loads". It is that a run split in half across two
fresh interpreters produces bit-identical weights to an uninterrupted one. The
two differ by exactly the state nobody writes down -- a global random stream, a
module-level cache, a counter in a closure -- so the halves run as subprocesses.

Three failure directions are covered: state that was never declared, a
checkpoint that was altered, and a checkpoint that belongs to a different run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from sentinel.wm.restart import (
    AUDITED_MODULES,
    CheckpointCorruption,
    DeclaredRunState,
    ProcessStateAudit,
    UndeclaredState,
    assert_restartable,
    key_to_tuple,
    load_run_state,
    save_run_state,
    verify_checkpoint,
)
from sentinel.wm.versioning import digest_of

WORKER = Path(__file__).parent / "_restart_worker.py"
UPDATES = 6


def run_worker(phase: str, workdir: Path, updates: int) -> dict:
    completed = subprocess.run(
        [sys.executable, str(WORKER), phase, str(workdir), str(updates)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if completed.returncode != 0:
        raise AssertionError(f"worker {phase} failed:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_a_run_split_across_two_processes_matches_an_uninterrupted_one(tmp_path):
    uninterrupted = run_worker("full", tmp_path / "full", UPDATES)
    first = run_worker("first", tmp_path / "split", UPDATES // 2)
    second = run_worker("second", tmp_path / "split", UPDATES - UPDATES // 2)

    assert first["data_digest"] == uninterrupted["data_digest"], "collection is not reproducible"
    assert second["updates"] == uninterrupted["updates"] == UPDATES
    assert second["loss_history"] == uninterrupted["loss_history"]
    assert second["parameters_digest"] == uninterrupted["parameters_digest"]
    assert second["prng_key"] == uninterrupted["prng_key"]


def state(**overrides) -> DeclaredRunState:
    base = dict(
        update_index=3,
        prng_key=key_to_tuple(mx.random.key(6600)),
        batch_cursor=3,
        permutation_digest=digest_of("permutation"),
        config_digest=digest_of("config"),
        objective_digest=digest_of("objective"),
        data_digest=digest_of("data"),
        split_manifest_digest=digest_of("splits"),
        planner_account={"invocations": 2},
        gate_ledger={"issued": 5},
        verifier_ledger={"verifications": 5},
    )
    base.update(overrides)
    return DeclaredRunState(**base)


def trained_pair():
    model = nn.Linear(16, 16)
    optimizer = optim.AdamW(learning_rate=1e-3)

    def loss(model):
        return mx.sum(model(mx.ones((2, 16))) ** 2)

    _, grads = nn.value_and_grad(model, loss)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return model, optimizer


def test_the_checkpoint_round_trips_weights_optimizer_and_declared_state(tmp_path):
    model, optimizer = trained_pair()
    original = state()
    save_run_state(tmp_path, original, model, optimizer)

    restored_model, restored_optimizer = trained_pair()
    loaded = load_run_state(tmp_path, restored_model, restored_optimizer)
    assert loaded.digest == original.digest
    assert bool(mx.all(model.weight == restored_model.weight).item())


@pytest.mark.parametrize("filename", ["model.safetensors", "optimizer.safetensors", "state.json"])
def test_a_corrupted_checkpoint_fails_before_anything_is_deserialised(tmp_path, filename):
    model, optimizer = trained_pair()
    save_run_state(tmp_path, state(), model, optimizer)
    path = tmp_path / filename
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 0xFF
    path.write_bytes(bytes(blob))
    with pytest.raises(CheckpointCorruption):
        verify_checkpoint(tmp_path)
    with pytest.raises(CheckpointCorruption):
        load_run_state(tmp_path, *trained_pair())


def test_a_missing_file_or_missing_checksum_manifest_fails_closed(tmp_path):
    model, optimizer = trained_pair()
    save_run_state(tmp_path, state(), model, optimizer)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(CheckpointCorruption, match="missing"):
        verify_checkpoint(tmp_path)
    (tmp_path / "checksums.json").unlink()
    with pytest.raises(CheckpointCorruption, match="checksums"):
        verify_checkpoint(tmp_path)


def test_the_declared_state_carries_the_stream_position_not_just_the_seed():
    """A seed says where the stream started; a restart needs where it is now."""
    first = state()
    advanced = state(prng_key=key_to_tuple(mx.random.split(mx.random.key(6600))[1]))
    assert first.digest != advanced.digest
    assert first.key.shape == (2,)


def test_a_forbidden_global_state_channel_is_detected():
    """Calibration arm: the audit must catch a planted global, or it proves nothing."""
    audit = ProcessStateAudit()
    audit.capture()
    assert audit.changed() == []
    audit.assert_no_undeclared_state()

    import sentinel.wm.cache as planted_module

    planted_module._PLANTED_FORBIDDEN_CACHE = {"answers": [1, 2, 3]}
    try:
        assert "sentinel.wm.cache" in audit.changed()
        with pytest.raises(UndeclaredState, match="sentinel.wm.cache"):
            audit.assert_no_undeclared_state()
        # An explicitly declared module is allowed to move.
        audit.assert_no_undeclared_state(declared=["sentinel.wm.cache"])
    finally:
        del planted_module._PLANTED_FORBIDDEN_CACHE
    assert audit.changed() == []


def test_the_audit_covers_every_module_a_run_touches():
    assert "sentinel.wm.models" in AUDITED_MODULES
    assert "sentinel.wm.cache" in AUDITED_MODULES
    assert "sentinel.env.adapters.synthetic_control" in AUDITED_MODULES
    audit = ProcessStateAudit()
    snapshot = audit.capture()
    assert set(snapshot) == set(AUDITED_MODULES)


def test_sampling_from_the_global_random_stream_is_refused_for_a_matrix_run():
    assert_restartable(False)
    with pytest.raises(UndeclaredState, match="global MLX random stream"):
        assert_restartable(True)


# ---- determinism findings, locked in -------------------------------------------


def _training_fixture(tmp_path, arm, updates: int = 4):
    """A small but structurally complete training setup."""
    from sentinel.env.adapters.synthetic_control import SyntheticControlAdapter
    from sentinel.wm.cache import LatentCache
    from sentinel.wm.collect import CollectionPlan, FeatureTable, SequenceSampler, collect
    from sentinel.wm.dataset import CollectorPolicy, Split, SplitManifest
    from sentinel.wm.encoder import CachedEncoder, DeterministicControlEncoder
    from sentinel.wm.models import build_model
    from sentinel.wm.objective import ObjectiveConfig
    from sentinel.wm.sizing import solve_config
    from sentinel.wm.trainer import Trainer, build_optimizer

    plan = CollectionPlan(
        environment="synthetic_control",
        transitions=600,
        mixture={
            CollectorPolicy.RANDOM: 180,
            CollectorPolicy.SCRIPTED_ORACLE: 150,
            CollectorPolicy.SENTINEL: 150,
            CollectorPolicy.UNCERTAINTY_SEEKING: 120,
        },
        episode_length=40,
    )
    split_manifest = SplitManifest(
        salt="determinism", weights={Split.TRAIN: 0.8, Split.DEV_HELD_OUT: 0.2}
    )
    encoder = CachedEncoder(
        DeterministicControlEncoder(feature_dimension=32),
        LatentCache(tmp_path / "cache"),
        digest_of("projector"),
    )
    collected = collect(
        lambda gate: SyntheticControlAdapter(gate=gate),
        plan,
        split_manifest,
        encoder,
        family="synthetic_control",
    )
    table = FeatureTable.from_mapping(collected.features)
    sized = solve_config(arm, 50_000_000, encoder_dimension=32, latent_width=256, action_count=4)

    def fresh():
        return Trainer(
            model=build_model(sized.config, seed=6600),
            optimizer=build_optimizer(),
            sampler=SequenceSampler.from_records(
                collected.records,
                split_manifest,
                split=Split.TRAIN,
                sequence_length=8,
                batch_size=4,
                seed=6600,
            ),
            table=table,
            objective=ObjectiveConfig(),
            seed=6600,
            data_digest=collected.transition_ids_digest,
            split_manifest_digest=split_manifest.digest,
        )

    return fresh


@pytest.mark.parametrize("arm_name", ["continuous", "discrete", "hybrid"])
def test_two_identical_runs_produce_bit_identical_weights(tmp_path, arm_name):
    """Regression for a non-deterministic backward that a matching loss hid.

    The action embedding was a gather, so its backward was a scatter-add over
    four rows receiving thousands of contributions. The ordering of that
    accumulation is not fixed, the weights diverged in the low bf16 bits, and
    the float32 loss printed identical for four updates before anything showed.
    """
    from sentinel.wm.latent_contract import RepresentationKind
    from sentinel.wm.trainer import parameter_digest

    fresh = _training_fixture(tmp_path, RepresentationKind(arm_name))
    first, second = fresh(), fresh()
    first.run(4, diagnose_every=0)
    second.run(4, diagnose_every=0)
    assert first.loss_history == second.loss_history
    assert parameter_digest(first.model) == parameter_digest(second.model)


@pytest.mark.parametrize("arm_name", ["continuous", "discrete", "hybrid"])
def test_a_checkpointed_half_run_finishes_where_the_whole_run_did(tmp_path, arm_name):
    from sentinel.wm.latent_contract import RepresentationKind
    from sentinel.wm.trainer import parameter_digest

    fresh = _training_fixture(tmp_path, RepresentationKind(arm_name))
    whole = fresh()
    whole.run(4, diagnose_every=0)

    half = fresh()
    half.run(2, diagnose_every=0)
    half.save(tmp_path / f"checkpoint-{arm_name}")

    resumed = fresh()
    resumed.restore(tmp_path / f"checkpoint-{arm_name}")
    resumed.run(2, diagnose_every=0)

    assert resumed.loss_history == whole.loss_history
    assert parameter_digest(resumed.model) == parameter_digest(whole.model)


def test_the_action_embedding_gradient_is_deterministic():
    """The narrow calibration arm for the fix above."""
    import numpy as np
    from mlx.utils import tree_flatten

    from sentinel.wm.latent_contract import RepresentationKind
    from sentinel.wm.models import build_model
    from sentinel.wm.objective import ObjectiveBatch, ObjectiveConfig, compute_objective
    from sentinel.wm.sizing import solve_config

    sized = solve_config(
        RepresentationKind.CONTINUOUS,
        50_000_000,
        encoder_dimension=32,
        latent_width=256,
        action_count=4,
    )
    mx.random.seed(0)
    batch = ObjectiveBatch(
        features=mx.random.normal((4, 8, 32)).astype(mx.bfloat16),
        actions=mx.random.randint(0, 4, (4, 8)),
        previous_rewards=mx.zeros((4, 8, 1), dtype=mx.bfloat16),
        rewards=mx.zeros((4, 8)),
        terminations=mx.zeros((4, 8)),
        event_targets=mx.zeros((4, 8), dtype=mx.int32),
    )
    key = mx.random.key(7)
    config = ObjectiveConfig()

    def gradients():
        model = build_model(sized.config, seed=6600)

        def loss_fn(model):
            output = model(batch.features, batch.actions, batch.previous_rewards, key=key)
            total, _, _, _ = compute_objective(model, output, batch, config)
            return total

        _, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(grads)
        return {
            name: np.asarray(tensor.astype(mx.float32)).tobytes()
            for name, tensor in tree_flatten(grads)
        }

    first, second, third = gradients(), gradients(), gradients()
    differing = sorted(name for name in first if first[name] != second[name])
    assert differing == [], differing
    assert first == third


def test_optimizer_accumulators_are_float32_over_bfloat16_weights():
    """The matrix freezes bf16 weights with fp32 accumulators; MLX defaults to
    bf16 for both, which is eight significand bits holding a running average."""
    from mlx.utils import tree_flatten

    from sentinel.wm.latent_contract import RepresentationKind
    from sentinel.wm.models import build_model
    from sentinel.wm.sizing import solve_config
    from sentinel.wm.trainer import build_optimizer

    sized = solve_config(
        RepresentationKind.HYBRID,
        50_000_000,
        encoder_dimension=32,
        latent_width=256,
        action_count=4,
    )
    model = build_model(sized.config, seed=6600)

    def loss_fn(model):
        output = model(
            mx.zeros((2, 4, 32), dtype=mx.bfloat16),
            mx.zeros((2, 4), dtype=mx.int32),
            mx.zeros((2, 4, 1), dtype=mx.bfloat16),
            key=mx.random.key(1),
        )
        return mx.sum(output.next_latent.astype(mx.float32) ** 2)

    optimizer = build_optimizer()
    _, grads = nn.value_and_grad(model, loss_fn)(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    weights = {name: tensor.dtype for name, tensor in tree_flatten(model.parameters())}
    moments = {
        name: tensor.dtype
        for name, tensor in tree_flatten(optimizer.state)
        if name.endswith(".m") or name.endswith(".v")
    }
    assert moments, "the optimizer exposed no per-parameter moments"
    assert set(weights.values()) == {mx.bfloat16}
    assert set(moments.values()) == {mx.float32}


def test_persistent_state_contains_no_future(tmp_path):
    """The no-answer-leakage gate, applied to what actually survives a restart.

    `EXPERIMENT-GATES.md` requires an audit that persistent state carries no
    target sequence, expected observation, hidden mechanic, evaluator predicate,
    clarification answer, final split label, or branch sibling from another
    split. The checkpoint is the only persistent state a run has, so the audit
    is: enumerate its keys, and require them to be exactly the declared set.

    An allowlist rather than a denylist, because a denylist only catches the
    leaks someone thought to name.
    """
    import json

    from sentinel.wm.latent_contract import HIDDEN_FIELD_NAMES

    model, optimizer = trained_pair()
    save_run_state(tmp_path, state(), model, optimizer)
    document = json.loads((tmp_path / "state.json").read_text())

    declared = {
        "update_index",
        "prng_key",
        "batch_cursor",
        "permutation_digest",
        "config_digest",
        "objective_digest",
        "data_digest",
        "split_manifest_digest",
        "planner_account",
        "gate_ledger",
        "verifier_ledger",
        "pending_counterexamples",
        "loss_history",
    }
    assert set(document) == declared, set(document) ^ declared

    # No hidden field name anywhere in the serialised state, at any depth.
    text = json.dumps(document)
    for name in HIDDEN_FIELD_NAMES:
        assert f'"{name}"' not in text, name

    # And nothing that could stand in for an answer: the state references data
    # only by digest, never by content.
    assert isinstance(document["data_digest"], str)
    assert document["data_digest"].startswith("sha256:")
    assert "records" not in document and "targets" not in document


def test_the_checkpoint_directory_holds_only_the_declared_files(tmp_path):
    model, optimizer = trained_pair()
    save_run_state(tmp_path, state(), model, optimizer)
    written = {p.name for p in tmp_path.iterdir() if p.is_file()}
    assert written == {"state.json", "model.safetensors", "optimizer.safetensors", "checksums.json"}
