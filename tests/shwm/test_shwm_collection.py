"""Scale-0 gate: collection produces the sealed data the matrix describes.

The mixture, the branch groups, the split order, and the feature table are all
checked here against the frozen contract, and the two families are checked
against *different* disjointness standards for a measured reason: one has a
finite reachable state space and one is generative.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.env.adapters.procedural_visual import ProceduralVisualAdapter
from sentinel.env.adapters.synthetic_control import SyntheticControlAdapter
from sentinel.wm.cache import LatentCache
from sentinel.wm.collect import (
    CollectionPlan,
    FeatureTable,
    SequenceSampler,
    collect,
    materialise,
    primary_event_index,
)
from sentinel.wm.dataset import (
    COLLECTION_MIXTURE,
    CollectorPolicy,
    LeakageError,
    Split,
    SplitManifest,
    assert_no_transition_overlap,
    assert_trainable,
    audit_splits,
    branch_coverage_report,
    mixture_report,
)
from sentinel.wm.encoder import CachedEncoder, DeterministicControlEncoder
from sentinel.wm.events import EVENT_INDEX, EventKind
from sentinel.wm.latent_contract import Taint
from sentinel.wm.versioning import digest_of

TRANSITIONS = 800


def plan_for(environment: str, transitions: int = TRANSITIONS) -> CollectionPlan:
    return CollectionPlan(
        environment=environment,
        transitions=transitions,
        mixture={
            policy: int(transitions * share) for policy, share in COLLECTION_MIXTURE.items()
        },
    )


def manifest() -> SplitManifest:
    return SplitManifest(salt="scale-0", weights={Split.TRAIN: 0.8, Split.DEV_HELD_OUT: 0.2})


def run(environment: str, tmp_path, width: int = 64, transitions: int = TRANSITIONS):
    factory = {
        "synthetic_control": lambda gate: SyntheticControlAdapter(gate=gate),
        "procedural_visual": lambda gate: ProceduralVisualAdapter(gate=gate),
    }[environment]
    cache = LatentCache(tmp_path / environment)
    encoder = CachedEncoder(
        DeterministicControlEncoder(feature_dimension=width), cache, digest_of("projector")
    )
    split_manifest = manifest()
    result = collect(
        factory, plan_for(environment, transitions), split_manifest, encoder, family=environment
    )
    return result, split_manifest, cache


@pytest.mark.parametrize("environment", ["synthetic_control", "procedural_visual"])
def test_collection_produces_exactly_the_requested_mixture(environment, tmp_path):
    result, _, _ = run(environment, tmp_path)
    assert len(result.records) == TRANSITIONS
    report = mixture_report(result.records)
    for policy, share in COLLECTION_MIXTURE.items():
        assert report[policy.value]["count"] == int(TRANSITIONS * share)
        assert report[policy.value]["share"] == pytest.approx(share)


@pytest.mark.parametrize("environment", ["synthetic_control", "procedural_visual"])
def test_the_structural_split_invariants_hold(environment, tmp_path):
    result, split_manifest, _ = run(environment, tmp_path)
    report = audit_splits(result.records, split_manifest)
    assert report["transitions"] == TRANSITIONS
    assert set(report["per_split"]) <= {"train", "dev_held_out"}
    assert report["branch_groups"] > 0


def test_the_generative_family_is_fully_disjoint_and_the_finite_one_is_measured(tmp_path):
    """Two families, two standards, and the reason is arithmetic rather than taste."""
    visual, visual_manifest, _ = run("procedural_visual", tmp_path)
    visual_report = audit_splits(visual.records, visual_manifest)
    assert_no_transition_overlap(visual_report, "procedural_visual")
    assert visual_report["observation_content_overlap_rate"] == 0.0

    finite, finite_manifest, _ = run("synthetic_control", tmp_path)
    finite_report = audit_splits(finite.records, finite_manifest)
    assert finite_report["distinct_observation_contents"] < TRANSITIONS
    assert finite_report["observation_content_overlap_rate"] > 0.0
    with pytest.raises(LeakageError):
        assert_no_transition_overlap(finite_report, "synthetic_control")


@pytest.mark.parametrize("environment", ["synthetic_control", "procedural_visual"])
def test_every_branch_group_actually_tried_more_than_one_action(environment, tmp_path):
    """A branch group of size one is a passive observation wearing a branch label."""
    result, _, _ = run(environment, tmp_path)
    coverage = branch_coverage_report(result.records)
    assert coverage["branch_groups"] > 0
    assert coverage["groups_with_multiple_actions"] == coverage["branch_groups"]


@pytest.mark.parametrize("environment", ["synthetic_control", "procedural_visual"])
def test_no_collected_record_carries_a_taint_training_may_not_see(environment, tmp_path):
    result, _, _ = run(environment, tmp_path)
    assert_trainable(result.records)
    assert all(r.taint == frozenset({Taint.DEVELOPMENT}) for r in result.records)


def test_the_oracle_reads_hidden_state_but_leaves_nothing_behind(tmp_path):
    """The oracle's lookahead is recorded as interactions, never as a record field."""
    result, _, _ = run("synthetic_control", tmp_path)
    oracle_records = [
        r for r in result.records if r.collector_policy is CollectorPolicy.SCRIPTED_ORACLE
    ]
    assert oracle_records
    assert result.oracle_lookaheads > 0
    for record in oracle_records:
        assert record.extra == {}
        assert record.taint == frozenset({Taint.DEVELOPMENT})


def test_propensity_distinguishes_a_policy_choice_from_a_forced_branch(tmp_path):
    """A branch action was forced by the collector, not selected by the policy.

    Recording the policy's own propensity on a forced action would misstate the
    behaviour distribution, and Theorem 1 makes that distribution part of what a
    later interventional claim rests on.
    """
    result, _, _ = run("synthetic_control", tmp_path)
    oracle = [r for r in result.records if r.collector_policy is CollectorPolicy.SCRIPTED_ORACLE]
    chosen = [r for r in oracle if r.branch_group_id is None]
    forced = [r for r in oracle if r.branch_group_id is not None]
    assert chosen and forced
    assert all(r.action_propensity == 1.0 for r in chosen), "the oracle is deterministic"
    assert all(0.0 < r.action_propensity < 1.0 for r in forced), "a forced branch is uniform"


def test_collection_interactions_are_reported_apart_from_a_run_s_zero(tmp_path):
    """A training run has zero online interactions; collection is not a run."""
    result, _, _ = run("synthetic_control", tmp_path)
    assert result.environment_interactions > len(result.records)


@pytest.mark.parametrize("environment", ["synthetic_control", "procedural_visual"])
def test_collection_is_reproducible_transition_for_transition(environment, tmp_path):
    first, _, _ = run(environment, tmp_path / "a")
    second, _, _ = run(environment, tmp_path / "b")
    assert first.transition_ids_digest == second.transition_ids_digest
    assert [r.digest for r in first.records] == [r.digest for r in second.records]


def test_the_cache_hits_on_a_finite_state_space_and_misses_on_a_generative_one(tmp_path):
    """The hit ratio is only meaningful once the key is the observation content."""
    _, _, finite_cache = run("synthetic_control", tmp_path / "a")
    _, _, visual_cache = run("procedural_visual", tmp_path / "b")
    assert finite_cache.stats.hit_ratio > 0.5
    assert finite_cache.stats.hit_ratio > visual_cache.stats.hit_ratio
    assert finite_cache.stats.stale_rejections == 0


def test_the_feature_table_matches_the_cache_it_was_built_from(tmp_path):
    result, _, _ = run("synthetic_control", tmp_path, width=32)
    table = FeatureTable.from_mapping(result.features)
    report = table.size_report()
    assert report["rows"] == len(result.features)
    assert report["width"] == 32
    assert report["bytes"] == report["rows"] * 32 * 4
    for digest, features in list(result.features.items())[:20]:
        assert np.array_equal(table.lookup([digest])[0], features)


def test_batches_have_the_frozen_shape_and_carry_branch_pairs(tmp_path):
    result, split_manifest, _ = run("synthetic_control", tmp_path, width=32)
    table = FeatureTable.from_mapping(result.features)
    sampler = SequenceSampler.from_records(
        result.records,
        split_manifest,
        split=Split.TRAIN,
        sequence_length=8,
        batch_size=4,
        seed=6600,
    )
    arrays = materialise(sampler.batch(0), table)
    assert arrays.features.shape == (4, 8, 32)
    assert arrays.actions.shape == (4, 8)
    assert arrays.previous_rewards.shape == (4, 8, 1)
    assert arrays.event_targets.max() < 12
    total_pairs = sum(
        len(materialise(sampler.batch(i), table).boundary_pairs) for i in range(8)
    )
    assert total_pairs > 0, "no batch contained a branch pair, so the boundary term is vacuous"


def test_the_sampler_permutation_depends_on_the_seed_and_nothing_else(tmp_path):
    result, split_manifest, _ = run("synthetic_control", tmp_path, width=16)
    def sampler(seed):
        return SequenceSampler.from_records(
            result.records, split_manifest, split=Split.TRAIN,
            sequence_length=8, batch_size=4, seed=seed,
        )
    assert sampler(6600).permutation_digest == sampler(6600).permutation_digest
    assert sampler(6600).permutation_digest != sampler(6601).permutation_digest


def test_sequences_never_span_two_episodes(tmp_path):
    result, split_manifest, _ = run("synthetic_control", tmp_path, width=16)
    sampler = SequenceSampler.from_records(
        result.records, split_manifest, split=Split.TRAIN,
        sequence_length=8, batch_size=4, seed=6600,
    )
    for sequence in sampler.sequences:
        assert len({r.episode_id for r in sequence.records}) == 1


def test_a_transition_with_nothing_observable_is_labelled_rather_than_dropped():
    assert primary_event_index(()) == EVENT_INDEX[EventKind.UNKNOWN_EVENT]


def test_the_event_label_prefers_the_probe_the_verifier_cares_about_most():
    from sentinel.wm.events import StructuredEvent

    events = (
        StructuredEvent(EventKind.ACTION_SUCCEEDED, witness="action_succeeded"),
        StructuredEvent(EventKind.CONSTRAINT_VIOLATED, witness="constraint_violation"),
        StructuredEvent(EventKind.FOCUS_MOVED, witness="observable_signature"),
    )
    assert primary_event_index(events) == EVENT_INDEX[EventKind.CONSTRAINT_VIOLATED]
