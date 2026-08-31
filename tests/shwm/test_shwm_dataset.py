"""Scale-0 gate: split order, branch groups, and taint.

Two orderings are the whole content of this file. Splits are assigned to episode
keys *before* collection, and branch siblings inherit the episode's split rather
than being divided afterwards -- because one sibling in train tells the model
exactly what the held-out sibling does, and the intervention audit would then be
measuring memorisation.
"""

from __future__ import annotations

import pytest

from sentinel.wm.dataset import (
    COLLECTION_MIXTURE,
    CollectorPolicy,
    EpisodeKey,
    LeakageError,
    SequenceBatch,
    assert_no_transition_overlap,
    Split,
    SplitManifest,
    TransitionRecord,
    assert_trainable,
    audit_splits,
    branch_coverage_report,
    mixture_report,
)
from sentinel.wm.events import EventKind, StructuredEvent
from sentinel.wm.latent_contract import ContractViolation, Taint
from sentinel.wm.versioning import digest_of


def key(seed: int = 0, family: str = "controlled", dynamic: str = "base") -> EpisodeKey:
    return EpisodeKey(family=family, seed=seed, dynamic=dynamic)


def record(
    *,
    episode_key: EpisodeKey | None = None,
    episode_id: str = "ep-0",
    step: int = 0,
    action: int = 0,
    branch_group_id: str | None = None,
    taint: frozenset[Taint] = frozenset({Taint.DEVELOPMENT}),
    obs_t: str | None = None,
    obs_t1: str | None = None,
    content_t: str | None = None,
    content_t1: str | None = None,
    policy: CollectorPolicy = CollectorPolicy.RANDOM,
    propensity: float = 0.25,
    extra: dict | None = None,
) -> TransitionRecord:
    return TransitionRecord(
        episode_key=episode_key or key(),
        episode_id=episode_id,
        step=step,
        observation_digest_t=obs_t or digest_of(f"{episode_id}:{step}"),
        content_digest_t=content_t or obs_t or digest_of(f"content:{episode_id}:{step}"),
        latent_digest_t=digest_of(f"latent:{episode_id}:{step}"),
        action=action,
        action_propensity=propensity,
        collector_policy=policy,
        collector_policy_digest=digest_of(policy.value),
        reward=0.0,
        termination=False,
        observation_digest_t1=obs_t1 or digest_of(f"{episode_id}:{step + 1}:{action}"),
        content_digest_t1=content_t1 or obs_t1 or digest_of(f"content:{episode_id}:{step + 1}:{action}"),
        latent_digest_t1=digest_of(f"latent:{episode_id}:{step + 1}:{action}"),
        structured_events=(StructuredEvent(EventKind.ACTION_SUCCEEDED, witness="action_succeeded"),),
        branch_group_id=branch_group_id,
        taint=taint,
        extra=extra or {},
    )


def manifest(**overrides) -> SplitManifest:
    base = dict(salt="scale-0", weights={Split.TRAIN: 0.8, Split.DEV_HELD_OUT: 0.2})
    base.update(overrides)
    return SplitManifest(**base)


# ---- split assignment happens first -----------------------------------------


def test_assignment_is_a_function_of_the_key_not_of_collection_order():
    a, b = manifest(), manifest()
    keys = [key(i) for i in range(50)]
    forward = [a.assign(k) for k in keys]
    backward = [b.assign(k) for k in reversed(keys)][::-1]
    assert forward == backward


def test_assignment_is_stable_and_never_reassigned():
    m = manifest()
    first = m.assign(key(7))
    assert m.assign(key(7)) is first
    assert m.split_of(key(7)) is first


def test_a_sealed_manifest_refuses_a_key_that_was_not_assigned_before_collection():
    m = manifest()
    m.assign(key(1))
    m.seal()
    with pytest.raises(LeakageError):
        m.assign(key(2))


def test_an_unassigned_key_cannot_be_inferred_from_collected_data():
    with pytest.raises(LeakageError):
        manifest().split_of(key(3))


def test_split_weights_must_be_a_distribution():
    with pytest.raises(ContractViolation):
        SplitManifest(salt="s", weights={Split.TRAIN: 0.9, Split.DEV_HELD_OUT: 0.2})


def test_the_manifest_digest_pins_the_assignment():
    m = manifest()
    m.assign(key(1))
    before = m.digest
    m.assign(key(2))
    assert m.digest != before


# ---- branch groups cannot cross splits ---------------------------------------


def build_branch_group(m: SplitManifest, seed: int, actions=(0, 1, 2)):
    episode_key = key(seed)
    m.assign(episode_key)
    group = digest_of({"episode": seed, "restore_point": 4})
    return [
        record(
            episode_key=episode_key,
            episode_id=f"ep-{seed}",
            step=4,
            action=a,
            branch_group_id=group,
            obs_t=digest_of(f"restored:{seed}:{a}"),
            content_t=digest_of(f"restored-content:{seed}"),
            obs_t1=digest_of(f"successor:{seed}:{a}"),
            content_t1=digest_of(f"successor-content:{seed}:{a}"),
        )
        for a in actions
    ]


def test_branch_siblings_collected_inside_one_assigned_split_pass_the_audit():
    m = manifest()
    records = build_branch_group(m, 1) + build_branch_group(m, 2)
    report = audit_splits(records, m)
    assert report["branch_groups"] == 2
    assert report["transitions"] == len(records)
    assert report["observation_contents_in_multiple_splits"] == 0


def test_a_branch_group_split_after_collection_is_caught():
    m = manifest()
    records = build_branch_group(m, 1)
    # Force the siblings apart the way a post-hoc split would.
    stolen = key(9999)
    m.assignments[stolen.digest] = (
        Split.DEV_HELD_OUT if m.split_of(records[0].episode_key) is Split.TRAIN else Split.TRAIN
    )
    records[-1] = record(
        episode_key=stolen,
        episode_id=records[-1].episode_id,
        step=4,
        action=records[-1].action,
        branch_group_id=records[-1].branch_group_id,
        obs_t=records[-1].observation_digest_t,
        obs_t1=records[-1].observation_digest_t1,
    )
    with pytest.raises(LeakageError, match="branch group"):
        audit_splits(records, m)


def test_a_branch_group_spanning_two_episodes_is_caught():
    m = manifest()
    a_key, b_key = key(1), key(2)
    m.assign(a_key)
    m.assignments[b_key.digest] = m.split_of(a_key)
    group = digest_of("shared-group")
    records = [
        record(episode_key=a_key, episode_id="ep-a", branch_group_id=group,
               obs_t=digest_of("x"), content_t=digest_of("cx")),
        record(episode_key=b_key, episode_id="ep-b", branch_group_id=group,
               obs_t=digest_of("y"), content_t=digest_of("cy")),
    ]
    with pytest.raises(LeakageError, match="more than one episode"):
        audit_splits(records, m)


def two_split_keys(m: SplitManifest):
    train_key = next(k for k in (key(i) for i in range(200)) if m.assign(k) is Split.TRAIN)
    held_key = next(
        k for k in (key(i) for i in range(200, 600)) if m.assign(k) is Split.DEV_HELD_OUT
    )
    return train_key, held_key


def test_a_complete_transition_tuple_in_two_splits_is_measured_and_can_be_forbidden():
    """The concrete form of "the held-out answer is in the training set".

    Measured always; fatal for a family that can actually be disjoint.
    """
    m = manifest()
    train_key, held_key = two_split_keys(m)
    before, after = digest_of("state-before"), digest_of("state-after")
    records = [
        record(episode_key=train_key, episode_id="ep-train", action=2,
               content_t=before, content_t1=after),
        record(episode_key=held_key, episode_id="ep-held", action=2,
               content_t=before, content_t1=after),
    ]
    report = audit_splits(records, m)
    assert report["transition_tuples_in_multiple_splits"] == 1
    with pytest.raises(LeakageError, match="generative"):
        assert_no_transition_overlap(report, "procedural_visual")


def test_a_disjoint_generative_family_passes_the_stricter_gate():
    m = manifest()
    train_key, held_key = two_split_keys(m)
    records = [
        record(episode_key=train_key, episode_id="ep-train", action=2,
               content_t=digest_of("a"), content_t1=digest_of("b")),
        record(episode_key=held_key, episode_id="ep-held", action=2,
               content_t=digest_of("c"), content_t1=digest_of("d")),
    ]
    report = audit_splits(records, m)
    assert report["transition_tuples_in_multiple_splits"] == 0
    assert_no_transition_overlap(report, "procedural_visual")


def test_the_same_episode_step_collected_into_two_splits_is_caught():
    m = manifest()
    train_key, held_key = two_split_keys(m)
    shared = digest_of("the-same-episode-step")
    records = [
        record(episode_key=train_key, episode_id="ep-train", obs_t=shared,
               content_t=digest_of("c1"), content_t1=digest_of("c2")),
        record(episode_key=held_key, episode_id="ep-held", obs_t=shared,
               content_t=digest_of("c3"), content_t1=digest_of("c4")),
    ]
    with pytest.raises(LeakageError, match="positional observation"):
        audit_splits(records, m)


def test_a_shared_observation_with_different_successors_is_reported_not_raised():
    """A finite observation space makes some overlap arithmetic, not leakage.

    The controlled adapter has eight visible states and many more episodes, so
    demanding disjoint observations would be a check that can only be satisfied
    by making the environment bigger. The overlap is measured instead.
    """
    m = manifest()
    train_key, held_key = two_split_keys(m)
    shared = digest_of("a-state-both-splits-can-reach")
    records = [
        record(episode_key=train_key, episode_id="ep-train", action=0,
               content_t=shared, content_t1=digest_of("successor-a")),
        record(episode_key=held_key, episode_id="ep-held", action=1,
               content_t=shared, content_t1=digest_of("successor-b")),
    ]
    report = audit_splits(records, m)
    assert report["observation_contents_in_multiple_splits"] == 1
    assert 0.0 < report["observation_content_overlap_rate"] <= 1.0


def test_an_environment_seed_reused_across_splits_is_caught():
    m = manifest()
    a = EpisodeKey("controlled", 5, "base")
    b = EpisodeKey("controlled", 5, "mechanic-swap")
    m.assign(a)
    m.assignments[b.digest] = (
        Split.DEV_HELD_OUT if m.split_of(a) is Split.TRAIN else Split.TRAIN
    )
    records = [
        record(episode_key=a, episode_id="ep-a", obs_t=digest_of("a"), content_t=digest_of("ca")),
        record(episode_key=b, episode_id="ep-b", obs_t=digest_of("b"), content_t=digest_of("cb")),
    ]
    with pytest.raises(LeakageError, match="seed"):
        audit_splits(records, m)


# ---- taint and evaluator fields ---------------------------------------------


@pytest.mark.parametrize(
    "taint", [Taint.FINAL, Taint.VALIDATION, Taint.ORACLE, Taint.EVALUATOR_ONLY]
)
def test_a_forbidden_taint_cannot_enter_a_training_step(taint):
    with pytest.raises(LeakageError):
        assert_trainable([record(taint=frozenset({Taint.DEVELOPMENT, taint}))])


def test_development_records_are_trainable():
    assert_trainable([record() for _ in range(3)])


@pytest.mark.parametrize(
    "field",
    ["hidden_state", "simulator_state", "target_program", "expected_observation", "evaluator_answer"],
)
def test_evaluator_only_fields_cannot_ride_along_in_a_transition(field):
    with pytest.raises(ContractViolation):
        record(extra={field: 1})


def test_a_zero_propensity_action_is_rejected():
    with pytest.raises(ContractViolation):
        record(propensity=0.0)
    with pytest.raises(ContractViolation):
        record(propensity=1.5)


# ---- sequences ---------------------------------------------------------------


def test_a_sequence_may_not_span_episodes_or_run_backwards():
    good = SequenceBatch(
        records=tuple(record(episode_id="ep", step=i) for i in range(4)), split=Split.TRAIN
    )
    assert len(good) == 4
    with pytest.raises(ContractViolation):
        SequenceBatch(
            records=(record(episode_id="a", step=0), record(episode_id="b", step=1)),
            split=Split.TRAIN,
        )
    with pytest.raises(ContractViolation):
        SequenceBatch(
            records=(record(episode_id="a", step=3), record(episode_id="a", step=1)),
            split=Split.TRAIN,
        )
    with pytest.raises(ContractViolation):
        SequenceBatch(records=(), split=Split.TRAIN)


# ---- reports -----------------------------------------------------------------


def test_mixture_report_names_the_target_share_next_to_the_realised_one():
    records = (
        [record(policy=CollectorPolicy.RANDOM) for _ in range(30)]
        + [record(policy=CollectorPolicy.SCRIPTED_ORACLE) for _ in range(25)]
        + [record(policy=CollectorPolicy.SENTINEL) for _ in range(25)]
        + [record(policy=CollectorPolicy.UNCERTAINTY_SEEKING) for _ in range(20)]
    )
    report = mixture_report(records)
    for policy, target in COLLECTION_MIXTURE.items():
        assert report[policy.value]["share"] == pytest.approx(target)
        assert report[policy.value]["target_share"] == target


def test_branch_coverage_separates_real_branches_from_singletons():
    m = manifest()
    records = build_branch_group(m, 1, actions=(0, 1, 2)) + build_branch_group(m, 2, actions=(0,))
    report = branch_coverage_report(records)
    assert report["branch_groups"] == 2
    assert report["groups_with_multiple_actions"] == 1


def test_transition_id_is_stable_and_action_sensitive():
    a = record(action=0)
    b = record(action=1)
    assert a.transition_id == record(action=0).transition_id
    assert a.transition_id != b.transition_id
