"""Scale-1A-0 D: the stratified split, and the leaks it must catch.

Scale 0's audit checked whether a branch *sibling* crossed a split. This checks
whether a *descendant* did, which is the case a clone makes possible and the
earlier audit could not express: continue from one branch of a cloned state and
the resulting states carry that clone's lineage even though they are not
siblings of anything.

Each clause is planted before it is trusted.
"""

from __future__ import annotations

import pytest

from sentinel.wm.dataset import LeakageError
from sentinel.wm.latent_contract import ContractViolation
from sentinel.wm.splits_v2 import (
    CANONICAL_APPEARANCE_SEED,
    EpisodeDescriptor,
    Lineage,
    Stratum,
    StratifiedRecord,
    StratifiedSplitManifest,
    assert_primary_stratum_is_disjoint,
    audit_v2_splits,
)
from sentinel.wm.versioning import digest_of


def manifest(**overrides) -> StratifiedSplitManifest:
    base = dict(
        train_layouts=frozenset(range(100, 200)),
        held_layouts=frozenset(range(200, 240)),
        train_appearances=frozenset({CANONICAL_APPEARANCE_SEED}),
        held_appearances=frozenset(range(500, 540)),
    )
    base.update(overrides)
    return StratifiedSplitManifest(**base)


def descriptor(layout=150, appearance=CANONICAL_APPEARANCE_SEED, goal=1, phase=1):
    return EpisodeDescriptor("procedural_visual_v2", layout, appearance, phase, goal)


def record(desc, lineage, action=0, before="a", after="b", trajectory="t"):
    return StratifiedRecord(
        descriptor=desc,
        lineage=lineage,
        content_digest_t=digest_of(before),
        action=action,
        content_digest_t1=digest_of(after),
        trajectory_id=trajectory,
    )


# ---- stratum assignment -----------------------------------------------------------


def test_the_four_strata_are_assigned_from_the_descriptor():
    m = manifest()
    assert m.assign(descriptor(layout=210)) is Stratum.DYNAMICS_CLEAN
    assert m.assign(descriptor(layout=150, appearance=505)) is Stratum.APPEARANCE_SHIFT
    assert m.assign(descriptor(layout=210, appearance=505)) is Stratum.CROSSED_SHIFT
    assert (
        m.assign(EpisodeDescriptor("procedural_visual", 1, 1, 1, 1))
        is Stratum.LEGACY_V1_REPLICATION
    )


def test_appearance_is_a_constant_inside_the_attribution_stratum():
    """dynamics_clean must pin appearance, or a result there is confounded by it."""
    m = manifest()
    for layout in (201, 205, 239):
        assert m.assign(descriptor(layout=layout)) is Stratum.DYNAMICS_CLEAN
    with pytest.raises(ContractViolation, match="neither the canonical palette"):
        m.assign(descriptor(layout=210, appearance=999))


def test_holding_out_the_canonical_appearance_is_refused():
    """Otherwise dynamics_clean is also an appearance shift and the two strata
    cannot be told apart, which is the entire point of separating them."""
    with pytest.raises(ContractViolation, match="canonical appearance is held out"):
        manifest(held_appearances=frozenset({CANONICAL_APPEARANCE_SEED}))


def test_a_seed_in_both_train_and_held_out_is_refused():
    with pytest.raises(ContractViolation, match="layout seeds appear in both"):
        manifest(train_layouts=frozenset({1, 2}), held_layouts=frozenset({2, 3}))
    with pytest.raises(ContractViolation, match="appearance seeds appear in both"):
        manifest(
            train_appearances=frozenset({CANONICAL_APPEARANCE_SEED, 7}),
            held_appearances=frozenset({7}),
        )


def test_a_sealed_manifest_refuses_a_descriptor_it_never_saw():
    m = manifest()
    m.assign(descriptor())
    m.seal()
    with pytest.raises(LeakageError):
        m.assign(descriptor(layout=151))


# ---- lineage ----------------------------------------------------------------------


def test_branches_of_one_clone_share_a_lineage_and_descendants_keep_it():
    root = Lineage(root="episode-1")
    cloned = root.clone_at("restore@4")
    branch_a, branch_b = cloned, cloned
    assert branch_a.lineage_hash == branch_b.lineage_hash
    descendant = branch_a.descend().descend()
    assert descendant.lineage_hash == cloned.lineage_hash
    assert descendant.depth == 2
    assert root.lineage_hash != cloned.lineage_hash


def test_a_descendant_of_a_clone_crossing_a_stratum_is_caught():
    """The case Scale 0's sibling check could not express."""
    m = manifest()
    clean_descriptor = descriptor(layout=210)
    shifted_descriptor = descriptor(layout=210, appearance=505)
    m.assign(clean_descriptor)
    m.assign(shifted_descriptor)
    cloned = Lineage(root="episode-1").clone_at("restore@4")
    records = [
        record(clean_descriptor, cloned, before="s0", after="s1", trajectory="ta"),
        # not a sibling: two steps further on, still carrying the clone's lineage
        record(shifted_descriptor, cloned.descend().descend(), before="s2", after="s3",
               trajectory="tb"),
    ]
    with pytest.raises(LeakageError, match="lineage"):
        audit_v2_splits(records, m)


def test_a_trajectory_split_across_strata_is_caught():
    m = manifest()
    a, b = descriptor(layout=210), descriptor(layout=210, appearance=505)
    m.assign(a)
    m.assign(b)
    records = [
        record(a, Lineage(root="e1"), before="s0", after="s1", trajectory="shared"),
        record(b, Lineage(root="e2"), before="s2", after="s3", trajectory="shared"),
    ]
    with pytest.raises(LeakageError, match="trajectory"):
        audit_v2_splits(records, m)


def test_a_clean_stratified_set_passes_and_reports_its_counts():
    m = manifest()
    records = []
    for index, layout in enumerate(range(200, 208)):
        desc = descriptor(layout=layout)
        m.assign(desc)
        records.append(
            record(desc, Lineage(root=f"e{index}"), before=f"a{index}", after=f"b{index}",
                   trajectory=f"t{index}")
        )
    for index, appearance in enumerate(range(500, 506)):
        desc = descriptor(layout=150 + index, appearance=appearance)
        m.assign(desc)
        records.append(
            record(desc, Lineage(root=f"f{index}"), before=f"c{index}", after=f"d{index}",
                   trajectory=f"u{index}")
        )
    report = audit_v2_splits(records, m)
    assert report["per_stratum"]["dynamics_clean"] == 8
    assert report["per_stratum"]["appearance_shift"] == 6
    assert report["transition_tuples_in_multiple_strata"] == 0
    assert_primary_stratum_is_disjoint(report)


def test_a_shared_transition_tuple_fails_the_primary_disjointness_gate():
    """v1's controlled family could not be disjoint by arithmetic. v2's primary
    stratum can be, so here it is required rather than measured."""
    m = manifest()
    a, b = descriptor(layout=210), descriptor(layout=211)
    m.assign(a)
    m.assign(b)
    records = [
        record(a, Lineage(root="e1"), before="same", after="also-same", trajectory="t1"),
        record(b, Lineage(root="e2"), before="same", after="also-same", trajectory="t2"),
    ]
    report = audit_v2_splits(records, m)
    assert report["transition_tuples_in_multiple_strata"] == 0  # same stratum, so not crossing

    m2 = manifest()
    c = descriptor(layout=212, appearance=505)
    m2.assign(a)
    m2.assign(c)
    crossing = [
        record(a, Lineage(root="e1"), before="same", after="also-same", trajectory="t1"),
        record(c, Lineage(root="e2"), before="same", after="also-same", trajectory="t2"),
    ]
    crossing_report = audit_v2_splits(crossing, m2)
    assert crossing_report["transition_tuples_in_multiple_strata"] == 1
    with pytest.raises(LeakageError, match="must be disjoint"):
        assert_primary_stratum_is_disjoint(crossing_report)


def test_clean_and_crossed_share_held_out_layouts_by_design():
    """crossed_shift is the same held-out layouts under a shifted appearance, so
    the shared layout is the design rather than a leak. What is a leak is a
    *trained* layout reaching a held-out stratum, checked below."""
    m = manifest()
    clean = descriptor(layout=210)
    crossed = descriptor(layout=210, appearance=505)
    m.assign(clean)
    m.assign(crossed)
    records = [
        record(clean, Lineage(root="e1"), before="a", after="b", trajectory="t1"),
        record(crossed, Lineage(root="e2"), before="c", after="d", trajectory="t2"),
    ]
    report = audit_v2_splits(records, m)
    assert report["layouts_shared_between_clean_and_crossed"] == 1
    assert report["distinct_layouts"] == 1


def test_a_trained_layout_reaching_a_held_out_stratum_is_caught():
    """appearance_shift uses trained layouts; dynamics_clean uses held-out ones.
    A layout in both means the model trained on what it is being scored against."""
    m = manifest()
    trained = descriptor(layout=150, appearance=505)          # appearance_shift
    m.assign(trained)
    # Force the same layout into a held-out-layout stratum.
    leaked = descriptor(layout=150, appearance=CANONICAL_APPEARANCE_SEED)
    m.assignments[leaked.episode_hash] = Stratum.DYNAMICS_CLEAN
    m.descriptors[leaked.episode_hash] = leaked.canonical_dict()
    records = [
        record(trained, Lineage(root="e1"), before="a", after="b", trajectory="t1"),
        record(leaked, Lineage(root="e2"), before="c", after="d", trajectory="t2"),
    ]
    with pytest.raises(LeakageError, match="trained stratum and a"):
        audit_v2_splits(records, m)


def test_the_manifest_digest_pins_the_assignment():
    m = manifest()
    before = m.digest
    m.assign(descriptor(layout=210))
    assert m.digest != before


def test_the_legacy_split_is_never_the_primary_stratum():
    from sentinel.wm.splits_v2 import NON_PRIMARY, PRIMARY_STRATUM

    assert PRIMARY_STRATUM is Stratum.DYNAMICS_CLEAN
    assert Stratum.LEGACY_V1_REPLICATION in NON_PRIMARY
    assert Stratum.APPEARANCE_SHIFT in NON_PRIMARY
