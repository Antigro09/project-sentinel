"""Scale 1A-0R-O2. Every objection this phase accepted, reproduced in code.

Phases O and O1 shipped no tests at all, so nothing stopped a claim from drifting between
the artifact and the prose -- which is exactly how "7 PASS" came to be printed over a
ledger holding eight. Each test below pins one measured correction.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/shwm"))
sys.path.insert(0, str(REPO / "src"))

o2_core = pytest.importorskip("o2_core")
ARTIFACTS = REPO / "artifacts/shwm/scale1"


@pytest.fixture(scope="module")
def collision_episodes():
    bijection = o2_core.sample_bijection(9_400)
    return bijection, o2_core.collect(range(110_000, 110_024), bijection,
                                      "COUNT_COLLISION", 9, seed=11, policy="uniform")


def test_token_fields_cover_the_width_exactly():
    covered = np.zeros(o2_core.TOKEN_WIDTH, dtype=int)
    for part in o2_core.FIELD_NAMES.values():
        covered[part] += 1
    assert (covered == 1).all(), f"token fields overlap or leave a gap: {covered}"


def test_moving_singleton_is_disambiguated_by_cardinality(collision_episodes):
    """before[a] == after[b] holds in BOTH directions when the agent steps from one
    empty cell to another. Reading the change set in scan order reversed the move on
    about half the steps and put the `entered` flag on the wrong colour."""
    bijection, episodes = collision_episodes
    registry = o2_core.ColourRegistry().scan(episodes)
    checked = 0
    for episode in episodes:
        tokens, _ = o2_core.episode_stream(episode, registry)
        for t in range(1, episode.length):
            role = int(episode.entered_role[t])
            if role < 0:
                continue
            checked += 1
            want = registry.of(o2_core.COLOUR_POOL[bijection[role]])
            entered = np.flatnonzero(tokens[t][:, o2_core.INTERACT][:, 0] > 0.5)
            assert list(entered) == [want], (
                f"step {t} of layout {episode.layout}: entered flag on {entered}, "
                f"true role {o2_core.ROLES[role]} is colour slot {want}")
    assert checked > 40


def test_the_reset_stripe_does_not_hide_the_move(collision_episodes):
    """The stripe repaints all twelve cells of row 0, so a pair straddling the reset
    frame has twelve changed cells. Requiring exactly two dropped every reset pair."""
    _, episodes = collision_episodes
    registry = o2_core.ColourRegistry().scan(episodes)
    found = 0
    for episode in episodes:
        cells = episode.cells
        if episode.entered_role[1] < 0:
            continue
        found += 1
        assert o2_core.moving_singleton(cells[0], cells[1]) is not None, (
            f"layout {episode.layout}: the reset pair's move was not detected")
    assert found > 0


def test_reset_flag_is_derived_from_pixels(collision_episodes):
    _, episodes = collision_episodes
    for episode in episodes:
        cells = episode.cells
        for t in range(episode.length):
            assert o2_core.is_reset_pair(cells[t]) == (t == 0)


def test_decoys_are_drawn_from_the_switch_generators_own_pool(collision_episodes):
    """Uniform placement left the ambiguous case almost unvisited; nearest-empty
    placement put decoys CLOSER to the start than switches and leaked through spatial
    moments. Both are measured in the module docstring; this pins the fix."""
    _, episodes = collision_episodes
    switch_distance, decoy_distance = [], []
    for episode in episodes:
        start = tuple(int(v) for v in episode.positions[0])
        grid = episode.roles[0]
        assert episode.decoy_cells == o2_core.SWITCH_COUNT
        assert int((grid == o2_core.SWITCH).sum()) >= o2_core.SWITCH_COUNT - 1
        for role, sink in ((o2_core.SWITCH, switch_distance),
                           (o2_core.DECOY, decoy_distance)):
            for cell in np.argwhere(grid == role):
                sink.append(abs(int(cell[0]) - start[0]) + abs(int(cell[1]) - start[1]))
    assert abs(np.mean(switch_distance) - np.mean(decoy_distance)) < 1.0, (
        f"switch mean distance {np.mean(switch_distance):.2f} vs decoy "
        f"{np.mean(decoy_distance):.2f}: the roles are not spatially exchangeable")


def test_the_contested_subset_is_actually_populated(collision_episodes):
    _, episodes = collision_episodes
    registry = o2_core.ColourRegistry().scan(episodes)
    data = o2_core.pair_dataset(episodes, registry)
    entered = data["meta"][:, 3]
    switches = int((entered == o2_core.SWITCH).sum())
    decoys = int((entered == o2_core.DECOY).sum())
    assert switches > 5 and decoys > 5, (switches, decoys)
    assert 0.3 < decoys / (switches + decoys) < 0.7, (
        f"{switches} switch entries against {decoys} decoy entries: the collision is "
        f"provable but unexercised, which is how a count-only lookup scored 0.975")


def test_the_truth_survives_its_own_evidence(collision_episodes):
    _, episodes = collision_episodes
    identity = tuple(range(o2_core.N_ROLES))
    assert identity in o2_core.survivors_over(episodes[:2])
    assert identity in o2_core.survivors_over([episodes[0]], steps=1)


def test_one_pair_is_ambiguous_and_one_calibration_episode_is_not():
    """The construction section G depends on: a lone transfer pair leaves the event
    class open, and a single calibration episode closes it."""
    bijection = o2_core.sample_bijection(9_400)
    calibration = o2_core.collect(range(116_000, 116_002), bijection, "COUNT_COLLISION",
                                  9, seed=71, policy="uniform")
    transfer = o2_core.collect(range(117_000, 117_002), bijection, "COUNT_COLLISION",
                               9, seed=162, policy="uniform")
    alone = o2_core.survivors_over([transfer[0]], steps=1)
    calibrated = o2_core.survivors_over(calibration[:1])
    assert o2_core.event_quotient_mass(alone) < 0.99, (
        "a single transfer pair must not identify the event class")
    assert o2_core.event_quotient_mass(calibrated) == 1.0, (
        "one calibration episode must pin AGENT and SWITCH")
    assert len(calibrated) < len(alone)


def test_goal_demonstration_retains_the_terminal_frame():
    """No ordinary episode has ever shown the goal marker occupied: the adapter
    terminates on arrival and collectors append the frame before stepping. Section K's
    demonstration is authored precisely to record it."""
    bijection = o2_core.sample_bijection(7_101)
    built = o2_core.goal_demonstration(115_000, bijection, "alpha")
    assert built is not None
    assert tuple(built["positions"][-1]) == built["target_cell"]
    assert built["terminal_frame_retained"]
    expected = tuple(int(v) for v in
                     o2_core.COLOUR_POOL[bijection[o2_core.GOAL_ALPHA]])
    assert tuple(built["named_marker_colour"]) == expected


def test_no_ordinary_episode_records_the_named_marker_occupied():
    import o_core as O

    episodes = O.collect_appearance(list(range(110_000, 110_012)),
                                    "HIDDEN_PALETTE_CONVENTION", [7_101], 1, 9, seed=11,
                                    policy="goal_directed")
    for episode in episodes:
        role = (O.ROLE_INDEX["GOAL_ALPHA"] if episode.goal_marker == "alpha"
                else O.ROLE_INDEX["GOAL_BETA"])
        where = np.argwhere(episode.roles[0] == role)
        if not len(where):
            continue
        cell = tuple(int(v) for v in where[0])
        assert not any(tuple(int(v) for v in p) == cell for p in episode.positions), (
            f"layout {episode.layout} recorded the named marker occupied; the "
            f"identifiability argument for section K assumes it never happens")


def test_the_binder_is_exactly_equivariant_in_the_colours(collision_episodes):
    import mlx.core as mx

    import o2_models as M

    _, episodes = collision_episodes
    registry = o2_core.ColourRegistry().scan(episodes)
    data = o2_core.pair_dataset(episodes, registry)
    model = M.build_stateless(7)
    order = np.random.default_rng(3).permutation(o2_core.MAX_COLOURS)
    remap = np.zeros(o2_core.MAX_COLOURS, dtype=np.int64)
    remap[order] = np.arange(o2_core.MAX_COLOURS)
    base = np.asarray(model(mx.array(data["tokens"]), mx.array(data["before_index"]),
                            mx.array(data["after_index"])))
    moved = np.asarray(model(mx.array(data["tokens"][:, order]),
                             mx.array(remap[data["before_index"]]),
                             mx.array(remap[data["after_index"]])))
    assert np.abs(base - moved).max() < 1e-4


def test_the_count_only_bayes_rule_is_at_chance_on_contested_collision_rows():
    """The calibration arm for the whole collision stratum. If this ever rises above
    chance the decoys stopped being exchangeable with the switches."""
    import o2_factorial as F

    bijection = o2_core.sample_bijection(9_400)
    train = o2_core.collect(range(110_000, 110_016), bijection, "COUNT_COLLISION", 9,
                            seed=11, policy="uniform")
    test = o2_core.collect(range(111_000, 111_016), o2_core.sample_bijection(9_401),
                           "COUNT_COLLISION", 9, seed=313, policy="uniform")
    registry = o2_core.ColourRegistry().scan(train + test)
    block = F.count_only_bayes(o2_core.pair_dataset(train, registry),
                               o2_core.pair_dataset(test, registry))
    assert abs(block["contested_balanced_accuracy"] - 0.5) < 0.03, block


def test_guard_a_passes_the_honest_generator_and_catches_the_plant():
    import o2_leakage as L

    honest = L.guard_a(range(9_500, 9_700), L.UNSEEN_PALETTES)
    planted = L.guard_a(range(9_500, 9_700), L.UNSEEN_PALETTES, plant=True)
    assert not honest["leak_detected"], honest
    assert planted["leak_detected"], planted
    assert planted["balanced_accuracy"] > honest["balanced_accuracy"] + 0.2


def test_guard_b_pools_the_global_block_instead_of_flattening_slots():
    """Flattening the slots leaves OCCUPANCY in the feature, and occupancy is which
    colours are on screen -- a palette fingerprint that is public by construction. That
    version scored 0.3509 on an honest generator."""
    import o2_leakage as L

    honest = L.build("identity", L.DEV_PALETTES[:4], L.TRAIN_LAYOUTS[:8], 11)
    registry = o2_core.ColourRegistry().scan(honest)
    data = o2_core.pair_dataset(honest, registry)
    label = data["meta"][:, 1]
    pooled = L.guard_b(data["tokens"], label, "global_only")
    planted = L.guard_b(data["tokens"], label, "global_only", plant=True)
    assert not pooled["leak_detected"], pooled
    assert planted["leak_detected"], planted


def test_the_o1_gate_tally_in_the_artifact_is_eight_not_seven():
    """The O1 report's headline said 7 PASS / 3 PARTIAL / 4 NOT_RUN / 1 FAIL. Its own
    table and its own artifact both say 8 / 3 / 3 / 1. This pins the correction."""
    path = ARTIFACTS / "p-gates.json"
    if not path.exists():
        pytest.skip("p-gates.json has not been produced in this checkout")
    gates = json.loads(path.read_text())["p_gates"]
    counts: dict[str, int] = {}
    for entry in gates.values():
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    assert counts == {"PASS": 8, "PARTIAL": 3, "NOT_RUN": 3, "FAIL": 1}, counts
    assert sum(counts.values()) == 15


def test_the_two_class_size_means_come_from_different_populations():
    """2.08 and 2.468 were both written as "the calibrated class size". They are the
    means of two different populations and this reproduces both from one function."""
    path = ARTIFACTS / "o2-equivalence.json"
    if not path.exists():
        pytest.skip("o2-equivalence.json has not been produced in this checkout")
    block = json.loads(path.read_text())["reconciliation"]
    assert abs(block["O_population_recomputed_mean"] - 50 / 24) < 1e-9
    assert abs(block["O1_population_recomputed_mean"] - 116 / 47) < 1e-9
    assert block["consistent"]
