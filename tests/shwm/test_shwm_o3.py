"""Scale 1A-0R-O3. Every correction this phase made, pinned in code.

O2 shipped the first tests for this track. O3 found five more defects that no test would
have caught, and each one below is the test that would have caught it. Three of them are
mistakes this phase made and then measured its way out of, so they are pinned in the
direction of the correction, not the original claim.
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

C = pytest.importorskip("o2_core")
M = pytest.importorskip("o2_models")
ch = pytest.importorskip("o3_change")
pop = pytest.importorskip("o3_population")
unc = pytest.importorskip("o3_uncertainty")
ARTIFACTS = REPO / "artifacts/shwm/scale1"


@pytest.fixture(scope="module")
def persistent_episode():
    plan = pop.palette_plan(20_000, 2, 2, 1)
    scenario = pop.palette_scenario(plan)
    episode = scenario.calibration[0]
    cells, _ = unc.render_regime(episode, plan["bijection"], "PERSISTENT_CONVENTION", 7)
    return plan, episode, cells


# ---- the view, which is what O3's central finding turned on ---------------------------


def test_no_rgb_is_equivariant_and_full_token_is_not():
    """O2 trained its memory on `full_token`, which carries the raw colour block, while
    O2's own factorial had selected a view without it. That is what made Q7 look like a
    perception limit.

    The property is EQUIVARIANCE, not invariance, and the distinction matters. Tokens are
    indexed by colour SLOT, so relabelling the palette moves each role's token to a
    different row; what must hold is that the multiset of rows is unchanged. Section B's
    0.00e+00 was measured on outputs already mapped back into semantic space, which is
    the same statement one level down. Asserting bitwise equality here fails on a
    correct pipeline, and this test was written that way first."""
    plan = pop.palette_plan(20_001, 2, 2, 1)
    scenario = pop.palette_scenario(plan)
    episode = scenario.calibration[0]
    registry = C.canonical_registry()
    other = C.sample_bijection(20_999)

    def tokens(bijection, view):
        cells, _ = unc.render_regime(episode, bijection, "PERSISTENT_CONVENTION", 3)
        stack = np.stack([C.pair_tokens(cells[t - 1], cells[t],
                                        int(episode.actions[t - 1]), registry)
                          for t in range(1, episode.length)])
        return M.mask_view(stack, view)

    def rows_up_to_slot_permutation(block):
        flat = block.reshape(block.shape[0], block.shape[1], -1)
        return np.sort(flat, axis=1)

    for view, equivariant in (("full_token", False), ("no_rgb", True)):
        a, b = tokens(plan["bijection"], view), tokens(other, view)
        assert not np.array_equal(a, b), f"{view} cannot be bitwise invariant"
        same = np.array_equal(rows_up_to_slot_permutation(a),
                              rows_up_to_slot_permutation(b))
        assert same is equivariant, (
            f"{view}: equal-up-to-slot-permutation was {same}, expected {equivariant}")


# ---- the change detector ---------------------------------------------------------------


def test_the_colour_set_alone_does_not_identify_a_palette():
    """The refuted v1 signature. A palette permutes ONE eight-colour pool, so the set of
    colours on screen is nearly invariant across palettes and varies with content."""
    a, b = C.sample_bijection(20_010), C.sample_bijection(20_011)
    shared = set(int(v) for v in a) & set(int(v) for v in b)
    assert len(shared) >= 6, (
        f"two palettes share {len(shared)} of 7 colours; the set cannot separate them")


def test_the_anchors_are_single_valued_and_true_under_a_persistent_convention(
        persistent_episode):
    plan, episode, cells = persistent_episode
    registry = C.canonical_registry()
    block = ch.episode_signature(cells, episode, registry)
    truth = ch._true_signature(plan["bijection"])
    assert not block["inconsistent"], block["counts"]
    for anchor, value in block["signature"].items():
        if value is not None:
            assert value == truth[anchor], f"{anchor} derived {value}, truth {truth}"


def test_field_excludes_the_border_so_a_dense_layout_cannot_flip_it():
    """The plain modal colour flips from EMPTY to WALL on a dense layout -- measured on
    palette 20002, where it took two values inside one palette."""
    plan = pop.palette_plan(20_002, 4, 2, 1)
    scenario = pop.palette_scenario(plan)
    registry = C.canonical_registry()
    seen = set()
    for index, episode in enumerate(scenario.calibration):
        cells, _ = unc.render_regime(episode, plan["bijection"],
                                     "PERSISTENT_CONVENTION", 7 * index)
        block = ch.episode_signature(cells, episode, registry)
        if block["signature"]["FIELD"] is not None:
            seen.add(block["signature"]["FIELD"])
    assert len(seen) == 1, f"FIELD took {len(seen)} values inside one palette: {seen}"


@pytest.mark.parametrize("regime,expect_support",
                         [("PER_CELL_NOISE", 0.0), ("PER_FRAME_BIJECTION", None)])
def test_both_uninformative_regimes_are_caught_by_consistency_not_magnitude(
        regime, expect_support):
    """The refuted v2 signal. `support <= 0` catches PER_CELL_NOISE and misses
    PER_FRAME_BIJECTION, which carries NONZERO support and still exhibits no map."""
    plan = pop.palette_plan(20_003, 2, 2, 1)
    scenario = pop.palette_scenario(plan)
    registry = C.canonical_registry()
    episode = scenario.calibration[0]
    cells, _ = unc.render_regime(episode, plan["bijection"], regime, 11)
    block = ch.episode_signature(cells, episode, registry)
    memory = ch.PaletteMemory()
    assert memory.observe(block, 0) == "MISSING_APPEARANCE"
    assert block["inconsistent"], "the regime must contradict itself inside the episode"
    if expect_support == 0.0:
        assert block["support"] == 0.0
    else:
        assert block["support"] > 0.0, (
            "PER_FRAME_BIJECTION must carry nonzero support, or it does not test the "
            "magnitude rule at all")


def test_a_quiet_honest_episode_is_not_mistaken_for_a_missing_appearance():
    """The control the magnitude rule would have failed: few interactions, honest
    convention."""
    plan = pop.palette_plan(20_004, 2, 2, 1)
    scenario = pop.palette_scenario(plan)
    registry = C.canonical_registry()
    full, _ = unc.render_regime(scenario.calibration[0], plan["bijection"],
                                "PERSISTENT_CONVENTION", 5)
    episode = ch._truncate(scenario.calibration[0], 3)
    block = ch.episode_signature(full[:3], episode, registry)
    assert ch.PaletteMemory().observe(block, 0) == "SAME_PALETTE"


def test_confirmed_memory_is_not_overwritten_before_promotion():
    memory = ch.PaletteMemory(promote_after=3, min_components=2)
    first = {"BORDER": (1, 1, 1), "FIELD": (2, 2, 2), "MOVER": (3, 3, 3)}
    second = {"BORDER": (9, 9, 9), "FIELD": (8, 8, 8), "MOVER": (7, 7, 7)}
    block = lambda s: {"signature": s, "inconsistent": [],       # noqa: E731
                       "derivable": list(s), "counts": {k: 1 for k in s},
                       "support": 1.0, "pairs": 1}
    assert memory.observe(block(first), 0) == "SAME_PALETTE"
    for step in range(1, 3):
        assert memory.observe(block(second), step) == "NEW_PALETTE"
        assert memory.confirmed == first, "confirmed changed before promotion"
        assert memory.provisional_open
    assert memory.observe(block(second), 3) == "NEW_PALETTE"
    assert memory.confirmed == second and memory.promotions == 1
    assert not memory.provisional_open


def test_the_drift_stays_a_bijection_at_every_stage():
    """Assigning a role its target colour outright lets two roles collide, which makes
    `moving_singleton` return None and turns a contradiction into a silent miss."""
    base, other = C.sample_bijection(20_020), C.sample_bijection(20_021)
    for stage in range(len(ch.DRIFT_REACHING_ANCHOR) + 1):
        moved = ch.drift_bijection(base, other, ch.DRIFT_REACHING_ANCHOR, stage)
        assert len(set(int(v) for v in moved)) == len(moved), (
            f"stage {stage} produced a non-injective palette: {moved}")


def test_the_non_anchor_transposition_is_invisible_to_the_signature():
    """The measured boundary of what section I closes, stated as a fact about the
    construction rather than as an empirical hope."""
    base = C.sample_bijection(20_030)
    swapped = np.array(base, copy=True)
    swapped[[C.SWITCH, C.DECOY]] = swapped[[C.DECOY, C.SWITCH]]
    assert ch._true_signature(base) == ch._true_signature(swapped)


# ---- the statistic ----------------------------------------------------------------------


def test_contested_rows_are_not_class_balanced_so_plain_accuracy_is_not_chance_at_half():
    """Each palette drawing its own content makes the SWITCH-against-DECOY base rate a
    per-palette quantity. Plain accuracy then rewards a constant answer."""
    persistent = pytest.importorskip("o3_persistent")
    mem = pytest.importorskip("o2_memory")
    registry = C.canonical_registry()
    groups = [persistent.group_of(pop.palette_plan(p, 4, 4, 1))
              for p in pop.VALIDATION_PALETTES[:6]]
    data = mem.stack_groups(groups, registry, "no_rgb")
    mask = mem.contested(data)
    rates = [float(data["event"][mask & (data["group"] == g.palette)].mean())
             for g in groups]
    assert max(rates) - min(rates) > 0.15, (
        f"base rates {rates} are too uniform for this test to be meaningful")

    truth = data["event"][mask].astype(float)
    constant = (truth == 1.0).astype(float)
    assert abs(persistent.balanced(constant, truth) - 0.5) < 1e-9, (
        "a constant answer must score exactly 0.5 balanced")
    assert constant.mean() > 0.5, (
        "and strictly more than 0.5 on plain accuracy, which is the trap")


def test_the_palette_bootstrap_is_wider_than_the_row_bootstrap():
    """Rows inside a palette are not independent replicates of the palette."""
    persistent = pytest.importorskip("o3_persistent")
    rng = np.random.default_rng(4)
    palettes = list(range(12))
    values, correct, truth = {}, [], []
    for index, palette in enumerate(palettes):
        # A per-palette skill level, which is exactly the structure row resampling hides.
        skill = 0.3 + 0.05 * index
        rows = (rng.random(80) < skill).astype(float)
        labels = np.array([0.0, 1.0] * 40)
        values[palette] = (rows, labels)
        correct.append(rows)
        truth.append(labels)
    _, low, high = persistent.palette_bootstrap(values, palettes, rng, 400)
    _, r_low, r_high = persistent.row_bootstrap(np.concatenate(correct),
                                                np.concatenate(truth), rng, 400)
    assert (high - low) > (r_high - r_low), (
        f"palette interval {high - low:.4f} is not wider than row {r_high - r_low:.4f}")


# ---- provenance -------------------------------------------------------------------------


def test_the_route_calibration_layouts_are_disjoint_from_the_memory_training_set():
    """The leak that produced route parity 1.0000 on every palette and masked the
    defect entirely."""
    ro = pytest.importorskip("o3_route_orbit")
    mem = pytest.importorskip("o2_memory")
    assert not (set(ro.ROUTE_CAL_LAYOUTS) & set(mem.CAL_LAYOUTS))


def test_the_language_replication_draws_only_fresh_ids():
    language = pytest.importorskip("o3_language")
    trace = language.provenance()
    assert trace["all_fresh"], {
        k: v for k, v in {**trace["layout_overlaps_with_spent_pools"],
                          **trace["palette_overlaps_with_spent_pools"]}.items() if v}
    assert trace["demonstration_layouts_disjoint_from_evaluation"]


def test_the_signature_collision_ceiling_is_reported_not_assumed():
    ceiling = ch.signature_collision_ceiling(range(21_000, 21_048))
    assert 0.0 <= ceiling["collision_rate"] < 0.05
    assert ceiling["detection_ceiling"] == pytest.approx(
        1.0 - ceiling["collision_rate"])


# ---- the artifacts say what the prose says -----------------------------------------------


@pytest.mark.skipif(not (ARTIFACTS / "o3-change.json").exists(),
                    reason="section I has not been run")
def test_section_i_thresholds_were_frozen_on_development_only():
    report = json.loads((ARTIFACTS / "o3-change.json").read_text())
    assert report["frozen"]["selected_on"] == "development palettes only"
    assert not (set(report["development_palettes"])
                & set(report["validation_palettes"]))


@pytest.mark.skipif(not (ARTIFACTS / "o3-change.json").exists(),
                    reason="section I has not been run")
def test_the_detector_never_fires_on_an_honest_control():
    report = json.loads((ARTIFACTS / "o3-change.json").read_text())
    for arm in ch.HONEST_ARMS:
        assert report["arms"][arm]["false_alarms_per_palette"] == 0.0, arm


@pytest.mark.skipif(not (ARTIFACTS / "o3-change.json").exists(),
                    reason="section I has not been run")
def test_recalibration_by_truncation_is_not_reported_as_a_gain():
    """The detector reaches the exact-change-point ceiling and is still WORSE than
    ignoring the change, because the clean history is shorter. The artifact has to carry
    both numbers or the pass would read as an improvement it is not."""
    report = json.loads((ARTIFACTS / "o3-change.json").read_text())
    detector = report["arms"]["6_silent_palette_change"]["recovery_event_accuracy"]
    ceiling = report["arms"]["4_exact_change_point_ceiling"]["recovery_event_accuracy"]
    none = report["arms"]["1_no_change_detector"]["recovery_event_accuracy"]
    assert detector == pytest.approx(ceiling, abs=1e-9), (
        "the detector must be at the oracle ceiling, or the loss is a detection failure")
    assert none is not None and detector is not None
