"""Regression pins for Scale 1A-0R-N.

Three pin instrument bugs that each produced a confident wrong answer before being
caught: a misaligned slot pooling, a scalar-supervised event head that could not fit its
own training set, and a degenerate multimodal target.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments/shwm"))

import n_core as core  # noqa: E402
import n_heads as heads  # noqa: E402
import n_interfaces as ifaces  # noqa: E402


def test_pool_to_slots_refuses_to_upsample():
    grid = np.zeros((2, 8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="upsampling"):
        ifaces.pool_to_slots(grid, 16)
    with pytest.raises(ValueError, match="exactly"):
        ifaces.pool_to_slots(np.zeros((2, 12, 12, 3), np.float32), 8)
    assert ifaces.pool_to_slots(np.zeros((2, 16, 16, 3), np.float32), 8).shape[1] == 8


def test_head_parameter_count_does_not_depend_on_the_slot_grid():
    """An 8x8 backbone and a 12x12 CNN must get literally the same head."""
    counts = set()
    for _ in range(1):
        model = heads.build_head(1, 7, "binary")
        counts.add(heads.parameter_count(model))
    assert len(counts) == 1
    import mlx.core as mx
    model = heads.build_head(1, 7, "binary")
    for side in (8, 12, 16):
        out = model(mx.array(np.zeros((2, side, side, ifaces.SLOT_WIDTH), np.float32)),
                    mx.array(np.zeros((2, 4), np.float32)))
        assert out.shape == (2, 1)


def test_mask_head_output_is_always_the_cell_grid():
    import mlx.core as mx
    model = heads.build_head(core.GRID * core.GRID, 7, "multilabel")
    for side in (8, 12, 16):
        out = model(mx.array(np.zeros((2, side, side, ifaces.SLOT_WIDTH), np.float32)),
                    mx.array(np.zeros((2, 4), np.float32)))
        assert out.shape == (2, core.GRID * core.GRID)


def test_event_and_entered_are_spatially_supervised():
    """A scalar-supervised event head could not fit even its TRAINING set (0.5301 at
    2500 updates). The spatial form is what made it learnable, so the kind is pinned."""
    assert heads.TARGETS["6_retrospective_event"][1] == "spatial_scalar"
    assert heads.TARGETS["5_entered_cell_switch"][1] == "spatial_scalar"


def test_unidentifiable_targets_declare_their_subset():
    assert heads.TARGETS["5_entered_cell_switch"][3] == "moved"
    assert heads.TARGETS["7_reset_stripe_state"][3] == "reset_pair"
    assert heads.TARGETS["6_retrospective_event"][3] is None


def test_entered_switch_is_genuinely_invisible_without_a_move():
    """The agent occludes its own cell in BOTH frames, so nothing in the pair says what
    is underneath. This is an identifiability fact, not a perception failure."""
    episodes = core.collect_visual(core.TRAIN_LAYOUTS[:4], 1, 9, seed=11)
    pairs = core.to_pairs(episodes)
    # Exclude the reset pair: its `before` carries the polarity stripe and its `after`
    # does not, so those two frames differ even when the agent stayed put.
    still = ((pairs.displacement == (core.N_DISPLACEMENT - 1))
             & (pairs.is_reset_pair < 0.5))
    assert still.sum() > 5
    same = [np.array_equal(pairs.before[i], pairs.after[i])
            for i in np.flatnonzero(still)]
    assert all(same), "an unmoved non-reset transition must leave the frame unchanged"


def test_switch_mask_target_respects_occlusion():
    episodes = core.collect_visual(core.TRAIN_LAYOUTS[:3], 1, 9, seed=11)
    for episode in episodes:
        for t in range(episode.length):
            r, c = episode.positions[t]
            assert episode.switch_mask[t][r, c] == 0.0


def test_event_map_reduces_to_the_scalar_event():
    episodes = core.collect_visual(core.TRAIN_LAYOUTS[:4], 1, 9, seed=11)
    pairs = core.to_pairs(episodes)
    assert np.array_equal(pairs.event_map.max(axis=1) > 0.5, pairs.event > 0.5)


def test_visual_dataflow_guards_catch_every_planted_channel():
    import n_dataflow as flow
    episodes = core.collect_visual(core.TRAIN_LAYOUTS[:3], 1, 6, seed=11)
    pairs = core.to_pairs(episodes)
    rows = flow.build_rows(pairs, episodes)
    width = pairs.before[0].size * 2 + 4 + flow.SLOTS
    detector = flow.Detector(width)
    for channel in flow.CHANNELS:
        guard = flow.channel_guard(channel)
        assert guard(flow.Pipeline(), rows, detector), channel
        assert not guard(flow.Pipeline(leaks=frozenset({channel})), rows, detector), channel


def test_goal_is_a_separate_dynamic_from_layout():
    """Section K needs the same rendered world under two different goals."""
    from sentinel.env.adapters.procedural_visual_v2 import ProceduralVisualV2Adapter
    from sentinel.wm.authority import AuthorityGate
    goals, frames = set(), []
    for draw in (0, 2):   # 0 and 1 both hash to alpha
        adapter = ProceduralVisualV2Adapter(gate=AuthorityGate(gate_id="t"))
        # ONE dynamic only: a second reset defaults the goal back and both draws collapse
        # to the same goal. That bug made the first section-K run measure nothing.
        adapter.reset(110_000, f"goal:{draw}")
        goals.add(adapter.goal_text())
        frames.append(adapter.frame().copy())
    assert len(goals) == 2, "the two goal draws must give two different goals"
    assert np.array_equal(frames[0], frames[1]), "the frame must not reveal the goal"
