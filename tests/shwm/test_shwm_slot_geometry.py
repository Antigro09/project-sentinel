"""The audit instrument gets the same scrutiny as the thing it measures.

If `pool_grid` silently upsampled, or a geometry misreported its alignment, the
slot-resolution audit would produce confident numbers about the wrong thing.
Every guard below has a calibration arm: an input it must reject.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.wm import slot_geometry as sg
from sentinel.wm.latent_contract import ContractViolation


# ---- geometry arithmetic -----------------------------------------------------------------


def test_matched_capacity_arms_hold_the_same_scalar_count() -> None:
    """A and B must be capacity-matched or the comparison is about size."""
    assert sg.GEOMETRY_A.scalars == sg.GEOMETRY_B.scalars == 4096


def test_high_capacity_arm_is_actually_higher_capacity() -> None:
    assert sg.GEOMETRY_C.scalars == 4 * sg.GEOMETRY_A.scalars


def test_alignment_is_reported_correctly_for_each_geometry() -> None:
    """Only a block that divides neither way straddles a cell."""
    assert sg.GEOMETRY_A.cell_aligned is True
    assert sg.GEOMETRY_B.cell_aligned is False
    assert sg.GEOMETRY_C.cell_aligned is False
    assert sg.GEOMETRY_D.cell_aligned is True
    assert sg.GEOMETRY_E.cell_aligned is True


def test_cells_per_block_matches_the_frame_and_grid() -> None:
    assert sg.GEOMETRY_A.cells_per_block == 3.0
    assert sg.GEOMETRY_B.cells_per_block == 1.5
    assert sg.GEOMETRY_D.cells_per_block == 1.0


# ---- never upsample ----------------------------------------------------------------------


def test_pool_grid_reduces_correctly() -> None:
    grid = np.arange(16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    assert sg.pool_grid(grid, 8).shape == (8, 8, 3)
    assert sg.pool_grid(grid, 4).shape == (4, 4, 3)


def test_pool_grid_refuses_to_upsample() -> None:
    """Calibration: the whole audit turns on not inventing resolution."""
    grid = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(ContractViolation, match="upsampling"):
        sg.pool_grid(grid, 12)


def test_pool_grid_refuses_a_ragged_division() -> None:
    grid = np.zeros((16, 16, 3), dtype=np.float32)
    with pytest.raises(ContractViolation, match="does not divide"):
        sg.pool_grid(grid, 3)


def test_pool_grid_at_native_resolution_is_the_identity() -> None:
    grid = np.random.default_rng(0).normal(size=(8, 8, 5)).astype(np.float32)
    assert np.allclose(sg.pool_grid(grid, 8), grid)


# ---- token grids -------------------------------------------------------------------------


def test_qwen_and_gemma_native_grids_match_the_frozen_record() -> None:
    assert sg.NATIVE_TOKEN_GRID["qwen3_vl_4b"] == 8
    assert sg.NATIVE_TOKEN_GRID["gemma3_4b"] == 16


def test_tokens_to_grid_refuses_an_unexpected_token_count() -> None:
    """Calibration: a changed backbone must not be reshaped into silence."""
    with pytest.raises(ContractViolation, match="visual tokens"):
        sg.tokens_to_grid(np.zeros((100, 32), dtype=np.float32), "qwen3_vl_4b")


def test_backbones_cannot_reach_the_twelve_grid() -> None:
    """Qwen is 8x8 natively, so 12x12 would have to be invented."""
    for encoder in ("qwen3_vl_4b", "gemma3_4b"):
        names = [g.name for g in sg.available_geometries(encoder)]
        assert "g12x12x64" not in names


def test_pixel_sources_can_reach_every_geometry() -> None:
    for source in ("raw", "cnn"):
        assert len(sg.available_geometries(source)) == len(sg.GEOMETRIES)


def test_sub_cell_diagnostic_is_capacity_matched_to_the_aligned_one() -> None:
    assert sg.GEOMETRY_D.scalars == sg.GEOMETRY_E.scalars == 9216
    assert sg.GEOMETRY_D.cells_per_block == 1.0
    assert sg.GEOMETRY_E.cells_per_block == 0.5


def test_a_block_finer_than_a_cell_refines_rather_than_straddles() -> None:
    """E is not a misalignment control, and an earlier report treated it as one.

    At one pixel per slot the slot boundaries contain every cell boundary, so the
    partition is refined and nothing straddles. Only 8x8 -- 3px blocks against 2px
    cells -- actually cuts cells.
    """
    assert sg.GEOMETRY_D.cell_aligned      # 2px block == 1 cell
    assert sg.GEOMETRY_E.cell_aligned      # 1px block refines the cell
    assert sg.GEOMETRY_A.cell_aligned      # 6px block == 3 cells
    assert not sg.GEOMETRY_B.cell_aligned  # 3px block straddles a 2px cell
    assert not sg.GEOMETRY_C.cell_aligned


def test_neither_backbone_can_supply_the_fine_diagnostics() -> None:
    for encoder in ("qwen3_vl_4b", "gemma3_4b"):
        names = [g.name for g in sg.available_geometries(encoder)]
        assert "g12x12x64" not in names and "g24x24x16" not in names


def test_qwen_reaches_its_native_grid_without_pooling() -> None:
    names = [g.name for g in sg.available_geometries("qwen3_vl_4b")]
    assert "g8x8x64" in names and "g8x8x256" in names


# ---- width fitting -----------------------------------------------------------------------


def test_narrow_sources_are_padded_and_values_survive_exactly() -> None:
    source = np.arange(4 * 4 * 10, dtype=np.float32).reshape(4, 4, 10)
    out = sg.fit_width(source, sg.GEOMETRY_A, "t")
    assert out.shape == (16, 256)
    assert np.array_equal(out[:, :10], source.reshape(16, 10))
    assert np.all(out[:, 10:] == 0)


def test_wide_sources_are_projected_to_the_slot_width() -> None:
    source = np.zeros((4, 4, 2560), dtype=np.float32)
    assert sg.fit_width(source, sg.GEOMETRY_A, "t").shape == (16, 256)


def test_projection_is_deterministic_for_a_given_tag() -> None:
    a = sg._fixed_projection(100, 16, "tag")
    b = sg._fixed_projection(100, 16, "tag")
    assert np.array_equal(a, b)


def test_different_tags_give_different_projections() -> None:
    """Calibration: one shared matrix would couple arms that must stay separate."""
    a = sg._fixed_projection(100, 16, "one")
    b = sg._fixed_projection(100, 16, "two")
    assert not np.array_equal(a, b)


# ---- end-to-end shapes -------------------------------------------------------------------


def test_backbone_slots_shape_for_every_available_geometry() -> None:
    for encoder, side in sg.NATIVE_TOKEN_GRID.items():
        tokens = np.random.default_rng(1).normal(size=(side * side, 2560)).astype(np.float32)
        for geometry in sg.available_geometries(encoder):
            out = sg.backbone_slots(tokens, encoder, geometry)
            assert out.shape == (geometry.slot_count, geometry.width)


def test_raw_and_projection_slots_shapes() -> None:
    frame = np.random.default_rng(2).integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    for geometry in sg.available_geometries("raw"):
        assert sg.raw_slots(frame, geometry).shape == (geometry.slot_count, geometry.width)
        assert sg.random_projection_slots(frame, geometry).shape == (
            geometry.slot_count,
            geometry.width,
        )


def test_raw_slots_preserve_frame_content_at_every_geometry() -> None:
    """A block partition loses nothing; only the layout changes."""
    frame = np.random.default_rng(3).integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    for geometry in (sg.GEOMETRY_A, sg.GEOMETRY_D):
        blocks = sg.frame_blocks(frame, geometry.grid)
        assert np.isclose(blocks.sum(), frame.astype(np.float32).sum() / 255.0)


def test_geometry_report_is_serialisable_and_complete() -> None:
    report = sg.geometry_report()
    assert len(report["geometries"]) == len(sg.GEOMETRIES) == 5
    assert set(report["availability"]) == {"qwen3_vl_4b", "gemma3_4b", "raw", "cnn"}
