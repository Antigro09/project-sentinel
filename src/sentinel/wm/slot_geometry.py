"""Variable-geometry slot construction, for the slot-resolution audit.

`interfaces.py` is frozen at one geometry -- (16, 256) -- and deliberately so:
equal shape is what makes its arms comparable. This module is the audit
instrument that varies that shape on purpose, and it lives separately so the
frozen interfaces and their digests are untouched.

Three facts here are load-bearing and none of them is cosmetic.

*Never upsample.* A backbone's token grid is its resolution. Qwen emits an 8x8
grid and Gemma a 16x16 one, so asking Qwen for 12x12 slots would manufacture
detail the encoder never produced, and any improvement measured afterwards
would be an artifact of the interpolation. Requesting a grid finer than the
source is a contract violation rather than a resize.

*Cell alignment is a measured property, not an assumption.* The environment is a
12x12 game grid rendered at 2 pixels per cell, so a 24x24 frame divides evenly
into 4x4 blocks (3 cells each) and 12x12 blocks (1 cell each), but an 8x8 block
is 3 pixels -- one and a half cells -- and straddles cell boundaries. That
matters because the audit asks whether a finer grid recovers switch events, and
a misaligned grid could fail for a reason that has nothing to do with
resolution. Every geometry reports its own alignment so the two explanations
stay separable.

*Capacity is reported, never equalised silently.* An 8x8x256 grid holds four
times the scalars of the current 4x4x256 one. That is exactly why the
specification calls it a diagnostic upper bound: a win there is uninterpretable
until the matched-capacity 8x8x64 arm is also read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from sentinel.wm.latent_contract import ContractViolation
from sentinel.wm.versioning import digest_of

FRAME_SIDE = 24
GAME_GRID = 12
CELL_PIXELS = FRAME_SIDE // GAME_GRID
assert CELL_PIXELS == 2


@dataclass(frozen=True, slots=True)
class Geometry:
    """One slot layout. `role` records why the audit contains it."""

    name: str
    grid: int
    width: int
    role: str

    @property
    def slot_count(self) -> int:
        return self.grid * self.grid

    @property
    def scalars(self) -> int:
        return self.slot_count * self.width

    @property
    def block_pixels(self) -> int:
        if FRAME_SIDE % self.grid:
            raise ContractViolation(
                f"a {FRAME_SIDE}px frame does not divide into {self.grid} blocks"
            )
        return FRAME_SIDE // self.grid

    @property
    def cells_per_block(self) -> float:
        return self.block_pixels / CELL_PIXELS

    @property
    def cell_aligned(self) -> bool:
        """True when block boundaries fall on game-cell boundaries."""
        return FRAME_SIDE % self.grid == 0 and self.block_pixels % CELL_PIXELS == 0

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "grid": self.grid,
            "width": self.width,
            "role": self.role,
            "slot_count": self.slot_count,
            "scalars": self.scalars,
            "block_pixels": self.block_pixels,
            "cells_per_block": self.cells_per_block,
            "cell_aligned": self.cell_aligned,
            "float32_bytes": self.scalars * 4,
        }


GEOMETRY_A = Geometry("g4x4x256", 4, 256, "current")
GEOMETRY_B = Geometry("g8x8x64", 8, 64, "matched_capacity_fine")
GEOMETRY_C = Geometry("g8x8x256", 8, 256, "diagnostic_high_capacity")
GEOMETRY_D = Geometry("g12x12x64", 12, 64, "cell_aligned_diagnostic")
"""D is an addition to the specification's A/B/C, and it earns its place.

A and D are cell-aligned; B and C are not. Without D, an 8x8 failure has two
explanations -- the resampling lost the event, or the block boundaries cut the
cells -- and no way to tell them apart. D is available only for the pixel-space
sources, because neither backbone emits a 12x12 grid and upsampling one to reach
it would answer the question by inventing the evidence.
"""

GEOMETRY_E = Geometry("g24x24x16", 24, 16, "sub_cell_diagnostic")
"""One slot per PIXEL: finer than a game cell, and capacity-matched to D.

D and E both hold 9,216 scalars, but D is exactly one slot per cell while E puts
four slots inside each cell. That pairing answers a question D alone cannot: if D
scores perfectly because its boundaries coincide with the simulator's cells, E
should not match it; if what matters is simply having at least one slot per cell,
E should match or beat it. Without E, "the 12x12 arm is perfect" and "the 12x12
arm is aligned to the environment" are the same sentence."""

GEOMETRIES: tuple[Geometry, ...] = (GEOMETRY_A, GEOMETRY_B, GEOMETRY_C, GEOMETRY_D,
                                    GEOMETRY_E)
SPECIFIED_GEOMETRIES: tuple[Geometry, ...] = (GEOMETRY_A, GEOMETRY_B, GEOMETRY_C)

NATIVE_TOKEN_GRID: Mapping[str, int] = {"qwen3_vl_4b": 8, "gemma3_4b": 16}
"""From the frozen token-geometry artifact, not re-derived here."""


def _fixed_projection(rows: int, columns: int, tag: str) -> np.ndarray:
    """A frozen random matrix, reproducible from its tag. Never trained."""
    seed = int(digest_of({"projection": tag, "rows": rows, "columns": columns})[7:15], 16)
    generator = np.random.default_rng(seed)
    return (generator.normal(size=(rows, columns)) / np.sqrt(rows)).astype(np.float32)


def pool_grid(grid: np.ndarray, target: int) -> np.ndarray:
    """Mean-pool a (S, S, W) grid to (target, target, W). Never upsamples."""
    side = grid.shape[0]
    if target > side:
        raise ContractViolation(
            f"cannot produce a {target}x{target} slot grid from a {side}x{side} source; "
            "upsampling would manufacture resolution the source never had"
        )
    if side % target:
        raise ContractViolation(
            f"a {side}x{side} grid does not divide into {target}x{target} blocks"
        )
    factor = side // target
    return grid.reshape(target, factor, target, factor, grid.shape[-1]).mean(axis=(1, 3))


def tokens_to_grid(tokens: np.ndarray, encoder_id: str) -> np.ndarray:
    """Reshape a flat visual-token sequence into its native square grid."""
    expected = NATIVE_TOKEN_GRID[encoder_id]
    if tokens.shape[0] != expected * expected:
        raise ContractViolation(
            f"{encoder_id} produced {tokens.shape[0]} visual tokens; the frozen "
            f"geometry record says {expected}x{expected}={expected * expected}"
        )
    return tokens.reshape(expected, expected, tokens.shape[-1])


def frame_blocks(frame: np.ndarray, grid: int) -> np.ndarray:
    """Split a frame into a (grid, grid, block*block*3) array of raw blocks."""
    side = frame.shape[0]
    if side % grid:
        raise ContractViolation(f"a {side}px frame does not divide into {grid} blocks")
    block = side // grid
    normalised = frame.astype(np.float32) / 255.0
    cells = normalised.reshape(grid, block, grid, block, 3).transpose(0, 2, 1, 3, 4)
    return cells.reshape(grid, grid, block * block * 3)


def fit_width(source: np.ndarray, geometry: Geometry, tag: str) -> np.ndarray:
    """Bring a (grid, grid, W) array to the geometry's slot width.

    Narrower than the target is zero-padded, which preserves the values exactly.
    Wider is projected through a frozen random matrix, which is the same
    treatment every source gets, so no source is advantaged by the reduction.
    """
    grid, _, width = source.shape
    flat = source.reshape(grid * grid, width)
    if width == geometry.width:
        out = flat
    elif width < geometry.width:
        out = np.zeros((grid * grid, geometry.width), dtype=np.float32)
        out[:, :width] = flat
    else:
        out = flat @ _fixed_projection(width, geometry.width, f"{tag}|{geometry.name}")
    return np.ascontiguousarray(out.astype(np.float32))


def backbone_slots(tokens: np.ndarray, encoder_id: str, geometry: Geometry) -> np.ndarray:
    grid = pool_grid(tokens_to_grid(tokens, encoder_id), geometry.grid)
    return fit_width(grid, geometry, f"backbone|{encoder_id}")


def raw_slots(frame: np.ndarray, geometry: Geometry) -> np.ndarray:
    return fit_width(frame_blocks(frame, geometry.grid), geometry, "raw")


def random_projection_slots(frame: np.ndarray, geometry: Geometry) -> np.ndarray:
    blocks = frame_blocks(frame, geometry.grid)
    flat = blocks.reshape(geometry.slot_count, blocks.shape[-1])
    projection = _fixed_projection(
        blocks.shape[-1], geometry.width, f"randproj|{geometry.name}"
    )
    return np.ascontiguousarray((flat @ projection).astype(np.float32))


def available_geometries(source: str) -> tuple[Geometry, ...]:
    """Which geometries a source can supply without upsampling."""
    if source in NATIVE_TOKEN_GRID:
        native = NATIVE_TOKEN_GRID[source]
        return tuple(
            g for g in GEOMETRIES if g.grid <= native and native % g.grid == 0
        )
    return tuple(g for g in GEOMETRIES if FRAME_SIDE % g.grid == 0)


def geometry_report() -> dict[str, Any]:
    return {
        "frame_side": FRAME_SIDE,
        "game_grid": GAME_GRID,
        "cell_pixels": CELL_PIXELS,
        "geometries": [g.canonical_dict() for g in GEOMETRIES],
        "native_token_grid": dict(NATIVE_TOKEN_GRID),
        "availability": {
            source: [g.name for g in available_geometries(source)]
            for source in ("qwen3_vl_4b", "gemma3_4b", "raw", "cnn")
        },
    }
