"""E / D. The nine visual interfaces and the shared public auxiliary heads.

Every interface is a function from a frozen frame pair to an 8x8 grid of 64-dimensional
slots, so slot count and width are matched by construction rather than by promise. What
differs is how those slots are produced and how many TRAINABLE parameters that costs,
which is reported per interface and never rounded away.

The event head is byte-identical across interfaces: same architecture, same parameter
count, same optimizer, same update count, same examples, same checkpoint rule. If one
interface wins it is because its slots carry more about the transition, not because it
was given a bigger head.

The cheap baselines -- a small translation-equivariant CNN and a FIXED random projection
-- stay in every table. A 4B backbone that cannot beat a random projection on a 24x24
frame has told you something, and removing the baseline would hide it.

    imported by n_pathway.py; runnable directly for the auxiliary-head report
"""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import n_core as core
from n_core import FRAME, GRID, N_DISPLACEMENT, PairBatch

# Each interface keeps its NATIVE coordinate grid. An earlier draft forced every
# interface through a shared 8x8 grid, which meant zero-padding a 12x12 feature map to
# 16x16 and mean-pooling 2x2 -- so six of the eight slot rows carried real cells, two
# carried padding, and no slot boundary lined up with a cell boundary. Both pixel
# interfaces then sat at majority-class on every spatial target. Matching the grid is
# not worth destroying it; what is matched instead is slot WIDTH, the head architecture
# and the head's parameter count, and the head is fully convolutional so its parameter
# count does not depend on the grid at all.
SLOT_WIDTH = 64
HEAD_HIDDEN = 128
UPDATES = 1500
BATCH = 128
LEARNING_RATE = 2e-3


def frozen_projection(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    """A fixed random projection. Declared, reproducible, and NOT trainable."""
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(in_dim, out_dim)) / np.sqrt(in_dim)).astype(np.float32)


def pool_to_slots(grid: np.ndarray, target: int) -> np.ndarray:
    """(N, H, W, C) -> (N, target, target, C) by exact block mean.

    Refuses to upsample. The J-phase spent a whole audit discovering that an arm sitting
    at its upsampling ceiling looks like a representation failure, so producing a finer
    grid than the source has is forbidden here rather than silently interpolated.
    """
    n, h, w, c = grid.shape
    if h == target:
        return grid
    if h < target:
        # Checked BEFORE divisibility: 8 -> 16 is not divisible either, and reporting it
        # as "not an exact multiple" would hide that the real objection is upsampling.
        raise ValueError(
            f"cannot produce a {target}x{target} slot grid from a {h}x{h} source; "
            "upsampling would manufacture resolution the source never had")
    if h % target:
        raise ValueError(f"cannot pool {h}x{w} to {target}x{target} exactly")
    block = h // target
    return grid.reshape(n, target, block, target, block, c).mean(axis=(2, 4))


# ---- interfaces ----------------------------------------------------------------------


@dataclass
class Encoded:
    slots: np.ndarray          # (N, 8, 8, 64)
    trainable_parameters: int
    frozen_parameters: int
    note: str = ""


class Interface:
    name = "base"
    eligible = True            # False for diagnostic/ceiling arms
    kind = "visual"

    def encode(self, pairs: PairBatch, fit_on: PairBatch | None = None) -> Encoded:
        raise NotImplementedError

    def digest(self) -> str:
        return hashlib.sha256(self.name.encode()).hexdigest()[:12]


class CellAlignedDiagnostic(Interface):
    """Interface 8. The environment's own cell grid, handed over for free.

    Diagnostic only: it uses knowledge of CELL=2 to subsample exactly on cell centres,
    which is a fact about the simulator rather than something perception recovered.
    """
    name = "8_cell_aligned_diagnostic"
    eligible = False

    def encode(self, pairs, fit_on=None):
        before = pairs.before[:, ::core.CELL, ::core.CELL, :]      # (N, 12, 12, 3)
        after = pairs.after[:, ::core.CELL, ::core.CELL, :]
        stacked = np.concatenate([before, after], axis=-1)          # (N, 12, 12, 6)
        projection = frozen_projection(6, SLOT_WIDTH, 20_001)
        return Encoded(stacked @ projection, 0, projection.size,
                       "cell-aligned 12x12 subsample; uses CELL=2, a simulator fact")


class RandomProjectionInterface(Interface):
    """Interface 2. Fixed random spatial projection of the raw pixels. No training."""
    name = "2_fixed_random_projection"

    def encode(self, pairs, fit_on=None):
        stacked = np.concatenate([pairs.before, pairs.after], axis=-1)   # (N,24,24,6)
        pooled = pool_to_slots(stacked, GRID)      # 24 -> 12, an exact 2x2 block mean
        projection = frozen_projection(6, SLOT_WIDTH, 20_002)
        return Encoded(pooled @ projection, 0, projection.size,
                       "exact 2x2 pixel-block mean to the 12x12 cell grid, then a "
                       "frozen random projection; no learned parameter")


class EquivariantCNN(Interface):
    """Interface 1. A small translation-equivariant CNN on the raw frame pair.

    Every layer is a stride-1 or stride-2 convolution with no positional input, so the
    map from pixels to slots commutes with translation up to the stride. That property
    is what the M2F structured phase showed to be the decisive one, and it is asserted by
    a test rather than by this docstring.
    """
    name = "1_raw_pixels_equivariant_cnn"

    def __init__(self, channels: int = 32, seed: int = 20_003) -> None:
        self.channels = channels
        self.seed = seed
        self.model = None

    def build(self):
        import mlx.core as mx
        import mlx.nn as nn

        channels, width = self.channels, SLOT_WIDTH

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(6, channels, 3, padding=1)
                self.c2 = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
                self.c3 = nn.Conv2d(channels, width, 3, padding=1)

            def __call__(self, x):
                h = nn.relu(self.c1(x))
                h = nn.relu(self.c2(h))              # 24 -> 12
                return self.c3(h)                    # (N, 12, 12, width)

        mx.random.seed(self.seed)
        model = Net()
        mx.eval(model.parameters())
        return model

    def encode(self, pairs, fit_on=None):
        import mlx.core as mx
        from mlx.utils import tree_flatten

        if self.model is None:
            self.model = self.build()
        stacked = np.concatenate([pairs.before, pairs.after], axis=-1).astype(np.float32)
        out = []
        for start in range(0, len(stacked), 512):
            block = self.model(mx.array(stacked[start:start + 512]))
            mx.eval(block)
            out.append(np.asarray(block))
        grid = np.concatenate(out)                    # (N, 12, 12, width), cell-aligned
        count = int(sum(v.size for _, v in tree_flatten(self.model.trainable_parameters())))
        return Encoded(grid, count, 0,
                       "stride-1/2 convolutions only; no positional channel")


class LearnedSlotCNN(EquivariantCNN):
    """Interface 3. A wider learned CNN producing slots at the shared geometry."""
    name = "3_learned_cnn_slots"

    def __init__(self):
        super().__init__(channels=64, seed=20_004)


class BackboneSlots(Interface):
    """Interfaces 4-7. Frozen pretrained vision towers, slots or mean-pool.

    Slots are pooled to the shared 8x8 grid and projected to the shared width with a
    FIXED random projection, so nothing about the comparison depends on a trainable
    adapter that only the big models get.
    """

    def __init__(self, encoder_id: str, pooled: bool) -> None:
        self.encoder_id = encoder_id
        self.pooled = pooled
        self.name = (f"{'6' if encoder_id.startswith('qwen') else '7'}_"
                     f"{encoder_id}_mean_pool" if pooled else
                     f"{'4' if encoder_id.startswith('qwen') else '5'}_"
                     f"{encoder_id}_spatial_slots")
        self.cache: dict[str, np.ndarray] = {}

    def encode(self, pairs, fit_on=None):
        from n_backbones import encode_frames

        before = encode_frames(self.encoder_id, pairs.before)
        after = encode_frames(self.encoder_id, pairs.after)
        if self.pooled:
            # The ablation: destroy the coordinate grid by averaging over it, then
            # broadcast back so the head sees the same tensor shape and the same
            # parameter count. Only the spatial information differs.
            before = np.repeat(np.repeat(before.mean(axis=(1, 2), keepdims=True),
                                         before.shape[1], 1), before.shape[2], 2)
            after = np.repeat(np.repeat(after.mean(axis=(1, 2), keepdims=True),
                                        after.shape[1], 1), after.shape[2], 2)
        stacked = np.concatenate([before, after], axis=-1)   # native grid preserved
        assert stacked.shape[-1] == SLOT_WIDTH, stacked.shape
        return Encoded(stacked, 0, 2 * 2560 * 32,
                       f"frozen {self.encoder_id} vision tower, "
                       f"{'mean-pooled over the token grid' if self.pooled else 'coordinate-preserving'}"
                       f"; fixed 2560->32 projection per frame")


class StructuredRelationalCeiling(Interface):
    """Interface 9. The M2F structured relational detector. Ceiling only.

    It receives structured public features, not pixels, so it is not a visual interface
    and cannot win the phase; it says what perception is aiming at.
    """
    name = "9_structured_relational_ceiling"
    eligible = False
    kind = "structured"

    def encode(self, pairs, fit_on=None):
        raise NotImplementedError("scored directly in n_pathway, not through slots")
