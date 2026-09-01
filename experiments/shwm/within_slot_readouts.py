"""E. Three decoders that can localise inside a slot, frozen before validation.

The J-phase slot head was a shared slotwise MLP followed by nearest upsampling.
Every cell inside a slot therefore received the same logit, `argmax` returned the
row-major first cell, and exact-cell accuracy was capped at
`P(true cell is first in its slot)` -- 0.1175 at 4x4 and 0.3650 at 8x8, which the
arms attained exactly. Nothing about an interface could be read through a decoder
whose output resolution was coarser than the quantity being decoded.

All three candidates below fix that in a different way, which is the point of
having three: if they agree, the conclusion is about the interface, and if they
disagree, the disagreement is informative about what kind of decoding the
representation supports.

1. `coordinate_query_cross_attention` -- one query per output cell, carrying its
   own normalised coordinate, attending over the slots. Within-slot position is
   available because the query coordinate is finer than the slot grid.
2. `token_grid_cnn` -- convolution over the slot grid with a *learned* upsample,
   so two cells in one slot can receive different values.
3. `hierarchical_slot_offset` -- predicts the coarse slot and a normalised offset
   inside it, which addresses the failure in the most direct way available.

All three share a parameter ceiling, optimiser, seeds, targets and data. The
comparison is between decoders and interfaces, not between training budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

PARAMETER_CEILING = 250_000
"""Pre-registered, matching the J-phase cap so the two phases stay comparable."""

GRID = 12
EPOCHS = 45
BATCH = 64
LEARNING_RATE = 2e-3
SEED = 6600


def _count(model) -> int:
    from mlx.utils import tree_flatten
    return int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))


def build_coordinate_query_cross_attention(grid: int, width: int):
    """Output-coordinate queries attend over the source slots.

    Each of the 144 output cells owns a query built from its normalised (row, col).
    Two cells inside one slot carry different coordinates, so they can attend
    identically and still decode differently -- which is exactly the capability the
    old head lacked.
    """
    import mlx.core as mx
    import mlx.nn as nn

    class CoordinateQueryCrossAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dim = 48
            rows = (np.arange(GRID) + 0.5) / GRID
            coords = np.stack(np.meshgrid(rows, rows, indexing="ij"), axis=-1)
            flat = coords.reshape(GRID * GRID, 2).astype(np.float32)
            # Sinusoidal coordinate features. The first version fed raw (row, col)
            # into a linear query and a linear output head, which cannot select an
            # arbitrary cell -- a capability test caught it before any validation
            # exposure: it reached 0.125 fitting 32 examples it should memorise,
            # against 1.000 for the grid CNN. Fourier features plus a nonlinear head
            # make the family fair rather than the comparison a formality.
            bands = np.concatenate([flat * (2 ** k) * np.pi for k in range(4)], axis=-1)
            features = np.concatenate([np.sin(bands), np.cos(bands)], axis=-1)
            self._coords = mx.array(features.astype(np.float32))
            self.query = nn.Linear(features.shape[1], self.dim)
            self.key = nn.Linear(width, self.dim)
            self.value = nn.Linear(width, self.dim)
            self.out = nn.Sequential(
                nn.Linear(self.dim + features.shape[1], self.dim), nn.ReLU(),
                nn.Linear(self.dim, 2))

        def __call__(self, slots: mx.array) -> mx.array:
            batch = slots.shape[0]
            flat = slots.reshape(batch, -1, slots.shape[-1])
            q = self.query(self._coords)                       # (144, d)
            k, v = self.key(flat), self.value(flat)            # (B, S, d)
            scores = mx.matmul(q[None], k.transpose(0, 2, 1)) / np.sqrt(self.dim)
            attended = mx.matmul(mx.softmax(scores, axis=-1), v)   # (B, 144, d)
            coords = mx.repeat(self._coords[None], batch, axis=0)
            return self.out(mx.concatenate([attended, coords], axis=-1))

    return CoordinateQueryCrossAttention


def build_token_grid_cnn(grid: int, width: int):
    """Convolution over the slot grid with a learned upsample to the cell grid.

    The upsample is learned rather than nearest, so cells inside one slot are not
    forced to share a value.
    """
    import mlx.core as mx
    import mlx.nn as nn

    class TokenGridCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = 48
            self.a = nn.Conv2d(width, hidden, 1)
            self.b = nn.Conv2d(hidden, hidden, 3, padding=1)
            factor = GRID // grid if GRID % grid == 0 else 2
            self.factor = factor
            self.expand = nn.Conv2d(hidden, hidden * factor * factor, 1)
            self.head = nn.Conv2d(hidden, 2, 3, padding=1)

        def __call__(self, slots: mx.array) -> mx.array:
            z = nn.relu(self.a(slots))
            z = nn.relu(self.b(z))
            z = self.expand(z)                                  # (B, g, g, h*f*f)
            b, g, _, c = z.shape
            f = self.factor
            hidden = c // (f * f)
            # pixel-shuffle: each slot expands into an f x f block with distinct values
            z = z.reshape(b, g, g, f, f, hidden).transpose(0, 1, 3, 2, 4, 5)
            z = z.reshape(b, g * f, g * f, hidden)
            if z.shape[1] != GRID:
                idx = mx.array(((np.arange(GRID) * z.shape[1]) // GRID).astype(np.int32))
                z = mx.take(mx.take(z, idx, axis=1), idx, axis=2)
            z = nn.relu(z)
            return self.head(z).reshape(b, GRID * GRID, 2)

    return TokenGridCNN


def build_hierarchical_slot_offset(grid: int, width: int):
    """Coarse slot logits plus a normalised within-slot offset.

    The most direct answer to the J-phase failure: the slot term reproduces what
    the old head could do, and the offset term supplies what it could not.
    """
    import mlx.core as mx
    import mlx.nn as nn

    class HierarchicalSlotOffset(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = 64
            self.trunk = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(),
                                       nn.Linear(hidden, hidden), nn.ReLU())
            self.slot_logit = nn.Linear(hidden, 2)     # agent, switch -- per slot
            self.offset = nn.Linear(hidden, 4)         # (dr, dc) per channel
            rows = (np.arange(GRID) + 0.5) / GRID
            coords = np.stack(np.meshgrid(rows, rows, indexing="ij"), axis=-1)
            self._cell = mx.array(coords.reshape(GRID * GRID, 2).astype(np.float32))
            owner = ((np.arange(GRID) * grid) // GRID)
            self._owner = mx.array((owner[:, None] * grid + owner[None, :])
                                   .reshape(-1).astype(np.int32))
            centres = (np.arange(grid) + 0.5) / grid
            cc = np.stack(np.meshgrid(centres, centres, indexing="ij"), axis=-1)
            self._slot_centre = mx.array(cc.reshape(grid * grid, 2).astype(np.float32))

        def __call__(self, slots: mx.array) -> mx.array:
            batch = slots.shape[0]
            flat = slots.reshape(batch, -1, slots.shape[-1])
            h = self.trunk(flat)                                  # (B, S, hidden)
            slot_logits = self.slot_logit(h)                      # (B, S, 2)
            offsets = self.offset(h).reshape(batch, -1, 2, 2)     # (B, S, 2ch, 2)
            per_cell_slot = mx.take(slot_logits, self._owner, axis=1)      # (B, 144, 2)
            per_cell_off = mx.take(offsets, self._owner, axis=1)           # (B, 144, 2, 2)
            centre = mx.take(self._slot_centre, self._owner, axis=0)       # (144, 2)
            delta = self._cell[None, :, None, :] - centre[None, :, None, :]
            # the offset head scores how well each cell's displacement from its slot
            # centre matches the predicted displacement: this is what breaks the tie
            # between cells that share a slot
            agreement = -mx.sum((delta - per_cell_off) ** 2, axis=-1) * float(GRID)
            return per_cell_slot + agreement

    return HierarchicalSlotOffset


DECODERS: dict[str, Callable[[int, int], Any]] = {
    "coordinate_query_cross_attention": build_coordinate_query_cross_attention,
    "token_grid_cnn": build_token_grid_cnn,
    "hierarchical_slot_offset": build_hierarchical_slot_offset,
}


def train_decoder(name, grid, width, train_slots, train_agent, train_switch,
                  train_visible, evaluate: dict[str, tuple]) -> dict[str, Any]:
    """Identical optimiser, epochs, batch and seed for every decoder and interface."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(SEED)
    model = DECODERS[name](grid, width)()
    mx.eval(model.parameters())
    parameters = _count(model)
    if parameters > PARAMETER_CEILING:
        raise ValueError(f"{name} at width {width}: {parameters} exceeds "
                         f"{PARAMETER_CEILING}")
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)
    rng = np.random.default_rng(4)
    n = len(train_slots)
    for _ in range(EPOCHS):
        for _ in range(max(1, n // BATCH)):
            idx = rng.integers(0, n, BATCH)
            xb = mx.array(train_slots[idx])
            ab = mx.array(train_agent[idx].astype(np.int32))
            sb = mx.array(train_switch[idx])
            vb = mx.array(train_visible[idx])

            def loss_fn(m):
                out = m(xb)
                agent_loss = nn.losses.cross_entropy(out[:, :, 0], ab, reduction="mean")
                per_cell = nn.losses.binary_cross_entropy(
                    out[:, :, 1], sb, with_logits=True, reduction="none")
                return agent_loss + (per_cell * vb).sum() / mx.maximum(vb.sum(), 1.0)

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    def decode(slots):
        agent, switch = [], []
        for k in range(0, len(slots), 256):
            out = model(mx.array(slots[k:k + 256]))
            mx.eval(out)
            values = np.asarray(out)
            e = np.exp(values[:, :, 0] - values[:, :, 0].max(axis=1, keepdims=True))
            agent.append(e / e.sum(axis=1, keepdims=True))
            # stable sigmoid: the naive form overflows on confident logits
            z = values[:, :, 1]
            switch.append(np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                                   np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z)))))
        return np.concatenate(agent), np.concatenate(switch)

    return {"parameters": parameters, "decode": decode}
