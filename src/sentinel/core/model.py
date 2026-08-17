"""The tiny recursive core.

A TRM-style reasoner: one small network applied over and over, carrying a
latent scratchpad `z` and a candidate answer `y`, refining both across
cycles rather than growing wider or deeper. TRM reached ~45% on ARC-AGI-1
with roughly 7M parameters and two layers, beating models thousands of
times its size, and that result is the entire reason this architecture is
worth trying here.

The shape of the bet, concretely:

- **Depth comes from recursion, not layers.** The same weights run for many
  cycles. Capacity to *reason* is bought with compute at inference rather
  than parameters at rest, which is what makes test-time adaptation
  affordable later — a 7M network can take real gradient steps mid-episode
  on this machine, and a 120B one cannot.
- **The answer is drafted, then revised.** `y` starts as a guess and is
  improved against the evidence, rather than emitted left-to-right in one
  pass. There is no autoregressive exposure bias because there is no
  autoregression.
- **The scratchpad is where the work happens.** `z` is never read out. It
  exists only to let the network hold intermediate structure across cycles
  — which is what inferring a hidden counter from a sequence of moves
  actually requires.

Deliberately small. If this needs to be scaled up to work, that is
evidence against the thesis, not a tuning step.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .encoding import CHANNELS, CROP, HEADS, MAX_TRANSITIONS, N_ACTIONS, N_CELL_VALUES


def _coord_grid(size: int) -> mx.array:
    """Normalised (x, y) per cell, shaped (1, 1, size, size, 2)."""
    axis = (mx.arange(size).astype(mx.float32) / (size - 1)) * 2.0 - 1.0
    xs = mx.broadcast_to(axis.reshape(1, size), (size, size))
    ys = mx.broadcast_to(axis.reshape(size, 1), (size, size))
    return mx.stack([xs, ys], axis=-1)[None, None]


def _sinusoidal(length: int, dim: int) -> mx.array:
    """Fixed sinusoidal positions.

    Fixed rather than learned so the ordering signal is present from step
    one. A learned encoding starting at zero has to discover that order
    matters before it can use it, and a network that has already settled
    into predicting the class prior has no gradient pointing it there.
    """
    pos = mx.arange(length).reshape(length, 1).astype(mx.float32)
    i = mx.arange(0, dim, 2).astype(mx.float32)
    freq = mx.exp(-mx.log(mx.array(10000.0)) * i / dim)
    angles = pos * freq
    return mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)[None, :, :dim]


@dataclass(frozen=True, slots=True)
class CoreConfig:
    d_model: int = 128
    n_heads: int = 4
    cell_embed: int = 16
    cycles: int = 8
    """Draft-revise passes. TRM uses ~16; fewer here because the sequence is
    short. Raising this buys reasoning depth at inference with no new
    parameters, which is the axis worth scaling later."""
    inner_steps: int = 3
    """Latent updates per cycle."""
    dropout: float = 0.0

    def describe(self) -> str:
        return (
            f"d_model={self.d_model} heads={self.n_heads} "
            f"cycles={self.cycles}x{self.inner_steps}"
        )


class TransitionEncoder(nn.Module):
    """One transition -> one token.

    Cell values are embedded rather than treated as numbers: value 3 is not
    "more" than value 1, they are different kinds of thing. Treating the
    palette as ordinal would invent a metric that does not exist and make
    the network fight it.
    """

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.cell = nn.Embedding(N_CELL_VALUES, cfg.cell_embed)
        self.action = nn.Embedding(N_ACTIONS + 1, cfg.d_model)

        # Coordinate channels, CoordConv-style. Flattening a 16x16 grid
        # through a Linear destroys spatial relationships, and the label that
        # matters here — a hidden counter making every Nth move travel two
        # cells — is recoverable only from the *distance* between the two
        # cells that changed. Measured directly on the data, the period shows
        # up as a +0.70 autocorrelation spike at exactly the right lag, so the
        # signal is strong; the encoder simply had no way to see it, because
        # nothing in a flat vector says which entries are adjacent.
        #
        # This hands over position, not displacement. Computing distance from
        # coordinates, and noticing it is periodic, is still the network's job.
        self.coords = _coord_grid(CROP)

        flat = CROP * CROP * (cfg.cell_embed * 2 + 3)
        self.proj = nn.Linear(flat, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def __call__(self, grids: mx.array, actions: mx.array) -> mx.array:
        # grids: (B, T, CROP, CROP, 3) -> before, after, changed
        before = self.cell(grids[..., 0])
        after = self.cell(grids[..., 1])
        changed = grids[..., 2][..., None].astype(before.dtype)

        b, t = before.shape[0], before.shape[1]
        coords = mx.broadcast_to(self.coords, (b, t, CROP, CROP, 2))

        joined = mx.concatenate([before, after, changed, coords], axis=-1)
        tokens = self.proj(joined.reshape(b, t, -1))

        # -1 marks padding; shift to the reserved final embedding row.
        action_ids = mx.where(actions < 0, N_ACTIONS, actions)
        return self.norm(tokens + self.action(action_ids))


class RecursiveBlock(nn.Module):
    """The single block applied repeatedly. All recursion shares these weights."""

    def __init__(self, cfg: CoreConfig) -> None:
        super().__init__()
        self.attn = nn.MultiHeadAttention(cfg.d_model, cfg.n_heads)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff1 = nn.Linear(cfg.d_model, cfg.d_model * 3)
        self.ff2 = nn.Linear(cfg.d_model * 3, cfg.d_model)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        x = x + self.attn(h, h, h)
        h = self.norm2(x)
        return x + self.ff2(nn.gelu(self.ff1(h)))


class TinyRecursiveCore(nn.Module):
    """Draft an answer, then revise it against the evidence, repeatedly."""

    def __init__(self, cfg: CoreConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CoreConfig()
        c = self.cfg

        self.encoder = TransitionEncoder(c)

        # Sinusoidal, and emphatically not zeros. Attention is permutation
        # invariant without positional information, and the whole point of
        # this network is to recover `charge_period` -- "every Nth move
        # travels two cells" -- which is a statement about ORDER. With a
        # zeroed positional signal that label is not merely hard to learn,
        # it is information-theoretically unrecoverable, and the network
        # correctly settled on predicting the class prior instead.
        self.pos = _sinusoidal(MAX_TRANSITIONS, c.d_model)

        self.block = RecursiveBlock(c)

        # Random rather than zero: identical zero-initialised seeds give
        # every batch element the same starting draft and symmetric
        # gradients, which is a slow way to never break symmetry.
        self.y_init = mx.random.normal((1, 1, c.d_model)) * 0.02
        self.z_init = mx.random.normal((1, 1, c.d_model)) * 0.02

        self.mix_z = nn.Linear(c.d_model * 2, c.d_model)
        self.mix_y = nn.Linear(c.d_model * 2, c.d_model)
        # Normalising after each mix keeps 24 stacked applications of the
        # same block from drifting in scale.
        self.norm_z = nn.LayerNorm(c.d_model)
        self.norm_y = nn.LayerNorm(c.d_model)
        self.norm_out = nn.LayerNorm(c.d_model)

        self.heads = [nn.Linear(c.d_model, n) for _, n in HEADS]

    def __call__(self, grids: mx.array, actions: mx.array) -> list[mx.array]:
        c = self.cfg
        b = grids.shape[0]

        evidence = self.encoder(grids, actions) + self.pos
        y = mx.broadcast_to(self.y_init, (b, 1, c.d_model))
        z = mx.broadcast_to(self.z_init, (b, 1, c.d_model))

        for _ in range(c.cycles):
            # Refine the scratchpad against evidence and the current draft.
            for _ in range(c.inner_steps):
                seq = mx.concatenate([z, y, evidence], axis=1)
                seq = self.block(seq)
                z = self.norm_z(self.mix_z(mx.concatenate([z, seq[:, :1]], axis=-1)))

            # Then revise the draft using the refined scratchpad.
            seq = mx.concatenate([y, z, evidence], axis=1)
            seq = self.block(seq)
            y = self.norm_y(self.mix_y(mx.concatenate([y, seq[:, :1]], axis=-1)))

        out = self.norm_out(y[:, 0])
        return [head(out) for head in self.heads]

    def parameter_count(self) -> int:
        def count(tree) -> int:
            if isinstance(tree, mx.array):
                return tree.size
            if isinstance(tree, dict):
                return sum(count(v) for v in tree.values())
            if isinstance(tree, (list, tuple)):
                return sum(count(v) for v in tree)
            return 0

        return count(self.parameters())


def loss_fn(model: TinyRecursiveCore, grids: mx.array, actions: mx.array, labels: mx.array) -> mx.array:
    """Mean cross-entropy across heads.

    Heads are weighted equally on purpose. `charge_period` is the hardest
    and most interesting label, but up-weighting it would let a model look
    good on the headline number while ignoring the rest — and the claim
    under test is that the architecture infers *structure*, not that it can
    be tuned to pass one metric.
    """
    logits = model(grids, actions)
    total = mx.zeros(())
    for i, head_logits in enumerate(logits):
        total = total + mx.mean(nn.losses.cross_entropy(head_logits, labels[:, i]))
    return total / len(logits)
