"""L. A frozen, bounded family of readouts for the language-conditioned goal decision.

Phase N passed its multimodal gate with a pooled-coordinate readout and phase O1 found
that readout at 0.5000 on contested keys -- for the CORRECT-language arm, which means the
test had no power and neither result said anything about language. A readout that cannot
express the task cannot falsify a claim about the input.

So the family is fixed here, before any learned visual arm is read, and it is qualified
against a positive control: the semantic-role oracle, which is handed the marker cells
outright. Any member that cannot solve the task from THAT input is disqualified, and the
failure is the readout's, not the representation's.

Every member takes the same four things -- a (12, 12, K) grid, a language one-hot, a
phase scalar and an action one-hot -- and returns one logit.

    imported by o2_goal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

GRID = 12
HIDDEN = 64
UPDATES = 2_500
BATCH = 128
LEARNING_RATE = 2e-3

FAMILY = ("1_oracle_coordinate", "2_soft_argmax_goal_relative",
          "3_coordinate_query_attention", "4_spatial_goal_heatmap",
          "5_pooled_coordinate_baseline")


def _soft_argmax(score, mx):
    """score (N, 144) -> (N, 2) expected cell coordinate, normalised to [0, 1]."""
    weight = mx.softmax(score, axis=-1)
    rows = mx.array((np.arange(GRID * GRID) // GRID) / GRID, dtype=mx.float32)
    columns = mx.array((np.arange(GRID * GRID) % GRID) / GRID, dtype=mx.float32)
    return mx.stack([mx.sum(weight * rows, axis=-1),
                     mx.sum(weight * columns, axis=-1)], axis=-1)


def build(name: str, channels: int, seed: int, language: int = 2):
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(seed)
    side = language + 1 + 4                    # language one-hot, phase, action one-hot

    class OracleCoordinate(nn.Module):
        """Soft-argmax coordinates of EVERY channel, then an MLP. The language has to
        learn which pair of coordinates it names."""

        def __init__(self):
            super().__init__()
            self.a = nn.Linear(2 * channels + side, HIDDEN)
            self.b = nn.Linear(HIDDEN, HIDDEN)
            self.out = nn.Linear(HIDDEN, 1)

        def __call__(self, grid, context):
            n = grid.shape[0]
            flat = grid.reshape(n, GRID * GRID, channels)
            points = mx.concatenate(
                [_soft_argmax(flat[:, :, k], mx) for k in range(channels)], axis=-1)
            h = nn.relu(self.a(mx.concatenate([points, context], axis=-1)))
            return self.out(nn.relu(self.b(h)))

    class SoftArgmaxGoalRelative(nn.Module):
        """A language-weighted channel mixture is soft-argmaxed into a goal coordinate,
        an unconditioned mixture into an agent coordinate, and the DIFFERENCE is what
        reaches the decision. This is the shape phase N's readout could not express: a
        3x3 convolution cannot compare two cells eleven apart."""

        def __init__(self):
            super().__init__()
            self.goal_mix = nn.Linear(language, channels)
            self.agent_mix = nn.Linear(1, channels)
            self.a = nn.Linear(6 + side, HIDDEN)
            self.b = nn.Linear(HIDDEN, HIDDEN)
            self.out = nn.Linear(HIDDEN, 1)

        def __call__(self, grid, context):
            n = grid.shape[0]
            flat = grid.reshape(n, GRID * GRID, channels)
            goal_weight = self.goal_mix(context[:, :language]).reshape(n, 1, channels)
            agent_weight = self.agent_mix(
                mx.ones((n, 1), mx.float32)).reshape(n, 1, channels)
            goal = _soft_argmax(mx.sum(flat * goal_weight, axis=-1), mx)
            agent = _soft_argmax(mx.sum(flat * agent_weight, axis=-1), mx)
            features = mx.concatenate([goal, agent, goal - agent, context], axis=-1)
            h = nn.relu(self.a(features))
            return self.out(nn.relu(self.b(h)))

    class CoordinateQueryAttention(nn.Module):
        """A query built from the language, the action and the phase attends over cells
        carrying their own positional encoding."""

        def __init__(self):
            super().__init__()
            self.key = nn.Linear(channels + 2, HIDDEN)
            self.value = nn.Linear(channels + 2, HIDDEN)
            self.query = nn.Linear(side, HIDDEN)
            self.a = nn.Linear(HIDDEN + side, HIDDEN)
            self.out = nn.Linear(HIDDEN, 1)

        def __call__(self, grid, context):
            n = grid.shape[0]
            flat = grid.reshape(n, GRID * GRID, channels)
            rows = mx.array((np.arange(GRID * GRID) // GRID) / GRID, dtype=mx.float32)
            columns = mx.array((np.arange(GRID * GRID) % GRID) / GRID, dtype=mx.float32)
            grid_position = mx.broadcast_to(
                mx.stack([rows, columns], axis=-1).reshape(1, GRID * GRID, 2),
                (n, GRID * GRID, 2))
            cells = mx.concatenate([flat, grid_position], axis=-1)
            key, value = self.key(cells), self.value(cells)
            query = self.query(context).reshape(n, 1, HIDDEN)
            weight = mx.softmax(mx.sum(key * query, axis=-1) / np.sqrt(HIDDEN), axis=-1)
            pooled = mx.sum(value * weight.reshape(n, GRID * GRID, 1), axis=1)
            h = nn.relu(self.a(mx.concatenate([pooled, context], axis=-1)))
            return self.out(h)

    class SpatialGoalHeatmap(nn.Module):
        """A goal heatmap and an agent heatmap, combined through the action delta.

        The decision is computed as an expected change in Manhattan distance under the
        two heatmaps, with the direction of the step modulated by a learned function of
        the phase -- which is exactly what the environment does.
        """

        def __init__(self):
            super().__init__()
            self.goal = nn.Linear(channels + language, 1)
            self.agent = nn.Linear(channels, 1)
            self.sign = nn.Linear(1, 1)
            self.scale = nn.Linear(2 + 4, 1)

        def __call__(self, grid, context):
            n = grid.shape[0]
            flat = grid.reshape(n, GRID * GRID, channels)
            language_field = mx.broadcast_to(
                context[:, :language].reshape(n, 1, language),
                (n, GRID * GRID, language))
            goal_map = self.goal(
                mx.concatenate([flat, language_field], axis=-1)).reshape(n, -1)
            agent_map = self.agent(flat).reshape(n, -1)
            goal = _soft_argmax(goal_map, mx) * GRID
            agent = _soft_argmax(agent_map, mx) * GRID
            action = context[:, language + 1:]
            deltas = mx.array(np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0],
                                        [0.0, -1.0]], dtype=np.float32))
            step = action @ deltas
            direction = mx.tanh(self.sign(context[:, language:language + 1]))
            landed = agent + step * direction
            before = mx.sum(mx.abs(goal - agent), axis=-1, keepdims=True)
            after = mx.sum(mx.abs(goal - landed), axis=-1, keepdims=True)
            return self.scale(mx.concatenate([before - after, before, action], axis=-1))

    class PooledCoordinateBaseline(nn.Module):
        """Phase N's readout: a small convolution, a global pool, then an MLP. Kept
        exactly so the family contains the arm that failed."""

        def __init__(self):
            super().__init__()
            self.first = nn.Conv2d(channels, HIDDEN, 3, padding=1)
            self.second = nn.Conv2d(HIDDEN, HIDDEN, 3, padding=1)
            self.a = nn.Linear(2 * HIDDEN + side, HIDDEN)
            self.out = nn.Linear(HIDDEN, 1)

        def __call__(self, grid, context):
            n = grid.shape[0]
            h = nn.relu(self.second(nn.relu(self.first(grid))))
            h = h.reshape(n, GRID * GRID, HIDDEN)
            pooled = mx.concatenate([mx.max(h, axis=1), mx.mean(h, axis=1)], axis=-1)
            return self.out(nn.relu(self.a(mx.concatenate([pooled, context], axis=-1))))

    model = {"1_oracle_coordinate": OracleCoordinate,
             "2_soft_argmax_goal_relative": SoftArgmaxGoalRelative,
             "3_coordinate_query_attention": CoordinateQueryAttention,
             "4_spatial_goal_heatmap": SpatialGoalHeatmap,
             "5_pooled_coordinate_baseline": PooledCoordinateBaseline}[name]()
    mx.eval(model.parameters())
    return model


def train(name: str, grid: np.ndarray, context: np.ndarray, target: np.ndarray,
          seed: int, updates: int = UPDATES):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    model = build(name, grid.shape[-1], seed)
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)
    rng = np.random.default_rng(seed)
    x, c, y = mx.array(grid), mx.array(context), mx.array(target)
    for _ in range(updates):
        pick = mx.array(rng.integers(0, len(target), min(BATCH, len(target))))

        def objective(m):
            return mx.mean(nn.losses.binary_cross_entropy(
                m(x[pick], c[pick])[:, 0], y[pick], with_logits=True))

        value, grads = nn.value_and_grad(model, objective)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, value)

    def infer(grid_in, context_in, batch: int = 1024):
        out = []
        for start in range(0, len(grid_in), batch):
            span = slice(start, start + batch)
            value = model(mx.array(grid_in[span]), mx.array(context_in[span]))
            mx.eval(value)
            out.append(np.asarray(value)[:, 0])
        return np.concatenate(out)
    return infer
