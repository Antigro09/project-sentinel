"""Shared models: one stateless binder, one persistent-memory binder, one budget.

Both consume the SAME per-colour token stream and both are supervised by the PUBLIC
event label alone -- no role name, no palette id, no evaluator state. The only
difference between them is whether anything survives from one frame pair to the next,
which is exactly the quantity section G is trying to measure. Matching everything else
is what makes that difference attributable.

The event is not predicted by a free head. It is the M2F relational expression lifted
onto the soft assignment,

    event = max over cells of  P(agent here now) * P(not agent here before)
                               * P(switch here before)

so the only way to move the loss is to move the assignment. A free head could fit the
event while leaving the assignment arbitrary, and section D's audit would then be
measuring nothing.

    imported by o2_leakage.py, o2_assignment.py, o2_factorial.py, o2_memory.py,
    o2_route.py, o2_unresolved.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import o2_core as C
from o2_core import (COUNT, GLOBAL, GRID, INTERACT, MAX_COLOURS, MOMENTS, MOTION,
                     N_ROLES, RGB, TOKEN_WIDTH)

WIDTH = 64
UPDATES = 2_000
BATCH = 128
LEARNING_RATE = 3e-3

# GLOBAL -- the action, the displacement and the moved flag -- is in EVERY view. It is
# the same vector for every colour, so it can never on its own say which colour is which
# and it cannot confound the factorial; withholding it from some cells would instead
# make those cells fail for a reason unrelated to the factor being varied. The three
# factors that actually vary are COUNT, MOTION and INTERACT, and all eight of their
# combinations are present, so main effects and interactions are exact rather than
# inferred from a partial design.
FACTORS = ("COUNT", "MOTION", "INTERACT")
VIEWS: dict[str, list[slice]] = {
    "none": [GLOBAL],
    "count_only": [COUNT, GLOBAL],
    "motion_only": [MOTION, GLOBAL],
    "moments_only": [MOMENTS, GLOBAL],
    "interaction_only": [INTERACT, GLOBAL],
    "count_plus_motion": [COUNT, MOTION, GLOBAL],
    "motion_plus_interaction": [MOTION, INTERACT, GLOBAL],
    "count_plus_interaction": [COUNT, INTERACT, GLOBAL],
    "count_motion_interaction": [COUNT, MOTION, INTERACT, GLOBAL],
    "full_token": [RGB, COUNT, MOMENTS, MOTION, INTERACT, GLOBAL],
    # Not part of the factorial: used by the leakage audit, where a view WITHOUT the raw
    # colour value must be invariant to any relabelling of the palette.
    "no_rgb": [COUNT, MOMENTS, MOTION, INTERACT, GLOBAL],
    # A single frame: no motion, no interaction, no action. Section G's arm 1.
    "single_frame": [RGB, COUNT, MOMENTS],
}
FACTORIAL_CELL = {
    (0, 0, 0): "none", (1, 0, 0): "count_only", (0, 1, 0): "motion_only",
    (0, 0, 1): "interaction_only", (1, 1, 0): "count_plus_motion",
    (0, 1, 1): "motion_plus_interaction", (1, 0, 1): "count_plus_interaction",
    (1, 1, 1): "count_motion_interaction",
}


def mask_view(tokens: np.ndarray, view: str) -> np.ndarray:
    out = np.zeros_like(tokens)
    for part in VIEWS[view]:
        out[..., part] = tokens[..., part]
    return out


def relational_event(role_before, role_after, before_index, after_index):
    """The M2F expression on soft roles. Returns a logit."""
    import mlx.core as mx

    n = before_index.shape[0]
    fan = mx.ones((1, 1, N_ROLES), mx.int32)
    pb = mx.take_along_axis(
        role_before, before_index.reshape(n, GRID * GRID, 1).astype(mx.int32) * fan,
        axis=1)
    pa = mx.take_along_axis(
        role_after, after_index.reshape(n, GRID * GRID, 1).astype(mx.int32) * fan,
        axis=1)
    agent_now = pa[:, :, C.AGENT]
    agent_before = pb[:, :, C.AGENT]
    switch_before = pb[:, :, C.SWITCH]
    evidence = mx.max(agent_now * (1.0 - agent_before) * switch_before, axis=1)
    return mx.log(mx.maximum(evidence, 1e-6) / mx.maximum(1.0 - evidence, 1e-6))


def build_stateless(seed: int, width: int = WIDTH):
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(seed)

    class Stateless(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(TOKEN_WIDTH, width)
            self.b = nn.Linear(2 * width, width)
            self.role = nn.Linear(width, N_ROLES)

        def assign(self, tokens):
            h = nn.relu(self.a(tokens))
            context = mx.broadcast_to(mx.mean(h, axis=1, keepdims=True), h.shape)
            h = nn.relu(self.b(mx.concatenate([h, context], axis=-1)))
            return mx.softmax(self.role(h), axis=-1)

        def __call__(self, tokens, before_index, after_index):
            # ONE assignment per frame pair, used for both endpoints. O1 assigned the
            # two frames independently, which let the model call the same colour AGENT
            # before and SWITCH after; the event expression then had a degree of freedom
            # that no role binding should have. It also makes the stateless arm and the
            # memory arm differ in exactly one thing -- persistence -- and nothing else.
            roles = self.assign(tokens)
            return relational_event(roles, roles, before_index, after_index)

    model = Stateless()
    mx.eval(model.parameters())
    return model


def build_memory(seed: int, width: int = WIDTH):
    """Persistent explicit assignment memory.

    A per-colour recurrent state is carried across the whole calibration segment and is
    READ OUT as a colour-to-role assignment that the transfer step then reuses. The
    colours are addressed by their public RGB value through a registry, so the memory
    survives a change of layout; nothing else does.
    """
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(seed)

    class Memory(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(TOKEN_WIDTH, width)
            self.b = nn.Linear(2 * width, width)
            self.cell = nn.GRU(width, width)
            self.role = nn.Linear(3 * width, N_ROLES)

        def embed(self, tokens):
            """(N, K, TOKEN_WIDTH) -> (N, K, W), permutation-equivariant in K."""
            h = nn.relu(self.a(tokens))
            context = mx.broadcast_to(mx.mean(h, axis=1, keepdims=True), h.shape)
            return nn.relu(self.b(mx.concatenate([h, context], axis=-1)))

        def integrate(self, sequence, mask):
            """sequence (N, T, K, TOKEN_WIDTH), mask (N, T) -> state (N, K, W).

            The state is the LAST hidden state, not a masked mean over time. A mean
            spreads thirty-odd steps evenly and drowns the two or three that carry the
            crossing evidence; measured, it left the memory arm at 0.6375 on contested
            rows against 0.5500 memoryless, with the interval spanning zero. Masked-out
            steps are zeroed at the input instead, so a reset memory really does see
            nothing.
            """
            n, t, k, _ = sequence.shape
            sequence = sequence * mask.reshape(n, t, 1, 1)
            flat = self.embed(sequence.reshape(n * t, k, TOKEN_WIDTH))
            flat = flat.reshape(n, t, k, -1).transpose(0, 2, 1, 3).reshape(n * k, t, -1)
            hidden = self.cell(flat)                       # (N*K, T, W)
            # Last state, plus a max and a mean over time. The last state alone has to
            # carry evidence across thirty-odd steps through a gate that is free to
            # forget it; the two pooled paths give the crossing evidence a route to the
            # readout that does not depend on the cell choosing to keep it.
            pooled = mx.concatenate([hidden[:, -1], mx.max(hidden, axis=1),
                                     mx.mean(hidden, axis=1)], axis=-1)
            return pooled.reshape(n, k, -1)

        def assignment(self, sequence, mask):
            return mx.softmax(self.role(self.integrate(sequence, mask)), axis=-1)

        def __call__(self, sequence, mask, before_index, after_index):
            roles = self.assignment(sequence, mask)
            return relational_event(roles, roles, before_index, after_index)

    model = Memory()
    mx.eval(model.parameters())
    return model


def _train(model, tensors, target, forward, seed: int, updates: int = UPDATES):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)
    rng = np.random.default_rng(seed)
    arrays = [mx.array(x) for x in tensors]
    y = mx.array(target)
    rows = len(target)
    for _ in range(updates):
        pick = mx.array(rng.integers(0, rows, min(BATCH, rows)))
        batch = [t[pick] for t in arrays]
        label = y[pick]

        def objective(m):
            return mx.mean(nn.losses.binary_cross_entropy(
                forward(m, batch), label, with_logits=True))

        value, grads = nn.value_and_grad(model, objective)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, value)
    return model


def train_stateless(train, seed: int, updates: int = UPDATES):
    """train = (tokens, before_index, after_index, event)."""
    import mlx.core as mx

    model = build_stateless(seed)
    _train(model, train[:3], train[3], lambda m, b: m(*b), seed, updates)

    def infer(block, batch: int = 2048):
        out = []
        for start in range(0, len(block[0]), batch):
            span = slice(start, start + batch)
            value = model(*[mx.array(x[span]) for x in block[:3]])
            mx.eval(value)
            out.append(np.asarray(value))
        return np.concatenate(out) if out else np.zeros(0, np.float32)
    return infer, model


def training_loss(model, train, forward, batch: int = 512) -> float:
    """Mean training-set loss. The selection criterion, and it touches no test row."""
    import mlx.core as mx
    import mlx.nn as nn

    total, rows = 0.0, len(train[-1])
    for start in range(0, rows, batch):
        span = slice(start, start + batch)
        block = [mx.array(x[span]) for x in train[:-1]]
        value = mx.mean(nn.losses.binary_cross_entropy(
            forward(model, block), mx.array(train[-1][span]), with_logits=True))
        mx.eval(value)
        total += float(value) * len(train[-1][span])
    return total / max(rows, 1)


def train_memory(train, seed: int, updates: int = UPDATES, restarts: int = 4):
    """train = (sequence, mask, before_index, after_index, event).

    Restarts are selected by TRAINING loss alone, the M2F rule. Measured without it, the
    contested-row gain over a memoryless binder was +0.2759 [-0.0244, +0.5127] across
    three seeds: the point estimate was large and the interval spanned zero, because
    some initialisations discover "entered, then the displacement sign flipped" and
    others never do. Selecting on the training objective is answer-free and turns a
    high-variance procedure into a reproducible one.
    """
    import mlx.core as mx

    ledger, best, best_loss = [], None, float("inf")
    for restart in range(restarts):
        candidate = build_memory(seed * 100 + restart)
        _train(candidate, train[:4], train[4], lambda m, b: m(*b),
               seed * 100 + restart, updates)
        loss = training_loss(candidate, train, lambda m, b: m(*b))
        ledger.append({"restart": restart, "training_loss": loss})
        if loss < best_loss:
            best, best_loss = candidate, loss
    model = best
    for row in ledger:
        row["selected"] = bool(row["training_loss"] == best_loss)

    def infer(block, batch: int = 256):
        out = []
        for start in range(0, len(block[0]), batch):
            span = slice(start, start + batch)
            value = model(*[mx.array(x[span]) for x in block[:4]])
            mx.eval(value)
            out.append(np.asarray(value))
        return np.concatenate(out) if out else np.zeros(0, np.float32)
    infer.restart_ledger = ledger
    return infer, model


def assignment_of(model, tokens: np.ndarray, batch: int = 2048) -> np.ndarray:
    import mlx.core as mx

    out = []
    for start in range(0, len(tokens), batch):
        value = model.assign(mx.array(tokens[start:start + batch]))
        mx.eval(value)
        out.append(np.asarray(value))
    return np.concatenate(out)


def memory_assignment_of(model, sequence: np.ndarray, mask: np.ndarray,
                         batch: int = 256) -> np.ndarray:
    import mlx.core as mx

    out = []
    for start in range(0, len(sequence), batch):
        span = slice(start, start + batch)
        value = model.assignment(mx.array(sequence[span]), mx.array(mask[span]))
        mx.eval(value)
        out.append(np.asarray(value))
    return np.concatenate(out)


def balanced_accuracy(logits: np.ndarray, truth: np.ndarray) -> float:
    predicted = (logits > 0).astype(float)
    out = []
    for value in (0.0, 1.0):
        mask = truth == value
        if mask.any():
            out.append(float((predicted[mask] == value).mean()))
    return float(np.mean(out)) if out else float("nan")
