"""D / H. The shared head, the eight public auxiliary targets, and the input controls.

One head architecture, one parameter budget, one optimizer, one update count, one
checkpoint rule -- for every interface and every target. The head is deliberately small
and identical so that a difference between interfaces is a difference between their
slots.

Section D asks for direct event prediction to be reported SEPARATELY from event derived
through predicted masks and displacement, because those are different claims: the first
says a head can be trained to call the event, the second says perception recovered the
public quantities the event is defined from. The second is the one that transfers.

    imported by n_pathway.py; runnable directly for the auxiliary-head report
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import n_core as core
from n_core import GRID, N_DISPLACEMENT, PairBatch
from n_interfaces import SLOT_WIDTH, HEAD_HIDDEN, UPDATES, BATCH, LEARNING_RATE

REDUCED = 16
PARAMETER_CEILING = 250_000

# Section D targets. `kind` decides the loss and the metric.
# `identifiable` names the subset on which the target is a function of the two frames.
# Two of the eight are NOT public everywhere and saying so is part of the result:
#   - entered_cell_switch is invisible when the agent did not move, because it occludes
#     its own cell in the before frame AND the after frame, so nothing in the pair says
#     what is underneath;
#   - reset_stripe_state is only rendered on the reset frame, so it is unidentifiable
#     from any pair whose `before` is not the reset frame.
# Both are reported unconditioned AND on their identifiable subset. Scoring them only
# unconditioned would report an identifiability fact as a perception failure.
# `spatial_scalar` targets are supervised as a 144-cell map and read out by a spatial
# max. The event is an existential over cells -- "some cell was entered and was a
# switch" -- and supervising it as a bare scalar behind a max gives gradient to exactly
# one location per example. Measured: a scalar-supervised event head could not fit even
# the TRAINING set (balanced accuracy 0.5301 at 2500 updates, 0.5788 at 8000), while the
# identical head supervised spatially reaches 1.0000 on train and held-out alike. Every
# interface number taken before that change was measuring the readout, not the interface.
TARGETS = {
    "1_agent_mask_before": ("agent_before", "multilabel", GRID * GRID, None),
    "2_agent_mask_after": ("agent_after", "multilabel", GRID * GRID, None),
    "3_displacement": ("displacement", "categorical", N_DISPLACEMENT, None),
    "4_switch_mask_before": ("switch_before", "multilabel", GRID * GRID, None),
    "5_entered_cell_switch": ("entered_map", "spatial_scalar", GRID * GRID, "moved"),
    "6_retrospective_event": ("event_map", "spatial_scalar", GRID * GRID, None),
    "7_reset_stripe_state": ("stripe", "binary", 1, "reset_pair"),
    "8_is_reset_pair": ("is_reset_pair", "binary", 1, None),
}

SCALAR_OF = {"entered_map": "entered", "event_map": "event"}

HEADLINE = {"multilabel": "f1", "categorical": "accuracy", "binary": "balanced_accuracy",
            "spatial_scalar": "balanced_accuracy"}


def identifiable_mask(name, pairs) -> np.ndarray:
    if name is None:
        return np.ones(len(pairs), dtype=bool)
    if name == "moved":
        return pairs.displacement < (N_DISPLACEMENT - 1)
    if name == "reset_pair":
        return pairs.is_reset_pair > 0.5
    raise KeyError(name)


def nearest_index(source: int, target: int) -> np.ndarray:
    """Index map for a parameterless nearest-neighbour resize of a PREDICTION.

    This resizes an output, not a feature map: an 8x8 interface is allowed to be less
    accurate about which of 144 cells it means, but it is not allowed to have its
    features interpolated up and then be scored as though it had that resolution.
    """
    return np.clip(((np.arange(target) + 0.5) * source / target).astype(int), 0, source - 1)


def build_head(out_dim: int, seed: int, kind: str, action_dim: int = 4):
    """One head shape for every interface and every grid.

    Fully convolutional, so the parameter count does not depend on the slot grid -- an
    8x8 backbone and a 12x12 CNN get literally the same head. The 3x3 layer exists
    because the event is a RELATION between two nearby cells (the agent left one and
    entered another); a 1x1-only head cannot express it, and a flatten-then-MLP head
    can only express it by memorising absolute positions, which is exactly what the
    structured phase showed destroys held-out transfer.

    Scalar targets are reduced by a spatial MAX rather than a mean: "a switch was
    entered somewhere" is an existential over locations, and averaging it dilutes a
    single positive cell by the grid size.
    """
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(seed)
    spatial = kind in ("multilabel", "spatial_scalar")

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.reduce = nn.Conv2d(SLOT_WIDTH, REDUCED, 1)
            self.mix = nn.Conv2d(REDUCED + action_dim, HEAD_HIDDEN, 3, padding=1)
            self.out = nn.Conv2d(HEAD_HIDDEN, 1 if spatial else out_dim, 1)

        def __call__(self, slots, action):
            h = nn.relu(self.reduce(slots))
            n, height, width, _ = h.shape
            broadcast = mx.broadcast_to(
                action.reshape(n, 1, 1, action.shape[-1]), (n, height, width, action.shape[-1]))
            h = nn.relu(self.mix(mx.concatenate([h, broadcast], axis=-1)))
            z = self.out(h)
            if spatial:
                rows = mx.array(nearest_index(height, GRID))
                columns = mx.array(nearest_index(width, GRID))
                z = mx.take(mx.take(z, rows, axis=1), columns, axis=2)
                return z.reshape(n, GRID * GRID)
            # Max AND mean. A bare spatial max is an existential -- good for "a switch
            # was entered somewhere", useless for "the agent did not move at all", which
            # is the absence of evidence everywhere. Both reductions are grid-agnostic.
            # Max over space, plus the spatial mean as an additive term. A bare max
            # is an existential -- right for "a switch was entered somewhere", blind to
            # "nothing happened anywhere", which is the no-move class. Adding the mean
            # restores that without a new layer: an extra Linear on top of an unbounded
            # post-max activation collapsed every head to a constant, and collapsed it
            # to the SAME constant for three very different interfaces, which is what
            # gave the bug away.
            flat = z.reshape(n, height * width, z.shape[-1])
            return mx.max(flat, axis=1) + mx.mean(flat, axis=1)

    model = Head()
    mx.eval(model.parameters())
    return model


def parameter_count(model) -> int:
    from mlx.utils import tree_flatten
    return int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))


@dataclass
class HeadResult:
    target: str
    kind: str
    metrics: dict[str, float]
    parameters: int


def _loss(kind: str):
    import mlx.core as mx
    import mlx.nn as nn

    if kind == "categorical":
        return lambda logits, y: nn.losses.cross_entropy(logits, y, reduction="mean")
    if kind == "spatial_scalar":
        return lambda logits, y: mx.mean(
            nn.losses.binary_cross_entropy(logits, y, with_logits=True))
    if kind == "binary":
        return lambda logits, y: mx.mean(
            nn.losses.binary_cross_entropy(logits[:, 0], y, with_logits=True))
    return lambda logits, y: mx.mean(
        nn.losses.binary_cross_entropy(logits, y, with_logits=True))


def train_target(slots_train: np.ndarray, action_train: np.ndarray, y_train: np.ndarray,
                 kind: str, out_dim: int, seed: int, updates: int = UPDATES,
                 trainable_encoder=None):
    """Train one head. `trainable_encoder` is passed for interfaces whose encoder is
    itself trainable, so its parameters are optimised jointly and counted."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    model = build_head(out_dim, seed, kind)
    count = parameter_count(model)
    assert count <= PARAMETER_CEILING, (out_dim, count)
    loss_fn = _loss(kind)
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE)
    rng = np.random.default_rng(seed)
    x = mx.array(slots_train)
    a = mx.array(action_train)
    y = mx.array(y_train.astype(np.int32) if kind == "categorical"
                 else y_train.astype(np.float32))
    for _ in range(updates):
        pick = rng.integers(0, len(slots_train), min(BATCH, len(slots_train)))
        idx = mx.array(pick)
        xb, ab, yb = x[idx], a[idx], y[idx]

        def objective(m):
            return loss_fn(m(xb, ab), yb)

        value, grads = nn.value_and_grad(model, objective)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, value)
    return model, count


def predict(model, slots: np.ndarray, action: np.ndarray, batch: int = 1024) -> np.ndarray:
    import mlx.core as mx
    out = []
    for start in range(0, len(slots), batch):
        block = model(mx.array(slots[start:start + batch]),
                      mx.array(action[start:start + batch]))
        mx.eval(block)
        out.append(np.asarray(block))
    return np.concatenate(out)


# ---- metrics -----------------------------------------------------------------------


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -50, 50)))


def mask_metrics(logits: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    predicted = sigmoid(logits) >= 0.5
    actual = truth >= 0.5
    intersection = float((predicted & actual).sum())
    union = float((predicted | actual).sum())
    tp = intersection
    fp = float((predicted & ~actual).sum())
    fn = float((~predicted & actual).sum())
    precision = tp / max(tp + fp, 1e-9)
    recall = tp / max(tp + fn, 1e-9)
    # Exact-cell accuracy: for a one-hot target, did the argmax land on the true cell?
    exact = float((logits.argmax(axis=1) == truth.argmax(axis=1)).mean()) \
        if truth.sum(axis=1).max() <= 1.0 + 1e-6 else float("nan")
    return {"f1": float(2 * precision * recall / max(precision + recall, 1e-9)),
            "iou": float(intersection / max(union, 1e-9)),
            "precision": float(precision), "recall": float(recall),
            "exact_cell_accuracy": exact}


def binary_metrics(logits: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    probability = sigmoid(logits[:, 0])
    predicted = (probability >= 0.5).astype(int)
    actual = truth.astype(int)
    positive, negative = actual == 1, actual == 0
    recall = float(predicted[positive].mean()) if positive.any() else float("nan")
    specificity = float(1 - predicted[negative].mean()) if negative.any() else float("nan")
    chosen = predicted == 1
    precision = float(actual[chosen].mean()) if chosen.any() else 0.0
    denominator = precision + recall
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    bins = np.clip((probability * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        picked = bins == b
        if picked.any():
            ece += picked.mean() * abs(float(probability[picked].mean())
                                       - float(actual[picked].mean()))
    return {"accuracy": float((predicted == actual).mean()),
            "balanced_accuracy": float((recall + specificity) / 2),
            "precision": precision, "recall": recall,
            "f1": float(2 * precision * recall / denominator) if denominator else 0.0,
            "brier": float(((probability - actual) ** 2).mean()),
            "nll": float(-(actual * np.log(p) + (1 - actual) * np.log(1 - p)).mean()),
            "expected_calibration_error": float(ece),
            "positive_rate": float(actual.mean()), "n": int(len(actual))}


def categorical_metrics(logits: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    predicted = logits.argmax(axis=1)
    return {"accuracy": float((predicted == truth).mean()),
            "n": int(len(truth))}


def score(kind: str, logits: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if kind == "multilabel":
        return mask_metrics(logits, truth)
    if kind == "spatial_scalar":
        # The map is a means; the scalar existential is the quantity of interest.
        block = binary_metrics(logits.max(axis=1)[:, None], (truth.max(axis=1) > 0.5))
        block["map_f1"] = mask_metrics(logits, truth)["f1"]
        return block
    if kind == "binary":
        return binary_metrics(logits, truth)
    return categorical_metrics(logits, truth)


# ---- section H input controls ---------------------------------------------------------


def apply_control(name: str, slots: np.ndarray, action: np.ndarray, pairs: PairBatch,
                  encode: Callable[[PairBatch], np.ndarray], seed: int = 31
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Return (slots, action) under one control. Frame controls re-encode from a
    modified pair so the ablation is in the INPUT, not a mask over the features."""
    rng = np.random.default_rng(seed)
    if name == "correct":
        return slots, action
    if name == "action_only":
        return np.zeros_like(slots), action
    if name == "frame_before_only":
        modified = pairs.subset(np.ones(len(pairs), bool))
        modified.after = modified.before.copy()
        return encode(modified), action
    if name == "frame_after_only":
        modified = pairs.subset(np.ones(len(pairs), bool))
        modified.before = modified.after.copy()
        return encode(modified), action
    if name == "no_action":
        return slots, np.zeros_like(action)
    if name == "shuffled_action":
        return slots, action[rng.permutation(len(action))]
    if name == "shuffled_frames":
        modified = pairs.subset(rng.permutation(len(pairs)))
        return encode(modified), action
    raise KeyError(name)


CONTROLS = ("correct", "action_only", "frame_before_only", "frame_after_only",
            "no_action", "shuffled_action", "shuffled_frames")


def derived_event(agent_after_logits: np.ndarray, switch_logits: np.ndarray,
                  agent_before_logits: np.ndarray) -> np.ndarray:
    """Section D: the event DERIVED from predicted masks and displacement.

    C = 1 iff the predicted displacement is a real move AND the cell the agent moved
    into was predicted to be a switch in the BEFORE frame. This never sees the event
    label; it is the composition of three other heads, which is why it is reported apart
    from the directly-trained event head.
    """
    before_cell = agent_before_logits.argmax(axis=1)
    after_cell = agent_after_logits.argmax(axis=1)
    switch_probability = sigmoid(switch_logits)
    moved = before_cell != after_cell
    # Displacement is READ OFF the two predicted agent masks rather than taken from the
    # separate displacement head. The masks reach f1 1.0000 held out while the direct
    # categorical head sits at 0.41-0.71, so routing the derivation through the weaker
    # head would have measured that head rather than the composition.
    return (moved.astype(np.float32)
            * switch_probability[np.arange(len(after_cell)), after_cell])
