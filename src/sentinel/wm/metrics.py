"""Diagnostics that are reported alongside the loss and never instead of it.

Two of these exist specifically because a good loss number can hide the failure
underneath it. `rollout_divergence_by_horizon` is the Lemma-3 check -- one-step
error times a sensitivity above one still diverges, so a single averaged number
says nothing about a rollout. `code_utilisation` is the codebook-collapse check
for the discrete arm, which can reach a fine reconstruction loss while using
four of a thousand codes.

`action_effect_discrimination` is the metric the action-intervention fixture
implies at scale: how often the model's predicted successors for two different
actions from the same state are further apart than the noise between two
predictions for the same action.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import mlx.core as mx
from mlx.utils import tree_flatten


def rollout_divergence_by_horizon(
    model, output, actions: mx.array, horizons: Sequence[int]
) -> dict[int, float]:
    """Imagined-versus-observed latent error at each requested depth."""
    from sentinel.wm.objective import multistep_prediction

    depth = max(horizons)
    pairs = multistep_prediction(model, output, actions, depth)
    result: dict[int, float] = {}
    for horizon in horizons:
        if horizon <= len(pairs):
            predicted, target = pairs[horizon - 1]
            error = mx.mean((predicted.astype(mx.float32) - target.astype(mx.float32)) ** 2)
            mx.eval(error)
            result[horizon] = float(error.item())
    return result


def fitted_sensitivity(divergence: Mapping[int, float]) -> float:
    """The `L` of the rollout recurrence, read off consecutive horizons.

    Reported rather than assumed. Below one the error is bounded; at one it is
    linear in the horizon; above one it compounds, and a low one-step loss buys
    nothing at depth 25.
    """
    horizons = sorted(divergence)
    ratios = [
        (divergence[b] / divergence[a]) ** 0.5
        for a, b in zip(horizons, horizons[1:])
        if divergence[a] > 0
    ]
    return float(sum(ratios) / len(ratios)) if ratios else float("nan")


def event_accuracy_and_coverage(
    event_logits: mx.array, targets: mx.array, abstain_index: int
) -> dict[str, float]:
    """Accuracy on committed predictions, and how often the head committed.

    Kept apart for the reason `contract.py` gives about abstention: a head that
    always answers `UNKNOWN_EVENT` is never wrong.
    """
    predicted = mx.argmax(event_logits, axis=-1)
    committed = predicted != abstain_index
    correct = (predicted == targets) & committed
    total = int(predicted.size)
    committed_count = int(mx.sum(committed).item())
    correct_count = int(mx.sum(correct).item())
    return {
        "event_coverage": committed_count / total if total else 0.0,
        "event_accuracy": correct_count / committed_count if committed_count else float("nan"),
        "event_committed": float(committed_count),
    }


def action_effect_discrimination(
    model, output, actions: mx.array, action_count: int
) -> float:
    """Fraction of states where changing the action moves the prediction more
    than re-predicting under the same action does.

    A model that scores at chance here has not learned that its actions matter,
    whatever its one-step loss says.
    """
    belief = mx.stop_gradient(output.belief)
    predictions = []
    for action in range(action_count):
        vectors = model.action_embedding(mx.full(actions.shape, action, dtype=actions.dtype))
        predictions.append(model.head_next_latent(model.dynamics(belief, vectors)))
    stacked = mx.stack(predictions).astype(mx.float32)  # (A, B, T, W)
    mean = mx.mean(stacked, axis=0, keepdims=True)
    between = mx.mean((stacked - mean) ** 2)
    within = mx.mean(mx.var(stacked, axis=-1))
    mx.eval(between, within)
    denominator = float(between.item()) + float(within.item())
    return float(between.item()) / denominator if denominator > 0 else 0.0


def code_utilisation(code_logits: mx.array | None, groups: int, categories: int) -> dict[str, float]:
    """How much of the codebook the discrete arm actually uses.

    Counted on the host with a bincount rather than with a loop of device ops:
    at 32 groups of 32 categories the loop version costs a thousand kernel
    launches per logging step, which would show up in the throughput report as a
    property of the model rather than of the instrumentation.
    """
    if code_logits is None:
        return {}
    import numpy as np

    grouped = code_logits.reshape(code_logits.shape[:-1] + (groups, categories))
    chosen = np.asarray(mx.argmax(grouped, axis=-1).reshape(-1, groups))
    used = 0
    entropy_total = 0.0
    for group in range(groups):
        counts = np.bincount(chosen[:, group], minlength=categories).astype(np.float64)
        used += int((counts > 0).sum())
        total = counts.sum()
        if total > 0:
            probabilities = counts / total
            positive = probabilities[probabilities > 0]
            entropy_total += float(-(positive * np.log(positive)).sum())
    return {
        "code_utilisation": used / (groups * categories),
        "code_entropy_nats": entropy_total / groups,
    }


def gradient_global_norm(gradients: Any) -> float:
    total = mx.array(0.0, dtype=mx.float32)
    for _, value in tree_flatten(gradients):
        total = total + mx.sum(value.astype(mx.float32) ** 2)
    mx.eval(total)
    return float(mx.sqrt(total).item())
