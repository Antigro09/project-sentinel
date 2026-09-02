"""B. Why does the 2-state filter collapse on one parity seed?

The collapse was reported as a rate. It needs a mechanism, because the mechanism
decides whether the fix is an initialization rule (cheap, legitimate) or a change
to the model class (which would be scaling by another name).

Hypothesis to test: the two latent states become behaviourally identical. A
two-state filter whose transition matrix maps both events to the same row
distribution has no way to represent "the phase flipped", and the belief carries no
information no matter how long the sequence.

Reports, per seed: latent occupancy, the learned transition matrices, state
entropy, gradient norms, the initial logits, and a direct test of whether the two
states became interchangeable.

    .venv-shwm/bin/python experiments/shwm/collapse_diagnosis.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity_exhaustive import TRAIN_LENGTHS, encode, enumerate_sequences  # noqa: E402
from parity_microcase import build  # noqa: E402

SEEDS = (6600, 6601, 6602)
UPDATES = 1024


def train_and_probe(kind, seed, pairs, max_length):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(seed)
    model = build(kind, 3)
    mx.eval(model.parameters())
    initial_logits = np.asarray(model.logits) if hasattr(model, "logits") else None
    x, y, mask = encode(pairs, max_length)
    optimizer = optim.AdamW(learning_rate=3e-3)
    rng = np.random.default_rng(seed)
    gradient_norms = []
    for step in range(UPDATES):
        pick = rng.integers(0, len(x), 64)
        xb, yb, mb = mx.array(x[pick]), mx.array(y[pick]), mx.array(mask[pick])

        def loss_fn(m):
            logits = m(xb)
            losses = nn.losses.cross_entropy(
                logits.reshape(-1, 2), yb.reshape(-1), reduction="none")
            return (losses * mb.reshape(-1)).sum() / mb.sum()

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        if step % 128 == 0 and "logits" in grads:
            gradient_norms.append(float(np.abs(np.asarray(grads["logits"])).max()))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    logits = model(mx.array(x))
    mx.eval(logits)
    predicted = np.asarray(logits).argmax(axis=-1)
    valid = mask.astype(bool)
    accuracy = float((predicted[valid] == y[valid]).mean())

    record: dict[str, Any] = {"arm": kind, "seed": seed, "accuracy": accuracy,
                             "gradient_norms_over_training": gradient_norms}
    if hasattr(model, "logits"):
        transition = np.asarray(mx.softmax(model.logits, axis=-1))
        record["transition_event_0"] = transition[0].tolist()
        record["transition_event_1"] = transition[1].tolist()
        # Are the two events behaviourally the same map? If so the belief cannot
        # register a flip and the filter is structurally blind to parity.
        record["event_maps_identical_l1"] = float(np.abs(transition[0] - transition[1]).sum())
        # Are the two STATES interchangeable? Compare each event map's rows.
        record["state_rows_identical_l1"] = float(
            np.abs(transition[0][0] - transition[0][1]).sum()
            + np.abs(transition[1][0] - transition[1][1]).sum())
        entropy = -(transition * np.log(np.maximum(transition, 1e-12))).sum(axis=-1)
        record["transition_row_entropy"] = entropy.tolist()
        record["initial_logits_spread"] = float(np.abs(initial_logits).max())

        # latent occupancy over the validation sweep
        beliefs = []
        belief = None
        import mlx.core as mxx
        z = mxx.array(x[:512])
        b = mxx.softmax(model.initial(z[:, 0]), axis=-1)
        t_matrix = mxx.softmax(model.logits, axis=-1)
        for t in range(z.shape[1]):
            e = z[:, t, 0][:, None, None]
            matrix = t_matrix[0][None] * (1.0 - e) + t_matrix[1][None] * e
            b = mxx.sum(b[:, :, None] * matrix, axis=1)
            b = b / mxx.maximum(mxx.sum(b, axis=-1, keepdims=True), 1e-9)
            beliefs.append(b)
        stacked = np.asarray(mxx.stack(beliefs, axis=1))
        record["mean_latent_occupancy"] = stacked.reshape(-1, stacked.shape[-1]).mean(
            axis=0).tolist()
        entropy_value = float(
            -(stacked * np.log(np.maximum(stacked, 1e-12))).sum(axis=-1).mean())
        states = stacked.shape[-1]
        record["mean_belief_entropy"] = entropy_value
        # Normalise by ln(states). An absolute threshold is meaningless across state
        # counts: 0.65 nats is near-maximal for two states and near-minimal for eight,
        # and an earlier version using the raw value labelled all three 8-state seeds
        # collapsed while they scored 1.000000.
        record["normalised_belief_entropy"] = entropy_value / float(np.log(states))
        record["collapsed"] = bool(record["normalised_belief_entropy"] > 0.9)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/collapse-diagnosis.json")
    arguments = parser.parse_args()
    pairs = list(enumerate_sequences(TRAIN_LENGTHS))
    max_length = max(TRAIN_LENGTHS)
    print(f"diagnosing on {len(pairs)} exhaustive trained-length pairs\n")

    rows = []
    for kind in ("two_state_filter", "eight_state_filter"):
        for seed in SEEDS:
            record = train_and_probe(kind, seed, pairs, max_length)
            rows.append(record)
            print(f"{kind:20s} seed {seed}  acc {record['accuracy']:.6f}  "
                  f"collapsed={record.get('collapsed')}", flush=True)
            if kind == "two_state_filter":
                print(f"    transition | event 0: {np.round(record['transition_event_0'], 3).tolist()}")
                print(f"               | event 1: {np.round(record['transition_event_1'], 3).tolist()}")
                print(f"    L1(event maps differ) {record['event_maps_identical_l1']:.4f}   "
                      f"L1(state rows differ) {record['state_rows_identical_l1']:.4f}")
                print(f"    belief entropy {record['mean_belief_entropy']:.4f}   "
                      f"occupancy {np.round(record['mean_latent_occupancy'], 3).tolist()}")
                print(f"    grad |max| over training: "
                      f"{[round(g, 6) for g in record['gradient_norms_over_training'][:5]]}")

    report = {"rows": rows}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(f"\nwrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
