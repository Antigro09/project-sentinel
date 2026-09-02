"""D / T3. Every binary sequence, lengths 1-16, both initial phases.

Sampling said all three temporal arms reach 1.0000. That is a claim about the
sampled sequences. The space is small enough to check completely -- 262,140
(sequence, initial phase) pairs -- so the claim can be made exact rather than
statistical, and any structured pocket of failure becomes visible by length, by
event count and by event position.

Training uses lengths 1-8 exhaustively (1,020 pairs). Lengths 9-16 are never
trained on and are the extrapolation set.

    .venv-shwm/bin/python experiments/shwm/parity_exhaustive.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parity_microcase import build, PARAMETER_CEILING  # noqa: E402

TRAIN_LENGTHS = range(1, 9)
HELD_LENGTHS = range(9, 17)
SEEDS = (6600, 6601, 6602)
UPDATES = 1024


def enumerate_sequences(lengths):
    """Every binary sequence of each length, with both initial phases."""
    for length in lengths:
        for value in range(2 ** length):
            events = np.array([(value >> k) & 1 for k in range(length)], dtype=np.float32)
            for h0 in (0, 1):
                yield events, h0


def encode(pairs, max_length):
    x = np.zeros((len(pairs), max_length, 3), dtype=np.float32)
    y = np.zeros((len(pairs), max_length), dtype=np.int32)
    mask = np.zeros((len(pairs), max_length), dtype=np.float32)
    for i, (events, h0) in enumerate(pairs):
        n = len(events)
        x[i, :n, 0] = events
        x[i, 0, 1] = float(h0)
        x[i, 0, 2] = 1.0
        y[i, :n] = (h0 + np.cumsum(events)) % 2
        mask[i, :n] = 1.0
    return x, y, mask


def evaluate(model, pairs, max_length, mx, batch=4096) -> dict[str, Any]:
    final_hits, step_hits, step_total = [], 0, 0
    by_length: dict[int, list] = {}
    by_count: dict[int, list] = {}
    by_position = np.zeros(max_length), np.zeros(max_length)
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        x, y, mask = encode(chunk, max_length)
        logits = model(mx.array(x))
        mx.eval(logits)
        predicted = np.asarray(logits).argmax(axis=-1)
        for i, (events, _) in enumerate(chunk):
            n = len(events)
            correct = predicted[i, :n] == y[i, :n]
            final_hits.append(float(correct[n - 1]))
            step_hits += int(correct.sum())
            step_total += n
            by_length.setdefault(n, []).append(float(correct[n - 1]))
            by_count.setdefault(int(events.sum()), []).append(float(correct[n - 1]))
            by_position[0][:n] += correct
            by_position[1][:n] += 1
    return {
        "pairs": len(pairs),
        "final_parity_accuracy": float(np.mean(final_hits)),
        "stepwise_trajectory_accuracy": step_hits / max(step_total, 1),
        "by_length": {str(k): float(np.mean(v)) for k, v in sorted(by_length.items())},
        "by_event_count": {str(k): float(np.mean(v)) for k, v in sorted(by_count.items())},
        "by_position": [float(a / b) if b else float("nan")
                        for a, b in zip(*by_position)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/parity-exhaustive.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    train_pairs = list(enumerate_sequences(TRAIN_LENGTHS))
    held_pairs = list(enumerate_sequences(HELD_LENGTHS))
    print(f"exhaustive: {len(train_pairs)} trained-length pairs, "
          f"{len(held_pairs)} extrapolation pairs, {len(train_pairs) + len(held_pairs)} total",
          flush=True)

    max_train = max(TRAIN_LENGTHS)
    max_held = max(HELD_LENGTHS)
    x, y, mask = encode(train_pairs, max_train)
    report: dict[str, Any] = {"train_pairs": len(train_pairs),
                              "held_pairs": len(held_pairs), "arms": {}}

    print(f"\n{'arm':22s} {'seed':>6s} {'trained len (exact)':>20s} {'extrapolation':>15s} "
          f"{'stepwise':>10s}")
    print("-" * 78)
    for kind in ("generic_recurrent", "two_state_filter", "eight_state_filter"):
        for seed in SEEDS:
            mx.random.seed(seed)
            model = build(kind, 3)
            mx.eval(model.parameters())
            optimizer = optim.AdamW(learning_rate=3e-3)
            rng = np.random.default_rng(seed)
            for _ in range(UPDATES):
                pick = rng.integers(0, len(x), 64)
                xb, yb, mb = mx.array(x[pick]), mx.array(y[pick]), mx.array(mask[pick])

                def loss_fn(m):
                    logits = m(xb)
                    losses = nn.losses.cross_entropy(
                        logits.reshape(-1, 2), yb.reshape(-1), reduction="none")
                    return (losses * mb.reshape(-1)).sum() / mb.sum()

                loss, grads = nn.value_and_grad(model, loss_fn)(model)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, loss)

            trained = evaluate(model, train_pairs, max_train, mx)
            held = evaluate(model, held_pairs, max_held, mx)
            report["arms"][f"{kind}@{seed}"] = {"trained": trained, "held": held}
            print(f"{kind:22s} {seed:6d} {trained['final_parity_accuracy']:20.6f} "
                  f"{held['final_parity_accuracy']:15.6f} "
                  f"{held['stepwise_trajectory_accuracy']:10.6f}", flush=True)

    exact = {k: v for k, v in report["arms"].items()
             if v["trained"]["final_parity_accuracy"] >= 1.0}
    report["t3_exhaustive_parity"] = len(exact) > 0
    report["arms_exact_on_trained_lengths"] = sorted(exact)
    report["wall_clock_seconds"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\narms exactly correct on ALL trained-length sequences: {len(exact)} of "
          f"{len(report['arms'])}")
    print(f"T3: {report['t3_exhaustive_parity']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
