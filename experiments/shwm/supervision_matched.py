"""G / N9. Is the factorized advantage factorization, or is it the extra label?

The M-phase comparison was confounded and the specification is right to say so. The
factorized arm was trained against event targets; the end-to-end GRU was not. Calling
that difference "factorization" is unsupported while one arm receives a label the other
never sees.

The 2x2 that settles it, all four cells sharing inputs, examples, budget, optimizer,
parameter ceiling and outcome head:

                      | outcome loss only | + auxiliary event loss
    end-to-end GRU    |        (1)        |         (2)
    factorized filter |        (3)        |         (4)

If (2) closes the gap, the gain was supervision and the honest claim is a supervised
event-factorization result, not autonomous hidden-state discovery. If (3) closes it, the
architecture alone suffices. If only (4) works, both are needed and the claim must say so.

The accumulator here is the LEARNED two-state filter, not the exact XOR. A hard XOR has no
gradient, so a factorized arm trained on outcome loss alone could not learn its event head
through it -- cell (3) would be untestable by construction.

    .venv-shwm/bin/python experiments/shwm/supervision_matched.py
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

from sentinel.env.adapters.procedural_visual_v2 import ACTIONS  # noqa: E402
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402
from readout_qualification import bootstrap_difference  # noqa: E402
from structured_calibration import CLASSES, collect  # noqa: E402
from belief_factorization import build_dataset  # noqa: E402

HIDDEN = 128
STATES = 2
PARAMETER_CEILING = 250_000
BUDGETS = (256, 1024)
SEEDS = (6600, 6601, 6602, 6603, 6604)
EVENT_LOSS_WEIGHT = 1.0


def pad(items):
    length = max(len(i["y"]) for i in items)
    width = items[0]["x"].shape[1]
    x = np.zeros((len(items), length, width), dtype=np.float32)
    y = np.zeros((len(items), length), dtype=np.int32)
    e = np.zeros((len(items), length), dtype=np.int32)
    m = np.zeros((len(items), length), dtype=np.float32)
    reset = np.zeros((len(items), 1), dtype=np.float32)
    for i, item in enumerate(items):
        n = len(item["y"])
        x[i, :n] = item["x"]
        y[i, :n] = item["y"]
        e[i, :n] = item["events"]
        m[i, :n] = 1.0
        reset[i, 0] = float(item["phases"][0])   # the rendered reset stripe, public
    return x, y, e, m, reset


def make_model(kind: str, width: int):
    import mlx.core as mx
    import mlx.nn as nn

    class EndToEnd(nn.Module):
        """A GRU over the sequence, with an optional auxiliary event head."""
        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Linear(width, HIDDEN)
            self.gru = nn.GRU(HIDDEN, HIDDEN)
            self.head = nn.Linear(HIDDEN, CLASSES)
            self.event_head = nn.Linear(HIDDEN, 2)

        def __call__(self, z, reset, true_event=None):
            h = self.gru(nn.relu(self.project(z)))
            return self.head(h), self.event_head(h)

    class Factorized(nn.Module):
        """event head -> learned two-state belief -> outcome head.

        The belief update is a learned row-stochastic matrix per event value. The filter
        knows only that there are two latent states; it is given no transition matrix, no
        state names and no phase input. The belief is initialised from the PUBLIC reset
        stripe rather than an evaluator bit.
        """
        def __init__(self) -> None:
            super().__init__()
            self.detect = nn.Sequential(nn.Linear(2 * width + 1, HIDDEN), nn.ReLU(),
                                        nn.Linear(HIDDEN, 2))
            self.logits = mx.random.normal((2, STATES, STATES)) * 0.5
            self.head = nn.Sequential(nn.Linear(width + STATES, HIDDEN), nn.ReLU(),
                                      nn.Linear(HIDDEN, CLASSES))

        def __call__(self, z, reset, true_event=None):
            batch, length, _ = z.shape
            previous = mx.concatenate([mx.zeros((batch, 1, z.shape[2])), z[:, :-1]], axis=1)
            flag = mx.concatenate(
                [mx.zeros((batch, 1, 1)), mx.ones((batch, length - 1, 1))], axis=1)
            event_logits = self.detect(mx.concatenate([previous, z, flag], axis=-1))
            event = (true_event if true_event is not None
                     else mx.softmax(event_logits, axis=-1)[:, :, 1])   # p(crossed)
            transition = mx.softmax(self.logits, axis=-1)
            belief = mx.concatenate([1.0 - reset, reset], axis=-1)    # public reset stripe
            beliefs = []
            for t in range(length):
                e = event[:, t][:, None, None]
                matrix = transition[0][None] * (1.0 - e) + transition[1][None] * e
                belief = mx.sum(belief[:, :, None] * matrix, axis=1)
                belief = belief / mx.maximum(mx.sum(belief, axis=-1, keepdims=True), 1e-9)
                beliefs.append(belief)
            stacked = mx.stack(beliefs, axis=1)
            return self.head(mx.concatenate([z, stacked], axis=-1)), event_logits

    class Memoryless(nn.Module):
        """No temporal state at all: the matched current-information baseline.

        Required by the reconciliation rule -- R_phase needs a *trained memoryless model*
        on the same population and metric, and a constant-phase arm may not be substituted
        for one unless the two are proven equivalent. It shares the outcome head's width
        and the same budget, and simply has no path from one step to the next.
        """
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(width, HIDDEN), nn.ReLU(),
                                     nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                     nn.Linear(HIDDEN, CLASSES))
            self.event_head = nn.Linear(width, 2)

        def __call__(self, z, reset, true_event=None):
            return self.net(z), self.event_head(z)

    if kind == "end_to_end":
        return EndToEnd()
    if kind == "memoryless":
        return Memoryless()
    return Factorized()


def run(kind, event_supervision, train, test, updates, seed, permute_events=False,
        true_event=False):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    mx.random.seed(seed)
    x, y, e, m, reset = pad(train)
    if permute_events:
        rng = np.random.default_rng(seed)
        e = e[rng.permutation(len(e))]
    model = make_model(kind, x.shape[2])
    mx.eval(model.parameters())
    parameters = int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))
    assert parameters <= PARAMETER_CEILING, (kind, parameters)
    optimizer = optim.AdamW(learning_rate=2e-3)
    rng = np.random.default_rng(seed)

    for _ in range(updates):
        pick = rng.integers(0, len(x), min(32, len(x)))
        xb, yb, eb = mx.array(x[pick]), mx.array(y[pick]), mx.array(e[pick])
        mb, rb = mx.array(m[pick]), mx.array(reset[pick])

        def loss_fn(mo):
            logits, event_logits = mo(xb, rb, eb.astype(mx.float32) if true_event else None)
            outcome = nn.losses.cross_entropy(
                logits.reshape(-1, CLASSES), yb.reshape(-1), reduction="none")
            total = (outcome * mb.reshape(-1)).sum() / mb.sum()
            if event_supervision:
                aux = nn.losses.cross_entropy(
                    event_logits.reshape(-1, 2), eb.reshape(-1), reduction="none")
                total = total + EVENT_LOSS_WEIGHT * (aux * mb.reshape(-1)).sum() / mb.sum()
            return total

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    xt, yt, et, mt, rt = pad(test)
    logits, _ = model(mx.array(xt), mx.array(rt),
                      mx.array(et.astype(np.float32)) if true_event else None)
    mx.eval(logits)
    predicted = np.asarray(logits).argmax(axis=-1)
    valid = mt.astype(bool)
    per_episode = np.array([
        float((predicted[i][valid[i]] == yt[i][valid[i]]).mean()) for i in range(len(xt))])
    groups = np.array([item["layout"] for item in test])
    return {
        "arm": kind, "event_supervision": bool(event_supervision),
        "true_event_input": bool(true_event),
        "permuted_events": bool(permute_events), "updates": updates, "seed": seed,
        "parameters": parameters,
        "accuracy": float(predicted[valid] == yt[valid]).__float__()
        if False else float((predicted[valid] == yt[valid]).mean()),
        "per_episode": per_episode, "groups": groups,
    }


def paired_interval(a, b, groups, resamples=4000, seed=99):
    """Paired resampling by layout: the difference between two arms on the same episodes."""
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for r in range(resamples):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[g] for g in picked])
        draws[r] = a[rows].mean() - b[rows].mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-layouts", type=int, default=60)
    parser.add_argument("--test-layouts", type=int, default=30)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/supervision-matched.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    appearance = CANONICAL_APPEARANCE_SEED

    train = build_dataset(collect(list(range(61_000, 61_000 + arguments.train_layouts)),
                                  arguments.trajectories, arguments.steps, appearance, 11), 5)
    test = build_dataset(collect(list(range(81_000, 81_000 + arguments.test_layouts)),
                                 2, arguments.steps, appearance, 777), 6)
    print(f"train {len(train)} trajectories, held-out {len(test)}", flush=True)

    cells = [
        ("0_memoryless_current_only", "memoryless", False, False),
        ("1_end_to_end_outcome_only", "end_to_end", False, False),
        ("2_end_to_end_plus_event_loss", "end_to_end", True, False),
        ("3_factorized_outcome_only", "factorized", False, False),
        ("4_factorized_plus_event_loss", "factorized", True, False),
        ("8_factorized_permuted_event_labels", "factorized", True, True),
    ]
    # N5: the learned filter given TRUE events, to separate "the filter cannot learn the
    # transition" from "the event detector is the bottleneck".
    n5 = ("5_TRUE_event_learned_filter", "factorized", False, False)
    report: dict[str, Any] = {"seeds": list(SEEDS), "budgets": list(BUDGETS), "cells": {}}
    store: dict[tuple, Any] = {}

    print(f"\n{'cell':38s} {'updates':>8s} {'mean acc':>9s} {'sd':>7s}  per-seed")
    print("-" * 96)
    for name, kind, supervision, permute in cells:
        for updates in BUDGETS:
            runs = [run(kind, supervision, train, test, updates, seed, permute)
                    for seed in SEEDS]
            accuracies = np.array([r["accuracy"] for r in runs])
            store[(name, updates)] = runs
            report["cells"][f"{name}@{updates}"] = {
                "cell": name, "updates": updates,
                "mean_accuracy": float(accuracies.mean()),
                "sd": float(accuracies.std()),
                "per_seed": [float(v) for v in accuracies],
                "parameters": runs[0]["parameters"],
            }
            print(f"{name:38s} {updates:8d} {accuracies.mean():9.4f} {accuracies.std():7.4f}  "
                  + " ".join(f"{v:.3f}" for v in accuracies), flush=True)

    for updates in BUDGETS:
        runs = [run(n5[1], n5[2], train, test, updates, seed, False, true_event=True)
                for seed in SEEDS]
        accuracies = np.array([r["accuracy"] for r in runs])
        store[(n5[0], updates)] = runs
        report["cells"][f"{n5[0]}@{updates}"] = {
            "cell": n5[0], "updates": updates, "mean_accuracy": float(accuracies.mean()),
            "sd": float(accuracies.std()), "per_seed": [float(v) for v in accuracies]}
        print(f"{n5[0]:38s} {updates:8d} {accuracies.mean():9.4f} {accuracies.std():7.4f}  "
              + " ".join(f"{v:.3f}" for v in accuracies), flush=True)

    best = max(BUDGETS)
    print(f"\npaired intervals by layout, at {best} updates:")
    comparisons = [
        ("factorized+event  vs  end-to-end+event",
         "4_factorized_plus_event_loss", "2_end_to_end_plus_event_loss"),
        ("factorized+event  vs  factorized outcome-only",
         "4_factorized_plus_event_loss", "3_factorized_outcome_only"),
        ("end-to-end+event  vs  end-to-end outcome-only",
         "2_end_to_end_plus_event_loss", "1_end_to_end_outcome_only"),
        ("factorized+event  vs  permuted event labels",
         "4_factorized_plus_event_loss", "8_factorized_permuted_event_labels"),
        ("factorized+event  vs  MEMORYLESS baseline",
         "4_factorized_plus_event_loss", "0_memoryless_current_only"),
        ("end-to-end+event  vs  MEMORYLESS baseline",
         "2_end_to_end_plus_event_loss", "0_memoryless_current_only"),
        ("N5: TRUE event + learned filter  vs  MEMORYLESS",
         "5_TRUE_event_learned_filter", "0_memoryless_current_only"),
        ("N5: TRUE event + learned filter  vs  learned event",
         "5_TRUE_event_learned_filter", "4_factorized_plus_event_loss"),
    ]
    report["paired"] = {}
    for label, left, right in comparisons:
        a = np.mean([r["per_episode"] for r in store[(left, best)]], axis=0)
        b = np.mean([r["per_episode"] for r in store[(right, best)]], axis=0)
        low, high = paired_interval(a, b, store[(left, best)][0]["groups"])
        excludes = low > 0 or high < 0
        report["paired"][label] = {"delta": float(a.mean() - b.mean()),
                                   "ci_low": low, "ci_high": high,
                                   "excludes_zero": bool(excludes)}
        print(f"  {label:46s} {a.mean() - b.mean():+.4f}  [{low:+.4f}, {high:+.4f}]"
              f"{'  *' if excludes else ''}")

    report["wall_clock_seconds"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(f"\nwrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
