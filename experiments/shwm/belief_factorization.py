"""D / E / F. Factorized belief against end-to-end recurrence, on the environment.

Three things are now known and they bracket the question tightly. The prediction
head reaches the oracle when phase is supplied (L3). The temporal model class
learns parity perfectly, including length extrapolation (M2). And the information
needed to infer phase is present in the structured history: comparing an observed
displacement against its action's expected delta reveals the polarity that was in
force, and crossings are inferable from the previous step's neighbour-switch bits.

So the failure is neither the head, nor the model class, nor the information. This
module tests the remaining candidate -- that end-to-end credit assignment is what
fails -- by factorizing the same computation into parts and measuring each.

One field is added to the structured encoding and it is a correction, not a
concession: the reset frame renders the polarity stripe, so initial polarity IS
public at step 0. Omitting it made phase unidentifiable up to a global flip for
reasons that had nothing to do with learning. It is supplied only at step 0, exactly
as the renderer supplies it.

    .venv-shwm/bin/python experiments/shwm/belief_factorization.py
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

from sentinel.env.adapters.procedural_visual_v2 import ACTIONS, GRID  # noqa: E402
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402
from readout_qualification import balanced_accuracy, bootstrap_difference, f1  # noqa: E402
from structured_calibration import (  # noqa: E402
    CLASSES, DELTAS_BY_INDEX, DISPLACEMENTS, analytic_memoryless_ceiling, collect,
    encode, target_cell,
)

HIDDEN = 128
PARAMETER_CEILING = 250_000
BUDGETS = (64, 256, 1024)
SEEDS = (6600, 6601, 6602)


def public_event(previous_row, row) -> int:
    """Did this transition enter a switch cell? Derived from PUBLIC state only.

    The destination's switch status is read from the previous step, where the agent
    was elsewhere and the cell showed its own colour -- the renderer paints the agent
    over the switch beneath it, so the current step cannot answer this.
    """
    if previous_row is None:
        return 0
    moved = row["position"] != previous_row["position"]
    return int(moved and row["position"] in previous_row["switches"])


def sequence_features(trajectory, query_actions, with_reset_polarity=True):
    """Structured public features, plus the reset stripe at step 0 only."""
    rows = trajectory["rows"]
    blocks = []
    for index, (row, action) in enumerate(zip(rows, query_actions)):
        base = encode(row, action, with_phase=False)
        extra = np.zeros(2, dtype=np.float32)
        if index == 0 and with_reset_polarity:
            extra[0] = float(row["polarity"])   # the rendered reset stripe: public
            extra[1] = 1.0
        blocks.append(np.concatenate([base, extra]))
    return np.stack(blocks).astype(np.float32)


def build_dataset(trajectories, seed):
    rng = np.random.default_rng(seed)
    out = []
    for trajectory in trajectories:
        rows = trajectory["rows"]
        query = [int(rng.integers(0, len(ACTIONS))) for _ in rows]
        events, phases = [], []
        previous = None
        for row in rows:
            events.append(public_event(previous, row))
            phases.append(row["polarity"])
            previous = row
        out.append({
            "x": sequence_features(trajectory, query),
            "y": np.array([target_cell(r, a) for r, a in zip(rows, query)]),
            "events": np.array(events),
            "phases": np.array(phases),
            "layout": trajectory["layout"],
            "rows": rows, "query": query,
        })
    return out


def exact_accumulator_phase(item):
    """h_t = h_0 XOR parity(events). h_0 is the rendered reset stripe."""
    h = int(item["phases"][0])
    out = []
    for index in range(len(item["events"])):
        if index:
            h ^= int(item["events"][index])
        out.append(h)
    return np.array(out)


def head_with_phase(train, test, phase_of, updates, seed):
    """The qualified L-phase head, fed a phase estimate instead of the truth."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(seed)

    def featurise(items):
        xs, ys = [], []
        for item in items:
            phase = phase_of(item)
            for t in range(len(item["y"])):
                xs.append(np.concatenate([item["x"][t], [float(phase[t])]]))
                ys.append(item["y"][t])
        return np.stack(xs).astype(np.float32), np.array(ys)

    x, y = featurise(train)
    xt, yt = featurise(test)

    class Head(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.a = nn.Linear(width, HIDDEN)
            self.b = nn.Linear(HIDDEN, HIDDEN)
            self.head = nn.Linear(HIDDEN, CLASSES)

        def __call__(self, z):
            return self.head(nn.relu(self.b(nn.relu(self.a(z)))))

    model = Head(x.shape[1])
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=2e-3)
    rng = np.random.default_rng(seed)
    for _ in range(updates):
        pick = rng.integers(0, len(x), 128)
        xb, yb = mx.array(x[pick]), mx.array(y[pick].astype(np.int32))

        def loss_fn(m):
            return nn.losses.cross_entropy(m(xb), yb, reduction="mean")

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    logits = model(mx.array(xt))
    mx.eval(logits)
    return float((np.asarray(logits).argmax(axis=1) == yt).mean())


def train_event_extractor(train, test, updates, seed):
    """Predict the PUBLIC crossing event from (state_{t-1}, action_{t-1}, state_t).

    Trained on evaluator-derived labels for a public quantity, which the
    specification permits as declared auxiliary supervision. It is never an input to
    the end-to-end arm.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(seed)

    def featurise(items):
        xs, ys = [], []
        for item in items:
            for t in range(len(item["events"])):
                previous = item["x"][t - 1] if t else np.zeros_like(item["x"][t])
                xs.append(np.concatenate([previous, item["x"][t], [float(t > 0)]]))
                ys.append(item["events"][t])
        return np.stack(xs).astype(np.float32), np.array(ys)

    x, y = featurise(train)
    xt, yt = featurise(test)

    class EventHead(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.a = nn.Linear(width, HIDDEN)
            self.head = nn.Linear(HIDDEN, 2)

        def __call__(self, z):
            return self.head(nn.relu(self.a(z)))

    model = EventHead(x.shape[1])
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=2e-3)
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(updates):
        pick = np.concatenate([rng.choice(positive, 64), rng.choice(negative, 64)])
        xb, yb = mx.array(x[pick]), mx.array(y[pick].astype(np.int32))

        def loss_fn(m):
            return nn.losses.cross_entropy(m(xb), yb, reduction="mean")

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    logits = model(mx.array(xt))
    mx.eval(logits)
    predicted = np.asarray(logits).argmax(axis=1)

    def predict_for(item):
        xs = []
        for t in range(len(item["events"])):
            previous = item["x"][t - 1] if t else np.zeros_like(item["x"][t])
            xs.append(np.concatenate([previous, item["x"][t], [float(t > 0)]]))
        out = model(mx.array(np.stack(xs).astype(np.float32)))
        mx.eval(out)
        return np.asarray(out).argmax(axis=1)

    return predict_for, {
        "balanced_accuracy": balanced_accuracy(yt, predicted),
        "f1": f1(yt, predicted),
        "precision": float((predicted[predicted == 1] == yt[predicted == 1]).mean())
        if (predicted == 1).any() else 0.0,
        "recall": float((predicted[yt == 1] == 1).mean()) if (yt == 1).any() else 0.0,
        "positive_rate": float(yt.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-layouts", type=int, default=60)
    parser.add_argument("--test-layouts", type=int, default=30)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/belief-factorization.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    appearance = CANONICAL_APPEARANCE_SEED

    train_t = collect(list(range(61_000, 61_000 + arguments.train_layouts)),
                      arguments.trajectories, arguments.steps, appearance, 11)
    test_t = collect(list(range(81_000, 81_000 + arguments.test_layouts)),
                     2, arguments.steps, appearance, 777)
    train = build_dataset(train_t, 5)
    test = build_dataset(test_t, 6)
    ceiling = analytic_memoryless_ceiling(test_t)
    print(f"train {len(train)} trajectories, held-out {len(test)}; "
          f"uniform-prior memoryless ceiling {ceiling:.4f}", flush=True)

    report: dict[str, Any] = {"ceiling": ceiling, "arms": {}, "seeds": list(SEEDS)}

    print("\nevent extractor (public label, declared auxiliary supervision)", flush=True)
    extractor, event_metrics = train_event_extractor(train, test, 1024, SEEDS[0])
    report["event_extractor"] = event_metrics
    print(f"  balanced accuracy {event_metrics['balanced_accuracy']:.4f}  "
          f"F1 {event_metrics['f1']:.4f}  precision {event_metrics['precision']:.4f}  "
          f"recall {event_metrics['recall']:.4f}", flush=True)

    def phase_true(item):
        return item["phases"]

    def phase_exact_from_true_events(item):
        return exact_accumulator_phase(item)

    def phase_exact_from_learned_events(item):
        predicted = extractor(item)
        h = int(item["phases"][0])          # reset stripe, public
        out = []
        for index in range(len(predicted)):
            if index:
                h ^= int(predicted[index])
            out.append(h)
        return np.array(out)

    def phase_constant(item):
        return np.zeros(len(item["y"]), dtype=int)

    def phase_shuffled_events(item):
        rng = np.random.default_rng(item["layout"])
        events = rng.permutation(item["events"])
        h = int(item["phases"][0])
        out = []
        for index in range(len(events)):
            if index:
                h ^= int(events[index])
            out.append(h)
        return np.array(out)

    arms = {
        "6_true_phase_oracle": phase_true,
        "1_true_event_exact_accumulator": phase_exact_from_true_events,
        "3_learned_event_exact_accumulator": phase_exact_from_learned_events,
        "control_shuffled_events": phase_shuffled_events,
        "control_constant_phase": phase_constant,
    }
    print(f"\n{'arm':38s} {'updates':>8s} {'displacement acc':>17s} {'phase acc':>10s}")
    print("-" * 80)
    for name, phase_of in arms.items():
        phase_accuracy = float(np.mean([
            (phase_of(item) == item["phases"]).mean() for item in test]))
        for updates in BUDGETS:
            score = head_with_phase(train, test, phase_of, updates, SEEDS[0])
            report["arms"][f"{name}@{updates}"] = {
                "arm": name, "updates": updates,
                "displacement_accuracy": score, "phase_accuracy": phase_accuracy,
            }
            print(f"{name:38s} {updates:8d} {score:17.4f} {phase_accuracy:10.4f}", flush=True)

    report["wall_clock_seconds"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
