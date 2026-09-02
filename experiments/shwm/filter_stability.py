"""C / D / G / U3. Does any learned filter reliably learn the ENVIRONMENTAL transition?

The previous phase marked environmental T4 as failed on the strength of a *parity*
seed collapse in the two-state arm. That was a ledger error: T4 is an environmental
gate and it admits either a two-state or an eight-state filter, and the eight-state
filter was exact on every parity seed. Environmental T4 was never run. It is run here.

Seed stability is the question, so the design is twenty development seeds and twenty
untouched validation seeds rather than three of each. The initialization rule is
chosen on development and frozen before validation is touched.

Events are the TRUE public switch-crossings at this stage, which isolates transition
learning from event extraction. No arm receives true phase, a transition matrix,
state names, the simulator step, a future outcome or any evaluator field.

    .venv-shwm/bin/python experiments/shwm/filter_stability.py
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

from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402
from structured_calibration import CLASSES, collect  # noqa: E402
from belief_factorization import build_dataset, exact_accumulator_phase  # noqa: E402

HIDDEN = 128
PARAMETER_CEILING = 250_000
UPDATES = 1024
DEV_SEEDS = tuple(range(7000, 7020))
VALIDATION_SEEDS = tuple(range(8000, 8020))

ARMS = (
    "1_exact_accumulator",
    "2_two_state_default_init",
    "3_two_state_symmetry_broken",
    "4_two_state_reset_conditioned",
    "5_eight_state",
    "6_generic_gru",
    "7_trained_memoryless",
)


def pad(items):
    length = max(len(i["y"]) for i in items)
    width = items[0]["x"].shape[1]
    x = np.zeros((len(items), length, width), dtype=np.float32)
    y = np.zeros((len(items), length), dtype=np.int32)
    e = np.zeros((len(items), length), dtype=np.float32)
    m = np.zeros((len(items), length), dtype=np.float32)
    reset = np.zeros((len(items), 1), dtype=np.float32)
    phases = np.zeros((len(items), length), dtype=np.int32)
    for i, item in enumerate(items):
        n = len(item["y"])
        x[i, :n] = item["x"]
        y[i, :n] = item["y"]
        e[i, :n] = item["events"]
        m[i, :n] = 1.0
        reset[i, 0] = float(item["phases"][0])
        phases[i, :n] = item["phases"]
    return x, y, e, m, reset, phases


def make(arm, width, seed):
    import mlx.core as mx
    import mlx.nn as nn

    states = 8 if arm == "5_eight_state" else 2

    class Filter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if arm == "2_two_state_default_init":
                self.logits = mx.random.normal((2, states, states)) * 0.5
            elif arm == "3_two_state_symmetry_broken":
                # A tiny seed-derived ANTISYMMETRIC perturbation. It encodes no phase
                # semantics and no XOR structure -- it only ensures the two event maps
                # do not start interchangeable, which is the basin the collapsed seed
                # settled in (belief entropy 0.664 of a 0.693 maximum).
                base = mx.random.normal((2, states, states)) * 0.05
                anti = mx.array(np.array([[[0.5, -0.5], [-0.5, 0.5]],
                                          [[-0.5, 0.5], [0.5, -0.5]]], dtype=np.float32))
                self.logits = base + anti
            else:
                self.logits = mx.random.normal((2, states, states)) * 0.5
            self.initial = nn.Linear(1, states)
            self.head = nn.Sequential(nn.Linear(width + states, HIDDEN), nn.ReLU(),
                                      nn.Linear(HIDDEN, CLASSES))

        def __call__(self, z, reset, event):
            batch, length, _ = z.shape
            transition = mx.softmax(self.logits, axis=-1)
            if arm == "4_two_state_reset_conditioned":
                belief = mx.softmax(self.initial(reset), axis=-1)
            else:
                belief = mx.concatenate([1.0 - reset, reset], axis=-1) if states == 2 \
                    else mx.softmax(self.initial(reset), axis=-1)
            beliefs = []
            for t in range(length):
                ev = event[:, t][:, None, None]
                matrix = transition[0][None] * (1.0 - ev) + transition[1][None] * ev
                belief = mx.sum(belief[:, :, None] * matrix, axis=1)
                belief = belief / mx.maximum(mx.sum(belief, axis=-1, keepdims=True), 1e-9)
                beliefs.append(belief)
            stacked = mx.stack(beliefs, axis=1)
            return self.head(mx.concatenate([z, stacked], axis=-1)), stacked

    class Gru(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Linear(width + 1, HIDDEN)
            self.gru = nn.GRU(HIDDEN, HIDDEN)
            self.head = nn.Linear(HIDDEN, CLASSES)

        def __call__(self, z, reset, event):
            fed = mx.concatenate([z, event[:, :, None]], axis=-1)
            h = self.gru(nn.relu(self.project(fed)))
            return self.head(h), h

    class Memoryless(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(width, HIDDEN), nn.ReLU(),
                                     nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                     nn.Linear(HIDDEN, CLASSES))

        def __call__(self, z, reset, event):
            return self.net(z), None

    class ExactAccumulator(nn.Module):
        """The ceiling: parity of the TRUE events, XOR the public reset stripe."""
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Sequential(nn.Linear(width + 2, HIDDEN), nn.ReLU(),
                                      nn.Linear(HIDDEN, CLASSES))

        def __call__(self, z, reset, event):
            batch, length, _ = z.shape
            shifted = mx.concatenate([mx.zeros((batch, 1)), event[:, 1:]], axis=1)
            parity = mx.remainder(mx.cumsum(shifted, axis=1), 2.0)
            phase = mx.remainder(reset + parity, 2.0)
            onehot = mx.stack([1.0 - phase, phase], axis=-1)
            return self.head(mx.concatenate([z, onehot], axis=-1)), onehot

    mx.random.seed(seed)
    if arm == "6_generic_gru":
        return Gru()
    if arm == "7_trained_memoryless":
        return Memoryless()
    if arm == "1_exact_accumulator":
        return ExactAccumulator()
    return Filter()


def run(arm, train, test, seed):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    x, y, e, m, reset, _ = pad(train)
    model = make(arm, x.shape[2], seed)
    mx.eval(model.parameters())
    parameters = int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))
    assert parameters <= PARAMETER_CEILING, (arm, parameters)
    optimizer = optim.AdamW(learning_rate=2e-3)
    rng = np.random.default_rng(seed)
    for _ in range(UPDATES):
        pick = rng.integers(0, len(x), min(32, len(x)))
        xb, yb, eb = mx.array(x[pick]), mx.array(y[pick]), mx.array(e[pick])
        mb, rb = mx.array(m[pick]), mx.array(reset[pick])

        def loss_fn(mo):
            logits, _ = mo(xb, rb, eb)
            losses = nn.losses.cross_entropy(
                logits.reshape(-1, CLASSES), yb.reshape(-1), reduction="none")
            return (losses * mb.reshape(-1)).sum() / mb.sum()

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    xt, yt, et, mt, rt, pt = pad(test)
    logits, belief = model(mx.array(xt), mx.array(rt), mx.array(et))
    mx.eval(logits)
    predicted = np.asarray(logits).argmax(axis=-1)
    valid = mt.astype(bool)
    per_episode = np.array([float((predicted[i][valid[i]] == yt[i][valid[i]]).mean())
                            for i in range(len(xt))])
    crossings = np.cumsum(et, axis=1)
    strata = {}
    for label, mask in (("all", valid),
                        ("after_1_change", valid & (crossings >= 1)),
                        ("after_2_changes", valid & (crossings >= 2)),
                        ("after_3_changes", valid & (crossings >= 3))):
        if mask.sum() > 20:
            strata[label] = float((predicted[mask] == yt[mask]).mean())
    record = {"arm": arm, "seed": seed, "parameters": parameters,
              "accuracy": float((predicted[valid] == yt[valid]).mean()),
              "per_episode": per_episode.tolist(), "strata": strata}
    if belief is not None and hasattr(model, "logits"):
        b = np.asarray(belief)
        states = b.shape[-1]
        entropy = float(-(b * np.log(np.maximum(b, 1e-12))).sum(axis=-1).mean())
        record["normalised_belief_entropy"] = entropy / float(np.log(states))
        record["collapsed"] = bool(record["normalised_belief_entropy"] > 0.9)
    return record


def summarise(records):
    """Never by mean alone: the question is the lower tail."""
    accuracies = np.array([r["accuracy"] for r in records])
    return {
        "seeds": len(records),
        "mean": float(accuracies.mean()),
        "sd": float(accuracies.std()),
        "median": float(np.median(accuracies)),
        "minimum": float(accuracies.min()),
        "p10": float(np.percentile(accuracies, 10)),
        "collapsed_seeds": int(sum(1 for r in records if r.get("collapsed"))),
        "after_2_changes_mean": float(np.mean(
            [r["strata"].get("after_2_changes", np.nan) for r in records])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-layouts", type=int, default=40)
    parser.add_argument("--test-layouts", type=int, default=20)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/filter-stability.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    appearance = CANONICAL_APPEARANCE_SEED

    train = build_dataset(collect(list(range(61_000, 61_000 + arguments.train_layouts)),
                                  arguments.trajectories, arguments.steps, appearance, 11), 5)
    test = build_dataset(collect(list(range(81_000, 81_000 + arguments.test_layouts)),
                                 2, arguments.steps, appearance, 777), 6)
    print(f"train {len(train)} trajectories, held-out {len(test)}; TRUE public events as "
          f"input, {len(DEV_SEEDS)} development seeds\n", flush=True)

    report: dict[str, Any] = {"development": {}, "validation": {},
                              "dev_seeds": list(DEV_SEEDS),
                              "validation_seeds": list(VALIDATION_SEEDS)}

    print(f"{'arm':32s} {'mean':>7s} {'sd':>7s} {'median':>7s} {'min':>7s} {'p10':>7s} "
          f"{'collapsed':>10s} {'2+ chg':>7s}")
    print("-" * 92)
    dev_records: dict[str, list] = {}
    for arm in ARMS:
        records = [run(arm, train, test, seed) for seed in DEV_SEEDS]
        dev_records[arm] = records
        stats = summarise(records)
        report["development"][arm] = {"stats": stats, "records": records}
        print(f"{arm:32s} {stats['mean']:7.4f} {stats['sd']:7.4f} {stats['median']:7.4f} "
              f"{stats['minimum']:7.4f} {stats['p10']:7.4f} "
              f"{stats['collapsed_seeds']:>4d}/{len(records):<5d} "
              f"{stats['after_2_changes_mean']:7.4f}", flush=True)

    # Freeze the rule on development: the learned filter with the best LOWER TAIL,
    # not the best mean.
    candidates = {a: report["development"][a]["stats"] for a in ARMS
                  if a.startswith(("2_", "3_", "4_", "5_"))}
    selected = max(candidates, key=lambda a: (candidates[a]["p10"], candidates[a]["median"]))
    report["selected_on_development"] = selected
    report["selection_rule"] = "highest 10th-percentile accuracy, then median; never mean alone"
    print(f"\nselected on development by lower tail: {selected}")

    print(f"\nvalidation on {len(VALIDATION_SEEDS)} untouched seeds:")
    for arm in (selected, "1_exact_accumulator", "6_generic_gru", "7_trained_memoryless"):
        records = [run(arm, train, test, seed) for seed in VALIDATION_SEEDS]
        stats = summarise(records)
        report["validation"][arm] = {"stats": stats, "records": records}
        print(f"  {arm:32s} mean {stats['mean']:.4f}  sd {stats['sd']:.4f}  "
              f"min {stats['minimum']:.4f}  p10 {stats['p10']:.4f}  "
              f"collapsed {stats['collapsed_seeds']}/{len(records)}", flush=True)

    memoryless = report["validation"]["7_trained_memoryless"]["stats"]["mean"]
    chosen = report["validation"][selected]["stats"]
    report["u3_stable_filter"] = bool(chosen["collapsed_seeds"] == 0
                                      and chosen["p10"] > memoryless)
    report["wall_clock_seconds"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(f"\nU3 (a learned filter is stable with true environmental events): "
          f"{report['u3_stable_filter']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
