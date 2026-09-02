"""C / M2. Pure parity, with the environment removed entirely.

The L phase established that the prediction head reaches the oracle the moment
phase is supplied, and that the same architecture given history instead reaches
nothing. That leaves six candidate explanations, and this module separates the
cheapest one first: can the temporal implementation learn parity at all, on a task
with no perception, no environment and no event extraction?

    h_T = h_0 XOR parity(c_1 ... c_T)

The recurrent arm is the *same* implementation the structured-history model used --
a linear projection, `nn.GRU`, a linear head -- so a failure here is a failure
there. Two learned finite-state filters are included because the specification is
explicit that a generic GRU need not be the winner: a belief over a small number of
unnamed states, updated by a learned stochastic matrix conditioned on the event, is
a legitimate temporal candidate and is scored only up to permutation.

Nothing receives a phase label at inference. `h_0` is an input at step 0 only,
exactly as the reset frame supplies it in the environment.

    .venv-shwm/bin/python experiments/shwm/parity_microcase.py
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

HIDDEN = 128          # matched to structured_calibration.HIDDEN
PARAMETER_CEILING = 250_000
BUDGETS = (64, 256, 1024)
SEEDS = (6600, 6601, 6602)
DEV_LENGTHS = range(1, 9)
EXTRAPOLATION_LENGTHS = range(9, 17)


def make_dataset(count, lengths, seed):
    """Sequences of (event, h0-at-step-0, first-step flag) with per-step parity targets."""
    rng = np.random.default_rng(seed)
    sequences, targets = [], []
    lengths = list(lengths)
    for _ in range(count):
        length = int(rng.choice(lengths))
        events = rng.integers(0, 2, size=length)
        h0 = int(rng.integers(0, 2))
        block = np.zeros((length, 3), dtype=np.float32)
        block[:, 0] = events
        block[0, 1] = float(h0)      # the initial value, visible once
        block[0, 2] = 1.0            # a "this is the first step" flag
        sequences.append(block)
        targets.append(((h0 + np.cumsum(events)) % 2).astype(np.int32))
    return sequences, targets


def corrupt(sequences, mode, seed=17):
    """Derive a control from an existing dataset, leaving the TARGETS untouched.

    Regenerating with a different mode was wrong: changing the corruption changes the
    number of RNG draws per example, so the length and event streams diverged and the
    control was scored against targets from different sequences.
    """
    rng = np.random.default_rng(seed)
    out = []
    for block in sequences:
        copy = np.array(block)
        if mode == "shuffled_events":
            copy[:, 0] = rng.permutation(copy[:, 0])
        elif mode == "reversed_events":
            copy[:, 0] = copy[::-1, 0]
        elif mode == "no_events":
            copy[:, 0] = 0.0
        out.append(copy)
    return out


def exact_accumulator(sequences, targets):
    """Closed form over the events actually fed to the model."""
    hits = []
    for block, target in zip(sequences, targets):
        h = int(block[0, 1])
        for t in range(len(block)):
            h ^= int(block[t, 0])
            hits.append(float(h == target[t]))
    return float(np.mean(hits))


def pad(sequences, targets):
    length = max(len(s) for s in sequences)
    width = sequences[0].shape[1]
    x = np.zeros((len(sequences), length, width), dtype=np.float32)
    y = np.zeros((len(sequences), length), dtype=np.int32)
    m = np.zeros((len(sequences), length), dtype=np.float32)
    for i, (s, t) in enumerate(zip(sequences, targets)):
        x[i, :len(s)] = s
        y[i, :len(t)] = t
        m[i, :len(s)] = 1.0
    return x, y, m


def build(kind: str, width: int):
    """The three learned temporal candidates."""
    import mlx.core as mx
    import mlx.nn as nn

    class GenericRecurrent(nn.Module):
        """Exactly the structured-history implementation."""
        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Linear(width, HIDDEN)
            self.gru = nn.GRU(HIDDEN, HIDDEN)
            self.head = nn.Linear(HIDDEN, 2)

        def __call__(self, z):
            return self.head(self.gru(nn.relu(self.project(z))))

    class LearnedFilter(nn.Module):
        """belief_t = normalize(T_e^T belief_{t-1}), with T_e learned per event.

        Knows only how many latent states there are, never their meaning and never
        the true transition matrix. Scored up to permutation, so a solution that
        labels the states the other way round counts.
        """
        def __init__(self, states: int) -> None:
            super().__init__()
            self.states = states
            # Zero-initialised transition logits are a symmetric fixed point: softmax
            # gives a uniform matrix for both event values, the belief becomes uniform
            # after one step, and the gradient to every event position is exactly zero.
            # The gradient diagnostic reported 0.00e+00 across the board, which is what
            # identified this as an implementation defect rather than a weak arm --
            # objective evidence independent of any validation score.
            self.logits = mx.random.normal((2, states, states)) * 0.5
            self.initial = nn.Linear(width, states)
            self.head = nn.Linear(states, 2)

        def __call__(self, z):
            batch, length, _ = z.shape
            transition = mx.softmax(self.logits, axis=-1)      # (2, S, S) row-stochastic
            belief = mx.softmax(self.initial(z[:, 0]), axis=-1)
            outputs = []
            for t in range(length):
                event = z[:, t, 0][:, None, None]
                matrix = transition[0][None] * (1.0 - event) + transition[1][None] * event
                belief = mx.sum(belief[:, :, None] * matrix, axis=1)
                belief = belief / mx.maximum(mx.sum(belief, axis=-1, keepdims=True), 1e-9)
                outputs.append(self.head(belief))
            return mx.stack(outputs, axis=1)

    if kind == "generic_recurrent":
        return GenericRecurrent()
    if kind == "two_state_filter":
        return LearnedFilter(2)
    if kind == "eight_state_filter":
        return LearnedFilter(8)
    raise ValueError(kind)


def train_and_score(kind, train, validation, extrapolation, updates, seed):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    mx.random.seed(seed)
    x, y, m = pad(*train)
    model = build(kind, x.shape[2])
    mx.eval(model.parameters())
    parameters = int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))
    assert parameters <= PARAMETER_CEILING, (kind, parameters)
    optimizer = optim.AdamW(learning_rate=3e-3)
    rng = np.random.default_rng(seed)
    xb_all, yb_all, mb_all = mx.array(x), mx.array(y), mx.array(m)

    for _ in range(updates):
        pick = rng.integers(0, len(x), min(64, len(x)))
        xb, yb, mb = mx.array(x[pick]), mx.array(y[pick]), mx.array(m[pick])

        def loss_fn(mo):
            logits = mo(xb)
            losses = nn.losses.cross_entropy(
                logits.reshape(-1, 2), yb.reshape(-1), reduction="none")
            return (losses * mb.reshape(-1)).sum() / mb.sum()

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    def score(dataset):
        xs, ys, ms = pad(*dataset)
        logits = model(mx.array(xs))
        mx.eval(logits)
        predicted = np.asarray(logits).argmax(axis=-1)
        valid = ms.astype(bool)
        return float((predicted[valid] == ys[valid]).mean())

    return {
        "arm": kind, "updates": updates, "seed": seed, "parameters": parameters,
        "train": score(train), "validation": score(validation),
        "extrapolation": score(extrapolation),
    }


def gradient_to_event_positions(kind, train, updates, seed):
    """How much does the final loss depend on each event position?

    A credit-assignment diagnostic: if the gradient at early events vanishes, the
    model cannot be learning to accumulate over them, whatever its accuracy says.
    """
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(seed)
    x, y, m = pad(*train)
    model = build(kind, x.shape[2])
    mx.eval(model.parameters())
    length = x.shape[1]
    xb = mx.array(x[:64])
    yb, mb = mx.array(y[:64]), mx.array(m[:64])

    def loss_of_input(inputs):
        logits = model(inputs)
        losses = nn.losses.cross_entropy(
            logits.reshape(-1, 2), yb.reshape(-1), reduction="none")
        return (losses * mb.reshape(-1)).sum() / mb.sum()

    grad = mx.grad(loss_of_input)(xb)
    mx.eval(grad)
    per_position = np.abs(np.asarray(grad)[:, :, 0]).mean(axis=0)
    return [float(v) for v in per_position[:length]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/parity-microcase.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    train = make_dataset(arguments.train, DEV_LENGTHS, 1)
    validation = make_dataset(arguments.validation, DEV_LENGTHS, 2)
    extrapolation = make_dataset(arguments.validation, EXTRAPOLATION_LENGTHS, 3)

    report: dict[str, Any] = {"budgets": list(BUDGETS), "seeds": list(SEEDS), "rows": []}

    exact = exact_accumulator(*validation)
    report["exact_xor_accumulator"] = exact
    print(f"exact XOR accumulator on validation: {exact:.4f}")
    for mode in ("shuffled_events", "reversed_events", "no_events"):
        # the targets still follow the TRUE events, so a corrupted stream must fail
        value = exact_accumulator(corrupt(validation[0], mode), validation[1])
        report[f"exact_accumulator_{mode}"] = value
        print(f"  same accumulator on {mode:16s}: {value:.4f}")

    print(f"\n{'arm':22s} {'updates':>8s} {'seed':>6s} {'train':>7s} {'val':>7s} {'extrap':>7s}")
    print("-" * 64)
    for kind in ("generic_recurrent", "two_state_filter", "eight_state_filter"):
        for updates in BUDGETS:
            for seed in SEEDS:
                row = train_and_score(kind, train, validation, extrapolation, updates, seed)
                report["rows"].append(row)
                print(f"{kind:22s} {updates:8d} {seed:6d} {row['train']:7.4f} "
                      f"{row['validation']:7.4f} {row['extrapolation']:7.4f}", flush=True)

    print("\ngradient magnitude from the final loss to each event position (untrained):")
    for kind in ("generic_recurrent", "two_state_filter"):
        grads = gradient_to_event_positions(kind, train, BUDGETS[-1], SEEDS[0])
        report[f"gradient_by_position_{kind}"] = grads
        print(f"  {kind:22s} " + " ".join(f"{g:.2e}" for g in grads[:8]))

    best = {}
    for row in report["rows"]:
        key = row["arm"]
        if key not in best or row["validation"] > best[key]["validation"]:
            best[key] = row
    report["best_by_arm"] = best
    report["m2_parity_learnable"] = any(
        r["validation"] > 0.90 for r in best.values())
    report["wall_clock_seconds"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nM2 (parity learnable by a non-oracle temporal model): "
          f"{report['m2_parity_learnable']}")
    for kind, row in best.items():
        print(f"  best {kind:22s} val {row['validation']:.4f} "
              f"(updates {row['updates']}, seed {row['seed']})")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
