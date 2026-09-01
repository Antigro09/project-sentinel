"""G / K9. The main arm: a world model given no privileged event bit.

Everything else in the K phase decodes observables and then applies a
hand-specified relation to obtain the switch-crossing event. That pipeline is
useful and it scores well, but it is not a world model: the relation encodes what
a switch crossing *is*, which is knowledge about the environment supplied by me
rather than inferred.

This arm gets none of it. Its input is the agent-visible packet sequence and
nothing else -- no event bit, no mask supervision, no hidden value, no derived
parity. Its target is a public observable: where each action would lead. If it
can rank same-action outcomes correctly on packet-alias pairs, it has inferred the
hidden phase from history on its own, because on those pairs the current packet is
identical and only history differs.

The controls are the point of the measurement. Correct history is compared
against a shuffled history and against a no-recurrence ablation running the same
parameters one step at a time, so a win has to come from using the past rather
than from capacity.

    .venv-shwm/bin/python experiments/shwm/main_arm.py
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

from sentinel.env.adapters.procedural_visual_v2 import (  # noqa: E402
    ACTIONS, GRID, ProceduralVisualV2Adapter,
)
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.slot_geometry import GEOMETRY_B, raw_slots  # noqa: E402
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402
from readout_qualification import bootstrap_difference  # noqa: E402

HIDDEN = 96
EPOCHS = 60
SEED = 6600
PARAMETER_CEILING = 250_000


def collect(layouts, trajectories, steps, appearance, seed):
    """Visible packets plus the public successor of every action. No hidden values."""
    gate = AuthorityGate(gate_id="main-arm")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    generator = np.random.default_rng(seed)
    out = []
    for layout in layouts:
        for _ in range(trajectories):
            adapter.reset(layout, f"appearance:{appearance}")
            rows, previous_action = [], -1
            for step in range(steps):
                snapshot = adapter.snapshot()
                successors = []
                for candidate in ACTIONS:
                    adapter.restore(snapshot)
                    adapter.step(candidate, gate.authorize_evaluator(candidate, "s"))
                    successors.append(
                        float(adapter.probes().values["observable_signature"]))
                adapter.restore(snapshot)
                truth = adapter.snapshot().reveal("evaluator")
                rows.append({
                    "slots": raw_slots(adapter.frame(), GEOMETRY_B).reshape(-1),
                    "previous_action": previous_action,
                    "blocked": float(truth["last_blocked"]),
                    "successors": successors,
                    # evaluator-only, never an input -- kept for reporting alone
                    "polarity": int(truth["polarity"]),
                    "position": tuple(int(v) for v in truth["position"]),
                })
                action = int(generator.integers(0, len(ACTIONS)))
                previous_action = action
                if adapter.step(action, gate.authorize_evaluator(action, "r")).terminated:
                    break
            if len(rows) >= 3:
                out.append({"layout": layout, "rows": rows})
    return out


SLOT_PROJECTION_WIDTH = 256


def _slot_projection(rows: int) -> np.ndarray:
    """A frozen random projection of the slot block, identical for every mode.

    Without it the input layer alone is 394k parameters and the arm cannot fit
    under the pre-registered ceiling. The projection is drawn once from a fixed
    seed and never trained, so it reduces dimensionality without adding capacity
    and cannot differ between the correct-history arm and its controls.
    """
    generator = np.random.default_rng(20250901)
    return (generator.normal(size=(rows, SLOT_PROJECTION_WIDTH))
            / np.sqrt(rows)).astype(np.float32)


def to_sequences(trajectories):
    """(T, F) per trajectory: projected slots, previous action one-hot, result, delta_t."""
    raw_width = len(trajectories[0]["rows"][0]["slots"])
    projection = _slot_projection(raw_width)
    width = SLOT_PROJECTION_WIDTH + len(ACTIONS) + 1 + 1 + 1
    sequences, targets, meta = [], [], []
    for trajectory in trajectories:
        rows = trajectory["rows"]
        block = np.zeros((len(rows), width), dtype=np.float32)
        target = np.zeros((len(rows), len(ACTIONS)), dtype=np.float32)
        for t, row in enumerate(rows):
            n = SLOT_PROJECTION_WIDTH
            block[t, :n] = np.asarray(row["slots"], dtype=np.float32) @ projection
            if row["previous_action"] >= 0:
                block[t, n + row["previous_action"]] = 1.0
            else:
                block[t, n + len(ACTIONS)] = 1.0
            block[t, n + len(ACTIONS) + 1] = row["blocked"]
            block[t, n + len(ACTIONS) + 2] = 1.0          # delta_t, the constant
            target[t] = np.asarray(row["successors"]) / float(GRID * GRID)
        sequences.append(block)
        targets.append(target)
        meta.append(trajectory)
    return sequences, targets, meta


def train(sequences, targets, mode: str, seed: int = SEED):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    mx.random.seed(seed)
    width = sequences[0].shape[1]

    class Forward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Linear(width, HIDDEN)
            self.gru = nn.GRU(HIDDEN, HIDDEN)
            self.head = nn.Linear(HIDDEN, len(ACTIONS))

        def __call__(self, x, recurrent: bool = True):
            z = nn.relu(self.project(x))
            if recurrent:
                h = self.gru(z)
            else:
                h = mx.concatenate(
                    [self.gru(z[:, t:t + 1]) for t in range(z.shape[1])], axis=1)
            return self.head(h)

    model = Forward()
    mx.eval(model.parameters())
    parameters = int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))
    assert parameters <= PARAMETER_CEILING, parameters
    optimizer = optim.AdamW(learning_rate=2e-3)

    length = max(len(s) for s in sequences)
    x = np.zeros((len(sequences), length, width), dtype=np.float32)
    y = np.zeros((len(sequences), length, len(ACTIONS)), dtype=np.float32)
    m = np.zeros((len(sequences), length), dtype=np.float32)
    for i, (s, t) in enumerate(zip(sequences, targets)):
        x[i, :len(s)] = s
        y[i, :len(t)] = t
        m[i, :len(s)] = 1.0
    if mode == "shuffled_history":
        rng = np.random.default_rng(5)
        for i in range(len(x)):
            valid = int(m[i].sum())
            x[i, :valid] = x[i, rng.permutation(valid)]
    recurrent = mode != "no_recurrence"

    xb, yb, mb = mx.array(x), mx.array(y), mx.array(m)

    def loss_fn(model_):
        prediction = model_(xb, recurrent)
        return (mx.mean((prediction - yb) ** 2, axis=-1) * mb).sum() / mb.sum()

    grad_fn = nn.value_and_grad(model, loss_fn)
    for _ in range(EPOCHS):
        loss, grads = grad_fn(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    def predict(seqs):
        out = []
        for s in seqs:
            p = model(mx.array(s[None]), recurrent)
            mx.eval(p)
            out.append(np.asarray(p)[0][:len(s)])
        return out

    return predict, parameters


def alias_pairs_with_routes(layouts, depth, appearance):
    """Pairs sharing the v2 AgentVisiblePacket, differing in phase, with routes.

    Reuses the audit's enumerator so the pairs here are the same objects the
    certificate counts, rather than a second construction that might differ.
    """
    from alias_audit import LEVELS, enumerate_states
    from collections import defaultdict

    states = enumerate_states(layouts, depth)
    groups = defaultdict(list)
    for state in states:
        groups[(state.layout, state.key("V2_agent_visible"))].append(state)
    pairs = []
    for members in groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a.polarity == b.polarity or a.successors == b.successors:
                    continue
                pairs.append((a, b))
    return pairs


def replay_sequence(adapter, gate, layout, route, appearance):
    adapter.reset(layout, f"appearance:{appearance}")
    rows, previous_action = [], -1
    for step in range(len(route) + 1):
        truth = adapter.snapshot().reveal("evaluator")
        rows.append({
            "slots": raw_slots(adapter.frame(), GEOMETRY_B).reshape(-1),
            "previous_action": previous_action,
            "blocked": float(truth["last_blocked"]),
            "successors": [0.0] * len(ACTIONS),
        })
        if step == len(route):
            break
        previous_action = route[step]
        adapter.step(route[step], gate.authorize_evaluator(route[step], "replay"))
    return {"layout": layout, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-layouts", type=int, default=90)
    parser.add_argument("--test-layouts", type=int, default=45)
    parser.add_argument("--pair-layouts", type=int, default=20)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/main-arm.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    appearance = CANONICAL_APPEARANCE_SEED

    train_t = collect(list(range(61_000, 61_000 + arguments.train_layouts)),
                      arguments.trajectories, arguments.steps, appearance, 11)
    test_t = collect(list(range(81_000, 81_000 + arguments.test_layouts)),
                     2, arguments.steps, appearance, 777)
    print(f"trajectories: train {len(train_t)}  held-out {len(test_t)}", flush=True)
    train_x, train_y, _ = to_sequences(train_t)
    test_x, test_y, test_meta = to_sequences(test_t)

    print("building packet-alias pairs (identical visible packet, different phase)",
          flush=True)
    pairs = alias_pairs_with_routes(
        list(range(90_000, 90_000 + arguments.pair_layouts)), 6, appearance)
    print(f"  {len(pairs)} pairs", flush=True)

    gate = AuthorityGate(gate_id="main-arm-pairs")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    pair_sequences, pair_truth = [], []
    for a, b in pairs[:400]:
        sa = replay_sequence(adapter, gate, a.layout, a.route, appearance)
        sb = replay_sequence(adapter, gate, b.layout, b.route, appearance)
        xa, _, _ = to_sequences([sa])
        xb, _, _ = to_sequences([sb])
        pair_sequences.append((xa[0], xb[0]))
        pair_truth.append((np.asarray(a.successors) / float(GRID * GRID),
                           np.asarray(b.successors) / float(GRID * GRID), a.layout))

    report: dict[str, Any] = {"modes": {}, "pairs": len(pair_sequences),
                              "privileged_event_bits": False,
                              "mask_supervision": False, "hidden_values_in_input": False}

    for mode in ("correct_history", "shuffled_history", "no_recurrence"):
        predict, parameters = train(train_x, train_y, mode)
        # held-out next-outcome error
        predictions = predict(test_x)
        errors = np.concatenate([
            np.abs(p - t[:len(p)]).mean(axis=1) for p, t in zip(predictions, test_y)])
        groups = np.concatenate([
            np.full(len(p), m["layout"]) for p, m in zip(predictions, test_meta)])

        # alias-pair ranking: identical packet, different history -> different outcome
        correct, pair_groups = [], []
        for (xa, xb), (ya, yb, layout) in zip(pair_sequences, pair_truth):
            pa, pb = predict([xa])[0][-1], predict([xb])[0][-1]
            hits = 0
            for action in range(len(ACTIONS)):
                if ya[action] == yb[action]:
                    continue
                delta_pred = pa[action] - pb[action]
                delta_true = ya[action] - yb[action]
                hits += 1 if delta_pred * delta_true > 0 else (
                    0.5 if delta_pred == 0 else 0)
            correct.append(hits / max(1, sum(ya != yb)))
            pair_groups.append(layout)
        correct = np.array(correct, dtype=float)
        low, high = bootstrap_difference(correct, np.array(pair_groups), 0.5)
        report["modes"][mode] = {
            "parameters": parameters,
            "held_out_mean_absolute_error": float(errors.mean()),
            "alias_pair_ranking_accuracy": float(correct.mean()),
            "chance": 0.5, "ci_low_vs_chance": low, "ci_high_vs_chance": high,
            "pairs_scored": int(len(correct)),
        }
        print(f"  {mode:18s} held-out MAE {errors.mean():.4f}  alias-pair ranking "
              f"{correct.mean():.4f}  CI[{low:+.3f},{high:+.3f}]  p={parameters}", flush=True)

    correct_mode = report["modes"]["correct_history"]
    report["k6_correct_history_improves_alias_ranking"] = all(
        correct_mode["alias_pair_ranking_accuracy"]
        > report["modes"][m]["alias_pair_ranking_accuracy"]
        for m in ("shuffled_history", "no_recurrence"))
    report["wall_clock_seconds"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nK6 (correct history beats controls on alias pairs): "
          f"{report['k6_correct_history_improves_alias_ranking']}")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
