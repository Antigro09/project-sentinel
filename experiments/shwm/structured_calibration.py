"""D / E / L2-L5. Qualify the architecture on structured PUBLIC state, before pixels.

The K-phase arm was used to draw a conclusion about hidden state without first
showing it could predict phase-independent dynamics: it scored 0.1492 MAE against
a memoryless lookup's 0.0354. This module refuses to reach any representation
question until the predictor clears the public memoryless ceiling on a target
whose ceiling is known exactly.

The input is structured public state -- what the packet actually shows: the agent
cell, the switch mask, the wall mask, the goal cell, the previous action and the
public action result. It contains no hidden phase, no simulator step, no crossing
label, no future outcome and no evaluator field. The oracle arm adds exactly one
scalar, the true phase, and changes nothing else, so the difference between the
two arms is that scalar and not a change of model.

The target is the next public outcome as a 144-way cell class, not a regression.
That matters: the K phase measured MAE, which mixes "wrong cell" with "wrong by a
little", and a categorical target has an exactly computable Bayes ceiling.

The budget ladder -- 64, 256, 1024 updates -- is frozen here before validation.
The K phase used 60 full-batch steps and read a convergence artifact as a property
of recurrence, so the curve is reported rather than a single point.

    .venv-shwm/bin/python experiments/shwm/structured_calibration.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402
from readout_qualification import bootstrap_difference  # noqa: E402

CELLS = GRID * GRID
BUDGET_LADDER = (64, 256, 1024)
"""Frozen before validation. The K phase used 60 and read the resulting
under-convergence as a fact about recurrence."""

HIDDEN = 128
PARAMETER_CEILING = 250_000
SEED = 6600
BATCH = 128


def collect(layouts, trajectories, steps, appearance, seed):
    """Structured public state plus every action's public successor."""
    gate = AuthorityGate(gate_id="structured")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    generator = np.random.default_rng(seed)
    out = []
    for layout in layouts:
        for _ in range(trajectories):
            adapter.reset(layout, f"appearance:{appearance}")
            level = adapter._require()
            switches = {tuple(int(v) for v in c) for c in level.switches}
            walls = np.asarray(level.walls, dtype=bool)
            goal = tuple(int(v) for v in level.markers[adapter._goal_marker])
            rows, previous_action = [], -1
            for step in range(steps):
                truth = adapter.snapshot().reveal("evaluator")
                position = tuple(int(v) for v in truth["position"])
                snapshot = adapter.snapshot()
                successors = []
                for candidate in ACTIONS:
                    adapter.restore(snapshot)
                    adapter.step(candidate, gate.authorize_evaluator(candidate, "s"))
                    successors.append(tuple(int(v) for v in
                                            adapter.snapshot().reveal("evaluator")["position"]))
                adapter.restore(snapshot)
                rows.append({
                    "position": position, "switches": switches, "walls": walls,
                    "goal": goal, "previous_action": previous_action,
                    "blocked": float(truth["last_blocked"]),
                    "successors": successors,
                    "polarity": int(truth["polarity"]),       # oracle / evaluator only
                    "crossings": int(truth["switch_crossings"]),
                })
                action = int(generator.integers(0, len(ACTIONS)))
                previous_action = action
                if adapter.step(action, gate.authorize_evaluator(action, "r")).terminated:
                    break
            if len(rows) >= 3:
                out.append({"layout": layout, "rows": rows})
    return out


DELTAS_BY_INDEX = ((-1, 0), (0, 1), (1, 0), (0, -1))
DISPLACEMENTS = DELTAS_BY_INDEX + ((0, 0),)
CLASSES = len(DISPLACEMENTS)


def encode(row, action, with_phase: bool) -> np.ndarray:
    """LOCAL structured public features, so the rule can transfer across layouts.

    A first version used absolute 144-way one-hots for agent, switches, walls and goal.
    That makes the mapping a per-layout lookup table: the predictor reached train
    accuracy 0.9661 and held-out 0.0655, because "the successor is adjacent to the
    agent" is not expressible in a form that survives a new wall pattern. Everything
    here is relative to the agent instead, and the target is a displacement rather than
    an absolute cell, so what is learned is a rule and not a table.

    Every field is genuinely visible in the packet. Note what is NOT here: whether the
    agent's own cell is a switch, because the renderer paints the agent over it.
    """
    row_index, column = row["position"]
    walls = row["walls"]
    neighbour_blocked, neighbour_switch = [], []
    for dr, dc in DELTAS_BY_INDEX:
        r, c = row_index + dr, column + dc
        inside = 0 <= r < GRID and 0 <= c < GRID
        neighbour_blocked.append(0.0 if inside and not walls[r, c] else 1.0)
        neighbour_switch.append(1.0 if inside and (r, c) in row["switches"] else 0.0)
    goal_direction = [
        float(np.sign(row["goal"][0] - row_index)), float(np.sign(row["goal"][1] - column)),
        abs(row["goal"][0] - row_index) / GRID, abs(row["goal"][1] - column) / GRID,
    ]
    previous = np.zeros(len(ACTIONS) + 1, dtype=np.float32)
    previous[row["previous_action"] + 1] = 1.0
    query = np.zeros(len(ACTIONS), dtype=np.float32)
    query[action] = 1.0
    parts = [
        np.asarray(neighbour_blocked, dtype=np.float32),
        np.asarray(neighbour_switch, dtype=np.float32),
        np.asarray(goal_direction, dtype=np.float32),
        previous, query,
        np.array([row["blocked"]], dtype=np.float32),
        np.array([row_index / GRID, column / GRID], dtype=np.float32),
    ]
    if with_phase:
        parts.append(np.array([float(row["polarity"])], dtype=np.float32))
    return np.concatenate(parts)


def target_cell(row, action) -> int:
    """The public displacement class, not an absolute cell.

    Displacement is what phase actually controls -- polarity negates the action delta --
    and it transfers across layouts, which an absolute cell index does not.
    """
    r0, c0 = row["position"]
    r1, c1 = row["successors"][action]
    delta = (r1 - r0, c1 - c0)
    return DISPLACEMENTS.index(delta) if delta in DISPLACEMENTS else CLASSES - 1


def flatten(trajectories, with_phase):
    x, y, groups, meta = [], [], [], []
    for trajectory in trajectories:
        for index, row in enumerate(trajectory["rows"]):
            for action in ACTIONS:
                x.append(encode(row, action, with_phase))
                y.append(target_cell(row, action))
                groups.append(trajectory["layout"])
                meta.append({"row": row, "action": action, "layout": trajectory["layout"],
                             "index": index, "trajectory": id(trajectory)})
    return np.stack(x), np.array(y), np.array(groups), meta


def train_mlp(x, y, updates, seed=SEED):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    mx.random.seed(seed)

    class Predictor(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.a = nn.Linear(width, HIDDEN)
            self.b = nn.Linear(HIDDEN, HIDDEN)
            self.head = nn.Linear(HIDDEN, CLASSES)

        def __call__(self, z):
            z = nn.relu(self.a(z))
            z = nn.relu(self.b(z))
            return self.head(z)

    model = Predictor(x.shape[1])
    mx.eval(model.parameters())
    parameters = int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))
    assert parameters <= PARAMETER_CEILING, parameters
    optimizer = optim.AdamW(learning_rate=2e-3)
    rng = np.random.default_rng(7)
    for _ in range(updates):
        idx = rng.integers(0, len(x), BATCH)
        xb, yb = mx.array(x[idx]), mx.array(y[idx].astype(np.int32))

        def loss_fn(m):
            return nn.losses.cross_entropy(m(xb), yb, reduction="mean")

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    def predict(features):
        out = []
        for k in range(0, len(features), 512):
            logits = model(mx.array(features[k:k + 512]))
            mx.eval(logits)
            out.append(np.asarray(logits))
        return np.concatenate(out)

    return predict, parameters


# ---- recurrent conditions -------------------------------------------------------------------


def train_gru(sequences, targets, updates, mode, seed=SEED):
    """Conditions 3 and 4: the same head, over a history instead of one step."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    mx.random.seed(seed)
    width = sequences[0].shape[1]

    class Recurrent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Linear(width, HIDDEN)
            self.gru = nn.GRU(HIDDEN, HIDDEN)
            self.head = nn.Linear(HIDDEN, CLASSES)

        def __call__(self, z, recurrent=True):
            z = nn.relu(self.project(z))
            h = self.gru(z) if recurrent else mx.concatenate(
                [self.gru(z[:, t:t + 1]) for t in range(z.shape[1])], axis=1)
            return self.head(h)

    model = Recurrent()
    mx.eval(model.parameters())
    parameters = int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))
    assert parameters <= PARAMETER_CEILING, parameters
    optimizer = optim.AdamW(learning_rate=2e-3)

    length = max(len(s) for s in sequences)
    x = np.zeros((len(sequences), length, width), dtype=np.float32)
    y = np.zeros((len(sequences), length), dtype=np.int32)
    m = np.zeros((len(sequences), length), dtype=np.float32)
    for i, (s, t) in enumerate(zip(sequences, targets)):
        x[i, :len(s)] = s
        y[i, :len(t)] = t
        m[i, :len(s)] = 1.0
    rng = np.random.default_rng(9)
    # The controls must disturb only the PAST. An earlier version permuted the whole
    # sequence, so the step being predicted was itself moved and the drop measured "the
    # model lost its own current input" rather than "the model lost its history". A GRU
    # reads a prefix, so corrupting the order of everything BEFORE the last step and
    # evaluating only at the last step isolates the history contribution.
    if mode in ("reversed_history", "shuffled_history"):
        for i in range(len(x)):
            valid = int(m[i].sum())
            if valid < 3:
                continue
            prefix = np.arange(valid - 1)
            order = prefix[::-1] if mode == "reversed_history" else rng.permutation(prefix)
            x[i, : valid - 1] = x[i, order]

    for _ in range(updates):
        pick = rng.integers(0, len(x), min(32, len(x)))
        xb, yb, mb = mx.array(x[pick]), mx.array(y[pick]), mx.array(m[pick])

        def loss_fn(mo):
            logits = mo(xb)
            losses = nn.losses.cross_entropy(
                logits.reshape(-1, CLASSES), yb.reshape(-1), reduction="none")
            return (losses * mb.reshape(-1)).sum() / mb.sum()

        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

    def predict(seqs):
        out = []
        for s in seqs:
            logits = model(mx.array(s[None]))
            mx.eval(logits)
            out.append(np.asarray(logits)[0][:len(s)])
        return out

    return predict, parameters


def to_sequences(trajectories, with_phase, action_mode="correct"):
    """One row per step, querying the action actually taken next."""
    rng = np.random.default_rng(3)
    sequences, targets = [], []
    for trajectory in trajectories:
        rows = trajectory["rows"]
        actions = [int(rng.integers(0, len(ACTIONS))) for _ in rows]
        if action_mode == "shuffled":
            actions = list(rng.permutation(actions))
        block = np.stack([encode(r, a, with_phase) for r, a in zip(rows, actions)])
        sequences.append(block.astype(np.float32))
        targets.append(np.array([target_cell(r, a) for r, a in zip(rows, actions)]))
    return sequences, targets


# ---- alias-pair candidate ranking ------------------------------------------------------------


def alias_evaluation_set(layouts, depth):
    """Exact alias pairs, encoded with the same structured encoder.

    Two candidate outcomes are presented per example -- the member's own successor and
    its partner's -- and the model must score its own above the other. The public
    packet is identical across the pair, so a memoryless model is at exactly 0.5 and
    only phase can lift it.
    """
    from alias_audit import enumerate_states
    states = enumerate_states(layouts, depth)
    classes = defaultdict(list)
    for state in states:
        classes[state.key("V2_agent_visible")].append(state)
    examples = []
    for members in classes.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a.polarity == b.polarity:
                    continue
                for action in ACTIONS:
                    if a.successors[action] == b.successors[action]:
                        continue
                    examples.append((a, b, action))
    return examples


def row_from_state(state) -> dict[str, Any]:
    """Rebuild the structured PUBLIC row for an enumerated state.

    Uses the level's own geometry, which is rendered and therefore public; polarity is
    attached for the oracle arm only and is never read unless `with_phase` is set.
    """
    from sentinel.env.adapters.procedural_visual_v2 import build_level_v2, MARKERS, _stream
    level = build_level_v2(state.layout, state.layout, state.layout)
    marker = MARKERS[int(_stream({"axis": "goal_v2", "seed": state.layout}, 1)[0]) % len(MARKERS)]
    return {
        "position": state.position,
        "switches": {tuple(int(v) for v in c) for c in level.switches},
        "walls": np.asarray(level.walls, dtype=bool),
        "goal": tuple(int(v) for v in level.markers[marker]),
        "previous_action": state.previous_action,
        "blocked": float(state.blocked),
        "polarity": state.polarity,
    }


def rank_alias(predict, examples, with_phase) -> dict[str, Any]:
    """Score each member's true successor against its partner's."""
    features, truths, others, groups = [], [], [], []
    cache: dict[Any, Any] = {}
    for a, b, action in examples:
        for me, partner in ((a, b), (b, a)):
            key = (me.layout, me.position, me.polarity, me.previous_action, me.blocked)
            if key not in cache:
                cache[key] = row_from_state(me)
            row = cache[key]
            # successors are observable_signature = r*GRID + c; the head predicts a
            # DISPLACEMENT class, so both candidates are converted the same way. The two
            # members share a position (they alias on the frame), so their displacements
            # differ exactly when their successors do.
            def to_class(signature, origin):
                cell = int(signature)
                delta = (cell // GRID - origin[0], cell % GRID - origin[1])
                return DISPLACEMENTS.index(delta) if delta in DISPLACEMENTS else CLASSES - 1

            features.append(encode(row, action, with_phase))
            truths.append(to_class(me.successors[action], me.position))
            others.append(to_class(partner.successors[action], me.position))
            groups.append(me.layout)
    if not features:
        return {"pairs": 0}
    logits = predict(np.stack(features))
    hits, margins = [], []
    for row_logits, t, o in zip(logits, truths, others):
        if t == o:
            continue          # the pair does not discriminate for this action
        margin = float(row_logits[t] - row_logits[o])
        margins.append(margin)
        hits.append(1.0 if margin > 0 else (0.5 if margin == 0 else 0.0))
    hits = np.array(hits)
    low, high = bootstrap_difference(hits, np.array(groups), 0.5)
    return {
        "pairs": int(len(hits)),
        "pairwise_accuracy": float(hits.mean()),
        "chance": 0.5,
        "mean_ranking_margin": float(np.mean(margins)),
        "ci_low_vs_chance": low, "ci_high_vs_chance": high,
    }


# ---- driver -----------------------------------------------------------------------------------


def analytic_memoryless_ceiling(trajectories) -> float:
    """The exact public memoryless ceiling for the displacement target.

    Not estimated by grouping inputs: with a 24-dimensional continuous encoding almost
    every input is unique, so that estimate returns ~1.0 and measures nothing -- the same
    singleton-class artifact that inflated an earlier version of this figure.

    Computed instead from the dynamics. Under phase h, action a moves by DELTAS[a] if
    h == 0 and by its negation otherwise, subject to walls. A memoryless predictor sees
    the same public state under both phases, so it is right with probability 1 when the
    two phases give the same displacement and 0.5 when they differ.
    """
    from sentinel.env.adapters.procedural_visual_v2 import build_level_v2
    hits = []
    for trajectory in trajectories:
        level = build_level_v2(trajectory["layout"], CANONICAL_APPEARANCE_SEED,
                               trajectory["layout"])
        walls = np.asarray(level.walls, dtype=bool)
        for row in trajectory["rows"]:
            r0, c0 = row["position"]
            for action in ACTIONS:
                outcomes = set()
                for phase in (0, 1):
                    dr, dc = DELTAS_BY_INDEX[action]
                    if phase:
                        dr, dc = -dr, -dc
                    r, c = r0 + dr, c0 + dc
                    blocked = not (0 <= r < GRID and 0 <= c < GRID) or bool(walls[r, c])
                    outcomes.add((0, 0) if blocked else (dr, dc))
                hits.append(1.0 if len(outcomes) == 1 else 0.5)
    return float(np.mean(hits))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-layouts", type=int, default=60)
    parser.add_argument("--test-layouts", type=int, default=30)
    parser.add_argument("--alias-layouts", type=int, default=10)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/structured-calibration.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    appearance = CANONICAL_APPEARANCE_SEED

    train_t = collect(list(range(61_000, 61_000 + arguments.train_layouts)),
                      arguments.trajectories, arguments.steps, appearance, 11)
    test_t = collect(list(range(81_000, 81_000 + arguments.test_layouts)),
                     2, arguments.steps, appearance, 777)
    print(f"trajectories: train {len(train_t)}  held-out {len(test_t)}", flush=True)

    print("building the exact alias-pair evaluation set", flush=True)
    examples = alias_evaluation_set(
        list(range(90_000, 90_000 + arguments.alias_layouts)), 6)
    print(f"  {len(examples)} (pair, action) examples where the outcome differs", flush=True)

    report: dict[str, Any] = {"budget_ladder": list(BUDGET_LADDER),
                              "alias_examples": len(examples), "conditions": {}}

    ceiling = analytic_memoryless_ceiling(test_t)
    report["held_out_memoryless_ceiling"] = ceiling
    print(f"  empirical public memoryless ceiling on held-out trajectories: {ceiling:.4f}",
          flush=True)

    print(f"\n{'condition':30s} {'updates':>8s} {'next-cell acc':>14s} {'NLL':>8s} "
          f"{'alias rank':>11s} {'CI low':>8s}")
    print("-" * 88)
    for with_phase, name in ((False, "1_structured_current"), (True, "2_plus_true_phase")):
        x_tr, y_tr, _, _ = flatten(train_t, with_phase)
        x_te, y_te, g_te, _ = flatten(test_t, with_phase)
        for updates in BUDGET_LADDER:
            predict, parameters = train_mlp(x_tr, y_tr, updates)
            logits = predict(x_te)
            accuracy = float((logits.argmax(axis=1) == y_te).mean())
            shifted = logits - logits.max(axis=1, keepdims=True)
            probability = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            nll = float(-np.log(np.maximum(probability[np.arange(len(y_te)), y_te], 1e-12)).mean())
            low, high = bootstrap_difference(
                (logits.argmax(axis=1) == y_te).astype(float), g_te, ceiling)
            alias = rank_alias(predict, examples, with_phase)
            report["conditions"][f"{name}@{updates}"] = {
                "condition": name, "updates": updates, "parameters": parameters,
                "next_cell_accuracy": accuracy, "nll": nll,
                "ci_low_vs_ceiling": low, "ci_high_vs_ceiling": high,
                "alias": alias,
            }
            print(f"{name:30s} {updates:8d} {accuracy:14.4f} {nll:8.4f} "
                  f"{alias.get('pairwise_accuracy', float('nan')):11.4f} "
                  f"{alias.get('ci_low_vs_chance', float('nan')):+8.3f}", flush=True)

    for mode in ("correct_history", "reversed_history", "shuffled_history"):
        sequences, targets = to_sequences(train_t, with_phase=False)
        test_sequences, test_targets = to_sequences(test_t, with_phase=False)
        for updates in BUDGET_LADDER:
            predict, parameters = train_gru(sequences, targets, updates, mode)
            preds = predict(test_sequences)
            # Scored at the FINAL step only. The order controls corrupt the prefix, so
            # every mode has its own current row intact there and the difference between
            # modes is the history and nothing else. Scoring all steps would compare
            # "lost its history" against "lost its own input".
            hits = np.array([float(p[-1].argmax() == t[len(p) - 1])
                             for p, t in zip(preds, test_targets)])
            groups = np.array([tr["layout"] for tr in test_t])
            low, high = bootstrap_difference(hits, groups, ceiling)
            report["conditions"][f"3_{mode}@{updates}"] = {
                "condition": f"3_{mode}", "updates": updates, "parameters": parameters,
                "next_cell_accuracy": float(hits.mean()),
                "ci_low_vs_ceiling": low, "ci_high_vs_ceiling": high,
            }
            print(f"{'3_' + mode:30s} {updates:8d} {hits.mean():14.4f} {'-':>8s} "
                  f"{'-':>11s} {'-':>8s}", flush=True)

    report["wall_clock_seconds"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    print(f"\nwrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
