"""J3-J7. Qualify a bounded readout family before it is allowed to judge anything.

The previous audit concluded that switch-event detection was unmeasurable. That
conclusion was drawn with a readout that could not recover agent position from
raw pixels — a representation that provably contains it — so its negative
findings were about the probe. This module fixes the probe first and only then
asks about representations.

Three design points, each answering a specific failure of the previous attempt.

*Appearance is held fixed.* The agent is rendered as `255 - palette[0]`, and the
palette is drawn per appearance seed, so with appearance tied to layout the agent
was a different colour in every layout and cross-layout localisation required an
appearance-invariant rule. Holding appearance at the canonical seed while letting
layouts vary gives a single agent colour with 12 distinct wall/switch
configurations, which is the systematic-generalisation question the specification
actually asks. Appearance shift is measured separately rather than folded in.

*Information presence and generalisation are separated.* Failure on held-out
layouts does not show that a representation destroyed information; it can equally
show that the readout cannot transfer a spatial rule. Split A keeps the layout
families and holds out trajectories, so it answers the information question.
Split B holds out layouts and answers the transfer question.

*Position R-squared is demoted to a diagnostic.* The gate is exact cell accuracy
and mask F1, because a regression score of 0.55 and an exact-accuracy of 0.55
mean very different things for a downstream difference.

    .venv-shwm/bin/python experiments/shwm/readout_qualification.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import (  # noqa: E402
    ACTIONS,
    CELL,
    GRID,
    ProceduralVisualV2Adapter,
    build_level_v2,
)
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402

PARAMETER_CAP = 250_000
"""Pre-registered cap for every learned readout, fixed before validation exposure."""

FRAME_SIDE = GRID * CELL


# ---- collection --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Step:
    layout: int
    trajectory: int
    step: int
    frame: np.ndarray
    previous_frame: np.ndarray
    position: tuple[int, int]
    previous_position: tuple[int, int]
    polarity: int
    previous_polarity: int
    crossings: int
    crossed_now: int
    previous_action: int
    switch_cells: tuple[tuple[int, int], ...]
    successors: tuple[float, ...]


def collect(layouts: Sequence[int], trajectories: int, steps: int,
            appearance: int | None, seed: int) -> list[Step]:
    """Rollouts under a fixed appearance seed unless one is requested otherwise."""
    gate = AuthorityGate(gate_id="readout-qual")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    generator = np.random.default_rng(seed)
    out: list[Step] = []
    for layout in layouts:
        dynamic = "base" if appearance is None else f"appearance:{appearance}"
        for trajectory in range(trajectories):
            adapter.reset(layout, dynamic)
            level = adapter._require()
            switches = tuple(sorted(tuple(int(v) for v in c) for c in level.switches))
            previous_frame = adapter.frame().copy()
            previous_position = tuple(int(v) for v in adapter._position)
            previous_polarity = int(adapter._polarity)
            for step in range(steps):
                truth = adapter.snapshot().reveal("evaluator")
                position = tuple(int(v) for v in truth["position"])
                polarity = int(truth["polarity"])
                snapshot = adapter.snapshot()
                successors = []
                for candidate in ACTIONS:
                    adapter.restore(snapshot)
                    adapter.step(candidate, gate.authorize_evaluator(candidate, "succ"))
                    successors.append(float(adapter.probes().values["observable_signature"]))
                adapter.restore(snapshot)
                out.append(Step(
                    layout=layout, trajectory=trajectory, step=step,
                    frame=adapter.frame().copy(), previous_frame=previous_frame,
                    position=position, previous_position=previous_position,
                    polarity=polarity, previous_polarity=previous_polarity,
                    crossings=int(truth["switch_crossings"]),
                    crossed_now=int(step > 0 and position != previous_position
                                    and position in set(switches)),
                    previous_action=-1 if step == 0 else previous_action_holder[0],
                    switch_cells=switches,
                    successors=tuple(successors),
                ))
                previous_frame = adapter.frame().copy()
                previous_position, previous_polarity = position, polarity
                action = int(generator.integers(0, len(ACTIONS)))
                previous_action_holder[0] = action
                if adapter.step(action, gate.authorize_evaluator(action, "roll")).terminated:
                    break
    return out


previous_action_holder = [-1]


# ---- readout 1: hand-coded renderer-aware oracle -------------------------------------------


def handcoded_decoder(appearance: int) -> Callable[[np.ndarray], dict[str, Any]]:
    """Calibration only. Knows the renderer and, since appearance is fixed, the palette.

    Its purpose is to establish that the visible state IS decodable from pixels.
    If this fails, the frames do not contain what the audit assumes and nothing
    else in the chain is interpretable.
    """
    level = build_level_v2(0, appearance, 0)
    agent_colour = (255 - level.palette[0].astype(int))
    switch_colour = level.palette[4].astype(int)

    def decode(frame: np.ndarray) -> dict[str, Any]:
        cells = frame.astype(int).reshape(GRID, CELL, GRID, CELL, 3).transpose(0, 2, 1, 3, 4)
        mean = cells.reshape(GRID, GRID, -1, 3).mean(axis=2)
        agent_distance = np.abs(mean - agent_colour[None, None, :]).sum(axis=2)
        switch_distance = np.abs(mean - switch_colour[None, None, :]).sum(axis=2)
        flat = int(np.argmin(agent_distance))
        return {
            "position": (flat // GRID, flat % GRID),
            "agent_mask": (agent_distance <= agent_distance.min() + 1e-6),
            "switch_mask": (switch_distance < 30),
        }

    return decode


# ---- readouts 2-3: learned raw-pixel decoders ----------------------------------------------


def build_heatmap_cnn(out_channels: int = 1):
    """Translation-equivariant: convolutions only, one logit per game cell.

    A dense head would let the readout memorise absolute positions per layout,
    which is exactly the overfitting that made the previous attempt unreadable
    (train 0.967, test 0.037).
    """
    import mlx.core as mx
    import mlx.nn as nn

    class HeatmapCNN(nn.Module):
        def __init__(self, in_channels: int) -> None:
            super().__init__()
            self.a = nn.Conv2d(in_channels, 32, 3, stride=1, padding=1)
            self.b = nn.Conv2d(32, 48, 3, stride=2, padding=1)   # 24 -> 12
            self.c = nn.Conv2d(48, 48, 3, stride=1, padding=1)
            self.d = nn.Conv2d(48, out_channels, 1, stride=1)

        def __call__(self, x: mx.array) -> mx.array:
            x = nn.relu(self.a(x))
            x = nn.relu(self.b(x))
            x = nn.relu(self.c(x))
            return self.d(x)                                      # (B, 12, 12, out)

    return HeatmapCNN


def count_parameters(model) -> int:
    from mlx.utils import tree_flatten
    return int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))


def train_heatmap(train_x, train_y, test_x, test_y, in_channels, epochs=40, seed=6600):
    """Cell-classification over a 12x12 grid, scored by exact-cell accuracy."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(seed)
    model = build_heatmap_cnn(1)(in_channels)
    mx.eval(model.parameters())
    parameters = count_parameters(model)
    if parameters > PARAMETER_CAP:
        raise ValueError(f"readout has {parameters} parameters, cap is {PARAMETER_CAP}")
    optimizer = optim.AdamW(learning_rate=2e-3)
    rng = np.random.default_rng(0)
    n = len(train_x)
    for _ in range(epochs):
        for k in range(0, n, 64):
            idx = rng.permutation(n)[k:k + 64] if k == 0 else rng.integers(0, n, 64)
            xb, yb = mx.array(train_x[idx]), mx.array(train_y[idx].astype(np.int32))

            def loss_fn(m):
                logits = m(xb).reshape(len(idx), GRID * GRID)
                return nn.losses.cross_entropy(logits, yb, reduction="mean")

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    def predict(x):
        outs = []
        for k in range(0, len(x), 256):
            logits = model(mx.array(x[k:k + 256])).reshape(-1, GRID * GRID)
            mx.eval(logits)
            outs.append(np.asarray(logits))
        return np.concatenate(outs)

    return predict(test_x).argmax(axis=1), predict(train_x).argmax(axis=1), parameters


def train_binary_cnn(train_x, train_y, test_x, in_channels, epochs=40, seed=6600):
    """Global binary decision (did this transition cross a switch) from stacked frames."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(seed)
    Base = build_heatmap_cnn(1)

    class EventCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = Base(in_channels)
            self.bias = nn.Linear(1, 2)

        def __call__(self, x: mx.array) -> mx.array:
            grid = self.body(x)                       # (B, 12, 12, 1)
            pooled = grid.reshape(grid.shape[0], -1).max(axis=1, keepdims=True)
            return self.bias(pooled)

    model = EventCNN()
    mx.eval(model.parameters())
    parameters = count_parameters(model)
    if parameters > PARAMETER_CAP:
        raise ValueError(f"readout has {parameters} parameters, cap is {PARAMETER_CAP}")
    optimizer = optim.AdamW(learning_rate=2e-3)
    rng = np.random.default_rng(1)
    positive = np.flatnonzero(train_y == 1)
    negative = np.flatnonzero(train_y == 0)
    for _ in range(epochs):
        for _ in range(max(1, len(train_x) // 64)):
            # Balanced batches: the positive rate is ~0.3 and an unbalanced batch
            # lets the readout score well by never predicting the event.
            idx = np.concatenate([rng.choice(positive, 32), rng.choice(negative, 32)])
            xb, yb = mx.array(train_x[idx]), mx.array(train_y[idx].astype(np.int32))

            def loss_fn(m):
                return nn.losses.cross_entropy(m(xb), yb, reduction="mean")

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    outs = []
    for k in range(0, len(test_x), 256):
        logits = model(mx.array(test_x[k:k + 256]))
        mx.eval(logits)
        outs.append(np.asarray(logits))
    logits = np.concatenate(outs)
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability = probability[:, 1] / probability.sum(axis=1)
    return logits.argmax(axis=1), probability, parameters



# ---- readout 3: object / relation decoder ---------------------------------------------------


def train_relation_decoder(train, test_sets, epochs=40, seed=6600):
    """Decode the objects, then relate them through a head with no parameters.

    The event "this transition entered a switch cell" is a relation between two
    decodable objects: where the agent is, and which cells are switches. This
    readout is trained ONLY on those two masks and never sees an event label; the
    event falls out of a fixed arithmetic combination afterwards. That ordering is
    what makes the result attributable -- the decoder cannot have fitted the event
    directly, so a good event score means the visible state was sufficient for it.

        p_moved     = 1 - sum_c  agent_t(c) * agent_{t-1}(c)
        p_on_switch = sum_c      agent_t(c) * switch_{t-1}(c)
        p_crossed   = p_moved * p_on_switch

    The switch mask is read from the PREVIOUS frame, and that is not a detail.
    `render_v2` draws the switches and then paints the agent over them, so the
    switch a agent is standing on is overwritten and simply absent from the
    current frame: measured directly, a switch cell reads (177,44,147) when the
    agent is away and (41,161,141) -- the agent colour -- when it is on it. "Am I
    on a switch" is therefore not a single-frame question in this environment. At
    t-1 the agent is elsewhere, so the destination cell still shows its own
    colour, and the relation is well posed again.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(seed)
    Base = build_heatmap_cnn(2)

    class ObjectDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = Base(3)

        def __call__(self, x: mx.array):
            grid = self.body(x)                                   # (B, 12, 12, 2)
            flat = grid.reshape(grid.shape[0], GRID * GRID, 2)
            return flat[:, :, 0], flat[:, :, 1]                   # agent logits, switch logits

    model = ObjectDecoder()
    mx.eval(model.parameters())
    parameters = count_parameters(model)
    if parameters > PARAMETER_CAP:
        raise ValueError(f"readout has {parameters} parameters, cap is {PARAMETER_CAP}")
    optimizer = optim.AdamW(learning_rate=2e-3)

    frames = stack_frames(train, False)
    agent_target = position_index(train)
    switch_target = np.zeros((len(train), GRID * GRID), dtype=np.float32)
    switch_visible = np.ones((len(train), GRID * GRID), dtype=np.float32)
    for i, step in enumerate(train):
        for cell in step.switch_cells:
            switch_target[i, cell[0] * GRID + cell[1]] = 1.0
        # The cell under the agent is overwritten by the agent, so its switch
        # state is not visible. Supervising it would train the decoder to
        # hallucinate a colour that is not in the image.
        switch_visible[i, step.position[0] * GRID + step.position[1]] = 0.0

    rng = np.random.default_rng(2)
    n = len(train)
    for _ in range(epochs):
        for _ in range(max(1, n // 64)):
            idx = rng.integers(0, n, 64)
            xb = mx.array(frames[idx])
            ab = mx.array(agent_target[idx].astype(np.int32))
            sb = mx.array(switch_target[idx])
            vb = mx.array(switch_visible[idx])

            def loss_fn(m):
                agent_logits, switch_logits = m(xb)
                agent_loss = nn.losses.cross_entropy(agent_logits, ab, reduction="mean")
                per_cell = nn.losses.binary_cross_entropy(
                    switch_logits, sb, with_logits=True, reduction="none")
                switch_loss = (per_cell * vb).sum() / vb.sum()
                return agent_loss + switch_loss

            loss, grads = nn.value_and_grad(model, loss_fn)(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

    def decode(steps):
        current = stack_frames(steps, False)
        previous = np.stack([s.previous_frame.astype(np.float32) / 255.0 for s in steps])
        agent_now, switch_prev, agent_prev = [], [], []
        for k in range(0, len(steps), 256):
            a1, _ = model(mx.array(current[k:k + 256]))
            a0, s0 = model(mx.array(previous[k:k + 256]))
            mx.eval(a1, a0, s0)
            agent_now.append(np.asarray(mx.softmax(a1, axis=-1)))
            switch_prev.append(np.asarray(mx.sigmoid(s0)))
            agent_prev.append(np.asarray(mx.softmax(a0, axis=-1)))
        return (np.concatenate(agent_now), np.concatenate(switch_prev),
                np.concatenate(agent_prev))

    results = {}
    for name, steps in test_sets.items():
        agent_now, switch_prev, agent_prev = decode(steps)
        # The relation head. No parameters, no fitting, no event labels.
        p_moved = 1.0 - (agent_now * agent_prev).sum(axis=1)
        p_on_switch = (agent_now * switch_prev).sum(axis=1)
        p_crossed = p_moved * p_on_switch
        predicted = (p_crossed > 0.5).astype(int)
        truth = np.array([s.crossed_now for s in steps])
        exact = float((agent_now.argmax(axis=1) == position_index(steps)).mean())
        switch_actual = np.zeros_like(switch_prev, dtype=bool)
        for i, step in enumerate(steps):
            for cell in step.switch_cells:
                switch_actual[i, cell[0] * GRID + cell[1]] = True
        groups = np.array([s.layout for s in steps])
        majority = float(max(np.bincount(truth, minlength=2)) / len(truth))
        low, high = bootstrap_difference((predicted == truth).astype(float), groups, majority)
        results[name] = {
            "readout": "object_relation_decoder_frozen_head",
            "parameters": parameters,
            "trained_on_event_labels": False,
            "agent_exact_cell_accuracy": exact,
            "switch_mask_f1": mask_f1(switch_prev > 0.5, switch_actual),
            "switch_mask_iou": mask_iou(switch_prev > 0.5, switch_actual),
            "switch_read_from": "previous frame (the agent occludes the switch it stands on)",
            "event_accuracy": float((predicted == truth).mean()),
            "event_balanced_accuracy": balanced_accuracy(truth, predicted),
            "event_f1": f1(truth, predicted),
            "event_brier": brier(truth, p_crossed),
            "majority_baseline": majority,
            "ci_low_vs_majority": low, "ci_high_vs_majority": high,
            "positive_rate": float(truth.mean()),
        }
    return results


# ---- scoring -------------------------------------------------------------------------------


def balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for label in (0, 1):
        mask = truth == label
        if mask.sum():
            scores.append(float((prediction[mask] == label).mean()))
    return float(np.mean(scores)) if scores else float("nan")


def f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    tp = float(((prediction == 1) & (truth == 1)).sum())
    fp = float(((prediction == 1) & (truth == 0)).sum())
    fn = float(((prediction == 0) & (truth == 1)).sum())
    return 0.0 if not tp else 2 * tp / (2 * tp + fp + fn)


def brier(truth: np.ndarray, probability: np.ndarray) -> float:
    return float(((probability - truth) ** 2).mean())


def mask_f1(predicted: np.ndarray, actual: np.ndarray) -> float:
    tp = float((predicted & actual).sum())
    fp = float((predicted & ~actual).sum())
    fn = float((~predicted & actual).sum())
    return 0.0 if not tp else 2 * tp / (2 * tp + fp + fn)


def mask_iou(predicted: np.ndarray, actual: np.ndarray) -> float:
    union = float((predicted | actual).sum())
    return 0.0 if not union else float((predicted & actual).sum()) / union


# ---- driver ---------------------------------------------------------------------------------


def stack_frames(steps: Sequence[Step], two_frame: bool) -> np.ndarray:
    current = np.stack([s.frame.astype(np.float32) / 255.0 for s in steps])
    if not two_frame:
        return current
    previous = np.stack([s.previous_frame.astype(np.float32) / 255.0 for s in steps])
    return np.concatenate([current, previous], axis=-1)


def position_index(steps: Sequence[Step]) -> np.ndarray:
    return np.array([s.position[0] * GRID + s.position[1] for s in steps])


def evaluate_handcoded(steps: Sequence[Step], appearance: int) -> dict[str, Any]:
    decode = handcoded_decoder(appearance)
    exact = agent_f1 = agent_iou = switch_f1 = switch_iou = 0.0
    for s in steps:
        got = decode(s.frame)
        exact += float(got["position"] == s.position)
        actual_agent = np.zeros((GRID, GRID), dtype=bool)
        actual_agent[s.position] = True
        actual_switch = np.zeros((GRID, GRID), dtype=bool)
        for cell in s.switch_cells:
            actual_switch[cell] = True
        agent_f1 += mask_f1(got["agent_mask"], actual_agent)
        agent_iou += mask_iou(got["agent_mask"], actual_agent)
        switch_f1 += mask_f1(got["switch_mask"], actual_switch)
        switch_iou += mask_iou(got["switch_mask"], actual_switch)
    n = len(steps)
    return {
        "readout": "handcoded_renderer_aware_oracle",
        "exact_cell_accuracy": exact / n,
        "agent_mask_f1": agent_f1 / n,
        "agent_mask_iou": agent_iou / n,
        "switch_mask_f1": switch_f1 / n,
        "switch_mask_iou": switch_iou / n,
        "observations": n,
    }


def bootstrap_difference(values: np.ndarray, groups: np.ndarray, baseline: float,
                         resamples: int = 2000, seed: int = 99) -> tuple[float, float]:
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for r in range(resamples):
        picked = generator.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[g] for g in picked])
        draws[r] = values[rows].mean() - baseline
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-layouts", type=int, default=90)
    parser.add_argument("--test-layouts", type=int, default=45)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/readout-qualification.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    appearance = CANONICAL_APPEARANCE_SEED

    train_layouts = list(range(61_000, 61_000 + arguments.train_layouts))
    held_layouts = list(range(81_000, 81_000 + arguments.test_layouts))

    print("collecting (appearance held at the canonical seed)", flush=True)
    # Split A: information presence -- same layouts, disjoint trajectories.
    a_train = collect(train_layouts, arguments.trajectories, arguments.steps, appearance, 11)
    a_test = collect(train_layouts, 2, arguments.steps, appearance, 999)
    # Split B: systematic generalisation -- held-out layouts, same fixed appearance.
    b_test = collect(held_layouts, 2, arguments.steps, appearance, 777)
    # Appearance shift, reported separately rather than folded in.
    shift_test = collect(held_layouts[:20], 2, arguments.steps, appearance + 1, 555)
    print(f"  A train {len(a_train)}  A test {len(a_test)}  B test {len(b_test)}  "
          f"shift {len(shift_test)}", flush=True)

    report: dict[str, Any] = {
        "parameter_cap": PARAMETER_CAP,
        "appearance_seed": appearance,
        "counts": {"a_train": len(a_train), "a_test": len(a_test),
                   "b_test": len(b_test), "shift_test": len(shift_test)},
        "handcoded": {}, "learned": {}, "events": {}, "controls": {},
    }

    print("J3a: hand-coded renderer-aware oracle", flush=True)
    for name, subset in (("A_information", a_test), ("B_generalisation", b_test),
                         ("appearance_shift", shift_test)):
        result = evaluate_handcoded(subset, appearance)
        report["handcoded"][name] = result
        print(f"  {name:18s} exact-cell {result['exact_cell_accuracy']:.4f}  "
              f"agent F1 {result['agent_mask_f1']:.4f}  switch F1 {result['switch_mask_f1']:.4f}",
              flush=True)

    print("J3b: learned translation-equivariant position readout", flush=True)
    x_train, y_train = stack_frames(a_train, False), position_index(a_train)
    for name, subset in (("A_information", a_test), ("B_generalisation", b_test),
                         ("appearance_shift", shift_test)):
        x_test, y_test = stack_frames(subset, False), position_index(subset)
        predicted, train_pred, parameters = train_heatmap(x_train, y_train, x_test, y_test, 3)
        exact = float((predicted == y_test).mean())
        train_exact = float((train_pred == y_train).mean())
        groups = np.array([s.layout for s in subset])
        low, high = bootstrap_difference((predicted == y_test).astype(float), groups, 1.0 / (GRID * GRID))
        report["learned"][f"position::{name}"] = {
            "readout": "translation_equivariant_cnn_heatmap",
            "parameters": parameters,
            "exact_cell_accuracy": exact,
            "train_exact_cell_accuracy": train_exact,
            "chance": 1.0 / (GRID * GRID),
            "ci_low": low, "ci_high": high,
        }
        print(f"  {name:18s} exact-cell {exact:.4f} (train {train_exact:.4f})  "
              f"chance {1/(GRID*GRID):.4f}  CI[{low:+.3f},{high:+.3f}]  params {parameters}",
              flush=True)

    print("J4: two-frame switch-crossing event classifier", flush=True)
    e_train_x = stack_frames(a_train, True)
    e_train_y = np.array([s.crossed_now for s in a_train])
    for name, subset in (("A_information", a_test), ("B_generalisation", b_test)):
        e_test_x = stack_frames(subset, True)
        e_test_y = np.array([s.crossed_now for s in subset])
        predicted, probability, parameters = train_binary_cnn(e_train_x, e_train_y, e_test_x, 6)
        groups = np.array([s.layout for s in subset])
        low, high = bootstrap_difference(
            (predicted == e_test_y).astype(float), groups,
            float(max(np.bincount(e_train_y, minlength=2)) / len(e_train_y)))
        record = {
            "readout": "two_frame_event_cnn",
            "parameters": parameters,
            "accuracy": float((predicted == e_test_y).mean()),
            "balanced_accuracy": balanced_accuracy(e_test_y, predicted),
            "f1": f1(e_test_y, predicted),
            "brier": brier(e_test_y, probability),
            "positive_rate": float(e_test_y.mean()),
            "majority_baseline": float(max(np.bincount(e_test_y, minlength=2)) / len(e_test_y)),
            "ci_low_vs_majority": low, "ci_high_vs_majority": high,
        }
        report["events"][name] = record
        print(f"  {name:18s} bal-acc {record['balanced_accuracy']:.4f}  F1 {record['f1']:.4f}  "
              f"Brier {record['brier']:.4f}  acc {record['accuracy']:.4f} vs majority "
              f"{record['majority_baseline']:.4f}  CI[{low:+.3f},{high:+.3f}]", flush=True)

    print("J4b/J6: object-relation decoder with a frozen relation head", flush=True)
    relation = train_relation_decoder(
        a_train, {"A_information": a_test, "B_generalisation": b_test})
    report["events"]["relation_decoder"] = relation
    for name, record in relation.items():
        print(f"  {name:18s} agent-exact {record['agent_exact_cell_accuracy']:.4f}  "
              f"switch F1 {record['switch_mask_f1']:.4f}  event bal-acc "
              f"{record['event_balanced_accuracy']:.4f}  F1 {record['event_f1']:.4f}  "
              f"Brier {record['event_brier']:.4f}  vs majority "
              f"{record['majority_baseline']:.4f}  CI[{record['ci_low_vs_majority']:+.3f},"
              f"{record['ci_high_vs_majority']:+.3f}]", flush=True)

    print("controls", flush=True)
    shuffled_y = np.random.default_rng(5).permutation(e_train_y)
    predicted, probability, _ = train_binary_cnn(
        e_train_x, shuffled_y, stack_frames(b_test, True), 6)
    b_y = np.array([s.crossed_now for s in b_test])
    report["controls"]["shuffled_event_labels"] = {
        "balanced_accuracy": balanced_accuracy(b_y, predicted),
        "f1": f1(b_y, predicted),
        "note": "must sit at chance; a positive score means the readout is reading something else",
    }
    constant = np.zeros_like(b_y)
    report["controls"]["constant_predictor"] = {
        "balanced_accuracy": balanced_accuracy(b_y, constant), "f1": f1(b_y, constant)}
    random_pred = np.random.default_rng(6).integers(0, 2, size=len(b_y))
    report["controls"]["random_predictor"] = {
        "balanced_accuracy": balanced_accuracy(b_y, random_pred), "f1": f1(b_y, random_pred)}
    for key, value in report["controls"].items():
        print(f"  {key:26s} bal-acc {value['balanced_accuracy']:.4f}  F1 {value['f1']:.4f}",
              flush=True)

    report["wall_clock_seconds"] = time.perf_counter() - started
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
