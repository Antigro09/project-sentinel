"""K / N12. Vision, language and action as separate channels, with a target that needs all.

The alias-pair outcome metric used everywhere else is displacement, and displacement is
goal-independent -- so it can never show language contributing anything, and reporting it
as a multimodal result would be vacuous. The target here is instead

    did the action just taken move the agent TOWARD the language-named goal marker?

A first version used "did the action land ON the goal marker". Under a random policy over
nine steps that fires on 0.67% of rows and leaves only 2 of 738 frame-action keys
contested, so every arm predicted all-negative and scored exactly 0.5000 balanced
accuracy. That is a degenerate target, not a finding about language, and reporting it as
one would have been the third withdrawn headline in this project's history. The
toward-the-goal target is dense and carries the same three-way dependence.

which needs three different things at once: the frame (where the agent and both markers
are), the language goal (WHICH marker counts -- the frame renders both and marks neither),
and the history (polarity negates the action delta, so where the action lands depends on
the hidden phase).

Section K's case is instantiable because the goal is a SEPARATE environment dynamic from
layout, appearance and phase: the same rendered world can carry either goal, so the same
frame has two different correct answers.

    .venv-shwm/bin/python experiments/shwm/n_multimodal.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import n_core as core
import n_heads as heads
import n_interfaces as ifaces
from m2d_core import ARTIFACTS, write
from n_core import GRID

SEEDS = (34_000, 34_001, 34_002)
ARMS = ("vision_language_history", "vision_shuffled_language_history",
        "vision_masked_language_history", "vision_language_no_history",
        "vision_language_shuffled_action_history")


def goal_conditioned(layouts, trajectories, steps, seed):
    """Both goal draws over the SAME rendered worlds, with a goal-dependent target."""
    from sentinel.env.adapters.procedural_visual_v2 import (
        ProceduralVisualV2Adapter, MARKERS, ACTIONS)
    from sentinel.wm.authority import AuthorityGate

    gate = AuthorityGate(gate_id="n-mm")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for layout in layouts:
        # Draws 0 and 1 both hash to the alpha marker, so the first version varied
        # nothing. 0 and 2 give different goals over a byte-identical frame, which is
        # the case section K actually asks for.
        for goal_draw in (0, 2):
            for _ in range(trajectories):
                # ONE dynamic per reset: `reset` defaults every other seed to `seed`,
                # so a second reset(appearance:...) silently reset the goal to its
                # default and both "goal draws" produced the same goal. Using the goal
                # dynamic alone leaves layout, appearance and phase all derived from the
                # layout seed, so the two draws are the SAME rendered world.
                adapter.reset(layout, f"goal:{goal_draw}")
                level = adapter._require()
                marker = adapter._goal_marker
                goal_cell = tuple(int(v) for v in level.markers[marker])
                for _step in range(steps):
                    truth = adapter.snapshot().reveal("evaluator")
                    frame = adapter.frame().copy()
                    position = tuple(int(v) for v in truth["position"])
                    phase = int(truth["polarity"])
                    action = int(rng.integers(0, len(ACTIONS)))
                    adapter.step(action, gate.authorize_evaluator(action, "n-mm"))
                    after = adapter.snapshot().reveal("evaluator")
                    landed = tuple(int(v) for v in after["position"])
                    before_distance = (abs(goal_cell[0] - position[0])
                                       + abs(goal_cell[1] - position[1]))
                    after_distance = (abs(goal_cell[0] - landed[0])
                                      + abs(goal_cell[1] - landed[1]))
                    rows.append({
                        "frame": frame, "action": action,
                        "goal_index": MARKERS.index(marker),
                        "phase": phase, "layout": layout,
                        "target": float(after_distance < before_distance)})
                    if after.get("reached"):
                        break
    return rows


def coordinate_head(slots_dim: int, seed: int, channels: int = 12):
    """Soft-argmax coordinates, then a small MLP.

    "Moved toward the goal" compares the agent's cell with a marker's cell, which may be
    eleven cells away. A 1x1/3x3 convolutional head cannot express a long-range distance
    comparison at all, and a first version of this section scored 0.5378 with correct
    language and 0.5547 with SHUFFLED language -- a readout limit reported as a language
    result would have been exactly the wrong conclusion. The auxiliary heads already show
    the masks are perfectly recoverable, so turning slots into coordinates and comparing
    coordinates is the honest bridge.
    """
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(seed)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.detect = nn.Conv2d(slots_dim, channels, 1)
            self.a = nn.Linear(2 * channels + 4 + 3, 128)
            self.b = nn.Linear(128, 1)

        def __call__(self, slots, action, side):
            maps = mx.softmax(self.detect(slots).reshape(
                slots.shape[0], side * side, channels), axis=1)
            grid = mx.arange(side, dtype=mx.float32)
            rows = mx.repeat(grid, side).reshape(1, side * side, 1)
            columns = mx.tile(grid, [side]).reshape(1, side * side, 1)
            y = mx.sum(maps * rows, axis=1) / side
            x = mx.sum(maps * columns, axis=1) / side
            return self.b(nn.relu(self.a(mx.concatenate([y, x, action], axis=-1))))

    model = Head()
    mx.eval(model.parameters())
    return model


def train_coordinate(slots, action, extra, y, seed, side, updates=2000):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    model = coordinate_head(slots.shape[-1], seed)
    optimizer = optim.AdamW(learning_rate=2e-3)
    rng = np.random.default_rng(seed)
    combined = np.concatenate([action, extra], axis=1).astype(np.float32)
    xs, ac, ys = mx.array(slots), mx.array(combined), mx.array(y)
    for _ in range(updates):
        pick = mx.array(rng.integers(0, len(slots), min(128, len(slots))))

        def objective(m):
            return mx.mean(nn.losses.binary_cross_entropy(
                m(xs[pick], ac[pick], side)[:, 0], ys[pick], with_logits=True))

        value, grads = nn.value_and_grad(model, objective)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, value)

    def infer(s, a, e):
        out = model(mx.array(s), mx.array(np.concatenate([a, e], axis=1)
                                          .astype(np.float32)), side)
        mx.eval(out)
        return np.asarray(out)
    return infer


def featurise(rows, arm: str, seed: int):
    """Slots from the frame; goal and history as separate broadcast channels."""
    rng = np.random.default_rng(seed)
    frames = np.stack([r["frame"] for r in rows]).astype(np.float32) / 255.0
    cells = frames[:, ::core.CELL, ::core.CELL, :]
    action = np.zeros((len(rows), 4), dtype=np.float32)
    for i, r in enumerate(rows):
        action[i, r["action"]] = 1.0
    goal = np.array([[1.0 - r["goal_index"], float(r["goal_index"])] for r in rows],
                    dtype=np.float32)
    history = np.array([[float(r["phase"])] for r in rows], dtype=np.float32)

    if arm == "vision_shuffled_language_history":
        goal = goal[rng.permutation(len(goal))]
    elif arm == "vision_masked_language_history":
        goal = np.zeros_like(goal)
    elif arm == "vision_language_no_history":
        history = np.zeros_like(history)
    elif arm == "vision_language_shuffled_action_history":
        history = history[rng.permutation(len(history))]

    extra = np.concatenate([goal, history], axis=1)          # (N, 3)
    projection = ifaces.frozen_projection(cells.shape[-1], ifaces.SLOT_WIDTH, 20_010)
    # Vision, language and action stay SEPARATE channels: the goal and the history are
    # not broadcast into the image, they enter alongside the coordinates the image
    # yields. Mixing them into the pixels would make the ablations uninterpretable.
    return cells @ projection, action, extra


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "n-multimodal.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    train = goal_conditioned(core.TRAIN_LAYOUTS, 2, 9, 11)
    held = goal_conditioned(core.HELD_OUT_LAYOUTS, 2, 9, 313)
    print(f"{len(train)} train rows / {len(held)} held-out rows; "
          f"toward-the-goal rate {np.mean([r['target'] for r in held]):.4f}", flush=True)

    # Does the same frame ever carry two different correct answers? Measure it.
    from collections import defaultdict
    by_frame = defaultdict(set)
    for r in held:
        key = (r["frame"].tobytes(), r["action"])
        by_frame[key].add(r["target"])
    contested = sum(1 for v in by_frame.values() if len(v) > 1)
    print(f"identical (frame, action) with DIFFERENT correct outcomes: "
          f"{contested} of {len(by_frame)}\n", flush=True)

    report: dict[str, Any] = {
        "train_rows": len(train), "held_out_rows": len(held),
        "contested_frame_action_pairs": contested,
        "distinct_frame_action_keys": len(by_frame),
        "target": ("the action moved the agent toward the language-named goal marker; "
                   "needs the frame for positions, the language for WHICH marker, and "
                   "the history because polarity negates the action delta"),
        "audio": "declared absent, as in every prior phase",
        "arms": {}}

    y_train = np.array([r["target"] for r in train], dtype=np.float32)
    y_held = np.array([r["target"] for r in held], dtype=np.float32)
    print(f"{'arm':46s} {'held-out balanced':>18s} {'F1':>8s}")
    print("-" * 76)
    for arm in ARMS:
        scores = []
        for seed in SEEDS:
            xt, at, et = featurise(train, arm, seed)
            xh, ah, eh = featurise(held, arm, seed)
            infer = train_coordinate(xt, at, et, y_train, seed, GRID)
            scores.append(heads.binary_metrics(infer(xh, ah, eh), y_held))
        report["arms"][arm] = {m: float(np.mean([s[m] for s in scores]))
                               for m in scores[0]}
        print(f"{arm:46s} {report['arms'][arm]['balanced_accuracy']:18.4f} "
              f"{report['arms'][arm]['f1']:8.4f}", flush=True)

    full = report["arms"]["vision_language_history"]["balanced_accuracy"]
    report["language_contributes"] = bool(
        full - report["arms"]["vision_shuffled_language_history"]["balanced_accuracy"] > 0.02
        and full - report["arms"]["vision_masked_language_history"]["balanced_accuracy"] > 0.02)
    report["history_contributes"] = bool(
        full - report["arms"]["vision_language_no_history"]["balanced_accuracy"] > 0.02
        and full - report["arms"]["vision_language_shuffled_action_history"]
        ["balanced_accuracy"] > 0.02)
    report["n12_both_nonvacuous"] = bool(report["language_contributes"]
                                         and report["history_contributes"])
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nlanguage contributes: {report['language_contributes']}")
    print(f"history contributes:  {report['history_contributes']}")
    print(f"N12: {report['n12_both_nonvacuous']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
