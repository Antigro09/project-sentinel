"""H / O12 / P11. The outcome-trained initial-state gauge. Owed since phase M2F.

Every earlier "learned gauge" was supervised on a number equal to the initial polarity,
which makes it a stripe reader, not a gauge learned from behaviour. This arm receives no
phase target and no phase input. It sees the reset frame, emits an initial belief, and is
trained ONLY through the likelihood of later displacement outcomes -- the belief has to
pay for itself downstream or it learns nothing.

The comparison that matters is against the authored stripe map. If the outcome-trained
gauge matches it, initial-state grounding stops being authored. If it does not, the
authored gauge is retained and the dependence is stated, which is what M2F and O both
had to do.

    .venv-shwm/bin/python experiments/shwm/p_gauge.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as m2d
import n_core as ncore
from m2d_core import ARTIFACTS, write
from n_core import GRID

SEEDS = (37_000, 37_001, 37_002)
VARIANTS = ("1_authored_public_stripe", "2_stripe_supervised", "3_phase_supervised",
            "4_outcome_trained", "5_stripe_masked", "6_reset_omitted",
            "7_shuffled_reset_frame", "8_false_stripe")


def transform(name: str, frames: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.array(frames, copy=True)
    if name == "5_stripe_masked":
        out[:, 0, :, :] = out[:, 1, :, :]
    elif name == "6_reset_omitted":
        out[:] = 0.0
    elif name == "7_shuffled_reset_frame":
        out = out[rng.permutation(len(out))]
    elif name == "8_false_stripe":
        out[:, 0, :, :] = 1.0 - out[:, 0, :, :]
    return out


def build_outcome_gauge(seed: int, width: int):
    """Reset frame -> initial belief -> displacement head. One graph, one loss.

    The belief is a two-state distribution and the ONLY path by which the reset frame can
    influence the loss, so any accuracy it acquires came from outcome likelihood.
    """
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(seed)

    class Gauge(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 16, 3, padding=1)
            self.c2 = nn.Conv2d(16, 16, 1)
            self.belief = nn.Linear(32, 2)
            self.head = nn.Sequential(nn.Linear(width + 2, 128), nn.ReLU(),
                                      nn.Linear(128, 5))

        def initial_belief(self, reset_frame):
            h = nn.relu(self.c1(reset_frame))
            h = nn.relu(self.c2(h))
            flat = h.reshape(h.shape[0], -1, h.shape[-1])
            pooled = mx.concatenate([mx.max(flat, axis=1), mx.mean(flat, axis=1)],
                                    axis=-1)
            return mx.softmax(self.belief(pooled), axis=-1)

        def __call__(self, reset_frame, features, parity):
            belief = self.initial_belief(reset_frame)
            # Propagate the initial belief by the PUBLIC event parity of the route: the
            # phase at step t is the initial state XOR the parity so far.
            flipped = mx.stack([belief[:, 1], belief[:, 0]], axis=-1)
            state = mx.where(parity.reshape(-1, 1) > 0.5, flipped, belief)
            return self.head(mx.concatenate([features, state], axis=-1)), belief

    model = Gauge()
    mx.eval(model.parameters())
    return model


def collect_gauge_data(layouts, trajectories, steps, seed):
    """Reset frames, per-step public features, route event parity, displacement target."""
    from structured_calibration import collect, encode
    from belief_factorization import public_event
    from sentinel.env.adapters.procedural_visual_v2 import ProceduralVisualV2Adapter
    from sentinel.wm.authority import AuthorityGate
    from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

    gate = AuthorityGate(gate_id="p-gauge")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    rng = np.random.default_rng(seed)
    resets, features, parity, target, truth = [], [], [], [], []
    for layout in layouts:
        for _ in range(trajectories):
            adapter.reset(layout, f"appearance:{CANONICAL_APPEARANCE_SEED}")
            level = adapter._require()
            switches = {tuple(int(v) for v in c) for c in level.switches}
            walls = np.asarray(level.walls, dtype=bool)
            goal = tuple(int(v) for v in level.markers[adapter._goal_marker])
            reset_frame = adapter.frame().astype(np.float32) / 255.0
            initial = int(level.initial_polarity)
            rows, crossings, previous_action = [], 0, -1
            for _step in range(steps):
                info = adapter.snapshot().reveal("evaluator")
                position = tuple(int(v) for v in info["position"])
                row = {"position": position, "switches": switches, "walls": walls,
                       "goal": goal, "previous_action": previous_action,
                       "blocked": float(info["last_blocked"])}
                if rows:
                    crossings += public_event(rows[-1], row)
                rows.append(row)
                action = int(rng.integers(0, 4))
                snapshot = adapter.snapshot()
                adapter.step(action, gate.authorize_evaluator(action, "g"))
                landed = tuple(int(v) for v in
                               adapter.snapshot().reveal("evaluator")["position"])
                adapter.restore(snapshot)
                from structured_calibration import DISPLACEMENTS
                delta = (landed[0] - position[0], landed[1] - position[1])
                resets.append(reset_frame)
                features.append(encode(row, action, with_phase=False))
                parity.append(float(crossings % 2))
                target.append(DISPLACEMENTS.index(delta) if delta in DISPLACEMENTS else 4)
                truth.append(float(initial))
                previous_action = action
                if adapter.step(action, gate.authorize_evaluator(action, "g")).terminated:
                    break
    return (np.stack(resets), np.stack(features).astype(np.float32),
            np.array(parity, np.float32), np.array(target, np.int32),
            np.array(truth, np.float32))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "p-gauge.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    train = collect_gauge_data(range(110_000, 110_040), 2, 9, 11)
    test = collect_gauge_data(range(111_000, 111_020), 2, 9, 313)
    print(f"{len(train[0])} train rows / {len(test[0])} test rows; "
          f"initial-polarity rate {test[4].mean():.3f}\n", flush=True)

    report: dict[str, Any] = {"variants": {}, "seeds": list(SEEDS[:arguments.seeds])}
    print(f"{'variant':30s} {'belief acc (up to perm)':>24s} {'displacement':>13s}")
    print("-" * 72)
    for variant in VARIANTS:
        if variant == "1_authored_public_stripe":
            read = (test[0][:, 0, :, :].mean(axis=(1, 2))
                    > test[0][:, 1, :, :].mean(axis=(1, 2))).astype(float)
            report["variants"][variant] = {
                "belief_accuracy_up_to_permutation": float((read == test[4]).mean()),
                "authored": True}
            print(f"{variant:30s} "
                  f"{report['variants'][variant]['belief_accuracy_up_to_permutation']:24.4f}"
                  f" {'-':>13s}")
            continue
        accuracies, displacements = [], []
        for seed in SEEDS[:arguments.seeds]:
            resets = transform(variant, train[0], seed)
            model = build_outcome_gauge(seed, train[1].shape[1])
            optimizer = optim.AdamW(learning_rate=2e-3)
            rng = np.random.default_rng(seed)
            tensors = [mx.array(resets), mx.array(train[1]), mx.array(train[2]),
                       mx.array(train[3]), mx.array(train[4])]
            for _ in range(2000):
                pick = mx.array(rng.integers(0, len(train[3]), 128))
                r, f, p, y, phase = [t[pick] for t in tensors]

                def objective(m):
                    logits, belief = m(r, f, p)
                    loss = nn.losses.cross_entropy(logits, y, reduction="mean")
                    if variant == "2_stripe_supervised":
                        loss = loss + nn.losses.binary_cross_entropy(
                            belief[:, 1], phase, with_logits=False).mean()
                    if variant == "3_phase_supervised":
                        loss = loss + 5.0 * nn.losses.binary_cross_entropy(
                            belief[:, 1], phase, with_logits=False).mean()
                    return loss

                value, grads = nn.value_and_grad(model, objective)(model)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, value)

            logits, belief = model(mx.array(transform(variant, test[0], seed)),
                                   mx.array(test[1]), mx.array(test[2]))
            mx.eval(logits, belief)
            predicted = np.asarray(belief).argmax(axis=1)
            direct = float((predicted == test[4]).mean())
            accuracies.append(max(direct, 1.0 - direct))    # up to state permutation
            displacements.append(float((np.asarray(logits).argmax(axis=1)
                                        == test[3]).mean()))
        report["variants"][variant] = {
            "belief_accuracy_up_to_permutation": float(np.mean(accuracies)),
            "belief_sd": float(np.std(accuracies)),
            "displacement_accuracy": float(np.mean(displacements)),
            "per_seed": accuracies}
        print(f"{variant:30s} {np.mean(accuracies):24.4f} "
              f"{np.mean(displacements):13.4f}", flush=True)

    authored = report["variants"]["1_authored_public_stripe"][
        "belief_accuracy_up_to_permutation"]
    outcome = report["variants"]["4_outcome_trained"]["belief_accuracy_up_to_permutation"]
    report["paired_difference_outcome_minus_authored"] = float(outcome - authored)
    report["o12_outcome_trained_matches_authored"] = bool(outcome >= authored - 0.02)
    report["o12_status"] = "PASS" if report["o12_outcome_trained_matches_authored"] else "FAIL"
    report["conditional_on_authored_grounding"] = bool(
        not report["o12_outcome_trained_matches_authored"])
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nauthored {authored:.4f} vs outcome-trained {outcome:.4f}  "
          f"(difference {outcome - authored:+.4f})")
    print(f"O12: {report['o12_status']}")
    print(f"still conditional on authored initial-state grounding: "
          f"{report['conditional_on_authored_grounding']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
