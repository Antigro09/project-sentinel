"""C / T2. The planted sequence-defect suite, and the guards that must catch them.

This is the oldest outstanding gap in the phase-2 work. I named it as the first
thing to close at the end of the M phase, did not close it, produced another
headline, and had that headline corrected. It is built here before anything else.

The discipline the specification asks for is precise and worth stating: a guard
that passes on both the honest implementation and its mutation is vacuous. So the
deliverable is not ten guards -- it is a matrix showing each guard passing the
honest pipeline and failing its own planted defect. A guard that catches nothing,
or that catches everything, is reported as broken.

Every corrupted arm is derived from the same frozen base examples. Regenerating
with a changed corruption mode shifts the RNG call path, which is how an earlier
control ended up scored against targets from different sequences.

    .venv-shwm/bin/python experiments/shwm/planted_defects.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402
from structured_calibration import CLASSES, collect  # noqa: E402
from belief_factorization import build_dataset  # noqa: E402

RESET_FLAG_COLUMN = -1      # the "this is step 0" indicator appended by sequence_features
RESET_VALUE_COLUMN = -2     # the rendered reset stripe


@dataclass
class Batch:
    """A frozen, inspectable training batch. Every arm is derived from one of these."""
    x: np.ndarray            # (episodes, steps, features)
    y: np.ndarray            # (episodes, steps) displacement class
    mask: np.ndarray         # (episodes, steps) 1 for real steps
    lengths: np.ndarray      # (episodes,) true lengths
    episode_id: np.ndarray   # (episodes,)

    def copy(self) -> "Batch":
        """Carries the planted flags forward.

        An earlier version dropped them, so a guard that perturbed a batch and re-ran
        the model silently compared the defective code path against the honest one and
        always saw a difference. That made `state_carries_within_episode` pass every
        arm -- a vacuous guard, caught by this matrix before it was trusted.
        """
        out = Batch(self.x.copy(), self.y.copy(), self.mask.copy(),
                    self.lengths.copy(), self.episode_id.copy())
        for key, value in self.__dict__.items():
            if key.startswith("_flag_"):
                setattr(out, key, value)
        return out


def base_batch(items) -> Batch:
    length = max(len(i["y"]) for i in items)
    width = items[0]["x"].shape[1]
    x = np.zeros((len(items), length, width), dtype=np.float32)
    y = np.zeros((len(items), length), dtype=np.int64)
    mask = np.zeros((len(items), length), dtype=np.float32)
    lengths = np.zeros(len(items), dtype=np.int64)
    for i, item in enumerate(items):
        n = len(item["y"])
        x[i, :n] = item["x"]
        y[i, :n] = item["y"]
        mask[i, :n] = 1.0
        lengths[i] = n
    return Batch(x, y, mask, lengths, np.arange(len(items)))


# ---- the ten planted defects, each derived from the same frozen base ----------------------


def defect_reset_state_every_step(batch: Batch) -> Batch:
    """Marked on the batch; the runner disables state carry for this arm."""
    out = batch.copy()
    out.__dict__["_flag_reset_every_step"] = True
    return out


def defect_detach_state(batch: Batch) -> Batch:
    out = batch.copy()
    out.__dict__["_flag_detach"] = True
    return out


def defect_cross_episode_state(batch: Batch) -> Batch:
    out = batch.copy()
    out.__dict__["_flag_cross_episode"] = True
    return out


def defect_reset_frame_omitted(batch: Batch) -> Batch:
    out = batch.copy()
    out.x[:, 0, RESET_FLAG_COLUMN] = 0.0
    out.x[:, 0, RESET_VALUE_COLUMN] = 0.0
    return out


def defect_reset_frame_duplicated(batch: Batch) -> Batch:
    out = batch.copy()
    for i, n in enumerate(out.lengths):
        if n > 1:
            out.x[i, 1, RESET_FLAG_COLUMN] = out.x[i, 0, RESET_FLAG_COLUMN]
            out.x[i, 1, RESET_VALUE_COLUMN] = out.x[i, 0, RESET_VALUE_COLUMN]
    return out


def defect_final_index_shifted(batch: Batch) -> Batch:
    """Targets rolled by one: the prediction at t is scored against t+1's transition."""
    out = batch.copy()
    for i, n in enumerate(out.lengths):
        out.y[i, :n] = np.roll(out.y[i, :n], -1)
    return out


def defect_padding_as_history(batch: Batch) -> Batch:
    out = batch.copy()
    out.mask[:] = 1.0        # every padded step now counts as real
    return out


def defect_batch_permuted(batch: Batch) -> Batch:
    """Rows permuted without permuting targets: histories no longer match their labels."""
    out = batch.copy()
    order = np.random.default_rng(4242).permutation(len(out.x))
    out.x = out.x[order]
    return out


def defect_current_step_removed(batch: Batch) -> Batch:
    """History intact, current row zeroed: the step being predicted is unavailable."""
    out = batch.copy()
    for i, n in enumerate(out.lengths):
        out.x[i, n - 1] = 0.0
    return out


def defect_target_in_padding(batch: Batch) -> Batch:
    """The target written into a feature column -- the classic silent leak."""
    out = batch.copy()
    for i, n in enumerate(out.lengths):
        out.x[i, :n, 0] = out.y[i, :n].astype(np.float32)
    return out


DEFECTS: dict[str, Callable[[Batch], Batch]] = {
    "1_reset_state_every_step": defect_reset_state_every_step,
    "2_detached_state": defect_detach_state,
    "3_cross_episode_state": defect_cross_episode_state,
    "4_reset_frame_omitted": defect_reset_frame_omitted,
    "5_reset_frame_duplicated": defect_reset_frame_duplicated,
    "6_final_index_shifted": defect_final_index_shifted,
    "7_padding_as_history": defect_padding_as_history,
    "8_batch_permuted": defect_batch_permuted,
    "9_current_step_removed": defect_current_step_removed,
    "10_target_in_features": defect_target_in_padding,
}


# ---- guards -------------------------------------------------------------------------------
# Each returns True when the pipeline looks correct. The matrix below requires every guard to
# pass the honest batch and fail its own defect; anything else is reported as a broken guard.


def guard_reset_present_exactly_once(batch: Batch, model=None) -> bool:
    """Exactly one step per episode carries the reset flag, and it is step 0."""
    for i, n in enumerate(batch.lengths):
        flags = batch.x[i, :n, RESET_FLAG_COLUMN]
        if flags.sum() != 1.0 or flags[0] != 1.0:
            return False
    return True


def guard_blocked_action_implies_no_movement(batch: Batch, model=None) -> bool:
    """Internal consistency, needing no reference copy.

    The encoder puts the four neighbour-blocked bits first and the queried action's
    one-hot later. If the queried action's neighbour is blocked in BOTH directions,
    the displacement must be the no-move class whatever the phase. A target rolled by
    one step, or a row permuted away from its label, breaks this.
    """
    # Column layout from `encode`: neighbour_blocked(4) | neighbour_switch(4) |
    # goal_direction(4) | previous_action(5) | query_action(4) | blocked(1) | position(2)
    # An earlier version used 12:16 for the query one-hot, which is goal_direction and
    # previous_action -- the guard then fired on the honest pipeline, which the matrix
    # reported as "guard fires on honest" rather than as a defect being caught.
    blocked = batch.x[:, :, 0:4]
    query = batch.x[:, :, 17:21]
    for i, n in enumerate(batch.lengths):
        for t in range(n):
            action = int(np.argmax(query[i, t]))
            opposite = (action + 2) % 4
            if blocked[i, t, action] > 0.5 and blocked[i, t, opposite] > 0.5:
                if int(batch.y[i, t]) != CLASSES - 1:
                    return False
    return True


def guard_mask_matches_lengths(batch: Batch, model=None) -> bool:
    return bool(np.all(batch.mask.sum(axis=1) == batch.lengths))


def guard_padding_is_empty(batch: Batch, model=None) -> bool:
    """Every step beyond an episode's length must be all zeros, and every real step
    must not be. Permuting rows across episodes of different lengths breaks this."""
    for i, n in enumerate(batch.lengths):
        if n < batch.x.shape[1] and np.abs(batch.x[i, n:]).sum() != 0.0:
            return False
        if np.abs(batch.x[i, n - 1]).sum() == 0.0:
            return False
    return True


def guard_current_step_present(batch: Batch, model=None) -> bool:
    """The step being predicted must carry content."""
    for i, n in enumerate(batch.lengths):
        if np.abs(batch.x[i, n - 1]).sum() == 0.0:
            return False
    return True


def guard_no_feature_equals_target(batch: Batch, model=None) -> bool:
    """No feature column may reproduce the target on the real steps."""
    for column in range(batch.x.shape[2]):
        agree = total = 0
        for i, n in enumerate(batch.lengths):
            agree += int(np.sum(np.round(batch.x[i, :n, column]) == batch.y[i, :n]))
            total += int(n)
        if total and agree / total > 0.95:
            return False
    return True


def guard_state_carries_within_episode(batch: Batch, model=None) -> bool:
    """Behavioural: the output at the last step must depend on an earlier step.

    Perturb step 0 and require the final output to move. A model whose state is reset
    or detached every step cannot see the change.
    """
    if model is None:
        return True
    a = model(batch.x, batch)
    perturbed = batch.copy()
    perturbed.x[:, 0, :12] += 1.0
    b = model(perturbed.x, perturbed)
    final = np.array([a[i, batch.lengths[i] - 1] for i in range(len(a))])
    final_b = np.array([b[i, batch.lengths[i] - 1] for i in range(len(b))])
    return bool(np.abs(final - final_b).max() > 1e-6)


def guard_gradient_reaches_first_step(batch: Batch, model=None) -> bool:
    """Behavioural: the loss at the final step must have gradient to step 0."""
    if model is None or not hasattr(model, "gradient_to_first_step"):
        return True
    return bool(model.gradient_to_first_step(batch) > 1e-12)


def guard_no_cross_episode_state(batch: Batch, model=None) -> bool:
    """Behavioural: episode i's output must not change when episode i-1 changes."""
    if model is None:
        return True
    a = model(batch.x, batch)
    perturbed = batch.copy()
    perturbed.x[0] += 1.0                      # disturb only the FIRST episode
    b = model(perturbed.x, perturbed)
    return bool(np.abs(a[1:] - b[1:]).max() < 1e-6)


GUARDS: dict[str, Callable[..., bool]] = {
    "reset_present_exactly_once": guard_reset_present_exactly_once,
    "blocked_action_implies_no_movement": guard_blocked_action_implies_no_movement,
    "mask_matches_lengths": guard_mask_matches_lengths,
    "padding_is_empty": guard_padding_is_empty,
    "current_step_present": guard_current_step_present,
    "no_feature_equals_target": guard_no_feature_equals_target,
    "state_carries_within_episode": guard_state_carries_within_episode,
    "gradient_reaches_first_step": guard_gradient_reaches_first_step,
    "no_cross_episode_state": guard_no_cross_episode_state,
}

INTENDED: dict[str, str] = {
    "1_reset_state_every_step": "state_carries_within_episode",
    "2_detached_state": "gradient_reaches_first_step",
    "3_cross_episode_state": "no_cross_episode_state",
    "4_reset_frame_omitted": "reset_present_exactly_once",
    "5_reset_frame_duplicated": "reset_present_exactly_once",
    "6_final_index_shifted": "blocked_action_implies_no_movement",
    "7_padding_as_history": "mask_matches_lengths",
    "8_batch_permuted": "padding_is_empty",
    "9_current_step_removed": "current_step_present",
    "10_target_in_features": "no_feature_equals_target",
}


# ---- a minimal recurrent model that honours the planted flags ------------------------------


class ProbeModel:
    """A small GRU used only to exercise the behavioural guards.

    It reads the flags a defect injector sets, so `reset every step`, `detached state`
    and `cross-episode carry` are real changes to the computation rather than
    annotations. Untrained: the guards test dataflow, not accuracy.
    """

    def __init__(self, width: int, seed: int = 6600) -> None:
        import mlx.core as mx
        import mlx.nn as nn

        mx.random.seed(seed)
        self.mx, self.nn = mx, nn
        self.project = nn.Linear(width, 32)
        self.gru = nn.GRU(32, 32)
        self.head = nn.Linear(32, CLASSES)
        mx.eval(self.project.parameters(), self.gru.parameters(), self.head.parameters())

    def _forward(self, x, batch):
        mx = self.mx
        z = self.nn.relu(self.project(x))
        if getattr(batch, "_flag_reset_every_step", False):
            steps = [self.gru(z[:, t:t + 1]) for t in range(z.shape[1])]
            h = mx.concatenate(steps, axis=1)
        elif getattr(batch, "_flag_cross_episode", False):
            # one long sequence across every episode: state leaks between them
            flat = z.reshape(1, -1, z.shape[2])
            h = self.gru(flat).reshape(z.shape[0], z.shape[1], -1)
        else:
            h = self.gru(z)
        if getattr(batch, "_flag_detach", False):
            h = mx.stop_gradient(h)
        return self.head(h)

    def __call__(self, x, batch):
        out = self._forward(self.mx.array(x), batch)
        self.mx.eval(out)
        return np.asarray(out)

    def gradient_to_first_step(self, batch) -> float:
        mx, nn = self.mx, self.nn

        def loss_of_input(inputs):
            logits = self._forward(inputs, batch)
            final = mx.stack([logits[i, int(batch.lengths[i]) - 1]
                              for i in range(len(batch.lengths))])
            target = mx.array(np.array([batch.y[i, int(batch.lengths[i]) - 1]
                                        for i in range(len(batch.lengths))]).astype(np.int32))
            return nn.losses.cross_entropy(final, target, reduction="mean")

        grad = mx.grad(loss_of_input)(mx.array(batch.x))
        mx.eval(grad)
        return float(np.abs(np.asarray(grad)[:, 0, :]).max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", type=int, default=25)
    parser.add_argument("--trajectories", type=int, default=2)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/planted-defects.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    items = build_dataset(collect(list(range(61_000, 61_000 + arguments.layouts)),
                                  arguments.trajectories, arguments.steps,
                                  CANONICAL_APPEARANCE_SEED, 11), 5)
    honest = base_batch(items)
    model = ProbeModel(honest.x.shape[2])
    print(f"frozen base batch: {honest.x.shape[0]} episodes x {honest.x.shape[1]} steps "
          f"x {honest.x.shape[2]} features\n", flush=True)

    arms = {"0_honest": honest}
    for name, injector in DEFECTS.items():
        arms[name] = injector(honest)

    matrix: dict[str, dict[str, bool]] = {}
    for arm_name, arm in arms.items():
        matrix[arm_name] = {}
        for guard_name, guard in GUARDS.items():
            try:
                matrix[arm_name][guard_name] = bool(guard(arm, model))
            except Exception:                                     # noqa: BLE001
                matrix[arm_name][guard_name] = False

    names = list(GUARDS)
    print(f"{'arm':28s} " + " ".join(f"{n[:11]:>12s}" for n in names))
    print("-" * (28 + 13 * len(names)))
    for arm_name in arms:
        row = " ".join(f"{'ok' if matrix[arm_name][n] else 'CATCH':>12s}" for n in names)
        print(f"{arm_name:28s} {row}")

    print()
    verdicts = {}
    for defect, guard_name in INTENDED.items():
        caught = not matrix[defect][guard_name]
        honest_ok = matrix["0_honest"][guard_name]
        # a guard that fires on the honest pipeline is broken, not strict
        verdicts[defect] = {
            "intended_guard": guard_name, "caught": caught,
            "guard_passes_honest": honest_ok, "valid": caught and honest_ok,
        }
        status = ("CAUGHT" if caught else "MISSED") + ("" if honest_ok else " (guard fires on honest!)")
        print(f"  {defect:28s} -> {guard_name:36s} {status}")

    vacuous = [n for n in names
               if all(matrix[a][n] for a in arms)]
    overbroad = [n for n in names
                 if not matrix["0_honest"][n]]
    all_caught = all(v["valid"] for v in verdicts.values())
    report = {
        "matrix": matrix, "verdicts": verdicts,
        "vacuous_guards": vacuous, "guards_failing_honest": overbroad,
        "t2_all_defects_caught": all_caught,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    print(f"\nvacuous guards (pass everything): {vacuous or 'none'}")
    print(f"guards that fire on the honest pipeline: {overbroad or 'none'}")
    print(f"T2 (every planted defect caught by its guard): {all_caught}")
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
