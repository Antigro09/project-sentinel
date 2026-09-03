"""I / J / K / L / M. Event-detector family, route-level fidelity, and the full pathway.

Runs only if F0-F9 pass; the driver refuses otherwise.

The family is bounded and fixed before any validation exposure. The interesting member is
the relational detector: the current one reads absolute position and goal direction, which
is exactly how it reaches 1.0000 on training layouts and 0.80 on held-out ones. A detector
restricted to relative quantities -- the displacement taken, and the switch bits of the
neighbours of the cell just left -- is translation-equivariant by construction and has no
layout-specific feature to memorise.

Section J's parity diagnostic is used in its non-identical form,

    P(correct final parity) = [1 + prod_t (1 - 2 e_t)] / 2

with e_t the detector's own probability of being wrong, min(p_t, 1 - p_t). No balanced
accuracy is plugged into it.

    .venv-shwm/bin/python experiments/shwm/m2f_events.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import m2d_core as m2d
import m2e_core as m2e
import m2f_core as core
from m2d_core import (ARTIFACTS, FilterSpec, NEIGHBOUR_SWITCH, PREVIOUS_ACTION,
                      POSITION, QUERY_ACTION, BLOCKED_SCALAR, write)
from m2d_coupling import EventDetector, corrupt, CORRUPTIONS, collect_goal_directed
from m2e_event_target import detector_nll, classification_metrics
from structured_calibration import collect, DELTAS_BY_INDEX
from belief_factorization import build_dataset
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

# New layout ranges for the event phase.
EVENT_VALIDATION_LAYOUTS = tuple(range(83_000, 83_020))
EVENT_HELD_OUT_LAYOUTS = tuple(range(97_000, 97_020))
NEW_ALIAS = tuple(range(97_000, 97_010))

COUPLING_CRITERION = ("primary: phase-sensitive NLL on development alias rows; "
                      "secondary: exact alias-pair accuracy; "
                      "calibration constraint: Brier no worse than the best rule "
                      "by more than 0.02")


class RelationalDetector(EventDetector):
    """Translation- and geometry-equivariant: only relative quantities reach it.

    Inputs are the displacement actually taken (derived from the two positions), the
    switch bits of the four neighbours of the PREVIOUS cell, the previous action and the
    blocked flag. Absolute position, goal direction and the query action are all absent,
    so there is no layout-specific feature for it to memorise and nothing to unlearn when
    the walls move.
    """

    def featurise(self, sequences: np.ndarray) -> np.ndarray:
        current = np.asarray(sequences, dtype=np.float32)
        previous = np.concatenate([np.zeros_like(current[:, :1]), current[:, :-1]], axis=1)
        position = current[..., POSITION] * m2d.GRID
        previous_position = previous[..., POSITION] * m2d.GRID
        delta = np.rint(position - previous_position).astype(int)
        displacement = np.zeros(current.shape[:2] + (len(DELTAS_BY_INDEX) + 1,),
                                dtype=np.float32)
        for index, (dr, dc) in enumerate(DELTAS_BY_INDEX):
            match = (delta[..., 0] == dr) & (delta[..., 1] == dc)
            displacement[..., index] = match.astype(np.float32)
        displacement[..., -1] = (displacement[..., :-1].sum(axis=-1) == 0).astype(np.float32)
        first = np.zeros(current.shape[:2] + (1,), dtype=np.float32)
        first[:, 1:, 0] = 1.0
        return np.concatenate([
            displacement,
            previous[..., NEIGHBOUR_SWITCH],
            previous[..., PREVIOUS_ACTION],
            current[..., BLOCKED_SCALAR][..., None],
            first], axis=-1)


class ExactDerivationDetector(EventDetector):
    """The public derivation, as a detector. The ceiling for the family."""

    def __init__(self) -> None:
        super().__init__("exact")

    def fit(self, items, seed: int = 0, updates: int = 0) -> "ExactDerivationDetector":
        return self

    def probabilities(self, sequences: np.ndarray, batch: int = 0) -> np.ndarray:
        current = np.asarray(sequences, dtype=np.float32)
        previous = np.concatenate([np.zeros_like(current[:, :1]), current[:, :-1]], axis=1)
        position = np.rint(current[..., POSITION] * m2d.GRID).astype(int)
        previous_position = np.rint(previous[..., POSITION] * m2d.GRID).astype(int)
        delta = position - previous_position
        out = np.zeros(current.shape[:2], dtype=np.float32)
        for index, (dr, dc) in enumerate(DELTAS_BY_INDEX):
            moved = (delta[..., 0] == dr) & (delta[..., 1] == dc)
            out += moved.astype(np.float32) * previous[..., NEIGHBOUR_SWITCH.start + index]
        out[:, 0] = 0.0
        return np.clip(out, 0.0, 1.0)


class CalibratedDetector(EventDetector):
    """Temperature-scaled, fitted on a split the detector did not train on."""

    def __init__(self, base: EventDetector, calibration_items) -> None:
        super().__init__(base.mode)
        self.base = base
        x, y, e, m, _ = m2d.pad(calibration_items)
        probability = np.clip(base.probabilities(x).astype(np.float64), 1e-6, 1 - 1e-6)
        valid = m.astype(bool)
        valid[:, 0] = False
        logit = np.log(probability[valid] / (1 - probability[valid]))
        truth = e[valid]
        best, self.temperature = np.inf, 1.0
        for temperature in np.linspace(0.25, 6.0, 116):
            scaled = 1.0 / (1.0 + np.exp(-logit / temperature))
            loss = detector_nll(truth, scaled)
            if loss < best:
                best, self.temperature = loss, float(temperature)

    def probabilities(self, sequences: np.ndarray, batch: int = 2048) -> np.ndarray:
        probability = np.clip(self.base.probabilities(sequences).astype(np.float64),
                              1e-6, 1 - 1e-6)
        logit = np.log(probability / (1 - probability))
        out = (1.0 / (1.0 + np.exp(-logit / self.temperature))).astype(np.float32)
        out[:, 0] = 0.0
        return out


def sequence_metrics(probability: np.ndarray, truth: np.ndarray, lengths: np.ndarray,
                     layouts: np.ndarray | None = None) -> dict[str, Any]:
    """Route-level fidelity, and the non-identical-error parity prediction."""
    hard = (probability >= 0.5).astype(np.float32)
    exact, parity_ok, first_error, bursts, predicted_parity = [], [], [], [], []
    for k, n in enumerate(lengths):
        span = slice(1, int(n))
        wrong = hard[k, span] != truth[k, span]
        exact.append(not wrong.any())
        parity_ok.append((hard[k, span].sum() % 2) == (truth[k, span].sum() % 2))
        first_error.append(int(np.argmax(wrong)) + 1 if wrong.any() else -1)
        run = best = 0
        for value in wrong:
            run = run + 1 if value else 0
            best = max(best, run)
        bursts.append(best)
        error = np.minimum(probability[k, span], 1.0 - probability[k, span])
        predicted_parity.append(float((1.0 + np.prod(1.0 - 2.0 * error)) / 2.0))
    flat_wrong = np.concatenate(
        [(hard[k, 1:int(n)] != truth[k, 1:int(n)]).astype(float)
         for k, n in enumerate(lengths)])
    lagged = np.concatenate(
        [(hard[k, 1:int(n) - 1] != truth[k, 1:int(n) - 1]).astype(float)
         for k, n in enumerate(lengths) if n > 2])
    lagged_next = np.concatenate(
        [(hard[k, 2:int(n)] != truth[k, 2:int(n)]).astype(float)
         for k, n in enumerate(lengths) if n > 2])
    autocorrelation = float(np.corrcoef(lagged, lagged_next)[0, 1]) if (
        len(lagged) > 2 and lagged.std() > 0 and lagged_next.std() > 0) else float("nan")
    out = {
        "exact_route_sequence_accuracy": float(np.mean(exact)),
        "final_event_parity_accuracy": float(np.mean(parity_ok)),
        "predicted_parity_non_identical_errors": float(np.mean(predicted_parity)),
        "mean_first_error_position": float(np.mean([f for f in first_error if f > 0]))
        if any(f > 0 for f in first_error) else float("nan"),
        "mean_error_burst_length": float(np.mean(bursts)),
        "max_error_burst_length": int(np.max(bursts)) if len(bursts) else 0,
        "error_autocorrelation_lag1": autocorrelation,
        "per_step_error_rate": float(flat_wrong.mean()),
        "routes": int(len(lengths)),
    }
    if layouts is not None:
        failure = {}
        for layout in np.unique(layouts):
            mask = layouts == layout
            failure[int(layout)] = float(1.0 - np.mean(np.array(parity_ok)[mask]))
        out["layout_conditioned_parity_failure_rate"] = failure
        out["worst_layout_parity_failure_rate"] = float(max(failure.values()))
    return out


def step_metrics(probability, truth, mask) -> dict[str, float]:
    block = classification_metrics(truth[mask], probability[mask])
    block["accuracy"] = float(((probability[mask] >= 0.5).astype(int)
                               == truth[mask].astype(int)).mean())
    block["nll"] = detector_nll(truth[mask], probability[mask])
    return block


def stratified(probability, truth, mask, x, lengths, layouts) -> dict[str, Any]:
    """Per-step and route-level metrics, sliced every way section J asks for."""
    action = np.argmax(x[..., PREVIOUS_ACTION], axis=-1) - 1
    crossings = np.cumsum(truth, axis=1)
    step = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))
    out: dict[str, Any] = {"overall": step_metrics(probability, truth, mask)}
    out["sequence"] = sequence_metrics(probability, truth, lengths, layouts)
    for a in range(4):
        picked = mask & (action == a)
        if picked.sum() > 20:
            out[f"action_{a}"] = step_metrics(probability, truth, picked)
    for label, low, high in (("events_0", 0, 1), ("events_1", 1, 2), ("events_2", 2, 3),
                             ("events_3", 3, 4), ("events_4plus", 4, 99)):
        picked = mask & (crossings >= low) & (crossings < high)
        if picked.sum() > 20:
            out[label] = step_metrics(probability, truth, picked)
    for s in range(1, 9):
        picked = mask & (step == s)
        if picked.sum() > 20:
            out[f"time_since_reset_{s}"] = step_metrics(probability, truth, picked)
    for length in sorted(set(int(n) for n in lengths)):
        chosen = lengths == length
        if chosen.sum() > 5:
            out[f"route_length_{length}"] = sequence_metrics(
                probability[chosen], truth[chosen], lengths[chosen])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--force", action="store_true",
                        help="run even though F0-F9 did not all pass (diagnostic only)")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2f-events.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    procedures = json.loads((ARTIFACTS / "m2f-procedures.json").read_text())
    if not procedures["f_gates_all_pass"] and not arguments.force:
        failed = [k for k, v in procedures["gates"].items() if not v["pass"]]
        print(f"F gates {failed} did not pass; section I is gated on F0-F9.")
        print("Refusing to run. Pass --force to produce non-qualifying diagnostics.")
        return 2
    qualifying = bool(procedures["f_gates_all_pass"])
    tau = procedures["tau"]

    appearance = CANONICAL_APPEARANCE_SEED
    train_t = collect(list(m2d.TRAIN_LAYOUTS), 3, 9, appearance, 11)
    train = build_dataset(train_t, 5)
    calibration = build_dataset(
        collect(list(m2d.DETECTOR_TEST_LAYOUTS), 2, 9, appearance, 777), 6)
    splits = {
        "development_layouts": (train, m2d.TRAIN_LAYOUTS),
        "validation_layouts": (build_dataset(
            collect(list(EVENT_VALIDATION_LAYOUTS), 2, 9, appearance, 515), 9),
            EVENT_VALIDATION_LAYOUTS),
        "held_out_layouts": (build_dataset(
            collect(list(EVENT_HELD_OUT_LAYOUTS), 2, 9, appearance, 616), 10),
            EVENT_HELD_OUT_LAYOUTS),
        "held_out_visitation_policy": (build_dataset(
            collect_goal_directed(list(EVENT_HELD_OUT_LAYOUTS), 2, 9, appearance, 717), 11),
            EVENT_HELD_OUT_LAYOUTS),
    }

    base = EventDetector("full").fit(train)
    family: dict[str, EventDetector] = {
        "1_current_structured": base,
        "2_action_only": EventDetector("action_only").fit(train),
        "3_state_pair_only": EventDetector("state_only").fit(train),
        "4_relational_equivariant": RelationalDetector("full").fit(train),
        "5_calibrated": CalibratedDetector(base, calibration),
        "6_exact_public_derivation": ExactDerivationDetector(),
    }

    report: dict[str, Any] = {
        "qualifying": qualifying, "tau": tau,
        "event_validation_layouts": list(EVENT_VALIDATION_LAYOUTS),
        "event_held_out_layouts": list(EVENT_HELD_OUT_LAYOUTS),
        "new_alias_layouts": list(NEW_ALIAS),
        "coupling_criterion": COUPLING_CRITERION,
        "detector_scope": EventDetector.RETROSPECTIVE,
        "held_out_dynamics": ("NOT INSTANTIABLE in v2: SWITCH_COUNT is constant and the "
                              "flip rule never varies. The visitation-policy split is "
                              "reported instead and labelled as such."),
        "detectors": {}}

    print(f"detector family on {len(splits)} splits "
          f"(qualifying={qualifying})\n")
    print(f"{'detector':28s} {'split':30s} {'bal':>7s} {'acc':>7s} {'Brier':>7s} "
          f"{'NLL':>7s} {'ECE':>7s} {'exact-seq':>9s} {'parity':>7s} {'pred':>7s}")
    print("-" * 122)
    for name, detector in family.items():
        block: dict[str, Any] = {}
        for split, (items, layouts) in splits.items():
            x, y, e, m, _ = m2d.pad(items)
            probability = detector.probabilities(x)
            valid = m.astype(bool)
            valid[:, 0] = False
            lengths = np.array([len(i["y"]) for i in items])
            layout_ids = np.array([i["layout"] for i in items])
            block[split] = stratified(probability, e, valid, x, lengths, layout_ids)
            o, sq = block[split]["overall"], block[split]["sequence"]
            print(f"{name:28s} {split:30s} {o['balanced_accuracy']:7.4f} "
                  f"{o['accuracy']:7.4f} {o['brier']:7.4f} {o['nll']:7.4f} "
                  f"{o['expected_calibration_error']:7.4f} "
                  f"{sq['exact_route_sequence_accuracy']:9.4f} "
                  f"{sq['final_event_parity_accuracy']:7.4f} "
                  f"{sq['predicted_parity_non_identical_errors']:7.4f}", flush=True)
        report["detectors"][name] = block
        print()
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
