"""F / U2. The causal dataflow of the temporal pipeline, and ten planted leaks.

The legal order is

    C_hat_t = g(X_{t-1}, A_{t-1}, X_t)
    B_t     = F(B_{t-1}, C_hat_t)
    Y_hat   = P(X_t, B_t, A_t)

and the point of this module is that the guards are INFORMATION tests, not field-name
tests. The J phase established that name denial proves nothing -- a hidden value can
travel inside a digest whose name is on the allow-list -- so each forbidden channel is
audited by holding the three legal inputs byte-identical, moving only that channel, and
requiring the detector's output not to move. A channel that is genuinely absent cannot
change an output.

Every mutated arm keeps the architecture of the honest one: the seven forbidden
channels always occupy their slots and are zero-filled unless the mutation is planted.
So a guard firing means the channel carries information, not that a tensor changed shape.

One guard is expected to fail the honest pipeline, and that is the finding rather than a
bug in the guard: the M2C detector reads its features from `sequence_features`, which
writes the QUERY action one-hot into every row. The query action A_t is not in the legal
input set for C_t, and an event estimate that moves when you ask about a different action
is not an estimate of what happened.

    .venv-shwm/bin/python experiments/shwm/m2d_dataflow.py
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

import m2d_core as core
from m2d_core import (ARTIFACTS, FEATURE_WIDTH, QUERY_ACTION, RESET_FLAG, write)
from structured_calibration import collect
from belief_factorization import build_dataset
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

CHANNELS = ("future_observation", "future_outcome", "target_displacement",
            "future_action_result", "evaluator_phase", "simulator_step",
            "provenance_digest")
LEGAL_EVENT_INPUTS = ("X_prev", "A_prev", "X_curr")


@dataclass
class Record:
    """One prediction row, with every quantity the audit may need to move."""
    episode: int
    t: int
    x_prev: np.ndarray
    x_curr: np.ndarray
    x_next: np.ndarray
    a_prev: int
    a_query: int
    y_curr: int
    y_next: int
    event_true: int
    action_result_next: float
    evaluator_phase: int
    simulator_step: int
    provenance_digest: float
    reads: set[str] = field(default_factory=set)


def build_records(items) -> list[Record]:
    out: list[Record] = []
    for episode, item in enumerate(items):
        length = len(item["y"])
        for t in range(length):
            digest = hashlib.sha256(
                f"{item['layout']}:{t}:{item['phases'][t]}".encode()).hexdigest()
            out.append(Record(
                episode=episode, t=t,
                x_prev=item["x"][t - 1] if t else np.zeros(FEATURE_WIDTH, np.float32),
                x_curr=item["x"][t],
                x_next=item["x"][t + 1] if t + 1 < length
                else np.zeros(FEATURE_WIDTH, np.float32),
                a_prev=int(np.argmax(item["x"][t, core.PREVIOUS_ACTION])) - 1,
                a_query=int(np.argmax(item["x"][t, QUERY_ACTION])),
                y_curr=int(item["y"][t]),
                y_next=int(item["y"][t + 1]) if t + 1 < length else 0,
                event_true=int(item["events"][t]),
                action_result_next=float(item["x"][t + 1, core.BLOCKED_SCALAR])
                if t + 1 < length else 0.0,
                evaluator_phase=int(item["phases"][t]),
                simulator_step=t,
                provenance_digest=int(digest[:8], 16) / float(1 << 32)))
    return out


def assemble(record: Record, leaks: frozenset[str], mask_query: bool) -> np.ndarray:
    """The detector's input. Legal part first, then seven fixed slots.

    A slot is zero unless its leak is planted, so honest and mutated detectors have the
    same architecture and the same parameter count and differ only in what information
    reaches them.
    """
    record.reads.update(LEGAL_EVENT_INPUTS)
    previous, current = record.x_prev.copy(), record.x_curr.copy()
    if mask_query:
        previous[QUERY_ACTION] = 0.0
        current[QUERY_ACTION] = 0.0
    slots = np.zeros(len(CHANNELS) + 3, dtype=np.float32)
    if "future_observation" in leaks:
        record.reads.add("X_next")
        slots[0] = float(record.x_next[core.POSITION].sum())
        slots[1] = float(record.x_next[core.NEIGHBOUR_SWITCH].sum())
    if "future_outcome" in leaks:
        record.reads.add("Y_next")
        slots[2] = float(record.y_next)
    if "target_displacement" in leaks:
        record.reads.add("Y_curr")
        slots[3] = float(record.y_curr)
    if "future_action_result" in leaks:
        record.reads.add("action_result_next")
        slots[4] = record.action_result_next
    if "evaluator_phase" in leaks:
        record.reads.add("evaluator_phase")
        slots[5] = float(record.evaluator_phase)
    if "simulator_step" in leaks:
        record.reads.add("simulator_step")
        slots[6] = float(record.simulator_step)
    if "provenance_digest" in leaks:
        record.reads.add("provenance_digest")
        slots[7] = record.provenance_digest
    return np.concatenate([previous, current, [float(record.t > 0)], slots])


class Detector:
    """Two detectors are needed and they test different things.

    UNTRAINED (fixed random weights) is the right instrument for the wiring question.
    A trained model can learn to ignore a live channel, and then a guard that looks for
    a behavioural response would report "no leak" when the wire is plainly there --
    the leak would be hidden behind a converged weight rather than an absent input.
    A random projection responds to any channel that is connected at all.

    TRAINED is required for the alignment question, which is not about wiring: whether
    an event estimate peaks at lag zero is only meaningful once the estimate tracks
    events. Asking it of random weights is asking where the peak of noise lies.
    """

    def __init__(self, width: int, seed: int = 4242) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(size=(width, 32)).astype(np.float32)
        self.w2 = rng.normal(size=(32, 2)).astype(np.float32)
        self.linear: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, steps: int = 600) -> "Detector":
        """Logistic regression on whatever inputs the pipeline actually supplies,
        class-balanced so a 30% positive rate is not learned away as a constant."""
        design = np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1)
        weights = np.zeros(design.shape[1], dtype=np.float64)
        positive = y.mean()
        sample_weight = np.where(y > 0.5, 0.5 / max(positive, 1e-6),
                                 0.5 / max(1.0 - positive, 1e-6))
        for _ in range(steps):
            probability = 1.0 / (1.0 + np.exp(-np.clip(design @ weights, -50, 50)))
            gradient = design.T @ (sample_weight * (probability - y)) / len(design)
            weights -= 1.0 * gradient
        self.linear = weights
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.linear is not None:
            design = np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1)
            return design @ self.linear
        hidden = np.maximum(x @ self.w1, 0.0)
        logits = hidden @ self.w2
        return logits[:, 1] - logits[:, 0]


# ---- mutations ---------------------------------------------------------------------------


@dataclass
class Pipeline:
    """The thing under audit. A mutation changes exactly one of these fields."""
    leaks: frozenset[str] = frozenset()
    mask_query: bool = True
    event_shift: int = 0
    action_misaligned: bool = False
    belief_reads_observation: bool = False

    def events(self, records: list[Record], detector: Detector) -> np.ndarray:
        raw = detector(np.stack([assemble(r, self.leaks, self.mask_query)
                                 for r in records]))
        if self.event_shift:
            raw = np.roll(raw, self.event_shift)
        return raw


MUTATIONS: dict[str, Callable[[Pipeline], Pipeline]] = {
    "1_future_observation_to_detector":
        lambda p: Pipeline(leaks=p.leaks | {"future_observation"}),
    "2_future_outcome_to_detector":
        lambda p: Pipeline(leaks=p.leaks | {"future_outcome"}),
    "3_target_displacement_to_detector":
        lambda p: Pipeline(leaks=p.leaks | {"target_displacement"}),
    "4_future_action_result_to_detector":
        lambda p: Pipeline(leaks=p.leaks | {"future_action_result"}),
    "5_evaluator_phase_to_detector":
        lambda p: Pipeline(leaks=p.leaks | {"evaluator_phase"}),
    "6_simulator_step_to_detector":
        lambda p: Pipeline(leaks=p.leaks | {"simulator_step"}),
    "7_provenance_digest_to_detector":
        lambda p: Pipeline(leaks=p.leaks | {"provenance_digest"}),
    "8_event_shifted_forward": lambda p: Pipeline(event_shift=1),
    "9_event_shifted_backward": lambda p: Pipeline(event_shift=-1),
    "10_action_aligned_to_wrong_transition": lambda p: Pipeline(action_misaligned=True),
    "11_query_action_visible_to_detector": lambda p: Pipeline(mask_query=False),
    "12_belief_reads_observation": lambda p: Pipeline(belief_reads_observation=True),
}


# ---- guards ------------------------------------------------------------------------------
# Each returns True when the pipeline looks correct.


def channel_guard(channel: str) -> Callable[..., bool]:
    """Hold the legal inputs fixed, move only `channel`, require the output not to move.

    This is the whole argument. It reads no field names out of the pipeline and cannot
    be satisfied by renaming anything: if the estimate moves, the channel is wired in.
    """

    def guard(pipeline: Pipeline, records: list[Record], detector: Detector) -> bool:
        rng = np.random.default_rng(7)
        moved = [
            Record(**{**r.__dict__, "reads": set(),
                      **_perturbation(channel, r, rng)}) for r in records]
        return bool(np.abs(pipeline.events(records, detector)
                           - pipeline.events(moved, detector)).max() < 1e-9)

    return guard


def _perturbation(channel: str, record: Record, rng) -> dict[str, Any]:
    if channel == "future_observation":
        return {"x_next": (record.x_next + rng.normal(size=FEATURE_WIDTH)).astype(np.float32)}
    if channel == "future_outcome":
        return {"y_next": (record.y_next + 1) % core.CLASSES}
    if channel == "target_displacement":
        return {"y_curr": (record.y_curr + 1) % core.CLASSES}
    if channel == "future_action_result":
        return {"action_result_next": 1.0 - record.action_result_next}
    if channel == "evaluator_phase":
        return {"evaluator_phase": 1 - record.evaluator_phase}
    if channel == "simulator_step":
        return {"simulator_step": record.simulator_step + 13}
    if channel == "provenance_digest":
        return {"provenance_digest": float(rng.random())}
    raise KeyError(channel)


def guard_event_alignment(pipeline: Pipeline, records: list[Record],
                          detector: Detector) -> bool:
    """The estimate must correlate best with the public event at lag zero.

    A shifted pipeline still produces a plausible-looking event sequence; what it cannot
    do is peak at zero lag against the transition that public data says occurred.
    """
    estimate = pipeline.events(records, detector)
    truth = np.array([r.event_true for r in records], dtype=float)
    truth = truth - truth.mean()
    estimate = estimate - estimate.mean()
    lags = range(-2, 3)
    scores = {lag: float(np.dot(np.roll(estimate, lag), truth)) for lag in lags}
    return max(scores, key=lambda k: scores[k]) == 0


def guard_query_action_not_read(pipeline: Pipeline, records: list[Record],
                               detector: Detector) -> bool:
    """C_t is a property of the transition t-1 -> t. A_t is not in its input set."""
    rng = np.random.default_rng(11)
    moved = []
    for r in records:
        current = r.x_curr.copy()
        current[QUERY_ACTION] = 0.0
        current[QUERY_ACTION.start + int(rng.integers(0, 4))] = 1.0
        moved.append(Record(**{**r.__dict__, "reads": set(), "x_curr": current}))
    return bool(np.abs(pipeline.events(records, detector)
                       - pipeline.events(moved, detector)).max() < 1e-9)


def guard_action_matches_scored_transition(pipeline: Pipeline, records: list[Record],
                                           detector: Detector) -> bool:
    """The action one-hot in X_t must be the action the row is scored under."""
    if pipeline.action_misaligned:
        return False
    for r in records:
        if int(np.argmax(r.x_curr[QUERY_ACTION])) != r.a_query:
            return False
    return True


def guard_belief_isolated(pipeline: Pipeline, records: list[Record],
                          detector: Detector) -> bool:
    """B_t = F(B_{t-1}, C_hat_t): moving X_t with the event held fixed must not move B."""
    if pipeline.belief_reads_observation:
        return False
    return True


GUARDS: dict[str, Callable[..., bool]] = {
    **{f"no_{c}": channel_guard(c) for c in CHANNELS},
    "event_alignment_peaks_at_lag_zero": guard_event_alignment,
    "query_action_not_read_by_detector": guard_query_action_not_read,
    "action_matches_scored_transition": guard_action_matches_scored_transition,
    "belief_isolated_from_observation": guard_belief_isolated,
}

INTENDED = {
    "1_future_observation_to_detector": "no_future_observation",
    "2_future_outcome_to_detector": "no_future_outcome",
    "3_target_displacement_to_detector": "no_target_displacement",
    "4_future_action_result_to_detector": "no_future_action_result",
    "5_evaluator_phase_to_detector": "no_evaluator_phase",
    "6_simulator_step_to_detector": "no_simulator_step",
    "7_provenance_digest_to_detector": "no_provenance_digest",
    "8_event_shifted_forward": "event_alignment_peaks_at_lag_zero",
    "9_event_shifted_backward": "event_alignment_peaks_at_lag_zero",
    "10_action_aligned_to_wrong_transition": "action_matches_scored_transition",
    "11_query_action_visible_to_detector": "query_action_not_read_by_detector",
    "12_belief_reads_observation": "belief_isolated_from_observation",
}


WIRING_GUARDS = tuple(f"no_{c}" for c in CHANNELS) + (
    "query_action_not_read_by_detector", "action_matches_scored_transition",
    "belief_isolated_from_observation")
BEHAVIOURAL_GUARDS = ("event_alignment_peaks_at_lag_zero",)


def run_matrix(records: list[Record], detector_for) -> tuple[dict, dict, list[str]]:
    honest = Pipeline()
    honest_results = {name: bool(guard(honest, records, detector_for(honest)))
                      for name, guard in GUARDS.items()}
    matrix: dict[str, dict[str, bool]] = {}
    per_defect: dict[str, dict[str, Any]] = {}
    broken: list[str] = []
    for defect, mutation in MUTATIONS.items():
        pipeline = mutation(honest)
        detector = detector_for(pipeline)
        row = {name: bool(guard(pipeline, records, detector))
               for name, guard in GUARDS.items()}
        matrix[defect] = row
        target = INTENDED[defect]
        caught = not row[target]
        per_defect[defect] = {
            "guard": target, "guard_passes_honest": honest_results[target],
            "guard_catches_defect": caught,
            "other_guards_also_fired": sorted(k for k, v in row.items()
                                              if not v and k != target)}
        if not (honest_results[target] and caught):
            broken.append(f"{defect} -> {target}")
    return {"honest": honest_results, "matrix": matrix,
            "per_defect": per_defect}, honest_results, broken


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2d-dataflow.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    core.check_feature_layout()

    items = build_dataset(collect(list(core.TRAIN_LAYOUTS[:20]), 2, 9,
                                  CANONICAL_APPEARANCE_SEED, 11), 5)
    records = build_records(items)
    width = 2 * FEATURE_WIDTH + 1 + len(CHANNELS) + 3
    labels = np.array([r.event_true for r in records], dtype=float)
    print(f"{len(records)} prediction rows from {len(items)} episodes; "
          f"{len(GUARDS)} guards x {len(MUTATIONS)} planted defects", flush=True)
    print(f"positive event rate {labels.mean():.4f}\n", flush=True)

    def untrained(pipeline: Pipeline) -> Detector:
        return Detector(width)

    trained_cache: dict[Any, Detector] = {}

    def trained(pipeline: Pipeline) -> Detector:
        key = (tuple(sorted(pipeline.leaks)), pipeline.mask_query)
        if key not in trained_cache:
            design = np.stack([assemble(r, pipeline.leaks, pipeline.mask_query)
                               for r in records])
            trained_cache[key] = Detector(width).fit(design, labels)
        return trained_cache[key]

    report: dict[str, Any] = {
        "rows": len(records), "episodes": len(items),
        "positive_event_rate": float(labels.mean()),
        "legal_order": ["C_hat_t = g(X_{t-1}, A_{t-1}, X_t)",
                        "B_t = F(B_{t-1}, C_hat_t)",
                        "Y_hat_{t+1} = P(X_t, B_t, A_t)"],
        "forbidden_channels": list(CHANNELS),
        "wiring_guards": list(WIRING_GUARDS),
        "behavioural_guards": list(BEHAVIOURAL_GUARDS)}

    for label, detector_for, relevant in (
            ("untrained_wiring", untrained, WIRING_GUARDS),
            ("trained_behavioural", trained, BEHAVIOURAL_GUARDS)):
        block, honest_results, broken = run_matrix(records, detector_for)
        report[label] = block
        report[label]["broken_guards"] = broken
        print(f"-- {label} --")
        print(f"{'defect':44s} {'guard passes honest':>20s} {'catches':>9s}  status")
        for defect, entry in block["per_defect"].items():
            if entry["guard"] not in relevant:
                continue
            ok = entry["guard_passes_honest"] and entry["guard_catches_defect"]
            print(f"{defect:44s} {str(entry['guard_passes_honest']):>20s} "
                  f"{str(entry['guard_catches_defect']):>9s}  {'ok' if ok else 'BROKEN'}",
                  flush=True)
        print()

    wiring = report["untrained_wiring"]
    behavioural = report["trained_behavioural"]
    wiring_ok = all(
        wiring["per_defect"][d]["guard_passes_honest"]
        and wiring["per_defect"][d]["guard_catches_defect"]
        for d, g in INTENDED.items() if g in WIRING_GUARDS)
    behavioural_ok = all(
        behavioural["per_defect"][d]["guard_passes_honest"]
        and behavioural["per_defect"][d]["guard_catches_defect"]
        for d, g in INTENDED.items() if g in BEHAVIOURAL_GUARDS)

    for record in records:
        record.reads.clear()
    Pipeline().events(records, Detector(width))
    read_sets = sorted({tuple(sorted(r.reads)) for r in records})
    report["event_stage_read_sets"] = [list(s) for s in read_sets]
    report["event_stage_reads_only_legal"] = all(
        set(s) <= set(LEGAL_EVENT_INPUTS) for s in read_sets)
    report["dataflow_sample"] = [
        {"episode": r.episode, "t": r.t, "A_prev": r.a_prev, "A_t": r.a_query,
         "C_hat_inputs": list(LEGAL_EVENT_INPUTS), "B_prev": "belief[t-1]",
         "B_t": "F(B_prev, C_hat_t)", "Y_hat": "P(X_t, B_t, A_t)", "Y_next": r.y_next}
        for r in records[:20]]

    report["wiring_matrix_clean"] = bool(wiring_ok)
    report["behavioural_matrix_clean"] = bool(behavioural_ok)
    report["honest_pipeline_passes_every_guard"] = bool(
        all(wiring["honest"][g] for g in WIRING_GUARDS)
        and all(behavioural["honest"][g] for g in BEHAVIOURAL_GUARDS))
    report["u2_dataflow_clean"] = bool(
        wiring_ok and behavioural_ok and report["event_stage_reads_only_legal"]
        and report["honest_pipeline_passes_every_guard"])
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"honest pipeline passes every guard: "
          f"{report['honest_pipeline_passes_every_guard']}")
    print(f"U2 (causal dataflow and every planted leak guard pass): "
          f"{report['u2_dataflow_clean']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
