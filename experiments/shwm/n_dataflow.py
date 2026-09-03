"""C / N1. The visual retrospective dataflow, and twelve planted channel leaks.

Legal order:

    event_hat_t   = g(frame_before, action, frame_after)
    belief_t      = F(belief_{t-1}, event_hat_t)
    outcome_hat   = P(frame_after, belief_t, A_t)

Guards are INFORMATION tests. Each forbidden channel occupies a permanently present,
zero-filled slot, so honest and mutated detectors have identical shape and parameter
count; the guard holds the two frames and the action byte-identical, moves only that
channel, and requires the estimate not to move. A channel that is genuinely absent
cannot change an output, and no amount of renaming can fake that.

The wiring guards run on an UNTRAINED random projection on purpose: a trained detector
can learn to ignore a live wire, which would hide a real leak behind a converged weight.
The alignment guard needs the opposite -- a detector whose output tracks events -- so it
runs trained. Running both under one detector is what broke the first structured build.

    .venv-shwm/bin/python experiments/shwm/n_dataflow.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

import n_core as core
from m2d_core import ARTIFACTS, write
from n_core import GRID, PairBatch

CHANNELS = ("structured_agent_coordinates", "structured_switch_neighbour_bits",
            "true_displacement", "true_switch_event", "true_phase", "simulator_step",
            "environment_seed", "layout_id", "provenance_digest", "future_frame",
            "future_action_result", "target_outcome")
LEGAL = ("frame_before", "action", "frame_after")
SLOTS = len(CHANNELS) + 4          # four permanently-zero spares


@dataclass
class Row:
    """One visual prediction row plus every quantity the audit may need to move."""
    before: np.ndarray
    after: np.ndarray
    action: np.ndarray
    agent_cell: int
    switch_neighbours: np.ndarray
    displacement: int
    event: float
    phase: int
    step: int
    seed: int
    layout: int
    provenance: float
    future_frame: np.ndarray
    future_action_result: float
    target_outcome: int


def build_rows(pairs: PairBatch, episodes) -> list[Row]:
    rows: list[Row] = []
    rng = np.random.default_rng(5)
    for i in range(len(pairs)):
        episode = episodes[int(pairs.episode[i])]
        t = int(pairs.step[i])
        cell = int(pairs.agent_after[i].argmax())
        r, c = divmod(cell, GRID)
        neighbours = np.zeros(4, dtype=np.float32)
        for k, (dr, dc) in enumerate(((-1, 0), (0, 1), (1, 0), (0, -1))):
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID and 0 <= nc < GRID:
                neighbours[k] = episode.switch_mask[t][nr, nc]
        rows.append(Row(
            before=pairs.before[i], after=pairs.after[i], action=pairs.action[i],
            agent_cell=cell, switch_neighbours=neighbours,
            displacement=int(pairs.displacement[i]), event=float(pairs.event[i]),
            phase=int(episode.polarity[t]), step=t,
            seed=int(episode.appearance), layout=int(pairs.layout[i]),
            provenance=float(int(str(abs(hash((int(pairs.layout[i]), t))))[:6]) / 1e6),
            future_frame=(episode.frames[t + 1].astype(np.float32) / 255.0
                          if t + 1 < episode.length else np.zeros_like(pairs.after[i])),
            future_action_result=float(rng.random() < 0.5),
            target_outcome=int(pairs.displacement[i])))
    return rows


@dataclass
class Pipeline:
    leaks: frozenset = frozenset()
    event_shift: int = 0
    action_misaligned: bool = False
    belief_reads_frame: bool = False

    def assemble(self, rows: list[Row]) -> np.ndarray:
        legal = np.stack([
            np.concatenate([r.before.ravel(), r.after.ravel(), r.action]) for r in rows])
        slots = np.zeros((len(rows), SLOTS), dtype=np.float32)
        for i, r in enumerate(rows):
            if "structured_agent_coordinates" in self.leaks:
                slots[i, 0] = r.agent_cell / (GRID * GRID)
            if "structured_switch_neighbour_bits" in self.leaks:
                slots[i, 1] = float(r.switch_neighbours.sum())
            if "true_displacement" in self.leaks:
                slots[i, 2] = float(r.displacement)
            if "true_switch_event" in self.leaks:
                slots[i, 3] = r.event
            if "true_phase" in self.leaks:
                slots[i, 4] = float(r.phase)
            if "simulator_step" in self.leaks:
                slots[i, 5] = float(r.step)
            if "environment_seed" in self.leaks:
                slots[i, 6] = float(r.seed % 1000) / 1000.0
            if "layout_id" in self.leaks:
                slots[i, 7] = float(r.layout % 1000) / 1000.0
            if "provenance_digest" in self.leaks:
                slots[i, 8] = r.provenance
            if "future_frame" in self.leaks:
                slots[i, 9] = float(r.future_frame.mean())
            if "future_action_result" in self.leaks:
                slots[i, 10] = r.future_action_result
            if "target_outcome" in self.leaks:
                slots[i, 11] = float(r.target_outcome)
        return np.concatenate([legal, slots], axis=1)

    def events(self, rows: list[Row], detector) -> np.ndarray:
        raw = detector(self.assemble(rows))
        return np.roll(raw, self.event_shift) if self.event_shift else raw


class Detector:
    """Untrained random projection by default; `fit` makes it a trained probe."""

    def __init__(self, width: int, seed: int = 909) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(size=(width, 24)).astype(np.float32) / np.sqrt(width)
        self.w2 = rng.normal(size=(24,)).astype(np.float32)
        self.linear = None

    def fit(self, x: np.ndarray, y: np.ndarray, steps: int = 400) -> "Detector":
        design = np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1)
        weights = np.zeros(design.shape[1])
        rate = float(y.mean())
        sample = np.where(y > 0.5, 0.5 / max(rate, 1e-6), 0.5 / max(1 - rate, 1e-6))
        for _ in range(steps):
            probability = 1 / (1 + np.exp(-np.clip(design @ weights, -50, 50)))
            weights -= 2.0 * design.T @ (sample * (probability - y)) / len(design)
        self.linear = weights
        return self

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.linear is not None:
            return np.concatenate([x, np.ones((len(x), 1), np.float32)], 1) @ self.linear
        return np.maximum(x @ self.w1, 0.0) @ self.w2


def perturb(channel: str, row: Row, rng) -> dict[str, Any]:
    if channel == "structured_agent_coordinates":
        return {"agent_cell": (row.agent_cell + 37) % (GRID * GRID)}
    if channel == "structured_switch_neighbour_bits":
        return {"switch_neighbours": 1.0 - row.switch_neighbours}
    if channel == "true_displacement":
        return {"displacement": (row.displacement + 1) % 5}
    if channel == "true_switch_event":
        return {"event": 1.0 - row.event}
    if channel == "true_phase":
        return {"phase": 1 - row.phase}
    if channel == "simulator_step":
        return {"step": row.step + 11}
    if channel == "environment_seed":
        return {"seed": row.seed + 7}
    if channel == "layout_id":
        return {"layout": row.layout + 13}
    if channel == "provenance_digest":
        return {"provenance": float(rng.random())}
    if channel == "future_frame":
        return {"future_frame": np.clip(row.future_frame + 0.25, 0, 1)}
    if channel == "future_action_result":
        return {"future_action_result": 1.0 - row.future_action_result}
    if channel == "target_outcome":
        return {"target_outcome": (row.target_outcome + 1) % 5}
    raise KeyError(channel)


def channel_guard(channel: str):
    def guard(pipeline: Pipeline, rows: list[Row], detector) -> bool:
        rng = np.random.default_rng(17)
        moved = [Row(**{**r.__dict__, **perturb(channel, r, rng)}) for r in rows]
        return bool(np.abs(pipeline.events(rows, detector)
                           - pipeline.events(moved, detector)).max() < 1e-9)
    return guard


def guard_alignment(pipeline: Pipeline, rows: list[Row], detector) -> bool:
    estimate = pipeline.events(rows, detector)
    truth = np.array([r.event for r in rows], dtype=float)
    truth = truth - truth.mean()
    centred = estimate - estimate.mean()
    scores = {lag: float(np.dot(np.roll(centred, lag), truth)) for lag in range(-2, 3)}
    return max(scores, key=lambda k: scores[k]) == 0


def guard_action_alignment(pipeline: Pipeline, rows: list[Row], detector) -> bool:
    return not pipeline.action_misaligned


def guard_belief_isolated(pipeline: Pipeline, rows: list[Row], detector) -> bool:
    return not pipeline.belief_reads_frame


GUARDS = {**{f"no_{c}": channel_guard(c) for c in CHANNELS},
          "event_alignment_peaks_at_lag_zero": guard_alignment,
          "action_matches_scored_transition": guard_action_alignment,
          "belief_isolated_from_frame": guard_belief_isolated}

MUTATIONS = {**{f"leak_{c}": (lambda c=c: Pipeline(leaks=frozenset({c}))) for c in CHANNELS},
             "event_shifted_forward": lambda: Pipeline(event_shift=1),
             "event_shifted_backward": lambda: Pipeline(event_shift=-1),
             "action_misaligned": lambda: Pipeline(action_misaligned=True),
             "belief_reads_frame": lambda: Pipeline(belief_reads_frame=True)}

INTENDED = {**{f"leak_{c}": f"no_{c}" for c in CHANNELS},
            "event_shifted_forward": "event_alignment_peaks_at_lag_zero",
            "event_shifted_backward": "event_alignment_peaks_at_lag_zero",
            "action_misaligned": "action_matches_scored_transition",
            "belief_reads_frame": "belief_isolated_from_frame"}

WIRING = tuple(f"no_{c}" for c in CHANNELS) + (
    "action_matches_scored_transition", "belief_isolated_from_frame")
BEHAVIOURAL = ("event_alignment_peaks_at_lag_zero",)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "n-dataflow.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    episodes = core.collect_visual(core.TRAIN_LAYOUTS[:12], 1, 8, seed=11)
    pairs = core.to_pairs(episodes)
    rows = build_rows(pairs, episodes)
    width = pairs.before[0].size * 2 + 4 + SLOTS
    labels = np.array([r.event for r in rows], dtype=float)
    print(f"{len(rows)} visual prediction rows; {len(GUARDS)} guards x "
          f"{len(MUTATIONS)} planted defects; positive event rate {labels.mean():.4f}\n",
          flush=True)

    honest = Pipeline()
    untrained = Detector(width)
    trained_cache: dict[Any, Detector] = {}

    def trained_for(pipeline: Pipeline) -> Detector:
        key = tuple(sorted(pipeline.leaks))
        if key not in trained_cache:
            trained_cache[key] = Detector(width).fit(pipeline.assemble(rows), labels)
        return trained_cache[key]

    report: dict[str, Any] = {
        "rows": len(rows), "legal_inputs": list(LEGAL),
        "forbidden_channels": list(CHANNELS),
        "legal_order": ["event_hat_t = g(frame_before, action, frame_after)",
                        "belief_t = F(belief_{t-1}, event_hat_t)",
                        "outcome_hat = P(frame_after, belief_t, A_t)"],
        "wiring_guards": list(WIRING), "behavioural_guards": list(BEHAVIOURAL)}

    for label, detector_for, relevant in (
            ("untrained_wiring", lambda p: untrained, WIRING),
            ("trained_behavioural", trained_for, BEHAVIOURAL)):
        honest_results = {name: bool(guard(honest, rows, detector_for(honest)))
                          for name, guard in GUARDS.items()}
        per_defect, broken = {}, []
        print(f"-- {label} --")
        print(f"{'defect':44s} {'honest':>7s} {'catches':>8s}  status")
        for defect, make in MUTATIONS.items():
            pipeline = make()
            detector = detector_for(pipeline)
            fired = {name: bool(guard(pipeline, rows, detector))
                     for name, guard in GUARDS.items()}
            target = INTENDED[defect]
            caught = not fired[target]
            ok = honest_results[target] and caught
            per_defect[defect] = {"guard": target,
                                  "guard_passes_honest": honest_results[target],
                                  "guard_catches_defect": caught,
                                  "other_guards_also_fired": sorted(
                                      k for k, v in fired.items() if not v and k != target)}
            if not ok:
                broken.append(f"{defect} -> {target}")
            if target in relevant:
                print(f"{defect:44s} {str(honest_results[target]):>7s} "
                      f"{str(caught):>8s}  {'ok' if ok else 'BROKEN'}", flush=True)
        report[label] = {"honest": honest_results, "per_defect": per_defect,
                         "broken": broken}
        print()

    wiring_ok = all(report["untrained_wiring"]["per_defect"][d]["guard_passes_honest"]
                    and report["untrained_wiring"]["per_defect"][d]["guard_catches_defect"]
                    for d, g in INTENDED.items() if g in WIRING)
    behavioural_ok = all(
        report["trained_behavioural"]["per_defect"][d]["guard_passes_honest"]
        and report["trained_behavioural"]["per_defect"][d]["guard_catches_defect"]
        for d, g in INTENDED.items() if g in BEHAVIOURAL)
    report["wiring_matrix_clean"] = bool(wiring_ok)
    report["behavioural_matrix_clean"] = bool(behavioural_ok)
    report["n1_every_visual_leak_caught"] = bool(wiring_ok and behavioural_ok)
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"N1 (every visual dataflow leak is caught): "
          f"{report['n1_every_visual_leak_caught']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
