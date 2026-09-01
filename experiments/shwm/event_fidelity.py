"""J5/§6. How accurate must event detection be for phase accumulation to work?

Phase is the parity of accumulated crossings, so an event detector with accuracy
p that makes independent errors gives

    P(correct parity | n events) = (1 + (2p - 1)^n) / 2

which decays geometrically in n. Averaging over the empirical distribution of n
turns that into a required event accuracy for any target phase accuracy.

The independence assumption is doing real work here and is probably optimistic --
detector errors are correlated with layout and with how ambiguous a crossing
looks -- so the derived number is a *lower bound* on what is needed. It is
reported as a design target, not as a gate, and the end-to-end recurrent result
is what actually decides J5. The previous audit's R^2 >= 0.99 heuristic is
retired: it was a prerequisite for one particular differencing readout, not a
property of the task.

    .venv/bin/python experiments/shwm/event_fidelity.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import ACTIONS, ProceduralVisualV2Adapter  # noqa: E402
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED  # noqa: E402

TARGET_PHASE_ACCURACY = 0.90
"""Pre-registered. The phase accuracy a qualified chain must reach.

Chosen before any event-detector number was measured: it is the level at which a
belief over phase is usable by a planner that can also observe the outcome of its
own next action, and it sits well clear of the ~0.55 majority baseline."""


def crossing_distribution(layouts, steps: int, appearance: int, seed: int) -> np.ndarray:
    gate = AuthorityGate(gate_id="fidelity")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    generator = np.random.default_rng(seed)
    counts = []
    for layout in layouts:
        adapter.reset(layout, f"appearance:{appearance}")
        for step in range(steps):
            counts.append(int(adapter.snapshot().reveal("evaluator")["switch_crossings"]))
            action = int(generator.integers(0, len(ACTIONS)))
            if adapter.step(action, gate.authorize_evaluator(action, "roll")).terminated:
                break
    return np.array(counts)


def phase_accuracy_given(p: float, counts: np.ndarray) -> float:
    """Expected parity accuracy under the independent-error model."""
    n = counts.astype(float)
    return float(np.mean((1.0 + (2.0 * p - 1.0) ** n) / 2.0))


def required_event_accuracy(counts: np.ndarray, target: float) -> float:
    low, high = 0.5, 1.0
    for _ in range(80):
        mid = (low + high) / 2
        if phase_accuracy_given(mid, counts) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def simulate(p: float, counts: np.ndarray, seed: int = 3) -> float:
    """Check the closed form against an actual accumulation, same error rate."""
    generator = np.random.default_rng(seed)
    correct = 0
    for n in counts:
        flips = generator.random(int(n)) > p
        correct += int(flips.sum() % 2 == 0)
    return correct / max(len(counts), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", type=int, default=200)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--measured", type=Path,
                        default=REPO / "artifacts/shwm/scale1/readout-qualification.json")
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/event-fidelity.json")
    arguments = parser.parse_args()

    counts = crossing_distribution(
        list(range(81_000, 81_000 + arguments.layouts)), arguments.steps,
        CANONICAL_APPEARANCE_SEED, 4242)
    histogram = {int(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))}
    required = required_event_accuracy(counts, TARGET_PHASE_ACCURACY)

    print(f"crossing counts over {len(counts)} validation states: mean {counts.mean():.3f}, "
          f"max {counts.max()}")
    print(f"  histogram {histogram}")
    print(f"\ntarget phase accuracy (pre-registered): {TARGET_PHASE_ACCURACY}")
    print(f"required event accuracy under independent errors: {required:.4f}\n")
    print(f"{'event acc p':>12s} | {'closed form':>12s} | {'simulated':>10s}")
    print("-" * 40)
    curve = {}
    for p in (0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 1.0):
        closed = phase_accuracy_given(p, counts)
        curve[p] = closed
        print(f"{p:12.3f} | {closed:12.4f} | {simulate(p, counts):10.4f}")

    measured: dict[str, Any] = {}
    if arguments.measured.exists():
        data = json.loads(arguments.measured.read_text())
        relation = data.get("events", {}).get("relation_decoder", {})
        for name, record in relation.items():
            accuracy = record.get("event_accuracy")
            if accuracy is None:
                continue
            measured[name] = {
                "measured_event_accuracy": accuracy,
                "measured_balanced_accuracy": record.get("event_balanced_accuracy"),
                "implied_phase_accuracy": phase_accuracy_given(accuracy, counts),
                "meets_target": phase_accuracy_given(accuracy, counts) >= TARGET_PHASE_ACCURACY,
                "shortfall_in_event_accuracy": round(required - accuracy, 4),
            }
        if measured:
            print("\nmeasured detectors:")
            for name, record in measured.items():
                print(f"  {name:18s} event acc {record['measured_event_accuracy']:.4f} -> "
                      f"implied phase {record['implied_phase_accuracy']:.4f}  "
                      f"(target {TARGET_PHASE_ACCURACY}: "
                      f"{'MET' if record['meets_target'] else 'short by '
                         + format(record['shortfall_in_event_accuracy'], '.4f')})")

    report = {
        "target_phase_accuracy": TARGET_PHASE_ACCURACY,
        "required_event_accuracy_independent_model": required,
        "crossing_count_histogram": histogram,
        "mean_crossings": float(counts.mean()),
        "curve_closed_form": {str(k): v for k, v in curve.items()},
        "measured": measured,
        "caveat": (
            "the independent-error model is optimistic; it is a design target and a "
            "lower bound, and J5 is decided by the end-to-end recurrent result"
        ),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nwrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
