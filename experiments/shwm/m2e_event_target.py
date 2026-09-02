"""6 / 8 / V5 / V6. The event target is public; and how good the detector has to be.

Two separate claims, both of which M2D asserted and neither of which it proved.

First, the event label. `public_event` reads `position` and `switches` off an evaluator
reveal, which is convenient and proves nothing: the question is whether any
evaluator-HIDDEN field is required. The adapter names its hidden set exactly --
`("polarity", "switch_crossings", "step")` -- so the derivation is published here as a
function over observable fields and then tested the only way that means anything: move
each hidden field and require the derived label not to move. A calibration arm that DOES
read a hidden field must be caught, or the test has no power.

Second, the fidelity requirement. Route parity is the product of per-step accuracies, not
their average, so

    P_correct_parity(n) = [1 + (2p - 1)^n] / 2

under independence. That gives a minimum detector accuracy for the coupling to retain any
advantage at all, derived on development and then checked against held-out fidelity.

    .venv-shwm/bin/python experiments/shwm/m2e_event_target.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import m2d_core as m2d
import m2e_core as core
from m2e_core import ARTIFACTS, write
from m2d_coupling import (EventDetector, classification_metrics, collect_goal_directed)
from structured_calibration import collect
from belief_factorization import build_dataset, public_event
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

HIDDEN_FIELDS = ("polarity", "switch_crossings", "step")
OBSERVABLE_FIELDS = ("position", "switches", "walls", "goal", "previous_action", "blocked")


def derive_event_public(previous: dict[str, Any], current: dict[str, Any]) -> int:
    """THE PUBLISHED DERIVATION. C_t = 1 iff the agent moved into a cell that was a
    switch at t-1.

    Reads `position` from both rows and `switches` from the PREVIOUS row only. The
    previous row is where the destination cell shows its own colour: the renderer paints
    the agent over the switch beneath it, so the current frame cannot answer the
    question about the cell the agent now occupies.

    Every field read is outside the adapter's declared hidden set. That is asserted by
    `assert_derivation_ignores_hidden_state`, not by this docstring.
    """
    if previous is None:
        return 0
    moved = current["position"] != previous["position"]
    return int(moved and current["position"] in previous["switches"])


def derive_event_leaky(previous: dict[str, Any], current: dict[str, Any]) -> int:
    """CALIBRATION ARM. Identical, except it consults a hidden field.

    If the invariance test cannot catch this, it cannot certify the honest derivation.
    """
    if previous is None:
        return 0
    moved = current["position"] != previous["position"]
    inferred = current["switch_crossings"] != previous["switch_crossings"]
    return int(moved and inferred)


def assert_derivation_ignores_hidden_state(
        derivation: Callable[..., int], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Move each hidden field, hold every observable field fixed, require no change."""
    rng = np.random.default_rng(17)
    base = [derivation(rows[t - 1] if t else None, rows[t]) for t in range(len(rows))]
    moved: dict[str, bool] = {}
    for field in HIDDEN_FIELDS:
        perturbed = []
        for row in rows:
            copy = dict(row)
            if field == "polarity":
                copy[field] = 1 - int(row.get(field, 0))
            else:
                copy[field] = int(row.get(field, 0)) + int(rng.integers(1, 7))
            perturbed.append(copy)
        after = [derivation(perturbed[t - 1] if t else None, perturbed[t])
                 for t in range(len(perturbed))]
        moved[field] = bool(after != base)
    return {"changed_under": [f for f, m in moved.items() if m],
            "invariant": not any(moved.values())}


def with_crossings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the hidden fields so the leaky arm has something to read."""
    out, crossings = [], 0
    previous = None
    for index, row in enumerate(rows):
        copy = dict(row)
        crossings += public_event(previous, row)
        copy["switch_crossings"] = crossings
        copy["step"] = index
        out.append(copy)
        previous = row
    return out


def detector_nll(truth: np.ndarray, probability: np.ndarray) -> float:
    """Cast to float64 BEFORE clipping.

    `np.clip(x_float32, 1e-9, 1 - 1e-9)` keeps float32, and 1 - 1e-9 rounds to exactly
    1.0 there -- so the upper clip is a no-op and a confident correct prediction yields
    log(0). The development split reported NLL = nan until this was fixed, which is the
    only reason it was noticed: a guard that silently does nothing.
    """
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    t = np.asarray(truth, dtype=np.float64)
    return float(-(t * np.log(p) + (1.0 - t) * np.log(1.0 - p)).mean())


def stratified_report(detector: EventDetector, items) -> dict[str, Any]:
    x, y, e, m, _ = m2d.pad(items)
    probability = detector.probabilities(x)
    valid = m.astype(bool)
    valid[:, 0] = False
    density = x[..., m2d.NEIGHBOUR_SWITCH].sum(axis=-1)
    action = np.argmax(x[..., m2d.PREVIOUS_ACTION], axis=-1) - 1
    crossings = np.cumsum(e, axis=1)
    step = np.tile(np.arange(x.shape[1]), (x.shape[0], 1))

    def block(mask):
        out = classification_metrics(e[mask], probability[mask])
        out["nll"] = detector_nll(e[mask], probability[mask])
        return out

    report = {"overall": block(valid)}
    for a in range(4):
        picked = valid & (action == a)
        if picked.sum() > 20:
            report[f"action_{a}"] = block(picked)
    for d in range(5):
        picked = valid & (density == d)
        if picked.sum() > 20:
            report[f"switch_density_{d}"] = block(picked)
    for label, low, high in (("event_count_0", 0, 1), ("event_count_1", 1, 2),
                             ("event_count_2", 2, 3), ("event_count_3", 3, 4),
                             ("event_count_4plus", 4, 99)):
        picked = valid & (crossings >= low) & (crossings < high)
        if picked.sum() > 20:
            report[label] = block(picked)
    for s in range(1, 9):
        picked = valid & (step == s)
        if picked.sum() > 20:
            report[f"time_since_reset_{s}"] = block(picked)
    return report


def parity_under_independence(p: float, n: int) -> float:
    return float((1.0 + (2.0 * p - 1.0) ** n) / 2.0)


def required_detector_accuracy(exponents: np.ndarray, target_parity: float) -> float:
    """Smallest per-step p whose expected route parity reaches `target_parity`.

    `exponents` must be the number of independent ERROR OPPORTUNITIES per route, which
    is the route length, not the number of true events. The specification writes the
    diagnostic with n = phase-changing events, and that form is reported too, but it
    over-predicts measured parity by 0.20 or more on every population: a detector can
    invert a route's parity with a false positive on a step where nothing happened, and
    the n-based form cannot see those. A route with n = 0 is not automatically correct.
    """
    weights = np.bincount(exponents) / len(exponents)
    grid = np.linspace(0.5, 1.0, 5001)
    for p in grid:
        expected = sum(w * parity_under_independence(p, n)
                       for n, w in enumerate(weights) if w > 0)
        if expected >= target_parity:
            return float(p)
    return 1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2e-event-target.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    m2d.check_feature_layout()
    appearance = CANONICAL_APPEARANCE_SEED

    train_t = collect(list(m2d.TRAIN_LAYOUTS), 3, 9, appearance, 11)
    validation_t = collect(list(m2d.DETECTOR_TEST_LAYOUTS), 2, 9, appearance, 777)
    held_out_t = collect(list(m2d.HELD_OUT_LAYOUTS), 2, 9, appearance, 313)
    policy_t = collect_goal_directed(list(m2d.HELD_OUT_LAYOUTS), 2, 9, appearance, 404)

    report: dict[str, Any] = {"hidden_fields": list(HIDDEN_FIELDS),
                              "observable_fields_read": ["position", "switches"]}

    # ---- the derivation is public ---------------------------------------------------------
    print("event-target derivation: exactness and hidden-field invariance")
    exact = leaky_caught = 0
    total = 0
    honest_invariance, leaky_invariance = [], []
    for trajectory in train_t + validation_t + held_out_t + policy_t:
        rows = with_crossings(trajectory["rows"])
        derived = [derive_event_public(rows[t - 1] if t else None, rows[t])
                   for t in range(len(rows))]
        label = [public_event(rows[t - 1] if t else None, rows[t])
                 for t in range(len(rows))]
        exact += int(derived == label)
        total += 1
        honest_invariance.append(
            assert_derivation_ignores_hidden_state(derive_event_public, rows))
        leaky_invariance.append(
            assert_derivation_ignores_hidden_state(derive_event_leaky, rows))
    leaky_caught = sum(1 for r in leaky_invariance if not r["invariant"])
    report["derivation"] = {
        "trajectories": total,
        "exactly_reproduces_label": exact,
        "honest_invariant_trajectories": sum(1 for r in honest_invariance
                                             if r["invariant"]),
        "leaky_calibration_caught": leaky_caught,
        "guard_has_power": bool(leaky_caught > 0),
        "public_derivation": "C_t = 1 iff position_t != position_{t-1} and "
                             "position_t in switches_{t-1}"}
    print(f"  exact on {exact}/{total} trajectories")
    print(f"  invariant to every hidden field on "
          f"{report['derivation']['honest_invariant_trajectories']}/{total}")
    print(f"  leaky calibration arm caught on {leaky_caught}/{total} "
          f"(guard has power: {report['derivation']['guard_has_power']})")
    report["v5_event_target_is_public"] = bool(
        exact == total
        and report["derivation"]["honest_invariant_trajectories"] == total
        and leaky_caught > 0)

    # ---- detector generalisation ----------------------------------------------------------
    train = build_dataset(train_t, 5)
    splits = {"development_layouts": train,
              "validation_layouts": build_dataset(validation_t, 6),
              "held_out_layouts": build_dataset(held_out_t, 7),
              "held_out_visitation_policy": build_dataset(policy_t, 8)}
    detector = EventDetector("full").fit(train)
    print("\ndetector p(C_t | X_{t-1}, A_{t-1}, X_t) -- RETROSPECTIVE")
    detection = {}
    for name, items in splits.items():
        detection[name] = stratified_report(detector, items)
        o = detection[name]["overall"]
        print(f"  {name:30s} bal {o['balanced_accuracy']:.4f}  F1 {o['f1']:.4f}  "
              f"Brier {o['brier']:.4f}  NLL {o['nll']:.4f}  ECE "
              f"{o['expected_calibration_error']:.4f}", flush=True)
    detection["scope"] = EventDetector.RETROSPECTIVE
    detection["held_out_dynamics"] = (
        "NOT INSTANTIABLE in v2: SWITCH_COUNT is a constant and the polarity flip rule "
        "never varies, so there is one transition function. What is held out instead is "
        "the visitation policy, reported above and labelled as such.")
    report["detector"] = detection
    report["v6_detector_generalises"] = bool(
        detection["held_out_layouts"]["overall"]["balanced_accuracy"] > 0.6
        and detection["held_out_visitation_policy"]["overall"]["balanced_accuracy"] > 0.6)

    # ---- fidelity requirement -------------------------------------------------------------
    print("\nevent fidelity: parity is the PRODUCT of per-step accuracies")
    fidelity: dict[str, Any] = {}
    for label, layouts in (("development", tuple(range(91_000, 91_010))),
                           ("validation", m2d.ALIAS_LAYOUTS),
                           ("held_out", tuple(range(95_000, 95_010))),
                           ("held_out_2", tuple(range(92_000, 92_010)))):
        population = m2d.build_population(layouts)
        features = m2d.RouteFeatures(population)
        tensors = m2d.build_tensors(population, features)
        hard = (detector.probabilities(tensors.z) >= 0.5).astype(np.float32)
        valid = np.zeros_like(hard, dtype=bool)
        for k, n in enumerate(tensors.lengths):
            valid[k, 1:n] = True
        per_step = float((hard[valid] == tensors.events_true[valid]).mean())
        estimated = np.array([hard[k, 1:tensors.lengths[k]].sum() % 2
                              for k in range(len(tensors.lengths))])
        truth = np.array([tensors.events_true[k, 1:tensors.lengths[k]].sum() % 2
                          for k in range(len(tensors.lengths))])
        counts = np.array([int(tensors.events_true[k, 1:tensors.lengths[k]].sum())
                           for k in range(len(tensors.lengths))])
        opportunities = np.array([int(tensors.lengths[k] - 1)
                                  for k in range(len(tensors.lengths))])
        by_count = {}
        for n in range(0, 5):
            mask = (counts >= n) & (counts < (n + 1 if n < 4 else 99))
            if mask.sum() > 20:
                by_count[f"n_{n}" if n < 4 else "n_4plus"] = {
                    "keys": int(mask.sum()),
                    "measured_parity_accuracy": float((estimated[mask]
                                                       == truth[mask]).mean()),
                    "prediction_from_event_count": parity_under_independence(per_step, n),
                    "prediction_from_route_length": float(np.mean(
                        [parity_under_independence(per_step, int(l))
                         for l in opportunities[mask]])),
                    "mean_route_length": float(opportunities[mask].mean())}
        fidelity[label] = {"per_step_accuracy": per_step,
                           "final_parity_accuracy": float((estimated == truth).mean()),
                           "prediction_from_event_count": float(np.mean(
                               [parity_under_independence(per_step, int(n))
                                for n in counts])),
                           "prediction_from_route_length": float(np.mean(
                               [parity_under_independence(per_step, int(l))
                                for l in opportunities])),
                           "by_event_count": by_count,
                           "mean_route_length": float(opportunities.mean()),
                           "event_count_distribution": np.bincount(
                               counts, minlength=8)[:8].tolist()}
        print(f"  {label:14s} per-step {per_step:.4f}  parity "
              f"{fidelity[label]['final_parity_accuracy']:.4f}  "
              f"n-based predicts {fidelity[label]['prediction_from_event_count']:.4f}  "
              f"length-based predicts "
              f"{fidelity[label]['prediction_from_route_length']:.4f}", flush=True)
        fidelity[label]["_exponents"] = opportunities

    # The smallest advantage seen with an interval excluding zero in M2D was +0.0921 on
    # held_out_2; alias accuracy tracks parity to within 0.01, so a target parity of
    # 0.55 is the development-frozen requirement.
    target_parity = 0.55
    exponents = fidelity["development"].pop("_exponents")
    minimum = required_detector_accuracy(exponents, target_parity)
    n_based = required_detector_accuracy(
        np.array([int(n) for n in range(1)] * 0 + list(
            np.repeat(np.arange(len(fidelity["development"]["event_count_distribution"])),
                      fidelity["development"]["event_count_distribution"]))),
        target_parity)
    errors = {k: abs(v["prediction_from_event_count"] - v["final_parity_accuracy"])
              for k, v in fidelity.items() if "final_parity_accuracy" in v}
    length_errors = {k: abs(v["prediction_from_route_length"] - v["final_parity_accuracy"])
                     for k, v in fidelity.items() if "final_parity_accuracy" in v}
    for label in fidelity:
        fidelity[label].pop("_exponents", None)
    report["fidelity"] = fidelity
    report["fidelity_requirement"] = {
        "target_parity_accuracy": target_parity,
        "derived_on": "development alias population route-length distribution",
        "minimum_per_step_accuracy": minimum,
        "minimum_per_step_accuracy_n_based_form": n_based,
        "mean_absolute_error_event_count_form": float(np.mean(list(errors.values()))),
        "mean_absolute_error_route_length_form": float(np.mean(list(length_errors.values()))),
        "diagnostic_form_supported_by_measurement":
            "route_length" if np.mean(list(length_errors.values()))
            < np.mean(list(errors.values())) else "event_count",
        "rationale": "alias-pair accuracy tracks final parity to within 0.01, so a "
                     "parity of 0.55 is the smallest that reproduced an interval "
                     "excluding zero in M2D"}
    below = {k: v["per_step_accuracy"] for k, v in fidelity.items()
             if v["per_step_accuracy"] < minimum}
    report["populations_below_requirement"] = below
    report["event_extraction_bottleneck"] = bool(below)
    print(f"\nrequired per-step accuracy for parity >= {target_parity}: {minimum:.4f}")
    print(f"populations below it: {below}")
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"V5 (event target is public and leak-free): {report['v5_event_target_is_public']}")
    print(f"V6 (detector generalises beyond training layouts): "
          f"{report['v6_detector_generalises']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
