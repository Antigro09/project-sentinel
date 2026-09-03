"""H. What is the authored initial-state gauge actually worth, and is it public?

The eligible filters map the rendered reset stripe onto a one-hot initial belief. That
stripe IS the initial polarity, drawn by the renderer on the reset frame, so the gauge
uses public information -- but "public" and "free" are different claims and M2E showed
the second one is false: replacing it with a learned encoder cost 0.0750 with an interval
excluding zero.

Six variants, kept separate. The phase-supervised arm is included precisely because it
should be INDISTINGUISHABLE from the authored one: if a gauge built from the rendered
stripe and a gauge built from the evaluator's polarity behave identically, that is the
demonstration that the stripe carries the initial phase exactly, and the authored gauge
is a rendering fact rather than a leak. Evaluator phase is never counted as public
grounding; it appears here only to establish that equivalence.

    .venv-shwm/bin/python experiments/shwm/m2f_gauge.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as m2d
import m2e_core as m2e
import m2f_core as core
from m2d_core import ARTIFACTS, FilterSpec, RESET_FLAG, RESET_VALUE, write
from structured_calibration import collect
from belief_factorization import build_dataset
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

SEEDS = core.VALIDATION_SEEDS[:20]
RESTARTS = 8

VARIANTS = {
    "1_authored_public_onehot": {"gauge": "reset_onehot", "reset": "identity",
                                 "eligible": True},
    "2_learned_from_outcome_only": {"gauge": "learned", "reset": "identity",
                                    "eligible": True},
    "3_phase_supervised_oracle": {"gauge": "reset_onehot", "reset": "oracle",
                                  "eligible": False},
    "4_reset_stripe_masked": {"gauge": "reset_onehot", "reset": "masked",
                              "eligible": True},
    "5_false_reset_stripe": {"gauge": "reset_onehot", "reset": "false",
                             "eligible": True},
    "6_random_gauge": {"gauge": "reset_onehot", "reset": "random", "eligible": True},
}


def transform_dataset(items, mode: str, seed: int):
    """Rewrite the reset stripe in the structured features.

    `oracle` writes the evaluator's initial polarity, which for this environment is the
    same number the renderer already drew -- that identity is the thing being tested.
    """
    if mode == "identity":
        return items
    rng = np.random.default_rng(seed)
    out = []
    for item in items:
        copy = dict(item)
        x = np.array(item["x"], copy=True)
        if mode == "masked":
            x[0, RESET_VALUE] = 0.0
            x[0, RESET_FLAG] = 0.0
        elif mode == "false":
            x[0, RESET_VALUE] = 1.0 - x[0, RESET_VALUE]
        elif mode == "random":
            x[0, RESET_VALUE] = float(rng.integers(0, 2))
        elif mode == "oracle":
            x[0, RESET_VALUE] = float(item["phases"][0])
        copy["x"] = x
        out.append(copy)
    return out


def transform_tensors(tensors, mode: str, seed: int):
    if mode == "identity":
        return tensors
    rng = np.random.default_rng(seed + 1)
    z = np.array(tensors.z, copy=True)
    if mode == "masked":
        z[:, 0, RESET_VALUE] = 0.0
        z[:, 0, RESET_FLAG] = 0.0
    elif mode == "false":
        z[:, 0, RESET_VALUE] = 1.0 - z[:, 0, RESET_VALUE]
    elif mode == "random":
        z[:, 0, RESET_VALUE] = rng.integers(0, 2, len(z)).astype(np.float32)
    # `oracle` is a no-op on the tensors: the alias routes already carry the rendered
    # stripe, and it equals the initial polarity. That is the point of the arm.
    replaced = m2d.AliasTensors(
        keys=tensors.keys, z=z, reset=z[:, :1, RESET_VALUE].copy(),
        final=tensors.final, lengths=tensors.lengths,
        events_true=tensors.events_true, row_key=tensors.row_key,
        target=tensors.target, other=tensors.other)
    return replaced


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--restarts", type=int, default=RESTARTS)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2f-gauge.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    train = build_dataset(collect(list(m2d.TRAIN_LAYOUTS), 3, 9,
                                  CANONICAL_APPEARANCE_SEED, 11), 5)
    population = m2d.build_population(core.VALIDATION_ALIAS)
    tensors = m2d.build_tensors(population, m2d.RouteFeatures(population))
    seeds = SEEDS[:arguments.seeds]
    strata = m2d.stratify(population)
    print(f"{len(seeds)} seeds x {arguments.restarts} generic restarts per variant; "
          f"true events; alias layouts {core.VALIDATION_ALIAS[0]}-"
          f"{core.VALIDATION_ALIAS[-1]}\n", flush=True)

    report: dict[str, Any] = {"seeds": list(seeds), "restarts": arguments.restarts,
                              "variants": {}}
    hits: dict[str, np.ndarray] = {}
    print(f"{'variant':32s} {'alias':>8s} {'p10':>8s} {'min':>8s} {'phase':>8s} "
          f"{'solved':>8s} elig")
    print("-" * 84)
    for name, setting in VARIANTS.items():
        items = transform_dataset(train, setting["reset"], 7)
        block = transform_tensors(tensors, setting["reset"], 7)
        per_seed, accuracy, phases, solved = [], [], [], 0
        for seed in seeds:
            candidates = []
            for restart in range(arguments.restarts):
                row, scored = core.run_restart(
                    items, block, population, seed, restart,
                    spec_for=lambda s, r: FilterSpec(
                        "gauge", "filter", 2, "generic", gauge=setting["gauge"],
                        perturbation=m2e.generic_antisymmetric(s * 1_000 + r, 2)))
                candidates.append((row, scored))
            row, scored = max(candidates, key=lambda c: c[0].training_log_likelihood)
            per_seed.append(scored["primary"])
            accuracy.append(row.alias_accuracy)
            phases.append(row.phase_accuracy)
            solved += int(row.alias_accuracy > 0.9)
        hits[name] = np.stack(per_seed)
        report["variants"][name] = {
            "eligible": setting["eligible"], "gauge": setting["gauge"],
            "reset_transform": setting["reset"],
            "mean": float(np.mean(accuracy)), "p10": float(np.percentile(accuracy, 10)),
            "minimum": float(np.min(accuracy)),
            "phase_accuracy": float(np.nanmean(phases)),
            "solved_seeds": solved, "per_seed": [float(a) for a in accuracy]}
        print(f"{name:32s} {np.mean(accuracy):8.4f} "
              f"{np.percentile(accuracy, 10):8.4f} {np.min(accuracy):8.4f} "
              f"{np.nanmean(phases):8.4f} {solved:>3d}/{len(seeds):<4d} "
              f"{'yes' if setting['eligible'] else 'NO'}", flush=True)

    rows = len(population.rows)
    seed_column = np.repeat(np.array(seeds), rows)
    layout_column = np.tile(strata["layout"], len(seeds))
    class_column = np.tile(strata["alias_class"], len(seeds))
    authored = hits["1_authored_public_onehot"].ravel()
    print("\npaired hierarchical intervals against the authored public gauge")
    for name in VARIANTS:
        if name == "1_authored_public_onehot":
            continue
        interval = m2d.hierarchical_paired_interval(
            hits[name].ravel(), authored, seed_column, layout_column, class_column)
        report["variants"][name]["vs_authored"] = interval
        print(f"  {name:32s} {interval['delta']:+.4f} "
              f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
              f"{'  DIFFERS' if interval['excludes_zero'] else ''}", flush=True)

    oracle = report["variants"]["3_phase_supervised_oracle"]["vs_authored"]
    learned = report["variants"]["2_learned_from_outcome_only"]["vs_authored"]
    report["stripe_equals_initial_polarity"] = bool(not oracle["excludes_zero"])
    report["learned_gauge_matches_authored"] = bool(not learned["excludes_zero"])
    report["result_is_conditional_on_authored_grounding"] = bool(
        not report["learned_gauge_matches_authored"])
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nrendered stripe is indistinguishable from evaluator initial polarity: "
          f"{report['stripe_equals_initial_polarity']}")
    print(f"outcome-trained learned gauge matches the authored one: "
          f"{report['learned_gauge_matches_authored']}")
    print(f"=> transition result must be reported as CONDITIONAL ON AUTHORED "
          f"INITIAL-STATE GROUNDING: "
          f"{report['result_is_conditional_on_authored_grounding']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
