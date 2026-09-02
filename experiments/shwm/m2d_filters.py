"""D / U4. True-event closure: which temporal mechanism wins on the alias population?

The population is the one the gate names and not a general transition set: identical
complete public packet, identical proposed action, different legal history, different
hidden phase, different public next outcome. A model with no temporal state receives
byte-identical input for the two directions of a pair, so it ties at exactly 0.5 there
by construction rather than by estimate. Everything above 0.5 is history.

Every arm sees the same trajectories, the same budget, the same TRUE event channel and
the same padded tensors. The only thing that varies is the mechanism holding the state,
which is what U4 asks about.

    .venv-shwm/bin/python experiments/shwm/m2d_filters.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as core
from m2d_core import (ARTIFACTS, ArmIdentity, FilterSpec, MECHANISM, ALIAS_LAYOUTS,
                      antisymmetric_two_state, build_population, build_tensors,
                      checkpoint_hash, fit_state_assignment, held_out_accuracy,
                      RouteFeatures, score_population, stratify, summarise_metric,
                      train_model, write)
from structured_calibration import collect
from belief_factorization import build_dataset
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

SEEDS = tuple(range(8000, 8020))

INPUTS_FILTER = ("public_row_features", "reset_stripe", "event_channel")
INPUTS_MEMORYLESS = ("public_row_features",)
SUPERVISION = ("displacement class; the TRUE public crossing event is supplied as an "
               "input channel, which is authored auxiliary information")


def arms() -> dict[str, FilterSpec]:
    anti = antisymmetric_two_state()
    return {
        "1_true_event_exact_accumulator": FilterSpec("1", "accumulator"),
        "2_true_event_learned_filter_2state": FilterSpec("2", "filter", 2,
                                                         "symmetry_broken",
                                                         perturbation=anti),
        "3_true_event_learned_filter_8state": FilterSpec("3", "filter", 8, "default"),
        "4_true_event_generic_gru": FilterSpec("4", "gru"),
        "5_trained_memoryless": FilterSpec("5", "memoryless"),
        "6_constant_event_filter": FilterSpec("6", "filter", 2, "symmetry_broken",
                                              perturbation=anti),
    }


def population_label(population) -> str:
    summary = population.summary()
    return (f"exact V2 agent-visible alias pairs, layouts "
            f"{ALIAS_LAYOUTS[0]}-{ALIAS_LAYOUTS[-1]}, depth 6: {summary['pairs']} pairs, "
            f"{summary['rows']} directed rows, {summary['alias_classes']} classes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2d-filters.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    core.check_feature_layout()

    train = build_dataset(collect(list(core.TRAIN_LAYOUTS), 3, 9,
                                  CANONICAL_APPEARANCE_SEED, 11), 5)
    test = build_dataset(collect(list(core.DETECTOR_TEST_LAYOUTS), 2, 9,
                                 CANONICAL_APPEARANCE_SEED, 777), 6)
    population = build_population()
    features = RouteFeatures(population)
    tensors = build_tensors(population, features)
    strata = stratify(population)
    seeds = SEEDS[:arguments.seeds]
    print(f"train {len(train)} trajectories; {population_label(population)}", flush=True)
    print(f"{len(seeds)} seeds; TRUE events everywhere\n", flush=True)

    constant_events = np.zeros_like(tensors.events_true)
    report: dict[str, Any] = {"population": population.summary(),
                              "population_label": population_label(population),
                              "alias_layouts": list(ALIAS_LAYOUTS),
                              "seeds": list(seeds), "arms": {}, "identities": {}}

    collected: dict[str, dict[str, np.ndarray]] = {}
    print(f"{'arm':38s} {'alias acc':>9s} {'p10':>7s} {'min':>7s} {'NLL':>7s} "
          f"{'Brier':>7s} {'margin':>8s} {'phase':>7s} {'collapse':>9s}")
    print("-" * 108)
    for name, spec in arms().items():
        events = constant_events if name.startswith("6_") else None
        per_seed_hits, per_seed_accuracy, records = [], [], []
        for seed in seeds:
            model, count = train_model(spec, train, seed,
                                       event_transform=(lambda e: e * 0.0)
                                       if name.startswith("6_") else None)
            scored = score_population(model, tensors, events)
            assignment = fit_state_assignment(model, train)
            phase = float("nan")
            collapsed = False
            if assignment is not None and scored["belief"].shape[1] > 1:
                predicted = assignment[scored["belief"].argmax(axis=1)]
                phase = float((predicted == np.array(
                    [population.states[r.self_index].polarity
                     for r in population.rows])).mean())
                entropy = -(scored["belief"] * np.log(np.maximum(
                    scored["belief"], 1e-12))).sum(axis=1).mean()
                collapsed = bool(entropy / np.log(scored["belief"].shape[1]) > 0.9)
            per_seed_hits.append(scored["hit"])
            per_seed_accuracy.append(float(scored["hit"].mean()))
            records.append({"seed": seed, "alias_accuracy": float(scored["hit"].mean()),
                            "nll": float(scored["nll"].mean()),
                            "brier": float(scored["brier"].mean()),
                            "margin": float(scored["margin"].mean()),
                            "phase_accuracy_up_to_permutation": phase,
                            "collapsed": collapsed,
                            "held_out_displacement_accuracy":
                                held_out_accuracy(model, test),
                            "parameters": count, "checkpoint": checkpoint_hash(model)})
            if seed == seeds[0]:
                report["identities"][name] = ArmIdentity(
                    arm_id=name,
                    event_source="constant" if name.startswith("6_") else "true",
                    temporal_mechanism=MECHANISM[spec.kind],
                    model_class=type(model).__name__,
                    checkpoint_hash=checkpoint_hash(model),
                    initialization_rule=spec.initialization_rule,
                    trainable_parameters=count,
                    supervision=SUPERVISION,
                    input_fields=(INPUTS_MEMORYLESS if spec.kind == "memoryless"
                                  else INPUTS_FILTER),
                    seed=seed, population=population_label(population),
                    metric="pairwise outcome accuracy on differing successors",
                    query_budget="one forward pass per (state, action); "
                                 "no environment queries at evaluation").to_dict()
        collected[name] = {"hit": np.concatenate(per_seed_hits),
                           "per_seed": np.stack(per_seed_hits)}
        stats = summarise_metric(np.array(per_seed_accuracy))
        stats.update({
            "nll": float(np.mean([r["nll"] for r in records])),
            "brier": float(np.mean([r["brier"] for r in records])),
            "margin": float(np.mean([r["margin"] for r in records])),
            "phase_accuracy_up_to_permutation": float(np.nanmean(
                [r["phase_accuracy_up_to_permutation"] for r in records]))
            if not all(np.isnan(r["phase_accuracy_up_to_permutation"]) for r in records)
            else float("nan"),
            "collapse_rate": float(np.mean([r["collapsed"] for r in records])),
            "held_out_displacement_accuracy": float(np.mean(
                [r["held_out_displacement_accuracy"] for r in records])),
        })
        report["arms"][name] = {"stats": stats, "records": records}
        print(f"{name:38s} {stats['mean']:9.4f} {stats['p10']:7.4f} "
              f"{stats['minimum']:7.4f} {stats['nll']:7.4f} {stats['brier']:7.4f} "
              f"{stats['margin']:8.4f} {stats['phase_accuracy_up_to_permutation']:7.4f} "
              f"{stats['collapse_rate']:9.2f}", flush=True)

    rows = len(population.rows)
    seed_column = np.repeat(np.array(seeds), rows)
    layout_column = np.tile(strata["layout"], len(seeds))
    class_column = np.tile(strata["alias_class"], len(seeds))
    print("\npaired hierarchical intervals (seed -> layout -> alias class)")
    for name in report["arms"]:
        for reference in ("5_trained_memoryless", "1_true_event_exact_accumulator"):
            if name == reference:
                continue
            interval = core.hierarchical_paired_interval(
                collected[name]["hit"], collected[reference]["hit"],
                seed_column, layout_column, class_column)
            report["arms"][name].setdefault("intervals", {})[f"vs_{reference}"] = interval
            if reference == "5_trained_memoryless":
                print(f"  {name:38s} vs memoryless {interval['delta']:+.4f}  "
                      f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
                      f"{'  *' if interval['excludes_zero'] else ''}", flush=True)

    memoryless = report["arms"]["5_trained_memoryless"]["stats"]
    filter_arm = report["arms"]["2_true_event_learned_filter_2state"]
    report["memoryless_is_exactly_chance"] = bool(
        abs(memoryless["mean"] - 0.5) < 1e-9 and memoryless["sd"] < 1e-9)
    report["c3_true_event_filter_beats_memoryless"] = bool(
        filter_arm["stats"]["p10"] > memoryless["mean"]
        and filter_arm["intervals"]["vs_5_trained_memoryless"]["ci_low"] > 0)
    frozen = ARTIFACTS / "m2d-filters-predictions.npz"
    report["frozen_predictions"] = {
        "path": str(frozen.relative_to(core.REPO)),
        "sha256_16": core.save_predictions(
            frozen, {**{f"hit::{k}": v["per_seed"] for k, v in collected.items()},
                     "seeds": np.array(seeds), "row_layout": strata["layout"],
                     "row_alias_class": strata["alias_class"],
                     "row_changes": strata["changes"]})}
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nmemoryless is exactly chance by construction: "
          f"{report['memoryless_is_exactly_chance']}")
    print(f"C3 (true-event learned filter robustly beats memoryless on alias pairs): "
          f"{report['c3_true_event_filter_beats_memoryless']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
