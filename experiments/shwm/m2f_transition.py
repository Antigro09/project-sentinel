"""D / E / F / G. A hundred fresh seeds, thirty-two restarts each, one table.

Every procedure in section E is a prefix operation over the same restart table, so
fixed K=8/16/32 and the adaptive rule are compared on IDENTICAL restarts rather than on
independently drawn ones. That is not only cheaper; it removes the possibility that a
procedure looks better because its draws happened to be luckier.

Only the prefix-argmax models can ever be selected, so their alias hits are snapshotted
at each block boundary (8, 16, 24, 32) instead of storing every restart's hits, which
would have been a quarter of a gigabyte per population for no gain.

    .venv-shwm/bin/python experiments/shwm/m2f_transition.py --split development
    .venv-shwm/bin/python experiments/shwm/m2f_transition.py --split validation \
        --long-updates <U> --gru-restarts <K>
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
from m2d_core import ARTIFACTS, FilterSpec, write
from m2e_core import population_manifest, episode_manifest
from structured_calibration import collect
from belief_factorization import build_dataset
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

BOUNDARIES = (8, 16, 24, 32)


def baseline_specs() -> dict[str, Any]:
    return {
        "7_trained_memoryless": lambda s, r: FilterSpec("m2f", "memoryless"),
        "8_exact_accumulator": lambda s, r: FilterSpec("m2f", "accumulator"),
        "9_answer_oriented_ineligible": lambda s, r: FilterSpec(
            "m2f", "filter", 2, "symmetry_broken", perturbation=m2e.ANSWER),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "validation"), required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--k-max", type=int, default=core.K_MAX)
    parser.add_argument("--long-updates", type=int, default=0)
    parser.add_argument("--gru-restarts", type=int, default=0)
    parser.add_argument("--no-extra-populations", action="store_true",
                        help="skip held-out alias scoring in this pass; "
                             "the pathway module scores the SELECTED "
                             "models there, so paying for it on every "
                             "restart buys nothing")
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()
    out = arguments.out or ARTIFACTS / f"m2f-restarts-{arguments.split}.json"
    started = time.perf_counter()
    m2d.check_feature_layout()

    train_t = collect(list(m2d.TRAIN_LAYOUTS), 3, 9, CANONICAL_APPEARANCE_SEED, 11)
    train = build_dataset(train_t, 5)

    if arguments.split == "development":
        seeds = core.DEV_SEEDS[:arguments.seeds]
        primary_layouts = core.DEV_ALIAS
        extra_layouts: dict[str, tuple] = {}
    else:
        seeds = core.VALIDATION_SEEDS[:arguments.seeds]
        primary_layouts = core.VALIDATION_ALIAS
        extra_layouts = ({} if arguments.no_extra_populations
                         else {"held_out": core.HELD_OUT_ALIAS,
                               "held_out_2": core.HELD_OUT_ALIAS_2})

    population = m2d.build_population(primary_layouts)
    tensors = m2d.build_tensors(population, m2d.RouteFeatures(population))
    strata = m2d.stratify(population)
    extra = {}
    manifests = {"primary": population_manifest(population, arguments.split,
                                                primary_layouts)}
    for label, layouts in extra_layouts.items():
        other = m2d.build_population(layouts)
        extra[label] = (m2d.build_tensors(other, m2d.RouteFeatures(other)), other)
        manifests[label] = population_manifest(other, label, layouts)

    print(f"{arguments.split}: {len(seeds)} seeds ({seeds[0]}-{seeds[-1]}) x "
          f"{arguments.k_max} restarts; primary alias layouts "
          f"{primary_layouts[0]}-{primary_layouts[-1]} "
          f"({manifests['primary']['rows']} rows, digest "
          f"{manifests['primary']['member_digest']})", flush=True)

    rows: list[core.RestartRow] = []
    snapshots: dict[str, np.ndarray] = {}
    for label in ["primary"] + list(extra):
        snapshots[label] = np.zeros((len(seeds), len(BOUNDARIES),
                                     len(population.rows) if label == "primary"
                                     else len(extra[label][1].rows)), dtype=np.float32)
    for index, seed in enumerate(seeds):
        best: core.RestartRow | None = None
        best_hits: dict[str, np.ndarray] = {}
        boundary = 0
        for restart in range(arguments.k_max):
            row, hits = core.run_restart(train, tensors, population, seed, restart,
                                         extra=extra)
            rows.append(row)
            if best is None or row.training_log_likelihood > best.training_log_likelihood:
                best, best_hits = row, hits
            if restart + 1 == BOUNDARIES[boundary]:
                for label in snapshots:
                    key = "primary" if label == "primary" else label
                    snapshots[label][index, boundary] = best_hits[key]
                boundary += 1
        if (index + 1) % 10 == 0:
            solved = sum(1 for r in rows if r.seed == seed and r.alias_accuracy > 0.9)
            print(f"  seed {seed} ({index + 1}/{len(seeds)}): best alias "
                  f"{best.alias_accuracy:.4f}, train LL "
                  f"{best.training_log_likelihood:.5f}, {solved}/{arguments.k_max} "
                  f"restarts solved  [{(time.perf_counter() - started) / 60:.1f} min]",
                  flush=True)

    report: dict[str, Any] = {
        "split": arguments.split, "seeds": list(seeds), "k_max": arguments.k_max,
        "boundaries": list(BOUNDARIES),
        "manifests": {k: {kk: vv for kk, vv in v.items()
                          if kk not in ("member_table", "member_routes")}
                      for k, v in manifests.items()},
        "episode_manifest": {k: v for k, v in episode_manifest(
            train, "train", m2d.TRAIN_LAYOUTS, 11).items() if k != "episode_table"},
        "restart_table": [r.to_dict() for r in rows],
        "restart_table_digest": core.digest_rows(rows),
    }

    # Baselines and equal-compute arms: one run per seed each.
    baselines: dict[str, np.ndarray] = {}
    print("\nbaselines, one run per seed")
    for name, spec_for in baseline_specs().items():
        hits, accuracy = [], []
        for seed in seeds:
            row, block = core.run_restart(train, tensors, population, seed, 0,
                                          spec_for=spec_for, extra=extra)
            hits.append(block["primary"])
            accuracy.append(row.alias_accuracy)
            for label in extra:
                baselines.setdefault(f"{name}::{label}", []).append(block[label])
        baselines[name] = np.stack(hits)
        report.setdefault("baselines", {})[name] = {
            "mean": float(np.mean(accuracy)),
            "p10": float(np.percentile(accuracy, 10)),
            "minimum": float(np.min(accuracy))}
        print(f"  {name:32s} {np.mean(accuracy):.4f} (p10 "
              f"{np.percentile(accuracy, 10):.4f})", flush=True)

    if arguments.long_updates:
        saved = m2d.UPDATES
        m2d.UPDATES = arguments.long_updates
        try:
            accuracy, hits = [], []
            for seed in seeds:
                row, block = core.run_restart(train, tensors, population, seed, 0,
                                              extra=extra)
                accuracy.append(row.alias_accuracy)
                hits.append(block["primary"])
                for label in extra:
                    baselines.setdefault(f"5_single_long_run::{label}", []).append(
                        block[label])
        finally:
            m2d.UPDATES = saved
        baselines["5_single_long_run"] = np.stack(hits)
        report.setdefault("baselines", {})["5_single_long_run"] = {
            "mean": float(np.mean(accuracy)),
            "p10": float(np.percentile(accuracy, 10)),
            "updates": arguments.long_updates}
        print(f"  {'5_single_long_run':32s} {np.mean(accuracy):.4f} "
              f"({arguments.long_updates} updates)", flush=True)

    if arguments.gru_restarts:
        accuracy, hits = [], []
        for seed in seeds:
            candidates = []
            for restart in range(arguments.gru_restarts):
                row, block = core.run_restart(
                    train, tensors, population, seed, restart,
                    spec_for=lambda s, r: FilterSpec("m2f", "gru"), extra=extra)
                candidates.append((row, block))
            row, block = max(candidates, key=lambda c: c[0].training_log_likelihood)
            accuracy.append(row.alias_accuracy)
            hits.append(block["primary"])
            for label in extra:
                baselines.setdefault(f"6_gru_multistart::{label}", []).append(block[label])
        baselines["6_gru_multistart"] = np.stack(hits)
        report.setdefault("baselines", {})["6_gru_multistart"] = {
            "mean": float(np.mean(accuracy)),
            "p10": float(np.percentile(accuracy, 10)),
            "restarts": arguments.gru_restarts}
        print(f"  {'6_gru_multistart':32s} {np.mean(accuracy):.4f} "
              f"({arguments.gru_restarts} restarts)", flush=True)

    payload = {f"snapshot::{k}": v for k, v in snapshots.items()}
    for name, value in baselines.items():
        payload[f"baseline::{name}"] = (np.stack(value) if isinstance(value, list)
                                        else value)
    payload["seeds"] = np.array(seeds)
    payload["boundaries"] = np.array(BOUNDARIES)
    payload["row_layout"] = strata["layout"]
    payload["row_alias_class"] = strata["alias_class"]
    payload["row_changes"] = strata["changes"]
    payload["member_table"] = manifests["primary"]["member_table"]
    for label in extra:
        other_strata = m2d.stratify(extra[label][1])
        payload[f"row_layout::{label}"] = other_strata["layout"]
        payload[f"row_alias_class::{label}"] = other_strata["alias_class"]
        payload[f"row_changes::{label}"] = other_strata["changes"]
        payload[f"member_table::{label}"] = manifests[label]["member_table"]
    frozen = ARTIFACTS / f"m2f-restarts-{arguments.split}.npz"
    report["frozen_predictions"] = {
        "path": str(frozen.relative_to(m2d.REPO)),
        "sha256_16": m2d.save_predictions(frozen, payload)}
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(out, report)
    print(f"\nwrote {out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
