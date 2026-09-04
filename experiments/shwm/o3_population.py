"""C / F / J. Fresh palette populations, with the palette as the randomization unit.

O2 resampled over seed, layout and alias class and never over palette, then reported a
per-palette split it could not have detected statistically. This builds the populations
that make palette-level inference possible: 64 development, 64 validation and 64
replication palettes on disjoint seed ranges, each drawing its OWN calibration layouts,
transfer layouts and alias routes from large pools.

One consequence of R1 has to be stated rather than hidden. The corrected pipeline is
bit-exact palette-equivariant, so on a FIXED semantic population every palette returns an
identical number and between-palette variance is exactly zero by construction. The
variance this module measures therefore comes from the CONTENT each palette happens to
draw, not from the colour convention -- and that is the honest reading of every
palette-level interval below.

    .venv-shwm/bin/python experiments/shwm/o3_population.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as m2d
import m2f_core as m2f
import n_pathway as npath
import o2_core as C
import o2_memory as mem
import o2_models as M
import o3_core as O3
import o2_route as o2route
import o3_route_orbit as ro
from m2d_core import ARTIFACTS, FilterSpec, write

SEED = 53_000
VIEW = "no_rgb"                      # the equivariant view; R1 is why
DEV_PALETTES = tuple(range(20_000, 20_064))
VALIDATION_PALETTES = tuple(range(21_000, 21_064))
REPLICATION_PALETTES = tuple(range(22_000, 22_064))
CAL_POOL = tuple(range(118_000, 118_200))
TRANSFER_POOL = tuple(range(119_000, 119_400))
ALIAS_POOL = m2f.HELD_OUT_ALIAS
ROUTE_PARITY_GATE = 0.75


def palette_plan(palette: int, calibration: int, transfer: int, alias: int) -> dict:
    """Each palette draws its own content. Deterministic in the palette id."""
    rng = np.random.default_rng(palette)
    return {
        "palette": palette,
        "bijection": C.sample_bijection(palette),
        "calibration_layouts": [int(v) for v in
                                rng.choice(CAL_POOL, calibration, replace=False)],
        "transfer_layouts": [int(v) for v in
                             rng.choice(TRANSFER_POOL, transfer, replace=False)],
        "alias_layouts": [int(v) for v in rng.choice(ALIAS_POOL, alias, replace=False)],
        "action_seed": int(palette),
    }


def palette_scenario(plan: dict, stratum: str = "COUNT_COLLISION") -> O3.Scenario:
    return O3.Scenario(
        label=f"palette-{plan['palette']}",
        calibration=C.collect(plan["calibration_layouts"], plan["bijection"], stratum, 9,
                              seed=plan["action_seed"], policy="uniform"),
        transfer=C.collect(plan["transfer_layouts"], plan["bijection"], stratum, 9,
                           seed=plan["action_seed"] + 91, policy="uniform"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--palettes", type=int, default=64)
    parser.add_argument("--train-palettes", type=int, default=32)
    parser.add_argument("--calibration", type=int, default=6)
    parser.add_argument("--transfer", type=int, default=20)
    parser.add_argument("--train-transfer", type=int, default=8)
    parser.add_argument("--alias", type=int, default=2)
    parser.add_argument("--replication", action="store_true",
                        help="also score the untouched replication palettes")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o3-population.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    from belief_factorization import build_dataset
    from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED
    from structured_calibration import collect as structured_collect

    registry = C.canonical_registry()
    print(f"building {arguments.train_palettes} development palette groups", flush=True)
    train_blocks = []
    for palette in DEV_PALETTES[:arguments.train_palettes]:
        plan = palette_plan(palette, arguments.calibration, arguments.train_transfer,
                            arguments.alias)
        block = O3.scenario_block(palette_scenario(plan), plan["bijection"], registry,
                                  VIEW, contested_only=False)
        train_blocks.append(block)
    train = {k: np.concatenate([b[k] for b in train_blocks])
             for k in ("sequence", "mask", "before", "after", "event")}
    print(f"  {len(train['event'])} training rows", flush=True)

    infer, model = M.train_memory(
        (train["sequence"], train["mask"], train["before"], train["after"],
         train["event"]), SEED, updates=mem.MEMORY_UPDATES)

    structured_train = build_dataset(
        structured_collect(list(m2d.TRAIN_LAYOUTS), 3, 9, CANONICAL_APPEARANCE_SEED, 11),
        5)
    filter_model, _ = m2d.train_model(FilterSpec("o3", "accumulator"), structured_train,
                                      SEED)

    print("replaying the alias pool once (palette-free)", flush=True)
    alias_cache: dict[int, Any] = {}
    for layout in ALIAS_POOL:
        population = m2d.build_population([layout])
        tensors = m2d.build_tensors(population, m2d.RouteFeatures(population))
        replays = {i: o2route.route_roles(population.states[i].layout,
                                          population.states[i].route)
                   for i in sorted({r.self_index for r in population.rows})}
        alias_cache[layout] = (population, tensors, replays)
        print(f"  layout {layout}: {population.summary()['states']} states", flush=True)

    groups = {"validation": VALIDATION_PALETTES[:arguments.palettes]}
    if arguments.replication:
        groups["replication"] = REPLICATION_PALETTES[:arguments.palettes]

    report: dict[str, Any] = {
        "view": VIEW, "seed": SEED,
        "development_palettes": list(DEV_PALETTES[:arguments.train_palettes]),
        "validation_palettes": list(groups["validation"]),
        "replication_palettes": list(groups.get("replication", [])),
        "calibration_pool": [CAL_POOL[0], CAL_POOL[-1]],
        "transfer_pool": [TRANSFER_POOL[0], TRANSFER_POOL[-1]],
        "alias_pool": list(ALIAS_POOL),
        "per_palette_content": {"calibration_layouts": arguments.calibration,
                                "transfer_layouts": arguments.transfer,
                                "alias_layouts": arguments.alias},
        "route_parity_gate": ROUTE_PARITY_GATE,
        "equivariance_note": (
            "the pipeline is bit-exact palette-equivariant (o3-orbit.json), so on a FIXED "
            "semantic population between-palette variance is zero by construction; the "
            "variance below is the variance of the CONTENT each palette drew"),
        "groups": {},
    }

    for name, palettes in groups.items():
        rows = []
        for palette in palettes:
            plan = palette_plan(palette, arguments.calibration, arguments.transfer,
                                arguments.alias)
            scenario = palette_scenario(plan)
            block = O3.scenario_block(scenario, plan["bijection"], registry, VIEW)
            logits = infer((block["sequence"], block["mask"], block["before"],
                            block["after"]))
            contested = float(((logits > 0).astype(float) == block["event"]).mean())

            history = block["sequence"][0, :-1]
            mask = np.ones(len(history), np.float32)
            parity, alias_hit, per_step, rates = [], [], [], []
            for layout in plan["alias_layouts"]:
                population, tensors, replays = alias_cache[layout]
                out = ro.route_parity_under(replays, population, tensors,
                                            plan["bijection"], registry, VIEW, infer,
                                            history, mask, filter_model)
                out.pop("events")
                parity.append(out["final_event_parity_accuracy"])
                alias_hit.append(out["alias_outcome_accuracy"])
                per_step.append(out["per_step_accuracy"])
                rates.append(out["predicted_event_rate"])
            rows.append({
                "palette": palette, "bijection": [int(v) for v in plan["bijection"]],
                "calibration_layouts": plan["calibration_layouts"],
                "transfer_layouts": plan["transfer_layouts"],
                "alias_layouts": plan["alias_layouts"],
                "contested_transfer_accuracy": contested,
                "contested_rows": int(len(block["event"])),
                "route_parity": float(np.mean(parity)),
                "per_step_event_accuracy": float(np.mean(per_step)),
                "alias_outcome_accuracy": float(np.mean(alias_hit)),
                "predicted_event_rate": float(np.mean(rates)),
                "collapsed": bool(np.mean(rates) < 0.01),
            })
            if len(rows) % 8 == 0:
                print(f"  {name}: {len(rows)}/{len(palettes)} palettes", flush=True)

        def summarise(key):
            values = np.array([r[key] for r in rows])
            rng = np.random.default_rng(7)
            draws = np.array([values[rng.integers(0, len(values), len(values))].mean()
                              for _ in range(4_000)])
            return {"mean": float(values.mean()), "median": float(np.median(values)),
                    "minimum": float(values.min()), "maximum": float(values.max()),
                    "p10": float(np.percentile(values, 10)),
                    "palette_ci_low": float(np.percentile(draws, 2.5)),
                    "palette_ci_high": float(np.percentile(draws, 97.5)),
                    "palettes": int(len(values))}

        parity_values = np.array([r["route_parity"] for r in rows])
        report["groups"][name] = {
            "palettes": len(rows),
            "contested_transfer_accuracy": summarise("contested_transfer_accuracy"),
            "route_parity": summarise("route_parity"),
            "per_step_event_accuracy": summarise("per_step_event_accuracy"),
            "alias_outcome_accuracy": summarise("alias_outcome_accuracy"),
            "palette_collapse_count": int(sum(r["collapsed"] for r in rows)),
            "palettes_at_or_above_gate": int((parity_values >= ROUTE_PARITY_GATE).sum()),
            "fraction_above_gate": float((parity_values >= ROUTE_PARITY_GATE).mean()),
            "distinct_route_parity_values": int(len(np.unique(np.round(parity_values, 6)))),
            "per_palette": rows,
        }
        block = report["groups"][name]
        print(f"\n{name}: {block['palettes']} palettes")
        print(f"  contested transfer "
              f"{block['contested_transfer_accuracy']['mean']:.4f} "
              f"[{block['contested_transfer_accuracy']['palette_ci_low']:.4f}, "
              f"{block['contested_transfer_accuracy']['palette_ci_high']:.4f}]  "
              f"p10 {block['contested_transfer_accuracy']['p10']:.4f}  "
              f"min {block['contested_transfer_accuracy']['minimum']:.4f}")
        print(f"  route parity {block['route_parity']['mean']:.4f} "
              f"[{block['route_parity']['palette_ci_low']:.4f}, "
              f"{block['route_parity']['palette_ci_high']:.4f}]  "
              f"p10 {block['route_parity']['p10']:.4f}  "
              f"min {block['route_parity']['minimum']:.4f}  "
              f"above gate {block['palettes_at_or_above_gate']}/{block['palettes']}  "
              f"collapsed {block['palette_collapse_count']}", flush=True)

    validation = report["groups"]["validation"]
    report["R3_memory_replicates_over_independent_palettes"] = bool(
        validation["contested_transfer_accuracy"]["palette_ci_low"] > 0.6)
    report["R6_route_parity_clears_the_gate_broadly"] = bool(
        validation["route_parity"]["mean"] >= ROUTE_PARITY_GATE
        and validation["fraction_above_gate"] >= 0.9)
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nR3 {report['R3_memory_replicates_over_independent_palettes']}   "
          f"R6 {report['R6_route_parity_clears_the_gate_broadly']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
