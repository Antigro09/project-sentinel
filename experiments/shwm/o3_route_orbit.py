"""B, route leg. The same alias routes under many palettes, end to end.

The transfer-pair orbit shows a real but sub-threshold defect: O2's `full_token` memory
deviates by 1.65e-01 in logit space across the palette orbit and flips no decisions on
that population. The route population is where O2 actually saw the failure, and it looks
different in kind: three of four palettes returned EXACTLY 0.5346 parity and one returned
1.0000. Three identical values to four decimals is not sensitivity, it is a degenerate
prediction -- a constant -- so this module runs the route leg over an orbit and asks which
of the two views collapses.

The temporal filter is palette-free by construction: it consumes events, never pixels. So
it is built once and held fixed, and the only thing varying across the orbit is what
perception recovered.

    .venv-shwm/bin/python experiments/shwm/o3_route_orbit.py
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
import o3_orbit as orbit
from m2d_core import ARTIFACTS, FilterSpec, write

SEED = 52_000
# Calibration for the ROUTE must be disjoint from anything the memory trained on. The
# first version of this module reused mem.CAL_LAYOUTS with seed 71 -- the same layouts AND
# the same action seed as the development training groups -- so `C.collect` produced
# byte-identical calibration trajectories and the model was handed histories it had
# memorised. It returned parity 1.0000 on every palette and every alias layout, which is
# what a leak looks like when it is total.
ROUTE_CAL_LAYOUTS = tuple(range(118_000, 118_006))
ROUTE_CAL_SEED = 613
ALIAS_LAYOUTS = m2f.VALIDATION_ALIAS[:2]
VIEWS = ("full_token", "no_rgb")


def route_parity_under(replays, population, tensors, bijection, registry, view,
                       infer, history, mask, filter_model, chunk: int = 2_048):
    """Events from frames under one palette, then the fixed filter."""
    tokens, before, after, owner, position = [], [], [], [], []
    for index, (grids, actions, stripe) in replays.items():
        cells = [C.cells_from_roles(grids[t], bijection, stripe if t == 0 else None)
                 for t in range(len(grids))]
        for t in range(1, len(grids)):
            tokens.append(C.pair_tokens(cells[t - 1], cells[t], actions[t - 1], registry))
            before.append(C.cell_index(cells[t - 1], registry))
            after.append(C.cell_index(cells[t], registry))
            owner.append(index)
            position.append(t)
    tokens = M.mask_view(np.stack(tokens).astype(np.float32), view)
    before, after = np.stack(before), np.stack(after)
    owner, position = np.array(owner), np.array(position)

    logits = []
    for start in range(0, len(tokens), chunk):
        span = slice(start, start + chunk)
        pairs = {"tokens": tokens[span], "before_index": before[span],
                 "after_index": after[span],
                 "event": np.zeros(len(tokens[span]), np.float32)}
        sequence, seq_mask, b, a, _ = C.sequence_dataset(pairs, history, mask)
        logits.append(infer((sequence, seq_mask, b, a)))
    probability = 1.0 / (1.0 + np.exp(-np.concatenate(logits)))

    per_state = {}
    for index in replays:
        rows = owner == index
        length = int(position[rows].max()) + 1
        row = np.zeros(length, np.float32)
        row[position[rows]] = probability[rows]
        per_state[index] = row
    hard = npath.events_into_tensor(tensors, population, per_state, hard=True)
    soft = npath.events_into_tensor(tensors, population, per_state, hard=False)
    metrics = npath.parity_metrics(soft, tensors.events_true, tensors.lengths)
    scored = m2d.score_population(filter_model, tensors, hard)
    return {
        "per_step_accuracy": metrics["per_step_accuracy"],
        "exact_route_sequence_accuracy": metrics["exact_route_sequence_accuracy"],
        "final_event_parity_accuracy": metrics["final_event_parity_accuracy"],
        "alias_outcome_accuracy": float(scored["hit"].mean()),
        "predicted_event_rate": float((hard[:, 1:] > 0.5).mean()),
        "events": hard,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--palettes", type=int, default=64)
    parser.add_argument("--alias", default="validation",
                        choices=("validation", "held_out"))
    parser.add_argument("--layouts", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o3-route-orbit.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    from belief_factorization import build_dataset
    from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED
    from structured_calibration import collect as structured_collect

    alias_layouts = ((m2f.VALIDATION_ALIAS if arguments.alias == "validation"
                      else m2f.HELD_OUT_ALIAS)[:arguments.layouts])
    population = m2d.build_population(alias_layouts)
    tensors = m2d.build_tensors(population, m2d.RouteFeatures(population))
    replays = {}
    import o2_route as route
    for index in sorted({r.self_index for r in population.rows}):
        state = population.states[index]
        replays[index] = route.route_roles(state.layout, state.route)
    print(f"alias layouts {list(alias_layouts)}: {population.summary()['states']} states, "
          f"{len(replays)} replayed, {len(population.rows)} rows", flush=True)

    structured_train = build_dataset(
        structured_collect(list(m2d.TRAIN_LAYOUTS), 3, 9, CANONICAL_APPEARANCE_SEED, 11),
        5)
    filter_model, _ = m2d.train_model(FilterSpec("o3", "accumulator"), structured_train,
                                      SEED)

    registry = C.canonical_registry()
    dev = [mem.build_group(p, 71) for p in mem.DEV_PALETTES]
    reference = O3.build_scenario("route-cal", ROUTE_CAL_LAYOUTS,
                                  mem.TRANSFER_LAYOUTS[:8], ROUTE_CAL_SEED)
    assert not (set(ROUTE_CAL_LAYOUTS) & set(mem.CAL_LAYOUTS)), (
        "route calibration layouts must be disjoint from the training calibration")
    palettes = O3.random_orbit(arguments.palettes, seed=4_242)

    report: dict[str, Any] = {
        "alias_layouts": list(alias_layouts), "alias_set": arguments.alias,
        "palettes": arguments.palettes,
        "states": int(population.summary()["states"]),
        "rows": int(len(population.rows)),
        "temporal_filter": "exact accumulator, palette-free by construction",
        "route_calibration_layouts": list(ROUTE_CAL_LAYOUTS),
        "route_calibration_seed": ROUTE_CAL_SEED,
        "training_calibration_layouts": list(mem.CAL_LAYOUTS),
        "calibration_disjoint_from_training": True,
        "views": {},
    }

    for view in VIEWS:
        train = mem.stack_groups(dev, registry, view=view)
        infer, model = M.train_memory(
            (train["sequence"], train["mask"], train["before"], train["after"],
             train["event"]), SEED, updates=mem.MEMORY_UPDATES)
        rows = []
        for bijection in palettes:
            history, mask = O3.scenario_block(
                reference, bijection, registry, view)["sequence"][0, :-1], None
            mask = np.ones(len(history), np.float32)
            block = route_parity_under(replays, population, tensors, bijection, registry,
                                       view, infer, history, mask, filter_model)
            events = block.pop("events")
            block["bijection"] = [int(v) for v in bijection]
            block["events_digest"] = int(events.sum())
            rows.append(block)
        parity = np.array([r["final_event_parity_accuracy"] for r in rows])
        alias = np.array([r["alias_outcome_accuracy"] for r in rows])
        rate = np.array([r["predicted_event_rate"] for r in rows])
        report["views"][view] = {
            "parity_mean": float(parity.mean()), "parity_min": float(parity.min()),
            "parity_max": float(parity.max()),
            "parity_spread": float(parity.max() - parity.min()),
            "distinct_parity_values": int(len(np.unique(np.round(parity, 6)))),
            "alias_mean": float(alias.mean()), "alias_min": float(alias.min()),
            "alias_max": float(alias.max()),
            "alias_spread": float(alias.max() - alias.min()),
            "predicted_event_rate_mean": float(rate.mean()),
            "predicted_event_rate_min": float(rate.min()),
            "collapsed_palettes": int((rate < 0.01).sum()),
            "palettes_at_or_above_075_parity": int((parity >= 0.75).sum()),
            "equivariant": bool(parity.max() - parity.min() < 1e-6
                                and alias.max() - alias.min() < 1e-6),
            "per_palette": rows,
        }
        block = report["views"][view]
        print(f"view {view:12s} parity {block['parity_mean']:.4f} "
              f"[{block['parity_min']:.4f}, {block['parity_max']:.4f}] spread "
              f"{block['parity_spread']:.4f}  {block['distinct_parity_values']} distinct  "
              f"alias {block['alias_mean']:.4f} spread {block['alias_spread']:.4f}  "
              f"collapsed {block['collapsed_palettes']}/{arguments.palettes}  "
              f"EQUIVARIANT {block['equivariant']}", flush=True)

    report["route_leg_is_equivariant"] = bool(report["views"]["no_rgb"]["equivariant"])
    report["o2_view_route_leg_is_equivariant"] = bool(
        report["views"]["full_token"]["equivariant"])
    report["diagnosis"] = (
        "the route leg reproduces O2's split under `full_token` and is exactly constant "
        "under `no_rgb`"
        if report["route_leg_is_equivariant"]
        and not report["o2_view_route_leg_is_equivariant"]
        else "see the per-view table")
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nroute leg equivariant under no_rgb: {report['route_leg_is_equivariant']}")
    print(f"route leg equivariant under O2's full_token: "
          f"{report['o2_view_route_leg_is_equivariant']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
