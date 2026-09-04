"""B / R1. Is the pipeline a function of the semantics, or of the colours?

O2 reported one unseen palette at 1.0000 route parity and three at 0.5346, and read it as
a calibration-sufficiency question. This module asks the prior question the specification
insists on: hold the semantics EXACTLY fixed -- same layouts, same actions, same
calibration interactions, same transfer route, same hidden phase, same goal -- vary only
the role-to-colour map, and see whether the pipeline's semantic predictions move.

They must not. The binder is a DeepSets over per-colour tokens addressed through a
canonical registry, so a pure relabelling permutes its inputs and must permute its outputs
identically. Anything that survives that permutation into the decision is a
canonicalization defect, and the one candidate visible in the source is the RGB block:
O2's factorial SELECTED `count_plus_motion`, and O2's memory model was nonetheless trained
on `full_token`, which carries the raw colour value.

Five defects are planted and each must be caught, because an equivariance test that
cannot fail is not a test.

    .venv-shwm/bin/python experiments/shwm/o3_orbit.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import o2_core as C
import o2_memory as mem
import o2_models as M
import o3_core as O3
from m2d_core import ARTIFACTS, write
from o3_core import N_POOL

SEED = 51_000
CAL_LAYOUTS = tuple(range(116_000, 116_006))
TRANSFER_LAYOUTS = tuple(range(117_000, 117_024))
VIEWS = ("full_token", "no_rgb")
# Five defects that must break equivariance. Two earlier candidates were tried and are
# recorded below as RECLASSIFIED rather than quietly dropped: a scan-order registry is
# only a different PERMUTATION of slots, and this architecture is permutation-equivariant,
# so within one scenario it changes nothing (it bites only when memory is written under
# one palette's slot order and read under another's -- which is why `cross_palette_memory`
# replaces it); and forcing a role onto a fixed pool colour breaks GENERATOR HONESTY,
# which the O2 leakage guard already catches at 1.1490 bits, not equivariance.
PLANTS = ("slot_index_bias", "rgb_sorted_tokens", "cross_palette_memory",
          "palette_index_feature", "memory_key_collision")
RECLASSIFIED = {
    "scan_order_registry": (
        "not an equivariance defect for this architecture: a scan-order registry is a "
        "permutation of slots and the binder is permutation-equivariant, so the orbit "
        "spread is exactly 0.00e+00. The hazard it stands for is cross-palette memory "
        "addressing, which `cross_palette_memory` tests directly."),
    "role_dependent_colour": (
        "not an equivariance defect: pinning SWITCH to a fixed pool colour collapses part "
        "of the orbit to a constant and leaves the outputs identical. It is a generator-"
        "honesty defect, caught by the O2 leakage audit's guard A at 1.1490 bits."),
}
TOLERANCE = 1e-4


def plant_registry(scenario: O3.Scenario, bijection: np.ndarray, kind: str):
    """Registries used by the plants. The honest one is canonical and palette-free."""
    if kind == "memory_key_collision":
        # THE PLANT: two pool colours share a slot, so a memory written for one is read
        # back for the other.
        registry = C.canonical_registry()
        registry.index[tuple(int(v) for v in C.COLOUR_POOL[1])] = registry.index[
            tuple(int(v) for v in C.COLOUR_POOL[0])]
        return registry
    return C.canonical_registry()


def plant_bijection(bijection: np.ndarray, kind: str) -> np.ndarray:
    return bijection


def plant_tokens(block: dict[str, np.ndarray], bijection: np.ndarray,
                 registry: C.ColourRegistry, kind: str) -> dict[str, np.ndarray]:
    if kind == "rgb_sorted_tokens":
        # THE PLANT: rows reordered by RGB, which destroys the correspondence between a
        # token's position and the cell indices that gather from it.
        order = np.lexsort(np.array([[c[2], c[1], c[0]] for c in registry.order]).T)
        block = dict(block)
        block["sequence"] = block["sequence"][:, :, order]
        return block
    if kind == "slot_index_bias":
        # THE PLANT: a constant that depends on the SLOT INDEX, which is exactly what a
        # permutation-equivariant network may not have.
        block = dict(block)
        sequence = np.array(block["sequence"], copy=True)
        sequence[:, :, :, C.GLOBAL.start] += (
            np.arange(C.MAX_COLOURS, dtype=np.float32) / C.MAX_COLOURS).reshape(1, 1, -1)
        block["sequence"] = sequence
        return block
    if kind == "palette_index_feature":
        # THE PLANT: the pool index written into the token, so the model can read which
        # colour it is looking at even from a colour-free view.
        block = dict(block)
        sequence = np.array(block["sequence"], copy=True)
        for role in range(C.N_ROLES):
            slot = registry.of(C.COLOUR_POOL[bijection[role]])
            sequence[:, :, slot, C.GLOBAL.start] = float(bijection[role]) / N_POOL
        block["sequence"] = sequence
        return block
    return block


def evaluate(scenario: O3.Scenario, bijection: np.ndarray, view: str, infer, model,
             kind: str = "honest") -> dict[str, Any]:
    registry = plant_registry(scenario, bijection, kind)
    rendered = plant_bijection(bijection, kind)
    if kind == "cross_palette_memory":
        # THE PLANT: the calibration history is rendered under a DIFFERENT palette from
        # the transfer pairs, so the memory is written against one set of colour keys and
        # read against another. This is the real hazard a non-canonical registry stands
        # for, and it is invisible to a within-scenario slot permutation.
        foreign = C.sample_bijection(int(rendered.sum()) + 7)
        history = O3.scenario_block(
            O3.Scenario(scenario.label, scenario.calibration, scenario.transfer),
            foreign, registry, view)["sequence"][0, :-1]
        block = O3.scenario_block(scenario, rendered, registry, view)
        block = dict(block)
        sequence = np.array(block["sequence"], copy=True)
        sequence[:, :-1] = history[None]
        block["sequence"] = sequence
    else:
        block = O3.scenario_block(scenario, rendered, registry, view)
    block = plant_tokens(block, rendered, registry, kind)
    logits = infer((block["sequence"], block["mask"], block["before"], block["after"]))
    assignment = M.memory_assignment_of(model, block["sequence"], block["mask"])
    semantic = O3.semantic_assignment(assignment[0], rendered, registry)
    entered = block["sequence"][:, -1, :, C.INTERACT][:, :, 0].argmax(axis=1)
    switch_mass = assignment[np.arange(len(assignment)), entered, C.SWITCH]
    resolved = np.abs(switch_mass - 0.5) >= mem.ABSTENTION_MARGIN
    return {
        "bijection": [int(v) for v in bijection],
        "event_accuracy": float(((logits > 0).astype(float) == block["event"]).mean()),
        "event_balanced": M.balanced_accuracy(logits, block["event"]),
        "assignment_entropy_bits": O3.entropy_bits(assignment[0]),
        "semantic_assignment": semantic.tolist(),
        "abstention_rate": float(1.0 - resolved.mean()),
        "rows": int(len(block["event"])),
        "logits": logits,
        "decisions": (logits > 0).astype(np.int8),
        "semantic_flat": semantic.ravel(),
    }


def orbit_spread(results: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy = np.array([r["event_accuracy"] for r in results])
    logits = np.stack([r["logits"] for r in results])
    decisions = np.stack([r["decisions"] for r in results])
    semantic = np.stack([r["semantic_flat"] for r in results])
    flips = (decisions != decisions[0]).any(axis=0)
    return {
        "decision_flip_fraction": float(flips.mean()),
        "rows_whose_decision_flips": int(flips.sum()),
        "permutations": len(results),
        "event_accuracy_mean": float(accuracy.mean()),
        "event_accuracy_min": float(accuracy.min()),
        "event_accuracy_max": float(accuracy.max()),
        "event_accuracy_spread": float(accuracy.max() - accuracy.min()),
        "distinct_accuracy_values": int(len(np.unique(np.round(accuracy, 6)))),
        "max_logit_deviation": float(np.abs(logits - logits[0]).max()),
        "max_semantic_assignment_deviation": float(
            np.abs(semantic - semantic[0]).max()),
        "abstention_spread": float(max(r["abstention_rate"] for r in results)
                                   - min(r["abstention_rate"] for r in results)),
        "equivariant": bool(np.abs(logits - logits[0]).max() < TOLERANCE
                            and np.abs(semantic - semantic[0]).max() < TOLERANCE),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", type=int, default=128)
    parser.add_argument("--exhaustive", type=int, default=720)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o3-orbit.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    scenario = O3.build_scenario("orbit", CAL_LAYOUTS, TRANSFER_LAYOUTS, 71)
    orbit = O3.base_role_orbit()[:arguments.exhaustive] + O3.random_orbit(
        arguments.random)
    print(f"scenario {scenario.semantic_digest}: {len(scenario.calibration)} calibration "
          f"and {len(scenario.transfer)} transfer episodes, held EXACTLY fixed\n"
          f"orbit: {arguments.exhaustive} exhaustive base-role permutations + "
          f"{arguments.random} random injections = {len(orbit)}\n", flush=True)

    report: dict[str, Any] = {
        "semantic_digest": scenario.semantic_digest,
        "calibration_layouts": list(CAL_LAYOUTS),
        "transfer_layouts": list(TRANSFER_LAYOUTS),
        "orbit_exhaustive": arguments.exhaustive, "orbit_random": arguments.random,
        "orbit_size": len(orbit), "tolerance": TOLERANCE,
        "held_fixed": ["semantic layout", "semantic object roles", "action sequence",
                       "calibration interactions", "transfer route",
                       "hidden phase dynamics", "language goal"],
        "views": {}, "plants": {},
    }

    dev = [mem.build_group(p, 71) for p in mem.DEV_PALETTES]
    registry = C.canonical_registry()
    for view in VIEWS:
        train = mem.stack_groups(dev, registry, view=view)
        infer, model = M.train_memory(
            (train["sequence"], train["mask"], train["before"], train["after"],
             train["event"]), SEED, updates=mem.MEMORY_UPDATES)
        results = [evaluate(scenario, b, view, infer, model) for b in orbit]
        spread = orbit_spread(results)
        report["views"][view] = {
            **spread,
            "per_permutation": [{k: v for k, v in r.items()
                                 if k not in ("logits", "semantic_flat", "decisions")}
                                for r in results[:32]],
            "accuracy_histogram": {
                f"{v:.4f}": int(n) for v, n in zip(
                    *np.unique(np.round([r["event_accuracy"] for r in results], 4),
                               return_counts=True))},
        }
        print(f"view {view:12s} accuracy {spread['event_accuracy_mean']:.4f} "
              f"[{spread['event_accuracy_min']:.4f}, {spread['event_accuracy_max']:.4f}]"
              f"  rows flipping {spread['rows_whose_decision_flips']}"
              f"/{results[0]['rows']}  max logit deviation "
              f"{spread['max_logit_deviation']:.2e}  "
              f"EQUIVARIANT {spread['equivariant']}", flush=True)
        if view == "no_rgb":
            honest_infer, honest_model = infer, model

    print("\nplanted canonicalization defects (each must break equivariance)", flush=True)
    small = orbit[:64]
    for kind in PLANTS:
        results = [evaluate(scenario, b, "no_rgb", honest_infer, honest_model, kind)
                   for b in small]
        spread = orbit_spread(results)
        report["plants"][kind] = {**spread, "caught": bool(not spread["equivariant"])}
        print(f"  {kind:26s} accuracy spread {spread['event_accuracy_spread']:.4f}  "
              f"rows flipping {spread['rows_whose_decision_flips']}  "
              f"max logit deviation {spread['max_logit_deviation']:.2e}  "
              f"caught {report['plants'][kind]['caught']}", flush=True)

    baseline = orbit_spread([evaluate(scenario, b, "no_rgb", honest_infer, honest_model)
                             for b in small])
    report["honest_on_the_plant_orbit"] = baseline
    report["reclassified_candidates"] = RECLASSIFIED
    report["R1_pipeline_is_palette_equivariant"] = bool(
        report["views"]["no_rgb"]["equivariant"])
    report["o2_memory_view_was_not_equivariant"] = bool(
        not report["views"]["full_token"]["equivariant"])
    report["plants_all_caught"] = bool(all(v["caught"] for v in report["plants"].values()))
    report["test_is_not_vacuous"] = bool(baseline["equivariant"]
                                         and report["plants_all_caught"])
    report["diagnosis"] = (
        "PALETTE-EQUIVARIANCE / CANONICALIZATION DEFECT in the O2 memory arm: it was "
        "trained on `full_token`, which carries the raw RGB block, while O2's own "
        "factorial had selected `count_plus_motion`. Removing the colour value from the "
        "view restores exact equivariance."
        if report["o2_memory_view_was_not_equivariant"]
        and report["R1_pipeline_is_palette_equivariant"]
        else "see the per-view table")
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nR1 (pipeline is palette-equivariant on the semantic orbit): "
          f"{report['R1_pipeline_is_palette_equivariant']}")
    print(f"O2's memory view was NOT equivariant: "
          f"{report['o2_memory_view_was_not_equivariant']}")
    print(f"every planted defect caught: {report['plants_all_caught']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
