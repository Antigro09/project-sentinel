"""I / Q7 / Q8 / Q9. Route parity and the certified transition, under an unseen palette.

Phase O1 recorded P7, P8 and P9 as NOT_DELIVERED: the binder was never carried to a
route, never coupled to the M2F certified transition, and never scored on the alias
population under a palette it had not seen. This closes all three.

The alias construction does the heavy lifting. Both members of a pair share a
byte-identical public packet AND, once rendered through the same hidden palette, a
byte-identical frame -- so every memoryless arm is pinned at exactly 0.5000 by
construction, and that is asserted rather than hoped for. What separates the pair is the
number of switch crossings behind it, which is only recoverable if perception recovers
the events AND a belief carries them.

The event source is the only thing that varies between the first arms; the belief filter
and the outcome head are the frozen M2F ones. A difference is therefore a difference in
what perception recovered under a palette nobody trained on.

    .venv-shwm/bin/python experiments/shwm/o2_route.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as m2d
import m2e_core as m2e
import m2f_core as m2f
import n_pathway as npath
import o_core as O
import o2_core as C
import o2_memory as mem
import o2_models as M
from m2d_core import ARTIFACTS, FilterSpec, write

SEEDS = (45_000, 45_001, 45_002)
CAL_LAYOUTS = mem.CAL_LAYOUTS
DEV_PALETTES = mem.DEV_PALETTES
UNSEEN_PALETTES = (9_400, 9_401, 9_402, 9_403)
STRATUM = "COUNT_COLLISION"

ARMS = {
    "1_palette_memory_event_exact_accumulator": ("accumulator", "memory"),
    "2_palette_memory_event_certified_transition": ("certified", "memory"),
    "3_palette_memory_event_generic_gru": ("gru", "memory"),
    "4_palette_memory_event_no_temporal_state": ("memoryless", None),
    "5_visual_memoryless_baseline": ("memoryless", None),
    "6_stateless_binder_event_certified_transition": ("certified", "stateless"),
    "7_true_event_certified_transition_ceiling": ("certified", "true"),
    "8_true_event_exact_accumulator_ceiling": ("accumulator", "true"),
}


def route_roles(layout: int, route, stratum: str = STRATUM):
    """Role grids and actions along an alias state's route. PALETTE-FREE, so the replay
    happens once and every palette is a lookup afterwards."""
    from sentinel.env.adapters.procedural_visual_v2 import ProceduralVisualV2Adapter
    from sentinel.wm.authority import AuthorityGate

    gate = AuthorityGate(gate_id="o2-route")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    adapter.reset(layout)
    level = adapter._require()
    start = tuple(int(v) for v in level.start)
    placement = np.random.default_rng(7_000_003 + layout)
    decoy = C.decoy_placement(level, start, C.decoy_count(stratum, placement), placement)
    stripe = int(level.initial_polarity)
    grids, actions = [], []
    for index in range(len(route) + 1):
        position = tuple(int(v) for v in
                         adapter.snapshot().reveal("evaluator")["position"])
        grid = O.role_grid(level, position)
        grid[decoy & (grid == C.EMPTY)] = C.DECOY
        grids.append(grid)
        if index == len(route):
            break
        action = int(route[index])
        actions.append(action)
        adapter.step(action, gate.authorize_evaluator(action, "o2-route"))
    return np.stack(grids), actions, stripe


def replay_population(population) -> dict[int, tuple]:
    needed = sorted({r.self_index for r in population.rows})
    return {index: route_roles(population.states[index].layout,
                               population.states[index].route)
            for index in needed}


def route_events(replays: dict[int, tuple], bijection: np.ndarray,
                 registry: C.ColourRegistry, history, history_mask, memory_infer,
                 pair_infer, chunk: int = 2_048) -> dict[str, dict[int, np.ndarray]]:
    """Per-state event probability along its route, for BOTH detectors in one pass.

    Chunked: a full population is twenty thousand frame pairs and the memory model's
    input is (rows, 33, 10, 22) floats, which is half a gigabyte if materialised whole.
    """
    tokens, before, after, owner, position = [], [], [], [], []
    for index, (grids, actions, stripe) in replays.items():
        cells = [C.cells_from_roles(grids[t], bijection, stripe if t == 0 else None)
                 for t in range(len(grids))]
        for t in range(1, len(grids)):
            tokens.append(C.pair_tokens(cells[t - 1], cells[t], actions[t - 1],
                                        registry))
            before.append(C.cell_index(cells[t - 1], registry))
            after.append(C.cell_index(cells[t], registry))
            owner.append(index)
            position.append(t)
    tokens = np.stack(tokens).astype(np.float32)
    before, after = np.stack(before), np.stack(after)
    owner, position = np.array(owner), np.array(position)

    memory_logits, stateless_logits = [], []
    for start in range(0, len(tokens), chunk):
        span = slice(start, start + chunk)
        pairs = {"tokens": tokens[span], "before_index": before[span],
                 "after_index": after[span],
                 "event": np.zeros(len(tokens[span]), np.float32)}
        sequence, mask, b, a, _ = C.sequence_dataset(pairs, history, history_mask)
        memory_logits.append(memory_infer((sequence, mask, b, a)))
        stateless_logits.append(pair_infer((pairs["tokens"], pairs["before_index"],
                                            pairs["after_index"])))
    out: dict[str, dict[int, np.ndarray]] = {"memory": {}, "stateless": {}}
    for source, logits in (("memory", np.concatenate(memory_logits)),
                           ("stateless", np.concatenate(stateless_logits))):
        probability = 1.0 / (1.0 + np.exp(-logits))
        for index in replays:
            rows = owner == index
            length = int(position[rows].max()) + 1
            row = np.zeros(length, np.float32)
            row[position[rows]] = probability[rows]
            out[source][index] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--palettes", type=int, default=len(UNSEEN_PALETTES))
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o2-route.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    procedures = json.loads((ARTIFACTS / "m2f-procedures.json").read_text())
    tau = procedures["tau"]
    from belief_factorization import build_dataset
    from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED
    from structured_calibration import collect as structured_collect

    print("training the palette memory on the transfer construction", flush=True)
    dev = [mem.build_group(p, 71) for p in DEV_PALETTES]
    unseen_groups = [mem.build_group(p, 313) for p in UNSEEN_PALETTES[:arguments.palettes]]
    registry = C.canonical_registry()
    train = mem.stack_groups(dev, registry)

    structured_train = build_dataset(
        structured_collect(list(m2d.TRAIN_LAYOUTS), 3, 9, CANONICAL_APPEARANCE_SEED, 11),
        5)

    report: dict[str, Any] = {
        "tau": tau, "seeds": list(SEEDS[:arguments.seeds]),
        "unseen_palettes": list(UNSEEN_PALETTES[:arguments.palettes]),
        "calibration_layouts": list(CAL_LAYOUTS), "stratum": STRATUM,
        "alias_populations": {}, "sequence": {}, "arms": {},
    }

    populations = {}
    for label, layouts in (("validation", m2f.VALIDATION_ALIAS),
                           ("held_out", m2f.HELD_OUT_ALIAS)):
        population = m2d.build_population(layouts)
        print(f"  replaying {label}: {population.summary()['states']} states",
              flush=True)
        tensors = m2d.build_tensors(population, m2d.RouteFeatures(population))
        populations[label] = {
            "population": population, "tensors": tensors,
            "replays": replay_population(population),
            "strata": m2d.stratify(population),
            "manifest": {k: v for k, v in
                         m2e.population_manifest(population, label, layouts).items()
                         if k not in ("member_table", "member_routes")}}
        report["alias_populations"][label] = populations[label]["manifest"]

    memory_infer, _ = M.train_memory(
        (train["sequence"], train["mask"], train["before"], train["after"],
         train["event"]), SEEDS[0], updates=mem.MEMORY_UPDATES)
    pair_infer, _ = M.train_stateless(
        (train["tokens"], train["before"], train["after"], train["event"]), SEEDS[0])

    print("\nroute-level event fidelity under unseen palettes", flush=True)
    for label, block in populations.items():
        per_palette: dict[str, list] = {}
        for group in unseen_groups:
            history, mask = mem.history_of(group, registry)
            events = route_events(block["replays"], group.bijection, registry,
                                  history, mask, memory_infer, pair_infer)
            for source, value in events.items():
                per_palette.setdefault(source, []).append(value)
        block["events"] = {}
        block["soft"] = {}
        for source, per in per_palette.items():
            # One palette per STATE, round-robin. Averaging the four palettes' event
            # probabilities would be an ensemble and would make the detector look better
            # than any single unseen palette it will actually meet.
            merged = {k: per[i % len(per)][k] for i, k in enumerate(sorted(per[0]))}
            block["events"][source] = npath.events_into_tensor(
                block["tensors"], block["population"], merged, hard=True)
            # SOFT probabilities for the sequence metrics. The hard metrics are
            # unchanged -- they threshold at 0.5 either way -- but the independence
            # diagnostic is a product over per-step error probabilities, and feeding it
            # hard 0/1 events makes every error probability exactly zero and the
            # prediction exactly 1.0000. That is what phase N printed and what this
            # module printed on its first run; it is a defect in the instrument, not a
            # property of the detector.
            block["soft"][source] = npath.events_into_tensor(
                block["tensors"], block["population"], merged, hard=False)
            metrics = npath.parity_metrics(
                block["soft"][source], block["tensors"].events_true,
                block["tensors"].lengths,
                np.array([block["population"].states[i].layout
                          for i, _ in block["tensors"].keys]))
            metrics["palette_assignment"] = "round-robin over states, one per state"
            metrics["per_palette"] = {
                str(group.bijection.tolist()): npath.parity_metrics(
                    npath.events_into_tensor(block["tensors"], block["population"],
                                             per[i], hard=True),
                    block["tensors"].events_true, block["tensors"].lengths
                )["final_event_parity_accuracy"]
                for i, group in enumerate(unseen_groups)}
            report["sequence"].setdefault(label, {})[source] = metrics
            print(f"  {label:11s} {source:10s} per-step "
                  f"{metrics['per_step_accuracy']:.4f}  exact-seq "
                  f"{metrics['exact_route_sequence_accuracy']:.4f}  parity "
                  f"{metrics['final_event_parity_accuracy']:.4f}", flush=True)
        block["events"]["true"] = block["tensors"].events_true

    print(f"\ncomplete pathway, {arguments.seeds} seeds")
    for label, block in populations.items():
        print(f"\n-- {label} --")
        print(f"{'arm':52s} {'alias':>8s} {'p10':>8s}")
        hits: dict[str, np.ndarray] = {}
        polarity = np.array([r.polarity_self for r in block["population"].rows])
        for name, (kind, source) in ARMS.items():
            per_seed, accuracy, belief_accuracy = [], [], []
            for seed in SEEDS[:arguments.seeds]:
                if kind == "certified":
                    model, _, _ = npath.certified_filter(structured_train, seed, tau)
                else:
                    model, _ = m2d.train_model(FilterSpec("o2", kind), structured_train,
                                               seed)
                events = None if source is None else block["events"][source]
                scored = m2d.score_population(model, block["tensors"], events)
                per_seed.append(scored["hit"])
                accuracy.append(float(scored["hit"].mean()))
                belief = scored["belief"]
                if belief.shape[-1] > 1:
                    # Up to permutation: the latent states are anonymous, so the label
                    # that matches better is the one the model meant.
                    argmax = belief.argmax(axis=-1)
                    belief_accuracy.append(max(float((argmax == polarity).mean()),
                                               float((argmax != polarity).mean())))
            hits[name] = np.stack(per_seed)
            summary = m2d.summarise_metric(np.array(accuracy))
            report["arms"].setdefault(label, {})[name] = {
                "stats": summary,
                "phase_belief_accuracy_up_to_permutation": (
                    float(np.mean(belief_accuracy)) if belief_accuracy else None)}
            print(f"{name:52s} {summary['mean']:8.4f} {summary['p10']:8.4f}", flush=True)

        assert abs(float(hits["5_visual_memoryless_baseline"].mean()) - 0.5) < 1e-9, (
            "the alias construction pins every memoryless arm at exactly 0.5000; "
            f"got {hits['5_visual_memoryless_baseline'].mean()}")

        rows = len(block["population"].rows)
        seed_column = np.repeat(np.array(SEEDS[:arguments.seeds]), rows)
        layout_column = np.tile(block["strata"]["layout"], arguments.seeds)
        class_column = np.tile(block["strata"]["alias_class"], arguments.seeds)
        changes = np.tile(block["strata"]["changes"], arguments.seeds)
        baseline = hits["5_visual_memoryless_baseline"].ravel()
        for name in ARMS:
            entry = {"vs_memoryless": m2d.hierarchical_paired_interval(
                hits[name].ravel(), baseline, seed_column, layout_column, class_column)}
            for tag, selector in (("changes_0", changes == 0), ("changes_1", changes == 1),
                                  ("changes_2", changes == 2), ("changes_3", changes == 3),
                                  ("changes_4plus", changes >= 4)):
                entry[tag] = {
                    "rows": int(selector.sum()),
                    "accuracy": (float(hits[name].ravel()[selector].mean())
                                 if selector.any() else float("nan")),
                    **m2d.hierarchical_paired_interval(
                        hits[name].ravel(), baseline, seed_column, layout_column,
                        class_column, mask=selector)}
            report["arms"][label][name]["intervals"] = entry
        main_arm = report["arms"][label]["2_palette_memory_event_certified_transition"][
            "intervals"]
        print(f"  memory+certified vs memoryless {main_arm['vs_memoryless']['delta']:+.4f} "
              f"[{main_arm['vs_memoryless']['ci_low']:+.4f}, "
              f"{main_arm['vs_memoryless']['ci_high']:+.4f}]"
              f"{' *' if main_arm['vs_memoryless']['excludes_zero'] else ''}", flush=True)
        for tag in ("changes_2", "changes_3", "changes_4plus"):
            block_tag = main_arm[tag]
            print(f"    {tag:14s} {block_tag['rows']:5d} rows  "
                  f"{block_tag['delta']:+.4f} [{block_tag['ci_low']:+.4f}, "
                  f"{block_tag['ci_high']:+.4f}]"
                  f"{' *' if block_tag['excludes_zero'] else ''}")

    held = report["arms"]["held_out"]["2_palette_memory_event_certified_transition"][
        "intervals"]
    report["Q7_route_parity_supported"] = bool(
        report["sequence"]["held_out"]["memory"]["final_event_parity_accuracy"] > 0.75)
    report["Q8_visual_event_plus_certified_beats_memoryless"] = bool(
        held["vs_memoryless"]["ci_low"] > 0)
    report["Q9_gain_survives_two_phase_changes"] = bool(
        held["changes_2"]["ci_low"] > 0
        and (held["changes_3"]["ci_low"] > 0 or held["changes_4plus"]["ci_low"] > 0))
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nQ7 {report['Q7_route_parity_supported']}  "
          f"Q8 {report['Q8_visual_event_plus_certified_beats_memoryless']}  "
          f"Q9 {report['Q9_gain_survives_two_phase_changes']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
