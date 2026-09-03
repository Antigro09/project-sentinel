"""I / J. Route-level visual event fidelity, and the complete visual temporal pathway.

The gate is decided by MEASURED sequence results, not by average per-step accuracy and
not by the independence formula, which appears only as a diagnostic. Route parity is the
product of per-step accuracies over error opportunities, so a detector at 0.95 per step
is nowhere near 0.95 on a six-step route.

The visual contribution is isolated by varying ONLY the event source. The belief filter
is the frozen M2F certified adaptive learner and the outcome head is the frozen M2F head,
so a difference between arms is a difference in what perception recovered. On the alias
population the two members of a pair share a byte-identical packet AND a byte-identical
frame, so every memoryless model -- structured or visual -- is pinned at exactly 0.5000
by construction, and that is asserted rather than assumed.

    .venv-shwm/bin/python experiments/shwm/n_pathway.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import m2d_core as m2d
import m2e_core as m2e
import m2f_core as m2f
import n_core as core
import n_heads as heads
import n_interfaces as ifaces
from m2d_core import ARTIFACTS, FilterSpec, write
from n_core import GRID

SEEDS = (32_000, 32_001, 32_002)


def replay_route_frames(layout: int, route, appearance: int) -> tuple[np.ndarray, list]:
    """Frames and public rows along an alias state's own route."""
    from sentinel.env.adapters.procedural_visual_v2 import ProceduralVisualV2Adapter
    from sentinel.wm.authority import AuthorityGate

    gate = AuthorityGate(gate_id="n-path")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    adapter.reset(layout, f"appearance:{appearance}")
    frames, actions = [], []
    for index in range(len(route) + 1):
        frames.append(adapter.frame().copy())
        if index == len(route):
            break
        action = int(route[index])
        actions.append(action)
        adapter.step(action, gate.authorize_evaluator(action, "n-path"))
    return np.stack(frames), actions


def visual_events_for_population(population, detector_fn, appearance: int,
                                 batch: int = 512) -> dict[int, np.ndarray]:
    """Per-state event probabilities along that state's route, from FRAMES only."""
    needed = sorted({r.self_index for r in population.rows})
    befores, afters, actions, owner, position = [], [], [], [], []
    for index in needed:
        state = population.states[index]
        frames, route_actions = replay_route_frames(state.layout, state.route, appearance)
        for t in range(1, len(frames)):
            befores.append(frames[t - 1])
            afters.append(frames[t])
            one_hot = np.zeros(4, dtype=np.float32)
            one_hot[route_actions[t - 1]] = 1.0
            actions.append(one_hot)
            owner.append(index)
            position.append(t)
    if not befores:
        return {}
    before = np.stack(befores).astype(np.float32) / 255.0
    after = np.stack(afters).astype(np.float32) / 255.0
    action = np.stack(actions)
    probability = detector_fn(before, after, action)
    out: dict[int, np.ndarray] = {}
    owner = np.array(owner)
    position = np.array(position)
    for index in needed:
        mask = owner == index
        length = int(position[mask].max()) + 1 if mask.any() else 1
        row = np.zeros(length, dtype=np.float32)
        row[position[mask]] = probability[mask]
        out[index] = row
    return out


def events_into_tensor(tensors, population, per_state: dict[int, np.ndarray],
                       hard: bool = True) -> np.ndarray:
    events = np.zeros_like(tensors.events_true)
    for k, (state_index, _action) in enumerate(tensors.keys):
        row = per_state.get(state_index)
        if row is None:
            continue
        n = min(len(row), events.shape[1])
        events[k, :n] = (row[:n] >= 0.5).astype(np.float32) if hard else row[:n]
    events[:, 0] = 0.0
    return events


def parity_metrics(estimated: np.ndarray, truth: np.ndarray, lengths: np.ndarray,
                   layouts: np.ndarray | None = None) -> dict[str, Any]:
    hard = (estimated >= 0.5).astype(np.float32)
    exact, parity, first_error, bursts, predicted = [], [], [], [], []
    for k, n in enumerate(lengths):
        span = slice(1, int(n))
        wrong = hard[k, span] != truth[k, span]
        exact.append(not wrong.any())
        parity.append((hard[k, span].sum() % 2) == (truth[k, span].sum() % 2))
        first_error.append(int(np.argmax(wrong)) + 1 if wrong.any() else -1)
        run = best = 0
        for value in wrong:
            run = run + 1 if value else 0
            best = max(best, run)
        bursts.append(best)
        error = np.minimum(estimated[k, span], 1.0 - estimated[k, span])
        predicted.append(float((1.0 + np.prod(1.0 - 2.0 * error)) / 2.0))
    flat = np.concatenate([(hard[k, 1:int(n)] != truth[k, 1:int(n)]).astype(float)
                           for k, n in enumerate(lengths)])
    a = np.concatenate([(hard[k, 1:int(n) - 1] != truth[k, 1:int(n) - 1]).astype(float)
                        for k, n in enumerate(lengths) if n > 2])
    b = np.concatenate([(hard[k, 2:int(n)] != truth[k, 2:int(n)]).astype(float)
                        for k, n in enumerate(lengths) if n > 2])
    correlation = (float(np.corrcoef(a, b)[0, 1])
                   if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan"))
    out = {"per_step_accuracy": float(1.0 - flat.mean()),
           "exact_route_sequence_accuracy": float(np.mean(exact)),
           "final_event_parity_accuracy": float(np.mean(parity)),
           "independence_diagnostic_only": float(np.mean(predicted)),
           "mean_first_error_position": float(np.mean([f for f in first_error if f > 0]))
           if any(f > 0 for f in first_error) else float("nan"),
           "mean_error_burst_length": float(np.mean(bursts)),
           "max_error_burst_length": int(np.max(bursts)) if len(bursts) else 0,
           "error_autocorrelation_lag1": correlation,
           "routes": int(len(lengths))}
    if layouts is not None:
        failure = {int(l): float(1 - np.mean(np.array(parity)[layouts == l]))
                   for l in np.unique(layouts)}
        out["layout_conditioned_parity_failure_rate"] = failure
        out["worst_layout_parity_failure_rate"] = float(max(failure.values()))
    return out


def certified_filter(train_items, seed: int, tau: float, k_max: int = 32,
                     block: int = 8):
    """The frozen M2F adaptive procedure, run live: draw blocks of generic restarts,
    keep the best by TRAINING likelihood, stop when it certifies."""
    used, best, best_score = 0, None, -np.inf
    while used < k_max:
        for restart in range(used, min(used + block, k_max)):
            model, _ = m2d.train_model(m2f.generic_spec(seed, restart), train_items,
                                       seed * 1_000 + restart)
            score = m2e.training_log_likelihood(model, train_items)
            if score > best_score:
                best, best_score = model, score
        used = min(used + block, k_max)
        if m2f.certify(best_score, tau) == "CERTIFIED":
            return best, used, "CERTIFIED"
    return best, used, "UNRESOLVED_TRANSITION"


def train_visual_detector(interface, train_pairs, seed: int):
    """The event head, spatially supervised, exactly as in the auxiliary sweep."""
    import n_aux
    trainable = isinstance(interface, ifaces.EquivariantCNN)
    if trainable:
        model, _ = n_aux.train_joint(interface, train_pairs, train_pairs.event_map,
                                     "spatial_scalar", GRID * GRID, seed)

        def detect(before, after, action):
            batch = core.PairBatch(**{**train_pairs.__dict__})
            batch.before, batch.after, batch.action = before, after, action
            logits = n_aux.predict_joint(model, batch)
            return heads.sigmoid(logits.max(axis=1))
    else:
        slots = interface.encode(train_pairs).slots
        model, _ = heads.train_target(slots, train_pairs.action, train_pairs.event_map,
                                      "spatial_scalar", GRID * GRID, seed)

        def detect(before, after, action):
            batch = core.PairBatch(**{**train_pairs.__dict__})
            batch.before, batch.after, batch.action = before, after, action
            logits = heads.predict(model, interface.encode(batch).slots, action)
            return heads.sigmoid(logits.max(axis=1))
    return detect


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="2")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--trajectories", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "n-pathway.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    procedures = json.loads((ARTIFACTS / "m2f-procedures.json").read_text())
    tau = procedures["tau"]
    from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED
    from structured_calibration import collect
    from belief_factorization import build_dataset

    print("training the visual event detector on frames only", flush=True)
    visual_train = core.to_pairs(core.collect_visual(
        core.TRAIN_LAYOUTS, arguments.trajectories, 9, CANONICAL_APPEARANCE_SEED, 11))
    import n_aux
    interface = n_aux.build_interfaces({arguments.interface})[arguments.interface]
    detect = train_visual_detector(interface, visual_train, 32_000)

    structured_train = build_dataset(
        collect(list(m2d.TRAIN_LAYOUTS), 3, 9, CANONICAL_APPEARANCE_SEED, 11), 5)

    report: dict[str, Any] = {"interface": interface.name, "tau": tau,
                              "seeds": list(SEEDS[:arguments.seeds]),
                              "populations": {}, "sequence": {}, "arms": {}}
    populations = {}
    for label, layouts in (("validation", m2f.VALIDATION_ALIAS),
                           ("held_out", m2f.HELD_OUT_ALIAS)):
        population = m2d.build_population(layouts)
        tensors = m2d.build_tensors(population, m2d.RouteFeatures(population))
        per_state = visual_events_for_population(population, detect,
                                                 CANONICAL_APPEARANCE_SEED)
        visual = events_into_tensor(tensors, population, per_state, hard=True)
        populations[label] = {"population": population, "tensors": tensors,
                              "visual": visual,
                              "strata": m2d.stratify(population),
                              "manifest": m2e.population_manifest(population, label,
                                                                  layouts)}
        layout_of_key = np.array([population.states[i].layout for i, _ in tensors.keys])
        block = parity_metrics(visual, tensors.events_true, tensors.lengths,
                               layout_of_key)
        report["sequence"][label] = block
        report["populations"][label] = {
            k: v for k, v in populations[label]["manifest"].items()
            if k not in ("member_table", "member_routes")}
        print(f"  {label:12s} per-step {block['per_step_accuracy']:.4f}  "
              f"exact-seq {block['exact_route_sequence_accuracy']:.4f}  "
              f"parity {block['final_event_parity_accuracy']:.4f}  "
              f"(independence diagnostic {block['independence_diagnostic_only']:.4f})",
              flush=True)

    arms = {
        "1_visual_event_exact_accumulator": ("accumulator", "visual"),
        "2_visual_event_certified_transition": ("certified", "visual"),
        "3_visual_event_generic_gru": ("gru", "visual"),
        "4_visual_event_no_temporal_state": ("memoryless", None),
        "5_visual_memoryless_baseline": ("memoryless", None),
        "6_structured_event_certified_ceiling": ("certified", "true"),
        "7_exact_event_exact_accumulator_ceiling": ("accumulator", "true"),
    }
    print(f"\ncomplete visual temporal pathway, {arguments.seeds} seeds")
    for label, block in populations.items():
        print(f"\n-- {label} --")
        print(f"{'arm':44s} {'alias':>8s} {'p10':>8s}")
        hits: dict[str, np.ndarray] = {}
        for name, (kind, source) in arms.items():
            per_seed, accuracy = [], []
            for seed in SEEDS[:arguments.seeds]:
                if kind == "certified":
                    model, _, _ = certified_filter(structured_train, seed, tau)
                else:
                    model, _ = m2d.train_model(FilterSpec("n", kind), structured_train,
                                               seed)
                events = (None if source is None
                          else block["visual"] if source == "visual" else None)
                scored = m2d.score_population(model, block["tensors"], events)
                per_seed.append(scored["hit"])
                accuracy.append(float(scored["hit"].mean()))
            hits[name] = np.stack(per_seed)
            summary = m2d.summarise_metric(np.array(accuracy))
            report["arms"].setdefault(label, {})[name] = {"stats": summary}
            print(f"{name:44s} {summary['mean']:8.4f} {summary['p10']:8.4f}", flush=True)

        rows = len(block["population"].rows)
        seed_column = np.repeat(np.array(SEEDS[:arguments.seeds]), rows)
        layout_column = np.tile(block["strata"]["layout"], arguments.seeds)
        class_column = np.tile(block["strata"]["alias_class"], arguments.seeds)
        changes = np.tile(block["strata"]["changes"], arguments.seeds)
        baseline = hits["5_visual_memoryless_baseline"].ravel()
        for name in arms:
            if name == "5_visual_memoryless_baseline":
                continue
            entry = {"vs_memoryless": m2d.hierarchical_paired_interval(
                hits[name].ravel(), baseline, seed_column, layout_column, class_column)}
            for tag, mask in (("changes_2plus", changes >= 2),
                              ("changes_4plus", changes >= 4)):
                entry[tag] = m2d.hierarchical_paired_interval(
                    hits[name].ravel(), baseline, seed_column, layout_column,
                    class_column, mask=mask)
            report["arms"][label][name]["intervals"] = entry
        main_arm = report["arms"][label]["2_visual_event_certified_transition"]["intervals"]
        print(f"  visual+certified vs memoryless "
              f"{main_arm['vs_memoryless']['delta']:+.4f} "
              f"[{main_arm['vs_memoryless']['ci_low']:+.4f}, "
              f"{main_arm['vs_memoryless']['ci_high']:+.4f}]"
              f"{' *' if main_arm['vs_memoryless']['excludes_zero'] else ''}   2+ "
              f"{main_arm['changes_2plus']['delta']:+.4f}"
              f"{' *' if main_arm['changes_2plus']['excludes_zero'] else ''}", flush=True)
        report["arms"][label]["memoryless_is_exactly_chance"] = bool(
            abs(float(baseline.mean()) - 0.5) < 1e-9)

    validation = report["arms"]["validation"]["2_visual_event_certified_transition"]
    held_out = report["arms"]["held_out"]["2_visual_event_certified_transition"]
    report["n6_validation"] = bool(validation["intervals"]["vs_memoryless"]["ci_low"] > 0)
    report["n7_two_changes"] = bool(
        validation["intervals"]["changes_2plus"]["ci_low"] > 0)
    report["n8_held_out"] = bool(held_out["intervals"]["vs_memoryless"]["ci_low"] > 0)
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nN6 {report['n6_validation']}  N7 {report['n7_two_changes']}  "
          f"N8 {report['n8_held_out']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
