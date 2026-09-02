"""7 / 9 / 10 / V7-V11. Coupling rule, the complete learned pathway, and what removes it.

M2D chose hard coupling over posterior mixture on a 0.006 accuracy difference while
ignoring an NLL difference of 1.5 nats. The criterion here is preregistered and joint:
phase-sensitive NLL first, alias accuracy second, and no worse calibration -- all fixed
on a development alias population before a validation seed is opened.

The complete pathway is only interesting if it survives layouts it has never seen, so
validation and held-out are both gates rather than one gate and one footnote.

    .venv-shwm/bin/python experiments/shwm/m2e_coupling.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as m2d
import m2e_core as core
from m2e_core import ARTIFACTS, VALIDATION_SEEDS, DEV_SEEDS, Arm, build_arms, write
from m2e_core import population_manifest
from m2d_core import build_tensors, save_predictions, score_population, stratify
from m2d_coupling import EventDetector, corrupt, CORRUPTIONS, collect_goal_directed
from m2e_transition import evaluate, summarise
from structured_calibration import collect
from belief_factorization import build_dataset
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

DEV_ALIAS = tuple(range(91_000, 91_010))
HELD_OUT_ALIAS = tuple(range(95_000, 95_010))
HELD_OUT_ALIAS_2 = tuple(range(92_000, 92_010))

COUPLING_CRITERION = ("primary: phase-sensitive NLL on development alias rows; "
                      "secondary: alias-pair accuracy; "
                      "constraint: ECE no worse than the best rule by more than 0.02")
PARTICLES = 32


def coupled_events(tensors, detector: EventDetector, rule: str, seed: int = 5) -> np.ndarray:
    probability = detector.probabilities(tensors.z)
    if rule == "hard":
        return (probability >= 0.5).astype(np.float32)
    if rule == "posterior":
        return probability
    if rule == "particle":
        rng = np.random.default_rng(seed)
        draws = (rng.random((PARTICLES,) + probability.shape) < probability)
        return draws.mean(axis=0).astype(np.float32)
    if rule == "exact":
        return tensors.events_true
    raise KeyError(rule)


def calibration_error(probability: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    index = np.clip((probability * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        picked = index == b
        if picked.any():
            total += picked.mean() * abs(float(probability[picked].mean())
                                         - float(correct[picked].mean()))
    return float(total)


def score_block(model, tensors, events) -> dict[str, float]:
    scored = score_population(model, tensors, events)
    probability = 1.0 / (1.0 + np.exp(-np.clip(scored["margin"], -50, 50)))
    return {"alias_accuracy": float(scored["hit"].mean()),
            "phase_sensitive_nll": float(scored["nll"].mean()),
            "brier": float(scored["brier"].mean()),
            "ece": calibration_error(probability, scored["hit"]),
            "margin": float(scored["margin"].mean()),
            "_hit": scored["hit"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-seeds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=len(VALIDATION_SEEDS))
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2e-coupling.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    m2d.check_feature_layout()
    appearance = CANONICAL_APPEARANCE_SEED

    transition = json.loads((ARTIFACTS / "m2e-transition.json").read_text())
    selected_k = transition["development"]["selected_k"]
    passing = transition["v3_passing_arms"]
    generic_key = passing[0] if passing else max(
        (k for k, v in transition["validation"].items() if v["eligible"]
         and k != "H_trained_memoryless"),
        key=lambda k: transition["validation"][k]["stats"]["mean"])
    print(f"V3 passing arms: {passing or 'NONE'}; generic filter used here: {generic_key}"
          f"{'' if passing else '  (BEST ELIGIBLE, NOT A V3 PASS)'}", flush=True)

    train_t = collect(list(m2d.TRAIN_LAYOUTS), 3, 9, appearance, 11)
    train = build_dataset(train_t, 5)
    detector = EventDetector("full").fit(train)
    action_only = EventDetector("action_only").fit(train)
    state_only = EventDetector("state_only").fit(train)

    populations: dict[str, Any] = {}
    for label, layouts in (("development", DEV_ALIAS), ("validation", m2d.ALIAS_LAYOUTS),
                           ("held_out", HELD_OUT_ALIAS), ("held_out_2", HELD_OUT_ALIAS_2)):
        population = m2d.build_population(layouts)
        features = m2d.RouteFeatures(population)
        populations[label] = {"population": population,
                              "tensors": build_tensors(population, features),
                              "strata": stratify(population),
                              "manifest": population_manifest(population, label, layouts)}
        m = populations[label]["manifest"]
        print(f"{label:14s} {m['rows']} rows / {m['pairs']} pairs / "
              f"{m['alias_classes']} classes  digest {m['member_digest']}", flush=True)

    arms = build_arms(selected_k)
    report: dict[str, Any] = {
        "generic_filter_arm": generic_key,
        "v3_passed": bool(passing),
        "selected_k": selected_k,
        "coupling_criterion": COUPLING_CRITERION,
        "manifests": {k: {kk: vv for kk, vv in v["manifest"].items()
                          if kk not in ("member_table", "member_routes")}
                      for k, v in populations.items()},
        "dev_seeds": list(DEV_SEEDS[:arguments.dev_seeds]),
        "validation_seeds": list(VALIDATION_SEEDS[:arguments.seeds])}

    # ---- section 7: choose the coupling rule on development -------------------------------
    dev = populations["development"]
    dev_seeds = DEV_SEEDS[:arguments.dev_seeds]
    print(f"\ncoupling rule selection on development alias layouts, {len(dev_seeds)} seeds")
    print(f"  {COUPLING_CRITERION}")
    print(f"{'rule':12s} {'alias':>8s} {'phase-NLL':>10s} {'Brier':>8s} {'ECE':>8s}")
    rules: dict[str, dict[str, float]] = {}
    dev_models = [core.train_arm(arms[generic_key], train, seed)[0] for seed in dev_seeds]
    for rule in ("hard", "posterior", "particle", "exact"):
        events = coupled_events(dev["tensors"], detector, rule)
        blocks = [score_block(model, dev["tensors"], events) for model in dev_models]
        rules[rule] = {k: float(np.mean([b[k] for b in blocks]))
                       for k in blocks[0] if not k.startswith("_")}
        print(f"{rule:12s} {rules[rule]['alias_accuracy']:8.4f} "
              f"{rules[rule]['phase_sensitive_nll']:10.4f} {rules[rule]['brier']:8.4f} "
              f"{rules[rule]['ece']:8.4f}", flush=True)
    eligible_rules = [r for r in ("hard", "posterior", "particle")]
    best_ece = min(rules[r]["ece"] for r in eligible_rules)
    allowed = [r for r in eligible_rules if rules[r]["ece"] <= best_ece + 0.02]
    selected_rule = min(allowed, key=lambda r: (rules[r]["phase_sensitive_nll"],
                                                -rules[r]["alias_accuracy"]))
    report["coupling_selection"] = {"rules": rules, "eligible": eligible_rules,
                                    "passed_calibration_constraint": allowed,
                                    "selected": selected_rule}
    print(f"selected coupling rule: {selected_rule}")

    # ---- section 9: the complete learned pathway ------------------------------------------
    seeds = VALIDATION_SEEDS[:arguments.seeds]
    pathway = {
        "1_learned_event_generic_filter": (generic_key, "learned"),
        "2_learned_event_exact_accumulator": ("A_exact_xor_accumulator", "learned"),
        "3_true_event_generic_filter": (generic_key, "exact"),
        "4_trained_memoryless": ("H_trained_memoryless", None),
        "5_generic_gru": ("G_generic_gru", "learned"),
        "6_detector_no_temporal_state": ("H_trained_memoryless", None),
        "7_m2d_answer_initialised_filter": ("B_answer_oriented_init", "learned"),
        "8_exact_event_exact_accumulator": ("A_exact_xor_accumulator", "exact"),
    }
    results: dict[str, Any] = {}
    print(f"\ncomplete learned coupling, {len(seeds)} untouched seeds, "
          f"rule={selected_rule}")
    # Training dominates and depends on neither the split nor the event source, so each
    # (arm, seed) is trained once and scored everywhere. Retraining per split would have
    # cost four times as much and, worse, would have let the arms drift apart between
    # splits for no reason other than the RNG.
    cache: dict[tuple[str, int], Any] = {}
    identities: dict[str, Any] = {}
    for arm_key in {k for k, _ in pathway.values()}:
        for seed in seeds:
            model, ledger, ident = core.train_arm(arms[arm_key], train, seed)
            cache[(arm_key, seed)] = model
            identities.setdefault(arm_key, ident)
    report["identities"] = identities
    for split in ("development", "validation", "held_out", "held_out_2"):
        block = populations[split]
        print(f"\n-- {split} --")
        print(f"{'arm':38s} {'alias':>8s} {'p10':>8s} {'NLL':>8s} {'ECE':>7s}")
        for name, (arm_key, source) in pathway.items():
            events = (None if source is None
                      else coupled_events(block["tensors"], detector,
                                          selected_rule if source == "learned" else "exact"))
            per_seed, blocks = [], []
            for seed in seeds:
                scored = score_block(cache[(arm_key, seed)], block["tensors"], events)
                per_seed.append(scored.pop("_hit"))
                blocks.append(scored)
            stacked = np.stack(per_seed)
            accuracy = np.array([b["alias_accuracy"] for b in blocks])
            summary = m2d.summarise_metric(accuracy)
            summary.update({k: float(np.mean([b[k] for b in blocks]))
                            for k in blocks[0]})
            summary["eligible"] = arms[arm_key].eligible
            results.setdefault(split, {})[name] = {"stats": summary, "hits": stacked}
            print(f"{name:38s} {summary['mean']:8.4f} {summary['p10']:8.4f} "
                  f"{summary['phase_sensitive_nll']:8.4f} {summary['ece']:7.4f}",
                  flush=True)

    # intervals per split
    print("\npaired hierarchical intervals vs trained memoryless")
    for split in results:
        block = populations[split]
        rows = len(block["population"].rows)
        seed_column = np.repeat(np.array(seeds), rows)
        layout_column = np.tile(block["strata"]["layout"], len(seeds))
        class_column = np.tile(block["strata"]["alias_class"], len(seeds))
        changes = np.tile(block["strata"]["changes"], len(seeds))
        baseline = results[split]["4_trained_memoryless"]["hits"].ravel()
        for name in results[split]:
            if name == "4_trained_memoryless":
                continue
            arm_hits = results[split][name]["hits"].ravel()
            entry = {"vs_memoryless": m2d.hierarchical_paired_interval(
                arm_hits, baseline, seed_column, layout_column, class_column)}
            for label, mask in (("changes_2plus", changes >= 2),
                                ("changes_4plus", changes >= 4)):
                entry[label] = m2d.hierarchical_paired_interval(
                    arm_hits, baseline, seed_column, layout_column, class_column,
                    mask=mask)
            results[split][name]["intervals"] = entry
        main_arm = results[split]["1_learned_event_generic_filter"]["intervals"]
        print(f"  {split:14s} learned+generic {main_arm['vs_memoryless']['delta']:+.4f} "
              f"[{main_arm['vs_memoryless']['ci_low']:+.4f}, "
              f"{main_arm['vs_memoryless']['ci_high']:+.4f}]"
              f"{' *' if main_arm['vs_memoryless']['excludes_zero'] else ''}   2+ "
              f"{main_arm['changes_2plus']['delta']:+.4f}"
              f"{' *' if main_arm['changes_2plus']['excludes_zero'] else ''}", flush=True)

    # ---- section 10: corruption -----------------------------------------------------------
    block = populations["validation"]
    rows = len(block["population"].rows)
    seed_column = np.repeat(np.array(seeds), rows)
    layout_column = np.tile(block["strata"]["layout"], len(seeds))
    class_column = np.tile(block["strata"]["alias_class"], len(seeds))
    baseline = results["validation"]["4_trained_memoryless"]["hits"].ravel()
    base_events = coupled_events(block["tensors"], detector, selected_rule)
    print("\nevent corruption, all derived from one frozen probability sequence")
    corruption: dict[str, Any] = {}
    rng = np.random.default_rng(909)
    extra = {
        "9_calibrated_random": np.clip(rng.permutation(base_events.ravel()).reshape(
            base_events.shape), 0.0, 1.0),
        "10_action_only_detector": coupled_events(block["tensors"], action_only,
                                                  selected_rule),
        "11_state_only_detector": coupled_events(block["tensors"], state_only,
                                                 selected_rule),
    }
    for control in list(CORRUPTIONS) + list(extra):
        events = (extra[control] if control in extra
                  else corrupt(control, base_events, block["tensors"].lengths, seed=515))
        per_seed = [score_population(cache[(generic_key, seed)], block["tensors"],
                                     events)["hit"] for seed in seeds]
        stacked = np.concatenate(per_seed)
        interval = m2d.hierarchical_paired_interval(
            stacked, baseline, seed_column, layout_column, class_column)
        corruption[control] = {"alias_accuracy": float(stacked.mean()), **interval}
        print(f"  {control:28s} {stacked.mean():8.4f}  {interval['delta']:+.4f} "
              f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
              f"{' *' if interval['excludes_zero'] else ''}", flush=True)
    report["corruption"] = corruption

    correct = corruption["1_correct"]["delta"]
    destroying = ("2_shift_forward", "3_shift_backward", "4_drop_one_event",
                  "5_flip_one_event", "6_cross_episode_shuffle",
                  "7_positionwise_permutation", "8_constant", "9_calibrated_random")
    report["v11_corruptions_remove_advantage"] = bool(
        correct > 0 and all(corruption[k]["delta"] < correct * 0.5 for k in destroying))
    report["v11_judged_on"] = list(destroying)

    report["pathway"] = {
        split: {name: {"stats": {k: v for k, v in entry["stats"].items()},
                       "intervals": entry.get("intervals")}
                for name, entry in block.items()}
        for split, block in results.items()}
    validation_main = results["validation"]["1_learned_event_generic_filter"]["intervals"]
    held_out_main = results["held_out"]["1_learned_event_generic_filter"]["intervals"]
    measured = {
        "v8_validation": bool(validation_main["vs_memoryless"]["ci_low"] > 0),
        "v9_two_changes": bool(validation_main["changes_2plus"]["ci_low"] > 0),
        "v10_held_out": bool(held_out_main["vs_memoryless"]["ci_low"] > 0)}
    report["section_9_measurements"] = measured
    # Section 9 is gated on a generic transition procedure passing section 4. It did not,
    # so these numbers are diagnostics and cannot qualify anything. They are run and
    # published because section 8 predicts a held-out failure and the prediction is worth
    # testing, not because the gate can be satisfied out of order.
    report["section_9_is_qualifying"] = bool(passing)
    for key, value in measured.items():
        report[key] = value if passing else None
    report["v8_v9_v10_status"] = ("MEASURED" if passing
                                  else "NOT_RUN as a gate: blocked by V3")

    frozen = ARTIFACTS / "m2e-coupling-predictions.npz"
    payload: dict[str, np.ndarray] = {"seeds": np.array(seeds)}
    for split, block in results.items():
        for name, entry in block.items():
            payload[f"hit::{split}::{name}"] = entry["hits"]
        payload[f"row_layout::{split}"] = populations[split]["strata"]["layout"]
        payload[f"row_class::{split}"] = populations[split]["strata"]["alias_class"]
        payload[f"row_changes::{split}"] = populations[split]["strata"]["changes"]
        payload[f"members::{split}"] = populations[split]["manifest"]["member_table"]
    report["frozen_predictions"] = {
        "path": str(frozen.relative_to(m2d.REPO)),
        "sha256_16": save_predictions(frozen, payload)}
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nV8 (validation): {report['v8_validation']}   "
          f"V9 (2+ changes): {report['v9_two_changes']}   "
          f"V10 (held-out layouts): {report['v10_held_out']}   "
          f"V11 (corruptions): {report['v11_corruptions_remove_advantage']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
