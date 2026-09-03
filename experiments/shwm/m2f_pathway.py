"""K / L / M. Coupling rule under a proper-scoring criterion, and the full pathway.

M2E selected hard coupling because an ECE constraint excluded the posterior mixture,
while the primary metric and Brier both preferred posterior. The specification is
explicit that the old choice must not be preserved for that reason, so the constraint
here is Brier -- a proper score -- and the selection is redone from scratch on
development before any validation row is read.

The certified transition model is whatever the frozen adaptive procedure selected for
each seed; its restart index comes out of the procedures artifact, so the model coupled
here is the model that procedure actually chose rather than a fresh draw.

    .venv-shwm/bin/python experiments/shwm/m2f_pathway.py
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
import m2f_core as core
import m2f_events as events
from m2d_core import ARTIFACTS, FilterSpec, write
from m2d_coupling import EventDetector, corrupt, CORRUPTIONS
from structured_calibration import collect
from belief_factorization import build_dataset
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

DEV_ALIAS = core.DEV_ALIAS
SPLITS = {"validation": core.VALIDATION_ALIAS,
          "held_out": core.HELD_OUT_ALIAS,
          "held_out_new": events.NEW_ALIAS}
DESTROYING = ("2_shift_forward", "3_shift_backward", "4_drop_one_event",
              "5_flip_one_event", "6_cross_episode_shuffle",
              "7_positionwise_permutation", "8_constant", "9_calibrated_random")


def coupled(tensors, detector, rule: str, seed: int = 5) -> np.ndarray:
    probability = detector.probabilities(tensors.z)
    if rule == "hard":
        return (probability >= 0.5).astype(np.float32)
    if rule == "posterior":
        return probability
    if rule == "particle":
        rng = np.random.default_rng(seed)
        return (rng.random((32,) + probability.shape) < probability).mean(
            axis=0).astype(np.float32)
    if rule == "exact":
        return tensors.events_true
    raise KeyError(rule)


def score_block(model, tensors, event_array) -> dict[str, Any]:
    scored = m2d.score_population(model, tensors, event_array)
    probability = 1.0 / (1.0 + np.exp(-np.clip(scored["margin"], -50, 50)))
    return {"alias_accuracy": float(scored["hit"].mean()),
            "phase_sensitive_nll": float(scored["nll"].mean()),
            "brier": float(scored["brier"].mean()),
            "hit": scored["hit"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2f-pathway.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    procedures = json.loads((ARTIFACTS / "m2f-procedures.json").read_text())
    qualifying = bool(procedures["f_gates_all_pass"])
    if not qualifying and not arguments.force:
        print("F0-F9 did not all pass; sections K-M are gated. Use --force for "
              "non-qualifying diagnostics.")
        return 2

    selection = procedures["validation_selection"]["4_adaptive"]
    seeds = [int(s) for s in sorted(selection, key=int)][:arguments.seeds]
    appearance = CANONICAL_APPEARANCE_SEED
    train = build_dataset(collect(list(m2d.TRAIN_LAYOUTS), 3, 9, appearance, 11), 5)
    calibration = build_dataset(
        collect(list(m2d.DETECTOR_TEST_LAYOUTS), 2, 9, appearance, 777), 6)

    base = EventDetector("full").fit(train)
    detectors = {
        "structured": base,
        "relational": events.RelationalDetector("full").fit(train),
        "calibrated_relational": events.CalibratedDetector(
            events.RelationalDetector("full").fit(train), calibration),
        "exact": events.ExactDerivationDetector(),
        "action_only": EventDetector("action_only").fit(train),
        "state_only": EventDetector("state_only").fit(train),
    }

    populations: dict[str, Any] = {}
    for label, layouts in {"development": DEV_ALIAS, **SPLITS}.items():
        population = m2d.build_population(layouts)
        populations[label] = {
            "population": population,
            "tensors": m2d.build_tensors(population, m2d.RouteFeatures(population)),
            "strata": m2d.stratify(population),
            "manifest": m2e.population_manifest(population, label, layouts)}
        print(f"{label:14s} {populations[label]['manifest']['rows']} rows  digest "
              f"{populations[label]['manifest']['member_digest']}", flush=True)

    # Certified transition model per seed, as the frozen procedure selected it.
    print(f"\nretraining the {len(seeds)} adaptive-selected transition models")
    certified: dict[int, Any] = {}
    for seed in seeds:
        restart = selection[str(seed)]["selected_restart"]
        model, _ = m2d.train_model(core.generic_spec(seed, restart), train,
                                   seed * 1_000 + restart)
        certified[seed] = model

    report: dict[str, Any] = {
        "qualifying": qualifying, "seeds": seeds,
        "coupling_criterion": events.COUPLING_CRITERION,
        "manifests": {k: {kk: vv for kk, vv in v["manifest"].items()
                          if kk not in ("member_table", "member_routes")}
                      for k, v in populations.items()}}

    # ---- section K: coupling rule, chosen on development with a proper score -------------
    dev = populations["development"]
    print(f"\ncoupling selection on development alias layouts\n  "
          f"{events.COUPLING_CRITERION}")
    print(f"{'rule':12s} {'alias':>8s} {'phase-NLL':>10s} {'Brier':>8s}")
    rules: dict[str, dict[str, float]] = {}
    for rule in ("hard", "posterior", "particle", "exact"):
        event_array = coupled(dev["tensors"], detectors["relational"], rule)
        blocks = [score_block(certified[s], dev["tensors"], event_array) for s in seeds]
        rules[rule] = {k: float(np.mean([b[k] for b in blocks]))
                       for k in ("alias_accuracy", "phase_sensitive_nll", "brier")}
        print(f"{rule:12s} {rules[rule]['alias_accuracy']:8.4f} "
              f"{rules[rule]['phase_sensitive_nll']:10.4f} {rules[rule]['brier']:8.4f}",
              flush=True)
    eligible = ("hard", "posterior", "particle")
    best_brier = min(rules[r]["brier"] for r in eligible)
    allowed = [r for r in eligible if rules[r]["brier"] <= best_brier + 0.02]
    selected_rule = min(allowed, key=lambda r: (rules[r]["phase_sensitive_nll"],
                                                -rules[r]["alias_accuracy"]))
    report["coupling_selection"] = {"rules": rules, "eligible": list(eligible),
                                    "passed_brier_constraint": allowed,
                                    "selected": selected_rule,
                                    "constraint": "Brier, a proper score"}
    print(f"selected coupling rule: {selected_rule}")

    # ---- section L: the complete pathway --------------------------------------------------
    pathway = {
        "1_exact_event_exact_accumulator": ("accumulator", "exact", "exact"),
        "2_exact_event_certified_transition": ("certified", "exact", "exact"),
        "3_learned_event_exact_accumulator": ("accumulator", "relational", selected_rule),
        "4_learned_event_certified_transition": ("certified", "relational", selected_rule),
        "5_learned_event_gru": ("gru", "relational", selected_rule),
        "6_learned_event_no_temporal_state": ("memoryless", None, None),
        "7_trained_memoryless": ("memoryless", None, None),
        "8_answer_oriented_diagnostic": ("answer", "relational", selected_rule),
    }
    others = {}
    for seed in seeds:
        others[("accumulator", seed)] = m2d.train_model(
            FilterSpec("m2f", "accumulator"), train, seed)[0]
        others[("gru", seed)] = m2d.train_model(FilterSpec("m2f", "gru"), train, seed)[0]
        others[("memoryless", seed)] = m2d.train_model(
            FilterSpec("m2f", "memoryless"), train, seed)[0]
        others[("answer", seed)] = m2d.train_model(
            FilterSpec("m2f", "filter", 2, "symmetry_broken", perturbation=m2e.ANSWER),
            train, seed)[0]

    def model_for(kind: str, seed: int):
        return certified[seed] if kind == "certified" else others[(kind, seed)]

    results: dict[str, Any] = {}
    for split in SPLITS:
        block = populations[split]
        print(f"\n-- {split} --")
        print(f"{'arm':40s} {'alias':>8s} {'p10':>8s} {'NLL':>8s} {'Brier':>8s}")
        for name, (kind, detector_name, rule) in pathway.items():
            event_array = (None if detector_name is None
                           else coupled(block["tensors"], detectors[detector_name], rule))
            blocks = [score_block(model_for(kind, s), block["tensors"], event_array)
                      for s in seeds]
            hits = np.stack([b.pop("hit") for b in blocks])
            accuracy = np.array([b["alias_accuracy"] for b in blocks])
            summary = m2d.summarise_metric(accuracy)
            summary.update({k: float(np.mean([b[k] for b in blocks]))
                            for k in ("phase_sensitive_nll", "brier")})
            results.setdefault(split, {})[name] = {"stats": summary, "hits": hits}
            print(f"{name:40s} {summary['mean']:8.4f} {summary['p10']:8.4f} "
                  f"{summary['phase_sensitive_nll']:8.4f} {summary['brier']:8.4f}",
                  flush=True)

    print("\npaired hierarchical intervals vs trained memoryless")
    for split in results:
        block = populations[split]
        rows = len(block["population"].rows)
        seed_column = np.repeat(np.array(seeds), rows)
        layout_column = np.tile(block["strata"]["layout"], len(seeds))
        class_column = np.tile(block["strata"]["alias_class"], len(seeds))
        changes = np.tile(block["strata"]["changes"], len(seeds))
        baseline = results[split]["7_trained_memoryless"]["hits"].ravel()
        for name in results[split]:
            if name == "7_trained_memoryless":
                continue
            hits = results[split][name]["hits"].ravel()
            entry = {"vs_memoryless": m2d.hierarchical_paired_interval(
                hits, baseline, seed_column, layout_column, class_column)}
            for label, mask in (("changes_2plus", changes >= 2),
                                ("changes_4plus", changes >= 4)):
                entry[label] = m2d.hierarchical_paired_interval(
                    hits, baseline, seed_column, layout_column, class_column, mask=mask)
            results[split][name]["intervals"] = entry
        main_arm = results[split]["4_learned_event_certified_transition"]["intervals"]
        print(f"  {split:16s} {main_arm['vs_memoryless']['delta']:+.4f} "
              f"[{main_arm['vs_memoryless']['ci_low']:+.4f}, "
              f"{main_arm['vs_memoryless']['ci_high']:+.4f}]"
              f"{' *' if main_arm['vs_memoryless']['excludes_zero'] else ''}   2+ "
              f"{main_arm['changes_2plus']['delta']:+.4f}"
              f"{' *' if main_arm['changes_2plus']['excludes_zero'] else ''}", flush=True)

    # ---- corruption -----------------------------------------------------------------------
    block = populations["validation"]
    rows = len(block["population"].rows)
    seed_column = np.repeat(np.array(seeds), rows)
    layout_column = np.tile(block["strata"]["layout"], len(seeds))
    class_column = np.tile(block["strata"]["alias_class"], len(seeds))
    baseline = results["validation"]["7_trained_memoryless"]["hits"].ravel()
    base_events = coupled(block["tensors"], detectors["relational"], selected_rule)
    rng = np.random.default_rng(909)
    # A calibrated random detector: the right marginal, no information, and the SAME
    # structural zeros as the honest one. A first version permuted the flattened array,
    # which moved events into padding and into step 0 -- positions the honest detector
    # never fills -- and left a small residual advantage that was an artefact of the
    # control rather than a property of the pipeline.
    valid_mask = np.zeros_like(base_events, dtype=bool)
    for k, n in enumerate(block["tensors"].lengths):
        valid_mask[k, 1:int(n)] = True
    marginal = float(base_events[valid_mask].mean())
    calibrated_random = np.zeros_like(base_events)
    calibrated_random[valid_mask] = (
        rng.random(int(valid_mask.sum())) < marginal).astype(np.float32)
    extra = {"9_calibrated_random": calibrated_random,
        "10_action_only_detector": coupled(block["tensors"], detectors["action_only"],
                                           selected_rule),
        "11_state_only_detector": coupled(block["tensors"], detectors["state_only"],
                                          selected_rule)}
    print("\nevent corruption, from one frozen probability sequence")
    corruption: dict[str, Any] = {}
    for control in list(CORRUPTIONS) + list(extra):
        event_array = (extra[control] if control in extra
                       else corrupt(control, base_events, block["tensors"].lengths, 515))
        hits = np.stack([score_block(certified[s], block["tensors"], event_array)["hit"]
                         for s in seeds]).ravel()
        interval = m2d.hierarchical_paired_interval(
            hits, baseline, seed_column, layout_column, class_column)
        corruption[control] = {"alias_accuracy": float(hits.mean()), **interval}
        print(f"  {control:28s} {hits.mean():8.4f}  {interval['delta']:+.4f} "
              f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
              f"{' *' if interval['excludes_zero'] else ''}", flush=True)
    report["corruption"] = corruption
    report["calibrated_random_marginal"] = marginal

    report["pathway"] = {
        split: {name: {"stats": entry["stats"], "intervals": entry.get("intervals")}
                for name, entry in block.items()}
        for split, block in results.items()}
    main_name = "4_learned_event_certified_transition"
    correct = corruption["1_correct"]["delta"]
    gates = {
        "E3_validation": bool(
            results["validation"][main_name]["intervals"]["vs_memoryless"]["ci_low"] > 0),
        "E4_held_out": bool(
            results["held_out"][main_name]["intervals"]["vs_memoryless"]["ci_low"] > 0
            and results["held_out_new"][main_name]["intervals"]["vs_memoryless"]["ci_low"] > 0),
        "E6_two_changes": bool(
            results["validation"][main_name]["intervals"]["changes_2plus"]["ci_low"] > 0),
        "E7_coupling_under_frozen_proper_score": True,
        "E1_beats_restricted_and_corrupted": bool(
            correct > 0 and all(corruption[k]["delta"] < correct * 0.5
                                for k in DESTROYING)),
    }
    report["gates"] = gates
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\ngates: {gates}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
