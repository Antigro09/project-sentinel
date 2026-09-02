"""4 / 5 / V3 / V4. Can a generic, semantics-free procedure learn the transition?

M2D's ceiling-reaching filter was initialised at event 0 -> stay, event 1 -> flip. Nothing
here is. Every eligible arm draws its perturbation from a seed with a random orientation,
and the one arm that keeps the answer-oriented matrix is marked `eligible=False` in the
Arm record so it cannot be selected by code rather than by promise.

The restart arm is the interesting one and it is also the one most easily fooled: a
procedure that trains K models and keeps the best has spent K times the compute, so it is
compared both per-model and under equal cumulative budget, and memoryless and GRU controls
are given the same K restarts and the same total updates.

K and the closeness margin are chosen on DEVELOPMENT seeds and frozen before any
validation seed is touched.

    .venv-shwm/bin/python experiments/shwm/m2e_transition.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as m2d
import m2e_core as core
from m2e_core import (ARTIFACTS, DEV_SEEDS, VALIDATION_SEEDS, Arm, ComputeLedger,
                      build_arms, episode_manifest, population_manifest, write)
from m2d_core import build_tensors, save_predictions, score_population, stratify
from structured_calibration import collect
from belief_factorization import build_dataset
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

# Preregistered, before any validation seed is opened.
K_CANDIDATES = (2, 4, 8, 16)
K_RULE = ("smallest K whose development alias p10 is within 0.01 of the best K; "
          "ties broken toward cheaper")
MARGIN_RULE = ("development gap to the exact accumulator, plus 0.02; frozen before "
               "validation")


def transition_statistics(model, train) -> dict[str, float]:
    """State occupancy and transition entropy, for filters that expose a belief."""
    import mlx.core as mx

    if not hasattr(model, "logits"):
        return {}
    transition = np.asarray(mx.softmax(model.logits, axis=-1))
    row_entropy = -(transition * np.log(np.maximum(transition, 1e-12))).sum(axis=-1)
    x, y, e, m, reset = m2d.pad(train)
    _, belief = model(mx.array(x), mx.array(reset), mx.array(e))
    mx.eval(belief)
    belief = np.asarray(belief)
    mask = m.astype(bool)
    used = np.unique(belief.argmax(axis=-1)[mask])
    entropy = -(belief * np.log(np.maximum(belief, 1e-12))).sum(axis=-1)
    states = belief.shape[-1]
    return {"transition_entropy": float(row_entropy.mean() / np.log(states)),
            "state_occupancy": float(len(used) / states),
            "normalised_belief_entropy": float(entropy[mask].mean() / np.log(states)),
            "stay_minus_flip_diagonal": float(
                transition[0].trace() - transition[1].trace()) if states == 2 else 0.0}


def evaluate(arm: Arm, train, tensors, population, seeds, updates=core.UPDATES):
    """Train and score one arm across seeds; returns per-seed hits and a merged ledger."""
    hits, records = [], []
    total = ComputeLedger()
    identity = None
    for seed in seeds:
        model, ledger, ident = core.train_arm(arm, train, seed, updates=updates)
        scored = score_population(model, tensors)
        assignment = m2d.fit_state_assignment(model, train)
        phase = float("nan")
        collapsed = False
        if assignment is not None and scored["belief"].shape[1] > 1:
            predicted = assignment[scored["belief"].argmax(axis=1)]
            truth = np.array([population.states[r.self_index].polarity
                              for r in population.rows])
            phase = float((predicted == truth).mean())
            entropy = -(scored["belief"] * np.log(np.maximum(
                scored["belief"], 1e-12))).sum(axis=1).mean()
            collapsed = bool(entropy / np.log(scored["belief"].shape[1]) > 0.9)
        hits.append(scored["hit"])
        record = {"seed": seed, "alias_accuracy": float(scored["hit"].mean()),
                  "nll": float(scored["nll"].mean()),
                  "brier": float(scored["brier"].mean()),
                  "margin": float(scored["margin"].mean()),
                  "phase_accuracy_up_to_permutation": phase, "collapsed": collapsed,
                  "compute": ledger.to_dict(), **ident}
        record.update(transition_statistics(model, train))
        records.append(record)
        total.merge(ledger)
        identity = identity or ident
    return np.stack(hits), records, total, identity


def summarise(records) -> dict[str, Any]:
    accuracy = np.array([r["alias_accuracy"] for r in records])
    out = m2d.summarise_metric(accuracy)
    for key in ("nll", "brier", "margin", "phase_accuracy_up_to_permutation",
                "transition_entropy", "state_occupancy", "stay_minus_flip_diagonal"):
        values = [r[key] for r in records if key in r and not np.isnan(r.get(key, np.nan))]
        if values:
            out[key] = float(np.mean(values))
    out["collapsed_seeds"] = int(sum(1 for r in records if r.get("collapsed")))
    out["seeds"] = len(records)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-seeds", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=len(VALIDATION_SEEDS))
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2e-transition.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    m2d.check_feature_layout()
    appearance = CANONICAL_APPEARANCE_SEED

    train_t = collect(list(m2d.TRAIN_LAYOUTS), 3, 9, appearance, 11)
    train = build_dataset(train_t, 5)
    population = m2d.build_population(m2d.ALIAS_LAYOUTS)
    features = m2d.RouteFeatures(population)
    tensors = build_tensors(population, features)
    strata = stratify(population)
    manifest = population_manifest(population, "validation_alias", m2d.ALIAS_LAYOUTS)
    episodes = episode_manifest(train, "train", m2d.TRAIN_LAYOUTS, 11)
    print(f"train {episodes['episodes']} episodes (digest {episodes['episode_digest']}); "
          f"alias {manifest['rows']} rows / {manifest['pairs']} pairs "
          f"(digest {manifest['member_digest']})", flush=True)

    report: dict[str, Any] = {
        "population_manifest": {k: v for k, v in manifest.items()
                                if k not in ("member_table", "member_routes")},
        "episode_manifest": {k: v for k, v in episodes.items() if k != "episode_table"},
        "dev_seeds": list(DEV_SEEDS[:arguments.dev_seeds]),
        "validation_seeds": list(VALIDATION_SEEDS[:arguments.seeds]),
        "k_candidates": list(K_CANDIDATES), "k_rule": K_RULE, "margin_rule": MARGIN_RULE,
        "development": {}, "validation": {}, "compute": {}}

    # ---- development: choose K, then freeze the closeness margin --------------------------
    dev_seeds = DEV_SEEDS[:arguments.dev_seeds]
    print(f"\ndevelopment ({len(dev_seeds)} seeds): choosing K by\n  {K_RULE}")
    print(f"{'K':>4s} {'alias p10':>10s} {'mean':>8s} {'updates':>10s}")
    dev_k: dict[int, dict[str, Any]] = {}
    for k in K_CANDIDATES:
        arm = build_arms(k)["E_generic_restarts"]
        hits, records, ledger, _ = evaluate(arm, train, tensors, population, dev_seeds)
        stats = summarise(records)
        dev_k[k] = {"stats": stats, "compute": ledger.to_dict()}
        print(f"{k:4d} {stats['p10']:10.4f} {stats['mean']:8.4f} "
              f"{ledger.optimizer_updates:10d}", flush=True)
    best = max(dev_k, key=lambda k: dev_k[k]["stats"]["p10"])
    selected_k = min(k for k in K_CANDIDATES
                     if dev_k[k]["stats"]["p10"] >= dev_k[best]["stats"]["p10"] - 0.01)
    report["development"]["k_search"] = {str(k): v for k, v in dev_k.items()}
    report["development"]["selected_k"] = selected_k
    print(f"selected K = {selected_k}")

    arms = build_arms(selected_k)
    dev_reference = {}
    for key in ("A_exact_xor_accumulator", "E_generic_restarts"):
        hits, records, ledger, _ = evaluate(arms[key], train, tensors, population,
                                            dev_seeds)
        dev_reference[key] = summarise(records)
    development_gap = (dev_reference["A_exact_xor_accumulator"]["mean"]
                       - dev_reference["E_generic_restarts"]["mean"])
    margin = float(development_gap + 0.02)
    report["development"]["gap_to_accumulator"] = float(development_gap)
    report["development"]["frozen_margin"] = margin
    report["development"]["reference"] = dev_reference
    print(f"development gap to the exact accumulator {development_gap:+.4f}; "
          f"frozen margin {margin:.4f}")

    # ---- validation: 20 untouched seeds ---------------------------------------------------
    seeds = VALIDATION_SEEDS[:arguments.seeds]
    print(f"\nvalidation on {len(seeds)} untouched seeds ({seeds[0]}-{seeds[-1]})")
    print(f"{'arm':30s} {'alias':>7s} {'p10':>7s} {'min':>7s} {'NLL':>7s} {'Brier':>7s} "
          f"{'phase':>7s} {'occ':>5s} {'Tent':>6s} {'coll':>5s} elig")
    print("-" * 108)
    stored: dict[str, np.ndarray] = {}
    for key, arm in arms.items():
        hits, records, ledger, identity = evaluate(arm, train, tensors, population, seeds)
        stats = summarise(records)
        stored[key] = hits
        report["validation"][key] = {"label": arm.label, "eligible": arm.eligible,
                                     "note": arm.note, "stats": stats,
                                     "records": records, "identity": identity}
        report["compute"][key] = ledger.to_dict()
        print(f"{key:30s} {stats['mean']:7.4f} {stats['p10']:7.4f} "
              f"{stats['minimum']:7.4f} {stats['nll']:7.4f} {stats['brier']:7.4f} "
              f"{stats.get('phase_accuracy_up_to_permutation', float('nan')):7.4f} "
              f"{stats.get('state_occupancy', float('nan')):5.2f} "
              f"{stats.get('transition_entropy', float('nan')):6.3f} "
              f"{stats['collapsed_seeds']:>2d}/{len(records):<2d} "
              f"{'yes' if arm.eligible else 'NO'}", flush=True)

    rows = len(population.rows)
    seed_column = np.repeat(np.array(seeds), rows)
    layout_column = np.tile(strata["layout"], len(seeds))
    class_column = np.tile(strata["alias_class"], len(seeds))
    changes = np.tile(strata["changes"], len(seeds))
    baseline = stored["H_trained_memoryless"].ravel()
    print("\npaired hierarchical intervals vs trained memoryless, and by phase changes")
    for key in arms:
        if key == "H_trained_memoryless":
            continue
        arm_hits = stored[key].ravel()
        interval = m2d.hierarchical_paired_interval(
            arm_hits, baseline, seed_column, layout_column, class_column)
        block = {"vs_memoryless": interval}
        for label, mask in (("changes_2", changes == 2), ("changes_3", changes == 3),
                            ("changes_4plus", changes >= 4), ("changes_2plus", changes >= 2)):
            block[label] = m2d.hierarchical_paired_interval(
                arm_hits, baseline, seed_column, layout_column, class_column, mask=mask)
        report["validation"][key]["intervals"] = block
        print(f"  {key:30s} {interval['delta']:+.4f} "
              f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
              f"{' *' if interval['excludes_zero'] else '  '}   2+ changes "
              f"{block['changes_2plus']['delta']:+.4f}"
              f"{' *' if block['changes_2plus']['excludes_zero'] else ''}", flush=True)

    # ---- equal cumulative compute ---------------------------------------------------------
    budget = selected_k * core.UPDATES
    print(f"\nequal cumulative compute: {selected_k} x {core.UPDATES} = {budget} updates")
    equal: dict[str, Any] = {}
    for key in ("C_zero_symmetric_init", "D_generic_random_single", "G_generic_gru",
                "H_trained_memoryless"):
        arm = arms[key]
        long_run = Arm(arm.key, arm.label, arm.spec_for, arm.eligible, restarts=1)
        saved = m2d.UPDATES
        m2d.UPDATES = budget
        try:
            hits, records, ledger, _ = evaluate(long_run, train, tensors, population,
                                                seeds, updates=budget)
        finally:
            m2d.UPDATES = saved
        equal[f"{key}__single_run_{budget}_updates"] = {
            "stats": summarise(records), "compute": ledger.to_dict()}
        print(f"  {key:30s} single run, {budget} updates: "
              f"{summarise(records)['mean']:.4f} "
              f"(p10 {summarise(records)['p10']:.4f})", flush=True)
    for key in ("G_generic_gru", "H_trained_memoryless"):
        arm = arms[key]
        restarts = Arm(arm.key, arm.label, arm.spec_for, arm.eligible,
                       restarts=selected_k)
        hits, records, ledger, _ = evaluate(restarts, train, tensors, population, seeds)
        equal[f"{key}__{selected_k}_restarts"] = {"stats": summarise(records),
                                                  "compute": ledger.to_dict()}
        print(f"  {key:30s} {selected_k} restarts: {summarise(records)['mean']:.4f} "
              f"(p10 {summarise(records)['p10']:.4f})", flush=True)
    report["equal_cumulative_compute"] = equal

    # ---- V3 -------------------------------------------------------------------------------
    eligible = {k: v for k, v in report["validation"].items() if v["eligible"]}
    accumulator = report["validation"]["A_exact_xor_accumulator"]["stats"]["mean"]
    verdicts = {}
    for key, block in eligible.items():
        if key == "H_trained_memoryless":
            continue
        stats = block["stats"]
        intervals = block["intervals"]
        verdicts[key] = {
            "beats_memoryless": bool(intervals["vs_memoryless"]["ci_low"] > 0),
            "within_frozen_margin": bool(accumulator - stats["mean"] <= margin),
            "survives_2_3_4plus": bool(all(
                intervals[f"changes_{s}"]["ci_low"] > 0
                for s in ("2", "3", "4plus"))),
            "lower_tail_stable": bool(stats["p10"] >= stats["mean"] - 0.05),
            "no_collapse": bool(stats["collapsed_seeds"] == 0)}
        verdicts[key]["passes_v3"] = bool(all(verdicts[key].values()))
    report["v3_verdicts"] = verdicts
    report["v3_generic_transition_learned"] = bool(
        any(v["passes_v3"] for v in verdicts.values()))
    report["v3_passing_arms"] = [k for k, v in verdicts.items() if v["passes_v3"]]

    frozen = ARTIFACTS / "m2e-transition-predictions.npz"
    report["frozen_predictions"] = {
        "path": str(frozen.relative_to(m2d.REPO)),
        "sha256_16": save_predictions(frozen, {
            **{f"hit::{k}": v for k, v in stored.items()},
            "seeds": np.array(seeds), "row_layout": strata["layout"],
            "row_alias_class": strata["alias_class"], "row_changes": strata["changes"],
            "member_table": manifest["member_table"],
            "episode_table": episodes["episode_table"]})}
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nV3 (a generic procedure learns the environmental transition): "
          f"{report['v3_generic_transition_learned']}  {report['v3_passing_arms']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
