"""C. Is the symmetry-breaking rule generic, or is it the answer in disguise?

The M2C rule adds a fixed perturbation to the transition logits and its docstring
claimed the perturbation "encodes no phase semantics and no XOR structure". Printing
the matrix it produces refutes that in one line: event 0 initialises at

    [[0.73, 0.27], [0.27, 0.73]]      -- stay

and event 1 at

    [[0.27, 0.73], [0.73, 0.27]]      -- flip

which is the XOR automaton at 73% confidence before a single gradient step. So the
question this module has to answer is not "does symmetry breaking help" but "does
symmetry breaking of a magnitude matched to that one help when it does NOT point at
stay/flip". The controls are built to separate those.

    5_random_antisymmetric  same Frobenius norm, same event-antisymmetry, random
                            orientation -- breaks interchangeability, names no automaton
    5b_random_independent   same norm, two independent draws -- breaks interchangeability
                            without even antisymmetry
    2_sign_reversed         the negated perturbation, which points at the WRONG automaton

If only the original orientation reaches the ceiling, the automaton was authored and
U3 has to be restated. Every arm is scored on the twenty untouched validation seeds.

    .venv-shwm/bin/python experiments/shwm/m2d_symmetry.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as core
from m2d_core import (ARTIFACTS, FilterSpec, antisymmetric_two_state, build_population,
                      checkpoint_hash, held_out_accuracy, summarise_metric, train_model,
                      write)
from structured_calibration import collect
from belief_factorization import build_dataset
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

VALIDATION_SEEDS = tuple(range(8000, 8020))
NORM = float(np.linalg.norm(antisymmetric_two_state()))


def matched_random(seed: int, states: int, antisymmetric: bool) -> np.ndarray:
    """A perturbation with the same Frobenius norm as the frozen one, random orientation."""
    rng = np.random.default_rng(20_000 + seed)
    if antisymmetric:
        first = rng.normal(size=(states, states))
        block = np.stack([first, -first])
    else:
        block = rng.normal(size=(2, states, states))
    return (block / np.linalg.norm(block) * NORM).astype(np.float32)


def specs_for(seed: int) -> dict[str, FilterSpec]:
    anti = antisymmetric_two_state()
    swap = np.array([[[0, 1], [1, 0]]])            # relabel the two latent states
    permuted = np.stack([swap[0].T @ anti[e] @ swap[0] for e in range(2)]).astype(np.float32)
    return {
        "1_original": FilterSpec("1_original", "filter", 2, "symmetry_broken",
                                 perturbation=anti),
        "2_sign_reversed": FilterSpec("2_sign_reversed", "filter", 2, "sign_reversed",
                                      perturbation=-anti),
        "3a_state_permutation": FilterSpec("3a_state_permutation", "filter", 2, "permuted",
                                           perturbation=permuted),
        "3b_gauge_permuted": FilterSpec("3b_gauge_permuted", "filter", 2, "symmetry_broken",
                                        gauge="reset_onehot_swapped", perturbation=anti),
        "3c_gauge_learned": FilterSpec("3c_gauge_learned", "filter", 2, "symmetry_broken",
                                       gauge="learned", perturbation=anti),
        "4_event_label_permuted": FilterSpec("4_event_label_permuted", "filter", 2,
                                             "symmetry_broken", perturbation=anti),
        "4b_event_permuted_matched_init": FilterSpec(
            "4b_event_permuted_matched_init", "filter", 2, "symmetry_broken",
            perturbation=anti[::-1]),
        "5_random_antisymmetric": FilterSpec("5_random_antisymmetric", "filter", 2,
                                             "random_antisymmetric",
                                             perturbation=matched_random(seed, 2, True)),
        "5b_random_independent": FilterSpec("5b_random_independent", "filter", 2,
                                            "random_independent",
                                            perturbation=matched_random(seed + 7, 2, False)),
        "6_zero_symmetry_breaking": FilterSpec("6_zero_symmetry_breaking", "filter", 2,
                                               "zero"),
        "7_eight_state_overcomplete": FilterSpec("7_eight_state_overcomplete", "filter", 8,
                                                 "symmetry_broken",
                                                 perturbation=matched_random(seed, 8, True)),
        # EXPLORATORY, and ineligible for selection or for any gate. It is added after
        # validation exposure, which the M2C rule forbids for a candidate; it is kept
        # because the question it answers -- can a procedure that names no automaton
        # find one -- is the question the failure of 5_random_antisymmetric raises, and
        # restart selection on TRAINING loss uses no privileged information.
        "8_random_restarts_exploratory": FilterSpec(
            "8_random_restarts_exploratory", "filter", 2, "random_antisymmetric",
            perturbation=matched_random(seed, 2, True)),
    }


def best_of_restarts(spec: FilterSpec, train, seed: int, restarts: int = 8):
    """Eight matched-magnitude random orientations, kept by TRAINING loss alone.

    No phase label, no held-out set and no automaton enters the choice, so if this
    reaches the ceiling then a generic procedure -- not an authored orientation -- can
    find the transition. Exploratory only; it is not eligible for any gate.
    """
    import mlx.core as mx
    import mlx.nn as nn

    x, y, e, m, reset = core.pad(train)
    best = None
    for restart in range(restarts):
        candidate = FilterSpec(spec.arm_id, "filter", spec.states, "random_antisymmetric",
                               perturbation=matched_random(seed * 31 + restart,
                                                           spec.states, True))
        model, count = train_model(candidate, train, seed * 101 + restart)
        logits, _ = model(mx.array(x), mx.array(reset), mx.array(e))
        losses = nn.losses.cross_entropy(logits.reshape(-1, core.CLASSES),
                                         mx.array(y).reshape(-1), reduction="none")
        loss = float((losses * mx.array(m).reshape(-1)).sum() / mx.array(m).sum())
        if best is None or loss < best[0]:
            best = (loss, model, count)
    return best[1], best[2]


def phase_accuracy_up_to_permutation(model, train, test) -> dict[str, float]:
    """Latent states are anonymous, so the mapping to polarity is fitted on TRAIN and
    then applied to TEST. Fitting it on the evaluation set would score the alignment."""
    import mlx.core as mx

    def beliefs_of(items):
        x, y, e, m, reset = core.pad(items)
        _, belief = model(mx.array(x), mx.array(reset), mx.array(e))
        if belief is None:
            return None, None, None
        mx.eval(belief)
        phases = np.zeros_like(y)
        for i, item in enumerate(items):
            phases[i, :len(item["phases"])] = item["phases"]
        return np.asarray(belief), phases, m.astype(bool)

    belief, phases, mask = beliefs_of(train)
    if belief is None:
        return {}
    states = belief.shape[-1]
    assignment = np.zeros(states, dtype=int)
    argmax = belief.argmax(axis=-1)
    for s in range(states):
        picked = mask & (argmax == s)
        assignment[s] = int(round(float(phases[picked].mean()))) if picked.any() else 0

    belief_t, phases_t, mask_t = beliefs_of(test)
    predicted = assignment[belief_t.argmax(axis=-1)]
    entropy = -(belief_t * np.log(np.maximum(belief_t, 1e-12))).sum(axis=-1)
    normalised = float(entropy[mask_t].mean() / np.log(states))
    return {"phase_accuracy_up_to_permutation": float(
        (predicted[mask_t] == phases_t[mask_t]).mean()),
        "normalised_belief_entropy": normalised,
        "collapsed": bool(normalised > 0.9),
        "distinct_states_used": int(len(np.unique(belief_t.argmax(axis=-1)[mask_t])))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(VALIDATION_SEEDS))
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2d-symmetry.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    train = build_dataset(collect(list(core.TRAIN_LAYOUTS), 3, 9,
                                  CANONICAL_APPEARANCE_SEED, 11), 5)
    test = build_dataset(collect(list(core.DETECTOR_TEST_LAYOUTS), 2, 9,
                                 CANONICAL_APPEARANCE_SEED, 777), 6)
    seeds = VALIDATION_SEEDS[:arguments.seeds]
    print(f"train {len(train)} trajectories, held-out {len(test)}; TRUE events; "
          f"{len(seeds)} untouched validation seeds", flush=True)
    print(f"frozen perturbation Frobenius norm {NORM:.4f}; every random control is "
          f"matched to it\n", flush=True)

    report: dict[str, Any] = {"validation_seeds": list(seeds), "perturbation_norm": NORM,
                              "arms": {}}
    names = list(specs_for(seeds[0]))
    print(f"{'arm':30s} {'held-out':>9s} {'p10':>7s} {'min':>7s} {'phase|perm':>11s} "
          f"{'collapsed':>10s}")
    print("-" * 82)
    for name in names:
        records = []
        for seed in seeds:
            spec = specs_for(seed)[name]
            permuted_events = name.startswith(("4_", "4b_"))
            transform = (lambda e: 1.0 - e) if permuted_events else None
            if name == "8_random_restarts_exploratory":
                model, count = best_of_restarts(spec, train, seed)
            else:
                model, count = train_model(spec, train, seed, event_transform=transform)
            evaluation = test
            if permuted_events:
                evaluation = [dict(item, events=1.0 - np.asarray(item["events"]))
                              for item in test]
            record = {"seed": seed, "accuracy": held_out_accuracy(model, evaluation),
                      "parameters": count, "checkpoint": checkpoint_hash(model)}
            record.update(phase_accuracy_up_to_permutation(
                model, train if not permuted_events
                else [dict(i, events=1.0 - np.asarray(i["events"])) for i in train],
                evaluation))
            records.append(record)
        accuracies = np.array([r["accuracy"] for r in records])
        phase = np.array([r.get("phase_accuracy_up_to_permutation", np.nan)
                          for r in records])
        stats = summarise_metric(accuracies)
        stats["phase_up_to_permutation_mean"] = float(np.nanmean(phase))
        stats["phase_up_to_permutation_p10"] = float(np.nanpercentile(phase, 10))
        stats["collapsed_seeds"] = int(sum(1 for r in records if r.get("collapsed")))
        report["arms"][name] = {"stats": stats, "records": records,
                                "initialization_rule": specs_for(seeds[0])[name]
                                .initialization_rule}
        print(f"{name:30s} {stats['mean']:9.4f} {stats['p10']:7.4f} "
              f"{stats['minimum']:7.4f} {stats['phase_up_to_permutation_mean']:11.4f} "
              f"{stats['collapsed_seeds']:>4d}/{len(records):<5d}", flush=True)

    original = report["arms"]["1_original"]["stats"]
    generic = report["arms"]["5_random_antisymmetric"]["stats"]
    report["c2_symmetry_breaking_is_generic"] = bool(generic["p10"] >= original["p10"] - 0.02)
    report["c2_orientation_invariant"] = bool(
        report["arms"]["2_sign_reversed"]["stats"]["p10"] >= original["p10"] - 0.02)
    report["c2_permutation_invariant"] = bool(
        report["arms"]["3a_state_permutation"]["stats"]["p10"] >= original["p10"] - 0.02
        and report["arms"]["3b_gauge_permuted"]["stats"]["p10"] >= original["p10"] - 0.02)
    report["c2_event_relabelling_invariant"] = bool(
        report["arms"]["4_event_label_permuted"]["stats"]["p10"] >= original["p10"] - 0.02)
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nC2 generic (matched random matches the frozen orientation): "
          f"{report['c2_symmetry_breaking_is_generic']}")
    print(f"C2 orientation-invariant: {report['c2_orientation_invariant']}")
    print(f"C2 permutation-invariant: {report['c2_permutation_invariant']}")
    print(f"C2 event-relabelling-invariant: {report['c2_event_relabelling_invariant']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
