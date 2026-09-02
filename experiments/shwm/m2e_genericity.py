"""3 / V2. Is every eligible initialisation actually generic?

M2D's rule failed this audit after the fact, so the audit runs first here. Three of the
eight checks are distributional rather than per-seed, because "contains no XOR structure"
is a property of the FAMILY a draw comes from, not of any one draw: a random orientation
will sometimes land near the answer, and that is not a leak. What would be a leak is a
family whose draws are closer to the answer than two random draws are to each other.

The last check is the one M2D could not have failed but which is worth pinning anyway:
restart selection sees `(arm, train, seed)` and nothing else, so no validation score can
reach it. That is asserted from the call signature and the syntax tree, not from prose.

    .venv-shwm/bin/python experiments/shwm/m2e_genericity.py
"""

from __future__ import annotations

import argparse
import ast
import inspect
import time
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as m2d
import m2e_core as core
from m2e_core import (ANSWER, ARTIFACTS, DEV_SEEDS, distances, generic_antisymmetric,
                      initialisation_digest, write)
from m2d_core import FilterSpec, build_tensors, score_population
from structured_calibration import collect
from belief_factorization import build_dataset
from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED

DRAWS = 400
SEEDS = DEV_SEEDS[:20]


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def family_statistics() -> dict[str, Any]:
    """Checks 5 and 6: the family names no orientation and no event label."""
    draws = [generic_antisymmetric(20_000 + i, 2) for i in range(DRAWS)]
    to_answer = np.array([distances(d)["to_answer_orientation"] for d in draws])
    diagonal = np.array([distances(d)["stay_minus_flip_diagonal"] for d in draws])
    between = np.array([np.linalg.norm(softmax(draws[i].astype(float))
                                       - softmax(draws[i + 1].astype(float)))
                        for i in range(0, DRAWS - 1, 2)])
    # Check 6: reversing the event axis maps the family onto itself, so the distribution
    # of any orientation statistic must be symmetric about zero.
    reversed_diagonal = np.array([
        distances(np.ascontiguousarray(d[::-1]))["stay_minus_flip_diagonal"]
        for d in draws])
    return {
        "draws": DRAWS,
        "mean_distance_to_answer": float(to_answer.mean()),
        "mean_distance_between_two_draws": float(between.mean()),
        "closer_to_answer_than_to_each_other": bool(
            to_answer.mean() < between.mean()),
        "mean_stay_minus_flip_diagonal": float(diagonal.mean()),
        "sd_stay_minus_flip_diagonal": float(diagonal.std()),
        "fraction_oriented_towards_stay_flip": float((diagonal > 0).mean()),
        "answer_stay_minus_flip_diagonal": float(distances(ANSWER)["stay_minus_flip_diagonal"]),
        "event_axis_reversal_is_measure_preserving": bool(
            abs(diagonal.mean() + reversed_diagonal.mean()) < 1e-9),
        "check_5_no_stay_swap_xor_in_initialisation": bool(
            abs(diagonal.mean()) < 0.1 and 0.4 < (diagonal > 0).mean() < 0.6),
        "check_6_no_event_name_dependence": bool(
            abs(diagonal.mean() + reversed_diagonal.mean()) < 1e-9),
    }


def static_checks() -> dict[str, Any]:
    """Checks 4, 7 and 8, from signatures and the syntax tree."""
    source = inspect.getsource(generic_antisymmetric)
    tree = ast.parse(inspect.getsource(core))
    forbidden = {"polarity", "phase", "switch_crossings", "answer", "ANSWER",
                 "stay", "flip", "xor", "XOR"}
    names_in_init = {n.id for n in ast.walk(ast.parse(source))
                     if isinstance(n, ast.Name)}
    selection = inspect.signature(core.train_arm).parameters
    likelihood_source = inspect.getsource(core.training_log_likelihood)
    validation_names = {n.id for n in ast.walk(ast.parse(likelihood_source))
                        if isinstance(n, ast.Name)}
    return {
        "check_4_no_phase_labels_in_initialisation": bool(
            not (names_in_init & forbidden)),
        "initialisation_free_names": sorted(names_in_init),
        "check_7_no_transition_target_serialised": bool(
            all(initialisation_digest(generic_antisymmetric(s, 2))
                != initialisation_digest(ANSWER) for s in range(20_000, 20_050))),
        "check_8_selection_inputs": sorted(selection),
        "check_8_no_validation_in_selection": bool(
            not any(("valid" in n) or ("test" in n) for n in validation_names)
            and set(selection) <= {"arm", "train", "seed", "updates",
                                   "event_transform"}),
    }


def paired_seed_interval(a: np.ndarray, b: np.ndarray, resamples: int = 4000,
                         seed: int = 3) -> dict[str, float]:
    """Bootstrap the per-seed difference.

    A bare tolerance was the first version and it was a bad instrument: a single generic
    draw solves roughly 40% of seeds, so the mean over twenty seeds carries a standard
    error near 0.05 and a "delta below 0.10" test cannot distinguish a real asymmetry
    from noise. An interval says which of those it is.
    """
    difference = a - b
    rng = np.random.default_rng(seed)
    draws = np.array([difference[rng.integers(0, len(difference), len(difference))].mean()
                      for _ in range(resamples)])
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"delta": float(difference.mean()), "ci_low": low, "ci_high": high,
            "excludes_zero": bool(low > 0 or high < 0)}


def trained_invariances(train, tensors, population) -> dict[str, Any]:
    """Checks 1, 2 and 3, on trained models rather than on initialisations."""
    out: dict[str, Any] = {}

    def run(spec_for, event_transform=None, events=None):
        scores = []
        for seed in SEEDS:
            spec = spec_for(seed)
            model, _ = m2d.train_model(spec, train, seed, event_transform=event_transform)
            scores.append(float(score_population(model, tensors, events)["hit"].mean()))
        return np.array(scores)

    base = run(lambda s: FilterSpec("g", "filter", 2, "generic",
                                    perturbation=generic_antisymmetric(s, 2)))
    swapped_gauge = run(lambda s: FilterSpec("g", "filter", 2, "generic",
                                             gauge="reset_onehot_swapped",
                                             perturbation=generic_antisymmetric(s, 2)))
    learned_gauge = run(lambda s: FilterSpec("g", "filter", 2, "generic", gauge="learned",
                                             perturbation=generic_antisymmetric(s, 2)))
    flipped = run(
        lambda s: FilterSpec("g", "filter", 2, "generic",
                             perturbation=np.ascontiguousarray(
                                 generic_antisymmetric(s, 2)[::-1])),
        event_transform=lambda e: 1.0 - e,
        events=1.0 - tensors.events_true)
    sign = run(lambda s: FilterSpec("g", "filter", 2, "generic",
                                    perturbation=-generic_antisymmetric(s, 2)))

    out["base"] = {"mean": float(base.mean()), "p10": float(np.percentile(base, 10)),
                   "seeds": len(base)}
    for name, values in (("latent_state_permutation", swapped_gauge),
                         ("learned_gauge", learned_gauge),
                         ("event_label_permutation_with_relabelled_init", flipped),
                         ("perturbation_sign_reversed", sign)):
        out[name] = {"mean": float(values.mean()),
                     "p10": float(np.percentile(values, 10)),
                     "delta_mean_vs_base": float(values.mean() - base.mean()),
                     "interval": paired_seed_interval(values, base)}
    out["check_1_latent_state_permutation_invariant"] = bool(
        not out["latent_state_permutation"]["interval"]["excludes_zero"])
    out["check_2_event_label_permutation_invariant"] = bool(
        not out["event_label_permutation_with_relabelled_init"]["interval"]
        ["excludes_zero"])
    # Sign invariance is a statement about the FAMILY, not about a seed: -P is itself a
    # legitimate generic draw, so the two distributions must agree in expectation.
    out["check_3_sign_invariant_in_expectation"] = bool(
        not out["perturbation_sign_reversed"]["interval"]["excludes_zero"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2e-genericity.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    train = build_dataset(collect(list(m2d.TRAIN_LAYOUTS), 3, 9,
                                  CANONICAL_APPEARANCE_SEED, 11), 5)
    population = m2d.build_population(m2d.ALIAS_LAYOUTS)
    tensors = build_tensors(population, m2d.RouteFeatures(population))

    family = family_statistics()
    static = static_checks()
    print("initial transition matrices, exactly as constructed")
    print(f"{'source':22s} {'event 0':>26s} {'event 1':>26s} "
          f"{'d(identity)':>11s} {'d(swap)':>8s} {'d(answer)':>10s}")
    matrices = {}
    for label, perturbation in (("M2D answer-oriented", ANSWER),
                                ("generic seed 13000", generic_antisymmetric(13_000, 2)),
                                ("generic seed 13001", generic_antisymmetric(13_001, 2)),
                                ("generic seed 13002", generic_antisymmetric(13_002, 2))):
        t = softmax(np.asarray(perturbation, dtype=float))
        d = distances(perturbation)
        matrices[label] = {"transition_event_0": t[0].round(4).tolist(),
                           "transition_event_1": t[1].round(4).tolist(), **d}
        print(f"{label:22s} {str(t[0].round(3).tolist()):>26s} "
              f"{str(t[1].round(3).tolist()):>26s} {d['to_identity_pair']:11.4f} "
              f"{d['to_swap_pair']:8.4f} {d['to_answer_orientation']:10.4f}")

    print(f"\nfamily over {DRAWS} draws:")
    print(f"  mean distance to the answer orientation   {family['mean_distance_to_answer']:.4f}")
    print(f"  mean distance between two draws           "
          f"{family['mean_distance_between_two_draws']:.4f}")
    print(f"  mean stay-minus-flip diagonal             "
          f"{family['mean_stay_minus_flip_diagonal']:+.4f} "
          f"(answer: {family['answer_stay_minus_flip_diagonal']:+.4f})")
    print(f"  fraction oriented toward stay/flip        "
          f"{family['fraction_oriented_towards_stay_flip']:.3f}")

    trained = trained_invariances(train, tensors, population)
    print(f"\ntrained invariances ({len(SEEDS)} seeds, alias-pair accuracy):")
    print(f"  base generic                              {trained['base']['mean']:.4f}")
    for key in ("latent_state_permutation", "learned_gauge",
                "event_label_permutation_with_relabelled_init",
                "perturbation_sign_reversed"):
        interval = trained[key]["interval"]
        print(f"  {key:44s} {trained[key]['mean']:.4f}  "
              f"{interval['delta']:+.4f} [{interval['ci_low']:+.4f}, "
              f"{interval['ci_high']:+.4f}]"
              f"{'  DIFFERS' if interval['excludes_zero'] else ''}")

    checks = {k: v for block in (family, static, trained) for k, v in block.items()
              if k.startswith("check_")}
    report = {"initial_matrices": matrices, "family": family, "static": static,
              "trained": trained, "checks": checks,
              "v2_all_eligible_initialisations_generic": bool(all(checks.values())),
              "wall_clock_seconds": time.perf_counter() - started}
    write(arguments.out, report)
    print("\nchecks:")
    for name, value in sorted(checks.items()):
        print(f"  {name:56s} {value}")
    print(f"\nV2 (every eligible initialisation is generic and permutation-invariant): "
          f"{report['v2_all_eligible_initialisations_generic']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
