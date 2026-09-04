"""D. Is the soft assignment a ROLE BINDING, or a latent code that happens to fit?

The binder is trained through the event alone, so nothing in its loss asks the
assignment to mean anything. This module asks what it means anyway: how close it is to a
permutation, how much of the true role map it recovers, how stable it is inside an
episode, and -- the decisive test -- whether swapping two roles in the assignment changes
the predictions those roles are supposed to control.

The swap test has a shape the audit must respect. An event-relevant swap
(AGENT <-> SWITCH, SWITCH <-> DECOY) MUST move the event. A goal-only swap
(GOAL_ALPHA <-> GOAL_BETA) NEED NOT, because the event does not mention the markers, and
scoring it as a failure would demand the model know something the event never taught it.
WALL <-> EMPTY is the same case.

If the assignment is far from any permutation and only the two event roles are recovered,
the honest label is `event-sufficient latent assignment`, not role binding.

    .venv-shwm/bin/python experiments/shwm/o2_assignment.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

import o2_core as C
import o2_models as M
from m2d_core import ARTIFACTS, write
from o2_core import COLOUR_POOL, N_ROLES, ROLES

SEEDS = (43_000, 43_001, 43_002)
TRAIN_LAYOUTS = tuple(range(110_000, 110_024))
TEST_LAYOUTS = tuple(range(111_000, 111_024))
DEV_PALETTES = tuple(range(9_300, 9_308))
UNSEEN_PALETTES = tuple(range(9_400, 9_408))
VIEW = "full_token"

SWAPS = {
    "AGENT_SWITCH": (C.AGENT, C.SWITCH, True),
    "SWITCH_DECOY": (C.SWITCH, C.DECOY, True),
    "GOAL_ALPHA_GOAL_BETA": (C.GOAL_ALPHA, C.GOAL_BETA, False),
    "WALL_EMPTY": (C.WALL, C.EMPTY, False),
}


def event_from_assignment(assignment: np.ndarray, before_index: np.ndarray,
                          after_index: np.ndarray) -> np.ndarray:
    """The M2F relational expression in numpy, on a per-row assignment (N, K, R)."""
    rows = np.arange(len(before_index))[:, None, None]
    pb = assignment[rows, before_index]
    pa = assignment[rows, after_index]
    evidence = pa[..., C.AGENT] * (1.0 - pb[..., C.AGENT]) * pb[..., C.SWITCH]
    return evidence.reshape(len(before_index), -1).max(axis=1)


def truth_map(episode: C.O2Episode, registry: C.ColourRegistry) -> dict[int, int]:
    """colour slot -> true role. Evaluator-only; used to SCORE, never to train."""
    return {registry.of(COLOUR_POOL[episode.bijection[role]]): role
            for role in range(N_ROLES)}


def nearest_permutation(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Hungarian projection of a soft (R, R) block onto a permutation matrix."""
    rows, columns = linear_sum_assignment(-matrix)
    hard = np.zeros_like(matrix)
    hard[rows, columns] = 1.0
    return hard, float(np.linalg.norm(matrix - hard))


def audit_stratum(stratum: str, seed: int) -> dict[str, Any]:
    train, test = [], []
    for palette in DEV_PALETTES:
        train.extend(C.collect(TRAIN_LAYOUTS, C.sample_bijection(palette), stratum, 9,
                               seed=11, policy="uniform"))
    for palette in UNSEEN_PALETTES:
        test.extend(C.collect(TEST_LAYOUTS, C.sample_bijection(palette), stratum, 9,
                              seed=313, policy="uniform"))
    registry = C.ColourRegistry().scan(train + test)
    train_data = C.pair_dataset(train, registry)
    block = C.as_block(train_data)
    block = (M.mask_view(block[0], VIEW),) + block[1:]
    infer, model = M.train_stateless(block, seed)

    per_palette: dict[int, list[np.ndarray]] = {}
    per_episode_spread: list[float] = []
    # The mean absolute spread is a bad stability measure here: most of a 10x7
    # assignment is zeros, so a row that relabels two colours barely moves it. The
    # honest question is whether the ARGMAX CODE is the same from pair to pair, and it
    # is not -- which is the difference between a role binding and a per-pair
    # computation dressed as one.
    stability: list[float] = []
    distinct: list[int] = []
    exact, event_ok, goal_ok, distance = [], [], [], []
    for episode in test:
        data = C.pair_dataset([episode], registry)
        if not len(data["event"]):
            continue
        assignment = M.assignment_of(model, M.mask_view(data["tokens"], VIEW))
        per_palette.setdefault(episode.palette_id, []).append(assignment.mean(axis=0))
        truth = truth_map(episode, registry)
        slots = sorted(truth)
        square = assignment.mean(axis=0)[slots]                  # (7, R)
        hard, gap = nearest_permutation(square)
        distance.append(gap)
        predicted = {slot: int(square[i].argmax()) for i, slot in enumerate(slots)}
        exact.append(float(all(predicted[s] == truth[s] for s in slots)))
        event_ok.append(float(all(
            predicted[s] == truth[s] for s in slots
            if truth[s] in (C.AGENT, C.SWITCH))))
        goal_ok.append(float(all(
            predicted[s] == truth[s] for s in slots
            if truth[s] in (C.AGENT, C.GOAL_ALPHA, C.GOAL_BETA))))
        per_episode_spread.append(float(assignment.std(axis=0).mean()))
        codes = [tuple(int(v) for v in row) for row in assignment.argmax(axis=-1)]
        modal = max(set(codes), key=codes.count)
        stability.append(float(np.mean([c == modal for c in codes])))
        distinct.append(len(set(codes)))

    test_data = C.pair_dataset(test, registry)
    truth_all = {}
    for episode in test:
        truth_all.update(truth_map(episode, registry))
    soft = M.assignment_of(model, M.mask_view(test_data["tokens"], VIEW))
    baseline = event_from_assignment(soft, test_data["before_index"],
                                     test_data["after_index"])
    hard_rows = np.zeros_like(soft)
    hard_rows[np.arange(len(soft))[:, None], np.arange(soft.shape[1])[None, :],
              soft.argmax(axis=-1)] = 1.0
    projected = event_from_assignment(hard_rows, test_data["before_index"],
                                      test_data["after_index"])

    swaps: dict[str, Any] = {}
    for name, (first, second, must_move) in SWAPS.items():
        swapped = soft.copy()
        swapped[:, :, [first, second]] = swapped[:, :, [second, first]]
        moved = event_from_assignment(swapped, test_data["before_index"],
                                      test_data["after_index"])
        changed = float(np.mean(np.abs(moved - baseline) > 1e-3))
        swaps[name] = {
            "fraction_of_rows_whose_event_changed": changed,
            "mean_absolute_change": float(np.abs(moved - baseline).mean()),
            "balanced_accuracy_after_swap": M.balanced_accuracy(
                moved - 0.5, test_data["event"]),
            "must_move_the_event": must_move,
            "behaved_as_required": bool(changed > 0.05 if must_move else True)}

    mean_matrix = {str(p): np.mean(v, axis=0).tolist() for p, v in per_palette.items()}
    row_entropy, column_entropy = [], []
    for value in per_palette.values():
        matrix = np.mean(value, axis=0)
        live = matrix[matrix.sum(axis=1) > 1e-6]
        row_entropy.append(float(np.mean(
            [-np.sum(r * np.log2(np.maximum(r, 1e-12))) for r in live])))
        column = live.sum(axis=0)
        column = column / max(column.sum(), 1e-9)
        column_entropy.append(float(-np.sum(column * np.log2(np.maximum(column, 1e-12)))))

    return {
        "stratum": stratum, "seed": seed,
        "soft_assignment_by_palette": mean_matrix,
        "mean_row_entropy_bits": float(np.mean(row_entropy)),
        "mean_column_entropy_bits": float(np.mean(column_entropy)),
        "maximum_row_entropy_bits": float(np.log2(N_ROLES)),
        "mean_distance_to_nearest_permutation": float(np.mean(distance)),
        "exact_role_map_accuracy": float(np.mean(exact)),
        "event_equivalence_accuracy": float(np.mean(event_ok)),
        "goal_equivalence_accuracy": float(np.mean(goal_ok)),
        "full_map_accuracy": float(np.mean(exact)),
        "event_balanced_accuracy_soft": M.balanced_accuracy(baseline - 0.5,
                                                            test_data["event"]),
        "event_balanced_accuracy_after_projection": M.balanced_accuracy(
            projected - 0.5, test_data["event"]),
        "within_episode_assignment_spread": float(np.mean(per_episode_spread)),
        "within_episode_argmax_stability": float(np.mean(stability)),
        "distinct_argmax_codes_per_episode": float(np.mean(distinct)),
        "counterfactual_swaps": swaps,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o2-assignment.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    report: dict[str, Any] = {"view": VIEW, "roles": list(ROLES),
                              "seeds": list(SEEDS[:arguments.seeds]), "strata": {}}
    for stratum in ("COUNT_INFORMATIVE", "COUNT_COLLISION"):
        blocks = [audit_stratum(stratum, seed) for seed in SEEDS[:arguments.seeds]]
        summary = {}
        for key in ("mean_row_entropy_bits", "mean_column_entropy_bits",
                    "mean_distance_to_nearest_permutation", "exact_role_map_accuracy",
                    "event_equivalence_accuracy", "goal_equivalence_accuracy",
                    "event_balanced_accuracy_soft",
                    "event_balanced_accuracy_after_projection",
                    "within_episode_assignment_spread",
                    "within_episode_argmax_stability",
                    "distinct_argmax_codes_per_episode"):
            summary[key] = float(np.mean([b[key] for b in blocks]))
        summary["counterfactual_swaps"] = {
            name: {
                "fraction_of_rows_whose_event_changed": float(np.mean(
                    [b["counterfactual_swaps"][name][
                        "fraction_of_rows_whose_event_changed"] for b in blocks])),
                "balanced_accuracy_after_swap": float(np.mean(
                    [b["counterfactual_swaps"][name]["balanced_accuracy_after_swap"]
                     for b in blocks])),
                "must_move_the_event": SWAPS[name][2],
                "behaved_as_required": bool(all(
                    b["counterfactual_swaps"][name]["behaved_as_required"]
                    for b in blocks))}
            for name in SWAPS}
        summary["per_seed"] = blocks
        report["strata"][stratum] = summary
        print(f"\n=== {stratum} ===")
        print(f"  exact role map          {summary['exact_role_map_accuracy']:.4f}")
        print(f"  event equivalence       {summary['event_equivalence_accuracy']:.4f}")
        print(f"  goal equivalence        {summary['goal_equivalence_accuracy']:.4f}")
        print(f"  distance to permutation {summary['mean_distance_to_nearest_permutation']:.4f}")
        print(f"  row entropy             {summary['mean_row_entropy_bits']:.4f} "
              f"of {np.log2(N_ROLES):.4f} bits")
        print(f"  within-episode spread   {summary['within_episode_assignment_spread']:.4f}")
        print(f"  argmax code stability   {summary['within_episode_argmax_stability']:.4f}"
              f"  ({summary['distinct_argmax_codes_per_episode']:.2f} distinct codes "
              f"per episode)")
        print(f"  event soft / projected  {summary['event_balanced_accuracy_soft']:.4f}"
              f" / {summary['event_balanced_accuracy_after_projection']:.4f}")
        for name, entry in summary["counterfactual_swaps"].items():
            print(f"  swap {name:22s} changed "
                  f"{entry['fraction_of_rows_whose_event_changed']:.4f}  "
                  f"accuracy {entry['balanced_accuracy_after_swap']:.4f}  "
                  f"required to move: {entry['must_move_the_event']}  "
                  f"ok: {entry['behaved_as_required']}")

    informative = report["strata"]["COUNT_INFORMATIVE"]
    report["verdict"] = {
        "event_roles_recovered": bool(informative["event_equivalence_accuracy"] > 0.8),
        "complete_role_binding": bool(informative["exact_role_map_accuracy"] > 0.8),
        "assignment_is_a_stable_map": bool(
            informative["within_episode_argmax_stability"] > 0.95),
        "label": ("complete role binding"
                  if informative["exact_role_map_accuracy"] > 0.8
                  else "event-sufficient latent assignment"),
        "basis": (f"exact role map {informative['exact_role_map_accuracy']:.4f}, event "
                  f"equivalence {informative['event_equivalence_accuracy']:.4f}, goal "
                  f"equivalence {informative['goal_equivalence_accuracy']:.4f}, distance "
                  f"to the nearest permutation "
                  f"{informative['mean_distance_to_nearest_permutation']:.4f}"),
    }
    report["swap_test_is_not_vacuous"] = bool(
        informative["counterfactual_swaps"]["AGENT_SWITCH"][
            "fraction_of_rows_whose_event_changed"] > 0.05)
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nverdict: {report['verdict']['label']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
