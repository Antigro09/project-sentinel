"""M / Q13. The regime where no answer exists, and what the system says instead.

Under a fresh bijection every frame there is no convention to infer, so the honest
output is not a guess. This module measures three things: what the exact enumeration says
is identifiable, what the learned system does with its abstention rule, and whether that
rule is doing any work -- a rule that abstains equally under a persistent palette is
measuring nothing, so the persistent case is run as the calibration arm.

The exact side enumerates PAIRS of per-frame permutations under the joint legality of a
transition, which is the right hypothesis space here: a per-frame regime does not have a
single mapping to be uncertain about, it has one per frame, and the only cross-frame
constraint left is that the hypothesised trajectory still has to be a legal trajectory.

    .venv-shwm/bin/python experiments/shwm/o2_unresolved.py
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np

import o_identifiability as ident
import o2_core as C
import o2_memory as mem
import o2_models as M
from m2d_core import ARTIFACTS, write

SEEDS = (47_000,)
LAYOUTS = tuple(range(111_000, 111_012))
DEV_PALETTES = mem.DEV_PALETTES
UNSEEN_PALETTES = mem.UNSEEN_PALETTES
STRATUM = "COUNT_COLLISION"
UNRESOLVED = "UNRESOLVED_APPEARANCE"


def frame_survivors(grid: np.ndarray) -> list[tuple[int, ...]]:
    return [pi for pi in itertools.permutations(range(C.N_ROLES))
            if C.cardinality_legal7(np.array(pi, np.int64)[grid])]


def per_frame_identifiability(episode: C.O2Episode, tied: bool) -> dict[str, Any]:
    """Exact, over PAIRS of per-frame permutations satisfying joint transition legality.

    `tied` restricts the pair space to its diagonal -- one mapping shared by both frames,
    which is what a persistent convention means. It is enforced explicitly, and the
    measurement then shows it makes NO difference, for a reason worth stating: static-
    scene legality already forces it. Over 672 legal pairs on a sample population, the
    number with distinct permutations is ZERO, because two frames whose hypothesised role
    grids differ by anything more than the agent's two cells cannot both be legal.

    That is a result about the regime, not about the code. Phase O introduced
    PER_FRAME_PERMUTATION as "the impossibility control: no persistent convention exists
    to infer". At the level of a frame PAIR that is false -- an observer can re-identify
    the relabelling by requiring the scene to be static. What a per-frame permutation
    actually destroys is the ability to ADDRESS a memory by colour value across episodes,
    which is a different and weaker claim, and it is the one the learned side measures.
    """
    event_identified, goal_identified, class_sizes, event_masses = [], [], [], []
    identical, distinct = 0, 0
    for t in range(1, episode.length):
        before_set = frame_survivors(episode.roles[t - 1])
        after_set = frame_survivors(episode.roles[t])
        legal = []
        for sigma in before_set:
            grid_before = np.array(sigma, np.int64)[episode.roles[t - 1]]
            for tau in ([sigma] if tied else after_set):
                if tied and sigma not in after_set:
                    continue
                grid_after = np.array(tau, np.int64)[episode.roles[t]]
                if (ident.static_scene_legal(grid_before, grid_after)
                        and ident.motion_legal(grid_before, grid_after,
                                               int(episode.actions[t - 1]))):
                    legal.append((sigma, tau))
                    if sigma == tau:
                        identical += 1
                    else:
                        distinct += 1
        if not legal:
            continue
        entered = int(episode.entered_role[t])
        calls = ({sigma[entered] == C.SWITCH for sigma, _ in legal}
                 if entered >= 0 else {False})
        event_identified.append(float(len(calls) == 1))
        goal_identified.append(float(len({(sigma[C.GOAL_ALPHA], sigma[C.GOAL_BETA])
                                          for sigma, _ in legal}) == 1))
        class_sizes.append(len(legal))
        event_masses.append(float(np.mean(
            [sigma[C.AGENT] == C.AGENT and sigma[C.SWITCH] == C.SWITCH
             for sigma, _ in legal])))
    return {
        "steps": len(event_identified),
        "event_identifiable_fraction": float(np.mean(event_identified)),
        "goal_identifiable_fraction": float(np.mean(goal_identified)),
        "mean_joint_class_size": float(np.mean(class_sizes)),
        "posterior_entropy_bits": float(np.mean(np.log2(class_sizes))),
        "event_equivalence_posterior_mass": float(np.mean(event_masses)),
        "full_map_posterior_mass": float(np.mean(1.0 / np.array(class_sizes))),
        "legal_pairs_with_identical_permutations": identical,
        "legal_pairs_with_distinct_permutations": distinct,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o2-unresolved.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    # ONE episode set. The role grids do not depend on the rendering regime -- only the
    # colours do -- so what separates the two regimes is the hypothesis space the
    # observer is entitled to, which is the `tied` flag. Collecting two identical sets
    # and enumerating over both produced two identical columns and would have been
    # reported as "the regimes are the same".
    episodes = []
    for index, palette in enumerate(UNSEEN_PALETTES):
        episodes.extend(C.collect(LAYOUTS, C.sample_bijection(palette), STRATUM, 9,
                                  seed=313 + index, policy="uniform"))

    print("exact identifiability, enumerated over permutation pairs", flush=True)
    exact = {}
    for name in ("HIDDEN_PALETTE_CONVENTION", "PER_FRAME_PERMUTATION"):
        blocks = [per_frame_identifiability(e, tied=(name != "PER_FRAME_PERMUTATION"))
                  for e in episodes[:24]]
        exact[name] = {k: float(np.mean([b[k] for b in blocks]))
                       for k in blocks[0] if k != "steps"}
        exact[name]["episodes"] = len(blocks)
        exact[name]["legal_pairs_with_distinct_permutations"] = int(
            sum(b["legal_pairs_with_distinct_permutations"] for b in blocks))
        exact[name]["legal_pairs_with_identical_permutations"] = int(
            sum(b["legal_pairs_with_identical_permutations"] for b in blocks))
        print(f"  {name:28s} event identified "
              f"{exact[name]['event_identifiable_fraction']:.4f}  goal "
              f"{exact[name]['goal_identifiable_fraction']:.4f}  entropy "
              f"{exact[name]['posterior_entropy_bits']:.3f} bits", flush=True)

    print("\ntraining the palette memory and applying its abstention rule", flush=True)
    dev_train = [mem.build_group(p, 71) for p in DEV_PALETTES]
    dev_threshold = [mem.build_group(p, 517) for p in mem.THRESHOLD_PALETTES]
    registry = C.canonical_registry()
    train = mem.stack_groups(dev_train, registry)
    threshold_data = mem.stack_groups(dev_threshold, registry)
    _, model = M.train_memory(
        (train["sequence"], train["mask"], train["before"], train["after"],
         train["event"]), SEEDS[0], updates=mem.MEMORY_UPDATES)

    held = M.memory_assignment_of(model, threshold_data["sequence"],
                                  threshold_data["mask"])
    held_slot = mem.entered_slot(threshold_data, registry, dev_threshold)
    held_contested = mem.contested(threshold_data)
    quantile_tau = float(np.quantile(
        np.abs(held[np.arange(len(held)), held_slot, C.SWITCH][held_contested] - 0.5),
        1.0 - mem.COVERAGE_TARGET))
    tau = mem.ABSTENTION_MARGIN

    learned = {}
    unresolved_rows: dict[str, list] = {}
    # The SAME construction as section G -- calibration on CAL_LAYOUTS, transfer pairs on
    # disjoint TRANSFER_LAYOUTS, same seeds and same trajectories -- rendered under each
    # regime. Evaluating the memory on a differently built population instead put its
    # answer-conditional accuracy at 0.5000 and would have been a statement about the
    # mismatch, not about the regime.
    for name, per_frame_flag in (("HIDDEN_PALETTE_CONVENTION", False),
                                 ("PER_FRAME_PERMUTATION", True)):
        groups = [mem.build_group(p, 313, per_frame=per_frame_flag)
                  for p in UNSEEN_PALETTES]
        data = mem.stack_groups(groups, registry)
        assignment = M.memory_assignment_of(model, data["sequence"], data["mask"])
        entered = data["sequence"][:, -1, :, C.INTERACT][:, :, 0].argmax(axis=1)
        probability = assignment[np.arange(len(assignment)), entered, C.SWITCH]
        contested = mem.contested(data)
        resolved = np.abs(probability - 0.5) >= tau
        truth = (data["meta"][:, 3] == C.SWITCH).astype(float)
        said = (probability > 0.5).astype(float)
        answered = contested & resolved
        learned[name] = {
            "tau": tau, "quantile_rule_tau": quantile_tau,
            "contested_rows": int(contested.sum()),
            "coverage": float(resolved[contested].mean()),
            "unresolved_rate": float(1.0 - resolved[contested].mean()),
            "accuracy_conditional_on_answering": (float((said == truth)[answered].mean())
                                                  if answered.any() else None),
            "accuracy_unconditional": float((said == truth)[contested].mean()),
            "false_confident_semantic_assignment": float(
                ((said != truth) & resolved)[contested].mean()),
            "event_balanced_accuracy": M.balanced_accuracy(
                (probability - 0.5)[contested], data["event"][contested]),
        }
        keep = contested & ~resolved
        if keep.any():
            unresolved_rows[name] = [{
                "layout": data["meta"][keep, 0], "step": data["meta"][keep, 2],
                "entered_role": data["meta"][keep, 3],
                "switch_probability": probability[keep], "event": data["event"][keep]}]
        print(f"  {name:28s} coverage {learned[name]['coverage']:.4f}  unresolved "
              f"{learned[name]['unresolved_rate']:.4f}  accuracy|answer "
              f"{learned[name]['accuracy_conditional_on_answering']}  false confident "
              f"{learned[name]['false_confident_semantic_assignment']:.4f}", flush=True)

    per_frame = exact["PER_FRAME_PERMUTATION"]
    persistent = exact["HIDDEN_PALETTE_CONVENTION"]
    report: dict[str, Any] = {
        "regimes": list(exact), "layouts": list(LAYOUTS), "stratum": STRATUM,
        "exact_hypothesis_space": (
            "pairs of per-frame permutations under joint transition legality; the "
            "persistent regime is the diagonal of that space, the per-frame regime is "
            "all of it"),
        "unseen_palettes": list(UNSEEN_PALETTES),
        "exact": exact, "learned": learned,
        "system_output_under_per_frame_permutation": UNRESOLVED,
        "scoped_event_prediction_permitted": bool(
            per_frame["event_identifiable_fraction"] >= 0.99),
        "scoped_event_basis": (
            f"a scoped event prediction is permitted only where event equivalence is "
            f"identified despite full semantic ambiguity; measured "
            f"{per_frame['event_identifiable_fraction']:.4f} of steps under a per-frame "
            f"permutation against {persistent['event_identifiable_fraction']:.4f} under "
            f"a persistent convention"),
        "per_frame_is_not_an_impossibility_control": {
            "legal_pairs_with_distinct_permutations":
                exact["PER_FRAME_PERMUTATION"]["legal_pairs_with_distinct_permutations"],
            "why": ("static-scene legality already forces the two frames to share one "
                    "mapping, so an observer can re-identify a per-frame relabelling. "
                    "Phase O called this regime the impossibility control; at the level "
                    "of a frame pair it is not one. What it destroys is colour-addressed "
                    "memory ACROSS episodes, which is what the learned side measures."),
        },
        "abstention_rule_is_not_vacuous": bool(
            learned["PER_FRAME_PERMUTATION"]["unresolved_rate"]
            > learned["HIDDEN_PALETTE_CONVENTION"]["unresolved_rate"] + 0.05),
    }
    # Q14: the unresolved cases themselves are kept, not just their rate.
    dump = {}
    for name, blocks in unresolved_rows.items():
        for field in ("layout", "step", "entered_role", "switch_probability", "event"):
            dump[f"{name}__{field}"] = np.concatenate([b[field] for b in blocks])
    np.savez_compressed(ARTIFACTS / "o2-unresolved-examples.npz", **dump)
    report["unresolved_examples_retained"] = {
        name: int(len(dump[f"{name}__layout"])) for name in unresolved_rows}
    # Two criteria, both reported, and the second was added AFTER seeing the first pass
    # on a system that is confidently wrong on 40% of per-frame contested rows. The
    # coded criterion -- "the rule abstains more here than under a persistent
    # convention, and the goal is never identified" -- is too weak for what the gate
    # actually says, which is that unidentifiable cases are unresolved RATHER THAN
    # confidently assimilated. The stricter test is stated separately and the gate is
    # graded on both; moving a criterion after the fact is only defensible in this
    # direction.
    report["Q13_weak_criterion_abstains_more_and_goal_never_identified"] = bool(
        report["abstention_rule_is_not_vacuous"]
        and per_frame["goal_identifiable_fraction"] < 0.01)
    report["Q13_strict_criterion_confident_assimilation_below_10pc"] = bool(
        learned["PER_FRAME_PERMUTATION"]["false_confident_semantic_assignment"] < 0.10)
    report["Q13_status"] = (
        "PASS" if (report["Q13_weak_criterion_abstains_more_and_goal_never_identified"]
                   and report["Q13_strict_criterion_confident_assimilation_below_10pc"])
        else "PARTIAL"
        if report["Q13_weak_criterion_abstains_more_and_goal_never_identified"]
        else "FAIL")
    report["Q13_per_frame_cases_are_unresolved_not_assimilated"] = bool(
        report["Q13_status"] == "PASS")
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nQ13 {report['Q13_status']}  (weak criterion "
          f"{report['Q13_weak_criterion_abstains_more_and_goal_never_identified']}, "
          f"strict criterion "
          f"{report['Q13_strict_criterion_confident_assimilation_below_10pc']}, "
          f"confident assimilation "
          f"{learned['PER_FRAME_PERMUTATION']['false_confident_semantic_assignment']:.4f})")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
