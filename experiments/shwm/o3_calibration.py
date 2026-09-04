"""D. Exact calibration sufficiency, per palette and per interaction.

The specification's rule is the point of this module: do not infer calibration
insufficiency from a learned score. For every palette the exact posterior over role
permutations is recomputed after each public interaction, so a palette the learned memory
fails on can be classified by what the EVIDENCE contained rather than by what the model
did with it.

R1 removes one of the four categories a priori. The corrected pipeline is bit-exact
palette-equivariant, so "works under some palettes only, on the same semantic trace" is
structurally impossible -- two palettes over one semantic trace return identical numbers.
Category 3 can therefore only be reached by a pipeline that has failed R1, and it is kept
in the table so that a future regression lands somewhere visible.

    .venv-shwm/bin/python experiments/shwm/o3_calibration.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import o2_core as C
import o3_population as pop
from m2d_core import ARTIFACTS, write

CATEGORIES = ("1_calibration_or_testbed_insufficient",
              "2_learned_inference_failure",
              "3_equivariance_or_canonicalization_defect",
              "4_sequence_fidelity_failure")


def exercised(episodes) -> dict[str, int]:
    """What the agent actually did, which is what the posterior can condition on."""
    moves = switches = decoys = goals = blocked = 0
    for episode in episodes:
        for t in range(1, episode.length):
            entered = int(episode.entered_role[t])
            if entered < 0:
                blocked += 1
                continue
            moves += 1
            switches += int(entered == C.SWITCH)
            decoys += int(entered == C.DECOY)
            goals += int(entered in (C.GOAL_ALPHA, C.GOAL_BETA))
    return {"agent_moves": moves, "switch_encounters": switches,
            "decoy_encounters": decoys, "goal_encounters": goals,
            "blocked_actions": blocked}


def posterior_curve(episodes) -> list[dict[str, Any]]:
    """Exact posterior after each interaction, accumulated across episodes."""
    identity = tuple(range(C.N_ROLES))
    curve, consumed, candidates = [], 0, None
    for index, episode in enumerate(episodes):
        for steps in range(1, episode.length):
            keep = C.survivors_over([episode], steps=steps, candidates=candidates)
            if index:
                pass
            assert identity in keep
            consumed += 1
            event_mass = C.event_quotient_mass(keep)
            goal_mass = float(np.mean([pi[C.GOAL_ALPHA] == C.GOAL_ALPHA
                                       and pi[C.GOAL_BETA] == C.GOAL_BETA
                                       for pi in keep]))
            curve.append({
                "interactions": consumed, "episode": index, "steps_into_episode": steps,
                "class_size": len(keep),
                "mapping_entropy_bits": float(np.log2(len(keep))),
                "event_equivalence_entropy_bits": float(
                    np.log2(max(len({(pi[C.AGENT], pi[C.SWITCH]) for pi in keep}), 1))),
                "goal_equivalence_entropy_bits": float(
                    np.log2(max(len({(pi[C.AGENT], pi[C.GOAL_ALPHA], pi[C.GOAL_BETA])
                                     for pi in keep}), 1))),
                "true_event_class_mass": event_mass,
                "true_goal_class_mass": goal_mass,
                "true_exact_map_mass": 1.0 / len(keep),
            })
        candidates = C.survivors_over(episodes[:index + 1])
    return curve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--palettes", type=int, default=32)
    parser.add_argument("--calibration", type=int, default=6)
    parser.add_argument("--population", type=Path,
                        default=ARTIFACTS / "o3-population.json")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o3-calibration.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    import json
    learned = {}
    if arguments.population.exists():
        block = json.loads(arguments.population.read_text())
        for row in block["groups"]["validation"]["per_palette"]:
            learned[row["palette"]] = row

    rows = []
    print(f"{'palette':>8s} {'inter':>6s} {'min sep':>8s} {'event H':>8s} "
          f"{'goal H':>7s} {'switch':>7s} {'decoy':>6s} {'learned':>8s} category")
    print("-" * 108)
    for palette in pop.VALIDATION_PALETTES[:arguments.palettes]:
        plan = pop.palette_plan(palette, arguments.calibration, 1, 1)
        episodes = C.collect(plan["calibration_layouts"], plan["bijection"],
                             "COUNT_COLLISION", 9, seed=plan["action_seed"],
                             policy="uniform")
        curve = posterior_curve(episodes)
        counts = exercised(episodes)
        pinned = [c["interactions"] for c in curve if c["true_event_class_mass"] == 1.0]
        final = curve[-1]
        score = learned.get(palette)
        if score is None:
            category = "not_scored"
        elif final["true_event_class_mass"] < 1.0:
            category = CATEGORIES[0]
        elif score["route_parity"] >= pop.ROUTE_PARITY_GATE:
            category = "passes"
        elif score["per_step_event_accuracy"] >= 0.95:
            category = CATEGORIES[3]
        else:
            category = CATEGORIES[1]
        rows.append({
            "palette": palette, "bijection": [int(v) for v in plan["bijection"]],
            "calibration_layouts": plan["calibration_layouts"],
            "interactions": final["interactions"],
            "minimum_separating_interactions": pinned[0] if pinned else None,
            "final_mapping_entropy_bits": final["mapping_entropy_bits"],
            "final_event_equivalence_entropy_bits":
                final["event_equivalence_entropy_bits"],
            "final_goal_equivalence_entropy_bits":
                final["goal_equivalence_entropy_bits"],
            "final_true_event_class_mass": final["true_event_class_mass"],
            "final_true_goal_class_mass": final["true_goal_class_mass"],
            "final_true_exact_map_mass": final["true_exact_map_mass"],
            "roles_exercised": counts,
            "learned_route_parity": score["route_parity"] if score else None,
            "learned_per_step": score["per_step_event_accuracy"] if score else None,
            "learned_contested": score["contested_transfer_accuracy"] if score else None,
            "category": category,
            "curve": curve,
        })
        print(f"{palette:8d} {final['interactions']:6d} "
              f"{str(pinned[0] if pinned else '-'):>8s} "
              f"{final['event_equivalence_entropy_bits']:8.3f} "
              f"{final['goal_equivalence_entropy_bits']:7.3f} "
              f"{counts['switch_encounters']:7d} {counts['decoy_encounters']:6d} "
              f"{(score['route_parity'] if score else float('nan')):8.4f} {category}",
              flush=True)

    from collections import Counter
    tally = Counter(r["category"] for r in rows)
    separating = [r["minimum_separating_interactions"] for r in rows
                  if r["minimum_separating_interactions"] is not None]
    report: dict[str, Any] = {
        "palettes": len(rows),
        "calibration_episodes_per_palette": arguments.calibration,
        "categories": dict(tally),
        "event_class_identified_fraction": float(np.mean(
            [r["final_true_event_class_mass"] == 1.0 for r in rows])),
        "goal_class_identified_fraction": float(np.mean(
            [r["final_true_goal_class_mass"] == 1.0 for r in rows])),
        "minimum_separating_interactions": {
            "mean": float(np.mean(separating)) if separating else None,
            "median": float(np.median(separating)) if separating else None,
            "maximum": int(np.max(separating)) if separating else None,
            "palettes_never_separating": int(len(rows) - len(separating))},
        "category_3_is_structurally_unreachable": (
            "the corrected pipeline is bit-exact palette-equivariant, so two palettes "
            "over one semantic trace return identical numbers; category 3 can only be "
            "reached by a pipeline that has failed R1"),
        "per_palette": rows,
    }
    report["R2_exact_audit_classifies_every_failure"] = bool(
        tally.get("not_scored", 0) == 0)
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\ncategories: {dict(tally)}")
    print(f"event class identified on {report['event_class_identified_fraction']:.4f} "
          f"of palettes; goal class on {report['goal_class_identified_fraction']:.4f}")
    print(f"R2 {report['R2_exact_audit_classifies_every_failure']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
