"""B / P1. The residual class histogram, and the event / goal / full quotients.

Phase O reported "the only residual is GOAL_ALPHA <-> GOAL_BETA" from three sampled
certificates. That was a claim about a histogram nobody had printed. It is printed here,
and the three quotients the specification asks for have an exact closed form:

    phi ~event phi'   iff  pi fixes AGENT and SWITCH
                           (every legal retrospective event query is a statement about
                            the agent entering a switch cell, so any pi fixing both
                            leaves all of them unchanged, and any pi moving either
                            changes at least one)
    phi ~goal  phi'   iff  pi fixes AGENT, GOAL_ALPHA and GOAL_BETA
    phi ~full  phi'   iff  pi is the identity

so the quotient masses are read straight off the survivor set rather than estimated.

    .venv-shwm/bin/python experiments/shwm/p_equivalence.py
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import o_core as O
import o_identifiability as ident
from m2d_core import ARTIFACTS, write
from o_core import N_ROLES, ROLES, ROLE_INDEX

A, S = ROLE_INDEX["AGENT"], ROLE_INDEX["SWITCH"]
GA, GB = ROLE_INDEX["GOAL_ALPHA"], ROLE_INDEX["GOAL_BETA"]
STAGES = ("1_one_frame", "2_frame_pair_and_action", "4_short_legal_history",
          "5_grounded_calibration_episode", "6_complete_permitted_history",
          "7_plus_language_goal")


def fixes(pi, indices) -> bool:
    return all(pi[i] == i for i in indices)


def survivors_for(episode: O.Episode, stage: str) -> list[tuple[int, ...]]:
    if stage == "7_plus_language_goal":
        base = ident.survivors(episode, "6_complete_permitted_history")
        # The instruction names the target ROLE, not a colour. It only binds a colour
        # when the episode also shows the agent reaching that marker -- otherwise the
        # two markers stay exchangeable and saying otherwise would overclaim what
        # language buys.
        reached = bool(episode.length and
                       tuple(episode.positions[-1]) == tuple(
                           np.argwhere(episode.roles[0] == (
                               GA if episode.goal_marker == "alpha" else GB))[0]))
        if not reached:
            return base
        target = GA if episode.goal_marker == "alpha" else GB
        return [pi for pi in base if pi[target] == target]
    return ident.survivors(episode, stage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "p-equivalence.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    episodes = O.collect_appearance(
        list(range(110_000, 110_000 + arguments.episodes)),
        "HIDDEN_PALETTE_CONVENTION", [7_101], 1, 9, seed=11, policy="goal_directed")
    identity = tuple(range(N_ROLES))
    print(f"{len(episodes)} grounded episodes; all 720 permutations per stage\n",
          flush=True)

    report: dict[str, Any] = {
        "episodes": len(episodes), "family": 720, "roles": list(ROLES),
        "quotient_definitions": {
            "event": "pi fixes AGENT and SWITCH",
            "goal": "pi fixes AGENT, GOAL_ALPHA and GOAL_BETA",
            "full": "pi is the identity"},
        "stages": {}}
    print(f"{'stage':34s} {'mean':>7s} {'med':>5s} {'min':>4s} {'max':>4s} "
          f"{'entropy':>8s} {'full':>7s} {'event':>7s} {'goal':>7s} {'>2':>4s}")
    print("-" * 100)
    for stage in STAGES:
        sizes, full, event, goal, oversize = [], [], [], [], []
        histogram: Counter = Counter()
        for episode in episodes:
            keep = survivors_for(episode, stage)
            assert identity in keep, (stage, episode.layout)
            n = len(keep)
            sizes.append(n)
            histogram[n] += 1
            full.append(1.0 / n)
            event.append(sum(1 for pi in keep if fixes(pi, (A, S))) / n)
            goal.append(sum(1 for pi in keep if fixes(pi, (A, GA, GB))) / n)
            if n > 2:
                oversize.append({"layout": episode.layout, "class_size": n,
                                 "non_identity_examples": [
                                     {ROLES[i]: ROLES[pi[i]] for i in range(N_ROLES)
                                      if pi[i] != i}
                                     for pi in keep if pi != identity][:3]})
        sizes = np.array(sizes, dtype=float)
        block = {
            "histogram": {str(k): v for k, v in sorted(histogram.items())},
            "mean": float(sizes.mean()), "median": float(np.median(sizes)),
            "min": int(sizes.min()), "max": int(sizes.max()),
            "entropy_bits": float(np.log2(sizes).mean()),
            "true_full_map_mass": float(np.mean(full)),
            "true_event_class_mass": float(np.mean(event)),
            "true_goal_class_mass": float(np.mean(goal)),
            "event_identified_fraction": float(np.mean([e == 1.0 for e in event])),
            "goal_identified_fraction": float(np.mean([g == 1.0 for g in goal])),
            "classes_larger_than_two": oversize,
            "episodes_with_class_over_two": len(oversize)}
        report["stages"][stage] = block
        print(f"{stage:34s} {block['mean']:7.2f} {block['median']:5.1f} "
              f"{block['min']:4d} {block['max']:4d} {block['entropy_bits']:8.4f} "
              f"{block['true_full_map_mass']:7.4f} {block['true_event_class_mass']:7.4f} "
              f"{block['true_goal_class_mass']:7.4f} "
              f"{block['episodes_with_class_over_two']:4d}", flush=True)

    grounded = report["stages"]["5_grounded_calibration_episode"]
    only_goal_swap = all(
        set(entry.keys()) <= {"GOAL_ALPHA", "GOAL_BETA"}
        for block in grounded["classes_larger_than_two"]
        for entry in block["non_identity_examples"])
    report["o_phase_claim_that_only_goal_markers_remain"] = {
        "claimed_in_O": True,
        "supported_by_histogram": bool(grounded["max"] <= 2 or only_goal_swap),
        "grounded_histogram": grounded["histogram"],
        "episodes_with_class_over_two": grounded["episodes_with_class_over_two"]}
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\ngrounded-stage histogram: {grounded['histogram']}")
    print(f"O's 'only the goal markers remain' claim is supported by the histogram: "
          f"{report['o_phase_claim_that_only_goal_markers_remain']['supported_by_histogram']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
