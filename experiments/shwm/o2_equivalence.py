"""B / C / Q1 / Q2. One episode table, reconciled arithmetic, and three quotients.

Phase O reported a mean calibrated class size of 2.08. Phase O1 printed the histogram
{2: 44, 4: 1, 12: 2}, whose mean is 116/47 = 2.468. Both numbers were written as though
they described the same thing. They do not, and this module settles it by recomputing
BOTH populations from one function and printing the manifest of each.

Everything downstream is derived from a single machine-readable episode table -- one row
per (population, stage, episode) carrying the complete survivor list -- so the histogram,
the mean, the median, the entropy and the quotient masses cannot disagree with each
other: they are all reductions of the same rows.

The three quotients are exact, not estimated. Two hypotheses agree on every retrospective
event query exactly when they place AGENT and SWITCH on the same cells, which for a
permutation pi read as `true role r appears as pi[r]` means pi^-1 agrees on those two
roles; the goal quotient adds the two markers, and the full quotient is the permutation
itself.

    .venv-shwm/bin/python experiments/shwm/o2_equivalence.py
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import o_core as O
import o_identifiability as ident
import o2_core as C
from m2d_core import ARTIFACTS, write
from o_core import N_ROLES, ROLES, ROLE_INDEX

A = ROLE_INDEX["AGENT"]
S = ROLE_INDEX["SWITCH"]
GA = ROLE_INDEX["GOAL_ALPHA"]
GB = ROLE_INDEX["GOAL_BETA"]
IDENTITY = tuple(range(N_ROLES))

# Section C's seven stages, mapped onto the exact legality levels that implement them.
STAGES = (
    ("1_one_frame", "1_one_frame"),
    ("2_frame_pair_and_action", "2_frame_pair_and_action"),
    ("3_short_legal_history", "4_short_legal_history"),
    ("4_grounded_event_calibration", "5_grounded_calibration_episode"),
    ("5_complete_permitted_visual_history", "6_complete_permitted_history"),
    ("6_visual_history_plus_language", None),
    ("7_goal_grounding_calibration", None),
)

# Phase O and phase O1 built DIFFERENT populations and both called the result "the
# calibrated class size". The two rows below are the whole explanation.
POPULATIONS = {
    "O_phase_24_uniform": dict(layouts=24, palette=7_001, policy="uniform",
                               source="o_identifiability.py at commit 953f052"),
    "O1_phase_48_goal_directed": dict(layouts=48, palette=7_101, policy="goal_directed",
                                      source="p_equivalence.py at commit 8e99ade"),
}


def inverse(pi: Sequence[int]) -> tuple[int, ...]:
    out = [0] * len(pi)
    for i, v in enumerate(pi):
        out[v] = i
    return tuple(out)


def reached_named_marker(episode: O.Episode) -> bool:
    """Did any RECORDED frame show the agent standing on the named marker?

    The answer is structurally no. The adapter terminates the instant the agent arrives
    and every collector appends the frame before stepping, so the terminal frame is
    discarded. Phase O1 asked the weaker question -- whether the last recorded position
    equalled the marker -- and got the same answer for a different reason, so the
    conclusion "language identifies nothing" was right while the reason given was not.
    """
    target = GA if episode.goal_marker == "alpha" else GB
    where = np.argwhere(episode.roles[0] == target)
    if not len(where):
        return False
    cell = tuple(int(v) for v in where[0])
    return any(tuple(int(v) for v in p) == cell for p in episode.positions)


def survivors_for(episode: O.Episode, stage: str, level: str | None,
                  goal_demonstrated: bool) -> list[tuple[int, ...]]:
    if level is not None:
        return ident.survivors(episode, level)
    base = ident.survivors(episode, "6_complete_permitted_history")
    target = GA if episode.goal_marker == "alpha" else GB
    if stage == "6_visual_history_plus_language":
        # Language NAMES the goal role. It binds a colour only when this episode's own
        # trajectory happens to end on that marker; otherwise the two markers stay
        # exchangeable and claiming otherwise would credit language with work it did not
        # do. That is why stage 6 moves almost nothing.
        return ([pi for pi in base if pi[target] == target]
                if reached_named_marker(episode) else base)
    # Stage 7: a CONSTRUCTED demonstration under the same palette on a disjoint layout
    # ends on the named marker, so the binding holds whether or not this episode reached
    # anything. Section K builds the protocol; this is its exact consequence.
    return [pi for pi in base if pi[target] == target] if goal_demonstrated else base


def episode_rows(episodes: Sequence[O.Episode], population: str,
                 goal_demonstrated: bool) -> list[dict[str, Any]]:
    rows = []
    for episode in episodes:
        for stage, level in STAGES:
            keep = survivors_for(episode, stage, level, goal_demonstrated)
            assert IDENTITY in keep, "the truth must survive its own evidence"
            inverses = [inverse(pi) for pi in keep]
            rows.append({
                "population": population,
                "layout": int(episode.layout),
                "palette_seed": int(episode.palette_seed),
                "goal_marker": episode.goal_marker,
                "steps": int(episode.length),
                "stage": stage,
                "class_size": len(keep),
                "event_classes": len({(q[A], q[S]) for q in inverses}),
                "goal_classes": len({(q[A], q[GA], q[GB]) for q in inverses}),
                "full_classes": len(keep),
                "true_event_members": sum(1 for pi in keep
                                          if pi[A] == A and pi[S] == S),
                "true_goal_members": sum(1 for pi in keep if pi[A] == A
                                         and pi[GA] == GA and pi[GB] == GB),
                "survivors": [list(pi) for pi in keep],
            })
    return rows


def reduce_stage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = np.array([r["class_size"] for r in rows], dtype=float)
    histogram = Counter(int(s) for s in sizes)
    event_mass = np.array([r["true_event_members"] / r["class_size"] for r in rows])
    goal_mass = np.array([r["true_goal_members"] / r["class_size"] for r in rows])
    return {
        "episodes": len(rows),
        "histogram": {str(k): histogram[k] for k in sorted(histogram)},
        "histogram_total": int(sum(histogram.values())),
        "histogram_weighted_sum": float(sum(k * v for k, v in histogram.items())),
        "arithmetic_mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "minimum": int(sizes.min()),
        "maximum": int(sizes.max()),
        "mean_log2_class_size": float(np.log2(sizes).mean()),
        "posterior_entropy_bits": float(np.log2(sizes).mean()),
        "entropy_equals_mean_log2": True,
        "true_exact_map_mass": float((1.0 / sizes).mean()),
        "event_quotient_classes_mean": float(np.mean([r["event_classes"] for r in rows])),
        "goal_quotient_classes_mean": float(np.mean([r["goal_classes"] for r in rows])),
        "full_quotient_classes_mean": float(sizes.mean()),
        "true_event_class_mass": float(event_mass.mean()),
        "true_goal_class_mass": float(goal_mass.mean()),
        "true_full_class_mass": float((1.0 / sizes).mean()),
        "event_identified_fraction": float(np.mean(event_mass == 1.0)),
        "goal_identified_fraction": float(np.mean(goal_mass == 1.0)),
        "full_identified_fraction": float(np.mean(sizes == 1)),
    }


def oversize_detail(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """EVERY episode above class 2, with EVERY surviving permutation spelled out."""
    out = []
    for row in rows:
        if row["class_size"] <= 2:
            continue
        out.append({
            "layout": row["layout"], "class_size": row["class_size"],
            "surviving_permutations": [
                {ROLES[i]: ROLES[pi[i]] for i in range(N_ROLES) if pi[i] != i} or
                {"__identity__": "identity"}
                for pi in row["survivors"]],
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o2-equivalence.json")
    parser.add_argument("--goal-calibration-layout", type=int, default=115_000)
    arguments = parser.parse_args()
    started = time.perf_counter()

    report: dict[str, Any] = {
        "quotient_definitions": {
            "event": "pi fixes AGENT and SWITCH; class key is (pi^-1[AGENT], pi^-1[SWITCH])",
            "goal": "pi fixes AGENT, GOAL_ALPHA and GOAL_BETA",
            "full": "pi is the identity",
        },
        "stages": [name for name, _ in STAGES],
        "populations": {}, "episode_table": [],
    }

    for name, spec in POPULATIONS.items():
        episodes = O.collect_appearance(
            list(range(110_000, 110_000 + spec["layouts"])),
            "HIDDEN_PALETTE_CONVENTION", [spec["palette"]], 1, 9, seed=11,
            policy=spec["policy"])
        rows = episode_rows(episodes, name, goal_demonstrated=True)
        block_shows_marker = sum(1 for e in episodes if reached_named_marker(e))
        report["episode_table"].extend(rows)
        block: dict[str, Any] = {
            "manifest": {**spec, "episodes": len(episodes),
                         "layouts_used": sorted({e.layout for e in episodes}),
                         "episodes_recording_the_named_marker_occupied":
                             int(block_shows_marker),
                         "digest": O.digest_episodes(episodes)},
            "stages": {},
        }
        print(f"\n=== {name}: {len(episodes)} episodes, palette {spec['palette']}, "
              f"policy {spec['policy']} ===", flush=True)
        print(f"{'stage':38s} {'mean':>7s} {'med':>5s} {'min':>4s} {'max':>4s} "
              f"{'H bits':>7s} {'full':>7s} {'event':>7s} {'goal':>7s}")
        print("-" * 100)
        for stage, _ in STAGES:
            stage_rows = [r for r in rows if r["stage"] == stage]
            reduced = reduce_stage(stage_rows)
            reduced["episodes_above_class_two"] = oversize_detail(stage_rows)
            block["stages"][stage] = reduced
            print(f"{stage:38s} {reduced['arithmetic_mean']:7.4f} "
                  f"{reduced['median']:5.1f} {reduced['minimum']:4d} "
                  f"{reduced['maximum']:4d} {reduced['posterior_entropy_bits']:7.4f} "
                  f"{reduced['true_full_class_mass']:7.4f} "
                  f"{reduced['true_event_class_mass']:7.4f} "
                  f"{reduced['true_goal_class_mass']:7.4f}", flush=True)
        report["populations"][name] = block

    # ---- the reconciliation ---------------------------------------------------------
    grounded = "4_grounded_event_calibration"
    o_block = report["populations"]["O_phase_24_uniform"]["stages"][grounded]
    o1_block = report["populations"]["O1_phase_48_goal_directed"]["stages"][grounded]
    report["reconciliation"] = {
        "reported_in_O": 2.08,
        "reported_histogram_in_O1": {"2": 44, "4": 1, "12": 2},
        "O_population_recomputed_mean": o_block["arithmetic_mean"],
        "O_population_recomputed_histogram": o_block["histogram"],
        "O1_population_recomputed_mean": o1_block["arithmetic_mean"],
        "O1_population_recomputed_histogram": o1_block["histogram"],
        "histogram_mean_check": {
            "weighted_sum": o1_block["histogram_weighted_sum"],
            "count": o1_block["histogram_total"],
            "quotient": o1_block["histogram_weighted_sum"] / o1_block["histogram_total"],
        },
        "cause": ("DIFFERENT POPULATIONS, not a stale report and not a different "
                  "weighting. 2.08 is the mean over the 24-episode uniform-policy "
                  "population phase O built at palette 7001; 2.468 is the mean over the "
                  "47-episode goal-directed population phase O1 built at palette 7101. "
                  "Both are arithmetic means of class size over their own episodes and "
                  "both are reproduced here from one function."),
        "why_the_larger_population_is_worse": (
            "the goal-directed policy reaches the marker and stops, so its episodes are "
            "shorter and two of them never move the agent at all, which leaves the "
            "one-frame class of 12 standing"),
        "consistent": bool(
            abs(o_block["arithmetic_mean"] - 2.0833333333333335) < 1e-9
            and abs(o1_block["arithmetic_mean"] - 116 / 47) < 1e-9),
    }

    # ---- section K's premise, verified rather than asserted -------------------------
    demonstration = C.goal_demonstration(arguments.goal_calibration_layout,
                                         C.sample_bijection(7_101), "alpha")
    report["goal_grounding_calibration"] = {
        "layout": arguments.goal_calibration_layout,
        "disjoint_from_evaluation_layouts": bool(
            arguments.goal_calibration_layout not in range(110_000, 110_048)),
        "built": demonstration is not None,
        "steps": demonstration["steps"] if demonstration else None,
        "terminal_frame_retained": bool(demonstration
                                        and demonstration["terminal_frame_retained"]),
        "named_marker_colour_matches_the_bijection": bool(
            demonstration and tuple(demonstration["named_marker_colour"])
            == tuple(int(v) for v in
                     O.COLOUR_POOL[C.sample_bijection(7_101)[GA]])),
        "why_ordinary_episodes_cannot_do_this": (
            "the adapter terminates on arrival and every collector appends the frame "
            "BEFORE stepping, so no recorded public history has ever depicted the goal "
            "marker occupied; language naming it therefore binds no colour, which is "
            "what stage 6 measures"),
        "minimum_separating_calibration_size": 1,
        "minimum_size_basis": ("one demonstration pins the named marker directly and the "
                               "other by elimination, because the surviving class at "
                               "stage 5 is generated by the alpha/beta transposition"),
    }
    q1 = report["reconciliation"]["consistent"]
    q2 = all("event_quotient_classes_mean" in block["stages"][s]
             for block in report["populations"].values() for s, _ in STAGES)
    report["Q1_class_arithmetic_internally_consistent"] = bool(q1)
    report["Q2_quotients_explicit"] = bool(q2)
    report["goal_identified_only_with_stage_7"] = {
        stage: report["populations"]["O1_phase_48_goal_directed"]["stages"][stage][
            "goal_identified_fraction"] for stage, _ in STAGES}
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)

    r = report["reconciliation"]
    print(f"\nO population mean {r['O_population_recomputed_mean']:.4f} "
          f"histogram {r['O_population_recomputed_histogram']}")
    print(f"O1 population mean {r['O1_population_recomputed_mean']:.4f} "
          f"histogram {r['O1_population_recomputed_histogram']}  "
          f"({r['histogram_mean_check']['weighted_sum']:.0f}/"
          f"{r['histogram_mean_check']['count']} = "
          f"{r['histogram_mean_check']['quotient']:.4f})")
    print(f"Q1 {report['Q1_class_arithmetic_internally_consistent']}   "
          f"Q2 {report['Q2_quotients_explicit']}")
    print(f"goal identified by stage: {report['goal_identified_only_with_stage_7']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
