"""E. Frozen calibration policies at equal interaction budget.

Section D already constrains what this section can find. On every one of the 64
validation palettes the exact event class is pinned after 2 to 6 of roughly 48 available
interactions, so almost the entire budget is spent after the question is already settled.
A comparison of calibration policies therefore has very little room to separate them, and
the honest result is most likely "no policy matters much, and here is the measurement
that shows why" rather than a ranking.

That is written down BEFORE the run so the result cannot be presented as a discovery
either way. What would falsify it is a policy that pins later than the others or fails to
pin at all inside the budget; the pinning curve is reported per policy so that is visible.

Eight policies, all FROZEN -- fixed in advance, none adapted to what the run sees -- and
all consuming the SAME number of frame pairs. They vary along two axes that are the only
ones the generator exposes: which action rule drives the agent (uniform, switch-seeking,
goal-seeking) and how the budget is spread over layouts (concentrated on one, spread
across many, many-short against few-long).

    .venv-shwm/bin/python experiments/shwm/o3_policy.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import o2_core as C
import o2_memory as mem
import o2_models as M
import o3_core as O3
import o3_population as pop
from m2d_core import ARTIFACTS, write

SEED = 77_000
VIEW = "no_rgb"
BUDGET_PAIRS = 48          # every policy consumes exactly this many frame pairs
STRATUM = "COUNT_COLLISION"

# (action rule, layouts, steps per episode). steps-1 pairs per episode, so
# layouts * (steps - 1) == BUDGET_PAIRS for every row.
POLICIES: dict[str, tuple[str, int, int]] = {
    "1_uniform_spread":             ("uniform",        6,  9),
    "2_uniform_concentrated":       ("uniform",        1, 49),
    "3_switch_seeking_spread":      ("switch_seeking", 6,  9),
    "4_switch_seeking_concentrated": ("switch_seeking", 1, 49),
    "5_goal_seeking_spread":        ("goal_seeking",   6,  9),
    "6_goal_seeking_concentrated":  ("goal_seeking",   1, 49),
    "7_many_short":                 ("uniform",       12,  5),
    "8_few_long":                   ("uniform",        3, 17),
}

PREREGISTERED = (
    "Section D measured the event class pinned after 2-6 of ~48 interactions on every "
    "validation palette. The expected result here is therefore that policies do not "
    "separate on the event query, and that any separation appears on the GOAL query, "
    "which section D found identified on 0 of 64 palettes. A policy that fails to pin "
    "inside the budget would falsify the expectation.")


def balanced(correct: np.ndarray, truth: np.ndarray) -> float:
    """Mean of the per-class accuracies on contested rows.

    Plain accuracy cannot be used here. Each palette draws its own layouts, so the
    SWITCH-against-DECOY base rate is a per-palette quantity -- measured at 0.5898
    overall and 0.2667 to 0.8000 across palettes -- and a constant answer scores the base
    rate rather than chance. On the balanced statistic a constant answer scores 0.5.
    """
    parts = [correct[truth == value].mean() for value in (0.0, 1.0)
             if (truth == value).any()]
    return float(np.mean(parts)) if parts else float("nan")


def build(plan: dict, policy: str, palette_seed: int) -> list[C.O2Episode]:
    """Calibration episodes under one frozen policy, at the fixed pair budget."""
    rule, layouts, steps = POLICIES[policy]
    action_policy = {"uniform": "uniform", "switch_seeking": "switch_seeking",
                     "goal_seeking": "goal"}[rule]
    pool = plan["calibration_layouts"]
    if layouts <= len(pool):
        chosen = pool[:layouts]
        episodes = C.collect(chosen, plan["bijection"], STRATUM, steps,
                             seed=palette_seed, policy=action_policy)
    else:
        # More layouts than the plan drew: extend from the shared calibration pool,
        # deterministically and disjointly from the transfer pool.
        extra = [int(v) for v in pop.CAL_POOL
                 if int(v) not in set(pool)][:layouts - len(pool)]
        chosen = list(pool) + extra
        episodes = C.collect(chosen, plan["bijection"], STRATUM, steps,
                             seed=palette_seed, policy=action_policy)
    if layouts == 1:
        # Concentration means repeating the SAME layout under fresh action seeds, not
        # one impossibly long episode: the adapter terminates when the goal is reached.
        episodes = []
        remaining = BUDGET_PAIRS
        repeat = 0
        while remaining > 0 and repeat < 40:
            block = C.collect(pool[:1], plan["bijection"], STRATUM, 9,
                              seed=palette_seed + 1_000 * repeat, policy=action_policy)
            episodes.extend(block)
            remaining -= sum(e.length - 1 for e in block)
            repeat += 1
    return episodes


def spend(episodes: list[C.O2Episode], budget: int) -> list[C.O2Episode]:
    """Truncate the stream to exactly `budget` frame pairs, so every policy is compared
    at the same interaction cost rather than at the same episode count."""
    kept, used = [], 0
    for episode in episodes:
        pairs = episode.length - 1
        if used + pairs <= budget:
            kept.append(episode)
            used += pairs
        if used >= budget:
            break
    return kept


def pinning_curve(episodes: list[C.O2Episode]) -> dict[str, Any]:
    """After how many INTERACTIONS the exact event class is pinned."""
    consumed, curve, pinned_at = 0, [], None
    survivors = None
    for episode in episodes:
        survivors = (C.survivors_over([episode]) if survivors is None
                     else C.survivors_over([episode], candidates=survivors))
        consumed += episode.length - 1
        mass = C.event_quotient_mass(survivors)
        curve.append({"interactions": consumed, "class_size": len(survivors),
                      "event_mass": float(mass)})
        if pinned_at is None and mass >= 1.0:
            pinned_at = consumed
    return {"curve": curve, "interactions_to_pin_event": pinned_at,
            "final_class_size": len(survivors) if survivors is not None else None,
            "final_event_mass": curve[-1]["event_mass"] if curve else None,
            "interactions_used": consumed}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-palettes", type=int, default=32)
    parser.add_argument("--palettes", type=int, default=32)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o3-policy.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    registry = C.canonical_registry()

    print(f"budget {BUDGET_PAIRS} frame pairs per policy", flush=True)
    for name, (rule, layouts, steps) in POLICIES.items():
        print(f"  {name:32s} {rule:15s} {layouts:3d} layouts x {steps:3d} steps",
              flush=True)

    print(f"\ntraining the memory on {arguments.train_palettes} development palettes",
          flush=True)
    dev_groups = [mem.Group(p, C.sample_bijection(p),
                            *(lambda s: (s.calibration, s.transfer))(
                                pop.palette_scenario(pop.palette_plan(p, 6, 20, 2))))
                  for p in pop.DEV_PALETTES[:arguments.train_palettes]]
    train = mem.stack_groups(dev_groups, registry, VIEW)
    infer, model = M.train_memory(
        (train["sequence"], train["mask"], train["before"], train["after"],
         train["event"]), SEED, updates=mem.MEMORY_UPDATES)

    report: dict[str, Any] = {
        "seed": SEED, "view": VIEW, "budget_pairs": BUDGET_PAIRS,
        "preregistered_expectation": PREREGISTERED,
        "policies": {k: {"action_rule": v[0], "layouts": v[1], "steps": v[2]}
                     for k, v in POLICIES.items()},
        "palettes": list(pop.VALIDATION_PALETTES[:arguments.palettes]),
        "results": {},
    }

    print(f"\n{'policy':32s} {'pairs':>7s} {'pin@':>7s} {'pinned':>8s} "
          f"{'class':>7s} {'contested':>10s}")
    print("-" * 82)
    for policy in POLICIES:
        rows = []
        for palette in report["palettes"]:
            plan = pop.palette_plan(palette, 6, 8, 1)
            episodes = spend(build(plan, policy, palette), BUDGET_PAIRS)
            if not episodes:
                continue
            pin = pinning_curve(episodes)
            group = mem.Group(palette, plan["bijection"], episodes,
                              pop.palette_scenario(plan).transfer)
            data = mem.stack_groups([group], registry, VIEW)
            logits = infer((data["sequence"], data["mask"], data["before"],
                            data["after"]))
            hit = ((logits > 0).astype(float) == data["event"]).astype(float)
            contested = mem.contested(data)
            answer = data["event"][contested]
            rows.append({
                "palette": palette,
                "episodes": len(episodes),
                "interactions": pin["interactions_used"],
                "interactions_to_pin_event": pin["interactions_to_pin_event"],
                "final_class_size": pin["final_class_size"],
                "final_event_mass": pin["final_event_mass"],
                "contested_balanced_accuracy": balanced(hit[contested], answer),
                "contested_plain_accuracy": float(hit[contested].mean()),
                "switch_base_rate": float(answer.mean()),
                "contested_rows": int(contested.sum()),
            })
        pins = [r["interactions_to_pin_event"] for r in rows
                if r["interactions_to_pin_event"] is not None]
        block = {
            "palettes": len(rows),
            "mean_interactions": float(np.mean([r["interactions"] for r in rows])),
            "mean_interactions_to_pin_event": float(np.mean(pins)) if pins else None,
            "pinned_fraction": float(len(pins) / len(rows)) if rows else None,
            "mean_final_class_size": float(np.mean(
                [r["final_class_size"] for r in rows])),
            "mean_final_event_mass": float(np.mean(
                [r["final_event_mass"] for r in rows])),
            "contested_balanced_accuracy": float(np.nanmean(
                [r["contested_balanced_accuracy"] for r in rows])),
            "contested_plain_accuracy": float(np.mean(
                [r["contested_plain_accuracy"] for r in rows])),
            "mean_switch_base_rate": float(np.mean(
                [r["switch_base_rate"] for r in rows])),
            "per_palette": rows,
        }
        report["results"][policy] = block
        print(f"{policy:32s} {block['mean_interactions']:7.1f} "
              f"{(block['mean_interactions_to_pin_event'] or -1):7.2f} "
              f"{(block['pinned_fraction'] or 0):8.4f} "
              f"{block['mean_final_class_size']:7.2f} "
              f"{block['contested_balanced_accuracy']:10.4f}", flush=True)

    accuracies = {k: v["contested_balanced_accuracy"]
                  for k, v in report["results"].items()}
    pin_rates = {k: v["pinned_fraction"] for k, v in report["results"].items()}
    best, worst = max(accuracies, key=accuracies.get), min(accuracies,
                                                           key=accuracies.get)
    report["spread"] = {
        "best_policy": best, "worst_policy": worst,
        "accuracy_spread": accuracies[best] - accuracies[worst],
        "all_policies_pin_the_event": bool(all(v == 1.0 for v in pin_rates.values())),
        "statistic": ("balanced accuracy on contested rows; a constant answer scores "
                      "0.5 whatever the palette's base rate"),
        "reading": ("a spread this size against a pinning budget already exhausted "
                    "after a handful of interactions means the calibration POLICY is "
                    "not the binding constraint; what binds is what the observations "
                    "can identify at all, which section D measured"),
    }
    # R5 asks whether policy choice matters at a fixed budget. It is answered either way:
    # a real separation OR a demonstrated absence of headroom both close it, and the
    # second is only credible if every policy actually pins.
    report["R5_calibration_policy_compared_at_equal_budget"] = bool(
        report["spread"]["all_policies_pin_the_event"])
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nbest {best} {accuracies[best]:.4f}   worst {worst} "
          f"{accuracies[worst]:.4f}   spread "
          f"{report['spread']['accuracy_spread']:.4f}")
    print(f"R5 {report['R5_calibration_policy_compared_at_equal_budget']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
