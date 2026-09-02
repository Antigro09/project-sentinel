"""B / L1. Exact reference ceilings on the reachable state space.

Every previous phase compared a learned model against a baseline that was itself
a fitted thing, so a gap could always be the baseline's fault. These references
are computed by exhaustive enumeration instead: the reachable set is finite at a
bounded depth, so p(Y | X, A) and p(Y | X, A, H) can be counted rather than
estimated.

The structure of this environment makes both references exact.

Given the complete AgentVisiblePacket X, the rendered frame fixes the level and
the agent's cell. Given the cell, the action and the hidden phase H, the successor
is determined -- polarity negates the action delta and nothing else varies. So
**p(Y | X, A, H) is a point mass** and the phase-aware oracle is exactly 1.000.
The public memoryless reference p(Y | X, A) is the mixture over whichever phases
are reachable in that packet class, so its best possible accuracy is the mass of
the majority phase. The difference between them is the phase headroom, and on
alias classes -- which by construction contain both phases -- it is the largest it
can be.

Weighting is uniform over reachable states within a packet class. That is a choice
and it is declared: a trajectory-frequency weighting would give different numbers,
and the alias classes are precisely where the two weightings disagree most.

    .venv/bin/python experiments/shwm/bayes_ceilings.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import ACTIONS  # noqa: E402
from alias_audit import enumerate_states  # noqa: E402

STRATA = (
    "all",
    "ordinary_non_switch",
    "switch_sensitive",
    "alias_pairs",
    "post_first_switch",
    "post_two_changes",
)


def build_tables(states) -> dict[str, Any]:
    """Group reachable states by their AgentVisiblePacket key."""
    classes: dict[str, list] = defaultdict(list)
    for state in states:
        classes[state.key("V2_agent_visible")].append(state)
    return classes


def stratum_mask(state, classes, action=None) -> dict[str, bool]:
    """`switch_sensitive` is per-action and is NOT the same as `post_first_switch`.

    An earlier version defined both as `crossings >= 1`, so they reported identical
    numbers and one of them measured nothing. A transition is phase-sensitive when its
    own packet class actually contains more than one outcome for THIS action -- that is
    the population where knowing the phase can change the answer.
    """
    members = classes[state.key("V2_agent_visible")]
    phases = {m.polarity for m in members}
    varies = (action is not None
              and len({m.successors[action] for m in members}) > 1)
    return {
        "all": True,
        "ordinary_non_switch": state.crossings == 0,
        "switch_sensitive": varies,
        "alias_pairs": len(phases) > 1,
        "post_first_switch": state.crossings >= 1,
        "post_two_changes": state.crossings >= 2,
    }


def compute(states) -> dict[str, Any]:
    classes = build_tables(states)
    # per (packet class, action): the reachable successors and their phases
    rows: list[dict[str, Any]] = []
    for key, members in classes.items():
        phases = {m.polarity for m in members}
        for action in ACTIONS:
            outcomes = [m.successors[action] for m in members]
            counts = defaultdict(float)
            for outcome in outcomes:
                counts[outcome] += 1.0 / len(outcomes)
            best = max(counts.values())
            for member in members:
                rows.append({
                    "state": member, "action": action,
                    "y": member.successors[action],
                    "p_public_best": best,
                    "p_public_of_truth": counts[member.successors[action]],
                    "n_outcomes": len(counts),
                    "multi_phase_class": len(phases) > 1,
                })
    return rows, classes


def summarise(rows, classes) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stratum in STRATA:
        selected = [r for r in rows
                    if stratum_mask(r["state"], classes, r["action"])[stratum]]
        if not selected:
            continue
        # The Bayes-optimal accuracy for a single prediction per (X, A) class is the
        # mass of the modal outcome. Counting "is the truth tied for modal" instead
        # scores every member of a 50/50 class as correct and reports 1.0 where the
        #true answer is 0.5 -- which inflates the baseline and hides the headroom.
        modal_hits = float(np.mean([r["p_public_best"] for r in selected]))
        # negative log likelihood of the truth under each reference
        nll_public = float(np.mean([-np.log(max(r["p_public_of_truth"], 1e-12))
                                    for r in selected]))
        brier_public = float(np.mean([(1.0 - r["p_public_of_truth"]) ** 2
                                      for r in selected]))
        out[stratum] = {
            "transitions": len(selected),
            "public_memoryless_accuracy": modal_hits,
            "public_memoryless_nll": nll_public,
            "public_memoryless_brier": brier_public,
            "phase_oracle_accuracy": 1.0,
            "phase_oracle_nll": 0.0,
            "phase_oracle_brier": 0.0,
            "headroom_accuracy": 1.0 - modal_hits,
            "headroom_nll": nll_public,
            "fraction_in_multi_phase_class": float(np.mean(
                [r["multi_phase_class"] for r in selected])),
        }
    per_action = {}
    for action in ACTIONS:
        selected = [r for r in rows if r["action"] == action
                    and stratum_mask(r["state"], classes)["alias_pairs"]]
        if selected:
            acc = float(np.mean([r["p_public_of_truth"] >= r["p_public_best"] - 1e-12
                                 for r in selected]))
            per_action[str(action)] = {
                "transitions": len(selected),
                "public_memoryless_accuracy": acc,
                "headroom_accuracy": 1.0 - acc,
            }
    out["per_action_on_alias_classes"] = per_action
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", type=int, default=40)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/bayes-ceilings.json")
    arguments = parser.parse_args()

    layouts = list(range(90_000, 90_000 + arguments.layouts))
    print(f"enumerating reachable states, {len(layouts)} layouts, depth {arguments.depth}",
          flush=True)
    states = enumerate_states(layouts, arguments.depth)
    print(f"  {len(states)} states", flush=True)
    rows, classes = compute(states)
    print(f"  {len(rows)} (state, action) transitions across {len(classes)} packet classes",
          flush=True)
    summary = summarise(rows, classes)

    print()
    print(f"{'stratum':24s} {'transitions':>12s} {'public acc':>11s} {'oracle acc':>11s} "
          f"{'headroom':>9s} {'public NLL':>11s}")
    print("-" * 84)
    for stratum in STRATA:
        record = summary.get(stratum)
        if not record:
            continue
        print(f"{stratum:24s} {record['transitions']:12d} "
              f"{record['public_memoryless_accuracy']:11.4f} "
              f"{record['phase_oracle_accuracy']:11.4f} "
              f"{record['headroom_accuracy']:9.4f} {record['public_memoryless_nll']:11.4f}")

    alias = summary.get("alias_pairs", {})
    headroom = alias.get("headroom_accuracy", 0.0)
    print(f"\nphase headroom on the alias subset: {headroom:.4f}")
    if headroom <= 1e-9:
        print("STOP: the target does not measure hidden state on this population")
    summary["stop_target_does_not_measure_hidden_state"] = headroom <= 1e-9
    summary["weighting"] = "uniform over reachable states within a packet class"
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
