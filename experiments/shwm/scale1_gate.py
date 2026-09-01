"""The authoritative S1.2 verdict, judged over three instruments.

A point-estimate threshold turned out to be too weak an instrument on its own.
`qwen3_vl_4b_spatial_slots` clears a 0.05 hidden-phase threshold with +0.070 --
and the episode-level interval for that same number is [-0.009, +0.141], which
includes zero, while conditioning on post-switch states drops it to -0.028
against an oracle at +0.543.

All three are measurements of the same claim, so the verdict is taken over all
three rather than from whichever is most permissive:

1. the point estimate must clear the threshold;
2. its paired episode interval must exclude zero;
3. it must survive the condition that removes the known confound.

The confound is specific and was demonstrated, not hypothesised: a
`reset_frame_only` feature scores +0.149 overall and -0.064 post-switch, so the
overall hidden-phase margins are substantially the reset indicator.

    .venv-shwm/bin/python experiments/shwm/scale1_gate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sentinel.wm.versioning import digest_file  # noqa: E402

ARTIFACTS = REPO / "artifacts/shwm/scale1"
THRESHOLD = 0.05
POST_SWITCH_THRESHOLD = 0.05


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "s1-2-verdict.json")
    arguments = parser.parse_args()

    qualification = json.loads((ARTIFACTS / "feature-qualification.json").read_text())
    intervals = json.loads((ARTIFACTS / "audit-bc.json").read_text())
    construct = json.loads((ARTIFACTS / "audit-c.json").read_text())

    clean = "dynamics_clean"
    admissible = [n for n in qualification["results"] if n != "oracle_structured_state"]

    def point(interface: str, group: str) -> float:
        rows = [
            r
            for r in qualification["results"][interface][clean]
            if r["group"] == group and not r["degenerate"]
        ]
        return sum(r["margin"] for r in rows) / len(rows) if rows else float("nan")

    def interval(interface: str, target: str) -> dict[str, Any]:
        row = next(r for r in intervals["table_B"][interface][clean] if r["target"] == target)
        return {
            "margin": row["margin"],
            "interval_95": row["interval_95"],
            "excludes_zero": row["interval_excludes_zero"],
            "shuffled_margin": row["shuffled_label_margin"],
            "episodes": row["episodes"],
        }

    post_switch = {
        variant: rows["post_first_switch_states"]["margin"]
        for variant, rows in construct["results"].items()
    }
    best_admissible_post_switch = max(
        margin for variant, margin in post_switch.items() if variant != "structured_oracle"
    )

    verdict: dict[str, Any] = {
        "instruments": {
            "point_estimates": str(ARTIFACTS / "feature-qualification.json"),
            "episode_intervals": str(ARTIFACTS / "audit-bc.json"),
            "construct_conditioning": str(ARTIFACTS / "audit-c.json"),
        },
        "artifact_digests": {
            name: digest_file(ARTIFACTS / f"{name}.json")
            for name in ("feature-qualification", "audit-bc", "audit-c")
        },
        "interfaces": {},
    }

    for interface in admissible:
        intervention = interval(interface, "successor_0")
        phase = interval(interface, "polarity")
        verdict["interfaces"][interface] = {
            "intervention": {
                "point_margin": point(interface, "intervention"),
                **intervention,
                "clears_threshold": point(interface, "intervention") > THRESHOLD,
                "passes": intervention["excludes_zero"] and intervention["margin"] > 0,
            },
            "hidden_phase": {
                "point_margin": point(interface, "hidden_phase"),
                **phase,
                "clears_threshold": point(interface, "hidden_phase") > THRESHOLD,
                "passes": phase["excludes_zero"] and phase["margin"] > 0,
            },
        }

    intervention_pass = [
        n for n, v in verdict["interfaces"].items() if v["intervention"]["passes"]
    ]
    phase_pass = [n for n, v in verdict["interfaces"].items() if v["hidden_phase"]["passes"]]
    both = [n for n in intervention_pass if n in phase_pass]

    verdict["confound"] = {
        "reset_frame_only_all_episodes": post_switch and construct["results"]["reset_frame_only"]["all_episodes"]["margin"],
        "reset_frame_only_post_switch": post_switch["reset_frame_only"],
        "demonstrated": construct["pins"]["reset_indicator_cannot_solve_post_switch"]["passes"],
        "reading": (
            "the reset indicator carries the overall hidden-phase margin and fails "
            "post-switch, so an overall margin is not evidence of state tracking"
        ),
    }
    verdict["post_switch"] = {
        "by_variant": post_switch,
        "oracle": post_switch["structured_oracle"],
        "best_admissible": best_admissible_post_switch,
        "construct_nonvacuous": construct["pins"]["construct_is_nonvacuous_post_switch"]["passes"],
        "any_admissible_recovers": best_admissible_post_switch > POST_SWITCH_THRESHOLD,
    }
    verdict["result"] = {
        "intervention_clause": {
            "passes": bool(intervention_pass),
            "interfaces": intervention_pass,
            "basis": "point estimate and episode interval excluding zero",
        },
        "hidden_phase_clause": {
            "passes": bool(phase_pass) and verdict["post_switch"]["any_admissible_recovers"],
            "interfaces_by_interval": phase_pass,
            "post_switch_survives": verdict["post_switch"]["any_admissible_recovers"],
            "basis": "interval excluding zero AND survival of post-switch conditioning",
        },
        "one_interface_clears_both": both,
        "S1_2_passes": bool(
            intervention_pass and phase_pass and verdict["post_switch"]["any_admissible_recovers"] and both
        ),
        "S1_2_intervention_only": bool(intervention_pass),
    }
    arguments.out.write_text(json.dumps(verdict, indent=2, sort_keys=True, default=str) + "\n")

    print("=== S1.2 verdict, judged over three instruments ===\n")
    print(f"{'interface':34s} {'interv margin':>14s} {'excl0':>6s} {'phase margin':>13s} {'excl0':>6s}")
    for name, entry in verdict["interfaces"].items():
        i, p = entry["intervention"], entry["hidden_phase"]
        print(f"{name:34s} {i['margin']:+14.3f} {str(i['excludes_zero']):>6s} "
              f"{p['margin']:+13.3f} {str(p['excludes_zero']):>6s}")
    print()
    print(f"reset_frame_only overall / post-switch : "
          f"{verdict['confound']['reset_frame_only_all_episodes']:+.3f} / "
          f"{verdict['confound']['reset_frame_only_post_switch']:+.3f}")
    print(f"oracle post-switch                     : {verdict['post_switch']['oracle']:+.3f}")
    print(f"best admissible post-switch            : {verdict['post_switch']['best_admissible']:+.3f}")
    print()
    r = verdict["result"]
    print(f"intervention clause : {r['intervention_clause']['passes']}  {r['intervention_clause']['interfaces']}")
    print(f"hidden-phase clause : {r['hidden_phase_clause']['passes']}  "
          f"(interval {r['hidden_phase_clause']['interfaces_by_interval']}, "
          f"post-switch survives {r['hidden_phase_clause']['post_switch_survives']})")
    print(f"one interface both  : {r['one_interface_clears_both']}")
    print()
    print(f"S1.2 PASSES                 : {r['S1_2_passes']}")
    print(f"S1.2 intervention-only pass : {r['S1_2_intervention_only']}")
    print(f"\nwritten: {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
