"""Turn the slot-resolution audit's raw records into the decision the rule asks for.

The non-inferiority margin below is fixed here, in a commit that precedes the
run that produces the numbers it judges. That ordering is the point: a margin
chosen after seeing the intervention deltas is not a margin, it is a description
of the result. Git history is what makes the claim checkable rather than
asserted.

    .venv/bin/python experiments/shwm/slot_audit_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

INTERVENTION_NON_INFERIORITY_MARGIN = -0.02
"""How far the intervention margin may fall before a finer grid is rejected.

Pre-registered. Two points of held-out R-squared is the largest drop that would
still leave the counterfactual-prediction result -- the one part of S1.2 that
qualified -- comfortably clear of its controls.
"""

SWITCH_TARGETS = ("crossed_now", "on_switch")
PHASE_TARGETS = ("polarity",)
OUTCOME_TARGETS = ("successor_0", "successor_1", "successor_2", "successor_3")
PRETRAINED = ("qwen3_vl_4b_spatial_slots", "gemma3_4b_spatial_slots")
PIXEL = ("raw_lowres_spatial", "learned_cnn_spatial_slots", "fixed_random_spatial_projection")
REFERENCE = "g4x4x256"


def recovers(rows, sources, targets, stratum="all") -> list[dict[str, Any]]:
    """Arms whose margin is positive with a bootstrap interval clear of zero."""
    return [
        r for r in rows
        if r["source"] in sources and r["target"] in targets and r["stratum"] == stratum
        and r["geometry"] == REFERENCE and r["ci_low"] > 0.0
    ]


def improved(deltas, sources, targets, geometry, stratum="all") -> list[dict[str, Any]]:
    return [
        d for d in deltas
        if d["source"] in sources and d["target"] in targets
        and d["geometry"] == geometry and d["stratum"] == stratum and d["improves"]
    ]


def worst_intervention_delta(deltas, geometry) -> float:
    relevant = [
        d for d in deltas
        if d["target"] in OUTCOME_TARGETS and d["geometry"] == geometry and d["stratum"] == "all"
    ]
    return min((d["ci_low"] for d in relevant), default=0.0)


def build_findings(report) -> dict[str, Any]:
    rows = report["ridge_results"]
    deltas = report.get("geometry_deltas", [])
    all_sources = PRETRAINED + PIXEL

    fine_switch = improved(deltas, all_sources, SWITCH_TARGETS, "g8x8x64")
    fine_phase = improved(deltas, all_sources, PHASE_TARGETS, "g8x8x64", "post_first_switch")
    fine_outcome = improved(deltas, all_sources, OUTCOME_TARGETS, "g8x8x64")
    high_switch = improved(deltas, all_sources, SWITCH_TARGETS, "g8x8x256")

    return {
        "fine_grid_improves_switch_detection": bool(fine_switch),
        "fine_grid_improves_phase_or_outcome": bool(fine_phase or fine_outcome),
        "intervention_non_inferior": worst_intervention_delta(deltas, "g8x8x64")
        > INTERVENTION_NON_INFERIORITY_MARGIN,
        "effect_in_a_pretrained_package": any(d["source"] in PRETRAINED for d in fine_switch),
        "high_capacity_works_matched_does_not": bool(high_switch) and not bool(fine_switch),
        "coarse_recovers_switch_events": bool(recovers(rows, all_sources, SWITCH_TARGETS)),
        "any_non_oracle_recovers_events": bool(recovers(rows, all_sources, SWITCH_TARGETS)),
        "pixel_sources_recover_events": bool(recovers(rows, PIXEL, SWITCH_TARGETS)),
    }


def best_by_link(rows) -> dict[str, dict[str, Any]]:
    """The strongest arm for each link in the chain, so a break is visible."""
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["source"] == "oracle_structured_state":
            continue
        key = f"{row['link']}|{row['stratum']}"
        if key not in best or row["margin"] > best[key]["margin"]:
            best[key] = row
    return best


def oracle_by_link(rows) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["source"] != "oracle_structured_state":
            continue
        key = f"{row['link']}|{row['stratum']}"
        if key not in best or row["margin"] > best[key]["margin"]:
            best[key] = row
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path,
                        default=REPO / "artifacts/shwm/scale1/slot-resolution-audit.json")
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/slot-resolution-decision.json")
    arguments = parser.parse_args()

    report = json.loads(arguments.audit.read_text())
    rows = report["ridge_results"]
    deltas = report.get("geometry_deltas", [])

    print("=" * 78)
    print("CAUSAL CHAIN  (best non-oracle arm, with the oracle as the ceiling)")
    print("=" * 78)
    best, oracle = best_by_link(rows), oracle_by_link(rows)
    for key in sorted(best):
        row = best[key]
        ceiling = oracle.get(key)
        ceiling_text = f"  oracle {ceiling['margin']:+.3f}" if ceiling else ""
        print(f"{key:34s} {row['margin']:+.3f} [{row['ci_low']:+.3f},{row['ci_high']:+.3f}]"
              f"  {row['source'][:26]:26s} {row['geometry']:10s} {row['condition'][:24]:24s}"
              f"{ceiling_text}")

    print()
    print("=" * 78)
    print(f"GEOMETRY vs {REFERENCE}  (paired by episode; only intervals clear of zero)")
    print("=" * 78)
    shown = 0
    for delta in sorted(deltas, key=lambda d: -abs(d["delta"])):
        if not delta["excludes_zero"]:
            continue
        shown += 1
        if shown > 30:
            continue
        direction = "better" if delta["improves"] else "WORSE "
        print(f"{delta['geometry']:10s} {direction} {delta['delta']:+.3f} "
              f"[{delta['ci_low']:+.3f},{delta['ci_high']:+.3f}]  {delta['source'][:26]:26s} "
              f"{delta['target'][:18]:18s} {delta['condition'][:22]:22s} {delta['stratum']}")
    total_excluding = sum(1 for d in deltas if d["excludes_zero"])
    print(f"\n{total_excluding} of {len(deltas)} geometry comparisons have intervals clear of zero"
          f"{f'; {shown - 30} more not shown' if shown > 30 else ''}")

    findings = build_findings(report)
    print()
    print("=" * 78)
    print("DECISION RULE")
    print("=" * 78)
    for name, value in findings.items():
        print(f"  {name:44s} {value}")

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from slot_resolution_audit import apply_decision_rule  # noqa: E402

    decision = apply_decision_rule(findings)
    decision["non_inferiority_margin"] = INTERVENTION_NON_INFERIORITY_MARGIN
    decision["worst_intervention_ci_low_at_8x8x64"] = worst_intervention_delta(deltas, "g8x8x64")
    print(f"\n  OUTCOME            {decision['outcome']}")
    print(f"  SELECTED GEOMETRY  {decision['selected_geometry']}")
    print(f"  SCREEN UNBLOCKED   {decision['screen_unblocked']}")

    arguments.out.write_text(json.dumps(decision, indent=1, sort_keys=True))
    print(f"\nwrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
