"""Emit the audit's tables as markdown, straight from the artifacts.

Hand-transcribing numbers into a report is how a report ends up disagreeing with
the run it describes. Every table below is generated from the JSON, so the
document and the artifact cannot drift apart.

    .venv/bin/python experiments/shwm/slot_audit_tables.py > tables.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

LINK_NAMES = {
    "0_position": "agent position (prerequisite)",
    "0_movement": "movement direction (prerequisite)",
    "1_switch_presence": "1. switch position / presence",
    "2_crossed_switch": "2. transition crossed a switch",
    "3_polarity_changed": "3. transition changed polarity",
    "4_accumulation": "4. parity / count of crossings",
    "5_hidden_phase": "5-6. hidden phase",
    "5a_reset_stripe": "5a. initial polarity (reset stripe)",
    "9_intervention": "9. counterfactual intervention",
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def chain_table(report: dict[str, Any]) -> str:
    rows_by_key: dict[tuple, dict[str, Any]] = {}
    oracle_by_key: dict[tuple, dict[str, Any]] = {}
    for row in report["ridge_results"]:
        key = (row["link"], row["stratum"], row["window"])
        target = oracle_by_key if row["source"] == "oracle_structured_state" else rows_by_key
        if key not in target or row["margin"] > target[key]["margin"]:
            target[key] = row
    lines = []
    for key in sorted(rows_by_key, key=lambda k: (k[0], k[1], k[2])):
        link, stratum, window = key
        row = rows_by_key[key]
        oracle = oracle_by_key.get((link, stratum, "full")) or oracle_by_key.get(key)
        lines.append([
            LINK_NAMES.get(link, link), stratum, window,
            f"{row['margin']:+.3f}", f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]",
            row["source"].replace("_spatial_slots", "").replace("_", " ")[:20],
            row["geometry"], row["condition"].replace("_", " ")[:22],
            f"{oracle['margin']:+.3f}" if oracle else "—",
        ])
    return table(
        ["link", "stratum", "window", "best margin", "95% CI", "arm", "geometry",
         "condition", "oracle"],
        lines,
    )


def geometry_table(report: dict[str, Any]) -> str:
    rows = [
        [g["name"], str(g["slot_count"]), str(g["width"]), str(g["scalars"]),
         f"{g['float32_bytes'] * 100_000 / 1e9:.2f}", f"{g['cells_per_block']:.1f}",
         "yes" if g["cell_aligned"] else "**no**", g["role"].replace("_", " ")]
        for g in report["geometry"]["geometries"]
    ]
    return table(
        ["geometry", "slots", "width", "scalars", "GB @100k steps", "cells/block",
         "cell-aligned", "role"],
        rows,
    )


def delta_table(report: dict[str, Any], limit: int = 24) -> str:
    deltas = [d for d in report.get("geometry_deltas", []) if d["excludes_zero"]]
    deltas.sort(key=lambda d: -abs(d["delta"]))
    rows = [
        [d["geometry"], d["source"].replace("_spatial_slots", "")[:20], d["target"],
         d["condition"].replace("_", " ")[:20], d["stratum"],
         f"{d['delta']:+.3f}", f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]",
         "better" if d["improves"] else "**worse**"]
        for d in deltas[:limit]
    ]
    return table(
        ["geometry", "arm", "target", "condition", "stratum", "Δ vs 4×4", "95% CI", "direction"],
        rows,
    ), len(deltas), len(report.get("geometry_deltas", []))


def recurrent_table(report: dict[str, Any]) -> str:
    best: dict[str, dict[str, Any]] = {}
    for row in report["recurrent_results"]:
        key = row["condition"]
        if key not in best or row["margin"] > best[key]["margin"]:
            best[key] = row
    rows = [
        [c.replace("_", " "), f"{r['score']:.3f}", f"{r['baseline']:.3f}",
         f"{r['margin']:+.3f}", r["source"].replace("_spatial_slots", "")[:20], r["geometry"]]
        for c, r in sorted(best.items(), key=lambda kv: -kv[1]["margin"])
    ]
    return table(["condition", "score", "baseline", "margin", "best arm", "geometry"], rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path,
                        default=REPO / "artifacts/shwm/scale1/slot-resolution-audit.json")
    parser.add_argument("--decision", type=Path,
                        default=REPO / "artifacts/shwm/scale1/slot-resolution-decision.json")
    arguments = parser.parse_args()
    report = json.loads(arguments.audit.read_text())
    decision = json.loads(arguments.decision.read_text()) if arguments.decision.exists() else {}

    print("## Provenance\n")
    print(table(
        ["item", "value"],
        [
            ["commit", f"`{git('rev-parse', 'HEAD')}`"],
            ["branch", f"`{git('rev-parse', '--abbrev-ref', 'HEAD')}`"],
            ["wall clock", f"{report['wall_clock_seconds'] / 60:.1f} min"],
            ["train / val / test rows", " / ".join(
                str(report["counts"][k]) for k in ("train", "val", "test"))],
            ["same-observation pairs", str(report["pairs"]["count"])],
        ],
    ))

    print("\n## Slot geometry, storage and compute\n")
    print(geometry_table(report))

    print("\n## Causal chain\n")
    print(chain_table(report))

    print("\n## Temporal positive controls\n")
    parity = report["parity_capability"]
    accumulator = report["exact_parity_accumulator"]
    print(table(
        ["control", "result", "verdict"],
        [
            ["exact parity accumulator reproduces recorded polarity",
             f"{accumulator['reproduces_recorded_polarity']:.4f}",
             "labels valid" if accumulator["valid"] else "**TESTBED INVALID**"],
            ["GRU accumulates parity from events + initial value",
             f"{parity['accuracy']:.4f}",
             "readout capable" if parity["capable"] else "**READOUT INCAPABLE**"],
        ],
    ))

    print("\n## Recurrent readout on real features\n")
    print(recurrent_table(report))

    print("\n## Geometry differences against the 4×4 reference\n")
    tbl, shown, total = delta_table(report)
    print(tbl)
    print(f"\n{shown} of {total} paired comparisons have an interval clear of zero.")

    print("\n## Same-observation pairs (items 7 and 8)\n")
    for row in report["paired_measurements"]:
        print(f"**{row['measurement']}** — {row['pairs']} pairs, score "
              f"{row['score']:.3f} against chance {row['chance']:.3f}; "
              f"features identical: {row['features_identical']}.\n")
        print(f"> {row['note']}\n")

    print("\n## Pins\n")
    pins = report["pins"]
    hidden = pins["hidden_state_absent"]
    print(table(
        ["pin", "result"],
        [
            ["public observation fields", ", ".join(f"`{f}`" for f in hidden["public_field_names"])],
            ["forbidden names present", str(hidden["forbidden_names_present"]) or "none"],
            ["planted-key calibration fires", str(hidden["calibration_planted_key_detected"])],
            ["pairs with differing content digest",
             f"{hidden['pairs_with_differing_content_digest']} of {hidden['pairs_checked']}"],
            ["correct histories replayed clean",
             f"{pins['correct_histories_reachable']['rows_replayed']} rows, "
             f"{pins['correct_histories_reachable']['mismatches']} mismatches"],
        ],
    ))

    if decision:
        print("\n## Decision\n")
        print(table(
            ["clause", "value"],
            [[k.replace("_", " "), str(v)] for k, v in decision["inputs"].items()]
            + [
                ["**outcome**", f"**{decision['outcome']}**"],
                ["selected geometry", str(decision["selected_geometry"])],
                ["pre-registered non-inferiority margin",
                 str(decision.get("non_inferiority_margin"))],
                ["worst intervention CI low at 8×8×64",
                 f"{decision.get('worst_intervention_ci_low_at_8x8x64', float('nan')):+.4f}"],
                ["**87-workload screen unblocked**", f"**{decision['screen_unblocked']}**"],
            ],
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
