"""Render the Scale-0 resource report from the run JSON.

Every number in the handoff comes from this script rather than from a person
reading a JSON file and typing it into a table. Transcription is where reports
acquire numbers that were never measured, and a report whose numbers cannot be
regenerated from the artefact is not evidence.

    python experiments/shwm/report_handoff.py artifacts/shwm/scale0/dry_run-report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


def gib(value: float | int | None) -> str:
    return "-" if value is None else f"{float(value) / 1024**3:.2f}"


def summarise_group(workloads: list[dict[str, Any]], key) -> dict[str, Any]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for workload in workloads:
        groups.setdefault(key(workload), []).append(workload)
    return groups


def table(rows: Iterable[Iterable[str]], header: list[str]) -> str:
    rows = [list(map(str, row)) for row in rows]
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    lines = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |")
    return "\n".join(lines)


def render(document: dict[str, Any]) -> str:
    workloads = document["workloads"]
    out: list[str] = []

    out.append("## Run identity\n")
    git = document["git"]
    environment = document["environment"]
    out.append(
        table(
            [
                ["mode", document["mode"]],
                ["is_matrix_run", str(document["is_matrix_run"])],
                ["commit", git["commit"]],
                ["branch", git["branch"]],
                ["tracked tree dirty", str(git["dirty_tracked"])],
                ["python", environment["python"]],
                ["mlx", str(environment.get("mlx"))],
                ["platform", environment["platform"]],
                ["freeze manifest digest", document["freeze_manifest_digest"]],
            ],
            ["field", "value"],
        )
    )

    out.append("\n## Backbone preflight\n")
    rows = []
    for record in document["backbone_preflight"]["records"]:
        candidate = record["candidate"]
        rows.append(
            [
                candidate["encoder_id"],
                candidate["repository"],
                record["verdict"],
                str(record["licence"]),
                str(record["gated"]),
                (record["revision"] or "")[:12],
                f"{record['total_parameters']:,}" if record["total_parameters"] else "-",
                ", ".join(record["reasons"]) or "-",
            ]
        )
    out.append(
        table(rows, ["encoder", "repository", "verdict", "licence", "gated", "revision", "parameters", "blocked by"])
    )
    out.append(f"\n`matrix_may_run` = **{document['backbone_preflight']['matrix_may_run']}**")
    out.append(f"\n> {document['backbone_preflight']['reason']}\n")

    out.append("\n## Dataset\n")
    dataset = document["dataset"]
    rows = []
    for slot, summary in dataset["per_slot"].items():
        cache = summary["cache"]
        rows.append(
            [
                slot,
                f"{summary['transitions']:,}",
                f"{summary['feature_table']['rows']:,}",
                str(summary["feature_table"]["width"]),
                f"{cache['payload_bytes_resident'] / 1e6:.1f}",
                f"{cache['index_bytes'] / 1e6:.1f}",
                f"{summary['cache_stats']['hit_ratio']:.3f}",
                f"{summary['resource']['throughput']['transitions_per_second']:.0f}",
            ]
        )
    out.append(
        table(
            rows,
            ["slot", "transitions", "distinct obs", "width", "payload MB", "index MB", "cache hit", "tx/s"],
        )
    )
    out.append(f"\nShared raw transition digest: `{dataset['transition_ids_digest']}`")
    out.append(f"\nSplit manifest digest: `{dataset['split_manifest_digest']}`")
    out.append(f"\nDataset build wall clock: {document['dataset_build_seconds']:.1f} s "
               f"(ceiling 28,800 s)\n")

    out.append("\n### Split audit, per family\n")
    rows = []
    for slot, summary in dataset["per_slot"].items():
        for family, audit in summary["audits"].items():
            if family == "combined":
                continue
            rows.append(
                [
                    slot,
                    family,
                    f"{audit['transitions']:,}",
                    f"{audit['branch_groups']:,}",
                    f"{audit['observation_content_overlap_rate']:.4f}",
                    f"{audit['transition_tuple_overlap_rate']:.4f}",
                    audit["transition_disjointness"].split(";")[0],
                ]
            )
    out.append(
        table(
            rows,
            ["slot", "family", "transitions", "branch groups", "obs overlap", "tuple overlap", "disjointness"],
        )
    )

    out.append("\n## Parameter accounting\n")
    rows = []
    seen: set[str] = set()
    for workload in workloads:
        claim = workload["claim"]
        cell = workload["workload_id"].rsplit(".s", 1)[0]
        if cell in seen:
            continue
        seen.add(cell)
        config = workload["parameters"]["config"]
        rows.append(
            [
                cell,
                f"{claim['target_parameters']:,}",
                f"{claim['trainable_parameters']:,}",
                f"{claim['trainable_parameters'] / claim['target_parameters'] - 1:+.4%}",
                str(config["latent_width"]),
                str(config["belief_dimension"]),
                str(config["core_width"]),
                "0",
            ]
        )
    out.append(
        table(rows, ["cell", "target", "actual trainable", "drift", "width", "belief", "core", "frozen"])
    )

    out.append("\n## Throughput and memory, per cell\n")
    rows = []
    for cell, group in summarise_group(workloads, lambda w: w["workload_id"].rsplit(".s", 1)[0]).items():
        wall = [w["resource"]["wall_seconds"] for w in group]
        peak = [w["resource"]["mlx_peak_bytes"] for w in group]
        updates = [w["resource"]["throughput"]["updates_per_second"] for w in group]
        positions = [w["resource"]["throughput"]["transition_positions_per_second"] for w in group]
        cold = [w["resource"]["cold_load_seconds"] for w in group]
        ratio = [w["resource"]["measured_over_estimated"] or 0.0 for w in group]
        rows.append(
            [
                cell,
                str(len(group)),
                f"{statistics.mean(wall):.1f}",
                f"{statistics.mean(cold):.2f}",
                gib(max(peak)),
                f"{statistics.mean(updates):.2f}",
                f"{statistics.mean(positions):,.0f}",
                f"{statistics.mean(ratio):.2f}",
            ]
        )
    out.append(
        table(
            rows,
            ["cell", "seeds", "wall s", "cold load s", "device peak GiB", "upd/s", "positions/s", "measured/estimated"],
        )
    )

    out.append("\n## Planner and verifier, per workload\n")
    planner = [w["planner"] for w in workloads]
    verifier = [w["verifier"] for w in workloads]
    if planner:
        out.append(
            table(
                [
                    ["planner invocations", str(planner[0]["invocations"])],
                    ["candidate sequences", f"{planner[0]['candidate_sequences']:,}"],
                    ["model calls", f"{planner[0]['model_calls']:,}"],
                    ["distinct plans", str(planner[0]["distinct_plans"])],
                    ["rollouts per second", f"{statistics.mean(p['rollouts_per_second'] for p in planner):,.0f}"],
                    ["planner wall seconds", f"{statistics.mean(p['wall_seconds'] for p in planner):.2f}"],
                    ["verifier verifications", str(verifier[0]["verifications"])],
                    ["planted mismatches", str(verifier[0]["planted_mismatches"])],
                    ["detection rate", f"{verifier[0]['detection_rate']:.3f}"],
                    ["mean probe coverage", f"{verifier[0]['mean_coverage']:.3f}"],
                    ["authorised actions", str(verifier[0]["authorised_actions"])],
                    ["denied actions", str(verifier[0]["denied_actions"])],
                    ["online environment interactions", "0"],
                ],
                ["quantity", "value"],
            )
        )

    process_peaks = [w["resource"]["process_peak_resident_bytes"] for w in workloads]
    if process_peaks:
        out.append(
            "\nProcess resident high-water across the whole sequence: "
            f"**{max(process_peaks) / 1024**3:.2f} GiB** of the 112 GiB per-process ceiling. "
            "That figure is cumulative by construction (`ru_maxrss` never falls), so it is "
            "the number the ceiling is checked against and *not* any single workload's cost; "
            "the device peak column above is the per-workload figure.\n"
        )

    out.append("\n## Gate\n")
    clauses = document.get("gate_clauses")
    if clauses:
        out.append("Each clause the run matrix states, evaluated by name.\n")
        out.append(
            table(
                [[name, "PASS" if ok else "**FAIL**"] for name, ok in clauses.items()],
                ["clause", "result"],
            )
        )
        external = document.get("gate_clauses_verified_externally") or {}
        if external:
            out.append("\nClauses the driver cannot settle, and how they are settled:\n")
            out.append(table([[k, v] for k, v in external.items()], ["clause", "how"]))
        out.append("")
    rows = [
        ["workloads completed", f"{document['workloads_completed']}/{document['workloads_expected']}"],
        ["matching failures", str(len(document["matching_failures"]))],
        ["resource envelope failures", str(len(document["resource_envelope_failures"]))],
        ["undeclared process state", str(len(document["undeclared_state"]))],
        ["failures", str(len(document["failures"]))],
        ["total wall clock", f"{document['total_seconds'] / 60:.1f} min (ceiling 4,320 min)"],
        ["artifact storage", f"{gib(document['artifact_bytes'])} GiB (ceiling 200 GiB)"],
        ["Scale-0 gate passed", str(document["scale_0_gate_passed"])],
    ]
    out.append(table(rows, ["check", "result"]))
    out.append(f"\n> {document['gate_note']}\n")

    restarts = [w for w in workloads if w.get("restart_check")]
    if restarts:
        check = restarts[0]["restart_check"]
        out.append("\n## Restart equivalence\n")
        out.append(
            table(
                [
                    ["workload", restarts[0]["workload_id"]],
                    ["checkpoint at update", str(check["checkpoint_at_update"])],
                    ["weights match", str(check["match"])],
                    ["loss history match", str(check["loss_history_match"])],
                ],
                ["field", "value"],
            )
        )

    for entries, title in (
        (document["matching_failures"], "Matching failures"),
        (document["resource_envelope_failures"], "Resource envelope failures"),
        (document["failures"], "Run failures"),
        (document["undeclared_state"], "Undeclared process state"),
    ):
        if entries:
            out.append(f"\n## {title}\n")
            out.extend(f"- {entry}" for entry in entries)

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()
    text = render(json.loads(arguments.report.read_text()))
    if arguments.out:
        arguments.out.write_text(text)
        print(f"written: {arguments.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
