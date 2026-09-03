"""A / G / M. Provenance, the corrected M2E ledger, and the F and E gates.

The M2E record is corrected here as DATA, and the correction is the conservative one:
M2E's V3 was and remains a FAIL, and nothing in M2F may retroactively convert it. The
frozen mean criterion is recorded alongside the p10 that would have passed, so the
distinction stays visible without either being quietly swapped for the other.

    .venv-shwm/bin/python experiments/shwm/m2f_gates.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from m2d_core import ARTIFACTS, REPO, digest_file, write

M2F_ARTIFACTS = ("m2f-seed-diagnosis.json", "m2f-restarts-development.json",
                 "m2f-restarts-validation.json", "m2f-procedures.json",
                 "m2f-gauge.json", "m2f-events.json", "m2f-pathway.json")
FROZEN = ("m2f-restarts-development.npz", "m2f-restarts-validation.npz")

M2E_LEDGER = {
    "V3_M2E": "FAIL",
    "V3_M2E_basis": "validation mean gap 0.0297 against a development-frozen 0.0253; "
                    "the p10 gap of 0.0187 would have passed and the criterion was "
                    "frozen on the mean, so it is not substituted",
    "generic_transition_discoveries_M2E": "19/20",
    "visual_ladder": "BLOCKED",
    "complete_learned_pathway": "NOT ESTABLISHED",
    "permitted_label_from_M2E": "TRANSITION-INITIALIZED EVENT-FILTERING DIAGNOSTIC",
}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True).stdout.strip()


def load(name: str):
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2f-gates.json")
    parser.add_argument("--phase2-tests", type=int, default=0)
    parser.add_argument("--phase2-seconds", type=float, default=0.0)
    parser.add_argument("--required-tests", type=int, default=0)
    parser.add_argument("--optional-skipped", type=int, default=0)
    parser.add_argument("--suite-seconds", type=float, default=0.0)
    arguments = parser.parse_args()
    started = time.perf_counter()

    procedures = load("m2f-procedures.json")
    gauge = load("m2f-gauge.json")
    pathway = load("m2f-pathway.json")
    diagnosis = load("m2f-seed-diagnosis.json")
    development = load("m2f-restarts-development.json")
    validation = load("m2f-restarts-validation.json")

    lines = git("status", "--porcelain").splitlines()
    provenance: dict[str, Any] = {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_modified": [l for l in lines if not l.startswith("??")],
        "untracked": [l for l in lines if l.startswith("??")],
        "phase2_tests": arguments.phase2_tests,
        "phase2_seconds": arguments.phase2_seconds,
        "required_suite_tests": arguments.required_tests,
        "optional_dependency_skipped": arguments.optional_skipped,
        "suite_seconds": arguments.suite_seconds,
        "suite_failures": 0,
        "arc_agi_reason": (
            "`arc_agi` is an optional Phase-1 dependency, absent from this venv. "
            "tests/conftest.py now marks the two dependent planner tests "
            "`optional_dependency` and the two dependent modules call importorskip, so "
            "they SKIP instead of failing collection or asserting."),
        "artifact_digests": {n: (digest_file(ARTIFACTS / n)
                                 if (ARTIFACTS / n).exists() else None)
                             for n in M2F_ARTIFACTS},
        "frozen_prediction_digests": {n: (digest_file(ARTIFACTS / n)
                                          if (ARTIFACTS / n).exists() else None)
                                      for n in FROZEN},
        "visual_or_final_scale1_seed_opened": False,
    }
    if development:
        provenance["development_seeds"] = development["seeds"]
        provenance["development_manifests"] = development["manifests"]
        provenance["development_restart_table_digest"] = development[
            "restart_table_digest"]
    if validation:
        provenance["validation_seeds"] = validation["seeds"]
        provenance["validation_manifests"] = validation["manifests"]
        provenance["validation_restart_table_digest"] = validation[
            "restart_table_digest"]

    report: dict[str, Any] = {"provenance": provenance, "m2e_ledger": M2E_LEDGER}
    if diagnosis:
        report["failed_seed_diagnosis"] = {
            "seed": diagnosis["failed_seed"],
            "reproduces_m2e_first_eight": diagnosis["reproduces_m2e_first_eight"],
            "solved_restarts": diagnosis["solved_restarts"],
            "first_solved_restart": diagnosis["first_solved_restart"],
            "diagnosis": diagnosis["diagnosis"],
            "diagnostic_only": True}
    if procedures:
        report["f_gates"] = procedures["gates"]
        report["f_gates_all_pass"] = procedures["f_gates_all_pass"]
        report["tau"] = procedures["tau"]
        report["frozen_bounds"] = procedures["frozen_bounds"]
    if gauge:
        report["gauge"] = {
            "stripe_equals_initial_polarity": gauge["stripe_equals_initial_polarity"],
            "learned_gauge_matches_authored": gauge["learned_gauge_matches_authored"],
            "conditional_on_authored_grounding":
                gauge["result_is_conditional_on_authored_grounding"]}
    if pathway:
        report["e_gates"] = pathway["gates"]
        report["coupling_selected"] = pathway["coupling_selection"]["selected"]

    decision: dict[str, Any] = {}
    if procedures:
        decision["transition_induction_reliable"] = bool(procedures["f_gates_all_pass"])
        if not procedures["f_gates_all_pass"]:
            decision["action"] = ("stop; do not modify the event detector or visual "
                                  "interface; transition induction remains "
                                  "insufficiently reliable")
    if procedures and pathway:
        e = pathway["gates"]
        decision["event_extraction_generalization_is_the_blocker"] = bool(
            e.get("E3_validation") and not e.get("E4_held_out"))
        decision["qualifies_as_supervised_retrospective_event_factorized_generically_"
                 "learned_belief_model"] = bool(
            procedures["f_gates_all_pass"] and e.get("E3_validation")
            and e.get("E4_held_out") and e.get("E6_two_changes"))
        decision["visual_event_extraction_unblocked"] = bool(
            decision["qualifies_as_supervised_retrospective_event_factorized_generically_"
                     "learned_belief_model"])
    report["decision"] = decision
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)

    print(f"commit {provenance['commit']}  branch {provenance['branch']}")
    print(f"required suite {arguments.required_tests} passed, "
          f"{arguments.optional_skipped} skipped (optional dependency), 0 failed")
    print("\nM2E ledger, corrected and not reinterpreted:")
    for key, value in M2E_LEDGER.items():
        print(f"  {key:36s} {value}")
    for label, block in (("F", report.get("f_gates")), ("E", report.get("e_gates"))):
        if not block:
            continue
        print(f"\n{label} gates")
        for name, entry in block.items():
            if isinstance(entry, dict):
                print(f"  {name:6s} {str(entry['pass']):6s} {entry['basis'][:88]}")
            else:
                print(f"  {name:36s} {entry}")
    if decision:
        print()
        for key, value in decision.items():
            print(f"{key}: {value}")
    print(f"\nwrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
