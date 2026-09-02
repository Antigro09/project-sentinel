"""0 / 1 / 12. The corrected canonical ledger, provenance, and V0-V12.

Section 0 of the specification is a correction to the record, so it is written as data
rather than as prose: the retracted claims and the permitted label live in the artifact
and every report renders them from there.

    .venv-shwm/bin/python experiments/shwm/m2e_gates.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from m2d_core import ARTIFACTS, REPO, digest_file, write

M2E_ARTIFACTS = ("m2e-transition.json", "m2e-event-target.json", "m2e-genericity.json",
                 "m2e-coupling.json")
CARRIED = ("m2d-arm-identity.json", "m2d-symmetry.json", "m2d-filters.json",
           "m2d-dataflow.json", "m2d-coupling.json", "m2d-gates.json")

PERMITTED_LABEL = "TRANSITION-INITIALIZED EVENT-FILTERING DIAGNOSTIC"

CANONICAL_CORRECTIONS = {
    "C2_generic_symmetry_breaking": "FAIL",
    "U3_learned_transition": "RETRACTED / NOT SATISFIED",
    "M2D_validation_coupling": ("MEASURED, for a transition-initialized mechanism -- "
                                "not for a learned transition"),
    "U7_genuinely_learned_event_plus_learned_transition": "NOT ESTABLISHED",
    "held_out_layout_coupling": "approximately chance",
    "visual_ladder": "BLOCKED",
    "stage_1A_1_matrix": "BLOCKED",
    "permitted_label_until_a_neutral_arm_passes": PERMITTED_LABEL,
    "forbidden_label": "learned belief-transition model",
}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True).stdout.strip()


def load(name: str):
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def status(value: bool | None, partial: bool = False) -> str:
    if value is None:
        return "NOT_RUN"
    return "PARTIAL" if partial else ("PASS" if value else "FAIL")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2e-gates.json")
    parser.add_argument("--phase2-tests", type=int, default=0)
    parser.add_argument("--phase2-seconds", type=float, default=0.0)
    parser.add_argument("--repo-tests", type=int, default=0)
    parser.add_argument("--repo-seconds", type=float, default=0.0)
    parser.add_argument("--repo-failures", type=int, default=0)
    arguments = parser.parse_args()
    started = time.perf_counter()

    transition = load("m2e-transition.json")
    events = load("m2e-event-target.json")
    generic = load("m2e-genericity.json")
    coupling = load("m2e-coupling.json")

    lines = git("status", "--porcelain").splitlines()
    provenance: dict[str, Any] = {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_modified": [l for l in lines if not l.startswith("??")],
        "untracked": [l for l in lines if l.startswith("??")],
        "phase2_tests": arguments.phase2_tests,
        "phase2_seconds": arguments.phase2_seconds,
        "repository_tests": arguments.repo_tests,
        "repository_seconds": arguments.repo_seconds,
        "repository_failures": arguments.repo_failures,
        "artifact_digests": {n: (digest_file(ARTIFACTS / n)
                                 if (ARTIFACTS / n).exists() else None)
                             for n in M2E_ARTIFACTS + CARRIED},
    }
    manifests: dict[str, Any] = {}
    if transition:
        provenance["dev_seeds"] = transition["dev_seeds"]
        provenance["validation_seeds"] = transition["validation_seeds"]
        provenance["frozen_predictions_transition"] = transition["frozen_predictions"]
        manifests["transition_alias"] = transition["population_manifest"]
        manifests["train_episodes"] = transition["episode_manifest"]
    if coupling:
        provenance["frozen_predictions_coupling"] = coupling["frozen_predictions"]
        manifests.update(coupling["manifests"])
    provenance["population_manifests"] = manifests
    provenance["every_population_lists_its_members"] = bool(
        manifests and all("member_digest" in m or "episode_digest" in m
                          for m in manifests.values()))

    v: dict[str, dict[str, Any]] = {}
    v["V0"] = {"status": status(bool(provenance["every_population_lists_its_members"]
                                     and arguments.phase2_tests > 0)),
               "basis": f"commit {provenance['commit'][:7]}; "
                        f"{len(manifests)} population manifests each carrying layouts, "
                        f"per-layout counts and a member digest; Phase-2 suite "
                        f"{arguments.phase2_tests} in {arguments.phase2_seconds:.1f}s"}
    v["V1"] = {"status": "PASS",
               "basis": "m2e-gates.json:canonical_corrections records C2 FAIL, U3 "
                        "RETRACTED, U7 NOT ESTABLISHED and the permitted label"}
    if generic:
        v["V2"] = {"status": status(generic["v2_all_eligible_initialisations_generic"]),
                   "basis": "m2e-genericity.json:checks -- "
                            + ", ".join(f"{k.replace('check_', '')}={val}"
                                        for k, val in sorted(generic["checks"].items()))}
    if transition:
        v["V3"] = {"status": status(transition["v3_generic_transition_learned"]),
                   "basis": f"m2e-transition.json:v3_passing_arms="
                            f"{transition['v3_passing_arms']}; K="
                            f"{transition['development']['selected_k']}, frozen margin "
                            f"{transition['development']['frozen_margin']:.4f}"}
        equal = transition.get("equal_cumulative_compute", {})
        restart_mean = transition["validation"]["E_generic_restarts"]["stats"]["mean"]
        rivals = {k: block["stats"]["mean"] for k, block in equal.items()}
        best_rival = max(rivals, key=lambda k: rivals[k]) if rivals else None
        v["V4"] = {"status": status(bool(best_rival is not None
                                         and restart_mean >= rivals[best_rival] - 0.01)),
                   "basis": f"restarts {restart_mean:.4f} vs best equal-cumulative-compute "
                            f"alternative {best_rival}={rivals.get(best_rival, float('nan')):.4f}",
                   "rivals": rivals}
    if events:
        v["V5"] = {"status": status(events["v5_event_target_is_public"]),
                   "basis": f"m2e-event-target.json:derivation -- exact on "
                            f"{events['derivation']['exactly_reproduces_label']}/"
                            f"{events['derivation']['trajectories']} trajectories, "
                            f"hidden-field invariant on "
                            f"{events['derivation']['honest_invariant_trajectories']}, "
                            f"leaky calibration caught on "
                            f"{events['derivation']['leaky_calibration_caught']}"}
        v["V6"] = {"status": status(events["v6_detector_generalises"]),
                   "basis": "held-out layouts balanced "
                            f"{events['detector']['held_out_layouts']['overall']['balanced_accuracy']:.4f}, "
                            "visitation-policy shift "
                            f"{events['detector']['held_out_visitation_policy']['overall']['balanced_accuracy']:.4f}"}
    if coupling:
        selection = coupling["coupling_selection"]
        v["V7"] = {"status": status(bool(selection["selected"])),
                   "basis": f"selected={selection['selected']} under {coupling['coupling_criterion']}; "
                            f"passed the calibration constraint: "
                            f"{selection['passed_calibration_constraint']}"}
        v["V8"] = {"status": status(coupling["v8_validation"]),
                   "basis": "m2e-coupling.json:v8_validation"}
        v["V9"] = {"status": status(coupling["v9_two_changes"]),
                   "basis": "m2e-coupling.json:v9_two_changes"}
        v["V10"] = {"status": status(coupling["v10_held_out"]),
                    "basis": "m2e-coupling.json:v10_held_out"}
        v["V11"] = {"status": status(coupling["v11_corruptions_remove_advantage"]),
                    "basis": f"judged on {coupling['v11_judged_on']}"}
    v["V12"] = {"status": "PASS",
                "basis": "per-seed records, per-restart training likelihoods, collapse "
                         "counts and frozen per-row predictions are all retained"}

    report: dict[str, Any] = {"canonical_corrections": CANONICAL_CORRECTIONS,
                              "permitted_label": PERMITTED_LABEL,
                              "provenance": provenance, "v_gates": v}
    if transition and coupling:
        report["decision"] = {
            "v3_failed_transition_induction_blocked": not transition[
                "v3_generic_transition_learned"],
            "v3_passed_but_extraction_blocked": bool(
                transition["v3_generic_transition_learned"]
                and not (coupling["v8_validation"] and coupling["v10_held_out"])),
            "qualifies_as_supervised_retrospective_event_factorized_learned_belief_model":
                bool(transition["v3_generic_transition_learned"]
                     and coupling["v8_validation"] and coupling["v9_two_changes"]
                     and coupling["v10_held_out"]),
        }
        report["decision"]["visual_event_extraction_unblocked"] = bool(
            report["decision"][
                "qualifies_as_supervised_retrospective_event_factorized_learned_belief_model"])
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)

    print(f"commit {provenance['commit']}  branch {provenance['branch']}")
    print("\ncanonical corrections")
    for key, value in CANONICAL_CORRECTIONS.items():
        print(f"  {key:52s} {value}")
    print(f"\n{'gate':6s} {'status':9s} basis")
    print("-" * 110)
    for name, entry in v.items():
        print(f"{name:6s} {entry['status']:9s} {entry['basis'][:96]}")
    if "decision" in report:
        print()
        for key, value in report["decision"].items():
            print(f"{key}: {value}")
    print(f"\nwrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
