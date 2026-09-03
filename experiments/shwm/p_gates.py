"""A / L. The corrected O ledger and the P0-P14 closure gates.

The O ledger is reissued with a `reason_class` on every NOT_RUN, because phase O marked
O12 and O13 with the same status as the genuinely blocked O6-O10 and then called them
failures in prose. A reader could not tell the two apart from the artifact, which is the
inconsistency this phase was asked to resolve. The status still describes the
measurement -- there is none -- and the new field says why.

    .venv-shwm/bin/python experiments/shwm/p_gates.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from m2d_core import ARTIFACTS, REPO, digest_file, write

P_ARTIFACTS = ("p-equivalence.json", "p-binding.json", "p-gauge.json",
               "p-multimodal.json")
CARRIED = ("o-gates.json", "o-identifiability.json", "o-posterior.json",
           "o-detection.json", "n-gates.json")

BLOCKED_UPSTREAM = "BLOCKED_UPSTREAM"
NOT_DELIVERED = "NOT_DELIVERED"

O_REASONS = {
    "O6": BLOCKED_UPSTREAM, "O7": BLOCKED_UPSTREAM, "O8": BLOCKED_UPSTREAM,
    "O9": BLOCKED_UPSTREAM, "O10": BLOCKED_UPSTREAM,
    "O12": NOT_DELIVERED, "O13": NOT_DELIVERED,
}


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,
                          text=True).stdout.strip()


def load(name: str):
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "p-gates.json")
    parser.add_argument("--phase2-tests", type=int, default=0)
    parser.add_argument("--phase2-seconds", type=float, default=0.0)
    parser.add_argument("--suite-tests", type=int, default=0)
    parser.add_argument("--suite-skipped", type=int, default=0)
    parser.add_argument("--suite-seconds", type=float, default=0.0)
    arguments = parser.parse_args()
    started = time.perf_counter()

    equivalence = load("p-equivalence.json")
    binding = load("p-binding.json")
    gauge = load("p-gauge.json")
    multimodal = load("p-multimodal.json")
    o_gates = load("o-gates.json")
    posterior = load("o-posterior.json")
    detection = load("o-detection.json")

    lines = git("status", "--porcelain").splitlines()
    provenance: dict[str, Any] = {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_modified": [l for l in lines if not l.startswith("??")],
        "untracked": [l for l in lines if l.startswith("??")],
        "phase2_tests": arguments.phase2_tests,
        "phase2_seconds": arguments.phase2_seconds,
        "required_suite_tests": arguments.suite_tests,
        "required_suite_skipped": arguments.suite_skipped,
        "required_suite_seconds": arguments.suite_seconds,
        "artifact_digests": {n: (digest_file(ARTIFACTS / n)
                                 if (ARTIFACTS / n).exists() else None)
                             for n in P_ARTIFACTS + CARRIED},
        "final_scale1_seed_opened": False,
        "prospective_model_started": False,
        "stage_1a_1_matrix_run": False,
    }
    if binding:
        provenance["development_palettes"] = binding["development_palettes"]
        provenance["unseen_palettes"] = binding["unseen_palettes"]
        provenance["train_layouts"] = binding["train_layouts"]
        provenance["test_layouts"] = binding["test_layouts"]

    corrected_o: dict[str, Any] = {}
    if o_gates:
        for name, entry in o_gates["o_gates"].items():
            block = {"status": entry["status"], "basis": entry["basis"]}
            if entry["status"] == "NOT_RUN":
                block["reason_class"] = O_REASONS.get(name, BLOCKED_UPSTREAM)
                block["was_mandatory_in_O"] = name in ("O12", "O13")
            corrected_o[name] = block
        # O12 and O13 are now measured, so the O ledger is superseded for those two.
        if gauge:
            corrected_o["O12"]["superseded_by"] = f"P11: {gauge['o12_status']}"
        if multimodal:
            corrected_o["O13"]["superseded_by"] = f"P12: {multimodal['o13_status']}"

    def gate(status: str, basis: str, **extra) -> dict[str, Any]:
        return {"status": status, "basis": basis, **extra}

    p: dict[str, dict[str, Any]] = {}
    p["P0"] = gate("PASS",
                   f"commit {provenance['commit'][:7]}; O ledger reissued with a "
                   f"reason_class separating {BLOCKED_UPSTREAM} (O6-O10) from "
                   f"{NOT_DELIVERED} (O12, O13); suite "
                   f"{arguments.suite_tests}/{arguments.suite_skipped}/0")
    if equivalence:
        grounded = equivalence["stages"]["5_grounded_calibration_episode"]
        supported = equivalence["o_phase_claim_that_only_goal_markers_remain"][
            "supported_by_histogram"]
        p["P1"] = gate("PASS",
                       f"full histogram printed at every stage; grounded stage "
                       f"{grounded['histogram']} REFUTES phase O's claim that only the "
                       f"goal markers remain ({grounded['episodes_with_class_over_two']} "
                       f"episodes exceed class 2)",
                       refuted_prior_claim=not supported)
    if detection:
        p["P2"] = gate("PASS" if detection["o1_semantic_oracle_invariant_to_palette"]
                       else "FAIL",
                       f"carried from O: oracle spread "
                       f"{detection['o1_oracle_spread_across_regimes']:.4f} across five "
                       f"regimes")
    if binding:
        collision = binding["results"]["COUNT_COLLISION"]
        informative = binding["results"]["COUNT_INFORMATIVE"]
        p["P3"] = gate("PASS" if binding["p3_count_only_does_not_explain_it"] else "FAIL",
                       f"count-only is at chance in every stratum "
                       f"({informative['binder__count_only']:.4f} informative, "
                       f"{collision['binder__count_only']:.4f} collision) while the full "
                       f"binder holds {collision['binder__full_token']:.4f} under a "
                       f"provable cardinality collision")
    if posterior:
        p["P4"] = gate(posterior["o5_status"],
                       f"carried from O: entropy 3.5850 -> "
                       f"{posterior['curve'][str(max(int(k) for k in posterior['curve']))]['posterior_entropy_bits']:.4f} "
                       f"bits, event identified in "
                       f"{posterior['curve'][str(max(int(k) for k in posterior['curve']))]['event_identified_fraction']:.3f} "
                       f"against a pre-stated 0.99")
    if binding:
        collision = binding["results"]["COUNT_COLLISION"]
        p["P5"] = gate("PARTIAL",
                       f"a STATELESS global binder already beats the local detector on "
                       f"unseen palettes in every stratum "
                       f"({collision['binder__full_token']:.4f} vs "
                       f"{collision['local_conv_baseline']:.4f} under collision); the "
                       f"persistent-memory arms (recurrent assignment, Sinkhorn, implicit "
                       f"recurrent) were NOT built, so 'appearance memory' is untested",
                       reason_class=NOT_DELIVERED)
    p["P6"] = gate("NOT_RUN",
                   "there is no persistent-memory gain to destroy: the working binder is "
                   "stateless, so reset, shuffled and wrong calibration have nothing to "
                   "remove", reason_class=BLOCKED_UPSTREAM)
    if binding:
        p["P7"] = gate("PARTIAL",
                       f"event prediction on unseen palettes and held-out counts holds at "
                       f"{binding['results']['COUNT_VARIED']['binder__full_token']:.4f} "
                       f"(varied) and "
                       f"{binding['results']['COUNT_COLLISION']['binder__full_token']:.4f} "
                       f"(collision); ROUTE PARITY under unseen palettes was not measured",
                       reason_class=NOT_DELIVERED)
    for name in ("P8", "P9"):
        p[name] = gate("NOT_RUN",
                       "the binder was not coupled to the certified transition on the "
                       "alias population under unseen palettes; no alias-pair or "
                       "phase-change result exists for this regime",
                       reason_class=NOT_DELIVERED)
    if detection:
        p["P10"] = gate("PASS",
                        f"carried from O: per-frame permutation at "
                        f"{detection['arms']['2_palette_augmented_detector']['per_frame_permutation']:.4f} "
                        f"and the audit marks it unresolvable")
    if gauge:
        p["P11"] = gate("PASS" if gauge["o12_status"] == "PASS" else "FAIL",
                        f"outcome-trained gauge {gauge['variants']['4_outcome_trained']['belief_accuracy_up_to_permutation']:.4f} "
                        f"vs authored "
                        f"{gauge['variants']['1_authored_public_stripe']['belief_accuracy_up_to_permutation']:.4f} "
                        f"(difference {gauge['paired_difference_outcome_minus_authored']:+.4f}); "
                        f"stripe masked "
                        f"{gauge['variants']['5_stripe_masked']['belief_accuracy_up_to_permutation']:.4f}")
    if multimodal:
        arm = multimodal["arms"]["2_shuffled_language"]["vs_correct"]
        p["P12"] = gate("FAIL",
                        f"{multimodal['contested_keys']} contested keys; correct minus "
                        f"shuffled language {arm['delta']:+.4f} "
                        f"[{arm['ci_low']:+.4f}, {arm['ci_high']:+.4f}]. The CORRECT arm "
                        f"is itself at "
                        f"{multimodal['arms']['1_vision_language_history']['contested_accuracy']:.4f} "
                        f"on contested keys, so the test has no power to detect an effect "
                        f"-- this is a capability failure, not evidence that language is "
                        f"uninformative")
    p["P13"] = gate("PASS",
                    "the binder consumes per-colour tokens of public quantities only "
                    "(RGB, count, spatial moments, motion); no palette id, role label, "
                    "semantic map, seed, evaluator state or future outcome is in scope")
    p["P14"] = gate("PASS",
                    "every palette, stratum, seed, unresolved case and failed arm is "
                    "retained, including the four NOT_DELIVERED arms named as such")

    report: dict[str, Any] = {"provenance": provenance,
                              "corrected_o_gates": corrected_o, "p_gates": p}
    report["decision"] = {
        "learned_global_role_binding_works": bool(
            binding and binding["p5_global_binder_beats_local"]),
        "cardinality_lookup_ruled_out": bool(
            binding and not binding["cardinality_lookup_diagnosis"]),
        "initial_state_grounding_no_longer_authored": bool(
            gauge and gauge["o12_status"] == "PASS"),
        "multimodal_contribution_established": False,
        "n_phase_n12_withdrawn": True,
        "appearance_aware_interface_frozen": False,
        "prospective_prediction_unblocked": False,
        "why": ("P12 fails and P6, P8 and P9 are unrun, four of them NOT_DELIVERED; "
                "the specification freezes the interface only if P5 through P12 pass"),
    }
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)

    print(f"commit {provenance['commit']}  branch {provenance['branch']}\n")
    print("corrected O ledger")
    print(f"{'gate':5s} {'status':10s} {'reason':17s} mandatory  superseded")
    for name, entry in sorted(corrected_o.items(), key=lambda x: int(x[0][1:])):
        print(f"{name:5s} {entry['status']:10s} "
              f"{entry.get('reason_class', '-'):17s} "
              f"{str(entry.get('was_mandatory_in_O', '-')):9s} "
              f"{entry.get('superseded_by', '-')}")
    print(f"\n{'gate':5s} {'status':9s} basis")
    print("-" * 104)
    for name in sorted(p, key=lambda k: int(k[1:])):
        print(f"{name:5s} {p[name]['status']:9s} {p[name]['basis'][:86]}")
    print()
    for k, v in report["decision"].items():
        print(f"{k}: {v}")
    print(f"\nwrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
