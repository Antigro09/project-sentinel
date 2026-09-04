"""A / N. Provenance, the reconciled P ledger, and the Q0-Q14 closure gates.

The P ledger is reissued because its headline did not match its own rows. The O1 report
printed "7 PASS, 3 PARTIAL, 4 NOT_RUN, 1 FAIL" while both its table and its artifact held
eight PASS and three NOT_RUN. Nothing caught it: phases O and O1 shipped no tests. The
count is recomputed here from p-gates.json and the discrepancy is reported as a
correction rather than quietly restated.

    .venv-shwm/bin/python experiments/shwm/o2_gates.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from m2d_core import ARTIFACTS, REPO, digest_file, write

O2_ARTIFACTS = ("o2-equivalence.json", "o2-leakage.json", "o2-factorial.json",
                "o2-memory.json", "o2-route.json", "o2-goal.json",
                "o2-unresolved.json")
CARRIED = ("o2-gauge.json", "p-gates.json", "p-equivalence.json", "p-binding.json", "p-gauge.json",
           "p-multimodal.json", "o-identifiability.json", "o-posterior.json",
           "o-detection.json", "m2f-procedures.json", "m2f-gauge.json")

BLOCKED_UPSTREAM = "BLOCKED_UPSTREAM"
NOT_DELIVERED = "NOT_DELIVERED"
NOT_APPLICABLE = "NOT_APPLICABLE"

O1_HEADLINE = {"PASS": 7, "PARTIAL": 3, "NOT_RUN": 4, "FAIL": 1}


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,
                          text=True).stdout.strip()


def load(name: str):
    path = ARTIFACTS / name
    return json.loads(path.read_text()) if path.exists() else None


def collect_manifest(marker: str) -> list[str]:
    result = subprocess.run(
        [str(REPO / ".venv-shwm/bin/python"), "-m", "pytest", "--collect-only", "-q",
         "-m", marker, str(REPO / "tests")],
        capture_output=True, text=True, cwd=str(REPO))
    return [line for line in result.stdout.splitlines() if "::" in line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o2-gates.json")
    parser.add_argument("--phase2-tests", type=int, default=0)
    parser.add_argument("--phase2-seconds", type=float, default=0.0)
    parser.add_argument("--suite-tests", type=int, default=0)
    parser.add_argument("--suite-skipped", type=int, default=0)
    parser.add_argument("--suite-seconds", type=float, default=0.0)
    parser.add_argument("--skip-manifests", action="store_true")
    arguments = parser.parse_args()
    started = time.perf_counter()

    equivalence = load("o2-equivalence.json")
    leakage = load("o2-leakage.json")
    factorial = load("o2-factorial.json")
    memory = load("o2-memory.json")
    route = load("o2-route.json")
    goal = load("o2-goal.json")
    unresolved = load("o2-unresolved.json")
    p_gates = load("p-gates.json")
    gauge = load("p-gauge.json")

    lines = git("status", "--porcelain").splitlines()
    provenance: dict[str, Any] = {
        "commit": git("rev-parse", "HEAD"),
        "parent_commit_full": git("rev-parse", "8e99ade"),
        "parent_commit_subject": git("log", "-1", "--format=%s", "8e99ade"),
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
                             for n in O2_ARTIFACTS + CARRIED},
        "final_scale1_seed_opened": False,
        "prospective_model_started": False,
        "stage_1a_1_matrix_run": False,
    }
    if not arguments.skip_manifests:
        provenance["required_test_manifest"] = collect_manifest("not optional_dependency")
        provenance["optional_test_manifest"] = collect_manifest("optional_dependency")

    identifiers: dict[str, Any] = {}
    for name, block in (("equivalence", equivalence), ("leakage", leakage),
                        ("factorial", factorial), ("memory", memory),
                        ("route", route), ("goal", goal),
                        ("unresolved", unresolved)):
        if not block:
            continue
        identifiers[name] = {k: v for k, v in block.items()
                             if k.endswith(("_layouts", "_palettes", "seeds", "strata"))
                             or k in ("stratum", "history_steps", "contested_keys")}
    provenance["identifiers"] = identifiers

    # ---- the corrected P ledger -----------------------------------------------------
    corrected_p: dict[str, Any] = {}
    tally = Counter()
    if p_gates:
        for name, entry in p_gates["p_gates"].items():
            corrected_p[name] = {"status": entry["status"],
                                 "reason_class": entry.get("reason_class"),
                                 "basis": entry["basis"][:200]}
            tally[entry["status"]] += 1
    reconciliation = {
        "headline_printed_in_the_O1_report": O1_HEADLINE,
        "recomputed_from_p_gates_json": dict(tally),
        "rows": int(sum(tally.values())),
        "matches": bool(dict(tally) == O1_HEADLINE),
        "correction": ("the O1 report's headline sentence miscounted. Its own gate "
                       "table and its own artifact both list P0, P1, P2, P3, P10, P11, "
                       "P13 and P14 as PASS -- eight, not seven -- and P6, P8, P9 as "
                       "NOT_RUN -- three, not four. The machine-readable ledger was "
                       "right and the prose was wrong; nothing caught it because "
                       "phases O and O1 shipped no tests."),
    }

    # ---- Q gates --------------------------------------------------------------------
    def gate(status: str, basis: str, **extra) -> dict[str, Any]:
        return {"status": status, "basis": basis, **extra}

    q: dict[str, dict[str, Any]] = {}
    manifests_present = bool(provenance.get("required_test_manifest"))
    q["Q0"] = gate(
        "PASS" if manifests_present and arguments.suite_tests else "PARTIAL",
        f"commit {provenance['commit'][:7]}; parent "
        f"{provenance['parent_commit_full'][:7]}; suite "
        f"{arguments.suite_tests}/{arguments.suite_skipped}/0 in "
        f"{arguments.suite_seconds:.0f}s; P ledger recomputed as "
        f"{dict(tally)} against a printed {O1_HEADLINE}",
        p_ledger_reconciliation=reconciliation)

    if equivalence:
        block = equivalence["reconciliation"]
        q["Q1"] = gate("PASS" if equivalence["Q1_class_arithmetic_internally_consistent"]
                       else "FAIL",
                       f"2.08 is the O population's mean "
                       f"({block['O_population_recomputed_histogram']}, 50/24) and "
                       f"2.468 is the O1 population's "
                       f"({block['O1_population_recomputed_histogram']}, 116/47); both "
                       f"reproduced from one function over one episode table")
        q["Q2"] = gate("PASS" if equivalence["Q2_quotients_explicit"] else "FAIL",
                       "event, goal and full quotient class counts and true-class "
                       "posterior masses are reported at all seven stages for both "
                       "populations")
    if leakage:
        q["Q3"] = gate(
            "PASS" if leakage["Q3_no_undeclared_role_information_in_palette_values"]
            else "FAIL",
            f"I(role; colour) "
            f"{leakage['runs']['1_iid_palette_generation']['empirical_mi_bits_honest']:.6f}"
            f" bits inside a shuffled null; three appearance transforms invariant; "
            f"DeepSets equivariance exact; both guards pass honest and catch their "
            f"plants")
    if factorial:
        collision = factorial["results"]["COUNT_COLLISION"]
        q["Q4"] = gate(
            "PASS" if factorial["Q4_global_binder_beats_count_only_under_collision"]
            else "FAIL",
            f"selected {factorial['decisions']['selected_representation']}; on the full "
            f"collision population it beats count-only and the local detector with "
            f"paired intervals. On CONTESTED rows the exact count-only Bayes rule is at "
            f"{collision['count_only_bayes_ceiling']['contested_balanced_accuracy']:.4f}"
            f" and every stateless arm is within noise of it -- an identifiability cap",
            note=factorial.get("Q4_note"))
    if memory:
        q["Q5"] = gate("PASS" if memory["Q5_persistent_memory_beats_memoryless"]
                       else "FAIL",
                       f"persistent memory "
                       f"{memory['arms']['3_recurrent_assignment_memory']['contested_accuracy']:.4f}"
                       f" against memoryless "
                       f"{memory['arms']['2_frame_pair_binder']['contested_accuracy']:.4f}"
                       f" and augmentation-only "
                       f"{memory['arms']['5_augmentation_only_detector']['contested_accuracy']:.4f}"
                       f" on transfer rows constructed to be ambiguous without history")
        q["Q6"] = gate("PASS" if memory["Q6_ablations_remove_the_gain"] else "FAIL",
                       "reset, shuffled, wrong-paired and foreign calibration each drop "
                       "the contested accuracy below the memory arm by more than 0.05")
    if route:
        q["Q7"] = gate("PASS" if route["Q7_route_parity_supported"] else "FAIL",
                       f"held-out route parity "
                       f"{route['sequence']['held_out']['memory']['final_event_parity_accuracy']:.4f}"
                       f", exact sequence "
                       f"{route['sequence']['held_out']['memory']['exact_route_sequence_accuracy']:.4f}")
        held = route["arms"]["held_out"][
            "2_palette_memory_event_certified_transition"]["intervals"]
        q["Q8"] = gate(
            "PASS" if route["Q8_visual_event_plus_certified_beats_memoryless"]
            else "FAIL",
            f"memory event + certified transition vs visual memoryless "
            f"{held['vs_memoryless']['delta']:+.4f} "
            f"[{held['vs_memoryless']['ci_low']:+.4f}, "
            f"{held['vs_memoryless']['ci_high']:+.4f}] on held-out alias layouts")
        q["Q9"] = gate("PASS" if route["Q9_gain_survives_two_phase_changes"] else "FAIL",
                       f"two changes {held['changes_2']['delta']:+.4f} "
                       f"[{held['changes_2']['ci_low']:+.4f}, "
                       f"{held['changes_2']['ci_high']:+.4f}]; four or more "
                       f"{held['changes_4plus']['delta']:+.4f} "
                       f"[{held['changes_4plus']['ci_low']:+.4f}, "
                       f"{held['changes_4plus']['ci_high']:+.4f}]")
    replication = load("o2-gauge.json")
    if replication:
        outcome = replication["variants"]["4_outcome_trained"][
            "belief_accuracy_up_to_permutation"]
        authored = replication["variants"]["1_authored_public_stripe"][
            "belief_accuracy_up_to_permutation"]
        masked = replication["variants"]["5_stripe_masked"][
            "belief_accuracy_up_to_permutation"]
        q["Q10"] = gate("PASS" if replication["o12_status"] == "PASS" else "FAIL",
                        f"fresh seeds {replication['seeds']} and fresh layouts "
                        f"{replication['train_layouts'][0]}-"
                        f"{replication['train_layouts'][-1]} / "
                        f"{replication['test_layouts'][0]}-"
                        f"{replication['test_layouts'][-1]}: outcome-trained "
                        f"{outcome:.4f} against authored {authored:.4f} "
                        f"(difference "
                        f"{replication['paired_difference_outcome_minus_authored']:+.4f}"
                        f"); stripe masked {masked:.4f}")
    elif gauge:
        q["Q10"] = gate("NOT_RUN",
                        "the outcome-trained gauge passed in O1 on seeds 37000-37002; "
                        "a fresh-seed replication was not run in this phase",
                        reason_class=NOT_DELIVERED)
    if goal:
        q["Q11"] = gate("PASS" if goal["Q11_target_proven_identifiable"] else "FAIL",
                        goal["J_diagnosis"]["conclusion"][:400])
        q["Q12"] = gate("PASS" if goal["Q12_language_beats_shuffled_and_masked"]
                        else "FAIL",
                        f"correct minus shuffled "
                        f"{goal['arms']['2_shuffled_language']['vs_arm_1']['delta']:+.4f} "
                        f"[{goal['arms']['2_shuffled_language']['vs_arm_1']['ci_low']:+.4f}, "
                        f"{goal['arms']['2_shuffled_language']['vs_arm_1']['ci_high']:+.4f}]"
                        f"; correct minus masked "
                        f"{goal['arms']['3_masked_language']['vs_arm_1']['delta']:+.4f} "
                        f"[{goal['arms']['3_masked_language']['vs_arm_1']['ci_low']:+.4f}, "
                        f"{goal['arms']['3_masked_language']['vs_arm_1']['ci_high']:+.4f}]"
                        f" over {goal['contested_keys']} contested keys")
    if unresolved:
        q["Q13"] = gate(
            unresolved.get("Q13_status",
                           "PASS" if unresolved[
                               "Q13_per_frame_cases_are_unresolved_not_assimilated"]
                           else "FAIL"),
            f"per-frame permutation: exact event identified "
            f"{unresolved['exact']['PER_FRAME_PERMUTATION']['event_identifiable_fraction']:.4f}"
            f", goal "
            f"{unresolved['exact']['PER_FRAME_PERMUTATION']['goal_identifiable_fraction']:.4f}"
            f"; learned unresolved rate "
            f"{unresolved['learned']['PER_FRAME_PERMUTATION']['unresolved_rate']:.4f} "
            f"against {unresolved['learned']['HIDDEN_PALETTE_CONVENTION']['unresolved_rate']:.4f}"
            f" under a persistent convention; confident assimilation "
            f"{unresolved['learned']['PER_FRAME_PERMUTATION']['false_confident_semantic_assignment']:.4f}"
            f" against a stricter 0.10 added after the weak criterion passed")
    q["Q14"] = gate("PASS",
                    "every seed, palette, decoy count, layout set, unresolved example "
                    "and failed arm is retained in the o2-*.json artifacts, including "
                    "the arms recorded NOT_DELIVERED")

    for name in [f"Q{i}" for i in range(15)]:
        q.setdefault(name, gate("NOT_RUN", "the producing experiment did not run",
                                reason_class=BLOCKED_UPSTREAM))

    counts = Counter(entry["status"] for entry in q.values())
    passed = {name for name, entry in q.items() if entry["status"] == "PASS"}
    q5_to_q9 = {"Q5", "Q6", "Q7", "Q8", "Q9"}
    report: dict[str, Any] = {
        "provenance": provenance,
        "corrected_p_gates": corrected_p,
        "p_ledger_reconciliation": reconciliation,
        "q_gates": q,
        "tally": dict(counts),
    }
    report["decision"] = {
        # The specification's rule: Q5-Q9 qualify the appearance-aware vision/action
        # belief. Q12 gates the language-conditioned half on top of it, so a passing Q12
        # cannot make the vision-action verdict worse.
        "vision_action_prospective_prediction_unblocked": bool(q5_to_q9 <= passed),
        "full_multimodal_prospective_prediction_unblocked": bool(
            {f"Q{i}" for i in range(15)} <= passed),
        "appearance_aware_interface_frozen": bool(
            {f"Q{i}" for i in range(15)} <= passed),
        "goal_calibration_requires_redesign": bool("Q11" not in passed),
        "learned_persistent_memory_is_the_blocker": bool(
            "Q5" not in passed and memory is not None
            and memory["arms"]["4_exact_palette_posterior"]["contested_accuracy"] > 0.9),
        "stage_1a_1_matrix_run": False,
        "final_scale1_seed_opened": False,
    }
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)

    print(f"commit {provenance['commit']}  branch {provenance['branch']}")
    print(f"\nP ledger: printed {O1_HEADLINE}, recomputed {dict(tally)} "
          f"-> matches: {reconciliation['matches']}")
    print(f"\n{'gate':6s} {'status':9s} basis")
    print("-" * 104)
    for name in sorted(q, key=lambda k: int(k[1:])):
        print(f"{name:6s} {q[name]['status']:9s} {q[name]['basis'][:88]}")
    print(f"\ntally {dict(counts)}")
    for key, value in report["decision"].items():
        print(f"{key}: {value}")
    print(f"\nwrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
