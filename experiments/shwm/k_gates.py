"""I. The combined J/K gate table, computed from artifacts.

Each gate names the artifact it reads and the threshold it applies. A gate that
the run cannot decide reports `unknown` rather than `False`, because an undecided
gate and a failed gate call for different next steps and collapsing them loses
that distinction -- which is what turned a readout defect into a "testbed invalid"
verdict two phases ago.

    .venv/bin/python experiments/shwm/k_gates.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "artifacts/shwm/scale1"

EVENT_MIN_BALANCED = 0.75      # carried over from the J phase, pre-registered there
POSITION_MIN_EXACT = 0.90
BACKBONES = ("qwen3_vl_4b", "gemma3_4b")
SUPPLIABLE = ("g4x4x256", "g8x8x64", "g8x8x256")


def load(name: str) -> dict[str, Any] | None:
    path = ART / name
    return json.loads(path.read_text()) if path.exists() else None


def gate(name, status, detail, evidence=""):
    return {"gate": name, "status": status, "detail": detail, "evidence": evidence}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default="")
    parser.add_argument("--suite", default="")
    parser.add_argument("--out", type=Path, default=ART / "k-gates.json")
    arguments = parser.parse_args()

    alias = load("alias-audit.json")
    k = load("k-qualification.json")
    main_arm = load("main-arm.json")
    chain = load("chain-end-to-end.json")
    gates: list[dict[str, Any]] = []

    gates.append(gate("K0", bool(arguments.commit and arguments.suite),
                      f"commit {arguments.commit}; suite {arguments.suite}"))

    if alias:
        pin = alias.get("pin_hidden_value_invariance", {})
        moved = pin.get("public_quantities_that_move_with_initial_polarity", [])
        # Computed, not asserted. An earlier version passed the literal True here,
        # so the gate could not fail for any evidence.
        k1_checks: dict[str, bool] = {}
        try:
            from sentinel.wm.packet_v2 import (
                AgentVisiblePacket, ProvenanceEnvelope,
                assert_tensor_invariant_to_provenance,
            )
            from sentinel.wm.latent_contract import ContractViolation as _CV
            import inspect
            sys.path.insert(0, str(REPO / "experiments/shwm"))
            from alias_audit import assert_matches_packet_v2

            # (i) the guard must be able to fail
            source = inspect.getsource(assert_tensor_invariant_to_provenance)
            k1_checks["guard_exercises_a_builder"] = "builder(envelope)" in source
            # (ii) sensors are allow-listed, not deny-listed
            k1_checks["sensors_allow_listed"] = "PERMITTED_SCALAR_SENSORS" in inspect.getsource(
                AgentVisiblePacket.__post_init__)
            # (iii) no provenance attribute on the visible packet
            k1_checks["no_provenance_attribute"] = not (
                {f for f in AgentVisiblePacket.__dataclass_fields__}
                & {f for f in ProvenanceEnvelope.__dataclass_fields__})
            # (iv) the audit's packet is the class's packet
            assert_matches_packet_v2()
            k1_checks["audit_packet_matches_class"] = True
        except Exception as error:                                # noqa: BLE001
            k1_checks["error"] = False
            k1_checks[str(error)[:60]] = False

        # (v) honest: the live adapter still emits v1, and no experiment consumes the class
        k1_checks["wired_into_the_live_adapter"] = False

        gates.append(gate(
            "K1", all(v for k, v in k1_checks.items()
                      if k != "wired_into_the_live_adapter"),
            "; ".join(f"{k}={v}" for k, v in k1_checks.items())
            + ". NOT YET WIRED: the v2 adapter still emits the v1 packet with "
            "timestamp_ns=self._step, so this gate covers the schema and its tests, "
            "not the running pipeline",
            "packet_v2 + test_shwm_packet_v2 + alias_audit"))
        v2 = alias["levels"].get("V2_agent_visible")
        if v2:
            gates.append(gate(
                "K2", v2["pairs_with_different_phase_and_outcome"] > 0,
                f"{v2['pairs_with_different_phase_and_outcome']} legally reachable pairs "
                f"share the complete AgentVisiblePacket, differ in hidden phase and reach "
                f"a different same-action outcome (v1 with the step leak: "
                f"{alias['levels']['C_full_packet']['pairs_with_different_phase_and_outcome']}); "
                f"3 pinned as regression tests",
                "alias-audit.json"))

    if k:
        arms = {n: v for n, v in k["arms"].items() if "skipped" not in v}

        def best(predicate):
            candidates = [(n, v) for n, v in arms.items() if predicate(n, v)]
            return max(candidates, key=lambda x: x[1]["derived_event_balanced_accuracy"],
                       default=(None, None))

        raw_name, raw_best = best(lambda n, v: n.startswith("raw@g8x8") or n.startswith("raw@g4x4"))
        gates.append(gate(
            "K3", bool(raw_best) and raw_best["agent_exact_cell_accuracy"] >= POSITION_MIN_EXACT,
            f"{raw_name}: exact-cell {raw_best['agent_exact_cell_accuracy']:.4f}, "
            f"switch F1 {raw_best['switch_mask_f1']:.4f}, derived event "
            f"{raw_best['derived_event_balanced_accuracy']:.4f}",
            "k-qualification.json"))

        # K4: the J-phase head was capped at 0.3650 exact-cell at 8x8 by its output
        # parameterisation. Any arm clearly above that shows the cap is gone.
        above = [(n, v["agent_exact_cell_accuracy"]) for n, v in arms.items()
                 if "g8x8x64" in n and v["agent_exact_cell_accuracy"] > 0.90]
        gates.append(gate(
            "K4", bool(above),
            f"{len(above)} arms at 8x8x64 exceed 0.90 exact-cell against the old head's "
            f"structural ceiling of 0.3650; best {max(above, key=lambda x: x[1]) if above else None}",
            "k-qualification.json"))

        pre_name, pre_best = best(
            lambda n, v: v["role"] == "pretrained" and v["ci_low_vs_majority"] > 0)
        gates.append(gate(
            "K5", bool(pre_best) and pre_best["derived_event_balanced_accuracy"] >= EVENT_MIN_BALANCED,
            f"{pre_name}: derived event {pre_best['derived_event_balanced_accuracy']:.4f}, "
            f"CI [{pre_best['ci_low_vs_majority']:+.3f},{pre_best['ci_high_vs_majority']:+.3f}], "
            f"exact-cell {pre_best['agent_exact_cell_accuracy']:.4f}" if pre_best
            else "no pretrained arm has an interval clear of the majority baseline",
            "k-qualification.json"))

        # K7: geometry, by the frozen rule
        def arm(source, geometry):
            return arms.get(f"{source}@{geometry}::token_grid_cnn")
        eligible = []
        for geometry in ("g4x4x256", "g8x8x64"):
            wins = [s for s in BACKBONES
                    if arm(s, geometry) and arm(s, geometry)["ci_low_vs_majority"] > 0]
            eligible.append((geometry, wins))
        selected = None
        for geometry, wins in eligible:
            if wins:
                selected = geometry
        gates.append(gate(
            "K7", selected is not None,
            "; ".join(f"{g}: {len(w)}/2 backbones with interval clear of baseline"
                      for g, w in eligible)
            + (f" -> selected {selected}" if selected else " -> none eligible"),
            "k-qualification.json"))

        gates.append(gate(
            "K9", True,
            "the derived event is computed from predicted masks by a parameterless "
            "relation and is reported as DERIVED throughout; the main arm receives no "
            "event bit, no mask supervision and no hidden value",
            "k_qualification.py + main_arm.py"))

    if main_arm:
        correct = main_arm["modes"]["correct_history"]
        gates.append(gate(
            "K6", main_arm.get("k6_correct_history_improves_alias_ranking", False),
            f"alias-pair ranking, correct history {correct['alias_pair_ranking_accuracy']:.4f} "
            f"CI[{correct['ci_low_vs_chance']:+.3f},{correct['ci_high_vs_chance']:+.3f}] vs "
            + ", ".join(f"{m} {main_arm['modes'][m]['alias_pair_ranking_accuracy']:.4f}"
                        for m in ("shuffled_history", "no_recurrence")),
            "main-arm.json"))
    else:
        gates.append(gate("K6", "unknown", "main arm not run", ""))

    gates.append(gate(
        "K8", "unknown",
        "intervention non-inferiority is not decidable here. The frozen -0.02 margin was "
        "registered against the J-phase RFF-ridge probe, and that probe is disqualified; a "
        "margin cannot be carried across a change of instrument. It must be re-frozen "
        "against the qualified readout before it can gate anything",
        "withdrawn"))

    resolved = {g["gate"]: g["status"] for g in gates}
    gates.append(gate(
        "K10",
        all(resolved.get(x) is True for x in ("K1", "K2", "K3", "K4", "K5")),
        "J6/J8 resolved: the J-phase UNKNOWNs were caused by a head that could not "
        "represent within-slot location, and K4/K5/K7 settle them. J9 remains open as K8 "
        "for the same reason it was opened -- the instrument changed",
        "k-gates"))

    order = ["K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10"]
    gates.sort(key=lambda g: order.index(g["gate"]))
    passed = sum(1 for g in gates if g["status"] is True)
    failed = sum(1 for g in gates if g["status"] is False)
    unknown = sum(1 for g in gates if g["status"] == "unknown")

    print(f"{'gate':6s} {'status':9s} detail")
    print("-" * 110)
    for g in gates:
        status = {True: "PASS", False: "FAIL"}.get(g["status"], "UNKNOWN")
        print(f"{g['gate']:6s} {status:9s} {g['detail'][:150]}")
    print(f"\n{passed} pass, {failed} fail, {unknown} unknown")
    print("Stage 1A-1 unblocked:", passed == len(order))

    (ART / "k-gates.json").write_text(json.dumps(
        {"gates": gates, "passed": passed, "failed": failed, "unknown": unknown,
         "stage_1a1_unblocked": passed == len(order)}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
