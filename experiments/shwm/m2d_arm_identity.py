"""B. Resolve which arm actually produced the 0.6503 row, from artifacts and AST.

The M2C report says "learned event + selected learned filter"; the M2C table says
"learned event + accumulator". A report string is not evidence of which arm ran, and
neither is a table heading, so neither is used here. The artifact is read for the
fields it holds, the frozen source is parsed for the fields it does not, and the
temporal mechanism is decided by asking the syntax tree two questions:

    is the selected filter's runner ever CALLED, or only imported?
    does the phase estimator accumulate with XOR?

Both are answerable without running anything and without trusting a label.

    .venv-shwm/bin/python experiments/shwm/m2d_arm_identity.py
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m2d_core import ARTIFACTS, digest_file, write  # noqa: E402

SOURCE = REPO / "experiments/shwm/learned_event_coupling.py"
ARTIFACT = ARTIFACTS / "learned-event-coupling.json"
ROW = "learned_event_accumulator"

FIELDS = ("arm_identifier", "event_source", "temporal_mechanism", "model_class",
          "checkpoint_hash", "initialization_rule", "trainable_parameters",
          "supervision", "input_fields", "seed", "population", "metric",
          "query_action_budget")


def imported_names(tree: ast.AST) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return out


def called_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                out.add(target.id)
            elif isinstance(target, ast.Attribute):
                out.add(target.attr)
    return out


def function_named(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def uses_xor_accumulation(function: ast.FunctionDef) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.BitXor):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2d-arm-identity.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    artifact = json.loads(ARTIFACT.read_text())
    tree = ast.parse(SOURCE.read_text())
    imports = imported_names(tree)
    calls = called_names(tree)

    filter_aliases = {alias for alias, origin in imports.items()
                      if origin.startswith("filter_stability.")}
    filter_called = sorted(a for a in filter_aliases if a in calls)
    estimator = function_named(tree, "phase_from_learned_events")
    xor = uses_xor_accumulation(estimator) if estimator else False

    mechanism = "exact_accumulator" if xor and not filter_called else (
        "learned_filter" if filter_called else "unknown")

    row = artifact["arms"][ROW]
    resolved: dict[str, dict[str, Any]] = {
        "arm_identifier": {"value": ROW, "source": "artifact: arms key"},
        "event_source": {"value": "learned, hard argmax",
                         "source": "source: extractor(item).argmax in "
                                   "belief_factorization.train_event_extractor"},
        "temporal_mechanism": {
            "value": mechanism,
            "source": f"AST: filter aliases {sorted(filter_aliases)} imported, "
                      f"called={filter_called or 'NONE'}; "
                      f"phase_from_learned_events XOR-accumulates={xor}"},
        "model_class": {"value": "int accumulator over predicted events (no model object)",
                        "source": "source: `h ^= int(predicted[index])`"},
        "checkpoint_hash": {"value": None, "source": "ABSENT FROM ARTIFACT"},
        "initialization_rule": {"value": None,
                                "source": "ABSENT FROM ARTIFACT; the recorded "
                                          "`selected_filter` string names an arm that "
                                          "was never instantiated"},
        "trainable_parameters": {"value": 0,
                                 "source": "source: the accumulator has no parameters; "
                                           "the prediction head is a separate object "
                                           "whose count is ABSENT FROM ARTIFACT"},
        "supervision": {"value": "displacement target; event detector trained on "
                                 "evaluator-derived labels for a public quantity",
                        "source": "source: train_event_extractor"},
        "input_fields": {"value": ["structured public row", "reset stripe",
                                   "predicted event"],
                         "source": "source: sequence_features + predicted event"},
        "seed": {"value": None,
                 "source": "ABSENT FROM ARTIFACT; source shows SEEDS[:3] = "
                           "(9000, 9001, 9002) for the head, extractor seed 6600"},
        "population": {"value": artifact.get("alias_examples"),
                       "source": "artifact: alias_examples count only; the LAYOUT SET "
                                 "is ABSENT FROM ARTIFACT"},
        "metric": {"value": "pairwise outcome accuracy",
                   "source": "source: evaluate_on_aliases"},
        "query_action_budget": {"value": None, "source": "ABSENT FROM ARTIFACT"},
    }

    print(f"row under audit: {ROW}  accuracy {row['pairwise_accuracy_mean']:.4f}  "
          f"advantage {row['vs_memoryless']['delta']:+.4f} "
          f"[{row['vs_memoryless']['ci_low']:+.4f}, {row['vs_memoryless']['ci_high']:+.4f}]")
    print(f"artifact records `selected_filter` = {artifact.get('selected_filter')!r} "
          f"as a STRING FIELD\n")
    print(f"{'field':24s} {'value':>10s}  provenance")
    print("-" * 100)
    for field in FIELDS:
        entry = resolved[field]
        value = "ABSENT" if entry["value"] is None else str(entry["value"])
        print(f"{field:24s} {value[:10]:>10s}  {entry['source']}")

    absent = [f for f in FIELDS if resolved[f]["value"] is None]
    verdict = "NOT_RUN" if mechanism == "exact_accumulator" else "REPRODUCE"
    report = {
        "row": ROW,
        "reported_accuracy": row["pairwise_accuracy_mean"],
        "reported_advantage": row["vs_memoryless"],
        "resolved_fields": resolved,
        "fields_absent_from_artifact": absent,
        "filter_aliases_imported": sorted(filter_aliases),
        "filter_aliases_called": filter_called,
        "phase_estimator_xor_accumulates": xor,
        "temporal_mechanism": mechanism,
        "u7_verdict": verdict,
        "frozen_predictions_available": False,
        "reproducible_without_retraining": False,
        "artifact_digest": digest_file(ARTIFACT),
        "source_digest": digest_file(SOURCE),
        "population_layout_set_recorded": False,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    write(arguments.out, report)
    print(f"\n{len(absent)} of {len(FIELDS)} identity fields are ABSENT from the artifact: "
          f"{absent}")
    print(f"temporal_mechanism resolves to: {mechanism}")
    print(f"U7 verdict: {verdict}")
    print(f"frozen predictions exist for this row: "
          f"{report['frozen_predictions_available']} -- the metric CANNOT be reproduced "
          f"without retraining")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
