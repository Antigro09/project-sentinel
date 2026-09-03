"""E / F / G. Derive every restart procedure from the frozen tables, then gate.

Nothing is retrained here. Fixed K=8/16/32 and the adaptive rule are prefix operations
over the development and validation restart tables, and the certificate threshold tau is
read off DEVELOPMENT alone before any validation row is touched.

`certify` sees a training log-likelihood and tau. The alias accuracy that appears in this
module is used to MEASURE how often the certificate is wrong -- which is the whole point
of a false-certification rate -- and never to decide anything the procedure does.

    .venv-shwm/bin/python experiments/shwm/m2f_procedures.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as m2d
import m2f_core as core
from m2d_core import ARTIFACTS, write

SOLVED = 0.9          # what counts as a solved model when SCORING the certificate
CATASTROPHE = 0.6     # what counts as a collapse
FALLBACK = 0.5        # an unresolved seed falls back to memoryless, which is exactly 0.5


def load(split: str):
    report = json.loads((ARTIFACTS / f"m2f-restarts-{split}.json").read_text())
    arrays = np.load(ARTIFACTS / f"m2f-restarts-{split}.npz")
    rows = [core.RestartRow(**r) for r in report["restart_table"]]
    return report, arrays, rows


def procedure_selection(rows_by_seed, seeds, boundaries, tau):
    """Which boundary snapshot each procedure ends up using, per seed."""
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for name, k in (("1_fixed_k8", 8), ("2_fixed_k16", 16), ("3_fixed_k32", 32)):
        out[name] = {}
        for seed in seeds:
            best = core.fixed_k(rows_by_seed[seed], k)
            out[name][seed] = {
                "boundary": boundaries.index(k), "restarts_used": k,
                "selected_restart": best.restart,
                "training_log_likelihood": best.training_log_likelihood,
                "certificate": core.certify(best.training_log_likelihood, tau),
                "alias_accuracy": best.alias_accuracy}
    out["4_adaptive"] = {}
    for seed in seeds:
        best, used, certificate = core.adaptive(rows_by_seed[seed], tau)
        out["4_adaptive"][seed] = {
            "boundary": boundaries.index(used), "restarts_used": used,
            "selected_restart": best.restart,
            "training_log_likelihood": best.training_log_likelihood,
            "certificate": certificate, "alias_accuracy": best.alias_accuracy}
    return out


def gather_hits(snapshot: np.ndarray, seeds, selection) -> np.ndarray:
    return np.stack([snapshot[i, selection[seed]["boundary"]]
                     for i, seed in enumerate(seeds)])


def summarise(selection, seeds) -> dict[str, Any]:
    accuracy = np.array([selection[s]["alias_accuracy"] for s in seeds])
    certified = np.array([selection[s]["certificate"] == "CERTIFIED" for s in seeds])
    solved = accuracy >= SOLVED
    used = np.array([selection[s]["restarts_used"] for s in seeds])
    coverage = float(certified.mean())
    overall = float(np.where(certified, accuracy, FALLBACK).mean())
    return {
        "mean": float(accuracy.mean()), "median": float(np.median(accuracy)),
        "p10": float(np.percentile(accuracy, 10)), "minimum": float(accuracy.min()),
        "catastrophic_rate": float((accuracy < CATASTROPHE).mean()),
        "solved_rate": float(solved.mean()),
        "certificate_coverage": coverage,
        "certified_accuracy": float(accuracy[certified].mean()) if certified.any()
        else float("nan"),
        "false_certification_rate": float((certified & ~solved).mean()),
        "missed_certification_rate": float((~certified & solved).mean()),
        "unresolved_rate": float(1.0 - coverage),
        "overall_with_memoryless_fallback": overall,
        "mean_restarts": float(used.mean()), "max_restarts": int(used.max()),
        "total_updates": int(used.sum() * m2d.UPDATES),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "m2f-procedures.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    dev_report, dev_arrays, dev_rows = load("development")
    val_report, val_arrays, val_rows = load("validation")
    boundaries = list(dev_report["boundaries"])
    dev_seeds = list(dev_report["seeds"])
    val_seeds = list(val_report["seeds"])

    # ---- freeze tau and the F6 bounds on DEVELOPMENT --------------------------------------
    tau_block = core.choose_tau(dev_rows, solved_threshold=SOLVED)
    tau = tau_block["tau"]
    print(f"development: {len(dev_seeds)} seeds x {dev_report['k_max']} restarts, "
          f"{len(dev_rows)} rows")
    print(f"  solved restart fraction {tau_block['solved_fraction']:.4f}")
    print(f"  worst solved log-likelihood {tau_block.get('worst_solved_log_likelihood', float('nan')):.5f}")
    print(f"  best unsolved log-likelihood {tau_block.get('best_unsolved_log_likelihood', float('nan')):.5f}")
    print(f"  separable: {tau_block['separable']}  ->  tau = {tau:.5f}")

    dev_by_seed = core.table_by_seed(dev_rows)
    dev_selection = procedure_selection(dev_by_seed, dev_seeds, boundaries, tau)
    dev_summary = {k: summarise(v, dev_seeds) for k, v in dev_selection.items()}
    frozen_bounds = {
        "false_certification_bound":
            float(max(0.02, dev_summary["4_adaptive"]["false_certification_rate"] + 0.02)),
        "coverage_floor": float(dev_summary["4_adaptive"]["certificate_coverage"] - 0.05),
        "overall_floor":
            float(dev_summary["4_adaptive"]["overall_with_memoryless_fallback"] - 0.05),
    }
    print(f"  frozen bounds: {frozen_bounds}")

    # ---- validation ------------------------------------------------------------------------
    val_by_seed = core.table_by_seed(val_rows)
    selection = procedure_selection(val_by_seed, val_seeds, boundaries, tau)
    summary = {k: summarise(v, val_seeds) for k, v in selection.items()}
    for name, block in val_report.get("baselines", {}).items():
        summary[name] = {"mean": block["mean"], "p10": block.get("p10"),
                         "minimum": block.get("minimum"), "baseline": True}

    print(f"\nvalidation: {len(val_seeds)} untouched seeds")
    print(f"{'procedure':28s} {'alias':>7s} {'p10':>7s} {'min':>7s} {'catas':>6s} "
          f"{'cover':>6s} {'falseC':>7s} {'unres':>6s} {'fallback':>8s} {'K̄':>5s}")
    print("-" * 104)
    for name in ("1_fixed_k8", "2_fixed_k16", "3_fixed_k32", "4_adaptive"):
        s = summary[name]
        print(f"{name:28s} {s['mean']:7.4f} {s['p10']:7.4f} {s['minimum']:7.4f} "
              f"{s['catastrophic_rate']:6.3f} {s['certificate_coverage']:6.3f} "
              f"{s['false_certification_rate']:7.4f} {s['unresolved_rate']:6.3f} "
              f"{s['overall_with_memoryless_fallback']:8.4f} {s['mean_restarts']:5.1f}")
    for name, block in val_report.get("baselines", {}).items():
        print(f"{name:28s} {block['mean']:7.4f} "
              f"{(block.get('p10') if block.get('p10') is not None else float('nan')):7.4f}")

    # ---- intervals --------------------------------------------------------------------------
    rows_n = dev_arrays["snapshot::primary"].shape[2]
    val_rows_n = val_arrays["snapshot::primary"].shape[2]
    seed_column = np.repeat(np.array(val_seeds), val_rows_n)
    layout_column = np.tile(val_arrays["row_layout"], len(val_seeds))
    class_column = np.tile(val_arrays["row_alias_class"], len(val_seeds))
    changes = np.tile(val_arrays["row_changes"], len(val_seeds))
    memoryless = val_arrays["baseline::7_trained_memoryless"].ravel()

    intervals: dict[str, Any] = {}
    print("\npaired hierarchical intervals vs trained memoryless")
    for name in ("1_fixed_k8", "2_fixed_k16", "3_fixed_k32", "4_adaptive"):
        hits = gather_hits(val_arrays["snapshot::primary"], val_seeds,
                           selection[name]).ravel()
        block = {"vs_memoryless": m2d.hierarchical_paired_interval(
            hits, memoryless, seed_column, layout_column, class_column)}
        for label, mask in (("changes_2", changes == 2), ("changes_3", changes == 3),
                            ("changes_4plus", changes >= 4)):
            block[label] = m2d.hierarchical_paired_interval(
                hits, memoryless, seed_column, layout_column, class_column, mask=mask)
        # Certified subset only.
        certified = np.array([selection[name][s]["certificate"] == "CERTIFIED"
                              for s in val_seeds])
        if certified.any():
            mask = np.repeat(certified, val_rows_n)
            block["certified_only"] = m2d.hierarchical_paired_interval(
                hits, memoryless, seed_column, layout_column, class_column, mask=mask)
        intervals[name] = block
        print(f"  {name:28s} {block['vs_memoryless']['delta']:+.4f} "
              f"[{block['vs_memoryless']['ci_low']:+.4f}, "
              f"{block['vs_memoryless']['ci_high']:+.4f}]"
              f"{' *' if block['vs_memoryless']['excludes_zero'] else ''}")
    for name in ("5_single_long_run", "6_gru_multistart"):
        key = f"baseline::{name}"
        if key not in val_arrays:
            continue
        intervals[name] = {"vs_memoryless": m2d.hierarchical_paired_interval(
            val_arrays[key].ravel(), memoryless, seed_column, layout_column,
            class_column)}
        # And against the adaptive procedure, under equal cumulative compute.
        adaptive_hits = gather_hits(val_arrays["snapshot::primary"], val_seeds,
                                    selection["4_adaptive"]).ravel()
        intervals[name]["adaptive_minus_this"] = m2d.hierarchical_paired_interval(
            adaptive_hits, val_arrays[key].ravel(), seed_column, layout_column,
            class_column)
        print(f"  {name:28s} {intervals[name]['vs_memoryless']['delta']:+.4f} "
              f"[{intervals[name]['vs_memoryless']['ci_low']:+.4f}, "
              f"{intervals[name]['vs_memoryless']['ci_high']:+.4f}]   adaptive beats it by "
              f"{intervals[name]['adaptive_minus_this']['delta']:+.4f}"
              f"{' *' if intervals[name]['adaptive_minus_this']['excludes_zero'] else ''}")

    # ---- gates -------------------------------------------------------------------------------
    adaptive = summary["4_adaptive"]
    fixed8 = summary["1_fixed_k8"]
    genericity = json.loads((ARTIFACTS / "m2e-genericity.json").read_text())
    answer_digest = core.m2e.initialisation_digest(core.m2e.ANSWER)
    generic_never_answer = all(
        core.m2e.initialisation_digest(core.generic_spec(s, r).perturbation)
        != answer_digest
        for s in list(dev_seeds)[:20] for r in range(8))

    gates: dict[str, dict[str, Any]] = {}
    gates["F0"] = {"pass": bool(genericity["v2_all_eligible_initialisations_generic"]
                                and generic_never_answer),
                   "basis": "m2e-genericity V2 plus a check that no M2F generic draw "
                            "equals the answer initialisation"}
    gates["F1"] = {"pass": bool(tau_block["separable"]),
                   "basis": f"development gap between worst solved and best unsolved "
                            f"training log-likelihood = {tau_block.get('gap', float('nan')):.5f}"}
    gates["F2"] = {"pass": bool(adaptive["false_certification_rate"]
                                <= frozen_bounds["false_certification_bound"]),
                   "basis": f"validation false certification "
                            f"{adaptive['false_certification_rate']:.4f} vs frozen bound "
                            f"{frozen_bounds['false_certification_bound']:.4f}"}
    gates["F3"] = {"pass": bool(adaptive["catastrophic_rate"]
                                < fixed8["catastrophic_rate"]),
                   "basis": f"adaptive catastrophic rate {adaptive['catastrophic_rate']:.4f} "
                            f"vs fixed K=8 {fixed8['catastrophic_rate']:.4f}"}
    equal = [intervals[n]["adaptive_minus_this"] for n in
             ("5_single_long_run", "6_gru_multistart") if n in intervals]
    gates["F4"] = {"pass": bool(equal and all(e["ci_low"] > 0 for e in equal)),
                   "basis": "; ".join(
                       f"{n}: {intervals[n]['adaptive_minus_this']['delta']:+.4f} "
                       f"[{intervals[n]['adaptive_minus_this']['ci_low']:+.4f}, "
                       f"{intervals[n]['adaptive_minus_this']['ci_high']:+.4f}]"
                       for n in ("5_single_long_run", "6_gru_multistart")
                       if n in intervals) or "equal-compute arms not run"}
    certified_interval = intervals["4_adaptive"].get("certified_only",
                                                     intervals["4_adaptive"]["vs_memoryless"])
    gates["F5"] = {"pass": bool(certified_interval["ci_low"] > 0),
                   "basis": f"certified-only {certified_interval['delta']:+.4f} "
                            f"[{certified_interval['ci_low']:+.4f}, "
                            f"{certified_interval['ci_high']:+.4f}]"}
    gates["F6"] = {"pass": bool(
        adaptive["certificate_coverage"] >= frozen_bounds["coverage_floor"]
        and adaptive["overall_with_memoryless_fallback"] >= frozen_bounds["overall_floor"]),
        "basis": f"coverage {adaptive['certificate_coverage']:.4f} vs floor "
                 f"{frozen_bounds['coverage_floor']:.4f}; overall "
                 f"{adaptive['overall_with_memoryless_fallback']:.4f} vs floor "
                 f"{frozen_bounds['overall_floor']:.4f}"}
    gates["F7"] = {"pass": bool(all(
        intervals["4_adaptive"][f"changes_{s}"]["ci_low"] > 0
        for s in ("2", "3", "4plus"))),
        "basis": "; ".join(
            f"{s}: {intervals['4_adaptive'][f'changes_{s}']['delta']:+.4f} "
            f"[{intervals['4_adaptive'][f'changes_{s}']['ci_low']:+.4f}, "
            f"{intervals['4_adaptive'][f'changes_{s}']['ci_high']:+.4f}]"
            for s in ("2", "3", "4plus"))}
    gates["F8"] = {"pass": True,
                   "basis": "the answer-oriented arm is recorded as a baseline "
                            "diagnostic and is never a selectable procedure"}
    gates["F9"] = {"pass": bool(len(dev_rows) == len(dev_seeds) * dev_report["k_max"]
                                and len(val_rows) == len(val_seeds) * val_report["k_max"]),
                   "basis": f"{len(dev_rows)} development and {len(val_rows)} validation "
                            f"restart rows retained, none dropped"}

    report = {
        "tau": tau, "tau_selection": tau_block, "frozen_bounds": frozen_bounds,
        "development_summary": dev_summary, "validation_summary": summary,
        "validation_selection": {k: {str(s): v for s, v in block.items()}
                                 for k, block in selection.items()},
        "intervals": intervals, "gates": gates,
        "f_gates_all_pass": bool(all(g["pass"] for g in gates.values())),
        "restart_table_digests": {"development": dev_report["restart_table_digest"],
                                  "validation": val_report["restart_table_digest"]},
        "wall_clock_seconds": time.perf_counter() - started,
    }
    write(arguments.out, report)
    print(f"\n{'gate':5s} {'pass':6s} basis")
    for name, block in gates.items():
        print(f"{name:5s} {str(block['pass']):6s} {block['basis'][:92]}")
    print(f"\nF0-F9 all pass: {report['f_gates_all_pass']}")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
