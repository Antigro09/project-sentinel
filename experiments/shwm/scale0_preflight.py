"""The Scale-0 driver: preflight, dataset, workloads, and the gate.

Two modes, and the difference is not cosmetic.

`matrix` runs the 48 frozen workloads against the two named frozen backbones. It
refuses to start unless both are runnable, because the run matrix says so: "If
either family cannot run faithfully, Scale 0 stops."

`dry_run` runs the same 48 workload *shapes* against the deterministic control
encoder. It measures the pipeline -- throughput, memory, restart, accounting --
and every artefact it writes is stamped `is_matrix_run: false`. It is not the
matrix and cannot be reported as the matrix. It exists because the pipeline can
be audited while an encoder is unavailable, and because the handoff asks for
fake-model dry-run resources to be reported separately.

    uv run --python .venv-shwm/bin/python python experiments/shwm/scale0_preflight.py
    ... --limit 4          # smoke test a few workloads
    ... --mode matrix      # the real thing, once both backbones are available
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.wm import matrix as M  # noqa: E402
from sentinel.wm.backbones import matrix_may_run, preflight_all  # noqa: E402
from sentinel.wm.latent_contract import RepresentationKind  # noqa: E402
from sentinel.wm.provenance import (  # noqa: E402
    FreezeManifest,
    environment_state,
    git_state,
)
from sentinel.wm.resource import directory_bytes  # noqa: E402
from sentinel.wm.restart import ProcessStateAudit  # noqa: E402
from sentinel.wm.versioning import digest_file, digest_of  # noqa: E402

import dataset as dataset_module  # noqa: E402
import workload as workload_module  # noqa: E402


def load_config(overrides: list[str]) -> dict[str, Any]:
    """Hydra composition, so the frozen values live in one versioned file."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = str(Path(__file__).resolve().parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        composed = compose(config_name="scale0", overrides=overrides)
    return OmegaConf.to_container(composed, resolve=True)  # type: ignore[return-value]


def cells_from_config(config: dict[str, Any]) -> list[M.MatrixCell]:
    """Build the cell list from config and check it against the frozen matrix."""
    matrix_config = config["matrix"]
    primary = [
        M.MatrixCell(encoder, RepresentationKind(arm), int(target), int(matrix_config["primary_width"]), "primary")
        for encoder in matrix_config["encoders"]
        for arm in matrix_config["representations"]
        for target in matrix_config["targets"]
    ]
    controls = [
        M.MatrixCell(encoder, RepresentationKind.HYBRID, M.DIMENSION_CONTROL_TARGET, int(width), "dimension_control")
        for encoder in matrix_config["encoders"]
        for width in matrix_config["control_widths"]
    ]
    cells = primary + controls
    expected = {c.cell_id for c in M.all_cells()}
    if {c.cell_id for c in cells} != expected:
        raise SystemExit(
            "the configured cells do not match the frozen matrix; a change here is a "
            "matrix amendment and must be committed before any workload runs"
        )
    return cells


def start_mlflow(config: dict[str, Any], output_root: Path):
    import mlflow

    uri = config["output"]["mlflow_uri"]
    if uri.startswith("sqlite:///") and not Path(uri[len("sqlite:///") :]).parent.exists():
        Path(uri[len("sqlite:///") :]).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["output"]["experiment"])
    return mlflow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry_run", "matrix"], default=None)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N workloads")
    parser.add_argument("--override", action="append", default=[], help="Hydra override")
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    config = load_config(arguments.override)
    if arguments.mode:
        config["mode"] = arguments.mode
    is_matrix_run = config["mode"] == "matrix"

    output_root = REPO / config["output"]["root"]
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = arguments.out or (output_root / f"{config['mode']}-report.json")

    audit = ProcessStateAudit()
    audit.capture()

    started = time.perf_counter()
    preflights = preflight_all()
    permitted, reason = matrix_may_run(preflights)
    print(f"backbone preflight: matrix_may_run={permitted}\n  {reason}\n")
    if is_matrix_run and not permitted:
        print("STOP: the frozen matrix cannot run. This is a pre-registered stop condition.")
        print("A replacement backbone requires a reviewed pre-run amendment to")
        print("SCALE-0-RUN-MATRIX.md; it is never selected after comparing results.")
        return 2

    print(f"building the sealed dataset ({config['data']['total_transitions']:,} transitions per slot)")
    dataset_started = time.perf_counter()
    datasets, dataset_summary = dataset_module.build_all(config, output_root)
    dataset_seconds = time.perf_counter() - dataset_started
    print(f"  built in {dataset_seconds:.1f}s; shared transition digest "
          f"{dataset_summary['transition_ids_digest'][:24]}...\n")

    cells = cells_from_config(config)
    seeds = [int(s) for s in config["matrix"]["seeds"]]
    workloads = [(cell, seed) for cell in cells for seed in seeds]
    if arguments.limit:
        workloads = workloads[: arguments.limit]

    mlflow = start_mlflow(config, output_root)
    outcomes: list[workload_module.WorkloadOutcome] = []
    failures: list[str] = []

    restart_target = workloads[0][0].cell_id if workloads else None
    # Measured once. Walking a cache of tens of thousands of files inside the
    # per-workload loop would put the filesystem into the throughput numbers.
    baseline_artifact_bytes = directory_bytes(output_root)
    cache_reports = {slot: d.cache.size_report() for slot, d in datasets.items()}

    for index, (cell, seed) in enumerate(workloads, start=1):
        workload_id = f"{cell.cell_id}.s{seed}"
        do_restart = cell.cell_id == restart_target and seed == seeds[0]
        print(f"[{index:2d}/{len(workloads)}] {workload_id}", flush=True)
        run_started = time.perf_counter()
        try:
            outcome = workload_module.run_workload(
                cell,
                seed,
                datasets[cell.encoder_id],
                config,
                output_root=output_root,
                is_matrix_run=is_matrix_run,
                restart_check=do_restart,
                cache_report=cache_reports[cell.encoder_id],
            )
        except Exception as exc:  # a failed workload stays in the report
            failures.append(f"{workload_id}: {type(exc).__name__}: {exc}")
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - run_started

        envelope = M.check_resource_envelope(
            outcome.resource.peak_unified_memory_bytes,
            baseline_artifact_bytes,
            elapsed,
        )
        if envelope:
            failures.extend(f"{workload_id}: {f}" for f in envelope)
        outcomes.append(outcome)

        with mlflow.start_run(run_name=workload_id):
            mlflow.log_params(
                {
                    "encoder_slot": cell.encoder_id,
                    "representation": cell.representation.value,
                    "target_parameters": cell.target_parameters,
                    "latent_width": cell.latent_width,
                    "seed": seed,
                    "role": cell.role,
                    "is_matrix_run": is_matrix_run,
                    "mode": config["mode"],
                }
            )
            mlflow.log_metrics(
                {
                    "trainable_parameters": outcome.claim.trainable_parameters,
                    "final_loss": outcome.training["final_loss"] or 0.0,
                    "wall_seconds": outcome.resource.wall_seconds,
                    "peak_unified_gib": outcome.resource.peak_unified_memory_bytes / 1024**3,
                    "updates_per_second": outcome.resource.throughput.get("updates_per_second", 0.0),
                    "planner_rollouts_per_second": outcome.planner["rollouts_per_second"],
                    "verifier_detection_rate": outcome.verifier["detection_rate"],
                    "mean_gradient_norm": outcome.training["mean_gradient_norm"],
                }
            )
        print(
            f"    {outcome.claim.trainable_parameters:,} params "
            f"({outcome.claim.trainable_parameters / cell.target_parameters - 1:+.4%}), "
            f"{elapsed:.1f}s, peak {outcome.resource.peak_unified_memory_bytes / 1024**3:.2f} GiB, "
            f"{outcome.resource.throughput.get('updates_per_second', 0):.2f} upd/s"
        )
        if outcome.restart_check:
            print(f"    restart equivalence: {outcome.restart_check['match']}")

    total_seconds = time.perf_counter() - started
    matching = M.check_match([o.claim for o in outcomes]) if outcomes else ["no workload completed"]
    artifact_bytes = directory_bytes(output_root)
    envelope = M.check_resource_envelope(
        max((o.resource.peak_unified_memory_bytes for o in outcomes), default=0),
        artifact_bytes,
        max((o.resource.wall_seconds for o in outcomes), default=0.0),
        matrix_seconds=total_seconds,
        cache_build_seconds=dataset_seconds,
    )

    try:
        audit.assert_no_undeclared_state()
        undeclared: list[str] = []
    except Exception as exc:
        undeclared = [str(exc)]

    git = git_state(REPO)
    manifest = FreezeManifest(
        phase="SHWM-SCALE-0",
        base_commit="5205543b110ba6da2e3f6da30630809941f821c4",
        implementation_commit=git["commit"],
        dirty_tracked=git["dirty_tracked"],
        dependency_lock_sha256=digest_file(REPO / "uv.lock"),
        encoder_identities=tuple(
            sorted(d.cache.size_report()["entries"] and slot for slot, d in datasets.items())
        ),
        environment_generator_sha256=digest_of(
            {
                "synthetic_control": digest_file(REPO / "src/sentinel/env/adapters/synthetic_control.py"),
                "procedural_visual": digest_file(REPO / "src/sentinel/env/adapters/procedural_visual.py"),
            }
        ),
        split_procedure_sha256=digest_file(REPO / "src/sentinel/wm/dataset.py"),
        evaluator_sha256=digest_file(REPO / "src/sentinel/wm/verifier_bridge.py"),
        config_sha256=digest_of(config),
        gate_document_sha256=digest_file(
            REPO / "docs/phase-2-continuous-world-model/SCALE-0-RUN-MATRIX.md"
        ),
    )

    gate_passed = (
        not failures
        and not matching
        and not envelope
        and not undeclared
        and len(outcomes) == len(workloads)
        and permitted
        and is_matrix_run
    )

    document = {
        "mode": config["mode"],
        "is_matrix_run": is_matrix_run,
        "run_label": config["run_label"],
        "git": git,
        "environment": environment_state(),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "backbone_preflight": {
            "matrix_may_run": permitted,
            "reason": reason,
            "records": [p.canonical_dict() for p in preflights],
        },
        "dataset": dataset_summary,
        "dataset_build_seconds": dataset_seconds,
        "workloads_expected": len(workloads),
        "workloads_completed": len(outcomes),
        "workloads": [o.canonical_dict() for o in outcomes],
        "matching_failures": matching,
        "resource_envelope_failures": envelope,
        "undeclared_state": undeclared,
        "failures": failures,
        "total_seconds": total_seconds,
        "artifact_bytes": artifact_bytes,
        "freeze_manifest": manifest.canonical_dict(),
        "freeze_manifest_digest": manifest.digest,
        "scale_0_gate_passed": gate_passed,
        "gate_note": (
            "PASS requires mode=matrix with both frozen backbones runnable. A dry run "
            "measures the pipeline and can never pass the Scale-0 gate."
        ),
    }
    report_path.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n")

    checksums = {
        str(path.relative_to(output_root)): digest_file(path)
        for path in sorted(output_root.rglob("*.json"))
        if path.is_file()
    }
    (output_root / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n"
    )

    print()
    print(f"workloads completed : {len(outcomes)}/{len(workloads)}")
    print(f"matching failures   : {len(matching)}")
    print(f"envelope failures   : {len(envelope)}")
    print(f"undeclared state    : {len(undeclared)}")
    print(f"total wall clock    : {total_seconds / 60:.1f} min")
    print(f"artifacts           : {artifact_bytes / 1024**3:.2f} GiB")
    print(f"Scale-0 gate passed : {gate_passed}")
    print(f"report              : {report_path}")
    return 0 if not failures and not matching and not envelope else 1


if __name__ == "__main__":
    raise SystemExit(main())
