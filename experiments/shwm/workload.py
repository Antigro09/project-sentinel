"""One Scale-0 workload: train, plan, verify, and account for all of it.

A workload is one matrix cell at one seed. It runs the frozen optimisation
budget, then the frozen planner dry run against fake dynamics, then an offline
verifier pass over recorded observables -- offline because the matrix fixes
online interactions at exactly zero and a verifier that stepped the environment
would be spending a budget it does not have.

What comes back is a `RunClaim`, which is the object the matching rule checks,
plus a resource report. Neither carries a loss comparison, because two hundred
updates at plumbing weights is a throughput measurement.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import mlx.core as mx

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sentinel.wm import matrix as M  # noqa: E402
from sentinel.wm.authority import AuthorityDenied, AuthorityGate  # noqa: E402
from sentinel.wm.collect import SequenceSampler  # noqa: E402
from sentinel.wm.dataset import Split, TransitionRecord  # noqa: E402
from sentinel.wm.latent_contract import (  # noqa: E402
    RepresentationKind,
    TransitionPrediction,
    UncertaintyTriple,
)
from sentinel.wm.models import build_model  # noqa: E402
from sentinel.wm.objective import ObjectiveConfig  # noqa: E402
from sentinel.wm.planner_bridge import (  # noqa: E402
    BeamPlanner,
    CEMPlanner,
    CountingRollout,
    FakeDynamicsRollout,
    MCTSPlanner,
)
from sentinel.wm.resource import (  # noqa: E402
    ResourceReport,
    estimate_training_memory,
    measure,
    process_resident_bytes,
)
from sentinel.wm.sizing import solve_config  # noqa: E402
from sentinel.wm.trainer import Trainer, build_optimizer, parameter_digest  # noqa: E402
from sentinel.wm.verifier_bridge import (  # noqa: E402
    Decision,
    DecisionController,
    ObservableVerifierBridge,
    VerificationContext,
    authorize_if_verified,
)
from sentinel.wm.versioning import digest_array, digest_of  # noqa: E402

PLANNERS = {"beam": BeamPlanner, "cem": CEMPlanner, "mcts": MCTSPlanner}


@dataclass
class WorkloadOutcome:
    workload_id: str
    claim: M.RunClaim
    resource: ResourceReport
    training: dict[str, Any]
    planner: dict[str, Any]
    verifier: dict[str, Any]
    parameters: dict[str, Any]
    restart_check: dict[str, Any] | None = None

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "claim": self.claim.canonical_dict(),
            "resource": self.resource.canonical_dict(),
            "training": self.training,
            "planner": self.planner,
            "verifier": self.verifier,
            "parameters": self.parameters,
            "restart_check": self.restart_check,
        }


def planner_dry_run(config: Mapping[str, Any]) -> tuple[dict[str, Any], int, int]:
    """The frozen planning workload against deterministic fake dynamics.

    Fake dynamics rather than the trained model, deliberately: the matching rule
    fixes planner invocations and candidates across arms, and a planner driven by
    each arm's own model would make the counts a property of the model.
    """
    settings = config["planner"]
    planner = PLANNERS[settings["adapter"]]()
    fake = FakeDynamicsRollout()
    rollout = CountingRollout(fake)
    started = time.perf_counter()
    plans = []
    for horizon in settings["horizons"]:
        for invocation in range(int(settings["invocations_per_horizon"])):
            # A distinct root per invocation. With one shared root the planner
            # would solve the same problem a hundred times and the reported
            # rollout rate would be a measurement of a cache rather than of
            # planning.
            plans.append(
                planner.plan(
                    rollout,
                    fake.root(state=invocation),
                    horizon,
                    int(settings["candidates_per_invocation"]),
                )
            )
    elapsed = time.perf_counter() - started
    account = rollout.account.canonical_dict()
    account["wall_seconds"] = elapsed
    account["plan_digest"] = digest_of([p.digest for p in plans])
    account["distinct_plans"] = len({p.digest for p in plans})
    account["rollouts_per_second"] = account["model_calls"] / elapsed if elapsed else 0.0
    return account, rollout.account.invocations, rollout.account.candidate_sequences


def verifier_dry_run(records: Sequence[TransitionRecord], sample: int = 512) -> dict[str, Any]:
    """Offline verification over recorded observables.

    Half of the sampled predictions are deliberately wrong on one probe, so the
    pass reports a detection rate against a known number of planted mismatches
    rather than a rate against whatever the model happened to do.
    """
    bridge = ObservableVerifierBridge(required=M.REQUIRED_PROBES)
    controller = DecisionController()
    gate = AuthorityGate(gate_id="scale-0-dry-run", required_probes=M.REQUIRED_PROBES)

    from sentinel.env.adapters.base import ProbeSet

    planted = 0
    detected = 0
    authorised = 0
    denied = 0
    coverage_total = 0.0
    considered = 0

    # Stride rather than a prefix: the records arrive one family at a time, so
    # the first `sample` of them would all come from the controlled adapter and
    # the visual family would never be verified at all.
    stride = max(1, len(records) // sample) if records else 1
    for index, record in enumerate(records[::stride][:sample]):
        if not record.probes_t1:
            continue
        actual = ProbeSet(dict(record.probes_t1))
        context = VerificationContext(
            episode_id=record.episode_id,
            step=record.step,
            available_probes=actual.names(),
            required_probes=M.REQUIRED_PROBES,
        )
        predicted = dict(record.probes_t1)
        corrupt = index % 2 == 1
        if corrupt:
            predicted["observable_signature"] = int(predicted["observable_signature"]) + 1
            planted += 1

        prediction = TransitionPrediction(
            next_latent=digest_array([0.0]),
            event_distribution={"ACTION_SUCCEEDED": 1.0},
            reward_mean=float(record.reward),
            reward_variance=0.0,
            termination_probability=1.0 if record.termination else 0.0,
            uncertainty=UncertaintyTriple(0.0, 0.0, 0.0),
            rollout_support_scope="development",
            model_version=digest_of("scale-0-dry-run"),
            action=record.action,
        )
        result = bridge.verify(prediction, predicted, actual, context)
        considered += 1
        coverage_total += result.coverage
        if corrupt and "observable_signature" in result.rejected_observables:
            detected += 1

        decision = controller.decide(
            prediction.uncertainty, float(bool(actual.values.get("constraint_violation")))
        )
        try:
            token = authorize_if_verified(gate, decision, record.action, result)
            gate.consume(token, record.action)
            authorised += 1
        except AuthorityDenied:
            denied += 1

    return {
        "verifications": considered,
        "planted_mismatches": planted,
        "detected_mismatches": detected,
        "detection_rate": detected / planted if planted else float("nan"),
        "mean_coverage": coverage_total / considered if considered else 0.0,
        "authorised_actions": authorised,
        "denied_actions": denied,
        "bridge": bridge.ledger(),
        "controller": controller.ledger(),
        "gate": gate.ledger(),
    }


def run_workload(
    cell: M.MatrixCell,
    seed: int,
    dataset,
    config: Mapping[str, Any],
    *,
    output_root: Path,
    is_matrix_run: bool,
    restart_check: bool = False,
    cache_report: Mapping[str, Any] | None = None,
) -> WorkloadOutcome:
    settings = config["optimisation"]
    workload_id = f"{cell.cell_id}.s{seed}"

    report = ResourceReport(label=workload_id)
    load_started = time.perf_counter()
    sized = solve_config(
        RepresentationKind(cell.representation.value),
        cell.target_parameters,
        encoder_dimension=int(config["encoder"]["feature_dimension"]),
        latent_width=cell.latent_width,
        action_count=4,
    )
    model = build_model(sized.config, seed=seed)
    report.cold_load_seconds = time.perf_counter() - load_started

    sampler = SequenceSampler.from_records(
        dataset.records,
        dataset.manifest,
        split=Split.TRAIN,
        sequence_length=int(settings["sequence_length"]),
        batch_size=int(settings["sequences_per_batch"]),
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        optimizer=build_optimizer(),
        sampler=sampler,
        table=dataset.table,
        objective=ObjectiveConfig(),
        seed=seed,
        data_digest=dataset.transition_ids_digest,
        split_manifest_digest=dataset.manifest.digest,
    )

    outcome = trainer.run(int(settings["updates"]), diagnose_every=100)

    restart_result: dict[str, Any] | None = None
    if restart_check:
        restart_result = _restart_equivalence(
            cell, seed, dataset, config, sized, outcome, output_root / "restart" / workload_id
        )

    planner_account, invocations, candidates = planner_dry_run(config)
    verifier = verifier_dry_run(dataset.records)

    estimate = estimate_training_memory(
        outcome.resource.trainable_parameters,
        batch_positions=int(settings["sequences_per_batch"]) * int(settings["sequence_length"]),
        activation_width=sized.config.belief_dimension,
        activation_layers=sized.config.core_depth * 3,
    )
    report.wall_seconds = outcome.resource.wall_seconds + planner_account["wall_seconds"]
    report.mlx_peak_bytes = outcome.resource.mlx_peak_bytes
    report.mlx_active_bytes = outcome.resource.mlx_active_bytes
    report.mlx_cache_bytes = outcome.resource.mlx_cache_bytes
    report.peak_resident_bytes = max(outcome.resource.peak_resident_bytes, process_resident_bytes())
    report.trainable_parameters = outcome.resource.trainable_parameters
    report.frozen_parameters = 0
    report.parameter_bytes_measured = outcome.resource.parameter_bytes_measured
    report.estimated_model_bytes = estimate["model_bytes"] + estimate["gradient_bytes"]
    report.estimated_optimizer_bytes = estimate["optimizer_bytes"]
    report.estimated_activation_bytes = estimate["activation_bytes"]
    report.throughput = dict(outcome.resource.throughput)
    report.throughput["planner_rollouts_per_second"] = planner_account["rollouts_per_second"]
    # Measured once by the caller: walking a payload tree of tens of thousands
    # of files per workload would put the filesystem into the throughput number.
    report.cache_report = dict(cache_report) if cache_report is not None else dataset.cache.size_report()
    report.planner_account = planner_account

    claim = M.RunClaim(
        workload_id=workload_id,
        seed=seed,
        trainable_parameters=outcome.resource.trainable_parameters,
        target_parameters=cell.target_parameters,
        transition_ids_digest=dataset.transition_ids_digest,
        split_manifest_digest=dataset.manifest.digest,
        optimizer_updates=outcome.updates,
        sequence_length=int(settings["sequence_length"]),
        sequences_per_batch=int(settings["sequences_per_batch"]),
        online_interactions=int(config["planner"]["online_actions"]),
        planner_invocations=invocations,
        planner_candidates=candidates,
        required_probe_digest=M.REQUIRED_PROBE_DIGEST,
        is_matrix_run=is_matrix_run,
    )

    return WorkloadOutcome(
        workload_id=workload_id,
        claim=claim,
        resource=report,
        training={
            **outcome.canonical_dict(),
            "parameters_digest": outcome.parameters_digest,
            "config_digest": sized.config.digest,
            "objective_digest": trainer.objective.digest,
            "permutation_digest": sampler.permutation_digest,
            "sequences_available": len(sampler.sequences),
        },
        planner=planner_account,
        verifier=verifier,
        parameters=sized.canonical_dict(),
        restart_check=restart_result,
    )


def _restart_equivalence(
    cell: M.MatrixCell,
    seed: int,
    dataset,
    config: Mapping[str, Any],
    sized,
    reference,
    directory: Path,
) -> dict[str, Any]:
    """Re-run the same budget in halves and compare the weights bit for bit."""
    settings = config["optimisation"]
    updates = int(settings["updates"])
    half = updates // 2

    def fresh() -> Trainer:
        model = build_model(sized.config, seed=seed)
        sampler = SequenceSampler.from_records(
            dataset.records,
            dataset.manifest,
            split=Split.TRAIN,
            sequence_length=int(settings["sequence_length"]),
            batch_size=int(settings["sequences_per_batch"]),
            seed=seed,
        )
        return Trainer(
            model=model,
            optimizer=build_optimizer(),
            sampler=sampler,
            table=dataset.table,
            objective=ObjectiveConfig(),
            seed=seed,
            data_digest=dataset.transition_ids_digest,
            split_manifest_digest=dataset.manifest.digest,
        )

    first = fresh()
    first.run(half, diagnose_every=0)
    first.save(directory)

    second = fresh()
    state = second.restore(directory)
    second.run(updates - half, diagnose_every=0)

    return {
        "checkpoint_at_update": half,
        "restored_update_index": state.update_index,
        "uninterrupted_digest": reference.parameters_digest,
        "restarted_digest": parameter_digest(second.model),
        "match": reference.parameters_digest == parameter_digest(second.model),
        "loss_history_match": [round(v, 6) for v in second.loss_history]
        == [round(v, 6) for v in reference.loss_history],
    }
