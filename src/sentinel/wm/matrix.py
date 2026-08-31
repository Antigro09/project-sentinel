"""The frozen Scale-0 run matrix, expressed as code.

`SCALE-0-RUN-MATRIX.md` is the single source of truth for what Scale 0 runs.
This module is that document in executable form, so that "the run matched the
matrix" is a test rather than a claim in a report. Every constant here is a
transcription; none of them is a tuning knob.

The matching rule is the load-bearing part. Eight quantities must hold
simultaneously -- parameters, data identity, updates, batch shape, online
interactions, planner counts, required probes, and seeds -- and `check_match`
returns every failure rather than the first, because a run that violates three
rules should not be fixed one restart at a time.

Wall time, memory, FLOPs, and cache size are pointedly *not* matched. They are
measured outcomes. Equalising them after the fact would be exactly the silent
adjustment the document forbids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from sentinel.wm.latent_contract import ContractViolation, Precision, RepresentationKind
from sentinel.wm.versioning import digest_of

MATRIX_VERSION = "scale-0-v1"

ENCODER_IDS: tuple[str, ...] = ("qwen3_vl_4b", "gemma3_4b")
REPRESENTATION_ARMS: tuple[RepresentationKind, ...] = (
    RepresentationKind.CONTINUOUS,
    RepresentationKind.DISCRETE,
    RepresentationKind.HYBRID,
)
TRAINABLE_TARGETS: tuple[int, ...] = (50_000_000, 200_000_000)
PARAMETER_TOLERANCE = 0.01

PRIMARY_WIDTH = 512
DIMENSION_CONTROL_WIDTHS: tuple[int, ...] = (256, 1024)
DIMENSION_CONTROL_ARM = RepresentationKind.HYBRID
DIMENSION_CONTROL_TARGET = 50_000_000

DEVELOPMENT_SEEDS: tuple[int, ...] = (6600, 6601, 6602)

TOTAL_TRANSITIONS = 100_000
TRANSITIONS_PER_ENVIRONMENT: Mapping[str, int] = {
    "synthetic_control": 50_000,
    "procedural_visual": 50_000,
}
"""Keyed by adapter name.

The run matrix writes the first family as "deterministic controlled fixtures";
the adapter that implements it is `synthetic_control`. Keying by the adapter
name means a rename cannot leave the config and the contract silently
disagreeing about which environment a count belongs to.
"""
MIXTURE_COUNTS: Mapping[str, int] = {
    "random": 15_000,
    "scripted_oracle": 12_500,
    "sentinel": 12_500,
    "uncertainty_seeking": 10_000,
}

SEQUENCE_LENGTH = 32
SEQUENCES_PER_BATCH = 32
OPTIMIZER_UPDATES = 200
TRANSITION_POSITIONS = SEQUENCE_LENGTH * SEQUENCES_PER_BATCH * OPTIMIZER_UPDATES  # 204,800
OPTIMIZER = "AdamW"
LEARNING_RATE = 3e-4
BETAS: tuple[float, float] = (0.9, 0.95)
EPSILON = 1e-8
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
COMPUTE_PRECISION = Precision.BF16
ACCUMULATOR_PRECISION = Precision.FP32
LOSS_PLUMBING_WEIGHT = 1.0
ONLINE_INTERACTIONS_PER_RUN = 0

PLANNER_HORIZONS: tuple[int, ...] = (5, 10, 25)
PLANNER_INVOCATIONS_PER_HORIZON = 100
PLANNER_CANDIDATES_PER_INVOCATION = 64
PLANNER_CANDIDATES_TOTAL = (
    len(PLANNER_HORIZONS) * PLANNER_INVOCATIONS_PER_HORIZON * PLANNER_CANDIDATES_PER_INVOCATION
)  # 19,200

PEAK_MEMORY_LIMIT_BYTES = 112 * (1024**3)
ARTIFACT_LIMIT_BYTES = 200 * (1024**3)
RUN_TIMEOUT_SECONDS = 2 * 3600
MATRIX_TIMEOUT_SECONDS = 72 * 3600
CACHE_BUILD_TIMEOUT_SECONDS = 8 * 3600

REQUIRED_PROBES: tuple[str, ...] = (
    "reward",
    "termination",
    "goal_progress",
    "constraint_violation",
    "action_succeeded",
    "observable_signature",
)
"""The evaluator-required probe set, frozen independently of model requests.

Model-requested probes may add diagnostic coverage. They can never remove or
replace anything on this tuple.
"""

REQUIRED_PROBE_DIGEST = digest_of(sorted(REQUIRED_PROBES))


@dataclass(frozen=True, slots=True)
class MatrixCell:
    """One (encoder, representation, size, width) combination."""

    encoder_id: str
    representation: RepresentationKind
    target_parameters: int
    latent_width: int
    role: str  # "primary" or "dimension_control"

    @property
    def cell_id(self) -> str:
        return (
            f"{self.encoder_id}.{self.representation.value}."
            f"{self.target_parameters // 1_000_000}M.w{self.latent_width}"
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "representation": self.representation.value,
            "target_parameters": self.target_parameters,
            "latent_width": self.latent_width,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class MatrixWorkload:
    """One cell at one development seed. The unit the 48-count refers to."""

    cell: MatrixCell
    seed: int

    @property
    def workload_id(self) -> str:
        return f"{self.cell.cell_id}.s{self.seed}"

    def canonical_dict(self) -> dict[str, Any]:
        return {"cell": self.cell.canonical_dict(), "seed": self.seed}


def primary_cells() -> tuple[MatrixCell, ...]:
    return tuple(
        MatrixCell(encoder, arm, target, PRIMARY_WIDTH, "primary")
        for encoder in ENCODER_IDS
        for arm in REPRESENTATION_ARMS
        for target in TRAINABLE_TARGETS
    )


def dimension_control_cells() -> tuple[MatrixCell, ...]:
    return tuple(
        MatrixCell(encoder, DIMENSION_CONTROL_ARM, DIMENSION_CONTROL_TARGET, width, "dimension_control")
        for encoder in ENCODER_IDS
        for width in DIMENSION_CONTROL_WIDTHS
    )


def all_cells() -> tuple[MatrixCell, ...]:
    return primary_cells() + dimension_control_cells()


def all_workloads() -> tuple[MatrixWorkload, ...]:
    return tuple(
        MatrixWorkload(cell, seed) for cell in all_cells() for seed in DEVELOPMENT_SEEDS
    )


def matrix_arithmetic() -> dict[str, int]:
    """The counts the matrix states in words, recomputed from the factors."""
    primary = len(primary_cells())
    controls = len(dimension_control_cells())
    return {
        "primary_cells": primary,
        "primary_workloads": primary * len(DEVELOPMENT_SEEDS),
        "dimension_control_cells": controls,
        "dimension_control_workloads": controls * len(DEVELOPMENT_SEEDS),
        "total_workloads": (primary + controls) * len(DEVELOPMENT_SEEDS),
        "transition_positions_per_run": TRANSITION_POSITIONS,
        "planner_candidates_per_run": PLANNER_CANDIDATES_TOTAL,
    }


def assert_matrix_encoder(encoder_id: str) -> str:
    """Refuse an encoder that is not one of the two named frozen families."""
    if encoder_id not in ENCODER_IDS:
        raise ContractViolation(
            f"{encoder_id!r} is not a frozen matrix encoder; the matrix names "
            f"{list(ENCODER_IDS)} and a replacement requires a committed pre-run amendment"
        )
    return encoder_id


def parameters_within_tolerance(actual: int, target: int) -> bool:
    return abs(actual - target) <= target * PARAMETER_TOLERANCE


@dataclass(frozen=True, slots=True)
class RunClaim:
    """What a completed run says about itself, for checking against the matrix."""

    workload_id: str
    seed: int
    trainable_parameters: int
    target_parameters: int
    transition_ids_digest: str
    split_manifest_digest: str
    optimizer_updates: int
    sequence_length: int
    sequences_per_batch: int
    online_interactions: int
    planner_invocations: int
    planner_candidates: int
    required_probe_digest: str
    optimizer: str = OPTIMIZER
    learning_rate: float = LEARNING_RATE
    is_matrix_run: bool = False

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "seed": self.seed,
            "trainable_parameters": self.trainable_parameters,
            "target_parameters": self.target_parameters,
            "transition_ids_digest": self.transition_ids_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "optimizer_updates": self.optimizer_updates,
            "sequence_length": self.sequence_length,
            "sequences_per_batch": self.sequences_per_batch,
            "online_interactions": self.online_interactions,
            "planner_invocations": self.planner_invocations,
            "planner_candidates": self.planner_candidates,
            "required_probe_digest": self.required_probe_digest,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "is_matrix_run": self.is_matrix_run,
        }


def check_match(claims: Sequence[RunClaim]) -> list[str]:
    """Every matching-rule violation across a set of runs, not just the first."""
    failures: list[str] = []
    if not claims:
        return ["no runs were claimed"]

    for claim in claims:
        if not parameters_within_tolerance(claim.trainable_parameters, claim.target_parameters):
            drift = claim.trainable_parameters / claim.target_parameters - 1.0
            failures.append(
                f"{claim.workload_id}: {claim.trainable_parameters:,} trainable parameters is "
                f"{drift:+.3%} from target {claim.target_parameters:,}, outside +/-1%"
            )
        if claim.optimizer_updates != OPTIMIZER_UPDATES:
            failures.append(
                f"{claim.workload_id}: {claim.optimizer_updates} optimizer updates, expected {OPTIMIZER_UPDATES}"
            )
        if claim.sequence_length != SEQUENCE_LENGTH or claim.sequences_per_batch != SEQUENCES_PER_BATCH:
            failures.append(
                f"{claim.workload_id}: batch shape {claim.sequences_per_batch}x{claim.sequence_length}, "
                f"expected {SEQUENCES_PER_BATCH}x{SEQUENCE_LENGTH}"
            )
        if claim.online_interactions != ONLINE_INTERACTIONS_PER_RUN:
            failures.append(
                f"{claim.workload_id}: {claim.online_interactions} online interactions, expected 0"
            )
        expected_invocations = len(PLANNER_HORIZONS) * PLANNER_INVOCATIONS_PER_HORIZON
        if claim.planner_invocations != expected_invocations:
            failures.append(
                f"{claim.workload_id}: {claim.planner_invocations} planner invocations, "
                f"expected {expected_invocations}"
            )
        if claim.planner_candidates != PLANNER_CANDIDATES_TOTAL:
            failures.append(
                f"{claim.workload_id}: {claim.planner_candidates} planner candidates, "
                f"expected {PLANNER_CANDIDATES_TOTAL}"
            )
        if claim.required_probe_digest != REQUIRED_PROBE_DIGEST:
            failures.append(
                f"{claim.workload_id}: required probe set does not match the frozen evaluator set"
            )
        if claim.optimizer != OPTIMIZER:
            failures.append(f"{claim.workload_id}: optimizer {claim.optimizer}, expected {OPTIMIZER}")

    data_digests = {c.transition_ids_digest for c in claims}
    if len(data_digests) > 1:
        failures.append(
            f"runs used {len(data_digests)} different transition-ID sets; all arms must see the same raw data"
        )
    split_digests = {c.split_manifest_digest for c in claims}
    if len(split_digests) > 1:
        failures.append(f"runs used {len(split_digests)} different split manifests")

    seen_seeds = {c.seed for c in claims}
    missing = set(DEVELOPMENT_SEEDS) - seen_seeds
    if missing and len(claims) >= len(DEVELOPMENT_SEEDS):
        failures.append(f"development seed(s) {sorted(missing)} are absent; no dropped failed seed is allowed")
    extra = seen_seeds - set(DEVELOPMENT_SEEDS)
    if extra:
        failures.append(f"non-development seed(s) used: {sorted(extra)}")

    return failures


def check_resource_envelope(
    peak_memory_bytes: int,
    artifact_bytes: int,
    run_seconds: float,
    matrix_seconds: float | None = None,
    cache_build_seconds: float | None = None,
) -> list[str]:
    """Hard ceilings. Exceeding one is a stop, never a reason to shrink an arm."""
    failures: list[str] = []
    if peak_memory_bytes > PEAK_MEMORY_LIMIT_BYTES:
        failures.append(
            f"peak process memory {peak_memory_bytes / 1024**3:.2f} GiB exceeds the "
            f"{PEAK_MEMORY_LIMIT_BYTES / 1024**3:.0f} GiB ceiling"
        )
    if artifact_bytes > ARTIFACT_LIMIT_BYTES:
        failures.append(
            f"artifact storage {artifact_bytes / 1024**3:.2f} GiB exceeds the "
            f"{ARTIFACT_LIMIT_BYTES / 1024**3:.0f} GiB ceiling"
        )
    if run_seconds > RUN_TIMEOUT_SECONDS:
        failures.append(f"run wall time {run_seconds / 3600:.2f} h exceeds the 2 h per-run ceiling")
    if matrix_seconds is not None and matrix_seconds > MATRIX_TIMEOUT_SECONDS:
        failures.append(f"matrix wall time {matrix_seconds / 3600:.2f} h exceeds the 72 h ceiling")
    if cache_build_seconds is not None and cache_build_seconds > CACHE_BUILD_TIMEOUT_SECONDS:
        failures.append(
            f"cache build wall time {cache_build_seconds / 3600:.2f} h exceeds the 8 h ceiling"
        )
    return failures
