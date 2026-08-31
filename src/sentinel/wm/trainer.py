"""The Scale-0 training loop: deterministic, restartable, and fully accounted.

This loop trains nothing worth reporting. Two hundred updates at plumbing
weights on a hundred thousand transitions is a throughput measurement, and any
loss number it produces is evidence about the pipeline rather than about a
representation. What it must do is run the exact frozen workload, spend exactly
the declared budget, and stop and continue without noticing.

Determinism comes from threading an explicit PRNG key rather than from seeding a
global stream. The key is split once per update, so update *k* draws the same
noise whether it is the *k*th update of an uninterrupted run or the first update
after a restart -- which is precisely the property the restart gate checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from sentinel.wm import matrix as M
from sentinel.wm.collect import FeatureTable, SequenceSampler, materialise
from sentinel.wm.metrics import (
    action_effect_discrimination,
    code_utilisation,
    fitted_sensitivity,
    gradient_global_norm,
    rollout_divergence_by_horizon,
)
from sentinel.wm.models import SHWMModel
from sentinel.wm.objective import (
    ObjectiveBatch,
    ObjectiveConfig,
    ObjectiveResult,
    compute_objective,
    finalise,
)
from sentinel.wm.resource import ResourceReport, measure, process_resident_bytes
from sentinel.wm.restart import (
    DeclaredRunState,
    assert_restartable,
    key_to_tuple,
    load_run_state,
    save_run_state,
)
from sentinel.wm.versioning import digest_of


def build_optimizer() -> optim.AdamW:
    """Exactly the optimizer the matrix freezes. No schedule, no warmup."""
    return optim.AdamW(
        learning_rate=M.LEARNING_RATE,
        betas=list(M.BETAS),
        eps=M.EPSILON,
        weight_decay=M.WEIGHT_DECAY,
    )


def to_mlx_batch(arrays, compute_dtype: mx.Dtype = mx.bfloat16) -> ObjectiveBatch:
    return ObjectiveBatch(
        features=mx.array(arrays.features).astype(compute_dtype),
        actions=mx.array(arrays.actions),
        previous_rewards=mx.array(arrays.previous_rewards).astype(compute_dtype),
        rewards=mx.array(arrays.rewards),
        terminations=mx.array(arrays.terminations),
        event_targets=mx.array(arrays.event_targets),
        boundary_pairs=arrays.boundary_pairs,
    )


@dataclass
class TrainingOutcome:
    updates: int = 0
    loss_history: list[float] = field(default_factory=list)
    last_result: ObjectiveResult | None = None
    gradient_norms: list[float] = field(default_factory=list)
    clipped_updates: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    resource: ResourceReport = field(default_factory=lambda: ResourceReport(label="training"))
    boundary_pairs_seen: int = 0

    @property
    def parameters_digest(self) -> str:
        return self._parameters_digest

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "updates": self.updates,
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "loss_history_digest": digest_of([round(v, 6) for v in self.loss_history]),
            "mean_gradient_norm": (
                sum(self.gradient_norms) / len(self.gradient_norms) if self.gradient_norms else 0.0
            ),
            "clipped_updates": self.clipped_updates,
            "boundary_pairs_seen": self.boundary_pairs_seen,
            "diagnostics": dict(self.diagnostics),
            "resource": self.resource.canonical_dict(),
        }


def parameter_digest(model: SHWMModel) -> str:
    """A content hash of the weights, for comparing two runs bit for bit."""
    from mlx.utils import tree_flatten

    import hashlib

    hasher = hashlib.sha256()
    for name, tensor in sorted(tree_flatten(model.parameters())):
        hasher.update(name.encode())
        hasher.update(np.asarray(tensor.astype(mx.float32)).tobytes())
    return "sha256:" + hasher.hexdigest()


@dataclass
class Trainer:
    """One matrix-shaped training workload."""

    model: SHWMModel
    optimizer: optim.AdamW
    sampler: SequenceSampler
    table: FeatureTable
    objective: ObjectiveConfig
    seed: int
    data_digest: str
    split_manifest_digest: str
    compute_dtype: mx.Dtype = mx.bfloat16
    key: mx.array = field(init=False)
    update_index: int = 0
    loss_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.key = mx.random.key(self.seed)

    # ---- state ---------------------------------------------------------

    def declared_state(
        self,
        planner_account: Mapping[str, Any],
        gate_ledger: Mapping[str, Any],
        verifier_ledger: Mapping[str, Any],
    ) -> DeclaredRunState:
        return DeclaredRunState(
            update_index=self.update_index,
            prng_key=key_to_tuple(self.key),
            batch_cursor=self.update_index,
            permutation_digest=self.sampler.permutation_digest,
            config_digest=self.model.config.digest,
            objective_digest=self.objective.digest,
            data_digest=self.data_digest,
            split_manifest_digest=self.split_manifest_digest,
            planner_account=dict(planner_account),
            gate_ledger=dict(gate_ledger),
            verifier_ledger=dict(verifier_ledger),
            loss_history=tuple(self.loss_history),
        )

    def save(self, directory: Path, **ledgers: Mapping[str, Any]) -> dict[str, str]:
        state = self.declared_state(
            ledgers.get("planner_account", {}),
            ledgers.get("gate_ledger", {}),
            ledgers.get("verifier_ledger", {}),
        )
        return save_run_state(directory, state, self.model, self.optimizer)

    def restore(self, directory: Path) -> DeclaredRunState:
        state = load_run_state(directory, self.model, self.optimizer)
        if state.config_digest != self.model.config.digest:
            raise ValueError(
                f"checkpoint was written by configuration {state.config_digest[:20]}..., "
                f"not {self.model.config.digest[:20]}..."
            )
        if state.data_digest != self.data_digest:
            raise ValueError("checkpoint was written against a different transition set")
        if state.permutation_digest != self.sampler.permutation_digest:
            raise ValueError("checkpoint was written against a different batch permutation")
        self.update_index = state.update_index
        self.key = state.key
        self.loss_history = list(state.loss_history)
        return state

    # ---- one update ----------------------------------------------------

    def step(self) -> ObjectiveResult:
        """One optimizer update at the frozen batch shape."""
        arrays = materialise(self.sampler.batch(self.update_index), self.table)
        batch = to_mlx_batch(arrays, self.compute_dtype)

        self.key, forward_key = mx.random.split(self.key)
        objective = self.objective
        model = self.model

        holder: dict[str, Any] = {}

        def loss_fn(model: SHWMModel) -> mx.array:
            output = model(batch.features, batch.actions, batch.previous_rewards, key=forward_key)
            assert_restartable(output.used_global_rng)
            total, components, coverage, extra = compute_objective(model, output, batch, objective)
            holder["components"] = components
            holder["coverage"] = coverage
            holder["extra"] = extra
            holder["output"] = output
            return total

        total, gradients = nn.value_and_grad(model, loss_fn)(model)
        gradients, norm = optim.clip_grad_norm(gradients, M.GRADIENT_CLIP_NORM)
        self.optimizer.update(model, gradients)
        mx.eval(model.parameters(), self.optimizer.state, total)

        result = finalise(total, holder["components"], holder["coverage"], objective, holder["extra"])
        self.loss_history.append(result.metrics["loss/total"])
        self.update_index += 1
        holder["gradient_norm"] = float(norm.item())
        self._last = holder
        return result

    # ---- the run -------------------------------------------------------

    def run(
        self,
        updates: int,
        *,
        diagnose_every: int = 50,
        checkpoint_directory: Path | None = None,
        checkpoint_every: int | None = None,
    ) -> TrainingOutcome:
        outcome = TrainingOutcome()
        report = ResourceReport(label="training")
        started = time.perf_counter()
        with measure("training", report):
            for _ in range(updates):
                result = self.step()
                outcome.updates += 1
                outcome.loss_history.append(result.metrics["loss/total"])
                outcome.gradient_norms.append(self._last["gradient_norm"])
                if self._last["gradient_norm"] > M.GRADIENT_CLIP_NORM:
                    outcome.clipped_updates += 1
                outcome.boundary_pairs_seen += int(result.coverage.get("boundary", 0))
                outcome.last_result = result
                if diagnose_every and self.update_index % diagnose_every == 0:
                    outcome.diagnostics.update(self._diagnostics())
                if (
                    checkpoint_directory is not None
                    and checkpoint_every
                    and self.update_index % checkpoint_every == 0
                ):
                    self.save(Path(checkpoint_directory))
        elapsed = max(time.perf_counter() - started, 1e-9)
        positions = updates * M.SEQUENCE_LENGTH * M.SEQUENCES_PER_BATCH
        report.throughput = {
            "updates_per_second": updates / elapsed,
            "transition_positions_per_second": positions / elapsed,
            "seconds_per_update": elapsed / max(updates, 1),
        }
        report.trainable_parameters = self.model.actual_trainable_parameters()
        report.parameter_bytes_measured = self.model.actual_parameter_bytes()
        report.peak_resident_bytes = max(report.peak_resident_bytes, process_resident_bytes())
        outcome.resource = report
        outcome._parameters_digest = parameter_digest(self.model)
        return outcome

    def _diagnostics(self) -> dict[str, Any]:
        output = self._last["output"]
        actions = mx.array(materialise(self.sampler.batch(self.update_index - 1), self.table).actions)
        divergence = rollout_divergence_by_horizon(self.model, output, actions, (1, 2, 4))
        diagnostics: dict[str, Any] = {
            f"rollout_divergence_h{h}": v for h, v in divergence.items()
        }
        diagnostics["rollout_fitted_sensitivity"] = fitted_sensitivity(divergence)
        diagnostics["action_effect_discrimination"] = action_effect_discrimination(
            self.model, output, actions, self.model.config.action_count
        )
        diagnostics.update(
            code_utilisation(
                output.code_logits, self.model.config.code_groups, self.model.config.code_categories
            )
        )
        return diagnostics


TrainingOutcome._parameters_digest = ""  # type: ignore[attr-defined]
