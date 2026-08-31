"""The nine declared loss components, and the plumbing that keeps them honest.

Scale 0 runs every component at weight 1.0. That is a plumbing choice, not a
coefficient selection, and no capability comparison may be drawn from the
resulting loss values. What Scale 0 *does* have to establish is that the
machinery is auditable:

* the reported total is exactly the weighted sum of the reported components,
  computed in a fixed order so the equality is bit-stable rather than
  approximate;
* disabling a component removes that component's contribution and its metric,
  and nothing else;
* a non-finite value fails the run instead of propagating into a checkpoint.

Each component also reports **coverage**: how many elements of the batch it was
actually able to score. The boundary term is the reason -- it needs branch
groups whose observable successors differ, and a batch containing none of those
contributes zero. A zero that means "nothing to measure" and a zero that means
"measured and satisfied" are different facts, and collapsing them is how a term
gets credit for work it never did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import mlx.core as mx
import mlx.nn as nn

from sentinel.wm.latent_contract import ContractViolation
from sentinel.wm.models import ForwardOutput
from sentinel.wm.versioning import digest_of

COMPONENT_NAMES: tuple[str, ...] = (
    "next",
    "multistep",
    "reward",
    "terminal",
    "inverse",
    "event",
    "calibration",
    "consistency",
    "boundary",
)
"""Fixed order. The total is summed in this order so the equality is exact."""

NONNEGATIVE_COMPONENTS: frozenset[str] = frozenset(
    {"next", "multistep", "terminal", "inverse", "event", "consistency", "boundary"}
)
"""Components that are non-negative by construction.

`reward` and `calibration` are Gaussian negative log-likelihoods, and a Gaussian
NLL can be negative. That matters because the mechanically checked
`weightedLoss_nonnegative` lemma is *conditional* on every component being
non-negative, and two of these are not. Rather than swap in a non-negative
surrogate -- which would let the variance head win by reporting certainty, the
exact failure the proper scoring rule exists to prevent -- the precondition is
checked at runtime and reported per step. Whether the lemma applies to a given
run becomes a measured fact instead of an assumption.
"""

BOUNDARY_MARGIN = 1.0


class NonFiniteLoss(RuntimeError):
    """A loss component produced NaN or infinity. The run stops."""


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    weights: Mapping[str, float] = field(
        default_factory=lambda: {name: 1.0 for name in COMPONENT_NAMES}
    )
    enabled: frozenset[str] = frozenset(COMPONENT_NAMES)
    multistep_horizon: int = 4

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(COMPONENT_NAMES)
        if unknown:
            raise ContractViolation(f"unknown loss component(s) {sorted(unknown)}")
        unknown = set(self.enabled) - set(COMPONENT_NAMES)
        if unknown:
            raise ContractViolation(f"cannot enable unknown component(s) {sorted(unknown)}")
        for name, weight in self.weights.items():
            if weight < 0.0:
                raise ContractViolation(
                    f"component {name} has negative weight {weight}; the "
                    "non-negativity result the objective relies on requires w >= 0"
                )

    def without(self, *names: str) -> "ObjectiveConfig":
        return ObjectiveConfig(
            weights=dict(self.weights),
            enabled=frozenset(self.enabled) - set(names),
            multistep_horizon=self.multistep_horizon,
        )

    @property
    def digest(self) -> str:
        return digest_of(
            {
                "weights": {k: float(v) for k, v in sorted(self.weights.items())},
                "enabled": sorted(self.enabled),
                "multistep_horizon": self.multistep_horizon,
            }
        )


@dataclass(frozen=True, slots=True)
class ObjectiveBatch:
    """The targets. Every field is derived from recorded transitions only."""

    features: mx.array           # (B, T, E) frozen encoder features
    actions: mx.array            # (B, T)    action taken at t
    previous_rewards: mx.array   # (B, T, 1)
    rewards: mx.array            # (B, T)    reward received for the action at t
    terminations: mx.array       # (B, T)    1.0 when the episode ended at t
    event_targets: mx.array      # (B, T)    index into the frozen event order
    boundary_pairs: tuple[tuple[int, int, int, int], ...] = ()
    """(batch_a, time_a, batch_b, time_b) for branch siblings whose observable
    successors differ. Empty when the batch contains no usable branch group."""


@dataclass(frozen=True, slots=True)
class ObjectiveResult:
    total: mx.array
    components: Mapping[str, float]
    weighted: Mapping[str, float]
    coverage: Mapping[str, int]
    metrics: Mapping[str, float]

    @property
    def nonnegativity_precondition_holds(self) -> bool:
        """Whether `weightedLoss_nonnegative` applies to this step's components."""
        return all(value >= 0.0 for value in self.components.values())

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "total": float(self.total.item()),
            "components": {k: float(v) for k, v in sorted(self.components.items())},
            "weighted": {k: float(v) for k, v in sorted(self.weighted.items())},
            "coverage": {k: int(v) for k, v in sorted(self.coverage.items())},
            "metrics": {k: float(v) for k, v in sorted(self.metrics.items())},
            "nonnegativity_precondition_holds": self.nonnegativity_precondition_holds,
        }


def _mse(prediction: mx.array, target: mx.array) -> mx.array:
    return mx.mean((prediction.astype(mx.float32) - target.astype(mx.float32)) ** 2)


def _gaussian_nll(mean: mx.array, log_variance: mx.array, target: mx.array) -> mx.array:
    """Proper scoring rule, so the variance head cannot win by predicting zero."""
    clipped = mx.clip(log_variance.astype(mx.float32), -8.0, 8.0)
    error = (target.astype(mx.float32) - mean.astype(mx.float32)) ** 2
    return mx.mean(0.5 * (clipped + error * mx.exp(-clipped)))


def _cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    return mx.mean(nn.losses.cross_entropy(logits.astype(mx.float32), targets))


def _binary_cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    return mx.mean(
        nn.losses.binary_cross_entropy(
            logits.astype(mx.float32).squeeze(-1), targets.astype(mx.float32), with_logits=True
        )
    )


def multistep_prediction(
    model, output: ForwardOutput, actions: mx.array, horizon: int
) -> list[tuple[mx.array, mx.array]]:
    """Imagine `horizon` steps without new observations.

    Compounding error is the point: Lemma 3 says a small one-step error and a
    sensitivity above one still diverge, so a one-step number on its own says
    nothing about a rollout. Pairs are returned per horizon so the metric can be
    reported by depth rather than averaged into one figure.
    """
    action_vectors = model.action_embedding(actions)
    pairs: list[tuple[mx.array, mx.array]] = []
    time_steps = output.belief.shape[1]
    state = output.belief
    for step in range(1, horizon + 1):
        if step >= time_steps:
            break
        state = model.dynamics(state, action_vectors)
        predicted = model.head_next_latent(state)[:, : time_steps - step]
        target = mx.stop_gradient(output.latent[:, step:])
        pairs.append((predicted, target))
    return pairs


def boundary_separation(
    output: ForwardOutput, pairs: Sequence[tuple[int, int, int, int]]
) -> tuple[mx.array, int, int, float]:
    """Contrastive margin on branch siblings with verifier-distinct successors.

    Penalises a model that collapses two states which look identical now but
    behave differently under the actions the branch data actually recorded. It
    is a surrogate, and it identifies nothing causal on its own: without action
    coverage and adequate probes there is no pair to contrast.

    The embeddings are L2-normalised first. Without that the margin is measured
    in whatever scale the projection happens to have -- at initialisation in 512
    dimensions that is a distance around 17, so a margin of 1.0 is satisfied by
    every pair and the term reports zero forever while appearing to work. On the
    unit sphere the distance lies in [0, 2] and the margin means something.

    Returns the loss, the number of pairs, the number where the hinge is
    actually active, and the mean pair distance. A zero from "no pairs", a zero
    from "pairs, none collapsed", and a real penalty are three different facts.
    """
    if not pairs:
        return mx.array(0.0, dtype=mx.float32), 0, 0, float("nan")
    left = mx.stack([output.boundary[b, t] for b, t, _, _ in pairs]).astype(mx.float32)
    right = mx.stack([output.boundary[b, t] for _, _, b, t in pairs]).astype(mx.float32)
    left = left / mx.sqrt(mx.sum(left**2, axis=-1, keepdims=True) + 1e-8)
    right = right / mx.sqrt(mx.sum(right**2, axis=-1, keepdims=True) + 1e-8)
    distance = mx.sqrt(mx.sum((left - right) ** 2, axis=-1) + 1e-8)
    hinge = mx.maximum(0.0, BOUNDARY_MARGIN - distance)
    active = int(mx.sum(hinge > 0.0).item())
    mean_distance = float(mx.mean(distance).item())
    return mx.mean(hinge), len(pairs), active, mean_distance


def compute_objective(
    model,
    output: ForwardOutput,
    batch: ObjectiveBatch,
    config: ObjectiveConfig,
) -> tuple[mx.array, dict[str, mx.array], dict[str, int], dict[str, float]]:
    """Return the total, the raw component values, their coverage, and diagnostics."""
    components: dict[str, mx.array] = {}
    coverage: dict[str, int] = {}
    extra: dict[str, float] = {}
    zero = mx.array(0.0, dtype=mx.float32)

    batch_size, time_steps = batch.actions.shape

    if "next" in config.enabled:
        predicted = output.next_latent[:, :-1]
        target = mx.stop_gradient(output.latent[:, 1:])
        components["next"] = _mse(predicted, target)
        coverage["next"] = batch_size * max(time_steps - 1, 0)

    if "multistep" in config.enabled:
        pairs = multistep_prediction(model, output, batch.actions, config.multistep_horizon)
        if pairs:
            components["multistep"] = mx.mean(mx.stack([_mse(p, t) for p, t in pairs]))
            coverage["multistep"] = sum(int(p.shape[0] * p.shape[1]) for p, _ in pairs)
        else:
            components["multistep"] = zero
            coverage["multistep"] = 0

    if "reward" in config.enabled:
        mean, log_variance = mx.split(output.reward, 2, axis=-1)
        components["reward"] = _gaussian_nll(
            mean.squeeze(-1), log_variance.squeeze(-1), batch.rewards
        )
        coverage["reward"] = batch_size * time_steps

    if "terminal" in config.enabled:
        components["terminal"] = _binary_cross_entropy(output.termination, batch.terminations)
        coverage["terminal"] = batch_size * time_steps

    if "inverse" in config.enabled:
        components["inverse"] = _cross_entropy(output.inverse_logits, batch.actions[:, :-1])
        coverage["inverse"] = batch_size * max(time_steps - 1, 0)

    if "event" in config.enabled:
        components["event"] = _cross_entropy(output.event_logits, batch.event_targets)
        coverage["event"] = batch_size * time_steps

    if "calibration" in config.enabled:
        # The uncertainty head is scored against the error it claims to predict,
        # under a proper rule, so it cannot minimise by reporting certainty.
        residual = mx.stop_gradient(
            mx.mean((output.next_latent[:, :-1] - output.latent[:, 1:]) ** 2, axis=-1)
        )
        aleatoric = output.uncertainty[:, :-1, 0]
        components["calibration"] = _gaussian_nll(
            mx.zeros_like(residual), aleatoric, mx.sqrt(residual + 1e-8)
        )
        coverage["calibration"] = batch_size * max(time_steps - 1, 0)

    if "consistency" in config.enabled:
        components["consistency"] = _mse(
            output.consistency, mx.stop_gradient(output.latent)
        )
        coverage["consistency"] = batch_size * time_steps

    if "boundary" in config.enabled:
        value, count, active, mean_distance = boundary_separation(output, batch.boundary_pairs)
        components["boundary"] = value
        coverage["boundary"] = count
        extra["boundary/active_pairs"] = float(active)
        extra["boundary/mean_pair_distance"] = mean_distance

    total = mx.array(0.0, dtype=mx.float32)
    for name in COMPONENT_NAMES:  # fixed order: the equality must be exact
        if name in components:
            total = total + config.weights.get(name, 1.0) * components[name]
    return total, components, coverage, extra


def finalise(
    total: mx.array,
    components: Mapping[str, mx.array],
    coverage: Mapping[str, int],
    config: ObjectiveConfig,
    extra: Mapping[str, float] | None = None,
) -> ObjectiveResult:
    """Materialise, check finiteness, and verify the total against its parts."""
    mx.eval(total, *components.values())
    raw = {name: float(value.item()) for name, value in components.items()}
    weighted = {name: config.weights.get(name, 1.0) * value for name, value in raw.items()}
    total_value = float(total.item())

    for name, value in raw.items():
        if not math.isfinite(value):
            raise NonFiniteLoss(f"loss component {name!r} is {value}; the run stops")
    if not math.isfinite(total_value):
        raise NonFiniteLoss(f"total loss is {total_value}; the run stops")

    recomputed = 0.0
    for name in COMPONENT_NAMES:
        if name in weighted:
            recomputed += weighted[name]
    if abs(recomputed - total_value) > 1e-4 * max(1.0, abs(total_value)):
        raise ContractViolation(
            f"reported total {total_value} does not equal the weighted sum of its "
            f"components {recomputed}"
        )

    metrics = {f"loss/{name}": value for name, value in raw.items()}
    metrics["loss/total"] = total_value
    negative = sorted(name for name, value in raw.items() if value < 0.0)
    metrics["loss/nonnegativity_precondition_holds"] = float(not negative)
    metrics["loss/negative_components"] = float(len(negative))
    for name, count in coverage.items():
        metrics[f"coverage/{name}"] = float(count)
    for name, value in (extra or {}).items():
        metrics[name] = float(value)
    return ObjectiveResult(
        total=total,
        components=raw,
        weighted=weighted,
        coverage=dict(coverage),
        metrics=metrics,
    )
