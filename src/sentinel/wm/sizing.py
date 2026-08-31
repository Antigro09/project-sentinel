"""Parameter accounting and the 50M/200M configuration solver.

The run matrix says a workload may be called 50M or 200M only when its *actual*
trainable tensor count lands within 1% of the target, and that nominal hidden
width is not a parameter count. So the number that decides is always the one
counted off the built model; the closed forms here exist only to make the search
cheap, and a test asserts they agree with the real count exactly, per arm and
per width.

The solver has two knobs and uses them in order. `belief_dimension` is coarse --
the count grows roughly quadratically in it -- and gets a binary search.
`core_width` is fine, worth a few thousand parameters per unit, and closes the
remaining gap. There is deliberately no third knob: an exactly-matching count
would need a ballast tensor that does no work, and a parameter that does nothing
is not a matched budget, it is a padded one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from sentinel.wm.latent_contract import ContractViolation, RepresentationKind
from sentinel.wm.versioning import digest_of

EVENT_COUNT = 12
UNCERTAINTY_OUTPUTS = 3
REWARD_OUTPUTS = 2


@dataclass(frozen=True, slots=True)
class WorldModelConfig:
    """Every shape that determines a trainable tensor."""

    representation: RepresentationKind
    encoder_dimension: int
    latent_width: int
    belief_dimension: int
    core_width: int
    core_depth: int
    action_count: int
    action_embedding: int = 32
    code_groups: int = 32
    code_categories: int = 32
    event_count: int = EVENT_COUNT

    def __post_init__(self) -> None:
        for name in ("encoder_dimension", "latent_width", "belief_dimension", "core_width"):
            if getattr(self, name) <= 0:
                raise ContractViolation(f"{name} must be positive")
        if self.core_depth <= 0:
            raise ContractViolation("core_depth must be positive")
        if self.representation is not RepresentationKind.CONTINUOUS:
            if self.code_groups <= 0 or self.code_categories <= 1:
                raise ContractViolation(
                    "a categorical arm needs at least one group of at least two categories"
                )
        if self.representation is RepresentationKind.HYBRID and self.latent_width % 2:
            raise ContractViolation("hybrid arm splits the latent width in two; it must be even")

    @property
    def continuous_width(self) -> int:
        """How much of the latent interface the continuous part occupies."""
        if self.representation is RepresentationKind.CONTINUOUS:
            return self.latent_width
        if self.representation is RepresentationKind.DISCRETE:
            return 0
        return self.latent_width // 2

    @property
    def discrete_width(self) -> int:
        return self.latent_width - self.continuous_width

    @property
    def code_width(self) -> int:
        return self.code_groups * self.code_categories

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation.value,
            "encoder_dimension": self.encoder_dimension,
            "latent_width": self.latent_width,
            "belief_dimension": self.belief_dimension,
            "core_width": self.core_width,
            "core_depth": self.core_depth,
            "action_count": self.action_count,
            "action_embedding": self.action_embedding,
            "code_groups": self.code_groups,
            "code_categories": self.code_categories,
            "event_count": self.event_count,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


def _linear(inputs: int, outputs: int) -> int:
    return inputs * outputs + outputs


def _layer_norm(width: int) -> int:
    return 2 * width


def _gru(inputs: int, hidden: int) -> int:
    """MLX GRU: Wx (3H, in), Wh (3H, H), b (3H,), bhn (H,)."""
    return 3 * hidden * inputs + 3 * hidden * hidden + 3 * hidden + hidden


def parameter_breakdown(config: WorldModelConfig) -> dict[str, int]:
    """Trainable parameters per component. The sum is the closed form."""
    width = config.latent_width
    belief = config.belief_dimension
    core = config.core_width
    embed = config.action_embedding
    codes = config.code_width

    parts: dict[str, int] = {
        "projector_in": _linear(config.encoder_dimension, width),
        "projector_norm": _layer_norm(width),
    }

    if config.representation is RepresentationKind.CONTINUOUS:
        parts["representation_continuous"] = _linear(width, 2 * width)
    elif config.representation is RepresentationKind.DISCRETE:
        parts["representation_codebook"] = _linear(width, codes)
        parts["representation_readout"] = _linear(codes, width)
    else:
        parts["representation_continuous"] = _linear(width, 2 * config.continuous_width)
        parts["representation_codebook"] = _linear(width, codes)
        parts["representation_readout"] = _linear(codes, config.discrete_width)

    parts["action_embedding"] = config.action_count * embed
    parts["belief_gru"] = _gru(width + embed + 1, belief)

    block = _layer_norm(belief) + _linear(belief + embed, core) + _linear(core, belief)
    parts["dynamics_core"] = config.core_depth * block

    parts["head_next_latent"] = _linear(belief, width)
    parts["head_event"] = _linear(belief, config.event_count)
    parts["head_reward"] = _linear(belief, REWARD_OUTPUTS)
    parts["head_termination"] = _linear(belief, 1)
    parts["head_uncertainty"] = _linear(belief, UNCERTAINTY_OUTPUTS)
    parts["head_inverse_action"] = _linear(2 * belief, config.action_count)
    parts["head_consistency"] = _linear(belief, width)
    parts["head_boundary"] = _linear(width, width)
    return parts


def count_parameters(config: WorldModelConfig) -> int:
    return sum(parameter_breakdown(config).values())


@dataclass(frozen=True, slots=True)
class SizedConfig:
    config: WorldModelConfig
    target: int
    counted: int

    @property
    def drift(self) -> float:
        return self.counted / self.target - 1.0

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.canonical_dict(),
            "target": self.target,
            "counted": self.counted,
            "drift": self.drift,
            "breakdown": parameter_breakdown(self.config),
        }


def solve_config(
    representation: RepresentationKind,
    target: int,
    *,
    encoder_dimension: int,
    latent_width: int,
    action_count: int,
    core_depth: int = 4,
    tolerance: float = 0.01,
    core_multiplier: int = 4,
) -> SizedConfig:
    """Find shapes whose counted parameters land inside the tolerance.

    Raises rather than returning a near miss. A workload that cannot be built at
    the target size is a stop condition, not something to relabel.
    """
    if latent_width % 2 and representation is RepresentationKind.HYBRID:
        raise ContractViolation("hybrid arm needs an even latent width")

    def build(belief: int, core: int) -> WorldModelConfig:
        return WorldModelConfig(
            representation=representation,
            encoder_dimension=encoder_dimension,
            latent_width=latent_width,
            belief_dimension=belief,
            core_width=core,
            core_depth=core_depth,
            action_count=action_count,
        )

    # Coarse: binary search the belief dimension with a fixed core multiplier.
    low, high = 8, 8
    while count_parameters(build(high, core_multiplier * high)) < target and high < 1 << 16:
        low, high = high, high * 2
    if high >= 1 << 16:
        raise ContractViolation(f"cannot reach {target:,} parameters at width {latent_width}")
    while low + 1 < high:
        middle = (low + high) // 2
        if count_parameters(build(middle, core_multiplier * middle)) < target:
            low = middle
        else:
            high = middle
    belief = low  # largest belief dimension still under target

    # Fine: widen the core until the count crosses the target, then keep whichever
    # side is closer. Overshooting by a hair beats undershooting by a lot.
    core = core_multiplier * belief
    step = max(1, belief // 64)
    while count_parameters(build(belief, core)) < target:
        core += step
    upper = build(belief, core)
    lower = build(belief, max(1, core - step))
    while count_parameters(build(belief, max(1, core - 1))) >= target and core > 1:
        core -= 1
        upper = build(belief, core)
        lower = build(belief, max(1, core - 1))

    best = min(
        (upper, lower),
        key=lambda candidate: abs(count_parameters(candidate) - target),
    )
    counted = count_parameters(best)
    if abs(counted - target) > target * tolerance:
        raise ContractViolation(
            f"{representation.value} at width {latent_width} solved to {counted:,} "
            f"parameters, {counted / target - 1:+.3%} from target {target:,}"
        )
    return SizedConfig(config=best, target=target, counted=counted)
