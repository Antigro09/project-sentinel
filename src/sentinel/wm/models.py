"""The three representation arms, built in MLX behind one forward interface.

Everything that differs between continuous, discrete, and hybrid is confined to
the projection from encoder features to the latent interface. Belief, dynamics,
and every head are shared and identically shaped, so a difference measured later
is a difference in representation rather than a difference in how much model
each arm was given.

The latent interface has the same width in all three arms. The discrete arm
reads its straight-through one-hot codes back out to that width, and the hybrid
arm splits it in half. Without that, "equal parameters" would still leave the
arms feeding different amounts of state into the same recurrence.

Sampling is reparameterised for the continuous part and straight-through for the
categorical part, so both arms have a gradient path. Neither choice is claimed
to be the right one; they are the standard ones, frozen for Scale 0 so that the
throughput number is not measuring a sampling experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from sentinel.wm.latent_contract import RepresentationKind
from sentinel.wm.sizing import (
    EVENT_COUNT,
    REWARD_OUTPUTS,
    UNCERTAINTY_OUTPUTS,
    WorldModelConfig,
    count_parameters,
)
from sentinel.wm.versioning import digest_of


def straight_through_codes(logits: mx.array, groups: int, categories: int) -> mx.array:
    """One-hot forward, softmax gradient backward.

    The one-hot is built by broadcasting an equality against the argmax rather
    than by scattering into a zero tensor. A scatter would be the obvious way to
    write it and MLX cannot take a VJP through scatter indices, so the obvious
    version silently makes the whole discrete arm untrainable.
    """
    shape = logits.shape[:-1] + (groups, categories)
    grouped = logits.reshape(shape)
    probabilities = mx.softmax(grouped, axis=-1)
    index = mx.argmax(grouped, axis=-1, keepdims=True)
    hard = (mx.arange(categories) == index).astype(probabilities.dtype)
    codes = hard + probabilities - mx.stop_gradient(probabilities)
    return codes.reshape(logits.shape[:-1] + (groups * categories,))


@dataclass(frozen=True, slots=True)
class ForwardOutput:
    """Everything the objective needs from one forward pass over a batch."""

    latent: mx.array           # (B, T, W)   projected observation latent
    belief: mx.array           # (B, T, D)   recurrent deterministic state
    core: mx.array             # (B, T, D)   action-conditioned dynamics state
    next_latent: mx.array      # (B, T, W)   predicted successor latent
    event_logits: mx.array     # (B, T, E)
    reward: mx.array           # (B, T, 2)   mean and log-variance
    termination: mx.array      # (B, T, 1)   logit
    uncertainty: mx.array      # (B, T, 3)   raw aleatoric/epistemic/inadequacy heads
    inverse_logits: mx.array   # (B, T-1, A)
    consistency: mx.array      # (B, T, W)
    boundary: mx.array         # (B, T, W)   metric space for the boundary margin
    code_logits: mx.array | None  # (B, T, G*C) when the arm has codes
    used_global_rng: bool = False
    """True when sampling fell back to the global stream.

    The global MLX random stream is process state that no checkpoint records, so
    a run that draws from it cannot be restarted bit-for-bit. Unit tests may use
    it for convenience; the restart audit requires every matrix-shaped run to
    have threaded an explicit key instead."""


class ActionEmbedding(nn.Module):
    """Action lookup implemented as a one-hot matmul rather than a gather.

    `nn.Embedding`'s backward is a scatter-add. In this model the action vector
    is consumed once per dynamics block and again for every imagined step of the
    multi-step term, so a handful of action indices accumulate thousands of
    gradient contributions into four rows, and the atomic ordering of that
    accumulation is not fixed. The result is a gradient that differs run to run
    in the low bf16 bits while the loss -- a float32 reduction -- prints
    identical, which is how the restart gate failed with everything apparently
    matching.

    With four actions the one-hot matrix is negligible, the parameter shape is
    unchanged, and the backward becomes a matmul, which is deterministic.
    """

    def __init__(self, count: int, dimension: int):
        super().__init__()
        scale = 1.0 / (dimension**0.5)
        self.weight = mx.random.uniform(-scale, scale, (count, dimension))
        self._count = count

    def __call__(self, indices: mx.array) -> mx.array:
        one_hot = (mx.arange(self._count) == indices[..., None]).astype(self.weight.dtype)
        return one_hot @ self.weight


class SHWMModel(nn.Module):
    """Action-conditioned latent world model with a pluggable representation."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        width = config.latent_width
        belief = config.belief_dimension

        self.projector_in = nn.Linear(config.encoder_dimension, width)
        self.projector_norm = nn.LayerNorm(width)

        kind = config.representation
        if kind is RepresentationKind.CONTINUOUS:
            self.representation_continuous = nn.Linear(width, 2 * width)
        elif kind is RepresentationKind.DISCRETE:
            self.representation_codebook = nn.Linear(width, config.code_width)
            self.representation_readout = nn.Linear(config.code_width, width)
        else:
            self.representation_continuous = nn.Linear(width, 2 * config.continuous_width)
            self.representation_codebook = nn.Linear(width, config.code_width)
            self.representation_readout = nn.Linear(config.code_width, config.discrete_width)

        self.action_embedding = ActionEmbedding(config.action_count, config.action_embedding)
        self.belief_gru = nn.GRU(width + config.action_embedding + 1, belief)

        self.core_norms = [nn.LayerNorm(belief) for _ in range(config.core_depth)]
        self.core_in = [
            nn.Linear(belief + config.action_embedding, config.core_width)
            for _ in range(config.core_depth)
        ]
        self.core_out = [nn.Linear(config.core_width, belief) for _ in range(config.core_depth)]

        self.head_next_latent = nn.Linear(belief, width)
        self.head_event = nn.Linear(belief, config.event_count)
        self.head_reward = nn.Linear(belief, REWARD_OUTPUTS)
        self.head_termination = nn.Linear(belief, 1)
        self.head_uncertainty = nn.Linear(belief, UNCERTAINTY_OUTPUTS)
        self.head_inverse_action = nn.Linear(2 * belief, config.action_count)
        self.head_consistency = nn.Linear(belief, width)
        self.head_boundary = nn.Linear(width, width)

    # ---- representation ------------------------------------------------

    def _sample(self, mean: mx.array, log_variance: mx.array, key: mx.array | None) -> mx.array:
        """Reparameterised draw, from an explicit key when one is supplied."""
        scale = mx.exp(0.5 * mx.clip(log_variance, -8.0, 8.0))
        if key is None:
            noise = mx.random.normal(mean.shape, dtype=mean.dtype)
        else:
            noise = mx.random.normal(mean.shape, dtype=mean.dtype, key=key)
        return mean + scale * noise

    def project(
        self, features: mx.array, key: mx.array | None = None
    ) -> tuple[mx.array, mx.array | None]:
        """Encoder features to the latent interface, in this arm's parameterisation."""
        hidden = self.projector_norm(self.projector_in(features))
        kind = self.config.representation
        if kind is RepresentationKind.CONTINUOUS:
            statistics = self.representation_continuous(hidden)
            mean, log_variance = mx.split(statistics, 2, axis=-1)
            return self._sample(mean, log_variance, key), None
        if kind is RepresentationKind.DISCRETE:
            logits = self.representation_codebook(hidden)
            codes = straight_through_codes(
                logits, self.config.code_groups, self.config.code_categories
            )
            return self.representation_readout(codes), logits
        statistics = self.representation_continuous(hidden)
        mean, log_variance = mx.split(statistics, 2, axis=-1)
        continuous = self._sample(mean, log_variance, key)
        logits = self.representation_codebook(hidden)
        codes = straight_through_codes(
            logits, self.config.code_groups, self.config.code_categories
        )
        return mx.concatenate([continuous, self.representation_readout(codes)], axis=-1), logits

    # ---- dynamics --------------------------------------------------------

    def dynamics(self, belief: mx.array, action_vector: mx.array) -> mx.array:
        """Residual action-conditioned core. The action enters every block."""
        state = belief
        for norm, project_in, project_out in zip(self.core_norms, self.core_in, self.core_out):
            normalised = norm(state)
            hidden = nn.gelu(project_in(mx.concatenate([normalised, action_vector], axis=-1)))
            state = state + project_out(hidden)
        return state

    def __call__(
        self,
        features: mx.array,       # (B, T, E)
        actions: mx.array,        # (B, T)     action taken *at* step t
        previous_rewards: mx.array,  # (B, T, 1)
        key: mx.array | None = None,
        recurrent: bool = True,
    ) -> ForwardOutput:
        latent, code_logits = self.project(features, key)
        action_vectors = self.action_embedding(actions)

        # The belief update sees the previous action and reward, never the
        # current one: a belief that already knows the action it is about to
        # take cannot be used to score that action.
        previous_actions = mx.concatenate(
            [mx.zeros_like(action_vectors[:, :1]), action_vectors[:, :-1]], axis=1
        )
        belief_input = mx.concatenate([latent, previous_actions, previous_rewards], axis=-1)
        if recurrent:
            belief = self.belief_gru(belief_input)
        else:
            # The same module and the same parameters, run one step at a time from
            # a zero state. Prior belief is unreachable and the budget is
            # untouched -- a smaller network here would confound "recurrence
            # matters" with "this model is smaller".
            steps = [
                self.belief_gru(belief_input[:, t : t + 1])
                for t in range(belief_input.shape[1])
            ]
            belief = mx.concatenate(steps, axis=1)

        core = self.dynamics(belief, action_vectors)

        inverse_input = mx.concatenate([belief[:, :-1], belief[:, 1:]], axis=-1)
        return ForwardOutput(
            latent=latent,
            belief=belief,
            core=core,
            next_latent=self.head_next_latent(core),
            event_logits=self.head_event(core),
            reward=self.head_reward(core),
            termination=self.head_termination(core),
            uncertainty=self.head_uncertainty(core),
            inverse_logits=self.head_inverse_action(inverse_input),
            consistency=self.head_consistency(core),
            boundary=self.head_boundary(latent),
            code_logits=code_logits,
            used_global_rng=key is None
            and self.config.representation is not RepresentationKind.DISCRETE,
        )

    # ---- accounting -------------------------------------------------------

    def actual_trainable_parameters(self) -> int:
        """The authoritative count: every trainable leaf, summed."""
        return int(sum(v.size for _, v in tree_flatten(self.trainable_parameters())))

    def actual_parameter_bytes(self) -> int:
        return int(sum(v.nbytes for _, v in tree_flatten(self.parameters())))

    def parameter_report(self) -> dict[str, Any]:
        counted = self.actual_trainable_parameters()
        predicted = count_parameters(self.config)
        return {
            "config": self.config.canonical_dict(),
            "trainable_parameters": counted,
            "closed_form_parameters": predicted,
            "closed_form_agrees": counted == predicted,
            "parameter_bytes": self.actual_parameter_bytes(),
            "tensors": len(tree_flatten(self.trainable_parameters())),
        }

    @property
    def version(self) -> str:
        """Identity of the architecture, not of the weights."""
        return digest_of(self.config.canonical_dict())


def build_model(config: WorldModelConfig, *, seed: int, dtype: mx.Dtype = mx.bfloat16) -> SHWMModel:
    """Construct and initialise. The seed controls initialisation only."""
    mx.random.seed(seed)
    model = SHWMModel(config)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model
