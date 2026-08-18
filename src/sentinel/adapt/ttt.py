"""Test-time training: change the weights during the task, using no labels.

Claim 3 of the plan says a system with frozen weights cannot be general.
This module is that claim in its smallest testable form, and it is built so
the claim can *fail visibly* rather than be assumed.

The loop, for one unfamiliar world:

    1. Act briefly and record what happened.
    2. Sample K rule sets from the core's head distributions.
    3. Build the world model each one implies and replay the episode
       through it. The verifier says how well each explained reality.
    4. Push probability mass toward the ones that explained it.

No ground truth is consulted anywhere. The verifier is not an oracle about
the world's true rules -- it only reports consistency with evidence already
collected -- which is exactly the signal a system in a new environment
actually has.

**Two mechanisms, deliberately separable.** Sampling K hypotheses and
keeping the verifier's favourite is *search*, and it changes no weights.
Updating the core toward what scored well is *learning*. Search alone will
improve results, so folding the two together would let a gradient step take
credit for a better sample. `select` and `adapt` are therefore separate
entry points and the benchmark reports them as separate conditions.

**Adaptation is reverted by default.** Weights changed for one world are
not obviously good for the next, and Phase 5 -- not this module -- is where
consolidation is argued for. `adapt` snapshots and restores unless the
caller explicitly keeps the result.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from sentinel.core.encoding import HEADS, encode_history
from sentinel.core.model import TinyRecursiveCore
from sentinel.env.history import History
from sentinel.gen.spec import LevelSpec, Mechanics

from .hypothesis import ScoredHypothesis, mechanics_from_classes, score_hypothesis


@dataclass
class AdaptConfig:
    """Knobs for one episode of test-time adaptation."""

    samples: int = 12
    """Hypotheses drawn per step. Each costs one verifier replay, not one
    environment action, so this is cheap in the currency the benchmark
    charges for."""

    steps: int = 3
    """Gradient steps. Kept small: the evidence is a single short episode
    and there is nothing here to prevent overfitting it."""

    learning_rate: float = 3e-4
    """An order of magnitude below training. These updates run on one
    episode with no validation set; large steps would simply destroy a
    core that took hours to train."""

    temperature: float = 1.0
    """Softmax temperature for sampling. Above 1 broadens the search."""

    keep: bool = False
    """Retain adapted weights after the episode. Off by default -- see the
    module docstring."""

    seed: int = 0


@dataclass
class AdaptResult:
    """What one episode of adaptation did, in enough detail to audit it."""

    best: ScoredHypothesis
    argmax: ScoredHypothesis
    scored: list[ScoredHypothesis] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    weights_changed: bool = False

    @property
    def improved_over_argmax(self) -> bool:
        return self.best.fitness > self.argmax.fitness

    @property
    def changed_decision(self) -> bool:
        return self.best.classes != self.argmax.classes

    def summary(self) -> str:
        mark = "=" if not self.changed_decision else "->"
        return (
            f"argmax {self.argmax.mechanics.summary()} (fit {self.argmax.fitness:.3f}) "
            f"{mark} best {self.best.mechanics.summary()} (fit {self.best.fitness:.3f})"
        )


def head_distributions(
    core: TinyRecursiveCore, history: History, temperature: float = 1.0
) -> list[np.ndarray]:
    """Per-head probability vectors for one episode."""
    grids, actions = encode_history(history)
    logits = core(
        mx.array(grids[None].astype(np.int32)), mx.array(actions[None].astype(np.int32))
    )
    mx.eval(logits)
    out = []
    for head in logits:
        z = np.array(head)[0].astype(np.float64) / max(1e-6, temperature)
        z = z - z.max()
        p = np.exp(z)
        out.append(p / p.sum())
    return out


def _sample_classes(
    probs: list[np.ndarray], n: int, rng: np.random.Generator
) -> list[tuple[int, ...]]:
    """Draw rule sets, always including the core's own best guess.

    Including argmax matters: it guarantees selection can never do worse
    than not searching at all, which keeps the ablation honest.
    """
    seen: list[tuple[int, ...]] = [tuple(int(np.argmax(p)) for p in probs)]
    for _ in range(max(0, n - 1)):
        draw = tuple(int(rng.choice(len(p), p=p)) for p in probs)
        if draw not in seen:
            seen.append(draw)
    return seen


def select(
    core: TinyRecursiveCore,
    history: History,
    observed: LevelSpec,
    field_size: int,
    config: AdaptConfig | None = None,
) -> AdaptResult:
    """Search only: sample rule sets, keep whichever explains the episode.

    Changes no weights. This is the control condition that any claimed
    benefit from `adapt` has to beat.
    """
    cfg = config or AdaptConfig()
    rng = np.random.default_rng(cfg.seed)
    probs = head_distributions(core, history, cfg.temperature)
    candidates = _sample_classes(probs, cfg.samples, rng)

    scored = [score_hypothesis(c, history, observed, field_size) for c in candidates]
    argmax = scored[0]
    best = max(scored, key=lambda s: (s.fitness, s.explained_prefix))
    return AdaptResult(best=best, argmax=argmax, scored=scored)


def _policy_loss(
    model: TinyRecursiveCore,
    grids: mx.array,
    actions: mx.array,
    chosen: mx.array,
    advantage: mx.array,
) -> mx.array:
    """REINFORCE with a mean baseline, averaged over heads.

    Verifier fitness is not differentiable -- it comes from executing a
    Python program -- so the gradient has to come through the *sampling*
    distribution rather than through the score. Subtracting the batch mean
    is what stops every hypothesis being reinforced merely for having
    positive fitness.
    """
    logits = model(grids, actions)
    total = mx.zeros(())
    for i, head in enumerate(logits):
        logp = head - mx.logsumexp(head, axis=-1, keepdims=True)
        picked = mx.take_along_axis(logp, chosen[:, i : i + 1], axis=-1)[:, 0]
        total = total + mx.mean(-advantage * picked)
    return total / len(logits)


@contextmanager
def frozen(core: TinyRecursiveCore):
    """Restore the core's weights on exit. Test-time updates are per-episode."""
    snapshot = tree_map(lambda a: mx.array(a) if isinstance(a, mx.array) else a, core.parameters())
    try:
        yield core
    finally:
        core.update(snapshot)
        mx.eval(core.parameters())


def adapt(
    core: TinyRecursiveCore,
    history: History,
    observed: LevelSpec,
    field_size: int,
    config: AdaptConfig | None = None,
) -> AdaptResult:
    """Take gradient steps on one episode, against verifier signal only.

    Returns the best hypothesis found across all steps. If `config.keep` is
    false the weights are restored before returning, so the caller gets the
    benefit of the search without the core silently drifting between worlds.
    """
    cfg = config or AdaptConfig()
    rng = np.random.default_rng(cfg.seed)
    grids_np, actions_np = encode_history(history)

    optimizer = optim.Adam(learning_rate=cfg.learning_rate)
    grad_fn = nn.value_and_grad(core, _policy_loss)

    snapshot = tree_map(lambda a: mx.array(a) if isinstance(a, mx.array) else a, core.parameters())
    all_scored: list[ScoredHypothesis] = []
    losses: list[float] = []
    argmax: ScoredHypothesis | None = None

    for step in range(max(1, cfg.steps)):
        probs = head_distributions(core, history, cfg.temperature)
        candidates = _sample_classes(probs, cfg.samples, rng)
        scored = [score_hypothesis(c, history, observed, field_size) for c in candidates]
        if argmax is None:
            argmax = scored[0]
        all_scored.extend(scored)

        fitness = np.array([s.fitness for s in scored], dtype=np.float32)
        spread = float(fitness.std())
        if spread < 1e-6:
            # Every hypothesis explained the evidence equally well. There is
            # nothing to learn from this episode, and a gradient step here
            # would be noise dressed as progress.
            break

        advantage = (fitness - fitness.mean()) / (spread + 1e-6)
        chosen = mx.array(np.array([s.classes for s in scored], dtype=np.int32))
        batch_grids = mx.array(np.repeat(grids_np[None], len(scored), axis=0).astype(np.int32))
        batch_actions = mx.array(
            np.repeat(actions_np[None], len(scored), axis=0).astype(np.int32)
        )

        loss, grads = grad_fn(core, batch_grids, batch_actions, chosen, mx.array(advantage))
        optimizer.update(core, grads)
        mx.eval(core.parameters(), optimizer.state)
        losses.append(float(loss))

    assert argmax is not None
    best = max(all_scored, key=lambda s: (s.fitness, s.explained_prefix))
    if not cfg.keep:
        core.update(snapshot)
        mx.eval(core.parameters())

    return AdaptResult(
        best=best,
        argmax=argmax,
        scored=all_scored,
        losses=losses,
        weights_changed=cfg.keep and bool(losses),
    )


def adapted_mechanics(result: AdaptResult) -> Mechanics:
    """The rule set to actually plan with."""
    return result.best.mechanics
