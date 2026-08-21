"""A ranker that scores hypotheses by BEHAVIOUR, not by domain labels.

The narrowness in this architecture is not where it looked. The verifier,
the version space and the planner are already domain agnostic -- refutation
on `domains/dials.py`, a world with no agent and no space, isolates the true
rule set in 100% of episodes with no code changed. What is narrow is the
learned part: `TinyRecursiveCore` predicts eight heads that are *names of
grid mechanics*. `charge_period` means nothing to a dial, so the core cannot
be carried anywhere, and a system whose learned component only works on the
dataset it was written for is a narrow tool however fast it runs.

The fix is to stop predicting labels. A hypothesis can be represented by
what it DOES -- roll it forward under a fixed probe sequence and encode the
frames it predicts -- and then the question becomes "does this behaviour
look like a plausible continuation of that evidence", which is a question
with the same shape in every domain.

Two properties follow, and both are the point:

- **No label vocabulary.** Nothing here enumerates mechanics, so a new
  domain needs no new heads, no new classes, and no retraining of a
  classifier that has never heard of its rules.
- **Counterfactual discrimination.** Survivors of refutation agree on
  everything already observed; that is what surviving means. They can only
  be told apart by what they predict for actions not yet taken, which is
  exactly what a rollout signature captures and a label never did.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sentinel.env.types import Action

from .encoding import CHANNELS, CROP, MAX_TRANSITIONS, encode_history
from .model import CoreConfig, TransitionEncoder, _sinusoidal

PROBE_ACTIONS: tuple[int, ...] = (1, 2, 3, 4, 5, 1, 2, 3)
"""A fixed action sequence every hypothesis is rolled forward under.

Fixed rather than chosen, so two hypotheses' signatures are directly
comparable, and covering every action so that a rule about any one of them
shows up somewhere in the rollout."""


def rollout_signature(model, state, crop_box, field_size: int,
                      probe: tuple[int, ...] = PROBE_ACTIONS) -> np.ndarray:
    """What a hypothesis PREDICTS, encoded like evidence.

    Deliberately the same shape and the same encoder as a real episode, so
    the ranker compares like with like and nothing in it knows whether the
    frames came from the world or from a model's imagination.
    """
    x0, y0 = crop_box
    frames = np.zeros((len(probe), CROP, CROP, CHANNELS), dtype=np.int8)
    actions = np.zeros((len(probe),), dtype=np.int32)

    current = state
    for i, aid in enumerate(probe):
        try:
            nxt = model.transition(current, Action(aid))
            before = model.render(current)
            after = model.render(nxt)
        except Exception:
            break
        b = np.array([row[x0:x0 + CROP] for row in before[y0:y0 + CROP]], dtype=np.int8)
        a = np.array([row[x0:x0 + CROP] for row in after[y0:y0 + CROP]], dtype=np.int8)
        if b.shape != (CROP, CROP) or a.shape != (CROP, CROP):
            pb = np.zeros((CROP, CROP), dtype=np.int8); pb[:b.shape[0], :b.shape[1]] = b; b = pb
            pa = np.zeros((CROP, CROP), dtype=np.int8); pa[:a.shape[0], :a.shape[1]] = a; a = pa
        frames[i, :, :, 0] = b
        frames[i, :, :, 1] = a
        frames[i, :, :, 2] = (b != a).astype(np.int8)
        actions[i] = aid
        current = nxt
    return frames, actions


class UniversalRanker(nn.Module):
    """Scores (evidence, predicted-behaviour) pairs. No label vocabulary.

    One shared encoder embeds both the observed episode and the hypothesis's
    rollout, which forces the two into the same space -- the comparison is
    then a learned similarity rather than a lookup of domain-specific
    classes.
    """

    def __init__(self, cfg: CoreConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CoreConfig()
        c = self.cfg
        self.encoder = TransitionEncoder(c)
        self.pos = _sinusoidal(MAX_TRANSITIONS, c.d_model)
        self.norm = nn.LayerNorm(c.d_model)
        self.score = nn.Sequential(
            nn.Linear(c.d_model * 3, c.d_model),
            nn.GELU(),
            nn.Linear(c.d_model, 1),
        )

    def embed(self, grids: mx.array, actions: mx.array) -> mx.array:
        n = grids.shape[1]
        tokens = self.encoder(grids, actions) + self.pos[:, :n]
        mask = (actions >= 0).astype(mx.float32)[..., None]
        pooled = (tokens * mask).sum(axis=1) / mx.maximum(mask.sum(axis=1), 1.0)
        return self.norm(pooled)

    def __call__(self, ev_grids, ev_actions, hy_grids, hy_actions) -> mx.array:
        e = self.embed(ev_grids, ev_actions)
        h = self.embed(hy_grids, hy_actions)
        joined = mx.concatenate([e, h, e * h], axis=-1)
        return self.score(joined)[:, 0]


def ranking_loss(model: UniversalRanker, ev_g, ev_a, hy_g, hy_a, labels) -> mx.array:
    """Softmax cross-entropy over a candidate set: the truth should win.

    Trained as a ranking problem rather than a classification, because
    ranking is what the system actually does with it -- choose among the
    hypotheses refutation left standing.
    """
    scores = model(ev_g, ev_a, hy_g, hy_a)
    return mx.mean(nn.losses.cross_entropy(scores[None, :], labels[None]))
