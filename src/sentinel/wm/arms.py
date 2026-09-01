"""F. The six Stage-1A-1 arms, defined so their differences are attributable.

Each ablation removes exactly one thing and holds the parameter count fixed,
because an arm with fewer parameters would confound "this component matters"
with "this model is smaller".

* `no_action` keeps the action embedding and every parameter in it, and drives it
  with a constant index. The module is present and carries no information.
* `no_recurrence` keeps the same recurrent module and the same parameters, and
  runs it one step at a time from a zero state. Prior belief is unreachable; the
  budget is untouched.
* `shuffled_action` permutes actions *within* a trajectory. That preserves the
  action marginal exactly and destroys the alignment between an action and its
  outcome, which is the thing being ablated. Shuffling across trajectories would
  also change the marginal per sequence and confound the two.

The continuous arm supplies all three ablations, so action conditioning and
recurrence are attributable there without spending three more cells per
representation. If the only planning-positive arm turns out to be discrete or
hybrid, its own ablations have to be run before it can be credited -- a
representation whose action and recurrence contribution was never measured has
not earned a claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from sentinel.wm.latent_contract import ContractViolation, RepresentationKind
from sentinel.wm.versioning import digest_of


class ActionMode(str, Enum):
    CONDITIONED = "conditioned"
    MASKED = "masked"
    SHUFFLED = "shuffled"


@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    representation: RepresentationKind
    action_mode: ActionMode
    recurrent: bool

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "representation": self.representation.value,
            "action_mode": self.action_mode.value,
            "recurrent": self.recurrent,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("continuous_action_recurrent", RepresentationKind.CONTINUOUS, ActionMode.CONDITIONED, True),
    ArmSpec("continuous_no_action_recurrent", RepresentationKind.CONTINUOUS, ActionMode.MASKED, True),
    ArmSpec("continuous_shuffled_action_recurrent", RepresentationKind.CONTINUOUS, ActionMode.SHUFFLED, True),
    ArmSpec("continuous_action_no_recurrence", RepresentationKind.CONTINUOUS, ActionMode.CONDITIONED, False),
    ArmSpec("discrete_action_recurrent", RepresentationKind.DISCRETE, ActionMode.CONDITIONED, True),
    ArmSpec("hybrid_action_recurrent", RepresentationKind.HYBRID, ActionMode.CONDITIONED, True),
)

ABLATION_ARMS = frozenset(
    {
        "continuous_no_action_recurrent",
        "continuous_shuffled_action_recurrent",
        "continuous_action_no_recurrence",
    }
)

MASKED_ACTION_INDEX = 0
"""The constant the masked arm's action embedding is driven with.

A constant index rather than a zeroed vector, so the embedding module runs
normally and its parameters stay in the graph -- zeroing the output would also
remove its gradient and quietly shrink the effective model."""


def apply_action_mode(
    actions: np.ndarray, mode: ActionMode, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transform the action array for an arm, and report what was done.

    Returns the marginal before and after so that `shuffled` can be checked to
    have preserved it rather than merely claimed to.
    """
    original = np.bincount(actions.reshape(-1).astype(int), minlength=8).tolist()
    if mode is ActionMode.CONDITIONED:
        transformed = actions
    elif mode is ActionMode.MASKED:
        transformed = np.full_like(actions, MASKED_ACTION_INDEX)
    elif mode is ActionMode.SHUFFLED:
        generator = np.random.default_rng(seed)
        transformed = actions.copy()
        for row in range(transformed.shape[0]):
            transformed[row] = generator.permutation(transformed[row])
    else:  # pragma: no cover - enum is closed
        raise ContractViolation(f"unknown action mode {mode}")

    after = np.bincount(transformed.reshape(-1).astype(int), minlength=8).tolist()
    return transformed, {
        "mode": mode.value,
        "marginal_before": original,
        "marginal_after": after,
        "marginal_preserved": original == after,
        "alignment_destroyed": mode is ActionMode.SHUFFLED,
        "information_removed": mode is ActionMode.MASKED,
    }


def assert_shuffle_is_trajectory_safe(before: np.ndarray, after: np.ndarray) -> None:
    """A shuffle must permute within rows, never across them.

    Across-row shuffling would move an action from one trajectory into another,
    which changes each sequence's marginal and leaks nothing useful -- it makes
    the control weaker and harder to interpret at the same time.
    """
    if before.shape != after.shape:
        raise ContractViolation("the shuffle changed the action array shape")
    for row in range(before.shape[0]):
        if sorted(before[row].tolist()) != sorted(after[row].tolist()):
            raise ContractViolation(
                f"row {row} does not contain the same multiset of actions after shuffling; "
                "the permutation crossed a trajectory boundary"
            )
