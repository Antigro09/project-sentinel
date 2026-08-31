"""Environment adapters for the Phase-2 learned world model.

These are additive and separate from `sentinel.env`, which is the exact Phase-1
environment layer. Nothing here changes exact replay semantics.
"""

from sentinel.env.adapters.base import (
    EnvironmentAdapter,
    EnvironmentIdentity,
    HiddenSnapshot,
    HiddenStateLeak,
    ProbeSet,
    StepResult,
    assert_no_hidden_state,
    assert_observation_invariant_to_hidden_field,
    declared_non_observable,
)

__all__ = [
    "EnvironmentAdapter",
    "EnvironmentIdentity",
    "HiddenSnapshot",
    "HiddenStateLeak",
    "ProbeSet",
    "StepResult",
    "assert_no_hidden_state",
    "assert_observation_invariant_to_hidden_field",
    "declared_non_observable",
]
