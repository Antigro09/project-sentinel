"""The adapter contract: reset, step, probe, branch, restore.

The single hardest requirement on an adapter is the one that is easiest to
violate by accident: the simulator's hidden state must be reachable by the
evaluator and unreachable by the model. So hidden state does not live in a
dictionary that an adapter author might forget to filter. It lives in
`HiddenSnapshot`, a type whose contents can only be read through a method that
names the caller, and which the observation builder refuses to accept.

`step` takes an `AuthorizationToken`. That is what makes "every path to an
external action crosses the gate" a property of the type signature rather than
a property of the paths someone remembered to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from sentinel.wm.authority import AuthorityGate, AuthorizationToken
from sentinel.wm.latent_contract import (
    ContractViolation,
    ObservationEnvelope,
    Taint,
)
from sentinel.wm.versioning import digest_of


class HiddenStateLeak(RuntimeError):
    """Hidden simulator state was routed somewhere the model can see it."""


@dataclass(frozen=True, slots=True)
class HiddenSnapshot:
    """Complete simulator state. Evaluator-only, and it says so.

    Two consumers are legitimate: `restore`, which needs the whole state to
    reproduce a branch point, and the evaluator, which is allowed to know what
    the model does not. `reveal` requires the caller to name itself, so a leak
    shows up in a grep for one string rather than in a diff of a dict literal.
    """

    payload: Mapping[str, Any]
    environment_version: str
    taint: frozenset[Taint] = frozenset({Taint.EVALUATOR_ONLY})

    _ALLOWED_READERS = frozenset({"restore", "evaluator", "fixture-builder"})

    def reveal(self, reader: str) -> Mapping[str, Any]:
        if reader not in HiddenSnapshot._ALLOWED_READERS:
            raise HiddenStateLeak(
                f"{reader!r} may not read hidden simulator state; "
                f"permitted readers are {sorted(HiddenSnapshot._ALLOWED_READERS)}"
            )
        return dict(self.payload)

    @property
    def digest(self) -> str:
        return digest_of({"payload": dict(self.payload), "env": self.environment_version})

    def canonical_dict(self) -> dict[str, Any]:
        """Deliberately does not include the payload.

        A snapshot that serialises itself into a report is a snapshot that can
        end up in a training record.
        """
        return {
            "hidden_snapshot_digest": self.digest,
            "taint": sorted(t.value for t in self.taint),
        }


@dataclass(frozen=True, slots=True)
class ProbeSet:
    """Exact observable values at one step.

    These are what the verifier compares against. They are computed by the
    environment, never by the model, and they are the only channel through which
    a prediction can be refuted.
    """

    values: Mapping[str, Any]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.values))

    def subset(self, names: tuple[str, ...]) -> "ProbeSet":
        missing = set(names) - set(self.values)
        if missing:
            raise ContractViolation(f"environment cannot supply probe(s) {sorted(missing)}")
        return ProbeSet({n: self.values[n] for n in names})

    def canonical_dict(self) -> dict[str, Any]:
        return {k: self.values[k] for k in sorted(self.values)}

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: ObservationEnvelope
    reward: float
    terminated: bool
    probes: ProbeSet
    legal_actions: tuple[int, ...]
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnvironmentIdentity:
    name: str
    version: str
    generator_digest: str
    supports_branching: bool

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "generator_digest": self.generator_digest,
            "supports_branching": self.supports_branching,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


@runtime_checkable
class EnvironmentAdapter(Protocol):
    @property
    def identity(self) -> EnvironmentIdentity: ...

    def reset(self, seed: int, dynamic: str = "base") -> StepResult: ...

    def step(self, action: int, token: AuthorizationToken) -> StepResult: ...

    def legal_actions(self) -> tuple[int, ...]: ...

    def probes(self) -> ProbeSet: ...

    def snapshot(self) -> HiddenSnapshot: ...

    def restore(self, snapshot: HiddenSnapshot) -> StepResult: ...


def declared_non_observable(hidden: HiddenSnapshot) -> tuple[str, ...]:
    """Which snapshot fields the environment says the model must not see.

    A snapshot holds the *complete* simulator state, and much of that state is
    legitimately observable -- the agent's own position, the goal it was given.
    Treating every snapshot field as secret would flag those and make the check
    useless, so the environment declares the genuinely hidden subset. When no
    declaration is present the conservative reading applies and every field
    counts as hidden.
    """
    payload = hidden.reveal("evaluator")
    declared = payload.get("_non_observable")
    if declared is None:
        return tuple(sorted(k for k in payload if not k.startswith("_")))
    return tuple(declared)


def assert_no_hidden_state(observation: ObservationEnvelope, hidden: HiddenSnapshot) -> None:
    """No declared-hidden field name appears in the model's observation."""
    visible = set(observation.structured_observation)
    leaked = sorted(set(declared_non_observable(hidden)) & visible)
    if leaked:
        raise HiddenStateLeak(
            f"declared non-observable field(s) {leaked} appear in the observation"
        )


def assert_observation_invariant_to_hidden_field(
    adapter: "EnvironmentAdapter",
    snapshot: HiddenSnapshot,
    field_name: str,
    values: tuple[Any, ...],
) -> tuple[str, ...]:
    """The real test: vary a hidden field and require the observation not to move.

    Name matching catches a hidden value copied under its own name. It cannot
    catch one copied under an innocuous name, and value matching cannot either
    without producing false positives on small integers. Invariance can: restore
    the same state with the hidden field set to each of its values, and require
    one observation digest. If the digest moves, the field reached model input,
    whatever it was called.

    Returns the probe digests, which do *not* have to be invariant -- the
    evaluator is allowed to see the consequences of hidden state.
    """
    payload = dict(snapshot.reveal("restore"))
    if field_name not in payload:
        raise HiddenStateLeak(f"{field_name!r} is not part of this snapshot")
    observation_digests: set[str] = set()
    probe_digests: list[str] = []
    for value in values:
        variant = HiddenSnapshot(
            payload={**payload, field_name: value},
            environment_version=snapshot.environment_version,
        )
        result = adapter.restore(variant)
        observation_digests.add(result.observation.digest)
        probe_digests.append(result.probes.digest)
    if len(observation_digests) > 1:
        raise HiddenStateLeak(
            f"changing hidden field {field_name!r} across {list(values)} changed the "
            f"observation ({len(observation_digests)} distinct digests); hidden simulator "
            "state is reaching model input"
        )
    return tuple(probe_digests)


def build_gate(required_probes: tuple[str, ...]) -> AuthorityGate:
    return AuthorityGate(required_probes=required_probes)
