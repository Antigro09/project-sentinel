"""The gate every external action passes through.

`ARCHITECTURE.md` requires that all externally consequential actions cross the
verifier bridge or an explicit authority gate. A test that merely *checks* this
can only observe the paths it thought to look at, so the property is enforced
structurally instead: `EnvironmentAdapter.step` will not accept a bare action.
It requires an `AuthorizationToken`, and tokens are minted only here.

Three authorities exist and they are not interchangeable. `COLLECTION` covers
data gathering under a declared collector policy. `VERIFIED_PLAN` covers a model
proposal that has already been through the verifier bridge, and minting one
requires the evaluator's required probes to have been satisfied -- a model
cannot authorise itself by asking for a cheaper probe set. `EVALUATOR` covers
replay and probing by the evaluator, which is allowed to see what the model may
not.

Tokens are single use. Replaying one is the same class of mistake as counting an
interaction twice, and the run matrix budgets interactions exactly.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from sentinel.wm.versioning import digest_of


class AuthorityDenied(RuntimeError):
    """An action was proposed that the gate will not authorise."""


class UnauthorizedAction(RuntimeError):
    """An action reached the environment without a valid token."""


class ActionAuthority(str, Enum):
    COLLECTION = "collection"
    VERIFIED_PLAN = "verified_plan"
    EVALUATOR = "evaluator"


@dataclass(frozen=True, slots=True)
class AuthorizationToken:
    """Permission for exactly one action, once."""

    authority: ActionAuthority
    action: int
    nonce: int
    gate_id: str
    justification_digest: str

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "authority": self.authority.value,
            "action": int(self.action),
            "nonce": int(self.nonce),
            "gate_id": self.gate_id,
            "justification_digest": self.justification_digest,
        }


@dataclass
class AuthorityGate:
    """Mints and retires action tokens, and keeps the count.

    The ledger is not decoration: `online_interactions` is one of the eight
    quantities the matrix matches, and it has to come from the object that
    actually let the actions through.
    """

    gate_id: str = "scale-0-gate"
    required_probes: tuple[str, ...] = ()
    _counter: itertools.count = field(default_factory=itertools.count, init=False, repr=False)
    _outstanding: dict[int, AuthorizationToken] = field(default_factory=dict, init=False, repr=False)
    issued: int = field(default=0, init=False)
    consumed: int = field(default=0, init=False)
    denied: int = field(default=0, init=False)
    by_authority: dict[str, int] = field(default_factory=dict, init=False)

    def _mint(self, authority: ActionAuthority, action: int, justification: Mapping[str, Any]) -> AuthorizationToken:
        token = AuthorizationToken(
            authority=authority,
            action=int(action),
            nonce=next(self._counter),
            gate_id=self.gate_id,
            justification_digest=digest_of(dict(justification)),
        )
        self._outstanding[token.nonce] = token
        self.issued += 1
        self.by_authority[authority.value] = self.by_authority.get(authority.value, 0) + 1
        return token

    def authorize_collection(self, action: int, collector_policy: str) -> AuthorizationToken:
        return self._mint(
            ActionAuthority.COLLECTION, action, {"collector_policy": collector_policy}
        )

    def authorize_evaluator(self, action: int, reason: str) -> AuthorizationToken:
        return self._mint(ActionAuthority.EVALUATOR, action, {"reason": reason})

    def authorize_plan(
        self,
        action: int,
        *,
        executed_probes: tuple[str, ...],
        verifier_version: str,
        abstained: bool = False,
    ) -> AuthorizationToken:
        """Authorise a model proposal, or refuse it.

        Refusal has two grounds, and they are different failures. An abstention
        is the controller declining to act, which is a legitimate outcome. A
        missing required probe is the model trying to act on a cheaper evidence
        set than the evaluator demands, which is the model-exploitation path the
        bridge exists to close.
        """
        if abstained:
            self.denied += 1
            raise AuthorityDenied(f"action {action} was proposed but the controller abstained")
        missing = set(self.required_probes) - set(executed_probes)
        if missing:
            self.denied += 1
            raise AuthorityDenied(
                f"action {action} was proposed without the evaluator-required probe(s) "
                f"{sorted(missing)}; model-requested probes may add coverage, never replace it"
            )
        return self._mint(
            ActionAuthority.VERIFIED_PLAN,
            action,
            {"executed_probes": sorted(executed_probes), "verifier_version": verifier_version},
        )

    def consume(self, token: AuthorizationToken, action: int) -> ActionAuthority:
        """Validate and retire a token. Called by the environment, not the caller."""
        if not isinstance(token, AuthorizationToken):
            raise UnauthorizedAction(
                f"action {action} reached the environment with "
                f"{type(token).__name__} instead of an AuthorizationToken"
            )
        if token.gate_id != self.gate_id:
            raise UnauthorizedAction(
                f"token was minted by gate {token.gate_id!r}, not {self.gate_id!r}"
            )
        if token.nonce not in self._outstanding:
            raise UnauthorizedAction(
                f"token {token.nonce} has already been consumed or was never issued; "
                "authorisation is single use"
            )
        if token.action != int(action):
            raise UnauthorizedAction(
                f"token authorises action {token.action}, but action {action} was taken"
            )
        del self._outstanding[token.nonce]
        self.consumed += 1
        return token.authority

    def ledger(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "issued": self.issued,
            "consumed": self.consumed,
            "denied": self.denied,
            "outstanding": len(self._outstanding),
            "by_authority": dict(sorted(self.by_authority.items())),
            "required_probes": list(self.required_probes),
        }
