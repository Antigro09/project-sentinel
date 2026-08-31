"""The bridge from predicted events to exact probes, and the decision controller.

The bridge never asks the latent model whether its own prediction was correct.
It compares predicted observables against actual ones and returns the mismatches
as counterexamples.

Two probe sets exist and they are not peers. `required_probes` comes from the
evaluator and is frozen independently of anything the model produces; it is
always executed and cannot be removed, weakened, or substituted.
`requested_probes` is what the model asks for, and it can only *add* diagnostic
coverage. Collapsing the two would let a model authorise itself by asking for a
cheaper evidence set, which is the exploitation path the bridge exists to close.

Lemma 4 sets the limit honestly: an equality verifier rejects every observable
that differs, and a non-injective probe can still hide a latent mismatch behind
an identical value. So correctness and coverage are reported as two numbers, and
the second one is where a missing probe shows up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from sentinel.env.adapters.base import ProbeSet
from sentinel.wm.authority import AuthorityDenied, AuthorityGate, AuthorizationToken
from sentinel.wm.events import EventKind
from sentinel.wm.latent_contract import (
    ContractViolation,
    Counterexample,
    TransitionPrediction,
    UncertaintyTriple,
    VerificationResult,
)
from sentinel.wm.versioning import digest_of

VERIFIER_VERSION_MATERIAL = {"name": "shwm-observable-equality", "version": 1}
VERIFIER_VERSION = digest_of(VERIFIER_VERSION_MATERIAL)

EVENT_PROBE_MAP: Mapping[EventKind, str] = {
    EventKind.ACTION_SUCCEEDED: "action_succeeded",
    EventKind.ACTION_FAILED: "action_succeeded",
    EventKind.CONSTRAINT_VIOLATED: "constraint_violation",
    EventKind.GOAL_PROGRESS_CHANGED: "goal_progress",
    EventKind.OBJECT_APPEARED: "observable_signature",
    EventKind.OBJECT_DISAPPEARED: "observable_signature",
    EventKind.INVENTORY_CHANGED: "observable_signature",
    EventKind.FOCUS_MOVED: "observable_signature",
    EventKind.FILE_STATE_CHANGED: "observable_signature",
    EventKind.UI_STATE_CHANGED: "observable_signature",
}
"""Which exact probe witnesses which predicted event.

`UNKNOWN_EVENT` and `MISSING_EVENT_REPRESENTATION` are deliberately absent. The
first is an in-schema abstention and witnesses nothing; the second is a claim
that the schema is inadequate, which is a representation obligation rather than
a prediction to check.
"""


class Decision(str, Enum):
    ACT = "act"
    ASK = "ask"
    OBSERVE = "observe"
    RUN_TEST = "run_test"
    REPLAN = "replan"
    SWITCH_MODEL = "switch_model"
    ABSTAIN = "abstain"
    EXPAND_REPRESENTATION = "expand_representation"


@dataclass(frozen=True, slots=True)
class VerificationContext:
    """What the evaluator knows about the situation being verified."""

    episode_id: str
    step: int
    available_probes: tuple[str, ...]
    required_probes: tuple[str, ...]

    def __post_init__(self) -> None:
        missing = set(self.required_probes) - set(self.available_probes)
        if missing:
            raise ContractViolation(
                f"environment cannot supply evaluator-required probe(s) {sorted(missing)}"
            )


@dataclass
class ObservableVerifierBridge:
    """Exact equality on observables, with coverage reported separately."""

    required: tuple[str, ...]
    tolerance: float = 1e-6
    verifier_version: str = VERIFIER_VERSION
    verifications: int = field(default=0, init=False)
    rejections: int = field(default=0, init=False)
    counterexample_count: int = field(default=0, init=False)

    def required_probes(self, context: VerificationContext) -> tuple[str, ...]:
        """Frozen by the evaluator. Independent of anything the model produced."""
        return tuple(sorted(set(self.required) | set(context.required_probes)))

    def requested_probes(self, prediction: TransitionPrediction) -> tuple[str, ...]:
        """Additive only: what the model's predicted events would witness."""
        requested: set[str] = set()
        for name, mass in prediction.event_distribution.items():
            if mass <= 0.0:
                continue
            try:
                kind = EventKind(name)
            except ValueError:  # pragma: no cover - guarded by the schema digest
                continue
            probe = EVENT_PROBE_MAP.get(kind)
            if probe is not None:
                requested.add(probe)
        return tuple(sorted(requested))

    def probe_plan(
        self, prediction: TransitionPrediction, context: VerificationContext
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """(executed, unprobed). Required probes are always in `executed`."""
        required = self.required_probes(context)
        requested = tuple(
            p for p in self.requested_probes(prediction) if p in context.available_probes
        )
        executed = tuple(sorted(set(required) | set(requested)))
        unprobed = tuple(sorted(set(context.available_probes) - set(executed)))
        return executed, unprobed

    def _matches(self, predicted: Any, actual: Any) -> bool:
        if isinstance(predicted, bool) or isinstance(actual, bool):
            return bool(predicted) == bool(actual)
        if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
            return abs(float(predicted) - float(actual)) <= self.tolerance
        return predicted == actual

    def verify(
        self,
        prediction: TransitionPrediction,
        predicted_probes: Mapping[str, Any],
        actual: ProbeSet,
        context: VerificationContext,
    ) -> VerificationResult:
        executed, unprobed = self.probe_plan(prediction, context)
        accepted: list[str] = []
        rejected: list[str] = []
        counterexamples: list[Counterexample] = []
        violations: list[str] = []

        for name in executed:
            if name not in actual.values:
                raise ContractViolation(f"environment did not supply probe {name!r}")
            observed = actual.values[name]
            if name not in predicted_probes:
                # A required probe the model made no prediction for is still
                # executed; it counts against coverage of the model, not of the
                # verifier, and is recorded as a rejection with no prediction.
                rejected.append(name)
                counterexamples.append(
                    Counterexample(name, predicted=None, actual=observed,
                                   step=context.step, episode_id=context.episode_id)
                )
                continue
            if self._matches(predicted_probes[name], observed):
                accepted.append(name)
            else:
                rejected.append(name)
                counterexamples.append(
                    Counterexample(name, predicted=predicted_probes[name], actual=observed,
                                   step=context.step, episode_id=context.episode_id)
                )
            if name == "constraint_violation" and bool(observed):
                violations.append(name)

        self.verifications += 1
        self.rejections += len(rejected)
        self.counterexample_count += len(counterexamples)
        return VerificationResult(
            accepted_observables=tuple(accepted),
            rejected_observables=tuple(rejected),
            unprobed_observables=unprobed,
            counterexamples=tuple(counterexamples),
            constraint_violations=tuple(violations),
            verifier_version=self.verifier_version,
            required_probe_names=self.required_probes(context),
        )

    def ledger(self) -> dict[str, Any]:
        return {
            "verifier_version": self.verifier_version,
            "verifications": self.verifications,
            "rejections": self.rejections,
            "counterexamples": self.counterexample_count,
            "required_probes": list(self.required),
        }


@dataclass(frozen=True, slots=True)
class ControlThresholds:
    """Frozen per experiment, and calibrated against always-act and always-abstain.

    A threshold set with no calibration arm measures nothing: always-abstain
    scores perfectly on accuracy and zero on coverage, and always-act does the
    reverse.
    """

    epistemic_ask: float = 0.6
    inadequacy_expand: float = 0.8
    constraint_abstain: float = 0.5
    disagreement_replan: float = 0.7

    def canonical_dict(self) -> dict[str, float]:
        return {
            "epistemic_ask": self.epistemic_ask,
            "inadequacy_expand": self.inadequacy_expand,
            "constraint_abstain": self.constraint_abstain,
            "disagreement_replan": self.disagreement_replan,
        }


@dataclass
class DecisionController:
    """Chooses among act, ask, observe, test, replan, switch, abstain, expand.

    Ordering matters and is deliberate: representation inadequacy is checked
    before epistemic uncertainty, because "no model in my class explains this"
    and "I have not seen enough of this" call for different responses and the
    second reading would quietly absorb the first.
    """

    thresholds: ControlThresholds = field(default_factory=ControlThresholds)
    counts: dict[str, int] = field(default_factory=dict)

    def decide(
        self,
        uncertainty: UncertaintyTriple,
        predicted_constraint_cost: float,
        *,
        can_ask: bool = True,
        can_test: bool = True,
    ) -> Decision:
        if uncertainty.inadequacy >= self.thresholds.inadequacy_expand:
            decision = Decision.EXPAND_REPRESENTATION
        elif predicted_constraint_cost >= self.thresholds.constraint_abstain:
            decision = Decision.RUN_TEST if can_test else Decision.ABSTAIN
        elif uncertainty.epistemic >= self.thresholds.epistemic_ask:
            decision = Decision.ASK if can_ask else Decision.OBSERVE
        else:
            decision = Decision.ACT
        self.counts[decision.value] = self.counts.get(decision.value, 0) + 1
        return decision

    def ledger(self) -> dict[str, Any]:
        total = sum(self.counts.values()) or 1
        return {
            "decisions": dict(sorted(self.counts.items())),
            "act_rate": self.counts.get("act", 0) / total,
            "abstain_rate": self.counts.get("abstain", 0) / total,
            "thresholds": self.thresholds.canonical_dict(),
        }


def authorize_if_verified(
    gate: AuthorityGate,
    decision: Decision,
    action: int,
    result: VerificationResult,
) -> AuthorizationToken:
    """The only route from a plan to an external action.

    Raises `AuthorityDenied` on abstention or on a missing required probe. The
    caller cannot obtain a token any other way, and the environment will not
    step without one.
    """
    return gate.authorize_plan(
        action,
        executed_probes=tuple(result.accepted_observables) + tuple(result.rejected_observables),
        verifier_version=result.verifier_version,
        abstained=decision is not Decision.ACT,
    )
