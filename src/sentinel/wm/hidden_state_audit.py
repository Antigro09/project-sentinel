"""Audits that decide whether a hidden variable is actually hidden.

v1 had a variable called hidden that was a deterministic function of the step
count, and a passing invariance test that certified it. The test varied the
variable while holding the step fixed -- a combination no trajectory reaches --
so it proved something true about an unreachable state and nothing about the
environment. The lesson generalises: an invariance argument is only as good as
the reachability of the states it varies over.

Each audit here returns a verdict rather than raising, so a caller can report
several at once, and each has a planted counterpart in the tests that it must
catch. An audit with no known-bad input measures nothing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class AuditVerdict:
    name: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


def audit_not_determined_by(
    observations: Sequence[tuple[int, int]], public_name: str, hidden_name: str
) -> AuditVerdict:
    """Is the hidden value a function of a public quantity?

    Takes `(public, hidden)` pairs gathered from real trajectories. If every
    public value maps to exactly one hidden value, the hidden variable is
    reconstructible from public data and is not hidden at all.
    """
    mapping: dict[int, set[int]] = defaultdict(set)
    for public, hidden in observations:
        mapping[public].add(hidden)
    ambiguous = {k: sorted(v) for k, v in mapping.items() if len(v) > 1}
    determined = not ambiguous and len(mapping) > 1
    return AuditVerdict(
        name=f"{hidden_name}_not_determined_by_{public_name}",
        passed=not determined,
        detail=(
            f"{hidden_name} is a deterministic function of {public_name}: every observed "
            f"{public_name} maps to exactly one value across {len(observations)} samples"
            if determined
            else f"{hidden_name} takes multiple values at the same {public_name} in "
            f"{len(ambiguous)} of {len(mapping)} observed values"
        ),
        evidence={
            "distinct_public_values": len(mapping),
            "ambiguous_public_values": len(ambiguous),
            "samples": len(observations),
            "example_ambiguous": dict(list(ambiguous.items())[:3]),
        },
    )


def audit_perturbation_reachable(
    reachable_states: Iterable[tuple[Any, ...]], perturbed_state: tuple[Any, ...]
) -> AuditVerdict:
    """Was the state an invariance test varied over actually reachable?

    This is the audit v1 needed and did not have. Perturbing a snapshot field
    produces a state; if no trajectory reaches it, an invariance observed there
    says nothing about the environment anyone will run.
    """
    catalogue = set(reachable_states)
    reachable = perturbed_state in catalogue
    return AuditVerdict(
        name="perturbation_reachable",
        passed=reachable,
        detail=(
            f"the perturbed state {perturbed_state} is reachable"
            if reachable
            else f"the perturbed state {perturbed_state} is NOT reachable; an invariance "
            "observed there is vacuous"
        ),
        evidence={"reachable_states": len(catalogue), "perturbed_state": list(perturbed_state)},
    )


def audit_hidden_not_rendered(
    frame_a: np.ndarray, frame_b: np.ndarray, hidden_a: Any, hidden_b: Any
) -> AuditVerdict:
    """Two reachable states with the same public position and different hidden
    value must render identically."""
    identical = bool(np.array_equal(frame_a, frame_b))
    differing = int(np.sum(frame_a != frame_b))
    return AuditVerdict(
        name="hidden_not_rendered",
        passed=identical,
        detail=(
            f"frames are identical with hidden {hidden_a} against {hidden_b}"
            if identical
            else f"frames differ in {differing} values with hidden {hidden_a} against "
            f"{hidden_b}; the hidden variable is drawn"
        ),
        evidence={"differing_values": differing, "hidden_a": hidden_a, "hidden_b": hidden_b},
    )


def audit_hidden_not_in_features(
    features_a: np.ndarray, features_b: np.ndarray, tolerance: float = 0.0
) -> AuditVerdict:
    """Encoded features for a public alias must not separate by hidden state.

    If they do, the hidden variable has entered the cache through the encoder,
    and every downstream model receives evaluator-only information for free.
    """
    difference = float(np.max(np.abs(np.asarray(features_a) - np.asarray(features_b))))
    clean = difference <= tolerance
    return AuditVerdict(
        name="hidden_not_in_cached_features",
        passed=clean,
        detail=(
            f"features agree to {difference:.3e}"
            if clean
            else f"features differ by {difference:.3e} for a public alias; evaluator-only "
            "state has entered the cache"
        ),
        evidence={"max_abs_difference": difference, "tolerance": tolerance},
    )


def audit_history_identifies(
    trace_a: Sequence[int], trace_b: Sequence[int], hidden_a: Any, hidden_b: Any
) -> AuditVerdict:
    """The other half: a hidden variable nothing can ever infer is not partial
    observability, it is noise. History must separate what the frame cannot."""
    separable = list(trace_a) != list(trace_b)
    return AuditVerdict(
        name="history_identifies_hidden_state",
        passed=separable,
        detail=(
            "the observation traces differ, so a history-conditioned model can "
            "distinguish the two hidden states"
            if separable
            else "the observation traces are identical; no history-conditioned model "
            "could recover the hidden state, which makes it noise rather than state"
        ),
        evidence={"trace_a": list(trace_a), "trace_b": list(trace_b),
                  "hidden_a": hidden_a, "hidden_b": hidden_b},
    )


def summarise(verdicts: Sequence[AuditVerdict]) -> dict[str, Any]:
    return {
        "all_passed": all(v.passed for v in verdicts),
        "failed": [v.name for v in verdicts if not v.passed],
        "verdicts": [v.canonical_dict() for v in verdicts],
    }
