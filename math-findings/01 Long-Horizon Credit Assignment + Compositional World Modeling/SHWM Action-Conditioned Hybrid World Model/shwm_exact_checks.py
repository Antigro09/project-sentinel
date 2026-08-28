"""Exact finite enumeration checks for SHWM boundary claims.

The checks intentionally use a two-action, two-model world. They falsify the
idea that passive prediction alone identifies unseen interventions; they are
not evidence that the proposed scaled architecture works.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "exact-checks.json"


def kernel_a(action: int) -> int:
    del action
    return 0


def kernel_b(action: int) -> int:
    return action


def exact_posterior(observations: list[tuple[int, int]]) -> tuple[Fraction, Fraction]:
    likelihoods = []
    for kernel in (kernel_a, kernel_b):
        likelihoods.append(
            Fraction(int(all(kernel(action) == outcome for action, outcome in observations)), 1)
        )
    masses = [Fraction(1, 2) * likelihood for likelihood in likelihoods]
    normalizer = sum(masses)
    if normalizer == 0:
        raise ValueError("observations are outside both deterministic candidates")
    return masses[0] / normalizer, masses[1] / normalizer


def noisy_posterior_b_after_identifying_interventions(
    count: int, reliability: Fraction
) -> Fraction:
    """Posterior on B after repeatedly observing B's distinct action-1 outcome."""
    b_mass = Fraction(1, 2) * reliability**count
    a_mass = Fraction(1, 2) * (1 - reliability) ** count
    return b_mass / (a_mass + b_mass)


def rollout_error(lipschitz: Fraction, epsilon: Fraction, horizon: int) -> Fraction:
    error = Fraction(0)
    for _ in range(horizon):
        error = epsilon + lipschitz * error
    return error


def main() -> None:
    # Any amount of data from the policy that only chooses action 0 leaves the
    # two candidates exactly tied.
    passive_posteriors = {}
    for count in (0, 1, 2, 8, 64):
        posterior = exact_posterior([(0, 0)] * count)
        assert posterior == (Fraction(1, 2), Fraction(1, 2))
        passive_posteriors[str(count)] = [float(value) for value in posterior]

    # One distinguishing intervention identifies the deterministic model.
    posterior_if_outcome_zero = exact_posterior([(1, 0)])
    posterior_if_outcome_one = exact_posterior([(1, 1)])
    assert posterior_if_outcome_zero == (Fraction(1), Fraction(0))
    assert posterior_if_outcome_one == (Fraction(0), Fraction(1))

    # Under bounded independent observation noise, intervention evidence
    # concentrates geometrically but never gives certainty at finite n.
    reliability = Fraction(9, 10)
    noisy_curve = {
        str(count): float(
            noisy_posterior_b_after_identifying_interventions(count, reliability)
        )
        for count in range(1, 9)
    }
    minimum_for_95 = min(
        count for count in range(1, 100)
        if noisy_posterior_b_after_identifying_interventions(count, reliability)
        >= Fraction(95, 100)
    )
    assert minimum_for_95 == 2

    # Enumerate action-sequence counts rather than relying on the formula.
    sequence_counts = {}
    for branching in (2, 3, 4):
        for horizon in (0, 1, 2, 5):
            enumerated = sum(1 for _ in product(range(branching), repeat=horizon))
            expected = branching**horizon
            assert enumerated == expected
            sequence_counts[f"b{branching}_h{horizon}"] = enumerated

    # Check the finite geometric rollout expression with exact rationals.
    rollout_cases = []
    for lipschitz in (Fraction(1, 2), Fraction(9, 10), Fraction(1), Fraction(11, 10)):
        for horizon in (0, 1, 5, 25):
            observed = rollout_error(lipschitz, Fraction(1, 100), horizon)
            closed = Fraction(1, 100) * sum(
                (lipschitz**index for index in range(horizon)), Fraction(0)
            )
            assert observed == closed
            rollout_cases.append(
                {
                    "lipschitz": float(lipschitz),
                    "horizon": horizon,
                    "error": float(observed),
                }
            )

    result = {
        "status": "MEASURED: exact finite counterexample and arithmetic checks passed; toy scope only",
        "passive_policy_posterior": passive_posteriors,
        "distinguishing_intervention": {
            "outcome_0_posterior_A_B": [
                float(value) for value in posterior_if_outcome_zero
            ],
            "outcome_1_posterior_A_B": [
                float(value) for value in posterior_if_outcome_one
            ],
        },
        "bounded_noise": {
            "reliability": float(reliability),
            "posterior_true_model": noisy_curve,
            "minimum_identifying_interventions_for_0_95": minimum_for_95,
            "independence_assumption": True,
        },
        "enumerated_sequence_counts": sequence_counts,
        "rollout_error_cases": rollout_cases,
        "sanity": {"all_outputs_finite": all(math.isfinite(case["error"]) for case in rollout_cases)},
        "limitations": [
            "The construction proves existence of non-identifiability, not that every passive dataset is non-identifying.",
            "The noisy concentration curve assumes a correct two-model class and conditionally independent observations.",
            "No neural model, environment benchmark, planner, or transfer mechanism is evaluated.",
        ],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
