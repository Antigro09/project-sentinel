"""SymPy checks for X64H posterior, conflict, and commitment identities."""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "symbolic-checks.json"


def main() -> None:
    mass, rho0, rho1, c0, c1 = sp.symbols(
        "mass rho0 rho1 c0 c1", positive=True
    )
    delta, epsilon, gamma = sp.symbols(
        "delta epsilon gamma", nonnegative=True
    )
    p_best, loss_wrong, ask_cost = sp.symbols(
        "p_best loss_wrong ask_cost", positive=True
    )

    # Under hard consistency, L0 is proportional to consistent language mass,
    # while a complementary mismatch model is proportional to 1 - mass.
    likelihood_match = c0 * mass
    likelihood_mismatch = c1 * (1 - mass)
    conflict_posterior = sp.cancel(
        rho1 * likelihood_mismatch
        / (rho0 * likelihood_match + rho1 * likelihood_mismatch)
    )

    posterior_odds = sp.cancel(conflict_posterior / (1 - conflict_posterior))
    bayes_odds = sp.cancel(
        rho1 * likelihood_mismatch / (rho0 * likelihood_match)
    )
    assert sp.simplify(posterior_odds - bayes_odds) == 0

    # X64E's score 1 - mass is exactly Bayesian only under balanced effective
    # prior/likelihood constants; otherwise it is merely monotone in the
    # Bayesian posterior.
    balanced = sp.simplify(
        conflict_posterior.subs(c1, rho0 * c0 / rho1) - (1 - mass)
    )
    assert balanced == 0

    # If the best in-model class is correct with probability 1-delta and the
    # model is out of class with probability epsilon, total error is bounded
    # by epsilon + delta - epsilon*delta, hence by epsilon + delta.
    calibrated_conditional_error = sp.expand(
        1 - (1 - epsilon) * (1 - delta)
    )
    assert sp.simplify(
        calibrated_conditional_error - (epsilon + delta - epsilon * delta)
    ) == 0
    calibrated_error = sp.expand(
        1 - (1 - gamma) * (1 - epsilon) * (1 - delta)
    )
    assert sp.simplify(
        calibrated_error
        - (gamma + (1 - gamma) * calibrated_conditional_error)
    ) == 0
    assert sp.simplify(
        (gamma + epsilon + delta) - calibrated_error
    ) == gamma * epsilon + gamma * delta + epsilon * delta - gamma * epsilon * delta

    # With zero loss for a correct execution and loss_wrong for a mistake,
    # acting is cheaper than asking exactly when p_best exceeds this threshold.
    execute_risk = loss_wrong * (1 - p_best)
    acting_threshold = 1 - ask_cost / loss_wrong
    assert sp.simplify(
        execute_risk.subs(p_best, acting_threshold) - ask_cost
    ) == 0

    result = {
        "status": "MEASURED: symbolic identities executed successfully",
        "conflict_posterior": str(conflict_posterior),
        "posterior_odds": str(posterior_odds),
        "bayes_odds_identity": True,
        "x64e_exact_under_balanced_effective_constants": True,
        "calibrated_conditional_error_given_match": str(
            calibrated_conditional_error
        ),
        "calibrated_total_error": str(calibrated_error),
        "acting_vs_asking_threshold": f"p_best >= {acting_threshold}",
        "sympy_version": sp.__version__,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
