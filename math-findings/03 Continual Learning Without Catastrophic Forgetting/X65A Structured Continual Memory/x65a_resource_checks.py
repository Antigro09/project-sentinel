"""Targeted checks across the X65A research resource stack.

These checks exercise finite optimization, differentiable retrieval utility,
hierarchical probability objects, frozen configuration, and local tracking.
They do not train or evaluate Sentinel.
"""

from __future__ import annotations

import json
from pathlib import Path

import cvxpy as cp
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import numpyro.distributions as dist
from hydra import compose, initialize_config_dir
from scipy.optimize import Bounds, LinearConstraint, milp


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "resource-checks.json"


def main() -> None:
    with initialize_config_dir(
        version_base=None, config_dir=str((HERE / "configs").resolve())
    ):
        cfg = compose(config_name="x65a")

    seed = int(cfg.seed)
    budget = float(cfg.retrieval_budget)
    costs = np.asarray([1.0, 2.0, 1.0, 1.0])
    additive_values = np.asarray([3.0, 5.0, 2.0, -4.0])

    # SciPy: exact binary retrieval for the deliberately additive restricted
    # case in which knapsack machinery is justified.
    scipy_result = milp(
        c=-additive_values,
        integrality=np.ones(len(costs)),
        bounds=Bounds(np.zeros(len(costs)), np.ones(len(costs))),
        constraints=LinearConstraint(costs[None, :], -np.inf, budget),
        options={"time_limit": 5.0},
    )
    assert scipy_result.success
    scipy_selection = np.rint(scipy_result.x).astype(int)
    assert float(costs @ scipy_selection) <= budget + 1e-8

    # CVXPy: LP relaxation gives an upper bound on the same integer retrieval
    # objective.  This is a diagnostic relaxation, not the main selection rule.
    relaxed = cp.Variable(len(costs))
    problem = cp.Problem(
        cp.Maximize(additive_values @ relaxed),
        [relaxed >= 0, relaxed <= 1, costs @ relaxed <= budget],
    )
    relaxed_value = problem.solve(solver="CLARABEL")
    assert problem.status == cp.OPTIMAL
    integer_value = float(additive_values @ scipy_selection)
    assert float(relaxed_value) + 1e-7 >= integer_value

    # JAX: differentiable independent-coverage surrogate.  This surrogate is
    # monotone submodular at binary selections before the cost penalty; the full
    # X65A utility need not be.
    feature_probabilities = jnp.asarray(
        [
            [0.9, 0.1, 0.0],
            [0.0, 0.8, 0.3],
            [0.2, 0.0, 0.9],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    feature_values = jnp.asarray([3.0, 2.0, 4.0], dtype=jnp.float32)

    def soft_coverage(weights: jax.Array) -> jax.Array:
        uncovered = jnp.prod(1.0 - weights[:, None] * feature_probabilities, axis=0)
        return jnp.sum(feature_values * (1.0 - uncovered)) - 0.05 * jnp.sum(weights)

    start = jnp.full((4,), 0.5, dtype=jnp.float32)
    gradient = np.asarray(jax.grad(soft_coverage)(start))
    assert np.all(np.isfinite(gradient))

    # NumPyro: exact finite convention-component posterior represented as a
    # categorical distribution after three persistent observations.
    reliability = float(cfg.observation_reliability)
    observations = np.asarray([1, 1, 0])
    likelihood = np.asarray(
        [
            np.prod(np.where(observations == latent, reliability, 1 - reliability))
            for latent in (0, 1)
        ]
    )
    posterior_probabilities = likelihood / likelihood.sum()
    posterior = dist.Categorical(probs=jnp.asarray(posterior_probabilities))
    assert np.isclose(float(np.sum(np.asarray(posterior.probs))), 1.0)

    mlflow.set_experiment(str(cfg.tracking.experiment_name))
    with mlflow.start_run(run_name="x65a-resource-checks") as run:
        mlflow.log_params(
            {
                "seed": seed,
                "retrieval_budget": budget,
                "memory_budget_bytes": int(cfg.memory_budget_bytes),
            }
        )
        mlflow.log_metrics(
            {
                "integer_retrieval_value": integer_value,
                "lp_upper_bound": float(relaxed_value),
                "soft_coverage_gradient_l2": float(np.linalg.norm(gradient)),
                "posterior_max": float(np.max(posterior_probabilities)),
            }
        )
        # The run ID is intentionally excluded from the DVC-tracked JSON: it is
        # traceable in the local MLflow store but random across reproductions.
        assert run.info.run_id

    result = {
        "status": "MEASURED: targeted resource checks executed; no X65A gate claim",
        "hydra_config": {
            "seed": seed,
            "retrieval_budget": budget,
            "memory_budget_bytes": int(cfg.memory_budget_bytes),
        },
        "scipy_binary_knapsack": {
            "success": bool(scipy_result.success),
            "selection": scipy_selection.tolist(),
            "cost": float(costs @ scipy_selection),
            "value": integer_value,
        },
        "cvxpy_relaxation": {
            "status": problem.status,
            "selection": np.asarray(relaxed.value).tolist(),
            "upper_bound": float(relaxed_value),
        },
        "jax_soft_coverage": {
            "value_at_half_selection": float(soft_coverage(start)),
            "gradient": gradient.tolist(),
            "gradient_finite": True,
        },
        "numpyro_finite_posterior": {
            "probabilities": posterior_probabilities.tolist(),
            "normalized": True,
        },
        "mlflow": {
            "experiment": str(cfg.tracking.experiment_name),
            "logged_to_local_store": True,
        },
        "versions": {
            "jax": jax.__version__,
            "cvxpy": cp.__version__,
            "mlflow": mlflow.__version__,
        },
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
