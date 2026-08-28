"""Targeted checks using the remaining OCI-relevant research stack.

SciPy solves a finite symbol-assignment problem, CVXPy recovers a minimal
Lipschitz contract from separating probes, JAX differentiates an unseen
composition, NumPyro represents noisy symbol uncertainty, Hydra loads the
frozen configuration, and MLflow records the run locally.
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
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "resource-checks.json"


def affine_signature(a: int, b: int, probes: list[int], prime: int) -> np.ndarray:
    return np.asarray([(a * value + b) % prime for value in probes], dtype=float)


def main() -> None:
    with initialize_config_dir(
        version_base=None, config_dir=str((HERE / "experiments").resolve())
    ):
        cfg = compose(config_name="oci")

    seed = int(cfg.seed)
    prime = int(cfg.field_prime)
    rng = np.random.default_rng(seed)

    # SciPy: recover a hidden surface-to-generator assignment from a separating
    # signature cost matrix.
    generators = [(1, 0), (1, 1), (2, 0), (2, 3), (4, 1), (5, 7), (7, 2), (9, 4)]
    probes = [int(value) for value in cfg.signature_probes]
    canonical = np.stack(
        [affine_signature(a, b, probes, prime) for a, b in generators]
    )
    hidden_permutation = rng.permutation(len(generators))
    observed = canonical[hidden_permutation]
    cost = np.sum(np.abs(observed[:, None, :] - canonical[None, :, :]), axis=2)
    rows, columns = linear_sum_assignment(cost)
    recovered = np.empty_like(hidden_permutation)
    recovered[rows] = columns
    assignment_correct = bool(np.array_equal(recovered, hidden_permutation))
    assert assignment_correct

    # CVXPy: infer the smallest coordinate-wise Lipschitz contract.  Axis
    # probes make the exact sensitivities identifiable in this finite check.
    left_true = float(cfg.contract.left_sensitivity)
    right_true = float(cfg.contract.right_sensitivity)
    differences = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    output_differences = np.abs(
        left_true * differences[:, 0] + right_true * differences[:, 1]
    )
    lipschitz = cp.Variable(2, nonneg=True)
    constraints = [
        output_differences
        <= cp.multiply(np.abs(differences[:, 0]), lipschitz[0])
        + cp.multiply(np.abs(differences[:, 1]), lipschitz[1])
    ]
    problem = cp.Problem(cp.Minimize(cp.sum(lipschitz)), constraints)
    optimum = problem.solve(solver="CLARABEL")
    fitted_contract = np.asarray(lipschitz.value, dtype=float)
    assert problem.status == cp.OPTIMAL
    assert np.allclose(fitted_contract, [left_true, right_true], atol=1e-7)

    # JAX: show that an unseen symbolic word remains differentiable with respect
    # to primitive contracts.  This is a mechanism check, not a learning result.
    params = jnp.asarray(
        [[1.05, 0.1], [0.8, -0.2], [1.2, 0.05]], dtype=jnp.float32
    )
    unseen_word = (0, 2, 1, 2, 0)

    def execute(parameter_matrix: jax.Array, initial: jax.Array) -> jax.Array:
        value = initial
        for generator_index in unseen_word:
            a, b = parameter_matrix[generator_index]
            value = a * value + b
        return value

    gradient = jax.grad(lambda parameter_matrix: execute(parameter_matrix, 0.3))(params)
    gradient_finite = bool(np.all(np.isfinite(np.asarray(gradient))))
    assert gradient_finite

    # NumPyro: form a calibrated finite posterior over generator identity under
    # a simple exponential mismatch likelihood.
    observed_noisy = observed[0].copy()
    observed_noisy[0] += float(cfg.probe_noise)
    mismatch = np.sum((canonical - observed_noisy[None, :]) ** 2, axis=1)
    logits = jnp.asarray(-mismatch)
    posterior = dist.Categorical(logits=logits)
    probabilities = np.asarray(posterior.probs)
    posterior_normalized = bool(np.isclose(np.sum(probabilities), 1.0))
    assert posterior_normalized

    mlflow.set_experiment(str(cfg.tracking.experiment_name))
    with mlflow.start_run(run_name="resource-checks") as run:
        mlflow.log_params(
            {
                "seed": seed,
                "prime": prime,
                "generator_count": len(generators),
                "probe_count": len(probes),
            }
        )
        mlflow.log_metrics(
            {
                "assignment_correct": float(assignment_correct),
                "contract_sum": float(optimum),
                "gradient_l2": float(np.linalg.norm(np.asarray(gradient))),
                "posterior_max": float(np.max(probabilities)),
            }
        )
        run_id = run.info.run_id

    result = {
        "status": "MEASURED: targeted resource checks executed",
        "hydra_config": {
            "seed": seed,
            "field_prime": prime,
            "signature_probes": probes,
        },
        "scipy_assignment": {
            "hidden_permutation": hidden_permutation.tolist(),
            "recovered_permutation": recovered.tolist(),
            "correct": assignment_correct,
        },
        "cvxpy_contract": {
            "status": problem.status,
            "fitted": fitted_contract.tolist(),
            "objective": float(optimum),
        },
        "jax_unseen_composition": {
            "word": list(unseen_word),
            "output": float(execute(params, 0.3)),
            "gradient_l2": float(np.linalg.norm(np.asarray(gradient))),
            "gradient_finite": gradient_finite,
        },
        "numpyro_symbol_posterior": {
            "probabilities": probabilities.tolist(),
            "normalized": posterior_normalized,
        },
        "mlflow": {
            "experiment": str(cfg.tracking.experiment_name),
            "run_id": run_id,
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
