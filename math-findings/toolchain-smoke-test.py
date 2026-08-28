"""Offline smoke tests for the isolated Sentinel math-research environment."""

from __future__ import annotations

import importlib.metadata
import json
import math
import tempfile
from pathlib import Path


def version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def main() -> None:
    import cvxpy as cp
    import jax
    import jax.numpy as jnp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow
    import networkx as nx
    import numpy as np
    import numpyro.distributions as dist
    import scipy.integrate
    import sympy as sp
    from hypothesis import find, strategies as st
    from sklearn.linear_model import LinearRegression

    checks: dict[str, object] = {}

    x, y = sp.symbols("x y")
    identity = sp.expand((x + y) ** 2) - (x**2 + 2 * x * y + y**2)
    assert sp.simplify(identity) == 0
    checks["sympy_identity"] = True

    matrix = np.array([[3.0, 1.0], [1.0, 2.0]])
    rhs = np.array([9.0, 8.0])
    solution = np.linalg.solve(matrix, rhs)
    assert np.allclose(matrix @ solution, rhs)
    checks["numpy_linear_solve"] = solution.tolist()

    ode = scipy.integrate.solve_ivp(
        lambda _t, state: -state,
        (0.0, 1.0),
        [1.0],
        rtol=1e-10,
        atol=1e-12,
    )
    assert abs(float(ode.y[0, -1]) - math.exp(-1.0)) < 1e-8
    checks["scipy_ode"] = float(ode.y[0, -1])

    variable = cp.Variable()
    problem = cp.Problem(cp.Minimize(cp.square(variable - 3.0)), [variable >= 0.0])
    optimum = problem.solve(solver="CLARABEL")
    assert problem.status == "optimal"
    assert abs(float(variable.value) - 3.0) < 1e-5
    checks["cvxpy_optimum"] = float(optimum)

    gradient = float(jax.grad(lambda value: value**2)(jnp.array(3.0)))
    assert abs(gradient - 6.0) < 1e-6
    checks["jax_gradient"] = gradient

    normal_mean = float(dist.Normal(0.0, 1.0).mean)
    assert normal_mean == 0.0
    checks["numpyro_distribution"] = normal_mean

    graph = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    assert nx.is_directed_acyclic_graph(graph)
    checks["networkx_topological_order"] = list(nx.topological_sort(graph))

    model = LinearRegression().fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
    assert abs(float(model.coef_[0]) - 2.0) < 1e-9
    checks["sklearn_slope"] = float(model.coef_[0])

    counterexample = find(st.integers(min_value=0), lambda value: value > 100)
    assert counterexample == 101
    checks["hypothesis_counterexample"] = counterexample

    with tempfile.TemporaryDirectory(prefix="sentinel-math-smoke-") as temp_dir:
        temp_path = Path(temp_dir)
        figure_path = temp_path / "smoke.png"
        figure, axis = plt.subplots(figsize=(3, 2))
        axis.plot([0, 1, 2], [0, 1, 4])
        figure.tight_layout()
        figure.savefig(figure_path, dpi=120)
        plt.close(figure)
        assert figure_path.stat().st_size > 0
        checks["matplotlib_render_bytes"] = figure_path.stat().st_size

        tracking_database = temp_path / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{tracking_database}")
        mlflow.set_experiment("toolchain-smoke")
        with mlflow.start_run():
            mlflow.log_metric("verified", 1.0)
        assert tracking_database.stat().st_size > 0
        checks["mlflow_local_tracking"] = True

    checks["versions"] = {
        name: version(name)
        for name in (
            "sympy",
            "numpy",
            "scipy",
            "cvxpy",
            "jax",
            "numpyro",
            "networkx",
            "matplotlib",
            "hydra-core",
            "mlflow",
            "dvc",
            "jupyterlab",
            "hypothesis",
        )
    }

    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
