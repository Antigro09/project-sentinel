"""Targeted numerical and resource-stack checks for SHWM.

This script exercises Hydra, NumPy, SciPy, JAX, CVXPy, NetworkX, NumPyro,
Matplotlib, and MLflow on finite diagnostics. It does not implement or train
the proposed 50M/200M world models.
"""

from __future__ import annotations

import json
from pathlib import Path

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
from hydra import compose, initialize_config_dir
from scipy.optimize import least_squares


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "resource-checks.json"
FIGURE_DIR = HERE / "figures"


def rollout_error(lipschitz: np.ndarray, epsilon: float, horizon: np.ndarray) -> np.ndarray:
    lipschitz = np.asarray(lipschitz, dtype=float)
    horizon = np.asarray(horizon, dtype=int)
    answer = np.empty(np.broadcast_shapes(lipschitz.shape, horizon.shape), dtype=float)
    broadcast_l = np.broadcast_to(lipschitz, answer.shape)
    broadcast_h = np.broadcast_to(horizon, answer.shape)
    for index in np.ndindex(answer.shape):
        value = 0.0
        for _ in range(int(broadcast_h[index])):
            value = epsilon + float(broadcast_l[index]) * value
        answer[index] = value
    return answer


def make_figures(cfg: object) -> list[str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []

    # 1. Parameter-state lower bounds. Activations and runtime overhead are
    # explicitly excluded; this is not a full hardware-feasibility estimate.
    parameter_billions = np.asarray([0.05, 0.2, 0.7, 1.5, 15.0, 70.0])
    fig, ax = plt.subplots(figsize=(8, 5))
    for bytes_per_parameter, label in ((12, "12 B/parameter"), (16, "16 B/parameter")):
        ax.plot(
            parameter_billions,
            parameter_billions * bytes_per_parameter,
            marker="o",
            label=label,
        )
    ax.axhline(
        float(cfg.resource_assumptions.machine_unified_memory_gb),
        color="black",
        linestyle="--",
        label="128 GB machine memory",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trainable parameters (billions)")
    ax.set_ylabel("Weights + gradients + optimizer state (decimal GB)")
    ax.set_title("Training-state lower bound; activations excluded")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    path = FIGURE_DIR / "training-state-memory-lower-bound.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path.name)

    # 2. Latent cache size.
    transitions = np.logspace(4, 7, 80)
    fig, ax = plt.subplots(figsize=(8, 5))
    for dimension in cfg.representation.dimensions:
        gib = transitions * int(dimension) * int(
            cfg.resource_assumptions.cache_bytes_per_coordinate
        ) / 2**30
        ax.plot(transitions, gib, label=f"d={dimension}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cached transitions")
    ax.set_ylabel("Latent cache (GiB)")
    ax.set_title("Frozen-feature cache footprint")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    path = FIGURE_DIR / "latent-cache-footprint.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path.name)

    # 3. Open-loop action-sequence growth.
    horizons = np.arange(1, 51)
    fig, ax = plt.subplots(figsize=(8, 5))
    for branching in (2, 4, 8, 16):
        ax.plot(horizons, horizons * np.log10(branching), label=f"b={branching}")
    ax.set_xlabel("Planning horizon H")
    ax.set_ylabel("log10 open-loop sequences")
    ax.set_title("Exhaustive sequence count b^H")
    ax.legend()
    ax.grid(True, alpha=0.25)
    path = FIGURE_DIR / "planning-sequence-growth.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path.name)

    # 4. Rollout-error stability map.
    lipschitz_values = np.linspace(0.5, 1.2, 141)
    horizon_values = np.arange(1, 51)
    grid_l, grid_h = np.meshgrid(lipschitz_values, horizon_values)
    errors = rollout_error(
        grid_l, float(cfg.toy_checks.one_step_error), grid_h
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(
        np.log10(np.maximum(errors, 1e-12)),
        origin="lower",
        aspect="auto",
        extent=[lipschitz_values.min(), lipschitz_values.max(), 1, 50],
        cmap="magma",
    )
    ax.axvline(1.0, color="cyan", linestyle="--", linewidth=1)
    ax.set_xlabel("Transport sensitivity L")
    ax.set_ylabel("Rollout horizon H")
    ax.set_title("log10 accumulated error for one-step error 0.01")
    fig.colorbar(image, ax=ax, label="log10 error")
    path = FIGURE_DIR / "rollout-error-stability-map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path.name)

    # 5. Passive versus interventional posterior concentration.
    reliability = float(cfg.toy_checks.posterior_reliability)
    counts = np.arange(0, 9)
    passive = np.full_like(counts, 0.5, dtype=float)
    active = np.asarray(
        [
            0.5
            if count == 0
            else reliability**count
            / (reliability**count + (1 - reliability) ** count)
            for count in counts
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(counts, passive, marker="o", label="passive action only")
    ax.plot(counts, active, marker="o", label="distinguishing interventions")
    ax.set_ylim(0.45, 1.01)
    ax.set_xlabel("Observations")
    ax.set_ylabel("Posterior mass on true candidate")
    ax.set_title("Intervention coverage changes identifiability")
    ax.legend()
    ax.grid(True, alpha=0.25)
    path = FIGURE_DIR / "intervention-identifiability.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(path.name)

    return generated


def main() -> None:
    with initialize_config_dir(
        version_base=None, config_dir=str((HERE / "configs").resolve())
    ):
        cfg = compose(config_name="shwm")

    np.random.seed(int(cfg.seed))

    # Exact arithmetic audit for the frozen Scale-0 matrix. This validates the
    # configuration contract only; it does not execute the 48 workloads.
    encoder_count = len(cfg.scale0.frozen_backbone_candidates)
    representation_count = len(cfg.representation.arms)
    size_count = len(cfg.scale0.trainable_parameter_counts)
    seed_count = len(cfg.development_seeds)
    primary_cells = encoder_count * representation_count * size_count
    primary_runs = primary_cells * seed_count
    dimension_control_runs = (
        encoder_count * len(cfg.representation.sensitivity_dimensions) * seed_count
    )
    total_training_workloads = primary_runs + dimension_control_runs
    transition_positions_per_run = (
        int(cfg.scale0.sequence_length)
        * int(cfg.scale0.sequences_per_batch)
        * int(cfg.scale0.optimizer_updates)
    )
    planner_invocations_per_run = len(cfg.scale0.rollout_horizons) * int(
        cfg.scale0.planner_invocations_per_horizon
    )
    planner_candidates_per_run = planner_invocations_per_run * int(
        cfg.scale0.candidate_sequences_per_invocation
    )
    assert primary_cells == int(cfg.scale0.primary_matrix_cells) == 12
    assert primary_runs == int(cfg.scale0.primary_matrix_runs) == 36
    assert dimension_control_runs == int(cfg.scale0.dimension_control_runs) == 12
    assert total_training_workloads == int(cfg.scale0.total_training_workloads) == 48
    assert transition_positions_per_run == 204_800
    assert planner_invocations_per_run == 300
    assert planner_candidates_per_run == 19_200

    # JAX: same latent state, opposite actions, opposite successors. An
    # action-blind affine model can only predict the mean, while an
    # action-conditioned affine model can represent both exactly.
    states = jnp.zeros((2,), dtype=jnp.float32)
    actions = jnp.asarray([-1.0, 1.0], dtype=jnp.float32)
    targets = actions

    def conditioned_loss(parameters: jax.Array) -> jax.Array:
        predictions = parameters[0] * states + parameters[1] * actions + parameters[2]
        return jnp.mean((predictions - targets) ** 2)

    conditioned_parameters = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32)
    conditioned_mse = float(conditioned_loss(conditioned_parameters))
    conditioned_gradient = np.asarray(
        jax.grad(conditioned_loss)(conditioned_parameters)
    )
    action_blind_prediction = float(np.mean(np.asarray(targets)))
    action_blind_mse = float(
        np.mean((np.asarray(targets) - action_blind_prediction) ** 2)
    )
    assert conditioned_mse == 0.0
    assert action_blind_mse == 1.0

    # Independent belief-history fixture: current observations and actions are
    # matched, while the hidden histories imply different successors. This is
    # deliberately separate from action conditioning.
    current_observations = jnp.zeros((2,), dtype=jnp.float32)
    histories = jnp.asarray([-1.0, 1.0], dtype=jnp.float32)
    matched_actions = jnp.zeros((2,), dtype=jnp.float32)
    history_targets = histories

    def history_conditioned_loss(parameters: jax.Array) -> jax.Array:
        predictions = (
            parameters[0] * current_observations
            + parameters[1] * histories
            + parameters[2] * matched_actions
            + parameters[3]
        )
        return jnp.mean((predictions - history_targets) ** 2)

    history_parameters = jnp.asarray([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
    history_conditioned_mse = float(history_conditioned_loss(history_parameters))
    history_conditioned_gradient = np.asarray(
        jax.grad(history_conditioned_loss)(history_parameters)
    )
    observation_only_prediction = float(np.mean(np.asarray(history_targets)))
    observation_only_mse = float(
        np.mean(
            (np.asarray(history_targets) - observation_only_prediction) ** 2
        )
    )
    assert history_conditioned_mse == 0.0
    assert observation_only_mse == 1.0

    # SciPy: recover a known rollout sensitivity from exact synthetic errors.
    true_lipschitz = 0.9
    fit_horizons = np.arange(1, 26)
    fit_errors = rollout_error(
        np.full_like(fit_horizons, true_lipschitz, dtype=float),
        float(cfg.toy_checks.one_step_error),
        fit_horizons,
    )

    def residual(parameter: np.ndarray) -> np.ndarray:
        predicted = rollout_error(
            np.full_like(fit_horizons, parameter[0], dtype=float),
            float(cfg.toy_checks.one_step_error),
            fit_horizons,
        )
        return predicted - fit_errors

    fit = least_squares(residual, x0=np.asarray([0.7]), bounds=(0.0, 1.5))
    assert fit.success and abs(float(fit.x[0]) - true_lipschitz) < 1e-8

    # CVXPy: a reproducible parameter-allocation scaffold for a 200M arm.
    # The target fractions are an engineering prior, not a learned optimum.
    total_budget = float(cfg.resource_assumptions.trainable_parameter_budget)
    target_fraction = np.asarray([0.70, 0.15, 0.10, 0.05])
    allocation = cp.Variable(4)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(allocation / total_budget - target_fraction)),
        [
            allocation >= np.asarray([20e6, 5e6, 5e6, 2e6]),
            cp.sum(allocation) == total_budget,
        ],
    )
    problem.solve(solver="CLARABEL")
    assert problem.status == cp.OPTIMAL
    allocation_value = np.asarray(allocation.value)
    assert abs(float(np.sum(allocation_value)) - total_budget) < 1.0

    # NetworkX: every route from either proposal source to external action
    # passes through the verifier in the specified control graph.
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("observation", "frozen_encoder"),
            ("frozen_encoder", "belief_state"),
            ("memory", "belief_state"),
            ("belief_state", "world_model"),
            ("world_model", "latent_planner"),
            ("belief_state", "fast_policy"),
            ("latent_planner", "verifier"),
            ("fast_policy", "verifier"),
            ("verifier", "action"),
            ("action", "environment"),
            ("environment", "observation"),
            ("environment", "counterexample"),
            ("counterexample", "memory"),
        ]
    )
    proposal_paths = []
    for source in ("latent_planner", "fast_policy"):
        for path in nx.all_simple_paths(graph, source=source, target="action"):
            proposal_paths.append(path)
            assert "verifier" in path

    # NumPyro: represent the finite posterior used by the intervention plot.
    reliability = float(cfg.toy_checks.posterior_reliability)
    count = 2
    masses = np.asarray([(1 - reliability) ** count, reliability**count])
    posterior_probabilities = masses / masses.sum()
    posterior = dist.Categorical(probs=jnp.asarray(posterior_probabilities))
    assert np.isclose(float(np.sum(np.asarray(posterior.probs))), 1.0)

    figures = make_figures(cfg)

    # Local-only MLflow log. The random run ID is intentionally excluded from
    # the deterministic JSON result and the store is gitignored.
    tracking_db = (HERE / "results" / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    mlflow.set_experiment(str(cfg.tracking.experiment_name))
    with mlflow.start_run(run_name="shwm-resource-audit") as run:
        mlflow.log_params(
            {
                "seed": int(cfg.seed),
                "parameter_budget": int(total_budget),
                "toy_reliability": reliability,
                "scale0_training_workloads": total_training_workloads,
            }
        )
        mlflow.log_metrics(
            {
                "conditioned_mse": conditioned_mse,
                "action_blind_mse": action_blind_mse,
                "history_conditioned_mse": history_conditioned_mse,
                "observation_only_alias_mse": observation_only_mse,
                "recovered_lipschitz": float(fit.x[0]),
                "posterior_true_model_after_two_interventions": float(
                    posterior_probabilities[1]
                ),
            }
        )
        assert run.info.run_id

    result = {
        "status": "MEASURED: targeted numerical/resource checks passed; no SHWM capability gate",
        "hydra": {
            "seed": int(cfg.seed),
            "development_seeds": list(cfg.development_seeds),
            "representation_arms": list(cfg.representation.arms),
            "dimensions": list(cfg.representation.dimensions),
        },
        "scale0_frozen_matrix": {
            "encoder_count": encoder_count,
            "representation_count": representation_count,
            "trainable_size_count": size_count,
            "development_seed_count": seed_count,
            "primary_cells": primary_cells,
            "primary_runs": primary_runs,
            "dimension_control_runs": dimension_control_runs,
            "total_training_workloads": total_training_workloads,
            "fixed_transition_count": int(cfg.scale0.fixed_transition_count),
            "transition_positions_per_run": transition_positions_per_run,
            "planner_invocations_per_run": planner_invocations_per_run,
            "planner_candidates_per_run": planner_candidates_per_run,
            "peak_unified_memory_ceiling_gib": int(
                cfg.resource_assumptions.peak_unified_memory_ceiling_gib
            ),
            "artifact_storage_ceiling_gib": int(
                cfg.resource_assumptions.artifact_storage_ceiling_gib
            ),
            "per_run_timeout_hours": int(
                cfg.resource_assumptions.per_run_timeout_hours
            ),
            "total_matrix_timeout_hours": int(
                cfg.resource_assumptions.total_matrix_timeout_hours
            ),
            "scope": "configuration arithmetic only; workloads not executed",
        },
        "jax_action_conditioning_toy": {
            "conditioned_mse": conditioned_mse,
            "conditioned_gradient": conditioned_gradient.tolist(),
            "best_action_blind_mse": action_blind_mse,
            "scope": "two contradictory transitions from one state",
        },
        "jax_belief_history_toy": {
            "history_conditioned_mse": history_conditioned_mse,
            "history_conditioned_gradient": history_conditioned_gradient.tolist(),
            "best_observation_only_mse": observation_only_mse,
            "scope": "same observation and action, different hidden histories",
        },
        "scipy_rollout_fit": {
            "success": bool(fit.success),
            "true_lipschitz": true_lipschitz,
            "recovered_lipschitz": float(fit.x[0]),
            "residual_l2": float(np.linalg.norm(fit.fun)),
        },
        "cvxpy_parameter_allocation": {
            "status": problem.status,
            "labels": ["dynamics", "projector", "event_heads", "uncertainty"],
            "parameters": allocation_value.tolist(),
            "total": float(np.sum(allocation_value)),
            "warning": "engineering allocation prior, not capability optimization",
        },
        "networkx_control_graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "proposal_to_action_paths": proposal_paths,
            "all_paths_include_verifier": True,
        },
        "numpyro_intervention_posterior": {
            "probabilities_A_B": posterior_probabilities.tolist(),
            "normalized": True,
        },
        "matplotlib_figures": figures,
        "mlflow": {
            "experiment": str(cfg.tracking.experiment_name),
            "logged_to_local_sqlite_store": True,
            "store_committed": False,
        },
        "versions": {
            "jax": jax.__version__,
            "cvxpy": cp.__version__,
            "networkx": nx.__version__,
            "numpy": np.__version__,
            "mlflow": mlflow.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
