"""Finite falsification experiments for Operadic Contract Induction (OCI).

The experiments are deliberately small and exact where possible.  They test
the boundary of the proposed theorem: transfer succeeds for novel trees only
after primitive symbols are identified, certified rewrites preserve meaning,
and approximation error follows the contract recurrence.  They are not an AGI
benchmark and do not establish effectiveness on natural language.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "falsification-results.json"
FIGURE_DIR = HERE / "figures"
SEED = 20260826
PRIME = 17
N_GENERATORS = 8
N_SIGNATURE_TRIALS = 600
N_TRANSFER_TRIALS = 180
TEST_WORDS_PER_TRIAL = 120


@dataclass(frozen=True)
class AffineMod:
    a: int
    b: int

    def __call__(self, value: int, prime: int = PRIME) -> int:
        return (self.a * value + self.b) % prime


def sample_generators(rng: np.random.Generator) -> list[AffineMod]:
    candidates = [
        AffineMod(a, b)
        for a in range(1, PRIME)
        for b in range(PRIME)
    ]
    indices = rng.choice(len(candidates), size=N_GENERATORS, replace=False)
    return [candidates[int(index)] for index in indices]


def signature(generator: AffineMod, probes: Iterable[int]) -> tuple[int, ...]:
    return tuple(generator(probe) for probe in probes)


def signatures_are_separating(
    generators: list[AffineMod], probes: Iterable[int]
) -> bool:
    signatures = [signature(generator, probes) for generator in generators]
    return len(set(signatures)) == len(signatures)


def mean_candidate_class_size(
    generators: list[AffineMod], probes: Iterable[int]
) -> float:
    signatures = [signature(generator, probes) for generator in generators]
    counts = {item: signatures.count(item) for item in set(signatures)}
    return float(np.mean([counts[item] for item in signatures]))


def signature_phase_diagram() -> dict[str, list[float]]:
    rng = np.random.default_rng(SEED)
    query_counts = list(range(0, 7))
    separating_probability: list[float] = []
    candidate_class_size: list[float] = []

    for query_count in query_counts:
        separated: list[float] = []
        class_sizes: list[float] = []
        for _ in range(N_SIGNATURE_TRIALS):
            generators = sample_generators(rng)
            probes = rng.choice(PRIME, size=query_count, replace=False).tolist()
            separated.append(float(signatures_are_separating(generators, probes)))
            class_sizes.append(mean_candidate_class_size(generators, probes))
        separating_probability.append(float(np.mean(separated)))
        candidate_class_size.append(float(np.mean(class_sizes)))

    fig, axis_left = plt.subplots(figsize=(7.4, 4.6))
    axis_left.plot(
        query_counts,
        separating_probability,
        marker="o",
        linewidth=2.2,
        color="#26547c",
        label="separating probability",
    )
    axis_left.set_xlabel("probe contexts per primitive")
    axis_left.set_ylabel("P(all generator signatures distinct)", color="#26547c")
    axis_left.set_ylim(-0.03, 1.03)
    axis_left.grid(alpha=0.25)
    axis_right = axis_left.twinx()
    axis_right.plot(
        query_counts,
        candidate_class_size,
        marker="s",
        linewidth=2.0,
        color="#ef476f",
        label="mean equivalence-class size",
    )
    axis_right.set_ylabel("mean observational class size", color="#ef476f")
    fig.suptitle("OCI identifiability boundary: persistent probe signatures")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "signature-identifiability.png", dpi=180)
    plt.close(fig)

    return {
        "query_counts": query_counts,
        "separating_probability": separating_probability,
        "mean_candidate_class_size": candidate_class_size,
    }


def infer_surface_mapping(
    generators: list[AffineMod],
    surface_to_generator: np.ndarray,
    probes: list[int],
) -> list[int | None]:
    canonical_signatures = [signature(generator, probes) for generator in generators]
    inferred: list[int | None] = []
    for surface_index in range(len(generators)):
        observed = canonical_signatures[int(surface_to_generator[surface_index])]
        matches = [
            index
            for index, candidate in enumerate(canonical_signatures)
            if candidate == observed
        ]
        inferred.append(matches[0] if len(matches) == 1 else None)
    return inferred


def execute_word(
    word: list[int], generators: list[AffineMod], start: int
) -> int:
    value = start
    for generator_index in word:
        value = generators[generator_index](value)
    return value


def transfer_experiment() -> dict[str, list[float]]:
    rng = np.random.default_rng(SEED + 1)
    query_counts = list(range(0, 7))
    coverage_by_query: list[float] = []
    exact_accuracy_by_query: list[float] = []
    conditional_accuracy_by_query: list[float | None] = []
    false_commitment_by_query: list[float] = []

    for query_count in query_counts:
        covered = 0
        correct = 0
        false_commitments = 0
        total = 0
        for _ in range(N_TRANSFER_TRIALS):
            generators = sample_generators(rng)
            surface_to_generator = rng.permutation(N_GENERATORS)
            probes = rng.choice(PRIME, size=query_count, replace=False).tolist()
            inferred = infer_surface_mapping(generators, surface_to_generator, probes)

            # The "training support" contains only words of length <= 2.
            # Every evaluation word has length 4--8 and is structurally OOD.
            for _ in range(TEST_WORDS_PER_TRIAL):
                length = int(rng.integers(4, 9))
                surface_word = rng.integers(0, N_GENERATORS, size=length).tolist()
                start = int(rng.integers(0, PRIME))
                true_word = [int(surface_to_generator[index]) for index in surface_word]
                true_output = execute_word(true_word, generators, start)
                decoded = [inferred[index] for index in surface_word]
                total += 1
                if all(index is not None for index in decoded):
                    covered += 1
                    predicted_output = execute_word(
                        [int(index) for index in decoded], generators, start
                    )
                    if predicted_output == true_output:
                        correct += 1
                    else:
                        false_commitments += 1

        coverage = covered / total
        coverage_by_query.append(coverage)
        exact_accuracy_by_query.append(correct / total)
        conditional_accuracy_by_query.append(correct / covered if covered else None)
        false_commitment_by_query.append(false_commitments / total)

    fig, axis = plt.subplots(figsize=(7.4, 4.6))
    axis.plot(
        query_counts,
        coverage_by_query,
        marker="o",
        linewidth=2.2,
        label="OCI commitment coverage",
    )
    axis.plot(
        query_counts,
        exact_accuracy_by_query,
        marker="s",
        linewidth=2.2,
        label="OCI unconditional exact accuracy",
    )
    axis.axhline(
        0.0,
        color="#ef476f",
        linestyle="--",
        linewidth=1.7,
        label="flat exact-word memorizer coverage",
    )
    axis.set_xlabel("probe contexts per primitive")
    axis.set_ylabel("fraction of OOD words")
    axis.set_ylim(-0.03, 1.03)
    axis.set_title("Novel words of length 4–8 after support only at length ≤2")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ood-composition-transfer.png", dpi=180)
    plt.close(fig)

    return {
        "query_counts": query_counts,
        "commitment_coverage": coverage_by_query,
        "unconditional_exact_accuracy": exact_accuracy_by_query,
        "conditional_accuracy_when_committed": conditional_accuracy_by_query,
        "false_commitment_rate": false_commitment_by_query,
        "flat_memorizer_ood_coverage": [0.0] * len(query_counts),
    }


def build_full_binary_contract_graph(
    depth: int,
    rho: float,
    epsilon: float,
    rng: np.random.Generator,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    next_id = itertools.count()

    def add_subtree(level: int) -> int:
        node_id = next(next_id)
        if level == depth:
            leaf_value = float(rng.uniform(-1.0, 1.0))
            graph.add_node(
                node_id,
                kind="leaf",
                exact=leaf_value,
                approximate=leaf_value,
                bound=0.0,
            )
            return node_id
        split = float(rng.uniform(0.15, 0.85))
        left_weight = rho * split
        right_weight = rho * (1.0 - split)
        graph.add_node(
            node_id,
            kind="operation",
            left_weight=left_weight,
            right_weight=right_weight,
            bias=float(rng.uniform(-0.1, 0.1)),
            local_noise=float(rng.uniform(-epsilon, epsilon)),
        )
        left_id = add_subtree(level + 1)
        right_id = add_subtree(level + 1)
        graph.add_edge(node_id, left_id, slot=0)
        graph.add_edge(node_id, right_id, slot=1)
        return node_id

    root = add_subtree(0)
    graph.graph["root"] = root
    assert nx.is_directed_acyclic_graph(graph)
    return graph


def evaluate_contract_graph(graph: nx.DiGraph, epsilon: float) -> tuple[float, float]:
    for node_id in reversed(list(nx.topological_sort(graph))):
        node = graph.nodes[node_id]
        if node["kind"] == "leaf":
            continue
        children = sorted(
            graph.successors(node_id), key=lambda child: graph.edges[node_id, child]["slot"]
        )
        left, right = (graph.nodes[child] for child in children)
        exact = (
            node["left_weight"] * left["exact"]
            + node["right_weight"] * right["exact"]
            + node["bias"]
        )
        approximate = (
            node["left_weight"] * left["approximate"]
            + node["right_weight"] * right["approximate"]
            + node["bias"]
            + node["local_noise"]
        )
        bound = (
            epsilon
            + abs(node["left_weight"]) * left["bound"]
            + abs(node["right_weight"]) * right["bound"]
        )
        node["exact"] = exact
        node["approximate"] = approximate
        node["bound"] = bound

    root = graph.nodes[graph.graph["root"]]
    return abs(root["approximate"] - root["exact"]), float(root["bound"])


def approximation_experiment() -> dict[str, object]:
    rng = np.random.default_rng(SEED + 2)
    epsilon = 0.02
    rho = 0.78
    depths = list(range(0, 11))
    actual_quantiles: list[list[float]] = []
    bounds: list[float] = []
    violations = 0

    for depth in depths:
        actual_errors: list[float] = []
        depth_bounds: list[float] = []
        for _ in range(180):
            graph = build_full_binary_contract_graph(depth, rho, epsilon, rng)
            actual, bound = evaluate_contract_graph(graph, epsilon)
            actual_errors.append(actual)
            depth_bounds.append(bound)
            if actual > bound + 1e-12:
                violations += 1
        actual_quantiles.append(
            [float(value) for value in np.quantile(actual_errors, [0.1, 0.5, 0.9])]
        )
        bounds.append(float(max(depth_bounds)))

    quantile_array = np.asarray(actual_quantiles)
    fig, axis = plt.subplots(figsize=(7.4, 4.6))
    axis.fill_between(
        depths,
        quantile_array[:, 0],
        quantile_array[:, 2],
        alpha=0.25,
        color="#118ab2",
        label="actual 10–90%",
    )
    axis.plot(depths, quantile_array[:, 1], color="#118ab2", marker="o", label="actual median")
    axis.plot(depths, bounds, color="#ef476f", marker="s", label="certified bound")
    axis.set_xlabel("composition-tree depth")
    axis.set_ylabel("root absolute error")
    axis.set_title(f"Contractive error transport (aggregate sensitivity ρ={rho})")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "error-bound-vs-depth.png", dpi=180)
    plt.close(fig)

    rho_values = np.linspace(0.1, 1.45, 120)
    depth_values = np.arange(1, 21)
    stability = np.empty((len(depth_values), len(rho_values)))
    for row, depth in enumerate(depth_values):
        for column, rho_value in enumerate(rho_values):
            if abs(rho_value - 1.0) < 1e-10:
                stability[row, column] = epsilon * depth
            else:
                stability[row, column] = epsilon * (
                    1.0 - rho_value**depth
                ) / (1.0 - rho_value)

    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    image = axis.imshow(
        np.log10(stability + 1e-12),
        aspect="auto",
        origin="lower",
        extent=[rho_values[0], rho_values[-1], depth_values[0], depth_values[-1]],
        cmap="magma",
    )
    axis.axvline(1.0, color="white", linestyle="--", linewidth=1.5, label="ρ = 1 boundary")
    axis.set_xlabel("aggregate child sensitivity ρ")
    axis.set_ylabel("tree depth")
    axis.set_title("log10 worst-case accumulated local error")
    fig.colorbar(image, ax=axis, label="log10 bound")
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "stability-region.png", dpi=180)
    plt.close(fig)

    return {
        "epsilon": epsilon,
        "rho": rho,
        "depths": depths,
        "actual_error_quantiles_10_50_90": actual_quantiles,
        "certified_bounds": bounds,
        "bound_violations": violations,
        "stability_grid_shape": list(stability.shape),
    }


def invalid_rewrite_experiment() -> dict[str, object]:
    rng = np.random.default_rng(SEED + 3)
    eta_values = np.linspace(0.0, 0.35, 80)
    scale_values = np.linspace(0.1, 2.5, 75)
    defects = np.empty((len(scale_values), len(eta_values)))

    def star(left: np.ndarray, right: np.ndarray, eta: float) -> np.ndarray:
        return left + right + eta * left**2 * right

    for row, scale in enumerate(scale_values):
        samples = rng.uniform(-scale, scale, size=(400, 3))
        x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
        for column, eta in enumerate(eta_values):
            left = star(star(x, y, eta), z, eta)
            right = star(x, star(y, z, eta), eta)
            defects[row, column] = float(np.max(np.abs(left - right)))

    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    image = axis.imshow(
        np.log10(defects + 1e-14),
        aspect="auto",
        origin="lower",
        extent=[eta_values[0], eta_values[-1], scale_values[0], scale_values[-1]],
        cmap="viridis",
    )
    axis.set_xlabel("nonassociative perturbation η")
    axis.set_ylabel("probe input scale")
    axis.set_title("Invalid associativity rewrite: log10 maximum defect")
    fig.colorbar(image, ax=axis, label="log10 max |(x⋆y)⋆z − x⋆(y⋆z)|")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "invalid-rewrite-boundary.png", dpi=180)
    plt.close(fig)

    positive_region = defects[:, 1:]
    return {
        "eta_range": [float(eta_values[0]), float(eta_values[-1])],
        "input_scale_range": [float(scale_values[0]), float(scale_values[-1])],
        "exact_associative_slice_max_defect": float(np.max(defects[:, 0])),
        "minimum_sampled_positive_eta_defect": float(np.min(positive_region)),
        "maximum_sampled_defect": float(np.max(defects)),
        "grid_shape": list(defects.shape),
    }


def fixed_family_minimal_probe_set() -> dict[str, object]:
    # Exact exhaustive check on a fixed, reproducible family.
    generators = [
        AffineMod(1, 0),
        AffineMod(1, 1),
        AffineMod(2, 0),
        AffineMod(2, 3),
        AffineMod(4, 1),
        AffineMod(5, 7),
        AffineMod(7, 2),
        AffineMod(9, 4),
    ]
    for size in range(PRIME + 1):
        separating_sets = [
            list(probes)
            for probes in itertools.combinations(range(PRIME), size)
            if signatures_are_separating(generators, probes)
        ]
        if separating_sets:
            return {
                "minimal_size": size,
                "number_of_minimal_sets": len(separating_sets),
                "first_five_sets": separating_sets[:5],
            }
    raise AssertionError("the full probe family must separate distinct functions")


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    signature_results = signature_phase_diagram()
    transfer_results = transfer_experiment()
    approximation_results = approximation_experiment()
    rewrite_results = invalid_rewrite_experiment()
    exact_probe_result = fixed_family_minimal_probe_set()

    assert approximation_results["bound_violations"] == 0
    assert all(rate == 0.0 for rate in transfer_results["false_commitment_rate"])
    assert rewrite_results["exact_associative_slice_max_defect"] < 1e-12

    result = {
        "status": "MEASURED: bounded synthetic experiments executed",
        "scientific_scope": (
            "Finite affine-symbol and linear-tree checks only; no natural-language, "
            "learned parser, or integrated Sentinel result."
        ),
        "seed": SEED,
        "versions": {
            "numpy": np.__version__,
            "matplotlib": __import__("matplotlib").__version__,
            "networkx": nx.__version__,
        },
        "signature_identifiability": signature_results,
        "ood_composition_transfer": transfer_results,
        "approximation_error": approximation_results,
        "invalid_rewrite": rewrite_results,
        "exact_fixed_family_probe_enumeration": exact_probe_result,
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
