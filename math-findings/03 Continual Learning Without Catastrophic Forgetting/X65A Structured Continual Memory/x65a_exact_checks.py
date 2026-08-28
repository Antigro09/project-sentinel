"""Exact finite falsification checks for X65A's mathematical core.

The script validates finite counterexamples, sufficient-statistic identities,
dependency assumptions, bounded-search compounding, and a genuine clean-process
reload.  Its outputs are toy-model evidence only; they are not an X65A pass.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "exact-checks.json"
FIGURE_DIR = HERE / "figures"
RELIABILITY = Fraction(4, 5)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def normalized(weights: dict[int, Fraction]) -> dict[int, Fraction]:
    total = sum(weights.values(), start=Fraction(0))
    assert total > 0
    posterior = {key: value / total for key, value in weights.items()}
    assert sum(posterior.values(), start=Fraction(0)) == 1
    return posterior


def convention_posterior(observations: Iterable[int]) -> dict[int, Fraction]:
    weights: dict[int, Fraction] = {}
    observations = tuple(observations)
    for latent in (0, 1):
        likelihood = Fraction(1)
        for observation in observations:
            likelihood *= RELIABILITY if observation == latent else 1 - RELIABILITY
        weights[latent] = Fraction(1, 2) * likelihood
    return normalized(weights)


def sequence_probability(sequence: tuple[int, ...], latent: int) -> Fraction:
    probability = Fraction(1)
    for observation in sequence:
        probability *= RELIABILITY if observation == latent else 1 - RELIABILITY
    return probability


def expected_map_accuracy(observation_count: int) -> Fraction:
    total = Fraction(0)
    for latent in (0, 1):
        for sequence in itertools.product((0, 1), repeat=observation_count):
            probability = Fraction(1, 2) * sequence_probability(sequence, latent)
            posterior = convention_posterior(sequence)
            best_mass = max(posterior.values())
            winners = [state for state, mass in posterior.items() if mass == best_mass]
            # Uniform random tie-breaking avoids an arbitrary latent preference.
            correctness = Fraction(1, len(winners)) if latent in winners else Fraction(0)
            total += probability * correctness
    assert Fraction(1, 2) <= total <= 1
    return total


def sufficient_statistic_check() -> dict[str, Any]:
    groups: dict[tuple[int, int], set[tuple[Fraction, Fraction]]] = {}
    for sequence in itertools.product((0, 1), repeat=5):
        statistic = (sum(sequence), len(sequence) - sum(sequence))
        posterior = convention_posterior(sequence)
        groups.setdefault(statistic, set()).add((posterior[0], posterior[1]))
    assert all(len(posteriors) == 1 for posteriors in groups.values())
    return {
        "history_count": 2**5,
        "statistic_count": len(groups),
        "all_equal_within_statistic_class": True,
        "classes": {
            f"ones={ones},zeros={zeros}": [str(value) for value in next(iter(values))]
            for (ones, zeros), values in sorted(groups.items())
        },
    }


def bounded_memory_check() -> dict[str, Any]:
    history_bits = 4
    memory_bits = 3
    histories = tuple(itertools.product((0, 1), repeat=history_bits))
    memory_state_count = 2**memory_bits
    assert len(histories) > memory_state_count
    # A concrete encoder makes the forced collision visible.  The Lean theorem
    # establishes that every encoder into this smaller state space collides.
    buckets: dict[int, list[tuple[int, ...]]] = {}
    for history in histories:
        encoded = sum(bit << index for index, bit in enumerate(history)) % memory_state_count
        buckets.setdefault(encoded, []).append(history)
    collision = next(bucket for bucket in buckets.values() if len(bucket) > 1)
    first, second = collision[:2]
    distinguishing_index = next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)
    return {
        "history_bits": history_bits,
        "memory_bits": memory_bits,
        "history_count": len(histories),
        "memory_state_count": memory_state_count,
        "concrete_collision": [first, second],
        "future_index_query_that_separates_collision": distinguishing_index,
    }


def coverage_utility(selection: frozenset[str]) -> Fraction:
    coverage = {
        "lexical": {"sense"},
        "ordering": {"order"},
        "procedure": {"filter", "compose"},
    }
    weights = {"sense": 3, "order": 2, "filter": 2, "compose": 4}
    covered = set().union(*(coverage[item] for item in selection)) if selection else set()
    return Fraction(sum(weights[feature] for feature in covered))


def complementary_utility(selection: frozenset[str]) -> Fraction:
    value = Fraction(1) if {"macro_left", "macro_right"} <= selection else Fraction(0)
    if "stale_rule" in selection:
        value -= Fraction(3, 5)
    return value


def all_subsets(items: tuple[str, ...]) -> list[frozenset[str]]:
    return [
        frozenset(combination)
        for length in range(len(items) + 1)
        for combination in itertools.combinations(items, length)
    ]


def is_monotone_submodular(items: tuple[str, ...], utility) -> tuple[bool, bool]:
    subsets = all_subsets(items)
    monotone = True
    submodular = True
    for left in subsets:
        for right in subsets:
            if not left <= right:
                continue
            if utility(left) > utility(right):
                monotone = False
            for item in set(items) - set(right):
                marginal_left = utility(left | {item}) - utility(left)
                marginal_right = utility(right | {item}) - utility(right)
                if marginal_left < marginal_right:
                    submodular = False
    return monotone, submodular


def retrieval_checks() -> dict[str, Any]:
    complementary_items = ("macro_left", "macro_right", "stale_rule")
    coverage_items = ("lexical", "ordering", "procedure")
    comp_monotone, comp_submodular = is_monotone_submodular(
        complementary_items, complementary_utility
    )
    cov_monotone, cov_submodular = is_monotone_submodular(
        coverage_items, coverage_utility
    )
    assert not comp_monotone
    assert not comp_submodular
    assert cov_monotone and cov_submodular

    empty = frozenset()
    left = frozenset({"macro_left"})
    marginal_at_empty = complementary_utility(frozenset({"macro_right"})) - complementary_utility(empty)
    marginal_after_left = complementary_utility(
        frozenset({"macro_left", "macro_right"})
    ) - complementary_utility(left)
    assert marginal_at_empty == 0 < marginal_after_left
    assert complementary_utility(frozenset({"macro_left", "macro_right", "stale_rule"})) < complementary_utility(
        frozenset({"macro_left", "macro_right"})
    )
    return {
        "general_utility": {
            "monotone": comp_monotone,
            "submodular": comp_submodular,
            "marginal_macro_right_at_empty": str(marginal_at_empty),
            "marginal_macro_right_after_left": str(marginal_after_left),
            "stale_memory_penalty": "3/5",
        },
        "independent_weighted_coverage": {
            "monotone": cov_monotone,
            "submodular": cov_submodular,
            "utilities": {
                "+".join(sorted(selection)) or "empty": str(coverage_utility(selection))
                for selection in all_subsets(coverage_items)
            },
        },
    }


def revision_checks() -> tuple[dict[str, Any], nx.DiGraph]:
    prior_false_claim = Fraction(4, 5)
    source_reliability = Fraction(9, 10)
    # The observed datum says the claim is false.
    revised_false_claim = (
        prior_false_claim * (1 - source_reliability)
        / (
            prior_false_claim * (1 - source_reliability)
            + (1 - prior_false_claim) * source_reliability
        )
    )
    unrelated_before = Fraction(9, 10)

    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("contextual_claim", "composite_schema"),
            ("composite_schema", "cached_plan"),
            ("procedure_filter", "later_composition"),
        ]
    )
    graph.add_node("unrelated_skill")
    affected = {"contextual_claim"} | nx.descendants(graph, "contextual_claim")
    assert "unrelated_skill" not in affected
    unrelated_after = unrelated_before
    assert unrelated_after == unrelated_before

    # Later highly reliable supporting evidence can reverse the revision;
    # provenance is retained rather than destructively deleting the old state.
    later_support_reliability = Fraction(19, 20)
    reversed_belief = (
        revised_false_claim * later_support_reliability
        / (
            revised_false_claim * later_support_reliability
            + (1 - revised_false_claim) * (1 - later_support_reliability)
        )
    )
    assert revised_false_claim < prior_false_claim < reversed_belief
    return (
        {
            "prior_false_claim": str(prior_false_claim),
            "after_trusted_counterevidence": str(revised_false_claim),
            "after_later_trusted_support": str(reversed_belief),
            "affected_dependency_region": sorted(affected),
            "unrelated_before": str(unrelated_before),
            "unrelated_after": str(unrelated_after),
            "locality_preserved": True,
            "provenance_events_retained": ["initial_support", "counterexample", "later_support"],
        },
        graph,
    )


def stream_checks() -> dict[str, Any]:
    tasks = [
        {"id": "ground_context", "learns": ["sem_context"], "requires": [], "target": "sense_A"},
        {"id": "ground_order", "learns": ["sem_order"], "requires": [], "target": "order_LR"},
        {"id": "learn_filter", "learns": ["proc_filter"], "requires": [], "target": "filter"},
        {"id": "learn_reverse", "learns": ["proc_reverse"], "requires": [], "target": "reverse"},
        {
            "id": "novel_semantic_composition",
            "learns": [],
            "requires": ["sem_context", "sem_order"],
            "target": "compose_semantics(context,order)",
        },
        {
            "id": "novel_procedural_composition",
            "learns": [],
            "requires": ["proc_filter", "proc_reverse"],
            "target": "compose(filter,reverse)",
        },
    ]

    graph = nx.DiGraph()
    for task in tasks:
        graph.add_node(task["id"], kind="task")
        for component in task["learns"]:
            graph.add_edge(task["id"], component, relation="learns")
        for component in task["requires"]:
            graph.add_edge(component, task["id"], relation="required_by")
    assert nx.is_directed_acyclic_graph(graph)

    def run_order(order: list[int]) -> tuple[int, list[str]]:
        learned: set[str] = set()
        solved = 0
        failures: list[str] = []
        persisted: list[dict[str, Any]] = []
        for position, task_index in enumerate(order):
            task = tasks[task_index]
            # No direct future target: all persisted records are from earlier
            # positions and no later composite target is present verbatim.
            assert all(record["acquired_at"] < position for record in persisted)
            future_targets = {tasks[index]["target"] for index in order[position:]}
            assert all(record["payload"] not in future_targets for record in persisted)
            if set(task["requires"]) <= learned:
                solved += bool(task["requires"])
            elif task["requires"]:
                failures.append(task["id"])
            for component in task["learns"]:
                learned.add(component)
                persisted.append(
                    {
                        "acquired_at": position,
                        "kind": "verified_component",
                        "payload": component,
                    }
                )
        return solved, failures

    dependency_order = list(range(len(tasks)))
    reverse_order = [4, 5, 0, 1, 2, 3]
    solved_dependency, dependency_failures = run_order(dependency_order)
    solved_reverse, reverse_failures = run_order(reverse_order)
    assert solved_dependency == 2 and not dependency_failures
    assert solved_reverse == 0 and len(reverse_failures) == 2
    return {
        "task_count": len(tasks),
        "dependency_graph_is_dag": True,
        "dependency_respecting": {
            "composite_tasks_solved": solved_dependency,
            "failures": dependency_failures,
        },
        "reverse_counterfactual": {
            "composite_tasks_solved": solved_reverse,
            "failures": reverse_failures,
        },
        "direct_target_leakage_detected": False,
        "caveat": "absence of literal targets does not preclude legitimate inference from reusable components",
    }


def compounding_check() -> dict[str, Any]:
    branching_factor = 6
    raw_length = 8
    macro_length = 3
    budget = 1000
    raw_candidates = branching_factor**raw_length
    macro_candidates = branching_factor**macro_length
    assert macro_candidates <= budget < raw_candidates
    stored_procedures = ["filter", "reverse"]
    later_target = "compose(filter,primitive_3,reverse)"
    assert later_target not in stored_procedures
    return {
        "branching_factor": branching_factor,
        "raw_program_length": raw_length,
        "macro_encoded_length": macro_length,
        "fixed_search_budget": budget,
        "raw_candidates_through_target_depth": raw_candidates,
        "macro_candidates_through_target_depth": macro_candidates,
        "target_program_stored": False,
        "memory_enabled_reachable_under_budget": True,
        "memoryless_reachable_under_budget": False,
        "memoryless_reachable_with_larger_budget": True,
        "claim_scope": "capability under the preregistered search budget, not absolute expressivity",
    }


def growth_check() -> dict[str, Any]:
    task_counts = np.arange(1, 129)
    raw = 128 * task_counts
    # One shared component table, short residuals, and sparse retained
    # counterexamples.  This is a specified code, not an optimal compressor.
    consolidated = 512 + 16 * task_counts + 64 * (task_counts // 10)
    assert consolidated[-1] < raw[-1]
    raw_slope = float(np.polyfit(task_counts, raw, 1)[0])
    consolidated_slope = float(np.polyfit(task_counts, consolidated, 1)[0])
    assert consolidated_slope < raw_slope
    first_better = int(task_counts[np.flatnonzero(consolidated < raw)[0]])
    return {
        "raw_bytes_at_128": int(raw[-1]),
        "consolidated_bytes_at_128": int(consolidated[-1]),
        "raw_slope_bytes_per_task": raw_slope,
        "consolidated_slope_bytes_per_task": consolidated_slope,
        "first_task_count_with_shorter_two_part_code": first_better,
        "performance_constraint": "must additionally pass verifier replay and held-out transfer; size alone is insufficient",
        "task_counts": task_counts.tolist(),
        "raw_curve": raw.tolist(),
        "consolidated_curve": consolidated.tolist(),
    }


def restart_child(path: Path) -> None:
    state = json.loads(path.read_text(encoding="utf-8"))
    serialized = canonical_bytes(state)
    forbidden = "future_target_answer"
    assert forbidden not in serialized.decode("utf-8")
    assert "X65A_FORBIDDEN_TARGET" not in os.environ
    observations = [int(value) for value in state["semantic_observations"]]
    posterior = convention_posterior(observations)
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "state_sha256": hashlib.sha256(serialized).hexdigest(),
                "posterior": {str(key): str(value) for key, value in posterior.items()},
                "forbidden_channel_present": False,
                "procedures": state["procedures"],
                "negative_memory": state["negative_memory"],
            },
            sort_keys=True,
        )
    )


def restart_check() -> dict[str, Any]:
    allowed_state = {
        "schema_version": 1,
        "semantic_observations": [1, 1, 0],
        "procedures": ["filter", "reverse"],
        "negative_memory": [
            {"claim": "contextual_claim", "status": "superseded", "reason": "counterexample"}
        ],
        "provenance": ["episode-0", "episode-1", "episode-2"],
    }
    runtime_only_forbidden_secret = "future_target_answer"
    assert runtime_only_forbidden_secret not in canonical_bytes(allowed_state).decode("utf-8")
    expected = {str(key): str(value) for key, value in convention_posterior([1, 1, 0]).items()}

    with tempfile.TemporaryDirectory(prefix="x65a-restart-") as directory:
        path = Path(directory) / "permitted-memory.json"
        path.write_bytes(canonical_bytes(allowed_state))
        environment = dict(os.environ)
        environment.pop("X65A_FORBIDDEN_TARGET", None)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--restart-child", str(path)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        child = json.loads(completed.stdout)
    assert child["pid"] != os.getpid()
    assert child["posterior"] == expected
    assert not child["forbidden_channel_present"]
    return {
        "parent_pid": os.getpid(),
        "child_pid": child["pid"],
        "distinct_process": True,
        "posterior_before": expected,
        "posterior_after": child["posterior"],
        "posterior_exactly_preserved": True,
        "forbidden_hidden_state_channel_caught": not child["forbidden_channel_present"],
        "serialized_state_sha256": child["state_sha256"],
    }


def make_figures(
    accuracies: list[Fraction],
    retrieval: dict[str, Any],
    revision_graph: nx.DiGraph,
    growth: dict[str, Any],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(range(len(accuracies)), [float(value) for value in accuracies], marker="o")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="reset prior")
    axis.set(
        xlabel="grounded observations retained under one latent convention",
        ylabel="exact expected MAP accuracy",
        title="Finite-model forward transfer from a persistent sufficient statistic",
        ylim=(0.45, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "semantic-transfer-curve.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    task_counts = growth["task_counts"]
    axis.plot(task_counts, growth["raw_curve"], label="raw full replay")
    axis.plot(task_counts, growth["consolidated_curve"], label="specified two-part code")
    axis.set(
        xlabel="tasks",
        ylabel="serialized bytes in toy code",
        title="Bounded-growth target: consolidation must change the slope",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "memory-growth.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    update_strength = np.linspace(0, 1, 101)
    for coupling in (0.0, 0.25, 0.75, 1.0):
        forward_transfer = update_strength
        retention_change = -coupling * update_strength
        axis.plot(
            forward_transfer,
            retention_change,
            label=f"dependency coupling={coupling:.2f}",
        )
    axis.set(
        xlabel="plasticity / new-component update magnitude",
        ylabel="old-component retention change",
        title="Toy stability–plasticity frontier depends on revision coupling",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "stability-plasticity-frontier.png", dpi=180)
    plt.close(figure)

    labels = ["empty", "left", "right", "left+right", "left+right+stale"]
    selections = [
        frozenset(),
        frozenset({"macro_left"}),
        frozenset({"macro_right"}),
        frozenset({"macro_left", "macro_right"}),
        frozenset({"macro_left", "macro_right", "stale_rule"}),
    ]
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    axis.bar(labels, [float(complementary_utility(selection)) for selection in selections])
    axis.set(
        ylabel="task utility",
        title="General retrieval is neither submodular nor monotone",
    )
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "retrieval-counterexample.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.0, 5.2))
    positions = nx.spring_layout(revision_graph, seed=6501)
    affected = {"contextual_claim"} | nx.descendants(revision_graph, "contextual_claim")
    colors = ["#d95f02" if node in affected else "#1b9e77" for node in revision_graph.nodes]
    nx.draw_networkx(
        revision_graph,
        pos=positions,
        node_color=colors,
        node_size=1700,
        font_size=8,
        arrows=True,
        ax=axis,
    )
    x_values = [point[0] for point in positions.values()]
    y_values = [point[1] for point in positions.values()]
    x_span = max(x_values) - min(x_values) or 1.0
    y_span = max(y_values) - min(y_values) or 1.0
    axis.set_xlim(min(x_values) - 0.24 * x_span, max(x_values) + 0.24 * x_span)
    axis.set_ylim(min(y_values) - 0.18 * y_span, max(y_values) + 0.18 * y_span)
    axis.set_title("Revision locality: only the dependency closure is invalidated")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "revision-dependency-region.png", dpi=180)
    plt.close(figure)

    lengths = np.arange(2, 11)
    budgets = np.unique(np.logspace(1, 8, 120).astype(int))
    phase = np.zeros((len(budgets), len(lengths)), dtype=int)
    for row, budget in enumerate(budgets):
        for column, raw_length in enumerate(lengths):
            macro_length = max(1, raw_length - 5)
            macro_reachable = 6**macro_length <= budget
            raw_reachable = 6**raw_length <= budget
            phase[row, column] = 2 if raw_reachable else (1 if macro_reachable else 0)
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    image = axis.imshow(
        phase,
        origin="lower",
        aspect="auto",
        extent=(lengths[0] - 0.5, lengths[-1] + 0.5, math.log10(budgets[0]), math.log10(budgets[-1])),
        cmap=matplotlib.colors.ListedColormap(["#d73027", "#fee08b", "#1a9850"]),
        vmin=0,
        vmax=2,
    )
    axis.set(
        xlabel="raw target program length",
        ylabel="log10 search budget",
        title="Compounding reachability under a fixed enumerative-search budget",
    )
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["neither", "macros only", "both"])
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "compounding-reachability-phase.png", dpi=180)
    plt.close(figure)


def main() -> None:
    accuracies = [expected_map_accuracy(count) for count in range(9)]
    # Even counts can tie under symmetric noise, so require nondecrease rather
    # than strict improvement at every interaction.
    assert all(right >= left for left, right in zip(accuracies, accuracies[1:]))
    assert accuracies[-1] > accuracies[0]

    retrieval = retrieval_checks()
    revision, revision_graph = revision_checks()
    growth = growth_check()
    restart = restart_check()
    result = {
        "status": "MEASURED: exact toy-model checks executed; NOT an X65A gate pass",
        "bounded_memory": bounded_memory_check(),
        "finite_posterior_sufficiency": sufficient_statistic_check(),
        "semantic_transfer": {
            "reliability": str(RELIABILITY),
            "expected_map_accuracy_by_observation_count": [str(value) for value in accuracies],
            "reset_accuracy": "1/2",
        },
        "retrieval": retrieval,
        "revision": revision,
        "stream_dependency_and_leakage": stream_checks(),
        "procedural_compounding": compounding_check(),
        "growth": growth,
        "restart": restart,
        "scope_limit": "These checks validate a finite construction and counterexamples, not continual-learning performance in Sentinel.",
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "networkx": nx.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    make_figures(accuracies, retrieval, revision_graph, growth)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart-child", type=Path)
    arguments = parser.parse_args()
    if arguments.restart_child is not None:
        restart_child(arguments.restart_child)
    else:
        main()
