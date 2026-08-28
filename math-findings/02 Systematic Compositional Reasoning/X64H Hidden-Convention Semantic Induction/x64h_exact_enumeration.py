"""Exact finite-model checks for X64H identifiability and clarification bounds."""

from __future__ import annotations

import functools
import itertools
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import scipy
from scipy.stats import binom

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "exact-enumeration.json"
FIGURE_DIR = HERE / "figures"


def signatures(k: int, contexts: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((context >> atom) & 1 for context in contexts)
        for atom in range(k)
    )


def is_separating(k: int, contexts: tuple[int, ...]) -> bool:
    return len(set(signatures(k, contexts))) == k


def exact_separating_probability(k: int, m: int) -> float:
    total = 0
    separating = 0
    for family in itertools.product(range(1 << k), repeat=m):
        total += 1
        separating += is_separating(k, family)
    return separating / total


def closed_form_separating_probability(k: int, m: int) -> float:
    m = int(m)
    signature_count = 1 << m
    if signature_count < k:
        return 0.0
    numerator = math.prod(range(signature_count - k + 1, signature_count + 1))
    return numerator / (signature_count**k)


def constructive_contexts(k: int) -> tuple[int, ...]:
    m = math.ceil(math.log2(k))
    return tuple(
        sum(1 << atom for atom in range(k) if (atom >> bit) & 1)
        for bit in range(m)
    )


def apply_context(permutation: tuple[int, ...], context: int) -> int:
    surface = 0
    for atom, word in enumerate(permutation):
        if (context >> atom) & 1:
            surface |= 1 << word
    return surface


def partition(
    hypotheses: tuple[tuple[int, ...], ...], query: int
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    parts: dict[int, list[tuple[int, ...]]] = {}
    for hypothesis in hypotheses:
        parts.setdefault(apply_context(hypothesis, query), []).append(hypothesis)
    return tuple(tuple(part) for _answer, part in sorted(parts.items()))


def entropy_of_query(
    hypotheses: tuple[tuple[int, ...], ...], query: int
) -> float:
    n = len(hypotheses)
    return -sum(
        (len(part) / n) * math.log2(len(part) / n)
        for part in partition(hypotheses, query)
    )


def query_statistics(k: int) -> dict[str, float | int]:
    hypotheses = tuple(itertools.permutations(range(k)))
    queries = tuple(range(1, (1 << k) - 1))

    @functools.lru_cache(maxsize=None)
    def optimal(state: tuple[tuple[int, ...], ...]) -> float:
        if len(state) <= 1:
            return 0.0
        values = []
        for query in queries:
            parts = partition(state, query)
            if len(parts) <= 1:
                continue
            values.append(
                1.0
                + sum((len(part) / len(state)) * optimal(part) for part in parts)
            )
        return min(values)

    @functools.lru_cache(maxsize=None)
    def greedy(state: tuple[tuple[int, ...], ...]) -> float:
        if len(state) <= 1:
            return 0.0
        candidates = [
            (entropy_of_query(state, query), -query, query)
            for query in queries
            if len(partition(state, query)) > 1
        ]
        query = max(candidates)[2]
        return 1.0 + sum(
            (len(part) / len(state)) * greedy(part)
            for part in partition(state, query)
        )

    @functools.lru_cache(maxsize=None)
    def random_disagreement(
        state: tuple[tuple[int, ...], ...], available: tuple[int, ...]
    ) -> float:
        if len(state) <= 1:
            return 0.0
        informative = tuple(
            query for query in available if len(partition(state, query)) > 1
        )
        if not informative:
            return math.inf
        per_query = []
        for query in informative:
            remaining = tuple(q for q in available if q != query)
            expected_tail = sum(
                (len(part) / len(state)) * random_disagreement(part, remaining)
                for part in partition(state, query)
            )
            per_query.append(1.0 + expected_tail)
        return sum(per_query) / len(per_query)

    max_answers = max(len(partition(hypotheses, query)) for query in queries)
    entropy = math.log2(math.factorial(k))
    lower_bound = entropy / math.log2(max_answers)
    return {
        "k": k,
        "hypotheses": len(hypotheses),
        "posterior_entropy_bits": entropy,
        "largest_answer_alphabet": max_answers,
        "entropy_lower_bound_questions": lower_bound,
        "optimal_expected_questions": optimal(hypotheses),
        "greedy_information_gain_expected_questions": greedy(hypotheses),
        "random_disagreement_expected_questions": random_disagreement(
            hypotheses, queries
        ),
    }


def majority_bit_error(repetitions: int, noise: float) -> float:
    threshold = repetitions // 2 + 1
    exact = sum(
        math.comb(repetitions, errors)
        * noise**errors
        * (1 - noise) ** (repetitions - errors)
        for errors in range(threshold, repetitions + 1)
    )
    scipy_check = float(binom.sf(repetitions // 2, repetitions, noise))
    assert np.isclose(exact, scipy_check, rtol=1e-13, atol=1e-15)
    return exact


def noise_recovery(k: int, m: int, noise: float, repetitions: int) -> dict[str, float | int]:
    bit_error = majority_bit_error(repetitions, noise)
    exact_success = (1 - bit_error) ** (k * m)
    hoeffding_union_failure = min(
        1.0,
        k * m * math.exp(-2 * repetitions * (0.5 - noise) ** 2),
    )
    return {
        "repetitions": repetitions,
        "exact_all_signature_success": exact_success,
        "hoeffding_success_lower_bound": 1 - hoeffding_union_failure,
    }


def indistinguishable_example() -> dict[str, object]:
    k = 4
    contexts = (0b0011,)
    sig = signatures(k, contexts)
    duplicate = next(
        (a, b)
        for a in range(k)
        for b in range(a + 1, k)
        if sig[a] == sig[b]
    )
    identity = tuple(range(k))
    swapped = list(identity)
    a, b = duplicate
    swapped[a], swapped[b] = swapped[b], swapped[a]
    surface_identity = tuple(apply_context(identity, c) for c in contexts)
    surface_swapped = tuple(apply_context(tuple(swapped), c) for c in contexts)
    assert surface_identity == surface_swapped
    return {
        "contexts": contexts,
        "duplicate_atoms": duplicate,
        "identity_convention": identity,
        "swapped_convention": tuple(swapped),
        "same_surface_observations": surface_identity,
    }


def make_figures(query_stats: dict[str, float | int]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for k in (4, 8, 16):
        ms = np.arange(1, 9)
        probabilities = [closed_form_separating_probability(k, m) for m in ms]
        axis.plot(ms, probabilities, marker="o", label=f"k={k}")
    axis.set(
        xlabel="number of persistent task contexts m",
        ylabel="P(all atom signatures distinct)",
        title="Random context families become identifying only after enough separation",
        ylim=(-0.03, 1.03),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "identifiability-probability.png", dpi=180)
    plt.close(figure)

    rows = [noise_recovery(8, 3, 0.1, r) for r in range(1, 22, 2)]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(
        [row["repetitions"] for row in rows],
        [row["exact_all_signature_success"] for row in rows],
        marker="o",
        label="exact majority-decoding success",
    )
    axis.plot(
        [row["repetitions"] for row in rows],
        [row["hoeffding_success_lower_bound"] for row in rows],
        marker="s",
        label="Hoeffding + union lower bound",
    )
    axis.set(
        xlabel="repetitions per context",
        ylabel="probability all signatures are recovered",
        title="Noisy identification: exact finite probability and theorem bound",
        ylim=(-0.03, 1.03),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "noisy-signature-recovery.png", dpi=180)
    plt.close(figure)

    labels = ["entropy bound", "optimal", "greedy IG", "random split"]
    values = [
        query_stats["entropy_lower_bound_questions"],
        query_stats["optimal_expected_questions"],
        query_stats["greedy_information_gain_expected_questions"],
        query_stats["random_disagreement_expected_questions"],
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(labels, values, color=["#777777", "#2A9D8F", "#457B9D", "#E76F51"])
    axis.set(
        ylabel="expected clarification questions",
        title="Exact k=4 permutation identification",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "query-policy-comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    exact_vs_formula = []
    for m in range(1, 5):
        exact = exact_separating_probability(4, m)
        formula = closed_form_separating_probability(4, m)
        assert abs(exact - formula) < 1e-15
        exact_vs_formula.append({"k": 4, "m": m, "exact": exact, "formula": formula})

    constructive = {}
    for k in range(2, 17):
        contexts = constructive_contexts(k)
        assert is_separating(k, contexts)
        assert len(contexts) == math.ceil(math.log2(k))
        constructive[str(k)] = {
            "contexts": contexts,
            "m": len(contexts),
            "cardinality_lower_bound": math.ceil(math.log2(k)),
        }

    query_stats = query_statistics(4)
    assert query_stats["greedy_information_gain_expected_questions"] == 2.0
    assert query_stats["optimal_expected_questions"] == 2.0
    assert (
        query_stats["random_disagreement_expected_questions"]
        > query_stats["greedy_information_gain_expected_questions"]
    )

    noise_rows = [noise_recovery(8, 3, 0.1, r) for r in range(1, 22, 2)]
    delta = 0.05
    raw_bound = math.ceil(
        math.log((8 * 3) / delta) / (2 * (0.5 - 0.1) ** 2)
    )
    repetition_bound = raw_bound if raw_bound % 2 else raw_bound + 1

    result = {
        "status": "MEASURED: exact finite enumeration and analytic probabilities",
        "tool_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "random_context_identifiability": exact_vs_formula,
        "constructive_minimum_contexts": constructive,
        "nonidentifiable_counterexample": indistinguishable_example(),
        "active_query": query_stats,
        "bounded_noise": {
            "k": 8,
            "m": 3,
            "bit_flip_probability": 0.1,
            "target_failure_probability": delta,
            "hoeffding_repetition_bound_odd": repetition_bound,
            "rows": noise_rows,
        },
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    make_figures(query_stats)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
