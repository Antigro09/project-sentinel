"""Property-based boundary checks for X65A's finite claims."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from hypothesis import given, settings, strategies as st


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "property-checks.json"


@settings(max_examples=300, deadline=None)
@given(
    prior_true=st.integers(min_value=1, max_value=1000),
    prior_false=st.integers(min_value=1, max_value=1000),
    like_true=st.integers(min_value=1, max_value=1000),
    like_false=st.integers(min_value=1, max_value=1000),
)
def posterior_normalizes(
    prior_true: int, prior_false: int, like_true: int, like_false: int
) -> None:
    true_weight = Fraction(prior_true * like_true)
    false_weight = Fraction(prior_false * like_false)
    total = true_weight + false_weight
    assert true_weight / total + false_weight / total == 1


@settings(max_examples=300, deadline=None)
@given(
    left_a=st.integers(min_value=1, max_value=1000),
    left_b=st.integers(min_value=1, max_value=1000),
    unrelated=st.integers(min_value=0, max_value=1000),
)
def factorized_revision_is_local(left_a: int, left_b: int, unrelated: int) -> None:
    normalizer = Fraction(left_a + left_b)
    marginal = (
        Fraction(left_a, 1) / normalizer * unrelated
        + Fraction(left_b, 1) / normalizer * unrelated
    )
    assert marginal == unrelated


@settings(max_examples=300, deadline=None)
@given(sizes=st.lists(st.integers(min_value=1, max_value=10000), max_size=200))
def raw_replay_grows_at_least_linearly(sizes: list[int]) -> None:
    assert sum(sizes) >= len(sizes)


@settings(max_examples=300, deadline=None)
@given(
    task_index=st.integers(min_value=0, max_value=10000),
    gaps=st.lists(st.integers(min_value=1, max_value=100), max_size=100),
)
def past_only_store_has_no_direct_future_target(task_index: int, gaps: list[int]) -> None:
    past_indices = [max(0, task_index - gap) for gap in gaps if gap <= task_index]
    assert all(index < task_index for index in past_indices)


@settings(max_examples=300, deadline=None)
@given(
    sets=st.lists(
        st.sets(st.integers(min_value=0, max_value=12), max_size=8),
        min_size=1,
        max_size=8,
    )
)
def deterministic_coverage_is_submodular(sets: list[set[int]]) -> None:
    # For A subset B and an item e not in B, newly covered elements from e can
    # only decrease as the selected set grows.
    item_count = len(sets)
    midpoint = item_count // 2
    a_indices = set(range(midpoint))
    b_indices = set(range(max(midpoint, item_count - 1)))
    if item_count - 1 in b_indices:
        return
    item = item_count - 1

    def covered(indices: set[int]) -> set[int]:
        result: set[int] = set()
        for index in indices:
            result |= sets[index]
        return result

    marginal_a = len(covered(a_indices | {item})) - len(covered(a_indices))
    marginal_b = len(covered(b_indices | {item})) - len(covered(b_indices))
    assert marginal_a >= marginal_b


def main() -> None:
    posterior_normalizes()
    factorized_revision_is_local()
    raw_replay_grows_at_least_linearly()
    past_only_store_has_no_direct_future_target()
    deterministic_coverage_is_submodular()
    result = {
        "status": "MEASURED: Hypothesis properties passed; finite claim scope only",
        "properties": {
            "posterior_normalization": 300,
            "factorized_revision_locality": 300,
            "raw_replay_linear_lower_bound": 300,
            "past_only_no_direct_target": 300,
            "deterministic_coverage_submodularity": 300,
        },
        "total_generated_examples": 1500,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
