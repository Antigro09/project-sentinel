"""Property-based checks and counterexample search for OCI's finite model."""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import find, given, settings, strategies as st


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "property-checks.json"
PRIME = 17


def apply_affine(pair: tuple[int, int], value: int) -> int:
    a, b = pair
    return (a * value + b) % PRIME


def compose_affine(
    outer: tuple[int, int], inner: tuple[int, int]
) -> tuple[int, int]:
    outer_a, outer_b = outer
    inner_a, inner_b = inner
    return (
        (outer_a * inner_a) % PRIME,
        (outer_a * inner_b + outer_b) % PRIME,
    )


@settings(max_examples=500, deadline=None, derandomize=True)
@given(
    st.tuples(st.integers(0, PRIME - 1), st.integers(0, PRIME - 1)),
    st.tuples(st.integers(0, PRIME - 1), st.integers(0, PRIME - 1)),
    st.tuples(st.integers(0, PRIME - 1), st.integers(0, PRIME - 1)),
    st.integers(0, PRIME - 1),
)
def check_affine_composition_associative(
    first: tuple[int, int],
    second: tuple[int, int],
    third: tuple[int, int],
    value: int,
) -> None:
    left = compose_affine(compose_affine(first, second), third)
    right = compose_affine(first, compose_affine(second, third))
    assert left == right
    assert apply_affine(left, value) == apply_affine(right, value)


def star(left: int, right: int, eta: float = 0.1) -> float:
    return left + right + eta * left**2 * right


def associativity_defect(triple: tuple[int, int, int]) -> float:
    x, y, z = triple
    return star(star(x, y), z) - star(x, star(y, z))


def main() -> None:
    check_affine_composition_associative()
    counterexample = find(
        st.tuples(
            st.integers(-3, 3),
            st.integers(-3, 3),
            st.integers(-3, 3),
        ),
        lambda triple: abs(associativity_defect(triple)) > 1e-9,
        settings=settings(max_examples=1000, deadline=None, derandomize=True),
    )
    result = {
        "status": "MEASURED: Hypothesis properties executed",
        "affine_associativity_examples": 500,
        "invalid_rewrite_counterexample": list(counterexample),
        "invalid_rewrite_defect": associativity_defect(counterexample),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
