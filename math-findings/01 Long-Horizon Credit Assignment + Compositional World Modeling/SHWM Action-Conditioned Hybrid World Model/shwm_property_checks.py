"""Property-based checks for SHWM's finite helper claims."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "property-checks.json"
COMMON = settings(max_examples=200, derandomize=True, database=None, deadline=None)


def rollout_error(lipschitz: Fraction, epsilon: Fraction, horizon: int) -> Fraction:
    value = Fraction(0)
    for _ in range(horizon):
        value = epsilon + lipschitz * value
    return value


@COMMON
@given(
    bits=st.integers(min_value=0, max_value=16),
    dimension=st.integers(min_value=0, max_value=16),
)
def test_finite_latent_cardinality(bits: int, dimension: int) -> None:
    assert (2**bits) ** dimension == 2 ** (bits * dimension)


@COMMON
@given(
    branching=st.integers(min_value=1, max_value=12),
    horizon=st.integers(min_value=0, max_value=20),
)
def test_sequence_count_monotone_in_horizon(branching: int, horizon: int) -> None:
    assert branching**horizon <= branching ** (horizon + 1)


@COMMON
@given(
    transitions=st.integers(min_value=0, max_value=10_000_000),
    dimension=st.integers(min_value=1, max_value=4096),
    bytes_per_coordinate=st.sampled_from([1, 2, 4, 8]),
)
def test_cache_arithmetic_linear(
    transitions: int, dimension: int, bytes_per_coordinate: int
) -> None:
    one = transitions * dimension * bytes_per_coordinate
    two = (2 * transitions) * dimension * bytes_per_coordinate
    assert two == 2 * one


@COMMON
@given(
    numerator_a=st.integers(min_value=1, max_value=10_000),
    numerator_b=st.integers(min_value=1, max_value=10_000),
)
def test_two_model_posterior_normalizes(numerator_a: int, numerator_b: int) -> None:
    evidence = Fraction(numerator_a + numerator_b)
    posterior_a = Fraction(numerator_a, 1) / evidence
    posterior_b = Fraction(numerator_b, 1) / evidence
    assert posterior_a + posterior_b == 1


@COMMON
@given(
    l_num=st.integers(min_value=0, max_value=15),
    e_num=st.integers(min_value=0, max_value=100),
    horizon=st.integers(min_value=0, max_value=40),
)
def test_rollout_recurrence_closed_form(l_num: int, e_num: int, horizon: int) -> None:
    lipschitz = Fraction(l_num, 10)
    epsilon = Fraction(e_num, 1000)
    expected = epsilon * sum(
        (lipschitz**index for index in range(horizon)), Fraction(0)
    )
    assert rollout_error(lipschitz, epsilon, horizon) == expected


@COMMON
@given(predicted=st.integers(), observed=st.integers())
def test_exact_verifier_is_equality(predicted: int, observed: int) -> None:
    accepted = predicted == observed
    if predicted != observed:
        assert not accepted
    else:
        assert accepted


def main() -> None:
    properties = [
        test_finite_latent_cardinality,
        test_sequence_count_monotone_in_horizon,
        test_cache_arithmetic_linear,
        test_two_model_posterior_normalizes,
        test_rollout_recurrence_closed_form,
        test_exact_verifier_is_equality,
    ]
    for property_test in properties:
        property_test()

    result = {
        "status": "MEASURED: deterministic Hypothesis checks passed; finite helper properties only",
        "properties": [test.__name__ for test in properties],
        "max_examples_per_property": 200,
        "derandomized": True,
        "total_configured_examples": len(properties) * 200,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
