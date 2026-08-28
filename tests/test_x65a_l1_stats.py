import sys
from fractions import Fraction

import pytest

sys.path.insert(0, "experiments")

from x65a import l1_stats as S


def test_paired_interval_is_exact_and_reproducible():
    a = [Fraction(1), Fraction(3, 4), Fraction(1, 2), Fraction(1)]
    b = [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1, 2)]
    x = S.paired_interval(a, b, reps=1000, seed=7)
    y = S.paired_interval(a, b, reps=1000, seed=7)
    assert x == y
    assert all(isinstance(x[k], Fraction) for k in ("lo", "delta", "hi"))
    assert x["delta"] == Fraction(1, 2)


def test_paired_interval_rejects_unpaired_or_float_metrics():
    with pytest.raises(ValueError):
        S.paired_interval([1], [])
    with pytest.raises(TypeError):
        S.paired_interval([0.5], [0])
