"""Calibration bookkeeping for the three-way uncertainty decomposition.

Scale 0 does not calibrate anything -- there is no trained model to calibrate --
but the plumbing has to exist and be tested, because the failure it guards
against only shows up later: an uncertainty head that tracks error in
distribution and says nothing useful outside it.

The reliability table here is the standard binned one. It is kept deliberately
plain so that the number it reports can be checked by hand against the bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sentinel.wm.latent_contract import ContractViolation, UncertaintyTriple


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    mean_correct: float

    @property
    def gap(self) -> float:
        return abs(self.mean_confidence - self.mean_correct)


@dataclass(frozen=True, slots=True)
class CalibrationTable:
    bins: tuple[CalibrationBin, ...]
    total: int

    @property
    def expected_calibration_error(self) -> float:
        if self.total == 0:
            return float("nan")
        return sum(b.count * b.gap for b in self.bins) / self.total

    def canonical_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "expected_calibration_error": self.expected_calibration_error,
            "bins": [
                {
                    "lower": b.lower,
                    "upper": b.upper,
                    "count": b.count,
                    "mean_confidence": b.mean_confidence,
                    "mean_correct": b.mean_correct,
                }
                for b in self.bins
            ],
        }


def build_calibration_table(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> CalibrationTable:
    if len(confidences) != len(correct):
        raise ContractViolation(
            f"{len(confidences)} confidences against {len(correct)} outcomes"
        )
    if n_bins <= 0:
        raise ContractViolation(f"n_bins must be positive, got {n_bins}")
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Last bin is closed on the right so confidence 1.0 is counted.
        members = [
            (c, o)
            for c, o in zip(confidences, correct)
            if (lo <= c < hi) or (i == n_bins - 1 and c == hi)
        ]
        if members:
            mean_conf = sum(c for c, _ in members) / len(members)
            mean_corr = sum(1.0 for _, o in members if o) / len(members)
        else:
            mean_conf = mean_corr = 0.0
        bins.append(CalibrationBin(lo, hi, len(members), mean_conf, mean_corr))
    return CalibrationTable(tuple(bins), len(confidences))


def ensemble_disagreement(predictions: Sequence[Sequence[float]]) -> float:
    """Mean squared deviation from the ensemble mean.

    This is the `U(b,a)` diagnostic of the strategy document, and Lean checks
    only that it is non-negative. That is all it is: a non-negative number that
    goes up when members differ. Shared misspecification drives it to zero while
    every member is wrong, so it is never sufficient evidence of open-world
    detection on its own.
    """
    import numpy as np

    matrix = np.asarray(predictions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ContractViolation(
            f"need at least two members with matching widths, got shape {matrix.shape}"
        )
    mean = matrix.mean(axis=0, keepdims=True)
    return float(((matrix - mean) ** 2).mean())


def triple_from_head(
    log_variance: float,
    ensemble_spread: float,
    residual_outside_class: float,
) -> UncertaintyTriple:
    """Map raw head outputs onto the three stored components."""
    import math

    return UncertaintyTriple(
        aleatoric=float(math.exp(min(log_variance, 60.0))),
        epistemic=float(max(ensemble_spread, 0.0)),
        inadequacy=float(max(residual_outside_class, 0.0)),
    )
