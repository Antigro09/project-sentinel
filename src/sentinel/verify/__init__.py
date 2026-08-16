"""Verification layer — the reward signal.

Deterministic, LLM-free, and fast. Everything the system learns is scored
here, so this package is deliberately strict and deliberately dull.
"""

from .evidence import EvidenceCoverage, evidence_coverage
from .report import CellStats, StepResult, VerificationReport
from .verifier import Verifier, compare, observed_outcome, verify

__all__ = [
    "EvidenceCoverage",
    "evidence_coverage",
    "CellStats",
    "StepResult",
    "VerificationReport",
    "Verifier",
    "compare",
    "observed_outcome",
    "verify",
]
