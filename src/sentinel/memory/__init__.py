"""Accumulation — claim 4, and the one personal AGI depends on entirely.

A skill library that reorders search rather than asserting answers, and the
cost-to-mastery curve that says whether any of it is working.
"""

from .curve import CurveResult, WorldCost, run_sequence
from .library import Entry, SkillLibrary
from .signature import Signature

__all__ = [
    "CurveResult",
    "Entry",
    "Signature",
    "SkillLibrary",
    "WorldCost",
    "run_sequence",
]
