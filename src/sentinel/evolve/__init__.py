"""Scaffold self-modification, with an archive and an overseer.

Configuration-space search, not code rewriting -- see `genome.py` for why
that boundary is where it is. Promotion is decided on held-out worlds and
every version is retained, because the documented failure mode of this
layer is a change that looks like an improvement and is not.
"""

from .archive import Archive, Version
from .genome import ScaffoldGenome
from .search import ACTION_PRICE, Evaluation, evaluate_genome, evolve

__all__ = [
    "ACTION_PRICE",
    "Archive",
    "Evaluation",
    "ScaffoldGenome",
    "Version",
    "evaluate_genome",
    "evolve",
]
