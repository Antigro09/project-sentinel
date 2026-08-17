"""The tiny recursive core — the heart of the architecture bet.

Everything before this was apparatus. This is the part that either learns
to infer a world's hidden structure from observation, or does not, and the
Phase 3 gate is written to tell those apart honestly.
"""

from .data import (
    Dataset,
    build_dataset,
    iterate_batches,
    load_dataset,
    load_split,
    majority_baseline,
    save_dataset,
)
from .encoding import (
    CROP,
    HEADS,
    MAX_TRANSITIONS,
    MechanicLabels,
    encode_history,
    encode_world,
    label_names,
)
from .model import CoreConfig, TinyRecursiveCore, loss_fn
from .train import EpochResult, GateResult, TrainConfig, evaluate, run_gate, train

__all__ = [
    "CROP",
    "HEADS",
    "MAX_TRANSITIONS",
    "CoreConfig",
    "Dataset",
    "EpochResult",
    "GateResult",
    "MechanicLabels",
    "TinyRecursiveCore",
    "TrainConfig",
    "build_dataset",
    "encode_history",
    "encode_world",
    "evaluate",
    "iterate_batches",
    "label_names",
    "load_dataset",
    "load_split",
    "loss_fn",
    "majority_baseline",
    "run_gate",
    "save_dataset",
    "train",
]
