"""Measured resource accounting.

Every number the Scale-0 report is required to carry is measured here rather
than estimated, and the two are kept in the same record so the gap between them
is visible. The strategy document is explicit that a raw four-bit weight array is
not a training-memory requirement and a nominal hidden width is not a parameter
count; the way to keep those honest is to print the estimate next to the
measurement and let the ratio speak.

Two peaks are reported and they answer different questions, which is why taking
the larger of them was wrong. `mlx_peak_bytes` is reset before each measurement,
so it is the cost *of that workload*. `process_peak_resident_bytes` comes from
`ru_maxrss`, which is a high-water mark for the life of the process and is
therefore monotonic by construction -- across a 48-workload sequence it climbs
whatever each workload does, and reporting it as a per-workload figure attributed
12 GiB to a model that used 1.6.

The per-process ceiling is checked against the process figure, because that is
what the ceiling is about. The per-workload table uses the device figure, because
that is what a reader comparing two arms needs.
"""

from __future__ import annotations

import os
import resource as _resource
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import mlx.core as mx


def process_resident_bytes() -> int:
    """Peak resident set of this process, in bytes.

    macOS reports `ru_maxrss` in bytes; Linux reports kibibytes. The platform
    check matters: getting it wrong is a factor of 1024 in a report that is
    checked against a hard ceiling.
    """
    peak = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    import sys

    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def mlx_memory() -> dict[str, int]:
    return {
        "active_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
    }


def directory_bytes(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


@dataclass
class ResourceReport:
    """One workload's measured cost, with the estimates it should be judged against."""

    label: str
    wall_seconds: float = 0.0
    cold_load_seconds: float = 0.0
    peak_resident_bytes: int = 0
    mlx_peak_bytes: int = 0
    mlx_active_bytes: int = 0
    mlx_cache_bytes: int = 0
    trainable_parameters: int = 0
    frozen_parameters: int = 0
    parameter_bytes_measured: int = 0
    estimated_model_bytes: int = 0
    estimated_optimizer_bytes: int = 0
    estimated_activation_bytes: int = 0
    throughput: dict[str, float] = field(default_factory=dict)
    cache_report: dict[str, Any] = field(default_factory=dict)
    planner_account: dict[str, Any] = field(default_factory=dict)
    artifact_bytes: int = 0
    failures: list[str] = field(default_factory=list)
    retries: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def peak_unified_memory_bytes(self) -> int:
        """This workload's peak device allocation. Reset before the measurement."""
        return self.mlx_peak_bytes

    @property
    def process_peak_resident_bytes(self) -> int:
        """Whole-process high-water mark. Monotonic; not a per-workload cost."""
        return self.peak_resident_bytes

    @property
    def estimated_total_bytes(self) -> int:
        return (
            self.estimated_model_bytes
            + self.estimated_optimizer_bytes
            + self.estimated_activation_bytes
        )

    def canonical_dict(self) -> dict[str, Any]:
        estimated = self.estimated_total_bytes
        measured = self.peak_unified_memory_bytes
        return {
            "label": self.label,
            "wall_seconds": self.wall_seconds,
            "cold_load_seconds": self.cold_load_seconds,
            "process_peak_resident_bytes": self.peak_resident_bytes,
            "process_peak_resident_gib": self.peak_resident_bytes / 1024**3,
            "process_peak_is_cumulative": True,
            "mlx_peak_bytes": self.mlx_peak_bytes,
            "mlx_peak_gib": self.mlx_peak_bytes / 1024**3,
            "mlx_active_bytes": self.mlx_active_bytes,
            "mlx_cache_bytes": self.mlx_cache_bytes,
            "workload_peak_gib": measured / 1024**3,
            "trainable_parameters": self.trainable_parameters,
            "frozen_parameters": self.frozen_parameters,
            "parameter_bytes_measured": self.parameter_bytes_measured,
            "estimated_model_bytes": self.estimated_model_bytes,
            "estimated_optimizer_bytes": self.estimated_optimizer_bytes,
            "estimated_activation_bytes": self.estimated_activation_bytes,
            "estimated_total_bytes": estimated,
            "measured_over_estimated": measured / estimated if estimated else None,
            "throughput": dict(self.throughput),
            "cache": dict(self.cache_report),
            "planner": dict(self.planner_account),
            "artifact_bytes": self.artifact_bytes,
            "artifact_gib": self.artifact_bytes / 1024**3,
            "failures": list(self.failures),
            "retries": self.retries,
            "notes": list(self.notes),
        }


def estimate_training_memory(
    trainable_parameters: int,
    *,
    weight_bytes: int = 2,
    gradient_bytes: int = 2,
    accumulator_bytes: int = 8,
    batch_positions: int = 0,
    activation_width: int = 0,
    activation_layers: int = 0,
    activation_bytes: int = 2,
) -> dict[str, int]:
    """The Adam-style accounting from the strategy document, applied to one model.

    bf16 weights and gradients with fp32 first and second moments come to twelve
    bytes per parameter. Activations are estimated separately and are the term
    most likely to be wrong, which is exactly why the report prints the measured
    peak beside it.
    """
    model = trainable_parameters * weight_bytes
    gradients = trainable_parameters * gradient_bytes
    accumulators = trainable_parameters * accumulator_bytes
    activations = batch_positions * activation_width * activation_layers * activation_bytes
    return {
        "model_bytes": model,
        "gradient_bytes": gradients,
        "optimizer_bytes": accumulators,
        "activation_bytes": activations,
        "total_bytes": model + gradients + accumulators + activations,
        "bytes_per_parameter": weight_bytes + gradient_bytes + accumulator_bytes,
    }


@contextmanager
def measure(label: str, report: ResourceReport | None = None) -> Iterator[ResourceReport]:
    """Time a block and record the memory peak it reached."""
    report = report or ResourceReport(label=label)
    mx.reset_peak_memory()
    started = time.perf_counter()
    try:
        yield report
    finally:
        report.wall_seconds += time.perf_counter() - started
        memory = mlx_memory()
        report.mlx_peak_bytes = max(report.mlx_peak_bytes, memory["peak_bytes"])
        report.mlx_active_bytes = memory["active_bytes"]
        report.mlx_cache_bytes = memory["cache_bytes"]
        report.peak_resident_bytes = max(report.peak_resident_bytes, process_resident_bytes())
