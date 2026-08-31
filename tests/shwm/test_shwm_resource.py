"""Scale-0 gate: the resource numbers mean what the report says they mean.

One of these exists because the first 48-workload run got it wrong. `ru_maxrss`
is a high-water mark for the life of the process, so using it as a per-workload
peak made a 50M model that used 1.6 GiB of device memory appear to cost 12.2 GiB
— the sum of everything that had run before it. A reader deciding whether the
matrix fits in 112 GiB would have been reading the wrong column.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from sentinel.wm import matrix as M
from sentinel.wm.resource import (
    ResourceReport,
    directory_bytes,
    estimate_training_memory,
    measure,
    mlx_memory,
    process_resident_bytes,
)


def test_the_workload_peak_is_the_device_peak_and_the_process_peak_is_separate():
    report = ResourceReport(label="probe")
    report.mlx_peak_bytes = 2 * 1024**3
    report.peak_resident_bytes = 30 * 1024**3
    assert report.peak_unified_memory_bytes == 2 * 1024**3
    assert report.process_peak_resident_bytes == 30 * 1024**3
    serialised = report.canonical_dict()
    assert serialised["workload_peak_gib"] == pytest.approx(2.0)
    assert serialised["process_peak_resident_gib"] == pytest.approx(30.0)
    assert serialised["process_peak_is_cumulative"] is True


def test_the_device_peak_is_reset_between_measurements():
    """The property that makes it a per-workload figure at all."""
    big = ResourceReport(label="big")
    with measure("big", big):
        held = mx.zeros((4096, 4096), dtype=mx.float32)  # 64 MiB
        mx.eval(held)
    del held

    small = ResourceReport(label="small")
    with measure("small", small):
        held = mx.zeros((256, 256), dtype=mx.float32)
        mx.eval(held)
    del held

    assert big.mlx_peak_bytes > small.mlx_peak_bytes, (
        "the device peak did not reset, so it is a cumulative figure and cannot be "
        "compared across workloads"
    )
    assert small.peak_resident_bytes >= big.peak_resident_bytes, (
        "ru_maxrss is expected to be monotonic; if it fell, this test's premise is wrong"
    )


def test_the_estimate_uses_twelve_bytes_per_parameter_and_reports_its_parts():
    estimate = estimate_training_memory(50_000_000)
    assert estimate["bytes_per_parameter"] == 12
    assert estimate["model_bytes"] == 50_000_000 * 2
    assert estimate["gradient_bytes"] == 50_000_000 * 2
    assert estimate["optimizer_bytes"] == 50_000_000 * 8
    assert estimate["total_bytes"] == sum(
        estimate[k] for k in ("model_bytes", "gradient_bytes", "optimizer_bytes", "activation_bytes")
    )


def test_a_four_bit_weight_array_is_not_reported_as_a_training_requirement():
    """The strategy document's warning, made a property of the accounting.

    Weight bytes are one of four terms and never stand alone; the estimator
    cannot be asked for a figure that omits gradients and accumulators.
    """
    estimate = estimate_training_memory(70_000_000_000, weight_bytes=1)
    assert estimate["model_bytes"] < estimate["total_bytes"]
    assert estimate["optimizer_bytes"] > estimate["model_bytes"]


def test_the_report_puts_the_measurement_next_to_the_estimate():
    report = ResourceReport(label="probe")
    report.mlx_peak_bytes = 4 * 1024**3
    report.estimated_model_bytes = 1 * 1024**3
    report.estimated_optimizer_bytes = 1 * 1024**3
    serialised = report.canonical_dict()
    assert serialised["measured_over_estimated"] == pytest.approx(2.0)


def test_an_empty_estimate_does_not_divide_by_zero():
    assert ResourceReport(label="p").canonical_dict()["measured_over_estimated"] is None


def test_directory_bytes_counts_files_and_tolerates_absence(tmp_path):
    assert directory_bytes(tmp_path / "missing") == 0
    (tmp_path / "a.bin").write_bytes(b"x" * 100)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"y" * 50)
    assert directory_bytes(tmp_path) == 150


def test_the_process_figure_is_the_one_checked_against_the_per_process_ceiling():
    """The ceiling is stated per process, so the cumulative figure is correct there."""
    assert M.check_resource_envelope(M.PEAK_MEMORY_LIMIT_BYTES, 0, 0.0) == []
    over = M.check_resource_envelope(M.PEAK_MEMORY_LIMIT_BYTES + 1, 0, 0.0)
    assert any("peak process memory" in f for f in over)


def test_mlx_memory_reports_active_peak_and_cache_separately():
    memory = mlx_memory()
    assert set(memory) == {"active_bytes", "peak_bytes", "cache_bytes"}
    assert all(isinstance(v, int) and v >= 0 for v in memory.values())
    assert process_resident_bytes() > 0
