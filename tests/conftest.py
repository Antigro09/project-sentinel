"""Shared test configuration.

`arc_agi` is an optional dependency: the Phase-1 ARC-AGI environments need it and
nothing in Phase-2 does. Before this file, two modules failed to COLLECT and two tests
FAILED when it was absent, so a clean checkout reported "963 passed, 2 failed" and the
two failures had to be explained in prose every time. A missing optional package is a
SKIP, not a failure, and the distinction is now made in code.

Run the required suite with `-m "not optional_dependency"`; run everything to see the
skips.
"""

from __future__ import annotations

import importlib.util

import pytest

OPTIONAL_PACKAGES = ("arc_agi",)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "optional_dependency(package): needs an optional package; skipped when absent")


def has(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


requires_arc_agi = pytest.mark.skipif(
    not has("arc_agi"),
    reason="optional dependency 'arc_agi' is not installed")


def pytest_collection_modifyitems(config: pytest.Config, items) -> None:
    """Tag anything marked `optional_dependency` so the manifests can be split."""
    for item in items:
        for marker in item.iter_markers(name="optional_dependency"):
            package = marker.args[0] if marker.args else "arc_agi"
            if not has(package):
                item.add_marker(pytest.mark.skip(
                    reason=f"optional dependency {package!r} is not installed"))
