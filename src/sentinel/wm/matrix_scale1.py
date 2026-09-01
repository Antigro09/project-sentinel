"""The frozen Scale-1A-1 screen matrix, expressed as code.

Same contract as `matrix.py` at Scale 0: the run matrix is a document, and this
module is that document in executable form, so "the run matched the matrix" is a
test rather than a sentence in a report.

Two things here are load-bearing and neither is obvious.

The first is the alias table. The specification names interfaces in short form
(`qwen_spatial_slots`, `learned_cnn_spatial_slots`); the implementation named
them independently and earlier (`qwen3_vl_4b_spatial_slots`,
`small_learned_cnn`), and those implementation names are already baked into four
frozen artifacts, including the freeze manifest. Renaming the classes to match
the specification would silently invalidate those digests. So the correspondence
is written down and *checked against the interfaces that actually build*, rather
than either side being quietly bent to fit the other. A typo in this table is a
loud failure, not a cell that can never run.

The second is that reactive controls carry a different role and no parameter
target. They are policies, not 50M world models, and the specification is
explicit that they are controls. Giving them a distinct role means a reactive
result cannot be summed into a world-model count by accident -- the arithmetic
below reports the three populations separately and never as one number.

Wall time, memory, and throughput are measured outcomes, not matched quantities,
for the same reason as at Scale 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from sentinel.wm.arms import ARMS, ArmSpec
from sentinel.wm.interfaces import build_interfaces
from sentinel.wm.latent_contract import ContractViolation
from sentinel.wm.versioning import digest_of

MATRIX_VERSION = "scale-1a-1-screen-v1"

# ---- 1. interfaces, under both names ---------------------------------------------------

INTERFACE_ALIASES: Mapping[str, str] = {
    "qwen_spatial_slots": "qwen3_vl_4b_spatial_slots",
    "gemma_spatial_slots": "gemma3_4b_spatial_slots",
    "learned_cnn_spatial_slots": "small_learned_cnn",
    "fixed_random_spatial_projection": "fixed_random_spatial_projection",
}
"""Specification name -> implemented `interface.name`.

The fourth entry is an identity mapping and is kept explicit rather than
special-cased, so the table can be read as the complete four-interface list.
"""

SCREEN_INTERFACES: tuple[str, ...] = (
    "qwen_spatial_slots",
    "gemma_spatial_slots",
    "learned_cnn_spatial_slots",
    "fixed_random_spatial_projection",
)

POOLING_ABLATION_INTERFACE = "qwen_mean_pool"
POOLING_ABLATION_IMPLEMENTATION = "qwen3_vl_4b_mean_pool"
POOLING_ABLATION_ARM = "continuous_action_recurrent"

TRAINABLE_TARGET = 50_000_000
PARAMETER_TOLERANCE = 0.01
SCREEN_SEEDS: tuple[int, ...] = (7700, 7701, 7702)
"""Screen seeds, deliberately disjoint from Scale 0's development seeds.

Final Scale-1 environment seeds are not named here and must not be. Access to
them runs through `provenance.FrozenRunContext.load_final_seeds`, which refuses
unless a manifest was committed after the freeze.
"""

SCALE0_DEVELOPMENT_SEEDS: tuple[int, ...] = (6600, 6601, 6602)


def resolve_interface(spec_name: str) -> str:
    """Map a specification interface name onto the implemented one."""
    try:
        return INTERFACE_ALIASES[spec_name]
    except KeyError:
        raise ContractViolation(
            f"{spec_name!r} is not a screen interface; the matrix names "
            f"{list(SCREEN_INTERFACES)}"
        ) from None


def assert_aliases_resolve() -> dict[str, str]:
    """Every alias must name an interface that `build_interfaces` actually returns.

    Without this the matrix could freeze a cell whose interface does not exist,
    and the failure would not surface until that cell was scheduled -- which, in
    an 87-cell screen, could be hours in.
    """
    built = {interface.name for interface in build_interfaces()}
    missing = {
        spec: impl for spec, impl in INTERFACE_ALIASES.items() if impl not in built
    }
    if POOLING_ABLATION_IMPLEMENTATION not in built:
        missing[POOLING_ABLATION_INTERFACE] = POOLING_ABLATION_IMPLEMENTATION
    if missing:
        raise ContractViolation(
            f"alias table names interfaces that do not build: {sorted(missing.items())}; "
            f"built interfaces are {sorted(built)}"
        )
    return dict(INTERFACE_ALIASES)


# ---- 2. cells and workloads -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scale1Cell:
    """One (interface, arm) combination at one role."""

    interface: str
    arm: str
    role: str  # "screen", "reactive_control", or "pooling_ablation"
    target_parameters: int | None

    @property
    def implementation(self) -> str:
        if self.interface == POOLING_ABLATION_INTERFACE:
            return POOLING_ABLATION_IMPLEMENTATION
        return resolve_interface(self.interface)

    @property
    def cell_id(self) -> str:
        return f"{self.interface}.{self.arm}.{self.role}"

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "interface": self.interface,
            "implementation": self.implementation,
            "arm": self.arm,
            "role": self.role,
            "target_parameters": self.target_parameters,
        }


@dataclass(frozen=True, slots=True)
class Scale1Workload:
    cell: Scale1Cell
    seed: int

    @property
    def workload_id(self) -> str:
        return f"{self.cell.cell_id}.s{self.seed}"

    @property
    def is_world_model(self) -> bool:
        """Reactive controls are policies. They never count as world-model runs."""
        return self.cell.role != "reactive_control"

    def canonical_dict(self) -> dict[str, Any]:
        return {"cell": self.cell.canonical_dict(), "seed": self.seed}


def screen_cells() -> tuple[Scale1Cell, ...]:
    """4 interfaces x 6 arms = 24 cells."""
    return tuple(
        Scale1Cell(interface, arm.name, "screen", TRAINABLE_TARGET)
        for interface in SCREEN_INTERFACES
        for arm in ARMS
    )


def reactive_control_cells() -> tuple[Scale1Cell, ...]:
    """One reactive policy per interface. Controls, so no parameter target."""
    return tuple(
        Scale1Cell(interface, "reactive_policy", "reactive_control", None)
        for interface in SCREEN_INTERFACES
    )


def pooling_ablation_cells() -> tuple[Scale1Cell, ...]:
    """Mean pooling is the spatial interface with position deleted."""
    return (
        Scale1Cell(
            POOLING_ABLATION_INTERFACE, POOLING_ABLATION_ARM, "pooling_ablation", TRAINABLE_TARGET
        ),
    )


def all_cells() -> tuple[Scale1Cell, ...]:
    return screen_cells() + reactive_control_cells() + pooling_ablation_cells()


def all_workloads() -> tuple[Scale1Workload, ...]:
    return tuple(
        Scale1Workload(cell, seed) for cell in all_cells() for seed in SCREEN_SEEDS
    )


def matrix_arithmetic() -> dict[str, int]:
    """The counts the specification states in words, recomputed from the factors."""
    seeds = len(SCREEN_SEEDS)
    screen = len(screen_cells())
    reactive = len(reactive_control_cells())
    pooling = len(pooling_ablation_cells())
    return {
        "interfaces": len(SCREEN_INTERFACES),
        "arms": len(ARMS),
        "seeds": seeds,
        "screen_cells": screen,
        "screen_workloads": screen * seeds,
        "reactive_control_cells": reactive,
        "reactive_control_workloads": reactive * seeds,
        "pooling_ablation_cells": pooling,
        "pooling_ablation_workloads": pooling * seeds,
        "world_model_workloads": (screen + pooling) * seeds,
        "total_workloads": (screen + reactive + pooling) * seeds,
    }


def assert_no_final_seed(seeds: tuple[int, ...]) -> None:
    """The screen runs on screen seeds. Overlap with Scale 0's is also a violation.

    Reusing a Scale-0 development seed would make the screen's environments
    partly ones whose outcomes have already been seen.
    """
    overlap = sorted(set(seeds) & set(SCALE0_DEVELOPMENT_SEEDS))
    if overlap:
        raise ContractViolation(
            f"screen seeds {overlap} collide with Scale-0 development seeds; "
            f"the screen must run on unseen environments"
        )
    unexpected = sorted(set(seeds) - set(SCREEN_SEEDS))
    if unexpected:
        raise ContractViolation(
            f"seeds {unexpected} are not screen seeds {list(SCREEN_SEEDS)}; "
            f"final seeds are reachable only through a post-freeze manifest"
        )


def matrix_digest() -> str:
    return digest_of(
        {
            "version": MATRIX_VERSION,
            "aliases": dict(sorted(INTERFACE_ALIASES.items())),
            "workloads": [w.canonical_dict() for w in all_workloads()],
            "arithmetic": matrix_arithmetic(),
        }
    )


def frozen_matrix_report() -> dict[str, Any]:
    return {
        "version": MATRIX_VERSION,
        "digest": matrix_digest(),
        "aliases": assert_aliases_resolve(),
        "arithmetic": matrix_arithmetic(),
        "seeds": list(SCREEN_SEEDS),
        "arms": [arm.name for arm in ARMS],
        "interfaces": list(SCREEN_INTERFACES),
        "workload_ids": [w.workload_id for w in all_workloads()],
    }
