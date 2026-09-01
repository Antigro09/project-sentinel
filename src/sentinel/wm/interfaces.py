"""F. Eight observation interfaces, all emitting the same slot shape.

Every non-oracle interface produces (16, 256). That constraint is what makes the
comparison a comparison: an arm with more slots or a wider slot has more
capacity, and more capacity is indistinguishable from a better representation.

Three of these are not learned and say so. A fixed random projection is a
frozen matrix drawn once; calling it a learned representation would credit
pretraining for something that never saw data. Only the small CNN has trainable
parameters, and they are counted into the trainable budget.

The mean-pool interfaces broadcast their single vector across all 16 slots. That
is not a trick to satisfy the shape -- it is what mean pooling *is* once the
slots are laid out, and it makes the ablation legible: the pooled arm is the
spatial arm with every slot forced equal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from sentinel.wm.latent_contract import ContractViolation, ObservationEnvelope
from sentinel.wm.packet import SLOT_COUNT, SLOT_WIDTH
from sentinel.wm.versioning import digest_of

SLOT_GRID = 4
assert SLOT_GRID * SLOT_GRID == SLOT_COUNT


@dataclass
class InterfaceContext:
    """One step's raw material, so each backbone is run once per observation."""

    observation: ObservationEnvelope
    frame: np.ndarray
    visual_tokens: dict[str, np.ndarray] = field(default_factory=dict)
    truth: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class SlotInterface(Protocol):
    name: str
    learned: bool
    evaluator_only: bool

    @property
    def trainable_parameters(self) -> int: ...

    def slots(self, context: InterfaceContext) -> np.ndarray: ...


def _fixed_projection(rows: int, columns: int, tag: str) -> np.ndarray:
    """A frozen random matrix, reproducible from its tag. Never trained."""
    seed = int(digest_of({"projection": tag, "rows": rows, "columns": columns})[7:15], 16)
    generator = np.random.default_rng(seed)
    return (generator.normal(size=(rows, columns)) / np.sqrt(rows)).astype(np.float32)


def _block_pool(grid: np.ndarray, target: int) -> np.ndarray:
    """Mean-pool a (S, S, W) token grid down to (target, target, W)."""
    side = grid.shape[0]
    if side % target:
        raise ContractViolation(
            f"a {side}x{side} token grid does not divide into {target}x{target} blocks"
        )
    factor = side // target
    return grid.reshape(target, factor, target, factor, grid.shape[-1]).mean(axis=(1, 3))


def _tokens_to_grid(tokens: np.ndarray) -> np.ndarray:
    side = int(round(tokens.shape[0] ** 0.5))
    if side * side != tokens.shape[0]:
        raise ContractViolation(
            f"{tokens.shape[0]} visual tokens do not form a square grid, so no "
            "coordinate mapping exists"
        )
    return tokens.reshape(side, side, tokens.shape[-1])


def _check(slots: np.ndarray, name: str) -> np.ndarray:
    if slots.shape != (SLOT_COUNT, SLOT_WIDTH):
        raise ContractViolation(
            f"{name} emitted {slots.shape}, every interface must emit "
            f"({SLOT_COUNT}, {SLOT_WIDTH})"
        )
    return slots.astype(np.float32)


# ---- 1. raw low-resolution spatial ------------------------------------------------


@dataclass
class RawLowResSpatial:
    name: str = "raw_lowres_spatial"
    learned: bool = False
    evaluator_only: bool = False

    @property
    def trainable_parameters(self) -> int:
        return 0

    def slots(self, context: InterfaceContext) -> np.ndarray:
        frame = context.frame.astype(np.float32) / 255.0
        side = frame.shape[0]
        block = side // SLOT_GRID
        cells = frame.reshape(SLOT_GRID, block, SLOT_GRID, block, 3).transpose(0, 2, 1, 3, 4)
        flat = cells.reshape(SLOT_COUNT, -1)
        if flat.shape[1] > SLOT_WIDTH:
            raise ContractViolation("a raw block does not fit in a slot")
        padded = np.zeros((SLOT_COUNT, SLOT_WIDTH), dtype=np.float32)
        padded[:, : flat.shape[1]] = flat
        return _check(padded, self.name)


# ---- 2. fixed random spatial projection --------------------------------------------


@dataclass
class FixedRandomSpatialProjection:
    name: str = "fixed_random_spatial_projection"
    learned: bool = False
    evaluator_only: bool = False
    _projection: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def trainable_parameters(self) -> int:
        return 0

    def slots(self, context: InterfaceContext) -> np.ndarray:
        frame = context.frame.astype(np.float32) / 255.0
        side = frame.shape[0]
        block = side // SLOT_GRID
        cells = frame.reshape(SLOT_GRID, block, SLOT_GRID, block, 3).transpose(0, 2, 1, 3, 4)
        flat = cells.reshape(SLOT_COUNT, -1)
        if self._projection is None:
            self._projection = _fixed_projection(flat.shape[1], SLOT_WIDTH, "raw-blocks")
        return _check(flat @ self._projection, self.name)


# ---- 3. small learned spatial encoder ----------------------------------------------


@dataclass
class SmallLearnedSpatialEncoder:
    """The only interface with trainable parameters, counted into the budget."""

    name: str = "small_learned_cnn"
    learned: bool = True
    evaluator_only: bool = False
    seed: int = 6600
    _model: Any = field(default=None, init=False, repr=False)

    def _build(self):
        if self._model is not None:
            return self._model
        import mlx.core as mx
        import mlx.nn as nn

        class Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Strides chosen so a 24x24 frame lands exactly on the 4x4 slot
                # grid: 24 -> 12 -> 4. Emitting 6x6 and pooling afterwards would
                # need a ragged pool, and a ragged pool would give this interface
                # a slightly different receptive field from the others.
                self.a = nn.Conv2d(3, 32, 3, stride=2, padding=1)
                self.b = nn.Conv2d(32, 64, 3, stride=3, padding=1)
                self.c = nn.Conv2d(64, SLOT_WIDTH, 3, stride=1, padding=1)

            def __call__(self, x: mx.array) -> mx.array:
                x = nn.gelu(self.a(x))
                x = nn.gelu(self.b(x))
                return self.c(x)

        mx.random.seed(self.seed)
        model = Encoder()
        mx.eval(model.parameters())
        self._model = model
        return model

    @property
    def trainable_parameters(self) -> int:
        from mlx.utils import tree_flatten

        return int(sum(v.size for _, v in tree_flatten(self._build().trainable_parameters())))

    def slots(self, context: InterfaceContext) -> np.ndarray:
        import mlx.core as mx

        model = self._build()
        frame = context.frame.astype(np.float32) / 255.0
        features = model(mx.array(frame)[None])
        mx.eval(features)
        grid = np.asarray(features, dtype=np.float32)[0]
        pooled = _block_pool(grid, SLOT_GRID)
        return _check(pooled.reshape(SLOT_COUNT, SLOT_WIDTH), self.name)


# ---- 4-7. backbone interfaces --------------------------------------------------------


@dataclass
class BackboneMeanPool:
    """The Scale-0 interface, kept as an ablation.

    Every slot is the same vector, which is what mean pooling leaves behind once
    the slots are laid out. Stated that way the ablation reads correctly: this is
    the spatial arm with position deleted.
    """

    encoder_id: str
    name: str = ""
    learned: bool = False
    evaluator_only: bool = False
    _projection: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.encoder_id}_mean_pool"

    @property
    def trainable_parameters(self) -> int:
        return 0

    def slots(self, context: InterfaceContext) -> np.ndarray:
        tokens = context.visual_tokens.get(self.encoder_id)
        if tokens is None:
            raise ContractViolation(f"{self.name}: no visual tokens supplied for this step")
        pooled = tokens.mean(axis=0)
        if self._projection is None:
            self._projection = _fixed_projection(pooled.shape[0], SLOT_WIDTH, f"pool-{self.encoder_id}")
        projected = pooled @ self._projection
        return _check(np.repeat(projected[None, :], SLOT_COUNT, axis=0), self.name)


@dataclass
class BackboneSpatialSlots:
    """Coordinate-preserving: the token grid pooled into a 4x4 slot layout."""

    encoder_id: str
    name: str = ""
    learned: bool = False
    evaluator_only: bool = False
    _projection: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.encoder_id}_spatial_slots"

    @property
    def trainable_parameters(self) -> int:
        return 0

    def slots(self, context: InterfaceContext) -> np.ndarray:
        tokens = context.visual_tokens.get(self.encoder_id)
        if tokens is None:
            raise ContractViolation(f"{self.name}: no visual tokens supplied for this step")
        grid = _block_pool(_tokens_to_grid(tokens), SLOT_GRID)
        if self._projection is None:
            self._projection = _fixed_projection(
                grid.shape[-1], SLOT_WIDTH, f"slots-{self.encoder_id}"
            )
        return _check((grid @ self._projection).reshape(SLOT_COUNT, SLOT_WIDTH), self.name)


# ---- 8. evaluator-only oracle ---------------------------------------------------------


@dataclass
class OracleStructuredState:
    """Evaluator-only upper bound. Never an admissible model input.

    It exists to calibrate the probes: a variable the oracle cannot deliver is
    one the probe cannot read, and a table where the oracle does not win is a
    table that is not measuring representation quality.
    """

    name: str = "oracle_structured_state"
    learned: bool = False
    evaluator_only: bool = True

    @property
    def trainable_parameters(self) -> int:
        return 0

    def slots(self, context: InterfaceContext) -> np.ndarray:
        truth = context.truth
        if not truth:
            raise ContractViolation("the oracle interface needs evaluator-only truth")
        values = np.zeros((SLOT_COUNT, SLOT_WIDTH), dtype=np.float32)
        keys = sorted(k for k, v in truth.items() if isinstance(v, (int, float, bool)))
        for index, key in enumerate(keys[:SLOT_COUNT]):
            values[index, 0] = float(truth[key])
            values[index, 1 + (index % (SLOT_WIDTH - 1))] = 1.0
        return _check(values, self.name)


def build_interfaces(encoder_ids: tuple[str, ...] = ("qwen3_vl_4b", "gemma3_4b")) -> list[Any]:
    interfaces: list[Any] = [
        RawLowResSpatial(),
        FixedRandomSpatialProjection(),
        SmallLearnedSpatialEncoder(),
    ]
    for encoder_id in encoder_ids:
        interfaces.append(BackboneMeanPool(encoder_id=encoder_id))
        interfaces.append(BackboneSpatialSlots(encoder_id=encoder_id))
    interfaces.append(OracleStructuredState())
    return interfaces


def interface_report(interfaces: list[Any]) -> dict[str, Any]:
    return {
        "slot_count": SLOT_COUNT,
        "slot_width": SLOT_WIDTH,
        "interfaces": [
            {
                "name": i.name,
                "learned": i.learned,
                "evaluator_only": i.evaluator_only,
                "trainable_parameters": i.trainable_parameters,
            }
            for i in interfaces
        ],
        "total_trainable_adapter_parameters": sum(
            i.trainable_parameters for i in interfaces if not i.evaluator_only
        ),
        "note": (
            "A fixed random projection is a frozen matrix drawn once. It is not a "
            "learned representation and its trainable count is zero."
        ),
    }
