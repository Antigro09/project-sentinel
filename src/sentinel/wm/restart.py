"""Declared run state, checkpointing, and detection of everything undeclared.

The Scale-0 restart gate is not "does a checkpoint load". It is "does an
interrupted run continue as if it had never stopped, using only state that was
written down". Those differ by exactly the things nobody remembers to save: a
global random stream, a module-level cache, a counter living in a closure.

So there are two mechanisms here. `DeclaredRunState` is the allowlist -- if it
is not in this record it does not survive, and the checksum file makes a
truncated or edited checkpoint fail closed rather than load a corrupted tensor.
`ProcessStateAudit` is the tripwire for the other direction: it digests the
mutable module-level containers of the packages a run touches, and reports any
that moved. A run that leans on an undeclared global shows up as a changed
module whose name is not on the declared list.

The MLX global random stream deserves a specific mention because it is the
easiest one to get wrong. `mx.random.state` cannot be round-tripped by
assignment, so every stochastic path in a matrix-shaped run threads an explicit
key, and that key is part of the declared state.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from sentinel.wm.latent_contract import ContractViolation
from sentinel.wm.versioning import canonical_json, digest_file, digest_of

STATE_FILENAME = "state.json"
MODEL_FILENAME = "model.safetensors"
OPTIMIZER_FILENAME = "optimizer.safetensors"
CHECKSUM_FILENAME = "checksums.json"


class CheckpointCorruption(RuntimeError):
    """A checkpoint does not match its recorded checksums. It is not loaded."""


class UndeclaredState(RuntimeError):
    """Process state changed that no checkpoint records."""


@dataclass(frozen=True, slots=True)
class DeclaredRunState:
    """Everything a restart is allowed to depend on.

    `prng_key` is a two-word MLX key rather than a seed: a seed says where the
    stream started, and a restart needs to know where it currently is.
    """

    update_index: int
    prng_key: tuple[int, int]
    batch_cursor: int
    permutation_digest: str
    config_digest: str
    objective_digest: str
    data_digest: str
    split_manifest_digest: str
    planner_account: Mapping[str, Any]
    gate_ledger: Mapping[str, Any]
    verifier_ledger: Mapping[str, Any]
    pending_counterexamples: tuple[Mapping[str, Any], ...] = ()
    loss_history: tuple[float, ...] = ()

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "update_index": int(self.update_index),
            "prng_key": [int(v) for v in self.prng_key],
            "batch_cursor": int(self.batch_cursor),
            "permutation_digest": self.permutation_digest,
            "config_digest": self.config_digest,
            "objective_digest": self.objective_digest,
            "data_digest": self.data_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "planner_account": dict(self.planner_account),
            "gate_ledger": dict(self.gate_ledger),
            "verifier_ledger": dict(self.verifier_ledger),
            "pending_counterexamples": [dict(c) for c in self.pending_counterexamples],
            "loss_history": [float(v) for v in self.loss_history],
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())

    @property
    def key(self) -> mx.array:
        return mx.array(list(self.prng_key), dtype=mx.uint32)


def key_to_tuple(key: mx.array) -> tuple[int, int]:
    import numpy as np

    values = np.asarray(key).reshape(-1).tolist()
    if len(values) != 2:
        raise ContractViolation(f"expected a two-word MLX key, got {len(values)} words")
    return int(values[0]), int(values[1])


def _atomic_write(path: Path, text: str) -> None:
    handle = tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, prefix=f".{path.name}-", suffix=".tmp"
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


def save_run_state(
    directory: Path,
    state: DeclaredRunState,
    model,
    optimizer,
) -> dict[str, str]:
    """Write the checkpoint and its checksums. Returns the checksum map."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    parameters = {k: v for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(str(directory / MODEL_FILENAME), parameters)

    optimizer_state = {
        k: v for k, v in tree_flatten(optimizer.state) if isinstance(v, mx.array)
    }
    mx.save_safetensors(str(directory / OPTIMIZER_FILENAME), optimizer_state)

    _atomic_write(directory / STATE_FILENAME, canonical_json(state.canonical_dict()) + "\n")

    checksums = {
        name: digest_file(directory / name)
        for name in (STATE_FILENAME, MODEL_FILENAME, OPTIMIZER_FILENAME)
    }
    _atomic_write(directory / CHECKSUM_FILENAME, canonical_json(checksums) + "\n")
    return checksums


def verify_checkpoint(directory: Path) -> dict[str, str]:
    """Recompute every checksum. Raises before anything is deserialised."""
    directory = Path(directory)
    checksum_path = directory / CHECKSUM_FILENAME
    if not checksum_path.exists():
        raise CheckpointCorruption(f"{directory} has no {CHECKSUM_FILENAME}")
    recorded = json.loads(checksum_path.read_text())
    for name, expected in recorded.items():
        path = directory / name
        if not path.exists():
            raise CheckpointCorruption(f"{name} is missing from {directory}")
        actual = digest_file(path)
        if actual != expected:
            raise CheckpointCorruption(
                f"{name} hashes to {actual[:24]}..., recorded {expected[:24]}...; "
                "the checkpoint is not loaded"
            )
    return recorded


def load_run_state(directory: Path, model, optimizer) -> DeclaredRunState:
    """Verify, then restore. Nothing is deserialised before the checksums pass."""
    directory = Path(directory)
    verify_checkpoint(directory)

    document = json.loads((directory / STATE_FILENAME).read_text())
    state = DeclaredRunState(
        update_index=document["update_index"],
        prng_key=tuple(document["prng_key"]),  # type: ignore[arg-type]
        batch_cursor=document["batch_cursor"],
        permutation_digest=document["permutation_digest"],
        config_digest=document["config_digest"],
        objective_digest=document["objective_digest"],
        data_digest=document["data_digest"],
        split_manifest_digest=document["split_manifest_digest"],
        planner_account=document["planner_account"],
        gate_ledger=document["gate_ledger"],
        verifier_ledger=document["verifier_ledger"],
        pending_counterexamples=tuple(document["pending_counterexamples"]),
        loss_history=tuple(document["loss_history"]),
    )

    parameters = mx.load(str(directory / MODEL_FILENAME))
    model.update(tree_unflatten(list(parameters.items())))

    optimizer_state = mx.load(str(directory / OPTIMIZER_FILENAME))
    if optimizer_state:
        merged = dict(tree_flatten(optimizer.state))
        merged.update(optimizer_state)
        optimizer.state = tree_unflatten(list(merged.items()))
    mx.eval(model.parameters(), optimizer.state)
    return state


# ---- undeclared process state ------------------------------------------------


AUDITED_MODULES: tuple[str, ...] = (
    "sentinel.wm.cache",
    "sentinel.wm.dataset",
    "sentinel.wm.models",
    "sentinel.wm.objective",
    "sentinel.wm.planner_bridge",
    "sentinel.wm.verifier_bridge",
    "sentinel.wm.encoder",
    "sentinel.env.adapters.synthetic_control",
    "sentinel.env.adapters.procedural_visual",
)

_MUTABLE_TYPES = (dict, list, set, bytearray)


def _module_state_digest(module_name: str) -> str:
    """Digest of the mutable module-level containers of one module.

    Constants -- tuples, frozensets, strings, numbers -- are excluded because
    they cannot be the channel; a run cannot smuggle state through an immutable
    global. Callables and classes are excluded for the same reason, and because
    including them would make every import order change look like a leak.
    """
    module = importlib.import_module(module_name)
    entries: dict[str, Any] = {}
    for name in dir(module):
        if name.startswith("__"):
            continue
        value = getattr(module, name, None)
        if isinstance(value, _MUTABLE_TYPES):
            try:
                entries[name] = canonical_json(value)
            except Exception:
                entries[name] = f"<unserialisable {type(value).__name__} len={len(value)}>"
    return digest_of(entries)


@dataclass
class ProcessStateAudit:
    """Snapshot the mutable globals of the audited modules and diff them."""

    modules: tuple[str, ...] = AUDITED_MODULES
    baseline: dict[str, str] = field(default_factory=dict)

    def capture(self) -> dict[str, str]:
        snapshot = {name: _module_state_digest(name) for name in self.modules}
        if not self.baseline:
            self.baseline = dict(snapshot)
        return snapshot

    def changed(self) -> list[str]:
        current = {name: _module_state_digest(name) for name in self.modules}
        return sorted(
            name for name, digest in current.items() if self.baseline.get(name) != digest
        )

    def assert_no_undeclared_state(self, declared: Iterable[str] = ()) -> None:
        offending = [name for name in self.changed() if name not in set(declared)]
        if offending:
            raise UndeclaredState(
                f"module-level mutable state changed during the run in {offending}; "
                "a restart cannot reproduce state that no checkpoint records"
            )


def assert_restartable(output_used_global_rng: bool) -> None:
    """Refuse a matrix-shaped run whose sampling used the global stream."""
    if output_used_global_rng:
        raise UndeclaredState(
            "the forward pass sampled from the global MLX random stream, which no "
            "checkpoint records; thread an explicit key instead"
        )
