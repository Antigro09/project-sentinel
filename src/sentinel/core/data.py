"""Training data for the core. No LLM required.

This is the part worth pausing on: **Phase 3 does not need the bootstrap
corpus.** The generator knows every world's true mechanics, so labels are
exact, free, and unlimited. The LLM corpus teaches a different skill —
writing world-model *programs* — and is not what the core is learning here.

That reframes the eight nights of corpus generation as optional for this
experiment, and it means the decisive test can run today rather than after
a fortnight of compute.

Two multipliers keep the data cheap. Worlds are reused from the cached
split (already generated and solvability-checked), and each world yields
several episodes under different exploration seeds — different evidence
about the same rules, which is exactly the invariance the core should
learn.

The holdout withholds whole mechanic *combinations*, not just seeds.
Unseen-seed accuracy measures interpolation over rules already taught;
only unseen-combination accuracy speaks to generalization, and that is the
number the Phase 3 gate is judged on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sentinel.bootstrap.teacher import make_training_history
from sentinel.gen.spec import WorldSpec
from sentinel.verify.evidence import evidence_coverage

from .encoding import encode_world


@dataclass(frozen=True, slots=True)
class Dataset:
    grids: np.ndarray
    actions: np.ndarray
    labels: np.ndarray
    world_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.labels)

    def summary(self) -> str:
        return f"{len(self)} episodes from {len(set(self.world_ids))} worlds"

    def label_balance(self) -> dict[str, dict[int, int]]:
        from .encoding import HEADS

        out: dict[str, dict[int, int]] = {}
        for i, (name, _) in enumerate(HEADS):
            values, counts = np.unique(self.labels[:, i], return_counts=True)
            out[name] = {int(v): int(c) for v, c in zip(values, counts)}
        return out


def load_split(path: str | Path) -> dict[str, list[WorldSpec]]:
    """Read cached world specs produced by build_corpus.py."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, list[WorldSpec]] = {}
    for entry in data["worlds"]:
        out.setdefault(entry["split"], []).append(WorldSpec.from_json(entry["spec"]))
    return out


def build_dataset(
    specs: list[WorldSpec],
    episodes_per_world: int = 3,
    limit: int | None = None,
    require_strong_evidence: bool = True,
    verbose: bool = False,
) -> Dataset:
    """Encode episodes for each world.

    Episodes whose evidence cannot distinguish action-dependent behaviour
    are dropped. Their labels would still be correct, but nothing in the
    observations supports them — training on those teaches the core to
    guess from priors rather than read the evidence, which is the exact
    habit this architecture is meant to avoid.
    """
    grids_out: list[np.ndarray] = []
    actions_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    ids_out: list[str] = []
    dropped = 0

    chosen = specs if limit is None else specs[:limit]
    for i, spec in enumerate(chosen):
        for episode in range(episodes_per_world):
            history = make_training_history(spec, seed=episode * 101 + 7)
            if history is None:
                dropped += 1
                continue
            if require_strong_evidence and evidence_coverage(history).unexercised():
                dropped += 1
                continue
            g, a, y = encode_world(spec, history)
            grids_out.append(g)
            actions_out.append(a)
            labels_out.append(y)
            ids_out.append(spec.world_id)
        if verbose and (i + 1) % 50 == 0:
            print(f"  encoded {i + 1}/{len(chosen)} worlds", flush=True)

    if not labels_out:
        raise RuntimeError("no usable episodes were produced")
    if verbose and dropped:
        print(f"  dropped {dropped} episodes with unusable evidence", flush=True)

    return Dataset(
        grids=np.stack(grids_out).astype(np.int32),
        actions=np.stack(actions_out).astype(np.int32),
        labels=np.stack(labels_out).astype(np.int32),
        world_ids=tuple(ids_out),
    )


def save_dataset(dataset: Dataset, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        grids=dataset.grids,
        actions=dataset.actions,
        labels=dataset.labels,
        world_ids=np.array(dataset.world_ids),
    )
    return p


def load_dataset(path: str | Path) -> Dataset:
    data = np.load(Path(path), allow_pickle=False)
    return Dataset(
        grids=data["grids"],
        actions=data["actions"],
        labels=data["labels"],
        world_ids=tuple(str(w) for w in data["world_ids"]),
    )


def iterate_batches(
    dataset: Dataset, batch_size: int, rng: np.random.Generator, shuffle: bool = True
):
    order = rng.permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        yield dataset.grids[idx], dataset.actions[idx], dataset.labels[idx]


def majority_baseline(train: Dataset, test: Dataset) -> dict[str, float]:
    """Accuracy from always predicting the most common training class.

    The floor the core must clear. Reporting raw accuracy without this is
    close to meaningless: with unbalanced labels, a network that has learned
    nothing can look competent by memorising the prior.
    """
    from .encoding import HEADS

    out: dict[str, float] = {}
    for i, (name, _) in enumerate(HEADS):
        values, counts = np.unique(train.labels[:, i], return_counts=True)
        guess = int(values[int(np.argmax(counts))])
        out[name] = float(np.mean(test.labels[:, i] == guess))
    return out
