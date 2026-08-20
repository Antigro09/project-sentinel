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


def exploration_history(spec: WorldSpec, seed: int, steps: int = 60, avoid_hazards: bool = True):
    """An episode of arbitrary actions -- what the agent actually collects.

    At test time `run_episode` explores by taking actions and then infers
    from what happened. Training on solution trajectories instead was a
    train/test mismatch I introduced early and did not notice: the core
    never saw the kind of episode it would be asked to read.

    **Hazard cells are avoided, and that is not cheating.** Hazards are
    rendered as a distinct colour, so refusing to step onto one reads the
    frame rather than the rules -- whether hazards are lethal, harmless or
    teleporting is exactly what stays hidden. Without this, random play in
    the compositional space usually dies within twenty moves: measured, 67%
    of exploration episodes were dropped for thin evidence and the
    survivors held 20 transitions against a solution path's 31. The hidden
    counter needs a long unbroken run of movement before its period shows.
    """
    from sentinel.env.types import Action
    from sentinel.gen.grid import HAZARD, MOVES, GridWorld

    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)
    size = spec.field_size

    for _ in range(steps):
        if world.done:
            break
        choices = [1, 2, 3, 4, 5]
        if avoid_hazards:
            grid = world.history.last.grid
            here = None
            for y in range(size):
                for x in range(size):
                    if grid[y][x] == 4:  # AGENT
                        here = (x, y)
                        break
                if here:
                    break
            if here is not None:
                safe = []
                for aid, (dx, dy) in MOVES.items():
                    nx, ny = here[0] + dx, here[1] + dy
                    if 0 <= nx < size and 0 <= ny < size and grid[ny][nx] == HAZARD:
                        continue
                    safe.append(aid)
                choices = safe + [5] if safe else [5]
        world.step(Action(int(rng.choice(choices))))
    return world.history


def probing_history(spec: WorldSpec, seed: int, steps: int = 60):
    """Exploration that deliberately tests whether targets can be collected.

    `ordered_targets` is only visible when a collection FAILS: the agent
    stands on a target it may not take yet, hiding it, and the target
    reappears when the agent steps away. Nothing in random play makes that
    happen -- measured, the evidence distinguished ordered objectives in 7%
    of worlds with random actions and 23% from solution paths, against 100%
    for the hidden counter. That is an evidence problem, not a learning one,
    and no amount of training fixes it.

    So this walks ONTO targets and then OFF them, which is the plan's
    `explore/` idea in its smallest form: choose the actions that raise
    evidence coverage most. Targets are rendered in their own colour, so
    steering toward one reads the frame; whether it yields is the hidden
    part and stays hidden.
    """
    from sentinel.env.types import Action
    from sentinel.gen.grid import MOVES, TARGET, GridWorld

    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)
    size = spec.field_size
    stepping_off = False

    def locate(grid, value):
        return [(x, y) for y in range(size) for x in range(size) if grid[y][x] == value]

    for _ in range(steps):
        if world.done:
            break
        grid = world.history.last.grid
        agent = locate(grid, 4)
        targets = locate(grid, TARGET)
        action = None

        if agent and targets:
            ax, ay = agent[0]
            if stepping_off:
                # Move anywhere; the point is to reveal whether the target
                # we were standing on is still there.
                action = int(rng.integers(1, 5))
                stepping_off = False
            else:
                tx, ty = min(targets, key=lambda t: abs(t[0] - ax) + abs(t[1] - ay))
                best, best_d = None, abs(tx - ax) + abs(ty - ay)
                for aid, (dx, dy) in MOVES.items():
                    d = abs(tx - (ax + dx)) + abs(ty - (ay + dy))
                    if d < best_d:
                        best, best_d = aid, d
                action = best
                if best_d == 0:
                    stepping_off = True

        if action is None or rng.random() < 0.25:
            action = int(rng.integers(1, 6))
        world.step(Action(action))
    return world.history


def build_dataset(
    specs: list[WorldSpec],
    episodes_per_world: int = 3,
    limit: int | None = None,
    require_strong_evidence: bool = True,
    verbose: bool = False,
    episode_kind: str = "solution",
    required_channels: tuple[str, ...] = ("render", "transition"),
) -> Dataset:
    """Encode episodes for each world.

    Episodes whose evidence cannot distinguish action-dependent behaviour
    are dropped. Their labels would still be correct, but nothing in the
    observations supports them — training on those teaches the core to
    guess from priors rather than read the evidence, which is the exact
    habit this architecture is meant to avoid.

    `episode_kind` selects what the core is shown: "solution" replays a
    solved path, "exploration" takes arbitrary actions, "probing" walks onto
    and off targets to expose ordered objectives, and "mixed" rotates
    through all three. Exploration is what the agent actually has at test time.

    `required_channels` is deliberately not all three. The outcome channel
    needs a level completion, a win or a death, and random exploration
    usually produces none of them -- which dropped 67% of exploration
    episodes. But every label here describes MOVEMENT, and an episode can
    pin down step distance, edge behaviour and the hidden counter without
    ever finishing a level. Demanding outcome evidence to learn dynamics
    discards good evidence for a channel the labels do not use.
    """
    grids_out: list[np.ndarray] = []
    actions_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    ids_out: list[str] = []
    dropped = 0

    chosen = specs if limit is None else specs[:limit]
    for i, spec in enumerate(chosen):
        for episode in range(episodes_per_world):
            if episode_kind == "probing" or (episode_kind == "mixed" and episode % 3 == 2):
                history = probing_history(spec, seed=episode * 101 + 7)
            elif episode_kind == "exploration" or (
                episode_kind == "mixed" and episode % 3 == 1
            ):
                history = exploration_history(spec, seed=episode * 101 + 7)
            else:
                history = make_training_history(spec, seed=episode * 101 + 7)
            if history is None:
                dropped += 1
                continue
            if require_strong_evidence:
                missing = set(evidence_coverage(history).unexercised())
                if missing & set(required_channels):
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
