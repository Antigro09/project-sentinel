"""The critique's checkpoint: can the selector work on an unrelated domain?

"If your core selector can pick the right solution out of those 49.4
survivors across completely diverse, unrelated tasks, then you have built a
genuinely generalized architecture."

Trained on GRID worlds only. Tested on grid worlds and on dial worlds --
no agent, no walls, no space, a value is a magnitude rather than a
position. The ranker never sees a mechanic name in either domain; it scores
(evidence, predicted-behaviour) pairs, so a domain it was not trained on
needs no new heads and no new classes.

Chance is 1/|survivors|. Beating it on dials is the whole claim.
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from sentinel.adapt.hypothesis import classes_from_mechanics, mechanics_from_classes, scorable_segment
from sentinel.core.agent import read_layout
from sentinel.core.data import exploration_history
from sentinel.core.encoding import crop_box, encode_history
from sentinel.core.model import CoreConfig
from sentinel.core.universal import UniversalRanker, ranking_loss, rollout_signature
from sentinel.core import load_split
from sentinel.domains.dials import DialModel, DialWorld, mechanic_space as dial_space
from sentinel.env.types import Action
from sentinel.explore import VersionSpace
from sentinel.gen.grid import GridWorldModel
from sentinel.gen.spec import WorldSpec

MAX_CANDIDATES = 12


def grid_example(spec, seed=0):
    """(evidence, candidate rollouts, index of the truth) for one grid world."""
    history = exploration_history(spec, seed, 60)
    segment = scorable_segment(history)
    if len(segment.steps) < 5:
        return None
    observed = read_layout(segment.initial.grid, spec.field_size)
    space = VersionSpace.over(observed, spec.field_size)
    for st in segment.steps:
        space.observe(st.action, st.settled)

    truth = classes_from_mechanics(spec.mechanics)
    survivors = space.candidates()
    if truth not in survivors or len(survivors) < 2:
        return None
    others = [c for c in survivors if c != truth][: MAX_CANDIDATES - 1]
    cands = [truth] + others

    ev_g, ev_a = encode_history(history)
    box = crop_box(history)
    rolls = []
    for classes in cands:
        sp = WorldSpec(world_id="r", seed=0, field_size=spec.field_size,
                       mechanics=mechanics_from_classes(classes), levels=(observed,))
        m = GridWorldModel(sp)
        rolls.append(rollout_signature(m, m.init_state(), box, spec.field_size))
    return ev_g, ev_a, rolls, 0


def dial_example(truth_mech, start, target, seed=0, steps=5):
    """Short episodes on purpose.

    A 40-step dial episode leaves exactly ONE survivor -- the evidence
    determines the rules completely -- so there is nothing to rank and the
    checkpoint cannot be measured. Ambiguity has to exist before a selector
    can be tested, so the episode is cut short deliberately.
    """
    world = DialWorld(truth_mech, start, target)
    world.reset()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        if world.done:
            break
        world.step(Action(int(rng.integers(1, 6))))
    history = world.history
    if len(history.steps) < 3:
        return None

    # Refute, exactly as in the grid domain.
    live = []
    for m in dial_space():
        model = DialModel(m, start, target)
        live.append((m, model, model.init_state()))
    for step in history.steps:
        actual = tuple(tuple(r) for r in step.settled.grid)
        surv = []
        for mech, model, state in live:
            nxt = model.transition(state, step.action)
            if tuple(tuple(r) for r in model.render(nxt)) == actual:
                surv.append((mech, model, nxt))
        if not surv:
            break
        live = surv
    names = [m.summary() for m, _, _ in live]
    if truth_mech.summary() not in names or len(live) < 2:
        return None

    ordered = [t for t in live if t[0].summary() == truth_mech.summary()]
    ordered += [t for t in live if t[0].summary() != truth_mech.summary()]
    ordered = ordered[:MAX_CANDIDATES]

    ev_g, ev_a = encode_history(history)
    box = crop_box(history)
    rolls = []
    for _, model, _ in ordered:
        rolls.append(rollout_signature(model, model.init_state(), box, 64))
    return ev_g, ev_a, rolls, 0


def batch(example):
    ev_g, ev_a, rolls, label = example
    n = len(rolls)
    ev_G = mx.array(np.repeat(ev_g[None], n, axis=0).astype(np.int32))
    ev_A = mx.array(np.repeat(ev_a[None], n, axis=0).astype(np.int32))
    hy_G = mx.array(np.stack([r[0] for r in rolls]).astype(np.int32))
    hy_A = mx.array(np.stack([r[1] for r in rolls]).astype(np.int32))
    return ev_G, ev_A, hy_G, hy_A, mx.array(np.array(label, dtype=np.int32))


def accuracy(model, examples):
    hits, chance = [], []
    for ex in examples:
        ev_G, ev_A, hy_G, hy_A, label = batch(ex)
        s = model(ev_G, ev_A, hy_G, hy_A)
        mx.eval(s)
        hits.append(int(np.argmax(np.array(s)) == int(np.array(label))))
        chance.append(1.0 / len(ex[2]))
    return float(np.mean(hits)), float(np.mean(chance)), len(hits)


def main() -> int:
    splits = load_split("corpus/split_wide.json")
    print("building grid training examples...")
    train_ex = []
    for spec in splits["train"][:400]:
        ex = grid_example(spec)
        if ex:
            train_ex.append(ex)
        if len(train_ex) >= 150:
            break
    grid_test = []
    for spec in splits["holdout_mechanics"][:120]:
        ex = grid_example(spec)
        if ex:
            grid_test.append(ex)
        if len(grid_test) >= 40:
            break
    print(f"  {len(train_ex)} train, {len(grid_test)} grid test")

    print("building dial test examples (never trained on)...")
    rng = np.random.default_rng(0)
    dial_test = []
    for i, m in enumerate(dial_space() * 6):
        start = tuple(int(v) for v in rng.integers(0, 6, 4))
        target = tuple(int(v) for v in rng.integers(0, 6, 4))
        ex = dial_example(m, start, target, seed=i, steps=int(rng.integers(3, 7)))
        if ex:
            dial_test.append(ex)
        if len(dial_test) >= 30:
            break
    print(f"  {len(dial_test)} dial test")

    model = UniversalRanker(CoreConfig(cycles=2))
    opt = optim.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    loss_and_grad = nn.value_and_grad(model, ranking_loss)

    print("\ntraining on GRID worlds only")
    for epoch in range(12):
        started = time.perf_counter()
        order = np.random.default_rng(epoch).permutation(len(train_ex))
        total = 0.0
        for i in order:
            ev_G, ev_A, hy_G, hy_A, label = batch(train_ex[i])
            loss, grads = loss_and_grad(model, ev_G, ev_A, hy_G, hy_A, label)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            total += float(loss)
        g_acc, g_ch, g_n = accuracy(model, grid_test)
        d_acc, d_ch, d_n = accuracy(model, dial_test)
        print(f"  epoch {epoch}  loss {total/len(order):.3f}  "
              f"grid {g_acc:.1%} (chance {g_ch:.1%})  dial {d_acc:.1%} (chance {d_ch:.1%})  "
              f"({time.perf_counter()-started:.0f}s)")

    g_acc, g_ch, _ = accuracy(model, grid_test)
    d_acc, d_ch, _ = accuracy(model, dial_test)
    print(f"\nTRAINED ON GRIDS ONLY")
    print(f"  grid worlds: {g_acc:.1%} against {g_ch:.1%} chance")
    print(f"  dial worlds: {d_acc:.1%} against {d_ch:.1%} chance   <- the checkpoint")
    print("\nbeating chance on dials means the SELECTOR generalises, not just the verifier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
