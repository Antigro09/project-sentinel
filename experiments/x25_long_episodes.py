"""X25: apply the window fix -- long episodes + MAX_TRANSITIONS=64.

X24 measured long periods at 42% under a 64-step window and recommended
this configuration. RESULT: NOT REPLICATED. Long episodes (400-step walks,
many targets) plus the 64-window trained on 400 worlds gives overall
charge accuracy 8% -- indistinguishable from the 32-window baseline.

The discrepancy with X24 PART 2 (42% on long periods, 200 worlds) is not
resolved by this run. Two runs of nominally similar configurations differ
by 34 points on 12 eval worlds: charge accuracy at DSL scale is HIGH
VARIANCE, and single small eval sets cannot rank these configurations.
What is established:

  - X24 PART 2 proves long periods CAN be read (42% once).
  - This run shows that reading is not STABLE across training runs.
  - Both facts point the same direction: periodicity reading at DSL scale
    sits at the edge of what this core size + window can do, and progress
    requires either many more training worlds, a fundamentally better
    period representation (hierarchical estimation), or larger evaluation
    sets to measure against.

HONEST STATUS: charge_period remains the open label of Level 4. The loop
still functions -- refutation handles charge whenever episodes tick it,
and the ranker wins on every other axis (X23: 6/12 exact). Charge is the
boundary of the current architecture, cleanly identified and measured.
"""

from __future__ import annotations

from __future__ import annotations

import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, "experiments")

# The window must be patched BEFORE sentinel.core.model is imported:
# RelativeAttention's relative-position embedding and the sinusoidal
# tables are sized from MAX_TRANSITIONS at class-definition time. This is
# exactly why X24's long-window run worked -- it reloaded the module chain.
import sentinel.core.encoding as _enc  # noqa: E402
import sentinel.core.model as _mdl  # noqa: E402
WINDOW = 64
_enc.MAX_TRANSITIONS = WINDOW
_mdl.MAX_TRANSITIONS = WINDOW
_mdl.MAX_REL = WINDOW + 2

from x19_identifiable_worlds import make_identifiable_level  # noqa: E402
from x21_derivation_core import mech_to_labels, sample_truth  # noqa: E402

from sentinel.core.encoding import encode_history  # noqa: E402
from sentinel.core.model import CoreConfig, TinyRecursiveCore  # noqa: E402
from sentinel.gen.grid import GridWorld  # noqa: E402
from sentinel.gen.spec import LevelSpec, WorldSpec  # noqa: E402

TRAIN_WORLDS = 400
EPISODES = 2
EPOCHS = 100


def long_random_episode(mech, observed, field_size, seed, steps=300):
    """Hazard-avoiding random walk, long horizon."""
    spec = WorldSpec(world_id="syn", seed=0, field_size=field_size,
                     mechanics=mech, levels=(observed,))
    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)
    size = field_size
    for _ in range(steps):
        if world.done:
            break
        grid = world.history.last.grid
        here = None
        for y in range(size):
            for x in range(size):
                if grid[y][x] == 4:
                    here = (x, y)
                    break
                if here:
                    break
        choices = [1, 2, 3, 4, 5]
        if here is not None:
            safe = []
            for aid in (1, 2, 3, 4):
                dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[aid]
                nx, ny = here[0] + dx, here[1] + dy
                if 0 <= nx < size and 0 <= ny < size and grid[ny][nx] != 2:
                    safe.append(aid)
            if safe and rng.random() > 0.15:
                choices = safe
        world.step(int(rng.choice(choices)))
    return world.history


def loss_fn(model, grids, actions, labels):
    logits = model(grids, actions)
    total = mx.zeros(())
    for i, head_logits in enumerate(logits):
        total = total + mx.mean(
            nn.losses.cross_entropy(head_logits, labels[:, i]))
    return total / len(logits)


def set_window(mt: int):
    """Patch MAX_TRANSITIONS in both encoding and model, then rebuild the
    positional tables. Must happen before TinyRecursiveCore is constructed:
    the sinusoidal tables are sized from these constants at call time via
    module globals, but RelativeAttention's embedding is sized at class
    definition -- so a full reload of the module chain is required when
    changing the window."""
    import importlib
    import sentinel.core.encoding as enc
    import sentinel.core.model as mdl
    enc.MAX_TRANSITIONS = mt
    mdl.MAX_TRANSITIONS = mt
    mdl.MAX_REL = mt + 2
    importlib.reload(mdl)
    global TinyRecursiveCore, CoreConfig
    from sentinel.core.model import TinyRecursiveCore, CoreConfig as CC
    TinyRecursiveCore = mdl.TinyRecursiveCore
    CoreConfig = CC


def build(count, seed, max_transitions: int):
    """Long-episode worlds: many targets so episodes run long."""
    import sentinel.core.encoding as enc
    old_mt = enc.MAX_TRANSITIONS
    enc.MAX_TRANSITIONS = max_transitions
    rng = np.random.default_rng(seed)
    out = []
    attempts = 0
    while len(out) < count and attempts < count * 5:
        attempts += 1
        truth = sample_truth(rng)
        size = int(rng.integers(9, 14))
        level = make_identifiable_level(rng, size, truth)
        observed = LevelSpec(
            start=level.start, walls=level.walls, hazards=level.hazards,
            targets=tuple(sorted(level.targets)),
            switches=level.switches, gates=level.gates,
        )
        episodes = []
        for ep in range(EPISODES):
            hist = long_random_episode(truth, observed, size,
                                       seed=int(rng.integers(0, 10**6)),
                                       steps=400)
            g, a = encode_history(hist)
            episodes.append((g, a))
        labels = mech_to_labels(truth)
        out.append((episodes, labels))
    enc.MAX_TRANSITIONS = old_mt
    return out


def train_and_eval(train_data, eval_data, tag: str, t0: float):
    from x17_dsl_search import AXES
    axes_sizes = tuple(len(a) for a in AXES)
    model = TinyRecursiveCore(CoreConfig())
    model.heads = [nn.Linear(model.cfg.d_model, k) for k in axes_sizes]
    mx.random.seed(0)
    opt = optim.AdamW(learning_rate=1e-3)

    grids_l, actions_l, labels_l = [], [], []
    for episodes, labels in train_data:
        for g, a in episodes:
            grids_l.append(g)
            actions_l.append(a)
            labels_l.append(labels)
    X = mx.array(np.stack(grids_l))
    A = mx.array(np.stack(actions_l))
    Y = mx.array(np.stack(labels_l))
    n = len(grids_l)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    trng = np.random.default_rng(0)
    for epoch in range(EPOCHS):
        perm = trng.permutation(n)
        losses = []
        for start in range(0, n, 32):
            idx = mx.array(perm[start:start + 32])
            loss, grads = loss_and_grad(model, X[idx], A[idx], Y[idx])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            losses.append(float(loss))
        if (epoch + 1) % 25 == 0:
            print(f"  [{tag}] epoch {epoch + 1}: "
                  f"loss {np.mean(losses):.4f} ({time.perf_counter() - t0:.0f}s)")

    buckets = {"none": [0, 0], "short(6-10)": [0, 0], "long(11-20)": [0, 0]}
    from x17_dsl_search import AXES as AX
    charge_values = AX[1]
    for episodes, labels in eval_data:
        g, a = episodes[0]
        logits = model(mx.array(g[None]), mx.array(a[None]))
        mx.eval(logits)
        pred = int(mx.argmax(logits[1][0]))
        true_cp = charge_values[labels[1]]
        key = ("none" if true_cp is None
               else ("short(6-10)" if true_cp <= 10 else "long(11-20)"))
        buckets[key][1] += 1
        buckets[key][0] += pred == labels[1]
    print(f"  [{tag}] charge accuracy by bucket:")
    for key, (c, t) in buckets.items():
        if t:
            print(f"     {key:12} {c:3d}/{t:3d} ({100.0 * c / t:4.0f}%)")
        else:
            print(f"     {key:12}  no samples")
    total_c = sum(c for c, _ in buckets.values())
    total_t = sum(t for _, t in buckets.values())
    return total_c / max(total_t, 1), model


def main() -> int:
    t0 = time.perf_counter()
    print("PART A: long single-level episodes (many targets), "
          "MAX_TRANSITIONS=64")
    train_a = build(TRAIN_WORLDS, 5000, 64)
    eval_a = build(24, 9000, 64)
    print(f"  {len(train_a)} train / {len(eval_a)} eval")
    acc_a, _ = train_and_eval(train_a, eval_a, "long-episodes", t0)

    print("\nverdict:")
    print(f"  overall charge accuracy with long episodes + 64-window: "
          f"{acc_a:.0%}")
    if acc_a >= 0.4:
        print("   CHARGE UNLOCKED at DSL scale: long episodes plus the wider")
        print("   window give the counter enough ticks to be read. The full")
        print("   Phase-3-style scaling of this configuration is the")
        print("   production configuration for the derivation core.")
    elif acc_a >= 0.25:
        print("   Charge improving; combine with more training worlds.")
    else:
        print("   Still limited; hierarchical period estimation remains the")
        print("   fallback.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
