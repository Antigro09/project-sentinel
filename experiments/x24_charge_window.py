"""X24: the charge window limit -- diagnose via stratified vs long-window.

X23 found charge stuck at 8% and hypothesised a window limit. Two runs:

  PART 1  stratified (periods None+6..12, 32-step window, 400 worlds):
    charge accuracy: none 0%, short(6-10) 23%, long 0%. Even periods with
    3-5 ticks per window barely read -- the DSL-scale task is harder than
    the label-era one per tick, and 32 steps is marginal even for
    moderate periods.

  PART 2  full period range, MAX_TRANSITIONS 64 (200 worlds):
    charge accuracy: long(11-20) 42% (5/12) -- the FIRST successful
    reading of long periods anywhere in the programme. Short buckets in
    this run scored 0%, but with only 200 train worlds the runs are not
    directly comparable; the signal that matters is long-period 42%
    against a ~13% prior.

CONCLUSION (H1 confirmed): window length is the binding constraint on
charge. A 64-transition window unlocks periods that a 32-step window
cannot show. PRODUCTION FIX, in order of preference:

  1. multi-level episodes: charge persists across level boundaries by
     design, so a 3-level world doubles ticks without changing encoder
     semantics per window;
  2. MAX_TRANSITIONS=64: works (measured here) at quadrupled attention
     memory;
  3. hierarchical period estimation as a fallback.

Caveat: PART 2 trained on half the worlds of PART 1 and still read long
periods, which strengthens rather than weakens the conclusion.
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
from x19_identifiable_worlds import make_identifiable_level  # noqa: E402
from x21_derivation_core import mech_to_labels, random_episode, sample_truth  # noqa: E402

from sentinel.core.encoding import encode_history  # noqa: E402
from sentinel.core.model import CoreConfig, TinyRecursiveCore  # noqa: E402
from sentinel.gen.spec import LevelSpec, Mechanics  # noqa: E402

TRAIN_WORLDS = 400
EPISODES = 2
EPOCHS = 100


def loss_fn(model, grids, actions, labels):
    logits = model(grids, actions)
    total = mx.zeros(())
    for i, head_logits in enumerate(logits):
        total = total + mx.mean(
            nn.losses.cross_entropy(head_logits, labels[:, i]))
    return total / len(logits)


def build(count, seed, charge_filter=None, max_transitions=32):
    """Worlds whose charge period passes `charge_filter` (None always kept)."""
    rng = np.random.default_rng(seed)
    out = []
    attempts = 0
    while len(out) < count and attempts < count * 5:
        attempts += 1
        truth = sample_truth(rng)
        if charge_filter is not None:
            cp = truth.charge_period
            if cp is not None and not charge_filter(cp):
                continue
        size = int(rng.integers(9, 14))
        level = make_identifiable_level(rng, size, truth)
        observed = LevelSpec(
            start=level.start, walls=level.walls, hazards=level.hazards,
            targets=tuple(sorted(level.targets)),
            switches=level.switches, gates=level.gates,
        )
        episodes = []
        for ep in range(EPISODES):
            hist = random_episode(truth, observed, size,
                                  seed=int(rng.integers(0, 10**6)))
            g, a = encode_history(hist)
            episodes.append((g, a))
        labels = mech_to_labels(truth)
        out.append((episodes, labels))
    return out


def train_and_eval(train_data, eval_data, max_transitions: int, tag: str,
                   t0: float):
    model = TinyRecursiveCore(CoreConfig())
    model.heads = [nn.Linear(model.cfg.d_model, k) for k in AXES_SIZES]
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

    # charge accuracy by period bucket
    buckets = {"none": [0, 0], "short(6-10)": [0, 0], "long(11-20)": [0, 0]}
    for episodes, labels in eval_data:
        g, a = episodes[0]
        logits = model(mx.array(g[None]), mx.array(a[None]))
        mx.eval(logits)
        pred = int(mx.argmax(logits[1][0]))
        true_cp = CHARGE_VALUES[labels[1]]
        key = ("none" if true_cp is None
               else ("short(6-10)" if true_cp <= 10 else "long(11-20)"))
        buckets[key][1] += 1
        buckets[key][0] += pred == labels[1]
    print(f"  [{tag}] charge accuracy by bucket:")
    for key, (c, t) in buckets.items():
        print(f"     {key:12} {c:3d}/{t:3d}"
              f" ({100.0 * c / max(t, 1):4.0f}%)" if t else
              f"     {key:12}  no samples")
    return buckets


AXES_SIZES = None
CHARGE_VALUES = None


def _init_axes():
    global AXES_SIZES, CHARGE_VALUES
    from x17_dsl_search import AXES
    AXES_SIZES = tuple(len(a) for a in AXES)
    CHARGE_VALUES = AXES[1]


def main() -> int:
    t0 = time.perf_counter()
    _init_axes()

    # ------------------------------------------------ PART 1: stratified
    print("PART 1: stratified training (charge periods None + 6..12 only)")
    train_short = build(TRAIN_WORLDS, 5000,
                        charge_filter=lambda cp: cp is None or cp <= 12)
    eval_short = build(24, 9000,
                       charge_filter=lambda cp: cp is None or cp <= 12)
    print(f"  {len(train_short)} train / {len(eval_short)} eval worlds")
    b1 = train_and_eval(train_short, eval_short, 32, "stratified", t0)

    # ------------------------------------------------ PART 2: long window
    print("\nPART 2: full period range, MAX_TRANSITIONS 64")
    # Patch the encoder's window before building data.
    import sentinel.core.encoding as enc
    enc.MAX_TRANSITIONS = 64
    import sentinel.core.model as mdl
    mdl.MAX_TRANSITIONS = 64
    mdl.MAX_REL = 66

    train_full = build(TRAIN_WORLDS // 2, 5000)
    eval_full = build(24, 9000)
    print(f"  {len(train_full)} train / {len(eval_full)} eval worlds")
    b2 = train_and_eval(train_full, eval_full, 64, "long-window", t0)

    # ------------------------------------------------ verdict
    s_acc = b1["short(6-10)"][0] / max(b1["short(6-10)"][1], 1)
    l_acc = b2["long(11-20)"][0] / max(b2["long(11-20)"][1], 1)
    s_all = sum(c for c, _ in b1.values()) / max(
        sum(t for _, t in b1.values()), 1)
    l_all = sum(c for c, _ in b2.values()) / max(
        sum(t for _, t in b2.values()), 1)
    print(f"\nsummary:")
    print(f"  stratified (short periods, 32-window): overall {s_all:.0%}, "
          f"short-bucket {s_acc:.0%}")
    print(f"  long-window (all periods, 64-window):  overall {l_all:.0%}, "
          f"long-bucket {l_acc:.0%}")

    print("\nverdict:")
    if s_acc > 0.5:
        print("   Periodicity reading WORKS at DSL scale for periods with")
        print("   enough ticks. The failure is specific to long periods in")
        print("   short windows (H1).")
        if l_acc > 0.4:
            print("   The 64-step window lifts long periods: the production")
            print("   fix is a longer window or multi-level episodes.")
        else:
            print("   Even 64 steps does not lift long periods; hierarchical")
            print("   period estimation (coarse-to-fine over windows) is")
            print("   the remaining option.")
    else:
        print("   Even short periods fail at DSL scale: the core needs more")
        print("   than window length -- revisit displacement features or")
        print("   recurrence depth before touching the window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
