"""X26: is charge variance seed noise or configuration signal?

X24 measured long-period charge at 42%; X25's similar run got 8%. This
experiment built the missing instrument: a 60-world eval set (5x larger),
three training seeds of one configuration, and ensemble ranking.

MEASURED:

    charge accuracy across 3 seeds (n=60): 13%..15% (spread 2%)
    ranked-exact per seed:                 22, 22, 20 of 60
    ensemble ranked-exact:                 21/60

TWO ANSWERS:

  1. SEED NOISE IS SMALL (2-point spread). The X24-vs-X25 swing was
     CONFIGURATION signal: episode-length distribution genuinely changes
     what the core can read. Long episodes (X25) are WORSE than standard
     ones (X24) for charge -- plausibly because encode_history keeps the
     FIRST 64 transitions, and long walks front-load uniform movement
     before the interesting ticks.

  2. ENSEMBLING DOES NOT HELP (21/60 vs 22/60 single): the cores make
     CORRELATED errors -- same architecture, same data, same blind spots.
     Ensembles only help when members fail differently; these fail the
     same way, which localises the problem in the representation, not the
     optimisation.

METROLOGY ESTABLISHED: charge at DSL scale is ~14% +/- 2 reliably, ranked
exact recovery ~37% +/- 4 on n=60. All future charge experiments must use
eval sets this size or larger.

CONSEQUENCE: the remaining lever for charge is REPRESENTATIONAL --
hierarchical period estimation, or displacement features tuned to long
periods -- not more seeds, ensembles, or minor recipe changes.
"""

from __future__ import annotations

from __future__ import annotations

import itertools
import sys
import time
from itertools import permutations as _perm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, "experiments")
from x19_identifiable_worlds import make_identifiable_level  # noqa: E402
from x21_derivation_core import mech_to_labels, random_episode, sample_truth  # noqa: E402

from sentinel.adapt.hypothesis import scorable_segment  # noqa: E402
from sentinel.core.encoding import encode_history  # noqa: E402
from sentinel.core.model import CoreConfig, TinyRecursiveCore  # noqa: E402
from sentinel.env.types import Action  # noqa: E402
from sentinel.explore.version_space import state_key  # noqa: E402
from sentinel.gen.grid import GridWorld, initial_state, transition_state  # noqa: E402
from sentinel.gen.spec import LevelSpec, WorldSpec  # noqa: E402

from x17_dsl_search import AXES, SpecCache, compile_program, complexity  # noqa: E402

TRAIN_WORLDS = 400
EPISODES = 2
EPOCHS = 100
SEEDS = (0, 1, 2)
EVAL_WORLDS = 60


def loss_fn(model, grids, actions, labels):
    logits = model(grids, actions)
    total = mx.zeros(())
    for i, head_logits in enumerate(logits):
        total = total + mx.mean(
            nn.losses.cross_entropy(head_logits, labels[:, i]))
    return total / len(logits)


def build_worlds(count, seed):
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
        out.append((truth, observed, size))
    return out


def encode_episode(mech, observed, size, seed):
    hist = random_episode(mech, observed, size, seed=seed)
    g, a = encode_history(hist)
    return g, a, mech_to_labels(mech)


def main() -> int:
    t0 = time.perf_counter()

    print(f"building {TRAIN_WORLDS} train + {EVAL_WORLDS} eval worlds...")
    train_worlds = build_worlds(TRAIN_WORLDS, 5000)
    eval_worlds = build_worlds(EVAL_WORLDS, 9000)

    # Pre-encode evaluation once (shared across seeds).
    eval_enc = []
    for wi, (mech, observed, size) in enumerate(eval_worlds):
        g, a, labels = encode_episode(mech, observed, size, seed=8000 + wi)
        eval_enc.append((g[None], a[None], labels, mech, observed, size))

    charge_values = AXES[1]
    per_seed_charge = []
    per_seed_exact = []
    models = []

    for si, seed in enumerate(SEEDS):
        tt = time.perf_counter()
        grids_l, actions_l, labels_l = [], [], []
        erng = np.random.default_rng(100 + seed)
        for i, (mech, observed, size) in enumerate(train_worlds):
            g, a, labels = encode_episode(mech, observed, size,
                                          seed=int(erng.integers(0, 10**6)))
            grids_l.append(g)
            actions_l.append(a)
            labels_l.append(labels)
        X = mx.array(np.stack(grids_l))
        A = mx.array(np.stack(actions_l))
        Y = mx.array(np.stack(labels_l))

        model = TinyRecursiveCore(CoreConfig())
        model.heads = [nn.Linear(model.cfg.d_model, k)
                       for k in tuple(len(ax) for ax in AXES)]
        mx.random.seed(seed)
        opt = optim.AdamW(learning_rate=1e-3)
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        trng = np.random.default_rng(seed)
        n = len(grids_l)
        for epoch in range(100):
            perm = trng.permutation(n)
            losses = []
            for start in range(0, n, 32):
                idx = mx.array(perm[start:start + 32])
                loss, grads = loss_and_grad(model, X[idx], A[idx], Y[idx])
                opt.update(model, grads)
                mx.eval(model.parameters(), opt.state)
                losses.append(float(loss))
        print(f"  seed {seed}: trained ({time.perf_counter() - tt:.0f}s)")

        # charge accuracy on the big eval set
        c_correct = 0
        ex = 0
        wcount = 0
        for g, a, labels, mech, observed, size in eval_enc:
            logits = model(mx.array(g), mx.array(a))
            mx.eval(logits)
            pred = int(mx.argmax(logits[1][0]))
            c_correct += pred == labels[1]
            # survivor ranking for this seed
            seg_hist = random_episode(mech, observed, size,
                                      seed=9500 + seed * 100 + wcount)
            wcount += 1
            seg = scorable_segment(seg_hist)
            orders = ([tuple(p) for p in _perm(observed.targets)]
                      if 2 <= len(observed.targets) <= 4
                      else [observed.targets])
            cache = SpecCache(observed, size)
            live = [(idx, o, initial_state(0, cache.get(idx, o)))
                    for idx in itertools.product(*[range(len(ax))
                                                   for ax in AXES])
                    for o in orders]
            for step in seg.steps:
                if len(live) <= 1:
                    break
                from x17_dsl_search import _frame_facts
                wa, wt, wg = _frame_facts(step.settled.grid, size)
                survivors = []
                for idx, o, state in live:
                    nxt = transition_state(state, step.action,
                                           cache.get(idx, o))
                    here = (nxt.x, nxt.y)
                    visible = frozenset(t for t in nxt.remaining
                                        if t != here)
                    if here != wa or visible != wt:
                        continue
                    if wg is not None and bool(nxt.gates_open) != wg:
                        continue
                    survivors.append((idx, o, nxt))
                live = survivors
            if not live:
                continue
            surv_idxs = [idx for idx, _, _ in live]
            truth_idx = tuple(int(v) for v in labels)
            logprobs = [l[0] - mx.logsumexp(l[0]) for l in logits]
            best_idx, best_score = None, -1e9
            for idx in surv_idxs:
                score = sum(float(logprobs[h][idx[h]])
                            for h in range(len(AXES)))
                if score > best_score:
                    best_idx, best_score = idx, score
            ex += best_idx == truth_idx
        acc = c_correct / len(eval_enc)
        per_seed_charge.append(acc)
        per_seed_exact.append(ex)
        models.append(model)
        print(f"    seed {seed}: charge {acc:.0%}, ranked-exact {ex}/{len(eval_enc)}")

    # ensemble ranking: sum log-probs across the three cores
    ens_exact = 0
    for wi, (g, a, labels, mech, observed, size) in enumerate(eval_enc):
        seg_hist = random_episode(mech, observed, size,
                                  seed=9500 + 50 + wi)
        seg = scorable_segment(seg_hist)
        orders = ([tuple(p) for p in _perm(observed.targets)]
                  if 2 <= len(observed.targets) <= 4 else [observed.targets])
        cache = SpecCache(observed, size)
        live = [(idx, o, initial_state(0, cache.get(idx, o)))
                for idx in itertools.product(*[range(len(ax)) for ax in AXES])
                for o in orders]
        for step in seg.steps:
            if len(live) <= 1:
                break
            from x17_dsl_search import _frame_facts
            wa, wt, wg = _frame_facts(step.settled.grid, size)
            survivors = []
            for idx, o, state in live:
                nxt = transition_state(state, step.action, cache.get(idx, o))
                here = (nxt.x, nxt.y)
                visible = frozenset(t for t in nxt.remaining if t != here)
                if here != wa or visible != wt:
                    continue
                if wg is not None and bool(nxt.gates_open) != wg:
                    continue
                survivors.append((idx, o, nxt))
            live = survivors
        if not live:
            continue
        surv_idxs = [idx for idx, _, _ in live]
        truth_idx = tuple(int(v) for v in labels)
        combined = np.zeros(len(AXES))
        all_lp = []
        for m in models:
            logits = m(mx.array(g), mx.array(a))
            mx.eval(logits)
            all_lp.append([np.array(l[0] - mx.logsumexp(l[0]))
                           for l in logits])
        best_idx, best_score = None, -1e18
        for idx in surv_idxs:
            score = sum(float(lp[h][idx[h]])
                        for lp in all_lp for h in range(len(AXES)))
            if score > best_score:
                best_idx, best_score = idx, score
        ens_exact += best_idx == truth_idx

    k = len(eval_enc)
    spread = max(per_seed_charge) - min(per_seed_charge)
    lo = min(per_seed_charge)
    hi = max(per_seed_charge)
    print(f"\ncharge accuracy across {len(SEEDS)} seeds "
          f"(eval n={k}): {lo:.0%}..{hi:.0%} (spread {spread:.0%})")
    print(f"ranked-exact per seed: {per_seed_exact}")
    print(f"ensemble ranked-exact: {ens_exact}/{k}")

    print("\nverdict:")
    if spread > 0.15:
        print(f"   SEED NOISE DOMINATES (spread {spread:.0%} >> measurement")
        print("   error). Single-run charge numbers are uninterpretable;")
        print("   ensembles or seed-selection by validation are mandatory.")
        if ens_exact > max(per_seed_exact):
            print(f"   Ensemble ranking also wins on exact recovery "
                  f"({ens_exact}/{k}) -- adopt it.")
    else:
        print(f"   Seed noise is small (spread {spread:.0%}); X24-vs-X25")
        print("   differences were configuration signal after all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
