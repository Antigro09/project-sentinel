"""X22: the scaled derivation core, integrated into the full Level 4 loop.

X21 proved ranking concept on a smoke test; this closes the training gap
(800 worlds x 2 episodes x 100 epochs -- the label-era recipe adapted)
and measures the SYSTEM result end to end.

MEASURED:

A. per-head accuracy on held-out extended worlds (the X21 gap closed):

    step    58%  (prior 14%)   edge     33%  (prior 27%)
    hazard  67%  (prior 48%)   gates   100%  (prior 50%)
    wait    58%  (prior 51%)   switch   50% / ordered 33%
    charge   8%  (prior 13%)   <- the hidden counter remains the hard
                                    label at this scale

B. full loop (DSL-QbC explore -> refute all -> rank survivors):

    simplicity pick: 0/12 exact
    core-ranked pick: 2/12 exact, 10/12 bisimilar

THE TRAINED RANKER LIFTS THE LOOP over the simplicity tie-break, and the
learned component now reads extended evidence (step=8, gates, hazard
presence) it was never label-trained on. Remaining gaps are specific and
known: charge_period accuracy (needs the full Phase-3 recipe: more
episodes/world, encoder displacement features already in TransitionEncoder,
longer training), and occasional young episode deaths under QbC (w04, w06:
30 and 15 steps).

LEVEL 4 STATUS: every component speaks DSL and the learned ranker earns
its place inside the loop. The path forward is training scale plus the
charge-specific fixes already invented -- not new architecture.
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
from x17_dsl_search import AXES, SpecCache, compile_program, complexity  # noqa: E402
from x19_identifiable_worlds import make_identifiable_level  # noqa: E402
from x21_derivation_core import (  # noqa: E402
    mech_to_labels,
    random_episode,
    sample_truth,
)

from sentinel.adapt.hypothesis import scorable_segment  # noqa: E402
from sentinel.core.encoding import encode_history  # noqa: E402
from sentinel.core.model import CoreConfig, TinyRecursiveCore  # noqa: E402
from sentinel.env.types import Action  # noqa: E402
from sentinel.explore.version_space import state_key  # noqa: E402
from sentinel.gen.grid import GridWorld, initial_state, transition_state  # noqa: E402
from sentinel.gen.spec import LevelSpec, WorldSpec  # noqa: E402

TRAIN_WORLDS = 800
EPISODES_PER_WORLD = 2
EVAL_WORLDS = 12
EPOCHS = 100
COMMITTEE = 300


def dsl_qbc_episode(mech, observed, field_size, seed, steps=200):
    """The X20 winning explorer: DSL-committee QbC + hazard-seeking."""
    spec = WorldSpec(world_id="syn", seed=0, field_size=field_size,
                     mechanics=mech, levels=(observed,))
    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)
    progs = [tuple(a[int(rng.integers(0, len(a)))] for a in AXES)
             for _ in range(COMMITTEE)]
    cache = SpecCache(observed, field_size)
    idxs = [tuple(a.index(v) for a, v in zip(AXES, p)) for p in progs]
    order = observed.targets
    live = [(i, initial_state(0, cache.get(i, order))) for i in idxs]

    size = field_size
    spent = 0
    MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    while spent < steps and not world.done:
        seg = scorable_segment(world.history)
        for step in seg.steps:
            if len(live) <= 1:
                break
            from x17_dsl_search import _frame_facts
            wa, wt, wg = _frame_facts(step.settled.grid, field_size)
            survivors = []
            for idx, state in live:
                nxt = transition_state(state, step.action,
                                       cache.get(idx, order))
                here = (nxt.x, nxt.y)
                visible = frozenset(t for t in nxt.remaining if t != here)
                if here != wa or visible != wt:
                    continue
                if wg is not None and bool(nxt.gates_open) != wg:
                    continue
                survivors.append((idx, nxt))
            live = survivors

        grid = world.history.last.grid
        here = None
        for y in range(size):
            for x in range(size):
                if grid[y][x] == 4:
                    here = (x, y)
                    break
                if here:
                    break
        cands = [1, 2, 3, 4, 5]

        aid_choice = None
        hazards = [(x, y) for y in range(size) for x in range(size)
                   if grid[y][x] == 2]
        if hazards and here is not None and spent % 15 == 14:
            hx, hy = min(hazards, key=lambda h: abs(h[0] - here[0])
                         + abs(h[1] - here[1]))
            best, best_d = None, abs(hx - here[0]) + abs(hy - here[1])
            for a, (dx, dy) in MOVES.items():
                d = abs(hx - (here[0] + dx)) + abs(hy - (here[1] + dy))
                if d < best_d:
                    best, best_d = a, d
            aid_choice = best

        if aid_choice is None:
            if len(live) <= 1 or rng.random() < 0.15:
                aid_choice = int(rng.choice(cands))
            else:
                best_aid, best_split = None, -1
                for aid in cands:
                    outs = set()
                    for idx, st in live:
                        try:
                            nxt = transition_state(st, Action(aid),
                                                   cache.get(idx, order))
                        except Exception:
                            continue
                        outs.add(state_key(nxt))
                    if len(outs) > best_split:
                        best_aid, best_split = aid, len(outs)
                aid_choice = (best_aid if best_aid
                              else int(rng.choice(cands)))
        world.step(Action(aid_choice))
        spent += 1
    return scorable_segment(world.history)


def loss_fn(model, grids, actions, labels):
    logits = model(grids, actions)
    total = mx.zeros(())
    for i, head_logits in enumerate(logits):
        total = total + mx.mean(
            nn.losses.cross_entropy(head_logits, labels[:, i]))
    return total / len(logits)


def main() -> int:
    t0 = time.perf_counter()

    # ------------------------------------------------ 1. train big
    print(f"building {TRAIN_WORLDS} train worlds "
          f"x {EPISODES_PER_WORLD} episodes...")
    rng = np.random.default_rng(5000)
    grids_l, actions_l, labels_l = [], [], []
    for i in range(TRAIN_WORLDS):
        truth = sample_truth(rng)
        size = int(rng.integers(9, 14))
        level = make_identifiable_level(rng, size, truth)
        observed = LevelSpec(
            start=level.start, walls=level.walls, hazards=level.hazards,
            targets=tuple(sorted(level.targets)),
            switches=level.switches, gates=level.gates,
        )
        for ep in range(EPISODES_PER_WORLD):
            hist = random_episode(truth, observed, size,
                                  seed=7000 + i * 10 + ep)
            g, a = encode_history(hist)
            grids_l.append(g)
            actions_l.append(a)
            labels_l.append(mech_to_labels(truth))
    X = mx.array(np.stack(grids_l))
    A = mx.array(np.stack(actions_l))
    Y = mx.array(np.stack(labels_l))
    print(f"  {len(grids_l)} episodes encoded ({time.perf_counter() - t0:.0f}s)")

    model = TinyRecursiveCore(CoreConfig())
    model.heads = [nn.Linear(model.cfg.d_model, n) for n in
                   tuple(len(a) for a in AXES)]
    mx.random.seed(0)
    opt = optim.AdamW(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(0)
    n = len(grids_l)
    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        losses = []
        for start in range(0, n, 32):
            idx = mx.array(perm[start:start + 32])
            loss, grads = loss_and_grad(model, X[idx], A[idx], Y[idx])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            losses.append(float(loss))
        if (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch + 1}: loss {np.mean(losses):.4f} "
                  f"({time.perf_counter() - t0:.0f}s)")

    # ------------------------------------------------ 2. per-head accuracy
    eval_rng = np.random.default_rng(9000)
    correct = np.zeros(len(AXES))
    counted = np.zeros(len(AXES))
    priors = np.zeros(len(AXES))
    train_label_matrix = np.stack(labels_l)
    eval_worlds = []
    print("\nper-head accuracy on held-out extended worlds:")
    for wi in range(EVAL_WORLDS):
        truth = sample_truth(eval_rng)
        size = int(eval_rng.integers(9, 14))
        level = make_identifiable_level(eval_rng, size, truth)
        observed = LevelSpec(
            start=level.start, walls=level.walls, hazards=level.hazards,
            targets=tuple(sorted(level.targets)),
            switches=level.switches, gates=level.gates,
        )
        hist = random_episode(truth, observed, size, seed=8000 + wi)
        g, a = encode_history(hist)
        labels = mech_to_labels(truth)
        eval_worlds.append((truth, observed, size, g, a, labels))
        logits = model(mx.array(g[None]), mx.array(a[None]))
        mx.eval(logits)
        for h in range(len(AXES)):
            pred = int(mx.argmax(logits[h][0]))
            correct[h] += pred == labels[h]
            counted[h] += 1
    for h, name in enumerate(("step", "charge", "edge", "hazard",
                              "switch", "ordered", "gates", "wait")):
        vals, counts = np.unique(train_label_matrix[:, h], return_counts=True)
        priors[h] = counts.max() / counts.sum()
        acc = correct[h] / max(counted[h], 1)
        mark = " <-- beats prior" if acc > priors[h] else ""
        print(f"   {name:8} {acc:6.0%}  (prior {priors[h]:.0%}){mark}")

    # ------------------------------------------------ 3. integrate
    print("\nfull Level 4 loop on fresh worlds:")
    simp_exact = core_exact = bisim_n = 0
    long_probe = None
    from sentinel.core.universal import PROBE_ACTIONS
    long_probe = tuple(list(PROBE_ACTIONS)
                       + [((i % 5) + 1) for i in range(32)])
    for wi in range(EVAL_WORLDS):
        truth, observed, size, g, a, labels = eval_worlds[wi]
        seg = dsl_qbc_episode(truth, observed, size, seed=6000 + wi)

        orders = ([tuple(p) for p in _perm(observed.targets)]
                  if 2 <= len(observed.targets) <= 4 else [observed.targets])
        cache = SpecCache(observed, size)
        live = [(idx, order, initial_state(0, cache.get(idx, order)))
                for idx in itertools.product(*[range(len(a_)) for a_ in AXES])
                for order in orders]
        for step in seg.steps:
            if len(live) <= 1:
                break
            from x17_dsl_search import _frame_facts
            wa, wt, wg = _frame_facts(step.settled.grid, size)
            survivors = []
            for idx, order, state in live:
                nxt = transition_state(state, step.action,
                                       cache.get(idx, order))
                here = (nxt.x, nxt.y)
                visible = frozenset(t for t in nxt.remaining if t != here)
                if here != wa or visible != wt:
                    continue
                if wg is not None and bool(nxt.gates_open) != wg:
                    continue
                survivors.append((idx, order, nxt))
            live = survivors
        if not live:
            continue
        surv_idxs = [idx for idx, _, _ in live]
        truth_idx = tuple(int(v) for v in labels)

        simp_pick = min(surv_idxs, key=lambda c: complexity(c))
        simp_exact += simp_pick == truth_idx

        logits = model(mx.array(g[None]), mx.array(a[None]))
        mx.eval(logits)
        logprobs = [l[0] - mx.logsumexp(l[0]) for l in logits]
        best_idx, best_score = None, -1e9
        for idx in surv_idxs:
            score = sum(float(logprobs[h][idx[h]]) for h in range(len(AXES)))
            if score > best_score:
                best_idx, best_score = idx, score
        core_exact += best_idx == truth_idx

        # bisimilarity of the core pick
        best_prog = tuple(a[i] for a, i in zip(AXES, best_idx))
        best_mech = compile_program(best_prog)
        ts = WorldSpec(world_id="ts", seed=0, field_size=size,
                       mechanics=truth, levels=(observed,))
        bs = WorldSpec(world_id="bs", seed=0, field_size=size,
                       mechanics=best_mech, levels=(observed,))
        ts_s, bs_s = initial_state(0, ts), initial_state(0, bs)
        b = True
        for aid in long_probe:
            try:
                ts_s = transition_state(ts_s, Action(aid), ts)
                bs_s = transition_state(bs_s, Action(aid), bs)
            except Exception:
                b = False
                break
            if state_key(ts_s) != state_key(bs_s):
                b = False
                break
        bisim_n += b
        status = ("EXACT" if best_idx == truth_idx
                  else ("bisim" if b else "MISS"))
        print(f"  w{wi:02d}: ep={len(seg.steps):3d} surv={len(live):4d} "
              f"{status:6} truth={truth.summary()}")

    k = EVAL_WORLDS
    print(f"\n{'pick':>10} {'exact':>8} {'bisimilar':>10}")
    print(f"{'simplicity':>10} {simp_exact:5d}/{k} "
          f"{'':>10}")
    print(f"{'core':>10} {core_exact:5d}/{k} {bisim_n:7d}/{k}")

    print("\nverdict:")
    if core_exact > simp_exact:
        print("   THE TRAINED RANKER LIFTS THE LOOP: core-ranked exact")
        print(f"   recovery {core_exact}/{k} vs simplicity {simp_exact}/{k}.")
        print("   The learned component earns its place inside Level 4;")
        print("   scaling training further is now the known lever.")
    elif core_exact == simp_exact and core_exact > 0:
        print("   PARITY: refutation already narrows to few survivors here;")
        print("   the ranker matters most when survivor sets are large.")
    else:
        print("   Ranker underperforms simplicity at this training scale;")
        print("   inspect which heads mislead before scaling further.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
