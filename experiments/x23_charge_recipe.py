"""X23: the full Phase-3 recipe on the derivation core -- charge gap audit.

Applied the label-era recipe in full: 800 worlds x 3 episodes x 200 epochs.

MEASURED:

A. per-head accuracy on held-out extended worlds:

    step    75%  (prior 14%)   hazard   75%  (prior 48%)
    switch  83%  (prior 52%)   gates   100%  (prior 50%)
    edge    67%  (prior 27%)   ordered  58%  (prior 52%)
    charge   8%  (prior 13%)   <- STILL stuck

B. full Level 4 loop on fresh worlds:

    simplicity pick: 0/12 exact
    core-ranked:     6/12 EXACT, 11/12 bisimilar  (X22 was 2/12, 10/12)

THE SURPRISE: end-to-end exact recovery TRIPLED even though charge head
accuracy stayed flat. The ranking gain came from sharper everywhere else
(step 58->75%, switch 50->83%, hazard 67->75%) -- refutation already
handles charge whenever the episode ticks it, so the ranker's leverage is
in every OTHER axis. A reminder that system-level gains do not require
every component to be individually solved.

CHARGE DIAGNOSIS (why 8% persists): the DSL asks for periods up to 20,
but MAX_TRANSITIONS=32 means a period-16 counter shows ~2 ticks and
period-20 shows 1 -- the label-era task (periods 2..5) had many ticks per
window. This is a WINDOW-SIZE limit, not an epoch limit: more training on
32-step windows cannot teach what the window does not show. Options:
longer MAX_TRANSITIONS (memory cost), multi-level episodes (charge
persists across boundaries by design), or hierarchical period estimation.
That is the next experiment if charge exactness matters downstream.
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
from x21_derivation_core import mech_to_labels, random_episode, sample_truth  # noqa: E402

from sentinel.adapt.hypothesis import scorable_segment  # noqa: E402
from sentinel.core.encoding import encode_history  # noqa: E402
from sentinel.core.model import CoreConfig, TinyRecursiveCore  # noqa: E402
from sentinel.env.types import Action  # noqa: E402
from sentinel.explore.version_space import state_key  # noqa: E402
from sentinel.gen.grid import GridWorld, initial_state, transition_state  # noqa: E402
from sentinel.gen.spec import LevelSpec, WorldSpec  # noqa: E402

TRAIN_WORLDS = 800
EPISODES_PER_WORLD = 3
EVAL_WORLDS = 12
EPOCHS = 200


def loss_fn(model, grids, actions, labels):
    logits = model(grids, actions)
    total = mx.zeros(())
    for i, head_logits in enumerate(logits):
        total = total + mx.mean(
            nn.losses.cross_entropy(head_logits, labels[:, i]))
    return total / len(logits)


def main() -> int:
    t0 = time.perf_counter()

    print(f"building {TRAIN_WORLDS} worlds x {EPISODES_PER_WORLD} episodes...")
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
    n = len(grids_l)
    print(f"  {n} episodes encoded ({time.perf_counter() - t0:.0f}s)")

    model = TinyRecursiveCore(CoreConfig())
    model.heads = [nn.Linear(model.cfg.d_model, k) for k in
                   tuple(len(a) for a in AXES)]
    mx.random.seed(0)
    opt = optim.AdamW(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    trng = np.random.default_rng(0)
    best_loss = float("inf")
    for epoch in range(EPOCHS):
        perm = trng.permutation(n)
        losses = []
        for start in range(0, n, 32):
            idx = mx.array(perm[start:start + 32])
            loss, grads = loss_and_grad(model, X[idx], A[idx], Y[idx])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            losses.append(float(loss))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
        if (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch + 1}: loss {mean_loss:.4f} "
                  f"(best {best_loss:.4f}, {time.perf_counter() - t0:.0f}s)")

    # ------------------------------------------------ evaluate
    eval_rng = np.random.default_rng(9000)
    correct = np.zeros(len(AXES))
    counted = np.zeros(len(AXES))
    priors = np.zeros(len(AXES))
    label_matrix = np.stack(labels_l)
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
    names = ("step", "charge", "edge", "hazard",
             "switch", "ordered", "gates", "wait")
    for h, name in enumerate(names):
        vals, counts = np.unique(label_matrix[:, h], return_counts=True)
        priors[h] = counts.max() / counts.sum()
        acc = correct[h] / max(counted[h], 1)
        mark = " <-- beats prior" if acc > priors[h] else ""
        print(f"   {name:8} {acc:6.0%}  (prior {priors[h]:.0%}){mark}")

    charge_acc = correct[1] / max(counted[1], 1)
    print(f"\ncharge accuracy at full recipe: {charge_acc:.0%} "
          f"(X22's scale run: 8%)")

    # ------------------------------------------------ loop integration
    from sentinel.core.universal import PROBE_ACTIONS
    long_probe = tuple(list(PROBE_ACTIONS)
                       + [((i % 5) + 1) for i in range(32)])
    simp_exact = core_exact = bisim_n = 0
    print("\nfull Level 4 loop on fresh worlds:")
    for wi in range(EVAL_WORLDS):
        truth, observed, size, g, a, labels = eval_worlds[wi]
        spec = WorldSpec(world_id="syn", seed=0, field_size=size,
                         mechanics=truth, levels=(observed,))
        world = GridWorld(spec)
        world.reset()
        erng = np.random.default_rng(6000 + wi)
        progs = [tuple(ax[int(erng.integers(0, len(ax)))] for ax in AXES)
                 for _ in range(300)]
        cache = SpecCache(observed, size)
        idxs = [tuple(ax.index(v) for ax, v in zip(AXES, p)) for p in progs]
        order = observed.targets
        live = [(idx, initial_state(0, cache.get(idx, order))) for idx in idxs]
        size_ = size
        spent = 0
        MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        while spent < 200 and not world.done:
            seg = scorable_segment(world.history)
            for step in seg.steps:
                if len(live) <= 1:
                    break
                wa, wt, wg = _frame_facts(step.settled.grid, size_)
                surv = []
                for idx, st in live:
                    nxt = transition_state(st, step.action,
                                           cache.get(idx, order))
                    here = (nxt.x, nxt.y)
                    vis = frozenset(t for t in nxt.remaining if t != here)
                    if here != wa or vis != wt:
                        continue
                    if wg is not None and bool(nxt.gates_open) != wg:
                        continue
                    surv.append((idx, nxt))
                live = surv
            grid = world.history.last.grid
            here = None
            for y in range(size_):
                for x in range(size_):
                    if grid[y][x] == 4:
                        here = (x, y)
                        break
                    if here:
                        break
            aid_choice = None
            hazards = [(x, y) for y in range(size_) for x in range(size_)
                       if grid[y][x] == 2]
            if hazards and here is not None and spent % 15 == 14:
                hx, hy = min(hazards, key=lambda hh: abs(hh[0] - here[0])
                             + abs(hh[1] - here[1]))
                best, best_d = None, abs(hx - here[0]) + abs(hy - here[1])
                for aa, (dx, dy) in MOVES.items():
                    d = abs(hx - (here[0] + dx)) + abs(hy - (here[1] + dy))
                    if d < best_d:
                        best, best_d = aa, d
                aid_choice = best
            if aid_choice is None:
                cands = [1, 2, 3, 4, 5]
                if len(live) <= 1 or erng.random() < 0.15:
                    aid_choice = int(erng.choice(cands))
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
                                  else int(erng.choice(cands)))
            world.step(Action(aid_choice))
            spent += 1

        seg = scorable_segment(world.history)
        orders = ([tuple(p) for p in _perm(observed.targets)]
                  if 2 <= len(observed.targets) <= 4 else [observed.targets])
        live = [(idx, o, initial_state(0, cache.get(idx, o)))
                for idx in itertools.product(*[range(len(ax)) for ax in AXES])
                for o in orders]
        for step in seg.steps:
            if len(live) <= 1:
                break
            wa, wt, wg = _frame_facts(step.settled.grid, size_)
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

        best_prog = tuple(ax[i] for ax, i in zip(AXES, best_idx))
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
    print(f"{'simplicity':>10} {simp_exact:5d}/{k}")
    print(f"{'core':>10} {core_exact:5d}/{k} {bisim_n:7d}/{k}")
    print(f"\n(total {time.perf_counter() - t0:.0f}s)")

    print("\nverdict:")
    if charge_acc > 0.3 and core_exact >= simp_exact:
        print("   CHARGE GAP CLOSING: the full recipe lifts the hidden")
        print("   counter above prior at DSL scale, and the ranked loop holds.")
        print("   Remaining charge accuracy is pure training scale.")
    elif charge_acc > 0.15:
        print("   Charge improving but not yet reliable; keep scaling epochs")
        print("   and episodes per world.")
    else:
        print("   Charge still stuck at DSL scale even with the recipe;")
        print("   inspect whether extended periods (6..20) need deeper")
        print("   recurrence or longer encoded windows than MAX_TRANSITIONS.")
    return 0


def _frame_facts(grid, field_size):
    from x17_dsl_search import _frame_facts as ff
    return ff(grid, field_size)


if __name__ == "__main__":
    raise SystemExit(main())
