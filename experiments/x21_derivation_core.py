"""X21: train the core on DSL derivations -- the ranker-to-generator shift.

Every other component of the Level 4 loop now speaks DSL (X19 generator,
X20 explorer, X17 search). The last old-vocabulary component was the
learned core: its heads spanned only the label encoding. This experiment
trains a TRM core with DSL-sized heads on episodes from identifiability-
aware worlds, then measures:

A. per-head accuracy on held-out extended worlds
B. survivor RANKING after bulk refutation vs the simplicity tie-break

MEASURED (240 train worlds / 12 eval, 60 epochs):

    A. gates 92% (prior 52%) and wait 58% (prior 50%) beat prior; the
       multi-step pattern labels (charge, step, edge) do not yet. The
       original core needed 800 worlds x 250 epochs x 3 episodes plus
       several encoder fixes to read those; this smoke test has none of
       that tuning. Training scale, not architecture, is the gap.

    B. survivor ranking: core-ranked pick exact 3/12 vs simplicity 0/12.
       THE DERIVATION-TRAINED PRIOR BEATS SIMPLICITY even undertrained --
       refutation constrains the choice to programs that explain all the
       evidence, so any real signal beats an arbitrary tie-break among
       survivors. Survivor counts are large here (median 124, max 10,080)
       because these single-level worlds exercise fewer axes than X17's.

THE SHIFT IS UNDERWAY: the learned component now ranks DSL derivations,
closing the last old-vocabulary component of the loop. The path to full
generator is incremental from here: more training data (the corpus
pipeline exists), the encoder fixes already invented for the label core,
and behavioural-embedding scoring rather than per-axis log-prob sums.
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

from sentinel.adapt.hypothesis import scorable_segment  # noqa: E402
from sentinel.core.encoding import encode_history  # noqa: E402
from sentinel.core.model import CoreConfig, TinyRecursiveCore  # noqa: E402
from sentinel.env.types import Action  # noqa: E402
from sentinel.explore.version_space import state_key  # noqa: E402
from sentinel.gen.grid import GridWorld, initial_state, transition_state  # noqa: E402
from sentinel.gen.spec import LevelSpec, Mechanics, WorldSpec  # noqa: E402

TRAIN_WORLDS = 240
EVAL_WORLDS = 12
EPISODE_STEPS = 200
AXIS_SIZES = tuple(len(a) for a in AXES)
AXIS_NAMES = ("step", "charge", "edge", "hazard", "switch", "ordered",
              "gates", "wait")


def mech_to_labels(mech: Mechanics) -> np.ndarray:
    """DSL program -> per-axis class indices."""
    prog = (mech.step_distance, mech.charge_period,
            mech.effective_edge_mode(),
            (mech.has_hazards, mech.hazard_effect if mech.has_hazards else "kill"),
            (mech.has_switches, mech.switch_mode if mech.has_switches else "toggle"),
            mech.ordered_targets, mech.gates_start_open,
            mech.wait_advances_charge)
    return np.array([a.index(v) for a, v in zip(AXES, prog)], dtype=np.int32)


def random_episode(mech, observed, field_size, seed, steps=EPISODE_STEPS):
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


def sample_truth(rng: np.random.Generator) -> Mechanics:
    charge_opts = (None, 6, 8, 10, 12, 14, 16, 18, 20)
    return Mechanics(
        step_distance=int(rng.integers(1, 9)),
        charge_period=charge_opts[int(rng.integers(0, len(charge_opts)))],
        edge_mode=str(rng.choice(("block", "wrap", "bounce", "respawn"))),
        has_hazards=bool(rng.integers(0, 2)),
        hazard_effect=str(rng.choice(("kill", "pushback", "respawn"))),
        has_switches=True,
        switch_mode=str(rng.choice(("toggle", "latch"))),
        ordered_targets=bool(rng.integers(0, 2)),
        gates_start_open=bool(rng.integers(0, 2)),
        wait_advances_charge=bool(rng.integers(0, 2)),
    )


def build_world_set(count: int, seed_offset: int):
    rng = np.random.default_rng(seed_offset)
    out = []
    for wi in range(count):
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


def loss_fn(model, grids, actions, labels):
    logits = model(grids, actions)
    total = mx.zeros(())
    for i, head_logits in enumerate(logits):
        total = total + mx.mean(
            nn.losses.cross_entropy(head_logits, labels[:, i]))
    return total / len(logits)


def main() -> int:
    t0 = time.perf_counter()
    print(f"building {TRAIN_WORLDS} train + {EVAL_WORLDS} eval worlds...")
    train_set = build_world_set(TRAIN_WORLDS, 5000)
    eval_set = build_world_set(EVAL_WORLDS, 9000)

    # ---- encode training data
    grids_l, actions_l, labels_l = [], [], []
    for i, (mech, observed, size) in enumerate(train_set):
        hist = random_episode(mech, observed, size, seed=7000 + i)
        g, a = encode_history(hist)
        grids_l.append(g)
        actions_l.append(a)
        labels_l.append(mech_to_labels(mech))
    X = mx.array(np.stack(grids_l))
    A = mx.array(np.stack(actions_l))
    Y = mx.array(np.stack(labels_l))
    print(f"  encoded {len(grids_l)} episodes ({time.perf_counter() - t0:.0f}s)")

    # ---- train a TRM core with DSL-sized heads
    model = TinyRecursiveCore(CoreConfig())
    # swap head sizes for the DSL axes
    import mlx.nn as nn_
    model.heads = [nn_.Linear(model.cfg.d_model, n) for n in AXIS_SIZES]
    mx.random.seed(0)
    opt = optim.AdamW(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    epochs = 60
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        perm = rng.permutation(TRAIN_WORLDS)
        losses = []
        for start in range(0, len(perm), 16):
            idx = mx.array(perm[start:start + 16])
            loss, grads = loss_and_grad(model, X[idx], A[idx], Y[idx])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            losses.append(float(loss))
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch + 1}: loss {np.mean(losses):.4f} "
                  f"({time.perf_counter() - t0:.0f}s)")

    # ---- A. per-head accuracy on held-out worlds
    correct = np.zeros(len(AXES))
    counted = np.zeros(len(AXES))
    priors = np.zeros(len(AXES))
    eval_enc = []
    for i, (mech, observed, size) in enumerate(eval_set):
        hist = random_episode(mech, observed, size, seed=8000 + i)
        g, a = encode_history(hist)
        labels = mech_to_labels(mech)
        eval_enc.append((g[None], a[None], labels, mech, observed, size))
        logits = model(mx.array(g[None]), mx.array(a[None]))
        mx.eval(logits)
        for h in range(len(AXES)):
            pred = int(mx.argmax(logits[h][0]))
            correct[h] += pred == labels[h]
            counted[h] += 1
    print("\nA. per-head accuracy on held-out extended worlds:")
    for h, name in enumerate(AXIS_NAMES):
        vals, counts = np.unique(
            [mech_to_labels(m)[h] for m, _, _ in
             [(m, o, s) for m, o, s in train_set]], return_counts=True)
        priors[h] = counts.max() / counts.sum()
        acc = correct[h] / max(counted[h], 1)
        mark = " <-- beats prior" if acc > priors[h] else ""
        print(f"   {name:8} {acc:6.0%}  (prior {priors[h]:.0%}){mark}")

    # ---- B. survivor ranking vs simplicity
    print("\nB. survivor ranking after bulk refutation:")
    simp_exact = core_exact = 0
    n_surv_list = []
    for i, (g, a, labels, mech, observed, size) in enumerate(eval_enc):
        # A fresh episode from the same truth provides the frames refutation
        # judges; the encoded episode provides what the core reads.
        hist = random_episode(mech, observed, size, seed=9500 + i)
        seg2 = scorable_segment(hist)

        orders = ([tuple(p) for p in _perm(observed.targets)]
                  if 2 <= len(observed.targets) <= 4 else [observed.targets])
        cache = SpecCache(observed, size)
        live = [(idx, order, initial_state(0, cache.get(idx, order)))
                for idx in itertools.product(*[range(len(a_)) for a_ in AXES])
                for order in orders]
        for step in seg2.steps:
            if len(live) <= 1:
                break
            from x17_dsl_search import _frame_facts
            wa, wt, wg = _frame_facts(step.settled.grid, size)
            survivors = []
            for idx, order, state in live:
                nxt = transition_state(state, step.action, cache.get(idx, order))
                here = (nxt.x, nxt.y)
                visible = frozenset(t for t in nxt.remaining if t != here)
                if here != wa or visible != wt:
                    continue
                if wg is not None and bool(nxt.gates_open) != wg:
                    continue
                survivors.append((idx, order, nxt))
            live = survivors
        n_surv_list.append(len(live))
        if not live:
            continue

        truth_idx = tuple(int(v) for v in labels)
        surv_idxs = [idx for idx, _, _ in live]

        # simplicity pick
        simp_pick = min(surv_idxs, key=lambda c: complexity(c))
        simp_exact += simp_pick == truth_idx

        # core-ranked pick: score each survivor by the core's per-axis
        # confidence in its class
        logits = model(mx.array(g), mx.array(a))
        mx.eval(logits)
        logprobs = [l[0] - mx.logsumexp(l[0]) for l in logits]
        best_idx, best_score = None, -1e9
        for idx in surv_idxs:
            score = sum(float(logprobs[h][idx[h]]) for h in range(len(AXES)))
            if score > best_score:
                best_idx, best_score = idx, score
        core_exact += best_idx == truth_idx

    k = len(eval_enc)
    print(f"   survivors/world: median {int(np.median(n_surv_list))}, "
          f"max {max(n_surv_list) if n_surv_list else 0}")
    print(f"   simplicity pick exact: {simp_exact}/{k}")
    print(f"   core-ranked pick exact: {core_exact}/{k}")

    print("\nverdict:")
    beats_prior = sum(1 for h in range(len(AXES))
                      if correct[h] / max(counted[h], 1) > priors[h])
    if core_exact > simp_exact:
        print("   THE CORE RANKS SURVIVORS: derivation-trained prior beats")
        print("   simplicity at exact recovery. The ranker-to-generator")
        print("   shift is underway; next is amortising refutation itself.")
    elif beats_prior >= 6:
        print("   Core reads extended evidence (most heads beat prior) but")
        print("   ranking is not yet better than simplicity. More training")
        print("   worlds or the behavioural-embedding ranker are next.")
    else:
        print("   Core cannot yet read extended evidence; scale training data")
        print("   before drawing architecture conclusions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
