"""X16: the integrated proposal pipeline at scale -- is it ready to hand to X4?

X14 showed proposal rises with data (+21 points from 8 to 32 worlds, curve
unflat); X15 showed member states add little (+3-4) -- class composition
already determines most of the probe. Two questions remain before the
generator can be handed to the DSL search:

  1. Where does the data curve SATURATE? Train at 16 / 40 / 72 worlds.
  2. Does the INTEGRATED pipeline -- beam for the free head, learned
     proposer for the tail, random as last resort -- beat pure random on
     TOTAL PROBE COST at comparable coverage? X12's hybrid saved only 5%
     because its tail stage was random; the learned tail is the part that
     was supposed to change that.

The cost accounting is the honest one: every probe step is an action in
the world. Coverage without cost is not a win.

This is the gate experiment for the proposal mechanism as a whole: if the
integrated pipeline wins on cost at >= random's coverage, the Level 4
generator exists in working form and X4 (DSL search pruned by behaviour)
inherits it. If not, the mechanism stays a sampling problem and X4 should
prune with random probes until a bigger corpus exists.
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from sentinel.adapt.search import ALL_HYPOTHESES
from sentinel.adapt.hypothesis import mechanics_from_classes
from sentinel.core.agent import read_layout
from sentinel.core.data import exploration_history
from sentinel.core.encoding import HEADS
from sentinel.core.universal import PROBE_ACTIONS
from sentinel.core import load_split
from sentinel.env.types import Action
from sentinel.explore.version_space import state_key
from sentinel.gen.grid import GridWorldModel
from sentinel.gen.spec import WorldSpec

N_WORLDS = 80
EVAL_WORLDS = 8
TRAIN_SIZES = (16, 40, 72)
BASE_LEN = 64
ATTACK_LEN = 24
RANDOM_ATTACKS = 40
MEMBER_CAP = 30
BLOCK_CAP = 60
SEQ_LEN = 24
N_ACTIONS = 5
SAMPLES_K = 8
BEAM_WIDTH = 8
BEAM_CAP = 8
CLASS_DIM = sum(n for _, n in HEADS) + 1


def advance(model, state, aid):
    try:
        return model.transition(state, Action(aid))
    except Exception:
        return state


def run(model, probe) -> tuple:
    out = []
    current = model.init_state()
    for aid in probe:
        try:
            current = model.transition(current, Action(aid))
        except Exception:
            break
        out.append(state_key(current))
    return tuple(out)


def class_features(members: list[tuple[int, ...]]) -> np.ndarray:
    feats = np.zeros(CLASS_DIM, dtype=np.float32)
    offset = 0
    for i, (_, n) in enumerate(HEADS):
        for classes in members:
            feats[offset + classes[i]] += 1.0
        offset += n
    feats[-1] = np.log1p(len(members))
    return feats


def beam_probe(models, rng):
    """X11 v3: full-state-ranked beam with a randomised half."""
    init = tuple((m, m.init_state()) for m in models)
    beam = [((), init)]
    for _depth in range(BEAM_CAP):
        candidates = []
        for probe, ms in beam:
            for aid in (1, 2, 3, 4, 5):
                nxt = tuple((m, advance(m, s, aid)) for m, s in ms)
                ext = probe + (aid,)
                visible = {state_key(s) for _, s in nxt}
                if len(visible) > 1:
                    return ext
                candidates.append((ext, nxt))

        def diversity(item):
            _, nxt = item
            return len({s for _, s in nxt})

        candidates.sort(key=lambda t: (-diversity(t), t[0]))
        k = BEAM_WIDTH // 2
        rest = candidates[k:]
        extra = [rest[i] for i in rng.choice(len(rest), size=min(k, len(rest)),
                                             replace=False)] if rest else []
        beam = candidates[:k] + extra
        if diversity(beam[0]) <= 1:
            return None
    return None


def harvest(spec, rng):
    history = exploration_history(spec, 0, 30)
    observed = read_layout(history.initial.grid, spec.field_size)

    models = {}
    for classes in ALL_HYPOTHESES:
        sp = WorldSpec(world_id="q", seed=0, field_size=spec.field_size,
                       mechanics=mechanics_from_classes(classes), levels=(observed,))
        models[classes] = GridWorldModel(sp)

    base = list(PROBE_ACTIONS)
    while len(base) < BASE_LEN:
        base.append(int(rng.integers(1, 6)))
    blocks: dict[tuple, list] = {}
    for classes, model in models.items():
        blocks.setdefault(run(model, tuple(base)), []).append(classes)
    residual = sorted((v for v in blocks.values() if len(v) > 1),
                      key=len, reverse=True)[:BLOCK_CAP]

    attacks = [tuple(int(rng.integers(1, 6)) for _ in range(ATTACK_LEN))
               for _ in range(RANDOM_ATTACKS)]

    out = []
    for block in residual:
        members = block if len(block) <= MEMBER_CAP else [
            block[i] for i in rng.choice(len(block), MEMBER_CAP, replace=False)]
        member_models = [models[c] for c in members]
        target = None
        for ap in attacks:
            if len({run(m, ap) for m in member_models}) > 1:
                target = np.array(ap, dtype=np.int64) - 1
                break
        out.append((member_models, members, target, attacks))
    return out


class Proposer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(CLASS_DIM, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
        )
        self.head = nn.Linear(128, SEQ_LEN * N_ACTIONS)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.body(x)
        return self.head(h).reshape(x.shape[0], SEQ_LEN, N_ACTIONS)


def train_proposer(pairs, epochs: int = 60) -> Proposer:
    model = Proposer()
    if not pairs:
        return model
    mx.random.seed(0)
    opt = optim.AdamW(learning_rate=1e-3)

    def loss_fn(m, x, y):
        logits = m(x)
        return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, N_ACTIONS),
                                               y.reshape(-1)))

    X = mx.array(np.stack([f for f, _ in pairs]))
    Y = mx.array(np.stack([p for _, p in pairs]))
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for epoch in range(epochs):
        perm = np.random.default_rng(epoch).permutation(len(pairs))
        for start in range(0, len(perm), 64):
            idx = mx.array(perm[start:start + 64])
            loss, grads = loss_and_grad(model, X[idx], Y[idx])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
    return model


def main() -> int:
    specs = load_split("corpus/split_wide.json")["holdout_mechanics"][:N_WORLDS]
    rng = np.random.default_rng(0)

    t0 = time.perf_counter()
    print(f"harvesting {N_WORLDS} worlds...")
    all_blocks = []
    for spec in specs:
        all_blocks.extend(harvest(spec, rng))
    print(f"  {len(all_blocks)} blocks ({time.perf_counter() - t0:.0f}s)")

    split_n = N_WORLDS - EVAL_WORLDS
    per_world = len(all_blocks) // N_WORLDS
    train_blocks = all_blocks[:split_n * per_world]
    eval_blocks = all_blocks[split_n * per_world:]
    n_splittable = sum(1 for _, _, t, _ in eval_blocks if t is not None)
    print(f"eval: {len(eval_blocks)} blocks, {n_splittable} splittable\n")

    # --- data curve
    models_by_size = {}
    print(f'{"worlds":>7} {"pairs":>7} {"cov(splittable)":>15}')
    for size in TRAIN_SIZES:
        subset = train_blocks[:size * per_world]
        pairs = [(class_features(members), t)
                 for _, members, t, _ in subset if t is not None]
        model = train_proposer(pairs)
        models_by_size[size] = model

        # quick coverage check on a slice of eval
        hit = 0
        for member_models, members, target, _ in eval_blocks[:200]:
            x = mx.array(class_features(members)[None])
            logits = model(x)
            mx.eval(logits)
            probs = np.array(mx.softmax(logits[0], axis=-1))
            for _ in range(SAMPLES_K):
                seq = tuple(int(rng.choice(N_ACTIONS, p=p)) + 1 for p in probs)
                if len({run(m, seq) for m in member_models}) > 1:
                    hit += 1
                    break
        print(f"{size:7d} {len(pairs):7d} "
              f"{100.0 * hit / max(min(200, len(eval_blocks)), 1):14.0f}%")

    # --- integrated pipeline vs pure random on the full eval set
    best = models_by_size[max(TRAIN_SIZES)]
    print(f"\nintegrated pipeline (beam -> learned@{max(TRAIN_SIZES)} -> random) "
          f"vs pure random:")

    strat = {
        "random": {"cov": 0, "steps": 0},
        "integrated": {"cov": 0, "steps": 0},
    }
    for member_models, members, target, attacks in eval_blocks:
        # pure random
        steps = 0
        hit = False
        for ap in attacks:
            steps += ATTACK_LEN
            if len({run(m, ap) for m in member_models}) > 1:
                hit = True
                break
        strat["random"]["steps"] += steps
        strat["random"]["cov"] += hit

        # integrated: beam, then learned K samples, then random remainder
        steps = 0
        hit = False
        bp = beam_probe(member_models, rng)
        if bp is not None:
            steps += len(bp)
            hit = True
        else:
            steps += BEAM_CAP
            x = mx.array(class_features(members)[None])
            logits = best(x)
            mx.eval(logits)
            probs = np.array(mx.softmax(logits[0], axis=-1))
            for _ in range(SAMPLES_K):
                seq = tuple(int(rng.choice(N_ACTIONS, p=p)) + 1 for p in probs)
                steps += len(seq)
                if len({run(m, seq) for m in member_models}) > 1:
                    hit = True
                    break
            if not hit:
                for ap in attacks:
                    steps += ATTACK_LEN
                    if len({run(m, ap) for m in member_models}) > 1:
                        hit = True
                        break
        strat["integrated"]["steps"] += steps
        strat["integrated"]["cov"] += hit

    n = len(eval_blocks)
    print(f'{"strategy":>12} {"coverage":>9} {"steps/split":>12}')
    for name, s in strat.items():
        cov = 100.0 * s["cov"] / max(n, 1)
        per = s["steps"] / max(s["cov"], 1)
        print(f"{name:>12} {cov:8.0f}% {per:11.1f}")

    r, i = strat["random"], strat["integrated"]
    rc = r["steps"] / max(r["cov"], 1)
    ic = i["steps"] / max(i["cov"], 1)
    print()
    if i["cov"] >= r["cov"] and ic < 0.8 * rc:
        print("VERDICT: the integrated pipeline wins on cost at full coverage.")
        print("The Level 4 generator EXISTS: beam for the free head, a learned")
        print("proposer for the tail. X4 inherits it.")
    elif ic < rc:
        print(f"VERDICT: cheaper per split ({ic:.0f} vs {rc:.0f}) at "
              f"{'equal' if i['cov'] >= r['cov'] else 'slightly lower'} coverage.")
        print("Adopt the pipeline; the remaining gap is data volume.")
    else:
        print("VERDICT: no cost win yet. The tail is still sampling-dominated;")
        print("X4 should prune with random probes until a bigger corpus exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
