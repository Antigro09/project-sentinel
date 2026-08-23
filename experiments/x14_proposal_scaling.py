"""X14: how does learned proposal scale with training data?

X13's first cut failed to beat sampling (65% vs 92% coverage) with two
confounds identified in its own harvest:

  - some eval blocks are unsplittable at len-24 (w2000003 had zero
    splitting probes), so 'proposer missed' and 'nothing could hit' were
    mixed in one number;
  - only 355 training pairs from 8 worlds.

This experiment removes the second confound directly: harvest MANY worlds,
train proposers on increasing prefixes (8 / 16 / 32 worlds), and evaluate
all of them on the SAME held-out set -- reporting coverage separately for
provably-splittable blocks (random found a probe) and unsplit ones.

The curve decides between X13's two explanations:

    rising with data   proposal is learnable; keep scaling worlds until it
                       beats sampling, then hand it to X4.
    flat from 8 on     block composition does not determine the probe; the
                       input features are wrong and must include member
                       STATES (where members sit, not just which rules
                       they carry) before any amount of data helps.
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

N_WORLDS = 40
EVAL_WORLDS = 8
TRAIN_SIZES = (8, 16, 32)
BASE_LEN = 64
ATTACK_LEN = 24
RANDOM_ATTACKS = 40
MEMBER_CAP = 30
BLOCK_CAP = 60
SEQ_LEN = 24
N_ACTIONS = 5
SAMPLES_K = 8
FEATURE_DIM = sum(n for _, n in HEADS) + 1


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


def block_features(members: list[tuple[int, ...]]) -> np.ndarray:
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    offset = 0
    for i, (_, n) in enumerate(HEADS):
        for classes in members:
            feats[offset + classes[i]] += 1.0
        offset += n
    feats[-1] = np.log1p(len(members))
    return feats


def harvest(spec, rng):
    """Blocks with member lists, features, and whether random splits them."""
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
        feats = block_features(members)
        target = None
        for ap in attacks:
            if len({run(m, ap) for m in member_models}) > 1:
                target = np.array(ap, dtype=np.int64) - 1
                break
        out.append((member_models, feats, target))
    return out


class Proposer(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(FEATURE_DIM, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
        )
        self.head = nn.Linear(dim, SEQ_LEN * N_ACTIONS)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.body(x)
        return self.head(h).reshape(x.shape[0], SEQ_LEN, N_ACTIONS)


def train_proposer(pairs, epochs: int = 60) -> Proposer:
    model = Proposer()
    mx.random.seed(0)
    opt = optim.AdamW(learning_rate=1e-3)

    def loss_fn(m, x, y):
        logits = m(x)
        return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, N_ACTIONS),
                                               y.reshape(-1)))

    if not pairs:
        return model
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
    print(f"  {len(all_blocks)} blocks total ({time.perf_counter() - t0:.0f}s)")

    split_n = N_WORLDS - EVAL_WORLDS
    per_world = len(all_blocks) // N_WORLDS
    train_blocks = all_blocks[:split_n * per_world]
    eval_blocks = all_blocks[split_n * per_world:]

    n_splittable = sum(1 for _, _, t in eval_blocks if t is not None)
    print(f"eval: {len(eval_blocks)} blocks, {n_splittable} provably splittable\n")

    print(f'{"train worlds":>12} {"pairs":>7} {"cov(all)":>9} {"cov(splittable)":>15} '
          f'{"steps/split":>11}')
    results = []
    for size in TRAIN_SIZES:
        subset = train_blocks[:size * per_world]
        pairs = [(f, t) for _, f, t in subset if t is not None]
        model = train_proposer(pairs)

        cov_all = 0
        cov_split = 0
        steps_total = 0
        for member_models, feats, target in eval_blocks:
            x = mx.array(feats[None])
            logits = model(x)
            mx.eval(logits)
            probs = np.array(mx.softmax(logits[0], axis=-1))
            steps = 0
            hit = False
            for _ in range(SAMPLES_K):
                seq = tuple(int(rng.choice(N_ACTIONS, p=p)) + 1 for p in probs)
                steps += len(seq)
                if len({run(m, seq) for m in member_models}) > 1:
                    hit = True
                    break
            steps_total += steps
            if hit:
                cov_all += 1
                if target is not None:
                    cov_split += 1
        results.append((size, len(pairs), cov_all, cov_split, steps_total))
        print(f"{size:12d} {len(pairs):7d} "
              f"{100.0 * cov_all / max(len(eval_blocks), 1):8.0f}% "
              f"{100.0 * cov_split / max(n_splittable, 1):14.0f}% "
              f"{steps_total / max(cov_all, 1):10.1f}")

    print("\nreading the curve:")
    first, last = results[0], results[-1]
    gain = (last[3] / max(n_splittable, 1)) - (first[3] / max(n_splittable, 1))
    if gain > 0.08:
        print("  RISING with data: proposal is learnable. Keep scaling worlds;")
        print("  when it beats sampling's cost curve, X4 inherits the generator.")
    else:
        print("  FLAT: block composition does not determine the probe. The")
        print("  input features are the bottleneck -- enrich with member")
        print("  full states (positions, charge) before adding more data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
