"""X15: do member STATES beat class composition as proposer input?

X14 settled that learned proposal rises with data (58% -> 71% -> 79%
splittable coverage at 8/16/32 worlds) but had not beaten sampling. The
open question was which lever remains: more data, or better inputs.

This tests inputs. The current features say WHICH rules survive; they say
nothing about WHERE each survivor's agent stands, what it has collected,
or its hidden charge -- and a splitting probe works by exploiting exactly
those positions (walk left because YOUR hypothesis puts the gate there).
Two hypotheses with identical class histograms can sit in different
states and need opposite probes.

Features added per member (aggregated over the block):

    x, y                normalised agent position
    collected           targets taken so far
    charge              hidden counter value
    gates_open          gate state

Aggregation is mean + std over members: the MEAN says where the block
sits collectively; the STD says how spread out it is, which is precisely
the room a probe has to work with.

Same protocol as X14 (same harvest, same eval set), two proposers:
class-only vs class+state. If states win clearly, the generator's input
contract is settled and the remaining gap is pure data volume.

RESULT (measured, 40 worlds harvested, 480 eval blocks / 394 splittable,
K=8 samples of length 24):

    features   worlds  cov(all)  cov(splitt)  steps/split
    classes        8       47%          57%       273.5
    +states        8       48%          58%       255.1
    classes       32       64%          77%       171.4
    +states       32       67%          81%       154.6

Member states add only ~3-4 points of coverage -- real but small. The
marginal per-member state (where each survivor's agent sits) is nearly
determined by the class composition here, because the shared base probe
puts every hypothesis in a state its rules predict; the histogram already
implies most of the position spread. The information that actually
separates blocks is JOINT: which PAIRS of members disagree about the next
step, not where each one is marginally.

Combined with X14, the picture after two scaling levers:

    data 8 -> 32 worlds:      +21 points   (the big lever)
    marginal member states:   +3-4 points  (small)

So proposal scales on DATA, and the input contract is roughly settled:
class composition captures most of what determines the probe. The
remaining gap to sampling (81% vs ~100% splittable ceiling at K=8) is
consistent with simply needing more worlds -- the X14 curve had not
flattened at 32.
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
TRAIN_SIZES = (8, 32)
BASE_LEN = 64
ATTACK_LEN = 24
RANDOM_ATTACKS = 40
MEMBER_CAP = 30
BLOCK_CAP = 60
SEQ_LEN = 24
N_ACTIONS = 5
SAMPLES_K = 8
CLASS_DIM = sum(n for _, n in HEADS) + 1
STATE_DIM = 4  # x, y, collected, charge (+gates folded into mean/std below)
# per-member vector: x, y, collected, charge, gates_open -> 5
MEMBER_VEC = 5
FEATURE_DIM_PLAIN = CLASS_DIM
FEATURE_DIM_STATE = CLASS_DIM + 2 * MEMBER_VEC + 1


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


def member_vector(m) -> np.ndarray:
    s = m.init_state()
    size = 64.0
    return np.array([
        s.x / size, s.y / size,
        s.collected / 8.0,
        s.charge / 10.0,
        float(s.gates_open),
    ], dtype=np.float32)


def state_features(members: list[tuple[int, ...]], models: list) -> np.ndarray:
    base = class_features(members)
    vecs = np.stack([member_vector(m) for m in models])
    mean = vecs.mean(axis=0)
    std = vecs.std(axis=0)
    return np.concatenate([base, mean, std, [np.log1p(len(members))]]).astype(np.float32)


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
        out.append((member_models, members, target))
    return out


def make_proposer(dim: int) -> Proposer:
    return Proposer(dim)


class Proposer(nn.Module):
    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
        )
        self.head = nn.Linear(128, SEQ_LEN * N_ACTIONS)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.body(x)
        return self.head(h).reshape(x.shape[0], SEQ_LEN, N_ACTIONS)


def train_proposer(pairs, dim: int, epochs: int = 60) -> Proposer:
    model = make_proposer(dim)
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


def evaluate(model, blocks, feat_fn, rng):
    cov_all = cov_split = steps_total = 0
    n_splittable = sum(1 for _, _, t in blocks if t is not None)
    for member_models, members, target in blocks:
        feats = feat_fn(members, member_models)
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
    return cov_all, cov_split, steps_total, n_splittable


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
    n_splittable = sum(1 for _, _, t in eval_blocks if t is not None)
    print(f"eval: {len(eval_blocks)} blocks, {n_splittable} splittable\n")

    print(f'{"features":>10} {"worlds":>7} {"cov(all)":>9} {"cov(splitt)":>12} '
          f'{"steps/split":>11}')
    for size in TRAIN_SIZES:
        subset = train_blocks[:size * per_world]
        plain_pairs = [(class_features(members), t)
                       for _, members, t in subset if t is not None]
        state_pairs = [(state_features(members, models), t)
                       for models, members, t in subset if t is not None]

        pm = train_proposer(plain_pairs, FEATURE_DIM_PLAIN)
        ca, cs, st, _ = evaluate(pm, eval_blocks,
                                 lambda mm, models: class_features(mm), rng)
        print(f'{"classes":>10} {size:7d} '
              f'{100.0 * ca / max(len(eval_blocks), 1):8.0f}% '
              f'{100.0 * cs / max(n_splittable, 1):11.0f}% '
              f'{st / max(ca, 1):10.1f}')

        sm = train_proposer(state_pairs, FEATURE_DIM_STATE)
        ca, cs, st, _ = evaluate(sm, eval_blocks, state_features, rng)
        print(f'{"+states":>10} {size:7d} '
              f'{100.0 * ca / max(len(eval_blocks), 1):8.0f}% '
              f'{100.0 * cs / max(n_splittable, 1):11.0f}% '
              f'{st / max(ca, 1):10.1f}')

    print("\ncompare the +states rows against their classes rows at the same")
    print("data volume: a clear lift settles the generator's input contract;")
    print("no lift means the missing information is elsewhere (probe history,")
    print("or the block's joint behaviour rather than marginal states).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
