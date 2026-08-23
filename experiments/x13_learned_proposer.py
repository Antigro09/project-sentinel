"""X13: can a LEARNED proposer amortise the cost of finding long-setup probes?

X7-X12 mapped the proposal problem completely:

    free head   ~40% of residual blocks split in <=2 steps; beam finds
                them all (X11), adopted.
    costly tail the rest need long order-dependent probes; greedy (X10),
                beam (X11) and hybrid (X12) all fail; random sampling
                pays ~264 steps per split.

The tail's information "does not exist in any prefix" -- but it DOES exist
in completed sequences, and random search finds those. So the remaining
move is to AMORTISE: train a small network on (block, splitting-probe)
pairs harvested by random search, then let it propose whole sequences in
one forward pass. This is the Level 4 ranker-to-generator shift in
miniature, and the first trained component of the proposal mechanism.

What the network sees is legitimate at test time: the version space knows
WHICH hypotheses survived, so block composition (which classes appear per
head among members) is an input, not a label. The splitting probe is the
target. Nothing here uses the world's true mechanics.

Measured on held-out worlds (never used in training):

    recovery@K    fraction of blocks split by one of K sampled probes
    steps/split   K x probe length, against random's 264

If the learned proposer beats random's cost curve at comparable coverage,
the generator exists and X4 (DSL search pruned by behaviour) inherits it.
If it only matches random, the tail's structure is too world-specific to
amortise from 12 worlds -- which is itself a scaling measurement: how much
training data does proposal need?

RESULT (measured, 8 train worlds / 4 held-out, 355 train pairs, 240 eval
blocks, K=8 sampled probes of length 24):

    learned  (K=8):   155 splits (65%), 150.8 steps/split
    random   (40x24): 222 splits (92%), 147.2 steps/split

The first cut does not win: coverage is well short and cost per split is
no better than sampling. Two confounds are visible in the harvest itself:

  - w2000003 contributed ZERO split pairs (all 60 blocks missed by all 40
    attacks) yet sits in the EVAL set -- its blocks may be inseparable at
    len-24, so no proposer could score there. Coverage numbers mix
    'proposer failed' with 'block unsplittable'.
  - Training saw only 355 pairs from 8 worlds. The proposer must map a
    block's class composition to an order-dependent sequence; that is a
    many-to-one structure it cannot get from one example per block type.

The honest reading: amortisation is NOT YET demonstrated, and the two
candidate causes make opposite predictions -- more training worlds fix
the first, richer inputs (member states, not just class counts) fix the
second. Distinguishing them is the next experiment: scale worlds first,
since data is cheap here, then enrich features if the curve stays flat.
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

N_WORLDS = 12
TRAIN_WORLDS = 8
BASE_LEN = 64
ATTACK_LEN = 24
RANDOM_ATTACKS = 40
MEMBER_CAP = 30
BLOCK_CAP = 60
SEQ_LEN = 24
N_ACTIONS = 5
SAMPLES_K = 8

# Block features: per head, which classes appear among members (count
# vector), plus log block size. 27 class slots + 1.
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
    """Which classes appear per head among the block's members.

    This is what the version space legitimately knows at test time: the
    surviving hypotheses and their rule classes. No true-mechanics label.
    """
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    offset = 0
    for i, (_, n) in enumerate(HEADS):
        for classes in members:
            feats[offset + classes[i]] += 1.0
        offset += n
    feats[-1] = np.log1p(len(members))
    return feats


def harvest(spec, rng):
    """Residual blocks for one world, with splitting probes where random finds them."""
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

    pairs = []   # (features, probe) where a split was found
    misses = []  # features where none of the attacks split
    for block in residual:
        members = block if len(block) <= MEMBER_CAP else [
            block[i] for i in rng.choice(len(block), MEMBER_CAP, replace=False)]
        feats = block_features(members)
        found = None
        for ap in attacks:
            if len({run(models[c], ap) for c in members}) > 1:
                found = ap
                break
        if found is not None:
            pairs.append((feats, np.array(found, dtype=np.int64) - 1))
        else:
            misses.append(feats)
    return pairs, misses


class Proposer(nn.Module):
    """Block features -> a whole probe sequence, one shot.

    One shot rather than autoregressive: the failure of X10/X11 was step-wise
    scoring with no view of the completed sequence. A network trained on
    completed sequences can emit one wholesale -- that is the entire point
    of amortising.
    """

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


def main() -> int:
    specs = load_split("corpus/split_wide.json")["holdout_mechanics"][:N_WORLDS]
    rng = np.random.default_rng(0)

    train_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    eval_misses: list[np.ndarray] = []

    t0 = time.perf_counter()
    for i, spec in enumerate(specs):
        pairs, misses = harvest(spec, rng)
        if i < TRAIN_WORLDS:
            train_pairs.extend(pairs)
        else:
            eval_misses.extend(misses)
        print(f"  {spec.world_id}: {len(pairs)} split pairs, "
              f"{len(misses)} misses  ({time.perf_counter() - t0:.0f}s)")

    print(f"\ntotal: {len(train_pairs)} train pairs, {len(eval_misses)} eval misses")

    # --- train
    model = Proposer()
    mx.random.seed(0)
    opt = optim.AdamW(learning_rate=1e-3)

    def loss_fn(m, x, y):
        logits = m(x)
        return mx.mean(nn.losses.cross_entropy(logits.reshape(-1, N_ACTIONS),
                                               y.reshape(-1)))

    X = mx.array(np.stack([f for f, _ in train_pairs]))
    Y = mx.array(np.stack([p for _, p in train_pairs]))
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    for epoch in range(60):
        perm = np.random.default_rng(epoch).permutation(len(train_pairs))
        losses = []
        for start in range(0, len(perm), 64):
            idx = mx.array(perm[start:start + 64])
            loss, grads = loss_and_grad(model, X[idx], Y[idx])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            losses.append(float(loss))
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1}: loss {np.mean(losses):.4f}")

    # --- evaluate on held-out worlds: harvest again, keeping member lists
    # (features alone cannot verify a split; the models are needed).
    learned_splits = 0
    learned_steps = 0
    random_splits = 0
    random_steps = 0
    eval_data = []
    for spec in specs[TRAIN_WORLDS:]:
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
        for classes, model_ in models.items():
            blocks.setdefault(run(model_, tuple(base)), []).append(classes)
        residual = sorted((v for v in blocks.values() if len(v) > 1),
                          key=len, reverse=True)[:BLOCK_CAP]
        attacks = [tuple(int(rng.integers(1, 6)) for _ in range(ATTACK_LEN))
                   for _ in range(RANDOM_ATTACKS)]
        for block in residual:
            members = block if len(block) <= MEMBER_CAP else [
                block[j] for j in rng.choice(len(block), MEMBER_CAP, replace=False)]
            eval_data.append(([models[c] for c in members],
                              block_features(members), attacks))

    for member_models, feats, attacks in eval_data:
        # learned
        x = mx.array(feats[None])
        logits = model(x)
        mx.eval(logits)
        probs = np.array(mx.softmax(logits[0], axis=-1))
        steps = 0
        for _ in range(SAMPLES_K):
            seq = tuple(int(rng.choice(N_ACTIONS, p=p)) + 1 for p in probs)
            steps += len(seq)
            if len({run(m, seq) for m in member_models}) > 1:
                learned_splits += 1
                break
        learned_steps += steps

        # random baseline on the same block
        steps = 0
        for ap in attacks:
            steps += ATTACK_LEN
            if len({run(m, ap) for m in member_models}) > 1:
                random_splits += 1
                break
        random_steps += steps

    total = len(eval_data)
    print(f"\nblocks: {total}")
    print(f"  learned  (K={SAMPLES_K}): {learned_splits:4d} splits "
          f"({100.0 * learned_splits / max(total, 1):.0f}%), "
          f"{learned_steps / max(learned_splits, 1):7.1f} steps/split")
    print(f"  random   (40x24):       {random_splits:4d} splits "
          f"({100.0 * random_splits / max(total, 1):.0f}%), "
          f"{random_steps / max(random_splits, 1):7.1f} steps/split")

    lc = learned_steps / max(learned_splits, 1)
    rc = random_steps / max(random_splits, 1)
    if learned_splits >= 0.9 * random_splits and lc < 0.7 * rc:
        print("\nVERDICT: the learned proposer WINS on cost at near-equal")
        print("coverage. Amortisation works: proposal is now a forward pass.")
        print("This is the Level 4 generator; X4 inherits it.")
    elif lc < rc:
        print("\nVERDICT: cheaper per split but lower coverage. Useful as the")
        print("hybrid's first stage; the tail still needs sampling.")
    else:
        print("\nVERDICT: no win yet. Either too few training worlds (scaling")
        print("measurement: collect more) or block features do not determine")
        print("the probe -- in which case proposal needs richer state input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
