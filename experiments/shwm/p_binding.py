"""C / E / F. Global role binding versus the local detector, across count strata.

The binder is a DeepSets over PER-COLOUR tokens, so it is permutation-equivariant in the
colours by construction: relabelling the palette permutes the tokens and permutes the
output identically. It emits a soft colour-to-role assignment, and the event is then the
M2F relational expression lifted onto soft roles --

    event = max over cells of  P(agent here now) * P(not agent here before)
                               * P(switch here before)

-- so it is supervised by the PUBLIC event label alone. No role name, no palette id and
no evaluator state ever reaches it, which is what makes regime B a grounding claim rather
than a lookup.

Feature-restricted arms answer the question phase O could not: if count-only works in
COUNT_INFORMATIVE and dies in COUNT_COLLISION, the earlier mechanism was cardinality.

    .venv-shwm/bin/python experiments/shwm/p_binding.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import n_heads as heads
import n_interfaces as ifaces
import o_core as O
import p_core as P
from m2d_core import ARTIFACTS, write
from p_core import GRID, N_ROLES, ROLE_INDEX, TOKEN_WIDTH

SEEDS = (36_000, 36_001)
TRAIN_LAYOUTS = tuple(range(110_000, 110_040))
TEST_LAYOUTS = tuple(range(111_000, 111_020))
DEV_PALETTES = tuple(range(9_300, 9_316))
UNSEEN_PALETTES = tuple(range(9_400, 9_416))

# Feature slices inside a per-colour token.
RGB = slice(0, 3)
COUNT = slice(3, 4)
MOMENTS = slice(4, 8)
MOTION = slice(8, 10)

VIEWS = {
    "count_only": [COUNT],
    "motion_only": [MOTION],
    "count_plus_motion": [COUNT, MOTION],
    "full_token": [RGB, COUNT, MOMENTS, MOTION],
}
MAX_COLOURS = 8


def episode_tokens(episode: P.StratumEpisode) -> tuple[np.ndarray, np.ndarray]:
    """(T, MAX_COLOURS, TOKEN_WIDTH) tokens and (T, 12, 12) colour indices."""
    tokens = np.zeros((episode.length, MAX_COLOURS, TOKEN_WIDTH), dtype=np.float32)
    index = np.zeros((episode.length, GRID, GRID), dtype=np.int64)
    for t in range(episode.length):
        block, colours = P.colour_tokens(episode, t)
        frame = episode.frames[t][::2, ::2, :]
        lookup = {tuple(int(v) for v in c): k for k, c in enumerate(colours)}
        n = min(len(block), MAX_COLOURS)
        tokens[t, :n] = block[:n]
        for r in range(GRID):
            for c in range(GRID):
                index[t, r, c] = min(lookup[tuple(int(v) for v in frame[r, c])],
                                     MAX_COLOURS - 1)
    return tokens, index


def mask_view(tokens: np.ndarray, view: str) -> np.ndarray:
    out = np.zeros_like(tokens)
    for part in VIEWS[view]:
        out[..., part] = tokens[..., part]
    return out


def build_pairs(episodes, view: str):
    before_t, after_t, before_i, after_i, event = [], [], [], [], []
    for episode in episodes:
        tokens, index = episode_tokens(episode)
        tokens = mask_view(tokens, view)
        for t in range(1, episode.length):
            before_t.append(tokens[t - 1]); after_t.append(tokens[t])
            before_i.append(index[t - 1]); after_i.append(index[t])
            event.append(episode.base.event[t])
    return (np.stack(before_t), np.stack(after_t), np.stack(before_i),
            np.stack(after_i), np.array(event, dtype=np.float32))


def build_binder(seed: int, width: int = 64):
    """DeepSets over colour tokens -> soft role assignment -> relational event."""
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(seed)

    class Binder(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(TOKEN_WIDTH, width)
            self.b = nn.Linear(2 * width, width)
            self.role = nn.Linear(width, N_ROLES)

        def assign(self, tokens):
            h = nn.relu(self.a(tokens))                      # (N, K, W)
            context = mx.mean(h, axis=1, keepdims=True)
            context = mx.broadcast_to(context, h.shape)
            h = nn.relu(self.b(mx.concatenate([h, context], axis=-1)))
            return mx.softmax(self.role(h), axis=-1)          # (N, K, R)

        def __call__(self, before_tokens, after_tokens, before_index, after_index):
            role_before = self.assign(before_tokens)
            role_after = self.assign(after_tokens)
            n = before_tokens.shape[0]
            rows = mx.arange(n).reshape(n, 1)
            flat_before = before_index.reshape(n, GRID * GRID)
            flat_after = after_index.reshape(n, GRID * GRID)
            # Gather each cell's role distribution from its colour's token.
            pb = mx.take_along_axis(
                role_before, flat_before.reshape(n, GRID * GRID, 1)
                .astype(mx.int32) * mx.ones((1, 1, N_ROLES), mx.int32), axis=1)
            pa = mx.take_along_axis(
                role_after, flat_after.reshape(n, GRID * GRID, 1)
                .astype(mx.int32) * mx.ones((1, 1, N_ROLES), mx.int32), axis=1)
            agent_now = pa[:, :, ROLE_INDEX["AGENT"]]
            agent_before = pb[:, :, ROLE_INDEX["AGENT"]]
            switch_before = pb[:, :, ROLE_INDEX["SWITCH"]]
            evidence = agent_now * (1.0 - agent_before) * switch_before
            return mx.log(mx.maximum(mx.max(evidence, axis=1), 1e-6)
                          / mx.maximum(1.0 - mx.max(evidence, axis=1), 1e-6))

    model = Binder()
    mx.eval(model.parameters())
    return model


def train_binder(train, seed: int, updates: int = 2000):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    bt, at, bi, ai, y = train
    model = build_binder(seed)
    optimizer = optim.AdamW(learning_rate=3e-3)
    rng = np.random.default_rng(seed)
    tensors = [mx.array(x) for x in (bt, at, bi, ai, y)]
    for _ in range(updates):
        pick = mx.array(rng.integers(0, len(y), min(128, len(y))))
        args = [t[pick] for t in tensors]

        def objective(m):
            return mx.mean(nn.losses.binary_cross_entropy(
                m(*args[:4]), args[4], with_logits=True))

        value, grads = nn.value_and_grad(model, objective)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, value)

    def infer(block):
        out = model(*[mx.array(x) for x in block[:4]])
        mx.eval(out)
        return np.asarray(out)
    return infer, model


def local_conv(train_eps, test_eps, seed: int) -> float:
    """Phase O's detector, unchanged, as the negative baseline."""
    def pack(episodes):
        before, after, action, event, maps = [], [], [], [], []
        for episode in episodes:
            for t in range(1, episode.length):
                before.append(episode.frames[t - 1].astype(np.float32) / 255.0)
                after.append(episode.frames[t].astype(np.float32) / 255.0)
                one = np.zeros(4, np.float32)
                one[episode.base.actions[t - 1]] = 1.0
                action.append(one)
                event.append(episode.base.event[t])
                block = np.zeros(GRID * GRID, np.float32)
                r, c = episode.base.positions[t]
                block[r * GRID + c] = episode.base.event[t]
                maps.append(block)
        return (np.stack(before), np.stack(after), np.stack(action),
                np.array(event, np.float32), np.stack(maps))

    tb, ta, tac, _, tm = pack(train_eps)
    eb, ea, eac, ee, _ = pack(test_eps)

    def encode(b, a):
        stacked = np.concatenate([ifaces.pool_to_slots(b, GRID),
                                  ifaces.pool_to_slots(a, GRID)], axis=-1)
        return stacked @ ifaces.frozen_projection(stacked.shape[-1],
                                                  ifaces.SLOT_WIDTH, 20_002)

    model, _ = heads.train_target(encode(tb, ta), tac, tm, "spatial_scalar",
                                  GRID * GRID, seed, updates=2000)
    logits = heads.predict(model, encode(eb, ea), eac)
    return heads.binary_metrics(logits.max(axis=1)[:, None], ee)["balanced_accuracy"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--palettes", type=int, default=8)
    parser.add_argument("--layouts", type=int, default=24)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "p-binding.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    dev = DEV_PALETTES[:arguments.palettes]
    unseen = UNSEEN_PALETTES[:arguments.palettes]
    train_layouts = TRAIN_LAYOUTS[:arguments.layouts]
    test_layouts = TEST_LAYOUTS[:arguments.layouts // 2]

    report: dict[str, Any] = {
        "strata": list(P.STRATA), "development_palettes": list(dev),
        "unseen_palettes": list(unseen), "train_layouts": list(train_layouts),
        "test_layouts": list(test_layouts), "views": list(VIEWS),
        "supervision": "public event label only (regime B); no role name reaches a model",
        "results": {}}

    for stratum in P.STRATA:
        print(f"\n=== {stratum} ===", flush=True)
        base_train = O.collect_appearance(train_layouts, "HIDDEN_PALETTE_CONVENTION",
                                          list(dev), 1, 9, seed=11)
        base_test = O.collect_appearance(test_layouts, "HIDDEN_PALETTE_CONVENTION",
                                         list(unseen), 1, 9, seed=313)
        train_eps = P.build_stratum(base_train, stratum, dev[0], seed=11)
        test_eps = P.build_stratum(base_test, stratum, unseen[0], seed=313)
        print(f"  {len(train_eps)} train / {len(test_eps)} test episodes; "
              f"decoy cells {train_eps[0].decoy_cells}", flush=True)

        block: dict[str, float] = {}
        for view in VIEWS:
            scores = []
            for seed in SEEDS[:arguments.seeds]:
                train = build_pairs(train_eps, view)
                test = build_pairs(test_eps, view)
                infer, _ = train_binder(train, seed)
                scores.append(heads.binary_metrics(infer(test)[:, None],
                                                   test[4])["balanced_accuracy"])
            block[f"binder__{view}"] = float(np.mean(scores))
            print(f"  {'binder ' + view:34s} {block[f'binder__{view}']:.4f}", flush=True)
        conv = [local_conv(train_eps, test_eps, seed) for seed in SEEDS[:arguments.seeds]]
        block["local_conv_baseline"] = float(np.mean(conv))
        print(f"  {'local conv (phase O baseline)':34s} "
              f"{block['local_conv_baseline']:.4f}", flush=True)
        report["results"][stratum] = block

    informative = report["results"]["COUNT_INFORMATIVE"]
    collision = report["results"]["COUNT_COLLISION"]
    report["p3_count_only_does_not_explain_it"] = bool(
        collision["binder__full_token"] > collision["binder__count_only"] + 0.05)
    report["cardinality_lookup_diagnosis"] = bool(
        informative["binder__count_only"] > 0.8 and collision["binder__count_only"] < 0.65)
    report["p5_global_binder_beats_local"] = bool(
        collision["binder__full_token"] > collision["local_conv_baseline"] + 0.05)
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nP3 (count-only does not explain the collision result): "
          f"{report['p3_count_only_does_not_explain_it']}")
    print(f"count-only is a cardinality lookup: "
          f"{report['cardinality_lookup_diagnosis']}")
    print(f"P5 (global binder beats the local detector on unseen palettes): "
          f"{report['p5_global_binder_beats_local']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
