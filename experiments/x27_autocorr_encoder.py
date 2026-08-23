"""X27: displacement autocorrelation features -- a representational fix
that did not carry the charge signal.

X26 established charge errors are correlated across seeds: the problem is
representational. The programme history records a '+0.70 autocorrelation
spike at the right lag' measured on charge data during Phase 3, so the
first candidate was per-value autocorrelation of displacement magnitude
over lags 1..8, added to a simplified TransitionEncoder.

MEASURED (800 worlds x 2 episodes x 100 epochs, n=60 eval):

    charge 15% vs X26 calibrated baseline 14% +/- 2 -- NO LIFT.
    (hazard 73%, gates 100%, switch 57% beat prior; step/edge low because
    this simplified encoder dropped the centroid machinery the original
    TransitionEncoder has.)

NEGATIVE RESULT, and why it is informative: the autocorrelated signal was
computed over MASS CHANGE (cells appearing/disappearing), not POSITION
CHANGE. A charge tick moves the agent one extra cell -- mass change per
value barely differs between tick and non-tick moves; the periodicity
lives in WHERE the agent lands, i.e. centroid displacement. The X24
long-window result showed position sequences do carry it (42% once).

CONSEQUENCE: the next attempt must autocorrelate CENTROID DISPLACEMENT
(dx, dy per value) rather than mass delta -- which requires porting the
full moment machinery from TransitionEncoder into the test encoder. The
simplified encoder used here also explains its own regressions elsewhere.

Also fixed en route: double-batching in data construction (g[None] then
stacked), a wrong moment_proj input size, and mx.repeat misuse.
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
from sentinel.core.model import CoreConfig, TinyRecursiveCore, _sinusoidal  # noqa: E402
from sentinel.core.model import RelativeAttention, RecursiveBlock  # noqa: E402
from sentinel.core.encoding import (  # noqa: E402
    CHANNELS,
    CROP,
    MAX_TRANSITIONS,
    N_CELL_VALUES,
)
from sentinel.env.types import Action  # noqa: E402
from sentinel.explore.version_space import state_key  # noqa: E402
from sentinel.gen.grid import GridWorld, initial_state, transition_state  # noqa: E402
from sentinel.gen.spec import LevelSpec, WorldSpec  # noqa: E402

from x17_dsl_search import AXES, SpecCache, compile_program, complexity  # noqa: E402

TRAIN_WORLDS = 800
EPISODES = 2
EPOCHS = 100
EVAL_WORLDS = 60
AC_LAGS = 8


def autocorr_features(disp: mx.array) -> mx.array:
    """Per-value lagged correlation of displacement magnitude.

    disp: (B, T, N_VALUES) displacement magnitudes.
    Returns (B, N_VALUES * AC_LAGS).
    """
    if len(disp.shape) != 3:
        raise ValueError(f"autocorr expected 3D (B,T,V), got {disp.shape}")
    b, t, nv = disp.shape
    feats = []
    for lag in range(1, AC_LAGS + 1):
        if t <= lag:
            feats.append(mx.zeros((b, nv)))
            continue
        x = disp[:, : t - lag, :]
        y = disp[:, lag:, :]
        xm = mx.mean(x, axis=1)
        ym = mx.mean(y, axis=1)
        dx = x - xm[:, None, :]
        dy = y - ym[:, None, :]
        num = mx.sum(dx * dy, axis=1)
        den = mx.sqrt(mx.maximum(
            mx.sum(dx * dx, axis=1) * mx.sum(dy * dy, axis=1), 1e-9))
        feats.append(num / den)
    return mx.concatenate(feats, axis=-1)


class AcEncoder(nn.Module):
    """TransitionEncoder + displacement autocorrelation features.

    Mirrors sentinel.core.model.TransitionEncoder exactly, adding one
    extra projection fed by the autocorrelation features.
    """

    def __init__(self, cfg: CoreConfig):
        super().__init__()
        from sentinel.core.model import DISP_BINS
        self.cell = nn.Embedding(N_CELL_VALUES, cfg.cell_embed)
        self.action = nn.Embedding(7, cfg.d_model)
        # log-mass before + log-mass after + mean changed mask = 33
        self.moment_proj = nn.Linear(N_CELL_VALUES * 2 + 1, cfg.d_model)
        self.ac_proj = nn.Linear(N_CELL_VALUES * AC_LAGS, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def __call__(self, grids, actions):
        if len(grids.shape) == 4:
            # single unbatched episode (T, C, C, CH): add the batch dim
            grids = grids[None]
            actions = actions[None]
        b, t = grids.shape[0], grids.shape[1]
        before = grids[..., 0]   # (B, T, C, C)
        after = grids[..., 1]

        def moments(plane):
            onehot = mx.stack([(plane == v).astype(mx.float32)
                               for v in range(N_CELL_VALUES)], axis=-1)
            mass = onehot.sum(axis=(2, 3))
            return mass

        mb = moments(before)
        ma = moments(after)
        changed = (before != after).astype(mx.float32)
        # Per-value displacement magnitude per transition: how much of
        # value v appeared/disappeared. Generic; not agent-specific.
        disp = mx.abs(ma - mb)  # (B, T, N_VALUES)

        ac = autocorr_features(disp)  # (B, N_VALUES * AC_LAGS)
        b_, t_ = mb.shape[0], mb.shape[1]
        tokens = self.moment_proj(
            mx.concatenate([mx.log1p(mb), mx.log1p(ma),
                            changed.mean(axis=(2, 3))[:, :, None]],
                           axis=-1))
        tokens = tokens + mx.repeat(self.ac_proj(ac)[:, None, :], t_, axis=1)

        ids = mx.where(actions < 0, 6, actions)
        return self.norm(tokens + self.action(ids))


class AcCore(nn.Module):
    """TinyRecursiveCore with the AcEncoder swapped in."""

    def __init__(self, cfg: CoreConfig, head_sizes):
        super().__init__()
        self.cfg = cfg
        c = cfg
        self.encoder = AcEncoder(c)
        self.pos = _sinusoidal(MAX_TRANSITIONS, c.d_model)
        self.block = RecursiveBlock(c)
        self.y_init = mx.random.normal((1, 1, c.d_model)) * 0.02
        self.z_init = mx.random.normal((1, 1, c.d_model)) * 0.02
        self.mix_z = nn.Linear(c.d_model * 2, c.d_model)
        self.mix_y = nn.Linear(c.d_model * 2, c.d_model)
        self.norm_z = nn.LayerNorm(c.d_model)
        self.norm_y = nn.LayerNorm(c.d_model)
        self.norm_out = nn.LayerNorm(c.d_model)
        self.heads = [nn.Linear(c.d_model, n) for n in head_sizes]

    def __call__(self, grids, actions):
        c = self.cfg
        if len(grids.shape) == 4:
            grids = grids[None]
            actions = actions[None]
        b = grids.shape[0]
        evidence = self.encoder(grids, actions) + self.pos
        y = mx.broadcast_to(self.y_init, (b, 1, c.d_model))
        z = mx.broadcast_to(self.z_init, (b, 1, c.d_model))
        for _ in range(c.cycles):
            for _ in range(c.inner_steps):
                seq = mx.concatenate([z, y, evidence], axis=1)
                seq = self.block(seq)
                z = self.norm_z(self.mix_z(mx.concatenate([z, seq[:, :1]], axis=-1)))
            seq = mx.concatenate([y, z, evidence], axis=1)
            seq = self.block(seq)
            y = self.norm_y(self.mix_y(mx.concatenate([y, seq[:, :1]], axis=-1)))
        out = self.norm_out(y[:, 0])
        return [head(out) for head in self.heads]


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
    rng = np.random.default_rng(5000)
    train = []
    for i in range(TRAIN_WORLDS):
        truth = sample_truth(rng)
        size = int(rng.integers(9, 14))
        level = make_identifiable_level(rng, size, truth)
        observed = LevelSpec(
            start=level.start, walls=level.walls, hazards=level.hazards,
            targets=tuple(sorted(level.targets)),
            switches=level.switches, gates=level.gates,
        )
        eps = []
        for ep in range(2):
            hist = random_episode(truth, observed, size,
                                  seed=int(rng.integers(0, 10**6)))
            g, a = encode_history(hist)
            eps.append((g, a))
        train.append((eps, mech_to_labels(truth)))

    erng = np.random.default_rng(9000)
    ev = []
    for wi in range(EVAL_WORLDS):
        truth = sample_truth(erng)
        size = int(erng.integers(9, 14))
        level = make_identifiable_level(erng, size, truth)
        observed = LevelSpec(
            start=level.start, walls=level.walls, hazards=level.hazards,
            targets=tuple(sorted(level.targets)),
            switches=level.switches, gates=level.gates,
        )
        g, a = encode_history(random_episode(truth, observed, size,
                                             seed=int(erng.integers(0, 10**6))))
        ev.append((g, a, mech_to_labels(truth), truth, observed, size))
    print(f"  built ({time.perf_counter() - t0:.0f}s)")

    grids_l, actions_l, labels_l = [], [], []
    for eps, labels in train:
        for g, a in eps:
            grids_l.append(g)
            actions_l.append(a)
            labels_l.append(labels)
    X = mx.array(np.stack(grids_l))
    A = mx.array(np.stack(actions_l))
    Y = mx.array(np.stack(labels_l))

    axes_sizes = tuple(len(ax) for ax in AXES)
    model = AcCore(CoreConfig(), axes_sizes)
    mx.random.seed(0)
    opt = optim.AdamW(learning_rate=1e-3)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    trng = np.random.default_rng(0)
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
        if (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch + 1}: loss {np.mean(losses):.4f} "
                  f"({time.perf_counter() - t0:.0f}s)")

    names = ("step", "charge", "edge", "hazard",
             "switch", "ordered", "gates", "wait")
    correct = np.zeros(len(AXES))
    counted = np.zeros(len(AXES))
    label_matrix = np.stack([lb for _, _, lb, *_ in ev])
    charge_by_bucket = {"none": [0, 0], "short": [0, 0], "long": [0, 0]}
    charge_values = AXES[1]
    for g, a, labels, *_ in ev:
        logits = model(mx.array(g), mx.array(a))
        mx.eval(logits)
        preds = [int(mx.argmax(l[0])) for l in logits]
        for h in range(len(AXES)):
            correct[h] += preds[h] == labels[h]
            counted[h] += 1
        true_cp = charge_values[labels[1]]
        key = ("none" if true_cp is None
               else ("short" if true_cp <= 10 else "long"))
        charge_by_bucket[key][1] += 1
        charge_by_bucket[key][0] += preds[1] == labels[1]

    print("\nper-head accuracy (autocorrelation encoder):")
    for h, name in enumerate(names):
        vals, counts = np.unique(label_matrix[:, h], return_counts=True)
        prior = counts.max() / counts.sum()
        acc = correct[h] / max(counted[h], 1)
        mark = " <-- beats prior" if acc > prior else ""
        print(f"   {name:8} {acc:6.0%}  (prior {prior:.0%}){mark}")
    print("  charge by bucket:")
    for key, (c, t) in charge_by_bucket.items():
        if t:
            print(f"     {key:6} {c:3d}/{t:3d} ({100.0 * c / t:4.0f}%)")

    charge_acc = correct[1] / max(counted[1], 1)
    print(f"\nbaseline from X26: 14% +/- 2 (n=60)")
    print(f"with autocorrelation features: {charge_acc:.0%}")
    print("\nverdict:")
    if charge_acc > 0.25:
        print("   REPRESENTATIONAL FIX CONFIRMED: autocorrelation features")
        print("   lift charge reading well above the calibrated baseline.")
        print("   Port into TransitionEncoder properly and re-run the loop.")
    elif charge_acc > 0.18:
        print("   Direction confirmed but small; combine with a longer window.")
    else:
        print("   Autocorrelation of appearance-displacement does not carry")
        print("   the periodicity signal; the next candidate is position-")
        print("   based displacement (centroid movement), which requires")
        print("   porting the full moment machinery rather than mass only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
