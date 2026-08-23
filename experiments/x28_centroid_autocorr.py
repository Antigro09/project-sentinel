"""X28: centroid-displacement autocorrelation -- the last representational
candidate, and it also fails.

Following X27's negative (mass-change autocorrelation carries no charge
signal), this ported the full moment machinery and autocorrelated CENTROID
displacement (dx, dy, |d| per value over lags 1..8), added alongside every
existing TransitionEncoder feature.

MEASURED (800 worlds x 2 episodes x 100 epochs, n=60):

    charge 3% (prior 15%) -- worse than baseline.
    ALL heads at or below prior; loss plateaued at 1.21 vs X23's 0.78 on
    identical data without the extra features.

The training itself destabilised: adding 384 autocorrelation features
projected into d_model appears to have swamped the smaller existing
feature paths. Whether a careful re-tuning (smaller ac_proj, layer norm
before sum) would recover base performance AND add the signal is untested;
two failures with different autocorrelation signals, plus X26's correlated-
errors finding, lower the prior that any single added statistic fixes
charge reading.

CHARGE ARC CLOSED WITH A RECOMMENDATION:

  - Refutation resolves charge whenever an episode ticks it (X17: truth
    always survived; X19: 12/12 bisimilar).
  - The ranker wins on every other axis (X23: 6/12 exact).
  - Charge head accuracy is bounded by window size (X24: 42% once at 64)
    and unstable across runs (X25/X26).
  - Three representational fixes failed (X27 mass-autocorr, X28 centroid-
    autocorr, X25 long episodes).

The honest engineering position: accept charge as REFUTATION-RESOLVED ONLY
at the current window, and revisit only if downstream tasks demonstrably
need the core to read it directly. The next programme investments with
clearer expected value are young-QbC-death prevention and the Level 5 /
Level 3 questions.
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

from sentinel.core.encoding import (  # noqa: E402
    CROP,
    encode_history,
)
from sentinel.core.model import (  # noqa: E402
    CoreConfig,
    N_CELL_VALUES,
    TinyRecursiveCore,
    TransitionEncoder,
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


class AcTransitionEncoder(TransitionEncoder):
    """The real TransitionEncoder + centroid-displacement autocorrelation.

    Everything from the original is preserved (moments, displacement bins,
    coordinate channels); one extra projection adds per-value autocor-
    relation of dx, dy and |d| over lags 1..AC_LAGS.
    """

    def __init__(self, cfg: CoreConfig):
        super().__init__(cfg)
        # 3 signals (dx, dy, dist) x N_VALUES x AC_LAGS
        self.ac_proj = nn.Linear(3 * N_CELL_VALUES * AC_LAGS, cfg.d_model)

    def _displacement_series(self, grids):
        """Per-value displacement components per transition.

        Returns dx, dy, dist each (B, T, N_VALUES), reusing the exact
        moment computation the base encoder uses.
        """
        coords = self.coords
        b, t = grids.shape[0], grids.shape[1]
        coords_b = mx.broadcast_to(coords, (b, t, CROP, CROP, 2))

        def centroids(plane):
            onehot = mx.stack([(plane == v).astype(mx.float32)
                               for v in range(N_CELL_VALUES)], axis=-1)
            mass = onehot.sum(axis=(2, 3))
            xs = coords_b[..., 0][..., None]
            ys = coords_b[..., 1][..., None]
            cx = (onehot * xs).sum(axis=(2, 3)) / mx.maximum(mass, 1e-6)
            cy = (onehot * ys).sum(axis=(2, 3)) / mx.maximum(mass, 1e-6)
            return cx, cy

        cxb, cyb = centroids(grids[..., 0])
        cxa, cya = centroids(grids[..., 1])
        present = ((cxb != 0) | (cyb != 0)).astype(mx.float32)
        scale = (CROP - 1) / 2.0
        dx = (cxa - cxb) * present * scale
        dy = (cya - cyb) * present * scale
        dist = mx.sqrt(dx * dx + dy * dy + 1e-9)
        return dx, dy, dist

    def _lagged_autocorr(self, series: mx.array) -> mx.array:
        """Pearson autocorrelation of a (B, T, V) series over lags."""
        b, t, v = series.shape
        feats = []
        for lag in range(1, AC_LAGS + 1):
            if t <= lag:
                feats.append(mx.zeros((b, v)))
                continue
            x = series[:, : t - lag, :]
            y = series[:, lag:, :]
            xm = mx.mean(x, axis=1)
            ym = mx.mean(y, axis=1)
            dx = x - xm[:, None, :]
            dy = y - ym[:, None, :]
            num = mx.sum(dx * dy, axis=1)
            den = mx.sqrt(mx.maximum(
                mx.sum(dx * dx, axis=1) * mx.sum(dy * dy, axis=1), 1e-9))
            feats.append(num / den)
        return mx.concatenate(feats, axis=-1)

    def __call__(self, grids, actions):
        if len(grids.shape) == 4:
            grids = grids[None]
            actions = actions[None]
        b, t = grids.shape[0], grids.shape[1]

        before = self.cell(grids[..., 0])
        after = self.cell(grids[..., 1])
        changed = grids[..., 2][..., None].astype(before.dtype)
        coords = mx.broadcast_to(self.coords, (b, t, CROP, CROP, 2))

        tokens = self.moment_proj(self._moment_features(grids))
        tokens = tokens + self.proj(
            mx.concatenate([before, after, changed, coords],
                           axis=-1).reshape(b, t, -1))

        # NEW: centroid-displacement autocorrelation.
        dx, dy, dist = self._displacement_series(grids)
        ac = mx.concatenate([
            self._lagged_autocorr(dx),
            self._lagged_autocorr(dy),
            self._lagged_autocorr(dist),
        ], axis=-1)
        tokens = tokens + self.ac_proj(ac)[:, None, :]

        ids = mx.where(actions < 0, 6,
                       mx.where(actions == -2, 6, actions))
        return self.norm(tokens + self.action(ids))


class AcCore(TinyRecursiveCore):
    """TinyRecursiveCore with AcTransitionEncoder and DSL-sized heads."""

    def __init__(self, cfg: CoreConfig, head_sizes):
        super().__init__(cfg)
        self.encoder = AcTransitionEncoder(cfg)
        self.heads = [nn.Linear(cfg.d_model, n) for n in head_sizes]


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
    for epoch in range(EPOCHS):
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
        logits = model(mx.array(g[None]), mx.array(a[None]))
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

    print("\nper-head accuracy (centroid-autocorrelation encoder):")
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
    step_acc = correct[0] / max(counted[0], 1)
    print(f"\nbaseline from X26: charge 14% +/- 2 (n=60)")
    print(f"with centroid autocorrelation: charge {charge_acc:.0%}, "
          f"step {step_acc:.0%}")
    print("\nverdict:")
    if charge_acc > 0.25 and step_acc > 0.4:
        print("   CENTROID AUTOCORRELATION CONFIRMED: charge reading lifts")
        print("   well above baseline with other heads intact. Port into")
        print("   TransitionEncoder properly and re-run the full loop.")
    elif charge_acc > 0.18:
        print("   Direction confirmed; combine with the 64-step window")
        print("   (X24 showed they should compose).")
    else:
        print("   Centroid autocorrelation also fails to lift charge. The")
        print("   remaining candidates are hierarchical period estimation")
        print("   or accepting charge as refutation-resolved only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
