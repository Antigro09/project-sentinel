"""Per-label accuracy of the trained core, scored honestly.

    uv run scripts/measure_core.py --seeds 3

Two things this does that a naive accuracy loop does not.

**Labels are scored only where they are defined.** `wait_advances_charge`
governs whether waiting ticks the hidden counter, so a world with no
counter cannot answer it; `gates_start_open` cannot be observed where there
are no gates. Scoring those rewards a lucky guess -- it inflated
gates_start_open across 46% of held-out episodes.

**The prior is computed on the same rows.** A label's majority-class
baseline changes once undefined rows are dropped, and comparing a masked
accuracy against an unmasked prior would flatter the model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from sentinel.core import CoreConfig, TrainConfig, load_dataset, save_core
from sentinel.core.encoding import HEADS, defined_mask
from sentinel.core.train import evaluate, train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="corpus/core_wide")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--save-to", default="corpus/cores")
    args = ap.parse_args()

    data = Path(args.data)
    tr = load_dataset(data / "train.npz")
    sd = load_dataset(data / "holdout_seed.npz")
    mech = load_dataset(data / "holdout_mechanics.npz")
    print(f"train={len(tr)}  holdout_seed={len(sd)}  holdout_mechanics={len(mech)}")

    prior, defined = {}, {}
    for i, (name, _) in enumerate(HEADS):
        mask = defined_mask(mech.labels, i)
        _, counts = np.unique(mech.labels[mask][:, i], return_counts=True)
        prior[name] = counts.max() / counts.sum()
        defined[name] = float(mask.mean())

    runs = []
    for seed in range(args.seeds):
        started = time.perf_counter()
        core, _ = train(
            tr, sd,
            core_config=CoreConfig(cycles=args.cycles),
            train_config=TrainConfig(batch_size=args.batch_size, seed=seed),
            verbose=False,
        )
        if args.save_to:
            save_core(core, Path(args.save_to) / f"seed{seed}.safetensors")
        runs.append(evaluate(core, mech))
        print(f"  seed {seed} ({time.perf_counter() - started:.0f}s)")

    print(f"\n{'label':24} {'core':>17} {'prior':>7} {'defined':>8}")
    for name, _ in HEADS:
        v = np.array([r[name] for r in runs])
        flag = "  <-- at/below prior" if v.mean() <= prior[name] + 0.01 else ""
        print(
            f"{name:24} {v.mean():.3f} +/- {v.std():.3f} {prior[name]:7.3f} "
            f"{defined[name]:7.0%}{flag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
