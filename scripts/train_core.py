"""Train the core and run the Phase 3 gate.

    uv run scripts/train_core.py --build-data     # first time: encode episodes
    uv run scripts/train_core.py                  # train on the cached dataset

Data is cached to disk because encoding replays episodes through the
engine, which is slow, while training reads the same arrays many times.

The gate is judged on held-out mechanic COMBINATIONS — rule mixtures the
core was never trained on. Unseen-seed accuracy is reported alongside it
for contrast: the gap between the two is the difference between
interpolating over taught rules and generalizing to untaught ones.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from sentinel.core import (
    CoreConfig,
    TrainConfig,
    build_dataset,
    evaluate,
    load_dataset,
    load_split,
    majority_baseline,
    run_gate,
    save_dataset,
    train,
)

DATA_DIR = Path("corpus/core")


def build(args: argparse.Namespace) -> int:
    split_path = Path(args.split)
    if not split_path.exists():
        print(
            f"{split_path} not found. Generate worlds first with:\n"
            "  uv run scripts/build_corpus.py --n-train 2000 --n-holdout-seed 200 "
            "--n-holdout-mechanics 200",
            file=sys.stderr,
        )
        return 1

    splits = load_split(split_path)
    print(f"world specs: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, limit in (
        ("train", args.train_worlds),
        ("holdout_seed", args.holdout_worlds),
        ("holdout_mechanics", args.holdout_worlds),
    ):
        specs = splits.get(name, [])
        if not specs:
            print(f"  {name}: none available, skipping")
            continue
        started = time.perf_counter()
        print(f"  encoding {name} ({min(len(specs), limit)} worlds)...")
        ds = build_dataset(
            specs,
            episodes_per_world=args.episodes,
            limit=limit,
            verbose=True,
        )
        save_dataset(ds, DATA_DIR / f"{name}.npz")
        print(f"  {name}: {ds.summary()}  ({time.perf_counter() - started:.0f}s)")
        print(f"    label balance: {ds.label_balance()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="corpus/split.json")
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--train-worlds", type=int, default=800)
    parser.add_argument("--holdout-worlds", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    args = parser.parse_args()

    if args.build_data:
        return build(args)

    paths = {n: DATA_DIR / f"{n}.npz" for n in ("train", "holdout_seed", "holdout_mechanics")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print(
            "Missing encoded data: " + ", ".join(missing) + "\nRun with --build-data first.",
            file=sys.stderr,
        )
        return 1

    train_set = load_dataset(paths["train"])
    seed_set = load_dataset(paths["holdout_seed"])
    mech_set = load_dataset(paths["holdout_mechanics"])

    # Early-stop against held-out SEEDS, never against held-out mechanics.
    # Two reasons. Tuning on the mechanics set is test-set leakage — it is
    # the thing the gate judges. And charge_period is not even measurable
    # there: the withheld combinations happen to contain no period-4 worlds,
    # so a model predicting the training majority scores exactly 0.000 and
    # the stopping signal is pure artifact.
    model, _ = train(
        train_set,
        seed_set,
        core_config=CoreConfig(d_model=args.d_model, cycles=args.cycles),
        train_config=TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr
        ),
    )

    print()
    seed_acc = evaluate(model, seed_set)
    seed_base = majority_baseline(train_set, seed_set)
    print("held-out SEEDS (same rules, new layouts) — measures interpolation:")
    for name in seed_acc:
        print(f"  {name:18} {seed_acc[name]:.3f}  (baseline {seed_base[name]:.3f})")

    print()
    print(run_gate(model, train_set, mech_set).report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
