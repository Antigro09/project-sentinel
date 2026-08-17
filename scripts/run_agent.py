"""Train the core, then use it to actually solve unseen worlds.

    uv run scripts/run_agent.py --worlds 60

Reports solve rate three ways — with the true rules, with rules the core
inferred by watching, and with a default guess. The gap between the last
two is the core's real contribution; the gap to the first is what remains.
"""

from __future__ import annotations

import argparse
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

from sentinel.core import CoreConfig, TrainConfig, load_dataset, load_split
from sentinel.core.agent import benchmark
from sentinel.core.train import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="corpus/split.json")
    parser.add_argument("--worlds", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--explore-steps", type=int, default=24)
    parser.add_argument("--cycles", type=int, default=8)
    args = parser.parse_args()

    train_set = load_dataset("corpus/core/train.npz")
    seed_set = load_dataset("corpus/core/holdout_seed.npz")

    print("training the core (no LLM involved)...")
    started = time.perf_counter()
    core, _ = train(
        train_set,
        seed_set,
        core_config=CoreConfig(cycles=args.cycles),
        train_config=TrainConfig(epochs=args.epochs, patience=20, batch_size=64),
        verbose=False,
    )
    print(f"  {core.parameter_count():,} parameters, {time.perf_counter() - started:.0f}s")

    splits = load_split(args.split)
    specs = splits["holdout_mechanics"][: args.worlds]
    print(f"\nsolving {len(specs)} worlds with rule combinations never seen in training\n")

    started = time.perf_counter()
    for result in benchmark(specs, core, explore_steps=args.explore_steps):
        print("  " + result.summary())
    print(f"\n({time.perf_counter() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
