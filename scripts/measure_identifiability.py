"""Which rules does the evidence actually determine?

    uv run scripts/measure_identifiability.py --worlds 30

A label the evidence cannot distinguish is not a training failure, and no
amount of learning fixes it. This separates the two, and the separation
matters: `charge_period` is determined in 100% of worlds and was simply
unread by the network, while `ordered_targets` is determined in 6% and a
core sitting near its prior is responding correctly.

The measure is the verifier's own: flip one label in the true rule set and
see whether the fitness falls. No learning is involved anywhere.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from sentinel.adapt.hypothesis import classes_from_mechanics, score_hypothesis, scorable_segment
from sentinel.core.agent import read_layout
from sentinel.core.data import exploration_history, load_split, probing_history
from sentinel.core.encoding import HEADS
from sentinel.explore import staged_exploration

EPISODES = {
    "random": lambda spec: exploration_history(spec, 0, 60),
    "greedy-probe": lambda spec: probing_history(spec, 0, 60),
    "staged": lambda spec: staged_exploration(spec, seed=0).history,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="corpus/split_wide.json")
    ap.add_argument("--worlds", type=int, default=25)
    ap.add_argument("--kind", default="random", choices=sorted(EPISODES))
    args = ap.parse_args()

    specs = load_split(args.split)["holdout_mechanics"][: args.worlds]
    make = EPISODES[args.kind]
    names = [n for n, _ in HEADS]
    nclass = [n for _, n in HEADS]

    detected = {n: [] for n in names}
    for spec in specs:
        history = make(spec)
        segment = scorable_segment(history)
        if len(segment.steps) < 5:
            continue
        observed = read_layout(segment.initial.grid, spec.field_size)
        truth = classes_from_mechanics(spec.mechanics)
        base = score_hypothesis(truth, history, observed, spec.field_size).fitness
        for i, name in enumerate(names):
            worst = base
            for value in range(nclass[i]):
                if value == truth[i]:
                    continue
                flipped = list(truth)
                flipped[i] = value
                worst = min(
                    worst,
                    score_hypothesis(tuple(flipped), history, observed, spec.field_size).fitness,
                )
            detected[name].append(base - worst > 1e-9)

    n = len(detected[names[0]])
    print(f"episodes: {args.kind}, {n} worlds\n")
    print(f"{'label':24} {'determined by evidence':>24}")
    for name in names:
        print(f"{name:24} {np.mean(detected[name]):24.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
