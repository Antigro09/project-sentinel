"""X1: does core-guided search scale sub-linearly with hypothesis-space size?

This is the experiment that decides whether the whole programme scales.

Transformers were validated small and then scaled, and that worked because
next-token prediction has a MEASURED scaling law -- the curve was known
before the money was spent. This architecture is search-and-verify, and
search has no established scaling law: the literature hypothesises one for
search-based synthesis but the comparative measurement does not exist.

So measure it. Cost is `rank of the true rule set` under each ordering,
which is what search must walk before it finds the answer. Vary the space
size and fit cost ~ |H|^alpha:

    alpha ~ 1     cost grows with the space; a prior buys a constant factor
                  and scaling this architecture buys nothing
    alpha < 1     the prior does more work as the space grows, which is what
                  "scaling the architecture" would have to mean

Sub-spaces are built by restricting classes per head, and each is scored
only on worlds whose TRUE rule set lies inside it -- otherwise the answer is
not in the space and rank is meaningless.
"""

from __future__ import annotations

import itertools
import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from sentinel.adapt.hypothesis import NCLASS, classes_from_mechanics, scorable_segment
from sentinel.adapt.search import core_order, replays_to_truth
from sentinel.core import CoreConfig, load_core, load_split
from sentinel.core.data import exploration_history
from sentinel.memory.library import Entry, SkillLibrary
from sentinel.memory.signature import Signature
from sentinel.core.agent import read_layout

# Space sizes to sweep. Every size uses EVERY world.
SIZES = (50, 100, 250, 500, 1000, 2500, 5760)

# Why subsampling rather than restricting classes per head: a restricted
# space only contains some worlds' true rule sets, and rank is meaningless
# for a world whose answer is not in the space. The first version of this
# experiment did that and left 5 worlds in the smallest space, which made
# the fit worthless -- random ordering came out at alpha 2.16 when its true
# exponent is exactly 1.00 by construction.
#
# That calibration is kept below as a CHECK. Random's expected rank is
# |H|/2 whatever the space, so if the estimator cannot recover alpha ~ 1.0
# for random, it cannot be trusted for the core either.


def sampled_space(full, size, truth, rng):
    """`size` hypotheses drawn from `full`, always including the truth."""
    if size >= len(full):
        return list(full)
    others = [c for c in full if c != truth]
    picked = [others[i] for i in rng.choice(len(others), size=size - 1, replace=False)]
    return picked + [truth]


def main() -> int:
    core = load_core("corpus/cores/seed0.safetensors", CoreConfig(cycles=8))
    specs = load_split("corpus/split_wide.json")["holdout_mechanics"][:120]
    full = [c for c in itertools.product(*[range(n) for n in NCLASS])]

    episodes = []
    for spec in specs:
        history = exploration_history(spec, 0, 60)
        segment = scorable_segment(history)
        if len(segment.steps) < 5:
            continue
        episodes.append((spec, history, classes_from_mechanics(spec.mechanics)))
    print(f"{len(episodes)} episodes, full space {len(full)}\n")

    # The core's ranking over the FULL space is computed once per episode;
    # a sampled space just filters that order, which keeps the sweep cheap
    # and guarantees the same prior is being measured at every size.
    core_orders = {}
    for i, (spec, history, truth) in enumerate(episodes):
        core_orders[i] = core_order(core, history)

    print(f'{"|H|":>6} {"random":>9} {"simplicity":>11} {"core":>9} {"core/random":>12}')
    rows = []
    for size in SIZES:
        rand_r, simp_r, core_r = [], [], []
        for i, (spec, history, truth) in enumerate(episodes):
            rng = np.random.default_rng(1000 + i)
            space = set(sampled_space(full, size, truth, rng))
            shuffled = [c for c in space]
            rng.shuffle(shuffled)
            simplicity = sorted(space, key=lambda c: (sum(c), c))
            filtered_core = [c for c in core_orders[i] if c in space]
            rand_r.append(replays_to_truth(shuffled, truth))
            simp_r.append(replays_to_truth(simplicity, truth))
            core_r.append(replays_to_truth(filtered_core, truth))
        rows.append((size, np.mean(rand_r), np.mean(simp_r), np.mean(core_r)))
        print(f"{size:6d} {np.mean(rand_r):9.1f} {np.mean(simp_r):11.1f} {np.mean(core_r):9.1f} "
              f"{np.mean(core_r)/max(1e-9, np.mean(rand_r)):12.3f}")

    sizes = np.log(np.array([r[0] for r in rows], dtype=float))
    print(f'\nfitted growth exponent alpha, cost ~ |H|^alpha')
    alphas = {}
    for i, name in ((1, "random"), (2, "simplicity"), (3, "core")):
        cost = np.log(np.maximum([r[i] for r in rows], 1e-9))
        alphas[name] = float(np.polyfit(sizes, cost, 1)[0])
        verdict = "sub-linear" if alphas[name] < 0.85 else (
            "linear" if alphas[name] < 1.15 else "super-linear")
        print(f"  {name:12} alpha = {alphas[name]:5.2f}   {verdict}")

    ok = abs(alphas["random"] - 1.0) < 0.15
    print(f'\nCALIBRATION: random must come out at alpha ~ 1.00 by construction.')
    print(f'  measured {alphas["random"]:.2f} -> {"estimator trustworthy" if ok else "ESTIMATOR BROKEN, ignore the rest"}')
    if ok:
        print(f'\n  core alpha {alphas["core"]:.2f}: '
              + ("SUB-LINEAR -- the prior does more work as the space grows, "
                 "which is the result that says scale this architecture."
                 if alphas["core"] < 0.85 else
                 "roughly LINEAR -- the core buys a large constant factor "
                 f"({np.mean([r[1]/max(1e-9,r[3]) for r in rows]):.0f}x) but not a better exponent."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
