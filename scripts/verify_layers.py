"""Does every layer still do its job on the compositional benchmark?

    uv run scripts/verify_layers.py --worlds 20

One check per claim in the plan, run against a saved core so the script
measures rather than trains:

  core    infers rules well enough to rank the truth near the top
  adapt   recovers a usable rule set at test time with no labels
  memory  reorders search without changing what search concludes
  evolve  promotes only what survives a held-out guard set
  explore designs experiments that raise evidence coverage

`memory` is checked for answer-preservation rather than for speed. A library
that reorders search into a different member of a scoring tie looks like a
13x cost win and is actually a 58% -> 29% accuracy loss, which is how this
layer failed the first time it was measured.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from sentinel.adapt.hypothesis import classes_from_mechanics, scorable_segment
from sentinel.adapt.search import (
    ALL_HYPOTHESES,
    SIMPLICITY_ORDER,
    core_order,
    exhaustive_search,
    replays_to_truth,
)
from sentinel.core import CoreConfig, load_core, load_split
from sentinel.core.agent import read_layout
from sentinel.core.data import exploration_history
from sentinel.evolve.search import evolve
from sentinel.explore import staged_exploration
from sentinel.memory.library import Entry, SkillLibrary
from sentinel.memory.signature import Signature


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="corpus/split_wide.json")
    ap.add_argument("--core", default="corpus/cores/seed0.safetensors")
    ap.add_argument("--worlds", type=int, default=20)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--skip-evolve", action="store_true")
    args = ap.parse_args()

    splits = load_split(args.split)
    specs = splits["holdout_mechanics"][: args.worlds]
    core = load_core(args.core, CoreConfig(cycles=args.cycles)) if Path(args.core).exists() else None
    if core is None:
        print(f"no core at {args.core}; run scripts/measure_core.py first", file=sys.stderr)

    library = SkillLibrary()
    recovered = ranks = 0
    replays: list[int] = []
    core_ranks: list[int] = []
    same_answer = 0
    scored = 0

    started = time.perf_counter()
    for spec in specs:
        history = exploration_history(spec, 0, 60)
        segment = scorable_segment(history)
        if len(segment.steps) < 5:
            continue
        observed = read_layout(segment.initial.grid, spec.field_size)
        signature = Signature.from_frame(segment.initial, spec.field_size)
        truth = classes_from_mechanics(spec.mechanics)
        scored += 1

        found = exhaustive_search(history, observed, spec.field_size)
        recovered += int(found.best.classes == truth)
        replays.append(found.replays)

        primed = exhaustive_search(
            history, observed, spec.field_size,
            order=library.rank(signature, list(SIMPLICITY_ORDER)),
        )
        same_answer += int(primed.best.classes == found.best.classes)
        library.add(Entry(spec.world_id, signature, found.best.classes, found.best.fitness))

        if core is not None:
            core_ranks.append(replays_to_truth(core_order(core, history), truth))

    print(f"\n=== adapt: test-time inference, no labels ({scored} worlds) ===")
    print(f"  exact rule set recovered: {recovered}/{scored} ({recovered / max(1, scored):.0%})")
    print(f"  verifier replays: mean {np.mean(replays):.0f} of {len(ALL_HYPOTHESES)}")

    print(f"\n=== memory: reordering must not change the answer ===")
    print(f"  same conclusion as unprimed search: {same_answer}/{scored}")
    print(f"  library holds {len(library)} verified worlds")

    if core_ranks:
        a = np.array(core_ranks)
        print(f"\n=== core: rank of the truth among {len(ALL_HYPOTHESES)} ===")
        print(f"  median {np.median(a):.0f}   mean {a.mean():.1f}   top-100 {(a <= 100).mean():.0%}")

    print(f"\n=== explore: experiments aimed with inferred rules ===")
    landed = aimed = 0
    for spec in specs[:8]:
        result = staged_exploration(spec, seed=0)
        landed += result.landings
        aimed += int(result.planned)
    print(f"  landings on a target: {landed} over 8 worlds; phase two aimed in {aimed}/8")

    if not args.skip_evolve:
        print(f"\n=== evolve: promotion gated on held-out worlds ===")
        archive = evolve(
            train=splits["train"][:5], guard=splits["holdout_seed"][:5],
            generations=1, population=2, verbose=False,
        )
        print(f"  {len(archive)} versions archived, {len(archive.promoted)} promoted")
        print(f"  rollback -> {archive.rollback().summary()}")

    print(f"\ntotal {time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
