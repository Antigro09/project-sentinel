"""How well does each prior rank the hypothesis space?

    uv run scripts/measure_search.py --worlds 60

The core's job in the plan is not to answer but to PRUNE -- to guess which
hypotheses are worth testing. That claim is only measurable once the space
is too large to enumerate cheaply, which is why the compositional generator
exists: 5,760 rule sets at ~17.5ms a verifier replay.

Accuracy is held constant by construction. The verifier still decides what
is true, so a confident wrong prior costs replays and nothing else, and
ranking quality shows up purely as how deep search must go to reach the
truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from sentinel.adapt.hypothesis import classes_from_mechanics, scorable_segment
from sentinel.adapt.search import ALL_HYPOTHESES, SIMPLICITY_ORDER, core_order, replays_to_truth
from sentinel.core import CoreConfig, load_core, load_dataset, load_split
from sentinel.core.agent import read_layout
from sentinel.core.data import exploration_history
from sentinel.memory.library import Entry, SkillLibrary
from sentinel.memory.signature import Signature

REPLAY_SECONDS = 0.0175


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="corpus/split_wide.json")
    ap.add_argument("--core", default="corpus/cores/seed0.safetensors")
    ap.add_argument("--worlds", type=int, default=60)
    ap.add_argument("--cycles", type=int, default=8)
    args = ap.parse_args()

    core = None
    if Path(args.core).exists():
        core = load_core(args.core, CoreConfig(cycles=args.cycles))
    else:
        print(f"no core at {args.core}; skipping the core arm", file=sys.stderr)

    specs = load_split(args.split)["holdout_mechanics"][: args.worlds]
    rng = np.random.default_rng(0)
    random_order = [ALL_HYPOTHESES[i] for i in rng.permutation(len(ALL_HYPOTHESES))]

    library = SkillLibrary()
    rows: dict[str, list[int]] = {}
    for spec in specs:
        history = exploration_history(spec, 0, 60)
        segment = scorable_segment(history)
        if len(segment.steps) < 5:
            continue
        signature = Signature.from_frame(segment.initial, spec.field_size)
        truth = classes_from_mechanics(spec.mechanics)

        rows.setdefault("random", []).append(replays_to_truth(random_order, truth))
        rows.setdefault("simplicity", []).append(replays_to_truth(SIMPLICITY_ORDER, truth))
        rows.setdefault("library", []).append(
            replays_to_truth(library.rank(signature, ALL_HYPOTHESES), truth)
        )
        if core is not None:
            rows.setdefault("core", []).append(
                replays_to_truth(core_order(core, history), truth)
            )
        library.add(Entry(spec.world_id, signature, truth, 1.0))

    n = len(rows["random"])
    print(f"\nrank of the TRUE rule set among {len(ALL_HYPOTHESES)}, {n} worlds")
    print(f"{'ordering':12} {'median':>8} {'mean':>9} {'top-10':>8} {'top-100':>8} {'seconds':>9}")
    for name, values in rows.items():
        a = np.array(values)
        print(
            f"{name:12} {np.median(a):8.0f} {a.mean():9.1f} {(a <= 10).mean():8.0%} "
            f"{(a <= 100).mean():8.0%} {a.mean() * REPLAY_SECONDS:9.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
