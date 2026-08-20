"""Does inference convert into solved worlds?

    uv run scripts/measure_end_to_end.py --worlds 60

Per-label accuracy has misled this project more than once, so this is the
number that decides things. Four conditions, identical verifier, identical
worlds; only the ordering of hypotheses and who breaks a tie changes.

Note that `rules exact` and `solve` disagree, and that the disagreement is
the point. A hypothesis differing from the truth only on behaviour the
episode never exercised plans identically to the truth, so exact-match
weights all labels equally while planning weights them by how much they
bend the trajectory.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from sentinel.adapt.hypothesis import scorable_segment
from sentinel.adapt.search import SIMPLICITY_ORDER, core_order, exhaustive_search
from sentinel.core import CoreConfig, load_core, load_split
from sentinel.core.agent import read_layout, run_episode
from sentinel.core.data import exploration_history
from sentinel.gen.spec import Mechanics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="corpus/split_wide.json")
    ap.add_argument("--core", default="corpus/cores/seed0.safetensors")
    ap.add_argument("--worlds", type=int, default=60)
    ap.add_argument("--cycles", type=int, default=8)
    args = ap.parse_args()

    if not Path(args.core).exists():
        print(f"no core at {args.core}; run scripts/measure_core.py first", file=sys.stderr)
        return 1
    core = load_core(args.core, CoreConfig(cycles=args.cycles))
    specs = load_split(args.split)["holdout_mechanics"][: args.worlds]
    default = Mechanics(step_distance=1, charge_period=None)

    rows = {
        k: {"solved": 0, "exact": 0, "replays": [], "n": 0}
        for k in ("true mechanics", "core-ordered", "simplicity", "default guess")
    }
    started = time.perf_counter()
    for i, spec in enumerate(specs):
        history = exploration_history(spec, 0, 60)
        segment = scorable_segment(history)
        if len(segment.steps) < 5:
            continue
        observed = read_layout(segment.initial.grid, spec.field_size)

        by_core = exhaustive_search(
            history, observed, spec.field_size,
            order=core_order(core, history), tie_break="order",
        )
        by_simple = exhaustive_search(
            history, observed, spec.field_size, order=SIMPLICITY_ORDER
        )
        for label, mech, found in (
            ("true mechanics", spec.mechanics, None),
            ("core-ordered", by_core.mechanics, by_core),
            ("simplicity", by_simple.mechanics, by_simple),
            ("default guess", default, None),
        ):
            row = rows[label]
            row["n"] += 1
            row["solved"] += int(run_episode(spec, mech, seed=0).solved)
            row["exact"] += int(mech.summary() == spec.mechanics.summary())
            if found is not None:
                row["replays"].append(found.replays)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(specs)}  ({time.perf_counter() - started:.0f}s)")

    print(f"\n{'condition':16} {'solve':>7} {'rules exact':>12} {'replays':>9}")
    for label, row in rows.items():
        replays = f"{np.mean(row['replays']):.0f}" if row["replays"] else "-"
        print(
            f"{label:16} {row['solved'] / max(1, row['n']):7.1%} "
            f"{row['exact'] / max(1, row['n']):12.1%} {replays:>9}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
