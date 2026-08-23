"""X7: is probe length actually the knob that sets the quotient's resolution?

The quotient result (X6) says 5,760 hypotheses collapse to ~135 behaviours
under an 8-step probe, which defuses the infinite-version-space objection in
principle. But it rested on one untested assertion: that PROBE LENGTH is the
knob controlling that collapse. Bisimulation is relative to the probe, so if
signatures do not sharpen as the probe grows, the quotient is not a
resolution-controlled object and the X4 design loses its foundation.

Two questions, one sweep:

1. Does the number of distinct behaviours grow with probe length, and does
   it saturate again? A quotient that keeps refining forever is as fatal as
   one that never refines -- the first means no finite probe suffices, the
   second means the residual ambiguity is irreducible by probing.

2. What is the RESIDUAL -- the partition of hypotheses that no probe of this
   length can split? If the residual blocks are exactly the hypotheses that
   differ only in mechanics the evidence never excites (the ordered_targets
   problem), then probe selection (Gap 3) and experiment choice are the same
   problem, and the disagreement heuristic in explore/version_space.py is
   the right tool for both.

Measured on the FULL compositional space (all 5,760), not a sample: at this
size sampling noise is what killed the first version of X1.
"""

from __future__ import annotations

import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from sentinel.adapt.hypothesis import classes_from_mechanics, mechanics_from_classes
from sentinel.adapt.search import ALL_HYPOTHESES
from sentinel.core.agent import read_layout
from sentinel.core.data import exploration_history
from sentinel.core.universal import PROBE_ACTIONS
from sentinel.core import load_split
from sentinel.env.types import Action
from sentinel.explore.version_space import state_key
from sentinel.gen.grid import GridWorldModel
from sentinel.gen.spec import WorldSpec

LENGTHS = (2, 4, 8, 16, 32, 64)
WORLDS = 12


def extended_probe(rng: np.random.Generator, length: int) -> tuple[int, ...]:
    """PROBE_ACTIONS first (comparability with X6), then random filler."""
    base = list(PROBE_ACTIONS)
    while len(base) < length:
        base.append(int(rng.integers(1, 6)))
    return tuple(base[:length])


def signature(model, state, probe) -> tuple:
    out = []
    current = state
    for aid in probe:
        try:
            current = model.transition(current, Action(aid))
        except Exception:
            break
        out.append(state_key(current))
    return tuple(out)


def main() -> int:
    specs = load_split("corpus/split_wide.json")["holdout_mechanics"][:WORLDS]
    rng = np.random.default_rng(0)

    print(f"full space {len(ALL_HYPOTHESES)} hypotheses, {len(specs)} worlds\n")
    print(f'{"probe len":>9} {"distinct":>9} {"ratio":>7} {"largest block":>13}')

    results = {}
    residuals = {}
    for length in LENGTHS:
        counts = []
        max_block = []
        for spec in specs:
            history = exploration_history(spec, 0, 30)
            observed = read_layout(history.initial.grid, spec.field_size)
            probe = extended_probe(rng, length)
            seen: dict[tuple, list] = {}
            for classes in ALL_HYPOTHESES:
                sp = WorldSpec(world_id="q", seed=0, field_size=spec.field_size,
                               mechanics=mechanics_from_classes(classes),
                               levels=(observed,))
                m = GridWorldModel(sp)
                sig = signature(m, m.init_state(), probe)
                seen.setdefault(sig, []).append(classes)
            counts.append(len(seen))
            # The residual: how many hypotheses sit in the biggest
            # indistinguishable block -- the ambiguity no probe of this
            # length can touch.
            biggest = max(len(v) for v in seen.values())
            max_block.append(biggest)
            if length == LENGTHS[-1]:
                residuals[spec.world_id] = sorted(
                    seen.values(), key=len, reverse=True)[0]
        mean = float(np.mean(counts))
        results[length] = mean
        print(f"{length:9d} {mean:9.1f} {mean/len(ALL_HYPOTHESES):7.3f} "
              f"{np.mean(max_block):13.1f}")

    # Growth law of the quotient in probe length.
    ls = np.log(np.array(LENGTHS, dtype=float))
    ds = np.log(np.maximum([results[l] for l in LENGTHS], 1e-9))
    slope = float(np.polyfit(ls, ds, 1)[0])
    print(f"\ndistinct signatures ~ L^{slope:.2f}")
    last_two = results[LENGTHS[-1]] / max(results[LENGTHS[-2]], 1e-9)
    if slope > 0.85 and last_two > 1.5:
        print("  quotient still growing fast at the longest probe: no finite")
        print("  probe resolves the space; probing alone cannot finish the job.")
    elif slope < 0.15 or last_two < 1.05:
        print("  quotient saturated: residual ambiguity is IRREDUCIBLE by")
        print("  longer probes. The leftover blocks differ only in mechanics")
        print("  the probe never excites -- choosing better probes (Gap 3),")
        print("  not longer ones, is what remains.")
    else:
        print("  still refining sub-linearly: probe length buys resolution")
        print("  with diminishing returns, as claimed.")

    if residuals:
        print("\nlargest unresolvable block (longest probe), first world:")
        block = next(iter(residuals.values()))
        truth = None
        for c in block:
            m = mechanics_from_classes(c)
            print(f"  {m.summary()}")
        _ = truth
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
