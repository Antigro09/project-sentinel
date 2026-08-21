"""Does the behavioural quotient stay finite as the hypothesis space grows?

The sharpest objection to this architecture: in a Turing-complete
representation the version space is infinite, so you cannot roll every
survivor forward under a probe. Enumeration dies, and with it the whole
method.

The answer, if there is one, is that hypotheses are only distinguishable up
to BEHAVIOUR. Two programs that predict identically under a probe are
bisimilar and need not both be carried. So the quantity that matters is not
how many hypotheses exist, but how many DISTINCT SIGNATURES they produce --
and if that saturates while the space keeps growing, an infinite space is
not fatal, because search is really over the quotient.

Measured here by sampling K hypotheses and counting distinct probe
signatures as K rises:

  signatures ~ K          the quotient is as big as the space; the objection
                          stands and Turing-completeness breaks this method
  signatures saturating   the space collapses onto a small set of behaviours
                          and only the quotient has to be searched
"""

from __future__ import annotations

import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from sentinel.adapt.hypothesis import mechanics_from_classes
from sentinel.adapt.search import ALL_HYPOTHESES
from sentinel.core.agent import read_layout
from sentinel.core.data import exploration_history
from sentinel.core.universal import PROBE_ACTIONS
from sentinel.core import load_split
from sentinel.env.types import Action
from sentinel.explore.version_space import state_key
from sentinel.gen.grid import GridWorldModel
from sentinel.gen.spec import WorldSpec

SAMPLES = (10, 25, 50, 100, 250, 500, 1000, 2500, 5760)


def signature(model, state, probe=PROBE_ACTIONS) -> tuple:
    """The behaviour a hypothesis exhibits under the probe."""
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
    specs = load_split("corpus/split_wide.json")["holdout_mechanics"][:12]
    rng = np.random.default_rng(0)

    print(f"probe length {len(PROBE_ACTIONS)}, {len(specs)} worlds\n")
    print(f'{"sampled":>8} {"signatures":>11} {"ratio":>7}')
    rows = []
    for k in SAMPLES:
        distinct = []
        for spec in specs:
            history = exploration_history(spec, 0, 30)
            observed = read_layout(history.initial.grid, spec.field_size)
            idx = rng.choice(len(ALL_HYPOTHESES), size=min(k, len(ALL_HYPOTHESES)), replace=False)
            seen = set()
            for i in idx:
                classes = ALL_HYPOTHESES[int(i)]
                sp = WorldSpec(world_id="q", seed=0, field_size=spec.field_size,
                               mechanics=mechanics_from_classes(classes), levels=(observed,))
                m = GridWorldModel(sp)
                seen.add(signature(m, m.init_state()))
            distinct.append(len(seen))
        mean = float(np.mean(distinct))
        rows.append((k, mean))
        print(f"{k:8d} {mean:11.1f} {mean/k:7.3f}")

    ks = np.log([r[0] for r in rows])
    sig = np.log(np.maximum([r[1] for r in rows], 1e-9))
    slope = float(np.polyfit(ks, sig, 1)[0])
    print(f"\nsignatures ~ K^{slope:.2f}")
    if slope < 0.5:
        print("  STRONGLY SUB-LINEAR: the space collapses onto few behaviours.")
        print("  An infinite hypothesis space is then not fatal -- search is over")
        print("  the quotient, which stays small.")
    elif slope < 0.85:
        print("  sub-linear: the quotient grows slower than the space, so bisimulation")
        print("  buys real headroom but does not by itself defeat an infinite space.")
    else:
        print("  ~linear: distinct behaviours grow with the space. The objection stands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
