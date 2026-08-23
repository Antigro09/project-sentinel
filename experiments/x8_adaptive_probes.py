"""X8: does ADAPTIVE probe selection collapse the quotient where fixed probes cannot?

X7 measured the claim that probe length sets the quotient's resolution and
found it half-true: signatures grow as L^1.17 with no saturation by L=64,
and the residual blocks differ only in mechanics a fixed probe never
excites -- `ordered`, `switches`, `wait-free`. Longer fixed probes cannot
finish the job.

But bisimulation is relative to the probe, and nothing says the probe must
be fixed in advance. The disagreement machinery in `explore/version_space`
already exists for exactly this: take the action the surviving hypotheses
most disagree about, because an action every hypothesis predicts identically
cannot split anything. If adaptive selection works, the residual blocks
should collapse in far fewer probe steps than a fixed sequence needs --
which is the corrected form of the quotient argument, and the mechanism X4
(search over a DSL, pruned by behaviour) would inherit.

RESULT (measured, 12 worlds, full 5,760-hypothesis space):

    step   adaptive   fixed (X7)
       2       5.8        15.3
       8     105.8       134.8
      64     327.5       828.6

Adaptive LOSES by 2.5x at every horizon. Diagnosis is more interesting than
the number: one-step disagreement saturates -- after ~16 steps EVERY action
scores exactly |behaviours| (each behaviour has a unique continuation for
every action), so the selector is choosing on noise from there on. The
residual blocks are NOT unreachable states; they are hypotheses that agree
on every state visited but disagree on ORDER-DEPENDENT structure
(`ordered_targets` vs not: same reachable set, different visit order).
A state-based signature cannot see them in principle.

That is a measurement about representation, not about probing: the
signature must encode TRANSITIONS, not just states -- or the residual must
be resolved by acting IN the world rather than probing the model, which is
what `planned_information_gain_history` does with sequences.
"""

from __future__ import annotations

import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from sentinel.adapt.search import ALL_HYPOTHESES
from sentinel.adapt.hypothesis import mechanics_from_classes
from sentinel.core.agent import read_layout
from sentinel.core.data import exploration_history
from sentinel.core.universal import PROBE_ACTIONS
from sentinel.core import load_split
from sentinel.env.types import Action
from sentinel.explore.version_space import state_key
from sentinel.gen.grid import GridWorldModel
from sentinel.gen.spec import WorldSpec

WORLDS = 12
STEPS = 64
LEGAL = (1, 2, 3, 4, 5)


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


def adaptive_curve(spec, rng) -> list[int]:
    """Distinct signatures after each adaptively chosen probe step.

    At every step, each surviving behaviour votes with its predicted next
    state; the action with the most distinct predictions is appended to the
    probe. This is Query-by-Committee over the quotient rather than over
    states -- the committee is the set of distinct behaviours, deduplicated
    so a block of 1,152 bisimilar hypotheses counts once.
    """
    history = exploration_history(spec, 0, 30)
    observed = read_layout(history.initial.grid, spec.field_size)

    models = []
    for classes in ALL_HYPOTHESES:
        sp = WorldSpec(world_id="q", seed=0, field_size=spec.field_size,
                       mechanics=mechanics_from_classes(classes), levels=(observed,))
        models.append((classes, GridWorldModel(sp), None))

    probe: list[int] = []
    curve = []
    for _ in range(STEPS):
        # Roll every hypothesis under the probe so far; keep one
        # representative per distinct signature.
        reps: dict[tuple, list] = {}
        for classes, model, _ in models:
            sig = signature(model, model.init_state(), probe)
            reps.setdefault(sig, []).append((classes, model))
        curve.append(len(reps))

        # Choose the next action by maximum disagreement among behaviours.
        # Ties go to the LEAST-USED action: pure greedy tie-breaking to the
        # lowest id makes the probe periodic (1,1,1,...) and the quotient
        # stops refining -- a probe with no action diversity cannot excite
        # rules about the actions it never takes.
        from collections import Counter
        used = Counter(probe)
        best_aid, best_key = LEGAL[0], (-1, 0)
        for aid in LEGAL:
            outcomes = set()
            for sig, group in reps.items():
                classes, model = group[0]
                try:
                    extended = signature(model, _state_after(model, probe), (aid,))
                    outcomes.add((sig, extended))
                except Exception:
                    continue
            key = (len(outcomes), -used[aid])
            if key > best_key:
                best_aid, best_key = aid, key
        probe.append(best_aid)
    return curve


def _state_after(model, probe):
    current = model.init_state()
    for aid in probe:
        try:
            current = model.transition(current, Action(aid))
        except Exception:
            break
    return current


def main() -> int:
    specs = load_split("corpus/split_wide.json")["holdout_mechanics"][:WORLDS]
    rng = np.random.default_rng(0)

    print(f"full space {len(ALL_HYPOTHESES)} hypotheses, {len(specs)} worlds, "
          f"{STEPS} probe steps\n")

    curves = []
    for spec in specs:
        curves.append(adaptive_curve(spec, rng))
        print(f"  {spec.world_id}: {curves[-1][7]} sigs @8 -> {curves[-1][-1]} @64")

    mean = np.mean(curves, axis=0)
    print(f'\n{"step":>5} {"adaptive":>9} {"fixed (X7)":>11}')
    fixed = {2: 15.3, 4: 42.8, 8: 134.8, 16: 329.9, 32: 515.1, 64: 828.6}
    for step in (2, 4, 8, 16, 32, 64):
        print(f"{step:5d} {mean[step-1]:9.1f} {fixed[step]:11.1f}")

    ratio = mean[63] / 828.6
    print(f"\nadaptive/fixed at 64 steps: {ratio:.2f}x")
    if ratio > 1.5:
        print("  ADAPTIVE WINS BIG: probe selection, not probe length, is the")
        print("  knob. The quotient argument survives in corrected form and")
        print("  X4 should prune the DSL space with adaptive probes.")
    elif ratio > 1.1:
        print("  adaptive helps modestly; the residual is partly irreducible.")
    else:
        print("  adaptive LOSES. Check whether one-step disagreement has")
        print("  saturated: if every action scores |behaviours|, the selector")
        print("  is choosing on noise and the residual is order-dependent")
        print("  structure a state-based signature cannot see.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
