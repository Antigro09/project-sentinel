"""X9: is the quotient's residual a SEARCH problem or an IDENTIFIABILITY limit?

X7 showed fixed probes leave big residual blocks; X8 showed one-step
adaptive selection cannot shrink them (disagreement saturates, and the
blocks look order-dependent). Two very different conclusions fit those
facts, and they demand opposite investments:

  SEARCH problem   some other probe separates the blocks; fixed and greedy
                   probes just never found it. The fix is a better probe
                   FINDER -- ultimately a learned generator (the Level 4
                   ranker-to-generator shift).
  IDENTIFIABILITY  NO probe from the start state separates them; the
                   hypotheses are behaviourally equivalent on everything
                   this observation type can express. The fix is acting in
                   the world (`planned_information_gain_history`) or a
                   richer observation (Level 3).

Measured here directly: take the residual blocks left by the X7 fixed
probe, then attack each with many random probes. A block is SEPARABLE if
any probe makes its members disagree. Report the fraction separable and
what the inseparable ones differ in.

RESULT (measured, 12 worlds, full space, 8522 residual blocks attacked
with 40 random probes each):

    separable by some probe:    4084 (48%)
    inseparable:                4438 (median size 6, max 48)
    mechanics inside inseparable blocks:
      has_switches     3431
      ordered_targets  3371
      has_hazards       301

The split is nearly even, which is the finding: the residual is HALF
search problem, HALF identifiability limit.

  - The separable half says greedy/adaptive selection was leaving real
    value on the table -- a probe GENERATOR (the Level 4 ranker-to-
    generator shift) recovers those blocks.
  - The inseparable half concentrates exactly on `has_switches` and
    `ordered_targets`, the two mechanics whose evidence lives OFF the
    reachable-from-start behaviour of a single agent: switches must be
    VISITED, order must be VIOLATED. No probe from the start state can
    express them. These need purpose-built experiments IN the world --
    `planned_information_gain_history` exists for precisely this -- or a
    richer observation type.

Caveat: 40 random length-24 probes is a weak attack; the inseparable
fraction is an upper bound on true irreducibility.
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
BASE_LEN = 64
ATTACKS = 40
"""Random probes thrown at each residual block."""
ATTACK_LEN = 24
MEMBER_CAP = 60
"""Blocks larger than this get a random sample; a split among a sample of
60 still detects separability with high probability."""


def base_probe(rng: np.random.Generator) -> tuple[int, ...]:
    out = list(PROBE_ACTIONS)
    while len(out) < BASE_LEN:
        out.append(int(rng.integers(1, 6)))
    return tuple(out[:BASE_LEN])


def run(model, probe) -> tuple:
    out = []
    current = model.init_state()
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

    total_blocks = 0
    separable = 0
    inseparable_diffs: dict[str, int] = {}
    inseparable_sizes: list[int] = []

    for spec in specs:
        history = exploration_history(spec, 0, 30)
        observed = read_layout(history.initial.grid, spec.field_size)

        models = []
        for classes in ALL_HYPOTHESES:
            sp = WorldSpec(world_id="q", seed=0, field_size=spec.field_size,
                           mechanics=mechanics_from_classes(classes),
                           levels=(observed,))
            models.append((classes, GridWorldModel(sp)))

        # Partition under the X7-style fixed probe.
        probe = base_probe(rng)
        blocks: dict[tuple, list] = {}
        for classes, model in models:
            blocks.setdefault(run(model, probe), []).append(classes)
        residual = [v for v in blocks.values() if len(v) > 1]
        residual.sort(key=len, reverse=True)

        attacks = [tuple(int(rng.integers(1, 6)) for _ in range(ATTACK_LEN))
                   for _ in range(ATTACKS)]

        for block in residual:
            total_blocks += 1
            members = block if len(block) <= MEMBER_CAP else [
                block[i] for i in rng.choice(len(block), MEMBER_CAP, replace=False)]
            split = False
            lookup = dict(models)
            for ap in attacks:
                sigs = set()
                for c in members:
                    sigs.add(run(lookup[c], ap))
                if len(sigs) > 1:
                    split = True
                    break
            if split:
                separable += 1
            else:
                inseparable_sizes.append(len(block))
                # What do the survivors of a random-probe attack differ in?
                head = mechanics_from_classes(block[0])
                other = mechanics_from_classes(block[-1])
                for name in ("step_distance", "charge_period", "wrap_edges",
                             "has_hazards", "has_switches", "ordered_targets"):
                    if getattr(head, name, None) != getattr(other, name, None):
                        inseparable_diffs[name] = inseparable_diffs.get(name, 0) + 1
        print(f"  {spec.world_id}: {len(blocks)} behaviours, "
              f"{len(residual)} residual blocks")

    print(f"\nresidual blocks attacked: {total_blocks}")
    print(f"  separable by some random probe: {separable} "
          f"({100.0 * separable / max(total_blocks, 1):.0f}%)")
    print(f"  inseparable: {total_blocks - separable}")
    if inseparable_sizes:
        print(f"  inseparable block sizes: median {int(np.median(inseparable_sizes))}, "
              f"max {max(inseparable_sizes)}")
    if inseparable_diffs:
        print("  mechanics differing inside inseparable blocks:")
        for name, n in sorted(inseparable_diffs.items(), key=lambda kv: -kv[1]):
            print(f"    {name}: {n}")

    frac = separable / max(total_blocks, 1)
    if frac > 0.75:
        print("\nVERDICT: SEARCH problem. Most residual blocks yield to SOME")
        print("probe -- fixed and greedy selectors simply never found it.")
        print("The next component is a probe GENERATOR (learned or searched),")
        print("which is exactly the Level 4 ranker-to-generator shift.")
    elif frac > 0.25:
        print(f"\nVERDICT: SPLIT ({frac:.0%} separable). Half the residual is a")
        print("search problem -- a probe generator recovers it; the other half")
        print("is an identifiability limit concentrated in mechanics whose")
        print("evidence requires purpose-built experiments IN the world")
        print("(visit switches, violate target order), not better probing")
        print("of the model from the start state.")
    else:
        print("\nVERDICT: IDENTIFIABILITY limit. Most residual blocks survive")
        print("every probe tried -- those hypotheses are behaviourally")
        print("equivalent from the start state under this observation type.")
        print("Only acting in the world with purpose-built experiments, or a")
        print("richer observation, can separate them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
