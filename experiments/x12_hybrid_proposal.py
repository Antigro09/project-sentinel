"""X12: beam-first-then-random -- does the hybrid beat either alone on COST?

X11's finding: beam search cracks 38% of residual blocks at median length
2, while random sampling reaches 82% but pays length-24 every time. If a
probe is executed in the world, LENGTH is the real currency -- each step
is an action spent. So the comparison that matters is not coverage but
TOTAL PROBE COST per block split.

The hybrid, in the order X11 implies:

    1. BEAM (width 8, cap 8): nearly free splits first.
    2. RANDOM (40 x len-24): only for the blocks beam missed.

Measured against pure-random and pure-beam on identical blocks:

    total steps spent per split   the cost currency
    coverage                      sanity check against X9/X11

This is also the last searched-only word on proposal: if the hybrid's
cost curve beats random clearly, the mechanism is settled enough to build
the learned proposer on top of it (train on the pairs this pipeline
emits); if not, proposal goes straight to learning.

RESULT (measured, 12 worlds, 720 residual blocks):

    strategy    coverage  total steps  steps/split
    random           83%       158136        264.4
    beam             38%         4295        15.8
    hybrid           83%       150608       251.9

The hybrid matches random's coverage but saves only 5% of cost: beam's
cheap splits (median 2) are a small fraction of the blocks, so the
expensive random tail dominates the budget either way. The cost
structure, not the coverage, is the finding:

  - ~40% of blocks split in <=2 steps (beam finds them all, nearly free).
  - The rest need long order-dependent probes, and NO searched proposer
    finds them cheaply -- greedy (X10) and beam (X11) both fail, and
    random pays 264 steps per split for them.

So the proposal problem has a bimodal cost structure: a free head and an
expensive tail. The free head is now solved (beam, adopted). The
expensive tail is the open problem, and the two remaining options are
(a) a LEARNED sequence prior -- train on the (block, probe) pairs random
search emits, amortising the 264-step cost into a network -- or (b)
handing the tail to the in-world experiment planner, since the tail's
mechanics (switches, order) are exactly X9's inseparable-from-probes set.
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
ATTACK_LEN = 24
RANDOM_ATTACKS = 40
BEAM_WIDTH = 8
BEAM_CAP = 8
MEMBER_CAP = 30
BLOCK_CAP = 60


def advance(model, state, aid):
    try:
        return model.transition(state, Action(aid))
    except Exception:
        return state


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


def beam_probe(models, rng) -> tuple[int, ...] | None:
    """As X11 v3: full-state-ranked beam with a randomised half."""
    init = tuple((m, m.init_state()) for m in models)
    beam = [((), init)]
    for _depth in range(BEAM_CAP):
        candidates = []
        for probe, ms in beam:
            for aid in (1, 2, 3, 4, 5):
                nxt = tuple((m, advance(m, s, aid)) for m, s in ms)
                ext = probe + (aid,)
                visible = {state_key(s) for _, s in nxt}
                if len(visible) > 1:
                    return ext
                candidates.append((ext, nxt))

        def diversity(item):
            _, nxt = item
            return len({s for _, s in nxt})

        candidates.sort(key=lambda t: (-diversity(t), t[0]))
        k = BEAM_WIDTH // 2
        rest = candidates[k:]
        extra = [rest[i] for i in rng.choice(len(rest), size=min(k, len(rest)),
                                             replace=False)] if rest else []
        beam = candidates[:k] + extra
        if diversity(beam[0]) <= 1:
            return None
    return None


def main() -> int:
    specs = load_split("corpus/split_wide.json")["holdout_mechanics"][:WORLDS]
    rng = np.random.default_rng(0)

    agg = {
        "random": {"splits": 0, "steps": 0},
        "beam": {"splits": 0, "steps": 0},
        "hybrid": {"splits": 0, "steps": 0},
    }
    total_blocks = 0

    for spec in specs:
        history = exploration_history(spec, 0, 30)
        observed = read_layout(history.initial.grid, spec.field_size)

        models = {}
        for classes in ALL_HYPOTHESES:
            sp = WorldSpec(world_id="q", seed=0, field_size=spec.field_size,
                           mechanics=mechanics_from_classes(classes),
                           levels=(observed,))
            models[classes] = GridWorldModel(sp)

        base = list(PROBE_ACTIONS)
        while len(base) < BASE_LEN:
            base.append(int(rng.integers(1, 6)))
        blocks: dict[tuple, list] = {}
        for classes, model in models.items():
            blocks.setdefault(run(model, tuple(base)), []).append(classes)
        residual = sorted((v for v in blocks.values() if len(v) > 1),
                          key=len, reverse=True)[:BLOCK_CAP]

        attacks = [tuple(int(rng.integers(1, 6)) for _ in range(ATTACK_LEN))
                   for _ in range(RANDOM_ATTACKS)]

        for block in residual:
            total_blocks += 1
            members = block if len(block) <= MEMBER_CAP else [
                block[i] for i in rng.choice(len(block), MEMBER_CAP, replace=False)]
            member_models = [models[c] for c in members]

            # --- pure random: pay len-24 per attack until split.
            steps = 0
            split = False
            for ap in attacks:
                steps += ATTACK_LEN
                if len({run(models[c], ap) for c in members}) > 1:
                    split = True
                    break
            if split:
                agg["random"]["splits"] += 1
            agg["random"]["steps"] += steps

            # --- pure beam: one attempt, cap 8.
            bp = beam_probe(member_models, rng)
            if bp is not None:
                agg["beam"]["splits"] += 1
                agg["beam"]["steps"] += len(bp)
            else:
                agg["beam"]["steps"] += BEAM_CAP

            # --- hybrid: beam first; random only on beam's misses.
            steps = 0
            split = False
            bp2 = beam_probe(member_models, rng)
            if bp2 is not None:
                steps += len(bp2)
                split = True
            else:
                steps += BEAM_CAP
                for ap in attacks:
                    steps += ATTACK_LEN
                    if len({run(models[c], ap) for c in members}) > 1:
                        split = True
                        break
            if split:
                agg["hybrid"]["splits"] += 1
            agg["hybrid"]["steps"] += steps

        print(f"  {spec.world_id}: attacked {len(residual)} blocks")

    print(f"\nblocks: {total_blocks}")
    print(f'{"strategy":10} {"coverage":>9} {"total steps":>12} {"steps/split":>12}')
    for name, s in agg.items():
        cov = 100.0 * s["splits"] / max(total_blocks, 1)
        per = s["steps"] / max(s["splits"], 1)
        print(f"{name:10} {cov:8.0f}% {s['steps']:12d} {per:12.1f}")

    r = agg["random"]
    h = agg["hybrid"]
    if h["splits"] >= r["splits"] and h["steps"] < 0.8 * r["steps"]:
        print("\nVERDICT: hybrid wins on BOTH axes. The two-stage proposal")
        print("pipeline is settled: cheap structural splits first, sampling")
        print("for the long-setup remainder. Emit its (block, probe) pairs as")
        print("training data for the learned Level 4 generator.")
    elif h["steps"] < r["steps"]:
        print("\nVERDICT: hybrid cheaper at equal coverage, but the margin is")
        print("small: the expensive random tail dominates either way. The free")
        print("head is solved; the long-setup tail needs a learned prior or")
        print("the in-world planner -- no searched proposer will find it.")
    else:
        print("\nVERDICT: no cost win. Beam's short splits are too rare to")
        print("offset its misses; go straight to a learned proposer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
