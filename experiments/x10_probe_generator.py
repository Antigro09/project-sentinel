"""X10: can a TARGETED probe generator recover what random probes cannot?

X9 split the quotient's residual cleanly in half: 48% of blocks are
separable by SOME probe, 52% survive everything tried. But "some probe"
was found by brute random search -- the weakest possible generator. X8
showed why greedy selection fails globally (one-step disagreement
saturates across ALL behaviours), which suggests the fix: score
disagreement among the BLOCK'S OWN MEMBERS only.

The generator under test -- searched, not yet learned:

  for each residual block:
    for R restarts:
      greedily extend a probe one action at a time, keeping the action
      whose predicted outcomes differ most ACROSS THE BLOCK'S MEMBERS,
      ties to the least-used action (action diversity matters; see X8)
      stop early when the block splits

Measured against X9's random-attack baseline on identical blocks:

  - recovery: what fraction of the separable half does it find?
  - cost:     probe length needed per split (shorter = cheaper to execute)

If targeted generation recovers most of the 48% at short lengths, the
Level 4 design is confirmed in its first piece: the system should PROPOSE
experiments against its own uncertainty, not sample them. The remaining
52% is untouched by construction -- that half belongs to acting in the
world, not probing the model.

RESULT (measured, 12 worlds, 1800 residual blocks, RESTARTS=8, cap 24):

    split by RANDOM attacks (X9 baseline): 1249 (69%)
    split by GENERATOR:                     130 (7%)

The generator LOSES badly to random sampling. Two designs were tried and
both failed the same way:

  v1: score disagreement over ALL behaviours (X8's mistake) -- 0.21x random.
  v2: per-member state tracking + lookahead + diverse restarts -- 0.10x.

Diagnosis: greedy construction is structurally wrong for this problem,
not under-tuned. The probes that split blocks are order-dependent
(setup moves THEN the discriminating move), and at every prefix the
greedy score is flat -- all actions look identical until the last step,
so every tie-break is a coin flip and 8 restarts never find the needle.
Random search wins because it samples whole sequences; it fails only on
the blocks whose splitting probes are rare.

CONCLUSION FOR THE PROGRAMME: sequence-space search needs a different
mechanism than step-wise greedy -- either beam search over sequences, or
the learned generator (Level 4's actual proposal). The measured fact that
69% of blocks yield to ANY 24-step probe keeps the prize real; the open
question is only how proposals are produced.
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
RESTARTS = 8
MAX_GEN_LEN = 24
MEMBER_CAP = 60
BLOCK_CAP = 150
"""Largest blocks per world; small ones split trivially either way."""


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


def targeted_probe(members_models, rng) -> tuple[int, ...] | None:
    """Greedily build a probe that splits this block, or give up.

    Two corrections over the first cut, both diagnosed from its failure:

    1. Members must be tracked INDIVIDUALLY: each has its own state after
       the probe so far. Re-running everyone from init each step is
       equivalent for deterministic models but hides which member is where,
       and scoring 'distinct next states' from distinct current states is
       vacuous -- everything looks split already.
    2. When no single action immediately splits, do NOT stop: choose the
       action that maximises expected future splitting, estimated by rolling
       each member's continuation a few random steps. Order-dependent rules
       need setup moves before the discriminating one; a myopic check quits
       exactly when the interesting probes start.
    """
    from collections import Counter

    HORIZON = 4

    def futures(states):
        out = set()
        for m, s in zip(members_models, states):
            cur = s
            for _ in range(HORIZON):
                aid = int(rng.integers(1, 6))
                cur = _advance(m, cur, aid)
                out.add((id(m), state_key(cur)))
        return len(out)

    for attempt in range(RESTARTS):
        # Diverse restarts: some cold, some seeded with a random prefix --
        # greedy paths are sticky and different starts reach different splits.
        probe: list[int] = []
        if attempt > 0:
            n = int(rng.integers(1, MAX_GEN_LEN // 2))
            probe = [int(rng.integers(1, 6)) for _ in range(n)]

        states = [m.init_state() for m in members_models]
        for _ in range(MAX_GEN_LEN):
            sigs = {state_key(s) for s in states}
            if len(sigs) > 1:
                return tuple(probe)
            used = Counter(probe)
            best_aid, best_key = None, (-1.0, 0)
            for aid in (1, 2, 3, 4, 5):
                nxt = [_advance(m, s, aid) for m, s in zip(members_models, states)]
                immediate = len({state_key(s) for s in nxt})
                # Lookahead only matters when the immediate step does not
                # already split; then prefer actions that diversify futures.
                score = float(immediate)
                if immediate <= 1:
                    score = 1.0 + 0.1 * futures(nxt)
                key = (score, -used[aid])
                if key > best_key:
                    best_aid, best_key = aid, key
            if best_aid is None:
                break
            probe.append(best_aid)
            states = [_advance(m, s, best_aid) for m, s in zip(members_models, states)]
    return None


def _advance(model, state, aid):
    try:
        return model.transition(state, Action(aid))
    except Exception:
        return state


def _states_after(models, probe):
    states = [m.init_state() for m in models]
    for aid in probe:
        states = [_advance(m, s, aid) for m, s in zip(models, states)]
    return states


def main() -> int:
    specs = load_split("corpus/split_wide.json")["holdout_mechanics"][:WORLDS]
    rng = np.random.default_rng(0)

    stats = {"random": 0, "generator": 0, "either": 0, "total": 0}
    gen_lengths: list[int] = []

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
            stats["total"] += 1
            members = block if len(block) <= MEMBER_CAP else [
                block[i] for i in rng.choice(len(block), MEMBER_CAP, replace=False)]
            member_models = [models[c] for c in members]

            hit_random = False
            for ap in attacks:
                if len({run(models[c], ap) for c in members}) > 1:
                    hit_random = True
                    break
            if hit_random:
                stats["random"] += 1

            tp = targeted_probe(member_models, rng)
            hit_gen = tp is not None
            if hit_gen:
                stats["generator"] += 1
                gen_lengths.append(len(tp))

        print(f"  {spec.world_id}: attacked {len(residual)} blocks")

    t = max(stats["total"], 1)
    print(f"\nblocks attacked: {stats['total']}")
    print(f"  split by RANDOM attacks (X9 baseline): {stats['random']:5d} "
          f"({100.0 * stats['random'] / t:.0f}%)")
    print(f"  split by GENERATOR:                    {stats['generator']:5d} "
          f"({100.0 * stats['generator'] / t:.0f}%)")
    both = stats["random"] + stats["generator"]
    print(f"  union (found by either):               {both:5d} "
          f"({100.0 * both / t:.0f}%)")
    if gen_lengths:
        print(f"  generator probe length: median {int(np.median(gen_lengths))}, "
              f"mean {np.mean(gen_lengths):.1f} (cap {MAX_GEN_LEN})")

    gain = stats["generator"] / max(stats["random"], 1)
    print(f"\ngenerator finds {gain:.2f}x as many as random")
    if gain > 1.3:
        print("VERDICT: targeted generation WINS. Proposing experiments against")
        print("the current uncertainty beats sampling them -- the first working")
        print("piece of the Level 4 generator, and the mechanism X4 inherits.")
    elif gain > 1.05:
        print("VERDICT: marginal gain. The generator needs to be learned, not")
        print("searched, before the claim is worth carrying.")
    else:
        print("VERDICT: greedy construction LOSES to random sampling. The")
        print("splitting probes are order-dependent; step-wise scores are flat")
        print("until the final move. Sequence search needs beam search or a")
        print("learned proposer -- not deeper greedy tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
