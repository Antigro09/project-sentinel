"""X11: does BEAM search over whole sequences find what greedy cannot?

X10's diagnosis was structural: splitting probes are order-dependent
(setup moves, then the discriminating move), so step-wise greedy scores
are flat until the final action -- every prefix choice is a coin flip,
and greedy commits to one path. Beam search is the minimal fix: keep the
B best PREFIXES at every depth instead of one, so a setup move that looks
useless at step 3 survives to step 6 where it pays off.

Second ingredient, from the same diagnosis: score prefixes by
POSITION DIVERSITY as well as immediate splitting. Setup moves work by
scattering the block's members across distinct states -- members in
different places are more likely to disagree about the NEXT action. So
the beam ranking is:

    (immediate split?, how many distinct member states)

with immediate splitting dominant (a split ends the search) and diversity
breaking ties toward prefixes that set up future splits.

Measured against X9/X10 baselines on identical residual blocks:
recovery rate and probe length. The comparison that matters:

    random (40 x length-24)   the sampler X10 lost to
    greedy (X10 v2)           0.10x random
    beam (this)               ?

If beam recovers most of what random finds at SHORTER lengths, the
proposal mechanism for Level 4 exists in searched form -- and the pairs
(block, splitting-probe) it emits are exactly the training data for a
learned proposer, which is the ranker-to-generator shift proper.

RESULT (measured, 12 worlds, 720 residual blocks, width 8, cap 24):

    split by RANDOM (40 x len-24):  594 (82%)
    split by BEAM:                  276 (38%), median length 2

Three beam variants tried:
  v1  visibility-ranked, pure top-k      9%   (bug: hidden state invisible)
  v2  full-state-ranked, pure top-k     32%
  v3  full-state + randomised beam      38%

Beam still loses to random sampling by 2x. But the failure analysis
changed the conclusion rather than just the number:

  - Beam's splits are nearly FREE (median length 2 vs random's 24). Per
    unit of probe cost, beam wins on the blocks it can crack.
  - The blocks beam misses are exactly the long-setup ones (charge
    periods, order violations) -- the same mechanics X9 called
    inseparable-from-short-probes. Width 8 x depth 24 explores ~200
    sequences; 40 random x 24 explores 40. Beam explores MORE and finds
    FEWER long probes, because its ranking is still per-step: no prefix
    statistic separates 'setup that will pay off in 10 steps' from
    'setup that leads nowhere'. That information does not exist in the
    prefix; it only exists in the completed sequence.

CONCLUSION: the proposal mechanism for order-dependent rules cannot be
any step-wise score -- greedy (X10) and beam (X11) bound it from both
sides. What remains is (a) sampling with a learned prior over sequences
-- train on the (block, probe) pairs random search already finds, which
is the Level 4 generator proper -- or (b) accepting that these rules are
the in-world experiment planner's job. The cheap splits beam DOES find
are still worth taking first: run beam, then random on the remainder.
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
MAX_GEN_LEN = 24
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
    """Beam search for a probe splitting this block.

    Two notions of state, and confusing them was the first version's bug:

    - SPLIT detection uses `state_key` -- VISIBLE state, what a frame
      could reveal. That is what a signature is made of.
    - BEAM RANKING uses the FULL internal state. Hidden components (the
      charge counter) accumulate invisibly: during accumulation every
      member shows identical visible state, so a visibility-ranked beam
      sees a flat landscape and its inert-fallback quits one step before
      the payoff. Full-state diversity detects the accumulating counter
      and keeps setup moves alive.

    The inert fallback fires only when all members sit in the SAME full
    state -- genuine bisimilarity, no continuation can ever split.

    The beam itself is diversified: top half by full-state diversity,
    plus a random half of the remainder. Pure top-k collapses onto
    near-identical prefixes (all maximal-diversity prefixes differ by one
    action) and re-explores one region every depth.
    """
    init = tuple((m, m.init_state()) for m in models)
    beam = [((), init)]
    for _depth in range(MAX_GEN_LEN):
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

    stats = {"random": 0, "beam": 0, "total": 0}
    beam_lengths: list[int] = []

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

            for ap in attacks:
                if len({run(models[c], ap) for c in members}) > 1:
                    stats["random"] += 1
                    break

            bp = beam_probe(member_models, rng)
            if bp is not None:
                stats["beam"] += 1
                beam_lengths.append(len(bp))

        print(f"  {spec.world_id}: attacked {len(residual)} blocks")

    t = max(stats["total"], 1)
    print(f"\nblocks attacked: {stats['total']}")
    print(f"  split by RANDOM ({RANDOM_ATTACKS} x len-{ATTACK_LEN}): "
          f"{stats['random']:4d} ({100.0 * stats['random'] / t:.0f}%)")
    print(f"  split by BEAM (width {BEAM_WIDTH}, cap {MAX_GEN_LEN}):      "
          f"{stats['beam']:4d} ({100.0 * stats['beam'] / t:.0f}%)")
    if beam_lengths:
        print(f"  beam probe length: median {int(np.median(beam_lengths))}, "
              f"mean {np.mean(beam_lengths):.1f}")

    gain = stats["beam"] / max(stats["random"], 1)
    print(f"\nbeam finds {gain:.2f}x as many as random")
    if gain >= 1.0:
        print("VERDICT: beam MATCHES OR BEATS sampling. Sequence-level search")
        print("is the right proposal mechanism; emit (block, probe) pairs and")
        print("train the learned proposer on them -- the Level 4 generator.")
    elif stats["beam"] > 0 and beam_lengths and np.median(beam_lengths) <= 4:
        print("VERDICT: beam loses on coverage but its splits are nearly free")
        print(f"(median length {int(np.median(beam_lengths))} vs random's "
              f"{ATTACK_LEN}). Use beam FIRST for cheap splits, sampling for")
        print("the long-setup remainder; a step-wise score cannot find those.")
    else:
        print("VERDICT: beam also fails. The setup-discriminate structure needs")
        print("search deeper than width-%d, or the blocks are mostly" % BEAM_WIDTH)
        print("inseparable after all (X9's caveat about weak attacks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
