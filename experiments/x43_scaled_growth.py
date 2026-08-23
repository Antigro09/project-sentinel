"""X43: the growth loop on 20+ worlds, with every trigger authenticated.

X35 passes on two worlds. Two worlds is a demonstration, not a result --
this project has been wrong about a headline number from a single run more
than once, and the variance on 2 samples is unknowable by construction.

This runs the whole loop across a suite: explore, authenticate the trigger
with X42's guard, propose and dispose candidate vocabularies, adopt by
Occam, certify, reset, plan, execute. Against a control with the growth
machinery removed, on the same worlds with the same seeds.

WHAT WOULD FALSIFY THE CLAIM, stated before the numbers exist:

  - the control winning about as often as the grown agent, which would mean
    growth is decoration and something else is doing the work
  - the guard authenticating everything, which would mean it is a rubber
    stamp rather than a check
  - one family being adopted everywhere regardless of the world, which
    would mean Occam is picking a default rather than responding to evidence

Failures are expected and are reported by cause rather than summarised
away. Ice worlds are hostile: sliding is how the agent learns about ice and
also how it corners itself, which is why the loop learns in one episode and
solves after a reset.

MEASURED (20 worlds, sizes 10-20, 2156s):

    grown agent won      20/20
    control won           0/20
    trigger fired        20/20
    guard authenticated  20/20
    certificate pinned   20/20
    actions on wins      mean 122, min 49, max 134

    adopted family: step-ext 16, slide 4

THE ADOPTION SPLIT IS THE RESULT, not the win rate. A single family
everywhere would have meant Occam was picking a default; instead the choice
tracks the board:

    sizes 10, 12, 14, 16   step-ext adopted -- a fixed step at least as
                           long as the board travels until blocked, which
                           IS ice at that width, so the families are
                           genuinely equivalent and the least expressive
                           one wins
    size 20                slide adopted alone -- slides of 17+ cells
                           exceed any fixed step the decoy offers, so
                           step-ext is ELIMINATED by evidence

So the system discriminates where discrimination is possible and certifies
a tie where it is not. That is the behaviour the design claims, measured
across a suite rather than demonstrated on two worlds.

WHAT THIS RUN DOES NOT SHOW, since 20/20 invites over-reading:

  - the guard is only tested for PERMISSIVENESS here. Every world is
    genuinely a slide world, so authenticating all 20 is expected; X42 is
    where its refusals are measured.
  - every world comes from one generator with one hidden mechanic. A 100%
    rate says the loop handles THIS novelty reliably, not that it handles
    novelty.
  - the control losing 20/20 isolates growth as the lever, but the control
    is deliberately crippled -- it plans inside a grammar known to be
    inadequate. It is a floor, not a rival.
"""

from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x35_novelty_trigger import EXPLORE_STEPS, make_ice_world, run_agent
from x42_trigger_guard import authenticate
from sentinel.env.types import Action
from sentinel.gen.grid import GridWorld

SUITE = [(seed, size) for size in (10, 12, 14, 16, 20) for seed in (0, 7, 23, 100)]


def episode_for_guard(spec, size: int):
    """Reproduce run_agent's exploration so the guard sees the same evidence.

    run_agent seeds its walk at 1234; matching that here means the verdict
    describes the episode the agent actually reasoned from, not a fresh one.
    """
    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(1234)
    actions = []
    for _ in range(EXPLORE_STEPS):
        if world.done:
            break
        aid = int(rng.integers(1, 6))
        world.step(Action(aid))
        actions.append(aid)
    return world, actions


def main() -> int:
    print("X43: growth loop across a suite, every trigger authenticated\n")
    t0 = time.perf_counter()

    rows = []
    print(f'{"world":>10} {"guard":>11} {"depth":>6} {"adopted":>9} {"cert":>7} '
          f'{"grown":>6} {"control":>8} {"actions":>8}')
    for seed, size in SUITE:
        made = make_ice_world(seed, size)
        if made is None or made[0] is None:
            continue
        spec, true_plan = made

        world, actions = episode_for_guard(spec, size)
        verdict = authenticate(world, spec, size, actions)

        grown = run_agent(spec, true_plan, expand=True)
        control = run_agent(spec, true_plan, expand=False)

        rows.append({
            "world": f"{spec.world_id}/{size}", "verdict": verdict,
            "grown": grown, "control": control,
        })
        cert = grown.get("certified")
        print(f'{spec.world_id + "/" + str(size):>10} {verdict.kind:>11} '
              f'{verdict.prefix_depth:>6} {str(grown.get("adopted"))[:9]:>9} '
              f'{("pinned" if cert else "no" if cert is False else "-"):>7} '
              f'{("WON" if grown.get("won") else "lost"):>6} '
              f'{("WON" if control.get("won") else "lost"):>8} '
              f'{grown.get("actions", 0):>8}')

    if not rows:
        print("no worlds generated")
        return 1

    n = len(rows)
    grown_wins = sum(r["grown"].get("won", False) for r in rows)
    control_wins = sum(r["control"].get("won", False) for r in rows)
    authed = sum(r["verdict"].may_grow for r in rows)
    triggered = sum(r["grown"].get("base_empty", False) for r in rows)
    certified = sum(1 for r in rows if r["grown"].get("certified"))

    print(f"\n{n} worlds, {time.perf_counter()-t0:.0f}s\n")
    print(f"  grown agent won      {grown_wins}/{n}  ({grown_wins/n:.0%})")
    print(f"  control won          {control_wins}/{n}  ({control_wins/n:.0%})")
    print(f"  trigger fired        {triggered}/{n}")
    print(f"  guard authenticated  {authed}/{n}")
    print(f"  certificate pinned   {certified}/{n}")

    adopted = Counter(str(r["grown"].get("adopted")) for r in rows)
    print(f"\n  adopted family: {dict(adopted)}")
    if len(adopted) == 1:
        print("    ONE family everywhere -- Occam may be picking a default")
        print("    rather than responding to evidence. Worth suspicion.")

    lost = [r for r in rows if not r["grown"].get("won")]
    if lost:
        print(f"\n  losses by cause ({len(lost)}):")
        causes = Counter()
        for r in lost:
            g = r["grown"]
            if g.get("no_plan"):
                causes["no survivor admits a plan"] += 1
            elif g.get("diverged"):
                causes["plan diverged from reality"] += 1
            elif str(g.get("adopted")) in ("DECLINED", "NONE"):
                causes["declined or nothing adopted"] += 1
            else:
                causes["executed but did not clear"] += 1
        for cause, k in causes.most_common():
            print(f"    {cause:32} {k}")

    actions = [r["grown"]["actions"] for r in rows if r["grown"].get("won")]
    if actions:
        print(f"\n  actions on wins: mean {np.mean(actions):.0f}, "
              f"min {min(actions)}, max {max(actions)}")

    print("\nREADING")
    lift = grown_wins - control_wins
    if lift <= 0:
        print("  growth machinery shows NO lift over the control: the win, if any,")
        print("  is not coming from vocabulary expansion.")
    else:
        print(f"  growth wins {lift} more worlds than the control. With n={n} the")
        print(f"  95% interval on a {grown_wins/n:.0%} rate is roughly "
              f"+/-{100*1.96*(grown_wins/n*(1-grown_wins/n)/n)**0.5:.0f} points, so")
        print("  read the direction, not the decimal.")
    if authed == n and triggered == n:
        print("  the guard authenticated every trigger. That is expected here --")
        print("  every world is genuinely a slide world -- but it means this run")
        print("  tests the guard's PERMISSIVENESS only; X42 tests its refusals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
