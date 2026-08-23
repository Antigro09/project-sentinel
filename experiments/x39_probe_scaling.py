"""X39: does self-generated testing scale, or was 2x an artefact of a small space?

X36 measured adversarial probe selection at 1.00 tests against 2.00 for
tests taken in arbitrary order. That is a real 2x, on 6,917 programs -- and
a 2x measured at one space size says nothing about what happens as the
space grows. The whole reason to care about Pillar 4 is the claim that a
system can test itself efficiently in a LARGE hypothesis space, so the
quantity worth measuring is the trend, not the ratio.

Method mirrors X1, including the part that saved it. X1's first attempt fit
an impossible exponent because it had no arm whose answer was known in
advance; the rewrite kept a random arm as a permanent calibration check.
Here the equivalent control is `sequential` -- probes in fixed order,
independent of what survives -- and a third arm, `random`, samples the pool
without replacement. If the two uninformed arms do not behave alike, the
harness is wrong rather than the finding interesting.

Cost is TESTS TO BEHAVIOURAL PINNING: how many probes before no remaining
input can split the surviving set. All arms stop by the same rule, which is
the specific bug that made X36's first control report a ten-billion-fold
win.

MEASURED: see main().
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x36_micro_vm import (
    Evidence,
    TASKS,
    VMError,
    adversarial_probe,
    behaviour,
    enumerate_programs,
    evaluate,
    refute,
    render,
    size,
)

SIZES = (5, 6, 7, 8, 9, 10)
TRAIN_INPUT = "A"
POOL = ("AB", "ABC", "ABCD", "XY", "Q", "MNOP", "ZZ", "PQRS", "K", "ABAB",
        "B", "CD", "WXYZ", "LMN", "OP", "RST")


def probes_to_pin(survivors, fn, rule: str, rng) -> int:
    """Tests until no remaining probe can split the survivors.

    Every arm stops the same way. Only the CHOICE of the next probe differs:
    `adversarial` picks the one the survivors most disagree about, the other
    two ignore the survivors entirely.
    """
    live = list(survivors)
    order = list(POOL)
    if rule == "random":
        order = [order[i] for i in rng.permutation(len(order))]
    used = 0
    while order:
        probe, split = adversarial_probe(live, tuple(order))
        if split <= 1:
            break
        if rule != "adversarial":
            probe = order[0]
        order = [q for q in order if q != probe]
        want = fn(probe)
        live = [p for p in live if behaviour(p, (probe,)) == (want,)]
        used += 1
    return used


def main() -> int:
    print("X39: how self-generated testing scales with the program space\n")
    rng = np.random.default_rng(0)

    print(f'{"max size":>9} {"programs":>10} {"adversarial":>12} {"sequential":>11} '
          f'{"random":>8} {"speedup":>8}')
    rows = []
    for max_size in SIZES:
        programs = enumerate_programs(max_size=max_size)
        adv, seq, rnd = [], [], []
        for name, fn in TASKS.items():
            evidence = Evidence(((TRAIN_INPUT, fn(TRAIN_INPUT)),))
            survivors = refute(programs, evidence)
            if len(survivors) < 2:
                continue
            adv.append(probes_to_pin(survivors, fn, "adversarial", rng))
            seq.append(probes_to_pin(survivors, fn, "sequential", rng))
            rnd.append(probes_to_pin(survivors, fn, "random", rng))
        if not adv:
            continue
        a, s, r = np.mean(adv), np.mean(seq), np.mean(rnd)
        rows.append((len(programs), a, s, r))
        print(f"{max_size:>9} {len(programs):>10,} {a:>12.2f} {s:>11.2f} "
              f"{r:>8.2f} {s/max(a, 1e-9):>7.2f}x")

    if len(rows) < 3:
        print("\ntoo few points to read a trend")
        return 0

    n = np.log([r[0] for r in rows])
    print(f"\ngrowth of cost with space size, cost ~ |H|^beta:")
    for idx, label in ((1, "adversarial"), (2, "sequential"), (3, "random")):
        y = np.log(np.maximum([r[idx] for r in rows], 1e-9))
        beta = float(np.polyfit(n, y, 1)[0])
        print(f"  {label:12} beta = {beta:+.2f}")

    # Calibration: the two uninformed arms should agree. If they do not, the
    # harness is measuring something other than what it claims.
    seqs = np.array([r[2] for r in rows])
    rnds = np.array([r[3] for r in rows])
    gap = float(np.mean(np.abs(seqs - rnds)))
    print(f"\nCALIBRATION: sequential vs random differ by {gap:.2f} probes on average")
    print("  the two uninformed arms should behave alike; a large gap would mean")
    print("  the pool order, not the selection rule, is doing the work")

    first, last = rows[0], rows[-1]
    print(f"\nspeedup at {first[0]:,} programs: {first[2]/max(first[1],1e-9):.2f}x")
    print(f"speedup at {last[0]:,} programs: {last[2]/max(last[1],1e-9):.2f}x")
    print(f"\ntotal {time.perf_counter():.0f}s process time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
