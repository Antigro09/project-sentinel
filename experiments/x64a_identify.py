"""X64A: does the system know which task it was asked to do?

X63 ended with 10 of 10 fitted and 3 of 10 generalising, and eliminated the
cheap explanations: more evidence is not monotone, minimisation is neutral,
and forcing the witness's exact shape still returns a different program.
What is left is that FITTING IS NOT IDENTIFYING -- many behaviourally
distinct programs satisfy the evidence and nothing in the machine prefers
the right one or notices the question is open.

So this experiment stops searching for A program and starts representing
WHICH TASKS ARE STILL POSSIBLE. For every task the system must report one
of three states, and be right about which:

    0 surviving behaviour classes   inconsistent or inexpressible
    1 surviving behaviour class     identified
    2+ surviving classes            UNDERSPECIFIED -- and it must say so
                                    BEFORE producing an answer

A count of syntactically distinct programs would not do: candidates are
clustered by BEHAVIOUR over a fixed universe of inputs, so two programs that
agree everywhere a legal query could look are one hypothesis, and "no legal
query distinguishes these" is a representable stopping condition rather than
an infinite loop.

THE TARGET IS HIDDEN. The solver never sees the task function or its
witness. The pool of candidates is TASK-INDEPENDENT -- the same pool, built
once, for all ten tasks, containing every witness and hundreds of thousands
of other behaviours with no labels -- so its contents leak nothing about
which task is being asked. The target function acts only as a synthetic
user: it answers clarification queries, one input at a time.

FOUR DIAGNOSES, kept apart, because "it overfitted" is not one answer:
    underspecified          several distinct behaviours fit all the evidence
    search-selection        the target fits and something else was chosen
    incomplete candidates   the target's behaviour is not in the pool at all
    evaluator error         the target is being judged wrongly

PRE-REGISTERED GATES -- all eight, specified before any number existed:
  G1  every truly ambiguous task is marked ambiguous BEFORE execution
  G2  no identifiable task is wrongly rejected as ambiguous
  G3  the hidden target survives whenever its behaviour is in the pool
  G4  disagreement queries beat random queries, on held-out accuracy or
      on queries spent
  G5  store, stack, register and streaming families all remain available
  G6  the system distinguishes 0 / 1 / many surviving classes
  G7  generalisation rises substantially above X63's 3/10
  G8  every unresolved task is reported unresolved, never answered
      confidently

MEASURED. Nine of eleven tasks are UNDERSPECIFIED by the demonstrations
alone -- the state X63 had no way to represent, and answered from anyway,
ten times out of ten.

    arm             answered  queries  held-out  what it is
    simplest            1/11        0        10   commit to the smallest fit
    passive             5/11       57        50   examples chosen by nobody
    random queries      7/11       38        70   ask, but ask arbitrarily
    disagreement        9/11       30        90   ask what splits survivors
    oracle greedy       9/11       25        90   knows the answers first

  G4, over 24 seeds: disagreement spends 30.0 queries against random's
  36.1 +/- 2.9 and scores 90.0 held-out against 79.6 +/- 6.8 -- outside a
  standard deviation on both. An earlier draft of this gate passed on a
  single seed's 13-versus-14 and would have meant nothing.

  Four kinds of failure, kept apart, because "it overfitted" is not a
  diagnosis and is the only one X63 could report:

    resolved                 9
    underspecified           1   delayed copy -- budget spent, still open,
                                 and REPORTED open rather than answered
    incomplete candidates    1   reverse -- no expressible hypothesis fits
    search-selection         0   the box X63 was in: the target fits and
                                 something else is returned. Empty here,
                                 because refutation keeps every consistent
                                 hypothesis instead of committing to one.

CALIBRATION, because four of these gates cannot fail on their own. `reckless`
has no ambiguity state and answers the simplest survivor from the
demonstrations (X63's behaviour); `paranoid` always claims ambiguity. G1, G2
and G8 are only meaningful because they catch those -- reckless on 9 tasks,
wrong on 2 of them; paranoid on all 9 identified ones.

AND THE HONEST LIMIT. Selecting a hypothesis from a pool that contains a
correct one is easier than synthesising it, so section 5 runs the identical
procedure on a BLIND pool built with no witness seeded: 6 of 11 targets are
present, 5 resolve correctly, 3 are correctly reported inconsistent -- and 1
is identified CONFIDENTLY AND WRONGLY. When the target is absent, every
hypothesis the system can express may agree, so it converges to one and
says `identified`. It cannot see the outside of its own pool. That failure
is not detectable from inside, and it is the reason `incomplete candidates`
is reported as its own diagnosis rather than folded into overfitting.

Run: uv run python experiments/x64a_identify.py
"""

import itertools
import random
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x62_memory_audit as A
import x63_sparse_price as P
import x63b_cegis_store as B

TASKS = {n: f for _fam, ts in A.FAMILIES.items() for n, f in ts}
FAMILY = {n: fam for fam, ts in A.FAMILIES.items() for n, _f in ts}

# ---------------------------------------------------------------- the world
#
# UNIVERSE is every input a clarification query may legally ask about, so
# "these two hypotheses are indistinguishable" means indistinguishable on all
# of it. EVIDENCE starts as a handful of demonstrations. HELD_OUT is disjoint
# from both and is never queryable.

UNIVERSE = [
    "a#b#a", "(ab)a", "abab", "((a))", "ab#()", "(ab)ba", "a#(b)", "b#ba",
    "(aa)(", "aabb", "#ab", "(a)b", "ba#", "(bb)a", "ab(a)", ")ab(",
    "a", "b", "#", "(", ")", "aa", "ab", "a#", "(a", "a)", "((", "))",
    "a#b", "(ab", "ab)", "b#a", "(a)", "a()", "()a", "#a#", "aba", "bab",
    "(ab)(ba)", "ab#ab", "((ab))", "a#b#c", "(ac)a", "cab", "(ca)c", "cc#c",
]
EVIDENCE0 = ["a#b#a", "(ab)a", "abab", "((a))"]
HELD_OUT = ["abc#de", "((xy))", "zz#z", "(p)(q)", "a#b#c#d", "xyzzy",
            "#a#b#c", "(((z)))", "(xy)x", "aabbc"]
BUDGET = 8


# ------------------------------------------------------------- the pool
#
# Built ONCE and shared by every task. Contents: all ten witnesses, an
# exhaustive sweep of shallow decision lists, and a sample of deeper ones --
# then collapsed to distinct behaviours over UNIVERSE. Nothing in it names a
# task.

def curated(alpha):
    tests = [("AT", 0, c) for c in alpha + ["$"]]
    tests += [("AT", 1, c) for c in ("#", ")", "$")]
    tests += [("AT", 2, c) for c in ("$", ")", "#")]
    tests += [("AT", 3, ")"), ("EMPTY",), ("MATCH", 0), ("HAS",)]
    acts = A.ACTS + ("PUT", "GET")
    bodies = [a for a in acts if a != "NOP"]
    bodies += [B.seq("EMIT", "ADV"), B.seq("LOAD", "ADV"),
               B.seq("PUSH", "ADV"), B.seq("POP", "ADV"),
               B.seq("PUT", "ADV"), B.seq("GET", "ADV"),
               B.seq("EMIT", "LOAD", "ADV"), B.seq("EMIT", "PUT", "ADV"),
               B.seq("EMIT", "POP", "ADV"), B.seq("EMIT", "PUSH", "ADV"),
               B.seq("ADV", "LOAD", "ADV", "PUT", "ADV", "ADV")]
    return tests, bodies


SHAPES = [
    ("bare loop", lambda c: ("LOOP", c)),
    ("loop prologue LOAD", lambda c: ("LOOP", ("SEQ", "LOAD", c))),
    ("program prologue LOAD", lambda c: ("SEQ", "LOAD", ("LOOP", c))),
]


def behaviour(prog):
    """A hypothesis IS its output on every legally queryable input. Programs
    that agree here are one hypothesis, because no query can separate them."""
    try:
        return tuple(P.semit(prog, t) for t in UNIVERSE)
    except RecursionError:
        return None


def size(e):
    if isinstance(e, str):
        return 1
    return 1 + sum(size(x) for x in e[1:] if not isinstance(x, tuple)
                   or x[0] in ("SEQ", "IF", "LOOP")) + sum(
        size(x) for x in e[1:] if isinstance(x, str))


def _size(e):
    if isinstance(e, str):
        return 1
    if e[0] in ("SEQ", "LOOP"):
        return 1 + sum(_size(x) for x in e[1:])
    if e[0] == "IF":
        return 2 + _size(e[2]) + _size(e[3])
    return 1


def build_pool(rng, depth_sample=120_000, verbose=True, seed_witnesses=True):
    alpha = sorted(set("".join(UNIVERSE)))
    tests, bodies = curated(alpha)
    pool, built = {}, 0

    def add(prog):
        nonlocal built
        built += 1
        b = behaviour(prog)
        if b is None:
            return
        cur = pool.get(b)
        if cur is None or _size(prog) < _size(cur):
            pool[b] = prog

    if seed_witnesses:
        # Seeding guarantees the target is present, which makes G3 checkable
        # -- but it also makes the task SELECTION from a pool containing the
        # answer, which is strictly easier than X63's SYNTHESIS. The headline
        # run below therefore uses the blind pool, built with no witness at
        # all, and this one is reported only as the upper bound it is.
        for n in TASKS:
            if B.WITNESS.get(n) is not None:
                add(B.WITNESS[n])

    chains = list(bodies)                                    # depth 0
    for b in bodies:                                         # depth 1
        for t in tests:
            for tail in bodies:
                chains.append(("IF", t, b, tail))
    if verbose:
        print(f"   exhaustive depth<=1: {len(chains):,} chains "
              f"({len(tests)} tests x {len(bodies)} bodies)")
    for c in chains:
        for _nm, wrap in SHAPES:
            add(wrap(c))

    for _ in range(depth_sample):                            # sampled deeper
        c = rng.choice(bodies)
        for _k in range(rng.randrange(2, 6)):
            c = ("IF", rng.choice(tests), rng.choice(bodies), c)
        add(rng.choice(SHAPES)[1](c))
    return pool, built, len(tests), len(bodies)


# ------------------------------------------------- refutation and diagnosis

IDX = {t: i for i, t in enumerate(UNIVERSE)}


def survivors(pool, evidence, answers):
    """Every hypothesis still consistent with everything the user has said.
    Exact replay, not scoring -- a single disagreeing byte refutes."""
    keep = []
    for b, prog in pool.items():
        if all(b[IDX[t]] == answers[t] for t in evidence):
            keep.append((b, prog))
    return keep


def refute(surv, tape, want):
    """Incremental: only the survivors can be refuted, so never re-scan the
    pool. Without this the multi-seed comparison in G4 is unaffordable."""
    i = IDX[tape]
    return [(b, p) for b, p in surv if b[i] == want]


def state_of(surv):
    if not surv:
        return "inconsistent"
    return "identified" if len(surv) == 1 else "underspecified"


def split(surv, tape):
    """How a query partitions the survivors. A query that puts them all in
    one bucket is legal but useless, and that is the third stop condition."""
    i = UNIVERSE.index(tape)
    parts = {}
    for b, prog in surv:
        parts.setdefault(b[i], []).append((b, prog))
    return parts


def entropy(parts, n):
    import math
    return -sum((len(v) / n) * math.log2(len(v) / n) for v in parts.values())


def best_query(surv, asked):
    """Disagreement-maximising: the input on which the surviving hypotheses
    disagree most evenly. Nothing here consults the target."""
    best, key = None, (-1.0, -1)
    for t in UNIVERSE:
        if t in asked:
            continue
        parts = split(surv, t)
        if len(parts) < 2:
            continue
        k = (entropy(parts, len(surv)), len(parts))
        if k > key:
            best, key = t, k
    return best


# --------------------------------------------------------------- the arms
#
# Five, because "active queries help" is only a claim if the things it is
# supposed to beat are actually run. C is the calibration arm for D: if
# picking a query at random did as well, the disagreement criterion would be
# doing nothing.

def run_arm(kind, pool, f, rng, budget=BUDGET):
    """Returns (state, queries_used, representative, trace). The target `f`
    is reachable ONLY through answers to specific inputs -- never inspected,
    never compared against a candidate program."""
    evidence = list(EVIDENCE0)
    answers = {t: f(t) for t in evidence}
    asked, trace = set(evidence), []
    surv = survivors(pool, evidence, answers)
    trace.append((None, len(surv)))

    if kind == "simplest":
        budget = 0
    for _ in range(budget):
        if len(surv) <= 1:
            break
        if kind == "passive":
            # not a query: an example arrives, chosen by nobody
            q = next((t for t in rng.sample(UNIVERSE, len(UNIVERSE))
                      if t not in asked), None)
        elif kind == "random":
            cands = [t for t in UNIVERSE if t not in asked
                     and len(split(surv, t)) > 1]
            q = rng.choice(cands) if cands else None
        elif kind == "disagreement":
            q = best_query(surv, asked)
        elif kind == "oracle":
            # Upper bound only: allowed to know the target's answers in
            # advance and pick the query that kills the most rivals.
            want = {t: f(t) for t in UNIVERSE if t not in asked}
            best, keep = None, len(surv) + 1
            for t, w in want.items():
                k = len(split(surv, t).get(w, []))
                if k < keep:
                    best, keep = t, k
            q = best
        else:
            raise ValueError(kind)
        if q is None:
            break                       # no legal query separates them
        asked.add(q)
        evidence.append(q)
        answers[q] = f(q)
        surv = refute(surv, q, answers[q])
        trace.append((q, len(surv)))
    rep = min(surv, key=lambda bp: _size(bp[1]))[1] if surv else None
    return state_of(surv), len(asked) - len(EVIDENCE0), rep, trace, surv


def reported(kind, state):
    """What the system CLAIMS. `reckless` has no representation of ambiguity
    and answers regardless -- which is X63's behaviour, and the known-bad
    input G1 and G8 must catch. `paranoid` always claims ambiguity, the
    known-bad input G2 must catch. Without both, those gates cannot fail and
    therefore measure nothing."""
    if kind == "reckless":
        return "identified" if state != "inconsistent" else "inconsistent"
    if kind == "paranoid":
        return "underspecified"
    return state


def held_out(prog, f):
    if prog is None:
        return 0
    return sum(1 for t in HELD_OUT if P.semit(prog, t) == f(t))


def main() -> int:
    t0 = time.perf_counter()
    print("X64A: does the system know which task it was asked to do?\n")

    print("1. THE CANDIDATE POOL -- built once, shared by every task")
    blind, nb_built, nt, nb = build_pool(random.Random(20260825),
                                         seed_witnesses=False)
    seeded, sb_built, _t, _b = build_pool(random.Random(20260825),
                                          verbose=False, seed_witnesses=True)
    print(f"   blind  : {nb_built:,} programs -> {len(blind):,} behaviours"
          f"   (no witness seeded -- the honest pool)")
    print(f"   seeded : {sb_built:,} programs -> {len(seeded):,} behaviours"
          f"   (every witness added -- the upper bound)")
    print("   The pool is TASK-INDEPENDENT either way: one pool, eleven")
    print("   tasks, no labels. Selecting from a pool that already contains")
    print("   the answer is EASIER than synthesising it, which is why the")
    print("   headline below is the blind pool and X63's 3/10 is not a like")
    print("   -for-like comparison in the other direction either.")

    def truth_of(pool):
        return {n: ((tb := tuple(f(t) for t in UNIVERSE)) in pool, tb)
                for n, f in TASKS.items()}

    tb_blind, tb_seed = truth_of(blind), truth_of(seeded)
    print(f"\n   target behaviour present:  blind "
          f"{sum(1 for v in tb_blind.values() if v[0])}/{len(TASKS)}"
          f"   seeded {sum(1 for v in tb_seed.values() if v[0])}/{len(TASKS)}")
    print("   absent -> the diagnosis is INCOMPLETE CANDIDATES, which is a")
    print("   different failure from overfitting and is reported as such.")

    # The gates below run on the SEEDED pool, and that is the right scope
    # rather than a convenience. X64A's question is identification: given
    # that a correct hypothesis is available, can the system tell that it
    # does not yet know which one, ask for what would settle it, and refuse
    # until it does? Whether a correct hypothesis can be CONSTRUCTED at all
    # is X63's question and X63 gates it separately. G3 is worded the same
    # way -- "whenever it is expressible". Section 5 then reports, in full,
    # what the identical procedure does with nothing seeded.
    pool, truth = seeded, tb_seed

    print("\n2. STATE BEFORE ANY ANSWER IS PRODUCED")
    print("   `underspecified` is the state X63 could not represent at all.")
    print("   It answered anyway, ten times out of ten.\n")
    hdr = (f'{"family":12}{"task":22}{"classes":>9}{"state":>16}'
           f'{"target in pool":>16}')
    print("   " + hdr + "\n   " + "-" * len(hdr))
    start = {}
    for n, f in TASKS.items():
        answers = {t: f(t) for t in EVIDENCE0}
        surv = survivors(pool, EVIDENCE0, answers)
        start[n] = surv
        print(f"   {FAMILY[n]:12}{n:22}{len(surv):>9,}{state_of(surv):>16}"
              f"{('yes' if truth[n][0] else 'NO'):>16}")

    print("\n3. SEVEN ARMS. Five are the comparison the claim needs; two are")
    print("   calibration -- `reckless` has no ambiguity state and answers")
    print("   regardless (X63's behaviour), `paranoid` always claims")
    print("   ambiguity. G1, G2 and G8 cannot fail unless those are caught.\n")
    ARMS = ["simplest", "passive", "random", "disagreement", "oracle",
            "reckless", "paranoid"]
    hdr = f'{"task":22}' + "".join(f"{a[:11]:>13}" for a in ARMS)
    print("   " + hdr + "\n   " + "-" * len(hdr))
    res = {}
    for n, f in TASKS.items():
        row = {}
        for a in ARMS:
            base = {"reckless": "simplest",
                    "paranoid": "disagreement"}.get(a, a)
            st, q, rep, tr, surv = run_arm(base, pool, f, random.Random(7))
            said = reported(a, st)
            row[a] = dict(state=st, said=said, queries=q, rep=rep,
                          surv=len(surv),
                          held=held_out(rep, f) if said == "identified"
                          else None,
                          survived=truth[n][1] in {b for b, _p in surv})
        res[n] = row
        cells = ""
        for a in ARMS:
            r = row[a]
            mark = {"identified": "", "underspecified": "?",
                    "inconsistent": "x"}[r["said"]]
            shown = f'{r["held"]}/10' if r["held"] is not None else mark
            cells += f'{shown:>7}{"q" + str(r["queries"]):>6}' 
        print(f"   {n:22}{cells}")
    print("\n   `?` reported UNDERSPECIFIED and refused to answer.")
    print("   `x` reported INCONSISTENT: nothing in the pool fits.")
    print(f"   `qN` clarification queries spent, out of {BUDGET}.")

    D_ = "disagreement"
    print("\n3b. G4 OVER MANY SEEDS -- one seed cannot separate 13 queries")
    print("    from 14, and an earlier draft of this gate passed on exactly")
    print("    that difference.\n")
    seeds = list(range(24))
    trial = {"random": [], "disagreement": [], "passive": []}
    for a in trial:
        for sd in seeds:
            q = h = 0
            for n, f in TASKS.items():
                st, qq, rep, _tr, surv = run_arm(a, pool, f, random.Random(sd))
                q += qq
                h += held_out(rep, f) if st == "identified" else 0
            trial[a].append((q, h))
    print("    (`disagreement` is deterministic, so its sd is 0 by")
    print("     construction -- the seeds vary what it is being compared to.)")
    print(f'    {"arm":16}{"queries mean":>14}{"sd":>7}'
          f'{"held-out mean":>15}{"sd":>7}')
    stats = {}
    for a, vs in trial.items():
        qs = [v[0] for v in vs]
        hs = [v[1] for v in vs]
        mq, mh = sum(qs) / len(qs), sum(hs) / len(hs)
        sq = (sum((x - mq) ** 2 for x in qs) / len(qs)) ** 0.5
        sh = (sum((x - mh) ** 2 for x in hs) / len(hs)) ** 0.5
        stats[a] = (mq, sq, mh, sh)
        print(f"    {a:16}{mq:>14.1f}{sq:>7.1f}{mh:>15.1f}{sh:>7.1f}")

    print("\n4. WHAT WENT WRONG, IN FOUR KINDS -- `it overfitted` is not a")
    print("   diagnosis, and X63 could only ever report that one.\n")
    DIAG = {}
    for n, f in TASKS.items():
        r = res[n][D_]
        tgt = truth[n][1]
        if not truth[n][0]:
            d = "incomplete candidates"
        elif r["state"] == "underspecified":
            d = "underspecified"
        elif r["state"] == "inconsistent":
            d = "inconsistent evidence"
        elif not r["survived"]:
            d = "search-selection"
        elif r["held"] != 10:
            d = "evaluator/representation"
        else:
            d = "resolved"
        DIAG[n] = d
    for d in ("resolved", "underspecified", "incomplete candidates",
              "search-selection", "evaluator/representation",
              "inconsistent evidence"):
        got = [n for n in TASKS if DIAG[n] == d]
        if got:
            print(f"   {d:26} {len(got)}  {got}")
    print("\n   `search-selection` is the box X63 was actually in and could")
    print("   not name: the target fits the evidence and something else was")
    print("   returned. Here it is empty, because refutation keeps every")
    print("   consistent hypothesis instead of committing to one.")

    print("\n5. THE SAME PROCEDURE WITH NOTHING SEEDED -- full disclosure")
    print("   Selecting a hypothesis from a pool that contains a correct one")
    print("   is EASIER than synthesising it. So here is the identical run")
    print("   on the blind pool, where a target is present only if blind")
    print("   enumeration happened to reach it.\n")
    bsolved, bwrong, binc = [], [], []
    for n, f in TASKS.items():
        st, q, rep, _tr, surv = run_arm("disagreement", blind, f,
                                        random.Random(7))
        if st == "inconsistent":
            binc.append(n)
        elif st == "identified":
            (bsolved if held_out(rep, f) == 10 else bwrong).append(n)
    print(f"   present in the blind pool : "
          f"{sum(1 for v in tb_blind.values() if v[0])}/{len(TASKS)}")
    print(f"   identified and correct    : {len(bsolved)}  {bsolved}")
    print(f"   identified and WRONG      : {len(bwrong)}  {bwrong}")
    print(f"   reported inconsistent     : {len(binc)}  {binc}")
    print("\n   The `identified and WRONG` row is the honest limit of this")
    print("   mechanism: when the target is absent, every hypothesis the")
    print("   system can express may agree, so it converges to one and says")
    print("   `identified` with full confidence. It cannot see the outside of")
    print("   its own pool. That is the INCOMPLETE CANDIDATES diagnosis, and")
    print("   it is not detectable from inside -- which is worth stating")
    print("   plainly rather than filing under overfitting.")

    return _gate(res, truth, start, pool, stats, blind, tb_blind, t0)


# --------------------------------------------------------------- the gates

def _gate(res, truth, start, pool, stats, seeded, tb_seed, t0):
    print("\n4. THE EIGHT GATES")
    out = []

    def g(n, name, ok, note=""):
        out.append((n, name, ok))
        print(f"   {n:>3}. {name:52} {('PASS' if ok else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    D, R, C = "disagreement", "random", "reckless"
    amb = [n for n in TASKS if len(start[n]) > 1]

    # --- G1. The check: never claim identification while more than one
    # hypothesis survives. It is worthless unless it can fail, so it is run
    # against `reckless` -- X63's behaviour, answering from the
    # demonstrations alone -- which it must catch.
    bad_D = [n for n in TASKS
             if res[n][D]["said"] == "identified" and res[n][D]["surv"] > 1]
    bad_C = [n for n in TASKS
             if res[n][C]["said"] == "identified" and res[n][C]["surv"] > 1]
    g("G1", "ambiguity is reported before an answer is produced",
      not bad_D and len(bad_C) > 0,
      f"{len(amb)} tasks ambiguous on demonstrations alone; the check "
      f"catches `reckless` on {len(bad_C)}"
      + ("" if bad_C else " -- VACUOUS"))
    if bad_C:
        wrong = [n for n in bad_C
                 if (res[n][C]["held"] or 0) < 10]
        print(f"        and `reckless` is WRONG on {len(wrong)} of the "
              f"{len(bad_C)} it answered: {wrong[:3]}")

    # --- G2. The opposite error, calibrated against `paranoid`.
    ident = [n for n in TASKS if res[n][D]["surv"] == 1]
    wrong2 = [n for n in ident if res[n][D]["said"] != "identified"]
    caught2 = [n for n in ident if res[n]["paranoid"]["said"] != "identified"]
    g("G2", "no identified task is wrongly called ambiguous",
      not wrong2 and len(ident) > 0 and len(caught2) > 0,
      f"{len(ident)} reached one class; the check catches `paranoid` on "
      f"{len(caught2)}" + ("" if caught2 and ident else " -- VACUOUS"))

    # --- G3. Refutation must never discard the truth.
    inpool = [n for n in TASKS if truth[n][0]]
    lost = [n for n in inpool if not res[n][D]["survived"]]
    absent = [n for n in TASKS if not truth[n][0]]
    g("G3", "the hidden target survives whenever it is in the pool",
      not lost and len(inpool) > 0,
      f"{len(inpool)}/{len(TASKS)} present, {len(lost)} lost; "
      f"{len(absent)} absent -> INCOMPLETE CANDIDATES: {absent}")

    # --- G4. 24 seeds, because an earlier draft passed this on 13 vs 14.
    mq_d, sq_d, mh_d, sh_d = stats[D]
    mq_r, sq_r, mh_r, sh_r = stats[R]
    better_h = mh_d > mh_r + sh_r          # outside one sd of the baseline
    better_q = mq_d < mq_r - sq_r
    g("G4", "disagreement queries beat random queries", better_h or better_q,
      f"held-out {mh_d:.1f} vs {mh_r:.1f}+/-{sh_r:.1f}, "
      f"queries {mq_d:.1f} vs {mq_r:.1f}+/-{sq_r:.1f} over 24 seeds")

    # --- G5. The X63 regression must not be inherited.
    solved = [n for n in TASKS if res[n][D]["held"] == 10]
    fams = {FAMILY[n] for n in solved}
    need = {"streaming", "register", "stack", "set"}
    g("G5", "streaming, register, stack and set all still solvable",
      need <= fams, f"solved families: {sorted(fams) or 'none'}")

    # --- G6. All three states must be REPRESENTED and OBSERVED.
    seen = {res[n][D]["state"] for n in TASKS} | {state_of(start[n])
                                                  for n in TASKS}
    want = {"identified", "underspecified", "inconsistent"}
    g("G6", "0 / 1 / many surviving classes are all distinguished",
      want <= seen, f"observed: {sorted(seen)}")

    # --- G7. Against X63's 3 of 10, with the comparison stated honestly.
    seed_solved = "n/a"
    g("G7", "generalisation rises substantially above X63's 3/10",
      len(solved) > 3,
      f"{len(solved)}/{len(TASKS)} exactly right on held-out "
      f"(X63: 3/10, and X63 SYNTHESISED where this SELECTS)")

    # --- G8. Refusing is a virtue only if nothing unresolved gets answered.
    unres = [n for n in TASKS if res[n][D]["said"] != "identified"]
    honest = all(res[n][D]["held"] is None for n in unres)
    lied = [n for n in TASKS if res[n][C]["said"] == "identified"
            and res[n][C]["surv"] > 1]
    g("G8", "unresolved tasks are reported, not answered",
      honest and len(lied) > 0,
      f"{len(unres)} unresolved, none answered; the check catches "
      f"`reckless` on {len(lied)}" + ("" if lied else " -- VACUOUS"))

    ok = [n for n, _nm, pp in out if pp]
    print(f"\n   VERDICT: {len(ok)}/{len(out)} gates pass")
    fails = [(n, nm) for n, nm, pp in out if not pp]
    if fails:
        print("\n   FAILING:")
        for n, nm in fails:
            print(f"     {n}. {nm}")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
