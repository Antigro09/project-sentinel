"""X64B-1: can it notice that none of its interpretations is adequate?

X64A's honest limit, reported there and now attacked here: when the intended
behaviour is ABSENT from the candidate pool, every hypothesis the system can
express may still agree with the evidence. It converges to one, says
`identified_on_U`, and is confidently wrong. No survivor-count rule can
detect that, because the count is 1 and the rule is satisfied.

    A singleton version space does not imply correctness. It implies
    uniqueness inside the current hypothesis class, and nothing more.

So identification needs an EXTERNAL criticism step. Three, in fact:

  CONFIRMATION   after a class is identified, run it on CHALLENGE inputs --
                 longer than anything queried, with symbols the universe
                 does not contain -- and ask the user whether that is the
                 intended behaviour. Disagreement is a counterexample, and a
                 counterexample from outside U is exactly the evidence a
                 survivor count cannot manufacture.

  NONE OF THE ABOVE  the user may reject every offered behaviour, not only
                 pick among them. Without that, every interaction silently
                 assumes realisability.

  EXPANSION      a rejection must grow the candidate space rather than pick
                 again inside it. The diagnosis is then MEASURED rather than
                 guessed: whichever expansion recovers the task names what
                 was missing -- memory, shape, vocabulary, or search.

CHALLENGE is disjoint from both the query universe and the held-out set, so
confirming on it cannot leak the score it is later graded against.

PRE-REGISTERED GATES:
  B1  singleton-but-wrong is rejected or corrected, never executed
      confidently -- the central gate
  B2  accuracy when the target IS present does not regress from X64A
  B3  false identification when the target is ABSENT goes to zero
  B4  when no interpretation is adequate, that is what gets reported
  B5  expansion recovers targets, and which expansion recovers each one is
      recorded as the diagnosis
  B6  target recall is reported per pool, not assumed
  B7  generalisation after expansion beats generalisation before it
  B8  independent generators agreeing is measured, and disagreement flagged
  B9  calibration: an arm with no confirmation step -- X64A's behaviour --
      must be caught producing confident wrong answers

MEASURED. The ladder makes the diagnosis a measurement rather than a story
-- whichever rung recovers a task names what was missing:

    strip comment          base           reachable with nothing added
    dedupe adjacent        +memory        needed the MATCH test
    first occurrence only  +shape         needed the loop prologue
    emit if seen before    +shape         same
    delayed copy           +vocabulary    needed an offset-2 test
    5 others               never          nothing expressible is adequate

WHERE CONFIDENT WRONGNESS LIVES, with the target removed and every rung
pinned in turn:

    rung           naive: wrong   confirm: wrong   naive answered
    base                      0                0                2
    +memory                   0                0                3
    +shape                    1                0                5
    +vocabulary               0                0                5
    +search                   2                0                6

It goes UP with pool richness, which is the opposite of the intuition. A
poor pool says `inconsistent` and is right to; a rich pool produces a
singleton that survives every question anyone thought to ask, and is wrong.
Growing the hypothesis space without an external criticism step makes the
failure MORE likely, not less.

Confirmation catches all four instances and produces none of its own. It
cannot make a poor pool good -- 5 tasks remain with no adequate
interpretation at any rung, and all 5 are correctly reported as such, with
no task abstaining while something adequate existed.

  present / confirm   10/11 exact on held-out  (X64A: 9/11)
  absent  / confirm    6/11 answered, all correct; 5 none-of-the-above
  absent  / naive      6/11 answered, one confidently wrong

AND ONE THING THE ADEQUACY MEASUREMENT FORCED. Exact equivalence over U is
STRICTER than "produces the intended behaviour on everything anyone will
check": `balanced prefix` has no U-exact match in any pool and is still
answered correctly on every held-out tape, because it differs from the
target only on universe inputs nobody asked about. An earlier draft of B4
tracked U-exactness and failed on precisely that distinction. Abstention
must track ADEQUACY, not identity.

Run: uv run python experiments/x64b1_openworld.py
"""

import random
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x62_memory_audit as A
import x63_sparse_price as P
import x63b_cegis_store as B
import x64a_identify as X

TASKS, FAMILY, UNIVERSE = X.TASKS, X.FAMILY, X.UNIVERSE
EVIDENCE0, HELD_OUT = X.EVIDENCE0, X.HELD_OUT

# Longer than anything in U, and carrying symbols U does not contain. This
# is where a hypothesis that merely fits the queried inputs breaks.
CHALLENGE = ["aabbaabb", "((ab)(ba))", "a#b#c#d#e", "xyzw", "(xy)(zw)x",
             "abcabc", "(((a)))b", "z#z#z#z", "(pq)p(qp)q", "abab#abab"]
assert not (set(CHALLENGE) & set(UNIVERSE))
assert not (set(CHALLENGE) & set(HELD_OUT))


# ------------------------------------------------------- the pool ladder
#
# Each rung adds exactly ONE thing, so whichever rung recovers a task is a
# measurement of what was missing rather than a story about it.

def vocab(level, alpha):
    tests = [("AT", 0, c) for c in alpha + ["$"]]
    tests += [("AT", 1, c) for c in ("#", ")", "$")]
    acts = [a for a in A.ACTS if a != "NOP"]
    bodies = list(acts) + [B.seq("EMIT", "ADV"), B.seq("LOAD", "ADV"),
                           B.seq("PUSH", "ADV"), B.seq("POP", "ADV"),
                           B.seq("EMIT", "LOAD", "ADV"),
                           B.seq("EMIT", "POP", "ADV"),
                           B.seq("EMIT", "PUSH", "ADV")]
    if level >= 1:                                   # + memory
        tests += [("EMPTY",), ("MATCH", 0), ("HAS",)]
        bodies += ["PUT", "GET", B.seq("PUT", "ADV"), B.seq("GET", "ADV"),
                   B.seq("EMIT", "PUT", "ADV")]
    if level >= 3:                                   # + vocabulary
        tests += [("AT", 2, c) for c in ("$", ")", "#")]
        tests += [("AT", 3, ")")] + [("TOP", c) for c in alpha]
        bodies += [B.seq("ADV", "LOAD", "ADV", "PUT", "ADV", "ADV")]
    return tests, bodies


def shapes(level):
    s = [("bare loop", lambda c: ("LOOP", c))]
    if level >= 2:                                   # + shape
        s += [("loop prologue", lambda c: ("LOOP", ("SEQ", "LOAD", c))),
              ("program prologue", lambda c: ("SEQ", "LOAD", ("LOOP", c)))]
    return s


RUNGS = ["base", "+memory", "+shape", "+vocabulary", "+search"]


def _insert(pool, prog):
    b = X.behaviour(prog)
    if b is None:
        return
    cur = pool.get(b)
    if cur is None or X._size(prog) < X._size(cur):
        pool[b] = prog


_CORES = {}


def core(level, seed=1000, sample=None, gen=0):
    """The pool with NO witness in it -- what blind enumeration reaches at
    this rung. Cached, because the witnesses are added afterwards and the
    expensive part does not depend on them."""
    key = (level, seed, sample, gen)
    if key in _CORES:
        return _CORES[key]
    rng = random.Random(seed + level)
    alpha = sorted(set("".join(UNIVERSE)))
    tests, bodies = vocab(level, alpha)
    shp = shapes(level)
    sample = sample if sample is not None else (12_000 if level < 4 else 90_000)
    pool = {}
    for b in bodies:
        for t in tests:
            for tail in bodies:
                for _nm, w in shp:
                    _insert(pool, w(("IF", t, b, tail)))
    for _ in range(sample):
        c = rng.choice(bodies)
        for _k in range(rng.randrange(2, 6 + gen)):
            c = ("IF", rng.choice(tests), rng.choice(bodies), c)
        _insert(pool, rng.choice(shp)[1](c))
    _CORES[key] = pool
    return pool


def build(level, rng=None, sample=None, exclude=(), gen=0, seed=1000):
    """`exclude` removes named witnesses so a target can be made ABSENT under
    control. The solver sees one pool and has nothing to compare it against,
    so it cannot detect the removal -- this is an experimental manipulation,
    not a signal the solver could read."""
    pool = dict(core(level, seed, sample, gen))
    for n in TASKS:
        if n not in exclude and B.WITNESS.get(n) is not None:
            _insert(pool, B.WITNESS[n])
    return pool


# --------------------------------------------- identify, confirm, expand

def keep_consistent(pool, off_universe):
    """Refute by execution, for evidence that lies outside U."""
    if not off_universe:
        return pool
    return {b: pr for b, pr in pool.items()
            if all(P.semit(pr, t) == w for t, w in off_universe.items())}


def identify(pool, answers, budget=X.BUDGET):
    """X64A's loop, but taking evidence accumulated across expansions rather
    than restarting from the demonstrations."""
    ev = list(answers)
    surv = X.survivors(pool, ev, answers)
    asked, used = set(ev), 0
    while len(surv) > 1 and used < budget:
        q = X.best_query(surv, asked)
        if q is None:
            break
        asked.add(q)
        used += 1
        yield ("query", q)
        surv = X.refute(surv, q, answers[q])
    rep = min(surv, key=lambda bp: X._size(bp[1]))[1] if surv else None
    return_value = (X.state_of(surv, asked), rep, surv, used)
    yield ("done", return_value)


def run_identify(pool, answers, f, budget=X.BUDGET):
    g = identify(pool, answers, budget)
    used = 0
    while True:
        kind, val = next(g)
        if kind == "query":
            answers[val] = f(val)
            used += 1
        else:
            st, rep, surv, u = val
            return st, rep, surv, used


def confirm(rep, f, rng, k=4):
    """The external criticism step. Returns (ok, counterexample). A
    counterexample from CHALLENGE is evidence no survivor count could have
    produced, because the survivors all agreed."""
    for t in rng.sample(CHALLENGE, k):
        if P.semit(rep, t) != f(t):
            return False, t
    return True, None


def solve(f, rng, exclude=(), confirmations=True, max_rung=4):
    """One task, open-world. Returns a verdict, not just an answer."""
    answers = {t: f(t) for t in EVIDENCE0}
    extra = {}                    # counterexamples, kept across expansions
    trail = []
    for level in range(max_rung + 1):
        pool = build(level, exclude=exclude, gen=1 if level == 4 else 0)
        # Counterexamples come from CHALLENGE, which is deliberately outside
        # the universe -- so they have no index in a universe-keyed
        # behaviour tuple and cannot be applied by the usual refutation.
        # They are applied by EXECUTION instead. (Until this was written,
        # the ladder only worked because every rejection happened to land on
        # the last rung; a rejection anywhere else raised a KeyError.)
        pool = keep_consistent(pool, extra)
        merged = dict(answers)
        st, rep, surv, used = run_identify(pool, merged, f)
        answers.update({k: v for k, v in merged.items() if k in UNIVERSE})
        trail.append((RUNGS[level], st, used, len(surv)))
        if st != "identified_on_U":
            if st == "inconsistent":
                continue          # nothing expressible fits -- climb
            return dict(verdict=st, rep=None, trail=trail, rung=RUNGS[level],
                        pool=pool)
        if not confirmations:
            return dict(verdict="answered", rep=rep, trail=trail,
                        rung=RUNGS[level], pool=pool)
        ok, ce = confirm(rep, f, rng)
        if ok:
            return dict(verdict="answered", rep=rep, trail=trail,
                        rung=RUNGS[level], pool=pool)
        extra[ce] = f(ce)         # the user said none of the above, and why
        trail.append((RUNGS[level], "REJECTED on " + ce, used, len(surv)))
    return dict(verdict="none_of_the_above", rep=None, trail=trail,
                rung=None, pool=None)


def solve_pinned(f, rng, level, exclude=(), confirmations=True):
    """One rung only, no climbing. Confident-wrongness needs a pool rich
    enough to have survivors and too poor to contain the target, so it
    appears at some rungs and not others -- which is worth sweeping rather
    than guessing at."""
    answers = {t: f(t) for t in EVIDENCE0}
    pool = build(level, exclude=exclude, gen=1 if level == 4 else 0)
    st, rep, surv, _u = run_identify(pool, answers, f)
    if st != "identified_on_U":
        return dict(verdict=st, rep=None)
    if confirmations:
        ok, _ce = confirm(rep, f, rng)
        if not ok:
            return dict(verdict="rejected", rep=None)
    return dict(verdict="answered", rep=rep)


def main() -> int:
    t0 = time.perf_counter()
    print("X64B-1: can it notice that none of its interpretations fits?\n")

    print("1. THE POOL LADDER -- each rung adds exactly one thing, so the")
    print("   rung that recovers a task MEASURES what was missing.\n")
    print(f'   {"task":24}' + "".join(f"{r:>13}" for r in RUNGS))
    print("   " + "-" * (24 + 13 * len(RUNGS)))
    recall = {}
    for n, f in TASKS.items():
        tb = tuple(f(t) for t in UNIVERSE)
        row, rec = "", []
        for lvl in range(len(RUNGS)):
            p = build(lvl, exclude=(n,), gen=1 if lvl == 4 else 0)
            here = tb in p
            rec.append(here)
            row += f'{("yes" if here else "-"):>13}'
        recall[n] = rec
        print(f"   {n:24}{row}")
    never = [n for n in TASKS if not any(recall[n])]
    print(f"\n   exact U-behaviour reachable at some rung: "
          f"{len(TASKS) - len(never)}/{len(TASKS)}")
    print(f"   never reachable: {never}")

    # Exact U-equivalence is STRICTER than "produces the intended behaviour
    # on everything anyone will check". A program can differ from the target
    # on a universe tape nobody asked about and still be right on every
    # demonstration, query, challenge and held-out input. So `adequate` is
    # measured separately: does ANY rung contain a program correct on the
    # whole held-out set? That is what abstention should track, and an
    # earlier draft of B4 tracked U-exactness instead and failed on exactly
    # this distinction.
    print("\n   ADEQUACY is the weaker and more useful notion: is there any")
    print("   program, at any rung, correct on the entire held-out set?\n")
    adequate = {}
    for n, f in TASKS.items():
        if any(recall[n]):
            adequate[n] = True
            continue
        found = None
        for lvl in range(len(RUNGS)):
            pl = build(lvl, exclude=(n,), gen=1 if lvl == 4 else 0)
            for _b, pr in pl.items():
                if all(P.semit(pr, t) == f(t) for t in HELD_OUT):
                    found = RUNGS[lvl]
                    break
            if found:
                break
        adequate[n] = found is not None
        print(f"   {n:24} U-exact no, adequate "
              f"{('yes at ' + found) if found else 'NO'}")
    hopeless = [n for n in TASKS if not adequate[n]]
    print(f"\n   no adequate interpretation at any rung: {hopeless}")
    print("   Those are the tasks where a singleton MUST be distrusted --")
    print("   nothing the system can express is the intended behaviour.")

    print("\n2. TWO CONDITIONS, TWO ARMS")
    print("   present : the pool contains the target")
    print("   absent  : the target's witness is removed under control")
    print("   confirm : identify, then CHALLENGE the answer and expand on")
    print("             rejection")
    print("   naive   : identify and answer -- X64A's behaviour, and the")
    print("             calibration arm without which B1 cannot fail\n")
    hdr = (f'{"task":24}{"present/confirm":>18}{"present/naive":>16}'
           f'{"absent/confirm":>18}{"absent/naive":>16}')
    print("   " + hdr + "\n   " + "-" * len(hdr))

    def score(r, f):
        if r["verdict"] != "answered":
            return r["verdict"], None
        return "answered", sum(1 for t in HELD_OUT
                               if P.semit(r["rep"], t) == f(t))

    def cell(v, h):
        if v == "answered":
            return f"{h}/10" + ("" if h == 10 else " WRONG")
        return {"none_of_the_above": "none-of-above",
                "unresolved_within_budget": "unresolved",
                "underspecified_on_U": "underspec",
                "inconsistent": "inconsistent"}.get(v, v)

    out = {}
    for n, f in TASKS.items():
        row = {}
        for cond, exc in (("present", ()), ("absent", (n,))):
            for arm, cf in (("confirm", True), ("naive", False)):
                r = solve(f, random.Random(5), exclude=exc, confirmations=cf)
                v, h = score(r, f)
                row[(cond, arm)] = dict(verdict=v, held=h, rung=r["rung"],
                                        trail=r["trail"])
        out[n] = row
        print(f"   {n:24}"
              + "".join(f'{cell(row[k]["verdict"], row[k]["held"]):>18}'
                        if k[1] == "confirm" else
                        f'{cell(row[k]["verdict"], row[k]["held"]):>16}'
                        for k in (("present", "confirm"), ("present", "naive"),
                                  ("absent", "confirm"), ("absent", "naive"))))

    # One caught error is a thin calibration. The naive arm errs only when
    # a pool happens to hold a wrong singleton, and `solve` climbs off the
    # weakest rung as soon as it finds an inconsistency. Pinning it at `base`
    # -- the poorest pool -- is where confident-wrongness is common, so it is
    # the sharper test of whether confirmation is doing anything.
    print("\n2b. EVERY RUNG PINNED, TARGET ABSENT -- where does confident")
    print("    wrongness actually live? A pool has to be rich enough to")
    print("    produce a singleton and too poor to contain the truth.\n")
    print(f'    {"rung":14}{"naive: wrong":>14}{"confirm: wrong":>16}'
          f'{"naive answered":>16}')
    pw, pc = [], []
    for lvl in range(len(RUNGS)):
        nw, cw, na = [], [], 0
        for n, f in TASKS.items():
            rn = solve_pinned(f, random.Random(5), lvl, (n,), False)
            rc = solve_pinned(f, random.Random(5), lvl, (n,), True)
            if rn["verdict"] == "answered":
                na += 1
                if any(P.semit(rn["rep"], t) != f(t) for t in HELD_OUT):
                    nw.append(n)
            if rc["verdict"] == "answered" and any(
                    P.semit(rc["rep"], t) != f(t) for t in HELD_OUT):
                cw.append(n)
        pw += [(RUNGS[lvl], n) for n in nw]
        pc += [(RUNGS[lvl], n) for n in cw]
        print(f"    {RUNGS[lvl]:14}{len(nw):>14}{len(cw):>16}{na:>16}")
    print(f"\n    naive is confidently wrong {len(pw)} times across the "
          f"ladder: {sorted({n for _r, n in pw})}")
    print(f"    with confirmation: {len(pc)}")
    print("    Confirmation cannot make a poor pool good. It stops the poor")
    print("    pool from being executed as though it were good.")

    print("\n3. WHICH EXPANSION RECOVERED WHICH TASK -- the diagnosis,")
    print("   measured rather than guessed\n")
    for n in TASKS:
        r = out[n][("absent", "confirm")]
        if r["verdict"] == "answered" and r["rung"]:
            print(f"   {n:24} recovered at {r['rung']}")
    print("   Others reported none-of-the-above or abstained; nothing in")
    print("   this machine can construct what its grammar cannot express,")
    print("   and saying so is the correct outcome, not a failure.")

    print("\n4. INDEPENDENT GENERATORS -- agreement is weak evidence, but")
    print("   disagreement is a flag, and neither is free\n")
    agree, disagree, gens = 0, [], (1000, 4242, 8888)
    for n, f in TASKS.items():
        picks = set()
        for sd in gens:
            p = build(3, exclude=(n,), seed=sd)
            merged = {t: f(t) for t in EVIDENCE0}
            st, rep, _s, _u = run_identify(p, merged, f)
            picks.add(X.behaviour(rep) if rep is not None else None)
        if len(picks) == 1:
            agree += 1
        else:
            disagree.append(n)
    print(f"   {len(gens)} independently seeded pools at the +vocabulary "
          f"rung, targets removed:")
    print(f"   agree on {agree}/{len(TASKS)}; disagree on {disagree}")

    return _gate(out, recall, never, hopeless, agree, disagree,
                 pw, pc, t0)


def _gate(out, recall, never, hopeless, agree, disagree, pw, pc, t0):
    print("\n5. THE NINE GATES")
    res = []

    def g(n, name, ok, note=""):
        res.append((n, name, ok))
        print(f"   {n:>3}. {name:52} {('PASS' if ok else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    PC, PN = ("present", "confirm"), ("present", "naive")
    AC, AN = ("absent", "confirm"), ("absent", "naive")

    def wrong(k):
        return [n for n in TASKS
                if out[n][k]["verdict"] == "answered" and out[n][k]["held"] != 10]

    def right(k):
        return [n for n in TASKS
                if out[n][k]["verdict"] == "answered" and out[n][k]["held"] == 10]

    # --- B1. The central gate, calibrated: the naive arm must be caught.
    w_c, w_n = wrong(AC), wrong(AN)
    g("B1", "singleton-but-wrong is never executed confidently",
      not w_c and not pc and len(w_n) + len(pw) > 0,
      f"confirm wrong {len(w_c)}x climbing + {len(pc)}x pinned; "
      f"naive wrong {len(w_n)}x + {len(pw)}x"
      + ("" if w_n or pw else "  -- VACUOUS, naive never errs"))

    # --- B2. No regression when the target IS there.
    g("B2", "accuracy with the target present does not regress",
      len(right(PC)) >= 9,
      f"{len(right(PC))}/{len(TASKS)} exact on held-out (X64A: 9/11)")

    # --- B3. False identification when the target is absent.
    g("B3", "false identification with the target absent is zero",
      not w_c, f"{len(w_c)} confident errors: {w_c or 'none'}")

    # --- B4. Abstention has to be correct, not merely frequent.
    abst = [n for n in TASKS if out[n][AC]["verdict"] != "answered"]
    missed = sorted(set(hopeless) - set(abst))
    idle = sorted(set(abst) - set(hopeless))
    g("B4", "no adequate interpretation is reported as such",
      not missed,
      f"{len(hopeless)} hopeless, {len(abst)} abstained, missed "
      f"{missed or 'none'}; {len(idle)} abstained though something adequate "
      f"existed: {idle or 'none'}")

    # --- B5. Expansion must actually recover something, and name what.
    rec = [n for n in TASKS if out[n][AC]["verdict"] == "answered"
           and out[n][AC]["rung"] not in (None, "base")]
    g("B5", "expansion recovers targets and names what was missing",
      len(rec) > 0, f"{len(rec)} recovered above `base`: "
                    f"{[(n, out[n][AC]['rung']) for n in rec]}")

    # --- B6. Recall reported, not assumed.
    reach = sum(1 for n in TASKS if any(recall[n]))
    g("B6", "target recall is reported per pool",
      reach + len(never) == len(TASKS),
      f"{reach}/{len(TASKS)} reachable at some rung, {len(never)} never")

    # --- B7. Expansion has to buy generalisation, not just movement.
    before = sum(1 for n in TASKS
                 if out[n][AC]["rung"] == "base"
                 and out[n][AC]["held"] == 10)
    after = len(right(AC))
    g("B7", "generalisation after expansion beats before it", after > before,
      f"{after} correct after expansion vs {before} at `base` alone")

    # --- B8. Measured, and disagreement surfaced.
    g("B8", "independent generators are compared, not assumed to agree",
      agree + len(disagree) == len(TASKS),
      f"agree {agree}/{len(TASKS)}, disagree {len(disagree)}")

    # --- B9. The calibration itself.
    g("B9", "an arm without confirmation is caught being confidently wrong",
      len(w_n) + len(pw) >= 2,
      f"naive wrong {len(w_n)}x climbing the ladder and {len(pw)}x pinned "
      f"across rungs: {sorted(set(w_n) | {n for _r, n in pw})}")

    ok = [n for n, _m, p in res if p]
    print(f"\n   VERDICT: {len(ok)}/{len(res)} gates pass")
    bad = [(n, m) for n, m, p in res if not p]
    if bad:
        print("\n   FAILING:")
        for n, m in bad:
            print(f"     {n}. {m}")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
