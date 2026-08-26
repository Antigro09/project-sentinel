"""X63b: the sparse store, and a search with no gradient at all.

X63a killed distance-ranked search in the executed setting: outputs resolve
8 levels where the behaviour table resolves 123, and the best of four
calibration metrics correlates r=0.190 with table agreement. A priority queue
needs an ordering over unrelated candidates, and outputs do not provide one.

So this drops the queue. The mechanism is counterexample-guided:

    run the candidate -> find the FIRST byte where its output diverges from
    the evidence -> read the machine state at that exact point -> prepend a
    rule whose test is TRUE in that state -> keep it only if the divergence
    moved LATER.

That last clause is the whole termination argument. `common prefix` scored
r=0.124 as a global gradient in X63a and is nonetheless the right progress
measure here, and those are not in conflict: a global gradient must order
candidates that have nothing to do with each other, while a repair measure
only has to certify that ONE edit advanced the frontier of correctness.
Total matched prefix is bounded by total output length, so strict increase
terminates.

THE MACHINE. X62's head + stack + register, plus:

    PUT    store[reg] = tape[pos]     GET    emit store[reg]
    HAS    reg is a key in the store

The store holds only the keys a run touches -- X63a measured that adding it
costs nothing per candidate, because a run touches what it touches whatever
the key space is. Emission is byte-valued, forced: `substitute` emits a value
that need not sit at the head, and index emission would make the store hold
POSITIONS, which grow with the tape and reintroduce exactly the unboundedness
the store was introduced to avoid.

PRE-REGISTERED BEFORE ANY NUMBER:

  (a) The three tasks X62 proved inexpressible -- first occurrence only, emit
      if seen before, substitute -- must become expressible, with witnesses
      that pass held-out. If they do not, the store is the wrong mechanism
      and the counting arguments in X62 need re-reading, not patching.
  (b) CEGIS must find at least as many tasks as X62's table search found
      (3 of 7 clean), at a lower evaluation count. Finding fewer means the
      gradient was load-bearing after all and X63a's conclusion was wrong.
  (c) Held-out generalisation is reported separately from finding, and
      inexpressibility separately from both. X62 found expressibility and
      findability come apart on more than half the suite.

MEASURED. The store closes all three tasks X62 proved inexpressible, and
CEGIS finds 10 of 10 at a fraction of the table search's cost:

                             expressible  found  generalises  worst evals
    X62, table + frontier              7      4            3      976,521
    X63b arm A, store + CEGIS         10      7            3       16,750
    X63b arm B, + plateau             10     10            3       50,829

The equivalence key was worth more than the search machinery. Keying on
OUTPUT ALONE merges programs with different store effects -- X63c clause 10
found 4 of 23 classes merged, up to 5 distinct stores in one -- so a plateau
move keeping `width` alternatives was keeping `width` copies of one
behaviour. Keying on (output, store) took arm B from 8 found to 10 and its
worst case from 12,830,685 evaluations to 50,829.

AND THEN THE REAL FINDING. 10 found, 3 generalising. Four explanations,
four experiments:

    task                      2    4    8  minimal  shape forced
    strip comment            10   10   10       10            10
    capture quoted            5    6    3        3             6
    dedupe adjacent           3    4   10        4             4
    emit matching first       0    3    3        3             3
    capture brackets          6    3    -        5             -
    balanced prefix          10   10   10       10            10
    first occurrence only     1    0   10        0             0
    emit if seen before       1    1    -        1             2
    delayed copy             10   10   10       10            10
    substitute                6    9    -        9             -
    TOTAL held-out           52   56   56       55            45   / 100

  H1 thin evidence      not monotone and nearly flat: 52 -> 56 -> 56 while
                        the evidence doubles twice; two tasks climb to 10
                        and two collapse.
  H2 no simplicity bias refuted -- 56 -> 55.
  H3 missing shape      refuted -- forcing LOOP(SEQ(LOAD, .)), the witness's
                        exact shape, still gives 0/10 on `first occurrence
                        only`, and lowers the total to 45.
  H4 weak search        refuted -- it finds 10/10.

FITTING IS NOT IDENTIFYING. A correct program exists in exactly the shape
the search was handed, and the search returns a different one that fits
every training tape. Nothing here prefers the general program, and nothing
here can notice the question is undetermined. That is X64, arrived at from
the other direction.

Run: uv run python experiments/x63b_cegis_store.py
"""

import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x62_memory_audit as A
import x63_sparse_price as P

NONE = A.NONE
SSt, srun, stest = P.SSt, P.srun, P.stest

# ------------------------------------------------------------------ tasks
FAMILIES = A.FAMILIES
TRAIN = ["a#b#a", "(aa)(", "abab", "(ab)a"]
HELD_OUT = ["abc#de", "((xy))", "zz#z", "(p)(q)", "a#b#c#d", "xyzzy",
            "#a#b#c", "(((z)))", "(ab)ba", "aabbc"]

E, AD, PU, PO, LO = "EMIT", "ADV", "PUSH", "POP", "LOAD"
PUT, GET = "PUT", "GET"
EA = ("SEQ", E, AD)
HASH, OPEN, CLOSE = ("AT", 0, "#"), ("AT", 0, "("), ("AT", 0, ")")
EMPTY, M0, HAS = ("EMPTY",), ("MATCH", 0), ("HAS",)


def seq(*xs):
    e = xs[-1]
    for x in reversed(xs[:-1]):
        e = ("SEQ", x, e)
    return e


# Witnesses. The first seven are X62's, re-expressed for byte-valued
# emission; the last four are the point of this experiment.
WITNESS = {
    "strip comment": ("LOOP", ("IF", HASH, "NOP", EA)),
    "capture quoted": ("LOOP", ("IF", HASH,
                                ("IF", EMPTY, seq(PU, AD), seq(PO, AD)),
                                ("IF", EMPTY, AD, EA))),
    "dedupe adjacent": ("LOOP", ("IF", M0, AD, seq(E, LO, AD))),
    "emit matching first": seq(LO, AD, ("LOOP", ("IF", M0, EA, AD))),
    "capture brackets": ("LOOP", ("IF", OPEN, seq(PU, AD),
                                  ("IF", CLOSE, seq(PO, AD),
                                   ("IF", EMPTY, AD, EA)))),
    "balanced prefix": ("LOOP", ("IF", CLOSE,
                                 ("IF", EMPTY, "NOP", seq(PO, E, AD)),
                                 ("IF", OPEN, seq(PU, E, AD), EA))),
    "delayed copy": ("LOOP", ("IF", ("AT", 2, "$"), "NOP", EA)),

    # --- the store earns its place here, or it does not earn it at all ---
    "first occurrence only": ("LOOP", seq(LO, ("IF", HAS, AD,
                                               seq(E, PUT, AD)))),
    "emit if seen before": ("LOOP", seq(LO, ("IF", HAS, seq(E, PUT, AD),
                                             seq(PUT, AD)))),
    "substitute": ("LOOP", ("IF", OPEN,
                            ("IF", ("AT", 3, ")"),
                             seq(AD, LO, AD, PUT, AD, AD),
                             seq(LO, ("IF", HAS, seq(GET, AD), EA))),
                            seq(LO, ("IF", HAS, seq(GET, AD), EA)))),
    # still claimed impossible: EMIT and GET both read forwards; nothing
    # emits FROM the stack, so the machine cannot run the tape backwards.
    "reverse": None,
}


# ------------------------------------------------- counterexample machinery
#
# A `case` is (tape, want, start) -- the start state carries the prologue's
# effect, including anything it already emitted, so the loop body is searched
# against evidence the prologue has genuinely consumed.


def trace(prog, tape, st0, fuel=8192):
    """Run, returning (output, the state at each emission, final state). The
    state list is what makes repair local: divergence i names the exact
    machine configuration that produced the wrong byte."""
    states = []
    st, _ = _trace_run(prog, tape, st0, fuel, states)
    return "".join(st.out), states, st


def _trace_run(expr, tape, st, fuel, states):
    if fuel <= 0 or not st.live:
        return st, fuel
    if isinstance(expr, str):
        before = st
        nxt, fuel = srun(expr, tape, st, fuel)
        if len(nxt.out) > len(st.out):
            states.extend([before] * (len(nxt.out) - len(st.out)))
        return nxt, fuel
    h = expr[0]
    if h == "SEQ":
        st, fuel = _trace_run(expr[1], tape, st, fuel, states)
        return _trace_run(expr[2], tape, st, fuel, states)
    if h == "IF":
        return _trace_run(expr[2] if stest(expr[1], tape, st) else expr[3],
                          tape, st, fuel - 1, states)
    if h == "LOOP":
        for _ in range(len(tape) + 2):
            mark = len(states)
            nxt, fuel = _trace_run(expr[1], tape, st, fuel, states)
            if nxt.key() == st.key():
                del states[mark:]      # a fixed-point pass emitted nothing
                return nxt, fuel
            if fuel <= 0:
                return nxt, fuel
            st = nxt
        return st, fuel
    return st, fuel - 1


def prefix(got, want):
    i = 0
    while i < min(len(got), len(want)) and got[i] == want[i]:
        i += 1
    return i


def progress(prog, cases):
    """Total matched prefix, and how many cases are exact. Bounded by total
    output length, which is why strict increase has to terminate."""
    tot = ex = 0
    for tape, want, st0 in cases:
        got, _, _ = trace(prog, tape, st0)
        tot += prefix(got, want)
        ex += got == want
    return tot, ex


def first_divergence(prog, cases):
    """Earliest failing case, and the machine state that produced the bad
    byte. When the program stopped emitting too soon there is no such state,
    so the repair has to fire at the configuration it halted in."""
    for tape, want, st0 in cases:
        got, states, final = trace(prog, tape, st0)
        if got == want:
            continue
        i = prefix(got, want)
        return tape, want, i, (states[i] if i < len(states) else final)
    return None


class Counter:
    __slots__ = ("n",)

    def __init__(self):
        self.n = 0


def behaviour(prog, cases):
    """The equivalence key. X63c clause 10: keying on OUTPUT ALONE merges
    programs with different store effects -- 4 of 23 output classes on the
    training tapes, up to 5 distinct stores collapsed into one class. A
    plateau move that keeps only `width` alternatives then throws away every
    store-distinct one and keeps `width` copies of the same behaviour.
    Including the store splits 23 classes into 30 and leaves 0 merged."""
    out, eff = [], []
    for tape, _want, st0 in cases:
        got, _s, final = trace(prog, tape, st0)
        out.append(got)
        eff.append(tuple(sorted(final.store)))
    return tuple(out), tuple(eff)


def _repairs(chain, wrap, cases, tests, bodies, base, ctr):
    """Every one-rule repair that fires where the program is WRONG, scored.

    Filtering by "this test is true in the divergent state" is what replaces
    the priority queue. It is not a ranking over unrelated candidates --
    X63a showed outputs cannot supply one -- it is the set of edits that can
    possibly address this counterexample."""
    d = first_divergence(wrap(chain), cases)
    ctr.n += 1
    if d is None:
        return None
    tape, _w, _i, st = d
    up, flat = [], []
    for t in tests:
        if not stest(t, tape, st):
            continue
        for b in bodies:
            cand = ("IF", t, b, chain)
            sc = progress(wrap(cand), cases)
            ctr.n += 1
            if sc > base:
                up.append((sc, cand))
            elif sc == base:
                flat.append((sc, cand))
    up.sort(key=lambda x: (-x[0][0], -x[0][1]))
    flat.sort(key=lambda x: (-x[0][0], -x[0][1]))
    # Ranking still reads output only -- there is no target store to compare
    # against. What changes is the EQUIVALENCE: distinct store effects are
    # distinct candidates, so the width cut cannot spend itself on copies.
    seen, uniq = set(), []
    for sc, cand in flat:
        k = behaviour(wrap(cand), cases)
        if k in seen:
            continue
        seen.add(k)
        uniq.append((sc, cand))
    return up, uniq


def cegis(cases, tests, bodies, ctr, wrap=None, max_steps=14):
    """ARM A -- strict. Every accepted rule must move the divergence later.
    Total matched prefix is bounded by total output length, so this cannot
    loop. It also cannot credit a rule that changes STATE without changing
    output, which is X63a's resolution loss arriving in a new costume."""
    wrap = wrap or (lambda c: ("LOOP", c))
    cur = AD
    base = progress(wrap(cur), cases)
    ctr.n += 1
    for _ in range(max_steps):
        r = _repairs(cur, wrap, cases, tests, bodies, base, ctr)
        if r is None:
            return wrap(cur)
        up, _flat = r
        if not up:
            return None
        base, cur = up[0]
    return None


def cegis_plateau(cases, tests, bodies, ctr, wrap=None, max_steps=7,
                  slack=2, width=3, budget=200_000):
    """ARM B -- strict, plus a bounded tolerance for rules that change state
    without yet changing output. `capture brackets` needs a PUSH before any
    EMIT can be right, and PUSH alone scores exactly zero.

    X57's measured lesson (three edits at once) as a depth-bounded DFS
    rather than a product: `slack` non-improving accepts, `width`
    alternatives each. Reported as a SEPARATE arm so the plateau move's
    contribution stays visible instead of folded into arm A's number."""
    wrap = wrap or (lambda c: ("LOOP", c))
    start = ctr.n

    def step(cur, base, depth, left):
        if ctr.n - start > budget:
            return None                # a search that cannot say "no"
        r = _repairs(cur, wrap, cases, tests, bodies, base, ctr)
        if r is None:
            return wrap(cur)
        if depth == 0:
            return None
        up, flat = r
        for sc, cand in up[:width]:
            got = step(cand, sc, depth - 1, left)
            if got is not None:
                return got
        if left > 0:
            for sc, cand in flat[:width]:
                got = step(cand, sc, depth - 1, left - 1)
                if got is not None:
                    return got
        return None

    ctr.n += 1
    return step(AD, progress(wrap(AD), cases), max_steps, slack)


def _rules(loop_body):
    """Split a decision list back into (tests+bodies, default)."""
    chain, e = [], loop_body
    while isinstance(e, tuple) and e[0] == "IF":
        chain.append((e[1], e[2]))
        e = e[3]
    return chain, e


def polish(prog, cases):
    """X50's minimality pass, rebuilt on execution instead of on the table.

    CEGIS prepends a rule per counterexample and never removes one, so a rule
    that was needed at step 3 can be dead by step 6. Deleting it cannot hurt
    the training fit -- that is checked -- and if generalisation improves,
    the overfitting was contorted programs rather than thin evidence."""
    pro, loop = (prog[1], prog[2]) if prog[0] == "SEQ" else (None, prog)
    chain, dflt = _rules(loop[1])

    def build(ch):
        e = dflt
        for t, b in reversed(ch):
            e = ("IF", t, b, e)
        w = ("LOOP", e)
        return w if pro is None else ("SEQ", pro, w)

    exact = lambda p: all(trace(p, t, s0)[0] == w for t, w, s0 in cases)
    changed = True
    while changed and chain:
        changed = False
        for i in range(len(chain)):
            trial = chain[:i] + chain[i + 1:]
            if exact(build(trial)):
                chain, changed = trial, True
                break
    return build(chain)


PROLOGUES = [None] + list(A.ACTS) + [PUT, GET]
# A LOOP prologue is a different shape from a program prologue: it runs at
# the top of EVERY pass, before the decision list, so a test can read state
# the same pass has just set. Without it `HAS` can only ever describe the
# PREVIOUS byte -- which is why the set tasks were found and generalised
# 0-1 of 10 while their witnesses passed all ten. Third time a task looked
# like a search problem and was a shape problem (X58 lookahead, X59 depth-2
# body, X60 prologue).
LOOP_PROLOGUES = [None, LO, PO, PUT, GET]
POLISH = False


def solve(pairs, tests, bodies, arm):
    """X60's lesson, kept: a task can fail because its SHAPE is not in the
    language, not because the search is weak. `emit matching first` must LOAD
    before the loop starts, and no amount of repair inside a bare LOOP will
    ever produce a prologue."""
    ctr = Counter()
    for pro, lpro in [(p, l) for l in LOOP_PROLOGUES for p in PROLOGUES]:
        wrap = ((lambda c: ("LOOP", c)) if lpro is None
                else (lambda c, _l=lpro: ("LOOP", ("SEQ", _l, c))))
        cases, ok = [], True
        for tape, want in pairs:
            st0 = SSt(0)
            if pro is not None:
                st0, _ = srun(pro, tape, st0)
                if not want.startswith("".join(st0.out)):
                    ok = False       # the prologue already emitted a wrong
                    break            # byte; no loop body can undo that
            cases.append((tape, want, st0))
        if not ok:
            continue
        inner = arm(cases, tests, bodies, ctr, wrap=wrap)
        if inner is None:
            continue
        whole = inner if pro is None else ("SEQ", pro, inner)
        if POLISH:
            whole = polish(whole, cases)
        if all(P.semit(whole, t) == w for t, w in pairs):
            return whole, ctr.n
    return None, ctr.n


def main() -> int:
    t0 = time.perf_counter()
    print("X63b: the sparse store, searched by counterexample\n")
    print(f"train {TRAIN}\nheld-out {len(HELD_OUT)} tapes\n")

    print("1. EXPRESSIBILITY -- does the store close X62's three gaps?")
    print(f'{"family":12} {"task":22} {"witness":>9} {"held-out":>9}  note')
    print("-" * 72)
    expressible, tasks = {}, []
    for fam, ts in FAMILIES.items():
        for name, f in ts:
            tasks.append((fam, name, f))
            wit = WITNESS[name]
            if wit is None:
                print(f"{fam:12} {name:22} {'none':>9} {'-':>9}  "
                      f"EMIT and GET both read forwards")
                expressible[name] = False
                continue
            tr = all(P.semit(wit, t) == f(t) for t in TRAIN)
            ho = all(P.semit(wit, t) == f(t) for t in HELD_OUT)
            expressible[name] = tr and ho
            note = "" if tr and ho else "  <- WITNESS IS WRONG"
            print(f"{fam:12} {name:22} {('ok' if tr else 'WRONG'):>9} "
                  f"{('ok' if ho else 'FAILS'):>9}{note}")

    was_impossible = ["first occurrence only", "emit if seen before",
                      "substitute"]
    closed = [n for n in was_impossible if expressible.get(n)]
    print(f"\n   X62 proved 3 tasks inexpressible. The store closes "
          f"{len(closed)}/3: {', '.join(closed) if closed else 'none'}")

    print("\n2. CEGIS -- no queue, no ranking, repair at the divergence")
    alpha = sorted(set("".join(TRAIN)))
    tests = [("AT", o, c) for o in (0, 1, 2, 3) for c in alpha + ["$"]]
    tests += [EMPTY, ("FULL",), M0, ("MATCH", 1), HAS]
    tests += [("TOP", c) for c in alpha]
    acts = A.ACTS + (PUT, GET)
    bodies = list(acts)
    bodies += [seq(a, b) for a in acts for b in acts if a != b]
    bodies += [seq(E, LO, AD), seq(LO, AD), seq(E, PUT, AD), seq(PUT, AD),
               seq(GET, AD), seq(AD, LO, AD, PUT, AD, AD)]
    print(f"   {len(tests)} tests, {len(bodies)} bodies, "
          f"{len(tests)*len(bodies):,} rules per repair step, "
          f"{len(PROLOGUES)} shapes\n")
    print("   arm A = strict: every rule must move the divergence later")
    print("   arm B = A + bounded tolerance for rules that change state "
          "only\n")
    hdr = (f'{"family":12} {"task":22} {"A":>4}{"evals":>9}{"held":>7}'
           f'{"":4}{"B":>4}{"evals":>9}{"held":>7}')
    print(hdr + "\n" + "-" * len(hdr))
    res = {}
    for fam, name, f in tasks:
        if not expressible.get(name):
            continue
        pairs = [(t, f(t)) for t in TRAIN]
        row = []
        for arm in (cegis, cegis_plateau):
            prog, ev = solve(pairs, tests, bodies, arm)
            ho = prog is not None and all(P.semit(prog, t) == f(t)
                                          for t in HELD_OUT)
            row.append((prog is not None, ev, ho))
        res[name] = row
        cells = ""
        for k, (fo, ev, ho) in enumerate(row):
            cells += (f'{("yes" if fo else "no"):>4}{ev:>9,}'
                      f'{("ok" if ho else ("NO" if fo else "-")):>7}'
                      + ("    " if k == 0 else ""))
        print(f"{fam:12} {name:22} {cells}")

    print("\n3. AGAINST X62's TABLE SEARCH")
    nA = sum(1 for v in res.values() if v[0][2])
    nB = sum(1 for v in res.values() if v[1][2])
    fA = sum(1 for v in res.values() if v[0][0])
    fB = sum(1 for v in res.values() if v[1][0])
    n_expr = sum(1 for v in expressible.values() if v)
    evA = max(v[0][1] for v in res.values())
    evB = max(v[1][1] for v in res.values())
    print(f'{"":28}{"expressible":>12}{"found":>7}{"generalises":>13}'
          f'{"worst evals":>13}')
    print(f'{"X62, table + frontier":28}{7:>12}{4:>7}{3:>13}{"976,521":>13}')
    print(f'{"X63b arm A, store + CEGIS":28}{n_expr:>12}{fA:>7}{nA:>13}'
          f'{evA:>13,}')
    print(f'{"X63b arm B, + plateau":28}{n_expr:>12}{fB:>7}{nB:>13}'
          f'{evB:>13,}')
    print(f"\n   evaluation budget cut {976521/max(evB,1):,.0f}x against "
          f"X62's worst case.")
    both = {n for n, v in res.items() if v[1][2]}
    x62_clean = {"strip comment", "emit matching first", "capture brackets"}
    print(f"   X62 solved cleanly : {sorted(x62_clean)}")
    print(f"   X63b arm B cleanly : {sorted(both)}")
    print(f"   lost against X62   : {sorted(x62_clean - both) or 'none'}")
    print(f"   gained             : {sorted(both - x62_clean) or 'none'}")

    # ------------------------------------------------------------------
    # 10 of 10 found and 3 of 10 generalising is not a search failure. The
    # search returns a program consistent with every tape it was given; if
    # that program is wrong on held-out tapes, the EVIDENCE did not determine
    # the task. X62 said the same thing about two tapes and treated it as a
    # footnote. It is measurable, so measure it.
    print("\n3b. WHY DOES 10-OF-10 FOUND GENERALISE 3 OF 10?")
    print("    Four explanations, each with a cheap experiment. `-` means")
    print("    not found; a number is held-out tapes passed, out of 10.\n")
    POOL = ["a#b#a", "(aa)(", "abab", "(ab)a", "ab#()", "(ab)ba", "a#(b)",
            "((a))"]
    global POLISH
    forced = lambda c: ("LOOP", ("SEQ", LO, c))
    hdr = (f'{"task":22}{"2":>5}{"4":>5}{"8":>5}{"minimal":>9}'
           f'{"shape forced":>14}')
    print("   " + hdr + "\n   " + "-" * len(hdr))
    curve = {}
    for _fam, name, f in tasks:
        if not expressible.get(name):
            continue
        row = []
        for k in (2, 4, 8):                      # H1: thin evidence
            pairs = [(t, f(t)) for t in POOL[:k]]
            pr, _e = solve(pairs, tests, bodies, cegis_plateau)
            row.append(None if pr is None else
                       sum(1 for t in HELD_OUT if P.semit(pr, t) == f(t)))
        pairs4 = [(t, f(t)) for t in POOL[:4]]
        POLISH = True                            # H2: no simplicity bias
        pr, _e = solve(pairs4, tests, bodies, cegis_plateau)
        POLISH = False
        row.append(None if pr is None else
                   sum(1 for t in HELD_OUT if P.semit(pr, t) == f(t)))
        # H3: the witness's shape is not in the language. Force it and see.
        ctr = Counter()
        pr = cegis_plateau([(t, f(t), SSt(0)) for t in POOL[:4]],
                           tests, bodies, ctr, wrap=forced)
        row.append(None if pr is None else
                   sum(1 for t in HELD_OUT if P.semit(pr, t) == f(t)))
        curve[name] = row
        print(f"   {name:22}"
              + "".join(f'{("-" if v is None else v):>5}' for v in row[:3])
              + f'{("-" if row[3] is None else row[3]):>9}'
              + f'{("-" if row[4] is None else row[4]):>14}')
    print("   " + "-" * len(hdr))
    cap = 10 * len(curve)
    tots = [sum(v for v in (curve[n][c] for n in curve) if v) for c in range(5)]
    print(f'   {"TOTAL held-out":22}{tots[0]:>5}{tots[1]:>5}{tots[2]:>5}'
          f'{tots[3]:>9}{tots[4]:>14}   / {cap}')
    print("\n   H1 thin evidence      NOT MONOTONE, and nearly flat in the")
    print("      aggregate: 52 -> 56 -> 56 of 100 while the evidence doubles")
    print("      twice. Per task it both helps and hurts -- `dedupe adjacent`")
    print("      3->10 and `first occurrence only` 1->10, against `capture")
    print("      quoted` 5->3 and `capture brackets` 6->not found. More")
    print("      evidence is not a reliable fix, which is a weaker and more")
    print("      accurate claim than refuting it outright.")
    print("   H2 no simplicity bias REFUTED -- deleting every rule that can")
    print("      be deleted moves the total from 56 to 55.")
    print("   H3 missing shape      REFUTED -- forcing the witness's exact")
    print("      shape, LOOP(SEQ(LOAD, .)), still gives 0/10 on `first")
    print("      occurrence only`; the found program even opens with the")
    print("      right rule and then piles on AT tests that fit by accident.")
    print("      Forcing it lowers the total to 45, because the shapes it")
    print("      displaces were carrying other tasks.")
    print("   H4 weak search        REFUTED in section 3 -- it finds 10/10.")
    print("\n   What is left is that FITTING IS NOT IDENTIFYING. The training")
    print("   tapes admit many consistent programs, nothing in this machine")
    print("   prefers the right one, and nothing in it can notice that the")
    print("   question is undetermined. `measure_identifiability.py` reports")
    print("   the same about ordered_targets elsewhere in this repo. That is")
    print("   the whole content of X64, reached from the other direction.")

    print("\n4. THE PRE-REGISTERED DECISION")
    a_ok = len(closed) == 3
    b_ok = nB >= 3
    print(f"   (a) the store closes X62's 3 inexpressible tasks : "
          f"{'YES' if a_ok else 'NO'} ({len(closed)}/3)")
    print(f"   (b) CEGIS finds >= X62's 3 clean, without a gradient : "
          f"{'YES' if b_ok else 'NO'} ({nB})")
    if a_ok and b_ok:
        print("\n   -> The store earns its place and the gradient was not")
        print("      load-bearing. X63's gate is met; X64 removes the target.")
    elif a_ok:
        print("\n   -> The store earns its place; the SEARCH does not yet.")
        print("      One rule per divergence is too weak a repair -- X58")
        print("      measured exactly that bound on the table side.")
    else:
        print("\n   -> The store did not close the gap it was chosen to "
              "close.\n      Re-read X62's counting arguments before "
              "patching anything.")
    print(f"\n({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
