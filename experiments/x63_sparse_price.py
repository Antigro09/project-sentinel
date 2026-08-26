"""X63a: the price of dropping the behaviour table -- and I priced the wrong risk.

X62's pre-registered rule fired: set 0/2 and associative 0/1 means an exact
sparse key-value store. The roadmap says price the thing it breaks before
building on it, because every experiment since X47 evaluates a candidate by
composing precomputed BEHAVIOUR TABLES: one row per situation, composition by
numpy fancy-indexing, dedup by signature bytes, and a search gradient of
`(sig == target).mean()` over ~27,000 numbers.

A store cannot be tabulated. Over 5 keys it has (5+1)^5 = 7,776
configurations, multiplying the situation space by the same factor:

    machine                              situations    MB/behaviour
    head + stack(2) + 1 register              2,232           0.107
      + sparse store, tabulated            17,356,032       833.090

So the store forces concrete execution. The pre-registered worry was that
execution would be too slow to carry X58-X62's ~10^6-evaluation budgets:

    evaluation                        us/candidate   vs the search step
    table, rebuilt from atoms                525.3                 2.4x
    table, one rule + re-wrap                218.4                 1.0x
    concrete execution                         6.3                0.03x
    concrete execution + store                 6.3                0.03x

THAT WORRY WAS BACKWARDS. Execution is 35x FASTER than the step the frontier
actually pays, and adding the store costs nothing measurable, because a run
touches the keys it touches whatever the key space is. The table was never a
speed device at this scale. It is a RESOLUTION device, and it is bought with
memory rather than time -- which is exactly why the store kills it.

WHAT THE TABLE SELLS. Over 300 sampled programs:

    distinct behaviours by full signature    211
    distinct behaviours by output only        22   -- 9.6x resolution lost

and as a search gradient toward `capture brackets`, with four output metrics
run as calibration arms so a crude one could not manufacture the answer:

    metric            levels  vs table   r (all)  r (top 10%)
    FULL TABLE           123     1.00x         -            -
    positional            10     0.08x     0.167        0.398
    exact match            2     0.02x    -0.012       -0.682
    common prefix          6     0.05x     0.124        0.294
    character bag          8     0.07x     0.190        0.555

None of them tracks the table. `exact match` is ANTI-correlated among near
misses (-0.682): matching one more tape exactly can mean moving further away
in the only ordering the search can act on.

AND THE RESOLUTION IS REAL, NOT PHANTOM. The obvious objection is that the
signature scores all 2,232 situations while a run from a fresh start reaches
almost none of them, so most of that precision would be agreement on states
nobody visits. Only 390 situations (17%) are reachable -- and restricted to
those, the table still separates 129 classes, slightly MORE than the 123 it
separates overall. The precision is about programs, not about dead states.

THE DECISION. Clause (a) passed by a factor of 1,600 and clause (b) failed:
speed survives, the gradient does not. Distance-ranked search is dead in the
executed setting however fast it runs -- 8 levels cannot order a queue that
123 levels barely ordered (X62 found 4 of 7 expressible tasks anyway). So
X63b is counterexample-guided: localise the FIRST divergence between the
candidate's output and the evidence, and repair at that point (X57's
mechanism), which needs no global gradient at all.

That is a different mechanism, not a patch on this one, which is why it is a
separate experiment rather than a flag on this one.

Run: uv run python experiments/x63_sparse_price.py    (~1s)
"""

import itertools
import random
import sys
import time

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x62_memory_audit as A

ALPHA = A.ALPHA
TAPES = A.TAPES


# ------------------------------------------------------ 1. the store machine
#
# The audited machine plus a key->byte store. Emission becomes BYTE-valued
# rather than tape-index-valued, which is itself forced: `substitute` emits a
# value that need not sit at the head, and the table encodes emission as
# counts over tape POSITIONS. Keeping index emission would mean storing
# positions instead of bytes -- and positions grow with the tape, so the store
# would inherit exactly the unboundedness the store was meant to avoid.

SACTS = A.ACTS + ("PUT", "GET")


class SSt:
    """State with a sparse store. `store` is a frozenset of (k, v) pairs so it
    stays hashable and holds only the keys a run actually touched."""
    __slots__ = ("pos", "stack", "reg", "out", "live", "store")

    def __init__(self, pos, stack=(), reg=A.NONE, out=(), live=True,
                 store=frozenset()):
        self.pos, self.stack, self.reg = pos, stack, reg
        self.out, self.live, self.store = out, live, store

    def key(self):
        return (self.pos, self.stack, self.reg, self.out, self.live,
                self.store)

    def copy(self, **kw):
        d = dict(pos=self.pos, stack=self.stack, reg=self.reg, out=self.out,
                 live=self.live, store=self.store)
        d.update(kw)
        return SSt(**d)


def lookup(store, k):
    for kk, vv in store:
        if kk == k:
            return vv
    return None


def stest(p, tape, st):
    if p[0] == "HAS":
        return st.reg != A.NONE and lookup(st.store, st.reg) is not None
    return A.test_pred(p, tape, st)


def srun(expr, tape, st, fuel=8192):
    """Same interpreter as X62, plus PUT/GET. Nothing here is bounded except
    fuel -- X62's own bug was bounding the machine as well as the abstraction."""
    if fuel <= 0 or not st.live:
        return st, fuel
    if isinstance(expr, str):
        if expr == "NOP":
            return st, fuel - 1
        if expr == "ADV":
            return st.copy(pos=min(st.pos + 1, len(tape))), fuel - 1
        if expr == "EMIT":
            if st.pos < len(tape):
                return st.copy(out=st.out + (tape[st.pos],)), fuel - 1
            return st, fuel - 1
        if expr == "HALT":
            return st.copy(live=False), fuel - 1
        if expr == "PUSH":
            if st.pos < len(tape):
                return st.copy(stack=st.stack + (tape[st.pos],)), fuel - 1
            return st, fuel - 1
        if expr == "POP":
            return st.copy(stack=st.stack[:-1]), fuel - 1
        if expr == "LOAD":
            if st.pos < len(tape):
                return st.copy(reg=tape[st.pos]), fuel - 1
            return st, fuel - 1
        if expr == "PUT":
            if st.reg != A.NONE and st.pos < len(tape):
                s2 = frozenset((k, v) for k, v in st.store if k != st.reg)
                return st.copy(store=s2 | {(st.reg, tape[st.pos])}), fuel - 1
            return st, fuel - 1
        if expr == "GET":
            # Guarded at the end of the tape like EMIT, LOAD, PUSH and PUT --
            # GET was the only act without it, and an unguarded emitting act
            # never reaches a LOOP fixed point: `substitute` on '(ab)a'
            # produced 'bbbbbb' because the register still held a key after
            # the head ran off the end.
            if st.pos >= len(tape) or st.reg == A.NONE:
                return st, fuel - 1
            v = lookup(st.store, st.reg)
            return (st.copy(out=st.out + (v,)) if v else st), fuel - 1
        return st, fuel - 1
    h = expr[0]
    if h == "SEQ":
        st, fuel = srun(expr[1], tape, st, fuel)
        return srun(expr[2], tape, st, fuel)
    if h == "IF":
        return srun(expr[2] if stest(expr[1], tape, st) else expr[3],
                    tape, st, fuel - 1)
    if h == "LOOP":
        for _ in range(len(tape) + 2):
            nxt, fuel = srun(expr[1], tape, st, fuel)
            if nxt.key() == st.key() or fuel <= 0:
                return nxt, fuel
            st = nxt
        return st, fuel
    return st, fuel - 1


def semit(expr, tape):
    res, _ = srun(expr, tape, SSt(0))
    return "".join(res.out)


# ---------------------------------------------------------- random programs

def sample_programs(n, rng, acts, preds, max_rules=3):
    """Decision lists in the shape the frontier actually builds: a chain of
    IF-tests over atom or two-atom bodies, wrapped in a LOOP."""
    def body():
        a = rng.choice(acts)
        if rng.random() < 0.5:
            b = rng.choice([x for x in acts if x != a])
            return ("SEQ", a, b)
        return a

    out = []
    for _ in range(n):
        e = body()
        for _ in range(rng.randrange(max_rules)):
            e = ("IF", rng.choice(preds), body(), e)
        out.append(("LOOP", e))
    return out


def main() -> int:
    t0 = time.perf_counter()
    rng = random.Random(20260825)
    print("X63a: what the behaviour table costs, and what dropping it costs\n")

    alpha = sorted(set("".join(TAPES)))
    space = A.Space(TAPES, alpha)
    w = space.w
    print(f"tapes {TAPES}, alphabet {''.join(alpha)!r}")
    print(f"X62's space: {space.n:,} situations, "
          f"{space.width * 4 / 1e6:.3f} MB per behaviour\n")

    # ------------------------------------------------ 1. tabulating a store
    print("1. WHY THE STORE CANNOT BE TABULATED")
    print(f'{"machine":34} {"situations":>14} {"MB/behaviour":>14}')
    print("-" * 64)
    k = len(alpha)
    store_states = (k + 1) ** k          # every key mapped to a byte or unset
    rows = [("head + stack(2) + 1 register", space.n),
            ("  + sparse store, tabulated", space.n * store_states)]
    for label, n in rows:
        mb = (2 * n + n * w) * 4 / 1e6
        print(f"{label:34} {n:>14,} {mb:>14,.3f}")
    print(f"\n   a store over {k} keys has ({k}+1)^{k} = {store_states:,} "
          f"configurations,")
    print(f"   so tabulating multiplies the space by {store_states:,}x. "
          f"That is the")
    print("   whole argument for executing instead: execution pays for the "
          "keys")
    print("   TOUCHED, tabulation pays for the keys POSSIBLE.")

    # -------------------------------------------- 2. per-candidate cost, real
    print("\n2. PER-CANDIDATE COST, MEASURED ON THE SAME PROGRAMS")
    preds = [("AT", o, c) for o in (0, 1) for c in alpha + ["$"]]
    preds += [("EMPTY",), ("FULL",), ("MATCH", 0)]
    progs = sample_programs(300, rng, A.ACTS, preds)

    t = time.perf_counter()
    sigs = [space.table(p) for p in progs]
    tab_s = (time.perf_counter() - t) / len(progs)

    # The frontier does NOT rebuild a program from atoms per candidate: it
    # adds one rule to a chain it already has, then re-wraps in the LOOP.
    # Timing `table()` against execution would flatter execution, so time the
    # step the search actually pays.
    k = max(len(t) for t in TAPES) + 2
    pm = space.pred(("AT", 0, "("))
    bt = space.atoms["ADV"]
    dt = space.table(("SEQ", "EMIT", "ADV"))
    t = time.perf_counter()
    for _ in range(len(progs)):
        space.loop(space.branch(pm, bt, dt), k)
    inc_s = (time.perf_counter() - t) / len(progs)

    t = time.perf_counter()
    outs = [tuple(A.emit(p, tp) for tp in TAPES) for p in progs]
    exe_s = (time.perf_counter() - t) / len(progs)

    t = time.perf_counter()
    souts = [tuple(semit(p, tp) for tp in TAPES) for p in progs]
    sexe_s = (time.perf_counter() - t) / len(progs)

    print(f'{"evaluation":34} {"us/candidate":>14} {"vs incremental":>15}')
    print("-" * 66)
    print(f'{"table, rebuilt from atoms":34} {tab_s*1e6:>14,.1f} '
          f'{tab_s/inc_s:>14,.1f}x')
    print(f'{"table, one rule + re-wrap  <- the":34} {inc_s*1e6:>14,.1f} '
          f'{"1.0x":>15}')
    print(f'{"   step search actually pays":34} {"":>14} {"":>15}')
    print(f'{"concrete execution":34} {exe_s*1e6:>14,.1f} '
          f'{exe_s/inc_s:>14,.2f}x')
    print(f'{"concrete execution + store":34} {sexe_s*1e6:>14,.1f} '
          f'{sexe_s/inc_s:>14,.2f}x')

    # ------------------------------------------------------ 3. how each scales
    print("\n3. HOW EACH SCALES -- the table with the SPACE, execution with "
          "the TAPES")
    print(f'{"situations":>12} {"MB/behaviour":>14} {"table us":>12} '
          f'{"exec us":>12}')
    print("-" * 54)
    small = sample_programs(60, rng, A.ACTS, preds)
    for tapes in (["a#b", "()a"], TAPES, ["a#b#a", "(aa)(", "ab#()"],
                  ["a#b#a#b", "(aa)()", "ab#()a", "#ab(a)"]):
        sp = A.Space(tapes, alpha)
        t = time.perf_counter()
        for p in small:
            sp.table(p)
        tt = (time.perf_counter() - t) / len(small)
        t = time.perf_counter()
        for p in small:
            for tp in tapes:
                A.emit(p, tp)
        et = (time.perf_counter() - t) / len(small)
        mb = sp.width * 4 / 1e6
        print(f"{sp.n:>12,} {mb:>14.3f} {tt*1e6:>12,.1f} {et*1e6:>12,.1f}")

    n0, n1 = 2, 3
    print("\n   The table's cost is a function of the situation space, which "
          "the")
    print("   store multiplies. Execution never reads a situation it does not "
          "reach.")

    # ------------------------------- 4. what the table buys: resolution + slope
    print("\n4. WHAT THE TABLE BUYS THAT EXECUTION DOES NOT")
    tab_classes = len({s.tobytes() for s in sigs})
    out_classes = len(set(outs))
    print(f"   {len(progs)} sampled programs")
    print(f"   distinct behaviours by full signature : {tab_classes:>6}")
    print(f"   distinct behaviours by output only    : {out_classes:>6}")
    print(f"   resolution lost                       : "
          f"{tab_classes / max(out_classes,1):>6.1f}x")

    # The gradient, with CALIBRATION ARMS. One crude output metric scoring
    # badly would prove nothing except that the metric was crude, so try four
    # and report the best. X1's random arm caught exactly this kind of
    # self-inflicted negative.
    wit = A.WITNESS["capture brackets"]
    target = space.table(wit)
    tgt_out = tuple(A.emit(wit, tp) for tp in TAPES)
    tab_agree = np.array([float((s == target).mean()) for s in sigs])

    def m_positional(o):
        num = den = 0
        for got, want in zip(o, tgt_out):
            m = max(len(got), len(want))
            den += m
            num += sum(1 for i in range(m)
                       if i < len(got) and i < len(want) and got[i] == want[i])
        return num / den if den else 1.0

    def m_exact(o):
        return sum(1 for g, w in zip(o, tgt_out) if g == w) / len(tgt_out)

    def m_prefix(o):
        num = den = 0
        for got, want in zip(o, tgt_out):
            m = max(len(got), len(want), 1)
            i = 0
            while i < min(len(got), len(want)) and got[i] == want[i]:
                i += 1
            num += i
            den += m
        return num / den

    def m_bag(o):
        """Character multiset overlap -- ignores order entirely."""
        num = den = 0
        for got, want in zip(o, tgt_out):
            keys = set(got) | set(want)
            den += sum(max(got.count(c), want.count(c)) for c in keys) or 1
            num += sum(min(got.count(c), want.count(c)) for c in keys)
        return num / den

    METRICS = [("positional", m_positional), ("exact match", m_exact),
               ("common prefix", m_prefix), ("character bag", m_bag)]

    # Ranking only matters among candidates that are already close, so report
    # the correlation on the top decile of table agreement as well as overall.
    order = np.argsort(-tab_agree)
    near = order[:max(10, len(order) // 10)]

    print("\n   gradient toward `capture brackets`, four output metrics:")
    hdr = (f'{"metric":16}{"levels":>8}{"vs table":>10}{"r (all)":>10}'
           f'{"r (top 10%)":>13}')
    print("   " + hdr)
    print("   " + "-" * len(hdr))
    tab_lv = len(set(np.round(tab_agree, 9)))
    print(f'   {"FULL TABLE":16}{tab_lv:>8,}{"1.00x":>10}{"-":>10}{"-":>13}')
    best_r, best_lv, best_name = -1.0, 0, ""
    for name, fn in METRICS:
        v = np.array([fn(o) for o in outs])
        lv = len(set(np.round(v, 9)))
        r_all = float(np.corrcoef(tab_agree, v)[0, 1]) if v.std() else 0.0
        sub = v[near]
        r_near = (float(np.corrcoef(tab_agree[near], sub)[0, 1])
                  if sub.std() else 0.0)
        print(f'   {name:16}{lv:>8,}{lv/tab_lv:>9.2f}x{r_all:>10.3f}'
              f'{r_near:>13.3f}')
        if r_all > best_r:
            best_r, best_lv, best_name = r_all, lv, name
    r, ex_lv = best_r, best_lv
    print(f"\n   best output metric: {best_name} (r={r:.3f}, "
          f"{ex_lv}/{tab_lv} levels)")

    # Is the table's extra resolution REAL, or is it agreement on situations
    # no run ever visits? The signature scores all 2,232 situations, and a
    # program starting at (tape, pos 0, empty stack, empty register) cannot
    # reach most of them. If restricting to the reachable subspace collapses
    # the table's resolution toward the output's, then the table was selling
    # phantom precision and the switch costs far less than section 4 implies.
    starts = [space.index[(ti, 0, (), A.NONE)] for ti in range(len(TAPES))]
    ends = {a: space.unpack(space.atoms[a])[0] for a in A.ACTS}
    seen_r, stack_r = set(starts), list(starts)
    while stack_r:
        i = stack_r.pop()
        for e in ends.values():
            j = int(e[i])
            if j not in seen_r:
                seen_r.add(j)
                stack_r.append(j)
    reach = np.array(sorted(seen_r), dtype=np.int32)
    n = space.n
    cols = np.concatenate([reach, n + reach,
                           (2 * n + (reach[:, None] * w
                                     + np.arange(w)[None, :])).ravel()])
    print(f"\n   reachable from a fresh start: {len(reach):,} of {n:,} "
          f"situations ({100*len(reach)/n:.0f}%)")
    rt = np.array([float((sg[cols] == target[cols]).mean()) for sg in sigs])
    rt_lv = len(set(np.round(rt, 9)))
    r_out = float(np.corrcoef(rt, np.array([m_bag(o) for o in outs]))[0, 1])
    print(f'   {"table, reachable only":16}{rt_lv:>8,}{rt_lv/tab_lv:>9.2f}x'
          f'{r_out:>10.3f}   <- vs character bag')
    if rt_lv > 4 * ex_lv:
        print("   The resolution survives the restriction, so it is real: "
              "the table")
        print("   separates programs that agree on every byte they emit.")
    else:
        print("   Most of the table's resolution was phantom -- agreement on")
        print("   situations no run reaches. Re-read section 4 before "
              "believing it.")

    # ------------------------------------------------------------- 5. verdict
    print("\n5. THE PRE-REGISTERED DECISION")
    speed_ok = sexe_s / inc_s < 50
    grad_ok = r > 0.5 and ex_lv >= 0.25 * tab_lv
    print(f"   (a) execution within 50x of the search step : "
          f"{'YES' if speed_ok else 'NO'} ({sexe_s/inc_s:.3f}x -- it is "
          f"{inc_s/sexe_s:.0f}x FASTER)")
    print(f"   (b) output gradient tracks the table        : "
          f"{'YES' if grad_ok else 'NO'} "
          f"(r={r:.3f}, {ex_lv}/{tab_lv} levels)")
    if speed_ok and grad_ok:
        print("\n   -> X63b builds the store on concrete execution and keeps "
              "the\n      frontier ranker unchanged.")
    elif speed_ok:
        print("\n   -> Speed survives, the GRADIENT DOES NOT. Distance-ranked")
        print("      search is dead in the executed setting however fast it "
              "runs:")
        print("      most candidates are tied, so the queue has nothing to "
              "order by.")
        print("      The replacement is counterexample-guided -- localise the")
        print("      first divergence and repair there (X57) -- which needs no")
        print("      global gradient. That is X63b, and it is a different")
        print("      mechanism, not a patch on this one.")
    else:
        print("\n   -> Execution is too slow to carry X58-X62's budgets. "
              "Before\n      any store is built, execution needs to get "
              "cheaper.")

    print(f"\n({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
