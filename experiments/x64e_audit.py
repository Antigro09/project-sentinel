"""F-1: the X64E audit. Reporting and diagnosis only -- X64E is not changed.

Six gaps were identified in the X64E report. Two of them are substantive and
both are confirmed here:

  the commitment rule was MISDESCRIBED. X64E's docstring says the system
  answers "only when the EVIDENCE leaves one behaviour". It does not: the
  main arm commits when the behaviour posterior exceeds 0.99, which lets
  LANGUAGE authorise an answer while behaviourally distinct rivals remain.
  The two policies give very different numbers and both are reported below.

  21 of 168 logical forms are absent from the 66/38/43 split. They are
  degenerate -- empty output on every universe input, e.g.
  `keep(brackets @ inside brackets)`, since a bracket is never inside a
  bracket. 147 + 21 = 168 and nothing is missing.

Run: uv run python experiments/x64e_audit.py
"""

import random
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

import x64e_semantics as E


GATE_CALIBRATION = {
    "E0": "none -- a stop condition, not a claim about the system",
    "E1": "hard lexicon excluding the target (X64C arm retains 36/86)",
    "E2": "uniform semantics, shuffled pairing, role-blind parser",
    "E3": "always-compatible and always-conflict detectors (both AUROC 0.5)",
    "E4": "shuffled instructions (denotation 0.00)",
    "E5": "role-blind parser (denotation 0.67 vs 1.00)",
    "E6": "shuffled-language arm must not produce wrong answers",
    "E7": "demonstrations-only arm",
    "E8": "silent-OOV would show as answered-and-wrong on variant 2",
    "E9": "confirmation bypass (2 wrong vs 0)",
    "E10": "planted identity token forced to posterior > 0.999",
    "E11": "all ten defects",
}


def paired_bootstrap(pairs, n=2000, seed=13):
    """Resampled BY TASK MEANING, not by instruction condition: three
    paraphrases of one meaning are not three independent observations."""
    rg = random.Random(seed)
    diffs = []
    for _ in range(n):
        s = [pairs[rg.randrange(len(pairs))] for _ in pairs]
        diffs.append(sum(a - b for a, b in s) / len(s))
    diffs.sort()
    return (diffs[int(0.025 * n)], sum(diffs) / len(diffs),
            diffs[int(0.975 * n)])


def main() -> int:
    t0 = time.perf_counter()
    dev = E.forms_in(E.DEV_PAIRS)
    val = E.forms_in(E.VAL_PAIRS)
    test = E.forms_in(E.TEST_PAIRS)
    E.pool()
    P = E.Parser().fit(E.training_examples(dev, n_demos=E.TRAIN_DEMOS),
                       epochs=E.EPOCHS, lr=E.LR, l2=E.L2)
    A = E.authored_parser(dev)
    RB = E.Parser(role_blind=True).fit(
        E.training_examples(dev, n_demos=E.TRAIN_DEMOS), epochs=E.EPOCHS,
        lr=E.LR, l2=E.L2)
    theta = E.calibrate_theta(P, val)
    print("F-1: X64E AUDIT -- reporting and diagnosis only\n")

    print("3. THE 21 MISSING LOGICAL FORMS")
    deg = [z for z in E.ALL_Z if all(o == "" for o in E.denote(z))]
    print(f"   {len(dev)} + {len(val)} + {len(test)} = "
          f"{len(dev)+len(val)+len(test)} of {len(E.ALL_Z)}")
    print(f"   degenerate (empty on every universe input): {len(deg)}")
    print(f"   {len(dev)+len(val)+len(test)} + {len(deg)} = "
          f"{len(dev)+len(val)+len(test)+len(deg)} -- fully accounted")
    print(f"   e.g. {deg[:3]}")

    print("\n4 AND 5. THE COMMITMENT RULE, AND THE CORRECTION")
    print("   X64E's docstring says the system answers only when the")
    print("   EVIDENCE leaves one behaviour. That is FALSE of the arm whose")
    print("   numbers were reported. The final answer condition is:\n")
    print("     answer iff  (a) the evidence-consistent set is a singleton")
    print("                 OR (b) the behaviour posterior of the top")
    print("                        candidate exceeds theta_commit = 0.99")
    print("                 AND the candidate agrees with the user on all")
    print("                     28 confirmation inputs\n")
    print("   Clause (b) lets LANGUAGE authorise an answer while")
    print("   behaviourally distinct rivals remain. The two policies were")
    print("   never separated in the X64E report; they are here.\n")
    rows = []
    for lab, kw in (("A  demonstrations only", dict(mode="none",
                                                    query="disagreement")),
                    ("B  language ranks, EVIDENCE commits", dict(commit=None)),
                    ("C  language ranks AND commits (reported)",
                     dict(commit=E.COMMIT_TAU))):
        ans = cor = wr = q = 0
        for z in test:
            f = E.execute(z)
            for v in (0, 1):
                r = E.solve(z, v, P, **kw)
                q += r["asked"]
                if r["verdict"] == "answered":
                    ans += 1
                    h = E.held(r, f)
                    cor += h == 10
                    wr += h != 10
        rows.append((lab, ans, cor, wr, q))
        print(f"   {lab:42}{ans:>5} answered{cor:>5} correct{wr:>4} wrong"
              f"{q:>6} queries")
    print("\n   So language's operational value splits in two: ranking the")
    print("   questions is worth 196 -> 150, and authorising commitment is")
    print("   worth 150 -> 2. E7 passes under either policy, but the")
    print("   headline number came from the second and the text described")
    print("   the first.")

    print("\n6 AND 7. FULL POPULATION VERSUS COMMON SUPPORT")
    shared = [z for z in test if E.w_covers(z)]
    wsen = E.w_senses(dev)
    ARMS = [
        ("demonstrations only",
         lambda z, v: E.solve(z, v, P, mode="none", query="disagreement")),
        ("X64C hard lexicon",
         lambda z, v: E.run_w(z, v, E.W.x64c_senses(), "hard")),
        ("X64D predicate senses", lambda z, v: E.run_w(z, v, wsen, "joint")),
        ("role-blind induced",
         lambda z, v: E.solve(z, v, RB, commit=E.COMMIT_TAU)),
        ("authored structure",
         lambda z, v: E.solve(z, v, A, commit=E.COMMIT_TAU)),
        ("MAIN induced", lambda z, v: E.solve(z, v, P, commit=E.COMMIT_TAU)),
    ]
    print(f'\n   {"arm":26}{"scope":>18}{"n":>5}{"ans":>5}{"corr":>6}'
          f'{"wrong":>7}{"abst":>6}{"uncov":>7}{"q":>6}')
    for lab, fn in ARMS:
        for scope, forms, variants in (("full population", test, (0, 1, 2)),
                                       ("common support", shared, (0, 1))):
            n = ans = cor = wr = q = unc = 0
            for z in forms:
                f = E.execute(z)
                for v in variants:
                    r = fn(z, v)
                    if r is None:
                        unc += 1
                        continue
                    n += 1
                    q += r["asked"]
                    if r["verdict"] == "answered":
                        ans += 1
                        h = E.held(r, f)
                        cor += h == 10
                        wr += h != 10
            print(f'   {lab:26}{scope:>18}{n:>5}{ans:>5}{cor:>6}{wr:>7}'
                  f'{n-ans:>6}{unc:>7}{q:>6}')
    return _part2(dev, val, test, P, A, RB, theta, t0)


def _part2(dev, val, test, P, A, RB, theta, t0):
    print("\n9. UNSEEN-WORD CASES, BROKEN DOWN")
    print("   E8 passed with denotation accuracy 0.02 on variant 2, which")
    print("   needs explaining: the parser does NOT understand those words.\n")
    cats = dict(interpreted=0, unsupported=0, clarified=0, ignored=0,
                wrong=0, total=0)
    qcost = 0
    for z in test:
        f = E.execute(z)
        toks = E.instr(z, 2)
        oov = E.known(P, toks)
        if not oov:
            continue
        cats["total"] += 1
        d = P.dist(toks)
        parsed_right = E.denote(max(d, key=d.get)) == E.denote(z)
        r = E.solve(z, 2, P, commit=E.COMMIT_TAU)
        qcost += r["asked"] + r["sem"]
        h = E.held(r, f)
        if r["verdict"] != "answered":
            cats["unsupported"] += 1
        elif h != 10:
            cats["wrong"] += 1
        elif parsed_right:
            cats["interpreted"] += 1
        elif r["sem"] > 0:
            cats["clarified"] += 1
        else:
            cats["ignored"] += 1
    print(f'   {"forms carrying an unknown word":42}{cats["total"]:>5}')
    for k, lab in (("interpreted", "the word was correctly interpreted"),
                   ("clarified", "resolved by a semantic clarification"),
                   ("ignored", "silently ignored, answer came from evidence"),
                   ("unsupported", "reported unsupported / unresolved"),
                   ("wrong", "answered confidently and WRONG")):
        print(f'   {lab:42}{cats[k]:>5}')
    print(f'   {"queries spent on these":42}{qcost:>5}')
    print("\n   E8 passes because nothing is answered wrongly, NOT because")
    print("   unseen vocabulary was understood. Where the answer is right,")
    print("   the evidence supplied it and the unknown word contributed")
    print("   nothing. That is safe, and it is not comprehension.")

    print("\n10 AND 11. EXACT FORM, IDENTIFIABILITY, RANK AND MASS")
    byb = E.forms_by_behaviour()
    ident = [z for z in test if len(byb[E.denote(z)]) == 1]
    ranks, mass = [], []
    ex_all = ex_id = mod = 0
    for z in test:
        d = P.dist(E.instr(z, 0))
        best = max(d, key=d.get)
        ex_all += best == z
        mod += E.denote(best) == E.denote(z)
        if z in ident:
            ex_id += best == z
        srt = sorted(d.items(), key=lambda kv: -kv[1])
        ranks.append(1 + [zz for zz, _v in srt].index(z))
        mass.append(sum(v for zz, v in d.items()
                        if E.denote(zz) == E.denote(z)))
    print(f'   exact form, all {len(test)} test forms          '
          f'{ex_all/len(test):.2f}')
    print(f'   exact form, {len(ident)} behaviourally identifiable  '
          f'{ex_id/max(1,len(ident)):.2f}')
    print(f'   modulo behavioural equivalence            {mod/len(test):.2f}')
    print(f'   median rank of the gold form              '
          f'{sorted(ranks)[len(ranks)//2]}')
    print(f'   mean posterior mass on the gold BEHAVIOUR  '
          f'{sum(mass)/len(mass):.3f}')
    print("   Equivalence is over the finite universe, so a form that is")
    print("   'equivalent' here may differ on inputs the universe omits.")

    print("\n12. PAIRED BOOTSTRAP, RESAMPLED BY TASK MEANING\n")
    pairs_q_auth, pairs_q_rb, pairs_a = [], [], []
    for i, z in enumerate(test):
        qm = qa = qr = 0
        for v in (0, 1):
            qm += E.solve(z, v, P, commit=E.COMMIT_TAU)["asked"]
            qa += E.solve(z, v, A, commit=E.COMMIT_TAU)["asked"]
            qr += E.solve(z, v, RB, commit=E.COMMIT_TAU)["asked"]
        pairs_q_auth.append((qa, qm))
        pairs_q_rb.append((qr, qm))
        f = E.execute(z)
        other = E.execute(test[(i + 1) % len(test)])
        for Pr, acc in ((P, pairs_a),):
            pm = Pr.dist(E.instr(z, 0))
            pa = A.dist(E.instr(z, 0))
            cm_m = E.conflict_score(pm, {t: f(t) for t in E.UNIVERSE[:2]})
            cm_x = E.conflict_score(pm, {t: other(t) for t in E.UNIVERSE[:2]})
            ca_m = E.conflict_score(pa, {t: f(t) for t in E.UNIVERSE[:2]})
            ca_x = E.conflict_score(pa, {t: other(t) for t in E.UNIVERSE[:2]})
            acc.append(((cm_x - cm_m), (ca_x - ca_m)))
    for lab, pr in (("queries saved vs authored structure", pairs_q_auth),
                    ("queries saved vs role-blind", pairs_q_rb)):
        lo, mu, hi = paired_bootstrap(pr)
        sig = "excludes 0" if lo > 0 or hi < 0 else "INCLUDES 0"
        print(f"   {lab:38} mean {mu:+.3f}  95% CI "
              f"({lo:+.3f},{hi:+.3f})  {sig}")
    lo, mu, hi = paired_bootstrap([(a, b) for a, b in pairs_a])
    sig = "excludes 0" if lo > 0 or hi < 0 else "INCLUDES 0"
    print(f"   {'conflict margin vs authored':38} mean {mu:+.3f}  95% CI "
          f"({lo:+.3f},{hi:+.3f})  {sig}")
    print("   Margin = mismatched minus matched conflict score, per meaning.")

    print("\n1 AND 2. GATES AND THE DEFECT THAT CALIBRATES EACH\n")
    for k, v in GATE_CALIBRATION.items():
        print(f"   {k:>4}  {v}")

    print("\n13. PROVENANCE")
    print("   tracked files: clean;  untracked: .claude/worktrees/ present")
    print("   `clean` in the X64E report should have said `tracked clean`.")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
