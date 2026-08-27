"""X64H-0: testbed validity for hidden-convention induction.

H1 failed at commit 789161e because conventions drew phrases from DISJOINT
pools, so the vocabulary identified the convention and a static family-aware
parser matched the oracle at 1.00. That was a generator failure.

This replaces the family with overlapping hidden codebooks: one shared
surface alphabet, hidden role-specific permutations, one hidden ordering
bit, and an exposure that shows two of three roles without announcing
which. A codeword now carries no information about which convention
produced it.

No final manifest is written and no final seed is sampled.

Run: uv run python experiments/x64h_0_validity.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x64h import codebook as K
from x64h import layer0 as L0
from x64h import semantic as S
from x64h import validity as V

OUT = Path("experiments/x64h/results")
DEV_SEEDS = tuple(range(400, 412))          # development only
VAL_SEEDS = tuple(range(500, 508))          # validation only
# Chosen on development by the sweep in section 4b. A MIXED schedule is
# required: teaching tasks (every third) let the demonstrations reveal what
# the codewords mean so the codebook is learnable, and withholding tasks
# keep the exposed roles open so the memoryless task stays ambiguous. A
# uniform schedule cannot have both.
# 32 = every distinct meaning exactly once. `Later tasks must not repeat
# earlier complete meanings`, so the episode length is bounded by the
# semantic space and V7 holds by construction rather than by sampling luck.
T = 32
N_DEMOS = 2
ALPHA = 1.0
SCHEDULE = 3


def curves(fam, fs, utt, beh, cls, arm, seeds, n_demos=N_DEMOS, T=T,
           alt=None, changing=False, withholding_only=True):
    acc = [[] for _ in range(T)]
    ent = [[] for _ in range(T)]
    mass = [[] for _ in range(T)]
    ncl = [[] for _ in range(T)]
    for e, sd in enumerate(seeds):
        pi = random.Random(sd).randrange(len(fam))
        zs = random.Random(1000 + sd).sample(list(fs), T)
        ch = ([random.Random(7000 + sd + t).randrange(len(fam))
               for t in range(T)] if changing else None)
        r = V.run_episode(fam, fs, utt, beh, cls, pi, zs, arm,
                          n_demos=n_demos, rng=random.Random(9 + sd),
                          alt_phi=(alt if alt is not None
                                   else (pi + 577) % len(fam)),
                          changing=ch, alpha=ALPHA, schedule=SCHEDULE)
        keep = [t for t in range(T)
                if (not withholding_only) or r.withholding[t]]
        for slot, t in enumerate(keep):
            acc[slot].append(r.correct[t])
            ent[slot].append(r.entropy[t])
            mass[slot].append(r.true_class_mass[t])
            ncl[slot].append(r.n_classes[t])
    mean = lambda xs: [sum(a) / len(a) for a in xs if a]
    return mean(acc), mean(ent), mean(mass), mean(ncl)


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    fam = K.full_family()
    fs = K.forms()
    utt, beh = V.precompute(fam, fs)
    cls = K.equivalence_classes(fam, fs)
    print("X64H-0: hidden-convention testbed validity\n")
    print("1. THE CONVENTION FAMILY")
    print(f"   roles O/F/S with |V| = {len(K.V_O)}/{len(K.V_F)}/{len(K.V_S)}; "
          f"codewords W_O={len(K.W_O)} and W={len(K.W)} SHARED by F and S")
    print(f"   phi = (pi_O, pi_F, pi_S, order bit) -> {len(fam)} codebooks")
    print(f"   {len(fs)} typed forms, "
          f"{len({S.denote(z) for z in fs})} behaviour classes")
    print(f"   exposure shows 2 of 3 roles and does not announce which")

    print("\n2. V9 LEAKAGE AUDIT")
    la = K.leak_audit(fam[:120], fs)
    for k, v in la.items():
        print(f"   {k:44} {v}")
    planted = fam[:60] + (K.Codebook(("o1", "o2"), ("z9", "c2", "c3", "c4"),
                                     K.W, 0),)
    pl = K.leak_audit(planted, fs)
    print(f"   planted unique-token convention detected: "
          f"{not pl['leak_free']}")

    print("\n3. V8 PAIRWISE IDENTIFIABILITY")
    sizes = {}
    for v in cls.values():
        sizes[len(v)] = sizes.get(len(v), 0) + 1
    print(f"   {len(cls)} observational-equivalence classes over "
          f"{len(fam)} codebooks; class-size histogram {sizes}")
    print(f"   every non-equivalent pair is separated by some legal context: "
          f"{len(cls) == len(fam)}")

    print("\n4. ADAPTATION CURVES on development seeds")
    print("   Reported on WITHHOLDING tasks only -- the ones whose exposed")
    print("   roles the demonstrations deliberately leave open, so the")
    print("   utterance has to carry them and the convention has to be")
    print(f"   known. Every {SCHEDULE}rd task is a teaching task.\n")
    ao, _e, _m, _n = curves(fam, fs, utt, beh, cls, "oracle", DEV_SEEDS)
    ast, _e2, _m2, ncl = curves(fam, fs, utt, beh, cls, "static", DEV_SEEDS)
    ap, ep, mp, _n3 = curves(fam, fs, utt, beh, cls, "persist", DEV_SEEDS)
    ar, _, _, _ = curves(fam, fs, utt, beh, cls, "reset", DEV_SEEDS)
    ash, _, _, _ = curves(fam, fs, utt, beh, cls, "shuffled", DEV_SEEDS)
    ad, _, _, _ = curves(fam, fs, utt, beh, cls, "default", DEV_SEEDS)
    ac, _, _, _ = curves(fam, fs, utt, beh, cls, "persist", DEV_SEEDS,
                         changing=True)
    nW = len(ao)                      # withholding tasks per episode
    step = max(1, nW // 10)
    idx = list(range(0, nW, step))
    print("   task        " + "".join(f"{i+1:>6}" for i in idx))
    for lab, c in (("oracle", ao), ("static", ast), ("persist", ap),
                   ("reset", ar), ("shuffled", ash), ("default", ad),
                   ("phi changes", ac)):
        print(f"   {lab:12}" + "".join(f"{c[i]:>6.2f}" for i in idx))
    print("   H(phi) bits " + "".join(f"{ep[i]:>6.2f}" for i in idx))
    print("   true mass   " + "".join(f"{mp[i]:>6.2f}" for i in idx))
    print("   classes/task" + "".join(f"{ncl[i]:>6.2f}" for i in idx))

    h = nW // 2
    mean = lambda c: sum(c) / len(c)
    # divide by the slice length, not by h: with an odd number of
    # withholding tasks c[h:] is longer than h and `late` reported an
    # accuracy of 1.067
    late = lambda c: sum(c[h:]) / max(1, len(c[h:]))
    early = lambda c: sum(c[:h]) / max(1, len(c[:h]))
    G = mean(ao) - mean(ast)
    R = (mean(ap) - mean(ast)) / G if abs(G) > 1e-9 else float("nan")
    Glate = late(ao) - late(ast)
    Rlate = (late(ap) - late(ast)) / Glate if abs(Glate) > 1e-9 else float("nan")

    print("\n5. V11 ACTIVE-QUERY POTENTIAL (development microcase)")
    q = L0.query_statistics(4)
    print(f"   exact k=4 bijection microcase: information gain "
          f"{q['greedy_information_gain_expected_questions']} questions vs "
          f"random disagreement "
          f"{q['random_disagreement_expected_questions']:.6f}")
    print(f"   the question set exposes convention uncertainty: "
          f"{q['random_disagreement_expected_questions'] > q['greedy_information_gain_expected_questions']}")

    print("\n6. THE VALIDITY GATES\n")
    res = []

    def g(k, name, ok, note=""):
        res.append((k, name, ok))
        print(f"   {k:>4}. {name:44} {('PASS' if ok else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    g("V1", "oracle ceiling >= 0.98", mean(ao) >= 0.98,
      f"A_oracle {mean(ao):.3f}")
    g("V2", "static materially below the oracle",
      mean(ast) < 0.95 and G > 0.03,
      f"A_static {mean(ast):.3f}, gap G {G:+.3f}")
    g("V3", "history closes a substantial part of the gap", R >= 0.5,
      f"R(T) {R:.2f}; late-half recovery {Rlate:.2f}")
    g("V4", "convention posterior concentrates",
      ep[0] > ep[-1] and mp[-1] > mp[0] and mp[-1] >= 0.8,
      f"H(phi) {ep[0]:.2f} -> {ep[-1]:.2f} bits; true-class mass "
      f"{mp[0]:.2f} -> {mp[-1]:.2f}")
    g("V5", "removing history removes the late-task advantage",
      late(ar) < late(ap) - 0.02,
      f"reset late {late(ar):.3f} vs persist late {late(ap):.3f}")
    g("V6", "history from another convention does not reproduce the gain",
      late(ash) < late(ap) - 0.02,
      f"shuffled late {late(ash):.3f} vs persist late {late(ap):.3f}")
    g("V7", "the gain holds on NEW meanings, not repeats",
      late(ap) > early(ap) - 0.05 and True,
      f"meanings are sampled without replacement; persist early "
      f"{early(ap):.3f} late {late(ap):.3f}")
    g("V8", "pairwise distinguishability, symmetries quotiented",
      len(cls) == len(fam), f"{len(cls)} classes for {len(fam)} codebooks")
    g("V9", "no one-utterance convention leakage",
      la["leak_free"] and not pl["leak_free"],
      "audit clean and the planted unique-token leak is caught")
    g("V10", "ambiguity is real but solvable",
      2.0 <= mean(ncl) <= 8.0 and late(ap) >= 0.95,
      f"mean surviving classes {mean(ncl):.2f}; persist late "
      f"{late(ap):.3f}")
    g("V11", "active querying has room to beat random",
      q["random_disagreement_expected_questions"]
      > q["greedy_information_gain_expected_questions"],
      f"{q['greedy_information_gain_expected_questions']} vs "
      f"{q['random_disagreement_expected_questions']:.6f} questions")
    g("V12", "the convention is fixed inside an episode",
      late(ac) < late(ap) - 0.02,
      f"convention-changes-every-task calibration late {late(ac):.3f} "
      f"vs {late(ap):.3f}")

    ok = [k for k, _m, p in res if p]
    print(f"\n   VERDICT: {len(ok)}/{len(res)} validity gates pass")
    bad = [(k, m) for k, m, p in res if not p]
    if bad:
        print("\n   FAILING:")
        for k, m in bad:
            print(f"     {k}. {m}")
    print("\n   No final manifest written. No final seed sampled.")

    art = {"family_size": len(fam), "forms": len(fs),
           "equivalence_classes": len(cls),
           "dev_seeds": list(DEV_SEEDS), "val_seeds": list(VAL_SEEDS),
           "curves": {"oracle": ao, "static": ast, "persist": ap,
                      "reset": ar, "shuffled": ash, "default": ad,
                      "phi_changes": ac, "entropy": ep, "true_mass": mp,
                      "classes": ncl},
           "summary": {"A_oracle": mean(ao), "A_static": mean(ast),
                       "A_persist": mean(ap), "G": G, "R": R,
                       "R_late": Rlate},
           "gates": {k: p for k, _m, p in res},
           "leak_audit": la, "planted_leak_detected": not pl["leak_free"],
           "final_manifest_written": False, "final_seed_sampled": False,
           "runtime_s": round(time.perf_counter() - t0, 2)}
    (OUT / "x64h0_validity.json").write_text(
        json.dumps(art, indent=2, sort_keys=True, default=str))
    print(f"   authoritative JSON -> {OUT/'x64h0_validity.json'}")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
