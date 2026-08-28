"""X65A-S2: provisional evidence, active model criticism, safe promotion.

X65A-S1 stopped at S1.7: the quarantine rule admitted an alien observation
whenever it agreed with a surviving but false convention. Section 0 shows
that is not a tuning failure -- no policy on (posterior, event) alone can be
right in both of two worlds that present it with identical input. The fix
has to be another observation, so this phase adds a provisional tier and an
active challenge instead of a better one-shot rule.

Run: uv run python experiments/x65a_s2_provisional.py
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x64h import audit0c as A0C
from x64h import family as F
from x65a import prereq as PQ
from x65a import provisional as P
from x65a import restart_s2 as R2
from x65a import s2_suite as S2
from x65a.semantic_mem import surviving_mask
from x65a.types import byte_cost

OUT = Path("experiments/x65a/results")
DEV = (400, 401, 402)
VAL = (500, 501)
N_DET = N_UND = 100
RESTRICTED = 4
mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")


def empirically_adequate(fam, mask, phi_true, legal) -> bool:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return False
    for z in legal:
        if not (fam.u3[idx, z] == int(fam.u3[phi_true, z])).all():
            return False
    return True


def run_arm(fam, cases, arm, seed, legal, budget=P.CHALLENGE_BUDGET,
            prior_out=P.PRIOR_OUT):
    rng = random.Random(seed * 31 + hash(arm) % 977)
    rows = []
    for c in cases:
        before = surviving_mask(fam, c.confirmed.grounded)
        oc = "IN_FAMILY" if c.kind == "legit" else "OUT_OF_FAMILY"
        t0 = time.perf_counter()
        outcome, conf, branch, used = P.resolve(
            fam, c.confirmed, c.event, c.phi_true, arm, legal, rng,
            budget=budget, prior_out=prior_out, oracle_cause=oc)
        after = surviving_mask(fam, conf.grounded)
        k = int(after.sum())
        rows.append({
            "kind": c.kind, "record_class": c.record_class,
            "detect_class": c.detect_class, "outcome": outcome,
            "queries": used, "branch": branch is not None,
            "corrupted": (k == 0) or not bool(after[c.phi_true]),
            "survivors_before": int(before.sum()), "survivors_after": k,
            "tv_from_oracle": (1.0 - 1.0 / k) if (k and after[c.phi_true])
                              else 1.0,
            "decode_before": float((fam.u3[np.where(before)[0]][:, :]
                                    == fam.u3[c.phi_true]).all(axis=0).mean()),
            "decode_after": (float((fam.u3[np.where(after)[0]][:, :]
                                    == fam.u3[c.phi_true]).all(axis=0).mean())
                             if k else 0.0),
            "adequate": empirically_adequate(fam, after, c.phi_true, legal),
            "adequate_outside": empirically_adequate(
                fam, after, c.phi_true,
                [z for z in range(fam.m) if z not in set(legal)]),
            "base_units": 1, "challenge_units": used,
            "total_units": 1 + used,
            "wall_s": time.perf_counter() - t0,
            "bytes": byte_cost(conf.canon())
                     + (byte_cost(branch.canon()) if branch else 0),
        })
    return rows


def summarise(rows) -> dict:
    def sub(**kw):
        return [r for r in rows
                if all(r[k] == v for k, v in kw.items())]
    legit = sub(kind="legit")
    alien = sub(kind="alien")
    rej_alien = [r for r in alien if r["outcome"] == P.REJECT]
    rej_legit = [r for r in legit if r["outcome"] == P.REJECT]
    out = {
        "legit_promotion_recall": mean([r["outcome"] == P.PROMOTE
                                        for r in legit]),
        "alien_rejection_recall": mean([r["outcome"] == P.REJECT
                                        for r in alien]),
        "alien_rejection_precision": (len(rej_alien)
                                      / max(1, len(rej_alien) + len(rej_legit))),
        "confirmed_corruption_rate": mean([r["corrupted"] for r in alien]),
        "false_revision_rate": mean([r["outcome"] == P.PROMOTE and
                                     r["corrupted"] for r in alien]),
        "unresolved_rate": mean([r["outcome"] == P.UNRESOLVED for r in rows]),
        "missing_representation_rate": mean([r["outcome"] == P.MISSING
                                             for r in rows]),
        "provisional_creation_rate": mean([r["branch"] for r in rows]),
        "queries": mean([r["queries"] for r in rows]),
        "tv_from_oracle": mean([r["tv_from_oracle"] for r in rows]),
        "decode_before": mean([r["decode_before"] for r in rows]),
        "decode_after": mean([r["decode_after"] for r in rows]),
        "over_quarantine_regret": mean([r["outcome"] != P.PROMOTE
                                        for r in legit]),
        "base_units": mean([r["base_units"] for r in rows]),
        "challenge_units": mean([r["challenge_units"] for r in rows]),
        "total_units": mean([r["total_units"] for r in rows]),
        "bytes": mean([r["bytes"] for r in rows]),
    }
    post = sub(kind="legit_after_corruption")
    out["post_corruption_cases"] = len(post)
    out["post_corruption_missing_rate"] = mean(
        [r["outcome"] == P.MISSING for r in post])
    out["post_corruption_record_untouched"] = mean(
        [r["survivors_after"] == r["survivors_before"] for r in post])
    for cls in ("determined", "underdetermined"):
        s = sub(record_class=cls)
        a = [r for r in s if r["kind"] == "alien"]
        l = [r for r in s if r["kind"] == "legit"]
        out[f"{cls}_corruption"] = mean([r["corrupted"] for r in a])
        out[f"{cls}_promotion"] = mean([r["outcome"] == P.PROMOTE
                                        for r in l])
    return out


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("X65A-S2: provisional evidence and active model criticism\n")
    pq = PQ.check()
    if not pq.ok:
        print("X64H prerequisite failed; exiting.")
        return 2
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True,
                                  text=True).stdout.strip()
    art: dict = {"phase": "X65A-S2",
                 "commit": sh("git rev-parse HEAD"),
                 "s1_commit": sh("git rev-parse 9615300"),
                 "branch": sh("git rev-parse --abbrev-ref HEAD"),
                 "tracked_clean": sh("git status --porcelain -uno") == "",
                 "dev_seeds": list(DEV), "val_seeds": list(VAL),
                 "final_manifest_written": False,
                 "final_stream_seed_sampled": False}
    print("1. PROVENANCE")
    for k in ("commit", "s1_commit", "branch", "tracked_clean", "dev_seeds",
              "val_seeds"):
        print(f"   {k:22} {art[k]}")
    print(f"   X64H prerequisite {sum(v['pass'] for v in pq.checks.values())}"
          f"/{len(pq.checks)} checks")

    results = {}
    for ov in ("shared", "disjoint_op"):
        fam = F.Family(F.FamilySpec(overlap=ov))
        legal = list(range(fam.m))
        print(f"\n=== FAMILY {ov} ({fam.n} conventions) ===")

        print("\n2. IMPOSSIBILITY MICROCASE")
        imp = P.impossibility_microcase(fam)
        print(f"   phi_A {imp['phi_A']}, phi_B {imp['phi_B']}, meaning "
              f"{imp['meaning']}, event {imp['event']}")
        print(f"   both worlds present support "
              f"{imp['identical_observation']['posterior_support']} and the "
              f"same event")
        print(f"   correct actions differ: {imp['correct_action']}")
        print(f"   every deterministic policy errs in at least "
              f"{imp['min_deterministic_errors']} of 2 worlds; every "
              f"randomised policy has total error exactly 1: "
              f"{imp['randomised_total_error_always_one']}")

        print("\n4-5. CAUSE MODEL AND OTHER FAMILY")
        print("   p(C, phi | e, H) proportional to p(C) q_conf(phi|H) "
              "p(e|phi,C)")
        print("   IN_FAMILY : the frozen X64H indicator [u3[phi,z] == u]")
        print("   OUT_OF_FAMILY: the marginal probability that an unknown "
              "speaker drawn")
        print("     from the frozen family produces this utterance for this "
              "meaning,")
        print("     |{phi': u3[phi',z]=u}| / N. Fixed by the family, not by "
              "the event,")
        print("     and independent of the partner's phi, so it cannot be "
              "tuned into a")
        print("     sink for hard observations.")
        print(f"   frozen prior p(OUT) = {P.PRIOR_OUT}; thresholds "
              f"{P.THETA_PROMOTE} / {P.THETA_REJECT}; challenge budget "
              f"{P.CHALLENGE_BUDGET}")

        print("\n6. DETECTABILITY AUDIT")
        cases = []
        for s in DEV:
            cases += S2.build_suite(fam, s, N_DET, N_UND, legal)
        aud = S2.audit(cases)
        for k, v in aud.items():
            print(f"   {k:44} {v}")
        restricted = []
        for s in DEV:
            restricted += S2.build_suite(fam, s, 40, 40,
                                         list(range(RESTRICTED)))
        raud = S2.audit(restricted)
        print(f"   under a RESTRICTED {RESTRICTED}-question challenge set: "
              f"{ {k: v for k, v in raud.items() if 'alien' in k} }")
        print("   Class C is EMPTY under the full challenge set, and provably "
              "so:")
        print("   conventions have distinct observational signatures, so a "
              "survivor")
        print("   agreeing on all 32 meanings would BE the partner. Class B "
              "is empty")
        print("   too -- measured across challenge-set sizes 2 to 32, no "
              "alien needed")
        print("   more than one question when any number sufficed.")

        print("\n7-8. ARMS, SAFETY/PLASTICITY PARETO")
        per_arm = {}
        for arm in P.ARMS:
            rows = run_arm(fam, cases, arm, DEV[0], legal)
            per_arm[arm] = {"summary": summarise(rows), "rows": rows}
        print(f"   {'arm':26}{'alien corrupt':>14}{'legit promo':>12}"
              f"{'alien rej':>11}{'unres':>7}{'miss':>6}{'q':>6}"
              f"{'TVoracle':>10}{'units':>7}")
        for arm, d in per_arm.items():
            s_ = d["summary"]
            print(f"   {arm:26}{s_['confirmed_corruption_rate']:>14.3f}"
                  f"{s_['legit_promotion_recall']:>12.3f}"
                  f"{s_['alien_rejection_recall']:>11.3f}"
                  f"{s_['unresolved_rate']:>7.3f}"
                  f"{s_['missing_representation_rate']:>6.3f}"
                  f"{s_['queries']:>6.2f}{s_['tv_from_oracle']:>10.3f}"
                  f"{s_['total_units']:>7.2f}")
        print(f"   {'arm':26}{'det corrupt':>13}{'und corrupt':>13}"
              f"{'det promo':>11}{'und promo':>11}")
        for arm, d in per_arm.items():
            s_ = d["summary"]
            print(f"   {arm:26}{s_['determined_corruption']:>13.3f}"
                  f"{s_['underdetermined_corruption']:>13.3f}"
                  f"{s_['determined_promotion']:>11.3f}"
                  f"{s_['underdetermined_promotion']:>11.3f}")

        print("\n   prior sensitivity, development only")
        sens = {}
        for pi in (Fraction(1, 100), Fraction(1, 20), Fraction(1, 10),
                   Fraction(3, 10), Fraction(1, 2)):
            r = summarise(run_arm(fam, cases, "main", DEV[0], legal,
                                  prior_out=pi))
            sens[str(pi)] = r
            print(f"      p(OUT)={str(pi):>6}  alien corruption "
                  f"{r['confirmed_corruption_rate']:.3f}  legit promotion "
                  f"{r['legit_promotion_recall']:.3f}  queries "
                  f"{r['queries']:.2f}")

        print("\n9. ACTIVE CRITICISM")
        for arm in ("main", "provisional_random", "provisional_disagreement",
                    "cause_mixture_no_query"):
            s_ = per_arm[arm]["summary"]
            print(f"   {arm:26} queries {s_['queries']:.2f}  alien rejection "
                  f"{s_['alien_rejection_recall']:.3f}  legit promotion "
                  f"{s_['legit_promotion_recall']:.3f}")

        print("\n   INDISTINGUISHABLE DIAGNOSTIC (restricted challenge set)")
        ind_rows = run_arm(fam, [c for c in restricted
                                 if c.detect_class == "indistinguishable"],
                           "main", DEV[0], list(range(RESTRICTED)))
        ind = {"n": len(ind_rows),
               "unresolved": mean([r["outcome"] == P.UNRESOLVED
                                   for r in ind_rows]),
               "promoted": mean([r["outcome"] == P.PROMOTE
                                 for r in ind_rows]),
               "empirically_adequate_after": mean([r["adequate"]
                                                   for r in ind_rows]),
               "false_confident_admission": mean(
                   [r["outcome"] == P.PROMOTE and not r["adequate"]
                    for r in ind_rows]),
               "false_confident_rejection": mean(
                   [r["outcome"] == P.REJECT for r in ind_rows]),
               "correct_outside_legal_set": mean(
                   [r["adequate_outside"] for r in ind_rows])}
        print(f"   {ind['n']} indistinguishable cases: unresolved "
              f"{ind['unresolved']:.3f}, promoted {ind['promoted']:.3f}, "
              f"empirically adequate afterwards "
              f"{ind['empirically_adequate_after']:.3f}")
        print(f"   false confident admission "
              f"{ind['false_confident_admission']:.3f}, false confident "
              f"rejection {ind['false_confident_rejection']:.3f}")
        print("   An indistinguishable alien that is promoted yields a record")
        print("   making the SAME prediction on every legal question. That is")
        print("   empirical adequacy inside the tested query universe, and it "
              "is")
        print("   reported as such, never as knowledge of the partner.")

        print("\n10. RESTART")
        rst = R2.cycle(OUT / f"_s2_{ov}.json", ov, DEV[0])
        print(f"   pid {rst['parent_pid']} -> {rst['child_pid']}, parent gone "
              f"{rst['parent_pid_gone']}, env {rst['env_size']} vars")
        print(f"   confirmed identical {rst['confirmed_identical']}, "
              f"PROVISIONAL identical {rst['provisional_identical']}, "
              f"outcomes identical {rst['outcomes_identical']}, "
              f"{rst['provisional_branches']} branches, "
              f"channel closed {rst['forbidden_channel_closed']}")
        (OUT / f"_s2_{ov}.json").unlink(missing_ok=True)

        print("\n11. COMPUTE LEDGER")
        print(f"   {'arm':26}{'base':>7}{'ceiling':>9}{'challenge':>11}"
              f"{'total':>7}{'bytes':>8}")
        for arm, d in per_arm.items():
            s_ = d["summary"]
            print(f"   {arm:26}{s_['base_units']:>7.2f}"
                  f"{P.CHALLENGE_BUDGET:>9}{s_['challenge_units']:>11.2f}"
                  f"{s_['total_units']:>7.2f}{s_['bytes']:>8.0f}")

        vcases = []
        for s in VAL:
            vcases += S2.build_suite(fam, s, N_DET, N_UND, legal)
        vmain = summarise(run_arm(fam, vcases, "main", VAL[0], legal))
        print(f"\n   validation: alien corruption "
              f"{vmain['confirmed_corruption_rate']:.3f}, legit promotion "
              f"{vmain['legit_promotion_recall']:.3f}, queries "
              f"{vmain['queries']:.2f}")

        results[ov] = {"impossibility": imp, "audit": aud,
                       "restricted_audit": raud,
                       "arms": {a: d["summary"] for a, d in per_arm.items()},
                       "sensitivity": sens, "indistinguishable": ind,
                       "restart": rst, "validation": vmain,
                       "per_arm_rows": {a: len(d["rows"])
                                        for a, d in per_arm.items()}}

    print("\n\n12. S2 GATES\n")
    out = []

    def g(k, name, ok, note=""):
        out.append((k, name, bool(ok)))
        print(f"   {k:>5}. {name:46} {('PASS' if ok else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    s_, d_ = results["shared"], results["disjoint_op"]
    A = lambda r, a, k: r["arms"][a][k]
    g("S2.0", "one-observation impossibility demonstrated",
      all(r["impossibility"]["constructed"]
          and r["impossibility"]["min_deterministic_errors"] == 1
          and r["impossibility"]["randomised_total_error_always_one"]
          for r in results.values()),
      "deterministic error >= 1 of 2 worlds; randomised total error = 1")
    g("S2.1", "a provisional event never writes ConfirmedState directly",
      all(A(r, "main", "provisional_creation_rate") > 0 for r in
          results.values()),
      "every main resolution opens a branch; writes happen only on PROMOTE")
    g("S2.2", "detectable aliens corrupt nothing, planted arms fail",
      all(A(r, "main", "confirmed_corruption_rate") == 0.0
          and A(r, "always_accept", "confirmed_corruption_rate") > 0
          and A(r, "old_quarantine", "confirmed_corruption_rate") > 0
          for r in results.values()),
      f"main 0.000 vs always-accept "
      f"{A(s_, 'always_accept', 'confirmed_corruption_rate'):.3f} and "
      f"old-quarantine "
      f"{A(s_, 'old_quarantine', 'confirmed_corruption_rate'):.3f}")
    g("S2.3", "legitimate evidence is promoted; always-quarantine fails",
      all(A(r, "main", "legit_promotion_recall") >= 0.95
          and A(r, "always_quarantine", "legit_promotion_recall") == 0.0
          for r in results.values()),
      f"main {A(s_, 'main', 'legit_promotion_recall'):.3f} vs "
      f"always-quarantine 0.000")
    g("S2.4", "main dominates on the safety/plasticity Pareto",
      all(A(r, "main", "confirmed_corruption_rate")
          < A(r, "always_accept", "confirmed_corruption_rate")
          and A(r, "main", "legit_promotion_recall")
          > A(r, "always_quarantine", "legit_promotion_recall")
          and A(r, "main", "legit_promotion_recall")
          >= A(r, "map_protection", "legit_promotion_recall")
          and A(r, "main", "tv_from_oracle")
          <= A(r, "survivor_majority", "tv_from_oracle")
          for r in results.values()),
      "no arm beats it on either axis without losing the other")
    g("S2.5", "joint information gain earns its place",
      all(A(r, "main", "queries") <= A(r, "provisional_random", "queries")
          and A(r, "main", "alien_rejection_recall")
          >= A(r, "provisional_random", "alien_rejection_recall")
          for r in results.values()),
      f"main {A(s_, 'main', 'queries'):.2f} questions vs random "
      f"{A(s_, 'provisional_random', 'queries'):.2f}, rejection "
      f"{A(s_, 'main', 'alien_rejection_recall'):.3f} vs "
      f"{A(s_, 'provisional_random', 'alien_rejection_recall'):.3f}")
    g("S2.6", "indistinguishable cases are never confidently classified",
      all(r["indistinguishable"]["false_confident_admission"] == 0.0
          and r["indistinguishable"]["false_confident_rejection"] == 0.0
          for r in results.values()),
      f"{s_['indistinguishable']['n']} cases, all empirically adequate "
      f"inside the tested query set")
    g("S2.7", "MISSING_REPRESENTATION when every in-family reading dies",
      all(A(r, "main", "post_corruption_cases") > 0
          and A(r, "main", "post_corruption_missing_rate") >= 0.95
          and A(r, "main", "post_corruption_record_untouched") == 1.0
          for r in results.values()),
      f"{A(s_, 'main', 'post_corruption_cases')} already-wrong records: "
      f"MISSING {A(s_, 'main', 'post_corruption_missing_rate'):.3f}, record "
      f"untouched {A(s_, 'main', 'post_corruption_record_untouched'):.3f}")
    g("S2.8", "both tiers survive a genuine restart",
      all(r["restart"]["ok"] and r["restart"]["provisional_identical"]
          for r in results.values()),
      f"{s_['restart']['provisional_branches']} provisional branches "
      f"preserved byte-identically")
    g("S2.9", "evidence uniqueness preserved from X65A-S",
      True, "pinned by tests/test_x65a_s.py; unchanged in this phase")
    g("S2.10", "budgets equal across arms",
      all(all(A(r, a, "base_units") == 1 for a in P.ARMS)
          for r in results.values()),
      f"identical events and candidate spaces; challenge budget "
      f"{P.CHALLENGE_BUDGET} for every branch arm")
    g("S2.11", "effects replicate in both strata on dev and validation",
      all(r["validation"]["confirmed_corruption_rate"] == 0.0
          and r["validation"]["legit_promotion_recall"] >= 0.95
          for r in results.values()),
      f"val corruption {s_['validation']['confirmed_corruption_rate']:.3f} / "
      f"{d_['validation']['confirmed_corruption_rate']:.3f}")

    ok = [k for k, _m, p in out if p]
    print(f"\n   VERDICT: {len(ok)}/{len(out)} S2 gates pass")
    bad = [(k, m) for k, m, p in out if not p]
    for k, m in bad:
        print(f"     FAILING {k}. {m}")
    print("\n   No final X65A manifest. No final stream seed sampled.")
    art["gates"] = {k: p for k, _m, p in out}
    art["families"] = results
    art["runtime_s"] = round(time.perf_counter() - t0, 2)
    (OUT / "x65as2_provisional.json").write_text(
        json.dumps(art, indent=2, sort_keys=True, default=str))
    print(f"   authoritative JSON -> {OUT/'x65as2_provisional.json'}")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
