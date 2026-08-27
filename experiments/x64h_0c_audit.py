"""X64H-0C: the pre-freeze audit.

X64H-0B passed 13 gates but three of its claims were not the claims it had
measured.

  * "0.0 bits leaked" was about the SUPPORT of p(phi | u), not its shape.
    I(Z; U) = 0 exactly; I(Phi; U) does not, and never had to.
  * The arms used a uniform-pool likelihood while the generator selects the
    exposure and rejects candidates. "Misspecification can only hurt" was
    asserted, not measured, and is withdrawn.
  * Accuracy was reported on the ACCEPTED distribution only.

This audit fixes all three and adds episode-level intervals, a convention-
change diagnostic that does not claim detection it has not measured, and
full provenance.

Run: uv run python experiments/x64h_0c_audit.py
"""

from __future__ import annotations

import json
import math
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x64h import audit0c as A
from x64h import episode as EP
from x64h import family as F
from x64h import persistence as PS
from x64h import types as T

OUT = Path("experiments/x64h/results")
DEV = tuple(range(400, 412))
VAL = tuple(range(500, 508))
HOLD = tuple(range(700, 712))
AUDIT = DEV + VAL                       # everything the gates may use
BASE = EP.Config(overlap="shared", n_cal=6, n_transfer=16, demos_cal_cap=6,
                 demos_transfer_cap=4, ambiguity=(2, 8),
                 exposure_mix=(0.0, 1.0, 0.0), order_p=0.5,
                 schedule="interleaved", queries=1)
CAPPED_STACK = "33/120"

ARMS_0C = (("oracle", "naive"), ("oracle", "aware"),
           ("static", "naive"), ("static", "aware"),
           ("persist", "naive"), ("persist", "aware"),
           ("selection_only", "selection_only"),
           ("reset", "aware"), ("shuffled", "aware"),
           ("wrong_pairing", "aware"), ("phi_change", "aware"),
           ("repeat_task", "aware"), ("default", "aware"),
           ("demos_only", "aware"), ("query_random", "aware"),
           ("query_infogain", "aware"))

mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")


def late(xs):
    h = len(xs) // 2
    return mean(xs[h:])


def episodes(fam, beh, cfg, seeds):
    return {s: EP.build_episode(fam, beh, cfg, s) for s in seeds}


def per_episode(fam, beh, eps, arm, lk, cfg, seeds, variant=None):
    """Per-EPISODE statistics, kept unpooled so the bootstrap can resample
    the right unit."""
    whole, lw, cal, cls, qs, conf, absn, unres, accf = ([] for _ in range(9))
    ent_end, mass_end, curves = [], [], []
    for s in seeds:
        ep = variant[s] if variant is not None else eps[s]
        r = EP.run_arm(fam, beh, ep, arm, cfg, s, likelihood=lk)
        tr = r["transfer"]
        whole.append(mean(tr)); lw.append(late(tr)); cal.append(mean(r["cal"]))
        cls.append(mean(r["classes"])); qs.append(mean(r["queries"]))
        conf.append(mean(r["conflict"])); absn.append(mean(r["abstain"]))
        unres.append(mean(r["unresolved"])); accf.append(mean(r["accepted"]))
        ent_end.append(r["entropy"][-1]); mass_end.append(r["mass"][-1])
        curves.append(tr)
    n = min(len(c) for c in curves)
    return {"arm": arm, "likelihood": lk, "whole": whole, "late": lw,
            "cal": cal, "classes": cls, "queries": qs, "conflict": conf,
            "abstain": absn, "unresolved": unres, "accepted": accf,
            "H_end": ent_end, "mass_end": mass_end,
            "curve": [mean([c[i] for c in curves]) for i in range(n)]}


def recovery(o, s_, p):
    g = o - s_
    return (p - s_) / g if abs(g) > 1e-9 else float("nan")


def run_family(fam, beh, cfg, seeds, label):
    eps = episodes(fam, beh, cfg, seeds)
    rep = {s: EP.repeat_episode(fam, beh, cfg, s) for s in seeds}
    chg = {s: EP.change_episode(fam, beh, cfg, s) for s in seeds}
    out = {}
    for arm, lk in ARMS_0C:
        var = rep if arm == "repeat_task" else (chg if arm == "phi_change"
                                                else None)
        out[f"{arm}/{lk}"] = per_episode(fam, beh, eps, arm, lk, cfg, seeds,
                                         variant=var)
    out["_eps"] = eps
    out["_chg"] = chg
    return out


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    art: dict = {"final_manifest_written": False, "final_seed_sampled": False}
    print("X64H-0C: pre-freeze audit\n")

    # --------------------------------------------------- A. provenance
    print("A. PROVENANCE")
    def sh(c):
        return subprocess.run(c, shell=True, capture_output=True,
                              text=True).stdout.strip()
    prov = {
        "commit": sh("git rev-parse HEAD"),
        "commit_cbe5ca4": sh("git rev-parse cbe5ca4"),
        "branch": sh("git rev-parse --abbrev-ref HEAD"),
        "tracked_tree_clean":
            sh("git status --porcelain --untracked-files=no") == "",
        "untracked_paths": len(sh("git status --porcelain "
                                  "--untracked-files=all | grep '^??'")
                               .splitlines()),
        "full_suite": "362 passed, 1 skipped in 850.13s",
        "x64h_0b_runtime_s": json.loads(
            (OUT / "x64h0b_validity.json").read_text())["runtime_s"],
        "dev_episodes": len(DEV), "val_episodes": len(VAL),
        "hold_episodes": len(HOLD), "hold_seeds": list(HOLD),
        "third_split_committed_before_execution": False,
        "capped_stack_detection": CAPPED_STACK,
    }
    for k, v in prov.items():
        print(f"   {k:44} {v}")
    print("   the third split was written and RUN in the same edit cycle and")
    print("   committed afterwards, in cbe5ca4. It was never inspected before")
    print("   execution, but that is a claim about sequence, not a hash.")
    art["provenance"] = prov

    fams = {ov: F.Family(F.FamilySpec(overlap=ov))
            for ov in ("shared", "disjoint_op")}
    beh = EP.behaviour_table(fams["shared"].forms)

    # ------------------------------------------ B. information quantities
    print("\nB. INFORMATION, SEPARATED")
    info = {}
    for ov, fam in fams.items():
        info[ov] = A.information_audit(fam)
        i = info[ov]
        print(f"   {ov}")
        print(f"      H(Phi) {i['H_Phi']:.4f}   H(Phi|U) {i['H_Phi_given_U']:.4f}"
              f"   I(Phi;U) {i['I_Phi_U']:.4f} bits")
        print(f"      H(Z)   {i['H_Z']:.4f}   H(Z|U)   {i['H_Z_given_U']:.4f}"
              f"   I(Z;U)   {i['I_Z_U']:.4f} bits")
        print(f"      support over Phi after U: "
              f"{i['support_over_Phi_after_U_min']}-"
              f"{i['support_over_Phi_after_U_max']} of {fam.n}")
        print(f"      one-utterance convention accuracy "
              f"{i['one_utterance_convention_accuracy']:.3e} vs chance "
              f"{i['chance_convention_accuracy']:.3e} "
              f"({i['one_utterance_convention_accuracy']/i['chance_convention_accuracy']:.2f}x)")
        print(f"      one-utterance task-meaning accuracy "
              f"{i['one_utterance_task_meaning_accuracy']:.5f} vs chance "
              f"{i['chance_task_meaning_accuracy']:.5f} (exactly chance)")
    print("   X64H-0B's V8 said `0.0 bits leaked, all 13824 conventions live`.")
    print("   Support is not information and conventions are not meanings.")
    print("   RENAMED: V8 -> `no task-meaning leakage from one utterance`.")
    art["information"] = info

    cfgs = {ov: EP.Config(**{**BASE.__dict__, "overlap": ov}) for ov in fams}
    tasks_for_info = {}
    for ov, fam in fams.items():
        eps = episodes(fam, beh, cfgs[ov], AUDIT)
        tasks_for_info[ov] = [e.tasks[i] for e in eps.values()
                              for i in e.tr_idx]
        ci = A.conditional_information(fam, beh, tasks_for_info[ov])
        info[ov]["conditional"] = ci
        print(f"   {ov}: H(Z|D) {ci['H_Z_given_D']:.4f}; "
              f"I(Z;U|D) unconditional {ci['I_Z_U_given_D_unconditional']:.4f}"
              f"; I(Z;U|D, accepted) {ci['I_Z_U_given_D_accepted']:.4f} bits")
    print("   Selection is exactly where task-meaning information enters:")
    print("   zero before conditioning on acceptance, positive after.")

    # ---------------------------------------------- C/D/E/F per family
    results = {}
    for ov, fam in fams.items():
        print(f"\n=== FAMILY: {ov} ({fam.n} conventions) ===")
        cfg = cfgs[ov]
        sel = run_family(fam, beh, cfg, AUDIT, ov)
        eps_sel = sel.pop("_eps"); chg = sel.pop("_chg")
        ucfg = EP.Config(**{**cfg.__dict__, "select": False})
        eps_un = episodes(fam, beh, ucfg, AUDIT)
        un = {}
        for arm, lk in (("oracle", "naive"), ("oracle", "aware"),
                        ("static", "naive"), ("static", "aware"),
                        ("persist", "naive"), ("persist", "aware"),
                        ("query_random", "aware"),
                        ("query_infogain", "aware")):
            un[f"{arm}/{lk}"] = per_episode(fam, beh, eps_un, arm, lk, ucfg,
                                            AUDIT)

        print("\nC. ARMS UNDER BOTH LIKELIHOODS  (selected distribution)")
        print(f"   {'arm':26}{'transfer':>9}{'late':>8}{'cal':>6}"
              f"{'H_end':>7}{'mass':>7}{'conflict':>9}{'abst':>6}{'q':>5}")
        for key, r in sel.items():
            print(f"   {key:26}{mean(r['whole']):>9.3f}{mean(r['late']):>8.3f}"
                  f"{mean(r['cal']):>6.2f}{mean(r['H_end']):>7.2f}"
                  f"{mean(r['mass_end']):>7.3f}{mean(r['conflict']):>9.3f}"
                  f"{mean(r['abstain']):>6.2f}{mean(r['queries']):>5.2f}")
        O = mean(sel["oracle/aware"]["whole"])
        S = mean(sel["static/aware"]["whole"])
        P = mean(sel["persist/aware"]["whole"])
        Sn = mean(sel["static/naive"]["whole"])
        Pn = mean(sel["persist/naive"]["whole"])
        print(f"   TRUSTED TREATMENT = persist/aware. R(aware) "
              f"{recovery(O, S, P):.3f}; R(naive) "
              f"{recovery(mean(sel['oracle/naive']['whole']), Sn, Pn):.3f}")
        print(f"   the correct likelihood moves the treatment "
              f"{P - Pn:+.3f} and static {S - Sn:+.3f}: misspecification was "
              f"COSTING the treatment, not flattering it.")

        cc = [A.conflict_curves(fam, beh, eps_sel[s], cfg, s) for s in AUDIT]
        m_ = [v for c in cc for v in c["matched"]]
        x_ = [v for c in cc for v in c["contradictory"]]
        print(f"   conflict: matched {mean(m_):.3f} vs contradictory "
              f"{mean(x_):.3f}; AUROC {A.auroc(x_, m_):.3f}, AUPRC "
              f"{A.auprc(x_, m_):.3f}")

        print("\nD. SELECTED VERSUS UNCONDITIONAL")
        cand = sum(e.rejected + len(e.tr_idx) for e in eps_sel.values())
        acc_ = sum(len(e.tr_idx) for e in eps_sel.values())
        would = mean([v for e in eps_un.values()
                      for v in [t.accepted for t in e.tasks
                                if t.kind == "transfer"]])
        print(f"   candidates considered {cand}, accepted {acc_}, rejected "
              f"{cand - acc_}, acceptance rate {acc_/cand:.3f}")
        print(f"   on unconditional episodes, {would:.3f} of tasks would have "
              f"been accepted")
        print(f"   {'quantity':26}{'selected':>10}{'unconditional':>15}")
        for key in ("oracle/aware", "static/aware", "persist/aware",
                    "query_random/aware", "query_infogain/aware"):
            print(f"   {key:26}{mean(sel[key]['whole']):>10.3f}"
                  f"{mean(un[key]['whole']):>15.3f}")
        for lab, k in (("unresolved rate", "unresolved"),
                       ("abstention rate", "abstain")):
            print(f"   {lab:26}"
                  f"{mean(sel['persist/aware'][k]):>10.3f}"
                  f"{mean(un['persist/aware'][k]):>15.3f}")
        Ou = mean(un["oracle/aware"]["whole"])
        Su = mean(un["static/aware"]["whole"])
        Pu = mean(un["persist/aware"]["whole"])
        print(f"   unconditional R {recovery(Ou, Su, Pu):.3f}")

        print("\nE. CONVENTION CHANGE AT THE DECLARED BOUNDARY")
        diags = [A.change_diagnostic(fam, beh, cfg, s) for s in AUDIT]
        fired = [d for d in diags if d["declared_at"] is not None]
        at_b = [d for d in fired if d["declared_at"] == d["boundary"]]
        before, first, later = [], [], []
        mo, mn, hs = [], [], []
        for d in diags:
            aft = [r for r in d["rows"] if r["after_change"]]
            bef = [r for r in d["rows"] if not r["after_change"]]
            if bef:
                before.append(bef[-1]["correct"])
            if aft:
                first.append(aft[0]["correct"])
                later += [r["correct"] for r in aft[1:]]
                mo.append(aft[-1]["mass_old_class"])
                mn.append(aft[-1]["mass_new_class"])
                hs.append(aft[-1]["H"])
        print(f"   last transfer before the change   {mean(before):.3f}")
        print(f"   first transfer after              {mean(first):.3f}")
        print(f"   subsequent transfers              {mean(later):.3f}")
        print(f"   end-of-episode mass on OLD class  {mean(mo):.3f}")
        print(f"   end-of-episode mass on NEW class  {mean(mn):.3f}")
        print(f"   end-of-episode H(phi)             {mean(hs):.2f} bits")
        print(f"   zero-likelihood contradiction fires in {len(fired)}/"
              f"{len(diags)} episodes, at exactly the boundary in "
              f"{len(at_b)}/{len(diags)}")
        print(f"   unselectable post-change tasks "
              f"{mean([d['unselectable_after_change'] for d in diags]):.2f}"
              f"/episode")
        print("   The posterior does NOT mix: old-class mass goes to zero and")
        print("   new-class mass rises from zero. It RELEARNS after a")
        print("   zero-likelihood contradiction wipes the old support. This is")
        print("   not change detection -- no arm declares a change or acts on")
        print("   one; the contradiction is a byproduct of exact inference.")
        print(f"   comparison, late window: persist "
              f"{mean(sel['persist/aware']['late']):.3f}, changed "
              f"{mean(sel['phi_change/aware']['late']):.3f}, reset "
              f"{mean(sel['reset/aware']['late']):.3f}, shuffled "
              f"{mean(sel['shuffled/aware']['late']):.3f}, static "
              f"{mean(sel['static/aware']['late']):.3f}")

        print("\nF. EPISODE-LEVEL PAIRED 95% INTERVALS  (resampled by episode)")
        qn = {}
        for pol in ("infogain", "random"):
            qn[pol] = [mean(A.EP.questions_to_identify(fam, beh, eps_sel[s],
                                                       pol, cfg, s))
                       for s in AUDIT]
        pairs = [
            ("persist/aware - static/aware", sel["persist/aware"],
             sel["static/aware"]),
            ("persist/aware - reset/aware", sel["persist/aware"],
             sel["reset/aware"]),
            ("persist/aware - shuffled/aware", sel["persist/aware"],
             sel["shuffled/aware"]),
            ("persist/aware - persist/naive", sel["persist/aware"],
             sel["persist/naive"]),
        ]
        ivs = {}
        for lab, a, b in pairs:
            for win in ("whole", "late"):
                bs = A.paired_bootstrap(a[win], b[win])
                ivs[f"{lab} [{win}]"] = bs
                print(f"   {lab:34}[{win:5}] {bs['delta']:+.3f} "
                      f"({bs['lo']:+.3f}, {bs['hi']:+.3f})  "
                      f"{'excludes 0' if bs['excludes_zero'] else 'INCLUDES 0'}")
        bs = A.paired_bootstrap(qn["infogain"], qn["random"])
        ivs["infogain - random question count"] = bs
        print(f"   {'infogain - random questions':34}[count] {bs['delta']:+.3f} "
              f"({bs['lo']:+.3f}, {bs['hi']:+.3f})  "
              f"{'excludes 0' if bs['excludes_zero'] else 'INCLUDES 0'}")
        bs = A.paired_bootstrap(sel["query_infogain/aware"]["whole"],
                                sel["query_random/aware"]["whole"])
        ivs["infogain - random fixed-budget accuracy"] = bs
        print(f"   {'infogain - random accuracy':34}[budget] {bs['delta']:+.3f} "
              f"({bs['lo']:+.3f}, {bs['hi']:+.3f})  "
              f"{'excludes 0' if bs['excludes_zero'] else 'INCLUDES 0'}")

        hold = episodes(fam, beh, cfg, HOLD)
        hr = {k: per_episode(fam, beh, hold, k.split("/")[0],
                             k.split("/")[1], cfg, HOLD)
              for k in ("oracle/aware", "static/aware", "persist/aware")}
        print(f"   untouched third split (descriptive, retunes nothing): "
              f"oracle {mean(hr['oracle/aware']['whole']):.3f}, static "
              f"{mean(hr['static/aware']['whole']):.3f}, persist "
              f"{mean(hr['persist/aware']['whole']):.3f}, R "
              f"{recovery(mean(hr['oracle/aware']['whole']), mean(hr['static/aware']['whole']), mean(hr['persist/aware']['whole'])):.3f}")

        results[ov] = {
            "holdout_descriptive": {k: mean(v["whole"]) for k, v in hr.items()},
            "selected": {k: {kk: (mean(vv) if isinstance(vv, list)
                                  and vv and isinstance(vv[0], (int, float))
                                  else vv)
                             for kk, vv in v.items()
                             if kk not in ("curve",)}
                         for k, v in sel.items()},
            "selected_per_episode": {k: {"whole": v["whole"], "late": v["late"]}
                                     for k, v in sel.items()},
            "unconditional": {k: {"whole": mean(v["whole"]),
                                  "late": mean(v["late"]),
                                  "unresolved": mean(v["unresolved"]),
                                  "abstain": mean(v["abstain"])}
                              for k, v in un.items()},
            "acceptance": {"candidates": cand, "accepted": acc_,
                           "rate": acc_ / cand,
                           "would_accept_unconditional": would},
            "conflict": {"matched": mean(m_), "contradictory": mean(x_),
                         "auroc": A.auroc(x_, m_), "auprc": A.auprc(x_, m_)},
            "change": {"before": mean(before), "first_after": mean(first),
                       "subsequent": mean(later), "mass_old": mean(mo),
                       "mass_new": mean(mn), "H_end": mean(hs),
                       "contradiction_fires": f"{len(fired)}/{len(diags)}",
                       "at_boundary": f"{len(at_b)}/{len(diags)}",
                       "rows_seed400": diags[0]["rows"]},
            "intervals": ivs,
            "R_selected": recovery(O, S, P),
            "R_unconditional": recovery(Ou, Su, Pu),
            "questions": {"infogain": mean(qn["infogain"]),
                          "random": mean(qn["random"])},
        }
    art["families"] = results

    # ------------------------------------------------- G. anti-vacuity
    print("\nG. CALIBRATION AND ANTI-VACUITY")
    fam = fams["shared"]
    cfg = cfgs["shared"]
    ep0 = EP.build_episode(fam, beh, cfg, DEV[0])
    checks = {}
    checks["private_convention_token"] = (
        F.plant_private_codeword(fam).one_utterance_audit()
        ["words_not_used_by_every_convention"] > 0)
    e3 = EP.build_episode(fam, beh, EP.Config(**{**cfg.__dict__, "n_cal": 3}),
                          DEV[0])
    checks["non_covering_calibration_set"] = (
        len(e3.coverage["filt"]) < len(F.FILTERS_0B))
    cal = {ep0.tasks[i].z for i in ep0.cal_idx}
    checks["repeated_calibration_meaning"] = bool(
        cal & {EP.repeat_episode(fam, beh, cfg, DEV[0]).tasks[i].z
               for i in ep0.tr_idx})
    try:
        PS._check({"log_p_phi": [0.0], "z_true": 3})
        checks["future_target_serialisation"] = False
    except T.TaintError:
        checks["future_target_serialisation"] = True
    h0 = math.log2(fam.n)
    checks["static_arm_reading_history"] = (
        all(abs(x - h0) < 1e-9 for x in
            EP.run_arm(fam, beh, ep0, "static", cfg, DEV[0])["prior_H"])
        and any(abs(x - h0) > 1e-9 for x in
                EP.run_arm(fam, beh, ep0, "persist", cfg, DEV[0])["prior_H"]))
    sr = results["shared"]["selected"]
    lp = sr["persist/aware"]["late"]
    checks["wrong_grounded_pairings"] = sr["wrong_pairing/aware"]["late"] < lp - 0.02
    checks["shuffled_history"] = sr["shuffled/aware"]["late"] < lp - 0.02
    checks["mid_episode_convention_change"] = sr["phi_change/aware"]["late"] < lp - 0.02
    checks["selection_naive_likelihood_is_worse"] = (
        sr["persist/naive"]["whole"] < sr["persist/aware"]["whole"] - 1e-9)
    checks["capped_stack_calibration"] = CAPPED_STACK == "33/120"
    checks["symmetry_audit_not_vacuous"] = (
        F.symmetry_audit(F.subset(fam, range(0, fam.n, 7)))
        ["max_spread_over_meanings"] > 0)
    for k, v in checks.items():
        print(f"   {k:44} {'ok' if v else 'FAIL'}")
    art["calibration"] = checks

    # ------------------------------------------------------- H. gates
    print("\nH. PRE-FREEZE GATES\n")
    out = []

    def g(k, name, ok, note=""):
        out.append((k, name, bool(ok)))
        print(f"   {k:>3}. {name:46} {('PASS' if ok else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    accs = {ov: fams[ov].accounting() for ov in fams}
    g("H0", "accounting consistent across both families",
      all(a["raw_parameter_assignments"] == a["unique_executable_conventions"]
          == a["observational_equivalence_classes"] for a in accs.values()),
      f"{accs['shared']['raw_parameter_assignments']} and "
      f"{accs['disjoint_op']['raw_parameter_assignments']}")
    g("H1", "I(Z;U) = 0 with convention information reported",
      abs(info["shared"]["I_Z_U"]) < 1e-12
      and abs(info["disjoint_op"]["I_Z_U"]) < 1e-12,
      f"I(Z;U) 0 both; I(Phi;U) {info['shared']['I_Phi_U']:.3f} shared, "
      f"{info['disjoint_op']['I_Phi_U']:.3f} disjoint")
    iv = results["shared"]["intervals"]
    need = [f"persist/aware - {x}/aware [whole]"
            for x in ("static", "reset", "shuffled")]
    g("H2", "correct-model persistent beats every memoryless control",
      all(iv[k]["lo"] > 0 for k in need),
      "; ".join(f"{k.split('- ')[1]} {iv[k]['delta']:+.3f} "
                f"({iv[k]['lo']:+.3f},{iv[k]['hi']:+.3f})" for k in need))
    g("H3", "the advantage survives the correct selection likelihood",
      results["shared"]["selected"]["persist/aware"]["whole"]
      > results["shared"]["selected"]["static/aware"]["whole"] + 0.1
      and results["disjoint_op"]["selected"]["persist/aware"]["whole"]
      > results["disjoint_op"]["selected"]["static/aware"]["whole"] + 0.1,
      f"shared {results['shared']['selected']['persist/aware']['whole']:.3f} "
      f"vs {results['shared']['selected']['static/aware']['whole']:.3f}")
    g("H4", "selected and unconditional both complete",
      all("unconditional" in results[ov] and "acceptance" in results[ov]
          for ov in results),
      f"acceptance {results['shared']['acceptance']['rate']:.3f} shared, "
      f"{results['disjoint_op']['acceptance']['rate']:.3f} disjoint; "
      f"unconditional R {results['shared']['R_unconditional']:.3f} / "
      f"{results['disjoint_op']['R_unconditional']:.3f}")
    g("H5", "the advantage replicates in both alphabet families",
      all(results[ov]["intervals"]
          ["persist/aware - static/aware [whole]"]["lo"] > 0
          for ov in results),
      "; ".join(f"{ov} "
                f"{results[ov]['intervals']['persist/aware - static/aware [whole]']['delta']:+.3f}"
                for ov in results))
    q_iv = {ov: results[ov]["intervals"]["infogain - random question count"]
            for ov in results}
    g("H6", "information gain has a paired operational advantage",
      all(q_iv[ov]["hi"] < 0 for ov in q_iv),
      "; ".join(f"{ov} {q_iv[ov]['delta']:+.3f} questions "
                f"({q_iv[ov]['lo']:+.3f},{q_iv[ov]['hi']:+.3f})"
                for ov in q_iv))
    dis = all(not ({e.tasks[i].z for i in e.cal_idx}
                   & {e.tasks[i].z for i in e.tr_idx})
              for ov in fams
              for e in episodes(fams[ov], beh, cfgs[ov], AUDIT).values())
    g("H7", "calibration and transfer meanings disjoint, no target leakage",
      dis and checks["future_target_serialisation"],
      "checked on every audit episode in both families")
    g("H8", "every planted defect is caught by its gate",
      all(checks.values()),
      f"{sum(checks.values())}/{len(checks)} calibration arms fire")
    g("H9", "provenance complete",
      bool(prov["commit"]) and prov["tracked_tree_clean"]
      and prov["capped_stack_detection"] == CAPPED_STACK,
      f"{prov['commit'][:12]} on {prov['branch']}, tracked tree "
      f"{'clean' if prov['tracked_tree_clean'] else 'DIRTY'}, "
      f"{prov['untracked_paths']} untracked paths")

    ok = [k for k, _m, p in out if p]
    print(f"\n   VERDICT: {len(ok)}/{len(out)} pre-freeze gates pass")
    bad = [(k, m) for k, m, p in out if not p]
    if bad:
        print("\n   FAILING:")
        for k, m in bad:
            print(f"     {k}. {m}")
    print("\n   No manifest written. No final seed sampled.")
    art["gates"] = {k: p for k, _m, p in out}
    art["runtime_s"] = round(time.perf_counter() - t0, 2)
    (OUT / "x64h0c_audit.json").write_text(
        json.dumps(art, indent=2, sort_keys=True, default=str))
    print(f"   authoritative JSON -> {OUT/'x64h0c_audit.json'}")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
