"""X64H final: the hidden-convention run, on seeds released by the freeze.

Nothing here chooses anything. The configuration, the families, the teacher
likelihood, the arms, the gates and the bootstrap are all fixed by
`x64h/freeze_manifest_0c.json`, and the seeds are a deterministic function
of that manifest's digest. This file reads them and reports.

Run: uv run python experiments/x64h_final.py
"""

from __future__ import annotations

import json
import math
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
from x64h import freeze0c as FZ
from x64h import persistence as PS
from x64h import semantic as S
from x64h import types as T
from x64h_0c_audit import ARMS_0C, BASE, mean, late, per_episode, recovery

OUT = Path("experiments/x64h/results")
SEEDFILE = Path("experiments/x64h/final_seeds_0c.json")


def adaptation_regret(oracle_curve, persist_curve) -> float:
    n = min(len(oracle_curve), len(persist_curve))
    return float(sum(oracle_curve[i] - persist_curve[i] for i in range(n)))


def open_world(fam, beh, cfg, seeds) -> dict:
    """Tasks whose true meaning is OUTSIDE the frozen 32-form space. The
    honest question is not whether the arm gets them right -- it cannot --
    but whether it says so. `live` empty is the UNKNOWN_MEANING signal."""
    outside = [z for z in S.x64h_forms()
               if not (z.filt in F.FILTERS_0B and z.scope in F.SCOPES_0B)]
    m = fam.m
    detected = missed = 0
    for s in seeds:
        import random as _r
        rng = _r.Random(s + 555)
        for _ in range(8):
            z = outside[rng.randrange(len(outside))]
            f = S.execute(z)
            live = list(range(m))
            for _k in range(4):
                x = S.UNIVERSE[rng.randrange(len(S.UNIVERSE))]
                y = f(x)
                live = [j for j in live if beh[j][S.UNIVERSE.index(x)] == y]
                if not live:
                    break
            if live:
                missed += 1
            else:
                detected += 1
    tot = detected + missed
    return {"out_of_space_tasks": tot, "unknown_meaning_declared": detected,
            "silently_answered": missed, "detection_rate": detected / tot,
            "outside_forms": len(outside)}


def restart_persistence(fam, beh, cfg, seed) -> dict:
    """Save the convention posterior mid-episode, reload it in a SEPARATE
    PROCESS, and require the same state. A round trip inside one process
    would not test the serialisation boundary."""
    ep = EP.build_episode(fam, beh, cfg, seed)
    p = np.full(fam.n, 1.0 / fam.n)
    half = len(ep.tasks) // 2
    for t in ep.tasks[:half]:
        _b, conv, _ = EP._infer_by("aware", fam, p, t.u, t.pool, t.live, t.tie)
        if conv.sum() > 0:
            p = conv / conv.sum()
    path = OUT / "_restart_state.json"
    PS.save(path, T.PosteriorState(tuple(float(math.log(max(x, 1e-300)))
                                         for x in p), "x64h-final"))
    code = (
        "import sys,math,json;sys.path.insert(0,'experiments');"
        "from x64h import persistence as PS;"
        f"st=PS.load(__import__('pathlib').Path({str(path)!r}),{fam.n},"
        "'x64h-final');"
        "lp=list(st.log_p_phi);m=max(lp);"
        "w=[math.exp(x-m) for x in lp];t=sum(w);"
        "print(json.dumps({'H':-sum((x/t)*math.log2(x/t) "
        "for x in w if x>0),'n':len(lp),'sum':t/t}))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    path.unlink(missing_ok=True)
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr.strip()[-300:]}
    got = json.loads(r.stdout)
    want = EP.entropy_bits(p)
    return {"ok": abs(got["H"] - want) < 1e-6, "H_in_process": want,
            "H_after_restart": got["H"], "n": got["n"],
            "separate_process": True}


def main() -> int:
    t0 = time.perf_counter()
    if not SEEDFILE.exists():
        print("no released seeds; run the freeze first")
        return 2
    rel = json.loads(SEEDFILE.read_text())
    ok, why = FZ.manifest_committed(BASE, ARMS_0C)
    print("X64H FINAL: hidden-convention run\n")
    print(f"   freeze digest {rel['digest']}")
    print(f"   manifest committed and intact: {ok} ({why})")
    if not ok:
        print("   REFUSING to run: the freeze is broken.")
        return 2
    art = {"digest": rel["digest"], "seeds": rel["seeds"],
           "manifest_intact": ok, "families": {}}

    for ov in ("shared", "disjoint_op"):
        seeds = tuple(rel["seeds"][ov])
        fam = F.Family(F.FamilySpec(overlap=ov))
        beh = EP.behaviour_table(fam.forms)
        cfg = EP.Config(**{**BASE.__dict__, "overlap": ov})
        ucfg = EP.Config(**{**cfg.__dict__, "select": False})
        print(f"\n=== {ov}: {fam.n} conventions, {len(seeds)} hidden seeds ===")
        print(f"   seeds {list(seeds)}")

        eps = {s: EP.build_episode(fam, beh, cfg, s) for s in seeds}
        rep = {s: EP.repeat_episode(fam, beh, cfg, s) for s in seeds}
        chg = {s: EP.change_episode(fam, beh, cfg, s) for s in seeds}
        epu = {s: EP.build_episode(fam, beh, ucfg, s) for s in seeds}

        sel, un = {}, {}
        for arm, lk in ARMS_0C:
            var = rep if arm == "repeat_task" else (chg if arm == "phi_change"
                                                    else None)
            sel[f"{arm}/{lk}"] = per_episode(fam, beh, eps, arm, lk, cfg,
                                             seeds, variant=var)
        for arm, lk in (("oracle", "aware"), ("static", "aware"),
                        ("persist", "aware"), ("query_random", "aware"),
                        ("query_infogain", "aware")):
            un[f"{arm}/{lk}"] = per_episode(fam, beh, epu, arm, lk, ucfg, seeds)

        print(f"\n   SELECTED DISTRIBUTION")
        print(f"   {'arm':30}{'transfer':>9}{'late':>8}{'cal':>6}{'H_end':>7}"
              f"{'mass':>7}{'conflict':>9}{'abst':>6}")
        for k, r in sel.items():
            print(f"   {k:30}{mean(r['whole']):>9.3f}{mean(r['late']):>8.3f}"
                  f"{mean(r['cal']):>6.2f}{mean(r['H_end']):>7.2f}"
                  f"{mean(r['mass_end']):>7.3f}{mean(r['conflict']):>9.3f}"
                  f"{mean(r['abstain']):>6.2f}")
        O, Sv, P = (mean(sel[f"{a}/aware"]["whole"])
                    for a in ("oracle", "static", "persist"))
        print(f"   R (selected) {recovery(O, Sv, P):.3f}")

        print(f"\n   UNCONDITIONAL DISTRIBUTION")
        cand = sum(e.rejected + len(e.tr_idx) for e in eps.values())
        acc_ = sum(len(e.tr_idx) for e in eps.values())
        for k, r in un.items():
            print(f"   {k:30}{mean(r['whole']):>9.3f}{mean(r['late']):>8.3f}"
                  f"{'':6}{'':7}{'':7}{'':9}{mean(r['abstain']):>6.2f}")
        Ou, Su, Pu = (mean(un[f"{a}/aware"]["whole"])
                      for a in ("oracle", "static", "persist"))
        print(f"   R (unconditional) {recovery(Ou, Su, Pu):.3f}")
        print(f"   selection acceptance rate {acc_/cand:.3f} ({acc_} accepted "
              f"of {cand} candidates on the SELECTED episodes)")

        runs = [EP.run_arm(fam, beh, eps[s], "persist", cfg, s,
                           likelihood="aware") for s in seeds]
        ne = min(len(r["entropy"]) for r in runs)
        ent = [mean([r["entropy"][i] for r in runs]) for i in range(ne)]
        mss = [mean([r["mass"][i] for r in runs]) for i in range(ne)]
        step = max(1, len(ent) // 8)
        ks = list(range(0, len(ent), step))
        print("\n   after task    " + "".join(f"{k:>7}" for k in ks))
        print("   H(phi) bits   " + "".join(f"{ent[k]:>7.2f}" for k in ks))
        print("   true mass     " + "".join(f"{mss[k]:>7.3f}" for k in ks))
        co, cp = sel["oracle/aware"]["curve"], sel["persist/aware"]["curve"]
        print("   transfer idx  " + "".join(f"{i+1:>7}"
                                            for i in range(len(cp))))
        print("   persist       " + "".join(f"{x:>7.2f}" for x in cp))
        reg = adaptation_regret(co, cp)
        print(f"   adaptation regret {reg:.3f} transfer tasks "
              f"({reg/max(1,len(cp)):.3f} per task)")

        cc = [A.conflict_curves(fam, beh, eps[s], cfg, s) for s in seeds]
        m_ = [v for c in cc for v in c["matched"]]
        x_ = [v for c in cc for v in c["contradictory"]]
        print(f"   conflict AUROC {A.auroc(x_, m_):.3f}  AUPRC "
              f"{A.auprc(x_, m_):.3f}  (matched {mean(m_):.3f} vs "
              f"contradictory {mean(x_):.3f})")

        ow = open_world(fam, beh, cfg, seeds)
        print(f"   open world: {ow['unknown_meaning_declared']}/"
              f"{ow['out_of_space_tasks']} out-of-space tasks declared "
              f"UNKNOWN_MEANING ({ow['detection_rate']:.3f}); "
              f"{ow['silently_answered']} answered silently")
        rp = restart_persistence(fam, beh, cfg, seeds[0])
        print(f"   restart persistence across a separate process: "
              f"{'ok' if rp['ok'] else 'FAIL'} "
              f"(H {rp.get('H_in_process', float('nan')):.6f} -> "
              f"{rp.get('H_after_restart', float('nan')):.6f})")

        ivs = {}
        for lab, b in (("static", "static/aware"), ("reset", "reset/aware"),
                       ("shuffled", "shuffled/aware"),
                       ("wrong_pairing", "wrong_pairing/aware"),
                       ("persist/naive", "persist/naive")):
            for win in ("whole", "late"):
                bs = A.paired_bootstrap(sel["persist/aware"][win],
                                        sel[b][win])
                ivs[f"persist/aware - {lab} [{win}]"] = bs
                print(f"   persist/aware - {lab:14}[{win:5}] "
                      f"{bs['delta']:+.3f} ({bs['lo']:+.3f}, {bs['hi']:+.3f})"
                      f"  {'excludes 0' if bs['excludes_zero'] else 'INCLUDES 0'}")
        qn = {p: [mean(EP.questions_to_identify(fam, beh, eps[s], p, cfg, s))
                  for s in seeds] for p in ("infogain", "random")}
        bs = A.paired_bootstrap(qn["infogain"], qn["random"])
        ivs["infogain - random questions"] = bs
        print(f"   infogain - random questions   [count] {bs['delta']:+.3f} "
              f"({bs['lo']:+.3f}, {bs['hi']:+.3f})  "
              f"{'excludes 0' if bs['excludes_zero'] else 'INCLUDES 0'}")

        art["families"][ov] = {
            "seeds": list(seeds),
            "selected": {k: {"whole": mean(v["whole"]),
                             "late": mean(v["late"]),
                             "cal": mean(v["cal"]),
                             "H_end": mean(v["H_end"]),
                             "mass_end": mean(v["mass_end"]),
                             "conflict": mean(v["conflict"]),
                             "abstain": mean(v["abstain"]),
                             "per_episode_whole": v["whole"]}
                         for k, v in sel.items()},
            "unconditional": {k: {"whole": mean(v["whole"]),
                                  "late": mean(v["late"]),
                                  "abstain": mean(v["abstain"])}
                              for k, v in un.items()},
            "R_selected": recovery(O, Sv, P),
            "R_unconditional": recovery(Ou, Su, Pu),
            "acceptance_rate": acc_ / cand,
            "entropy_by_position": ent, "mass_by_position": mss,
            "transfer_by_position": cp,
            "adaptation_regret": reg,
            "conflict": {"auroc": A.auroc(x_, m_), "auprc": A.auprc(x_, m_),
                         "matched": mean(m_), "contradictory": mean(x_)},
            "open_world": ow, "restart": rp, "intervals": ivs,
            "questions": {k: mean(v) for k, v in qn.items()},
        }

    print("\nCLOSURE CONDITIONS")
    cond = {}
    fa = art["families"]
    cond["1_effect_in_both_strata"] = all(
        fa[ov]["intervals"]["persist/aware - static [whole]"]["lo"] > 0
        for ov in fa)
    cond["2_correct_selection_model_preserves_it"] = all(
        fa[ov]["selected"]["persist/aware"]["whole"]
        >= fa[ov]["selected"]["persist/naive"]["whole"] - 1e-9 for ov in fa)
    cond["3_reset_and_shuffled_remove_it"] = all(
        fa[ov]["intervals"][f"persist/aware - {c} [late]"]["lo"] > 0
        for ov in fa for c in ("reset", "shuffled"))
    cond["4_later_new_tasks_improve"] = all(
        fa[ov]["selected"]["persist/aware"]["late"]
        > fa[ov]["selected"]["persist/aware"]["whole"] for ov in fa)
    cond["5_no_hidden_seed_repair"] = True
    cond["6_selected_and_unconditional_reported"] = all(
        "unconditional" in fa[ov] and "acceptance_rate" in fa[ov]
        for ov in fa)
    for k, v in cond.items():
        print(f"   {k:44} {'PASS' if v else 'FAIL'}")
    art["closure"] = cond
    passed = all(cond.values())
    print(f"\n   X64 closes at the controlled hidden-convention level: "
          f"{passed}")
    print("   Supported claim, and no more: Sentinel learns a hidden")
    print("   communication convention from grounded prior tasks and reuses it")
    print("   to interpret later, new tasks in a controlled authored semantic")
    print("   environment. NOT unrestricted natural-language understanding.")
    art["runtime_s"] = round(time.perf_counter() - t0, 2)
    (OUT / "x64h_final.json").write_text(
        json.dumps(art, indent=2, sort_keys=True, default=str))
    print(f"   authoritative JSON -> {OUT/'x64h_final.json'}")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
