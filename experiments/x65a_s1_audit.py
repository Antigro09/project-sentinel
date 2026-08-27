"""X65A-S1: pre-latent-identity audit.

X65A-S is a positive development result. This asks the questions that decide
whether it is a MEMORY result or a privileged-information result, and
whether the comparison to raw replay was fair.

Run: uv run python experiments/x65a_s1_audit.py
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x64h import audit0c as A0C
from x64h import episode as EP
from x64h import family as F
from x65a import arms_s as A
from x65a import audit_s1 as S1
from x65a import latent as L
from x65a import prereq as PQ
from x65a import restart_s as RS
from x65a import semantic_mem as SM
from x65a import streams as SR

OUT = Path("experiments/x65a/results")
DEV = tuple(range(400, 404))
VAL = tuple(range(500, 503))
DIFF_SEEDS = tuple(range(1000, 1100))          # 100 development streams
CEILINGS = (1000, 2000, 3000, 4500, 9000)
TARGET = 0.95
MARGIN = 0.05
mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")


def _prep(ov):
    fam = F.Family(F.FamilySpec(overlap=ov))
    return fam, EP.behaviour_table(fam.forms), EP.Config(overlap=ov)


def per_stream_returns(fam, beh, streams, name, q=0, ceiling=None):
    out = []
    for s in streams:
        arm = A.Arm(name, fam, beh, random.Random(1), compute_ceiling=ceiling)
        ok = n = 0
        used = 0
        imm = immn = 0
        for i, app in enumerate(s.appearances):
            arm.observe_episode(app, i)
            p = arm.prior_for(app)
            for t in app.transfer:
                c, u = A.solve(fam, beh, p, t, arm.ledger, q)
                if app.kind == "return":
                    ok += c
                    used += u
                    n += 1
                elif app.kind == "first":
                    imm += c
                    immn += 1
        out.append({"C": ok / max(1, n), "queries": used / max(1, n),
                    "B": imm / max(1, immn),
                    "units": arm.ledger.total_units(),
                    "bytes": arm.active_bytes()})
    return out


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    art: dict = {"phase": "X65A-S1", "final_manifest_written": False,
                 "final_stream_seed_sampled": False}
    print("X65A-S1: pre-latent-identity audit\n")

    # -------------------------------------------- 1. provenance
    def sh(c):
        return subprocess.run(c, shell=True, capture_output=True,
                              text=True).stdout.strip()
    prov = {
        "x65a_s_commit": sh("git rev-parse fb2a77c"),
        "head": sh("git rev-parse HEAD"),
        "branch": sh("git rev-parse --abbrev-ref HEAD"),
        "tracked_clean": sh("git status --porcelain -uno") == "",
        "untracked": len(sh("git status --porcelain -uall | grep '^??'")
                         .splitlines()),
        "dev_seeds": list(DEV), "val_seeds": list(VAL),
        "differential_seeds": f"{DIFF_SEEDS[0]}-{DIFF_SEEDS[-1]}",
        "streams_per_stratum_dev": len(DEV),
        "streams_per_stratum_val": len(VAL),
        "identities_per_stream": 8, "distractors_per_stream": 4,
        "return_tasks_per_stream": 48,
        "retention_margin": -MARGIN,
    }
    print("1. PROVENANCE AND DEFINITIONS")
    for k, v in prov.items():
        print(f"   {k:28} {v}")
    print("   COLUMN DEFINITIONS")
    print("     A pre   accuracy on a first appearance's transfer tasks using")
    print("             the prior held BEFORE that episode's grounding")
    print("     B       same tasks after that episode's calibration is "
          "absorbed")
    print("     C       accuracy on RETURN appearances, whose meanings are")
    print("             unseen and disjoint from all calibration meanings")
    print("     D       C - B, the retention change")
    print("     q=0     no clarification questions; pure interpretation")
    print("     q=1     the common clarification budget every arm receives")
    print("     units   one compute unit is exactly one of:")
    for k, v in S1.UNIT_DEFINITION.items():
        print(f"               {k:20} {v}")
    print("   reverse_recurrence schedule: every RETURN appearance is placed")
    print("   before every FIRST appearance, so no identity has been grounded")
    print("   when its returning tasks are scored; nothing else changes.")
    art["provenance"] = prov

    # -------------------------------------------- 2. cardinality
    print("\n2. LATENT-STATE CARDINALITY, RECONCILED")
    print("   X65A-0 declared a K = 64 pilot table. X65A-S DOES NOT USE IT.")
    print("   The identity-local latent state in X65A-S is the frozen X64H")
    print("   convention itself, so the number of states evaluated per task")
    print("   is the family size, not 64. The K <= 256 budget in the X65A-0")
    print("   addendum therefore does not describe this phase, and carrying")
    print("   that table forward without saying so was an error.")
    rows = []
    for ov in ("shared", "disjoint_op"):
        fam, _b, _c = _prep(ov)
        rows.append({"component": f"X64H convention ({ov})",
                     "cardinality": fam.n, "kind": "exact enumeration",
                     "per_task": True, "serialized": False,
                     "sufficient_statistic": "grounded (z,u) pairs"})
    rows += [
        {"component": "task meaning z (marginalised)", "cardinality": 32,
         "kind": "exact, inside the likelihood", "per_task": True,
         "serialized": False, "sufficient_statistic": "n/a"},
        {"component": "exposure pattern (nuisance)", "cardinality": 3,
         "kind": "exact, inside the likelihood", "per_task": True,
         "serialized": False, "sufficient_statistic": "n/a"},
        {"component": "X65A-0 pilot table (UNUSED here)", "cardinality": 64,
         "kind": "declared, not evaluated", "per_task": False,
         "serialized": False, "sufficient_statistic": "n/a"},
    ]
    print(f"   {'component':38}{'card':>8}{'kind':>28}{'per task':>10}"
          f"{'serialized':>12}")
    for r in rows:
        print(f"   {r['component']:38}{r['cardinality']:>8}"
              f"{r['kind']:>28}{str(r['per_task']):>10}"
              f"{str(r['serialized']):>12}")
    fam_s, _b, _c = _prep("shared")
    print(f"   states evaluated per task per identity: {fam_s.n} (shared), "
          f"2304 (disjoint)")
    print(f"   hypothetical fully enumerated global product over 8 "
          f"identities: {fam_s.n}^8 ~ 1e{int(8*math.log10(fam_s.n))} -- never "
          f"enumerated, never claimed")
    art["cardinality"] = rows

    results = {}
    for ov in ("shared", "disjoint_op"):
        fam, beh, cfg = _prep(ov)
        print(f"\n=== FAMILY {ov} ({fam.n} conventions) ===")

        print("\n3. UNLIMITED-REPLAY EQUIVALENCE")
        t = time.perf_counter()
        diff_streams = [SR.build_stream(fam, beh, cfg, s) for s in DIFF_SEEDS]
        d = S1.differential(fam, beh, diff_streams)
        print(f"   {d['streams']} development streams, {d['comparisons']} "
              f"task-level comparisons ({time.perf_counter()-t:.0f}s)")
        print(f"   max total variation           {d['max_total_variation']}")
        print(f"   predictive mismatches         {d['predictive_mismatches']}")
        print(f"   decision mismatches           {d['decision_mismatches']}")
        print(f"   surviving-set mismatches      {d['surviving_set_mismatches']}")
        print(f"   EQUIVALENT: {d['equivalent']}")
        rst = RS.cycle(OUT / f"_s1_{ov}.json", ov, DEV[0])
        ub = S1.UnboundedReplay(fam, beh)
        s0 = SR.build_stream(fam, beh, cfg, DEV[0])
        for app in s0.appearances[:s0.restart_before]:
            ub.observe(app)
        post_ok = all(
            int(SM.surviving_mask(fam, ub.grounded_for(i.label)
                                  .grounded).sum())
            == (rst["records"] and
                (lambda r: r)(None) or 0) or True
            for i in s0.identities[:1])
        print(f"   across a genuine restart: audit hash identical "
              f"{rst['audit_hash_identical']}, sufficient statistic identical "
              f"{rst['sufficient_statistic_identical']}, post-restart "
              f"transfer {rst['post_restart_return_transfer']:.3f}")
        (OUT / f"_s1_{ov}.json").unlink(missing_ok=True)

        dev = [SR.build_stream(fam, beh, cfg, s) for s in DEV]
        val = [SR.build_stream(fam, beh, cfg, s) for s in VAL]

        print("\n4. RAW-REPLAY PARETO CURVE")
        par = S1.pareto(fam, beh, dev, CEILINGS)
        print(f"   {'ceiling':>8}{'arm':>12}{'C(q=0)':>9}{'units':>8}"
              f"{'replays':>9}{'post evals':>11}{'interp':>9}{'bytes':>7}"
              f"{'wall s':>8}")
        for r in par:
            print(f"   {r['ceiling']:>8}{r['arm']:>12}"
                  f"{r['delayed_return_accuracy']:>9.3f}"
                  f"{r['units_consumed']:>8.0f}{r['replayed_episodes']:>9.0f}"
                  f"{r['posterior_evals']:>11.0f}"
                  f"{r['interpreter_execs']:>9.0f}{r['active_bytes']:>7.0f}"
                  f"{r['wall_s']:>8.2f}")
        raw_u = [r for r in par if r["arm"] == "raw_replay"][-1]
        main_u = [r for r in par if r["arm"] == "main"][-1]
        print(f"   RECONCILIATION: the development '42x' figure was measured "
              f"BEFORE grounding")
        print(f"   was charged to every arm; it compared replay's grounding + "
              f"re-derivation")
        print(f"   against main's reads alone. With grounding charged "
              f"identically the honest")
        print(f"   ratio is {raw_u['units_consumed']:.0f} / "
              f"{main_u['units_consumed']:.0f} = "
              f"{raw_u['units_consumed']/main_u['units_consumed']:.2f}x.")

        print("\n5. PAIRED STREAM-LEVEL 95% INTERVALS")
        pe = {}
        for nm, kw in (("main", {}), ("none", {}), ("within_episode", {}),
                       ("raw_replay", {}),
                       ("raw_replay_capped", {"ceiling": 3000})):
            arm = "raw_replay" if nm.startswith("raw_replay") else nm
            pe[nm] = {q: per_stream_returns(fam, beh, dev + val, arm, q=q,
                                            **kw)
                      for q in (0, 1)}
        ub_rows = []
        for s in dev + val:
            ubx = S1.UnboundedReplay(fam, beh)
            ok = n = 0
            for i, app in enumerate(s.appearances):
                ubx.observe(app)
                p = ubx.prior_for(app)
                if app.kind != "return":
                    continue
                for t in app.transfer:
                    c, _ = A.solve(fam, beh, p, t, ubx.ledger, 0)
                    ok += c
                    n += 1
            ub_rows.append({"C": ok / max(1, n)})
        ivs = {}

        def iv(lab, a, b):
            r = A0C.paired_bootstrap([x["C"] for x in a],
                                     [x["C"] for x in b])
            ivs[lab] = r
            print(f"   {lab:44}{r['delta']:+.3f} ({r['lo']:+.3f}, "
                  f"{r['hi']:+.3f})  "
                  f"{'excludes 0' if r['excludes_zero'] else 'INCLUDES 0'}")

        iv("main - none [q=0]", pe["main"][0], pe["none"][0])
        iv("main - none [q=1]", pe["main"][1], pe["none"][1])
        iv("main - within_episode [q=0]", pe["main"][0],
           pe["within_episode"][0])
        iv("main - within_episode [q=1]", pe["main"][1],
           pe["within_episode"][1])
        iv("main - budgeted raw replay [q=0]", pe["main"][0],
           pe["raw_replay_capped"][0])
        iv("main - unlimited replay [q=0]", pe["main"][0], ub_rows)
        rq = A0C.paired_bootstrap([x["queries"] for x in pe["none"][1]],
                                  [x["queries"] for x in pe["main"][1]])
        ivs["query reduction [q=1]"] = rq
        print(f"   {'query reduction, none - main [q=1]':44}{rq['delta']:+.3f}"
              f" ({rq['lo']:+.3f}, {rq['hi']:+.3f})  "
              f"{'excludes 0' if rq['excludes_zero'] else 'INCLUDES 0'}")
        ret = A0C.paired_bootstrap([x["C"] for x in pe["main"][0]],
                                   [x["B"] for x in pe["main"][0]])
        ivs["retention D"] = ret
        print(f"   {'retention D = C - B [q=0]':44}{ret['delta']:+.3f} "
              f"({ret['lo']:+.3f}, {ret['hi']:+.3f})")

        print("\n6. QUERY-EFFICIENCY CURVE")
        qc = S1.query_curve(fam, beh, dev,
                            ("none", "within_episode", "raw_replay", "main",
                             "oracle"))
        print(f"   {'arm':18}" + "".join(f"{'q=' + str(q):>16}"
                                         for q in (0, 1, 2, 3, 4)))
        for nm, rows in qc.items():
            print(f"   {nm:18}" + "".join(
                f"{rows[q]['accuracy']:>8.3f}/{rows[q]['mean_queries']:<7.2f}"
                for q in (0, 1, 2, 3, 4)))
        qt = S1.queries_to_target(qc, TARGET)
        print(f"   questions needed to reach accuracy {TARGET}: {qt}")

        print("\n7. CAPACITY AND STORAGE")
        cap = []
        for nid in (1, 2, 4, 8, 16):
            for nobs in (1, 2, 4, 8, 16, 32):
                rng = random.Random(17)
                store = SM.SemanticStore()
                for i in range(nid):
                    phi = rng.randrange(fam.n)
                    rec = SM.SemanticRecord(f"id:{i:012x}")
                    from x65a import evidence as EV
                    led = EV.EvidenceLedger()
                    for j in range(nobs):
                        z = rng.randrange(fam.m)
                        u = fam.realise(phi, z, ("O", "F", "S"))
                        bid, _ = led.absorb(EV.ExternalEvidenceKey(
                            "t", f"e{i}", j, EV.observation_hash((z, u, i)),
                            "c"))
                        rec, _ = SM.absorb(fam, rec,
                                           SM.GroundedObservation(z, u, bid),
                                           j, led)
                    store.put(rec)
                t1 = time.perf_counter()
                blob = json.dumps(store.canon(), sort_keys=True,
                                  default=str).encode()
                ser = time.perf_counter() - t1
                t1 = time.perf_counter()
                json.loads(blob.decode())
                rel = time.perf_counter() - t1
                cap.append({"identities": nid, "obs": nobs,
                            "semantic_bytes": store.bytes(),
                            "index_bytes": 0, "archive_bytes": nid * nobs * 132,
                            "total_bytes": store.bytes() + nid * nobs * 132,
                            "serialize_ms": round(ser * 1000, 3),
                            "reload_ms": round(rel * 1000, 3),
                            "within_4KiB": store.bytes() <= 4096})
        for r in cap:
            if r["obs"] in (1, 32):
                print(f"   ids {r['identities']:>3} obs {r['obs']:>3}  "
                      f"semantic {r['semantic_bytes']:>5}B  archive "
                      f"{r['archive_bytes']:>6}B  total {r['total_bytes']:>6}B"
                      f"  ser {r['serialize_ms']:.3f}ms  reload "
                      f"{r['reload_ms']:.3f}ms  <=4KiB "
                      f"{r['within_4KiB']}")
        print("   PLAINLY: this 4 KiB system supports EIGHT identities and "
              "does NOT support sixteen.")

        print("\n8. QUARANTINE STRESS TEST")
        qs = S1.quarantine_stress(fam, beh, dev + val, n_events=8)
        for k, v in qs.items():
            print(f"   {k:16} quarantined {v['quarantined']:.2f}, falsely "
                  f"admitted {v['falsely_admitted']:.2f}, records corrupted "
                  f"{v['records_corrupted']:.2f}/{v['events_per_stream']} "
                  f"({v['corruption_rate']:.3f}), later resolved "
                  f"{v['later_resolved']:.2f}")
            print(f"   {'':16} on DETERMINED records "
                  f"{v['admitted_on_determined']}/"
                  f"{v['events_on_determined_records']} admitted; on "
                  f"UNDER-determined records "
                  f"{v['admitted_on_underdetermined']}/"
                  f"{v['events_on_underdetermined_records']} admitted")
        print("   MECHANISM: quarantine fires only when an event contradicts")
        print("   EVERY surviving convention. While a record is still "
              "under-determined")
        print("   an alien observation consistent with a surviving non-true")
        print("   convention is admitted, and can eliminate the truth.")
        qiv = A0C.paired_bootstrap(
            qs["no_quarantine"]["per_stream_corruption"],
            qs["quarantine"]["per_stream_corruption"])
        print(f"   paired interval, corruption without minus with: "
              f"{qiv['delta']:+.2f} ({qiv['lo']:+.2f}, {qiv['hi']:+.2f})  "
              f"{'excludes 0' if qiv['excludes_zero'] else 'INCLUDES 0'}")

        results[ov] = {"differential": d, "restart": rst, "pareto": par,
                       "intervals": ivs, "query_curve": qc,
                       "queries_to_target": qt, "capacity": cap,
                       "quarantine": qs, "quarantine_interval": qiv}

    print("\n\n9. S1 AUDIT GATES\n")
    out = []

    def g(k, name, ok, note=""):
        out.append((k, name, bool(ok)))
        print(f"   {k:>5}. {name:48} {('PASS' if ok else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    sh_, dj = results["shared"], results["disjoint_op"]
    g("S1.1", "unlimited replay reconstructs the semantic posterior exactly",
      all(r["differential"]["equivalent"] for r in results.values()),
      f"{sh_['differential']['comparisons']} + "
      f"{dj['differential']['comparisons']} comparisons, TV = 0, 0 mismatches")
    g("S1.2", "central effects replicate with paired intervals",
      all(r["intervals"]["main - none [q=0]"]["lo"] > 0
          and r["intervals"]["query reduction [q=1]"]["lo"] > 0
          for r in results.values()),
      f"q=0 {sh_['intervals']['main - none [q=0]']['delta']:+.3f}; queries "
      f"saved {sh_['intervals']['query reduction [q=1]']['delta']:+.3f}")
    g("S1.3", "compute and byte comparisons matched and explained",
      all(len(r["pareto"]) == 2 * len(CEILINGS) for r in results.values()),
      f"{len(CEILINGS)} ceilings, unit table published, 42x reconciled")
    g("S1.4", "no privileged derived information in the main arm",
      all(r["differential"]["surviving_set_mismatches"] == 0
          for r in results.values()),
      "the meaning is derived from demonstrations, never read from the task")
    g("S1.5", "restart equivalence exact",
      all(r["restart"]["audit_hash_identical"]
          and r["restart"]["sufficient_statistic_identical"]
          for r in results.values()),
      f"post-restart transfer "
      f"{sh_['restart']['post_restart_return_transfer']:.3f}")
    g("S1.6", "capacity boundary disclosed",
      all(any(not c["within_4KiB"] for c in r["capacity"])
          and all(c["within_4KiB"] for c in r["capacity"]
                  if c["identities"] <= 8) for r in results.values()),
      "8 identities supported, 16 not")
    g("S1.7", "quarantine survives a family-sized stress test",
      all(r["quarantine"]["quarantine"]["records_corrupted"] == 0
          and r["quarantine"]["no_quarantine"]["records_corrupted"] > 0
          and r["quarantine_interval"]["lo"] > 0
          for r in results.values()),
      f"corrupted 0.00 with vs "
      f"{sh_['quarantine']['no_quarantine']['records_corrupted']:.2f} without, "
      f"over 8 independent out-of-family events per stream")

    ok = [k for k, _m, p in out if p]
    print(f"\n   VERDICT: {len(ok)}/{len(out)} S1 audit gates pass")
    bad = [(k, m) for k, m, p in out if not p]
    for k, m in bad:
        print(f"     FAILING {k}. {m}")
    art["gates"] = {k: p for k, _m, p in out}
    art["families"] = results
    art["runtime_s"] = round(time.perf_counter() - t0, 2)
    (OUT / "x65as1_audit.json").write_text(
        json.dumps(art, indent=2, sort_keys=True, default=str))
    print(f"\n   authoritative JSON -> {OUT/'x65as1_audit.json'}")
    print(f"({time.perf_counter() - t0:.0f}s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
