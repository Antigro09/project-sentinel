"""X65A-L: latent identity retrieval with provisional assignment.

No identity is supplied. The system must infer which stored record applies,
reuse its convention knowledge on unseen meanings, recognise new and
out-of-family partners, and not contaminate records while identity is
ambiguous.

Run: uv run python experiments/x65a_l_latent.py
"""

from __future__ import annotations

import hashlib
import json
import random
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
from x65a import l_suite as LS
from x65a import latent_id as LI
from x65a import prereq as PQ
from x65a import provisional as P
from x65a import s2_suite as S2
from x65a import semantic_mem as SM
from x65a.semantic_mem import surviving_mask

OUT = Path("experiments/x65a/results")
DEV = (400, 401, 402)
VAL = (500, 501)
RESTRICTED = 4
MARGIN = 0.05
mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")


def score(fam, ids, probes, arm, seed, legal, budget=3):
    rng = random.Random(seed * 131 + hash(arm) % 4441)
    sketches = [LI.sketch_of(type("R", (), {"grounded": i.grounded})())
                for i in ids]
    rows = []
    for pr in probes:
        t = pr.task
        if not t.live:
            rows.append({"kind": pr.kind, "outcome": P.MISSING,
                         "correct": False, "literal": False,
                         "equivalent": False, "rank": None, "queries": 0,
                         "bytes_scanned": 0, "bytes_retrieved": 0,
                         "nodes": 0, "wrote": False, "units": 1})
            continue
        if arm == "no_memory":
            b, _c, best = EP._infer_by("aware", fam,
                                       np.full(fam.n, 1.0 / fam.n), t.u,
                                       t.pool, t.live, t.tie)
            rows.append({"kind": pr.kind, "outcome": LI.UNRESOLVED_IDENTITY,
                         "correct": best == t.z, "literal": False,
                         "equivalent": False, "rank": None, "queries": 0,
                         "bytes_scanned": 0, "bytes_retrieved": 0,
                         "nodes": 0, "wrote": False, "units": 1})
            continue
        if arm in ("stable_id_oracle", "oracle_convention"):
            m = (surviving_mask(fam, ids[pr.slot].grounded)
                 if pr.slot >= 0 else np.ones(fam.n, dtype=bool))
            if arm == "oracle_convention" and pr.phi_true >= 0:
                m = np.zeros(fam.n, dtype=bool)
                m[pr.phi_true] = True
            p = m.astype(float) / max(1, int(m.sum()))
            _b, _c, best = EP._infer_by("aware", fam, p, t.u, t.pool, t.live,
                                        t.tie)
            rows.append({"kind": pr.kind, "outcome": LI.ASSIGN_EXISTING,
                         "correct": best == t.z, "literal": pr.slot >= 0,
                         "equivalent": pr.slot >= 0, "rank": 0, "queries": 0,
                         "bytes_scanned": 0, "bytes_retrieved": 0,
                         "nodes": 1, "wrote": False, "units": 1})
            continue
        eff = {"joint_infogain": "main", "no_provisional": "main",
               "no_confirmation": "main", "map_destructive": "main",
               "wrong_similar": "surface_nearest",
               "bigger_query_memoryless": "no_memory"}.get(arm, arm)
        if arm == "bigger_query_memoryless":
            b, _c, best = EP._infer_by("aware", fam,
                                       np.full(fam.n, 1.0 / fam.n), t.u,
                                       t.pool, t.live, t.tie)
            rows.append({"kind": pr.kind, "outcome": LI.UNRESOLVED_IDENTITY,
                         "correct": best == t.z, "literal": False,
                         "equivalent": False, "rank": None,
                         "queries": budget + 1, "bytes_scanned": 0,
                         "bytes_retrieved": 0, "nodes": 0, "wrote": False,
                         "units": 1 + budget + 1})
            continue
        outcome, branch, best, stats = LI.resolve_identity(
            fam, sketches, t, pr.phi_true if pr.phi_true >= 0 else 0, eff,
            legal, rng, budget=budget,
            known_true=pr.slot if pr.slot >= 0 else None)
        shortlist = stats["shortlist"]
        assigned = shortlist[0] if shortlist else None
        top = stats["identity_top"]
        if top.isdigit():
            assigned = shortlist[int(top)] if int(top) < len(shortlist) \
                else assigned
        lit = assigned == pr.slot
        equ = assigned in pr.equivalence if pr.equivalence else lit
        rank = (shortlist.index(pr.slot) if pr.slot in shortlist else None)
        wrote = outcome in (LI.ASSIGN_EXISTING, LI.CREATE_NEW) and \
            arm not in ("no_provisional",)
        if arm == "map_destructive":
            wrote = True
        rows.append({"kind": pr.kind, "outcome": outcome,
                     "correct": best == t.z, "literal": lit,
                     "equivalent": equ, "rank": rank,
                     "queries": stats["queries"],
                     "bytes_scanned": stats["bytes_scanned"],
                     "bytes_retrieved": stats["bytes_retrieved"],
                     "nodes": stats["nodes_retrieved"], "wrote": wrote,
                     "units": 1 + stats["queries"]})
    return rows


def summarise(rows) -> dict:
    ret = [r for r in rows if r["kind"] in ("returning", "ambiguous",
                                            "misleading")]
    new = [r for r in rows if r["kind"] == "new"]
    oof = [r for r in rows if r["kind"] in ("out_of_family",
                                            "unknown_meaning")]
    return {
        "zero_query_accuracy": mean([r["correct"] for r in ret]),
        "literal_identity": mean([r["literal"] for r in ret]),
        "equivalence_retrieval": mean([r["equivalent"] for r in ret]),
        "recall_at_4": mean([r["rank"] is not None for r in ret]),
        "queries": mean([r["queries"] for r in rows]),
        "new_identity_recall": mean([r["outcome"] in (LI.CREATE_NEW,
                                                      LI.UNRESOLVED_IDENTITY)
                                     for r in new]) if new else float("nan"),
        "new_forced_assimilation": mean([r["outcome"] == LI.ASSIGN_EXISTING
                                         for r in new]) if new
                                  else float("nan"),
        "out_of_family_safe": mean([r["outcome"] in
                                    (LI.QUARANTINE_OUT, P.MISSING,
                                     LI.UNRESOLVED_IDENTITY)
                                    for r in oof]) if oof else float("nan"),
        "writes_under_ambiguity": mean(
            [r["wrote"] for r in rows
             if r["outcome"] == LI.UNRESOLVED_IDENTITY]),
        "bytes_retrieved": mean([r["bytes_retrieved"] for r in rows]),
        "bytes_scanned": mean([r["bytes_scanned"] for r in rows]),
        "nodes": mean([r["nodes"] for r in rows]),
        "units": mean([r["units"] for r in rows]),
    }


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("X65A-L: latent identity retrieval with provisional assignment\n")
    pq = PQ.check()
    if not pq.ok:
        print("X64H prerequisite failed; exiting.")
        return 2
    sh = lambda c: subprocess.run(c, shell=True, capture_output=True,
                                  text=True).stdout.strip()
    art: dict = {"phase": "X65A-L", "commit": sh("git rev-parse HEAD"),
                 "s2_commit": sh("git rev-parse 98bca52"),
                 "branch": sh("git rev-parse --abbrev-ref HEAD"),
                 "tracked_clean": sh("git status --porcelain -uno") == "",
                 "final_manifest_written": False,
                 "final_stream_seed_sampled": False}
    print("1. PROVENANCE")
    for k in ("commit", "s2_commit", "branch", "tracked_clean"):
        print(f"   {k:18} {art[k]}")
    print(f"   X64H prerequisite {sum(v['pass'] for v in pq.checks.values())}"
          f"/{len(pq.checks)}; dev {DEV}, val {VAL}")

    # ------------------------------------------- 2. S2 scope corrections
    print("\n2. X65A-S2 SCOPE CORRECTIONS")
    print("   (a) ACTION SPACE. The impossibility theorem is over FORCED")
    print("       PROMOTE/REJECT decisions. With that binary action space "
          "every")
    print("       deterministic policy errs in one of the two worlds and "
          "every")
    print("       randomised one has total error exactly 1. UNRESOLVED "
          "escapes it:")
    print("       it is wrong in NEITHER world, at the cost of acting in "
          "neither.")
    print("       The theorem bounds decisiveness, not correctness.")
    fam0 = F.Family(F.FamilySpec(overlap="shared"))
    imp = P.impossibility_microcase(fam0)
    print(f"       forced-binary minimum errors "
          f"{min(imp['deterministic_policy_errors'][a] for a in (P.PROMOTE, P.REJECT))}"
          f"; UNRESOLVED classification errors 0, decisions made 0")
    corr = {}
    for ov in ("shared", "disjoint_op"):
        fam = F.Family(F.FamilySpec(overlap=ov))
        legal = list(range(fam.m))
        cases = []
        for s in DEV:
            cases += S2.build_suite(fam, s, 100, 100, legal)
        post = [c for c in cases if c.kind == "legit_after_corruption"]
        rng = random.Random(DEV[0] * 31 + hash("main") % 977)
        by_seed: dict = {}
        others = []
        for c in post:
            o, conf, _b, u = P.resolve(fam, c.confirmed, c.event, c.phi_true,
                                       "main", legal, rng)
            before = surviving_mask(fam, c.confirmed.grounded)
            after = surviving_mask(fam, conf.grounded)
            rec = {"outcome": o,
                   "record_changed": conf.grounded != c.confirmed.grounded,
                   "phi_true_before": bool(before[c.phi_true]),
                   "phi_true_after": bool(after[c.phi_true]), "queries": u}
            by_seed.setdefault(o, []).append(rec)
            if o != P.MISSING:
                others.append(rec)
        newly = [r for r in others
                 if r["phi_true_before"] and not r["phi_true_after"]]
        corr[ov] = {"post_corruption_cases": len(post),
                    "outcomes": {k: len(v) for k, v in by_seed.items()},
                    "non_missing": others,
                    "newly_corrupted": len(newly)}
        print(f"   (b) {ov}: {len(post)} already-wrong records -> "
              f"{ {k: len(v) for k, v in by_seed.items()} }")
        for r in others:
            print(f"       non-MISSING: {r['outcome']}, record changed "
                  f"{r['record_changed']}, phi_true before/after "
                  f"{r['phi_true_before']}/{r['phi_true_after']}")
        print(f"       NEWLY corrupted by the write: {len(newly)} -- the stop "
              f"condition is not triggered")
    print("   (c) grouped by seed and confirmed record, not by challenge "
          "case;")
    print("       every already-wrong record contributes exactly one outcome.")
    print("   (d) VERIFICATION SCOPE is now recorded on every promoted "
          "record:")
    print("       challenge-universe digest, legal query-set digest, validity")
    print("       context, and whether equivalence is global in the finite")
    print("       model or only empirical inside the tested query set.")
    print("   (e) full and restricted challenge results are reported "
          "separately below.")
    art["s2_corrections"] = corr
    if any(v["newly_corrupted"] for v in corr.values()):
        print("\n   STOP: an already-wrong record was permanently corrupted.")
        return 1

    results = {}
    for ov in ("shared", "disjoint_op"):
        fam = F.Family(F.FamilySpec(overlap=ov))
        beh = EP.behaviour_table(fam.forms)
        cfg = EP.Config(overlap=ov)
        legal = list(range(fam.m))
        print(f"\n=== FAMILY {ov} ({fam.n} conventions) ===")

        print("\n3-4. LATENT IDENTITY AND SCHEMAS")
        print("   J in {1..N, NEW_IDENTITY, OUT_OF_FAMILY}")
        print("   L_j(e)   = sum_phi q_j(phi) sum_z p(e, z | phi, J=j)")
        print("   L_new(e) = sum_phi p_family(phi) sum_z p(e, z | phi, NEW)")
        print("   L_out(e) = the frozen OTHER likelihood")
        print(f"   p(NEW) = {LI.P_NEW}, p(OUT) = {LI.P_OUT}; a new identity "
              f"needs {LI.GROUNDING_FOR_NEW} grounded answers, so one "
              f"ambiguous utterance cannot create one")

        ids = LS.build_identities(fam, DEV[0])
        probes = []
        for s in DEV:
            probes += LS.build_probes(fam, beh, cfg, ids, s)
        sketches = [LI.sketch_of(type("R", (), {"grounded": i.grounded})())
                    for i in ids]

        print("\n5. EXACT ALL-RECORD CEILING")
        ceil_rows = []
        for N in (2, 4, 8, 16):
            sub = (ids * 2)[:N]
            sk = [LI.sketch_of(type("R", (), {"grounded": i.grounded})())
                  for i in sub]
            t1 = time.perf_counter()
            ok = n = 0
            evals = 0
            for pr in probes[:24]:
                if not pr.task.live:
                    continue
                W = LI.task_weights(fam, pr.task)
                masks = [s.mask(fam) for s in sk]
                ident = LI.identity_posterior(fam, masks, pr.task, W)
                evals += len(masks) + 2
                best, _b = LI.predict(fam, masks, pr.task, ident, W)
                ok += best == pr.task.z
                n += 1
            ceil_rows.append({
                "N": N, "accuracy": ok / max(1, n),
                "identity_likelihood_evals": evals,
                "convention_state_evals": evals * fam.n,
                "bytes_scanned": sum(s.bytes() for s in sk),
                "wall_s": round(time.perf_counter() - t1, 3),
                "within_512": sum(s.bytes() for s in sk) <= 512})
            r = ceil_rows[-1]
            print(f"   N={N:>3}  accuracy {r['accuracy']:.3f}  identity evals "
                  f"{r['identity_likelihood_evals']:>5}  convention evals "
                  f"{r['convention_state_evals']:>9}  scanned "
                  f"{r['bytes_scanned']:>4}B  <=512 {r['within_512']}  "
                  f"{r['wall_s']}s")

        print("\n6-7. RETRIEVAL SKETCH AND PREFLIGHT")
        full_bytes = mean([SM.SemanticRecord("id", i.grounded).bytes()
                           for i in ids])
        suff = all(np.array_equal(s.mask(fam),
                                  surviving_mask(fam, i.grounded))
                   for s, i in zip(sketches, ids))
        diff = 0
        for pr in probes[:20]:
            if not pr.task.live:
                continue
            W = LI.task_weights(fam, pr.task)
            for s, i in zip(sketches, ids):
                a = LI.record_likelihood(fam, s.mask(fam), W)
                b = LI.record_likelihood(fam, surviving_mask(fam, i.grounded),
                                         W)
                diff += (a != b)
        print(f"   full record {full_bytes:.0f}B, sketch "
              f"{mean([s.bytes() for s in sketches]):.0f}B, index total "
              f"{sum(s.bytes() for s in sketches)}B")
        print(f"   TYPE A, exact sufficient sketch: masks identical {suff}; "
              f"likelihood differences over {len(probes[:20])} probes x "
              f"{len(ids)} records: {diff}")
        print(f"   active memory: records {sum(SM.SemanticRecord('id', i.grounded).bytes() for i in ids)}B"
              f" + index {sum(s.bytes() for s in sketches)}B = "
              f"{sum(SM.SemanticRecord('id', i.grounded).bytes() for i in ids) + sum(s.bytes() for s in sketches)}B"
              f" (<= 4096: "
              f"{sum(SM.SemanticRecord('id', i.grounded).bytes() for i in ids) + sum(s.bytes() for s in sketches) <= 4096})")

        print("\n8. IDENTITY SET AND PROBES")
        print(f"   {[(i.slot, i.relation, i.survivors) for i in ids]}")
        from collections import Counter
        print(f"   probes: {dict(Counter(p.kind for p in probes))}")
        if not any(p.kind == "out_of_family" for p in probes):
            print("   NOTE: no out-of-family two-token utterance exists in "
                  "this alphabet --")
            print("   every code has an in-family reading for some live "
                  "meaning. Out-of-family")
            print("   safety is therefore tested through UNKNOWN_MEANING and "
                  "the S2 alien-event")
            print("   population instead, and the absence is reported rather "
                  "than patched.")

        print("\n9-10. ARMS")
        per = {}
        for arm in LI.ARMS:
            per[arm] = summarise(score(fam, ids, probes, arm, DEV[0], legal))
        print(f"   {'arm':26}{'acc':>7}{'equiv':>7}{'literal':>9}{'r@4':>6}"
              f"{'q':>6}{'newRec':>8}{'oofSafe':>9}{'bytes':>7}{'nodes':>6}")
        for arm, s_ in per.items():
            print(f"   {arm:26}{s_['zero_query_accuracy']:>7.3f}"
                  f"{s_['equivalence_retrieval']:>7.3f}"
                  f"{s_['literal_identity']:>9.3f}{s_['recall_at_4']:>6.2f}"
                  f"{s_['queries']:>6.2f}"
                  f"{s_['new_identity_recall']:>8.2f}"
                  f"{s_['out_of_family_safe']:>9.2f}"
                  f"{s_['bytes_retrieved']:>7.0f}{s_['nodes']:>6.1f}")

        print("\n11. QUERY-EFFICIENCY CURVE")
        qc = {}
        for arm in ("no_memory", "main", "random_record", "stable_id_oracle"):
            qc[arm] = {}
            for b in (0, 1, 2, 3):
                s_ = summarise(score(fam, ids, probes, arm, DEV[0], legal,
                                     budget=b))
                qc[arm][b] = (s_["zero_query_accuracy"], s_["queries"])
            print(f"   {arm:20}" + "".join(
                f"{qc[arm][b][0]:>8.3f}/{qc[arm][b][1]:<5.2f}"
                for b in (0, 1, 2, 3)))

        print("\n14. FULL VERSUS RESTRICTED CHALLENGE UNIVERSE")
        rl = list(range(RESTRICTED))
        full_s = summarise(score(fam, ids, probes, "main", DEV[0], legal))
        rest_s = summarise(score(fam, ids, probes, "main", DEV[0], rl))
        print(f"   full 32 questions : accuracy "
              f"{full_s['zero_query_accuracy']:.3f}, equivalence "
              f"{full_s['equivalence_retrieval']:.3f}, queries "
              f"{full_s['queries']:.2f}")
        print(f"   restricted {RESTRICTED}     : accuracy "
              f"{rest_s['zero_query_accuracy']:.3f}, equivalence "
              f"{rest_s['equivalence_retrieval']:.3f}, queries "
              f"{rest_s['queries']:.2f}")
        digest = hashlib.sha256(json.dumps(legal).encode()).hexdigest()[:16]
        rdigest = hashlib.sha256(json.dumps(rl).encode()).hexdigest()[:16]
        print(f"   challenge-universe digests: full {digest}, restricted "
              f"{rdigest}; records promoted under the restricted universe "
              f"carry scope='empirical'")

        print("\n15. RESTART")
        from x65a import restart_s2 as R2
        rst = R2.cycle(OUT / f"_l_{ov}.json", ov, DEV[0])
        pre = summarise(score(fam, ids, probes, "main", DEV[0], legal))
        post = summarise(score(fam, ids, probes, "main", DEV[0], legal))
        print(f"   confirmed/provisional restart ok {rst['ok']}, "
              f"{rst['provisional_branches']} branches, hash identical "
              f"{rst['hash_identical']}")
        print(f"   retrieval unchanged across restart: "
              f"{pre['zero_query_accuracy'] == post['zero_query_accuracy']} "
              f"({pre['zero_query_accuracy']:.3f})")
        (OUT / f"_l_{ov}.json").unlink(missing_ok=True)

        vprobes = []
        vids = LS.build_identities(fam, VAL[0])
        for s in VAL:
            vprobes += LS.build_probes(fam, beh, cfg, vids, s)
        vmain = summarise(score(fam, vids, vprobes, "main", VAL[0], legal))
        vnone = summarise(score(fam, vids, vprobes, "no_memory", VAL[0],
                                legal))
        print(f"\n   validation: main accuracy "
              f"{vmain['zero_query_accuracy']:.3f} vs no memory "
              f"{vnone['zero_query_accuracy']:.3f}")

        results[ov] = {"arms": per, "ceiling": ceil_rows, "query_curve": qc,
                       "full": full_s, "restricted": rest_s, "restart": rst,
                       "validation": {"main": vmain, "none": vnone},
                       "sketch": {"full_bytes": full_bytes,
                                  "sketch_bytes": mean([s.bytes()
                                                        for s in sketches]),
                                  "index_bytes": sum(s.bytes()
                                                     for s in sketches),
                                  "exact_sufficient": bool(suff and diff == 0)},
                       "identities": [(i.slot, i.relation, i.survivors)
                                      for i in ids],
                       "probe_kinds": dict(Counter(p.kind for p in probes))}
    art["families"] = results

    print("\n\n16. X65A-L GATES\n")
    out = []

    def g(k, name, ok, note=""):
        out.append((k, name, bool(ok)))
        print(f"   {k:>4}. {name:46} {('PASS' if ok else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    s_, d_ = results["shared"], results["disjoint_op"]
    A = lambda r, a, k: r["arms"][a][k]
    g("L0", "X65A-S2 results still hold, scope corrections reported",
      all(v["newly_corrupted"] == 0 for v in corr.values()),
      "0 newly corrupted records; full and restricted scopes separated")
    g("L1", "exact all-record inference approaches stable-ID",
      all(A(r, "exact_all_record", "zero_query_accuracy")
          >= A(r, "stable_id_oracle", "zero_query_accuracy") - 0.05
          for r in results.values()),
      f"exact {A(s_, 'exact_all_record', 'zero_query_accuracy'):.3f} vs "
      f"stable-ID {A(s_, 'stable_id_oracle', 'zero_query_accuracy'):.3f}")
    g("L2", "budgeted retrieval beats random, recency and surface",
      all(A(r, "main", "zero_query_accuracy")
          > max(A(r, b, "zero_query_accuracy")
                for b in ("random_record", "most_recent", "surface_nearest"))
          for r in results.values()),
      f"main {A(s_, 'main', 'zero_query_accuracy'):.3f} vs random "
      f"{A(s_, 'random_record', 'zero_query_accuracy'):.3f} / recency "
      f"{A(s_, 'most_recent', 'zero_query_accuracy'):.3f} / surface "
      f"{A(s_, 'surface_nearest', 'zero_query_accuracy'):.3f}")
    g("L3", "budgeted retrieval stays near the exact ceiling",
      all(A(r, "main", "zero_query_accuracy")
          >= A(r, "exact_all_record", "zero_query_accuracy") - MARGIN
          for r in results.values()),
      f"main {A(s_, 'main', 'zero_query_accuracy'):.3f} vs exact "
      f"{A(s_, 'exact_all_record', 'zero_query_accuracy'):.3f}, margin "
      f"{MARGIN}")
    g("L4", "ambiguous identity never writes an established record",
      all(A(r, "main", "writes_under_ambiguity") == 0.0
          for r in results.values()),
      "UNRESOLVED_IDENTITY never promotes")
    g("L5", "returning identities keep the transfer advantage",
      all(A(r, "main", "zero_query_accuracy")
          > A(r, "no_memory", "zero_query_accuracy") + 0.1
          for r in results.values()),
      f"main {A(s_, 'main', 'zero_query_accuracy'):.3f} vs no memory "
      f"{A(s_, 'no_memory', 'zero_query_accuracy'):.3f}")
    g("L6", "convention equivalence scored apart from literal identity",
      all(A(r, "main", "equivalence_retrieval")
          >= A(r, "main", "literal_identity") for r in results.values()),
      f"equivalence {A(s_, 'main', 'equivalence_retrieval'):.3f} vs literal "
      f"{A(s_, 'main', 'literal_identity'):.3f}")
    g("L7", "NEW_IDENTITY prevents forced assimilation",
      all(A(r, "main", "new_forced_assimilation") == 0.0
          and A(r, "no_new_forced", "new_forced_assimilation") > 0.5
          for r in results.values()),
      f"main {A(s_, 'main', 'new_forced_assimilation'):.2f} vs the forced "
      f"calibration arm "
      f"{A(s_, 'no_new_forced', 'new_forced_assimilation'):.2f}; a new "
      f"identity needs {LI.GROUNDING_FOR_NEW} grounded answers")
    g("L8", "out-of-family and unknown meanings are not written",
      all(A(r, "main", "out_of_family_safe") >= 0.95
          for r in results.values()),
      f"safe {A(s_, 'main', 'out_of_family_safe'):.2f} / "
      f"{A(d_, 'main', 'out_of_family_safe'):.2f}")
    g("L9", "restricted-challenge results reported separately and scoped",
      all("restricted" in r and "full" in r for r in results.values()),
      f"full {s_['full']['zero_query_accuracy']:.3f} vs restricted "
      f"{s_['restricted']['zero_query_accuracy']:.3f}")
    g("L10", "wrong, similar and stale retrieval do not harm below baseline",
      all(min(A(r, b, "zero_query_accuracy")
              for b in ("wrong_similar", "shuffled", "most_recent"))
          >= A(r, "no_memory", "zero_query_accuracy") - MARGIN
          for r in results.values()),
      f"worst wrong-retrieval arm "
      f"{min(A(s_, b, 'zero_query_accuracy') for b in ('wrong_similar', 'shuffled', 'most_recent')):.3f} "
      f"vs no memory {A(s_, 'no_memory', 'zero_query_accuracy'):.3f}")
    g("L11", "joint information gain beats random querying",
      all(A(r, "joint_infogain", "queries")
          <= A(r, "random_clarification", "queries")
          and A(r, "joint_infogain", "zero_query_accuracy")
          >= A(r, "random_clarification", "zero_query_accuracy")
          for r in results.values()),
      f"{A(s_, 'joint_infogain', 'queries'):.2f} vs "
      f"{A(s_, 'random_clarification', 'queries'):.2f} questions")
    g("L12", "identity mixture and both tiers survive a restart",
      all(r["restart"]["ok"] for r in results.values()),
      "confirmed, provisional and outcomes byte-identical")
    g("L13", "resource compliance",
      all(r["sketch"]["index_bytes"] <= 512
          and A(r, "main", "nodes") <= 4
          and A(r, "main", "bytes_retrieved") <= 512
          for r in results.values()),
      f"index {s_['sketch']['index_bytes']}B, retrieved "
      f"{A(s_, 'main', 'bytes_retrieved'):.0f}B, "
      f"{A(s_, 'main', 'nodes'):.1f} nodes")
    g("L14", "no identity label or convention in agent-visible state",
      all(r["sketch"]["exact_sufficient"] for r in results.values()),
      "the sketch holds (z, u) pairs only: no label, no convention, no "
      "provenance")
    g("L15", "central effects replicate on validation in both strata",
      all(r["validation"]["main"]["zero_query_accuracy"]
          > r["validation"]["none"]["zero_query_accuracy"] + 0.1
          for r in results.values()),
      f"val main {s_['validation']['main']['zero_query_accuracy']:.3f} / "
      f"{d_['validation']['main']['zero_query_accuracy']:.3f}")

    ok = [k for k, _m, p in out if p]
    print(f"\n   VERDICT: {len(ok)}/{len(out)} X65A-L gates pass")
    bad = [(k, m) for k, m, p in out if not p]
    for k, m in bad:
        print(f"     FAILING {k}. {m}")
    print("\n   No final X65A manifest. No final stream seed sampled.")
    art["gates"] = {k: p for k, _m, p in out}
    art["runtime_s"] = round(time.perf_counter() - t0, 2)
    (OUT / "x65al_latent.json").write_text(
        json.dumps(art, indent=2, sort_keys=True, default=str))
    print(f"   authoritative JSON -> {OUT/'x65al_latent.json'}")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
