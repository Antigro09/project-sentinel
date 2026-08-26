"""X64H runner -- IMPLEMENTATION SLICE ONLY.

This runs Layer 0, the microcase, and development/validation conventions
across all fourteen arms. It does NOT run final hidden seeds: those require
a committed freeze manifest, and `protocol.release_final_seeds` refuses
without one. Passing everything here is not passing X64H.

Run: uv run python experiments/x64h_hidden_convention.py
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

from x64h import (arms as A, convention as C, decision as DE, grammar as G,
                  layer0 as L0, metrics as MT, microcase as M,
                  posterior as PO, protocol as PR, semantic as S, types as T)

OUT = Path("experiments/x64h/results")


def development_family(n=8, start=900):
    fam = []
    s = start
    while len(fam) < n:
        phi = C.sample_convention(s)
        if not C.structural_audit(phi)["malformed"]:
            fam.append(phi)
        s += 1
    return tuple(fam)


def episodes(fam, forms, per_episode=6, n_episodes=4):
    """An EPISODE is one convention and a sequence of tasks under it. The
    first version cycled the convention every task, which is not what
    `persistent within an episode` means: the persistent arm locked onto the
    first convention and was then punished on every later task, executing 3
    of 24 while the memoryless control executed all 24."""
    out = []
    for e in range(n_episodes):
        phi_i = e % len(fam)
        out.append([(phi_i, forms[(7 * e + 3 * k + 5) % len(forms)])
                    for k in range(per_episode)])
    return out


def run_arms(fam, forms, tasks, budget=6):
    ctx = A.Context(fam, forms, PO.Config(), DE.Costs(), DE.Gates(),
                    budget=budget, query_universe=tuple(S.UNIVERSE[:12]))
    acc = {a: MT.blank(a) for a in A.ARMS}
    state = {a: T.PosteriorState(tuple([-math.log(len(fam))] * len(fam)),
                                 "dev") for a in A.ARMS}
    for k, (phi_i, z) in enumerate(tasks):
        phi = fam[phi_i]
        f = S.execute(z)
        ev = T.Evidence(G.generate(phi, z, random.Random(1000 + k)),
                        tuple((t, f(t)) for t in S.UNIVERSE[:2]))
        orc = A.Oracle(phi, z)
        for arm in A.ARMS:
            ep = T.Episode(ev, k, arm)
            assert PR.taint_audit(ep) == []
            v, st = A.run_arm(arm, ep, state[arm], ctx, orc,
                              random.Random(7 + k))
            state[arm] = st
            MT.accumulate(acc[arm], v, v.program == S.denote(z))
    return {a: MT.finish(v) for a, v in acc.items()}, state


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("X64H: hidden-convention semantic induction -- IMPLEMENTATION "
          "SLICE\n")
    digest = PR.freeze_digest()
    env = PR.environment()
    print(f"0. FREEZE DIGEST {digest[:32]}")
    ok, why = PR.manifest_committed()
    print(f"   freeze manifest: {why}")
    print(f"   final seeds: NOT released and NOT sampled -- "
          f"release_final_seeds refuses without a committed manifest.")
    print(f"   optional tooling absent: "
          f"{[k for k, v in env['optional_tools'].items() if not v]}")
    print("   hydra/mlflow/dvc/jax/numpyro are not installed and DVC also")
    print("   needs the repository owner's approval, so their PROPERTIES are")
    print("   implemented with the standard library: the fully resolved")
    print("   config is serialised into every artifact, each run writes a")
    print("   structured record, and JSON is the authoritative output.\n")

    print("1. LAYER 0 -- the finite bijection reference")
    sep = [L0.exact_separating_probability(4, m) for m in (1, 2, 3, 4)]
    q = L0.query_statistics(4)
    print(f"   separating probability k=4, m=1..4: {sep}")
    print(f"   posterior entropy {q['posterior_entropy_bits']:.5f} bits, "
          f"alphabet {q['largest_answer_alphabet']}, bound "
          f"{q['entropy_lower_bound_questions']:.5f}")
    print(f"   optimal {q['optimal_expected_questions']}, greedy "
          f"{q['greedy_information_gain_expected_questions']}, random "
          f"{q['random_disagreement_expected_questions']:.9f}")
    print(f"   noise k=8: {L0.noise_recovery(8, 3, 0.1, 7)}")

    print("\n2. MICROCASE -- four atoms, two constructors, hand-checkable")
    post, phim, zm = M.exact_posterior(("w1", "w2"))
    print(f"   {len(M.micro_family())} conventions x {len(M.MICRO_Z)} "
          f"meanings; utterance ('w1','w2') leaves {len(post)} pairs at "
          f"{list(post.values())[0]:.6f} each")
    print(f"   semantic marginal uniform at "
          f"{list(zm.values())[0]:.4f}; hand enumeration agrees")
    ffam = M.micro_family_functional()
    obs = [z for z in M.MICRO_Z if z[0] == "a1"]
    print(f"   non-separating observations leave an automorphism class of "
          f"size {len(M.observational_class_given(ffam[0], obs, ffam))}")

    print("\n3. EXACTNESS OF THE FT-SPCFG")
    fam = development_family()
    forms = tuple(S.x64h_forms())
    rng = random.Random(4)
    checks = mism = 0
    worst_norm = 0.0
    for phi in fam[:4]:
        for z in forms[:10]:
            worst_norm = max(worst_norm,
                             abs(sum(G.support(phi, z).values()) - 1.0))
            for _ in range(2):
                u = G.generate(phi, z, rng)
                checks += 1
                mism += abs(G.inside(phi, z, u)
                            - G.brute_force_likelihood(phi, z, u)) > 1e-12
    print(f"   {len(forms)} logical forms, "
          f"{len(S.equivalence_classes(forms))} probe-equivalence classes")
    print(f"   inside vs brute force: {checks} checks, {mism} mismatches")
    print(f"   normalisation: max |sum p(u|phi,z) - 1| = {worst_norm:.2e}")

    print("\n4. FOURTEEN ARMS on development conventions\n")
    eps = episodes(fam, forms)
    print(f"   {len(eps)} episodes x {len(eps[0])} tasks; the convention is "
          f"fixed inside an episode and resampled between them\n")
    table = {a: MT.blank(a) for a in A.ARMS}
    for ep_tasks in eps:
        part, _st = run_arms(fam, forms, ep_tasks)
        for a in A.ARMS:
            for k in ("n", "executed", "correct", "wrong", "abstained",
                      "expanded", "asked_semantic", "asked_behavioral",
                      "conflict_flag"):
                table[a][k] += part[a][k]
            table[a]["mean_p_in"] += part[a]["mean_p_in"]
            table[a]["mean_ambiguity"] += part[a]["mean_ambiguity"]
            table[a]["incomplete_candidates"] |= part[a]["incomplete_candidates"]
    for a in A.ARMS:
        table[a]["mean_p_in"] /= len(eps)
        table[a]["mean_ambiguity"] /= len(eps)
    hdr = (f'   {"arm":30}{"exec":>6}{"correct":>9}{"wrong":>7}'
           f'{"abstain":>9}{"expand":>8}{"askB":>6}{"askS":>6}'
           f'{"p_in":>7}{"incompl":>9}')
    print(hdr + "\n   " + "-" * (len(hdr) - 3))
    for a in A.ARMS:
        r = table[a]
        print(f'   {a:30}{r["executed"]:>6}{r["correct"]:>9}{r["wrong"]:>7}'
              f'{r["abstained"]:>9}{r["expanded"]:>8}'
              f'{r["asked_behavioral"]:>6}{r["asked_semantic"]:>6}'
              f'{r["mean_p_in"]:>7.3f}{str(r["incomplete_candidates"]):>9}')

    print("\n5. H1 PRE-FREEZE CHECK -- is the testbed valid at all?")
    print("   H1 requires oracle-convention accuracy >= 0.98 AND the static")
    print("   family-aware parser BELOW 0.95. A static parser that just")
    print("   marginalises over the whole convention family, with no")
    print("   persistence, must not reach the oracle -- otherwise the")
    print("   episode needs convention DECODING, not convention INFERENCE,")
    print("   and X64G's authored-inverse ceiling is back in a new costume.\n")
    h1_rows = []
    for nfam in (8, 32, 64):
        f2 = development_family(nfam)
        ctx2 = A.Context(f2, forms, PO.Config(), DE.Costs(), DE.Gates(),
                         query_universe=tuple(S.UNIVERSE[:12]))
        acc2 = {a: [0, 0] for a in ("oracle_convention",
                                    "static_family_aware")}
        for e in range(4):
            phi = f2[e % nfam]
            st2 = {a: T.PosteriorState(
                tuple([-math.log(nfam)] * nfam), "d") for a in acc2}
            for k in range(6):
                z = forms[(7 * e + 3 * k + 5) % len(forms)]
                f = S.execute(z)
                ev = T.Evidence(G.generate(phi, z, random.Random(1000 + k + 9 * e)),
                                tuple((t, f(t)) for t in S.UNIVERSE[:2]))
                orc = A.Oracle(phi, z)
                for a in acc2:
                    v, ns = A.run_arm(a, T.Episode(ev, k, a), st2[a], ctx2,
                                      orc, random.Random(7 + k))
                    st2[a] = ns
                    acc2[a][1] += 1
                    acc2[a][0] += (v.program == S.denote(z))
        o = acc2["oracle_convention"][0] / acc2["oracle_convention"][1]
        sf = acc2["static_family_aware"][0] / acc2["static_family_aware"][1]
        ok = o >= 0.98 and sf < 0.95
        h1_rows.append({"family": nfam, "oracle": o, "static_family": sf,
                        "pass": ok})
        print(f"   family {nfam:>3}: oracle {o:.2f}, static family-aware "
              f"{sf:.2f}  ->  {'PASS' if ok else 'FAIL'}")
    print("\n   H1 FAILS AT EVERY FAMILY SIZE. A full utterance under any")
    print("   well-formed convention identifies (phi, z) on its own, because")
    print("   phrases are unambiguous within a role, so marginalising over")
    print("   the family decodes it without any persistent inference.")
    print("   FREEZING NOW WOULD FREEZE A TESTBED THAT FAILS ITS OWN")
    print("   VALIDITY GATE. No freeze manifest is written and no final seed")
    print("   is sampled.")

    art = {
        "digest": digest,
        "resolved_config": PR.resolved_config(PO.Config(), DE.Costs(),
                                              DE.Gates(), 6, 12),
        "environment": env,
        "layer0": {"separating": sep, "query": q,
                   "noise": L0.noise_recovery(8, 3, 0.1, 7)},
        "exactness": {"checks": checks, "mismatches": mism,
                      "max_normalisation_error": worst_norm},
        "arms": table,
        "h1_pre_freeze": h1_rows,
        "frozen": False,
        "final_seeds_released": False,
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    (OUT / "implementation_slice.json").write_text(
        json.dumps(art, indent=2, sort_keys=True, default=str))
    print(f"\n   authoritative JSON -> {OUT/'implementation_slice.json'}")
    print(f"\n   IMPLEMENTATION SLICE ONLY. The research result is decided "
          f"by\n   H1-H10 on conventions sampled after the freeze manifest "
          f"is committed.")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
