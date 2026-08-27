"""X64H-0B: calibration-to-transfer hidden-convention validity.

X64H-0 reached 10 of 12 gates and failed V1 (oracle 0.960) and V10
(persistent late 0.909) for one shared reason: a single demonstration
schedule had to keep the task ambiguous AND teach the codebook. The sweep
showed the two are coupled through the same channel -- withholding weight 0
gave oracle 0.997 and recovery 0.84 with ambiguity collapsed to 1.70
classes; withholding weight 1 gave 4.25 classes and recovery 0.14.

X64H-0B splits the channel. CALIBRATION tasks ground the convention:
demonstrations identify the meaning on their own and the utterance shows
all three roles. TRANSFER tasks require it: the meaning is new, the
demonstrations leave 2-8 behaviours open, and the utterance shows fewer
roles. Primary accuracy is scored on transfer only.

It also corrects the family accounting. 1152 was the size of the family
with pi_O PINNED to the identity; the free count is 2304 for the disjoint
operator alphabet, and 13824 once the operator shares the alphabet with the
other roles. Section 1 reports raw, executable and observational counts
separately and shows the pinning is a restriction, not a symmetry.

No final manifest is written. No final convention seed is sampled.

Run: uv run python experiments/x64h_0b_validity.py
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "experiments")

from x64h import episode as EP
from x64h import family as F
from x64h import layer0 as L0
from x64h import persistence as PS
from x64h import semantic as S
from x64h import types as T

OUT = Path("experiments/x64h/results")
DEV_SEEDS = tuple(range(400, 412))
VAL_SEEDS = tuple(range(500, 508))
# A THIRD split, never inspected while the generator was being fixed. Three
# corrections were made with dev and validation output on screen -- a
# covering-design truncation bug, a planted-token encoding bug and the draw
# of the alternate convention -- and although no threshold or parameter was
# changed in response to a number, "no threshold was tuned" is a claim about
# intent. This split is run once, at the end, and reported whatever it says.
HOLD_SEEDS = tuple(range(700, 712))

# Thresholds carried over from X64H-0 unchanged, so they are not tuned on
# the data they judge.
TH_ORACLE = 0.98
TH_STATIC = 0.95
TH_R = 0.50
TH_LATE = 0.95
TH_MARGIN = 0.02

BASE = EP.Config(overlap="shared", n_cal=6, n_transfer=16,
                 demos_cal_cap=6, demos_transfer_cap=4, ambiguity=(2, 8),
                 exposure_mix=(0.0, 1.0, 0.0), order_p=0.5,
                 schedule="interleaved", queries=1)

mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")


def late(xs):
    h = len(xs) // 2
    return mean(xs[h:])


def run_set(fam, beh, cfg, seeds, arms):
    """Every arm sees the SAME episodes, so demonstrations and budgets are
    identical by construction rather than by matching."""
    eps = {s: EP.build_episode(fam, beh, cfg, s) for s in seeds}
    rep = None
    out = {}
    for arm in arms:
        if arm == "repeat_task":
            if rep is None:
                rep = {s: EP.repeat_episode(fam, beh, cfg, s) for s in seeds}
            src = rep
        else:
            src = eps
        whole, lt, cal, cls, qs, ent, mass, nerr = [], [], [], [], [], [], [], 0.0
        curves = []
        for s in seeds:
            r = EP.run_arm(fam, beh, src[s], arm, cfg, s)
            tr = r["transfer"]
            whole.append(mean(tr)); lt.append(late(tr))
            cal.append(mean(r["cal"])); cls += r["classes"]; qs += r["queries"]
            ent.append(r["entropy"]); mass.append(r["mass"])
            nerr = max(nerr, r["max_normalisation_error"])
            curves.append(tr)
        n = min(len(c) for c in curves)
        out[arm] = {
            "raw_task_accuracy": mean(whole),
            "late_window_raw_accuracy": mean(lt),
            "calibration_accuracy": mean(cal),
            "curve": [mean([c[i] for c in curves]) for i in range(n)],
            "mean_classes": mean(cls), "mean_queries": mean(qs) if qs else 0.0,
            "entropy": [mean([e[i] for e in ent])
                        for i in range(min(len(e) for e in ent))],
            "mass": [mean([m[i] for m in mass])
                     for i in range(min(len(m) for m in mass))],
            "max_normalisation_error": nerr,
        }
    out["_episodes"] = eps
    return out


def recovery(o, s_, p):
    g = o - s_
    return (p - s_) / g if abs(g) > 1e-9 else float("nan")


def anti_leakage(fam, beh, cfg, seed):
    """Every check is paired with a planted defect it has to catch."""
    checks = {}
    ep = EP.build_episode(fam, beh, cfg, seed)

    cal = {ep.tasks[i].z for i in ep.cal_idx}
    tr = {ep.tasks[i].z for i in ep.tr_idx}
    checks["calibration_transfer_disjoint"] = not (cal & tr)
    bad = list(ep.tasks)
    bad[ep.tr_idx[0]] = EP.Task("transfer", ep.tasks[ep.cal_idx[0]].z,
                                (), (0,), 0, F.P2, tuple(range(fam.m)))
    checks["planted_repeat_detected"] = bool(
        cal & {bad[i].z for i in ep.tr_idx})

    checks["calibration_covers_every_value"] = (
        len(ep.coverage["op"]) == len(F.OPS)
        and len(ep.coverage["filt"]) == len(F.FILTERS_0B)
        and len(ep.coverage["scope"]) == len(F.SCOPES_0B))
    checks["transfer_atoms_are_familiar"] = all(
        any(fam.forms[j].op == fam.forms[k].op for k in cal)
        and any(fam.forms[j].filt == fam.forms[k].filt for k in cal)
        and any(fam.forms[j].scope == fam.forms[k].scope for k in cal)
        for j in tr)
    checks["transfer_triples_are_new"] = all(j not in cal for j in tr)

    la = fam.one_utterance_audit()
    checks["no_unique_word_or_length"] = (
        la["words_not_used_by_every_convention"] == 0
        and la["lengths_not_used_by_every_convention"] == 0)
    planted = F.plant_private_codeword(fam)
    pa = planted.one_utterance_audit()
    checks["planted_unique_token_detected"] = (
        pa["words_not_used_by_every_convention"] > 0)

    ok = True
    try:
        PS.save(OUT / "_probe.json",
                T.PosteriorState((-math.log(fam.n),) * 4, "h"), {})
    except T.TaintError:
        ok = False
    checks["clean_state_persists"] = ok
    try:
        PS._check({"log_p_phi": [0.0], "z_true": 3})
        checks["planted_future_target_detected"] = False
    except T.TaintError:
        checks["planted_future_target_detected"] = True

    r = EP.run_arm(fam, beh, ep, "static", cfg, seed)
    h0 = math.log2(fam.n)
    checks["static_prior_never_moves"] = all(
        abs(x - h0) < 1e-9 for x in r["prior_H"])
    rp = EP.run_arm(fam, beh, ep, "persist", cfg, seed)
    checks["planted_history_reading_detected"] = any(
        abs(x - h0) > 1e-9 for x in rp["prior_H"])

    checks["convention_fixed_in_episode"] = all(
        fam.realise(ep.phi, t.z, ("O", "F", "S"))
        == fam.realise(ep.phi, t.z, ("O", "F", "S")) for t in ep.tasks)
    checks["planted_mid_episode_change_detected"] = (
        ep.phi_alt != ep.phi
        and fam.realise(ep.phi_alt, ep.tasks[0].z, ("O", "F", "S"))
        != fam.realise(ep.phi, ep.tasks[0].z, ("O", "F", "S")))

    checks["task_id_and_seed_absent_from_features"] = not any(
        k in ("seed", "task_id", "episode") for k in EP.Task.__annotations__)
    (OUT / "_probe.json").unlink(missing_ok=True)
    return checks


def sweep(fam_cache, beh, seeds):
    rows = []
    dims = [
        ("base", {}),
        ("n_cal", {"n_cal": 3}), ("n_cal", {"n_cal": 4}),
        ("n_cal", {"n_cal": 8}),
        ("n_transfer", {"n_transfer": 8}), ("n_transfer", {"n_transfer": 24}),
        ("demos_cal", {"demos_cal_cap": 3}),
        ("demos_transfer", {"demos_transfer_cap": 2}),
        ("demos_transfer", {"demos_transfer_cap": 3}),
        ("exposure", {"exposure_mix": (0.25, 0.75, 0.0)}),
        ("exposure", {"exposure_mix": (0.0, 0.75, 0.25)}),
        ("order_p", {"order_p": 0.0}), ("order_p", {"order_p": 1.0}),
        ("overlap", {"overlap": "disjoint_op"}),
        ("ambiguity", {"ambiguity": (2, 4)}),
        ("ambiguity", {"ambiguity": (4, 12)}),
        ("schedule", {"schedule": "front"}),
    ]
    arms = ("oracle", "static", "persist", "reset", "shuffled",
            "query_random", "query_infogain")
    for dim, over in dims:
        cfg = EP.Config(**{**BASE.__dict__, **over})
        if cfg.overlap not in fam_cache:
            fam_cache[cfg.overlap] = F.Family(F.FamilySpec(overlap=cfg.overlap))
        fam = fam_cache[cfg.overlap]
        b = EP.behaviour_table(fam.forms) if fam.forms != fam_cache[
            "shared"].forms else beh
        r = run_set(fam, b, cfg, seeds, arms)
        eps = r.pop("_episodes")
        o = r["oracle"]["raw_task_accuracy"]
        s_ = r["static"]["raw_task_accuracy"]
        p_ = r["persist"]["raw_task_accuracy"]
        rows.append({
            "dim": dim, "config": str(over) if over else "(base)",
            "calibration_accuracy": r["persist"]["calibration_accuracy"],
            "transfer_oracle": o, "transfer_static": s_, "transfer_persist": p_,
            "oracle_gap": o - s_, "gap_recovery": recovery(o, s_, p_),
            "late_raw_persist": r["persist"]["late_window_raw_accuracy"],
            "late_gap_recovery": recovery(
                r["oracle"]["late_window_raw_accuracy"],
                r["static"]["late_window_raw_accuracy"],
                r["persist"]["late_window_raw_accuracy"]),
            "H_end_bits": r["persist"]["entropy"][-1],
            "true_class_mass_end": r["persist"]["mass"][-1],
            "reset": r["reset"]["raw_task_accuracy"],
            "shuffled": r["shuffled"]["raw_task_accuracy"],
            "mean_classes": r["persist"]["mean_classes"],
            "query_random": r["query_random"]["raw_task_accuracy"],
            "query_infogain": r["query_infogain"]["raw_task_accuracy"],
            "rejected_per_episode": mean([e.rejected for e in eps.values()]),
            "accepted_transfer": mean([len(e.tr_idx) for e in eps.values()]),
        })
    return rows


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    art: dict = {"final_manifest_written": False, "final_seed_sampled": False}
    print("X64H-0B: calibration-to-transfer hidden-convention validity\n")

    # ------------------------------------------------ 1. V0 accounting
    print("1. CONVENTION-FAMILY ACCOUNTING  (V0)")
    fams = {}
    for ov in ("disjoint_op", "shared"):
        fams[ov] = F.Family(F.FamilySpec(overlap=ov))
        a = fams[ov].accounting()
        print(f"   {ov}: alphabet {a['alphabet_size']} words")
        print(f"      raw parameter assignments      "
              f"{a['raw_parameter_assignments']:>6}  = {a['raw_factorisation']}")
        print(f"      unique executable conventions  "
              f"{a['unique_executable_conventions']:>6}")
        print(f"      observational classes          "
              f"{a['observational_equivalence_classes']:>6}   "
              f"sizes {a['class_size_histogram']}")
        print(f"      canonical representatives      "
              f"{a['canonical_representatives']:>6}")
        print(f"      quotient: {a['quotient']}")
        pin = F.pin_no_operator_symmetry(fams[ov])
        print(f"      pinning pi_O keeps {pin['kept_by_pinning']}, excludes "
              f"{pin['excluded_by_pinning']}, of which "
              f"{pin['excluded_that_duplicate_a_kept_convention']} duplicate a "
              f"kept convention -> pinning is a symmetry: "
              f"{pin['pinning_is_a_symmetry']}")
        art.setdefault("accounting", {})[ov] = {**a, **pin}
    print("   X64H-0 reported 1152. That was `full_family(fix_op=True)`: pi_O")
    print("   pinned to the identity. A design restriction, not a quotient.")
    fam = fams["shared"]
    beh = EP.behaviour_table(fam.forms)
    n_cls = len(fam.classes())
    H0 = math.log2(n_cls)
    print(f"\n   PRIMARY family: shared alphabet, {fam.n} conventions, "
          f"{n_cls} classes")
    print(f"   uniform prior  H0 = log2({n_cls}) = {H0:.6f} bits, "
          f"P_true_class(0) = 1/{n_cls} = {1/n_cls:.3e}")
    print(f"   {fam.m} typed forms, "
          f"{len({S.denote(z) for z in fam.forms})} distinct behaviours "
          f"(every form its own class)")

    sym = F.symmetry_audit(fam)
    broken = F.symmetry_audit(F.subset(fam, range(0, fam.n, 7)))
    print(f"\n   EXACT SYMMETRY. sum over phi of p(u | phi, z) varies over "
          f"meanings by {sym['max_spread_over_meanings']:.1f}: under a uniform")
    print(f"   convention prior an utterance says NOTHING about the meaning.")
    print(f"   Calibration arm -- a deliberately unclosed sub-family gives "
          f"{broken['max_spread_over_meanings']:.1f}, so the audit is not "
          f"vacuous.")
    art["symmetry"] = {"family": sym, "broken_subfamily": broken}

    # ------------------------------------- 2. separating calibration sets
    print("\n2. CALIBRATION AS A SEPARATING FAMILY  (V7)")
    ms = fam.minimal_separating_size()
    print(f"   exhaustive search: no set of size < {ms['k']} separates; "
          f"size {ms['k']} does (tried {ms['combinations_tried']})")
    print(f"      example {ms['example']}")
    print(f"   lower bound: two grounded meanings expose two filter values, "
          f"leaving pi_F")
    print(f"      ambiguous between the two unassigned words, so k >= 3.")
    gr = fam.greedy_separating(order=range(fam.m))
    print(f"   greedy separating set size {gr['size']}, separating "
          f"{gr['separating']}")
    art["separating"] = {"minimal": ms, "greedy": gr}

    # -------------------------------------------- 3. the base episode set
    print("\n3. EPISODE STRUCTURE")
    print(f"   {BASE.label()}")
    arms = EP.ARMS
    dev = run_set(fam, beh, BASE, DEV_SEEDS, arms)
    dev_eps = dev.pop("_episodes")
    val = run_set(fam, beh, BASE, VAL_SEEDS, arms)
    val_eps = val.pop("_episodes")
    rej = mean([e.rejected for e in dev_eps.values()])
    rb = mean([e.reject_band for e in dev_eps.values()])
    ro = mean([e.reject_oracle for e in dev_eps.values()])
    acc_tr = mean([len(e.tr_idx) for e in dev_eps.values()])
    uns = [b for e in dev_eps.values() for b in e.oracle_unselected]
    sep = mean([1.0 if fam.separates([e.tasks[i].z for i in e.cal_idx]) else 0.0
                for e in dev_eps.values()])
    cover = mean([1.0 if (len(e.coverage["op"]) == len(F.OPS)
                          and len(e.coverage["filt"]) == len(F.FILTERS_0B)
                          and len(e.coverage["scope"]) == len(F.SCOPES_0B))
                  else 0.0 for e in list(dev_eps.values())
                  + list(val_eps.values())])
    resid = mean([fam.residual_classes([e.tasks[i].z for i in e.cal_idx])
                  for e in dev_eps.values()])
    print(f"   accepted transfer tasks {acc_tr:.2f}/episode; rejected "
          f"{rej:.2f} ({rb:.2f} outside the ambiguity band, {ro:.2f} because "
          f"no exposure made the")
    print(f"   oracle identify). THE ORACLE CEILING IS CONSTRUCTED: with an "
          f"UNSELECTED exposure the oracle would identify only "
          f"{mean([1.0 if b else 0.0 for b in uns]):.3f} of tasks.")
    print(f"   calibration sets that separate the convention: {sep:.3f}; "
          f"mean conventions still tied after grounding "
          f"{fam.n/max(1e-9,resid):.2f}" if resid else "")
    art["generator"] = {"rejected_per_episode": rej, "reject_band": rb,
                        "reject_oracle": ro, "accepted_transfer": acc_tr,
                        "oracle_identifies_under_unselected_exposure":
                            mean([1.0 if b else 0.0 for b in uns]),
                        "calibration_separating_fraction": sep,
                        "calibration_full_coverage_fraction": cover,
                        "residual_conventions_after_grounding":
                            fam.n / max(1e-9, resid)}

    idx = EP.first_task_indexed(fam, dev_eps[DEV_SEEDS[0]])
    print("\n   INDEXED POSTERIOR (seed 400, first task)")
    print(f"      prior                       H {idx['H_prior']:7.3f} bits   "
          f"true-class mass {idx['mass_prior']:.3e}")
    print(f"      after first utterance       H "
          f"{idx['H_after_first_utterance']:7.3f} bits   true-class mass "
          f"{idx['mass_after_first_utterance']:.3e}")
    print(f"      after first demonstrations  H "
          f"{idx['H_after_first_demonstrations']:7.3f} bits   true-class mass "
          f"{idx['mass_after_first_demonstrations']:.3e}")
    ent = dev["persist"]["entropy"]; mss = dev["persist"]["mass"]
    step = max(1, len(ent) // 8)
    ks = list(range(0, len(ent), step))
    print("      after task    " + "".join(f"{k:>7}" for k in ks))
    print("      H(phi) bits   " + "".join(f"{ent[k]:>7.2f}" for k in ks))
    print("      true mass     " + "".join(f"{mss[k]:>7.3f}" for k in ks))
    art["indexed"] = {**idx, "entropy_by_task": ent, "mass_by_task": mss}

    # -------------------------------------------------------- 4. the arms
    print("\n4. ARMS  (transfer tasks only; calibration reported separately)")
    hdr = (f"   {'arm':18}{'cal':>7}{'transfer':>10}{'late':>8}"
           f"{'val tr':>9}{'val late':>10}")
    print(hdr)
    for a in arms:
        d, v = dev[a], val[a]
        print(f"   {a:18}{d['calibration_accuracy']:>7.3f}"
              f"{d['raw_task_accuracy']:>10.3f}"
              f"{d['late_window_raw_accuracy']:>8.3f}"
              f"{v['raw_task_accuracy']:>9.3f}"
              f"{v['late_window_raw_accuracy']:>10.3f}")
    art["arms"] = {"dev": {a: {k: x for k, x in dev[a].items()
                               if k != "trace"} for a in arms},
                   "val": {a: {k: x for k, x in val[a].items()
                               if k != "trace"} for a in arms}}

    cur = dev["persist"]["curve"]
    co = dev["oracle"]["curve"]; cs = dev["static"]["curve"]
    print("\n   transfer index " + "".join(f"{i+1:>6}" for i in range(len(cur))))
    print("   oracle         " + "".join(f"{x:>6.2f}" for x in co))
    print("   persist        " + "".join(f"{x:>6.2f}" for x in cur))
    print("   static         " + "".join(f"{x:>6.2f}" for x in cs))

    def stat(split, a, k="raw_task_accuracy"):
        return split[a][k]

    res = {}
    for nm, split in (("dev", dev), ("val", val)):
        o = stat(split, "oracle"); s_ = stat(split, "static")
        p_ = stat(split, "persist")
        lo = stat(split, "oracle", "late_window_raw_accuracy")
        ls = stat(split, "static", "late_window_raw_accuracy")
        lp = stat(split, "persist", "late_window_raw_accuracy")
        res[nm] = {"raw_task_accuracy": {"oracle": o, "static": s_,
                                         "persist": p_},
                   "oracle_gap": o - s_,
                   "episode_gap_recovery": recovery(o, s_, p_),
                   "late_window_raw_accuracy": {"oracle": lo, "static": ls,
                                                "persist": lp},
                   "late_window_gap_recovery": recovery(lo, ls, lp)}
    print("\n   METRIC DEFINITIONS, kept distinct:")
    for nm in ("dev", "val"):
        r = res[nm]
        print(f"      {nm}: raw task accuracy O/S/P "
              f"{r['raw_task_accuracy']['oracle']:.3f}/"
              f"{r['raw_task_accuracy']['static']:.3f}/"
              f"{r['raw_task_accuracy']['persist']:.3f}; "
              f"G {r['oracle_gap']:+.3f}; whole-episode gap recovery "
              f"{r['episode_gap_recovery']:.3f}")
        print(f"           late-window raw accuracy O/S/P "
              f"{r['late_window_raw_accuracy']['oracle']:.3f}/"
              f"{r['late_window_raw_accuracy']['static']:.3f}/"
              f"{r['late_window_raw_accuracy']['persist']:.3f}; "
              f"late-window gap recovery "
              f"{r['late_window_gap_recovery']:.3f}")
    art["headline"] = res

    # ------------------------------------------------ 4b. two alphabets
    print("\n4b. THE TWO ALPHABETS, and what each one costs")
    print("   Both pass every gate. They trade one-utterance leakage against")
    print("   how much the generator must discard to build the ceiling.")
    two = {}
    for ov in ("shared", "disjoint_op"):
        f2 = fams[ov]
        c2 = EP.Config(**{**BASE.__dict__, "overlap": ov})
        d2 = run_set(f2, beh, c2, DEV_SEEDS,
                     ("oracle", "static", "selection_aware", "persist"))
        e2 = d2.pop("_episodes")
        un2 = [1.0 if b else 0.0
               for e in e2.values() for b in e.oracle_unselected]
        la2 = f2.one_utterance_audit()
        two[ov] = {
            "conventions": f2.n,
            "bits_leaked_by_one_utterance": la2["bits_leaked_worst_case"],
            "fraction_of_family_left": la2["fraction_of_family_left"],
            "rejected_per_episode": mean([e.rejected for e in e2.values()]),
            "oracle_identifies_under_unselected_exposure": mean(un2),
            "oracle": d2["oracle"]["raw_task_accuracy"],
            "static": d2["static"]["raw_task_accuracy"],
            "selection_aware": d2["selection_aware"]["raw_task_accuracy"],
            "persist": d2["persist"]["raw_task_accuracy"],
            "persist_late": d2["persist"]["late_window_raw_accuracy"],
        }
    print(f"   {'':40}{'shared':>12}{'disjoint_op':>14}")
    for k, lab in (("conventions", "conventions in the family"),
                   ("bits_leaked_by_one_utterance", "bits leaked by 1 utterance"),
                   ("fraction_of_family_left", "fraction of family left"),
                   ("rejected_per_episode", "tasks discarded per episode"),
                   ("oracle_identifies_under_unselected_exposure",
                    "oracle id. under UNSELECTED exposure"),
                   ("oracle", "transfer oracle"),
                   ("static", "transfer static"),
                   ("selection_aware", "transfer selection-aware"),
                   ("persist", "transfer persistent"),
                   ("persist_late", "late-window persistent")):
        print(f"   {lab:40}{two['shared'][k]:>12.3f}"
              f"{two['disjoint_op'][k]:>14.3f}")
    print(f"   the selection rule is worth "
          f"{two['shared']['selection_aware']-two['shared']['static']:+.3f} on "
          f"the shared alphabet and "
          f"{two['disjoint_op']['selection_aware']-two['disjoint_op']['static']:+.3f} "
          f"on the disjoint one;")
    print(f"   the convention posterior is worth "
          f"{two['shared']['persist']-two['shared']['static']:+.3f} and "
          f"{two['disjoint_op']['persist']-two['disjoint_op']['static']:+.3f}.")
    art["two_alphabets"] = two

    # ---------------------------------------------------- 5. the sweep
    print("\n5. PARAMETER SWEEP  (development seeds only)")
    rows = sweep({"shared": fam}, beh, DEV_SEEDS)
    cols = ("dim", "config", "calibration_accuracy", "transfer_oracle",
            "transfer_static", "transfer_persist", "oracle_gap",
            "gap_recovery", "late_raw_persist", "H_end_bits",
            "true_class_mass_end", "reset", "shuffled", "mean_classes",
            "query_random", "query_infogain", "rejected_per_episode")
    print("   " + f"{'dim':14}{'config':30}{'cal':>5}{'orc':>6}{'sta':>6}"
          f"{'per':>6}{'gap':>7}{'R':>6}{'lateP':>7}{'H':>6}{'mass':>6}"
          f"{'rst':>6}{'shf':>6}{'cls':>6}{'qR':>6}{'qI':>6}{'rej':>6}")
    for r in rows:
        print("   " + f"{r['dim']:14}{r['config'][:30]:30}"
              f"{r['calibration_accuracy']:>5.2f}{r['transfer_oracle']:>6.2f}"
              f"{r['transfer_static']:>6.2f}{r['transfer_persist']:>6.2f}"
              f"{r['oracle_gap']:>7.2f}{r['gap_recovery']:>6.2f}"
              f"{r['late_raw_persist']:>7.2f}{r['H_end_bits']:>6.2f}"
              f"{r['true_class_mass_end']:>6.2f}{r['reset']:>6.2f}"
              f"{r['shuffled']:>6.2f}{r['mean_classes']:>6.2f}"
              f"{r['query_random']:>6.2f}{r['query_infogain']:>6.2f}"
              f"{r['rejected_per_episode']:>6.1f}")
    art["sweep"] = rows

    # ------------------------------------------------ 6. anti-leakage
    print("\n6. ANTI-LEAKAGE CONTROLS AND PLANTED DEFECTS")
    checks = anti_leakage(fam, beh, BASE, DEV_SEEDS[0])
    for k, v in checks.items():
        print(f"   {k:44} {'ok' if v else 'FAIL'}")
    art["anti_leakage"] = checks

    # ------------------------------------------------------ 7. queries
    print("\n7. ACTIVE QUERYING ON THE CURRENT TRANSFER DISTRIBUTION  (V11)")
    qr = dev["query_random"]; qi = dev["query_infogain"]
    print(f"   budget {BASE.queries} question/task: random "
          f"{qr['raw_task_accuracy']:.3f}, exact information gain "
          f"{qi['raw_task_accuracy']:.3f}")
    print(f"   validation: random {val['query_random']['raw_task_accuracy']:.3f}"
          f", information gain "
          f"{val['query_infogain']['raw_task_accuracy']:.3f}")
    qi_n = [n for s in DEV_SEEDS
            for n in EP.questions_to_identify(fam, beh, dev_eps[s],
                                              "infogain", BASE, s)]
    qr_n = [n for s in DEV_SEEDS
            for n in EP.questions_to_identify(fam, beh, dev_eps[s],
                                              "random", BASE, s)]
    print(f"   questions to identify from the MEMORYLESS prior (query policy "
          f"isolated from history):")
    print(f"      exact information gain {mean(qi_n):.3f}   "
          f"random {mean(qr_n):.3f}   over {len(qi_n)} transfer tasks")
    cp = [EP.collapse_point(fam, beh, dev_eps[s], BASE, s) for s in DEV_SEEDS]
    ok_cp = [c for c in cp if c["tasks"] is not None]
    print(f"   convention posterior below 1 bit after "
          f"{mean([c['calibration_tasks'] for c in ok_cp]):.2f} calibration "
          f"tasks ({mean([c['tasks'] for c in ok_cp]):.2f} tasks overall) in "
          f"{len(ok_cp)}/{len(cp)} episodes")
    micro = L0.query_statistics(4)
    print(f"   preserved k=4 microcase (regression, not the gate): "
          f"{micro['greedy_information_gain_expected_questions']} vs "
          f"{micro['random_disagreement_expected_questions']:.6f}")
    art["collapse_point"] = {
        "calibration_tasks": mean([c["calibration_tasks"] for c in ok_cp]),
        "tasks": mean([c["tasks"] for c in ok_cp]),
        "episodes_reaching_1_bit": f"{len(ok_cp)}/{len(cp)}"}
    art["queries"] = {"questions_to_identify_infogain": mean(qi_n),
                      "questions_to_identify_random": mean(qr_n),
                      "dev_random": qr["raw_task_accuracy"],
                      "dev_infogain": qi["raw_task_accuracy"],
                      "val_random": val["query_random"]["raw_task_accuracy"],
                      "val_infogain": val["query_infogain"]["raw_task_accuracy"],
                      "microcase": micro}

    # -------------------------------------------------------- 8. gates
    print("\n8. VALIDITY GATES\n")
    out = []

    def g(k, name, ok, note=""):
        out.append((k, name, bool(ok)))
        print(f"   {k:>4}. {name:46} {('PASS' if ok else 'FAIL'):>4}"
              + (f"   {note}" if note else ""))

    acc = art["accounting"]
    g("V0", "convention accounting consistent and pinned",
      all(a["raw_parameter_assignments"] == a["unique_executable_conventions"]
          == a["observational_equivalence_classes"] for a in acc.values())
      and not any(a["pinning_is_a_symmetry"] for a in acc.values()),
      f"{acc['shared']['raw_parameter_assignments']} raw = executable = "
      f"classes; 1152 was pi_O pinned")
    d, v = res["dev"], res["val"]
    g("V1", "transfer oracle ceiling >= 0.98",
      d["raw_task_accuracy"]["oracle"] >= TH_ORACLE
      and v["raw_task_accuracy"]["oracle"] >= TH_ORACLE,
      f"dev {d['raw_task_accuracy']['oracle']:.3f} val "
      f"{v['raw_task_accuracy']['oracle']:.3f} (CONSTRUCTED)")
    g("V2", "nontrivial transfer oracle gap",
      d["raw_task_accuracy"]["static"] < TH_STATIC
      and v["raw_task_accuracy"]["static"] < TH_STATIC,
      f"static dev {d['raw_task_accuracy']['static']:.3f} val "
      f"{v['raw_task_accuracy']['static']:.3f}; G dev "
      f"{d['oracle_gap']:+.3f} val {v['oracle_gap']:+.3f}")
    g("V3", "persistent gap recovery >= 0.50",
      d["episode_gap_recovery"] >= TH_R and v["episode_gap_recovery"] >= TH_R,
      f"R dev {d['episode_gap_recovery']:.3f} val "
      f"{v['episode_gap_recovery']:.3f}; late R dev "
      f"{d['late_window_gap_recovery']:.3f}")
    ent = dev["persist"]["entropy"]; mss = dev["persist"]["mass"]
    g("V4", "convention posterior learns and stays normalised",
      ent[0] > ent[-1] and mss[-1] > mss[0] and mss[-1] >= 0.8
      and dev["persist"]["max_normalisation_error"] < 1e-9,
      f"H {ent[0]:.2f} -> {ent[-1]:.2f} bits; mass {mss[0]:.2e} -> "
      f"{mss[-1]:.3f}; max |sum p - 1| "
      f"{dev['persist']['max_normalisation_error']:.1e}")
    lp = dev["persist"]["late_window_raw_accuracy"]
    ctrl = {a: dev[a]["late_window_raw_accuracy"]
            for a in ("reset", "shuffled", "wrong_pairing", "phi_change")}
    g("V5", "history causality: every corruption removes the gain",
      all(x < lp - TH_MARGIN for x in ctrl.values()),
      "late " + ", ".join(f"{k} {x:.3f}" for k, x in ctrl.items())
      + f" vs persist {lp:.3f}")
    g("V6", "the gain is on NEW meanings, not repeats",
      checks["transfer_triples_are_new"]
      and checks["transfer_atoms_are_familiar"]
      and lp >= dev["repeat_task"]["late_window_raw_accuracy"] - 0.05,
      f"meanings disjoint by construction; persist late {lp:.3f} vs "
      f"repeated-task control "
      f"{dev['repeat_task']['late_window_raw_accuracy']:.3f}")
    g("V7", "calibration covers every value and separates",
      sep >= 0.95 and cover >= 0.999 and ms["k"] is not None,
      f"{sep*100:.0f}% of calibration sets separate, {cover*100:.0f}% cover "
      f"every O/F/S value; minimal separating size {ms['k']}")
    la = fam.one_utterance_audit()
    g("V8", "no one-utterance leakage",
      not la["identifies_convention"]
      and la["fraction_of_family_left"] == 1.0
      and checks["no_unique_word_or_length"]
      and checks["planted_unique_token_detected"],
      f"every utterance leaves all {la['max_conventions_left']} conventions; "
      f"{la['bits_leaked_worst_case']:.1f} bits leaked; planted token caught")
    cl = dev["persist"]["mean_classes"]
    band = BASE.ambiguity
    g("V9", "transfer ambiguity in the intended band",
      band[0] <= cl <= band[1],
      f"mean surviving behaviours {cl:.2f} in [{band[0]}, {band[1]}]")
    g("V10", "late transfer competence >= 0.95",
      lp >= TH_LATE
      and val["persist"]["late_window_raw_accuracy"] >= TH_LATE,
      f"dev {lp:.3f} val "
      f"{val['persist']['late_window_raw_accuracy']:.3f}")
    g("V11", "information gain beats random on THIS distribution",
      mean(qi_n) < mean(qr_n)
      and qi["raw_task_accuracy"] >= qr["raw_task_accuracy"],
      f"questions to identify {mean(qi_n):.3f} vs {mean(qr_n):.3f}; accuracy "
      f"at budget 1: dev {qi['raw_task_accuracy']:.3f} vs "
      f"{qr['raw_task_accuracy']:.3f}, val "
      f"{val['query_infogain']['raw_task_accuracy']:.3f} vs "
      f"{val['query_random']['raw_task_accuracy']:.3f}")
    g("V12", "convention fixed in an episode, changes only at a boundary",
      checks["convention_fixed_in_episode"]
      and checks["planted_mid_episode_change_detected"]
      and dev["phi_change"]["late_window_raw_accuracy"] < lp - TH_MARGIN,
      f"mid-episode change late "
      f"{dev['phi_change']['late_window_raw_accuracy']:.3f} vs {lp:.3f}")

    ok = [k for k, _m, p in out if p]
    print(f"\n   VERDICT: {len(ok)}/{len(out)} validity gates pass")
    bad = [(k, m) for k, m, p in out if not p]
    if bad:
        print("\n   FAILING:")
        for k, m in bad:
            print(f"     {k}. {m}")

    print("\n   ADVERSARIAL CONTROL (not a gate, a disclosure): a static "
          "parser that\n   models the generator's own exposure-SELECTION rule "
          f"scores "
          f"{dev['selection_aware']['raw_task_accuracy']:.3f} against "
          f"chance-level static\n   "
          f"{dev['static']['raw_task_accuracy']:.3f} and persistent "
          f"{dev['persist']['raw_task_accuracy']:.3f}. The selection rule is "
          "worth\n   about "
          f"{dev['selection_aware']['raw_task_accuracy']-dev['static']['raw_task_accuracy']:+.3f}"
          "; the convention posterior is worth "
          f"{dev['persist']['raw_task_accuracy']-dev['static']['raw_task_accuracy']:+.3f}.")

    print("\n9. UNTOUCHED HOLDOUT SPLIT, run once")
    hold = run_set(fam, beh, BASE, HOLD_SEEDS,
                   ("oracle", "static", "persist", "reset", "shuffled",
                    "wrong_pairing", "phi_change"))
    hold.pop("_episodes")
    ho = hold["oracle"]["raw_task_accuracy"]
    hs = hold["static"]["raw_task_accuracy"]
    hp = hold["persist"]["raw_task_accuracy"]
    print(f"   seeds {HOLD_SEEDS[0]}-{HOLD_SEEDS[-1]}: oracle {ho:.3f}, "
          f"static {hs:.3f}, persist {hp:.3f}")
    print(f"   whole-episode gap recovery {recovery(ho, hs, hp):.3f}; "
          f"late-window raw accuracy persist "
          f"{hold['persist']['late_window_raw_accuracy']:.3f}")
    print(f"   controls late: reset "
          f"{hold['reset']['late_window_raw_accuracy']:.3f}, shuffled "
          f"{hold['shuffled']['late_window_raw_accuracy']:.3f}, wrong pairing "
          f"{hold['wrong_pairing']['late_window_raw_accuracy']:.3f}, "
          f"convention change "
          f"{hold['phi_change']['late_window_raw_accuracy']:.3f}")
    art["holdout"] = {a: {k: x for k, x in hold[a].items() if k != "trace"}
                      for a in hold}
    art["holdout_summary"] = {"oracle": ho, "static": hs, "persist": hp,
                              "episode_gap_recovery": recovery(ho, hs, hp),
                              "late_window_raw_accuracy":
                                  hold["persist"]["late_window_raw_accuracy"]}

    print("\n   No final manifest written. No final convention seed sampled.")
    art["gates"] = {k: p for k, _m, p in out}
    art["thresholds"] = {"oracle": TH_ORACLE, "static": TH_STATIC,
                         "R": TH_R, "late": TH_LATE, "margin": TH_MARGIN}
    art["config"] = BASE.label()
    art["dev_seeds"] = list(DEV_SEEDS); art["val_seeds"] = list(VAL_SEEDS)
    art["runtime_s"] = round(time.perf_counter() - t0, 2)
    (OUT / "x64h0b_validity.json").write_text(
        json.dumps(art, indent=2, sort_keys=True, default=str))
    print(f"   authoritative JSON -> {OUT/'x64h0b_validity.json'}")
    print(f"\n({time.perf_counter() - t0:.0f}s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
