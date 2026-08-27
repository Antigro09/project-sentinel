"""X65A-S1: the audit that has to pass before latent identity is attempted.

The central question is whether the semantic sufficient statistic contains
ANYTHING that an unbounded replay of the raw episodes could not reconstruct.
If it does, the "structured memory" result is really a privileged-information
result and X65A-L would inherit the flaw.

The unbounded diagnostic keeps every externally observed episode, every
demonstration, every clarification and every public outcome, with no byte,
retrieval or compute cap, and re-derives the posterior using the SAME
grounding algorithm the semantic arm uses -- including quarantine. The
earlier 0.898-versus-0.989 gap came from the budgeted replay arm lacking
quarantine, not from lacking information.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from x64h import episode as EP

from . import arms_s as A
from . import evidence as EV
from . import semantic_mem as SM

# One compute unit is exactly one of:
UNIT_DEFINITION = {
    "posterior_eval": "one exact joint evaluation over the whole convention "
                      "family for one task",
    "likelihood_eval": "one task-likelihood evaluation",
    "query": "one clarification question asked and answered",
    "interpreter_exec": "one trusted execution of one candidate form on one "
                        "input tape",
    "replayed_episode": "one stored episode read back and re-derived",
    "serialization": "one canonical serialize or deserialize of memory",
    "archive_read": "one lookup into persisted memory",
}


class UnboundedReplay:
    """Everything, forever, with no caps -- and the same grounding rule."""

    def __init__(self, fam, beh):
        self.fam, self.beh = fam, beh
        self.episodes: list = []
        self.ledger = A.Ledger()

    def observe(self, app) -> None:
        self.episodes.append({
            "identity": app.label, "episode": app.index, "kind": app.kind,
            "cal": [{"u": t.u,
                     "demos": [[EP.UNIVERSE[d], self.beh[t.z][d]]
                               for d in t.demos]}
                    for t in app.cal],
            "transfer": [{"u": t.u, "live": list(t.live)}
                         for t in app.transfer],
            "questions": [], "answers": [], "outcomes": [],
        })
        self.ledger.serializations += 1

    def grounded_for(self, label: str):
        led = EV.EvidenceLedger()
        rec = SM.SemanticRecord(label)
        for ep in self.episodes:
            if ep["identity"] != label:
                continue
            self.ledger.replayed_episodes += 1
            for i, c in enumerate(ep["cal"]):
                live = list(range(self.fam.m))
                for x, y in c["demos"]:
                    k = EP.UNIVERSE.index(x)
                    self.ledger.interpreter_execs += len(live)
                    live = [j for j in live if self.beh[j][k] == y]
                if len(live) != 1:
                    continue
                bid, _ = led.absorb(EV.ExternalEvidenceKey(
                    "teacher", f"ep{ep['episode']}", i,
                    EV.observation_hash((live[0], c["u"])), label))
                rec, _why = SM.absorb(self.fam, rec,
                                      SM.GroundedObservation(live[0], c["u"],
                                                             bid),
                                      ep["episode"], led, quarantine=True)
        return rec

    def prior_for(self, app):
        rec = self.grounded_for(app.label)
        if not rec.grounded:
            return np.full(self.fam.n, 1.0 / self.fam.n)
        return SM.prior_from(self.fam, rec.grounded)


def differential(fam, beh, streams, verbose=False) -> dict:
    """Compare posterior_from_full_raw_history with
    posterior_from_semantic_sufficient_statistic at every task and identity."""
    tv_max = 0.0
    pred_mismatch = dec_mismatch = compared = 0
    grounded_mismatch = 0
    for stream in streams:
        main = A.Arm("main", fam, beh, random.Random(0))
        ub = UnboundedReplay(fam, beh)
        for i, app in enumerate(stream.appearances):
            main.observe_episode(app, i)
            ub.observe(app)
            pm = main.prior_for(app)
            pu = ub.prior_for(app)
            tv = 0.5 * float(np.abs(pm - pu).sum())
            tv_max = max(tv_max, tv)
            rec = main.store.get(app.label)
            urec = ub.grounded_for(app.label)
            if rec is not None:
                a = SM.surviving_mask(fam, rec.grounded)
                b = SM.surviving_mask(fam, urec.grounded)
                if not np.array_equal(a, b):
                    grounded_mismatch += 1
            for t in app.transfer:
                compared += 1
                ba, _c, da = EP._infer_by("aware", fam, pm, t.u, t.pool,
                                          t.live, t.tie)
                bb, _c2, db = EP._infer_by("aware", fam, pu, t.u, t.pool,
                                           t.live, t.tie)
                if not np.allclose(ba, bb, atol=0, rtol=0):
                    pred_mismatch += 1
                if da != db:
                    dec_mismatch += 1
    return {"streams": len(streams), "comparisons": compared,
            "max_total_variation": tv_max,
            "predictive_mismatches": pred_mismatch,
            "decision_mismatches": dec_mismatch,
            "surviving_set_mismatches": grounded_mismatch,
            "equivalent": (tv_max == 0.0 and pred_mismatch == 0
                           and dec_mismatch == 0
                           and grounded_mismatch == 0)}


def query_curve(fam, beh, streams, arms, budgets=(0, 1, 2, 3, 4)) -> dict:
    out: dict = {}
    for name in arms:
        rows = {}
        for q in budgets:
            ok = n = 0
            used = 0
            for stream in streams:
                arm = A.Arm(name, fam, beh, random.Random(1))
                for i, app in enumerate(stream.appearances):
                    arm.observe_episode(app, i)
                    p = arm.prior_for(app)
                    if app.kind != "return":
                        continue
                    for t in app.transfer:
                        c, u = A.solve(fam, beh, p, t, arm.ledger, q)
                        ok += c
                        used += u
                        n += 1
            rows[q] = {"accuracy": ok / max(1, n),
                       "mean_queries": used / max(1, n)}
        out[name] = rows
    return out


def queries_to_target(curve: dict, target: float) -> dict:
    """The central semantic-memory metric: how many clarification questions
    each arm needs to reach a frozen target accuracy."""
    out = {}
    for name, rows in curve.items():
        hit = next((q for q in sorted(rows) if rows[q]["accuracy"] >= target),
                   None)
        out[name] = hit
    return out


def pareto(fam, beh, streams, ceilings) -> list:
    rows = []
    for ceil in ceilings:
        for name in ("raw_replay", "main"):
            acc = []
            units = []
            reps = []
            pev = []
            iexec = []
            abytes = []
            wall = []
            for stream in streams:
                t0 = time.perf_counter()
                arm = A.Arm(name, fam, beh, random.Random(1),
                            compute_ceiling=ceil)
                ok = n = 0
                for i, app in enumerate(stream.appearances):
                    arm.observe_episode(app, i)
                    p = arm.prior_for(app)
                    if app.kind != "return":
                        continue
                    for t in app.transfer:
                        c, _ = A.solve(fam, beh, p, t, arm.ledger, 0)
                        ok += c
                        n += 1
                acc.append(ok / max(1, n))
                units.append(arm.ledger.total_units())
                reps.append(arm.ledger.replayed_episodes)
                pev.append(arm.ledger.posterior_evals)
                iexec.append(arm.ledger.interpreter_execs)
                abytes.append(arm.active_bytes())
                wall.append(time.perf_counter() - t0)
            m = lambda x: sum(x) / len(x)
            rows.append({"ceiling": ceil, "arm": name,
                         "delayed_return_accuracy": m(acc),
                         "units_consumed": m(units),
                         "replayed_episodes": m(reps),
                         "posterior_evals": m(pev),
                         "interpreter_execs": m(iexec),
                         "active_bytes": m(abytes), "wall_s": m(wall)})
    return rows


def quarantine_stress(fam, beh, streams, n_events: int = 8) -> dict:
    """Many independently generated out-of-family events, not one."""
    per_stream = []
    for stream in streams:
        rng = random.Random(stream.seed * 7 + 5)
        for quarantine in (True, False):
            arm = A.Arm("main" if quarantine else "main_no_quarantine",
                        fam, beh, random.Random(1))
            for i, app in enumerate(stream.appearances):
                if app.kind == "unknown":
                    continue
                arm.observe_episode(app, i)
            admitted = corrupted = quarantined = resolved = 0
            # WHEN does quarantine fail? It fires only when an event
            # contradicts EVERY surviving convention. While a record is
            # under-determined, an alien observation consistent with a
            # surviving non-true convention is admitted and can eliminate
            # the truth. Split the events by that condition rather than
            # reporting one rate.
            det_seen = det_adm = und_seen = und_adm = 0
            led = EV.EvidenceLedger()
            for e in range(n_events):
                k = e % 8
                lab = stream.identities[k].label
                rec = arm.store.get(lab)
                if rec is None or not rec.grounded:
                    continue
                phi = stream.phis[k]
                alien = rng.randrange(fam.n)
                z = rng.randrange(fam.m)
                tries = 0
                while fam.u3[alien, z] == fam.u3[phi, z] and tries < 50:
                    alien = rng.randrange(fam.n)
                    tries += 1
                surv_before = int(SM.surviving_mask(fam,
                                                    rec.grounded).sum())
                bid, _ = led.absorb(EV.ExternalEvidenceKey(
                    "unknown", f"oof{e}", e,
                    EV.observation_hash((z, int(fam.u3[alien, z]))), lab))
                new, why = SM.absorb(fam, rec,
                                     SM.GroundedObservation(
                                         z, int(fam.u3[alien, z]), bid),
                                     100 + e, led, quarantine=quarantine)
                if surv_before == 1:
                    det_seen += 1
                    det_adm += (why == "absorbed")
                else:
                    und_seen += 1
                    und_adm += (why == "absorbed")
                if why == "contradiction_quarantined":
                    quarantined += 1
                elif why == "absorbed":
                    admitted += 1
                arm.store.put(new)
                if not bool(SM.surviving_mask(fam, new.grounded)[phi]):
                    corrupted += 1
                else:
                    # a later consistent observation restores CONFIRMED
                    z2 = rng.randrange(fam.m)
                    bid2, _ = led.absorb(EV.ExternalEvidenceKey(
                        "teacher", f"fix{e}", e,
                        EV.observation_hash((z2, int(fam.u3[phi, z2]))), lab))
                    fixed, why2 = SM.absorb(
                        fam, new, SM.GroundedObservation(
                            z2, int(fam.u3[phi, z2]), bid2), 200 + e, led,
                        quarantine=quarantine)
                    if why2 == "absorbed":
                        resolved += 1
                    arm.store.put(fixed)
            per_stream.append({"quarantine": quarantine, "events": n_events,
                               "quarantined": quarantined,
                               "falsely_admitted": admitted,
                               "records_corrupted": corrupted,
                               "later_resolved": resolved,
                               "events_on_determined_records": det_seen,
                               "admitted_on_determined": det_adm,
                               "events_on_underdetermined_records": und_seen,
                               "admitted_on_underdetermined": und_adm})
    out = {}
    for q in (True, False):
        rows = [r for r in per_stream if r["quarantine"] is q]
        m = lambda k: sum(r[k] for r in rows) / len(rows)
        out["quarantine" if q else "no_quarantine"] = {
            "streams": len(rows), "events_per_stream": n_events,
            "quarantined": m("quarantined"),
            "falsely_admitted": m("falsely_admitted"),
            "records_corrupted": m("records_corrupted"),
            "later_resolved": m("later_resolved"),
            "corruption_rate": m("records_corrupted") / n_events,
            "events_on_determined_records": sum(
                r["events_on_determined_records"] for r in rows),
            "admitted_on_determined": sum(
                r["admitted_on_determined"] for r in rows),
            "events_on_underdetermined_records": sum(
                r["events_on_underdetermined_records"] for r in rows),
            "admitted_on_underdetermined": sum(
                r["admitted_on_underdetermined"] for r in rows),
            "per_stream_corruption": [r["records_corrupted"] for r in rows]}
    return out
