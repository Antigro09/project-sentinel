"""X65A-S1 audit regressions.

The load-bearing test here is the unlimited-replay equivalence: if a raw
episode history cannot reconstruct the semantic posterior, the memory result
is a privileged-information result. The second is the quarantine mechanism,
which is pinned as a LIMIT rather than as a guarantee.
"""

import random
import sys

import numpy as np
import pytest

sys.path.insert(0, "experiments")

from x64h import episode as EP
from x64h import family as F
from x65a import arms_s as A
from x65a import audit_s1 as S1
from x65a import evidence as EV
from x65a import semantic_mem as SM
from x65a import streams as SR


@pytest.fixture(scope="module")
def fam():
    return F.Family(F.FamilySpec(overlap="shared"))


@pytest.fixture(scope="module")
def beh(fam):
    return EP.behaviour_table(fam.forms)


@pytest.fixture(scope="module")
def streams(fam, beh):
    cfg = EP.Config(overlap="shared")
    return [SR.build_stream(fam, beh, cfg, s) for s in range(1000, 1012)]


# -------------------------------------------- S1.1 / S1.4 equivalence

def test_unlimited_replay_reconstructs_the_semantic_posterior(fam, beh,
                                                              streams):
    d = S1.differential(fam, beh, streams)
    assert d["max_total_variation"] == 0.0
    assert d["predictive_mismatches"] == 0
    assert d["decision_mismatches"] == 0
    assert d["surviving_set_mismatches"] == 0
    assert d["equivalent"] and d["comparisons"] > 500


def test_the_semantic_arm_derives_the_meaning_rather_than_reading_it(fam, beh,
                                                                     streams):
    """S1.4. Reading `task.z` would give the main arm information no raw
    episode contains, and the equivalence above would be comparing two
    different observers."""
    arm = A.Arm("main", fam, beh, random.Random(1))
    s = streams[0]
    for app in s.appearances:
        for t in app.cal:
            live = list(range(fam.m))
            for d in t.demos:
                live = [j for j in live if beh[j][d] == beh[t.z][d]]
            want = live[0] if len(live) == 1 else None
            assert arm.derive_meaning(t) == want


def test_the_equivalence_check_is_not_vacuous(fam, beh, streams):
    """PLANTED: an unbounded arm that drops quarantine must diverge, or the
    differential is measuring nothing."""
    class NoQuarantine(S1.UnboundedReplay):
        def grounded_for(self, label):
            led = EV.EvidenceLedger()
            rec = SM.SemanticRecord(label)
            for ep in self.episodes:
                if ep["identity"] != label:
                    continue
                for i, c in enumerate(ep["cal"]):
                    live = list(range(self.fam.m))
                    for x, y in c["demos"]:
                        k = EP.UNIVERSE.index(x)
                        live = [j for j in live if self.beh[j][k] == y]
                    if len(live) != 1:
                        continue
                    bid, _ = led.absorb(EV.ExternalEvidenceKey(
                        "teacher", f"ep{ep['episode']}", i,
                        EV.observation_hash((live[0], c["u"])), label))
                    rec, _ = SM.absorb(self.fam, rec,
                                       SM.GroundedObservation(live[0],
                                                              c["u"], bid),
                                       ep["episode"], led, quarantine=False)
            return rec

    s = streams[0]
    main = A.Arm("main", fam, beh, random.Random(0))
    bad = NoQuarantine(fam, beh)
    diverged = False
    for i, app in enumerate(s.appearances):
        main.observe_episode(app, i)
        bad.observe(app)
        if 0.5 * float(np.abs(main.prior_for(app)
                              - bad.prior_for(app)).sum()) > 0:
            diverged = True
    assert diverged


# ------------------------------------------------------ S1.3 resources

def test_the_unit_table_covers_every_counted_component():
    led = A.Ledger()
    counted = {k for k in led.__dict__ if k != "wall_s"}
    named = {"posterior_evals": "posterior_eval",
             "likelihood_evals": "likelihood_eval", "queries": "query",
             "interpreter_execs": "interpreter_exec",
             "replayed_episodes": "replayed_episode",
             "serializations": "serialization",
             "archive_reads": "archive_read"}
    assert set(named) == counted
    assert set(named.values()) <= set(S1.UNIT_DEFINITION)


def test_the_ceiling_changes_what_replay_can_do(fam, beh, streams):
    rows = S1.pareto(fam, beh, streams[:3], (1000, 9000))
    low = [r for r in rows if r["arm"] == "raw_replay"
           and r["ceiling"] == 1000][0]
    high = [r for r in rows if r["arm"] == "raw_replay"
            and r["ceiling"] == 9000][0]
    assert high["delayed_return_accuracy"] > low["delayed_return_accuracy"]
    assert high["units_consumed"] > low["units_consumed"]


def test_main_is_cheaper_than_replay_at_every_ceiling(fam, beh, streams):
    rows = S1.pareto(fam, beh, streams[:3], (1000, 9000))
    for c in (1000, 9000):
        m = [r for r in rows if r["arm"] == "main" and r["ceiling"] == c][0]
        rp = [r for r in rows if r["arm"] == "raw_replay"
              and r["ceiling"] == c][0]
        assert m["units_consumed"] < rp["units_consumed"]


# ------------------------------------------------- S1.2 query efficiency

def test_the_query_curve_is_monotone_in_budget(fam, beh, streams):
    qc = S1.query_curve(fam, beh, streams[:3], ("none", "main"))
    for name, rows in qc.items():
        accs = [rows[q]["accuracy"] for q in (0, 1, 2, 3, 4)]
        assert accs == sorted(accs)


def test_memory_reaches_the_target_with_fewer_questions(fam, beh, streams):
    qc = S1.query_curve(fam, beh, streams[:3], ("none", "main"))
    qt = S1.queries_to_target(qc, 0.95)
    assert qt["main"] is not None and qt["none"] is not None
    assert qt["main"] < qt["none"]


def test_a_memoryless_arm_closes_the_accuracy_gap_with_enough_questions(
        fam, beh, streams):
    """The finding against interest, pinned so it cannot quietly vanish."""
    qc = S1.query_curve(fam, beh, streams[:3], ("none", "main"))
    assert qc["none"][0]["accuracy"] < qc["main"][0]["accuracy"] - 0.5
    assert qc["none"][3]["accuracy"] >= qc["main"][3]["accuracy"] - 0.02


# --------------------------------------------------- S1.7 the quarantine

def test_quarantine_holds_on_determined_records_and_fails_otherwise(fam, beh,
                                                                    streams):
    """Pinned as a LIMIT. Quarantine fires only when an event contradicts
    every surviving convention, so a record that is not yet determined can
    admit an alien observation consistent with a surviving non-true
    convention."""
    qs = S1.quarantine_stress(fam, beh, streams[:8], n_events=8)
    q = qs["quarantine"]
    assert q["admitted_on_determined"] == 0
    assert q["events_on_determined_records"] > 0
    assert qs["no_quarantine"]["records_corrupted"] > q["records_corrupted"]


def test_removing_quarantine_admits_everything(fam, beh, streams):
    qs = S1.quarantine_stress(fam, beh, streams[:4], n_events=8)
    n = qs["no_quarantine"]
    assert n["falsely_admitted"] == n["events_per_stream"]
    assert n["corruption_rate"] == 1.0


def test_an_alien_event_on_an_underdetermined_record_can_be_admitted(fam):
    """The failure mode, constructed directly rather than sampled."""
    led = EV.EvidenceLedger()
    rec = SM.SemanticRecord("id:x")
    z0 = 0
    u0 = int(fam.u3[0, z0])
    bid, _ = led.absorb(EV.ExternalEvidenceKey("t", "e", 0, "h", "c"))
    rec, _ = SM.absorb(fam, rec, SM.GroundedObservation(z0, u0, bid), 0, led)
    surv = SM.surviving_mask(fam, rec.grounded)
    assert int(surv.sum()) > 1                      # under-determined
    other = int(np.where(surv)[0][1])
    z1 = next(z for z in range(fam.m) if fam.u3[other, z] != fam.u3[0, z])
    bid2, _ = led.absorb(EV.ExternalEvidenceKey("u", "oof", 1, "h2", "c"))
    new, why = SM.absorb(fam, rec, SM.GroundedObservation(
        z1, int(fam.u3[other, z1]), bid2), 1, led, quarantine=True)
    assert why == "absorbed"
    assert not bool(SM.surviving_mask(fam, new.grounded)[0])


# -------------------------------------------------------- S1.6 capacity

def test_eight_identities_fit_and_sixteen_do_not(fam):
    def store_bytes(nid, nobs):
        rng = random.Random(17)
        store = SM.SemanticStore()
        for i in range(nid):
            phi = rng.randrange(fam.n)
            rec = SM.SemanticRecord(f"id:{i:012x}")
            led = EV.EvidenceLedger()
            for j in range(nobs):
                z = rng.randrange(fam.m)
                u = fam.realise(phi, z, ("O", "F", "S"))
                bid, _ = led.absorb(EV.ExternalEvidenceKey(
                    "t", f"e{i}", j, EV.observation_hash((z, u, i)), "c"))
                rec, _ = SM.absorb(fam, rec,
                                   SM.GroundedObservation(z, u, bid), j, led)
            store.put(rec)
        return store.bytes()

    assert store_bytes(8, 32) <= 4096
    assert store_bytes(16, 32) > 4096
