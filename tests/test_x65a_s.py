"""X65A-S regressions: evidence identity, opaque identities, the exact
sufficient statistic, stream construction, the arms, and the restart.

Latent identity, procedural memory, general retrieval, revision and
consolidation are out of scope for this phase and have no tests here,
deliberately.
"""

import json
import random
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, "experiments")

from x64h import episode as EP
from x64h import family as F
from x65a import arms_s as A
from x65a import evidence as EV
from x65a import identity as ID
from x65a import posterior as PO
from x65a import restart_s as RS
from x65a import semantic_mem as SM
from x65a import streams as SR
from x65a.types import Status, TaintError, byte_cost


@pytest.fixture(scope="module")
def fam():
    return F.Family(F.FamilySpec(overlap="shared"))


@pytest.fixture(scope="module")
def beh(fam):
    return EP.behaviour_table(fam.forms)


@pytest.fixture(scope="module")
def stream(fam, beh):
    return SR.build_stream(fam, beh, EP.Config(overlap="shared"), 400)


# ------------------------------------------------- 1. evidence identity

def _lik():
    return {0: Fraction(4, 5), 1: Fraction(1, 5)}


def _key(seq=0, content=7, ep="ep0"):
    return EV.ExternalEvidenceKey("teacher", ep, seq,
                                  EV.observation_hash({"u": content}), "ctx")


def test_the_same_event_counts_once():
    p = PO.ExactPosterior.uniform((0, 1))
    k = _key()
    one = p.update([PO.Likelihood(k.base_id(), True, _lik())])
    twice = p.update([PO.Likelihood(k.base_id(), True, _lik()),
                      PO.Likelihood(_key().base_id(), True, _lik())])
    assert one.q == twice.q == {0: Fraction(4, 5), 1: Fraction(1, 5)}


def test_a_new_caller_supplied_memory_id_does_not_create_evidence():
    """Identity comes from the EVENT, not from the id a caller passes."""
    led = EV.EvidenceLedger()
    k = _key()
    bid, new1 = led.absorb(k)
    _bid2, new2 = led.absorb(_key())
    led.reference(bid, "semantic-entry-uuid-A")
    led.reference(bid, "episodic-entry-uuid-B")
    assert new1 and not new2
    assert led.contribution_count(bid) == 1
    assert len(led.references[bid]) == 2


def test_two_independent_events_with_equal_content_count_twice():
    """Content is not identity."""
    p = PO.ExactPosterior.uniform((0, 1))
    a, b = _key(seq=0), _key(seq=1)
    assert a.observation_hash == b.observation_hash
    assert a.base_id() != b.base_id()
    out = p.update([PO.Likelihood(a.base_id(), True, _lik()),
                    PO.Likelihood(b.base_id(), True, _lik())])
    assert out.q[0] == Fraction(16, 17)


def test_a_summary_of_an_event_adds_no_factor():
    p = PO.ExactPosterior.uniform((0, 1))
    k = _key()
    base = PO.Likelihood(k.base_id(), True, _lik())
    assert p.update([base]).q == p.update(
        [base, PO.deterministic_summary(base)]).q


def test_an_event_reintroduced_after_restart_still_counts_once():
    p = PO.ExactPosterior.uniform((0, 1))
    k = _key()
    after = p.update([PO.Likelihood(k.base_id(), True, _lik())])
    revived = PO.ExactPosterior(after.states, dict(after.q), after.absorbed)
    assert revived.update([PO.Likelihood(k.base_id(), True, _lik())]).q \
        == after.q


def test_referencing_an_unabsorbed_event_is_refused():
    with pytest.raises(TaintError):
        EV.EvidenceLedger().reference("ev:deadbeef", "entry")


# ------------------------------------------------ 2. opaque identities

def test_the_honest_assigner_is_exactly_independent_of_the_convention():
    r = ID.functional_independence(ID.assign, range(400, 420), 8,
                                   list(range(2000)))
    assert r["zero"] and r["I_label_convention_bits_exact"] == 0.0
    assert r["seeds_where_output_changed"] == 0


@pytest.mark.parametrize("plant", [ID.assign_leaky_index,
                                   ID.assign_leaky_token,
                                   ID.assign_leaky_order])
def test_every_planted_identity_leak_is_caught(plant):
    r = ID.functional_independence(plant, range(400, 420), 8,
                                   list(range(2000)))
    assert r["depends_on_convention"]
    assert r["seeds_where_output_changed"] == r["seeds_tested"]


def test_the_probe_set_refuses_to_be_degenerate():
    """A constant convention vector induces one ranking, and an order leak
    escapes. The first version of this audit was handed exactly that."""
    with pytest.raises(ValueError):
        ID.probe_vectors(8, [5] * 8, count=8)      # no ranking diversity
    with pytest.raises(ValueError):
        ID.probe_vectors(8, [5], count=8)          # pool too small
    v = ID.probe_vectors(8, list(range(2000)))
    assert len({tuple(sorted(range(8), key=lambda i: x[i])) for x in v}) >= 6


def test_labels_are_remapped_between_seeds():
    a = {i.label for i in ID.assign(400, 8, [0] * 8)}
    b = {i.label for i in ID.assign(401, 8, [0] * 8)}
    assert not (a & b)


# --------------------------------------- 3/4. the exact sufficient statistic

def test_minimize_preserves_the_surviving_set_exactly(fam):
    rng = random.Random(3)
    phi = rng.randrange(fam.n)
    g = tuple(SM.GroundedObservation(z, fam.realise(phi, z, ("O", "F", "S")),
                                     f"e{z}")
              for z in rng.sample(range(fam.m), 8))
    small = SM.minimize(fam, g)
    assert len(small) < len(g)
    assert np.array_equal(SM.surviving_mask(fam, g),
                          SM.surviving_mask(fam, small))
    assert np.array_equal(SM.prior_from(fam, g), SM.prior_from(fam, small))


def test_the_statistic_saturates_rather_than_growing(fam):
    rng = random.Random(5)
    phi = rng.randrange(fam.n)
    rec = SM.SemanticRecord("id:x")
    led = EV.EvidenceLedger()
    sizes = []
    for j in range(32):
        z = rng.randrange(fam.m)
        u = fam.realise(phi, z, ("O", "F", "S"))
        k = EV.ExternalEvidenceKey("t", "ep", j, EV.observation_hash((z, u)),
                                   "c")
        bid, _ = led.absorb(k)
        rec, _ = SM.absorb(fam, rec, SM.GroundedObservation(z, u, bid), j, led)
        sizes.append(len(rec.grounded))
    assert max(sizes) <= 4
    assert sizes[-1] == sizes[len(sizes) // 2]


def test_the_record_pins_the_true_convention(fam):
    rng = random.Random(7)
    phi = rng.randrange(fam.n)
    rec = SM.SemanticRecord("id:x")
    led = EV.EvidenceLedger()
    for j, z in enumerate(rng.sample(range(fam.m), 6)):
        u = fam.realise(phi, z, ("O", "F", "S"))
        bid, _ = led.absorb(EV.ExternalEvidenceKey(
            "t", "ep", j, EV.observation_hash((z, u)), "c"))
        rec, _ = SM.absorb(fam, rec, SM.GroundedObservation(z, u, bid), j, led)
    m = SM.surviving_mask(fam, rec.grounded)
    assert int(m.sum()) == 1 and bool(m[phi])


def test_a_contradicting_event_quarantines_rather_than_overwrites(fam):
    rng = random.Random(11)
    phi = rng.randrange(fam.n)
    alien = (phi + 1) % fam.n
    led = EV.EvidenceLedger()
    rec = SM.SemanticRecord("id:x")
    for j, z in enumerate(rng.sample(range(fam.m), 4)):
        u = fam.realise(phi, z, ("O", "F", "S"))
        bid, _ = led.absorb(EV.ExternalEvidenceKey(
            "t", "ep", j, EV.observation_hash((z, u)), "c"))
        rec, _ = SM.absorb(fam, rec, SM.GroundedObservation(z, u, bid), j, led)
    z = next(z for z in range(fam.m)
             if fam.u3[alien, z] != fam.u3[phi, z])
    bad = SM.GroundedObservation(z, int(fam.u3[alien, z]), "ev:alien")
    kept, why = SM.absorb(fam, rec, bad, 9, led, quarantine=True)
    assert why == "contradiction_quarantined"
    assert kept.status is Status.QUARANTINED
    assert bool(SM.surviving_mask(fam, kept.grounded)[phi])
    loose, why2 = SM.absorb(fam, rec, bad, 9, led, quarantine=False)
    assert why2 == "absorbed"
    assert not bool(SM.surviving_mask(fam, loose.grounded)[phi])


def test_a_duplicate_event_is_not_absorbed_twice(fam):
    rec = SM.SemanticRecord("id:x")
    led = EV.EvidenceLedger()
    o = SM.GroundedObservation(0, int(fam.u3[0, 0]), "ev:one")
    rec, w1 = SM.absorb(fam, rec, o, 0, led)
    rec2, w2 = SM.absorb(fam, rec, o, 1, led)
    assert w1 == "absorbed" and w2 == "duplicate_event"
    assert rec2 is rec


def test_the_store_stays_inside_the_preregistered_budget(fam):
    rng = random.Random(13)
    store = SM.SemanticStore()
    led = EV.EvidenceLedger()
    for i in range(8):
        phi = rng.randrange(fam.n)
        rec = SM.SemanticRecord(f"id:{i:012x}")
        for j in range(32):
            z = rng.randrange(fam.m)
            u = fam.realise(phi, z, ("O", "F", "S"))
            bid, _ = led.absorb(EV.ExternalEvidenceKey(
                "t", f"ep{i}", j, EV.observation_hash((z, u, i)), "c"))
            rec, _ = SM.absorb(fam, rec, SM.GroundedObservation(z, u, bid),
                               j, led)
        store.put(rec)
    assert not store.over_budget()
    assert store.bytes() <= 4096


# --------------------------------------------------- 6. stream structure

def test_calibration_and_transfer_meanings_never_overlap(stream):
    assert SR.schedule_summary(stream)["meaning_overlap_cal_vs_transfer"] == 0


def test_every_appearance_uses_new_complete_meanings(stream):
    seen: dict = {}
    for a in stream.appearances:
        s = seen.setdefault(a.identity, set())
        for t in a.transfer:
            assert t.z not in s
            s.add(t.z)


def test_returns_are_separated_by_long_gaps(stream):
    s = SR.schedule_summary(stream)
    assert s["returns"] >= 16 and s["long_gap_returns"] >= 8
    assert 24 <= s["episodes"] <= 48


def test_the_unknown_event_genuinely_disagrees(fam, stream):
    unk = [a for a in stream.appearances if a.kind == "unknown"][0]
    z = unk.cal[0].z
    assert fam.u3[unk.phi, z] != fam.u3[stream.phis[unk.identity], z]


def test_a_near_convention_differs_in_exactly_one_component(fam):
    rng = random.Random(2)
    phi = rng.randrange(fam.n)
    near = SR.near_convention(fam, phi, rng)
    d = (int((fam.PO[near] != fam.PO[phi]).any())
         + int((fam.PF[near] != fam.PF[phi]).any())
         + int((fam.PS[near] != fam.PS[phi]).any())
         + int(fam.ORD[near] != fam.ORD[phi]))
    assert d == 1


# --------------------------------------------------------- 7/8. the arms

def test_raw_replay_stores_the_episode_not_the_derived_meaning(fam, beh,
                                                               stream):
    """Storing `z` would make this semantic memory with extra steps, which
    is what the first version of this arm accidentally was."""
    arm = A.Arm("raw_replay", fam, beh, random.Random(1))
    for i, app in enumerate(stream.appearances[:3]):
        arm.observe_episode(app, i)
    blob = json.dumps(arm.raw)
    assert arm.raw and '"z"' not in blob
    assert '"demos"' in blob and '"u"' in blob


def test_raw_replay_pays_to_re_derive_what_memory_stored(fam, beh, stream):
    raw = A.Arm("raw_replay", fam, beh, random.Random(1))
    main = A.Arm("main", fam, beh, random.Random(1))
    for i, app in enumerate(stream.appearances):
        raw.observe_episode(app, i)
        main.observe_episode(app, i)
        raw.prior_for(app)
        main.prior_for(app)
    assert raw.ledger.total_units() > 2 * main.ledger.total_units()


def test_the_compute_ceiling_actually_binds(fam, beh, stream):
    capped = A.Arm("raw_replay", fam, beh, random.Random(1),
                   compute_ceiling=100)
    for i, app in enumerate(stream.appearances):
        capped.observe_episode(app, i)
        capped.prior_for(app)
    assert capped.truncated_replays > 0


def test_every_arm_sees_the_same_tasks(fam, beh, stream):
    tasks = {a.index: [t.z for t in a.transfer] for a in stream.appearances}
    for name in ("none", "main", "raw_replay", "oracle"):
        arm = A.Arm(name, fam, beh, random.Random(1))
        for i, app in enumerate(stream.appearances):
            arm.observe_episode(app, i)
            assert [t.z for t in app.transfer] == tasks[app.index]


def test_only_the_declared_control_gets_a_larger_query_budget(fam, beh):
    for name in A.ARMS:
        arm = A.Arm(name, fam, beh, random.Random(1))
        assert arm.query_budget() == (4 if name == "bigger_query_budget"
                                      else 1)


def test_main_beats_no_memory_on_returning_identities(fam, beh, stream):
    got = {}
    for name in ("none", "main"):
        arm = A.Arm(name, fam, beh, random.Random(1))
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
        got[name] = ok / n
    assert got["main"] > got["none"] + 0.5


# ------------------------------------------------------------ 13. restart

def test_a_genuine_restart_preserves_the_store_and_the_effect(tmp_path):
    r = RS.cycle(tmp_path / "s.json", "shared", 400)
    assert r["ok"], r
    assert r["parent_pid_gone"] and r["parent_pid"] != r["child_pid"]
    assert r["audit_hash_identical"]
    assert r["sufficient_statistic_identical"]
    assert r["forbidden_channel_closed"]
    assert r["post_restart_return_transfer"] > 0.9
    assert r["env_size"] <= 8
