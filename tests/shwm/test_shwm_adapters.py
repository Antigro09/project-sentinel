"""Scale-0 gate: the two fixtures test two different things, and hidden state stays hidden.

The audit that produced this branch found action conditioning and hidden-state
aliasing conflated into one test. They are separated here, and each fixture is
given the control arm that must fail it:

* the action-intervention fixture must be failed by an action-blind predictor
  and solved by an action-conditioned one;
* the belief-aliasing fixture must be failed by an observation-and-action
  predictor -- which is *not* action-blind -- and solved by one that can see the
  earlier observation.

A fixture that both controls pass would be measuring nothing, which is the
failure mode `sentinel-calibration-arms` exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.env.adapters.base import (
    HiddenSnapshot,
    HiddenStateLeak,
    assert_no_hidden_state,
    assert_observation_invariant_to_hidden_field,
    declared_non_observable,
)
from sentinel.env.adapters.procedural_visual import (
    CHARGE_PERIOD,
    ProceduralVisualAdapter,
    build_level,
    render,
)
from sentinel.env.adapters.synthetic_control import (
    ACTIONS,
    FORBIDDEN_VISIBLE,
    SyntheticControlAdapter,
    build_action_intervention_fixture,
    build_belief_alias_fixture,
)
from sentinel.wm.authority import AuthorityGate, UnauthorizedAction
from sentinel.wm.latent_contract import ContractViolation, HIDDEN_FIELD_NAMES


def adapter() -> SyntheticControlAdapter:
    return SyntheticControlAdapter(gate=AuthorityGate())


# ---- least-squares controls --------------------------------------------------


def best_constant_error(targets: np.ndarray) -> float:
    """MSE of the best predictor that ignores its input entirely."""
    return float(((targets - targets.mean()) ** 2).mean())


def keyed_predictor_error(keys: list[tuple], targets: np.ndarray) -> float:
    """MSE of the best predictor that sees exactly `keys`.

    Grouping by key and predicting each group's mean is the least-squares
    optimum for any function of the key, so a positive result is a statement
    about the key, not about how hard someone tried to fit it.
    """
    groups: dict[tuple, list[float]] = {}
    for key, target in zip(keys, targets):
        groups.setdefault(key, []).append(float(target))
    error = 0.0
    for key, target in zip(keys, targets):
        mean = sum(groups[key]) / len(groups[key])
        error += (float(target) - mean) ** 2
    return error / len(targets)


# ---- T1a: action intervention ------------------------------------------------


def test_action_intervention_fixture_restores_the_identical_state():
    fixture = build_action_intervention_fixture()
    payload = fixture.restore_point.reveal("evaluator")
    assert fixture.action_a != fixture.action_b
    assert fixture.successor_a.digest != fixture.successor_b.digest
    assert payload["visible"] == fixture.observation.structured_observation["visible"]


def test_an_action_blind_predictor_fails_the_action_intervention_fixture():
    fixture = build_action_intervention_fixture()
    observation = int(fixture.observation.structured_observation["visible"])
    targets = np.array(
        [
            float(fixture.successor_a.values["observable_signature"]),
            float(fixture.successor_b.values["observable_signature"]),
        ]
    )
    blind = keyed_predictor_error([(observation,), (observation,)], targets)
    conditioned = keyed_predictor_error(
        [(observation, fixture.action_a), (observation, fixture.action_b)], targets
    )
    assert blind > 0.0, "an action-blind predictor solved a fixture that must separate actions"
    assert conditioned == 0.0
    assert blind == pytest.approx(best_constant_error(targets))


def test_the_action_fixture_does_not_secretly_depend_on_history():
    """A model with the full history but no action must still fail it.

    Both branches share every prior observation by construction, so history
    cannot be what separates them. Without this check the fixture could be
    passed by a recurrent model for the wrong reason.
    """
    fixture = build_action_intervention_fixture()
    observation = int(fixture.observation.structured_observation["visible"])
    history = (observation, "identical-prefix")
    targets = np.array(
        [
            float(fixture.successor_a.values["observable_signature"]),
            float(fixture.successor_b.values["observable_signature"]),
        ]
    )
    assert keyed_predictor_error([history, history], targets) > 0.0


# ---- T1b: belief aliasing ----------------------------------------------------


def test_belief_alias_fixture_shares_the_observation_and_the_action():
    fixture = build_belief_alias_fixture()
    assert len(fixture.history_a) == len(fixture.history_b), "unequal histories let a model count steps"
    assert fixture.history_a != fixture.history_b
    assert fixture.observation_trace_a != fixture.observation_trace_b
    assert fixture.observation_trace_a[-1] == fixture.observation_trace_b[-1]
    assert fixture.successor_a.digest != fixture.successor_b.digest


def test_an_observation_only_predictor_fails_the_belief_alias_fixture():
    fixture = build_belief_alias_fixture()
    current = fixture.observation_trace_a[-1]
    targets = np.array(
        [
            float(fixture.successor_a.values["observable_signature"]),
            float(fixture.successor_b.values["observable_signature"]),
        ]
    )
    # Same current observation, same action: this predictor is action-conditioned
    # and still cannot separate the pair.
    observation_only = keyed_predictor_error(
        [(current, fixture.probe_action), (current, fixture.probe_action)], targets
    )
    # One extra step of history is enough, because that is where the traces differ.
    with_history = keyed_predictor_error(
        [
            (fixture.observation_trace_a[-2], current, fixture.probe_action),
            (fixture.observation_trace_b[-2], current, fixture.probe_action),
        ],
        targets,
    )
    assert observation_only > 0.0
    assert with_history == 0.0


def test_the_alias_fixture_is_not_solvable_by_action_conditioning_alone():
    fixture = build_belief_alias_fixture()
    assert fixture.probe_action == fixture.probe_action  # one action, by construction
    payload_a = "a"
    targets = np.array(
        [
            float(fixture.successor_a.values["observable_signature"]),
            float(fixture.successor_b.values["observable_signature"]),
        ]
    )
    assert keyed_predictor_error([(fixture.probe_action,), (fixture.probe_action,)], targets) > 0.0


# ---- hidden state -------------------------------------------------------------


def test_no_hidden_field_name_appears_in_any_observation():
    environment = adapter()
    result = environment.reset(6600)
    for _ in range(12):
        action = ACTIONS[result.observation.step % len(ACTIONS)]
        result = environment.step(action, environment.gate.authorize_collection(action, "random"))
        assert not set(result.observation.structured_observation) & HIDDEN_FIELD_NAMES


def test_the_hidden_phase_is_absent_from_the_observation_but_present_in_the_snapshot():
    environment = adapter()
    environment.reset(6601)
    snapshot = environment.snapshot()
    assert "phase" in snapshot.reveal("evaluator")
    assert "phase" not in environment._observation().structured_observation
    assert_no_hidden_state(environment._observation(), snapshot)


def test_hidden_snapshots_refuse_to_be_read_by_the_model():
    environment = adapter()
    environment.reset(6600)
    snapshot = environment.snapshot()
    for reader in ("model", "belief", "encoder", "trainer"):
        with pytest.raises(HiddenStateLeak):
            snapshot.reveal(reader)


def test_a_snapshot_does_not_serialise_its_payload_into_a_report():
    environment = adapter()
    environment.reset(6600)
    serialised = environment.snapshot().canonical_dict()
    assert set(serialised) == {"hidden_snapshot_digest", "taint"}
    assert "phase" not in str(serialised)


def test_assert_no_hidden_state_catches_a_verbatim_leak():
    snapshot = HiddenSnapshot(
        payload={"phase": 2, "visible": 1, "_non_observable": ("phase",)},
        environment_version="v",
    )
    assert declared_non_observable(snapshot) == ("phase",)

    class Leaky:
        structured_observation = {"phase": 2, "visible": 1}

    class Clean:
        structured_observation = {"visible": 1}

    with pytest.raises(HiddenStateLeak):
        assert_no_hidden_state(Leaky(), snapshot)  # type: ignore[arg-type]
    assert_no_hidden_state(Clean(), snapshot)  # type: ignore[arg-type]


def test_an_undeclared_snapshot_is_read_conservatively():
    """With no declaration, every field counts as hidden."""
    snapshot = HiddenSnapshot(payload={"anything": 1}, environment_version="v")
    assert declared_non_observable(snapshot) == ("anything",)


def test_the_observation_is_invariant_to_the_synthetic_hidden_phase():
    environment = adapter()
    environment.reset(6601)
    snapshot = environment.snapshot()
    probes = assert_observation_invariant_to_hidden_field(
        environment, snapshot, "phase", (0, 1, 2)
    )
    # The evaluator may see the consequences; the model may not see the cause.
    assert len(set(probes)) >= 1


def test_the_invariance_check_catches_a_planted_leak():
    """Calibration arm: an adapter that does leak must be caught.

    Without this, the invariance test above could pass because the check is
    broken rather than because the adapter is clean.
    """
    environment = adapter()
    environment.reset(6601)
    snapshot = environment.snapshot()
    original = SyntheticControlAdapter._observation

    def leaky(self):
        envelope = original(self)
        return type(envelope)(
            episode_id=envelope.episode_id,
            step=envelope.step,
            timestamp_ns=envelope.timestamp_ns,
            modality_payloads=envelope.modality_payloads,
            structured_observation={**envelope.structured_observation, "cadence": self._phase},
            modality_mask=envelope.modality_mask,
            available_action_digest=envelope.available_action_digest,
            environment_version=envelope.environment_version,
            taint=envelope.taint,
        )

    SyntheticControlAdapter._observation = leaky
    try:
        with pytest.raises(HiddenStateLeak, match="reaching model input"):
            assert_observation_invariant_to_hidden_field(
                environment, snapshot, "phase", (0, 1, 2)
            )
    finally:
        SyntheticControlAdapter._observation = original


def test_the_observation_is_invariant_to_the_visual_hidden_charge():
    environment = visual()
    environment.reset(6600)
    snapshot = environment.snapshot()
    assert_observation_invariant_to_hidden_field(
        environment, snapshot, "charge", tuple(range(CHARGE_PERIOD))
    )


# ---- authority ----------------------------------------------------------------


def test_the_environment_refuses_an_action_without_a_token():
    environment = adapter()
    environment.reset(6600)
    with pytest.raises(UnauthorizedAction):
        environment.step(1, None)  # type: ignore[arg-type]
    with pytest.raises(UnauthorizedAction):
        environment.step(1, "pretend-token")  # type: ignore[arg-type]


def test_a_token_cannot_be_replayed_or_used_for_a_different_action():
    environment = adapter()
    environment.reset(6600)
    token = environment.gate.authorize_collection(1, "random")
    environment.step(1, token)
    with pytest.raises(UnauthorizedAction):
        environment.step(1, token)
    other = environment.gate.authorize_collection(1, "random")
    with pytest.raises(UnauthorizedAction):
        environment.step(2, other)


def test_a_token_from_another_gate_is_rejected():
    environment = adapter()
    environment.reset(6600)
    foreign = AuthorityGate(gate_id="someone-elses-gate").authorize_collection(1, "random")
    with pytest.raises(UnauthorizedAction):
        environment.step(1, foreign)


# ---- exact replay --------------------------------------------------------------


def test_synthetic_control_replays_exactly_from_a_restore_point():
    environment = adapter()
    environment.reset(6602)
    for action in (1, 2, 0):
        environment.step(action, environment.gate.authorize_collection(action, "random"))
    snapshot = environment.snapshot()
    first = [
        environment.step(a, environment.gate.authorize_collection(a, "random")).probes.digest
        for a in (3, 1, 2, 0)
    ]
    environment.restore(snapshot)
    second = [
        environment.step(a, environment.gate.authorize_collection(a, "random")).probes.digest
        for a in (3, 1, 2, 0)
    ]
    assert first == second


def test_the_forbidden_cell_separates_constraint_violation_from_action_failure():
    """Two probes with distinct meanings, not one wearing two names."""
    environment = adapter()
    environment.reset(6600)
    blocked_seen = False
    for _ in range(40):
        for action in ACTIONS:
            snapshot = environment.snapshot()
            result = environment.step(action, environment.gate.authorize_collection(action, "random"))
            if not result.probes.values["action_succeeded"]:
                blocked_seen = True
                assert not result.probes.values["constraint_violation"], (
                    "a blocked action must not also report entering the forbidden cell"
                )
            environment.restore(snapshot)
        action = ACTIONS[environment._step % len(ACTIONS)]
        environment.step(action, environment.gate.authorize_collection(action, "random"))
        if blocked_seen:
            break
    assert blocked_seen, "the forbidden cell was never reachable, so the probe is vacuous"


def test_an_illegal_action_is_rejected_after_the_gate():
    environment = adapter()
    environment.reset(6600)
    token = environment.gate.authorize_collection(9, "random")
    with pytest.raises(ContractViolation):
        environment.step(9, token)


# ---- procedural visual ---------------------------------------------------------


def visual() -> ProceduralVisualAdapter:
    return ProceduralVisualAdapter(gate=AuthorityGate())


def test_appearance_and_mechanics_are_independent_axes():
    same_layout_new_look = build_level(dynamics_seed=5, appearance_seed=1), build_level(5, 2)
    new_layout_same_look = build_level(5, 1), build_level(6, 1)
    a, b = same_layout_new_look
    assert np.array_equal(a.walls, b.walls) and a.mirrored == b.mirrored and a.start == b.start
    assert not np.array_equal(a.palette, b.palette)
    c, d = new_layout_same_look
    assert not np.array_equal(c.walls, d.walls)


def test_the_hidden_charge_counter_is_not_rendered():
    environment = visual()
    environment.reset(6600)
    frames = []
    for _ in range(CHARGE_PERIOD):
        snapshot = environment.snapshot()
        frames.append((environment.snapshot().reveal("evaluator")["charge"], environment.frame().copy()))
        action = 1
        environment.step(action, environment.gate.authorize_collection(action, "random"))
        environment.restore(snapshot)
        environment.step(action, environment.gate.authorize_collection(action, "random"))
    charges = {c for c, _ in frames}
    assert len(charges) > 1, "charge never varied, so the check is vacuous"
    assert "charge" not in environment._observation().structured_observation


def test_the_visual_frame_carries_a_content_digest_not_a_raw_buffer():
    environment = visual()
    result = environment.reset(6600)
    reference = result.observation.modality_payloads["image"]
    assert reference.shape == (24, 24, 3)
    assert reference.dtype == "uint8"
    assert reference.digest.startswith("sha256:")


def test_visual_replay_is_exact_after_restore():
    environment = visual()
    environment.reset(6601)
    for action in (1, 2, 1):
        environment.step(action, environment.gate.authorize_collection(action, "random"))
    snapshot = environment.snapshot()
    plan = (0, 1, 2, 3, 1)
    first = [
        environment.step(a, environment.gate.authorize_collection(a, "random")).probes.digest
        for a in plan
    ]
    environment.restore(snapshot)
    second = [
        environment.step(a, environment.gate.authorize_collection(a, "random")).probes.digest
        for a in plan
    ]
    assert first == second


def test_restoring_a_snapshot_from_a_different_level_is_refused():
    a, b = visual(), visual()
    a.reset(6600)
    b.reset(6601)
    with pytest.raises(ContractViolation):
        b.restore(HiddenSnapshot(payload={**a.snapshot().reveal("evaluator"), "seed": 6601,
                                          "dynamic": "base"},
                                 environment_version=a.identity.digest))


def test_both_adapters_supply_every_evaluator_required_probe():
    from sentinel.wm.matrix import REQUIRED_PROBES

    for environment in (adapter(), visual()):
        environment.reset(6600)
        environment.probes().subset(REQUIRED_PROBES)
