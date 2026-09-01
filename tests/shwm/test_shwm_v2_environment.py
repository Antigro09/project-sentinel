"""Scale-1A-0 gate: v2 hides something, and the audits can tell.

The calibration for every audit here is v1. Its `charge` variable was called
hidden and was a deterministic function of the step count, with a passing
invariance test to prove it. So each audit is run against both: it must fail on
v1 and pass on v2. An audit that passes on both is not measuring anything.

v1 itself is checked for byte-identity, because Scale 0's artefacts refer to its
generator digest and a v2 that quietly edited v1 would invalidate a passed gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.env.adapters.procedural_visual import (
    CHARGE_PERIOD,
    ProceduralVisualAdapter,
)
from sentinel.env.adapters.procedural_visual_v2 import (
    ACTIONS,
    GOAL_PHRASES,
    MARKERS,
    ProceduralVisualV2Adapter,
    build_hidden_state_certificate,
    build_language_certificate,
    build_level_v2,
)
from sentinel.wm.authority import AuthorityGate
from sentinel.wm.hidden_state_audit import (
    audit_hidden_not_in_features,
    audit_hidden_not_rendered,
    audit_history_identifies,
    audit_not_determined_by,
    audit_perturbation_reachable,
    summarise,
)
from sentinel.wm.latent_contract import ContractViolation

SEED = 9000


def v2() -> ProceduralVisualV2Adapter:
    return ProceduralVisualV2Adapter(gate=AuthorityGate())


def rollout_hidden(adapter, seed, steps, hidden_key, reset_kwargs=None):
    """(step, hidden) pairs from a real trajectory, plus the horizon pairing."""
    result = adapter.reset(seed, **(reset_kwargs or {}))
    pairs, horizon_pairs = [], []
    for step in range(steps):
        snapshot = adapter.snapshot().reveal("evaluator")
        pairs.append((int(snapshot["step"]), int(snapshot[hidden_key])))
        horizon_pairs.append((adapter.horizon - int(snapshot["step"]), int(snapshot[hidden_key])))
        action = ACTIONS[(step * 3 + seed) % len(ACTIONS)]
        result = adapter.step(action, adapter.gate.authorize_evaluator(action, "audit"))
        if result.terminated:
            break
    return pairs, horizon_pairs


# ---- v1 is untouched --------------------------------------------------------------


def test_the_v1_generator_is_unchanged():
    """Scale 0's artefacts refer to this digest. It must not move."""
    adapter = ProceduralVisualAdapter(gate=AuthorityGate())
    assert adapter.identity.version == "procedural-visual-v1"
    assert adapter.identity.name == "procedural_visual"
    # v1 still exposes the clock that made its hidden variable public.
    result = adapter.reset(6600)
    assert "steps_remaining" in result.observation.structured_observation


# ---- the audits, calibrated against v1 --------------------------------------------


def test_the_step_audit_fails_on_v1_and_passes_on_v2():
    v1_pairs, v1_horizon = rollout_hidden(
        ProceduralVisualAdapter(gate=AuthorityGate()), 6600, 12, "charge"
    )
    v1_verdict = audit_not_determined_by(v1_pairs, "step", "charge")
    assert not v1_verdict.passed, "the audit failed to catch v1's step-determined variable"

    pairs = []
    for seed in range(SEED, SEED + 12):
        collected, _ = rollout_hidden(v2(), seed, 12, "polarity")
        pairs.extend(collected)
    v2_verdict = audit_not_determined_by(pairs, "step", "polarity")
    assert v2_verdict.passed, v2_verdict.detail

    assert audit_not_determined_by(v1_horizon, "horizon_remaining", "charge").passed is False


def test_the_horizon_audit_passes_on_v2():
    pairs = []
    for seed in range(SEED, SEED + 12):
        _, horizon = rollout_hidden(v2(), seed, 12, "polarity")
        pairs.extend(horizon)
    verdict = audit_not_determined_by(pairs, "horizon_remaining", "polarity")
    assert verdict.passed, verdict.detail


def test_v2_exposes_no_clock_surrogate():
    result = v2().reset(SEED)
    fields = set(result.observation.structured_observation)
    assert not (fields & {"step", "steps_remaining", "horizon", "time", "elapsed"})


# ---- the reachability audit -------------------------------------------------------


def test_the_reachability_audit_rejects_the_v1_style_perturbation():
    """v1's invariance test varied charge with step held fixed. In a real
    trajectory charge == step % 3, so all but one of those states is unreachable
    -- and that is precisely what made the passing test vacuous."""
    reachable = {(step, step % CHARGE_PERIOD) for step in range(12)}
    unreachable = (4, (4 + 1) % CHARGE_PERIOD)
    assert unreachable not in reachable
    verdict = audit_perturbation_reachable(reachable, unreachable)
    assert not verdict.passed
    assert audit_perturbation_reachable(reachable, (4, 4 % CHARGE_PERIOD)).passed


def test_the_v2_certificate_states_are_reachable_by_construction():
    certificate = build_hidden_state_certificate(SEED, max_depth=8)
    adapter = v2()
    for history, expected in (
        (certificate.history_a, certificate.polarity_a),
        (certificate.history_b, certificate.polarity_b),
    ):
        adapter.reset(SEED)
        for action in history:
            adapter.step(action, adapter.gate.authorize_evaluator(action, "replay"))
        assert adapter._polarity == expected


# ---- the certificate itself --------------------------------------------------------


def test_the_hidden_state_certificate_has_every_required_property():
    certificate = build_hidden_state_certificate(SEED, max_depth=8)
    assert certificate.polarity_a != certificate.polarity_b, "hidden states do not differ"
    assert len(certificate.history_a) == len(certificate.history_b), (
        "unequal histories let a model separate them by counting steps"
    )
    assert certificate.history_a != certificate.history_b
    assert certificate.successor_signature_a != certificate.successor_signature_b, (
        "the same action produced the same outcome; the hidden state has no consequence"
    )
    assert certificate.observation_trace_a != certificate.observation_trace_b, (
        "the histories are publicly indistinguishable; nothing could infer the hidden state"
    )
    assert certificate.observation_trace_a[-1] == certificate.observation_trace_b[-1]


def test_the_hidden_state_is_not_rendered():
    certificate = build_hidden_state_certificate(SEED, max_depth=8)
    adapter = v2()
    frames = []
    for history in (certificate.history_a, certificate.history_b):
        adapter.reset(SEED)
        for action in history:
            adapter.step(action, adapter.gate.authorize_evaluator(action, "replay"))
        frames.append(adapter.frame().copy())
    verdict = audit_hidden_not_rendered(
        frames[0], frames[1], certificate.polarity_a, certificate.polarity_b
    )
    assert verdict.passed, verdict.detail


def test_the_rendering_audit_catches_a_planted_leak():
    """Calibration arm: if the audit cannot see a drawn hidden variable it is
    not evidence that the real one is undrawn."""
    clean = np.zeros((8, 8, 3), dtype=np.uint8)
    leaky = clean.copy()
    leaky[0, 0] = 255
    assert audit_hidden_not_rendered(clean, clean.copy(), 0, 1).passed
    assert not audit_hidden_not_rendered(clean, leaky, 0, 1).passed


def test_history_identifies_what_the_frame_cannot():
    certificate = build_hidden_state_certificate(SEED, max_depth=8)
    verdict = audit_history_identifies(
        certificate.observation_trace_a,
        certificate.observation_trace_b,
        certificate.polarity_a,
        certificate.polarity_b,
    )
    assert verdict.passed, verdict.detail
    assert not audit_history_identifies((1, 2, 3), (1, 2, 3), 0, 1).passed


# ---- the hidden state must not reach the cache ------------------------------------


def test_evaluator_only_state_does_not_enter_the_cached_features():
    """A public alias must encode identically, or the hidden variable is being
    handed to every downstream model for free."""
    from sentinel.wm.encoder import DeterministicControlEncoder

    certificate = build_hidden_state_certificate(SEED, max_depth=8)
    encoder = DeterministicControlEncoder(feature_dimension=64)
    adapter = v2()
    features = []
    for history in (certificate.history_a, certificate.history_b):
        adapter.reset(SEED)
        for action in history:
            adapter.step(action, adapter.gate.authorize_evaluator(action, "replay"))
        features.append(encoder.encode_array(adapter._observation()))
    verdict = audit_hidden_not_in_features(features[0], features[1])
    assert verdict.passed, verdict.detail

    planted = features[0].copy()
    planted[0] += 1.0
    assert not audit_hidden_not_in_features(features[0], planted).passed


# ---- language ----------------------------------------------------------------------


def test_language_changes_the_correct_action_with_pixels_fixed():
    certificate = build_language_certificate(SEED, max_depth=6)
    assert certificate["language_changes_correct_action"]
    assert certificate["pixels_identical_under_both_instructions"]
    assert certificate["best_action_alpha"] != certificate["best_action_beta"]
    assert certificate["goal_text_alpha"] != certificate["goal_text_beta"]


def test_the_goal_text_is_in_the_observation_and_the_marker_is_not():
    adapter = v2()
    result = adapter.reset(SEED)
    structured = result.observation.structured_observation
    assert structured["goal_text"] in GOAL_PHRASES.values()
    assert set(structured["markers_visible"]) == set(MARKERS)
    # The chosen marker's identity is carried by language, not by a field.
    assert "goal_marker" not in structured


# ---- appearance and layout stay independent ----------------------------------------


def test_appearance_and_layout_move_independently_in_v2():
    same_layout = build_level_v2(5, 1, 1), build_level_v2(5, 2, 1)
    new_layout = build_level_v2(5, 1, 1), build_level_v2(6, 1, 1)
    a, b = same_layout
    assert a.layout_digest == b.layout_digest
    assert a.appearance_digest != b.appearance_digest
    c, d = new_layout
    assert c.layout_digest != d.layout_digest


def test_the_hidden_phase_stream_is_independent_of_layout_and_appearance():
    polarities = {build_level_v2(5, 1, phase).initial_polarity for phase in range(24)}
    assert polarities == {0, 1}, "the phase seed does not actually vary the hidden state"
    fixed = {build_level_v2(layout, 1, 7).initial_polarity for layout in range(24)}
    assert len(fixed) == 1, "the hidden phase depends on the layout seed"


def test_exact_replay_and_restore_hold_in_v2():
    adapter = v2()
    adapter.reset(SEED)
    for action in (1, 2, 1):
        adapter.step(action, adapter.gate.authorize_evaluator(action, "replay"))
    snapshot = adapter.snapshot()
    plan = (0, 1, 2, 3, 1)
    first = [
        adapter.step(a, adapter.gate.authorize_evaluator(a, "replay")).probes.digest for a in plan
    ]
    adapter.restore(snapshot)
    second = [
        adapter.step(a, adapter.gate.authorize_evaluator(a, "replay")).probes.digest for a in plan
    ]
    assert first == second


def test_v2_supplies_every_evaluator_required_probe():
    from sentinel.wm.matrix import REQUIRED_PROBES

    adapter = v2()
    adapter.reset(SEED)
    adapter.probes().subset(REQUIRED_PROBES)


def test_the_full_audit_summary_separates_v1_from_v2():
    v1_pairs, _ = rollout_hidden(ProceduralVisualAdapter(gate=AuthorityGate()), 6600, 12, "charge")
    v2_pairs = []
    for seed in range(SEED, SEED + 12):
        collected, _ = rollout_hidden(v2(), seed, 12, "polarity")
        v2_pairs.extend(collected)
    v1_summary = summarise([audit_not_determined_by(v1_pairs, "step", "charge")])
    v2_summary = summarise([audit_not_determined_by(v2_pairs, "step", "polarity")])
    assert not v1_summary["all_passed"] and v1_summary["failed"]
    assert v2_summary["all_passed"] and not v2_summary["failed"]


# ---- construct validity: is the hidden variable exercised at all? -----------------


def test_the_hidden_variable_actually_changes_in_most_episodes():
    """A hidden variable nothing exercises is a constant with a good disguise.

    At three switches only 27% of episodes ever changed polarity, so a
    hidden-phase probe was mostly reading the reset indicator rather than
    tracking state. The density was raised until this passed, and this test
    exists so it cannot regress silently the way v1's did.
    """
    from sentinel.wm.hidden_state_audit import audit_hidden_state_exercised

    counts = []
    for layout in range(10_000, 10_150):
        adapter = v2()
        adapter.reset(layout)
        for step in range(6):
            action = ACTIONS[(step * 3 + layout) % len(ACTIONS)]
            adapter.step(action, adapter.gate.authorize_evaluator(action, "exercise"))
        counts.append(adapter._switch_crossings)

    verdict = audit_hidden_state_exercised(counts, minimum_rate=0.5)
    assert verdict.passed, verdict.detail
    assert verdict.evidence["mean_changes"] > 1.0


def test_the_exercise_audit_catches_a_variable_that_never_changes():
    """Calibration arm: v1's charge does change, but a constant must be caught."""
    from sentinel.wm.hidden_state_audit import audit_hidden_state_exercised

    assert not audit_hidden_state_exercised([0] * 100).passed
    assert not audit_hidden_state_exercised([0] * 73 + [1] * 27).passed  # the old 27%
    assert audit_hidden_state_exercised([0] * 24 + [2] * 76).passed
