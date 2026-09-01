"""Scale-1A-0R D and E: the slots are vision-only, and the task needs both channels.

The D pin is the one that could have invalidated the headline. If language
reached the visual slots, the +0.400 intervention margin might be partly a
language effect, and the raw and CNN controls -- which have no language path --
would have been handicapped by construction rather than by their features. They
are bit-identical across a goal change, and this test holds them to it.

The E pins are the two halves of a multimodal claim. A task solvable from vision
alone does not need language; one solvable from language alone does not need
vision. Both directions are certified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from sentinel.env.adapters.procedural_visual_v2 import (
    ACTIONS,
    GOAL_PHRASES,
    ProceduralVisualV2Adapter,
    build_hidden_state_certificate,
    build_language_certificate,
    build_vision_necessity_certificate,
)
from sentinel.wm.authority import AuthorityGate
from sentinel.wm.packet import MAX_GOAL_TOKENS, build_vocabulary, tokenise_goal

SEED = 9000
BACKBONES = Path(__file__).resolve().parents[2] / "artifacts/shwm/backbones"

# The backbone-dependent pins need both the weights and mlx-vlm, which lives in
# the Phase-2 environment only. The exact suite must not require either, so they
# skip there and run under .venv-shwm. A skip is not a pass: the Scale-1A-0R
# report records which environment executed them.
import importlib.util as _finder

HAVE_WEIGHTS = (
    (BACKBONES / "qwen3_vl_4b").exists()
    and (BACKBONES / "gemma3_4b").exists()
    and _finder.find_spec("mlx_vlm") is not None
)


def v2() -> ProceduralVisualV2Adapter:
    return ProceduralVisualV2Adapter(gate=AuthorityGate())


# ---- D: the visual span is vision-only ---------------------------------------------


@pytest.mark.skipif(not HAVE_WEIGHTS, reason="backbone weights or mlx-vlm are unavailable in this environment")
@pytest.mark.parametrize("encoder_id", ["qwen3_vl_4b", "gemma3_4b"])
def test_visual_slots_are_bit_identical_across_a_language_goal_change(encoder_id):
    """If this ever fails, the slots are multimodal-fused and every interface
    without a language path is being compared unfairly."""
    from sentinel.wm.backbone_encoder import BackboneSpec, MlxVlmBackboneEncoder
    from sentinel.wm.backbones import FROZEN_CANDIDATES

    import yaml

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "experiments/shwm/configs/scale0.yaml").read_text()
    )
    candidate = next(c for c in FROZEN_CANDIDATES if c.encoder_id == encoder_id)
    encoder = MlxVlmBackboneEncoder(
        BackboneSpec(
            encoder_id,
            candidate.repository,
            config["encoder"]["revisions"][encoder_id],
            config["encoder"]["licences"][encoder_id],
            BACKBONES / encoder_id,
        )
    )
    adapter = v2()
    adapter.reset(SEED)
    frame = adapter.frame().copy()
    spans = []
    for marker in GOAL_PHRASES:
        adapter._goal_marker = marker
        spans.append(encoder.encode_visual_tokens(adapter._observation(), frame))
    encoder.release()
    assert np.array_equal(spans[0], spans[1]), (
        "the visual span moved with the instruction; these are multimodal-fused slots "
        "and the raw and CNN controls are owed a matched language-fusion module"
    )


@pytest.mark.skipif(not HAVE_WEIGHTS, reason="backbone weights or mlx-vlm are unavailable in this environment")
@pytest.mark.parametrize("encoder_id,expected", [("qwen3_vl_4b", 64), ("gemma3_4b", 256)])
def test_the_visual_span_is_exactly_the_expected_token_count(encoder_id, expected):
    """Gemma's marker token appears once and its soft tokens 256 times; taking
    the wrong one gave a span of one."""
    from sentinel.wm.backbone_encoder import BackboneSpec, MlxVlmBackboneEncoder
    from sentinel.wm.backbones import FROZEN_CANDIDATES

    import yaml

    config = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "experiments/shwm/configs/scale0.yaml").read_text()
    )
    candidate = next(c for c in FROZEN_CANDIDATES if c.encoder_id == encoder_id)
    encoder = MlxVlmBackboneEncoder(
        BackboneSpec(
            encoder_id, candidate.repository,
            config["encoder"]["revisions"][encoder_id],
            config["encoder"]["licences"][encoder_id],
            BACKBONES / encoder_id,
        )
    )
    adapter = v2()
    adapter.reset(SEED)
    span = encoder.encode_visual_tokens(adapter._observation(), adapter.frame())
    encoder.release()
    assert span.shape[0] == expected
    side = int(round(expected**0.5))
    assert side * side == expected, "the span does not form a square coordinate grid"


# ---- E: both channels are necessary -------------------------------------------------


def test_vision_alone_is_insufficient():
    """Same pixels, different instruction, different correct action."""
    certificate = build_language_certificate(SEED, max_depth=6)
    assert certificate["pixels_identical_under_both_instructions"]
    assert certificate["best_action_alpha"] != certificate["best_action_beta"]


def test_language_alone_is_insufficient():
    """Same instruction, different visual state, different correct action."""
    certificate = build_vision_necessity_certificate(SEED, max_depth=5)
    assert certificate["identical_instruction"]
    assert certificate["visual_state_differs"]
    assert certificate["best_action_a"] != certificate["best_action_b"]


def test_shuffling_the_instruction_changes_the_correct_action():
    certificate = build_language_certificate(SEED, max_depth=6)
    vocabulary = build_vocabulary(GOAL_PHRASES.values())
    a = tokenise_goal(certificate["goal_text_alpha"], vocabulary)
    b = tokenise_goal(certificate["goal_text_beta"], vocabulary)
    assert a != b, "the two instructions tokenise identically, so shuffling is a no-op"
    assert len(a) == len(b) == MAX_GOAL_TOKENS
    assert certificate["language_changes_correct_action"]


def test_action_history_is_required_on_the_hidden_phase_case():
    """Same frame, same action, different outcome -- separable only by history."""
    certificate = build_hidden_state_certificate(SEED, max_depth=8)
    assert certificate.successor_signature_a != certificate.successor_signature_b
    assert certificate.observation_trace_a != certificate.observation_trace_b
    assert certificate.observation_trace_a[-1] == certificate.observation_trace_b[-1]
    assert len(certificate.history_a) == len(certificate.history_b)


def test_audio_stays_declared_and_absent_in_this_phase():
    adapter = v2()
    result = adapter.reset(SEED)
    declared = {m.value for m in result.observation.modality_mask.declared}
    present = {m.value for m in result.observation.modality_mask.present}
    assert "audio" in declared
    assert "audio" not in present
    assert result.observation.structured_observation["audio_present"] is False
