"""Scale-0 gate: the feature cache cannot serve one identity's features to another.

A cache that misses is slow. A cache that returns the wrong encoder's features
is a silently invalid comparison across every arm that shared it, which is why
the stale case raises instead of falling back to a miss.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.wm.cache import (
    CacheCorruptionError,
    CacheIdentity,
    LatentCache,
    StaleCacheError,
    latent_cache_key,
)
from sentinel.wm.encoder import CachedEncoder, DeterministicControlEncoder
from sentinel.wm.latent_contract import (
    EncoderIdentity,
    Modality,
    ModalityMask,
    ObservationEnvelope,
    Precision,
)
from sentinel.wm.versioning import digest_of

MASK = ModalityMask((Modality.IMAGE, Modality.STRUCTURED), (Modality.STRUCTURED,))
ALT_MASK = ModalityMask((Modality.IMAGE, Modality.STRUCTURED), (Modality.IMAGE, Modality.STRUCTURED))


def identity(
    *, revision="r1", weights="w1", preprocessing="p1", precision=Precision.BF16,
    projector="proj-1", mask=MASK,
) -> CacheIdentity:
    encoder = EncoderIdentity(
        provider="prov",
        model_name="model",
        revision=revision,
        weight_digest=digest_of(weights),
        preprocessing_digest=digest_of(preprocessing),
        precision=precision,
        license_record="licence",
        feature_dimension=8,
    )
    return CacheIdentity(encoder, digest_of(projector), mask)


OBSERVATION = digest_of("observation-1")


def test_key_depends_on_the_observation_and_every_identity_field():
    base = latent_cache_key(OBSERVATION, identity())
    variants = {
        "observation": latent_cache_key(digest_of("observation-2"), identity()),
        "revision": latent_cache_key(OBSERVATION, identity(revision="r2")),
        "weights": latent_cache_key(OBSERVATION, identity(weights="w2")),
        "preprocessing": latent_cache_key(OBSERVATION, identity(preprocessing="p2")),
        "precision": latent_cache_key(OBSERVATION, identity(precision=Precision.FP32)),
        "projector": latent_cache_key(OBSERVATION, identity(projector="proj-2")),
        "modality_mask": latent_cache_key(OBSERVATION, identity(mask=ALT_MASK)),
    }
    for name, key in variants.items():
        assert key != base, f"{name} did not change the cache key"
    assert len(set(variants.values())) == len(variants)


def test_miss_then_hit(tmp_path):
    cache = LatentCache(tmp_path)
    assert cache.get(OBSERVATION, identity()) is None
    assert cache.stats.misses == 1
    values = np.arange(8, dtype=np.float32)
    cache.put(OBSERVATION, identity(), values)
    loaded = cache.get(OBSERVATION, identity())
    assert np.array_equal(loaded, values)
    assert cache.stats.hits == 1
    assert cache.stats.hit_ratio == pytest.approx(0.5)


@pytest.mark.parametrize(
    "changed",
    [
        {"revision": "r2"},
        {"weights": "w2"},
        {"preprocessing": "p2"},
        {"precision": Precision.FP32},
        {"projector": "proj-2"},
        {"mask": ALT_MASK},
    ],
)
def test_a_changed_identity_is_a_stale_rejection_not_a_silent_miss(tmp_path, changed):
    cache = LatentCache(tmp_path)
    cache.put(OBSERVATION, identity(), np.zeros(8, dtype=np.float32))
    with pytest.raises(StaleCacheError):
        cache.get(OBSERVATION, identity(**changed))
    assert cache.stats.stale_rejections == 1


def test_writing_over_a_different_identity_is_refused(tmp_path):
    cache = LatentCache(tmp_path)
    cache.put(OBSERVATION, identity(), np.zeros(8, dtype=np.float32))
    with pytest.raises(StaleCacheError):
        cache.put(OBSERVATION, identity(revision="r2"), np.ones(8, dtype=np.float32))


def test_a_corrupted_payload_is_detected_rather_than_returned(tmp_path):
    cache = LatentCache(tmp_path)
    cache.put(OBSERVATION, identity(), np.zeros(8, dtype=np.float32))
    payload = next((tmp_path / "payload").rglob("*.npy"))
    blob = bytearray(payload.read_bytes())
    blob[-1] ^= 0xFF
    payload.write_bytes(bytes(blob))
    with pytest.raises(CacheCorruptionError):
        cache.get(OBSERVATION, identity())


def test_the_index_survives_a_restart(tmp_path):
    first = LatentCache(tmp_path)
    first.put(OBSERVATION, identity(), np.arange(8, dtype=np.float32))
    first.flush()
    second = LatentCache(tmp_path)
    assert np.array_equal(second.get(OBSERVATION, identity()), np.arange(8, dtype=np.float32))
    assert second.stats.hits == 1


def test_size_report_separates_payload_from_index_and_metadata(tmp_path):
    cache = LatentCache(tmp_path)
    for i in range(4):
        cache.put(digest_of(f"obs-{i}"), identity(), np.zeros(512, dtype=np.float16))
    cache.flush()
    report = cache.size_report()
    assert report["entries"] == 4
    assert report["payload_bytes_resident"] == report["payload_bytes_indexed"] > 0
    assert report["index_bytes"] > 0
    assert report["total_bytes"] == report["payload_bytes_resident"] + report["index_bytes"]


# ---- the adapter over the cache ---------------------------------------------


def envelope(step: int) -> ObservationEnvelope:
    return ObservationEnvelope(
        episode_id="ep",
        step=step,
        timestamp_ns=step,
        modality_payloads={},
        structured_observation={"visible": step},
        modality_mask=MASK,
        available_action_digest=digest_of([0, 1]),
        environment_version=digest_of("env"),
    )


def test_control_encoder_is_deterministic_across_instances():
    a = DeterministicControlEncoder(feature_dimension=32)
    b = DeterministicControlEncoder(feature_dimension=32)
    assert np.array_equal(a.encode_array(envelope(1)), b.encode_array(envelope(1)))
    assert not np.array_equal(a.encode_array(envelope(1)), a.encode_array(envelope(2)))
    assert a.identity.digest == b.identity.digest
    assert a.health_check().ok


def test_two_control_variants_have_different_identities_and_features():
    a = DeterministicControlEncoder(feature_dimension=32, variant="control-a")
    b = DeterministicControlEncoder(feature_dimension=32, variant="control-b")
    assert a.identity.digest != b.identity.digest
    assert not np.array_equal(a.encode_array(envelope(1)), b.encode_array(envelope(1)))


def test_control_encoder_declares_bfloat16_and_actually_quantises_to_it():
    encoder = DeterministicControlEncoder(feature_dimension=64, precision=Precision.BF16)
    values = encoder.encode_array(envelope(3))
    as_int = values.view(np.uint32)
    assert np.all((as_int & 0x0000FFFF) == 0), "declared bf16 but kept float32 mantissa bits"


def test_cached_encoder_serves_the_second_call_from_disk(tmp_path):
    cache = LatentCache(tmp_path)
    wrapped = CachedEncoder(
        DeterministicControlEncoder(feature_dimension=16), cache, digest_of("projector")
    )
    first = wrapped.encode_array(envelope(0))
    second = wrapped.encode_array(envelope(0))
    assert np.array_equal(first, second)
    assert cache.stats.writes == 1
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
