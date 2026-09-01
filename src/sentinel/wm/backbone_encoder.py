"""Real frozen-backbone encoders behind the same adapter as the control.

The matrix requires two independent encoder families to satisfy one adapter with
no evaluator change, and that constraint decides the feature definition. Qwen3-VL
exposes a `hidden_state_at_layer` that would give a deeply contextualised
representation; Gemma 3 does not. Using it for one family and something else for
the other would mean the two arms were not doing the same thing, which is exactly
the confound the matched-arm rule exists to prevent.

So the feature both families compute is the one they both expose: the **fused
multimodal input embedding**, mean-pooled over the sequence, at the language
model's hidden width.

That choice has a limitation worth stating plainly rather than discovering later.
For an image the fused embedding is the output of the full vision tower and the
multimodal projector -- real pretrained perceptual computation, 415M parameters
in Qwen3-VL's case. For text it is an embedding lookup with no contextualisation
at all. So inherited backbone capability enters this pipeline almost entirely
through the visual path, and any attribution argument at Scale 1 has to account
for the two families' text paths being nearly as weak as the control encoder's.

Precision is bfloat16 throughout, matching the declared identity. The pooled
result is returned as float32 because that is what the cache and the feature
table store; the computation that produced it was bf16.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sentinel.wm.latent_contract import (
    ContractViolation,
    EncodedObservation,
    EncoderIdentity,
    ObservationEnvelope,
    Precision,
    Taint,
)
from sentinel.wm.versioning import digest_array, digest_file, digest_of

POOLING = "mean-over-sequence"
IMAGE_SIDE = 224
"""Frames are resized to this square before the processor sees them.

Part of the preprocessing digest: changing it changes every cached feature, and
the cache is built to refuse to serve across that change.
"""


def _safetensors_parameter_count(path: Path) -> int:
    """Sum of tensor element counts, from the file's JSON header alone."""
    import struct

    with open(path, "rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    total = 0
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        shape = entry.get("shape") or []
        count = 1
        for axis in shape:
            count *= int(axis)
        total += count if shape else 0
    return total


@dataclass(frozen=True, slots=True)
class BackboneSpec:
    encoder_id: str
    repository: str
    revision: str
    licence: str
    local_path: Path

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "repository": self.repository,
            "revision": self.revision,
            "licence": self.licence,
        }


@dataclass
class MlxVlmBackboneEncoder:
    """A frozen multimodal backbone, loaded once, used read-only.

    The model is loaded lazily so that constructing the identity -- which the
    cache key needs -- costs a few file digests rather than nine gigabytes.
    """

    spec: BackboneSpec
    precision: Precision = Precision.BF16
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)
    _identity: EncoderIdentity | None = field(default=None, init=False, repr=False)
    _feature_dimension: int = field(default=0, init=False, repr=False)
    calls: int = field(default=0, init=False)
    last_geometry: dict[str, Any] = field(default_factory=dict, init=False)
    _last_input_ids: Any = field(default=None, init=False, repr=False)
    """Shape facts from the most recent encode, for the feature-geometry audit.

    A diagnostic, not state the output depends on: what a reader needs to know
    is how many tokens existed before pooling and how many survived it, and that
    is not recoverable from the pooled vector."""

    # ---- identity ------------------------------------------------------

    def _config(self) -> dict[str, Any]:
        return json.loads((self.spec.local_path / "config.json").read_text())

    def _declared_width(self) -> int:
        config = self._config()
        text = config.get("text_config", {})
        width = text.get("hidden_size")
        if not width:
            raise ContractViolation(
                f"{self.spec.encoder_id}: config declares no text hidden_size, so the "
                "fused embedding width is unknown"
            )
        return int(width)

    @property
    def identity(self) -> EncoderIdentity:
        if self._identity is None:
            weights = sorted(self.spec.local_path.glob("*.safetensors"))
            if not weights:
                raise ContractViolation(
                    f"{self.spec.encoder_id}: no safetensors under {self.spec.local_path}"
                )
            preprocessing_files = [
                self.spec.local_path / name
                for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json")
                if (self.spec.local_path / name).exists()
            ]
            self._feature_dimension = self._declared_width()
            self._identity = EncoderIdentity(
                provider=self.spec.repository.split("/", 1)[0],
                model_name=self.spec.repository,
                revision=self.spec.revision,
                weight_digest=digest_of(
                    {p.name: digest_file(p) for p in weights}
                ),
                preprocessing_digest=digest_of(
                    {
                        "files": {p.name: digest_file(p) for p in preprocessing_files},
                        "pooling": POOLING,
                        "image_side": IMAGE_SIDE,
                        "feature": "fused_multimodal_input_embedding",
                    }
                ),
                precision=self.precision,
                license_record=self.spec.licence,
                frozen=True,
                feature_dimension=self._feature_dimension,
                notes="frozen backbone; inherited capability, reported separately",
            )
        return self._identity

    # ---- loading -------------------------------------------------------

    def load(self) -> None:
        if self._model is not None:
            return
        import mlx.core as mx
        from mlx_vlm import load

        identity = self.identity  # digests before weights, so a mismatch fails cheap
        model, processor = load(str(self.spec.local_path), lazy=False)
        model.freeze()
        mx.eval(model.parameters())
        self._model, self._processor = model, processor
        del identity

    def release(self) -> None:
        """Drop the weights. Two 4B backbones held at once is 16.5 GiB for nothing."""
        import mlx.core as mx

        self._model = None
        self._processor = None
        mx.clear_cache()

    @property
    def frozen_parameters(self) -> int:
        """Inherited parameter count, read from the weight headers.

        Counted without loading the tensors, because with a warm cache the model
        is never loaded at all -- and a report that said "0 frozen parameters"
        because nothing happened to be resident would be worse than no report.
        Each safetensors file begins with a JSON header giving every tensor's
        shape, which is exact and costs a few kilobytes to read.
        """
        total = 0
        for weights in sorted(self.spec.local_path.glob("*.safetensors")):
            total += _safetensors_parameter_count(weights)
        return total

    # ---- encoding ------------------------------------------------------

    def _observation_text(self, observation: ObservationEnvelope) -> str:
        """Canonical text for the structured part. Deterministic and key-sorted."""
        return json.dumps(
            dict(observation.structured_observation), sort_keys=True, separators=(",", ":")
        )

    def _image_placeholder(self) -> str:
        """The token the processor expects to mark where an image goes.

        Discovered from the processor rather than hard-coded, because the two
        families spell it differently and a hard-coded placeholder would make the
        adapter family-specific -- the one thing it must not be.
        """
        for attribute in ("image_token", "boi_token", "image_placeholder"):
            token = getattr(self._processor, attribute, None)
            if token:
                return str(token)
        raise ContractViolation(
            f"{self.spec.encoder_id}: the processor declares no image placeholder token, "
            "so an image cannot be positioned in the prompt deterministically"
        )

    def _forward_tokens(self, observation: ObservationEnvelope, frame: np.ndarray | None):
        """The fused multimodal embedding, before pooling."""
        import mlx.core as mx

        self.load()
        processor = self._processor
        text = self._observation_text(observation)

        if frame is not None:
            from PIL import Image

            image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize(
                (IMAGE_SIDE, IMAGE_SIDE), Image.NEAREST
            )
            inputs = processor(
                text=[f"{self._image_placeholder()}{text}"],
                images=[image],
                return_tensors="np",
                padding=True,
            )
            pixel_values = mx.array(inputs["pixel_values"]).astype(mx.bfloat16)
            # Family-specific extras, passed only when the processor produced
            # them: Qwen3-VL needs the patch grid, Gemma 3 does not emit one.
            extra = {
                key: mx.array(np.asarray(inputs[key]))
                for key in ("image_grid_thw",)
                if key in inputs
            }
        else:
            inputs = processor(text=[text], return_tensors="np", padding=True)
            pixel_values = None
            extra = {}

        if "attention_mask" in inputs:
            # Gemma 3 threads the mask into its multimodal fusion; Qwen3-VL uses
            # it for rope indices. Supplying it is correct for both.
            extra["mask"] = mx.array(np.asarray(inputs["attention_mask"]))

        input_ids = mx.array(np.asarray(inputs["input_ids"]))
        self._last_input_ids = np.asarray(inputs["input_ids"])
        features = self._model.get_input_embeddings(input_ids, pixel_values, **extra)
        embeddings = features.inputs_embeds if hasattr(features, "inputs_embeds") else features
        self.last_geometry = {
            "modality": "image+text" if frame is not None else "text",
            "embedding_shape": [int(d) for d in embeddings.shape],
            "tokens_before_pooling": int(
                embeddings.size // embeddings.shape[-1] if embeddings.size else 0
            ),
            "tokens_after_pooling": 1,
            "pooling_axes": list(range(embeddings.ndim - 1)),
        }
        return embeddings

    def encode_array(
        self, observation: ObservationEnvelope, frame: np.ndarray | None = None
    ) -> np.ndarray:
        import mlx.core as mx

        embeddings = self._forward_tokens(observation, frame)
        pooled = mx.mean(embeddings.astype(mx.float32), axis=tuple(range(embeddings.ndim - 1)))
        mx.eval(pooled)
        self.calls += 1
        values = np.asarray(pooled, dtype=np.float32)
        if values.shape[-1] != self.identity.feature_dimension:
            raise ContractViolation(
                f"{self.spec.encoder_id}: pooled width {values.shape[-1]} disagrees with the "
                f"declared {self.identity.feature_dimension}"
            )
        return values

    def encode_with_tokens(
        self, observation: ObservationEnvelope, frame: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the pooled vector *and* the token sequence it was pooled from.

        The feature-sufficiency probe needs both from one forward pass. Whether a
        controllable variable is missing from the pooled vector but present in
        the tokens is the difference between the encoder having destroyed the
        information and the pooling having destroyed it, and those two findings
        have different remedies.
        """
        import mlx.core as mx

        tokens = self._forward_tokens(observation, frame)
        pooled = mx.mean(tokens.astype(mx.float32), axis=tuple(range(tokens.ndim - 1)))
        mx.eval(pooled, tokens)
        return (
            np.asarray(pooled, dtype=np.float32),
            np.asarray(tokens.astype(mx.float32), dtype=np.float32).reshape(-1, tokens.shape[-1]),
        )

    def _visual_token_id(self, ids: np.ndarray) -> int:
        """The id that actually fills the visual span, chosen by multiplicity.

        Naming it from one attribute does not work across families. Gemma 3's
        `processor.image_token_id` is `<start_of_image>`, a single marker that
        appears once, while the 256 visual slots carry `<image_soft_token>` from
        `config.image_token_index`. Taking the processor's answer gave one visual
        token instead of 256 and would have handed the slot interface a marker.

        So every candidate is collected and the one that actually repeats is
        used, which is self-correcting if a third family spells it differently
        again.
        """
        candidates: list[int] = []
        for attribute in ("image_token_id", "boi_token_id"):
            value = getattr(self._processor, attribute, None)
            if value is not None:
                candidates.append(int(value))
        config = self._config()
        for key in ("image_token_index", "image_token_id"):
            if key in config:
                candidates.append(int(config[key]))
        if not candidates:
            raise ContractViolation(
                f"{self.spec.encoder_id}: no image token id is discoverable, so the "
                "visual span cannot be located"
            )
        counts = {candidate: int((ids == candidate).sum()) for candidate in set(candidates)}
        best = max(counts, key=lambda candidate: counts[candidate])
        if counts[best] <= 1:
            raise ContractViolation(
                f"{self.spec.encoder_id}: no candidate image token repeats in the "
                f"sequence (counts {counts}); the visual span cannot be isolated"
            )
        return best

    def encode_visual_tokens(
        self, observation: ObservationEnvelope, frame: np.ndarray
    ) -> np.ndarray:
        """Only the visual tokens, in sequence order.

        A slot interface needs the image span alone. Taking a prefix of the
        sequence would mix language tokens into the slots for one family and not
        the other.
        """
        import mlx.core as mx

        tokens = self._forward_tokens(observation, frame)
        mx.eval(tokens)
        ids = np.asarray(self._last_input_ids).reshape(-1)
        mask = ids == self._visual_token_id(ids)
        if not mask.any():
            raise ContractViolation(
                f"{self.spec.encoder_id}: no image token found in the sequence; the "
                "visual span cannot be isolated"
            )
        flat = np.asarray(tokens.astype(mx.float32), dtype=np.float32).reshape(-1, tokens.shape[-1])
        return flat[mask]

    def encode(self, observation: ObservationEnvelope) -> EncodedObservation:
        values = self.encode_array(observation)
        return EncodedObservation(
            encoder_identity=self.identity,
            source_observation_digest=observation.digest,
            features=digest_array(values),
            modality_mask=observation.modality_mask,
            taint=frozenset({Taint.DEVELOPMENT, Taint.INHERITED_PRETRAINED}),
        )

    def health_check(self) -> dict[str, Any]:
        import time

        started = time.perf_counter()
        self.load()
        return {
            "ok": True,
            "encoder_id": self.spec.encoder_id,
            "revision": self.spec.revision,
            "load_seconds": time.perf_counter() - started,
            "feature_dimension": self.identity.feature_dimension,
            "frozen_parameters": self.frozen_parameters,
            "pooling": POOLING,
            "detail": (
                "fused multimodal input embedding; the image path runs the vision tower "
                "and projector, the text path is an embedding lookup"
            ),
        }
