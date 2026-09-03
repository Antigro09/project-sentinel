"""E interfaces 4-7. Frozen pretrained vision towers as coordinate-preserving slots.

Frames repeat heavily -- consecutive pairs share a frame, and the same layout is revisited
-- so every distinct frame is encoded ONCE, keyed by its content hash, and the cache is
persisted. Encoding each pair independently would have run the tower twice per frame for
no gain.

The measured geometry is taken from the model configs rather than assumed: Qwen3-VL 4B
emits an 8x8 visual token grid, Gemma 3 4B a 16x16 grid after its projector pools 4096
patches to 256. Neither is forced onto the other's grid.

    imported by n_interfaces.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sentinel.wm.backbone_encoder import BackboneSpec, MlxVlmBackboneEncoder  # noqa: E402
from sentinel.wm.backbones import FROZEN_CANDIDATES  # noqa: E402

ROOT = REPO / "artifacts/shwm/backbones"
CACHE = REPO / "artifacts/shwm/scale1/n-backbone-cache"
_ENCODERS: dict[str, Any] = {}
_MEMO: dict[str, dict[str, np.ndarray]] = {}

# Each frame is projected to PER_FRAME dims by a FIXED random projection before it is
# cached, so a before+after concatenation lands exactly on the shared slot width and no
# second projection is applied. Caching the raw 2560-wide tokens would have needed about
# eight gigabytes for Gemma alone, and the projection is frozen and declared rather than
# learned, so it costs no trainable parameter to either backbone.
PER_FRAME = 32
_PROJECTIONS: dict[str, np.ndarray] = {}


def projection_for(encoder_id: str, width: int) -> np.ndarray:
    if encoder_id not in _PROJECTIONS:
        rng = np.random.default_rng(30_000 + (0 if encoder_id.startswith("qwen") else 1))
        _PROJECTIONS[encoder_id] = (
            rng.normal(size=(width, PER_FRAME)) / np.sqrt(width)).astype(np.float32)
    return _PROJECTIONS[encoder_id]


def spec_for(encoder_id: str) -> BackboneSpec:
    """Positional construction, matching slot_resolution_audit.py.

    The weights are read from the repository's own artifacts/shwm/backbones tree, which
    is where the frozen matrix put them; the HF cache is not consulted.
    """
    import json

    candidate = next(c for c in FROZEN_CANDIDATES if c.encoder_id == encoder_id)
    local = ROOT / encoder_id
    revision = "local"
    manifest = REPO / "artifacts/shwm/scale0/freeze-manifest.json"
    if manifest.exists():
        blob = json.loads(manifest.read_text())
        revision = str(blob.get("encoders", {}).get(encoder_id, {}).get(
            "revision", revision))
    return BackboneSpec(encoder_id, candidate.repository, revision,
                        getattr(candidate, "licence", "see-repository"), local)


def encoder(encoder_id: str):
    if encoder_id not in _ENCODERS:
        instance = MlxVlmBackboneEncoder(spec_for(encoder_id))
        instance.load()
        _ENCODERS[encoder_id] = instance
    return _ENCODERS[encoder_id]


_CANONICAL = None


def canonical_observation():
    """One valid envelope, reused for every frame.

    `encode_visual_tokens(observation, frame)` takes the pixels from `frame`; the
    envelope only supplies the text placeholder and the modality declaration. Building a
    fresh envelope per frame would have re-derived the same object thousands of times,
    and hand-rolling one risks disagreeing with the adapter's contract -- so the adapter
    builds it, once.
    """
    global _CANONICAL
    if _CANONICAL is None:
        from sentinel.env.adapters.procedural_visual_v2 import ProceduralVisualV2Adapter
        from sentinel.wm.authority import AuthorityGate
        adapter = ProceduralVisualV2Adapter(gate=AuthorityGate(gate_id="n-backbone"))
        adapter.reset(110_000)
        _CANONICAL = adapter._observation()
    return _CANONICAL


def encode_frames(encoder_id: str, frames: np.ndarray, batch_note: str = "") -> np.ndarray:
    """(N, 24, 24, 3) float in [0,1] -> (N, g, g, width) with the tower's native grid."""
    instance = encoder(encoder_id)
    memo = _MEMO.setdefault(encoder_id, {})
    CACHE.mkdir(parents=True, exist_ok=True)
    store = CACHE / f"{encoder_id}.npz"
    if not memo and store.exists():
        with np.load(store) as data:
            memo.update({k: data[k] for k in data.files})

    out: list[np.ndarray] = []
    fresh = 0
    for frame in frames:
        u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        key = hashlib.sha256(u8.tobytes()).hexdigest()[:24]
        if key not in memo:
            tokens = instance.encode_visual_tokens(canonical_observation(), u8)
            side = int(round(tokens.shape[0] ** 0.5))
            if side * side != tokens.shape[0]:
                raise ValueError(
                    f"{encoder_id}: {tokens.shape[0]} visual tokens is not a square grid; "
                    "the coordinate mapping would be a guess")
            grid = tokens.reshape(side, side, tokens.shape[-1]).astype(np.float32)
            memo[key] = grid @ projection_for(encoder_id, grid.shape[-1])
            fresh += 1
        out.append(memo[key])
    if fresh:
        np.savez_compressed(store, **memo)
    return np.stack(out)
