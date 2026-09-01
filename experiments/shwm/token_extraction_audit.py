"""D. Exactly where the visual slots come from, and what is inside them.

The question that mattered most here was whether the slots are vision-only or
multimodal-fused. If language reaches them, the +0.400 intervention margin could
be partly a language effect, and the raw and CNN controls -- which have no
language path at all -- would have been handicapped by construction rather than
by their features.

They are vision-only, bit-identical across a goal change, and this script is the
evidence rather than the argument.

    .venv-shwm/bin/python experiments/shwm/token_extraction_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sentinel.env.adapters.procedural_visual_v2 import (  # noqa: E402
    GOAL_PHRASES,
    ProceduralVisualV2Adapter,
)
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.backbone_encoder import BackboneSpec, MlxVlmBackboneEncoder  # noqa: E402
from sentinel.wm.backbones import FROZEN_CANDIDATES  # noqa: E402
from sentinel.wm.interfaces import SLOT_GRID  # noqa: E402
from sentinel.wm.packet import SLOT_COUNT, SLOT_WIDTH  # noqa: E402
from sentinel.wm.versioning import digest_of  # noqa: E402


def audit(encoder_id: str, config: dict[str, Any]) -> dict[str, Any]:
    candidate = next(c for c in FROZEN_CANDIDATES if c.encoder_id == encoder_id)
    encoder = MlxVlmBackboneEncoder(
        BackboneSpec(
            encoder_id,
            candidate.repository,
            config["encoder"]["revisions"][encoder_id],
            config["encoder"]["licences"][encoder_id],
            REPO / "artifacts/shwm/backbones" / encoder_id,
        )
    )
    gate = AuthorityGate()
    adapter = ProceduralVisualV2Adapter(gate=gate)
    adapter.reset(9000)
    frame = adapter.frame().copy()

    variants: dict[str, np.ndarray] = {}
    id_counts: dict[str, dict[int, int]] = {}
    for marker in GOAL_PHRASES:
        adapter._goal_marker = marker
        variants[marker] = encoder.encode_visual_tokens(adapter._observation(), frame)
        ids = np.asarray(encoder._last_input_ids).reshape(-1)
        unique, counts = np.unique(ids, return_counts=True)
        id_counts[marker] = {int(u): int(c) for u, c in zip(unique, counts)}

    alpha, beta = variants["alpha"], variants["beta"]
    identical = bool(np.array_equal(alpha, beta))
    tokens = alpha.shape[0]
    side = int(round(tokens**0.5))

    model_config = json.loads(
        (REPO / "artifacts/shwm/backbones" / encoder_id / "config.json").read_text()
    )
    vision = model_config.get("vision_config", {})
    selected_id = encoder._visual_token_id(np.asarray(encoder._last_input_ids).reshape(-1))
    marker_ids = {
        "processor_image_token_id": getattr(encoder._processor, "image_token_id", None),
        "config_image_token_index": model_config.get("image_token_index"),
    }
    counts = id_counts["alpha"]

    record = {
        "encoder_id": encoder_id,
        "extraction_stage": (
            "multimodal projector output, written into the language model's input "
            "embedding sequence; no decoder layer has run"
        ),
        "extraction_stage_is_not": [
            "vision tower output (that is 1024/1152 wide, before the projector)",
            "a full transformer hidden state (no decoder layer is executed)",
        ],
        "extraction_width": int(alpha.shape[1]),
        "language_model_hidden_width": model_config.get("text_config", {}).get("hidden_size"),
        "vision_tower_hidden_width": vision.get("hidden_size"),
        "processor_patch_count": (
            (vision["image_size"] // vision["patch_size"]) ** 2
            if vision.get("image_size")
            else None
        ),
        "image_marker_count": int(counts.get(int(marker_ids["processor_image_token_id"] or -1), 0)),
        "soft_visual_token_count": int(counts.get(selected_id, 0)),
        "adapter_selected_token_count": tokens,
        "selected_token_id": selected_id,
        "candidate_token_ids": marker_ids,
        "token_spatial_coordinates": (
            f"{side}x{side} row-major; token i maps to (row=i//{side}, col=i%{side})"
        ),
        "slot_resampling_rule": (
            f"{side}x{side} token grid mean-pooled in "
            f"{side // SLOT_GRID}x{side // SLOT_GRID} blocks to a {SLOT_GRID}x{SLOT_GRID} slot grid"
        ),
        "projection_rule": (
            f"one frozen random matrix {alpha.shape[1]}x{SLOT_WIDTH}, drawn once from a "
            "tag-derived seed; never trained"
        ),
        "cached_slot_shape": [SLOT_COUNT, SLOT_WIDTH],
        "same_image_different_goal": {
            "bit_identical": identical,
            "max_abs_difference": float(np.max(np.abs(alpha - beta))),
            "verdict": "VISION-ONLY" if identical else "MULTIMODAL-FUSED",
            "consequence": (
                "no language-fusion module is owed to the raw or CNN controls"
                if identical
                else "raw and CNN controls must be given a matched language-fusion module"
            ),
        },
        "span_purity": {
            "selected_id_repeats": int(counts.get(selected_id, 0)),
            "marker_excluded": selected_id != marker_ids["processor_image_token_id"]
            or int(counts.get(selected_id, 0)) > 1,
            "no_language_token_in_span": True,
            "no_padding_token_in_span": True,
            "reason": (
                "the span is exactly the positions whose input id equals the selected "
                "visual token id, so every other id -- language, marker, padding -- is "
                "excluded by construction"
            ),
        },
    }
    encoder.release()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/shwm/scale1/token-extraction-audit.json")
    arguments = parser.parse_args()

    import yaml

    config = yaml.safe_load((Path(__file__).parent / "configs" / "scale0.yaml").read_text())
    records = {c.encoder_id: audit(c.encoder_id, config) for c in FROZEN_CANDIDATES}
    document = {"encoders": records, "digest": digest_of(records)}
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n")

    for encoder_id, record in records.items():
        print(f"=== {encoder_id} ===")
        print(f"  extraction stage        : {record['extraction_stage']}")
        print(f"  extraction width        : {record['extraction_width']}"
              f"  (vision tower is {record['vision_tower_hidden_width']})")
        print(f"  processor patches       : {record['processor_patch_count']}")
        print(f"  image markers in seq    : {record['image_marker_count']}")
        print(f"  soft visual tokens      : {record['soft_visual_token_count']}")
        print(f"  adapter-selected tokens : {record['adapter_selected_token_count']}")
        print(f"  coordinates             : {record['token_spatial_coordinates']}")
        print(f"  slot resampling         : {record['slot_resampling_rule']}")
        print(f"  same image, other goal  : {record['same_image_different_goal']['verdict']}"
              f"  (max diff {record['same_image_different_goal']['max_abs_difference']:.3e})")
        print(f"  -> {record['same_image_different_goal']['consequence']}")
    print(f"\nwritten: {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
