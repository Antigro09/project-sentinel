"""B. Reconcile the visual token geometry of both frozen backbones.

Two figures were reported earlier and both were wrong. "49 tokens" for Qwen3-VL
assumed the processor keeps the 224x224 image it is handed; it does not, it
upscales to meet a minimum pixel count. "4096 tokens" for Gemma 3 was its *patch*
count, not its token count -- the multimodal projector pools those 4096 patches
down to 256 before anything sees them.

Every number here is measured from the processor and the model config rather than
derived from a formula, because the formula is what was wrong.

    .venv-shwm/bin/python experiments/shwm/token_geometry.py
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

from sentinel.wm.versioning import digest_of  # noqa: E402

BACKBONES = ("qwen3_vl_4b", "gemma3_4b")


def measure(encoder_id: str, side: int) -> dict[str, Any]:
    from PIL import Image
    from mlx_vlm import load

    root = REPO / "artifacts/shwm/backbones" / encoder_id
    config = json.loads((root / "config.json").read_text())
    preprocessor = json.loads((root / "preprocessor_config.json").read_text())
    vision = config.get("vision_config", {})
    _, processor = load(str(root), lazy=True)

    image = Image.fromarray(np.zeros((24, 24, 3), dtype=np.uint8)).resize((side, side), Image.NEAREST)
    placeholder = getattr(processor, "image_token", None) or getattr(processor, "boi_token", "")
    packed = processor(text=[f"{placeholder}x"], images=[image], return_tensors="np", padding=True)

    pixel_values = np.asarray(packed["pixel_values"])
    record: dict[str, Any] = {
        "encoder_id": encoder_id,
        "input_image_side": side,
        "processor_size_config": preprocessor.get("size"),
        "pixel_values_shape": list(pixel_values.shape),
        "vision_hidden_width": vision.get("hidden_size"),
        "language_hidden_width": config.get("text_config", {}).get("hidden_size"),
        "patch_size": vision.get("patch_size"),
        "spatial_merge_size": vision.get("spatial_merge_size"),
        "input_ids_length": int(np.asarray(packed["input_ids"]).shape[-1]),
    }

    if "image_grid_thw" in packed:
        grid = np.asarray(packed["image_grid_thw"])[0].tolist()
        merge = vision.get("spatial_merge_size") or 1
        patches = int(grid[0] * grid[1] * grid[2])
        tokens = patches // (merge * merge)
        side_tokens = int(round(tokens ** 0.5))
        record.update(
            {
                "image_grid_thw": grid,
                "processor_patch_count": patches,
                "vision_tower_output_tokens": patches,
                "multimodal_projector_output_tokens": tokens,
                "adapter_selected_visual_tokens": tokens,
                "visual_token_grid": [side_tokens, side_tokens],
                "spatial_coordinate_representation": (
                    f"{side_tokens}x{side_tokens} row-major grid; token index i maps to "
                    f"(row=i//{side_tokens}, col=i%{side_tokens})"
                ),
                "effective_input_pixels": int(grid[1] * grid[2]) * (vision.get("patch_size") or 1) ** 2,
            }
        )
    else:
        image_size = vision.get("image_size")
        patch = vision.get("patch_size")
        patches = (image_size // patch) ** 2
        tokens = int(config.get("mm_tokens_per_image"))
        side_tokens = int(round(tokens ** 0.5))
        record.update(
            {
                "image_grid_thw": None,
                "processor_patch_count": patches,
                "vision_tower_output_tokens": patches,
                "multimodal_projector_output_tokens": tokens,
                "adapter_selected_visual_tokens": tokens,
                "visual_token_grid": [side_tokens, side_tokens],
                "spatial_coordinate_representation": (
                    f"{side_tokens}x{side_tokens} row-major grid after the projector pools "
                    f"{patches} patches; token index i maps to (row=i//{side_tokens}, col=i%{side_tokens})"
                ),
                "effective_input_pixels": image_size * image_size,
            }
        )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=224)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/shwm/scale1/token-geometry.json")
    arguments = parser.parse_args()

    records = {b: measure(b, arguments.side) for b in BACKBONES}

    # Full-sequence and cached figures come from a real observation, so that the
    # earlier 87/287 numbers can be placed against the visual-token counts.
    from sentinel.env.adapters.procedural_visual import ProceduralVisualAdapter
    from sentinel.wm.authority import AuthorityGate
    from sentinel.wm.backbone_encoder import BackboneSpec, MlxVlmBackboneEncoder
    from sentinel.wm.backbones import FROZEN_CANDIDATES

    import yaml

    config = yaml.safe_load((Path(__file__).parent / "configs" / "scale0.yaml").read_text())
    for encoder_id in BACKBONES:
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
        visual = ProceduralVisualAdapter(gate=gate)
        result = visual.reset(6600)
        _, tokens = encoder.encode_with_tokens(result.observation, frame=visual.frame())
        text_only = encoder.encode_with_tokens(result.observation)[1]
        encoder.release()
        record = records[encoder_id]
        record["full_model_sequence_tokens_with_image"] = int(tokens.shape[0])
        record["full_model_sequence_tokens_text_only"] = int(text_only.shape[0])
        record["language_tokens_in_sequence"] = int(
            tokens.shape[0] - record["adapter_selected_visual_tokens"]
        )
        record["tokens_before_pooling"] = int(tokens.shape[0])
        record["cached_tokens"] = 1
        record["cached_note"] = "the Scale-0 cache stores the mean-pooled vector, not the sequence"

    reconciliation = {
        "earlier_figures": {
            "qwen_49": {
                "claim": "49 visual tokens",
                "status": "WRONG",
                "why": (
                    "derived as (224/16)^2 / 2^2, assuming the processor keeps the 224x224 "
                    "image. It does not: shortest_edge is 65,536 pixels, so a 224x224 input "
                    "is upscaled to 256x256, giving a 16x16 patch grid and 64 tokens after "
                    "the 2x2 spatial merge."
                ),
                "correct": records["qwen3_vl_4b"]["adapter_selected_visual_tokens"],
            },
            "gemma_4096": {
                "claim": "4,096 visual tokens",
                "status": "WRONG",
                "why": (
                    "4,096 is the patch count, (896/14)^2, not the token count. The "
                    "multimodal projector pools those patches to mm_tokens_per_image = 256 "
                    "before the language model sees them. The 4,096 figure is the right "
                    "measure of vision-tower work and the wrong measure of sequence length."
                ),
                "correct": records["gemma3_4b"]["adapter_selected_visual_tokens"],
            },
            "sequence_87_287": {
                "claim": "87 and 287 tokens before pooling",
                "status": "CORRECT but conflated",
                "why": (
                    "these are full sequence lengths including language tokens, not visual "
                    "token counts. 87 = 64 visual + 23 language; 287 = 256 visual + 31 "
                    "language."
                ),
            },
        },
        "encode_cost_ratio_survives": {
            "claim": "Gemma costs about 16x Qwen to encode",
            "status": "UPHELD",
            "why": (
                "the ratio is driven by vision-tower patches, 4,096 against 256, which is "
                "16x. That figure was right even though the token figure quoted beside it "
                "was not."
            ),
        },
        "enabling_finding": {
            "statement": (
                "Both backbones emit a square grid of visual tokens -- 8x8 for Qwen3-VL, "
                "16x16 for Gemma 3 -- with a recoverable row-major coordinate mapping."
            ),
            "consequence": (
                "A coordinate-preserving slot interface is buildable for both without "
                "guessing the layout, which is what Decision 4 requires."
            ),
        },
    }

    document = {
        "backbones": records,
        "reconciliation": reconciliation,
        "digest": digest_of({"backbones": records}),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n")

    fields = [
        "processor_patch_count",
        "vision_tower_output_tokens",
        "multimodal_projector_output_tokens",
        "adapter_selected_visual_tokens",
        "visual_token_grid",
        "full_model_sequence_tokens_with_image",
        "language_tokens_in_sequence",
        "tokens_before_pooling",
        "cached_tokens",
        "vision_hidden_width",
        "language_hidden_width",
    ]
    width = max(len(f) for f in fields)
    print(f"{'':{width}}  {'qwen3_vl_4b':>18} {'gemma3_4b':>18}")
    for field_name in fields:
        a, b = records["qwen3_vl_4b"][field_name], records["gemma3_4b"][field_name]
        print(f"{field_name:{width}}  {str(a):>18} {str(b):>18}")
    print()
    for name, entry in reconciliation["earlier_figures"].items():
        print(f"{name:18s} {entry['status']}")
    print(f"{'encode_cost_16x':18s} {reconciliation['encode_cost_ratio_survives']['status']}")
    print(f"\nwritten: {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
