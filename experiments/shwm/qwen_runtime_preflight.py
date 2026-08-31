"""Runtime preflight for the one frozen backbone whose licence is clear.

The metadata preflight answers "may we use it". This answers the question the
resource plan actually turns on: can a 4B multimodal backbone encode the sealed
development set on this machine inside the eight-hour cache-build ceiling, and
at what memory cost.

That is worth measuring even though the matrix is stopped on the other family,
because it is the cheapest way to learn now whether the backbone half of Scale 0
is affordable at all. If a single frozen encoder cannot finish 50,000 visual
observations in eight hours, granting Gemma access would not unblock anything
and the honest response is a narrower design, not a larger download.

Nothing here is a matrix run. It loads one model, encodes a sample, and reports.

    .venv-shwm/bin/python experiments/shwm/qwen_runtime_preflight.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from sentinel.env.adapters.procedural_visual import ProceduralVisualAdapter, render  # noqa: E402
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.latent_contract import Precision  # noqa: E402
from sentinel.wm.matrix import CACHE_BUILD_TIMEOUT_SECONDS, TRANSITIONS_PER_ENVIRONMENT  # noqa: E402
from sentinel.wm.resource import mlx_memory, process_resident_bytes  # noqa: E402
from sentinel.wm.versioning import digest_file, digest_of  # noqa: E402

REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DEFAULT_PATH = REPO / "artifacts/shwm/backbones/qwen3_vl_4b"


def sample_frames(count: int) -> list[np.ndarray]:
    """Real frames from the procedural visual adapter, not synthetic noise."""
    gate = AuthorityGate(gate_id="preflight")
    adapter = ProceduralVisualAdapter(gate=gate)
    frames: list[np.ndarray] = []
    seed = 6600
    while len(frames) < count:
        adapter.reset(seed)
        for step in range(8):
            frames.append(adapter.frame().copy())
            if len(frames) >= count:
                break
            action = step % 4
            adapter.step(action, gate.authorize_collection(action, "preflight"))
        seed += 1
    return frames


def weight_identity(path: Path) -> dict[str, Any]:
    """Digest every file that determines the features."""
    files = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix in {".safetensors", ".json", ".txt"}
    )
    digests = {p.name: digest_file(p) for p in files}
    weights = {name: d for name, d in digests.items() if name.endswith(".safetensors")}
    preprocessing = {
        name: d
        for name, d in digests.items()
        if name in {"preprocessor_config.json", "video_preprocessor_config.json", "config.json"}
    }
    return {
        "files": digests,
        "weight_digest": digest_of(weights),
        "preprocessing_digest": digest_of(preprocessing),
        "total_bytes": sum(p.stat().st_size for p in files),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/shwm/scale0/qwen-runtime-preflight.json")
    arguments = parser.parse_args()

    record: dict[str, Any] = {
        "encoder_id": "qwen3_vl_4b",
        "repository": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": REVISION,
        "licence": "apache-2.0",
        "declared_precision": Precision.BF16.value,
        "is_matrix_run": False,
        "note": "runtime feasibility probe; the frozen matrix remains stopped on gemma3_4b",
    }

    if not arguments.path.exists():
        record["verdict"] = "blocked"
        record["detail"] = f"weights are not present at {arguments.path}"
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(json.dumps(record, indent=2))
        return 2

    record["identity"] = weight_identity(arguments.path)

    mx.reset_peak_memory()
    resident_before = process_resident_bytes()
    load_started = time.perf_counter()
    from mlx_vlm import load

    model, processor = load(str(arguments.path), lazy=False)
    mx.eval(model.parameters())
    record["cold_load_seconds"] = time.perf_counter() - load_started
    record["memory_after_load"] = mlx_memory()
    record["resident_after_load_bytes"] = process_resident_bytes()
    record["resident_delta_bytes"] = record["resident_after_load_bytes"] - resident_before

    from mlx.utils import tree_flatten

    leaves = tree_flatten(model.parameters())
    record["total_parameters"] = int(sum(v.size for _, v in leaves))
    record["parameter_bytes"] = int(sum(v.nbytes for _, v in leaves))
    record["parameter_dtypes"] = sorted({str(v.dtype) for _, v in leaves})

    vision = None
    for attribute in ("vision_tower", "visual", "vision_model"):
        vision = getattr(model, attribute, None)
        if vision is not None:
            record["vision_attribute"] = attribute
            break
    if vision is not None:
        vision_leaves = tree_flatten(vision.parameters())
        record["vision_parameters"] = int(sum(v.size for _, v in vision_leaves))
        record["vision_parameter_bytes"] = int(sum(v.nbytes for _, v in vision_leaves))

    from PIL import Image

    frames = sample_frames(arguments.frames)
    images = [Image.fromarray(frame).resize((224, 224), Image.NEAREST) for frame in frames]

    mx.reset_peak_memory()
    encode_started = time.perf_counter()
    widths: set[int] = set()
    encoded = 0
    failure: str | None = None
    try:
        for image in images:
            inputs = processor.image_processor(images=[image], return_tensors="np")
            pixel_values = mx.array(inputs["pixel_values"]).astype(mx.bfloat16)
            grid = inputs.get("image_grid_thw")
            features = vision(pixel_values, mx.array(np.asarray(grid))) if grid is not None else vision(pixel_values)
            if isinstance(features, tuple):
                features = features[0]
            mx.eval(features)
            widths.add(int(features.shape[-1]))
            encoded += 1
    except Exception as exc:  # the failure is the finding
        failure = f"{type(exc).__name__}: {exc}"
    encode_seconds = time.perf_counter() - encode_started

    record["frames_encoded"] = encoded
    record["encode_seconds"] = encode_seconds
    record["frames_per_second"] = encoded / encode_seconds if encode_seconds and encoded else 0.0
    record["feature_widths"] = sorted(widths)
    record["memory_after_encode"] = mlx_memory()
    record["peak_resident_bytes"] = process_resident_bytes()
    record["encode_failure"] = failure

    visual_observations = TRANSITIONS_PER_ENVIRONMENT["procedural_visual"]
    rate = record["frames_per_second"]
    projected = visual_observations / rate if rate else float("inf")
    record["projected_seconds_for_visual_split"] = projected
    record["cache_build_ceiling_seconds"] = CACHE_BUILD_TIMEOUT_SECONDS
    record["fits_cache_build_ceiling"] = projected <= CACHE_BUILD_TIMEOUT_SECONDS
    record["verdict"] = "runnable" if failure is None and encoded else "blocked"

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")

    print(f"verdict                : {record['verdict']}")
    print(f"cold load              : {record['cold_load_seconds']:.1f} s")
    print(f"total parameters       : {record['total_parameters']:,}")
    print(f"vision parameters      : {record.get('vision_parameters', 0):,}")
    print(f"parameter bytes        : {record['parameter_bytes'] / 1024**3:.2f} GiB")
    print(f"peak resident          : {record['peak_resident_bytes'] / 1024**3:.2f} GiB")
    print(f"frames encoded         : {encoded} in {encode_seconds:.2f} s "
          f"({record['frames_per_second']:.1f} frames/s)")
    print(f"feature widths         : {record['feature_widths']}")
    print(f"projected 50k frames   : {projected / 3600:.2f} h "
          f"(ceiling {CACHE_BUILD_TIMEOUT_SECONDS / 3600:.0f} h) -> fits: {record['fits_cache_build_ceiling']}")
    if failure:
        print(f"encode failure         : {failure}")
    print(f"written                : {arguments.out}")
    return 0 if record["verdict"] == "runnable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
