"""F / N9. Initial-state grounding, read from the reset FRAME rather than handed over.

The structured phases used an authored gauge: the rendered polarity stripe, mapped to a
one-hot initial belief. M2F established that the stripe is genuinely public -- a
phase-supervised gauge was bit-identical to it -- but not free, costing 0.0916 when
replaced by a learned encoder.

Here the gauge must be recovered from pixels. The stripe is row 0 of the reset frame and
nothing else, so a visual gauge that works is evidence the grounding is perceptual; a
visual gauge that only works when told where to look is evidence it is authored. The
MAIN pathway never sees evaluator phase; the phase-supervised arm exists solely to show
what the stripe is worth and is marked ineligible.

    .venv-shwm/bin/python experiments/shwm/n_gauge.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import n_core as core
import n_heads as heads
from m2d_core import ARTIFACTS, write

SEEDS = (33_000, 33_001, 33_002)
VARIANTS = {
    "A_authored_public_stripe_map": {"transform": "identity", "eligible": True,
                                     "note": "the M2F authored gauge, for comparison"},
    # Trained on the RENDERED stripe value. That value equals the initial polarity, so
    # this arm is supervised by the same number the evaluator holds -- it is a
    # stripe-reading result, not an outcome-trained gauge, and it is labelled as such.
    # The outcome-only-trained visual gauge that section F actually asks for is NOT_RUN.
    "B_stripe_supervised_visual_reader": {
        "transform": "identity", "eligible": True,
        "note": "supervised on the rendered stripe, which equals the initial polarity; "
                "NOT an outcome-only-trained gauge"},
    "C_reset_stripe_masked": {"transform": "mask_stripe", "eligible": True},
    "D_reset_frame_omitted": {"transform": "omit_frame", "eligible": True},
    "E_false_stripe": {"transform": "false_stripe", "eligible": True},
    "F_shuffled_reset_frames": {"transform": "shuffle", "eligible": True},
    "G_phase_supervised_visual_decoder": {
        "transform": "identity", "eligible": False,
        "note": "diagnostic; identical in construction to B, which is the point: a "
                "gauge read from the rendered stripe and one read from evaluator phase "
                "are the same computation because the stripe IS the polarity"},
}


def reset_frames(episodes) -> tuple[np.ndarray, np.ndarray]:
    """The reset frame of every episode and its true initial polarity."""
    frames = np.stack([e.frames[0] for e in episodes]).astype(np.float32) / 255.0
    truth = np.array([float(e.polarity[0]) for e in episodes], dtype=np.float32)
    return frames, truth


def transform(name: str, frames: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.array(frames, copy=True)
    if name == "identity":
        return out
    if name == "mask_stripe":
        out[:, 0, :, :] = out[:, 1, :, :]          # overwrite the stripe row with row 1
        return out
    if name == "omit_frame":
        return np.zeros_like(out)
    if name == "false_stripe":
        out[:, 0, :, :] = 1.0 - out[:, 0, :, :]
        return out
    if name == "shuffle":
        return out[rng.permutation(len(out))]
    raise KeyError(name)


def gauge_head(frames: np.ndarray, truth: np.ndarray, seed: int, updates: int = 1200):
    """A tiny convolutional reader over the reset frame. No coordinate input."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(seed)

    class Gauge(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(3, 16, 3, padding=1)
            self.c2 = nn.Conv2d(16, 16, 1)
            self.out = nn.Linear(32, 1)

        def __call__(self, x):
            h = nn.relu(self.c1(x))
            h = nn.relu(self.c2(h))
            flat = h.reshape(h.shape[0], -1, h.shape[-1])
            return self.out(mx.concatenate([mx.max(flat, axis=1),
                                            mx.mean(flat, axis=1)], axis=-1))

    model = Gauge()
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=2e-3)
    rng = np.random.default_rng(seed)
    x, y = mx.array(frames), mx.array(truth)
    for _ in range(updates):
        pick = mx.array(rng.integers(0, len(frames), min(64, len(frames))))

        def objective(m):
            return mx.mean(nn.losses.binary_cross_entropy(
                m(x[pick])[:, 0], y[pick], with_logits=True))

        value, grads = nn.value_and_grad(model, objective)(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, value)
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "n-gauge.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    train_eps = core.collect_visual(core.TRAIN_LAYOUTS, 3, 9, seed=11)
    held_eps = core.collect_visual(core.HELD_OUT_LAYOUTS, 3, 9, seed=313)
    train_frames, train_truth = reset_frames(train_eps)
    held_frames, held_truth = reset_frames(held_eps)
    print(f"{len(train_frames)} reset frames train / {len(held_frames)} held out; "
          f"positive polarity rate {held_truth.mean():.3f}\n", flush=True)

    # The authored gauge, for comparison: read the stripe by construction.
    authored = (held_frames[:, 0, :, :].mean(axis=(1, 2))
                > held_frames[:, 1, :, :].mean(axis=(1, 2))).astype(float)
    report: dict[str, Any] = {
        "reset_frames_train": len(train_frames), "reset_frames_held_out": len(held_frames),
        "authored_gauge_accuracy": float((authored == held_truth).mean()),
        "variants": {}}
    print(f"{'variant':40s} {'held-out accuracy':>18s}  eligible")
    print("-" * 72)
    print(f"{'A_authored_public_stripe_map':40s} "
          f"{report['authored_gauge_accuracy']:18.4f}  yes")

    for name, setting in VARIANTS.items():
        if name == "A_authored_public_stripe_map":
            report["variants"][name] = {"accuracy": report["authored_gauge_accuracy"],
                                        "eligible": True,
                                        "note": setting.get("note", "")}
            continue
        scores = []
        for seed in SEEDS:
            x = transform(setting["transform"], train_frames, seed)
            y = train_truth
            model = gauge_head(x, y, seed)
            import mlx.core as mx
            logits = np.asarray(model(mx.array(
                transform(setting["transform"], held_frames, seed))))
            scores.append(float(((logits[:, 0] > 0).astype(float) == held_truth).mean()))
        report["variants"][name] = {
            "accuracy": float(np.mean(scores)), "per_seed": scores,
            "eligible": setting["eligible"], "note": setting.get("note", ""),
            "transform": setting["transform"]}
        print(f"{name:40s} {np.mean(scores):18.4f}  "
              f"{'yes' if setting['eligible'] else 'NO'}")

    learned = report["variants"]["B_stripe_supervised_visual_reader"]["accuracy"]
    masked = report["variants"]["C_reset_stripe_masked"]["accuracy"]
    report["stripe_is_readable_from_pixels"] = bool(learned > 0.95)
    report["depends_entirely_on_the_stripe"] = bool(masked < 0.7)
    report["outcome_trained_visual_gauge"] = "NOT_RUN"
    report["outcome_trained_visual_gauge_reason"] = (
        "every arm here is supervised on the stripe value, which equals the initial "
        "polarity; a gauge trained only through the downstream outcome likelihood was "
        "not built, so no claim is made that visual grounding is outcome-learnable")
    report["false_stripe_is_a_relabelling_not_a_corruption"] = (
        "arm E scores 1.0000 because the reader is RETRAINED on the inverted stripe and "
        "simply learns the inverted convention; it is a positive invariance check, not "
        "an information-destroying control. Arms C, D and F are the destroying ones")
    report["conditional_on_authored_grounding"] = True
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nstripe is readable from pixels: "
          f"{report['stripe_is_readable_from_pixels']}")
    print(f"masking the stripe destroys it: {report['depends_entirely_on_the_stripe']}")
    print(f"outcome-only-trained visual gauge: "
          f"{report['outcome_trained_visual_gauge']}")
    print(f"still conditional on AUTHORED grounding: "
          f"{report['conditional_on_authored_grounding']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
