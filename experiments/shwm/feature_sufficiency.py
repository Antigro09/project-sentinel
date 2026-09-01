"""S1.2 feature-sufficiency stop gate.

The question this answers is whether the cached encoder interface still contains
the state an action-conditioned model would need. If it does not, no dynamics
architecture can recover it, and training one would be measuring the encoder's
loss rather than the model's ability.

The design point that matters is the third representation arm. Comparing the
pooled vector against pixels tells you whether information survived; comparing
it against the *token sequence it was pooled from* tells you where it was lost.
Those are different findings with different remedies -- a worse encoder versus a
worse pooling -- and the brief asks failures to be classified into exactly that
distinction.

Probes are linear and fitted on development levels only. Linear is deliberate: a
deep probe can manufacture structure that a downstream model would also have to
manufacture, so a linear probe is the honest question -- is the variable *there*,
in a form something simple can read.

    .venv-shwm/bin/python experiments/shwm/feature_sufficiency.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sentinel.env.adapters.procedural_visual import (  # noqa: E402
    ACTIONS,
    CHARGE_PERIOD,
    DELTAS,
    GRID,
    ProceduralVisualAdapter,
)
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.backbone_encoder import BackboneSpec, MlxVlmBackboneEncoder  # noqa: E402
from sentinel.wm.backbones import FROZEN_CANDIDATES  # noqa: E402
from sentinel.wm.versioning import digest_of  # noqa: E402

TRAIN_SEEDS = range(7000, 7000 + 300)
HELD_OUT_SEEDS = range(8000, 8000 + 100)
STEPS_PER_LEVEL = 8
TOKEN_CHUNKS = 8
CHUNK_PROJECTION = 128


# ---- probe targets ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Target:
    name: str
    kind: str  # "regression" | "classification"
    classes: int = 0
    needs_history: bool = False
    note: str = ""


TARGETS: tuple[Target, ...] = (
    Target("agent_row", "classification", GRID, note="controllable object position"),
    Target("agent_col", "classification", GRID, note="controllable object position"),
    Target("goal_row", "classification", GRID, note="relevant object position"),
    Target("goal_col", "classification", GRID, note="relevant object position"),
    Target("delta_row", "regression", note="relative direction, signed"),
    Target("delta_col", "regression", note="relative direction, signed"),
    Target("manhattan", "regression", note="relative distance"),
    Target("blocked_up", "classification", 2, note="immediate action-effect class"),
    Target("blocked_right", "classification", 2, note="immediate action-effect class"),
    Target("blocked_down", "classification", 2, note="immediate action-effect class"),
    Target("blocked_left", "classification", 2, note="immediate action-effect class"),
    Target("legal_action_count", "regression", note="legal-action mask size"),
    Target("reward", "regression", note="reward / progress"),
    Target("terminated", "classification", 2, note="termination / failure"),
    Target(
        "charge",
        "classification",
        CHARGE_PERIOD,
        needs_history=True,
        note="hidden progress variable; not in one frame, only in history",
    ),
    Target(
        "moved",
        "classification",
        2,
        needs_history=True,
        note="velocity / change across consecutive observations",
    ),
)


def collect_probe_set(seeds, steps: int) -> list[dict[str, Any]]:
    """Observations with evaluator-only ground truth attached.

    The truth comes from the hidden snapshot and never enters any representation;
    it exists only to score the probes.
    """
    gate = AuthorityGate(gate_id="probe")
    adapter = ProceduralVisualAdapter(gate=gate)
    samples: list[dict[str, Any]] = []
    for seed in seeds:
        result = adapter.reset(seed)
        level = adapter._require_level()
        previous_position = None
        for step in range(steps):
            snapshot = adapter.snapshot().reveal("evaluator")
            position = tuple(int(v) for v in snapshot["position"])
            blocked = {}
            for action in ACTIONS:
                row_delta, column_delta = DELTAS[action]
                if level.mirrored:
                    row_delta, column_delta = -row_delta, -column_delta
                candidate = (position[0] + row_delta, position[1] + column_delta)
                inside = 0 <= candidate[0] < GRID and 0 <= candidate[1] < GRID
                blocked[action] = int(not inside or bool(level.walls[candidate]))
            samples.append(
                {
                    "seed": seed,
                    "step": step,
                    "observation": result.observation,
                    "frame": adapter.frame().copy(),
                    "truth": {
                        "agent_row": position[0],
                        "agent_col": position[1],
                        "goal_row": level.goal[0],
                        "goal_col": level.goal[1],
                        "delta_row": float(level.goal[0] - position[0]),
                        "delta_col": float(level.goal[1] - position[1]),
                        "manhattan": float(
                            abs(level.goal[0] - position[0]) + abs(level.goal[1] - position[1])
                        ),
                        "blocked_up": blocked[0],
                        "blocked_right": blocked[1],
                        "blocked_down": blocked[2],
                        "blocked_left": blocked[3],
                        "legal_action_count": float(4 - sum(blocked.values())),
                        "reward": float(result.reward),
                        "terminated": int(result.terminated),
                        "charge": int(snapshot["charge"]),
                        "moved": int(previous_position is not None and position != previous_position),
                    },
                }
            )
            previous_position = position
            action = ACTIONS[(step * 3 + seed) % len(ACTIONS)]
            result = adapter.step(action, gate.authorize_evaluator(action, "probe"))
            if result.terminated:
                break
    return samples


# ---- representations -------------------------------------------------------------


def chunk_tokens(tokens: np.ndarray, chunks: int, projection: np.ndarray) -> np.ndarray:
    """Mean-pool within contiguous chunks, keeping coarse sequence order.

    Global mean pooling is permutation-invariant and loses order entirely.
    Chunked pooling keeps a coarse version of it without needing to know the
    encoder's 2D token layout, which differs between the two families.
    """
    count = tokens.shape[0]
    bounds = np.linspace(0, count, chunks + 1).astype(int)
    pieces = []
    for i in range(chunks):
        lo, hi = bounds[i], max(bounds[i + 1], bounds[i] + 1)
        pieces.append(tokens[lo:hi].mean(axis=0))
    stacked = np.stack(pieces)              # (chunks, width)
    return (stacked @ projection).reshape(-1)  # (chunks * projection_dim,)


def build_representations(
    samples: list[dict[str, Any]], config: dict[str, Any], encoders: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(6600)
    frames = np.stack([s["frame"] for s in samples]).astype(np.float32) / 255.0
    flat = frames.reshape(len(samples), -1)

    # Raw low-resolution pixels: 24x24 -> 12x12 by 2x2 mean.
    small = frames.reshape(len(samples), GRID, 2, GRID, 2, 3).mean(axis=(2, 4))
    representations: dict[str, np.ndarray] = {
        "raw_lowres_pixels": small.reshape(len(samples), -1),
    }
    # A random frozen projection of the pixels. Preserves linear structure, so it
    # is a real floor rather than a trivially uninformative one.
    projection = rng.normal(size=(flat.shape[1], 2560)).astype(np.float32) / np.sqrt(flat.shape[1])
    representations["random_projection_pixels"] = flat @ projection

    geometry: dict[str, Any] = {}
    root = REPO / config["encoder"]["weights_root"]
    for encoder_id in encoders:
        candidate = next(c for c in FROZEN_CANDIDATES if c.encoder_id == encoder_id)
        encoder = MlxVlmBackboneEncoder(
            BackboneSpec(
                encoder_id,
                candidate.repository,
                config["encoder"]["revisions"][encoder_id],
                config["encoder"]["licences"][encoder_id],
                root / encoder_id,
            )
        )
        chunk_projection = rng.normal(size=(2560, CHUNK_PROJECTION)).astype(np.float32) / np.sqrt(2560)
        pooled_rows, chunk_rows = [], []
        started = time.perf_counter()
        for index, sample in enumerate(samples):
            pooled, tokens = encoder.encode_with_tokens(sample["observation"], frame=sample["frame"])
            pooled_rows.append(pooled)
            chunk_rows.append(chunk_tokens(tokens, TOKEN_CHUNKS, chunk_projection))
            if index == 0:
                geometry[encoder_id] = dict(encoder.last_geometry)
        elapsed = time.perf_counter() - started
        geometry[encoder_id]["observations"] = len(samples)
        geometry[encoder_id]["encode_seconds"] = elapsed
        geometry[encoder_id]["observations_per_second"] = len(samples) / elapsed
        representations[f"pooled_{encoder_id}"] = np.stack(pooled_rows)
        representations[f"spatial_tokens_{encoder_id}"] = np.stack(chunk_rows)
        encoder.release()
    return representations, geometry


# ---- linear probes ----------------------------------------------------------------


def ridge_fit(features: np.ndarray, targets: np.ndarray, penalty: float) -> np.ndarray:
    n, d = features.shape
    design = np.hstack([features, np.ones((n, 1), dtype=features.dtype)])
    gram = design.T @ design
    gram[np.diag_indices_from(gram)] += penalty
    gram[-1, -1] -= penalty  # do not penalise the intercept
    return np.linalg.solve(gram, design.T @ targets)


def ridge_predict(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.hstack([features, np.ones((features.shape[0], 1), dtype=features.dtype)])
    return design @ weights


PENALTIES = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)


def probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    held_x: np.ndarray,
    held_y: np.ndarray,
    target: Target,
) -> dict[str, Any]:
    """Fit on train, select the penalty on validation, report on held-out."""
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    tr = (train_x - mean) / scale
    va = (validation_x - mean) / scale
    te = (held_x - mean) / scale

    # A target with almost no variation is unreadable: every probe and the
    # baseline both score near-perfectly and the margin means nothing. Flag it
    # rather than let a degenerate 1.000 look like a result.
    if target.kind == "classification":
        counts = np.bincount(held_y.astype(int), minlength=max(target.classes, 1))
        degeneracy = float(counts.max() / max(counts.sum(), 1))
        degenerate = degeneracy > 0.95
    else:
        degeneracy = float(held_y.var())
        degenerate = degeneracy < 1e-3

    if target.kind == "classification":
        encode = lambda y: np.eye(target.classes, dtype=np.float32)[y.astype(int)]
        score = lambda pred, y: float((pred.argmax(axis=1) == y.astype(int)).mean())
        train_t, validation_t = encode(train_y), validation_y
        baseline = float(
            (np.bincount(train_y.astype(int), minlength=target.classes).argmax() == held_y.astype(int)).mean()
        )
        baseline_name = "majority class"
    else:
        train_t, validation_t = train_y.reshape(-1, 1), validation_y
        variance = float(((held_y - train_y.mean()) ** 2).mean())
        score = lambda pred, y: float(1.0 - ((y - pred.reshape(-1)) ** 2).mean() / (variance + 1e-12))
        baseline = 0.0
        baseline_name = "train mean (R2 = 0)"

    best = None
    for penalty in PENALTIES:
        weights = ridge_fit(tr, train_t, penalty)
        value = score(ridge_predict(va, weights), validation_t)
        if best is None or value > best[0]:
            best = (value, penalty, weights)
    validation_score, penalty, weights = best
    held_score = score(ridge_predict(te, weights), held_y)
    return {
        "target": target.name,
        "kind": target.kind,
        "note": target.note,
        "needs_history": target.needs_history,
        "penalty": penalty,
        "validation_score": validation_score,
        "held_out_score": held_score,
        "baseline": baseline,
        "baseline_name": baseline_name,
        "margin": held_score - baseline,
        "held_out_degeneracy": degeneracy,
        "degenerate": degenerate,
        "degeneracy_meaning": (
            "share of the most common held-out class" if target.kind == "classification"
            else "held-out variance"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="*", default=list(M_ENCODERS))
    parser.add_argument("--steps", type=int, default=STEPS_PER_LEVEL)
    parser.add_argument("--train-levels", type=int, default=300)
    parser.add_argument("--held-levels", type=int, default=100)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/shwm/scale1/feature-sufficiency.json")
    arguments = parser.parse_args()

    import yaml

    config = yaml.safe_load((Path(__file__).parent / "configs" / "scale0.yaml").read_text())

    train_seeds = list(range(7000, 7000 + arguments.train_levels))
    held_seeds = list(range(8000, 8000 + arguments.held_levels))
    assert not set(train_seeds) & set(held_seeds)

    print(f"collecting {len(train_seeds)} train levels and {len(held_seeds)} held-out levels")
    train_samples = collect_probe_set(train_seeds, arguments.steps)
    held_samples = collect_probe_set(held_seeds, arguments.steps)
    samples = train_samples + held_samples
    split = len(train_samples)
    print(f"  {len(train_samples)} train observations, {len(held_samples)} held-out")

    representations, geometry = build_representations(samples, config, arguments.encoders)

    # Validation is carved out of train by level, never from held-out.
    train_level_ids = np.array([s["seed"] for s in train_samples])
    unique_levels = np.unique(train_level_ids)
    validation_levels = set(unique_levels[: max(1, len(unique_levels) // 5)].tolist())
    is_validation = np.array([s["seed"] in validation_levels for s in train_samples])

    results: dict[str, Any] = {}
    for name, matrix in representations.items():
        train_block, held_block = matrix[:split], matrix[split:]
        rows = []
        for target in TARGETS:
            y_train = np.array([s["truth"][target.name] for s in train_samples], dtype=np.float32)
            y_held = np.array([s["truth"][target.name] for s in held_samples], dtype=np.float32)
            rows.append(
                probe(
                    train_block[~is_validation],
                    y_train[~is_validation],
                    train_block[is_validation],
                    y_train[is_validation],
                    held_block,
                    y_held,
                    target,
                )
            )
        results[name] = {"dimension": int(matrix.shape[1]), "probes": rows}

    document = {
        "gate": "S1.2 feature sufficiency",
        "train_levels": len(train_seeds),
        "held_out_levels": len(held_seeds),
        "train_observations": len(train_samples),
        "held_out_observations": len(held_samples),
        "probe": "linear ridge, penalty selected on a level-disjoint validation slice",
        "token_chunks": TOKEN_CHUNKS,
        "chunk_projection_dim": CHUNK_PROJECTION,
        "encoder_geometry": geometry,
        "results": results,
        "config_digest": digest_of(config),
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n")

    names = list(results)
    width = max(len(n) for n in names)
    print(f"\n{'target':22s} " + " ".join(f"{n[:16]:>16s}" for n in names))
    for index, target in enumerate(TARGETS):
        cells = []
        for name in names:
            row = results[name]["probes"][index]
            cells.append(f"{row['held_out_score']:+.3f}/{row['margin']:+.3f}")
        flag = " (history)" if target.needs_history else ""
        if results[names[0]]["probes"][index]["degenerate"]:
            flag += " DEGENERATE"
        print(f"{target.name:22s} " + " ".join(f"{c:>16s}" for c in cells) + flag)
    print("\nheld-out score / margin over baseline")
    print(f"written: {arguments.out}")
    return 0


M_ENCODERS = ("qwen3_vl_4b", "gemma3_4b")

if __name__ == "__main__":
    raise SystemExit(main())
