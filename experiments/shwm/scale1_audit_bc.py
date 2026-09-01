"""B and C. The full qualification table with intervals, and the hidden-phase
construct audit.

B exists because margins alone hide two things: whether a margin is larger than
its own sampling noise, and whether the target had any variation to predict. So
every cell carries an absolute score, its frozen baseline, an episode-level
paired interval, the shuffled-label control, and a degeneracy flag.

C exists because "recovers the hidden phase" was already shown once to mean
"reads the reset indicator". Splitting by switch-crossing history is what
separates the two, and comparing correct against shuffled history is what shows
whether the recovery uses history at all.

Intervals are bootstrapped over **episodes**, not frames. Frames within an
episode share a level, a palette and a trajectory, so resampling them would
report an interval far narrower than the evidence supports.

    .venv-shwm/bin/python experiments/shwm/scale1_audit_bc.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import ACTIONS  # noqa: E402
from sentinel.wm.splits_v2 import Stratum  # noqa: E402
from sentinel.wm.versioning import digest_of  # noqa: E402

from feature_qualification import (  # noqa: E402
    READOUT_WIDTH,
    RFF_BANDWIDTHS,
    RFF_WIDTH,
    TARGETS,
    build_slot_matrices,
    collect,
    readout,
    with_history,
)
from feature_sufficiency import PENALTIES, RandomFourier, ridge_fit, ridge_predict  # noqa: E402

BOOTSTRAP = 400


def fit_predict(train_x, train_y, val_x, val_y, test_x, target, shuffled=False):
    """Fit on train, select on validation, return per-sample test predictions."""
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    tr, va, te = (train_x - mean) / scale, (val_x - mean) / scale, (test_x - mean) / scale
    if shuffled:
        train_y = np.random.default_rng(11).permutation(train_y)

    if target.kind == "classification":
        encode = lambda y: np.eye(target.classes, dtype=np.float32)[y.astype(int)]
        pick = lambda p: p.argmax(axis=1)
        score = lambda p, y: float((pick(p) == y.astype(int)).mean())
        train_t, val_t = encode(train_y), val_y
    else:
        train_t, val_t = train_y.reshape(-1, 1), val_y
        pick = lambda p: p.reshape(-1)
        score = None

    best = None
    for bandwidth in RFF_BANDWIDTHS:
        expansion = RandomFourier(RFF_WIDTH, bandwidth)
        expansion.fit_shape(tr.shape[1], seed=6600)
        tr_e, va_e, te_e = expansion(tr), expansion(va), expansion(te)
        for penalty in PENALTIES:
            weights = ridge_fit(tr_e, train_t, penalty)
            prediction = ridge_predict(va_e, weights)
            if target.kind == "classification":
                value = score(prediction, val_t)
            else:
                value = -float(((val_t - prediction.reshape(-1)) ** 2).mean())
            if best is None or value > best[0]:
                best = (value, bandwidth, penalty, ridge_predict(te_e, weights))
    _, bandwidth, penalty, test_prediction = best
    return pick(test_prediction), {"bandwidth": bandwidth, "penalty": penalty}


def per_sample_quality(prediction, y, target, train_y):
    """Correctness per sample, so an episode bootstrap has something to resample."""
    if target.kind == "classification":
        correct = (prediction == y.astype(int)).astype(np.float64)
        majority = int(np.bincount(train_y.astype(int), minlength=target.classes).argmax())
        base = (np.full_like(y, majority) == y.astype(int)).astype(np.float64)
        return correct, base
    error = (y - prediction) ** 2
    base = (y - train_y.mean()) ** 2
    return -error, -base


def episode_bootstrap(quality, baseline, episodes, draws=BOOTSTRAP, seed=17):
    """Paired interval on (quality - baseline), resampling whole episodes."""
    unique = np.unique(episodes)
    generator = np.random.default_rng(seed)
    index = {e: np.where(episodes == e)[0] for e in unique}
    samples = np.empty(draws)
    for draw in range(draws):
        chosen = generator.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[e] for e in chosen])
        if quality.mean() < 0:  # regression: report an R2-like ratio
            denominator = -baseline[rows].mean()
            samples[draw] = (
                1.0 - (-quality[rows].mean()) / denominator if denominator > 0 else 0.0
            )
        else:
            samples[draw] = quality[rows].mean() - baseline[rows].mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-levels", type=int, default=150)
    parser.add_argument("--shift-levels", type=int, default=60)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--encoders", nargs="*", default=["qwen3_vl_4b", "gemma3_4b"])
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/shwm/scale1/audit-bc.json")
    arguments = parser.parse_args()

    import yaml

    from sentinel.wm.splits_v2 import CANONICAL_APPEARANCE_SEED, EpisodeDescriptor, StratifiedSplitManifest

    config = yaml.safe_load((Path(__file__).parent / "configs" / "scale0.yaml").read_text())
    train_layouts = list(range(10_000, 10_000 + arguments.clean_levels))
    held_layouts = list(range(20_000, 20_000 + arguments.clean_levels // 2))
    held_appearances = list(range(30_000, 30_000 + arguments.shift_levels))
    manifest = StratifiedSplitManifest(
        train_layouts=frozenset(train_layouts),
        held_layouts=frozenset(held_layouts),
        train_appearances=frozenset({CANONICAL_APPEARANCE_SEED}),
        held_appearances=frozenset(held_appearances),
    )

    def make(layouts, appearance_of):
        out = []
        for layout in layouts:
            d = EpisodeDescriptor("procedural_visual_v2", layout, appearance_of(layout), layout, layout)
            manifest.assign(d)
            out.append(d)
        return out

    groups = {
        "train": make(train_layouts, lambda _: CANONICAL_APPEARANCE_SEED),
        Stratum.DYNAMICS_CLEAN.value: make(held_layouts, lambda _: CANONICAL_APPEARANCE_SEED),
        Stratum.APPEARANCE_SHIFT.value: make(
            train_layouts[: arguments.shift_levels],
            lambda l: held_appearances[l % len(held_appearances)],
        ),
        Stratum.CROSSED_SHIFT.value: make(
            held_layouts[: arguments.shift_levels],
            lambda l: held_appearances[l % len(held_appearances)],
        ),
    }
    manifest.seal()

    print("collecting")
    collected = {name: collect(items, arguments.steps) for name, items in groups.items()}
    samples, boundaries, cursor = [], {}, 0
    for name in groups:
        boundaries[name] = (cursor, cursor + len(collected[name]))
        cursor += len(collected[name])
        samples.extend(collected[name])
    for name, block in collected.items():
        print(f"  {name:18s} {len(block):5d} observations")

    cache = arguments.out.parent / "qualification-slots.npz"
    signature = digest_of(
        {"clean": arguments.clean_levels, "shift": arguments.shift_levels,
         "steps": arguments.steps, "encoders": sorted(arguments.encoders)}
    )
    matrices = meta = None
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        if str(stored["signature"]) == signature:
            matrices = {k[5:]: stored[k] for k in stored.files if k.startswith("mat::")}
            meta = json.loads(str(stored["meta"]))
            print(f"  reusing cached slots from {cache.name}")
    if matrices is None:
        matrices, meta = build_slot_matrices(samples, config, arguments.encoders)
        np.savez_compressed(cache, signature=signature, meta=json.dumps(meta, default=str),
                            **{f"mat::{k}": v for k, v in matrices.items()})

    episodes = np.array([s["descriptor"].episode_hash for s in samples])
    lineages = np.array([f"{s['descriptor'].episode_hash}" for s in samples])
    train_lo, train_hi = boundaries["train"]
    validation = np.zeros(len(samples), dtype=bool)
    validation[train_lo : train_lo + (train_hi - train_lo) // 5] = True
    fit_rows = np.zeros(len(samples), dtype=bool)
    fit_rows[train_lo:train_hi] = True
    fit_rows &= ~validation

    probe_spec = {
        "architecture": "random-Fourier ridge (closed form, no optimiser)",
        "rff_width": RFF_WIDTH,
        "rff_bandwidths": list(RFF_BANDWIDTHS),
        "ridge_penalties": list(PENALTIES),
        "readout_width": READOUT_WIDTH,
        "selection": "bandwidth and penalty chosen on a level-disjoint validation slice",
        "solver": "exact linear solve of the regularised normal equations",
        "bootstrap": {"draws": BOOTSTRAP, "unit": "episode", "paired": True},
    }

    # ---- B: full table ----------------------------------------------------
    table: dict[str, Any] = {}
    for name, matrix in matrices.items():
        reduced = readout(matrix, name)
        historical = with_history(reduced, np.array([hash(e) for e in episodes]), arguments.steps)
        per_stratum: dict[str, Any] = {}
        for stratum in (
            Stratum.DYNAMICS_CLEAN.value,
            Stratum.APPEARANCE_SHIFT.value,
            Stratum.CROSSED_SHIFT.value,
        ):
            lo, hi = boundaries[stratum]
            test_rows = np.zeros(len(samples), dtype=bool)
            test_rows[lo:hi] = True
            rows = []
            for target in TARGETS:
                features = historical if target.history else reduced
                y = np.array([s["truth"][target.name] for s in samples], dtype=np.float32)
                prediction, params = fit_predict(
                    features[fit_rows], y[fit_rows], features[validation], y[validation],
                    features[test_rows], target,
                )
                quality, base = per_sample_quality(prediction, y[test_rows], target, y[fit_rows])
                low, high = episode_bootstrap(quality, base, episodes[test_rows])
                shuffled_prediction, _ = fit_predict(
                    features[fit_rows], y[fit_rows], features[validation], y[validation],
                    features[test_rows], target, shuffled=True,
                )
                shuffled_quality, _ = per_sample_quality(
                    shuffled_prediction, y[test_rows], target, y[fit_rows]
                )
                if target.kind == "classification":
                    absolute, baseline = float(quality.mean()), float(base.mean())
                    shuffled_score = float(shuffled_quality.mean())
                    counts = np.bincount(y[test_rows].astype(int), minlength=max(target.classes, 1))
                    degenerate = bool(counts.max() / max(counts.sum(), 1) > 0.95)
                else:
                    denominator = float((-base).mean())
                    absolute = 1.0 - float((-quality).mean()) / (denominator + 1e-12)
                    baseline = 0.0
                    shuffled_score = 1.0 - float((-shuffled_quality).mean()) / (denominator + 1e-12)
                    degenerate = bool(float(y[test_rows].var()) < 1e-3)
                rows.append(
                    {
                        "target": target.name,
                        "group": target.group,
                        "kind": target.kind,
                        "absolute_score": absolute,
                        "baseline_score": baseline,
                        "margin": absolute - baseline,
                        "interval_95": [low, high],
                        "interval_excludes_zero": bool(low > 0 or high < 0),
                        "shuffled_label_score": shuffled_score,
                        "shuffled_label_margin": shuffled_score - baseline,
                        "episodes": int(len(np.unique(episodes[test_rows]))),
                        "lineages": int(len(np.unique(lineages[test_rows]))),
                        "observations": int(test_rows.sum()),
                        "development_observations": int(fit_rows.sum()),
                        "validation_observations": int(validation.sum()),
                        "degenerate": degenerate,
                        "probe": {**probe_spec, **params},
                    }
                )
            per_stratum[stratum] = rows
        table[name] = per_stratum

    document = {
        "probe_specification": probe_spec,
        "split_manifest_digest": manifest.digest,
        "observations": {k: len(v) for k, v in collected.items()},
        "meta": meta,
        "table_B": table,
        "legacy_v1_replication": {
            "status": "not evaluated in this audit",
            "reason": (
                "the v1 split is replication-only and its 36.3% transition-tuple overlap "
                "makes a capability number from it uninterpretable; it is retained for "
                "reproducing Scale-0, not for qualifying an interface"
            ),
        },
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n")

    clean = Stratum.DYNAMICS_CLEAN.value
    print(f"\n=== B: dynamics_clean, intervention and hidden phase, 95% episode intervals ===")
    print(f"{'interface':34s} {'group':13s} {'abs':>7s} {'base':>7s} {'margin':>8s} "
          f"{'interval':>18s} {'excl 0':>7s} {'shuf':>7s}")
    for name, strata in table.items():
        for row in strata[clean]:
            if row["group"] not in ("intervention", "hidden_phase"):
                continue
            if row["target"] not in ("successor_0", "polarity"):
                continue
            lo, hi = row["interval_95"]
            print(f"{name:34s} {row['group']:13s} {row['absolute_score']:+7.3f} "
                  f"{row['baseline_score']:+7.3f} {row['margin']:+8.3f} "
                  f"[{lo:+.3f},{hi:+.3f}]".ljust(0) + f" {str(row['interval_excludes_zero']):>7s} "
                  f"{row['shuffled_label_margin']:+7.3f}")
    print(f"\nwritten: {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
