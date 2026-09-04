"""E / Q3. Does a colour VALUE carry role information before any behaviour is seen?

Seven runs, and two of them are plants that the guards must catch. A guard that passes
both the honest generator and the planted one measures nothing, so each guard is stated
with the calibration arm it must fail on.

    generator honesty     I(role ; colour) exactly zero, analytically and empirically
    invariance            relabelling the palette must not move a colour-free model
    guard A               role predicted from the colour VALUE -- chance when honest,
                          above chance under the role-dependent generation plant
    guard B               palette id predicted from the NON-colour token fields --
                          chance when honest, high under the palette-id plant

The equivariance check is exact rather than statistical: the binder is a DeepSets, so
permuting the colour tokens must permute its assignment and leave its event logit
bit-identical. That is a property of the architecture and it is asserted, not sampled.

    .venv-shwm/bin/python experiments/shwm/o2_leakage.py
"""

from __future__ import annotations

import argparse
import itertools
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as m2d
import o_core as O
import o2_core as C
import o2_models as M
from m2d_core import ARTIFACTS, write

SEEDS = (41_000, 41_001, 41_002)
TRAIN_LAYOUTS = tuple(range(110_000, 110_024))
TEST_LAYOUTS = tuple(range(111_000, 111_024))
DEV_PALETTES = tuple(range(9_300, 9_308))
UNSEEN_PALETTES = tuple(range(9_400, 9_408))
STRATUM = "COUNT_COLLISION"

CHANNEL_PERMUTATION = (1, 2, 0)
INVERTIBLE = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64)   # det = 2


@contextmanager
def palette_pool(pool: np.ndarray):
    """Swap the colour pool the renderer draws from. `o_core.render_roles` reads the
    module global, so this changes appearance and nothing else."""
    original = O.COLOUR_POOL
    O.COLOUR_POOL = pool
    try:
        yield
    finally:
        O.COLOUR_POOL = original


def remapped_pool(kind: str) -> np.ndarray:
    base = np.array([[220, 40, 40], [40, 200, 60], [50, 80, 230], [235, 200, 40],
                     [200, 60, 210], [40, 210, 215], [250, 140, 30], [150, 150, 150]],
                    dtype=np.uint8)
    if kind == "identity":
        return base
    if kind == "index_permutation":
        return base[np.random.default_rng(4_242).permutation(len(base))]
    if kind == "channel_permutation":
        return base[:, list(CHANNEL_PERMUTATION)].copy()
    if kind == "invertible_remap":
        out = (base.astype(np.int64) @ INVERTIBLE.T) % 251
        assert len({tuple(r) for r in out}) == len(base), "remap collided two colours"
        return out.astype(np.uint8)
    raise KeyError(kind)


def build(pool_kind: str, palettes, layouts, seed: int, role_plant: bool = False):
    pool = remapped_pool(pool_kind)
    episodes = []
    with palette_pool(pool):
        for palette in palettes:
            bijection = C.sample_bijection(palette)
            if role_plant:
                # THE PLANT: the generator stops being blind to the role. AGENT always
                # gets pool entry 0 and SWITCH always entry 1, so the colour value alone
                # names two roles and guard A must see it.
                bijection = plant_bijection(palette)
            episodes.extend(C.collect(layouts, bijection, STRATUM, 9, seed=seed,
                                      policy="switch_seeking"))
    return episodes


def plant_bijection(palette: int) -> np.ndarray:
    rng = np.random.default_rng(palette)
    rest = [i for i in rng.permutation(8) if i not in (0, 1)]
    out = np.zeros(C.N_ROLES, dtype=np.int64)
    out[C.AGENT], out[C.SWITCH] = 0, 1
    for role in range(C.N_ROLES):
        if role not in (C.AGENT, C.SWITCH):
            out[role] = rest.pop()
    return out


# ---- run 1: is the generator blind to the role? -------------------------------------


def mutual_information(pairs: np.ndarray, n_role: int, n_colour: int) -> float:
    joint = np.zeros((n_role, n_colour))
    for role, colour in pairs:
        joint[role, colour] += 1
    joint /= joint.sum()
    row, column = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = joint * np.log2(joint / (row * column))
    return float(np.nansum(term))


def generator_audit(draws: int = 20_000) -> dict[str, Any]:
    honest, planted = [], []
    for palette in range(draws):
        for role, colour in enumerate(C.sample_bijection(palette)):
            honest.append((role, int(colour)))
        for role, colour in enumerate(plant_bijection(palette)):
            planted.append((role, int(colour)))
    honest = np.array(honest)
    planted = np.array(planted)
    # A finite sample gives a positive MI even when the truth is zero, so the honest
    # number is compared against a label-shuffled null rather than against 0.
    rng = np.random.default_rng(7)
    null = [mutual_information(
        np.stack([honest[:, 0], rng.permutation(honest[:, 1])], axis=1),
        C.N_ROLES, 8) for _ in range(200)]
    honest_mi = mutual_information(honest, C.N_ROLES, 8)
    return {
        "draws": draws,
        "analytic_mi_bits": 0.0,
        "analytic_basis": ("the bijection is a uniform random injection of 7 roles into "
                           "8 pool entries, so P(colour | role) = 1/8 for every pair and "
                           "the mutual information is identically zero"),
        "empirical_mi_bits_honest": honest_mi,
        "shuffled_null_mean_bits": float(np.mean(null)),
        "shuffled_null_p95_bits": float(np.percentile(null, 95)),
        "honest_within_null": bool(honest_mi <= float(np.percentile(null, 95))),
        "empirical_mi_bits_role_dependent_plant": mutual_information(planted,
                                                                    C.N_ROLES, 8),
    }


# ---- guards --------------------------------------------------------------------------


def guard_a(fit_palettes, score_palettes, plant: bool = False) -> dict[str, Any]:
    """Predict the ROLE from the colour VALUE alone, with an exact lookup Bayes rule.

    Fitted on one set of palettes and scored on a DISJOINT set, one row per
    (palette, role). Fitting and scoring on the same rows put an honest generator at
    0.3036 -- the argmax of a noisy 8-palette histogram is right far more often than
    chance ON ITS OWN DATA -- which is a property of the estimator, not of the
    generator. Held out, an honest generator has nothing to offer and lands on chance.
    """
    def rows_for(palettes):
        out = []
        for palette in palettes:
            bijection = plant_bijection(palette) if plant else C.sample_bijection(palette)
            for role in range(C.N_ROLES):
                out.append((tuple(int(v) for v in O.COLOUR_POOL[bijection[role]]), role))
        return out

    fit, score = rows_for(fit_palettes), rows_for(score_palettes)
    table: dict[tuple, np.ndarray] = {}
    for colour, role in fit:
        table.setdefault(colour, np.zeros(C.N_ROLES))[role] += 1
    rng = np.random.default_rng(11)
    fallback = int(rng.integers(0, C.N_ROLES))
    predicted = {c: int(v.argmax()) for c, v in table.items()}
    hits = np.array([float(predicted.get(c, fallback) == r) for c, r in score])
    per_role = [float(np.mean([h for (c, r), h in zip(score, hits) if r == role]))
                for role in range(C.N_ROLES)]
    draws = [float(np.mean(hits[rng.integers(0, len(hits), len(hits))]))
             for _ in range(4_000)]
    return {"balanced_accuracy": float(np.mean(per_role)),
            "chance": 1.0 / C.N_ROLES,
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)),
            "fit_palettes": len(list(fit_palettes)),
            "scored_palettes": len(list(score_palettes)),
            "distinct_colours_in_fit": len(table),
            "leak_detected": bool(np.percentile(draws, 2.5) > 1.0 / C.N_ROLES)}


def guard_b(tokens: np.ndarray, palette_label: np.ndarray, fields: str,
            plant: bool = False) -> dict[str, Any]:
    """Predict the PALETTE ID from a named field group, by nearest centroid.

    The field group matters and the first version of this guard got it wrong. Offered
    the per-colour statistics, an honest generator scores 0.93 -- and correctly so: the
    count and motion pattern ACROSS colour slots is precisely the behavioural evidence
    that identifies the bijection, and it is the same evidence the exact posterior uses.
    Calling that a leak would condemn the mechanism the phase is trying to build.

    The guard therefore asks the well-posed question instead: can the palette be read
    from the GLOBAL block -- action, displacement, moved -- which is colour-free and
    byte-identical across palettes for the same trajectory? Nothing honest can. The
    per-colour result is still reported, labelled as the intended channel.
    """
    if fields == "global_only":
        # Max-pooled over slots, so the feature is the SHARED global vector and nothing
        # else. Flattening the slots instead leaves slot OCCUPANCY in the feature, and
        # occupancy is which colours are on screen -- a palette fingerprint that is
        # public by construction. That version scored 0.3509 on an honest generator and
        # was measuring the colour set, not a hidden channel.
        features = tokens[:, :, C.GLOBAL].max(axis=1)
    elif fields == "occupancy":
        features = (np.abs(tokens).sum(axis=2) > 0).astype(np.float32)
    else:
        features = M.mask_view(tokens, fields).reshape(len(tokens), -1)
    if plant:
        # THE PLANT: the palette id, scaled, written into the global block.
        features = features.copy()
        features[:, 0] = palette_label.astype(np.float32) / 10.0
    unique = np.unique(palette_label)
    split = np.random.default_rng(3).random(len(features)) < 0.5
    centroids = np.stack([features[split & (palette_label == p)].mean(axis=0)
                          for p in unique])
    held = ~split
    distance = ((features[held][:, None, :] - centroids[None]) ** 2).sum(-1)
    predicted = unique[distance.argmin(axis=1)]
    truth = palette_label[held]
    per_class = [float((predicted[truth == p] == p).mean()) for p in unique
                 if (truth == p).any()]
    accuracy = float(np.mean(per_class))
    chance = 1.0 / len(unique)
    return {"fields": fields, "balanced_accuracy": accuracy, "chance": chance,
            "palettes": int(len(unique)),
            "leak_detected": bool(accuracy > chance + 0.15)}


def equivariance_check(seed: int, tokens: np.ndarray, before, after) -> dict[str, Any]:
    """Permuting the colour tokens must permute the assignment and leave the event
    logit unchanged. Exact, not sampled."""
    import mlx.core as mx

    model = M.build_stateless(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(C.MAX_COLOURS)
    inverse = np.argsort(order)
    permuted_tokens = tokens[:, order]
    remap = np.zeros(C.MAX_COLOURS, dtype=np.int64)
    remap[order] = np.arange(C.MAX_COLOURS)
    base = np.asarray(model(mx.array(tokens), mx.array(before), mx.array(after)))
    moved = np.asarray(model(mx.array(permuted_tokens), mx.array(remap[before]),
                             mx.array(remap[after])))
    role_base = M.assignment_of(model, tokens)
    role_moved = M.assignment_of(model, permuted_tokens)[:, inverse]
    return {"max_logit_difference": float(np.abs(base - moved).max()),
            "max_assignment_difference": float(np.abs(role_base - role_moved).max()),
            "equivariant": bool(np.abs(base - moved).max() < 1e-4
                                and np.abs(role_base - role_moved).max() < 1e-5)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o2-leakage.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    report: dict[str, Any] = {
        "stratum": STRATUM, "seeds": list(SEEDS[:arguments.seeds]),
        "development_palettes": list(DEV_PALETTES),
        "unseen_palettes": list(UNSEEN_PALETTES),
        "train_layouts": list(TRAIN_LAYOUTS), "test_layouts": list(TEST_LAYOUTS),
        "runs": {},
    }

    print("1. generator audit", flush=True)
    report["runs"]["1_iid_palette_generation"] = generator_audit()
    block = report["runs"]["1_iid_palette_generation"]
    print(f"   I(role; colour) honest {block['empirical_mi_bits_honest']:.6f} bits "
          f"(null p95 {block['shuffled_null_p95_bits']:.6f}); "
          f"role-dependent plant {block['empirical_mi_bits_role_dependent_plant']:.4f}",
          flush=True)

    accuracy: dict[str, np.ndarray] = {}
    meta_reference = None
    for kind in ("identity", "index_permutation", "invertible_remap",
                 "channel_permutation"):
        per_seed = []
        for seed in SEEDS[:arguments.seeds]:
            train = build(kind, DEV_PALETTES, TRAIN_LAYOUTS, 11)
            test = build(kind, UNSEEN_PALETTES, TEST_LAYOUTS, 313)
            registry = C.ColourRegistry().scan(train + test)
            train_data = C.pair_dataset(train, registry)
            test_data = C.pair_dataset(test, registry)
            block = C.as_block(train_data)
            block = (M.mask_view(block[0], "no_rgb"),) + block[1:]
            infer, _ = M.train_stateless(block, seed)
            probe = C.as_block(test_data)
            probe = (M.mask_view(probe[0], "no_rgb"),) + probe[1:]
            per_seed.append((infer(probe) > 0).astype(float)
                            == test_data["event"])
            meta_reference = test_data["meta"]
        accuracy[kind] = np.concatenate([p.astype(float) for p in per_seed])
        print(f"2-4. pool {kind:20s} accuracy {accuracy[kind].mean():.4f}", flush=True)

    n_seeds = arguments.seeds
    rows = len(meta_reference)
    seed_column = np.repeat(np.array(SEEDS[:n_seeds]), rows)
    layout_column = np.tile(meta_reference[:, 0], n_seeds)
    class_column = np.tile(meta_reference[:, 2], n_seeds)
    for kind in ("index_permutation", "invertible_remap", "channel_permutation"):
        interval = m2d.hierarchical_paired_interval(
            accuracy[kind], accuracy["identity"], seed_column, layout_column,
            class_column)
        report["runs"][f"{kind}_vs_identity"] = {
            "accuracy": float(accuracy[kind].mean()),
            "baseline": float(accuracy["identity"].mean()), **interval,
            "invariant": bool(not interval["excludes_zero"])}
        print(f"   {kind:22s} vs identity {interval['delta']:+.4f} "
              f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
              f"{'  MOVED' if interval['excludes_zero'] else '  invariant'}", flush=True)

    print("5-7. guards", flush=True)
    fit_palettes = range(9_500, 9_700)
    report["runs"]["5_colour_value_only_classifier_honest"] = guard_a(
        fit_palettes, UNSEEN_PALETTES)
    report["runs"]["7_role_dependent_colour_plant"] = guard_a(
        fit_palettes, UNSEEN_PALETTES, plant=True)
    honest = build("identity", DEV_PALETTES, TRAIN_LAYOUTS, 11)
    registry = C.ColourRegistry().scan(honest)
    data = C.pair_dataset(honest, registry)
    label = data["meta"][:, 1]
    report["runs"]["6_palette_id_plant"] = {
        "honest": guard_b(data["tokens"], label, "global_only", plant=False),
        "planted": guard_b(data["tokens"], label, "global_only", plant=True),
        "slot_occupancy_reference": {
            **guard_b(data["tokens"], label, "occupancy", plant=False),
            "is_a_leak": False,
            "why": ("which colour slots are occupied is which colours are on screen, "
                    "which a viewer can see; it identifies the palette and is meant to")},
        "per_colour_statistics_reference": {
            **guard_b(data["tokens"], label, "no_rgb", plant=False),
            "is_a_leak": False,
            "why": ("the count and motion pattern across colour slots IS the public "
                    "behavioural evidence that identifies the bijection; the exact "
                    "posterior reads the same thing, so recovering the palette from it "
                    "is the mechanism, not a leak")}}
    report["runs"]["equivariance"] = equivariance_check(
        99, data["tokens"][:256], data["before_index"][:256], data["after_index"][:256])

    a_honest = report["runs"]["5_colour_value_only_classifier_honest"]
    a_plant = report["runs"]["7_role_dependent_colour_plant"]
    b = report["runs"]["6_palette_id_plant"]
    eq = report["runs"]["equivariance"]
    print(f"   guard A honest  {a_honest['balanced_accuracy']:.4f} "
          f"[{a_honest['ci_low']:.4f}, {a_honest['ci_high']:.4f}] vs chance "
          f"{a_honest['chance']:.4f}  leak={a_honest['leak_detected']}")
    print(f"   guard A plant   {a_plant['balanced_accuracy']:.4f}  "
          f"leak={a_plant['leak_detected']}")
    print(f"   guard B honest  {b['honest']['balanced_accuracy']:.4f} vs chance "
          f"{b['honest']['chance']:.4f}  leak={b['honest']['leak_detected']}")
    print(f"   guard B plant   {b['planted']['balanced_accuracy']:.4f}  "
          f"leak={b['planted']['leak_detected']}")
    print(f"   DeepSets equivariance exact: {eq['equivariant']} "
          f"(max logit difference {eq['max_logit_difference']:.2e})")

    report["guards_are_not_vacuous"] = {
        "guard_a_passes_honest_and_catches_plant": bool(
            not a_honest["leak_detected"] and a_plant["leak_detected"]),
        "guard_b_passes_honest_and_catches_plant": bool(
            not b["honest"]["leak_detected"] and b["planted"]["leak_detected"]),
    }
    report["Q3_no_undeclared_role_information_in_palette_values"] = bool(
        report["runs"]["1_iid_palette_generation"]["honest_within_null"]
        and not a_honest["leak_detected"]
        and not b["honest"]["leak_detected"]
        and all(report["runs"][f"{k}_vs_identity"]["invariant"]
                for k in ("index_permutation", "invertible_remap",
                          "channel_permutation"))
        and eq["equivariant"]
        and all(report["guards_are_not_vacuous"].values()))
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nQ3: {report['Q3_no_undeclared_role_information_in_palette_values']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
