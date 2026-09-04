"""F. Persistent-memory replication, resampled by PALETTE rather than by row.

O2 reported the memory arm at 0.865 contested against 0.413 for the frame-pair binder,
with an interval computed over ROWS. That interval is not the one the claim needs. Rows
inside a palette share a convention, a calibration history and a layout pool, so thousands
of them do not supply thousands of independent tests of "the memory generalises to a
palette it has not seen". They supply as many independent tests as there are palettes.

Everything here therefore resamples the PALETTE as the primary unit, and the row-level
interval is reported beside it so the difference is visible rather than argued.

The arms are O2's eleven, minus the two palette-change arms, which section I now covers
with a real detector, a provisional branch and honest controls. The view is `no_rgb`:
section B proved `full_token` is not palette-equivariant, so replicating on it would
measure the defect rather than the memory.

The claim passes only if the memory beats BOTH required controls -- current-frame-only
and augmentation-only -- under palette-level paired resampling.

    .venv-shwm/bin/python experiments/shwm/o3_persistent.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

import o2_core as C
import o2_memory as mem
import o2_models as M
import o3_core as O3
import o3_population as pop
from m2d_core import ARTIFACTS, write

SEED = 66_000
VIEW = "no_rgb"
BOOTSTRAP = 4_000
FOREIGN_OFFSET = 500_000       # a fresh id space, disjoint from every reserved pool

ARMS = (
    "0_majority_class",
    "1_current_frame_binder",
    "2_frame_pair_binder",
    "3_recurrent_assignment_memory",
    "4_exact_palette_posterior",
    "5_augmentation_only_detector",
    "6_memory_reset_before_transfer",
    "7_shuffled_calibration",
    "8_wrong_colour_pairings",
    "9_calibration_from_another_palette",
    "10_no_persistent_memory",
    "11_oracle_palette_map",
)

# The two controls the claim has to beat. Named here rather than chosen after the fact.
REQUIRED_CONTROLS = ("1_current_frame_binder", "5_augmentation_only_detector")
CLAIM = "3_recurrent_assignment_memory"

# WHY BALANCED ACCURACY, MEASURED BEFORE THE ARMS WERE SCORED.
#
# O2 drew every palette's transfer rows from ONE shared layout pool, so the contested
# population had a single fixed SWITCH-against-DECOY base rate. Section C requires each
# palette to draw its own content, and that makes the base rate a per-palette quantity:
# measured over 16 validation palettes it is 0.5898 overall and ranges from 0.2667 to
# 0.8000. A constant "always SWITCH" therefore scores 0.5898 on plain accuracy, and the
# first run of this section returned 0.6182 for the stateless frame-pair binder -- which
# is that base rate, not a capability, and which would otherwise have appeared to breach
# O2's exact count-only ceiling of 0.5000 on contested rows.
#
# Every headline here is BALANCED accuracy, on which the majority-class strategy scores
# exactly 0.5. Arm 0 is that strategy, carried as the calibration arm: if it does not
# land on 0.5, the statistic is wrong and nothing below it means anything.
BALANCED = ("mean of the per-class accuracies on contested rows, so a constant answer "
            "scores 0.5 whatever the palette's base rate")


def group_of(plan: dict, stratum: str = "COUNT_COLLISION") -> mem.Group:
    """An o2_memory.Group over an o3 palette plan, so each palette draws its own
    content. Section C established that this is required: once the pipeline is bit-exact
    palette-equivariant, palettes that share content are identical by construction and
    between-palette variance would be exactly zero."""
    scenario = pop.palette_scenario(plan, stratum)
    return mem.Group(plan["palette"], plan["bijection"], scenario.calibration,
                     scenario.transfer)


def palette_bootstrap(values: dict[int, tuple[np.ndarray, np.ndarray]],
                      palettes: list[int], rng, draws: int
                      ) -> tuple[float, float, float]:
    """Hierarchical resample: palette first, then rows within class within the palette.

    Resampling rows alone would treat a palette's rows as independent replicates of the
    palette, which is the error this section exists to correct.
    """
    means = np.empty(draws)
    for b in range(draws):
        drawn = rng.choice(palettes, len(palettes), replace=True)
        pooled = [resample_within(*values[int(p)], rng) for p in drawn
                  if len(values[int(p)][0])]
        pooled = [v for v in pooled if np.isfinite(v)]
        means[b] = float(np.mean(pooled)) if pooled else np.nan
    finite = means[np.isfinite(means)]
    point = [balanced(c, t) for c, t in values.values() if len(c)]
    point = [v for v in point if np.isfinite(v)]
    return (float(np.mean(point)),
            float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)))


def row_bootstrap(correct: np.ndarray, truth: np.ndarray, rng, draws: int
                  ) -> tuple[float, float, float]:
    """The interval O2 reported: rows pooled across palettes, treated as exchangeable.
    Kept so the two can be compared, not to be believed."""
    means = np.array([resample_within(correct, truth, rng) for _ in range(draws)])
    return (balanced(correct, truth), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def paired_palette_contrast(a: dict[int, tuple], b: dict[int, tuple],
                            palettes: list[int], rng, draws: int) -> dict[str, float]:
    """A vs B on balanced accuracy, resampling PALETTES and pairing inside each one."""
    per_palette = np.array([balanced(*a[p]) - balanced(*b[p]) for p in palettes
                            if len(a[p][0]) and len(b[p][0])])
    per_palette = per_palette[np.isfinite(per_palette)]
    means = np.array([per_palette[rng.integers(0, len(per_palette),
                                               len(per_palette))].mean()
                      for _ in range(draws)])
    low, high = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    return {"delta": float(per_palette.mean()), "low": low, "high": high,
            "excludes_zero": bool(low > 0.0),
            "palettes_where_a_wins": int((per_palette > 0).sum()),
            "palettes": int(len(per_palette))}


def by_palette(correct: np.ndarray, groups: np.ndarray, mask: np.ndarray,
               palettes: list[int], truth: np.ndarray
               ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {p: (correct[mask & (groups == p)], truth[mask & (groups == p)])
            for p in palettes}


def balanced(correct: np.ndarray, truth: np.ndarray) -> float:
    """Mean of the per-class accuracies. A constant answer scores exactly 0.5."""
    parts = [correct[truth == value].mean() for value in (0.0, 1.0)
             if (truth == value).any()]
    return float(np.mean(parts)) if parts else float("nan")


def resample_within(correct: np.ndarray, truth: np.ndarray, rng) -> float:
    """Bootstrap a palette's rows WITHIN each class, so the resample cannot change the
    class balance and reintroduce the skew the balanced statistic removes."""
    parts = []
    for value in (0.0, 1.0):
        rows = correct[truth == value]
        if len(rows):
            parts.append(rows[rng.integers(0, len(rows), len(rows))].mean())
    return float(np.mean(parts)) if parts else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-palettes", type=int, default=32)
    parser.add_argument("--palettes", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o3-persistent.json")
    arguments = parser.parse_args()
    started = time.perf_counter()
    registry = C.canonical_registry()
    rng = np.random.default_rng(SEED)

    print(f"building {arguments.train_palettes} development groups", flush=True)
    dev_groups = [group_of(pop.palette_plan(p, 6, 20, 2))
                  for p in pop.DEV_PALETTES[:arguments.train_palettes]]
    # Built through o2_memory's own stacker so this is a replication of that pipeline
    # and not a reimplementation of it. Only the view and the populations change.
    train = mem.stack_groups(dev_groups, registry, VIEW)

    print(f"building {arguments.palettes} validation groups", flush=True)
    plans = [pop.palette_plan(p, 6, 8, 1)
             for p in pop.VALIDATION_PALETTES[:arguments.palettes]]
    groups = [group_of(plan) for plan in plans]
    palettes = [g.palette for g in groups]

    # A foreign population for the wrong-history arm. Deliberately NOT drawn from
    # REPLICATION_PALETTES: section C reserves those as untouched until every decision is
    # frozen, and spending them on an ablation would forfeit the replication.
    foreign = [group_of(pop.palette_plan(FOREIGN_OFFSET + p, 6, 8, 1))
               for p in palettes]

    evaluation = {
        CLAIM: mem.stack_groups(groups, registry, VIEW),
        "6_memory_reset_before_transfer": mem.stack_groups(groups, registry, VIEW,
                                                           history=False),
        "7_shuffled_calibration": mem.stack_groups(groups, registry, VIEW,
                                                   shuffle_seed=5),
        "8_wrong_colour_pairings": mem.stack_groups(groups, registry, VIEW,
                                                    slot_permutation=17),
        "9_calibration_from_another_palette": mem.stack_groups(
            groups, registry, VIEW, history_source=foreign),
    }
    base = evaluation[CLAIM]
    mask = mem.contested(base)
    group_ids = base["group"]
    truth = base["event"].astype(float)

    # The calibration arm: answer each palette's majority class on every row. Balanced
    # accuracy MUST come back at 0.5 -- if it does not, the statistic is broken and no
    # arm below it can be read.
    majority = np.zeros(len(truth), np.float32)
    base_rates = {}
    for palette in [g.palette for g in groups]:
        rows = mask & (group_ids == palette)
        if not rows.any():
            continue
        rate = float(truth[rows].mean())
        base_rates[str(palette)] = rate
        majority[rows] = (truth[rows] == (1.0 if rate >= 0.5 else 0.0)).astype(float)
    print(f"{len(base['event'])} transfer pairs, {int(mask.sum())} contested, "
          f"over {len(palettes)} palettes", flush=True)

    # ---- memoryless and oracle arms ---------------------------------------------------
    correct: dict[str, np.ndarray] = {}

    correct["0_majority_class"] = majority

    exact = mem.exact_posterior_arm(groups, base)
    correct["4_exact_palette_posterior"] = (
        (exact > 0.5).astype(float) == base["event"]).astype(float)

    truth_tables = mem.truth_assignment(groups, registry)
    oracle = np.stack([truth_tables[int(p)] for p in group_ids])
    oracle_event = mem.numpy_event(oracle, base["before"], base["after"])
    correct["11_oracle_palette_map"] = (
        (oracle_event > 0.5).astype(float) == base["event"]).astype(float)

    # ---- the learned arms, averaged over restarts by the M2F rule ---------------------
    print("training the memory and the memoryless binders", flush=True)
    seed_hits: dict[str, list[np.ndarray]] = {}
    frame_train = M.mask_view(train["tokens"], "single_frame")
    frame_evaluation = M.mask_view(base["tokens"], "single_frame")
    for seed in range(arguments.seeds):
        infer, _ = M.train_memory(
            (train["sequence"], train["mask"], train["before"], train["after"],
             train["event"]), SEED + seed, updates=mem.MEMORY_UPDATES)
        for arm, data in evaluation.items():
            logits = infer((data["sequence"], data["mask"], data["before"],
                            data["after"]))
            seed_hits.setdefault(arm, []).append(
                ((logits > 0).astype(float) == data["event"]).astype(float))

        pair_infer, _ = M.train_stateless(
            (train["tokens"], train["before"], train["after"], train["event"]), seed)
        frame_infer, _ = M.train_stateless(
            (frame_train, train["before"], train["after"], train["event"]), seed)
        for arm, logits in (
                ("2_frame_pair_binder",
                 pair_infer((base["tokens"], base["before"], base["after"]))),
                ("1_current_frame_binder",
                 frame_infer((frame_evaluation, base["before"], base["after"])))):
            seed_hits.setdefault(arm, []).append(
                ((logits > 0).astype(float) == base["event"]).astype(float))

        # Palette augmentation is the standard answer to appearance shift and consumes
        # the raw frames directly, which is exactly why it is the control the memory
        # claim has to beat.
        seed_hits.setdefault("5_augmentation_only_detector", []).append(
            ((mem.augmentation_detector(dev_groups, groups, registry, seed) > 0
              ).astype(float) == base["event"]).astype(float))
        print(f"  restart {seed}: memory {seed_hits[CLAIM][-1][mask].mean():.4f}  "
              f"pair {seed_hits['2_frame_pair_binder'][-1][mask].mean():.4f}  "
              f"augmentation "
              f"{seed_hits['5_augmentation_only_detector'][-1][mask].mean():.4f}",
              flush=True)
    # Per-seed values are kept as well as the average. `train_memory` already selects
    # the best of four restarts by TRAINING loss (the M2F rule), so what remains here is
    # variance BETWEEN those selected runs -- and on the memory arm it is large. The
    # headline is the seed AVERAGE, never the best seed, and the spread is reported so a
    # reader can see how much of any gap is restart luck.
    per_seed = {}
    for arm, runs in seed_hits.items():
        correct[arm] = np.mean(np.stack(runs), axis=0)
        per_seed[arm] = [float(balanced(run[mask], truth[mask])) for run in runs]

    # O2's arm 10 is the frame-pair binder scored on the same rows: no persistent state.
    correct["10_no_persistent_memory"] = correct["2_frame_pair_binder"]

    # ---- intervals --------------------------------------------------------------------
    report: dict[str, Any] = {
        "seed": SEED, "view": VIEW,
        "why_no_rgb": ("section B proved full_token is not palette-equivariant; "
                       "replicating on it would measure the defect, not the memory"),
        "primary_randomisation_unit": "palette",
        "bootstrap_draws": BOOTSTRAP,
        "palettes": palettes,
        "foreign_palettes": [g.palette for g in foreign],
        "restarts": arguments.seeds,
        "contested_rows": int(mask.sum()),
        "rows": int(len(base["event"])),
        "headline_statistic": BALANCED,
        "switch_base_rate_overall": float(truth[mask].mean()),
        "switch_base_rate_per_palette": base_rates,
        "switch_base_rate_range": [min(base_rates.values()), max(base_rates.values())],
        "arms": {},
    }

    print(f"\nswitch base rate {float(truth[mask].mean()):.4f} overall, "
          f"{min(base_rates.values()):.4f}-{max(base_rates.values()):.4f} per palette; "
          f"headline is BALANCED accuracy")
    print(f"\n{'arm':38s} {'balanced':>13s} {'palette 95%':>22s} "
          f"{'row 95%':>22s} {'plain':>8s}")
    print("-" * 110)
    for arm in ARMS:
        values = correct.get(arm)
        if values is None:
            report["arms"][arm] = {"status": "NOT_RUN",
                                   "reason_class": "helper_absent"}
            continue
        per_palette = by_palette(values, group_ids, mask, palettes, truth)
        mean, low, high = palette_bootstrap(per_palette, palettes, rng, BOOTSTRAP)
        r_mean, r_low, r_high = row_bootstrap(values[mask], truth[mask], rng, BOOTSTRAP)
        scores = {p: balanced(*per_palette[p]) for p in palettes
                  if len(per_palette[p][0])}
        collapsed = int(sum(1 for v in scores.values() if v <= 0.5))
        report["arms"][arm] = {
            "balanced_accuracy_palette_mean": mean,
            "palette_ci": [low, high],
            "palette_ci_width": high - low,
            "balanced_accuracy_row_pooled": r_mean,
            "row_ci": [r_low, r_high],
            "row_ci_width": r_high - r_low,
            "interval_width_ratio": ((high - low) / (r_high - r_low)
                                     if r_high > r_low else None),
            "plain_accuracy_contested": float(values[mask].mean()),
            "palettes_at_or_below_chance": collapsed,
            "per_seed_balanced_accuracy": per_seed.get(arm),
            "seed_spread": (max(per_seed[arm]) - min(per_seed[arm])
                            if per_seed.get(arm) else None),
            "per_palette": {str(p): float(v) for p, v in scores.items()},
        }
        print(f"{arm:38s} {mean:13.4f} [{low:9.4f}, {high:9.4f}] "
              f"[{r_low:9.4f}, {r_high:9.4f}] "
              f"{report['arms'][arm]['plain_accuracy_contested']:8.4f}", flush=True)

    # ---- the required contrasts -------------------------------------------------------
    claim_rows = by_palette(correct[CLAIM], group_ids, mask, palettes, truth)
    contrasts = {}
    for control in REQUIRED_CONTROLS:
        if correct.get(control) is None:
            contrasts[control] = {"status": "NOT_RUN", "reason_class": "helper_absent"}
            continue
        control_rows = by_palette(correct[control], group_ids, mask, palettes, truth)
        contrasts[control] = paired_palette_contrast(claim_rows, control_rows,
                                                     palettes, rng, BOOTSTRAP)
        print(f"\n{CLAIM} - {control}: {contrasts[control]['delta']:+.4f} "
              f"[{contrasts[control]['low']:+.4f}, {contrasts[control]['high']:+.4f}]  "
              f"wins on {contrasts[control]['palettes_where_a_wins']}/"
              f"{contrasts[control]['palettes']} palettes", flush=True)

    report["required_contrasts"] = contrasts
    calibration = report["arms"]["0_majority_class"]["balanced_accuracy_palette_mean"]
    report["calibration_arm_lands_on_chance"] = bool(abs(calibration - 0.5) < 1e-6)
    usable = [c for c in contrasts.values() if "excludes_zero" in c]
    # The calibration arm gates the whole section: a statistic on which a constant
    # answer does NOT score 0.5 is not measuring contested-row capability at all.
    report["R4_persistent_memory_replicates_at_palette_level"] = bool(
        report["calibration_arm_lands_on_chance"]
        and usable and len(usable) == len(REQUIRED_CONTROLS)
        and all(c["excludes_zero"] for c in usable))
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nR4 {report['R4_persistent_memory_replicates_at_palette_level']}")
    print(f"wrote {arguments.out}  "
          f"({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
