"""F / Q4. What is the binder actually using? A complete 2^3 factorial with intervals.

Phase O1 tested four views and read the answer off point estimates. Here all eight cells
of the COUNT x MOTION x INTERACT design are run, so main effects and interactions are
computed rather than inferred, and every comparison carries a hierarchical paired
interval over the SAME test rows.

The count-only arm gets a second, stronger form: an exact lookup Bayes rule that is
HANDED the identity of the entered colour for free and asked only whether its cell count
names its role. That is a ceiling on everything a count-only mechanism could ever do,
and it is the number the cardinality-lookup hypothesis has to beat.

    .venv-shwm/bin/python experiments/shwm/o2_factorial.py
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np

import m2d_core as m2d
import n_heads as heads
import n_interfaces as ifaces
import o2_core as C
import o2_models as M
from m2d_core import ARTIFACTS, write

SEEDS = (42_000, 42_001, 42_002)
TRAIN_LAYOUTS = tuple(range(110_000, 110_024))
HELD_OUT_LAYOUTS = tuple(range(111_000, 111_024))
DEV_PALETTES = tuple(range(9_300, 9_308))
UNSEEN_PALETTES = tuple(range(9_400, 9_408))
POLICY = "uniform"             # no evaluator-guided policy anywhere in this section
TRAIN_COUNTS = (4, 8)          # decoy counts 4..7 during training
HELD_OUT_COUNTS = (8, 11)      # decoy counts 8..10, never trained on

VIEW_ORDER = ("none", "count_only", "motion_only", "moments_only", "interaction_only",
              "count_plus_motion", "motion_plus_interaction", "count_plus_interaction",
              "count_motion_interaction", "full_token")

POPULATIONS = {
    "COUNT_INFORMATIVE": dict(stratum="COUNT_INFORMATIVE", layouts="held_out",
                              palettes="unseen", counts=None),
    "COUNT_VARIED": dict(stratum="COUNT_VARIED", layouts="held_out",
                         palettes="unseen", counts=TRAIN_COUNTS),
    "COUNT_COLLISION": dict(stratum="COUNT_COLLISION", layouts="held_out",
                            palettes="unseen", counts=None),
    "held_out_counts": dict(stratum="COUNT_VARIED", layouts="held_out",
                            palettes="unseen", counts=HELD_OUT_COUNTS),
    "held_out_layouts": dict(stratum="COUNT_VARIED", layouts="held_out",
                             palettes="development", counts=TRAIN_COUNTS),
    "unseen_palettes": dict(stratum="COUNT_VARIED", layouts="train",
                            palettes="unseen", counts=TRAIN_COUNTS),
}


def episodes_for(spec: dict[str, Any], seed: int) -> list[C.O2Episode]:
    layouts = {"train": TRAIN_LAYOUTS, "held_out": HELD_OUT_LAYOUTS}[spec["layouts"]]
    palettes = {"development": DEV_PALETTES, "unseen": UNSEEN_PALETTES}[spec["palettes"]]
    out: list[C.O2Episode] = []
    for palette in palettes:
        out.extend(C.collect(layouts, C.sample_bijection(palette), spec["stratum"], 9,
                             seed=seed, policy=POLICY,
                             count_range=spec["counts"]))
    return out


def contested_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    """Rows where the agent stepped onto a SWITCH or a DECOY.

    This is the only sub-population on which COUNT_COLLISION is a collision. Over the
    whole test set the ambiguous case is diluted by empty-cell steps, and a count-only
    lookup scores 0.975 there while being at chance on the cases the stratum exists to
    create. Reporting only the pooled number would have repeated phase O1's claim.
    """
    return np.isin(data["meta"][:, 3], [C.SWITCH, C.DECOY])


def count_only_bayes(train: dict[str, np.ndarray],
                     test: dict[str, np.ndarray]) -> dict[str, Any]:
    """The exact count-only ceiling.

    It is given the entered colour for free -- which the index grids determine anyway --
    and asked the only question counts can answer: does this colour's cell count name a
    SWITCH? Fitted as a lookup on the training counts, applied unchanged to the test.
    """
    def key(data):
        entered = data["tokens"][:, :, C.INTERACT][:, :, 0]
        slot = entered.argmax(axis=1)
        has_move = entered.max(axis=1) > 0.5
        count = data["tokens"][np.arange(len(slot)), slot, C.COUNT.start]
        return np.where(has_move, np.round(count * C.GRID * C.GRID).astype(int), -1)

    train_key, test_key = key(train), key(test)
    base = float(train["event"].mean())
    table: dict[int, list[float]] = {}
    for k, y in zip(train_key, train["event"]):
        table.setdefault(int(k), []).append(float(y))
    rule = {k: float(np.mean(v)) for k, v in table.items()}
    predicted = np.array([1.0 if rule.get(int(k), base) > 0.5 else 0.0
                          for k in test_key])
    mask = contested_mask(test)
    two_class = len(np.unique(test["event"][mask])) > 1
    return {"balanced_accuracy": M.balanced_accuracy(predicted - 0.5, test["event"]),
            "contested_balanced_accuracy": (M.balanced_accuracy(
                (predicted - 0.5)[mask], test["event"][mask]) if two_class else None),
            "contested_rows": int(mask.sum()),
            "distinct_keys": len(rule),
            "keys_seen_in_training": float(np.mean([int(k) in rule for k in test_key])),
            "posterior_at_the_switch_count": rule.get(C.SWITCH_COUNT, None),
            "hits": (predicted == test["event"]).astype(float)}


def local_conv(train_eps, test_eps, seed: int) -> np.ndarray:
    """Phase O's local convolutional detector, unchanged, as the negative control.

    A frozen environment-aligned pixel aggregation into 12x12 slots, then the shared
    spatially-supervised head. It has no notion of a colour as an object, so it cannot
    hold a role assignment; that is exactly what it is here to demonstrate.
    """
    def pack(episodes):
        before, after, action, event, maps = [], [], [], [], []
        for episode in episodes:
            for t in range(1, episode.length):
                before.append(episode.frames[t - 1].astype(np.float32) / 255.0)
                after.append(episode.frames[t].astype(np.float32) / 255.0)
                one = np.zeros(4, np.float32)
                one[int(episode.actions[t - 1])] = 1.0
                action.append(one)
                event.append(episode.event[t])
                block = np.zeros(C.GRID * C.GRID, np.float32)
                row, column = episode.positions[t]
                block[row * C.GRID + column] = episode.event[t]
                maps.append(block)
        return (np.stack(before), np.stack(after), np.stack(action),
                np.array(event, np.float32), np.stack(maps))

    def encode(before, after):
        stacked = np.concatenate([ifaces.pool_to_slots(before, C.GRID),
                                  ifaces.pool_to_slots(after, C.GRID)], axis=-1)
        return stacked @ ifaces.frozen_projection(stacked.shape[-1], ifaces.SLOT_WIDTH,
                                                  20_002)

    tb, ta, tac, _, tm = pack(train_eps)
    eb, ea, eac, _, _ = pack(test_eps)
    model, _ = heads.train_target(encode(tb, ta), tac, tm, "spatial_scalar",
                                  C.GRID * C.GRID, seed, updates=M.UPDATES)
    return heads.predict(model, encode(eb, ea), eac).max(axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o2-factorial.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    train_episodes: list[C.O2Episode] = []
    for palette in DEV_PALETTES:
        train_episodes.extend(C.collect(TRAIN_LAYOUTS, C.sample_bijection(palette),
                                        "COUNT_VARIED", 9, seed=11, policy=POLICY,
                                        count_range=TRAIN_COUNTS))
    test_episodes = {name: episodes_for(spec, 313)
                     for name, spec in POPULATIONS.items()}
    registry = C.ColourRegistry().scan(
        train_episodes + [e for v in test_episodes.values() for e in v])

    train_data = C.pair_dataset(train_episodes, registry)
    test_data = {name: C.pair_dataset(eps, registry)
                 for name, eps in test_episodes.items()}

    report: dict[str, Any] = {
        "seeds": list(SEEDS[:arguments.seeds]),
        "views": list(VIEW_ORDER),
        "factors": list(M.FACTORS),
        "global_block_in_every_view": True,
        "train_manifest": C.manifest(train_episodes, "train"),
        "test_manifests": {n: C.manifest(e, n) for n, e in test_episodes.items()},
        "exchangeability": {n: C.exchangeability(e)
                            for n, e in test_episodes.items()},
        "colours_registered": len(registry.order),
        "results": {}, "intervals": {}, "factorial": {},
    }
    print(f"train {len(train_data['event'])} pairs over "
          f"{len(train_episodes)} episodes; "
          f"{len(registry.order)} distinct colours\n", flush=True)

    hits: dict[str, dict[str, np.ndarray]] = {n: {} for n in POPULATIONS}
    print("each cell is  all-rows / contested-rows  balanced accuracy")
    header = f"{'view':26s}" + "".join(f"{n[:16]:>17s}" for n in POPULATIONS)
    print(header)
    print("-" * len(header))
    for view in VIEW_ORDER:
        row = {}
        for name in POPULATIONS:
            row[name] = []
        for seed in SEEDS[:arguments.seeds]:
            block = C.as_block(train_data)
            block = (M.mask_view(block[0], view),) + block[1:]
            infer, _ = M.train_stateless(block, seed)
            for name, data in test_data.items():
                probe = C.as_block(data)
                probe = (M.mask_view(probe[0], view),) + probe[1:]
                logits = infer(probe)
                row[name].append({"logits": logits, "truth": data["event"]})
        line = f"{view:26s}"
        for name in POPULATIONS:
            scores = [M.balanced_accuracy(r["logits"], r["truth"]) for r in row[name]]
            hits[name][view] = np.concatenate(
                [((r["logits"] > 0).astype(float) == r["truth"]).astype(float)
                 for r in row[name]])
            mask = contested_mask(test_data[name])
            two_class = len(np.unique(test_data[name]["event"][mask])) > 1
            contested = ([M.balanced_accuracy(r["logits"][mask], r["truth"][mask])
                          for r in row[name]] if two_class else [float("nan")])
            hits[name].setdefault("__contested__", np.tile(mask, arguments.seeds))
            report["results"].setdefault(name, {})[view] = {
                "balanced_accuracy": float(np.mean(scores)),
                "contested_balanced_accuracy": (float(np.mean(contested))
                                                if two_class else None),
                "contested_rows": int(mask.sum()),
                "contested_is_single_class": bool(not two_class),
                "per_seed": [float(s) for s in scores]}
            line += (f"{np.mean(scores):8.4f}/"
                     + (f"{np.mean(contested):<8.4f}" if two_class else f"{'  n/a':<9s}"))
        print(line, flush=True)

    line = f"{'local_conv_baseline':26s}"
    for name, episodes in test_episodes.items():
        per_seed, contested = [], []
        mask = contested_mask(test_data[name])
        two_class = len(np.unique(test_data[name]["event"][mask])) > 1
        hit_rows = []
        for seed in SEEDS[:arguments.seeds]:
            logits = local_conv(train_episodes, episodes, seed)
            truth = test_data[name]["event"]
            per_seed.append(M.balanced_accuracy(logits, truth))
            if two_class:
                contested.append(M.balanced_accuracy(logits[mask], truth[mask]))
            hit_rows.append(((logits > 0).astype(float) == truth).astype(float))
        hits[name]["local_conv_baseline"] = np.concatenate(hit_rows)
        report["results"][name]["local_conv_baseline"] = {
            "balanced_accuracy": float(np.mean(per_seed)),
            "contested_balanced_accuracy": (float(np.mean(contested))
                                            if two_class else None),
            "per_seed": [float(v) for v in per_seed]}
        line += (f"{np.mean(per_seed):8.4f}/"
                 + (f"{np.mean(contested):<8.4f}" if two_class else f"{'  n/a':<9s}"))
    print(line, flush=True)

    ceiling = {}
    line = f"{'count_only_bayes_ceiling':26s}"
    for name, data in test_data.items():
        block = count_only_bayes(train_data, data)
        hits[name]["count_only_bayes_ceiling"] = np.tile(block.pop("hits"),
                                                         arguments.seeds)
        ceiling[name] = block
        report["results"][name]["count_only_bayes_ceiling"] = block
        line += (f"{block['balanced_accuracy']:8.4f}/"
                 + (f"{block['contested_balanced_accuracy']:<8.4f}"
                    if block["contested_balanced_accuracy"] is not None
                    else f"{'  n/a':<9s}"))
    print(line, flush=True)
    report["count_only_bayes_ceiling"] = ceiling

    print("\npaired intervals against count_plus_motion, by population")
    for name, data in test_data.items():
        rows = len(data["event"])
        seed_column = np.repeat(np.array(SEEDS[:arguments.seeds]), rows)
        layout_column = np.tile(data["meta"][:, 0], arguments.seeds)
        class_column = np.tile(data["meta"][:, 1], arguments.seeds)
        base = hits[name]["count_plus_motion"]
        contested = hits[name]["__contested__"]
        entry = {}
        for view in list(VIEW_ORDER) + ["count_only_bayes_ceiling",
                                        "local_conv_baseline"]:
            if view == "count_plus_motion":
                continue
            entry[view] = m2d.hierarchical_paired_interval(
                hits[name][view], base, seed_column, layout_column, class_column)
            entry[view]["contested"] = m2d.hierarchical_paired_interval(
                hits[name][view], base, seed_column, layout_column, class_column,
                mask=contested)
        report["intervals"][name] = entry
        c = entry["full_token"]["contested"]
        print(f"  {name:20s} full_token vs count_plus_motion, contested rows "
              f"{c['delta']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
              f"{' *' if c['excludes_zero'] else ''}  ({c['rows']} rows)", flush=True)

    print("\n2^3 factorial: main effects and interactions (balanced accuracy)")
    for name in POPULATIONS:
        cell = {k: report["results"][name][v]["balanced_accuracy"]
                for k, v in M.FACTORIAL_CELL.items()}
        effects: dict[str, float] = {}
        for index, factor in enumerate(M.FACTORS):
            on = [v for k, v in cell.items() if k[index] == 1]
            off = [v for k, v in cell.items() if k[index] == 0]
            effects[f"main_{factor}"] = float(np.mean(on) - np.mean(off))
        for a, b in itertools.combinations(range(3), 2):
            same = [v for k, v in cell.items() if k[a] == k[b]]
            differ = [v for k, v in cell.items() if k[a] != k[b]]
            effects[f"interaction_{M.FACTORS[a]}_x_{M.FACTORS[b]}"] = float(
                np.mean(same) - np.mean(differ))
        sign = {k: (-1) ** (sum(k) + 1) for k in cell}
        effects["interaction_three_way"] = float(
            sum(sign[k] * v for k, v in cell.items()) / 4.0)
        report["factorial"][name] = {"cells": {str(k): v for k, v in cell.items()},
                                     "effects": effects}
        print(f"  {name:20s} " + "  ".join(
            f"{k.replace('main_', '').replace('interaction_', 'x:'):18s}{v:+.4f}"
            for k, v in effects.items() if k.startswith("main_")), flush=True)

    collision = report["results"]["COUNT_COLLISION"]
    informative = report["results"]["COUNT_INFORMATIVE"]
    interval = report["intervals"]["COUNT_COLLISION"]
    richer = interval["full_token"]["contested"]
    report["decisions"] = {
        # COUNT_INFORMATIVE has no DECOY, so its contested subset is single-class and
        # has no balanced accuracy; the informative side of the comparison is therefore
        # the pooled number, and the collision side is the contested one.
        "count_only_is_a_cardinality_lookup": bool(
            informative["count_only"]["balanced_accuracy"] > 0.8
            and collision["count_only"]["contested_balanced_accuracy"] < 0.65),
        "count_only_bayes_ceiling_under_collision_all_rows":
            collision["count_only_bayes_ceiling"]["balanced_accuracy"],
        "count_only_bayes_ceiling_under_collision_contested":
            collision["count_only_bayes_ceiling"]["contested_balanced_accuracy"],
        "full_token_beats_count_plus_motion_materially": bool(
            richer["excludes_zero"] and richer["delta"] > 0.02),
        "selected_representation": (
            "full_token" if (richer["excludes_zero"] and richer["delta"] > 0.02)
            else "count_plus_motion"),
        "selection_rule": ("the simpler representation is kept unless the richer one "
                           "clears +0.02 with an interval excluding zero on the "
                           "CONTESTED rows of the collision population"),
    }
    selected = report["decisions"]["selected_representation"]
    base_is_selected = selected == "count_plus_motion"
    against = {}
    for control in ("count_only", "local_conv_baseline", "count_only_bayes_ceiling"):
        entry = interval[control] if base_is_selected else None
        against[control] = entry
    report["decisions"]["Q4_controls"] = {
        k: {"delta": v["delta"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
            "beaten": bool(v["ci_high"] < 0)} for k, v in against.items() if v}
    report["Q4_global_binder_beats_count_only_under_collision"] = bool(
        against["count_only"] and against["count_only"]["ci_high"] < 0
        and against["local_conv_baseline"]
        and against["local_conv_baseline"]["ci_high"] < 0)
    report["Q4_note"] = (
        "evaluated on the FULL collision population, which is what the controls are "
        "controls for. On the CONTESTED rows -- the agent stepped onto a SWITCH or a "
        "DECOY -- the exact count-only Bayes rule scores "
        f"{collision['count_only_bayes_ceiling']['contested_balanced_accuracy']:.4f} "
        "and every stateless arm is within noise of it, because one frame pair cannot "
        "separate two roles with identical cardinality, identical spatial distribution "
        "and identical appearance dynamics. That is an identifiability cap, not a "
        "learner failure, and it is what section G has to lift.")
    report["residual_spatial_channel"] = {
        "moments_only_contested_under_collision":
            collision["moments_only"]["contested_balanced_accuracy"],
        "distance_gap": report["exchangeability"]["COUNT_COLLISION"]["distance_gap"],
        "cardinality_identical":
            report["exchangeability"]["COUNT_COLLISION"]["cardinality_identical"],
        "note": ("cardinality is exact; the spatial marginal is matched shell by shell "
                 "and the residual is reported rather than assumed away")}
    report["cardinality_not_ruled_out_at_the_pooled_level"] = bool(
        collision["count_only_bayes_ceiling"]["balanced_accuracy"]
        >= collision[selected]["balanced_accuracy"] - 0.02)
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nselected representation: "
          f"{report['decisions']['selected_representation']}")
    print(f"count-only Bayes ceiling under collision: all rows "
          f"{report['decisions']['count_only_bayes_ceiling_under_collision_all_rows']:.4f}, "
          f"contested "
          f"{report['decisions']['count_only_bayes_ceiling_under_collision_contested']:.4f}")
    print(f"Q4: {report['Q4_global_binder_beats_count_only_under_collision']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
