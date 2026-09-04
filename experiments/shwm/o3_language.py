"""K. Controlled language replication on fresh palettes, layouts and contested keys.

O2's Q12 recorded correct-minus-shuffled at +0.2209 [+0.2030, +0.2389] and
correct-minus-masked at +0.2161 [+0.2001, +0.2321] over 344 contested keys. Two things
about that need testing rather than repeating.

  the populations were not fresh
      four development palettes, four unseen palettes, and layout ranges that O3 has
      since reused for its own calibration pool. Everything here is drawn from id spaces
      that no earlier phase has touched.

  the interval was computed over KEYS
      a contested key is (palette, layout, step). Keys inside one palette share a
      convention and a grounding demonstration, so resampling keys asks "would another
      key from these palettes agree", not "would another palette agree". Section C
      established that the palette is the unit that has to be resampled, and both
      intervals are reported here so the difference is visible rather than argued.

This runs O2's own goal pipeline -- the same build, the same readout family, the same
qualification threshold -- with the constants repointed. It is a replication, not a
reimplementation: if the arms were to be rewritten here, a disagreement would not
distinguish "the claim does not replicate" from "the rewrite differs".

    .venv-shwm/bin/python experiments/shwm/o3_language.py
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import o2_goal as G
from m2d_core import ARTIFACTS, write

# Fresh id spaces. Checked against every pool any earlier phase used, in `provenance()`.
FRESH_DEV_LAYOUTS = tuple(range(120_000, 120_040))
FRESH_TEST_LAYOUTS = tuple(range(121_000, 121_032))
FRESH_DEMO_LAYOUTS = tuple(range(116_500, 116_512))
FRESH_DEV_PALETTES = tuple(range(23_000, 23_012))
FRESH_UNSEEN_PALETTES = tuple(range(23_100, 23_112))
FRESH_SEEDS = (47_000, 47_001, 47_002)

# Pools spent by earlier phases. A fresh range must intersect none of them.
SPENT_LAYOUTS = {
    "o2_goal_dev": range(118_000, 118_040),
    "o2_goal_test": range(119_000, 119_032),
    "o2_goal_demo": range(115_000, 115_012),
    "o2_memory_calibration": range(116_000, 116_006),
    "o2_memory_transfer": range(117_000, 117_048),
    "o3_population_calibration": range(118_000, 118_200),
    "o3_population_transfer": range(119_000, 119_400),
}
SPENT_PALETTES = {
    "o2_goal": tuple(range(9_300, 9_304)) + tuple(range(9_400, 9_404)),
    "o2_memory_thresholds": tuple(range(9_700, 9_708)),
    "o3_development": tuple(range(20_000, 20_064)),
    "o3_validation": tuple(range(21_000, 21_064)),
    "o3_replication_reserved": tuple(range(22_000, 22_064)),
}


def provenance() -> dict[str, Any]:
    """Every fresh pool against every spent pool. Recorded, not asserted in prose."""
    fresh_layouts = (set(FRESH_DEV_LAYOUTS) | set(FRESH_TEST_LAYOUTS)
                     | set(FRESH_DEMO_LAYOUTS))
    fresh_palettes = set(FRESH_DEV_PALETTES) | set(FRESH_UNSEEN_PALETTES)
    layout_overlaps = {name: sorted(fresh_layouts & set(pool))
                       for name, pool in SPENT_LAYOUTS.items()}
    palette_overlaps = {name: sorted(fresh_palettes & set(pool))
                        for name, pool in SPENT_PALETTES.items()}
    return {
        "fresh_development_layouts": [FRESH_DEV_LAYOUTS[0], FRESH_DEV_LAYOUTS[-1]],
        "fresh_test_layouts": [FRESH_TEST_LAYOUTS[0], FRESH_TEST_LAYOUTS[-1]],
        "fresh_demonstration_layouts": [FRESH_DEMO_LAYOUTS[0], FRESH_DEMO_LAYOUTS[-1]],
        "fresh_development_palettes": list(FRESH_DEV_PALETTES),
        "fresh_unseen_palettes": list(FRESH_UNSEEN_PALETTES),
        "layout_overlaps_with_spent_pools": layout_overlaps,
        "palette_overlaps_with_spent_pools": palette_overlaps,
        "all_fresh": bool(not any(layout_overlaps.values())
                          and not any(palette_overlaps.values())),
        "demonstration_layouts_disjoint_from_evaluation": bool(
            not (set(FRESH_DEMO_LAYOUTS)
                 & (set(FRESH_DEV_LAYOUTS) | set(FRESH_TEST_LAYOUTS)))),
    }


def palette_paired_interval(a: np.ndarray, b: np.ndarray, keys, seeds,
                            resamples: int = 4_000, seed: int = 99) -> dict[str, Any]:
    """O2's key-level interval, plus the palette-level one the claim actually needs.

    The key-level number is computed by O2's own function so the two are guaranteed
    comparable. The palette-level number resamples palettes with replacement and pools
    the drawn palettes' rows, which is the interval that answers "would another palette
    agree".
    """
    out = dict(G._key_level_interval(a, b, keys, seeds, resamples, seed))
    out["unit"] = "contested key (palette, layout, step) -- O2's unit"

    by_palette: dict[int, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        by_palette[int(key[0])].append(i)
    palettes = sorted(by_palette)
    difference = a - b
    per_palette = np.array([difference[by_palette[p]].mean() for p in palettes])
    rng = np.random.default_rng(seed + 1)
    draws = np.array([per_palette[rng.integers(0, len(per_palette),
                                               len(per_palette))].mean()
                      for _ in range(resamples)])
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    out["palette_level"] = {
        "unit": "palette",
        "delta": float(per_palette.mean()),
        "ci_low": low, "ci_high": high,
        "excludes_zero": bool(low > 0 or high < 0),
        "palettes": int(len(palettes)),
        "palettes_where_correct_wins": int((per_palette > 0).sum()),
        "per_palette": {str(p): float(difference[by_palette[p]].mean())
                        for p in palettes},
        "width_ratio_to_key_level": (
            (high - low) / (out["ci_high"] - out["ci_low"])
            if out["ci_high"] > out["ci_low"] else None),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--palettes", type=int, default=len(FRESH_DEV_PALETTES))
    parser.add_argument("--dev-layouts", type=int, default=len(FRESH_DEV_LAYOUTS))
    parser.add_argument("--test-layouts", type=int, default=len(FRESH_TEST_LAYOUTS))
    parser.add_argument("--seeds", type=int, default=len(FRESH_SEEDS))
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o3-language.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    trace = provenance()
    print(f"fresh pools: {trace['all_fresh']}  "
          f"demo disjoint: {trace['demonstration_layouts_disjoint_from_evaluation']}",
          flush=True)
    if not trace["all_fresh"]:
        for name, hits in {**trace["layout_overlaps_with_spent_pools"],
                           **trace["palette_overlaps_with_spent_pools"]}.items():
            if hits:
                print(f"  OVERLAP with {name}: {hits[:8]}", flush=True)
        raise SystemExit("fresh pools are not fresh; refusing to replicate on spent ids")

    # Repoint O2's own module and run O2's own pipeline. Nothing about the arms, the
    # readout family or the qualification threshold is touched.
    G.DEV_LAYOUTS = FRESH_DEV_LAYOUTS
    G.TEST_LAYOUTS = FRESH_TEST_LAYOUTS
    G.DEMO_LAYOUTS = FRESH_DEMO_LAYOUTS
    G.DEV_PALETTES = FRESH_DEV_PALETTES
    G.UNSEEN_PALETTES = FRESH_UNSEEN_PALETTES
    G.SEEDS = FRESH_SEEDS
    G._key_level_interval = G.paired_interval
    G.paired_interval = palette_paired_interval

    import sys
    argv = sys.argv
    sys.argv = ["o2_goal",
                "--palettes", str(arguments.palettes),
                "--dev-layouts", str(arguments.dev_layouts),
                "--test-layouts", str(arguments.test_layouts),
                "--seeds", str(arguments.seeds),
                "--out", str(arguments.out)]
    try:
        status = G.main()
    finally:
        sys.argv = argv
    if status != 0:
        return status

    import json
    report = json.loads(Path(arguments.out).read_text())
    report["replication"] = {
        "of": "O2 Q12, correct language against shuffled and masked",
        "o2_headline": {
            "correct_minus_shuffled": {"delta": 0.2209, "ci": [0.2030, 0.2389]},
            "correct_minus_masked": {"delta": 0.2161, "ci": [0.2001, 0.2321]},
            "unit": "contested key", "contested_keys": 344},
        "provenance": trace,
        "seeds": list(FRESH_SEEDS),
    }
    arms = report.get("arms", {})
    verdicts = {}
    for control in ("2_shuffled_language", "3_masked_language"):
        block = arms.get(control, {}).get("vs_arm_1")
        if not block:
            verdicts[control] = {"status": "NOT_RUN", "reason_class": "arm_absent"}
            continue
        palette = block.get("palette_level", {})
        verdicts[control] = {
            "key_level_delta": block["delta"],
            "key_level_ci": [block["ci_low"], block["ci_high"]],
            "key_level_excludes_zero": block["excludes_zero"],
            "palette_level_delta": palette.get("delta"),
            "palette_level_ci": [palette.get("ci_low"), palette.get("ci_high")],
            "palette_level_excludes_zero": palette.get("excludes_zero"),
            "palettes_where_correct_wins": palette.get("palettes_where_correct_wins"),
            "palettes": palette.get("palettes"),
            "interval_width_ratio": palette.get("width_ratio_to_key_level"),
        }
        print(f"\n1 - {control}: key {block['delta']:+.4f} "
              f"[{block['ci_low']:+.4f}, {block['ci_high']:+.4f}]   "
              f"palette {palette.get('delta', float('nan')):+.4f} "
              f"[{palette.get('ci_low', float('nan')):+.4f}, "
              f"{palette.get('ci_high', float('nan')):+.4f}]  "
              f"wins on {palette.get('palettes_where_correct_wins')}/"
              f"{palette.get('palettes')}", flush=True)
    report["replication"]["verdicts"] = verdicts
    usable = [v for v in verdicts.values() if "palette_level_excludes_zero" in v]
    report["R13_language_replicates_on_fresh_palettes"] = bool(
        len(usable) == 2 and all(v["palette_level_excludes_zero"] for v in usable))
    report["wall_clock_seconds_o3"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nR13 {report['R13_language_replicates_on_fresh_palettes']}")
    print(f"wrote {arguments.out}  "
          f"({report['wall_clock_seconds_o3'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
