"""I / O13 / P12. The balanced paired multimodal population, with intervals.

Phase N passed its multimodal gate on a +0.021 point estimate with no interval and 65
contested keys. That is not evidence, and the specification says so. Here the scene is
held byte-identical between the two goals -- same layout, same palette, same trajectory,
same history -- and only the language instruction differs, so a contested key is a
literal pair of rows that share every input except the goal and disagree on the answer.

Intervals are paired by contested key: the two members of a pair are resampled together,
which is the only resampling that respects the construction.

    .venv-shwm/bin/python experiments/shwm/p_multimodal.py
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import n_interfaces as ifaces
import n_heads as heads
from m2d_core import ARTIFACTS, write
from n_core import GRID

SEEDS = (38_000, 38_001, 38_002)
GOAL_DRAWS = (0, 2)          # 0 and 1 both hash to alpha; 0 and 2 differ
ARMS = ("1_vision_language_history", "2_shuffled_language", "3_masked_language",
        "4_wrong_goal_convention", "5_no_history", "6_shuffled_history",
        "9_semantic_oracle")


def build_population(layouts, steps: int, seed: int):
    """The SAME scene under two goals. The scene is never regenerated between them."""
    from sentinel.env.adapters.procedural_visual_v2 import (
        ACTIONS, MARKERS, ProceduralVisualV2Adapter)
    from sentinel.wm.authority import AuthorityGate

    gate = AuthorityGate(gate_id="p-mm")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    rows = []
    for layout in layouts:
        rng = np.random.default_rng(seed * 7919 + layout)
        plan = [int(rng.integers(0, len(ACTIONS))) for _ in range(steps)]
        per_goal = {}
        for draw in GOAL_DRAWS:
            adapter.reset(layout, f"goal:{draw}")
            level = adapter._require()
            marker = adapter._goal_marker
            goal_cell = tuple(int(v) for v in level.markers[marker])
            trace = []
            for step in range(steps):
                info = adapter.snapshot().reveal("evaluator")
                position = tuple(int(v) for v in info["position"])
                frame = adapter.frame().copy()
                phase = int(info["polarity"])
                action = plan[step]
                adapter.step(action, gate.authorize_evaluator(action, "p"))
                landed = tuple(int(v) for v in
                               adapter.snapshot().reveal("evaluator")["position"])
                before = abs(goal_cell[0] - position[0]) + abs(goal_cell[1] - position[1])
                after = abs(goal_cell[0] - landed[0]) + abs(goal_cell[1] - landed[1])
                trace.append({"frame": frame, "action": action, "phase": phase,
                              "goal_index": MARKERS.index(marker), "layout": layout,
                              "step": step, "target": float(after < before),
                              "markers": level.markers})
            per_goal[draw] = trace
        # Pair step-by-step. The plan is shared, so the trajectories coincide exactly.
        for step in range(min(len(per_goal[GOAL_DRAWS[0]]), len(per_goal[GOAL_DRAWS[1]]))):
            a, b = per_goal[GOAL_DRAWS[0]][step], per_goal[GOAL_DRAWS[1]][step]
            if not np.array_equal(a["frame"], b["frame"]):
                continue                                  # scenes must be identical
            key = (layout, step)
            a["key"], b["key"] = key, key
            a["contested"] = b["contested"] = bool(a["target"] != b["target"])
            rows.extend([a, b])
    return rows


def featurise(rows, arm: str, seed: int, roles=None):
    rng = np.random.default_rng(seed)
    frames = np.stack([r["frame"] for r in rows]).astype(np.float32) / 255.0
    cells = frames[:, ::2, ::2, :]
    action = np.zeros((len(rows), 4), np.float32)
    for i, r in enumerate(rows):
        action[i, r["action"]] = 1.0
    goal = np.array([[1.0 - r["goal_index"], float(r["goal_index"])] for r in rows],
                    np.float32)
    history = np.array([[float(r["phase"])] for r in rows], np.float32)
    if arm == "2_shuffled_language":
        goal = goal[rng.permutation(len(goal))]
    elif arm == "3_masked_language":
        goal = np.zeros_like(goal)
    elif arm == "4_wrong_goal_convention":
        goal = goal[:, ::-1].copy()          # the naming convention is inverted
    elif arm == "5_no_history":
        history = np.zeros_like(history)
    elif arm == "6_shuffled_history":
        history = history[rng.permutation(len(history))]
    projection = ifaces.frozen_projection(cells.shape[-1], ifaces.SLOT_WIDTH, 20_010)
    return cells @ projection, action, np.concatenate([goal, history], axis=1)


def train_and_score(train_rows, test_rows, arm: str, seed: int) -> np.ndarray:
    import n_multimodal as nmm

    xt, at, et = featurise(train_rows, arm, seed)
    xh, ah, eh = featurise(test_rows, arm, seed)
    y = np.array([r["target"] for r in train_rows], np.float32)
    infer = nmm.train_coordinate(xt, at, et, y, seed, GRID)
    logits = infer(xh, ah, eh)
    truth = np.array([r["target"] for r in test_rows], np.float32)
    return ((logits[:, 0] > 0).astype(float) == truth).astype(float)


def paired_interval(a: np.ndarray, b: np.ndarray, keys, seeds,
                    resamples: int = 4000, seed: int = 99) -> dict[str, float]:
    """Resample contested KEYS, carrying both members of each pair together."""
    groups = defaultdict(list)
    for i, (k, s) in enumerate(zip(keys, seeds)):
        groups[(k, s)].append(i)
    index = list(groups.values())
    difference = a - b
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples)
    for r in range(resamples):
        picked = np.concatenate([index[i] for i in
                                 rng.integers(0, len(index), len(index))])
        draws[r] = difference[picked].mean()
    low, high = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return {"delta": float(difference.mean()), "ci_low": low, "ci_high": high,
            "excludes_zero": bool(low > 0 or high < 0), "keys": len(index)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--train-layouts", type=int, default=90)
    parser.add_argument("--test-layouts", type=int, default=60)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "p-multimodal.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    train_rows = build_population(range(113_000, 113_000 + arguments.train_layouts), 9, 11)
    test_rows = build_population(range(114_000, 114_000 + arguments.test_layouts), 9, 313)
    contested = [r for r in test_rows if r["contested"]]
    keys = {r["key"] for r in contested}
    print(f"{len(train_rows)} train rows / {len(test_rows)} test rows; "
          f"{len(contested)} contested rows over {len(keys)} contested keys\n", flush=True)

    report: dict[str, Any] = {
        "train_rows": len(train_rows), "test_rows": len(test_rows),
        "contested_rows": len(contested), "contested_keys": len(keys),
        "target": "the action moved the agent toward the language-named goal",
        "construction": ("the scene is held byte-identical between the two goals; the "
                         "action plan is shared, so the paired rows differ only in the "
                         "language instruction"),
        "arms": {}}

    scores: dict[str, np.ndarray] = {}
    key_list, seed_list = [], []
    print(f"{'arm':30s} {'contested accuracy':>19s}")
    print("-" * 52)
    for arm in ARMS:
        if arm == "9_semantic_oracle":
            continue
        per_seed = []
        for seed in SEEDS[:arguments.seeds]:
            per_seed.append(train_and_score(train_rows, contested, arm, seed))
        scores[arm] = np.concatenate(per_seed)
        report["arms"][arm] = {"contested_accuracy": float(scores[arm].mean())}
        print(f"{arm:30s} {scores[arm].mean():19.4f}", flush=True)
    for seed in SEEDS[:arguments.seeds]:
        key_list.extend([r["key"] for r in contested])
        seed_list.extend([seed] * len(contested))

    print("\npaired intervals by contested key")
    base = scores["1_vision_language_history"]
    for other in ("2_shuffled_language", "3_masked_language", "4_wrong_goal_convention",
                  "5_no_history", "6_shuffled_history"):
        interval = paired_interval(base, scores[other], key_list, seed_list)
        report["arms"][other]["vs_correct"] = interval
        print(f"  correct minus {other:26s} {interval['delta']:+.4f} "
              f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
              f"{' *' if interval['excludes_zero'] else ''}", flush=True)

    language = [report["arms"][k]["vs_correct"] for k in
                ("2_shuffled_language", "3_masked_language")]
    history = [report["arms"][k]["vs_correct"] for k in
               ("5_no_history", "6_shuffled_history")]
    report["p12_language_interval_excludes_zero"] = bool(
        all(i["ci_low"] > 0 for i in language))
    report["history_interval_excludes_zero"] = bool(all(i["ci_low"] > 0 for i in history))
    report["o13_status"] = ("PASS" if report["p12_language_interval_excludes_zero"]
                            else "FAIL")
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nO13 / P12 (language contributes with an interval excluding zero): "
          f"{report['o13_status']}")
    print(f"history contributes with an interval excluding zero: "
          f"{report['history_interval_excludes_zero']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
