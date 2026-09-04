"""J / K / L / Q11 / Q12. Is the goal decision identifiable at all, and can it be read?

Phase O1 called P12 "a readout capability failure, not evidence that language is
uninformative". The specification for this phase says not to accept that without
checking, and checking overturns half of it.

The audit in o2_equivalence.py shows goal-equivalence is identified in NONE of 47
episodes at any evidence stage, language included -- and the reason is structural: the
adapter terminates the instant the agent reaches its goal marker and every collector
appends the frame BEFORE stepping, so no recorded public history has ever depicted the
goal marker occupied. Nothing binds the word "alpha" to a colour, so the correct answer
on a contested key is not a function of the permitted evidence. The target was
unidentifiable and BOTH earlier results were measuring nothing.

Section K therefore authors a grounding protocol: a demonstration on a disjoint layout
under the same palette, ending WITH the terminal frame retained, paired with the
instruction that names the marker. Its public residue is two per-cell indicator
channels -- this cell's colour is the colour the alpha (or beta) demonstration ended on
-- which every arm receives equally and none is told how to use.

Ceilings are run before any learned arm is interpreted, and the readout family is
qualified on the semantic oracle against a threshold frozen in this file.

    .venv-shwm/bin/python experiments/shwm/o2_goal.py
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import o_core as O
import o2_core as C
import o2_memory as mem
import o2_models as M
import o2_readouts as R
from m2d_core import ARTIFACTS, write
from o2_core import GRID, N_ROLES

SEEDS = (46_000, 46_001, 46_002)
DEV_LAYOUTS = tuple(range(118_000, 118_040))
TEST_LAYOUTS = tuple(range(119_000, 119_032))
DEMO_LAYOUTS = tuple(range(115_000, 115_012))
DEV_PALETTES = (9_300, 9_301, 9_302, 9_303)
UNSEEN_PALETTES = (9_400, 9_401, 9_402, 9_403)
STEPS = 9
STRATUM = "COUNT_COLLISION"

# FROZEN BEFORE ANY ARM IS RUN. A readout that cannot reach this on the semantic-role
# oracle -- which is handed the marker cells outright -- is disqualified, and no
# statement about a learned representation may be made through it.
QUALIFICATION_THRESHOLD = 0.80

CEILINGS = {
    "1_semantic_oracle_correct_language": ("semantic_oracle", "correct", True),
    "2_semantic_oracle_shuffled_language": ("semantic_oracle", "shuffled", True),
    "3_exact_posterior_correct_language": ("exact_posterior", "correct", False),
    "4_exact_posterior_plus_goal_mapping": ("exact_posterior_grounded", "correct", True),
    "5_learned_binder_correct_language": ("learned_binder", "correct", False),
}

ARMS = {
    "1_vision_language_history": ("raw_cells", "correct", "true"),
    "2_shuffled_language": ("raw_cells", "shuffled", "true"),
    "3_masked_language": ("raw_cells", "masked", "true"),
    "4_wrong_lexical_convention": ("raw_cells", "wrong", "true"),
    "5_no_history": ("raw_cells", "correct", "none"),
    "6_shuffled_history": ("raw_cells", "correct", "shuffled"),
    "7_exact_palette_posterior": ("exact_posterior_grounded", "correct", "true"),
    "8_learned_palette_posterior": ("learned_binder", "correct", "true"),
    "9_semantic_oracle": ("semantic_oracle", "correct", "true"),
}


def build_population(layouts, bijection, palette: int, seed: int):
    """The same scene under two language goals. The scene is never regenerated: one
    action plan is executed once and both goals are scored against it."""
    from sentinel.env.adapters.procedural_visual_v2 import (
        ACTIONS, ProceduralVisualV2Adapter)
    from sentinel.wm.authority import AuthorityGate

    gate = AuthorityGate(gate_id="o2-goal")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    rows = []
    for layout in layouts:
        rng = np.random.default_rng(seed * 7919 + layout)
        adapter.reset(layout)
        level = adapter._require()
        start = tuple(int(v) for v in level.start)
        placement = np.random.default_rng(9_000_003 + layout)
        decoy = C.decoy_placement(level, start,
                                  C.decoy_count(STRATUM, placement), placement)
        markers = {m: tuple(int(v) for v in level.markers[m]) for m in ("alpha", "beta")}
        stripe = int(level.initial_polarity)
        frames, roles, positions, phases, actions = [], [], [], [], []
        for step in range(STEPS):
            truth = adapter.snapshot().reveal("evaluator")
            position = tuple(int(v) for v in truth["position"])
            grid = O.role_grid(level, position)
            grid[decoy & (grid == C.EMPTY)] = C.DECOY
            frames.append(O.render_roles(grid, bijection, stripe if step == 0 else None))
            roles.append(grid)
            positions.append(position)
            phases.append(int(truth["polarity"]))
            action = int(rng.integers(0, len(ACTIONS)))
            actions.append(action)
            adapter.step(action, gate.authorize_evaluator(action, "o2-goal"))
        positions.append(tuple(int(v) for v in
                               adapter.snapshot().reveal("evaluator")["position"]))
        episode = {"layout": layout, "palette": palette, "frames": np.stack(frames),
                   "roles": np.stack(roles), "actions": np.array(actions),
                   "positions": np.array(positions[:STEPS]),
                   "phases": np.array(phases), "markers": markers,
                   "bijection": bijection}
        for step in range(STEPS):
            before, after = positions[step], positions[step + 1]
            targets = {}
            for index, marker in enumerate(("alpha", "beta")):
                cell = markers[marker]
                targets[marker] = float(
                    abs(cell[0] - after[0]) + abs(cell[1] - after[1])
                    < abs(cell[0] - before[0]) + abs(cell[1] - before[1]))
            contested = targets["alpha"] != targets["beta"]
            for index, marker in enumerate(("alpha", "beta")):
                rows.append({"episode": episode, "step": step, "goal_index": index,
                             "target": targets[marker], "contested": contested,
                             "key": (palette, layout, step)})
    return rows


def posterior_grid(episode, grounded: bool, cache: dict) -> np.ndarray:
    """Per-cell posterior over roles from the exact survivor set."""
    key = (episode["layout"], episode["palette"], grounded)
    if key not in cache:
        fake = C.O2Episode(
            layout=episode["layout"], stratum=STRATUM, palette_id=episode["palette"],
            bijection=episode["bijection"], frames=episode["frames"],
            roles=episode["roles"], actions=episode["actions"],
            positions=episode["positions"],
            polarity=episode["phases"], event=np.zeros(STEPS, np.float32),
            entered_role=np.zeros(STEPS, np.int64), goal_marker="alpha",
            goal_cells=episode["markers"], stripe=0, decoy_cells=0)
        keep = C.survivors_over([fake])
        if grounded:
            keep = [pi for pi in keep
                    if pi[C.GOAL_ALPHA] == C.GOAL_ALPHA and pi[C.GOAL_BETA] == C.GOAL_BETA]
        table = np.zeros((N_ROLES, N_ROLES), np.float32)
        for pi in keep:
            for role in range(N_ROLES):
                table[role, pi[role]] += 1.0
        cache[key] = table / max(len(keep), 1)
    return cache[key]


_DEMO_CACHE: dict = {}


def demonstration_channels(episode, demonstrations, registry) -> np.ndarray:
    """Section K's public residue: is this cell's colour the colour each demonstration
    ended on? Two binary channels, computed from the demonstration frames and the
    instruction, and given identically to every arm."""
    key = (episode["layout"], episode["palette"])
    if key in _DEMO_CACHE:
        return _DEMO_CACHE[key]
    out = np.zeros((STEPS, GRID, GRID, 2), np.float32)
    cells = episode["frames"][:, ::2, ::2, :]
    for index, marker in enumerate(("alpha", "beta")):
        colour = demonstrations[episode["palette"]].get(marker)
        if colour is None:
            continue
        out[..., index] = np.all(cells == np.array(colour, np.uint8), axis=-1)
    _DEMO_CACHE[key] = out
    return out


def featurise(rows, representation: str, language: str, history: str,
              grounded: bool, binder_assignment, registry, demonstrations,
              posterior_cache, seed: int):
    grids, context, target = [], [], []
    rng = np.random.default_rng(seed)
    for row in rows:
        episode, step = row["episode"], row["step"]
        cells = episode["frames"][step, ::2, ::2, :]
        if representation == "raw_cells":
            grid = cells.astype(np.float32) / 255.0
        elif representation == "semantic_oracle":
            grid = np.zeros((GRID, GRID, N_ROLES), np.float32)
            r, c = np.indices((GRID, GRID))
            grid[r, c, episode["roles"][step]] = 1.0
        elif representation in ("exact_posterior", "exact_posterior_grounded"):
            table = posterior_grid(episode, representation.endswith("grounded"),
                                   posterior_cache)
            grid = table[episode["roles"][step]]
        elif representation == "learned_binder":
            index = C.cell_index(cells, registry)
            grid = binder_assignment[episode["palette"]][index]
        else:
            raise KeyError(representation)
        if grounded:
            grid = np.concatenate(
                [grid, demonstration_channels(episode, demonstrations,
                                              registry)[step]], axis=-1)
        grids.append(grid)
        one_hot = np.zeros(2, np.float32)
        one_hot[row["goal_index"]] = 1.0
        phase = np.array([float(episode["phases"][step])], np.float32)
        action = np.zeros(4, np.float32)
        action[int(episode["actions"][step])] = 1.0
        context.append(np.concatenate([one_hot, phase, action]))
        target.append(row["target"])
    grids = np.stack(grids).astype(np.float32)
    context = np.stack(context).astype(np.float32)
    if language == "shuffled":
        context[:, :2] = context[rng.permutation(len(context)), :2]
    elif language == "masked":
        context[:, :2] = 0.0
    elif language == "wrong":
        context[:, :2] = context[:, [1, 0]]
    if history == "none":
        context[:, 2] = 0.0
    elif history == "shuffled":
        context[:, 2] = context[rng.permutation(len(context)), 2]
    return grids, context, np.array(target, np.float32)


def paired_interval(a: np.ndarray, b: np.ndarray, keys, seeds, resamples: int = 4000,
                    seed: int = 99) -> dict[str, float]:
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


def balanced(prediction: np.ndarray, truth: np.ndarray) -> float:
    out = []
    for value in (0.0, 1.0):
        mask = truth == value
        if mask.any():
            out.append(float((prediction[mask] == value).mean()))
    return float(np.mean(out)) if out else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--dev-layouts", type=int, default=len(DEV_LAYOUTS))
    parser.add_argument("--test-layouts", type=int, default=len(TEST_LAYOUTS))
    parser.add_argument("--palettes", type=int, default=len(DEV_PALETTES))
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o2-goal.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    dev_palettes = DEV_PALETTES[:arguments.palettes]
    unseen_palettes = UNSEEN_PALETTES[:arguments.palettes]
    dev_layouts = DEV_LAYOUTS[:arguments.dev_layouts]
    test_layouts = TEST_LAYOUTS[:arguments.test_layouts]
    bijections = {p: C.sample_bijection(p) for p in dev_palettes + unseen_palettes}
    train_rows, test_rows = [], []
    for palette in dev_palettes:
        train_rows.extend(build_population(dev_layouts, bijections[palette], palette, 11))
    for palette in unseen_palettes:
        test_rows.extend(build_population(test_layouts, bijections[palette], palette, 313))
    contested_test = [r for r in test_rows if r["contested"]]
    keys = {r["key"] for r in contested_test}
    print(f"{len(train_rows)} train rows, {len(test_rows)} test rows, "
          f"{len(contested_test)} contested rows over {len(keys)} contested keys",
          flush=True)

    # ---- section K: the grounding protocol -----------------------------------------
    demonstrations: dict[int, dict[str, tuple]] = {}
    demonstration_manifest = {}
    for palette, bijection in bijections.items():
        table, used = {}, {}
        for marker in ("alpha", "beta"):
            for layout in DEMO_LAYOUTS:
                built = C.goal_demonstration(layout, bijection, marker, STRATUM)
                if built:
                    table[marker] = built["named_marker_colour"]
                    used[marker] = {"layout": layout, "steps": built["steps"]}
                    break
        demonstrations[palette] = table
        demonstration_manifest[str(palette)] = used
    disjoint = not (set(DEMO_LAYOUTS) & (set(DEV_LAYOUTS) | set(TEST_LAYOUTS)))

    registry = C.canonical_registry()

    # ---- the learned binder's assignment, one per palette ---------------------------
    print("training the palette binder for the learned representation", flush=True)
    groups = [mem.build_group(p, 71) for p in dev_palettes]
    unseen_groups = [mem.build_group(p, 313) for p in unseen_palettes]
    binder_registry = registry
    binder_train = mem.stack_groups(groups, binder_registry)
    memory_infer, memory_model = M.train_memory(
        (binder_train["sequence"], binder_train["mask"], binder_train["before"],
         binder_train["after"], binder_train["event"]), SEEDS[0],
        updates=mem.MEMORY_UPDATES)
    binder_assignment = {}
    for group in groups + unseen_groups:
        history, mask = mem.history_of(group, binder_registry)
        sequence = np.concatenate([history, np.zeros((1,) + history.shape[1:],
                                                     np.float32)])[None]
        full_mask = np.concatenate([mask, np.zeros(1, np.float32)])[None]
        # One canonical registry everywhere, so no remapping step exists to be wrong.
        binder_assignment[group.palette] = M.memory_assignment_of(
            memory_model, sequence, full_mask)[0]

    posterior_cache: dict = {}
    report: dict[str, Any] = {
        "contested_keys": len(keys), "contested_rows": len(contested_test),
        "train_rows": len(train_rows), "test_rows": len(test_rows),
        "qualification_threshold": QUALIFICATION_THRESHOLD,
        "development_layouts": list(dev_layouts), "test_layouts": list(test_layouts),
        "development_palettes": list(dev_palettes),
        "unseen_palettes": list(unseen_palettes), "seeds": list(SEEDS[:arguments.seeds]),
        "stratum": STRATUM,
        "goal_grounding_calibration": {
            "demonstration_layouts": list(DEMO_LAYOUTS),
            "disjoint_from_evaluation": bool(disjoint),
            "per_palette": demonstration_manifest,
            "public_residue": ("two per-cell indicator channels: is this cell's colour "
                               "the colour the alpha (or beta) demonstration ended on"),
            "palette_map_exposed": False,
            "minimum_separating_calibration_size": 1},
        "readout_family": list(R.FAMILY),
    }

    # ---- L: qualify the readout family on the semantic oracle -----------------------
    print(f"\nqualifying the readout family on the semantic oracle "
          f"(threshold {QUALIFICATION_THRESHOLD}, frozen)", flush=True)
    dev_contested = [r for r in train_rows if r["contested"]]
    holdout = dev_contested[: len(dev_contested) // 3]
    fit = dev_contested[len(dev_contested) // 3:]
    qualification = {}
    for name in R.FAMILY:
        scores = []
        for seed in SEEDS[:2]:
            g, c, y = featurise(fit, "semantic_oracle", "correct", "true", True,
                                binder_assignment, registry, demonstrations,
                                posterior_cache, seed)
            infer = R.train(name, g, c, y, seed)
            gh, ch, yh = featurise(holdout, "semantic_oracle", "correct", "true", True,
                                   binder_assignment, registry, demonstrations,
                                   posterior_cache, seed)
            scores.append(balanced((infer(gh, ch) > 0).astype(float), yh))
        qualification[name] = {"contested_balanced_accuracy": float(np.mean(scores)),
                               "qualified": bool(np.mean(scores)
                                                 >= QUALIFICATION_THRESHOLD)}
        print(f"  {name:34s} {np.mean(scores):.4f}  "
              f"{'QUALIFIED' if qualification[name]['qualified'] else 'disqualified'}",
              flush=True)
    report["qualification"] = qualification
    qualified = [n for n in R.FAMILY if qualification[n]["qualified"]]
    report["qualified_readouts"] = qualified
    if not qualified:
        report["stop"] = ("no readout in the frozen family solves the task from the "
                          "semantic-role oracle; the target or the evaluator is invalid "
                          "and no learned arm may be interpreted")
        report["Q11_target_proven_identifiable"] = False
        report["wall_clock_seconds"] = time.perf_counter() - started
        write(arguments.out, report)
        print(f"\nSTOP: {report['stop']}")
        return 0
    chosen = max(qualified,
                 key=lambda n: qualification[n]["contested_balanced_accuracy"])
    report["selected_readout"] = chosen
    print(f"  selected readout: {chosen}", flush=True)

    # ---- J: ceilings ----------------------------------------------------------------
    print("\nJ. identifiability ceilings on contested test keys", flush=True)
    ceilings = {}
    for name, (representation, language, grounded) in CEILINGS.items():
        scores = []
        for seed in SEEDS[:arguments.seeds]:
            g, c, y = featurise(train_rows, representation, language, "true", grounded,
                                binder_assignment, registry, demonstrations,
                                posterior_cache, seed)
            infer = R.train(chosen, g, c, y, seed)
            gh, ch, yh = featurise(contested_test, representation, language, "true",
                                   grounded, binder_assignment, registry,
                                   demonstrations, posterior_cache, seed)
            scores.append(balanced((infer(gh, ch) > 0).astype(float), yh))
        ceilings[name] = {"contested_balanced_accuracy": float(np.mean(scores)),
                          "per_seed": [float(s) for s in scores],
                          "representation": representation, "language": language,
                          "goal_grounding": grounded}
        print(f"  {name:42s} {np.mean(scores):.4f}", flush=True)
    report["ceilings"] = ceilings

    oracle = ceilings["1_semantic_oracle_correct_language"]["contested_balanced_accuracy"]
    ungrounded = ceilings["3_exact_posterior_correct_language"][
        "contested_balanced_accuracy"]
    grounded_posterior = ceilings["4_exact_posterior_plus_goal_mapping"][
        "contested_balanced_accuracy"]
    report["J_diagnosis"] = {
        "semantic_oracle_solves_it": bool(oracle >= QUALIFICATION_THRESHOLD),
        "exact_posterior_without_grounding_is_at_chance": bool(ungrounded < 0.60),
        "goal_grounding_recovers_it": bool(grounded_posterior >= 0.75),
        "conclusion": (
            "the evaluator is valid and the readout can express the task; without the "
            "section K calibration the exact palette posterior is at "
            f"{ungrounded:.4f} because no permitted evidence binds a marker name to a "
            f"colour, and with it the same posterior reaches {grounded_posterior:.4f}. "
            "Goal-role appearance grounding was the missing piece, not readout capacity."
            if oracle >= QUALIFICATION_THRESHOLD and ungrounded < 0.60
            else "see the ceiling table"),
    }
    report["Q11_target_proven_identifiable"] = bool(
        oracle >= QUALIFICATION_THRESHOLD and grounded_posterior >= 0.75)

    # ---- L: the nine arms -----------------------------------------------------------
    print("\nL. arms on contested test keys, with the qualified readout", flush=True)
    scores: dict[str, np.ndarray] = {}
    key_list, seed_list = [], []
    for arm, (representation, language, history) in ARMS.items():
        per_seed = []
        for seed in SEEDS[:arguments.seeds]:
            g, c, y = featurise(train_rows, representation, language, history, True,
                                binder_assignment, registry, demonstrations,
                                posterior_cache, seed)
            infer = R.train(chosen, g, c, y, seed)
            gh, ch, yh = featurise(contested_test, representation, language, history,
                                   True, binder_assignment, registry, demonstrations,
                                   posterior_cache, seed)
            per_seed.append(((infer(gh, ch) > 0).astype(float) == yh).astype(float))
        scores[arm] = np.concatenate(per_seed)
        print(f"  {arm:34s} {scores[arm].mean():.4f}", flush=True)
    for seed in SEEDS[:arguments.seeds]:
        key_list.extend([r["key"] for r in contested_test])
        seed_list.extend([seed] * len(contested_test))

    report["arms"] = {a: {"contested_accuracy": float(scores[a].mean())} for a in ARMS}
    print("\npaired intervals by contested key, against arm 1")
    base = scores["1_vision_language_history"]
    for other in ARMS:
        if other == "1_vision_language_history":
            continue
        interval = paired_interval(base, scores[other], key_list, seed_list)
        report["arms"][other]["vs_arm_1"] = interval
        print(f"  arm 1 minus {other:34s} {interval['delta']:+.4f} "
              f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
              f"{' *' if interval['excludes_zero'] else ''}", flush=True)

    language_arms = [report["arms"][k]["vs_arm_1"] for k in
                     ("2_shuffled_language", "3_masked_language")]
    report["Q12_language_beats_shuffled_and_masked"] = bool(
        all(i["ci_low"] > 0 for i in language_arms))
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nQ11 {report['Q11_target_proven_identifiable']}   "
          f"Q12 {report['Q12_language_beats_shuffled_and_masked']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
