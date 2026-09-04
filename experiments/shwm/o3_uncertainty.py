"""G / H. Query-scoped uncertainty, and an appearance process that is genuinely unreadable.

Two things are wrong with how O2 handled uncertainty and both are fixed here.

First, O2 used ONE global per-row confidence margin for every question. The evidence does
not work that way: on all 64 validation palettes the exact posterior identifies the EVENT
class (0.000 bits) while leaving the GOAL class at exactly 1.000 bit, so a single margin
must be either over-confident about goals or under-confident about events. Uncertainty is
maintained per query type here -- EVENT, GOAL, FULL -- with thresholds frozen on
development.

Second, O2's PER_FRAME_PERMUTATION was not an impossibility control. Its own audit found
0 of 5904 legal permutation pairs with distinct per-frame mappings, because static-scene
legality pins the relative relabelling between two frames: an observer can undo a
per-frame recolouring by requiring the scene to hold still. So a per-frame BIJECTION is
run here and reported for what it is -- still event-identifiable -- and a genuine negative
control is added alongside it, in which appearance is not a function of role at all and
nothing is identifiable by construction.

    .venv-shwm/bin/python experiments/shwm/o3_uncertainty.py
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np

import o_identifiability as ident
import o2_core as C
import o2_memory as mem
import o2_models as M
import o3_core as O3
import o3_population as pop
from m2d_core import ARTIFACTS, write

SEED = 54_000
VIEW = "no_rgb"
REGIMES = ("PERSISTENT_CONVENTION", "PER_FRAME_BIJECTION", "PER_CELL_NOISE")
QUERIES = ("EVENT", "GOAL", "FULL")
OUTPUTS = ("EVENT_IDENTIFIED", "EVENT_UNRESOLVED", "GOAL_IDENTIFIED", "GOAL_UNRESOLVED",
           "FULL_MAPPING_UNRESOLVED", "PALETTE_CHANGE_SUSPECTED")


def render_regime(episode: C.O2Episode, bijection: np.ndarray, regime: str,
                  seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Cells for one episode under an appearance regime, plus the per-frame maps used."""
    rng = np.random.default_rng(seed)
    cells, maps = [], []
    for t in range(episode.length):
        if regime == "PERSISTENT_CONVENTION":
            current = bijection
        elif regime == "PER_FRAME_BIJECTION":
            current = C.sample_bijection(int(rng.integers(0, 2 ** 31)))
        else:
            current = None
        if current is None:
            # THE GENUINE NEGATIVE CONTROL: colour is not a function of role. Every cell
            # is painted independently, so no colour-to-role map exists to be inferred
            # and nothing about the scene is recoverable from appearance.
            grid = rng.integers(0, len(C.COLOUR_POOL),
                                size=(C.GRID, C.GRID))
            frame = C.COLOUR_POOL[grid]
            if t == 0:
                frame = frame.copy()
                frame[0, :] = 255 if episode.stripe else 0
            cells.append(frame.astype(np.uint8))
            maps.append(None)
            continue
        cells.append(C.cells_from_roles(episode.roles[t], current,
                                        episode.stripe if t == 0 else None))
        maps.append(current)
    return np.stack(cells), maps


def maps_genuinely_differ(maps) -> dict[str, Any]:
    """Verify the per-frame process actually uses distinct mappings."""
    real = [m for m in maps if m is not None]
    if not real:
        return {"frames": len(maps), "distinct_maps": 0,
                "all_frames_share_one_map": False,
                "appearance_is_a_function_of_role": False}
    distinct = {tuple(int(v) for v in m) for m in real}
    return {"frames": len(real), "distinct_maps": len(distinct),
            "all_frames_share_one_map": len(distinct) == 1,
            "appearance_is_a_function_of_role": True}


def exact_queries(episode: C.O2Episode, regime: str, tied: bool) -> dict[str, float]:
    """Exact identifiability of each query from a frame pair, over permutation pairs.

    `tied` restricts the pair space to its diagonal, which is what a persistent
    convention means. Under PER_CELL_NOISE no map exists at all, so every query is
    unidentifiable by construction and the enumeration is skipped rather than faked.
    """
    if regime == "PER_CELL_NOISE":
        return {"event_identifiable": 0.0, "goal_identifiable": 0.0,
                "full_identifiable": 0.0, "mean_class_size": float("inf"),
                "by_construction": True}
    event, goal, full, sizes = [], [], [], []
    for t in range(1, episode.length):
        before = [pi for pi in itertools.permutations(range(C.N_ROLES))
                  if C.cardinality_legal7(np.array(pi, np.int64)[episode.roles[t - 1]])]
        after = [pi for pi in itertools.permutations(range(C.N_ROLES))
                 if C.cardinality_legal7(np.array(pi, np.int64)[episode.roles[t]])]
        legal = []
        for sigma in before:
            grid_before = np.array(sigma, np.int64)[episode.roles[t - 1]]
            for tau in ([sigma] if tied else after):
                if tied and sigma not in after:
                    continue
                grid_after = np.array(tau, np.int64)[episode.roles[t]]
                if (ident.static_scene_legal(grid_before, grid_after)
                        and ident.motion_legal(grid_before, grid_after,
                                               int(episode.actions[t - 1]))):
                    legal.append((sigma, tau))
        if not legal:
            continue
        entered = int(episode.entered_role[t])
        calls = ({sigma[entered] == C.SWITCH for sigma, _ in legal} if entered >= 0
                 else {False})
        event.append(float(len(calls) == 1))
        goal.append(float(len({(sigma[C.GOAL_ALPHA], sigma[C.GOAL_BETA])
                               for sigma, _ in legal}) == 1))
        full.append(float(len({sigma for sigma, _ in legal}) == 1))
        sizes.append(len(legal))
    return {"event_identifiable": float(np.mean(event)) if event else float("nan"),
            "goal_identifiable": float(np.mean(goal)) if goal else float("nan"),
            "full_identifiable": float(np.mean(full)) if full else float("nan"),
            "mean_class_size": float(np.mean(sizes)) if sizes else float("nan"),
            "by_construction": False}


def regime_block(scenario, bijection, registry, view, regime, seed, history_steps=32):
    """A memory input whose frames are rendered under the given appearance regime.

    The first version of this function zeroed the transfer token instead of re-rendering,
    which made PER_FRAME_BIJECTION and PER_CELL_NOISE return numbers identical to sixteen
    digits. Rendering is done properly here: calibration and transfer both pass through
    `render_regime`, so a regime that destroys the colour-to-role map destroys the tokens
    the memory is built from.
    """
    steps = []
    for index, episode in enumerate(scenario.calibration):
        cells, _ = render_regime(episode, bijection, regime, seed * 101 + index)
        for t in range(1, episode.length):
            steps.append(C.pair_tokens(cells[t - 1], cells[t],
                                       int(episode.actions[t - 1]), registry))
    history = np.zeros((history_steps, C.MAX_COLOURS, C.TOKEN_WIDTH), np.float32)
    mask = np.zeros(history_steps, np.float32)
    take = min(len(steps), history_steps)
    history[:take] = np.stack(steps[:take])
    mask[:take] = 1.0
    history = M.mask_view(history, view)

    tokens, before, after, event, meta = [], [], [], [], []
    for index, episode in enumerate(scenario.transfer):
        cells, _ = render_regime(episode, bijection, regime, seed * 977 + index)
        for t in range(1, episode.length):
            entered = int(episode.entered_role[t])
            if entered not in (C.SWITCH, C.DECOY):
                continue
            tokens.append(C.pair_tokens(cells[t - 1], cells[t],
                                        int(episode.actions[t - 1]), registry))
            before.append(C.cell_index(cells[t - 1], registry))
            after.append(C.cell_index(cells[t], registry))
            event.append(episode.event[t])
            meta.append((episode.layout, t, entered))
    pairs = {"tokens": M.mask_view(np.stack(tokens).astype(np.float32), view),
             "before_index": np.stack(before), "after_index": np.stack(after),
             "event": np.array(event, np.float32)}
    sequence, seq_mask, b, a, y = C.sequence_dataset(pairs, history, mask)
    return {"sequence": sequence, "mask": seq_mask, "before": b, "after": a,
            "event": y, "meta": np.array(meta, np.int64)}


def learned_queries(assignment: np.ndarray, entered: np.ndarray,
                    thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    """Per-query answers from the memory's assignment. No single global margin."""
    rows = np.arange(len(assignment))
    switch_mass = assignment[rows, entered, C.SWITCH]
    goal_gap = np.abs(assignment[:, :, C.GOAL_ALPHA] - assignment[:, :, C.GOAL_BETA]
                      ).max(axis=1)
    peak = assignment.max(axis=-1)
    full_confidence = peak.min(axis=1)
    return {
        "event_resolved": np.abs(switch_mass - 0.5) >= thresholds["EVENT"],
        "event_answer": (switch_mass > 0.5).astype(float),
        "goal_resolved": goal_gap >= thresholds["GOAL"],
        "full_resolved": full_confidence >= thresholds["FULL"],
        "switch_mass": switch_mass, "goal_gap": goal_gap,
        "full_confidence": full_confidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--palettes", type=int, default=24)
    parser.add_argument("--exact-episodes", type=int, default=12)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o3-uncertainty.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    registry = C.canonical_registry()
    print("training the memory on the development palettes", flush=True)
    train_blocks = []
    for palette in pop.DEV_PALETTES[:32]:
        plan = pop.palette_plan(palette, 6, 20, 2)
        train_blocks.append(O3.scenario_block(pop.palette_scenario(plan),
                                              plan["bijection"], registry, VIEW,
                                              contested_only=False))
    train = {k: np.concatenate([b[k] for b in train_blocks])
             for k in ("sequence", "mask", "before", "after", "event")}
    infer, model = M.train_memory(
        (train["sequence"], train["mask"], train["before"], train["after"],
         train["event"]), SEED, updates=mem.MEMORY_UPDATES)

    # ---- thresholds frozen on development ------------------------------------------
    dev_plan = pop.palette_plan(pop.DEV_PALETTES[0], 6, 20, 2)
    dev_block = O3.scenario_block(pop.palette_scenario(dev_plan), dev_plan["bijection"],
                                  registry, VIEW)
    dev_assignment = M.memory_assignment_of(model, dev_block["sequence"],
                                            dev_block["mask"])
    dev_entered = dev_block["sequence"][:, -1, :, C.INTERACT][:, :, 0].argmax(axis=1)
    dev_rows = np.arange(len(dev_assignment))
    thresholds = {
        # EVENT: the tenth percentile of the development margin, so nine in ten
        # development rows are answered.
        "EVENT": float(np.quantile(
            np.abs(dev_assignment[dev_rows, dev_entered, C.SWITCH] - 0.5), 0.10)),
        # GOAL: the exact posterior leaves 1.000 bit of goal ambiguity on every palette,
        # so the honest threshold is one the assignment can essentially never clear
        # without a grounding demonstration.
        "GOAL": 0.50,
        "FULL": 0.90,
    }
    thresholds["EVENT"] = min(thresholds["EVENT"], 0.40)
    print(f"frozen thresholds {thresholds}", flush=True)

    report: dict[str, Any] = {
        "view": VIEW, "seed": SEED, "regimes": list(REGIMES), "queries": list(QUERIES),
        "allowed_outputs": list(OUTPUTS),
        "thresholds": thresholds,
        "thresholds_frozen_on": f"development palette {pop.DEV_PALETTES[0]}",
        "palettes": arguments.palettes, "regime_results": {},
    }

    for regime in REGIMES:
        exact_rows, learned_rows, verification = [], [], []
        for palette in pop.VALIDATION_PALETTES[:arguments.palettes]:
            plan = pop.palette_plan(palette, 6, 20, 2)
            scenario = pop.palette_scenario(plan)
            episodes = scenario.transfer[:2]
            for index, episode in enumerate(episodes[:arguments.exact_episodes]):
                cells, maps = render_regime(episode, plan["bijection"], regime,
                                            seed=palette * 31 + index)
                verification.append(maps_genuinely_differ(maps))
                exact_rows.append(exact_queries(
                    episode, regime, tied=(regime == "PERSISTENT_CONVENTION")))

            block = regime_block(scenario, plan["bijection"], registry, VIEW, regime,
                                 seed=palette)
            assignment = M.memory_assignment_of(model, block["sequence"], block["mask"])
            entered = block["sequence"][:, -1, :, C.INTERACT][:, :, 0].argmax(axis=1)
            answers = learned_queries(assignment, entered, thresholds)
            truth = (block["meta"][:, 2] == C.SWITCH).astype(float)
            learned_rows.append({
                "palette": palette, "rows": int(len(truth)),
                "event_coverage": float(answers["event_resolved"].mean()),
                "event_accuracy_given_answer": (
                    float((answers["event_answer"] == truth)[
                        answers["event_resolved"]].mean())
                    if answers["event_resolved"].any() else None),
                "event_false_confident": float(
                    ((answers["event_answer"] != truth)
                     & answers["event_resolved"]).mean()),
                "goal_coverage": float(answers["goal_resolved"].mean()),
                "full_coverage": float(answers["full_resolved"].mean()),
            })

        def mean_of(key, source):
            values = [r[key] for r in source if r.get(key) is not None]
            return float(np.mean(values)) if values else None

        report["regime_results"][regime] = {
            "appearance_verification": {
                "episodes": len(verification),
                "mean_distinct_maps_per_episode": float(np.mean(
                    [v["distinct_maps"] for v in verification])),
                "episodes_where_all_frames_share_one_map": int(sum(
                    v["all_frames_share_one_map"] for v in verification)),
                "appearance_is_a_function_of_role": bool(
                    verification[0]["appearance_is_a_function_of_role"]),
            },
            "exact": {
                "event_identifiable": mean_of("event_identifiable", exact_rows),
                "goal_identifiable": mean_of("goal_identifiable", exact_rows),
                "full_identifiable": mean_of("full_identifiable", exact_rows),
                "by_construction_unidentifiable": bool(exact_rows[0]["by_construction"]),
            },
            "learned": {
                "event_coverage": mean_of("event_coverage", learned_rows),
                "event_accuracy_given_answer": mean_of("event_accuracy_given_answer",
                                                       learned_rows),
                "event_false_confident": mean_of("event_false_confident", learned_rows),
                "goal_coverage": mean_of("goal_coverage", learned_rows),
                "full_coverage": mean_of("full_coverage", learned_rows),
                "palettes": len(learned_rows),
            },
            "per_palette": learned_rows,
        }
        block = report["regime_results"][regime]
        print(f"\n{regime}")
        print(f"  appearance: {block['appearance_verification']['mean_distinct_maps_per_episode']:.2f} "
              f"distinct maps per episode; colour is a function of role: "
              f"{block['appearance_verification']['appearance_is_a_function_of_role']}")
        print(f"  exact:   event {block['exact']['event_identifiable']}  "
              f"goal {block['exact']['goal_identifiable']}  "
              f"full {block['exact']['full_identifiable']}")
        print(f"  learned: event coverage {block['learned']['event_coverage']:.4f}  "
              f"accuracy|answer {block['learned']['event_accuracy_given_answer']}  "
              f"false confident {block['learned']['event_false_confident']:.4f}  "
              f"goal coverage {block['learned']['goal_coverage']:.4f}  "
              f"full coverage {block['learned']['full_coverage']:.4f}", flush=True)

    persistent = report["regime_results"]["PERSISTENT_CONVENTION"]
    noise = report["regime_results"]["PER_CELL_NOISE"]
    per_frame = report["regime_results"]["PER_FRAME_BIJECTION"]
    report["per_frame_bijection_is_not_an_impossibility_control"] = {
        "distinct_maps_per_episode":
            per_frame["appearance_verification"]["mean_distinct_maps_per_episode"],
        "exact_event_identifiable": per_frame["exact"]["event_identifiable"],
        "why": ("the mappings genuinely differ frame to frame, and the event stays "
                "identifiable anyway: static-scene legality pins the relative "
                "relabelling, so an observer can undo a per-frame recolouring. It is a "
                "control against colour-ADDRESSED memory across episodes, not against "
                "event identifiability."),
    }
    report["R9_uncertainty_is_query_scoped"] = bool(
        persistent["learned"]["goal_coverage"] < 0.05
        and persistent["learned"]["event_coverage"] > 0.5)
    report["R10_unidentifiable_appearance_is_unresolved"] = bool(
        noise["learned"]["event_false_confident"] < 0.10)
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nR9 {report['R9_uncertainty_is_query_scoped']}   "
          f"R10 {report['R10_unidentifiable_appearance_is_unresolved']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
