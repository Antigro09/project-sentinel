"""D / O3. Which palette mappings does the permitted evidence actually separate?

The renderer is a per-cell lookup, so two hypotheses produce byte-identical pixels
exactly when one's semantic trajectory is a role-permutation of the other's:

    render(S, phi) == render(S', phi')   iff   S' = pi . S  with  pi = phi'^-1 . phi

So the observational equivalence class of the truth is precisely the set of role
permutations pi whose permuted trajectory pi.S is still a LEGAL trajectory of this
environment. That makes the audit exact and enumerable -- 720 permutations of six roles
-- rather than a sampling estimate, and every surviving pi is a literal collision
certificate: same pixels, different semantics.

Legality is checked in cascading evidence levels, so the output says which evidence
buys which separation.

    .venv-shwm/bin/python experiments/shwm/o_identifiability.py
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np

import o_core as O
from m2d_core import ARTIFACTS, write
from o_core import GRID, N_ROLES, ROLES, ROLE_INDEX
from structured_calibration import DELTAS_BY_INDEX

SWITCH_COUNT = 7
LEVELS = ("1_one_frame", "2_frame_pair_and_action", "3_reset_frame",
          "4_short_legal_history", "5_grounded_calibration_episode",
          "6_complete_permitted_history")


def permuted(roles: np.ndarray, pi: tuple[int, ...]) -> np.ndarray:
    table = np.array(pi, dtype=np.int64)
    return table[roles]


def cardinality_legal(grid: np.ndarray) -> bool:
    """What one frame can check: how many cells carry each role."""
    counts = np.bincount(grid.ravel(), minlength=N_ROLES)
    if counts[ROLE_INDEX["AGENT"]] != 1:
        return False
    # The agent occludes whatever it stands on, INCLUDING a goal marker, so a marker
    # count of 0 is legal. Requiring exactly one rejected the true trajectory on every
    # episode where the agent stepped onto a marker -- caught by the assertion that the
    # truth must survive its own evidence.
    if counts[ROLE_INDEX["GOAL_ALPHA"]] not in (0, 1):
        return False
    if counts[ROLE_INDEX["GOAL_BETA"]] not in (0, 1):
        return False
    if counts[ROLE_INDEX["SWITCH"]] not in (SWITCH_COUNT - 1, SWITCH_COUNT):
        return False
    if counts[ROLE_INDEX["WALL"]] < 1 or counts[ROLE_INDEX["EMPTY"]] < 1:
        return False
    return True


def agent_cell(grid: np.ndarray) -> tuple[int, int] | None:
    where = np.argwhere(grid == ROLE_INDEX["AGENT"])
    return tuple(int(v) for v in where[0]) if len(where) == 1 else None


def motion_legal(before: np.ndarray, after: np.ndarray, action: int) -> bool:
    """A frame pair plus the action: the agent must move by +/- the action delta or stay.

    The sign is unknown because polarity is hidden, so both are admitted; what is NOT
    admitted is a move to a cell the action could never reach.
    """
    a, b = agent_cell(before), agent_cell(after)
    if a is None or b is None:
        return False
    delta = (b[0] - a[0], b[1] - a[1])
    dr, dc = DELTAS_BY_INDEX[action]
    return delta in ((0, 0), (dr, dc), (-dr, -dc))


def static_scene_legal(before: np.ndarray, after: np.ndarray) -> bool:
    """Everything except the agent is static, so only the two agent cells may change."""
    changed = np.argwhere(before != after)
    if len(changed) == 0:
        return True
    if len(changed) > 2:
        return False
    cells = {tuple(int(v) for v in c) for c in changed}
    return cells <= {agent_cell(before), agent_cell(after)}


def switch_consistency_legal(roles: np.ndarray, actions: np.ndarray) -> bool:
    """Over a whole episode: entering a SWITCH must flip the sign of the action delta.

    This is the only evidence that separates SWITCH from EMPTY, and it is behavioural --
    no colour says it.
    """
    sign = None
    for t in range(1, len(roles)):
        a, b = agent_cell(roles[t - 1]), agent_cell(roles[t])
        if a is None or b is None:
            return False
        delta = (b[0] - a[0], b[1] - a[1])
        dr, dc = DELTAS_BY_INDEX[int(actions[t - 1])]
        if delta == (0, 0):
            observed = None
        elif delta == (dr, dc):
            observed = +1
        elif delta == (-dr, -dc):
            observed = -1
        else:
            return False
        if observed is not None:
            if sign is not None and observed != sign:
                return False        # the sign changed without a crossing to explain it
            sign = observed
        # Wall grounding: once the sign is known, a blocked move says the cell the
        # action pointed at is a WALL (or off-grid), and a completed move says the
        # destination is NOT a WALL. Without this the audit leaves WALL and EMPTY
        # exchangeable, because nothing else in the pixels distinguishes them.
        if sign is not None:
            target = (a[0] + sign * dr, a[1] + sign * dc)
            inside = 0 <= target[0] < GRID and 0 <= target[1] < GRID
            if delta == (0, 0):
                if inside and roles[t - 1][target] != ROLE_INDEX["WALL"]:
                    return False
            else:
                if roles[t - 1][b] == ROLE_INDEX["WALL"]:
                    return False
        # A crossing is the agent entering a cell that was SWITCH in the previous grid.
        if delta != (0, 0) and roles[t - 1][b] == ROLE_INDEX["SWITCH"]:
            sign = None if sign is None else -sign
    return True


def survivors(episode: O.Episode, level: str) -> list[tuple[int, ...]]:
    """Every role permutation whose permuted trajectory is legal at this evidence level."""
    out = []
    roles, actions = episode.roles, episode.actions
    for pi in itertools.permutations(range(N_ROLES)):
        grids = permuted(roles, pi)
        if level == "3_reset_frame":
            if cardinality_legal(grids[0]):
                out.append(pi)
            continue
        if not cardinality_legal(grids[0]):
            continue
        if level == "1_one_frame":
            out.append(pi)
            continue
        pairs = {"2_frame_pair_and_action": 1, "4_short_legal_history": 3}.get(
            level, len(grids) - 1)
        ok = True
        for t in range(1, min(pairs, len(grids) - 1) + 1):
            if not (cardinality_legal(grids[t])
                    and static_scene_legal(grids[t - 1], grids[t])
                    and motion_legal(grids[t - 1], grids[t], int(actions[t - 1]))):
                ok = False
                break
        if ok and level in ("5_grounded_calibration_episode", "6_complete_permitted_history"):
            ok = switch_consistency_legal(grids, actions)
        if ok:
            out.append(pi)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o-identifiability.json")
    arguments = parser.parse_args()
    started = time.perf_counter()

    episodes = O.collect_appearance(
        list(range(110_000, 110_000 + arguments.episodes)),
        "HIDDEN_PALETTE_CONVENTION", [7_001], 1, 9, seed=11)
    print(f"{len(episodes)} hidden-palette episodes; enumerating all 720 role "
          f"permutations per evidence level\n", flush=True)

    identity = tuple(range(N_ROLES))
    report: dict[str, Any] = {
        "roles": list(ROLES), "episodes": len(episodes),
        "permutations_enumerated": 720, "exhaustive": True,
        "why_exact": ("render is a per-cell lookup, so identical pixels means the "
                      "semantic trajectories differ by a role permutation; the class is "
                      "therefore enumerable rather than sampled"),
        "levels": {}}
    print(f"{'evidence level':34s} {'mean class':>11s} {'median':>8s} {'max':>5s} "
          f"{'identified':>11s} {'switch pinned':>14s}")
    print("-" * 92)
    for level in LEVELS:
        sizes, pinned_switch, pinned_agent = [], 0, 0
        certificates = []
        for episode in episodes:
            keep = survivors(episode, level)
            assert identity in keep, "the truth must always survive its own evidence"
            sizes.append(len(keep))
            # A role is pinned when every surviving permutation maps it to itself.
            for role, counter in (("SWITCH", "switch"), ("AGENT", "agent")):
                index = ROLE_INDEX[role]
                if all(pi[index] == index for pi in keep):
                    if counter == "switch":
                        pinned_switch += 1
                    else:
                        pinned_agent += 1
            if len(keep) > 1 and len(certificates) < 3:
                other = next(pi for pi in keep if pi != identity)
                certificates.append({
                    "layout": episode.layout,
                    "alternative_permutation": list(other),
                    "reads": {ROLES[i]: ROLES[other[i]] for i in range(N_ROLES)},
                    "same_pixels": True,
                    "changes_the_event": bool(other[ROLE_INDEX["SWITCH"]]
                                              != ROLE_INDEX["SWITCH"])})
        report["levels"][level] = {
            "mean_class_size": float(np.mean(sizes)),
            "median_class_size": float(np.median(sizes)),
            "max_class_size": int(np.max(sizes)),
            "min_class_size": int(np.min(sizes)),
            "identified_fraction": float(np.mean([s == 1 for s in sizes])),
            "switch_role_pinned_fraction": pinned_switch / len(episodes),
            "agent_role_pinned_fraction": pinned_agent / len(episodes),
            "collision_certificates": certificates}
        block = report["levels"][level]
        print(f"{level:34s} {block['mean_class_size']:11.2f} "
              f"{block['median_class_size']:8.1f} {block['max_class_size']:5d} "
              f"{block['identified_fraction']:11.3f} "
              f"{block['switch_role_pinned_fraction']:14.3f}", flush=True)

    separating = [l for l in LEVELS
                  if report["levels"][l]["switch_role_pinned_fraction"] >= 0.99]
    report["minimum_separating_evidence_for_switch"] = separating[0] if separating else None
    report["event_target_status"] = (
        "IDENTIFIABLE" if separating else "UNRESOLVED_APPEARANCE")
    report["wall_clock_seconds"] = time.perf_counter() - started
    write(arguments.out, report)
    print(f"\nminimum evidence that pins SWITCH: "
          f"{report['minimum_separating_evidence_for_switch'] or 'NONE OF THE PERMITTED LEVELS'}")
    print(f"event target under a hidden palette: {report['event_target_status']}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
