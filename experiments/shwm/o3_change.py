"""I. Cause hypotheses, provisional branching, and an evidence-side unresolved signal.

Section G showed that a confidence-margin abstention rule cannot express "I have no
information": a collapsed assignment is SATURATED, so the margin reads maximum confidence
exactly where the model knows least. Measured, that rule answers 100% of rows under an
appearance process containing no information at all and is wrong on 0.5598 of them. The
signal therefore has to come from the evidence rather than from the model's self-report.

TWO EARLIER VERSIONS OF THAT SIGNAL WERE MEASURED AND BOTH FAILED. They are kept in
`REFUTED` rather than deleted, because each failed for a reason that constrains the
design.

  set-equality on on-screen colours
      A palette is a permutation of ONE eight-colour pool, so two palettes share six or
      seven of their seven colours and the on-screen colour SET is very nearly palette
      invariant. What the set does track is CONTENT: which roles a layout happens to
      instantiate. Measured, the no-change control false-alarmed at 0.25 per palette and
      mean detection delay went NEGATIVE, both of which are content variation being read
      as palette change.

  magnitude of interaction support
      `support <= 0` classifies PER_CELL_NOISE (support 0) but not PER_FRAME_BIJECTION
      (support 6), which carries nonzero support and still exhibits no colour-to-role
      map. A magnitude threshold low enough to catch it also fires on an honest but quiet
      episode, so magnitude cannot separate "no map exists" from "little happened".

What survives is CONSISTENCY, not magnitude, of a behavioural signature. Three anchors
are derivable from geometry and behaviour with no model and no role labels:

    BORDER  modal colour of the outer ring          -> WALL
    FIELD   modal interior colour, border excluded  -> EMPTY
    MOVER   colour of the moving singleton          -> AGENT

Measured over 24 palettes each anchor is single-valued within a palette and equals the
true role colour on 24/24. Under PER_FRAME_BIJECTION they take 7-8 distinct values within
one episode, and under PER_CELL_NOISE 7-10, so inconsistency separates both from an
honest quiet episode without ever consulting the magnitude of anything.

The signature has a measured intrinsic ceiling: two independently drawn palettes agree on
all three anchors on 0.0072 of pairs, and such a change is undetectable by construction.
That is reported as a ceiling, not charged to the mechanism.

A contradiction opens a PROVISIONAL branch; confirmed palette memory is never overwritten
until the branch is promoted, which is what stops a silent change from corrupting what was
already established.

    .venv-shwm/bin/python experiments/shwm/o3_change.py
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import o2_core as C
import o2_memory as mem
import o2_models as M
import o3_core as O3
import o3_population as pop
import o3_uncertainty as unc
from m2d_core import ARTIFACTS, write

SEED = 55_000
VIEW = "no_rgb"
HISTORY_STEPS = 32

CAUSES = ("SAME_PALETTE", "NEW_PALETTE", "MISSING_APPEARANCE")
ANCHORS = ("BORDER", "FIELD", "MOVER")

ARMS = (
    "1_no_change_detector",
    "2_declared_reset",
    "3_contradiction_detector",
    "4_exact_change_point_ceiling",
    "5_false_change_control",
    "6_silent_palette_change",
    "7_non_anchor_transposition",
    "8_gradual_drift_reaching_anchor",
    "9_low_interaction_persistent",
    "10_missing_appearance_zero_support",
    "11_missing_appearance_nonzero_support",
)

# What each arm is FOR. Written down so a bare 0.0000 in the table is read as the
# designed outcome rather than as a failure.
EXPECTATION = {
    "1_no_change_detector": "no detector at all; measures corruption when a change lands",
    "2_declared_reset": "told the change out of band, so it never needs to DETECT one; "
                        "bounds clean handling",
    "3_contradiction_detector": "the mechanism under test",
    "4_exact_change_point_ceiling": "told the truth; the achievable ceiling",
    "5_false_change_control": "no change is present; any detection is a false alarm",
    "6_silent_palette_change": "the operational test: a change with no announcement",
    "7_non_anchor_transposition":
        "SWITCH and DECOY exchange colours and nothing else moves. This is a legal "
        "palette change that provably leaves BORDER/FIELD/MOVER fixed, so a model-free "
        "signature cannot see it -- and it is the ADVERSARIAL worst case, because "
        "SWITCH against DECOY is the very distinction the task turns on. Detection 0 "
        "here is the measured boundary of what section I closes",
    "8_gradual_drift_reaching_anchor":
        "a drift that reaches an anchor role at stage 3, so the expected delay is the "
        "number of episodes before an anchor moves",
    "9_low_interaction_persistent":
        "honest, persistent, and deliberately quiet. This is the arm the REFUTED "
        "magnitude rule would have failed",
    "10_missing_appearance_zero_support": "PER_CELL_NOISE: no map exists, support 0",
    "11_missing_appearance_nonzero_support":
        "PER_FRAME_BIJECTION: no persistent map exists, but support is NONZERO. The "
        "case a support-magnitude threshold cannot reach",
}

# Arms that must NEVER report NEW_PALETTE: no change is present in any of them.
HONEST_ARMS = ("5_false_change_control", "9_low_interaction_persistent")
# Arms whose appearance process exhibits no colour-to-role map at all.
UNRESOLVABLE_ARMS = ("10_missing_appearance_zero_support",
                     "11_missing_appearance_nonzero_support")
# The drift arm the signature cannot see. Excluded from the honest-control false-alarm
# statistic because a change IS present; excluded from the detection requirement because
# no anchor moves.
BLIND_BY_CONSTRUCTION = ("7_non_anchor_transposition",)

REFUTED = {
    "set_equality_signature": (
        "frozenset of on-screen colours. A palette permutes one eight-colour pool, so "
        "the set is nearly palette invariant and tracks content instead. Measured: "
        "false alarms 0.25 per palette on the no-change control, mean delay -0.33."),
    "support_magnitude_threshold": (
        "support <= 0 as the unresolved signal. Catches PER_CELL_NOISE (support 0) but "
        "not PER_FRAME_BIJECTION (support 6), and any threshold low enough to catch the "
        "latter also fires on an honest quiet episode."),
}


# ---- the model-free signature ---------------------------------------------------------


def _modal(pixels: np.ndarray) -> tuple[int, int, int] | None:
    if len(pixels) == 0:
        return None
    unique, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
    return tuple(int(v) for v in unique[counts.argmax()])


def anchors(before: np.ndarray, after: np.ndarray) -> dict[str, tuple | None]:
    """BORDER / FIELD / MOVER from one frame pair. No model, no role labels.

    FIELD excludes the border colour before taking the mode. The first version took the
    plain modal colour of the whole frame, which flips from EMPTY to WALL on a dense
    layout -- measured on palette 20002, where it took two distinct values inside a
    single palette and would have produced a spurious contradiction.
    """
    ring = np.concatenate([before[0], before[-1], before[1:-1, 0], before[1:-1, -1]])
    border = _modal(ring)
    inner = before[1:-1, 1:-1].reshape(-1, 3)
    if border is not None:
        inner = inner[~np.all(inner == np.array(border, dtype=inner.dtype), axis=-1)]
    step = C.moving_singleton(before, after)
    mover = None if step is None else tuple(int(v) for v in before[step[0]])
    return {"BORDER": border, "FIELD": _modal(inner), "MOVER": mover}


def evidence_support(tokens: np.ndarray) -> float:
    """Object-like behaviour exhibited by the appearance. Public and model-free.

    Reported for every arm because the REFUTED magnitude rule has to stay falsifiable:
    arm 10 carries nonzero support and is still unresolvable, and arm 8 carries low
    support and is still honest. Both numbers appear in the table.
    """
    return float(np.abs(tokens[..., C.INTERACT]).sum())


def episode_signature(cells: np.ndarray, episode: C.O2Episode, registry
                      ) -> dict[str, Any]:
    """One episode's signature, its within-episode consistency, and its support."""
    observed: dict[str, set] = {a: set() for a in ANCHORS}
    tokens = []
    for t in range(1, episode.length):
        found = anchors(cells[t - 1], cells[t])
        for key, value in found.items():
            if value is not None:
                observed[key].add(value)
        tokens.append(C.pair_tokens(cells[t - 1], cells[t],
                                    int(episode.actions[t - 1]), registry))
    stacked = np.stack(tokens) if tokens else np.zeros((0, 1, C.TOKEN_WIDTH), np.float32)
    consistent = {a: (next(iter(observed[a])) if len(observed[a]) == 1 else None)
                  for a in ANCHORS}
    inconsistent = [a for a in ANCHORS if len(observed[a]) > 1]
    derivable = [a for a in ANCHORS if len(observed[a]) >= 1]
    return {
        "signature": consistent,
        "inconsistent": inconsistent,
        "derivable": derivable,
        "counts": {a: len(observed[a]) for a in ANCHORS},
        "support": evidence_support(stacked),
        "pairs": len(tokens),
    }


# ---- confirmed memory with a provisional branch ---------------------------------------


@dataclass
class PaletteMemory:
    """Confirmed signature plus a provisional branch.

    The branch NEVER overwrites the confirmed state until it is promoted. `since` is the
    index of the first episode consistent with the currently confirmed signature, which
    is what lets a recalibration drop the contaminated prefix instead of averaging over
    it.
    """

    promote_after: int = 2
    min_components: int = 2
    confirmed: dict | None = None
    since: int = 0
    provisional: dict | None = None
    provisional_since: int = 0
    provisional_steps: int = 0
    promotions: int = 0
    history: list = field(default_factory=list)

    @staticmethod
    def _compare(a: dict, b: dict) -> tuple[int, int]:
        """(agreements, disagreements) over anchors present and consistent in both."""
        agree = dis = 0
        for key in ANCHORS:
            if a.get(key) is None or b.get(key) is None:
                continue
            if a[key] == b[key]:
                agree += 1
            else:
                dis += 1
        return agree, dis

    def observe(self, block: dict, index: int) -> str:
        # An appearance process that exhibits no stable colour-to-role map at all.
        # Consistency, not magnitude: an honest quiet episode has few anchors but no
        # CONTRADICTORY ones, while a per-frame process contradicts itself inside the
        # episode.
        if block["inconsistent"] or not block["derivable"]:
            self.history.append("MISSING_APPEARANCE")
            return "MISSING_APPEARANCE"

        signature = block["signature"]
        if self.confirmed is None:
            self.confirmed, self.since = dict(signature), index
            self.history.append("SAME_PALETTE")
            return "SAME_PALETTE"

        agree, dis = self._compare(signature, self.confirmed)
        if dis == 0:
            # Corroboration fills in anchors the confirmed signature had not yet seen,
            # without ever changing one that is already set.
            for key in ANCHORS:
                if self.confirmed.get(key) is None and signature.get(key) is not None:
                    self.confirmed[key] = signature[key]
            self.provisional, self.provisional_steps = None, 0
            self.history.append("SAME_PALETTE")
            return "SAME_PALETTE"

        if agree + dis < self.min_components:
            # Too little overlap to call a contradiction. Neither confirm nor branch.
            self.history.append("SAME_PALETTE")
            return "SAME_PALETTE"

        if self.provisional is not None and self._compare(
                signature, self.provisional)[1] == 0:
            self.provisional_steps += 1
            for key in ANCHORS:
                if self.provisional.get(key) is None and signature.get(key) is not None:
                    self.provisional[key] = signature[key]
        else:
            self.provisional = dict(signature)
            self.provisional_since = index
            self.provisional_steps = 1

        if self.provisional_steps >= self.promote_after:
            self.confirmed = dict(self.provisional)
            self.since = self.provisional_since
            self.provisional, self.provisional_steps = None, 0
            self.promotions += 1
        self.history.append("NEW_PALETTE")
        return "NEW_PALETTE"

    @property
    def provisional_open(self) -> bool:
        return self.provisional is not None


# ---- one arm --------------------------------------------------------------------------


ANCHOR_ROLES = (C.WALL, C.EMPTY, C.AGENT)          # BORDER, FIELD, MOVER
DRIFT_NON_ANCHOR = (C.SWITCH, C.GOAL_ALPHA, C.GOAL_BETA, C.DECOY)
DRIFT_REACHING_ANCHOR = (C.SWITCH, C.GOAL_ALPHA, C.AGENT, C.WALL, C.EMPTY)


def drift_bijection(base: np.ndarray, other: np.ndarray, order, stage: int
                    ) -> np.ndarray:
    """Relabel roles one at a time, SWAPPING so the result stays a bijection.

    Assigning `out[role] = other[role]` would let two roles share a colour mid-drift. A
    collision on the agent colour makes `moving_singleton` return None, which surfaces as
    a MISS rather than as the contradiction it should be -- so the drift is applied as a
    transposition instead and every stage is a legal palette.

    Two orders are used. DRIFT_NON_ANCHOR touches only roles that carry no anchor, so a
    signature built on BORDER/FIELD/MOVER is blind to it by construction; that arm
    measures the blind spot. DRIFT_REACHING_ANCHOR moves the agent at stage 3, so the
    expected detection delay there is two episodes.
    """
    out = np.array(base, copy=True)
    for role in order[:stage]:
        target = int(other[role])
        if int(out[role]) == target:
            continue
        holder = np.flatnonzero(out == target)
        if len(holder):
            index = int(holder[0])
            out[role], out[index] = target, int(out[role])
        else:
            out[role] = target
    return out



def arm_stream(arm: str, plan: dict, scenario: O3.Scenario, other: np.ndarray
               ) -> tuple[list, list[str], int | None]:
    """(per-episode bijection, per-episode regime, true change index) for one arm."""
    n = len(scenario.calibration)
    base = plan["bijection"]
    change_at = n // 2

    if arm in HONEST_ARMS:
        return [base] * n, ["PERSISTENT_CONVENTION"] * n, None
    if arm == "10_missing_appearance_zero_support":
        return [base] * n, ["PER_CELL_NOISE"] * n, None
    if arm == "11_missing_appearance_nonzero_support":
        return [base] * n, ["PER_FRAME_BIJECTION"] * n, None
    if arm == "7_non_anchor_transposition":
        swapped = np.array(base, copy=True)
        swapped[[C.SWITCH, C.DECOY]] = swapped[[C.DECOY, C.SWITCH]]
        bijections = [base if i < change_at else swapped for i in range(n)]
        return bijections, ["PERSISTENT_CONVENTION"] * n, change_at
    if arm == "8_gradual_drift_reaching_anchor":
        bijections = [base if i < change_at
                      else drift_bijection(base, other, DRIFT_REACHING_ANCHOR,
                                           i - change_at + 1)
                      for i in range(n)]
        return bijections, ["PERSISTENT_CONVENTION"] * n, change_at
    bijections = [base if i < change_at else other for i in range(n)]
    return bijections, ["PERSISTENT_CONVENTION"] * n, change_at


def first_anchor_visible_stage(base: np.ndarray, other: np.ndarray, order) -> int | None:
    """Which drift stage first moves an anchor role off its original colour."""
    for stage in range(1, len(order) + 1):
        moved = drift_bijection(base, other, order, stage)
        if any(moved[r] != base[r] for r in ANCHOR_ROLES):
            return stage
    return None


def transfer_accuracy(scenario: O3.Scenario, bijection: np.ndarray, registry, model,
                      calibration_episodes: list[C.O2Episode],
                      calibration_cells: list[np.ndarray], keep_from: int,
                      thresholds: dict[str, float], seed: int,
                      forced: bool) -> dict[str, Any]:
    """Score EVENT / GOAL / FULL separately on transfer, gated by the controller.

    `keep_from` is the first calibration episode the memory is allowed to use. That is
    the whole operational value of the provisional branch: a detected change lets the
    history drop its contaminated prefix instead of averaging across two conventions.
    """
    steps = []
    for index in range(keep_from, len(calibration_cells)):
        cells = calibration_cells[index]
        episode = calibration_episodes[index]
        for t in range(1, episode.length):
            steps.append(C.pair_tokens(cells[t - 1], cells[t],
                                       int(episode.actions[t - 1]), registry))
    history = np.zeros((HISTORY_STEPS, C.MAX_COLOURS, C.TOKEN_WIDTH), np.float32)
    mask = np.zeros(HISTORY_STEPS, np.float32)
    take = min(len(steps), HISTORY_STEPS)
    if take:
        history[:take] = np.stack(steps[:take])
        mask[:take] = 1.0
    history = M.mask_view(history, VIEW)

    tokens, before, after, event = [], [], [], []
    for index, episode in enumerate(scenario.transfer):
        cells, _ = unc.render_regime(episode, bijection, "PERSISTENT_CONVENTION",
                                     seed * 977 + index)
        for t in range(1, episode.length):
            entered = int(episode.entered_role[t])
            if entered not in (C.SWITCH, C.DECOY):
                continue
            tokens.append(C.pair_tokens(cells[t - 1], cells[t],
                                        int(episode.actions[t - 1]), registry))
            before.append(C.cell_index(cells[t - 1], registry))
            after.append(C.cell_index(cells[t], registry))
            event.append(episode.event[t])
    if not tokens:
        return {"rows": 0}

    pairs = {"tokens": M.mask_view(np.stack(tokens).astype(np.float32), VIEW),
             "before_index": np.stack(before), "after_index": np.stack(after),
             "event": np.array(event, np.float32)}
    sequence, seq_mask, b, a, y = C.sequence_dataset(pairs, history, mask)
    assignment = M.memory_assignment_of(model, sequence, seq_mask)
    # The entered SLOT, read off the last history step's interaction block exactly as
    # section G reads it. `a` from `sequence_dataset` is the after-cell grid, not a slot.
    entered = sequence[:, -1, :, C.INTERACT][:, :, 0].argmax(axis=1)
    queries = unc.learned_queries(assignment, entered, thresholds)

    out: dict[str, Any] = {"rows": int(len(y))}
    correct = queries["event_answer"] == y
    if forced:
        # The controller has declared the cause unresolvable: nothing is answered.
        out.update({
            "event_coverage": 0.0, "event_accuracy_given_answer": None,
            "event_false_confident": 0.0,
            "goal_coverage": 0.0, "full_coverage": 0.0,
            "unresolved_rate": 1.0,
        })
        return out
    resolved = queries["event_resolved"]
    out.update({
        "event_coverage": float(resolved.mean()),
        "event_accuracy_given_answer": (float(correct[resolved].mean())
                                        if resolved.any() else None),
        "event_false_confident": float((resolved & ~correct).mean()),
        "goal_coverage": float(queries["goal_resolved"].mean()),
        "full_coverage": float(queries["full_resolved"].mean()),
        "unresolved_rate": float((~resolved).mean()),
    })
    return out


def run_arm(arm: str, plan: dict, scenario: O3.Scenario, registry, model,
            thresholds: dict[str, float], promote_after: int, min_components: int,
            seed: int) -> dict[str, Any]:
    other = C.sample_bijection(plan["palette"] + 7_777)
    bijections, regimes, change_at = arm_stream(arm, plan, scenario, other)
    n = len(scenario.calibration)
    anchor_stage = reachable = None
    if arm == "8_gradual_drift_reaching_anchor":
        anchor_stage = first_anchor_visible_stage(plan["bijection"], other,
                                                  DRIFT_REACHING_ANCHOR)
        reachable = bool(anchor_stage is not None and anchor_stage <= n - change_at)

    if arm == "9_low_interaction_persistent":
        # Honest, persistent, and deliberately quiet: only the first two frame pairs of
        # each episode are shown. This is the false-alarm control the magnitude rule
        # would have failed.
        limit = 3
    else:
        limit = None

    memory = PaletteMemory(promote_after=promote_after, min_components=min_components)
    detections, blocks, cells_per_episode, episodes_seen = [], [], [], []
    for index, episode in enumerate(scenario.calibration):
        cells, _ = unc.render_regime(episode, bijections[index], regimes[index],
                                     seed * 31 + index)
        if limit is not None:
            cells = cells[:limit]
            episode = _truncate(episode, limit)
        cells_per_episode.append(cells)
        episodes_seen.append(episode)
        block = episode_signature(cells, episode, registry)
        blocks.append(block)

        if arm == "1_no_change_detector":
            detections.append("SAME_PALETTE")
        elif arm == "2_declared_reset":
            if change_at is not None and index == change_at:
                memory = PaletteMemory(promote_after=promote_after,
                                       min_components=min_components)
            detections.append(memory.observe(block, index))
        elif arm == "4_exact_change_point_ceiling":
            memory.observe(block, index)
            detections.append("NEW_PALETTE"
                              if (change_at is not None and index >= change_at)
                              else "SAME_PALETTE")
        else:
            detections.append(memory.observe(block, index))

    truth = ["SAME_PALETTE" if change_at is None or i < change_at else "NEW_PALETTE"
             for i in range(n)]
    if arm in UNRESOLVABLE_ARMS:
        truth = ["MISSING_APPEARANCE"] * n

    first_true = next((i for i, t in enumerate(truth) if t != "SAME_PALETTE"), None)
    target = "MISSING_APPEARANCE" if arm in UNRESOLVABLE_ARMS else "NEW_PALETTE"
    first_detected = next((i for i, d in enumerate(detections) if d == target), None)
    false_alarms = sum(1 for t, d in zip(truth, detections)
                       if t == "SAME_PALETTE" and d != "SAME_PALETTE")

    # -- old-memory corruption ----------------------------------------------------------
    # The confirmed signature is corrupt if it does not match the convention actually in
    # force at the end of the stream, on anchors both define.
    final_true = _true_signature(bijections[-1])
    if arm == "1_no_change_detector":
        # No detector: the memory is whatever the first episode set, never revised.
        held = _true_signature(bijections[0])
    else:
        held = memory.confirmed
    stale = 0
    if arm not in UNRESOLVABLE_ARMS and held is not None:
        stale = int(PaletteMemory._compare(held, final_true)[1] > 0)

    # -- operational scoring through the controller -------------------------------------
    unresolvable = arm in UNRESOLVABLE_ARMS or (
        detections and detections[-1] == "MISSING_APPEARANCE")
    forced = bool(unresolvable or memory.provisional_open)
    keep_from = 0 if arm == "1_no_change_detector" else memory.since
    scored = transfer_accuracy(scenario, bijections[-1], registry, model,
                               episodes_seen, cells_per_episode, keep_from, thresholds,
                               seed, forced=forced)

    # CORRUPTION IS A HARM, NOT A STALENESS. The first version scored "confirmed no
    # longer matches the convention in force" as corruption, which charged the drift
    # arms for holding a branch open and REFUSING to answer -- the exact behaviour the
    # mechanism exists to produce. Corruption is answering from a memory that is wrong.
    corrupt = int(bool(stale) and not forced)

    # The signature-level test above cannot see a change that leaves all three anchors
    # fixed, so it cannot see arm 7 either. This one can: it asks whether the history the
    # memory is actually reading spans more than one convention while the controller is
    # answering. It is derived from the generator, so it is an EVALUATOR statistic and
    # never available to the detector.
    span = {tuple(int(v) for v in bijections[i])
            for i in range(keep_from, len(bijections))}
    mixed = int(len(span) > 1 and not forced)
    recalibration = None if first_detected is None else int(
        sum(blocks[i]["pairs"] for i in range(memory.since, n)))

    return {
        "arm": arm,
        "change_at": change_at,
        "detections": detections,
        "detected": bool(first_detected is not None),
        "detection_delay": (None if first_true is None or first_detected is None
                            else int(first_detected - first_true)),
        "false_alarms": int(false_alarms),
        "old_memory_corrupt": corrupt,
        "confirmed_stale": stale,
        "answered": int(not forced),
        "answered_from_mixed_convention_history": mixed,
        "conventions_in_history": len(span),
        "promotions": int(memory.promotions),
        "provisional_open": bool(memory.provisional_open),
        "mean_support": float(np.mean([b["support"] for b in blocks])),
        "mean_pairs": float(np.mean([b["pairs"] for b in blocks])),
        "anchor_counts": {a: float(np.mean([b["counts"][a] for b in blocks]))
                          for a in ANCHORS},
        "recalibration_interactions": recalibration,
        "keep_from": int(keep_from),
        "first_anchor_visible_stage": anchor_stage,
        "anchor_stage_reached_by_stream": reachable,
        # Whether THIS palette pair moves an anchor at all. A pair that collides on all
        # three anchors cannot be detected by this signature, and is scored against the
        # ceiling rather than against the mechanism.
        "change_detectable_in_principle": (
            None if change_at is None else
            bool(PaletteMemory._compare(_true_signature(bijections[0]),
                                        _true_signature(bijections[-1]))[1] > 0)),
        **{f"transfer_{k}": v for k, v in scored.items()},
    }


def _truncate(episode: C.O2Episode, limit: int) -> C.O2Episode:
    """A genuinely shorter episode: quiet because little happened, not because it was
    masked. Everything downstream indexes by t, so the arrays are cut consistently."""
    import dataclasses
    original = episode.length
    changes = {}
    for f in dataclasses.fields(episode):
        value = getattr(episode, f.name)
        if isinstance(value, np.ndarray) and value.shape[:1] == (original,):
            changes[f.name] = value[:limit]
    # `length` is a property over `frames`, so cutting the per-step arrays IS the cut.
    return dataclasses.replace(episode, **changes)


def _true_signature(bijection: np.ndarray) -> dict[str, tuple]:
    return {
        "BORDER": tuple(int(v) for v in C.COLOUR_POOL[bijection[C.WALL]]),
        "FIELD": tuple(int(v) for v in C.COLOUR_POOL[bijection[C.EMPTY]]),
        "MOVER": tuple(int(v) for v in C.COLOUR_POOL[bijection[C.AGENT]]),
    }


def signature_collision_ceiling(palettes) -> dict[str, float]:
    """How often two independent palettes agree on all three anchors.

    Such a change cannot be detected by this signature at all, so it bounds the
    detection rate from above and is not charged to the mechanism.
    """
    signatures = [tuple(_true_signature(C.sample_bijection(p)).values())
                  for p in palettes]
    pairs = list(itertools.combinations(signatures, 2))
    collide = sum(a == b for a, b in pairs)
    return {"pairs": len(pairs), "collisions": int(collide),
            "collision_rate": collide / len(pairs),
            "detection_ceiling": 1.0 - collide / len(pairs)}


# ---- driver ---------------------------------------------------------------------------


def evaluate(palettes, registry, model, thresholds, promote_after, min_components,
             label: str, verbose: bool = True) -> dict[str, Any]:
    table: dict[str, Any] = {}
    if verbose:
        print(f"\n[{label}]  {'arm':38s} {'detect':>7s} {'delay':>6s} {'false':>6s} "
              f"{'corrupt':>8s} {'unres':>7s} {'falseconf':>10s} {'support':>8s}")
        print("-" * 108)
    for arm in ARMS:
        rows = []
        for palette in palettes:
            plan = pop.palette_plan(palette, 6, 8, 1)
            scenario = pop.palette_scenario(plan)
            rows.append(run_arm(arm, plan, scenario, registry, model, thresholds,
                                promote_after, min_components, seed=palette))
        delays = [r["detection_delay"] for r in rows if r["detection_delay"] is not None]
        unres = [r.get("transfer_unresolved_rate") for r in rows
                 if r.get("transfer_unresolved_rate") is not None]
        fconf = [r.get("transfer_event_false_confident") for r in rows
                 if r.get("transfer_event_false_confident") is not None]
        acc = [r["transfer_event_accuracy_given_answer"] for r in rows
               if r.get("transfer_event_accuracy_given_answer") is not None]
        recal = [r["recalibration_interactions"] for r in rows
                 if r["recalibration_interactions"] is not None]
        detectable = [r for r in rows if r["change_detectable_in_principle"]]
        block = {
            "palettes": len(rows),
            "detection_rate": float(np.mean([r["detected"] for r in rows])),
            "detectable_palettes": len(detectable),
            "detection_rate_among_detectable": (
                float(np.mean([r["detected"] for r in detectable]))
                if detectable else None),
            "mean_detection_delay": float(np.mean(delays)) if delays else None,
            "false_alarms_per_palette": float(np.mean([r["false_alarms"] for r in rows])),
            "old_memory_corruption_rate": float(np.mean(
                [r["old_memory_corrupt"] for r in rows])),
            "confirmed_stale_rate": float(np.mean([r["confirmed_stale"] for r in rows])),
            "answered_from_mixed_convention_rate": float(np.mean(
                [r["answered_from_mixed_convention_history"] for r in rows])),
            "answered_rate": float(np.mean([r["answered"] for r in rows])),
            "unresolved_rate": float(np.mean(unres)) if unres else None,
            "false_confident_rate": float(np.mean(fconf)) if fconf else None,
            "recovery_event_accuracy": float(np.mean(acc)) if acc else None,
            "mean_recalibration_interactions": float(np.mean(recal)) if recal else None,
            "goal_coverage": float(np.mean([r["transfer_goal_coverage"] for r in rows
                                            if "transfer_goal_coverage" in r])),
            "full_coverage": float(np.mean([r["transfer_full_coverage"] for r in rows
                                            if "transfer_full_coverage" in r])),
            "mean_support": float(np.mean([r["mean_support"] for r in rows])),
            "mean_promotions": float(np.mean([r["promotions"] for r in rows])),
            "expectation": EXPECTATION[arm],
            "anchor_stage_reached_fraction": (
                float(np.mean([bool(r["anchor_stage_reached_by_stream"]) for r in rows]))
                if rows[0]["first_anchor_visible_stage"] is not None else None),
            "per_palette": rows,
        }
        table[arm] = block
        if verbose:
            delay = block["mean_detection_delay"]
            unresolved = block["unresolved_rate"]
            false_confident = block["false_confident_rate"]
            print(f"{'':9s} {arm:38s} {block['detection_rate']:7.4f} "
                  f"{'  None' if delay is None else format(delay, '6.2f')} "
                  f"{block['false_alarms_per_palette']:6.2f} "
                  f"{block['old_memory_corruption_rate']:8.2f} "
                  f"{-1.0 if unresolved is None else unresolved:7.4f} "
                  f"{-1.0 if false_confident is None else false_confident:10.4f} "
                  f"{block['mean_support']:8.1f}", flush=True)
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-palettes", type=int, default=12)
    parser.add_argument("--palettes", type=int, default=16)
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "o3-change.json")
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
    thresholds = {"EVENT": 0.40, "GOAL": 0.50, "FULL": 0.90}

    ceiling = signature_collision_ceiling(pop.VALIDATION_PALETTES)
    print(f"signature collision ceiling: {ceiling['collision_rate']:.4f} of palette "
          f"pairs agree on all three anchors -> detection ceiling "
          f"{ceiling['detection_ceiling']:.4f}", flush=True)

    # ---- development: choose promote_after and min_components, then FREEZE -------------
    dev_palettes = list(pop.DEV_PALETTES[:arguments.dev_palettes])
    grid, chosen = [], None
    for promote_after in (1, 2, 3):
        for min_components in (1, 2):
            table = evaluate(dev_palettes, registry, model, thresholds, promote_after,
                             min_components, f"dev pa={promote_after} mc={min_components}",
                             verbose=False)
            detector = table["3_contradiction_detector"]
            honest = max(table[a]["false_alarms_per_palette"] for a in HONEST_ARMS)
            missing = min(table[a]["detection_rate"] for a in UNRESOLVABLE_ARMS)
            recovers = table["6_silent_palette_change"]["answered_rate"]
            # FROZEN SELECTION RULE, stated before the grid is run and not revisited
            # after validation is seen, in this order:
            #   1  detection among palettes whose change is detectable in principle
            #   2  fewest false alarms on the honest controls
            #   3  the unresolved signal on both uninformative arms
            #   4  the silent-change arm must still RECOVER inside the stream: a
            #      promote_after larger than the number of post-change episodes can
            #      never promote, so the branch stays open forever and "never corrupts"
            #      would be bought by never answering
            #   5  shortest detection delay
            #   6  on an exact tie, the MORE CONSERVATIVE setting (larger
            #      promote_after, larger min_components), because promote_after=1
            #      promotes on the first contradiction and the provisional branch would
            #      then never hold anything provisionally -- which is the property
            #      section I exists to demonstrate
            # The grid is expected to tie on 1-3, since an exact signature makes
            # detection easy; 4-6 are what actually choose.
            score = (detector["detection_rate_among_detectable"] or 0.0, -honest,
                     missing, recovers,
                     -(detector["mean_detection_delay"] or 0.0),
                     promote_after, min_components)
            grid.append({"promote_after": promote_after,
                         "min_components": min_components,
                         "detection_rate": detector["detection_rate"],
                         "detection_rate_among_detectable":
                             detector["detection_rate_among_detectable"],
                         "worst_honest_false_alarms": honest,
                         "missing_appearance_rate": missing,
                         "silent_arm_answered_rate": recovers,
                         "mean_detection_delay": detector["mean_detection_delay"],
                         "score": list(score)})
            print(f"  dev pa={promote_after} mc={min_components}: "
                  f"detect {detector['detection_rate']:.4f} "
                  f"false {honest:.4f} missing {missing:.4f} "
                  f"recovers {recovers:.4f} "
                  f"delay {detector['mean_detection_delay']}", flush=True)
            if chosen is None or score > tuple(chosen["score"]):
                chosen = grid[-1]

    promote_after = chosen["promote_after"]
    min_components = chosen["min_components"]
    print(f"\nFROZEN on development: promote_after={promote_after} "
          f"min_components={min_components}", flush=True)

    # ---- validation: FRESH palettes, fresh episodes, nothing further tuned -------------
    validation = list(pop.VALIDATION_PALETTES[:arguments.palettes])
    table = evaluate(validation, registry, model, thresholds, promote_after,
                     min_components, "validation")

    silent = table["6_silent_palette_change"]
    drift = table["8_gradual_drift_reaching_anchor"]
    blind = table["7_non_anchor_transposition"]
    none = table["1_no_change_detector"]
    controls = {a: table[a]["false_alarms_per_palette"] for a in HONEST_ARMS}
    missing = {a: table[a]["detection_rate"] for a in UNRESOLVABLE_ARMS}
    unresolved_missing = {a: table[a]["unresolved_rate"] for a in UNRESOLVABLE_ARMS}
    fconf_missing = {a: table[a]["false_confident_rate"] for a in UNRESOLVABLE_ARMS}

    # R10: an appearance process carrying no colour-to-role map must come back
    # UNRESOLVED, on BOTH the zero-support and the nonzero-support case, with no
    # confident wrong answers.
    r10 = bool(all(v == 1.0 for v in missing.values())
               and all(v == 1.0 for v in unresolved_missing.values())
               and all(v == 0.0 for v in fconf_missing.values()))
    # R11: a silent change is detected on every palette whose change is detectable in
    # principle, confirmed memory is never corrupted, and neither honest control fires.
    # Palettes that collide on all three anchors are charged to the measured ceiling.
    r11 = bool(silent["detection_rate_among_detectable"] == 1.0
               and drift["detection_rate_among_detectable"] == 1.0
               and silent["old_memory_corruption_rate"] == 0.0
               and drift["old_memory_corruption_rate"] == 0.0
               and max(controls.values()) == 0.0)

    report: dict[str, Any] = {
        "seed": SEED, "view": VIEW,
        "cause_hypotheses": list(CAUSES),
        "anchors": list(ANCHORS),
        "refuted_signals": REFUTED,
        "signature_collision_ceiling": ceiling,
        "frozen": {"promote_after": promote_after, "min_components": min_components,
                   "thresholds": thresholds,
                   "selected_on": "development palettes only",
                   "grid": grid},
        "development_palettes": dev_palettes,
        "validation_palettes": validation,
        "arms": table,
        "mechanism": ("a contradiction opens a PROVISIONAL branch; confirmed palette "
                      "memory is never overwritten until the branch is promoted"),
        "R10_unresolved_signal_on_uninformative_appearance": r10,
        "R11_silent_change_detected_or_held_provisional": r11,
        "no_detector_corruption_rate": none["old_memory_corruption_rate"],
        "measured_blind_spot": {
            "arm": "7_non_anchor_transposition",
            "detection_rate": blind["detection_rate"],
            "old_memory_corruption_rate": blind["old_memory_corruption_rate"],
            "answered_from_mixed_convention_rate":
                blind["answered_from_mixed_convention_rate"],
            "false_confident_rate": blind["false_confident_rate"],
            "statement": ("exchanging the SWITCH and DECOY colours is a legal palette "
                          "change that leaves BORDER, FIELD and MOVER fixed, so a "
                          "model-free signature cannot see it and the system answers "
                          "confidently from a memory bound to the old convention. This "
                          "bounds what section I closes: the mechanism detects changes "
                          "that move a BEHAVIOURALLY ANCHORED role, not every "
                          "relabelling, and the unseen case is the adversarial one."),
        },
        "honest_control_false_alarms": controls,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    write(arguments.out, report)
    print(f"\nR10 {r10}   R11 {r11}")
    print(f"wrote {arguments.out}  ({report['wall_clock_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
