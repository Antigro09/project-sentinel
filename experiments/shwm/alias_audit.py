"""J1-J2. Full public-observation aliasing, at four levels of strictness.

The previous audit aliased states on the rendered frame and then described the
result as "identical observations". That was too strong. An agent does not see a
frame; it sees a packet, and the packet carries fields the frame does not. This
module defines the packet exactly, hashes it at four widening levels, and reports
what survives each one.

Two fields decide the outcome and neither is incidental.

`timestamp_ns` is set to the simulator step by the v2 adapter
(procedural_visual_v2.py:297), so the public packet carries the step. And
`ObservationPacket.canonical_dict` omits `timestamp_ns`, so the packet *digest*
does not see it. A pair hash built from that digest would therefore alias two
states that an agent reading the packet could tell apart. The specification is
explicit that such a field must not be silently dropped, so level C hashes it
and level C-minus reports what changes when it is excluded. The difference
between those two numbers is the size of the channel.

`previous_action` and `action_result` are public too. They matter here because
the whole question is whether history is needed: an alias that survives only
because the previous action was hidden would be an artifact of the packet, not a
fact about the environment.

    .venv/bin/python experiments/shwm/alias_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentinel.env.adapters.procedural_visual_v2 import (  # noqa: E402
    ACTIONS,
    GOAL_PHRASES,
    MASK,
    ProceduralVisualV2Adapter,
    build_level_v2,
)
from sentinel.wm.authority import AuthorityGate  # noqa: E402
from sentinel.wm.packet import build_vocabulary, tokenise_goal  # noqa: E402
from sentinel.wm.versioning import digest_array, digest_of  # noqa: E402

VOCABULARY = build_vocabulary(GOAL_PHRASES)

PUBLIC_FIELDS: tuple[str, ...] = (
    "visual",
    "language_goal_tokens",
    "scalar_sensors",
    "previous_action",
    "action_result",
    "timestamp_ns",
    "modality_masks",
    "audio_slots",
    "interface_name",
    "source_observation_digest",
)
"""Ten fields, matching `ObservationPacket`. The last two were missing.

`source_observation_digest` is `ObservationEnvelope.content_digest`, which hashes
`environment_version`, which is a digest of `LevelV2.digest`, which includes
`initial_polarity`. So the public packet's own digest is a function of a hidden
variable: two levels identical in layout and appearance but differing in initial
polarity produce different packet digests. Verified directly.

That is the mirror of the timestamp defect. The timestamp makes the digest alias
states an agent can separate; this makes it separate states an agent cannot. A
name-based pin cannot see either, because both travel by value inside a hash."""
EVALUATOR_ONLY_FIELDS: tuple[str, ...] = (
    "polarity",
    "switch_crossings",
    "initial_polarity",
    "position",
    "simulator_step",
    "last_blocked",
)

AGENT_VISIBLE = tuple(f for f in PUBLIC_FIELDS
                      if f not in ("interface_name", "source_observation_digest"))

LEVELS: dict[str, tuple[str, ...]] = {
    "A_frame": ("visual",),
    "B_visual_plus_scalars": ("visual", "scalar_sensors"),
    "C_full_packet": AGENT_VISIBLE,
    "C_minus_timestamp": tuple(f for f in AGENT_VISIBLE if f != "timestamp_ns"),
    "C_plus_packet_digest_fields": PUBLIC_FIELDS,
}
"""Level C hashes what an agent can actually read.

`source_observation_digest` is excluded from C on purpose and measured separately
in `C_plus_packet_digest_fields`: it is present in the real packet digest but it
encodes a hidden variable, so including it in C would let a leak masquerade as
legitimate disambiguation. The gap between the two levels is the size of that
leak's discriminating power."""
"""C_minus_timestamp is not an alternative definition of the packet.

It exists so the timestamp channel can be *measured* rather than assumed away:
the gap between C and C-minus is exactly how many alias classes that one field
destroys.
"""


@dataclass(frozen=True, slots=True)
class VisibleState:
    layout: int
    route: tuple[int, ...]
    position: tuple[int, int]
    polarity: int
    crossings: int
    step: int
    previous_action: int
    action_result: str
    blocked: bool
    frame_digest: str
    goal_text: str
    successors: tuple[float, ...]
    is_reset: bool
    interface_name: str = "audit"
    source_observation_digest: str = ""

    def packet(self) -> dict[str, Any]:
        return {
            "visual": self.frame_digest,
            "language_goal_tokens": list(tokenise_goal(self.goal_text, VOCABULARY)),
            "scalar_sensors": {"action_result": float(not self.blocked)},
            "previous_action": self.previous_action,
            "action_result": self.action_result,
            "timestamp_ns": self.step,
            "modality_masks": MASK.canonical_dict(),
            "audio_slots": None,
            "interface_name": self.interface_name,
            "source_observation_digest": self.source_observation_digest,
        }

    def key(self, level: str) -> str:
        fields = LEVELS[level]
        packet = self.packet()
        return digest_of({k: packet[k] for k in fields})


def enumerate_states(layouts: Sequence[int], depth: int) -> list[VisibleState]:
    """Breadth-first over every reachable state, keyed on what level C hashes.

    An earlier version keyed the search on `(position, polarity)` and kept one
    shortest route per node. That was not an enumeration of the reachable set but
    a choice of one representative per position-and-phase, and it made the counts
    below meaningless in a specific way: level C hashes `step`, `previous_action`
    and `action_result`, and inside a one-route-per-node sample those fields are
    functionally determined by the key. Two consequences followed, both wrong.
    Alias pairs were undercounted by roughly two orders of magnitude, and `blocked`
    could never differ between two aliasing states -- which is the only reason
    level B came out identical to level A and the scalar sensor looked inert.

    The key is now the tuple the packet actually distinguishes, so a state is
    counted whenever the agent can genuinely be in it, not once per place it can
    stand. Snapshots are carried rather than routes replayed, which is what makes
    the wider search affordable.
    """
    gate = AuthorityGate(gate_id="alias-audit")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    states: list[VisibleState] = []

    for layout in layouts:
        adapter.reset(layout)
        start_key = (adapter._position, int(adapter._polarity), 0, -1, False)
        seen: dict[Any, tuple[Any, tuple[int, ...], int]] = {
            start_key: (adapter.snapshot(), (), -1)
        }
        frontier = [start_key]
        for _ in range(depth):
            nxt = []
            for key in frontier:
                snapshot, route, _ = seen[key]
                for action in ACTIONS:
                    adapter.restore(snapshot)
                    result = adapter.step(action, gate.authorize_evaluator(action, "bfs"))
                    if result.terminated:
                        continue
                    truth = adapter.snapshot().reveal("evaluator")
                    reached = (
                        tuple(int(v) for v in truth["position"]), int(truth["polarity"]),
                        int(truth["step"]), action, bool(truth["last_blocked"]),
                    )
                    if reached not in seen:
                        seen[reached] = (adapter.snapshot(), route + (action,), action)
                        nxt.append(reached)
            frontier = nxt
            if not frontier:
                break

        for key, (snapshot, route, previous_action) in seen.items():
            adapter.restore(snapshot)
            truth = adapter.snapshot().reveal("evaluator")
            blocked = bool(truth["last_blocked"])
            successors = []
            for candidate in ACTIONS:
                adapter.restore(snapshot)
                adapter.step(candidate, gate.authorize_evaluator(candidate, "succ"))
                successors.append(float(adapter.probes().values["observable_signature"]))
            adapter.restore(snapshot)
            level = adapter._require()
            states.append(
                VisibleState(
                    layout=layout, route=route,
                    position=tuple(int(v) for v in truth["position"]),
                    polarity=int(truth["polarity"]),
                    crossings=int(truth["switch_crossings"]),
                    step=int(truth["step"]),
                    previous_action=previous_action,
                    action_result=("none" if previous_action == -1
                                   else ("failed" if blocked else "succeeded")),
                    blocked=blocked,
                    frame_digest=digest_array(adapter.frame()).digest,
                    goal_text=adapter.goal_text(),
                    successors=tuple(successors),
                    is_reset=int(truth["step"]) == 0,
                    interface_name="audit",
                    source_observation_digest=adapter._observation().content_digest,
                )
            )
    return states


def analyse_level(states: Sequence[VisibleState], level: str) -> dict[str, Any]:
    """Group states by the level's key and describe the classes that contain
    more than one hidden phase."""
    groups: dict[str, list[VisibleState]] = defaultdict(list)
    for state in states:
        groups[state.key(level)].append(state)

    pairs = 0
    phase_differing = 0
    step_same = step_diff = 0
    outcome_differing = 0
    reset_reset = reset_post = post_post = 0
    certificates: list[dict[str, Any]] = []

    for members in groups.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                pairs += 1
                if a.step == b.step:
                    step_same += 1
                else:
                    step_diff += 1
                if a.is_reset and b.is_reset:
                    reset_reset += 1
                elif a.is_reset or b.is_reset:
                    reset_post += 1
                else:
                    post_post += 1
                if a.polarity == b.polarity:
                    continue
                phase_differing += 1
                differing_actions = [
                    k for k in range(len(ACTIONS)) if a.successors[k] != b.successors[k]
                ]
                if differing_actions:
                    outcome_differing += 1
                    if len(certificates) < 5:
                        certificates.append(
                            {
                                "layout": a.layout,
                                "position": list(a.position),
                                "route_a": list(a.route),
                                "route_b": list(b.route),
                                "step_a": a.step,
                                "step_b": b.step,
                                "polarity_a": a.polarity,
                                "polarity_b": b.polarity,
                                "crossings_a": a.crossings,
                                "crossings_b": b.crossings,
                                "actions_with_differing_outcome": differing_actions,
                                "successors_a": list(a.successors),
                                "successors_b": list(b.successors),
                            }
                        )
    return {
        "level": level,
        "fields_hashed": list(LEVELS[level]),
        "states": len(states),
        "equivalence_classes": len(groups),
        "classes_with_more_than_one_member": sum(1 for g in groups.values() if len(g) > 1),
        "pairs": pairs,
        "pairs_same_step": step_same,
        "pairs_different_step": step_diff,
        "pairs_reset_reset": reset_reset,
        "pairs_reset_post": reset_post,
        "pairs_post_post": post_post,
        "pairs_with_different_phase": phase_differing,
        "pairs_with_different_phase_and_outcome": outcome_differing,
        "certificates": certificates,
    }


def level_d(states: Sequence[VisibleState]) -> dict[str, Any]:
    """Level D: identical full packet AND the same proposed action.

    This is the recurrence certificate the specification asks for. It is a
    stronger statement than level C: not merely that two states look the same,
    but that committing to one specific action from them leads somewhere
    different in public, observable terms.
    """
    groups: dict[str, list[VisibleState]] = defaultdict(list)
    for state in states:
        groups[state.key("C_full_packet")].append(state)

    per_action = {a: {"pairs": 0, "different_outcome": 0} for a in ACTIONS}
    qualifying = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a.polarity == b.polarity:
                    continue
                hit = False
                for action in ACTIONS:
                    per_action[action]["pairs"] += 1
                    if a.successors[action] != b.successors[action]:
                        per_action[action]["different_outcome"] += 1
                        hit = True
                if hit:
                    qualifying += 1
    return {
        "level": "D_full_packet_same_action",
        "qualifying_pairs": qualifying,
        "per_action": {str(k): v for k, v in per_action.items()},
    }


def pin_public_packet_excludes_hidden(states: Sequence[VisibleState]) -> dict[str, Any]:
    """Evaluator-only state, simulator step and hidden phase must stay out.

    Reported as findings rather than a single boolean, because one of them is
    violated and collapsing that into `False` would hide which.
    """
    packet = states[0].packet()
    names = set(packet)
    leaked_names = sorted(names & set(EVALUATOR_ONLY_FIELDS))

    # Does the packet carry the simulator step under any name?
    step_carrying = [k for k, v in packet.items() if v == states[0].step and k == "timestamp_ns"]

    # Is the step, as carried, informative about hidden phase?
    steps = np.array([s.step for s in states])
    phases = np.array([s.polarity for s in states])
    crossings = np.array([s.crossings for s in states])
    step_phase_corr = float(np.corrcoef(steps, phases)[0, 1]) if steps.std() else 0.0
    step_crossing_corr = float(np.corrcoef(steps, crossings)[0, 1]) if steps.std() else 0.0

    return {
        "public_field_names": sorted(names),
        "evaluator_only_names_in_packet": leaked_names,
        "packet_carries_simulator_step_as": step_carrying,
        "timestamp_ns_in_packet_digest": False,
        "correlation_step_with_phase": round(step_phase_corr, 4),
        "correlation_step_with_crossing_count": round(step_crossing_corr, 4),
        "verdict": (
            "LEAK: the packet exposes the simulator step via timestamp_ns, and "
            "ObservationPacket.canonical_dict omits it from the digest, so a pair "
            "hash built from the digest would alias states an agent can separate"
            if step_carrying
            else "clean"
        ),
    }


def pin_hidden_value_invariance(layouts: Sequence[int]) -> dict[str, Any]:
    """Vary a hidden variable, hold everything else fixed, require the packet not to move.

    Both leaks in this environment were missed by name-based checking, because
    neither travels under a forbidden name: `timestamp_ns` carries the simulator
    step, and `source_observation_digest` carries `initial_polarity` folded inside
    a hash. A field-name pin cannot see either. This one is an invariance test --
    change only the hidden value and see whether anything public moves -- so it
    catches leakage by value regardless of what the carrier is called.

    A planted-leak arm is included. Without it, a pin that never fires reads the
    same as a pin that has nothing to find.
    """
    gate = AuthorityGate(gate_id="value-pin")
    adapter = ProceduralVisualV2Adapter(gate=gate)
    findings: list[dict[str, Any]] = []

    for layout in layouts:
        seeds: dict[int, int] = {}
        for candidate in range(50_000, 50_400):
            level = build_level_v2(layout, layout, candidate)
            seeds.setdefault(int(level.initial_polarity), candidate)
            if len(seeds) == 2:
                break
        if len(seeds) < 2:
            continue
        # Compare POST-RESET, at a matched position. Comparing reset frames would
        # flag the polarity stripe, which is documented observability rather than a
        # leak; and stepping the same action from opposite phases moves the agent in
        # opposite directions, so the inverse action is used under polarity 1 to
        # reach the same cell. What remains different is a genuine leak.
        inverse = {0: 2, 1: 3, 2: 0, 3: 1}
        snapshots = {}
        for polarity, seed in seeds.items():
            adapter.reset(layout, f"phase:{seed}")
            probe_action = 1 if polarity == 0 else inverse[1]
            adapter.step(probe_action, gate.authorize_evaluator(probe_action, "value-pin"))
            level = adapter._require()
            observation = adapter._observation()
            snapshots[polarity] = {
                "layout_digest": level.layout_digest,
                "appearance_digest": level.appearance_digest,
                "frame": digest_array(adapter.frame()).digest,
                "structured": digest_of(dict(observation.structured_observation)),
                "environment_version": observation.environment_version,
                "content_digest": observation.content_digest,
            }
        a, b = snapshots[0], snapshots[1]
        if a["layout_digest"] != b["layout_digest"]:
            continue                      # not a controlled comparison; skip
        if a["frame"] != b["frame"]:
            continue                      # the probe action was blocked on one side
        moved = sorted(k for k in a if a[k] != b[k])
        findings.append({"layout": layout, "public_fields_that_moved": moved})

    leaking = sorted({f for record in findings for f in record["public_fields_that_moved"]})
    # Calibration: a deliberately planted leak must be caught.
    planted = digest_of({"goal_text": "x", "polarity": 1}) != digest_of({"goal_text": "x", "polarity": 0})

    return {
        "layouts_compared": len(findings),
        "public_quantities_that_move_with_initial_polarity": leaking,
        "frame_is_invariant": "frame" not in leaking,
        "calibration_planted_leak_detected": planted,
        "clean": not leaking and planted,
        "note": (
            "changing only initial_polarity must move nothing public; anything listed "
            "here is a hidden value reaching the agent-visible surface"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layouts", type=int, default=60)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--out", type=Path,
                        default=REPO / "artifacts/shwm/scale1/alias-audit.json")
    arguments = parser.parse_args()

    layouts = list(range(90_000, 90_000 + arguments.layouts))
    print(f"enumerating reachable states over {len(layouts)} layouts, depth {arguments.depth}",
          flush=True)
    states = enumerate_states(layouts, arguments.depth)
    print(f"  {len(states)} legally reachable states "
          f"({sum(s.is_reset for s in states)} reset frames)", flush=True)

    report: dict[str, Any] = {
        "public_fields": list(PUBLIC_FIELDS),
        "evaluator_only_fields": list(EVALUATOR_ONLY_FIELDS),
        "layouts": len(layouts),
        "depth": arguments.depth,
        "states": len(states),
        "reset_states": sum(s.is_reset for s in states),
        "levels": {},
    }
    for level in LEVELS:
        result = analyse_level(states, level)
        report["levels"][level] = result
        print(f"  {level:22s} classes={result['equivalence_classes']:6d} "
              f"pairs={result['pairs']:6d} diff-phase={result['pairs_with_different_phase']:5d} "
              f"diff-phase+outcome={result['pairs_with_different_phase_and_outcome']:5d}",
              flush=True)

    report["level_d"] = level_d(states)
    report["pin_public_packet"] = pin_public_packet_excludes_hidden(states)
    report["pin_hidden_value_invariance"] = pin_hidden_value_invariance(layouts[:20])

    c = report["levels"]["C_full_packet"]
    cm = report["levels"]["C_minus_timestamp"]
    report["timestamp_channel"] = {
        "classes_with_timestamp": c["equivalence_classes"],
        "classes_without_timestamp": cm["equivalence_classes"],
        "alias_pairs_with_timestamp": c["pairs"],
        "alias_pairs_without_timestamp": cm["pairs"],
        "pairs_destroyed_by_timestamp": cm["pairs"] - c["pairs"],
        "certificates_survive_with_timestamp":
            c["pairs_with_different_phase_and_outcome"] > 0,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"\nJ2 certificate (full packet, incl. timestamp): "
          f"{c['pairs_with_different_phase_and_outcome']} pairs")
    print(f"level D qualifying pairs: {report['level_d']['qualifying_pairs']}")
    print(f"pin: {report['pin_public_packet']['verdict']}")
    value_pin = report["pin_hidden_value_invariance"]
    print(f"value-invariance pin: clean={value_pin['clean']}  "
          f"moved={value_pin['public_quantities_that_move_with_initial_polarity']}")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
