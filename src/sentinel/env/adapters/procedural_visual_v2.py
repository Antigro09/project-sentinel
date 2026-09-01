"""Adapter B v2: genuine partial observability and a language-selected goal.

v1 is preserved untouched, because Scale-0's artefacts refer to its generator
digest and altering it would invalidate a passed gate. This is a separate
environment, and the reason it exists is a defect v1's own tests could not see.

v1's "hidden" variable was `charge`, which equals `step % 3` while `steps_remaining`
sat in the observation. It was therefore a deterministic function of a public
quantity, and the invariance test that certified it passed only by varying charge
with step held fixed -- a combination no trajectory ever reaches. True and vacuous.

v2 hides something that cannot be reconstructed that way:

* `polarity` is drawn at reset from a seed stream independent of layout and
  appearance, so it is not a function of the level;
* it flips when the agent enters a switch cell, so it is event-driven rather
  than time-driven;
* it is rendered **only on the reset frame**, so a full history determines it and
  no later single frame does;
* its consequence is sparse -- it mirrors the movement delta, which is invisible
  until the agent moves.

Nothing about the step index is observable. There is no `steps_remaining` field
and no horizon surrogate, because those are exactly what let v1's hidden variable
be reconstructed.

The goal is selected by language. Two markers are rendered and the instruction
names one of them, so the same visual state has different correct actions under
different instructions -- which is the multimodal requirement, and also stops the
goal from being inferable from pixels alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from sentinel.env.adapters.base import (
    EnvironmentIdentity,
    HiddenSnapshot,
    ProbeSet,
    StepResult,
)
from sentinel.wm.authority import AuthorityGate, AuthorizationToken
from sentinel.wm.latent_contract import (
    ContractViolation,
    Modality,
    ModalityMask,
    ObservationEnvelope,
    Taint,
)
from sentinel.wm.versioning import digest_array, digest_of

GRID = 12
CELL = 2
FRAME = GRID * CELL
ACTIONS: tuple[int, ...] = (0, 1, 2, 3)
DELTAS: Mapping[int, tuple[int, int]] = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
DEFAULT_HORIZON = 32
SWITCH_COUNT = 7
GENERATOR_VERSION = "procedural-visual-v2"

MARKERS = ("alpha", "beta")
GOAL_PHRASES: Mapping[str, str] = {
    "alpha": "reach the alpha marker",
    "beta": "reach the beta marker",
}

MASK = ModalityMask(
    declared=(Modality.IMAGE, Modality.STRUCTURED, Modality.GOAL, Modality.TEXT, Modality.AUDIO),
    present=(Modality.IMAGE, Modality.STRUCTURED, Modality.GOAL, Modality.TEXT),
)
"""Audio is declared and absent. A declared-but-absent modality is a mask bit,
never a zero tensor, so a later audio channel does not change the schema."""


def _stream(material: Mapping[str, Any], count: int) -> np.ndarray:
    seed = digest_of(dict(material)).encode()
    blocks: list[bytes] = []
    counter = 0
    while sum(len(b) for b in blocks) < count * 4:
        blocks.append(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1
    return np.frombuffer(b"".join(blocks)[: count * 4], dtype=np.uint32)


@dataclass(frozen=True, slots=True)
class LevelV2:
    walls: np.ndarray
    palette: np.ndarray
    texture: np.ndarray
    start: tuple[int, int]
    markers: dict[str, tuple[int, int]]
    switches: tuple[tuple[int, int], ...]
    initial_polarity: int

    @property
    def digest(self) -> str:
        return digest_of(
            {
                "walls": digest_array(self.walls).canonical_dict(),
                "palette": digest_array(self.palette).canonical_dict(),
                "texture": digest_array(self.texture).canonical_dict(),
                "start": list(self.start),
                "markers": {k: list(v) for k, v in sorted(self.markers.items())},
                "switches": [list(s) for s in self.switches],
                "initial_polarity": self.initial_polarity,
            }
        )

    @property
    def layout_digest(self) -> str:
        """Layout and dynamics only. Excludes appearance, so an appearance shift
        can be constructed that provably leaves the layout identical."""
        return digest_of(
            {
                "walls": digest_array(self.walls).canonical_dict(),
                "start": list(self.start),
                "markers": {k: list(v) for k, v in sorted(self.markers.items())},
                "switches": [list(s) for s in self.switches],
            }
        )

    @property
    def appearance_digest(self) -> str:
        return digest_of(
            {
                "palette": digest_array(self.palette).canonical_dict(),
                "texture": digest_array(self.texture).canonical_dict(),
            }
        )


def build_level_v2(layout_seed: int, appearance_seed: int, phase_seed: int) -> LevelV2:
    """Three independent seed streams: layout, appearance, and hidden phase.

    The phase stream is separate so that the hidden variable is not a function of
    anything the agent can see, which is the property v1 lacked.
    """
    layout = _stream({"axis": "layout_v2", "seed": int(layout_seed)}, GRID * GRID + 32)
    looks = _stream({"axis": "appearance_v2", "seed": int(appearance_seed)}, GRID * GRID + 24)
    phase = _stream({"axis": "phase_v2", "seed": int(phase_seed)}, 4)

    walls = (layout[: GRID * GRID] % 100 < 12).reshape(GRID, GRID)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True

    taken: set[tuple[int, int]] = set()

    def free_cell(offset: int) -> tuple[int, int]:
        for i in range(GRID * GRID):
            index = int(layout[GRID * GRID + (offset % 16)] + i * 7) % (GRID * GRID)
            cell = divmod(index, GRID)
            if not walls[cell] and cell not in taken:
                taken.add(cell)
                return cell
        raise ContractViolation("generated level has no free cell")

    start = free_cell(0)
    markers = {"alpha": free_cell(3), "beta": free_cell(6)}

    # Switches are placed near the start, and densely, on purpose. Scattered
    # uniformly they were unreachable inside any practical search depth, so
    # polarity never flipped. Sparse but reachable was not enough either: at
    # three switches only 27% of episodes crossed one, which left polarity equal
    # to its initial value in the large majority and made "recovering the hidden
    # state from history" mostly a matter of reading the reset frame's
    # indicator. A hidden variable that is almost never exercised tests almost
    # nothing, whatever a probe scores on it.
    candidates = [
        (row, column)
        for row in range(GRID)
        for column in range(GRID)
        if not walls[row, column] and (row, column) not in taken
    ]
    candidates.sort(
        key=lambda cell: (abs(cell[0] - start[0]) + abs(cell[1] - start[1]), cell)
    )
    near = [c for c in candidates if 1 <= abs(c[0] - start[0]) + abs(c[1] - start[1]) <= 3]
    if len(near) < SWITCH_COUNT:
        near = candidates[:SWITCH_COUNT]
    picks: list[tuple[int, int]] = []
    for i in range(SWITCH_COUNT):
        if not near:
            break
        index = int(layout[GRID * GRID + 12 + i]) % len(near)
        picks.append(near.pop(index))
    switches = tuple(picks)
    taken.update(switches)

    palette = (looks[:15] % 200 + 30).astype(np.uint8).reshape(5, 3)
    texture = (looks[24 : 24 + GRID * GRID] % 40).astype(np.uint8).reshape(GRID, GRID)
    return LevelV2(
        walls=walls,
        palette=palette,
        texture=texture,
        start=start,
        markers=markers,
        switches=switches,
        initial_polarity=int(phase[0] % 2),
    )


def render_v2(level: LevelV2, position: tuple[int, int], show_polarity: int | None) -> np.ndarray:
    """RGB frame. `show_polarity` is set only on the reset frame.

    After reset the polarity is not drawn anywhere, so a single frame cannot
    reveal it, while a history that includes the reset frame and the switch cells
    crossed since determines it exactly.
    """
    frame = np.zeros((GRID, GRID, 3), dtype=np.uint8)
    frame[:, :] = level.palette[0]
    frame[level.walls] = level.palette[1]
    frame += level.texture[:, :, None] // 4
    for cell in level.switches:
        frame[cell] = level.palette[4]
    frame[level.markers["alpha"]] = level.palette[2]
    frame[level.markers["beta"]] = level.palette[3]
    frame[position] = 255 - level.palette[0]
    rendered = np.repeat(np.repeat(frame, CELL, axis=0), CELL, axis=1)
    if show_polarity is not None:
        # A one-pixel border stripe, present only at reset.
        rendered[0, :] = 255 if show_polarity else 0
    return rendered


@dataclass
class ProceduralVisualV2Adapter:
    """v2: hidden polarity, switch events, language-selected goal, no clock."""

    gate: AuthorityGate
    horizon: int = DEFAULT_HORIZON
    _level: LevelV2 | None = field(default=None, init=False)
    _position: tuple[int, int] = field(default=(0, 0), init=False)
    _polarity: int = field(default=0, init=False)
    _step: int = field(default=0, init=False)
    _layout_seed: int = field(default=0, init=False)
    _appearance_seed: int = field(default=0, init=False)
    _phase_seed: int = field(default=0, init=False)
    _goal_marker: str = field(default="alpha", init=False)
    _dynamic: str = field(default="base", init=False)
    _last_blocked: bool = field(default=False, init=False)
    _reached: bool = field(default=False, init=False)
    _interactions: int = field(default=0, init=False)
    _switch_crossings: int = field(default=0, init=False)

    @property
    def identity(self) -> EnvironmentIdentity:
        return EnvironmentIdentity(
            name="procedural_visual_v2",
            version=GENERATOR_VERSION,
            generator_digest=digest_of(
                {
                    "grid": GRID,
                    "cell": CELL,
                    "actions": list(ACTIONS),
                    "switches": SWITCH_COUNT,
                    "markers": list(MARKERS),
                    "version": GENERATOR_VERSION,
                }
            ),
            supports_branching=True,
        )

    @property
    def interactions(self) -> int:
        return self._interactions

    @property
    def episode_id(self) -> str:
        return (
            f"procedural_visual_v2:{self._dynamic}:{self._layout_seed}"
            f":{self._appearance_seed}:{self._phase_seed}:{self._goal_marker}"
        )

    def _require(self) -> LevelV2:
        if self._level is None:
            raise ContractViolation("adapter used before reset")
        return self._level

    def frame(self) -> np.ndarray:
        return render_v2(
            self._require(), self._position, self._polarity if self._step == 0 else None
        )

    def goal_text(self) -> str:
        return GOAL_PHRASES[self._goal_marker]

    def _observation(self) -> ObservationEnvelope:
        level = self._require()
        frame = self.frame()
        return ObservationEnvelope(
            episode_id=self.episode_id,
            step=self._step,
            timestamp_ns=self._step,
            modality_payloads={"image": digest_array(frame)},
            structured_observation={
                # Deliberately no step, horizon, or remaining-step field: those are
                # what let v1's hidden variable be reconstructed from public data.
                "goal_text": self.goal_text(),
                "frame_shape": list(frame.shape),
                "markers_visible": list(MARKERS),
                "audio_present": False,
            },
            modality_mask=MASK,
            available_action_digest=digest_of(list(ACTIONS)),
            environment_version=digest_of(
                {"identity": self.identity.digest, "level": level.digest}
            ),
            taint=frozenset({Taint.DEVELOPMENT}),
        )

    def probes(self) -> ProbeSet:
        level = self._require()
        goal = level.markers[self._goal_marker]
        distance = abs(self._position[0] - goal[0]) + abs(self._position[1] - goal[1])
        return ProbeSet(
            {
                "reward": 1.0 if self._reached else 0.0,
                "termination": self._reached or self._step >= self.horizon,
                "goal_progress": float(1.0 - distance / (2 * GRID)),
                "constraint_violation": bool(level.walls[self._position]),
                "action_succeeded": not self._last_blocked,
                "observable_signature": int(self._position[0] * GRID + self._position[1]),
            }
        )

    def legal_actions(self) -> tuple[int, ...]:
        return ACTIONS

    def reset(self, seed: int, dynamic: str = "base") -> StepResult:
        """`dynamic` selects which seed streams shift, so appearance and layout
        can be moved independently and provably."""
        self._dynamic = dynamic
        layout_seed = appearance_seed = phase_seed = int(seed)
        goal_draw = int(seed)
        if dynamic.startswith("appearance:"):
            appearance_seed = int(dynamic.split(":", 1)[1])
        elif dynamic.startswith("layout:"):
            layout_seed = int(dynamic.split(":", 1)[1])
        elif dynamic.startswith("phase:"):
            phase_seed = int(dynamic.split(":", 1)[1])
        elif dynamic.startswith("goal:"):
            goal_draw = int(dynamic.split(":", 1)[1])

        self._layout_seed, self._appearance_seed, self._phase_seed = (
            layout_seed,
            appearance_seed,
            phase_seed,
        )
        self._level = build_level_v2(layout_seed, appearance_seed, phase_seed)
        self._goal_marker = MARKERS[
            int(_stream({"axis": "goal_v2", "seed": goal_draw}, 1)[0]) % len(MARKERS)
        ]
        self._position = self._level.start
        self._polarity = self._level.initial_polarity
        self._step = 0
        self._reached = False
        self._last_blocked = False
        self._switch_crossings = 0
        return self._result()

    def _result(self) -> StepResult:
        probes = self.probes()
        return StepResult(
            observation=self._observation(),
            reward=float(probes.values["reward"]),
            terminated=bool(probes.values["termination"]),
            probes=probes,
            legal_actions=self.legal_actions(),
            info={"dynamic": self._dynamic},
        )

    def step(self, action: int, token: AuthorizationToken) -> StepResult:
        self.gate.consume(token, action)
        if action not in ACTIONS:
            raise ContractViolation(f"action {action} is not legal here; legal actions {ACTIONS}")
        level = self._require()
        row_delta, column_delta = DELTAS[action]
        if self._polarity:
            row_delta, column_delta = -row_delta, -column_delta

        candidate = (self._position[0] + row_delta, self._position[1] + column_delta)
        blocked = not (0 <= candidate[0] < GRID and 0 <= candidate[1] < GRID) or bool(
            level.walls[candidate]
        )
        if not blocked:
            self._position = candidate
            if candidate in level.switches:
                self._polarity ^= 1
                self._switch_crossings += 1
        self._last_blocked = blocked
        self._step += 1
        self._interactions += 1
        if self._position == level.markers[self._goal_marker]:
            self._reached = True
        return self._result()

    def snapshot(self) -> HiddenSnapshot:
        level = self._require()
        return HiddenSnapshot(
            payload={
                "position": list(self._position),
                "polarity": self._polarity,
                "step": self._step,
                "layout_seed": self._layout_seed,
                "appearance_seed": self._appearance_seed,
                "phase_seed": self._phase_seed,
                "goal_marker": self._goal_marker,
                "dynamic": self._dynamic,
                "reached": self._reached,
                "last_blocked": self._last_blocked,
                "switch_crossings": self._switch_crossings,
                "level_digest": level.digest,
                "_non_observable": ("polarity", "switch_crossings", "step"),
            },
            environment_version=self.identity.digest,
        )

    def restore(self, snapshot: HiddenSnapshot) -> StepResult:
        payload = snapshot.reveal("restore")
        if (
            self._level is None
            or payload["layout_seed"] != self._layout_seed
            or payload["appearance_seed"] != self._appearance_seed
            or payload["phase_seed"] != self._phase_seed
        ):
            self._level = build_level_v2(
                int(payload["layout_seed"]),
                int(payload["appearance_seed"]),
                int(payload["phase_seed"]),
            )
            self._layout_seed = int(payload["layout_seed"])
            self._appearance_seed = int(payload["appearance_seed"])
            self._phase_seed = int(payload["phase_seed"])
        if self._require().digest != payload["level_digest"]:
            raise ContractViolation(
                "restore target was generated by a different level; the snapshot cannot "
                "be replayed into this environment"
            )
        self._position = tuple(int(v) for v in payload["position"])  # type: ignore[assignment]
        self._polarity = int(payload["polarity"])
        self._step = int(payload["step"])
        self._goal_marker = str(payload["goal_marker"])
        self._dynamic = str(payload["dynamic"])
        self._reached = bool(payload["reached"])
        self._last_blocked = bool(payload["last_blocked"])
        self._switch_crossings = int(payload["switch_crossings"])
        return self._result()


# ---- reachable paired-state certificates -------------------------------------------


@dataclass(frozen=True, slots=True)
class HiddenStateCertificate:
    """Evidence that two *reachable* histories alias publicly and diverge under
    the same action.

    Reachable is the load-bearing word. Perturbing a hidden field in a snapshot
    and observing that the frame does not change proves nothing if no trajectory
    can produce that combination -- which is exactly how v1's charge test passed
    while charge was publicly determined. Both histories here are executed.
    """

    layout_seed: int
    appearance_seed: int
    phase_seed: int
    history_a: tuple[int, ...]
    history_b: tuple[int, ...]
    shared_content_digest: str
    polarity_a: int
    polarity_b: int
    switch_crossings_a: int
    switch_crossings_b: int
    probe_action: int
    successor_signature_a: int
    successor_signature_b: int
    observation_trace_a: tuple[int, ...]
    observation_trace_b: tuple[int, ...]

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "layout_seed": self.layout_seed,
            "appearance_seed": self.appearance_seed,
            "phase_seed": self.phase_seed,
            "history_a": list(self.history_a),
            "history_b": list(self.history_b),
            "shared_content_digest": self.shared_content_digest,
            "polarity_a": self.polarity_a,
            "polarity_b": self.polarity_b,
            "switch_crossings_a": self.switch_crossings_a,
            "switch_crossings_b": self.switch_crossings_b,
            "probe_action": self.probe_action,
            "successor_signature_a": self.successor_signature_a,
            "successor_signature_b": self.successor_signature_b,
            "observation_trace_a": list(self.observation_trace_a),
            "observation_trace_b": list(self.observation_trace_b),
        }


def _replay(adapter: "ProceduralVisualV2Adapter", seed: int, dynamic: str, actions) -> StepResult:
    result = adapter.reset(seed, dynamic)
    for action in actions:
        result = adapter.step(action, adapter.gate.authorize_evaluator(action, "certificate"))
    return result


def build_hidden_state_certificate(
    seed: int = 9000, dynamic: str = "base", max_depth: int = 7
) -> HiddenStateCertificate:
    """Search executed trajectories for a public alias with different polarity.

    Breadth-first over action words, keeping the shortest history that reaches
    each (position, polarity) pair. A pair of histories that reach the same
    position with opposite polarity aliases publicly at every step after reset,
    because polarity is drawn only on the reset frame.
    """
    adapter = ProceduralVisualV2Adapter(gate=AuthorityGate())
    reached: dict[tuple[tuple[int, int], int], tuple[int, ...]] = {}
    frontier: list[tuple[int, ...]] = [()]
    for _ in range(max_depth):
        next_frontier: list[tuple[int, ...]] = []
        for history in frontier:
            for action in ACTIONS:
                candidate = history + (action,)
                _replay(adapter, seed, dynamic, candidate)
                key = (adapter._position, adapter._polarity)
                if key not in reached:
                    reached[key] = candidate
                    next_frontier.append(candidate)
        frontier = next_frontier
        positions = {position for position, _ in reached}
        for position in positions:
            if (position, 0) in reached and (position, 1) in reached:
                history_a = reached[(position, 0)]
                history_b = reached[(position, 1)]
                if not history_a or not history_b:
                    continue  # a reset frame reveals polarity; it does not alias
                result_a = _replay(adapter, seed, dynamic, history_a)
                content = result_a.observation.content_digest
                snap_a = adapter.snapshot().reveal("fixture-builder")
                trace_a = _signature_trace(adapter, seed, dynamic, history_a)
                result_b = _replay(adapter, seed, dynamic, history_b)
                if result_b.observation.content_digest != content:
                    continue
                snap_b = adapter.snapshot().reveal("fixture-builder")
                trace_b = _signature_trace(adapter, seed, dynamic, history_b)
                for probe_action in ACTIONS:
                    _replay(adapter, seed, dynamic, history_a)
                    token = adapter.gate.authorize_evaluator(probe_action, "certificate")
                    after_a = adapter.step(probe_action, token).probes.values["observable_signature"]
                    _replay(adapter, seed, dynamic, history_b)
                    token = adapter.gate.authorize_evaluator(probe_action, "certificate")
                    after_b = adapter.step(probe_action, token).probes.values["observable_signature"]
                    if after_a != after_b:
                        return HiddenStateCertificate(
                            layout_seed=seed,
                            appearance_seed=seed,
                            phase_seed=seed,
                            history_a=history_a,
                            history_b=history_b,
                            shared_content_digest=content,
                            polarity_a=int(snap_a["polarity"]),
                            polarity_b=int(snap_b["polarity"]),
                            switch_crossings_a=int(snap_a["switch_crossings"]),
                            switch_crossings_b=int(snap_b["switch_crossings"]),
                            probe_action=probe_action,
                            successor_signature_a=int(after_a),
                            successor_signature_b=int(after_b),
                            observation_trace_a=trace_a,
                            observation_trace_b=trace_b,
                        )
    raise ContractViolation(
        f"no reachable public alias with differing polarity within depth {max_depth} "
        f"for seed {seed}; this environment cannot certify hidden state"
    )


def _signature_trace(
    adapter: "ProceduralVisualV2Adapter", seed: int, dynamic: str, actions
) -> tuple[int, ...]:
    result = adapter.reset(seed, dynamic)
    trace = [int(result.probes.values["observable_signature"])]
    for action in actions:
        result = adapter.step(action, adapter.gate.authorize_evaluator(action, "certificate"))
        trace.append(int(result.probes.values["observable_signature"]))
    return tuple(trace)


def build_language_certificate(seed: int = 9000, max_depth: int = 6) -> dict[str, Any]:
    """Evidence that the instruction changes the correct action with vision fixed.

    An earlier version compared only the best action at the start state and found
    none, concluding the task did not need language. It did not: both markers lie
    in the same direction from the start, so the *first* move agrees while later
    moves do not. Testing one state answered a narrower question than the one
    asked.

    This searches reachable states instead, and reports the first at which the
    two instructions disagree. The pixels are identical by construction -- the
    frame does not depend on which marker is named -- so any difference in the
    correct action is carried by language alone.
    """
    adapter = ProceduralVisualV2Adapter(gate=AuthorityGate())
    adapter.reset(seed)
    reachable: dict[tuple[tuple[int, int], int], tuple[int, ...]] = {}
    frontier: list[tuple[int, ...]] = [()]
    for _ in range(max_depth):
        next_frontier: list[tuple[int, ...]] = []
        for history in frontier:
            for action in ACTIONS:
                candidate = history + (action,)
                _replay(adapter, seed, "base", candidate)
                key = (adapter._position, adapter._polarity)
                if key not in reachable:
                    reachable[key] = candidate
                    next_frontier.append(candidate)
        frontier = next_frontier

    def best_action(snapshot: HiddenSnapshot, marker: str) -> tuple[int, dict[int, float]]:
        scores: dict[int, float] = {}
        for action in ACTIONS:
            adapter.restore(snapshot)
            adapter._goal_marker = marker  # evaluator-only: the goal is not part of the level
            token = adapter.gate.authorize_evaluator(action, "language-certificate")
            scores[action] = float(adapter.step(action, token).probes.values["goal_progress"])
        return max(scores, key=lambda a: scores[a]), scores

    for (position, polarity), history in sorted(reachable.items()):
        _replay(adapter, seed, "base", history)
        snapshot = adapter.snapshot()
        frame_alpha = None
        results = {}
        for marker in MARKERS:
            adapter.restore(snapshot)
            adapter._goal_marker = marker
            frame = adapter.frame().copy()
            if frame_alpha is None:
                frame_alpha = frame
            elif not np.array_equal(frame_alpha, frame):
                raise ContractViolation(
                    "the rendered frame depends on which marker is named; the goal is "
                    "visible in pixels and the language claim would be unearned"
                )
            results[marker] = best_action(snapshot, marker)
        action_a, scores_a = results["alpha"]
        action_b, scores_b = results["beta"]
        if action_a != action_b:
            return {
                "seed": seed,
                "history": list(history),
                "position": list(position),
                "polarity": polarity,
                "goal_text_alpha": GOAL_PHRASES["alpha"],
                "goal_text_beta": GOAL_PHRASES["beta"],
                "pixels_identical_under_both_instructions": True,
                "best_action_alpha": action_a,
                "best_action_beta": action_b,
                "scores_alpha": {str(k): v for k, v in scores_a.items()},
                "scores_beta": {str(k): v for k, v in scores_b.items()},
                "language_changes_correct_action": True,
                "states_searched": len(reachable),
            }
    raise ContractViolation(
        f"no reachable state at seed {seed} within depth {max_depth} has the instruction "
        "change the correct action; the task does not require language there"
    )


def build_vision_necessity_certificate(seed: int = 9000, max_depth: int = 5) -> dict[str, Any]:
    """The mirror of the language certificate: language alone is not enough.

    Same instruction, two different reachable visual states, different correct
    actions. Together with the language certificate this pins that the task needs
    both channels -- a task solvable from either one alone would make a
    multimodal claim unearned in the other direction.
    """
    adapter = ProceduralVisualV2Adapter(gate=AuthorityGate())
    adapter.reset(seed)
    marker = adapter._goal_marker
    reachable: dict[tuple[tuple[int, int], int], tuple[int, ...]] = {}
    frontier: list[tuple[int, ...]] = [()]
    for _ in range(max_depth):
        next_frontier: list[tuple[int, ...]] = []
        for history in frontier:
            for action in ACTIONS:
                candidate = history + (action,)
                _replay(adapter, seed, "base", candidate)
                key = (adapter._position, adapter._polarity)
                if key not in reachable:
                    reachable[key] = candidate
                    next_frontier.append(candidate)
        frontier = next_frontier

    def best_action(history) -> tuple[int, str]:
        _replay(adapter, seed, "base", history)
        snapshot = adapter.snapshot()
        content = adapter._observation().content_digest
        scores = {}
        for action in ACTIONS:
            adapter.restore(snapshot)
            token = adapter.gate.authorize_evaluator(action, "vision-certificate")
            scores[action] = float(adapter.step(action, token).probes.values["goal_progress"])
        return max(scores, key=lambda a: scores[a]), content

    entries = sorted(reachable.items())
    for i, (_, history_a) in enumerate(entries):
        action_a, content_a = best_action(history_a)
        for _, history_b in entries[i + 1 :]:
            action_b, content_b = best_action(history_b)
            if action_a != action_b and content_a != content_b:
                return {
                    "seed": seed,
                    "goal_text": GOAL_PHRASES[marker],
                    "identical_instruction": True,
                    "history_a": list(history_a),
                    "history_b": list(history_b),
                    "visual_state_differs": True,
                    "best_action_a": action_a,
                    "best_action_b": action_b,
                    "vision_changes_correct_action": True,
                    "states_searched": len(reachable),
                }
    raise ContractViolation(
        f"no pair of reachable visual states at seed {seed} changes the correct action "
        "under one instruction; language alone would suffice and the multimodal claim "
        "would be unearned"
    )
