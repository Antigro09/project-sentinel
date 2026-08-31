"""Adapter A: a deterministic controlled environment with two separate fixtures.

The independent reader audit that produced this branch caught a conflation
between two questions that look alike and are not. This adapter exists to keep
them apart, and it builds one fixture for each.

**T1a, action intervention.** Restore the *identical* full simulator state, force
two different legal actions, and require different observable successors. What
fails here is an action-blind model. Nothing about history is involved.

**T1b, belief aliasing.** Construct the *same* current observation from two
different hidden histories, apply the *same* action, and require different
observable successors. What fails here is an observation-only model. Nothing
about action conditioning is involved, and a recurrent belief can represent the
distinction because the two histories differ in their earlier observations.

The dynamics are chosen so both fixtures exist and neither needs a special case:

    visible' = (visible + 1 + action + phase) mod 8
    phase'   = (phase + 1) mod 3   if action == PHASE_ACTION else phase

`phase` is never rendered. It is recoverable from the action history, which the
model does receive, so the aliasing fixture tests whether recurrence is *used*
rather than whether the information was withheld.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

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
from sentinel.wm.versioning import digest_of

VISIBLE_STATES = 8
PHASES = 3
ACTIONS: tuple[int, ...] = (0, 1, 2, 3)
PHASE_ACTION = 3
"""The only action that advances the hidden phase. Everything else leaves it."""

DEFAULT_HORIZON = 24
FORBIDDEN_VISIBLE = 7
"""A visible value the agent is constrained not to enter. Gives the verifier a
constraint probe with a real referent rather than a placeholder."""

GENERATOR_VERSION = "synthetic-control-v1"

MASK = ModalityMask(
    declared=(Modality.STRUCTURED, Modality.GOAL),
    present=(Modality.STRUCTURED, Modality.GOAL),
)


def _dynamic_offset(dynamic: str) -> int:
    """A named mechanics variant. Appearance is unchanged; the rule shifts."""
    if dynamic == "base":
        return 0
    return int(digest_of({"dynamic": dynamic})[8:12], 16) % VISIBLE_STATES


@dataclass
class SyntheticControlAdapter:
    """Tiny, deterministic, fully replayable.

    Exact replay is the point: the Phase-1 verifier's guarantee is that the same
    actions from the same state produce the same observations, and a Phase-2
    adapter that cannot promise that cannot be used to falsify anything.
    """

    gate: AuthorityGate
    horizon: int = DEFAULT_HORIZON
    _visible: int = field(default=0, init=False)
    _phase: int = field(default=0, init=False)
    _step: int = field(default=0, init=False)
    _goal: int = field(default=0, init=False)
    _seed: int = field(default=0, init=False)
    _dynamic: str = field(default="base", init=False)
    _last_action: int | None = field(default=None, init=False)
    _last_blocked: bool = field(default=False, init=False)
    _interactions: int = field(default=0, init=False)

    # ---- identity ------------------------------------------------------

    @property
    def identity(self) -> EnvironmentIdentity:
        return EnvironmentIdentity(
            name="synthetic_control",
            version=GENERATOR_VERSION,
            generator_digest=digest_of(
                {
                    "visible_states": VISIBLE_STATES,
                    "phases": PHASES,
                    "actions": list(ACTIONS),
                    "phase_action": PHASE_ACTION,
                    "forbidden_visible": FORBIDDEN_VISIBLE,
                    "version": GENERATOR_VERSION,
                }
            ),
            supports_branching=True,
        )

    @property
    def interactions(self) -> int:
        return self._interactions

    # ---- pure dynamics -------------------------------------------------

    def _successor(self, visible: int, phase: int, action: int, dynamic: str) -> tuple[int, int]:
        offset = _dynamic_offset(dynamic)
        next_visible = (visible + 1 + action + phase + offset) % VISIBLE_STATES
        next_phase = (phase + 1) % PHASES if action == PHASE_ACTION else phase
        return next_visible, next_phase

    # ---- observation ---------------------------------------------------

    def _observation(self) -> ObservationEnvelope:
        return ObservationEnvelope(
            episode_id=self.episode_id,
            step=self._step,
            timestamp_ns=self._step,
            modality_payloads={},
            structured_observation={
                "visible": self._visible,
                "goal": self._goal,
                "steps_remaining": self.horizon - self._step,
                "forbidden": FORBIDDEN_VISIBLE,
            },
            modality_mask=MASK,
            available_action_digest=digest_of(list(ACTIONS)),
            environment_version=self.identity.digest,
            taint=frozenset({Taint.DEVELOPMENT}),
        )

    @property
    def episode_id(self) -> str:
        return f"synthetic_control:{self._dynamic}:{self._seed}"

    def probes(self) -> ProbeSet:
        return ProbeSet(
            {
                "reward": 1.0 if self._visible == self._goal else 0.0,
                "termination": self._step >= self.horizon,
                "goal_progress": float(
                    1.0 - min(abs(self._visible - self._goal), VISIBLE_STATES - abs(self._visible - self._goal))
                    / (VISIBLE_STATES // 2)
                ),
                "constraint_violation": self._visible == FORBIDDEN_VISIBLE,
                "action_succeeded": not self._last_blocked,
                "observable_signature": int(self._visible),
            }
        )

    def legal_actions(self) -> tuple[int, ...]:
        return ACTIONS

    # ---- lifecycle -----------------------------------------------------

    def reset(self, seed: int, dynamic: str = "base") -> StepResult:
        self._seed = int(seed)
        self._dynamic = dynamic
        draw = int(digest_of({"seed": int(seed), "dynamic": dynamic})[7:19], 16)
        self._visible = draw % VISIBLE_STATES
        self._phase = (draw // VISIBLE_STATES) % PHASES
        self._goal = (draw // (VISIBLE_STATES * PHASES)) % VISIBLE_STATES
        self._step = 0
        self._last_action = None
        self._last_blocked = False
        return self._result()

    def _result(self) -> StepResult:
        return StepResult(
            observation=self._observation(),
            reward=float(self.probes().values["reward"]),
            terminated=bool(self.probes().values["termination"]),
            probes=self.probes(),
            legal_actions=self.legal_actions(),
            info={"dynamic": self._dynamic},
        )

    def step(self, action: int, token: AuthorizationToken) -> StepResult:
        self.gate.consume(token, action)
        if action not in ACTIONS:
            raise ContractViolation(f"action {action} is not legal here; legal actions {ACTIONS}")
        proposed, next_phase = self._successor(self._visible, self._phase, action, self._dynamic)
        # The constraint is enforced by the environment, not by the model: an
        # action into the forbidden cell is refused and recorded as a failure,
        # which gives `action_succeeded` and `constraint_violation` distinct
        # meanings instead of one being a rename of the other.
        if proposed == FORBIDDEN_VISIBLE:
            self._last_blocked = True
        else:
            self._visible = proposed
            self._phase = next_phase
            self._last_blocked = False
        self._step += 1
        self._last_action = action
        self._interactions += 1
        return self._result()

    # ---- branching -----------------------------------------------------

    def snapshot(self) -> HiddenSnapshot:
        return HiddenSnapshot(
            payload={
                "visible": self._visible,
                "phase": self._phase,
                "step": self._step,
                "goal": self._goal,
                "seed": self._seed,
                "dynamic": self._dynamic,
                "last_blocked": self._last_blocked,
                "_non_observable": ("phase",),
            },
            environment_version=self.identity.digest,
        )

    def restore(self, snapshot: HiddenSnapshot) -> StepResult:
        payload = snapshot.reveal("restore")
        self._visible = int(payload["visible"])
        self._phase = int(payload["phase"])
        self._step = int(payload["step"])
        self._goal = int(payload["goal"])
        self._seed = int(payload["seed"])
        self._dynamic = str(payload["dynamic"])
        self._last_blocked = bool(payload["last_blocked"])
        return self._result()


# ---- the two fixtures -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActionInterventionFixture:
    """T1a. One restored state, two forced actions, two different successors."""

    seed: int
    dynamic: str
    restore_point: HiddenSnapshot
    action_a: int
    action_b: int
    observation: ObservationEnvelope
    successor_a: ProbeSet
    successor_b: ProbeSet

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "kind": "action_intervention",
            "seed": self.seed,
            "dynamic": self.dynamic,
            "restore_point": self.restore_point.canonical_dict(),
            "action_a": self.action_a,
            "action_b": self.action_b,
            "observation_digest": self.observation.digest,
            "successor_a": self.successor_a.canonical_dict(),
            "successor_b": self.successor_b.canonical_dict(),
        }


@dataclass(frozen=True, slots=True)
class BeliefAliasFixture:
    """T1b. Two histories, one shared observation, one action, two successors."""

    seed: int
    dynamic: str
    history_a: tuple[int, ...]
    history_b: tuple[int, ...]
    shared_observation: ObservationEnvelope
    observation_trace_a: tuple[int, ...]
    observation_trace_b: tuple[int, ...]
    probe_action: int
    successor_a: ProbeSet
    successor_b: ProbeSet

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "kind": "belief_alias",
            "seed": self.seed,
            "dynamic": self.dynamic,
            "history_a": list(self.history_a),
            "history_b": list(self.history_b),
            "shared_observation_digest": self.shared_observation.digest,
            "observation_trace_a": list(self.observation_trace_a),
            "observation_trace_b": list(self.observation_trace_b),
            "probe_action": self.probe_action,
            "successor_a": self.successor_a.canonical_dict(),
            "successor_b": self.successor_b.canonical_dict(),
        }


def _rollout(
    adapter: SyntheticControlAdapter, seed: int, dynamic: str, actions: Iterable[int]
) -> tuple[list[int], StepResult]:
    """Replay a history under evaluator authority and return the visible trace."""
    result = adapter.reset(seed, dynamic)
    trace = [int(result.observation.structured_observation["visible"])]
    for action in actions:
        token = adapter.gate.authorize_evaluator(action, "fixture-construction")
        result = adapter.step(action, token)
        trace.append(int(result.observation.structured_observation["visible"]))
    return trace, result


def build_action_intervention_fixture(
    seed: int = 6600, dynamic: str = "base", prefix: tuple[int, ...] = (0, 1)
) -> ActionInterventionFixture:
    """Search for two legal actions whose observable successors differ.

    A search rather than a hard-coded pair, so the fixture survives a change to
    the dynamics: if no such pair exists the environment is action-blind and the
    fixture refuses to be built, which is the honest failure.
    """
    adapter = SyntheticControlAdapter(gate=AuthorityGate())
    _rollout(adapter, seed, dynamic, prefix)
    restore_point = adapter.snapshot()
    observation = adapter._observation()

    outcomes: dict[int, ProbeSet] = {}
    for action in ACTIONS:
        adapter.restore(restore_point)
        token = adapter.gate.authorize_evaluator(action, "fixture-construction")
        outcomes[action] = adapter.step(action, token).probes

    for i, action_a in enumerate(ACTIONS):
        for action_b in ACTIONS[i + 1 :]:
            if outcomes[action_a].digest != outcomes[action_b].digest:
                return ActionInterventionFixture(
                    seed=seed,
                    dynamic=dynamic,
                    restore_point=restore_point,
                    action_a=action_a,
                    action_b=action_b,
                    observation=observation,
                    successor_a=outcomes[action_a],
                    successor_b=outcomes[action_b],
                )
    raise ContractViolation(
        "no two legal actions produce different observable successors from this state; "
        "the environment cannot support an action-intervention fixture"
    )


def build_belief_alias_fixture(
    seed: int = 6600, dynamic: str = "base", history_length: int = 2
) -> BeliefAliasFixture:
    """Search for two equal-length histories that alias in the observation.

    Equal length matters. If one history were shorter, a model could separate
    the two by counting steps rather than by remembering what it saw, and the
    fixture would pass for the wrong reason.
    """
    adapter = SyntheticControlAdapter(gate=AuthorityGate())

    def enumerate_histories(length: int):
        if length == 0:
            yield ()
            return
        for head in enumerate_histories(length - 1):
            for action in ACTIONS:
                yield head + (action,)

    seen: dict[tuple[int, ...], list[tuple[tuple[int, ...], list[int], HiddenSnapshot]]] = {}
    for history in enumerate_histories(history_length):
        trace, _ = _rollout(adapter, seed, dynamic, history)
        snapshot = adapter.snapshot()
        observation_key = (
            int(adapter._observation().structured_observation["visible"]),
            int(adapter._step),
        )
        seen.setdefault(observation_key, []).append((history, trace, snapshot))

    for _observation_key, group in sorted(seen.items()):
        for i, (history_a, trace_a, snap_a) in enumerate(group):
            for history_b, trace_b, snap_b in group[i + 1 :]:
                phase_a = snap_a.reveal("fixture-builder")["phase"]
                phase_b = snap_b.reveal("fixture-builder")["phase"]
                if phase_a == phase_b or trace_a == trace_b:
                    continue
                for probe_action in ACTIONS:
                    adapter.restore(snap_a)
                    shared_observation = adapter._observation()
                    token = adapter.gate.authorize_evaluator(probe_action, "fixture-construction")
                    outcome_a = adapter.step(probe_action, token).probes
                    adapter.restore(snap_b)
                    token = adapter.gate.authorize_evaluator(probe_action, "fixture-construction")
                    outcome_b = adapter.step(probe_action, token).probes
                    if outcome_a.digest != outcome_b.digest:
                        return BeliefAliasFixture(
                            seed=seed,
                            dynamic=dynamic,
                            history_a=history_a,
                            history_b=history_b,
                            shared_observation=shared_observation,
                            observation_trace_a=tuple(trace_a),
                            observation_trace_b=tuple(trace_b),
                            probe_action=probe_action,
                            successor_a=outcome_a,
                            successor_b=outcome_b,
                        )
    raise ContractViolation(
        f"no aliasing pair of length {history_length} exists for seed {seed}; "
        "the environment cannot support a belief-aliasing fixture at this depth"
    )
