"""Sealed development data: collection, feature tables, and batch sampling.

The order in this module is the load-bearing part, and it is the order the
implementation plan specifies:

1. enumerate environment family, seed, and dynamic;
2. assign the split to that key;
3. collect every branch inside the assigned split;
4. seal the episode by hashing it;
5. never divide a branch group afterwards.

Four collector policies make up the preregistered mixture. None of them needs a
trained model, which matters at Scale 0: `sentinel` is a heuristic that repeats
what moved the observable signature, and `uncertainty_seeking` is count-based
novelty. Calling either of them a Sentinel policy in the capability sense would
be an overclaim, and the report says so.

`scripted_oracle` is the one that touches hidden state. The oracle reads the
snapshot to choose a good action; what enters the record is the action and the
resulting observation, never the oracle's reasoning or its answer. That is why
the trajectory is `DEVELOPMENT` while the oracle's own output is not stored at
all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from sentinel.env.adapters.base import EnvironmentAdapter, HiddenSnapshot, StepResult
from sentinel.wm.authority import AuthorityGate
from sentinel.wm.dataset import (
    CollectorPolicy,
    EpisodeKey,
    SequenceBatch,
    Split,
    SplitManifest,
    TransitionRecord,
)
from sentinel.wm.encoder import CachedEncoder
from sentinel.wm.events import EventKind, StructuredEvent
from sentinel.wm.latent_contract import ContractViolation, Taint
from sentinel.wm.versioning import canonical_json, digest_array, digest_of


@dataclass(frozen=True, slots=True)
class CollectionPlan:
    """How many transitions to draw, from where, under which policies."""

    environment: str
    transitions: int
    mixture: Mapping[CollectorPolicy, int]
    episode_length: int = 24
    branch_every: int = 6
    branch_actions: int = 3
    dynamics: tuple[str, ...] = ("base",)

    def __post_init__(self) -> None:
        total = sum(self.mixture.values())
        if total != self.transitions:
            raise ContractViolation(
                f"mixture sums to {total} transitions but the plan asks for {self.transitions}"
            )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "transitions": self.transitions,
            "mixture": {p.value: c for p, c in sorted(self.mixture.items(), key=lambda kv: kv[0].value)},
            "episode_length": self.episode_length,
            "branch_every": self.branch_every,
            "branch_actions": self.branch_actions,
            "dynamics": list(self.dynamics),
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


def _draw(material: Mapping[str, Any], bound: int) -> int:
    """Deterministic integer draw. No global RNG anywhere in collection."""
    return int(digest_of(dict(material))[7:23], 16) % bound


class CollectorPolicyFn:
    """Picks an action. Only `scripted_oracle` may look at the snapshot."""

    def __init__(self, policy: CollectorPolicy):
        self.policy = policy
        self.seen: dict[tuple[str, int], int] = {}
        self.last_signature: float | None = None
        self.last_action: int | None = None

    @property
    def digest(self) -> str:
        return digest_of({"policy": self.policy.value, "version": 1})

    def choose(
        self,
        adapter: EnvironmentAdapter,
        result: StepResult,
        step: int,
        episode_id: str,
    ) -> tuple[int, float]:
        """Return (action, propensity). Propensity is recorded on the transition."""
        legal = result.legal_actions
        signature = result.probes.values["observable_signature"]

        if self.policy is CollectorPolicy.RANDOM:
            index = _draw({"episode": episode_id, "step": step, "policy": "random"}, len(legal))
            return legal[index], 1.0 / len(legal)

        if self.policy is CollectorPolicy.SCRIPTED_ORACLE:
            # The oracle is allowed to see hidden state. Its *answer* never
            # enters a record; only the action taken and what followed do.
            best_action, best_value = legal[0], float("-inf")
            snapshot = adapter.snapshot()
            for action in legal:
                token = adapter.gate.authorize_evaluator(action, "oracle-lookahead")
                probe = adapter.step(action, token).probes
                value = float(probe.values["goal_progress"]) + float(probe.values["reward"])
                adapter.restore(snapshot)
                if value > best_value:
                    best_action, best_value = action, value
            return best_action, 1.0

        if self.policy is CollectorPolicy.SENTINEL:
            # Repeat what moved the observable signature; otherwise rotate.
            if self.last_action is not None and self.last_signature is not None:
                if signature != self.last_signature:
                    action = self.last_action
                else:
                    action = legal[(legal.index(self.last_action) + 1) % len(legal)]
            else:
                action = legal[_draw({"episode": episode_id, "step": step, "policy": "s"}, len(legal))]
            self.last_signature = signature
            self.last_action = action
            return action, 1.0 / len(legal)

        # Uncertainty-seeking: the least-tried action from this observation.
        counts = [(self.seen.get((str(signature), a), 0), a) for a in legal]
        counts.sort()
        action = counts[0][1]
        self.seen[(str(signature), action)] = self.seen.get((str(signature), action), 0) + 1
        return action, 1.0 / len(legal)


def _events_for(before: StepResult, after: StepResult) -> tuple[StructuredEvent, ...]:
    """Observable events, each named with the probe that witnesses it."""
    events: list[StructuredEvent] = []
    if after.probes.values["action_succeeded"]:
        events.append(StructuredEvent(EventKind.ACTION_SUCCEEDED, witness="action_succeeded"))
    else:
        events.append(StructuredEvent(EventKind.ACTION_FAILED, witness="action_succeeded"))
    if after.probes.values["constraint_violation"]:
        events.append(StructuredEvent(EventKind.CONSTRAINT_VIOLATED, witness="constraint_violation"))
    if after.probes.values["goal_progress"] != before.probes.values["goal_progress"]:
        events.append(StructuredEvent(EventKind.GOAL_PROGRESS_CHANGED, witness="goal_progress"))
    if after.probes.values["observable_signature"] != before.probes.values["observable_signature"]:
        events.append(StructuredEvent(EventKind.FOCUS_MOVED, witness="observable_signature"))
    return tuple(events)


def primary_event_index(events: Sequence[StructuredEvent]) -> int:
    """One index per transition for the event head. Ordered by informativeness.

    A constraint violation outranks a movement because the verifier cares more
    about it, and a transition with nothing observable becomes `UNKNOWN_EVENT`
    rather than being dropped -- an unlabelled transition would silently shrink
    the batch.
    """
    from sentinel.wm.events import EVENT_INDEX

    priority = (
        EventKind.CONSTRAINT_VIOLATED,
        EventKind.ACTION_FAILED,
        EventKind.GOAL_PROGRESS_CHANGED,
        EventKind.FOCUS_MOVED,
        EventKind.ACTION_SUCCEEDED,
    )
    present = {e.kind for e in events}
    for kind in priority:
        if kind in present:
            return EVENT_INDEX[kind]
    return EVENT_INDEX[EventKind.UNKNOWN_EVENT]


@dataclass
class CollectionResult:
    records: list[TransitionRecord] = field(default_factory=list)
    features: dict[str, np.ndarray] = field(default_factory=dict)
    episodes: int = 0
    branch_groups: int = 0
    oracle_lookaheads: int = 0
    encoder_calls: int = 0
    environment_interactions: int = 0
    """Steps actually taken during collection, including oracle lookahead.

    Separate from a training run's online interactions, which the matrix fixes
    at exactly zero. Collection happens once, before any cell runs; conflating
    the two would make every arm look as though it had been allowed to explore.
    """

    @property
    def transition_ids_digest(self) -> str:
        """Identity of the raw data every arm must share."""
        return digest_of([r.transition_id for r in self.records])

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "transitions": len(self.records),
            "episodes": self.episodes,
            "branch_groups": self.branch_groups,
            "oracle_lookaheads": self.oracle_lookaheads,
            "encoder_calls": self.encoder_calls,
            "environment_interactions": self.environment_interactions,
            "distinct_observations": len(self.features),
            "transition_ids_digest": self.transition_ids_digest,
        }


def collect(
    adapter_factory: Callable[[AuthorityGate], EnvironmentAdapter],
    plan: CollectionPlan,
    manifest: SplitManifest,
    encoder: CachedEncoder,
    *,
    family: str,
    first_seed: int = 0,
) -> CollectionResult:
    """Collect exactly `plan.transitions` transitions under the frozen order."""
    result = CollectionResult()
    gate = AuthorityGate(gate_id=f"collection:{family}")
    adapter = adapter_factory(gate)

    remaining = {policy: count for policy, count in plan.mixture.items() if count > 0}
    seed = first_seed
    dynamic_index = 0

    while remaining:
        policy = max(remaining, key=lambda p: (remaining[p], p.value))
        key = EpisodeKey(family=family, seed=seed, dynamic=plan.dynamics[dynamic_index % len(plan.dynamics)])
        split = manifest.assign(key)
        seed += 1
        dynamic_index += 1

        chooser = CollectorPolicyFn(policy)
        current = adapter.reset(key.seed, key.dynamic)
        episode_id = f"{family}:{key.dynamic}:{key.seed}:{policy.value}"
        result.episodes += 1

        for step in range(plan.episode_length):
            if remaining.get(policy, 0) <= 0:
                break
            features_before = encoder.encode_array(current.observation)
            result.encoder_calls += 1
            result.features[current.observation.content_digest] = features_before
            latent_before = digest_array(features_before).digest

            branching = step > 0 and step % plan.branch_every == 0 and remaining[policy] >= plan.branch_actions
            if branching:
                snapshot = adapter.snapshot()
                group = digest_of({"episode": episode_id, "restore_step": step})
                result.branch_groups += 1
                legal = current.legal_actions[: plan.branch_actions]
                for branch_index, action in enumerate(legal):
                    if remaining.get(policy, 0) <= 0:
                        break
                    token = gate.authorize_collection(action, policy.value)
                    after = adapter.step(action, token)
                    result.records.append(
                        _record(
                            key, episode_id, step, action, 1.0 / len(current.legal_actions),
                            policy, chooser.digest, current, after, encoder,
                            latent_before, group, branch_index, result.features,
                        )
                    )
                    result.encoder_calls += 1
                    remaining[policy] -= 1
                    adapter.restore(snapshot)
                current = adapter.restore(snapshot)
                if remaining.get(policy, 0) <= 0:
                    break

            action, propensity = chooser.choose(adapter, current, step, episode_id)
            if policy is CollectorPolicy.SCRIPTED_ORACLE:
                result.oracle_lookaheads += len(current.legal_actions)
            token = gate.authorize_collection(action, policy.value)
            after = adapter.step(action, token)
            result.records.append(
                _record(
                    key, episode_id, step, action, propensity, policy, chooser.digest,
                    current, after, encoder, latent_before, None, 0, result.features,
                )
            )
            result.encoder_calls += 1
            remaining[policy] -= 1
            current = after
            if after.terminated:
                break

        if remaining.get(policy, 0) <= 0:
            remaining.pop(policy, None)

    result.environment_interactions = adapter.interactions
    return result


def _record(
    key: EpisodeKey,
    episode_id: str,
    step: int,
    action: int,
    propensity: float,
    policy: CollectorPolicy,
    policy_digest: str,
    before: StepResult,
    after: StepResult,
    encoder: CachedEncoder,
    latent_before: str,
    branch_group_id: str | None,
    branch_index: int,
    feature_store: dict[str, np.ndarray],
) -> TransitionRecord:
    features_after = encoder.encode_array(after.observation)
    feature_store[after.observation.content_digest] = features_after
    return TransitionRecord(
        episode_key=key,
        episode_id=episode_id,
        step=step,
        observation_digest_t=before.observation.digest,
        content_digest_t=before.observation.content_digest,
        latent_digest_t=latent_before,
        action=action,
        action_propensity=propensity,
        collector_policy=policy,
        collector_policy_digest=policy_digest,
        reward=float(after.reward),
        termination=bool(after.terminated),
        observation_digest_t1=after.observation.digest,
        content_digest_t1=after.observation.content_digest,
        latent_digest_t1=digest_array(features_after).digest,
        structured_events=_events_for(before, after),
        probes_t1=dict(after.probes.canonical_dict()),
        branch_group_id=branch_group_id,
        branch_index=branch_index,
        taint=frozenset({Taint.DEVELOPMENT}),
    )


# ---- feature table -----------------------------------------------------------


@dataclass
class FeatureTable:
    """One packed array of frozen features, addressed by observation digest.

    The cache is the store of record; this is the read path. Two hundred
    thousand individual cache reads per run would make the throughput report a
    measurement of the filesystem, so the features are packed once into a single
    array whose digest is recorded, and the training loop indexes into it.
    """

    rows: dict[str, int]
    values: np.ndarray
    packed: bool

    @classmethod
    def from_mapping(cls, features: Mapping[str, np.ndarray]) -> "FeatureTable":
        digests = sorted(features)
        if not digests:
            raise ContractViolation("cannot build a feature table from no observations")
        width = int(np.asarray(features[digests[0]]).shape[-1])
        values = np.zeros((len(digests), width), dtype=np.float32)
        rows: dict[str, int] = {}
        for index, digest in enumerate(digests):
            rows[digest] = index
            values[index] = np.asarray(features[digest], dtype=np.float32)
        return cls(rows=rows, values=values, packed=False)

    def lookup(self, digests: Sequence[str]) -> np.ndarray:
        return self.values[[self.rows[d] for d in digests]]

    @property
    def digest(self) -> str:
        return digest_array(self.values).digest

    def size_report(self) -> dict[str, Any]:
        return {
            "rows": int(self.values.shape[0]),
            "width": int(self.values.shape[1]),
            "bytes": int(self.values.nbytes),
            "digest": self.digest,
        }


# ---- batch sampling ----------------------------------------------------------


@dataclass
class SequenceSampler:
    """Seeded, restartable sampling of fixed-shape sequence batches.

    The permutation is a function of the seed alone, and the cursor is an
    integer, so a restart resumes on the same batch rather than on a batch that
    merely looks similar.
    """

    sequences: list[SequenceBatch]
    seed: int
    batch_size: int
    cursor: int = 0
    _order: list[int] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_records(
        cls,
        records: Sequence[TransitionRecord],
        manifest: SplitManifest,
        *,
        split: Split,
        sequence_length: int,
        batch_size: int,
        seed: int,
    ) -> "SequenceSampler":
        by_episode: dict[str, list[TransitionRecord]] = {}
        for record in records:
            if manifest.split_of(record.episode_key) is not split:
                continue
            by_episode.setdefault(record.episode_id, []).append(record)

        sequences: list[SequenceBatch] = []
        for episode_id in sorted(by_episode):
            ordered = sorted(by_episode[episode_id], key=lambda r: (r.step, r.branch_index))
            for start in range(0, len(ordered) - sequence_length + 1, sequence_length):
                window = ordered[start : start + sequence_length]
                sequences.append(
                    SequenceBatch(records=tuple(window), split=split)
                )
        if not sequences:
            raise ContractViolation(
                f"no sequence of length {sequence_length} exists in split {split.value}; "
                "either the episodes are too short or the split is empty"
            )
        return cls(sequences=sequences, seed=seed, batch_size=batch_size)

    @property
    def permutation(self) -> list[int]:
        """Seed-determined order, computed once.

        Recomputing it costs one hash per sequence, and `batch` reads it on
        every optimizer update -- which put a few thousand hashes per update
        into a throughput number that is supposed to be measuring the model.
        """
        if self._order is None:
            self._order = sorted(
                range(len(self.sequences)),
                key=lambda i: digest_of({"seed": self.seed, "index": i}),
            )
        return self._order

    @property
    def permutation_digest(self) -> str:
        return digest_of({"seed": self.seed, "order": self.permutation})

    def batch(self, index: int) -> list[SequenceBatch]:
        order = self.permutation
        start = (index * self.batch_size) % len(order)
        chosen = [order[(start + offset) % len(order)] for offset in range(self.batch_size)]
        return [self.sequences[i] for i in chosen]


@dataclass(frozen=True, slots=True)
class SequenceBatchArrays:
    features: np.ndarray
    actions: np.ndarray
    previous_rewards: np.ndarray
    rewards: np.ndarray
    terminations: np.ndarray
    event_targets: np.ndarray
    boundary_pairs: tuple[tuple[int, int, int, int], ...]


def materialise(batch: Sequence[SequenceBatch], table: FeatureTable) -> SequenceBatchArrays:
    """Turn a list of sequences into the arrays the objective consumes."""
    batch_size = len(batch)
    time_steps = len(batch[0])
    width = table.values.shape[1]

    features = np.zeros((batch_size, time_steps, width), dtype=np.float32)
    actions = np.zeros((batch_size, time_steps), dtype=np.int32)
    previous_rewards = np.zeros((batch_size, time_steps, 1), dtype=np.float32)
    rewards = np.zeros((batch_size, time_steps), dtype=np.float32)
    terminations = np.zeros((batch_size, time_steps), dtype=np.float32)
    events = np.zeros((batch_size, time_steps), dtype=np.int32)

    by_group: dict[str, list[tuple[int, int]]] = {}
    for row, sequence in enumerate(batch):
        if len(sequence) != time_steps:
            raise ContractViolation("ragged batch: every sequence must have the same length")
        features[row] = table.lookup([r.content_digest_t for r in sequence.records])
        for column, record in enumerate(sequence.records):
            actions[row, column] = record.action
            rewards[row, column] = record.reward
            terminations[row, column] = 1.0 if record.termination else 0.0
            events[row, column] = primary_event_index(record.structured_events)
            if column > 0:
                previous_rewards[row, column, 0] = sequence.records[column - 1].reward
            if record.branch_group_id is not None:
                by_group.setdefault(record.branch_group_id, []).append((row, column))

    pairs: list[tuple[int, int, int, int]] = []
    for positions in by_group.values():
        for i in range(len(positions) - 1):
            (row_a, column_a), (row_b, column_b) = positions[i], positions[i + 1]
            pairs.append((row_a, column_a, row_b, column_b))

    return SequenceBatchArrays(
        features=features,
        actions=actions,
        previous_rewards=previous_rewards,
        rewards=rewards,
        terminations=terminations,
        event_targets=events,
        boundary_pairs=tuple(pairs),
    )
