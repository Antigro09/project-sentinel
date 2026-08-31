"""Transition and sequence schemas, branch groups, splits, and leak audits.

The dataset is where the two most expensive Phase-2 mistakes would be made, and
both are ordering mistakes rather than modelling mistakes.

**Branch siblings must not cross splits.** A branch group is several actions
taken from one restored simulator state. If the group is collected first and
split afterwards, one sibling in train tells the model exactly what the held-out
sibling does, and the intervention audit measures memorisation. So the split is
assigned to the *episode key* -- family, seed, dynamic -- before a single step
is collected, and every branch inherits it.

**Evaluator fields must not ride along.** Hidden mechanics, expected successors,
and oracle answers are useful for scoring and fatal in a training record. They
are rejected by field name and by taint at construction time.

`collector_propensity` is recorded on every transition because Theorem 1 makes
the collection policy part of what a later causal claim depends on: a dataset
whose behaviour policy never tried an action cannot identify what it does, and
that fact has to be visible in the data rather than remembered by the author.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from sentinel.wm.events import StructuredEvent
from sentinel.wm.latent_contract import (
    ContractViolation,
    TRAIN_FORBIDDEN_TAINTS,
    Taint,
    reject_hidden_fields,
)
from sentinel.wm.versioning import digest_of, require_digest


class Split(str, Enum):
    TRAIN = "train"
    DEV_HELD_OUT = "dev_held_out"
    VALIDATION = "validation"
    FINAL = "final"


class CollectorPolicy(str, Enum):
    """The four collection strata of the preregistered mixture."""

    RANDOM = "random"
    SCRIPTED_ORACLE = "scripted_oracle"
    SENTINEL = "sentinel"
    UNCERTAINTY_SEEKING = "uncertainty_seeking"


COLLECTION_MIXTURE: Mapping[CollectorPolicy, float] = {
    CollectorPolicy.RANDOM: 0.30,
    CollectorPolicy.SCRIPTED_ORACLE: 0.25,
    CollectorPolicy.SENTINEL: 0.25,
    CollectorPolicy.UNCERTAINTY_SEEKING: 0.20,
}
"""Preregistered starting values from the strategy document. To be ablated."""


@dataclass(frozen=True, slots=True)
class EpisodeKey:
    """The unit that receives a split, before any data exists.

    A `dynamic` is a named variation of the environment's mechanics. It is part
    of the key because holding out a mechanic is one of the independent holdout
    axes, and a mechanic that appears in two splits is not held out.
    """

    family: str
    seed: int
    dynamic: str

    def __post_init__(self) -> None:
        if not self.family or not self.dynamic:
            raise ContractViolation(f"EpisodeKey has an empty component: {self}")

    def canonical_dict(self) -> dict[str, Any]:
        return {"family": self.family, "seed": int(self.seed), "dynamic": self.dynamic}

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """One `(observation, action, outcome, next observation)` tuple, sealed.

    Digests rather than payloads: the record is small enough to keep a hundred
    thousand of them in memory, and every array it refers to is recoverable from
    the content-addressed cache.
    """

    episode_key: EpisodeKey
    episode_id: str
    step: int
    observation_digest_t: str
    content_digest_t: str
    latent_digest_t: str
    action: int
    action_propensity: float
    collector_policy: CollectorPolicy
    collector_policy_digest: str
    reward: float
    termination: bool
    observation_digest_t1: str
    content_digest_t1: str
    latent_digest_t1: str
    structured_events: tuple[StructuredEvent, ...] = ()
    probes_t1: Mapping[str, Any] = field(default_factory=dict)
    """The exact observables the environment reported after the action.

    Recorded so that the verifier bridge can be exercised entirely offline. The
    matrix fixes online interactions at exactly zero per run, and a verifier
    that had to step the environment to obtain a probe value would be spending
    interactions the budget does not have.

    These are observables, not hidden state: they are the same quantities the
    event head predicts, so storing them adds no channel that training did not
    already have.
    """
    branch_group_id: str | None = None
    branch_index: int = 0
    taint: frozenset[Taint] = frozenset({Taint.DEVELOPMENT})
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "observation_digest_t",
            "content_digest_t",
            "latent_digest_t",
            "observation_digest_t1",
            "content_digest_t1",
            "latent_digest_t1",
            "collector_policy_digest",
        ):
            require_digest(getattr(self, name), f"TransitionRecord.{name}")
        if not 0.0 < self.action_propensity <= 1.0:
            raise ContractViolation(
                f"action_propensity must lie in (0,1], got {self.action_propensity}; "
                "a zero-propensity action was never actually collectable"
            )
        if self.step < 0:
            raise ContractViolation(f"step must be non-negative, got {self.step}")
        reject_hidden_fields(self.extra, "TransitionRecord.extra")
        reject_hidden_fields(self.probes_t1, "TransitionRecord.probes_t1")

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "episode_key": self.episode_key.canonical_dict(),
            "episode_id": self.episode_id,
            "step": self.step,
            "observation_digest_t": self.observation_digest_t,
            "content_digest_t": self.content_digest_t,
            "latent_digest_t": self.latent_digest_t,
            "action": int(self.action),
            "action_propensity": float(self.action_propensity),
            "collector_policy": self.collector_policy.value,
            "collector_policy_digest": self.collector_policy_digest,
            "reward": float(self.reward),
            "termination": bool(self.termination),
            "observation_digest_t1": self.observation_digest_t1,
            "content_digest_t1": self.content_digest_t1,
            "latent_digest_t1": self.latent_digest_t1,
            "structured_events": [e.canonical_dict() for e in self.structured_events],
            "probes_t1": {k: self.probes_t1[k] for k in sorted(self.probes_t1)},
            "branch_group_id": self.branch_group_id,
            "branch_index": int(self.branch_index),
            "taint": sorted(t.value for t in self.taint),
            "extra": dict(self.extra),
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())

    @property
    def transition_id(self) -> str:
        """Stable identity used to prove all arms saw the same raw transitions."""
        return digest_of(
            {
                "episode_id": self.episode_id,
                "step": self.step,
                "action": int(self.action),
                "observation_digest_t": self.observation_digest_t,
            }
        )


class LeakageError(RuntimeError):
    """A split or taint invariant was violated. Never downgraded to a warning."""


@dataclass
class SplitManifest:
    """Split assignment for episode keys, fixed before collection begins.

    `assign` is deterministic in the key and the salt, so the manifest can be
    rebuilt from the seed list alone, and a key that has already been assigned
    cannot be reassigned to a friendlier split later.
    """

    salt: str
    weights: Mapping[Split, float]
    assignments: dict[str, Split] = field(default_factory=dict)
    keys: dict[str, dict[str, Any]] = field(default_factory=dict)
    sealed: bool = False

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ContractViolation(f"split weights sum to {total}, not 1")

    def assign(self, key: EpisodeKey) -> Split:
        digest = key.digest
        if digest in self.assignments:
            return self.assignments[digest]
        if self.sealed:
            raise LeakageError(
                f"manifest is sealed; {key} was not assigned before collection began"
            )
        # Deterministic uniform draw from the key digest, so the assignment is a
        # function of the key rather than of collection order.
        bucket = int(digest_of({"salt": self.salt, "key": key.canonical_dict()})[7:23], 16)
        position = (bucket % 1_000_000) / 1_000_000
        cumulative = 0.0
        chosen = list(self.weights)[-1]
        for split, weight in self.weights.items():
            cumulative += weight
            if position < cumulative:
                chosen = split
                break
        self.assignments[digest] = chosen
        self.keys[digest] = key.canonical_dict()
        return chosen

    def split_of(self, key: EpisodeKey) -> Split:
        digest = key.digest
        if digest not in self.assignments:
            raise LeakageError(
                f"{key} has no split assignment; splits are assigned before "
                "collection, never inferred from collected data"
            )
        return self.assignments[digest]

    def seal(self) -> None:
        self.sealed = True

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "salt": self.salt,
            "weights": {s.value: float(w) for s, w in self.weights.items()},
            "assignments": {k: v.value for k, v in sorted(self.assignments.items())},
            "sealed": self.sealed,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    """A contiguous window of transitions from one episode.

    Sequences never span episodes. A batch that stitches two episodes together
    teaches the recurrent state that an episode boundary is an ordinary step,
    which is exactly the distinction the belief-aliasing fixture tests.
    """

    records: tuple[TransitionRecord, ...]
    split: Split

    def __post_init__(self) -> None:
        if not self.records:
            raise ContractViolation("SequenceBatch is empty")
        episodes = {r.episode_id for r in self.records}
        if len(episodes) != 1:
            raise ContractViolation(f"sequence spans {len(episodes)} episodes: {sorted(episodes)}")
        steps = [r.step for r in self.records]
        if steps != sorted(steps):
            raise ContractViolation(f"sequence is not in step order: {steps}")

    def __len__(self) -> int:
        return len(self.records)


def assert_trainable(records: Iterable[TransitionRecord]) -> None:
    """Refuse a training batch carrying a taint training is not allowed to see."""
    for record in records:
        forbidden = record.taint & TRAIN_FORBIDDEN_TAINTS
        if forbidden:
            raise LeakageError(
                f"transition {record.transition_id[:16]}... carries "
                f"{sorted(t.value for t in forbidden)}; it may not enter a training step"
            )


def audit_splits(
    records: Sequence[TransitionRecord],
    manifest: SplitManifest,
) -> dict[str, Any]:
    """Structural violations raise; content overlap is measured.

    The distinction is not a softening. A structural violation means the split
    procedure itself was broken and no amount of environment design can excuse
    it: a branch group divided between splits, a branch group spanning episodes,
    an environment seed appearing in two splits, or one episode step landing in
    two splits at once. Each of those is a bug in the ordering, and each raises.

    Content overlap is different. The controlled adapter reaches a small set of
    observations, so two splits inevitably share some of them and some complete
    `(observation, action, successor)` tuples as well. Raising there would be a
    check satisfiable only by enlarging the environment, so the overlap is
    counted and returned. A family that *can* be disjoint -- a generated visual
    level, where a shared tuple really would be the answer in the training set --
    asserts it explicitly with `assert_no_transition_overlap`.
    """
    branch_splits: dict[str, set[Split]] = defaultdict(set)
    branch_episodes: dict[str, set[str]] = defaultdict(set)
    content_splits: dict[str, set[Split]] = defaultdict(set)
    latent_splits: dict[str, set[Split]] = defaultdict(set)
    tuple_splits: dict[tuple[str, int, str], set[Split]] = defaultdict(set)
    positional_splits: dict[str, set[Split]] = defaultdict(set)
    seed_splits: dict[tuple[str, int], set[Split]] = defaultdict(set)
    per_split: Counter[Split] = Counter()

    for record in records:
        split = manifest.split_of(record.episode_key)
        per_split[split] += 1
        if record.branch_group_id is not None:
            branch_splits[record.branch_group_id].add(split)
            branch_episodes[record.branch_group_id].add(record.episode_id)
        for digest in (record.content_digest_t, record.content_digest_t1):
            content_splits[digest].add(split)
        for digest in (record.observation_digest_t, record.observation_digest_t1):
            positional_splits[digest].add(split)
        for digest in (record.latent_digest_t, record.latent_digest_t1):
            latent_splits[digest].add(split)
        tuple_splits[
            (record.content_digest_t, int(record.action), record.content_digest_t1)
        ].add(split)
        seed_splits[(record.episode_key.family, record.episode_key.seed)].add(split)

    crossing_branches = sorted(g for g, s in branch_splits.items() if len(s) > 1)
    if crossing_branches:
        raise LeakageError(
            f"{len(crossing_branches)} branch group(s) span more than one split, "
            f"first: {crossing_branches[0]}; branch siblings must share a split"
        )
    split_branches = sorted(g for g, e in branch_episodes.items() if len(e) > 1)
    if split_branches:
        raise LeakageError(
            f"branch group(s) span more than one episode: {split_branches[:3]}"
        )
    reused_seeds = sorted(k for k, s in seed_splits.items() if len(s) > 1)
    if reused_seeds:
        raise LeakageError(f"environment seed(s) reused across splits: {reused_seeds[:3]}")
    shared_positional = sorted(d for d, s in positional_splits.items() if len(s) > 1)
    if shared_positional:
        raise LeakageError(
            f"{len(shared_positional)} positional observation identit(ies) appear in more "
            f"than one split, first: {shared_positional[0][:24]}...; the same episode step "
            "was collected into two splits"
        )

    shared_contents = sum(1 for s in content_splits.values() if len(s) > 1)
    shared_latents = sum(1 for s in latent_splits.values() if len(s) > 1)
    shared_tuples = sum(1 for s in tuple_splits.values() if len(s) > 1)
    return {
        "transitions": len(records),
        "per_split": {s.value: c for s, c in sorted(per_split.items(), key=lambda kv: kv[0].value)},
        "branch_groups": len(branch_splits),
        "distinct_observation_contents": len(content_splits),
        "distinct_latents": len(latent_splits),
        "distinct_transition_tuples": len(tuple_splits),
        "environment_seeds": len(seed_splits),
        "observation_contents_in_multiple_splits": shared_contents,
        "observation_content_overlap_rate": (
            shared_contents / len(content_splits) if content_splits else 0.0
        ),
        "latents_in_multiple_splits": shared_latents,
        "transition_tuples_in_multiple_splits": shared_tuples,
        "transition_tuple_overlap_rate": (
            shared_tuples / len(tuple_splits) if tuple_splits else 0.0
        ),
    }


def assert_no_transition_overlap(report: Mapping[str, Any], family: str) -> None:
    """Require complete transition disjointness for a family that can achieve it.

    Called for generative domains, where a shared `(observation, action,
    successor)` tuple is the held-out answer sitting in the training set rather
    than an artefact of a small state space.
    """
    shared = int(report.get("transition_tuples_in_multiple_splits", 0))
    if shared:
        raise LeakageError(
            f"{family}: {shared} complete transition tuple(s) appear in more than one "
            f"split ({report.get('transition_tuple_overlap_rate', 0.0):.4%} of "
            f"{report.get('distinct_transition_tuples')}); this family is generative and "
            "must be disjoint"
        )


def mixture_report(records: Sequence[TransitionRecord]) -> dict[str, Any]:
    """Realised collection mixture against the preregistered target."""
    counts: Counter[CollectorPolicy] = Counter(r.collector_policy for r in records)
    total = len(records) or 1
    return {
        policy.value: {
            "count": counts.get(policy, 0),
            "share": counts.get(policy, 0) / total,
            "target_share": COLLECTION_MIXTURE[policy],
        }
        for policy in CollectorPolicy
    }


def branch_coverage_report(records: Sequence[TransitionRecord]) -> dict[str, Any]:
    """How many restored states actually got more than one action tried.

    This is the number that decides whether an interventional claim is even
    admissible: branch groups of size one are passive observations wearing a
    branch label.
    """
    sizes: dict[str, set[int]] = defaultdict(set)
    for record in records:
        if record.branch_group_id is not None:
            sizes[record.branch_group_id].add(record.action)
    distribution = Counter(len(actions) for actions in sizes.values())
    multi = sum(count for size, count in distribution.items() if size > 1)
    return {
        "branch_groups": len(sizes),
        "groups_with_multiple_actions": multi,
        "size_distribution": {str(k): v for k, v in sorted(distribution.items())},
    }
