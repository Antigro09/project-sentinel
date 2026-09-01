"""Stratified v2 splits, with lineage tracking that survives cloning.

Scale 0 split by episode key and audited afterwards. That was enough to catch a
branch group divided between splits, and not enough to express what Scale 1
needs: appearance and dynamics have to move independently, so that a result on
held-out appearance can be told apart from a result on held-out mechanics.

Four strata, and only the first carries a world-model claim:

* `dynamics_clean` -- appearance pinned to one canonical palette, layouts,
  dynamics and goals held out. This is the attribution stratum.
* `appearance_shift` -- layouts and dynamics seen, palettes and textures held
  out. A perception diagnostic. The S1.2 probes showed every representation
  including raw pixels collapses here, so a failure on this stratum says
  something about perception and nothing about a dynamics model.
* `crossed_shift` -- both shifted. Reported, never primary.
* `legacy_v1_replication` -- the 36.3%-contaminated v1 split, replication only.

Lineage is the part Scale 0 lacked. A cloned state's branches, and any
deterministic descendant of them, carry the lineage of the state they came from,
so the audit can ask whether a *descendant* crossed a split rather than only
whether a sibling did.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from sentinel.wm.dataset import LeakageError
from sentinel.wm.latent_contract import ContractViolation
from sentinel.wm.versioning import digest_of

CANONICAL_APPEARANCE_SEED = 4242
"""The one palette `dynamics_clean` uses, so appearance is a constant there."""


class Stratum(str, Enum):
    DYNAMICS_CLEAN = "dynamics_clean"
    APPEARANCE_SHIFT = "appearance_shift"
    CROSSED_SHIFT = "crossed_shift"
    LEGACY_V1_REPLICATION = "legacy_v1_replication"


PRIMARY_STRATUM = Stratum.DYNAMICS_CLEAN
NON_PRIMARY = frozenset(
    {Stratum.APPEARANCE_SHIFT, Stratum.CROSSED_SHIFT, Stratum.LEGACY_V1_REPLICATION}
)


@dataclass(frozen=True, slots=True)
class EpisodeDescriptor:
    """Everything that determines an episode, factored so shifts are separable."""

    family: str
    layout_seed: int
    appearance_seed: int
    phase_seed: int
    goal_draw: int
    dynamics_name: str = "base"

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "layout_seed": int(self.layout_seed),
            "appearance_seed": int(self.appearance_seed),
            "phase_seed": int(self.phase_seed),
            "goal_draw": int(self.goal_draw),
            "dynamics_name": self.dynamics_name,
        }

    @property
    def environment_hash(self) -> str:
        return digest_of({"family": self.family})

    @property
    def layout_hash(self) -> str:
        return digest_of({"family": self.family, "layout_seed": int(self.layout_seed)})

    @property
    def appearance_hash(self) -> str:
        return digest_of({"family": self.family, "appearance_seed": int(self.appearance_seed)})

    @property
    def dynamics_hash(self) -> str:
        return digest_of(
            {
                "family": self.family,
                "dynamics_name": self.dynamics_name,
                "phase_seed": int(self.phase_seed),
            }
        )

    @property
    def goal_hash(self) -> str:
        return digest_of({"family": self.family, "goal_draw": int(self.goal_draw)})

    @property
    def episode_hash(self) -> str:
        return digest_of(self.canonical_dict())


@dataclass(frozen=True, slots=True)
class Lineage:
    """Where a transition came from, including through a clone.

    `root` is the episode. `clone_point` names the restored state a branch group
    came from, and `depth` counts deterministic descendants of it. Two branches
    of one clone share a root and a clone point; a state reached by continuing
    from a branch shares them too, which is what makes descendant leakage
    visible.
    """

    root: str
    clone_point: str | None = None
    depth: int = 0

    @property
    def lineage_hash(self) -> str:
        return digest_of({"root": self.root, "clone_point": self.clone_point})

    def descend(self) -> "Lineage":
        return Lineage(root=self.root, clone_point=self.clone_point, depth=self.depth + 1)

    def clone_at(self, marker: str) -> "Lineage":
        return Lineage(root=self.root, clone_point=marker, depth=0)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "clone_point": self.clone_point,
            "depth": self.depth,
            "lineage_hash": self.lineage_hash,
        }


@dataclass
class StratifiedSplitManifest:
    """Assigns a stratum from the descriptor, before anything is collected."""

    train_layouts: frozenset[int]
    held_layouts: frozenset[int]
    train_appearances: frozenset[int]
    held_appearances: frozenset[int]
    assignments: dict[str, Stratum] = field(default_factory=dict)
    descriptors: dict[str, dict[str, Any]] = field(default_factory=dict)
    sealed: bool = False

    def __post_init__(self) -> None:
        if CANONICAL_APPEARANCE_SEED in self.held_appearances:
            raise ContractViolation(
                "the canonical appearance is held out, so dynamics_clean would also be "
                "an appearance shift and the two strata could not be told apart"
            )
        if self.train_layouts & self.held_layouts:
            raise ContractViolation(
                f"layout seeds appear in both train and held-out: "
                f"{sorted(self.train_layouts & self.held_layouts)[:5]}"
            )
        if self.train_appearances & self.held_appearances:
            raise ContractViolation(
                f"appearance seeds appear in both train and held-out: "
                f"{sorted(self.train_appearances & self.held_appearances)[:5]}"
            )

    def stratum_for(self, descriptor: EpisodeDescriptor) -> Stratum:
        if descriptor.family == "procedural_visual":
            return Stratum.LEGACY_V1_REPLICATION
        layout_held = descriptor.layout_seed in self.held_layouts
        appearance_held = descriptor.appearance_seed in self.held_appearances
        canonical = descriptor.appearance_seed == CANONICAL_APPEARANCE_SEED
        if layout_held and appearance_held:
            return Stratum.CROSSED_SHIFT
        if appearance_held:
            return Stratum.APPEARANCE_SHIFT
        if canonical:
            return Stratum.DYNAMICS_CLEAN
        raise ContractViolation(
            f"{descriptor} falls in no stratum: appearance {descriptor.appearance_seed} is "
            "neither the canonical palette nor a held-out one, so the result could not be "
            "attributed to dynamics or to perception"
        )

    def assign(self, descriptor: EpisodeDescriptor) -> Stratum:
        key = descriptor.episode_hash
        if key in self.assignments:
            return self.assignments[key]
        if self.sealed:
            raise LeakageError(f"manifest is sealed; {descriptor} was never assigned")
        stratum = self.stratum_for(descriptor)
        self.assignments[key] = stratum
        self.descriptors[key] = descriptor.canonical_dict()
        return stratum

    def stratum_of(self, descriptor: EpisodeDescriptor) -> Stratum:
        key = descriptor.episode_hash
        if key not in self.assignments:
            raise LeakageError(
                f"{descriptor} has no stratum; strata are assigned before collection"
            )
        return self.assignments[key]

    def seal(self) -> None:
        self.sealed = True

    @property
    def digest(self) -> str:
        return digest_of(
            {
                "train_layouts": sorted(self.train_layouts),
                "held_layouts": sorted(self.held_layouts),
                "train_appearances": sorted(self.train_appearances),
                "held_appearances": sorted(self.held_appearances),
                "assignments": {k: v.value for k, v in sorted(self.assignments.items())},
                "sealed": self.sealed,
            }
        )


@dataclass(frozen=True, slots=True)
class StratifiedRecord:
    """The minimum a split audit needs, kept separate from the training record."""

    descriptor: EpisodeDescriptor
    lineage: Lineage
    content_digest_t: str
    action: int
    content_digest_t1: str
    trajectory_id: str


def audit_v2_splits(
    records: Sequence[StratifiedRecord], manifest: StratifiedSplitManifest
) -> dict[str, Any]:
    """Structural violations raise; content overlap is measured.

    The clause Scale 0 did not have is the lineage one: a branch of a cloned
    state, and anything descended from that branch, must stay in the stratum its
    root was assigned. Checking siblings alone misses a descendant that wandered.
    """
    lineage_strata: dict[str, set[Stratum]] = defaultdict(set)
    trajectory_strata: dict[str, set[Stratum]] = defaultdict(set)
    layout_strata: dict[str, set[Stratum]] = defaultdict(set)
    appearance_strata: dict[str, set[Stratum]] = defaultdict(set)
    goal_strata: dict[str, set[Stratum]] = defaultdict(set)
    tuple_strata: dict[tuple[str, int, str], set[Stratum]] = defaultdict(set)
    per_stratum: Counter[Stratum] = Counter()

    for record in records:
        stratum = manifest.stratum_of(record.descriptor)
        per_stratum[stratum] += 1
        lineage_strata[record.lineage.lineage_hash].add(stratum)
        trajectory_strata[record.trajectory_id].add(stratum)
        layout_strata[record.descriptor.layout_hash].add(stratum)
        appearance_strata[record.descriptor.appearance_hash].add(stratum)
        goal_strata[record.descriptor.goal_hash].add(stratum)
        tuple_strata[(record.content_digest_t, int(record.action), record.content_digest_t1)].add(
            stratum
        )

    def crossing(mapping: dict[Any, set[Stratum]]) -> list[Any]:
        return sorted((k for k, v in mapping.items() if len(v) > 1), key=str)

    for label, mapping in (
        ("cloned-state lineage", lineage_strata),
        ("trajectory", trajectory_strata),
    ):
        offenders = crossing(mapping)
        if offenders:
            raise LeakageError(
                f"{len(offenders)} {label}(s) appear in more than one stratum, first: "
                f"{offenders[0]}; a descendant of a cloned state carries its lineage and "
                "must stay where its root was assigned"
            )

    # dynamics_clean and crossed_shift share held-out layouts *by design* -- that
    # sharing is what makes crossed_shift a shift of the same layouts rather than
    # a different experiment. The violation is a layout appearing in both a
    # train-layout stratum and a held-layout stratum, which means a layout the
    # model trained on is being scored as held out.
    held_layout_strata = {Stratum.DYNAMICS_CLEAN, Stratum.CROSSED_SHIFT}
    train_layout_strata = {Stratum.APPEARANCE_SHIFT}
    leaked_layouts = [
        k
        for k, v in layout_strata.items()
        if (v & held_layout_strata) and (v & train_layout_strata)
    ]
    if leaked_layouts:
        raise LeakageError(
            f"{len(leaked_layouts)} layout(s) appear in both a trained stratum and a "
            f"held-out-layout stratum, first: {leaked_layouts[0]}; a layout the model "
            "trained on is being scored as held out"
        )

    shared_tuples = sum(1 for v in tuple_strata.values() if len(v) > 1)
    return {
        "records": len(records),
        "per_stratum": {s.value: c for s, c in sorted(per_stratum.items(), key=lambda kv: kv[0].value)},
        "distinct_lineages": len(lineage_strata),
        "distinct_trajectories": len(trajectory_strata),
        "distinct_layouts": len(layout_strata),
        "distinct_appearances": len(appearance_strata),
        "distinct_goals": len(goal_strata),
        "transition_tuples_in_multiple_strata": shared_tuples,
        "transition_tuple_overlap_rate": shared_tuples / len(tuple_strata) if tuple_strata else 0.0,
        "layouts_shared_between_clean_and_crossed": sum(
            1
            for v in layout_strata.values()
            if {Stratum.DYNAMICS_CLEAN, Stratum.CROSSED_SHIFT} <= v
        ),
    }


def assert_primary_stratum_is_disjoint(report: Mapping[str, Any]) -> None:
    """dynamics_clean is the attribution stratum, so it must be fully disjoint."""
    shared = int(report.get("transition_tuples_in_multiple_strata", 0))
    if shared:
        raise LeakageError(
            f"{shared} transition tuple(s) appear in more than one stratum; the primary "
            "stratum must be disjoint or a world-model claim rests on seen transitions"
        )
