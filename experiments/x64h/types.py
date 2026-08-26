"""Immutable records and the taint lattice.

The whole experiment turns on one invariant: the sampled convention and the
target logical form must never reach an agent-facing object or persistent
state. Taint is therefore a type-level property carried by every field, and
the persistence writer enforces it rather than trusting call sites.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Taint(enum.Enum):
    """Who may see a field.

    PUBLIC       given to the agent and safe to persist
    OBSERVED     produced by an interaction the agent actually had
    ORACLE_ONLY  visible to arm 11 (oracle convention) or 12 (oracle meaning)
    TARGET_ONLY  evaluator only; never given to any arm
    FUTURE       exists but has not happened yet for this episode
    """
    PUBLIC = "PUBLIC"
    OBSERVED = "OBSERVED"
    ORACLE_ONLY = "ORACLE_ONLY"
    TARGET_ONLY = "TARGET_ONLY"
    FUTURE = "FUTURE"


PERSISTABLE = frozenset({Taint.PUBLIC, Taint.OBSERVED})


class TaintError(RuntimeError):
    """Raised when a forbidden field crosses a boundary. Never caught inside
    the package: a leak is a failed run, not a recoverable condition."""


@dataclass(frozen=True)
class Tainted:
    value: Any
    taint: Taint

    def read(self, allowed: frozenset[Taint]) -> Any:
        if self.taint not in allowed:
            raise TaintError(
                f"field with taint {self.taint.value} read where only "
                f"{sorted(t.value for t in allowed)} is permitted")
        return self.value


class OpenWorldKind(enum.Enum):
    IN = "IN"
    UNKNOWN_REALIZATION = "UNKNOWN_REALIZATION"
    UNKNOWN_MEANING = "UNKNOWN_MEANING"
    UNKNOWN_PROGRAM = "UNKNOWN_PROGRAM"


class Decision(enum.Enum):
    EXECUTE = "EXECUTE"
    ASK_SEMANTIC = "ASK_SEMANTIC"
    ASK_BEHAVIORAL = "ASK_BEHAVIORAL"
    ABSTAIN = "ABSTAIN"
    EXPAND = "EXPAND"


@dataclass(frozen=True)
class Evidence:
    """Everything the agent is entitled to see for the current task."""
    utterance: tuple[str, ...]
    demonstrations: tuple[tuple[str, str], ...]
    qa: tuple[tuple[Any, Any], ...] = ()

    def key(self) -> tuple:
        return (self.utterance, self.demonstrations, self.qa)


@dataclass(frozen=True)
class PosteriorState:
    """Persistent convention evidence. `log_p_phi` is unnormalised so that
    restart cannot silently change the prior's scale."""
    log_p_phi: tuple[float, ...]
    model_hash: str
    observation_hashes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Episode:
    """The agent-facing object. It has no field for the convention or the
    target form, by construction rather than by convention."""
    evidence: Evidence
    task_index: int
    arm: str

    def __post_init__(self) -> None:
        for k in ("convention", "phi", "target", "z_true", "gold"):
            if k in self.__dict__:
                raise TaintError(f"episode carries a forbidden field {k!r}")


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    program: Any = None
    open_world: dict[str, float] = field(default_factory=dict)
    conflict: float = 0.0
    ambiguity: float = 0.0
    asked: int = 0
    semantic_asked: int = 0
    incomplete_candidates: bool = False
    notes: tuple[str, ...] = ()
