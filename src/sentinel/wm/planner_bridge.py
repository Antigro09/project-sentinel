"""Planner adapters over a pure rollout interface, with exact accounting.

Scale 0 does not ask whether planning works. It asks whether planning can be
*counted*, because the cheapest way to manufacture an apparent world-model
advantage is to give one arm more planner compute than another. So every planner
here evaluates exactly the number of complete action sequences the matrix
specifies, and the account it returns is the number the report uses.

The rollout interface is pure. A planner receives an `expand` and a `score` and
cannot reach the model's parameters, its optimiser, or the environment. The
combinatorics are not solved by that -- Lemma 2 says an open-loop search over
`B` actions and horizon `H` contains `B**H` sequences, and at B=4, H=25 that is
about 1.1e15 -- which is exactly why a fixed candidate budget is the honest way
to compare arms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from sentinel.wm.latent_contract import ContractViolation, UncertaintyTriple
from sentinel.wm.versioning import digest_of


@dataclass(frozen=True, slots=True)
class RolloutNode:
    """One expanded state in a latent rollout."""

    state: Any
    depth: int
    cumulative_reward: float
    uncertainty: UncertaintyTriple
    constraint_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class UtilityDistribution:
    """Score of a candidate, kept decomposed so a penalty cannot hide in the mean."""

    expected_return: float
    uncertainty_penalty: float
    constraint_penalty: float

    @property
    def total(self) -> float:
        return self.expected_return - self.uncertainty_penalty - self.constraint_penalty


@runtime_checkable
class LatentRollout(Protocol):
    def expand(self, node: RolloutNode, action: int) -> RolloutNode: ...

    def score(self, node: RolloutNode) -> UtilityDistribution: ...

    @property
    def actions(self) -> tuple[int, ...]: ...


@dataclass
class PlannerAccount:
    """What the planner spent. Reported per run, never averaged away."""

    invocations: int = 0
    candidate_sequences: int = 0
    model_calls: int = 0
    expanded_nodes: int = 0
    score_calls: int = 0
    wall_seconds: float = 0.0

    def merge(self, other: "PlannerAccount") -> None:
        self.invocations += other.invocations
        self.candidate_sequences += other.candidate_sequences
        self.model_calls += other.model_calls
        self.expanded_nodes += other.expanded_nodes
        self.score_calls += other.score_calls
        self.wall_seconds += other.wall_seconds

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "invocations": self.invocations,
            "candidate_sequences": self.candidate_sequences,
            "model_calls": self.model_calls,
            "expanded_nodes": self.expanded_nodes,
            "score_calls": self.score_calls,
            "wall_seconds": self.wall_seconds,
            "calls_per_second": self.model_calls / self.wall_seconds if self.wall_seconds else 0.0,
        }


@dataclass
class CountingRollout:
    """Wraps a rollout and counts every call. The planner cannot avoid it."""

    inner: LatentRollout
    account: PlannerAccount = field(default_factory=PlannerAccount)

    @property
    def actions(self) -> tuple[int, ...]:
        return self.inner.actions

    def expand(self, node: RolloutNode, action: int) -> RolloutNode:
        self.account.model_calls += 1
        self.account.expanded_nodes += 1
        return self.inner.expand(node, action)

    def score(self, node: RolloutNode) -> UtilityDistribution:
        self.account.score_calls += 1
        return self.inner.score(node)


@dataclass(frozen=True, slots=True)
class Plan:
    actions: tuple[int, ...]
    utility: UtilityDistribution
    terminal_uncertainty: UncertaintyTriple
    account: PlannerAccount

    @property
    def digest(self) -> str:
        return digest_of(
            {
                "actions": list(self.actions),
                "expected_return": self.utility.expected_return,
                "uncertainty_penalty": self.utility.uncertainty_penalty,
                "constraint_penalty": self.utility.constraint_penalty,
            }
        )


def _deterministic_sequence(
    actions: tuple[int, ...], horizon: int, index: int, salt: str
) -> tuple[int, ...]:
    """A reproducible action word. No global RNG anywhere in the planner."""
    digest = digest_of({"salt": salt, "index": index, "horizon": horizon})[7:]
    body = int(digest, 16)
    sequence = []
    for _ in range(horizon):
        sequence.append(actions[body % len(actions)])
        body //= len(actions)
        if body == 0:
            body = int(digest_of({"salt": salt, "index": index, "extend": len(sequence)})[7:23], 16)
    return tuple(sequence)


class Planner(Protocol):
    name: str

    def plan(self, rollout: LatentRollout, root: RolloutNode, horizon: int, candidates: int) -> Plan: ...


def _evaluate(
    rollout: CountingRollout, root: RolloutNode, sequence: Sequence[int]
) -> tuple[RolloutNode, UtilityDistribution]:
    node = root
    for action in sequence:
        node = rollout.expand(node, action)
    return node, rollout.score(node)


@dataclass
class BeamPlanner:
    """Breadth-limited search that still evaluates the full candidate budget.

    A beam that scored fewer sequences than CEM would look cheaper and better at
    once, so the budget is spent exactly: the beam is refilled with deterministic
    completions until `candidates` complete sequences have been scored.
    """

    width: int = 8
    name: str = "beam"

    def plan(self, rollout: LatentRollout, root: RolloutNode, horizon: int, candidates: int) -> Plan:
        counting = rollout if isinstance(rollout, CountingRollout) else CountingRollout(rollout)
        started = time.perf_counter()
        actions = counting.actions
        beam: list[tuple[tuple[int, ...], RolloutNode]] = [((), root)]
        scored: list[tuple[tuple[int, ...], RolloutNode, UtilityDistribution]] = []

        for _ in range(horizon):
            expanded: list[tuple[tuple[int, ...], RolloutNode, float]] = []
            for prefix, node in beam:
                for action in actions:
                    child = counting.expand(node, action)
                    expanded.append((prefix + (action,), child, child.cumulative_reward))
            expanded.sort(key=lambda item: -item[2])
            beam = [(prefix, node) for prefix, node, _ in expanded[: self.width]]

        for prefix, node in beam:
            scored.append((prefix, node, counting.score(node)))

        index = 0
        while len(scored) < candidates:
            sequence = _deterministic_sequence(actions, horizon, index, "beam-refill")
            node, utility = _evaluate(counting, root, sequence)
            scored.append((sequence, node, utility))
            index += 1
        scored = scored[:candidates]

        counting.account.invocations += 1
        counting.account.candidate_sequences += len(scored)
        counting.account.wall_seconds += time.perf_counter() - started
        best = max(scored, key=lambda item: item[2].total)
        return Plan(best[0], best[2], best[1].uncertainty, counting.account)


@dataclass
class CEMPlanner:
    """Cross-entropy method with a deterministic proposal stream."""

    iterations: int = 4
    elite_fraction: float = 0.25
    name: str = "cem"

    def plan(self, rollout: LatentRollout, root: RolloutNode, horizon: int, candidates: int) -> Plan:
        counting = rollout if isinstance(rollout, CountingRollout) else CountingRollout(rollout)
        started = time.perf_counter()
        actions = counting.actions
        per_iteration = max(1, candidates // self.iterations)
        remaining = candidates
        elite_prefix: tuple[int, ...] = ()
        best: tuple[tuple[int, ...], RolloutNode, UtilityDistribution] | None = None

        for iteration in range(self.iterations):
            draw = min(per_iteration, remaining) if iteration < self.iterations - 1 else remaining
            population: list[tuple[tuple[int, ...], RolloutNode, UtilityDistribution]] = []
            for index in range(draw):
                sequence = _deterministic_sequence(
                    actions, horizon, index, f"cem-{iteration}-{elite_prefix}"
                )
                if elite_prefix:
                    keep = min(len(elite_prefix), horizon // 2)
                    sequence = tuple(elite_prefix[:keep]) + sequence[keep:]
                node, utility = _evaluate(counting, root, sequence)
                population.append((sequence, node, utility))
            remaining -= draw
            if not population:
                break
            population.sort(key=lambda item: -item[2].total)
            if best is None or population[0][2].total > best[2].total:
                best = population[0]
            elite_count = max(1, int(len(population) * self.elite_fraction))
            elite_prefix = population[0][0][:elite_count]
            if remaining <= 0:
                break

        if best is None:  # pragma: no cover - candidates >= 1 by contract
            raise ContractViolation("CEM was given a zero candidate budget")
        counting.account.invocations += 1
        counting.account.candidate_sequences += candidates
        counting.account.wall_seconds += time.perf_counter() - started
        return Plan(best[0], best[2], best[1].uncertainty, counting.account)


@dataclass
class MCTSPlanner:
    """Deterministic best-first tree search, budgeted in complete sequences."""

    exploration: float = 1.0
    name: str = "mcts"

    def plan(self, rollout: LatentRollout, root: RolloutNode, horizon: int, candidates: int) -> Plan:
        counting = rollout if isinstance(rollout, CountingRollout) else CountingRollout(rollout)
        started = time.perf_counter()
        actions = counting.actions
        visits: dict[tuple[int, ...], int] = {(): 0}
        values: dict[tuple[int, ...], float] = {(): 0.0}
        best: tuple[tuple[int, ...], RolloutNode, UtilityDistribution] | None = None

        for simulation in range(candidates):
            prefix: tuple[int, ...] = ()
            node = root
            for depth in range(horizon):
                scores = []
                for action in actions:
                    child_key = prefix + (action,)
                    seen = visits.get(child_key, 0)
                    mean = values.get(child_key, 0.0) / seen if seen else 0.0
                    bonus = self.exploration * ((visits[prefix] + 1) ** 0.5) / (1 + seen)
                    scores.append((mean + bonus, action))
                scores.sort(key=lambda item: (-item[0], item[1]))
                action = scores[0][1]
                prefix = prefix + (action,)
                node = counting.expand(node, action)
                visits.setdefault(prefix, 0)
                values.setdefault(prefix, 0.0)
            utility = counting.score(node)
            for depth in range(len(prefix) + 1):
                key = prefix[:depth]
                visits[key] = visits.get(key, 0) + 1
                values[key] = values.get(key, 0.0) + utility.total
            if best is None or utility.total > best[2].total:
                best = (prefix, node, utility)

        if best is None:  # pragma: no cover
            raise ContractViolation("MCTS was given a zero candidate budget")
        counting.account.invocations += 1
        counting.account.candidate_sequences += candidates
        counting.account.wall_seconds += time.perf_counter() - started
        return Plan(best[0], best[2], best[1].uncertainty, counting.account)


@dataclass
class FakeDynamicsRollout:
    """Deterministic fake dynamics, so planner accounting can be tested alone.

    Scale 0's planner gate is explicitly a dry run: the point is that the counts,
    the uncertainty penalty, the probe requests, and the authority path all work
    before a learned model is attached. A fake with a closed form makes the
    expected counts checkable by hand.
    """

    action_set: tuple[int, ...] = (0, 1, 2, 3)
    reward_scale: float = 1.0
    uncertainty_growth: float = 0.05
    constraint_action: int | None = 3

    @property
    def actions(self) -> tuple[int, ...]:
        return self.action_set

    def root(self, state: int = 0) -> RolloutNode:
        return RolloutNode(
            state=state, depth=0, cumulative_reward=0.0, uncertainty=UncertaintyTriple(0.0, 0.0, 0.0)
        )

    def expand(self, node: RolloutNode, action: int) -> RolloutNode:
        state = (int(node.state) * 7 + action * 13 + 1) % 251
        reward = self.reward_scale * ((state % 11) - 5) / 5.0
        depth = node.depth + 1
        return RolloutNode(
            state=state,
            depth=depth,
            cumulative_reward=node.cumulative_reward + reward,
            uncertainty=UncertaintyTriple(
                aleatoric=node.uncertainty.aleatoric + self.uncertainty_growth,
                epistemic=node.uncertainty.epistemic + self.uncertainty_growth * depth,
                inadequacy=node.uncertainty.inadequacy,
            ),
            constraint_cost=node.constraint_cost
            + (1.0 if action == self.constraint_action else 0.0),
        )

    def score(self, node: RolloutNode) -> UtilityDistribution:
        return UtilityDistribution(
            expected_return=node.cumulative_reward,
            uncertainty_penalty=node.uncertainty.scalar((0.1, 0.2, 1.0)),
            constraint_penalty=node.constraint_cost,
        )


def planner_registry() -> dict[str, Callable[[], Planner]]:
    return {"beam": BeamPlanner, "cem": CEMPlanner, "mcts": MCTSPlanner}
