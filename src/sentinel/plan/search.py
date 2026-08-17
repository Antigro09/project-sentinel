"""Planning by simulating the model forward.

The point of building an executable world model is that you can *search
inside it* — spending simulated actions instead of real ones. On a real
ARC game the engine runs at roughly 2,500 steps/sec, but a Python world
model runs far faster than that and, more importantly, is not scored.
Environment actions are the currency the benchmark actually charges for.

Plans produced here are **advisory**. Every plan carries the frame the
model predicted at each step, and the executor checks reality against
that prediction after every single action. The moment they disagree, the
plan is abandoned — because a divergence means the model is wrong, and
continuing to follow a plan derived from a wrong model is worse than
having no plan. That divergence is also the single most valuable piece of
evidence the system can collect, so it is captured, not discarded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable

from sentinel.env.types import Action
from sentinel.wm.contract import Outcome, RenderedGrid, WorldModel

DEFAULT_ACTIONS: tuple[int, ...] = (1, 2, 3, 4, 5)


@dataclass(frozen=True, slots=True)
class Plan:
    """An action sequence with the model's prediction at each step."""

    actions: tuple[Action, ...]
    predicted: tuple[RenderedGrid, ...]
    """predicted[i] is the frame the model expects *after* actions[i]."""
    goal: Outcome
    nodes_expanded: int
    exhausted: bool = False
    """True if search covered the whole reachable space without finding a goal."""

    def __len__(self) -> int:
        return len(self.actions)

    def __bool__(self) -> bool:
        return bool(self.actions)

    def summary(self) -> str:
        return (
            f"plan[{len(self.actions)} actions -> {self.goal.value}, "
            f"{self.nodes_expanded} nodes expanded]"
        )


@dataclass
class SearchStats:
    expanded: int = 0
    generated: int = 0
    pruned_dead: int = 0
    pruned_seen: int = 0
    hit_limit: bool = False
    exhausted: bool = False

    def summary(self) -> str:
        return (
            f"expanded={self.expanded} generated={self.generated} "
            f"seen-pruned={self.pruned_seen} dead-pruned={self.pruned_dead} "
            + ("LIMIT" if self.hit_limit else "exhausted" if self.exhausted else "found")
        )


class BFSPlanner:
    """Breadth-first search over an executable world model.

    Breadth-first because the score on ARC-AGI-3 is
    `(human_actions / ai_actions)^2` — action count is squared in the
    reward, so the *shortest* plan is worth substantially more than a
    merely valid one. BFS returns a shortest path by construction.

    States are deduplicated by hash, which is why the contract requires
    State to be hashable. That requirement is doing real work here: on the
    toy world, dedup is the difference between a few thousand nodes and an
    unbounded walk, since the agent can revisit any cell freely.
    """

    def __init__(
        self,
        max_nodes: int = 200_000,
        max_depth: int = 200,
        actions: Iterable[int] | None = None,
    ) -> None:
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.actions = tuple(actions) if actions is not None else DEFAULT_ACTIONS

    def _actions_for(self, model: WorldModel, state: Any) -> tuple[Action, ...]:
        ids = model.available_actions(state)
        if ids is None:
            ids = self.actions
        # ACTION6 takes coordinates, so it cannot be enumerated blindly —
        # a model that wants it planned must expose it another way rather
        # than have the planner guess 4096 coordinate pairs.
        return tuple(Action(i) for i in ids if i != 6)

    def plan(
        self,
        model: WorldModel,
        start: Any | None = None,
        goal: Outcome = Outcome.LEVEL_COMPLETE,
        stats: SearchStats | None = None,
    ) -> Plan | None:
        """Shortest action sequence reaching `goal`, or None."""
        stats = stats or SearchStats()
        root = model.init_state() if start is None else start

        try:
            if model.outcome(root) is goal:
                return Plan((), (), goal, 0)
        except Exception:  # noqa: BLE001 - a broken model simply cannot be planned through
            return None

        queue: deque[tuple[Any, tuple[Action, ...]]] = deque([(root, ())])
        seen: set[Hashable] = {root}

        while queue:
            state, path = queue.popleft()
            stats.expanded += 1

            if stats.expanded > self.max_nodes:
                stats.hit_limit = True
                return None
            if len(path) >= self.max_depth:
                continue

            for action in self._actions_for(model, state):
                try:
                    nxt = model.transition(state, action)
                    result = model.outcome(nxt)
                except Exception:  # noqa: BLE001
                    continue

                stats.generated += 1

                if result is Outcome.GAME_OVER:
                    stats.pruned_dead += 1
                    continue
                if result is goal:
                    actions = (*path, action)
                    return Plan(
                        actions=actions,
                        predicted=self._predict(model, root, actions),
                        goal=goal,
                        nodes_expanded=stats.expanded,
                    )
                if nxt in seen:
                    stats.pruned_seen += 1
                    continue

                seen.add(nxt)
                queue.append((nxt, (*path, action)))

        stats.exhausted = True
        return None

    def _predict(
        self, model: WorldModel, root: Any, actions: tuple[Action, ...]
    ) -> tuple[RenderedGrid, ...]:
        """Re-simulate to capture the predicted frame after each action.

        Done as a second pass rather than carried through the frontier:
        storing a 64x64 grid per queued node would dominate memory, and
        only the winning path's predictions are ever needed.
        """
        frames: list[RenderedGrid] = []
        state = root
        for action in actions:
            state = model.transition(state, action)
            frames.append(model.render(state))
        return tuple(frames)


@dataclass
class ExecutionResult:
    """What happened when a plan met reality."""

    executed: int
    diverged_at: int | None
    """Index of the first action whose predicted frame did not match."""
    completed: bool
    predicted: RenderedGrid | None = None
    observed: Any = None
    mismatched_cells: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def diverged(self) -> bool:
        return self.diverged_at is not None

    def summary(self) -> str:
        if self.diverged:
            return (
                f"executed {self.executed}, DIVERGED at action {self.diverged_at} "
                f"({self.mismatched_cells} cells wrong)"
            )
        return f"executed {self.executed}, {'goal reached' if self.completed else 'plan ended'}"


class PlanExecutor:
    """Runs a plan against reality, one action at a time, checking as it goes.

    The check after every action is the whole point. A model good enough to
    produce a plan is not necessarily good enough to be trusted for the
    plan's full length, and executing blindly past a divergence spends real
    actions on a hypothesis already known to be false.
    """

    def __init__(self, abstain_tolerant: bool = True) -> None:
        self.abstain_tolerant = abstain_tolerant

    def execute(
        self,
        plan: Plan,
        step_fn: Callable[[Action], Any],
        is_done: Callable[[], bool] | None = None,
    ) -> ExecutionResult:
        """Execute `plan`, halting at the first prediction mismatch.

        `step_fn` takes an Action and returns the observed Observation, so
        this works against the ARC engine, the toy world, or anything else
        satisfying that shape.
        """
        from sentinel.verify.verifier import compare

        levels_completed: int | None = None

        for i, action in enumerate(plan.actions):
            observed = step_fn(action)
            predicted = plan.predicted[i] if i < len(plan.predicted) else None

            # Completing a level swaps in a layout the model has never seen,
            # so a mismatch on that step is the plan *succeeding*, not the
            # model failing. Without this the correct model reports a false
            # divergence on the final action of every level it solves.
            advanced = (
                levels_completed is not None
                and observed.levels_completed > levels_completed
            )
            levels_completed = observed.levels_completed
            if advanced:
                return ExecutionResult(
                    executed=i + 1,
                    diverged_at=None,
                    completed=True,
                    notes=["level boundary reached; model state is now stale"],
                )

            if predicted is not None:
                cells, matched = compare(predicted, observed.grid)
                if not matched:
                    return ExecutionResult(
                        executed=i + 1,
                        diverged_at=i,
                        completed=False,
                        predicted=predicted,
                        observed=observed,
                        mismatched_cells=cells.predicted - cells.correct,
                        notes=[f"model mispredicted after {action}"],
                    )

            if is_done is not None and is_done():
                return ExecutionResult(executed=i + 1, diverged_at=None, completed=True)

        return ExecutionResult(
            executed=len(plan.actions), diverged_at=None, completed=True
        )
