"""The propose → verify → repair loop.

The teacher writes a world model, the verifier scores it automatically, and
the divergence point is fed back as the next prompt. No human ever labels
anything — which is the whole reason Phase 1 built the verifier first.

Two decisions worth naming:

**Best, not last.** The loop keeps the highest-fitness attempt across all
rounds rather than the final one. Repair is not monotonic: a model asked to
explain a new failure will sometimes trade away something it had right, and
without this the corpus would fill with third-round regressions.

**Rounds are few.** Three by default. Rodionov's failure analysis found
agents refining forever inside a hypothesis that was wrong from the start,
and more rounds mostly buy deeper commitment to that mistake. The repair
prompt asks for a different hypothesis rather than a patch for the same
reason.
"""

from __future__ import annotations

import random
import time
from dataclasses import replace
from dataclasses import dataclass, field
from typing import Any

from sentinel.env.history import History
from sentinel.env.types import Action
from sentinel.gen.grid import GridWorld, solve_level, solve_world
from sentinel.gen.spec import WorldSpec
from sentinel.verify import Verifier, evidence_coverage
from sentinel.verify.report import VerificationReport

from .client import LLMError, OllamaClient
from .loader import extract_code
from .prompts import SYSTEM, build_initial_prompt, build_repair_prompt
from .sandbox import Sandbox


@dataclass(slots=True)
class Attempt:
    """One proposal and what became of it."""

    round_index: int
    source: str
    fitness: float = 0.0
    load_error: str | None = None
    report_json: dict[str, Any] | None = None
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def loaded(self) -> bool:
        return self.load_error is None

    def summary(self) -> str:
        if self.load_error:
            return f"round {self.round_index}: LOAD FAILED — {self.load_error}"
        return f"round {self.round_index}: fitness={self.fitness:.4f}"


@dataclass(slots=True)
class InductionResult:
    """Everything one world produced."""

    world_id: str
    solved: bool
    best_source: str | None
    best_fitness: float
    best_report: VerificationReport | None
    attempts: list[Attempt] = field(default_factory=list)
    error: str | None = None

    @property
    def usable(self) -> bool:
        """Worth putting in the corpus.

        Includes imperfect models on purpose. A model that explains 80% of
        transitions is a genuine partial hypothesis and is exactly the kind
        of example the core needs to learn from — training only on perfect
        answers would teach it that induction either succeeds completely or
        not at all, which is not how induction works.
        """
        return self.best_source is not None and self.best_fitness > 0.0

    def summary(self) -> str:
        if self.error:
            return f"{self.world_id}: ERROR — {self.error}"
        mark = "SOLVED" if self.solved else f"partial {self.best_fitness:.3f}"
        return f"{self.world_id}: {mark} in {len(self.attempts)} round(s)"


def make_training_history(
    spec: WorldSpec, explore_steps: int = 12, seed: int = 0, attempts: int = 4
) -> History | None:
    """Produce evidence, retrying until it can actually falsify a model.

    Retries matter because weak evidence is not a neutral outcome. In the
    first smoke run one world produced a history whose transition channel
    was untestable — no two identical grids with differing successors, no
    action-dependent effect anywhere — and it still received a fitness
    score. That score was meaningless: nothing in the evidence could have
    contradicted an action-blind model. Rather than record an unreliable
    number, vary the exploration and try again.
    """
    for attempt in range(attempts):
        history = _build_history(spec, explore_steps, seed + attempt * 7919)
        if history is None:
            return None
        if not evidence_coverage(history).unexercised():
            return history
    return history


def _probe_targets(world: GridWorld, spec: WorldSpec, rng: random.Random, rounds: int = 3) -> None:
    """Walk onto targets in arbitrary order, so failed collections are seen."""
    from sentinel.gen.grid import TARGET, GridWorldModel, initial_state
    from sentinel.gen.spec import LevelSpec, WorldSpec as WS
    from sentinel.plan import BFSPlanner, PlanExecutor

    planner = BFSPlanner(max_nodes=20_000)
    for _ in range(rounds):
        if world.done:
            return
        grid = world.history.last.grid
        size = spec.field_size
        targets = [
            (x, y)
            for y in range(size)
            for x in range(size)
            if grid[y][x] == TARGET
        ]
        if not targets:
            return
        goal = targets[rng.randrange(len(targets))]
        level = spec.levels[world.state.level]
        probe = WS(
            world_id=spec.world_id,
            seed=spec.seed,
            field_size=size,
            # Route as if unordered, so the walk reaches whichever target was
            # picked rather than refusing to plan for an out-of-turn one.
            mechanics=replace(spec.mechanics, ordered_targets=False),
            levels=(
                LevelSpec(
                    start=(world.state.x, world.state.y),
                    walls=level.walls,
                    hazards=level.hazards,
                    targets=(goal,),
                    switches=level.switches,
                    gates=level.gates,
                ),
            ),
        )
        model = GridWorldModel(probe, level_index=0)
        plan = planner.plan(model, start=model.init_state())
        if plan is None:
            return
        for action in plan.actions:
            if world.done:
                return
            world.step(action)


def _build_history(
    spec: WorldSpec, explore_steps: int = 12, seed: int = 0
) -> History | None:
    """Produce evidence rich enough to actually falsify a wrong model.

    Random play is prepended for action variety, then the known solution is
    played so the history contains level completions. Phase 1 established
    why this matters: a history with no level boundary cannot refute a model
    claiming levels never end, and the verifier would pass it silently.

    Using the known solution is legitimate here — Phase 2 is building a
    corpus that teaches induction from good evidence. Choosing which actions
    to take when no solution is known is the exploration problem, and it
    belongs to a later phase.
    """
    if solve_world(spec) is None:
        return None

    world = GridWorld(spec)
    world.reset()
    rng = random.Random(seed)

    for _ in range(explore_steps):
        if world.done:
            break
        world.step(Action(rng.choice([1, 2, 3, 4, 5])))

    if world.done:
        # Exploration killed the agent before it demonstrated anything.
        # Restart clean rather than record an episode that ends in a hazard
        # having shown no level completion.
        world = GridWorld(spec)
        world.reset()

    # Probe targets out of order before solving.
    #
    # Without this, `ordered_targets` is not merely hard to infer, it is
    # unlearnable. A solution trajectory collects targets in the correct
    # sequence by construction, so an ordered world and an unordered one
    # produce byte-identical evidence -- measured directly, zero
    # failed-collection events in either. The rule that distinguishes them
    # is never exercised.
    #
    # Walking to targets in arbitrary order fixes that: in an ordered world
    # the wrong target does nothing when stepped on, and that non-event is
    # the only observable difference between the two rules.
    #
    # Kept deliberately brief. Only 32 transitions are encoded, and probing
    # harder -- 8 rounds, repeated per level -- filled that window with long
    # BFS routes and crowded out the varied movement that reveals the hidden
    # counter, collapsing charge_period from 0.883 to 0.405. Evidence has to
    # exercise every rule, and a window this small means covering one rule
    # better can cost another.
    _probe_targets(world, spec, rng)

    # Re-solve from wherever the agent now stands, level by level. A
    # solution computed from the opening position is invalid once
    # exploration has moved the agent, and replaying it walks into hazards —
    # producing exactly the evidence-poor history Phase 1 showed cannot
    # falsify a wrong model.
    for _ in range(spec.num_levels):
        if world.done:
            break
        path = solve_level(spec, world.state.level, start=world.state)
        if path is None:
            break
        for action in path:
            if world.done:
                break
            world.step(action)

    return world.history


class Teacher:
    """Induces world models from histories, using a local LLM."""

    def __init__(
        self,
        client: OllamaClient | None = None,
        max_rounds: int = 3,
        model_timeout: float = 2.0,
        max_steps_in_prompt: int = 40,
        sandbox: Sandbox | None = None,
    ) -> None:
        self.client = client or OllamaClient()
        self.verifier = Verifier(stop_on_crash=True)
        # Defaults to auto: containers when a runtime is present, in-process
        # otherwise. The same call site works before and after Docker exists.
        self.sandbox = sandbox if sandbox is not None else Sandbox(mode="auto")
        self.max_rounds = max_rounds
        self.model_timeout = model_timeout
        self.max_steps_in_prompt = max_steps_in_prompt
        self.effort_ladder = ("low", "medium", "medium")
        """Escalation, capped deliberately.

        The first attempt is cheap because most worlds are straightforward;
        effort is spent only where a cheap attempt already failed. But the
        cap is not arbitrary — an earlier ladder ending in "high" made things
        strictly worse: reasoning consumed most of the token budget and the
        actual code was truncated mid-file, so a harder-thinking round scored
        zero where an easier one had scored something. More thinking is only
        useful if there is room left to write the answer.
        """

    def induce(self, history: History, world_id: str | None = None) -> InductionResult:
        """Propose, verify, repair. Returns the best model found."""
        world_id = world_id or history.game_id
        result = InductionResult(
            world_id=world_id,
            solved=False,
            best_source=None,
            best_fitness=0.0,
            best_report=None,
        )

        coverage = evidence_coverage(history)
        if coverage.unexercised():
            # Not fatal, but recorded: a model scored against evidence that
            # cannot test a channel is unfalsifiable in that channel, and the
            # corpus should not pretend otherwise.
            result.error = f"weak evidence: {coverage.summary()}"

        prompt = build_initial_prompt(history, max_steps=self.max_steps_in_prompt)
        best_source: str | None = None

        for round_index in range(self.max_rounds):
            started = time.perf_counter()
            effort = self.effort_ladder[min(round_index, len(self.effort_ladder) - 1)]
            try:
                completion = self.client.complete(prompt, system=SYSTEM, think=effort)
            except LLMError as exc:
                result.error = str(exc)
                return result

            source = extract_code(completion.text)
            attempt = Attempt(
                round_index=round_index,
                source=source,
                prompt_tokens=completion.prompt_tokens,
                output_tokens=completion.output_tokens,
                seconds=time.perf_counter() - started,
            )

            outcome = self.sandbox.verify(
                source, history, name=f"{world_id}-r{round_index}"
            )
            if not outcome.ok:
                detail = outcome.error or "verification failed"
                if completion.truncated:
                    detail += " (generation hit the token limit — the code was cut off)"
                attempt.load_error = detail
                result.attempts.append(attempt)
                # Rebuild from the base prompt rather than appending. Appending
                # grew the prompt every round, which crowded out the output
                # budget and caused the very truncation being complained about.
                prompt = (
                    build_initial_prompt(history, max_steps=self.max_steps_in_prompt)
                    + f"\n\nA previous attempt failed to load: {detail}\n"
                    "Return ONE Python code block defining exactly the four required "
                    "top-level functions. Keep the code compact and complete — do not "
                    "add commentary, examples, or tests."
                )
                continue

            assert outcome.report is not None
            report = outcome.report
            attempt.fitness = report.fitness
            attempt.report_json = report.to_json()
            result.attempts.append(attempt)

            # Best-so-far, because repair is not monotonic.
            if report.fitness > result.best_fitness or best_source is None:
                result.best_fitness = report.fitness
                result.best_report = report
                best_source = source

            if report.is_perfect:
                result.solved = True
                break

            prompt = build_repair_prompt(
                history, source, report, max_steps=self.max_steps_in_prompt
            )

        result.best_source = best_source
        return result

    def induce_world(self, spec: WorldSpec, seed: int = 0) -> InductionResult:
        """Generate evidence for a world, then induce a model of it."""
        history = make_training_history(spec, seed=seed)
        if history is None:
            return InductionResult(
                world_id=spec.world_id,
                solved=False,
                best_source=None,
                best_fitness=0.0,
                best_report=None,
                error="world could not be solved, so no evidence could be produced",
            )
        return self.induce(history, world_id=spec.world_id)
