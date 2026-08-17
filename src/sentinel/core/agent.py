"""The closed loop: watch, infer, model, plan, act.

This is the payoff. Everything before it was apparatus or measurement; here
the core is finally *used*.

    1. Explore an unknown world briefly, taking arbitrary actions.
    2. Infer its hidden rules from what happened, using the trained core.
    3. Build an executable world model from those inferred rules.
    4. Plan inside that model with BFS, spending simulated actions.
    5. Execute the plan in the real world.

No LLM is involved at any point. A 1.3M-parameter network trained from
scratch is the only learned component, and the environment is one it has
never seen.

**What this measures that accuracy could not.** A label being 78% correct
says nothing about whether the resulting model is *usable*. A plan built on
a wrong hidden-state assumption does not degrade gracefully — it desyncs on
the third move and walks the agent into a wall or a hazard. Solve rate is
therefore the honest test, and it is bracketed on both sides:

    true mechanics    -> the ceiling: what perfect inference would achieve
    inferred mechanics-> the core's actual contribution
    default guess     -> the floor: assume the simplest rules and hope

If inferred lands at the floor, the core's accuracy was decorative. If it
lands near the ceiling, the inference is doing real work.

**One thing is handed over, and it should be stated plainly.** The layout
reader below knows the colour convention — that value 1 is a wall, 3 a
target, and so on. That is observable structure rather than hidden rules,
and learning the palette is a separate problem from learning the dynamics.
The variable under test is the *mechanics*, which is what the core infers
and what the three-way comparison isolates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from sentinel.env.types import Action
from sentinel.gen.grid import (
    AGENT,
    GATE_CLOSED,
    GATE_OPEN,
    HAZARD,
    SWITCH,
    TARGET,
    WALL,
    GridWorld,
    GridWorldModel,
)
from sentinel.gen.grid import GridState, initial_state
from sentinel.gen.grid import GridState
from sentinel.gen.spec import LevelSpec, Mechanics, WorldSpec
from sentinel.plan import BFSPlanner, PlanExecutor

from .encoding import encode_history
from .model import TinyRecursiveCore

CHARGE_FROM_CLASS = {0: None, 1: 3, 2: 4}


def read_layout(grid, field_size: int) -> LevelSpec:
    """Recover a level's static layout from one observed frame."""
    walls, hazards, targets, switches, gates = [], [], [], [], []
    start = (0, 0)
    for y in range(field_size):
        for x in range(field_size):
            value = grid[y][x]
            if value == WALL:
                walls.append((x, y))
            elif value == HAZARD:
                hazards.append((x, y))
            elif value == TARGET:
                targets.append((x, y))
            elif value == AGENT:
                start = (x, y)
            elif value == SWITCH:
                switches.append((x, y))
            elif value in (GATE_CLOSED, GATE_OPEN):
                gates.append((x, y))
    return LevelSpec(
        start=start,
        walls=frozenset(walls),
        hazards=frozenset(hazards),
        targets=tuple(targets),
        switches=frozenset(switches),
        gates=frozenset(gates),
    )


def infer_mechanics(core: TinyRecursiveCore, history) -> Mechanics:
    """Ask the core what rules govern the world it just watched."""
    grids, actions = encode_history(history)
    logits = core(mx.array(grids[None].astype(np.int32)), mx.array(actions[None].astype(np.int32)))
    mx.eval(logits)
    pred = [int(np.array(mx.argmax(head, axis=-1))[0]) for head in logits]
    return Mechanics(
        step_distance=pred[0] + 1,
        charge_period=CHARGE_FROM_CLASS.get(pred[1]),
        wrap_edges=bool(pred[2]),
        has_hazards=bool(pred[3]),
        has_switches=bool(pred[4]),
        ordered_targets=bool(pred[5]),
    )


@dataclass
class EpisodeOutcome:
    world_id: str
    solved: bool
    levels_completed: int
    total_levels: int
    actions_used: int
    diverged: bool
    inferred: Mechanics | None = None
    truth: Mechanics | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def mechanics_correct(self) -> bool:
        if self.inferred is None or self.truth is None:
            return False
        return self.inferred.summary() == self.truth.summary()

    def summary(self) -> str:
        mark = "SOLVED" if self.solved else f"{self.levels_completed}/{self.total_levels}"
        return f"{self.world_id}: {mark} in {self.actions_used} actions"


def _spec_with(spec: WorldSpec, mechanics: Mechanics, observed: LevelSpec, level: int) -> WorldSpec:
    """A one-level spec built from what was observed plus assumed rules."""
    return WorldSpec(
        world_id=spec.world_id,
        seed=spec.seed,
        field_size=spec.field_size,
        mechanics=mechanics,
        levels=(observed,),
    )


def run_episode(
    spec: WorldSpec,
    mechanics: Mechanics | None,
    core: TinyRecursiveCore | None = None,
    explore_steps: int = 24,
    max_actions: int = 400,
    seed: int = 0,
) -> EpisodeOutcome:
    """Solve a world using the given rules, or rules inferred by the core.

    Replanning happens per level and after any divergence. Divergence is
    informative rather than fatal: it means the assumed rules mispredicted,
    and the honest response is to look again rather than keep executing a
    plan derived from a model reality has already contradicted.
    """
    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)
    notes: list[str] = []

    if mechanics is None:
        if core is None:
            raise ValueError("need either explicit mechanics or a core to infer them")
        # Explore first: the core cannot infer dynamics from a still frame.
        for _ in range(explore_steps):
            if world.done:
                break
            world.step(Action(int(rng.integers(1, 6))))
        mechanics = infer_mechanics(core, world.history)
        notes.append(f"inferred: {mechanics.summary()}")
        # Rebuild so planning starts from a clean state; exploration may have
        # collected targets or tripped switches.
        world = GridWorld(spec)
        world.reset()

    planner = BFSPlanner(max_nodes=120_000)
    executor = PlanExecutor()
    actions_used = 0
    actions_this_level = 0
    level_marker = world.history.last.levels_completed
    stalled = 0

    # Target ORDER is not readable from a frame. A frame shows which cells
    # hold targets, never which must be taken first, and under
    # `ordered_targets` aiming at the wrong one silently does nothing — the
    # real world ignores it while the model believes it was collected, so the
    # two desync at the very first action. Measured with the TRUE mechanics,
    # this alone produced "diverged at action 0" on 10 of 25 worlds.
    #
    # So when order is believed to matter, it is DISCOVERED: aim at one
    # target and see whether the target count actually fell. Counting targets
    # is the correct progress signal. An earlier attempt tested whether the
    # grid had changed at all, which any movement satisfies, and it therefore
    # scored stalls as successes and made everything worse.
    def target_count() -> int:
        grid = world.history.last.grid
        return sum(
            1
            for y in range(spec.field_size)
            for x in range(spec.field_size)
            if grid[y][x] == TARGET
        )

    while not world.done and actions_used < max_actions and stalled < 4:
        observed = read_layout(world.history.last.grid, spec.field_size)
        if not observed.targets:
            break

        candidates = (
            [(t,) for t in observed.targets]
            if mechanics.ordered_targets
            else [observed.targets]
        )
        before_targets = target_count()
        before_levels = world.history.last.levels_completed
        progressed = False

        for chosen in candidates:
            if world.done or actions_used >= max_actions:
                break

            # Re-read before every attempt. A failed candidate still MOVED
            # the agent -- it walked somewhere and found the target did not
            # yield -- so planning the next candidate from the position read
            # at the top of the loop starts the plan from where the agent no
            # longer is. Every such plan then diverges on its first action.
            # This was the entire cause of the ordered-targets failure: with
            # true mechanics, ordered worlds solved 1/25 while every other
            # rule combination solved 25/25.
            current = read_layout(world.history.last.grid, spec.field_size)
            if not current.targets:
                break
            aim = chosen if set(chosen) <= set(current.targets) else (current.targets[0],)
            attempt = LevelSpec(
                start=current.start,
                walls=current.walls,
                hazards=current.hazards,
                targets=aim,
                switches=current.switches,
                gates=current.gates,
            )
            model = GridWorldModel(_spec_with(spec, mechanics, attempt, 0), level_index=0)

            plan = planner.plan(model, start=model.init_state())
            if plan is None:
                continue

            result = executor.execute(plan, world.step, lambda: world.done)
            actions_used += result.executed
            actions_this_level += result.executed
            if world.history.last.levels_completed != level_marker:
                level_marker = world.history.last.levels_completed
                actions_this_level = 0
            if result.diverged:
                notes.append(f"diverged at action {result.diverged_at}")

            if (
                target_count() < before_targets
                or world.history.last.levels_completed != before_levels
                or world.done
            ):
                progressed = True
                break

        stalled = 0 if progressed else stalled + 1
        if not progressed:
            notes.append("no target could be collected under the believed rules")

    last = world.history.last
    return EpisodeOutcome(
        world_id=spec.world_id,
        solved=last.levels_completed >= last.win_levels,
        levels_completed=last.levels_completed,
        total_levels=last.win_levels,
        actions_used=actions_used,
        diverged=any("diverged" in n for n in notes),
        inferred=mechanics,
        truth=spec.mechanics,
        notes=notes,
    )


@dataclass
class BenchmarkResult:
    label: str
    outcomes: list[EpisodeOutcome]

    @property
    def solve_rate(self) -> float:
        return sum(o.solved for o in self.outcomes) / max(1, len(self.outcomes))

    @property
    def level_rate(self) -> float:
        done = sum(o.levels_completed for o in self.outcomes)
        total = sum(o.total_levels for o in self.outcomes)
        return done / max(1, total)

    @property
    def mechanics_accuracy(self) -> float:
        judged = [o for o in self.outcomes if o.inferred and o.truth]
        if not judged:
            return 0.0
        return sum(o.mechanics_correct for o in judged) / len(judged)

    @property
    def mean_actions(self) -> float:
        solved = [o.actions_used for o in self.outcomes if o.solved]
        return float(np.mean(solved)) if solved else 0.0

    def summary(self) -> str:
        return (
            f"{self.label:22} solved {self.solve_rate:6.1%}  "
            f"levels {self.level_rate:6.1%}  "
            f"mechanics-exact {self.mechanics_accuracy:6.1%}  "
            f"mean actions {self.mean_actions:5.1f}"
        )


def benchmark(
    specs: list[WorldSpec],
    core: TinyRecursiveCore,
    explore_steps: int = 24,
    seed: int = 0,
) -> list[BenchmarkResult]:
    """Three-way comparison: ceiling, core, floor."""
    default = Mechanics(step_distance=1, charge_period=None)

    ceiling = [run_episode(s, s.mechanics, seed=seed) for s in specs]
    inferred = [
        run_episode(s, None, core=core, explore_steps=explore_steps, seed=seed) for s in specs
    ]
    floor = [run_episode(s, default, seed=seed) for s in specs]

    return [
        BenchmarkResult("true mechanics", ceiling),
        BenchmarkResult("core-inferred", inferred),
        BenchmarkResult("default guess", floor),
    ]
