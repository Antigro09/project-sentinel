"""Experiments designed using what has already been inferred.

Evidence coverage is not a property of a world, it is a property of what you
chose to do in it. Two labels in this system make that concrete, and they
fail in opposite ways:

  charge_period    the evidence determines it in 100% of worlds; the
                   network simply could not read it
  ordered_targets  the evidence determines it in 7% of worlds under random
                   play and 23% from solution paths -- there is nothing to
                   read, and no amount of training fixes that

Ordered objectives are visible only when a collection FAILS, which requires
standing on a target that may not be taken yet. So the agent has to land
exactly on a target cell. That is where staging becomes forced rather than
elegant: collection happens only at the FINAL cell of a move, so an agent
that travels three cells per action sails straight over its goal. Measured
with a greedy prober, landings per episode fall 2.81 -> 1.19 -> 0.43 as
step_distance goes 1 -> 2 -> 3.

The movement rule therefore has to be inferred BEFORE an experiment about
order can even be aimed. Phase one moves arbitrarily and lets the verifier
pin down the dynamics; phase two plans inside the model that produced,
landing on a target and stepping off to see whether it comes back. This is
the ordinary shape of experiment design -- you need a theory of your
instrument before you can point it at anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sentinel.adapt.hypothesis import scorable_segment
from sentinel.adapt.search import SIMPLICITY_ORDER, exhaustive_search
from sentinel.core.agent import CollectOneModel, read_layout
from sentinel.env.history import History
from sentinel.env.types import Action
from sentinel.gen.grid import TARGET, GridWorld
from sentinel.gen.spec import LevelSpec, Mechanics, WorldSpec
from sentinel.plan import BFSPlanner, PlanExecutor


@dataclass(frozen=True, slots=True)
class StagedResult:
    history: History
    believed: Mechanics
    landings: int
    """Times the agent finished a move on a cell that held a target."""
    planned: bool
    """False if phase two never managed to aim, which is itself a finding."""


def _locate(grid, size: int, value: int) -> list[tuple[int, int]]:
    return [(x, y) for y in range(size) for x in range(size) if grid[y][x] == value]


def _count_landings(history: History, size: int) -> int:
    landings = 0
    previous = history.initial
    for step in history.steps:
        before, after = previous.grid, step.settled.grid
        for y in range(size):
            for x in range(size):
                if before[y][x] == TARGET and after[y][x] == 4:
                    landings += 1
        previous = step.settled
    return landings


def staged_exploration(
    spec: WorldSpec,
    seed: int = 0,
    free_steps: int = 30,
    aimed_steps: int = 40,
    order=None,
) -> StagedResult:
    """Move freely, infer the dynamics, then aim experiments with them."""
    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)
    size = spec.field_size

    for _ in range(free_steps):
        if world.done:
            break
        world.step(Action(int(rng.integers(1, 6))))

    segment = scorable_segment(world.history)
    if len(segment.steps) < 5 or world.done:
        return StagedResult(world.history, Mechanics(), _count_landings(world.history, size), False)

    observed = read_layout(segment.initial.grid, spec.field_size)
    found = exhaustive_search(
        world.history, observed, spec.field_size,
        order=list(order) if order is not None else list(SIMPLICITY_ORDER),
    )
    believed = found.mechanics

    planner = BFSPlanner(max_nodes=40_000)
    executor = PlanExecutor()
    planned = False
    spent = 0

    while spent < aimed_steps and not world.done:
        grid = world.history.last.grid
        targets = _locate(grid, size, TARGET)
        agent = _locate(grid, size, 4)
        if not targets or not agent:
            break

        current = read_layout(grid, spec.field_size)
        # Aim at one target: the model treats reaching ANY target as success,
        # so the plan is a landing rather than a full solve.
        attempt = LevelSpec(
            start=current.start, walls=current.walls, hazards=current.hazards,
            targets=current.targets, switches=current.switches, gates=current.gates,
        )
        model = CollectOneModel(
            WorldSpec(world_id=spec.world_id, seed=spec.seed, field_size=spec.field_size,
                      mechanics=believed, levels=(attempt,)),
            level_index=0,
        )
        plan = planner.plan(model, start=model.init_state())
        if plan is None:
            break
        result = executor.execute(plan, world.step, lambda: world.done)
        spent += result.executed
        planned = planned or result.executed > 0

        # Step away, so a target that was merely hidden can reappear.
        for _ in range(2):
            if world.done or spent >= aimed_steps:
                break
            world.step(Action(int(rng.integers(1, 5))))
            spent += 1

    return StagedResult(
        history=world.history,
        believed=believed,
        landings=_count_landings(world.history, size),
        planned=planned,
    )
