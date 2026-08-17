"""Planning and execution.

The Phase 1 end-to-end claim: a hand-written model, searched with BFS,
solves a game with zero learning and zero LLM calls. And the safety
property that makes plans usable at all — reality is checked after every
single action, so a wrong model is caught in a few steps rather than
followed off a cliff.
"""

from __future__ import annotations

import pytest

from sentinel.env import Runner
from sentinel.env.types import Action
from sentinel.gen import (
    GridOnlyToyModel,
    ToyModel,
    ToyWorld,
    default_levels,
)
from sentinel.plan import BFSPlanner, PlanExecutor, SearchStats
from sentinel.wm import Outcome
from sentinel.wm.ls20 import Ls20Model
from sentinel.wm.reference import AbstainModel


@pytest.fixture(scope="module")
def levels():
    return default_levels()


def test_planner_solves_every_toy_level(levels) -> None:
    """THE PHASE 1 END-TO-END GATE.

    Model -> plan -> act -> solved, with nothing learned and nothing asked
    of an LLM.
    """
    planner = BFSPlanner()
    world = ToyWorld(levels)
    world.reset()
    actions_used = 0

    for level_index in range(len(levels)):
        model = ToyModel(levels)
        model._level_index = level_index
        plan = planner.plan(model, start=model.init_state())
        assert plan is not None, f"no plan found for level {level_index}"

        result = PlanExecutor().execute(plan, world.step, lambda: world.done)
        assert not result.diverged, f"level {level_index}: {result.summary()}"
        actions_used += result.executed

    final = world.history.last
    assert final.levels_completed == len(levels)
    assert final.state.value == "WIN"
    assert actions_used < 200


def test_plan_is_shortest(levels) -> None:
    """BFS must return a shortest path: the ARC score squares action count."""
    model = ToyModel(levels)
    planner = BFSPlanner()
    plan = planner.plan(model, start=model.init_state())
    assert plan is not None

    shorter = BFSPlanner(max_depth=len(plan) - 1)
    assert shorter.plan(model, start=model.init_state()) is None


def test_wrong_model_is_caught_by_reality(levels) -> None:
    """A model that denies hidden state plans confidently and is refuted fast.

    This is the architecture's safety property in one test: the plan is
    advisory, reality is authoritative, and the disagreement is detected
    within a few real actions rather than at the end.
    """
    world = ToyWorld(levels)
    world.reset()
    bad = GridOnlyToyModel(levels)

    plan = BFSPlanner().plan(bad, start=bad.init_state())
    assert plan is not None, "the wrong model should still produce a plan"

    result = PlanExecutor().execute(plan, world.step, lambda: world.done)
    assert result.diverged, "reality must refute a model that denies hidden state"
    assert result.diverged_at is not None and result.diverged_at < 5
    assert result.mismatched_cells > 0


def test_level_boundary_is_not_a_divergence(levels) -> None:
    """Completing a level swaps in an unseen layout; that is success."""
    world = ToyWorld(levels)
    world.reset()
    model = ToyModel(levels)
    plan = BFSPlanner().plan(model, start=model.init_state())
    assert plan is not None

    result = PlanExecutor().execute(plan, world.step, lambda: world.done)
    assert result.completed
    assert not result.diverged


def test_planner_prunes_fatal_moves(levels) -> None:
    """Plans must never route through a state the model calls game over."""
    model = ToyModel(levels)
    model._level_index = 2  # the hazard gauntlet
    stats = SearchStats()
    plan = BFSPlanner().plan(model, start=model.init_state(), stats=stats)

    assert plan is not None
    assert stats.pruned_dead > 0, "hazards should have pruned some branches"

    state = model.init_state()
    for action in plan.actions:
        state = model.transition(state, action)
        assert model.outcome(state) is not Outcome.GAME_OVER


def test_unplannable_model_returns_none() -> None:
    """A model that never reports success yields no plan, rather than hanging."""
    planner = BFSPlanner(max_nodes=500)
    assert planner.plan(AbstainModel(), start=0) is None


def test_search_respects_node_budget(levels) -> None:
    stats = SearchStats()
    planner = BFSPlanner(max_nodes=10)
    planner.plan(ToyModel(levels), start=ToyModel(levels).init_state(), stats=stats)
    assert stats.hit_limit
    assert stats.expanded <= 11


def test_predictions_accompany_every_action(levels) -> None:
    """Every action must carry a predicted frame, or execution cannot check it."""
    model = ToyModel(levels)
    plan = BFSPlanner().plan(model, start=model.init_state())
    assert plan is not None
    assert len(plan.predicted) == len(plan.actions)


def test_partial_ls20_model_plans_on_the_real_game() -> None:
    """The partial model is incomplete but must still be usable for search.

    It cannot detect level completion — it says so by always reporting
    ONGOING — so no plan to a goal exists. What must hold is that its
    movement model is sound enough to simulate forward without crashing,
    which is the precondition for a proposer to repair it later.
    """
    runner = Runner("ls20", seed=0)
    first = runner.reset()
    model = Ls20Model(first)

    state = model.init_state()
    for action_id in (4, 4, 2, 3, 1):
        state = model.transition(state, Action(action_id))
        rendered = model.render(state)
        assert len(rendered) == 64 and len(rendered[0]) == 64

    assert BFSPlanner(max_nodes=2000).plan(model, start=model.init_state()) is None


def test_partial_model_beats_knowing_nothing() -> None:
    """Ranking must prefer understanding dynamics over predicting stasis."""
    import random

    from sentinel.verify import verify
    from sentinel.wm.reference import StaticModel

    runner = Runner("ls20", seed=0)
    first = runner.reset()
    rng = random.Random(11)
    for _ in range(80):
        if runner.done:
            break
        runner.step(Action(rng.choice([1, 2, 3, 4])))
    history = runner.history

    partial = verify(Ls20Model(first), history)
    static = verify(StaticModel(first), history)
    nothing = verify(AbstainModel(), history)

    assert partial.fitness > static.fitness > nothing.fitness
    assert partial.transition_match > static.transition_match


# --------------------------------------------------------------------------
# The closed loop
# --------------------------------------------------------------------------


def test_agent_model_renders_every_target() -> None:
    """THE ORDERED-TARGETS REGRESSION TEST.

    When aiming at one target at a time, the model must still render the
    whole level. Building a level containing only the target being pursued
    makes the renderer omit the others, so the model predicts background
    where reality shows a target and the plan diverges before the agent has
    moved. That single bug held ordered worlds to 1/25 while every other
    rule combination solved 25/25.
    """
    from sentinel.core.agent import CollectOneModel, read_layout
    from sentinel.gen import GridWorld, generate
    from sentinel.gen.spec import LevelSpec, WorldSpec
    from sentinel.verify.verifier import compare

    spec = next(
        s
        for s in (generate(i) for i in range(40))
        if s and s.mechanics.ordered_targets and len(s.levels[0].targets) > 1
    )
    world = GridWorld(spec)
    world.reset()
    observed = read_layout(world.history.last.grid, spec.field_size)

    # Aim at the second target; the first must still be drawn.
    chosen = (observed.targets[1],)
    rest = tuple(t for t in observed.targets if t not in set(chosen))
    level = LevelSpec(
        start=observed.start,
        walls=observed.walls,
        hazards=observed.hazards,
        targets=chosen + rest,
        switches=observed.switches,
        gates=observed.gates,
    )
    believed = WorldSpec(
        world_id=spec.world_id,
        seed=spec.seed,
        field_size=spec.field_size,
        mechanics=spec.mechanics,
        levels=(level,),
    )
    model = CollectOneModel(believed, level_index=0)
    _, matched = compare(model.render(model.init_state()), world.history.last.grid)
    assert matched, "model must reproduce the opening frame exactly, all targets included"


def test_agent_solves_worlds_given_true_rules() -> None:
    """With correct rules the loop must actually solve things."""
    from sentinel.core.agent import run_episode
    from sentinel.gen import generate

    specs = [s for s in (generate(i) for i in range(30)) if s is not None][:12]
    solved = sum(run_episode(s, s.mechanics, seed=0).solved for s in specs)
    assert solved >= 4, f"only {solved}/{len(specs)} solved with true mechanics"
