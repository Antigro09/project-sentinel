"""The generated environment distribution.

Three properties matter. Worlds must be solvable, or the corpus teaches
the teacher that giving up is sometimes correct. Generation must be
deterministic, or the corpus cannot be rebuilt from a seed list. And the
holdout must genuinely withhold — a split that leaks mechanics measures
interpolation while claiming to measure generalization.
"""

from __future__ import annotations

import pytest

from sentinel.env.types import Action
from sentinel.gen import (
    GridWorld,
    GridWorldModel,
    Mechanics,
    generate,
    generate_many,
    make_split,
    mechanic_space,
    solve_world,
)
from sentinel.gen.grid import initial_state, transition_state
from sentinel.verify import evidence_coverage, verify


def test_generation_is_deterministic() -> None:
    a = generate(42)
    b = generate(42)
    assert a is not None and b is not None
    assert a.to_json() == b.to_json()


def test_generated_worlds_are_solvable() -> None:
    """The gate: nothing unsolvable may enter the corpus."""
    for spec in generate_many(15, start_seed=100):
        assert solve_world(spec) is not None, f"{spec.world_id} is unsolvable"


def test_spec_roundtrips_through_json() -> None:
    spec = generate(7)
    assert spec is not None
    from sentinel.gen.spec import WorldSpec

    assert WorldSpec.from_json(spec.to_json()).to_json() == spec.to_json()


def test_exact_model_is_exact() -> None:
    """The reference model must be correct by construction, or every
    measurement taken against it inherits an unknown error."""
    for spec in generate_many(6, start_seed=300):
        solution = solve_world(spec)
        assert solution is not None
        world = GridWorld(spec)
        world.reset()
        world.play(solution)
        report = verify(GridWorldModel(spec), world.history)
        assert report.is_perfect, f"{spec.world_id}: {report.summary()}"


def test_solution_produces_falsifiable_evidence() -> None:
    """Playing the solution must exercise every channel the verifier scores."""
    for spec in generate_many(6, start_seed=400):
        solution = solve_world(spec)
        assert solution is not None
        world = GridWorld(spec)
        world.reset()
        world.play(solution)
        coverage = evidence_coverage(world.history)
        assert coverage.has_level_boundary, f"{spec.world_id} never completes a level"


def test_hidden_state_makes_worlds_non_markov() -> None:
    """The charge mechanic must actually break the Markov property.

    If identical grids always had identical successors, a grid-only model
    would be correct and the central difficulty would be absent from the
    training distribution.
    """
    mech = Mechanics(step_distance=1, charge_period=3)
    spec = generate(11, mechanics=mech)
    assert spec is not None

    state = initial_state(0, spec)
    distances = []
    for _ in range(6):
        nxt = transition_state(state, Action(4), spec)
        distances.append(abs(nxt.x - state.x))
        state = nxt

    assert len(set(distances)) > 1 or 2 in distances, (
        "charge_period should make some moves travel two cells"
    )


def test_mechanic_space_is_varied() -> None:
    space = mechanic_space()
    assert len(space) >= 20
    assert len({m.summary() for m in space}) == len(space), "duplicate combinations"
    assert any(m.charge_period for m in space)
    assert any(m.has_switches for m in space)
    assert any(m.ordered_targets for m in space)


def test_split_withholds_mechanics_completely() -> None:
    """THE SPLIT INTEGRITY CHECK.

    A withheld combination must appear in *no* training world at any seed.
    If it leaks, Phase 3's generalization number is measuring interpolation
    and would overstate the result.
    """
    split = make_split(
        n_train=24, n_holdout_seed=6, n_holdout_mechanics=6, withhold=4, seed=1
    )
    withheld = {m.summary() for m in split.withheld}
    assert withheld, "nothing was withheld"

    train_mechanics = {s.mechanics.summary() for s in split.train}
    assert not (train_mechanics & withheld), (
        f"withheld mechanics leaked into training: {train_mechanics & withheld}"
    )

    holdout_mechanics = {s.mechanics.summary() for s in split.holdout_mechanics}
    assert holdout_mechanics <= withheld


def test_split_worlds_are_disjoint() -> None:
    split = make_split(n_train=20, n_holdout_seed=6, n_holdout_mechanics=6, seed=2)
    train_ids = {s.world_id for s in split.train}
    seed_ids = {s.world_id for s in split.holdout_seed}
    mech_ids = {s.world_id for s in split.holdout_mechanics}

    assert not (train_ids & seed_ids)
    assert not (train_ids & mech_ids)
    assert not (seed_ids & mech_ids)


@pytest.mark.parametrize("seed", [0, 5, 13, 29])
def test_worlds_have_targets_on_every_level(seed: int) -> None:
    spec = generate(seed)
    assert spec is not None
    for level in spec.levels:
        assert level.targets, "a level with no targets is trivially complete"
