"""The compositional mechanics, each checked against its stated meaning.

These widen the hypothesis space from 96 rule sets toward the ~17,000 where
exhaustive verifier search stops being viable. That target matters: at 96,
brute force beats the trained core on every identifiable rule, so the space
is too small to test whether a learned prior is worth anything.

Each test pins one rule's behaviour. The last one pins the property the
whole widening depends on -- that defaults are unchanged, so every earlier
measurement still means what it meant.
"""

from __future__ import annotations

from sentinel.env.types import Action
from sentinel.gen.grid import GridState, transition_state
from sentinel.gen.spec import LevelSpec, Mechanics, WorldSpec

SIZE = 8


def _spec(mech: Mechanics, **level_kw) -> WorldSpec:
    level = LevelSpec(
        start=level_kw.pop("start", (4, 4)),
        walls=frozenset(level_kw.pop("walls", ())),
        hazards=frozenset(level_kw.pop("hazards", ())),
        targets=tuple(level_kw.pop("targets", ())),
        switches=frozenset(level_kw.pop("switches", ())),
        gates=frozenset(level_kw.pop("gates", ())),
    )
    return WorldSpec(world_id="t", seed=0, field_size=SIZE, mechanics=mech, levels=(level,))


def _state(spec: WorldSpec, x: int, y: int) -> GridState:
    level = spec.levels[0]
    return GridState(
        level=0, x=x, y=y, collected=0, remaining=frozenset(level.targets),
        charge=0, gates_open=False, dead=False, cleared=False,
    )


LEFT, RIGHT = Action(3), Action(4)


def test_block_is_the_default_and_stops_at_the_edge():
    spec = _spec(Mechanics())
    out = transition_state(_state(spec, 0, 4), LEFT, spec)
    assert (out.x, out.y) == (0, 4)


def test_wrap_reenters_from_the_far_side():
    spec = _spec(Mechanics(edge_mode="wrap"))
    out = transition_state(_state(spec, 0, 4), LEFT, spec)
    assert (out.x, out.y) == (SIZE - 1, 4)


def test_legacy_wrap_edges_still_wins():
    """Specs written before edge_mode existed must keep their meaning."""
    spec = _spec(Mechanics(wrap_edges=True))
    out = transition_state(_state(spec, 0, 4), LEFT, spec)
    assert (out.x, out.y) == (SIZE - 1, 4)


def test_bounce_reverses_at_the_edge():
    spec = _spec(Mechanics(edge_mode="bounce"))
    out = transition_state(_state(spec, 0, 4), LEFT, spec)
    assert (out.x, out.y) == (1, 4)


def test_respawn_returns_to_the_start():
    spec = _spec(Mechanics(edge_mode="respawn"), start=(6, 6))
    out = transition_state(_state(spec, 0, 4), LEFT, spec)
    assert (out.x, out.y) == (6, 6)


def test_hazard_kill_is_the_default():
    spec = _spec(Mechanics(has_hazards=True), hazards=[(3, 4)])
    out = transition_state(_state(spec, 4, 4), LEFT, spec)
    assert out.dead


def test_hazard_pushback_undoes_the_move():
    spec = _spec(Mechanics(has_hazards=True, hazard_effect="pushback"), hazards=[(3, 4)])
    out = transition_state(_state(spec, 4, 4), LEFT, spec)
    assert not out.dead and (out.x, out.y) == (4, 4)


def test_hazard_respawn_sends_the_agent_home():
    spec = _spec(
        Mechanics(has_hazards=True, hazard_effect="respawn"), start=(7, 7), hazards=[(3, 4)]
    )
    out = transition_state(_state(spec, 4, 4), LEFT, spec)
    assert not out.dead and (out.x, out.y) == (7, 7)


def test_switch_latch_does_not_close_again():
    # A target is required or the level counts as cleared and every
    # further transition returns early -- which would make this test pass
    # without the rule ever running.
    spec = _spec(
        Mechanics(has_switches=True, switch_mode="latch"),
        switches=[(3, 4), (2, 4)], targets=[(7, 7)],
    )
    once = transition_state(_state(spec, 4, 4), LEFT, spec)
    assert once.gates_open
    twice = transition_state(once, LEFT, spec)
    assert twice.gates_open, "latch must not toggle back"


def test_switch_toggle_flips_each_time():
    spec = _spec(Mechanics(has_switches=True), switches=[(3, 4), (2, 4)], targets=[(7, 7)])
    once = transition_state(_state(spec, 4, 4), LEFT, spec)
    assert once.gates_open
    twice = transition_state(once, LEFT, spec)
    assert not twice.gates_open


def test_slide_continues_until_blocked():
    spec = _spec(Mechanics(slide=True), walls=[(0, 4)])
    out = transition_state(_state(spec, 6, 4), LEFT, spec)
    assert (out.x, out.y) == (1, 4)


def test_slide_terminates_on_a_wrap_board():
    """An unbounded slide with no obstacle must not hang."""
    spec = _spec(Mechanics(slide=True, edge_mode="wrap"))
    out = transition_state(_state(spec, 4, 4), LEFT, spec)
    assert 0 <= out.x < SIZE


def test_defaults_are_byte_for_byte_the_old_behaviour():
    """Every prior measurement depends on this."""
    spec = _spec(
        Mechanics(step_distance=2, charge_period=3, has_hazards=True, has_switches=True),
        walls=[(1, 4)], hazards=[(6, 4)], switches=[(4, 2)], targets=[(4, 6)],
    )
    state = _state(spec, 4, 4)
    for action in (LEFT, RIGHT, LEFT, LEFT):
        state = transition_state(state, action, spec)
    assert (state.x, state.y) == (2, 4)
    assert state.charge == 4


def test_wait_advances_the_hidden_counter_by_default():
    """The default keeps the counter hidden; waiting must still tick it."""
    spec = _spec(Mechanics(charge_period=3), targets=[(7, 7)])
    out = transition_state(_state(spec, 4, 4), Action(5), spec)
    assert out.charge == 1
    assert (out.x, out.y) == (4, 4)


def test_wait_free_worlds_do_not_tick_on_wait():
    """These are the easier worlds: the period can be pinned by waiting."""
    spec = _spec(Mechanics(charge_period=3, wait_advances_charge=False), targets=[(7, 7)])
    out = transition_state(_state(spec, 4, 4), Action(5), spec)
    assert out.charge == 0


def test_gates_start_open_is_reflected_in_the_initial_state():
    from sentinel.gen.grid import initial_state

    shut = _spec(Mechanics(has_switches=True), switches=[(1, 1)], gates=[(2, 2)],
                 targets=[(7, 7)])
    assert not initial_state(0, shut).gates_open

    open_spec = _spec(Mechanics(has_switches=True, gates_start_open=True),
                      switches=[(1, 1)], gates=[(2, 2)], targets=[(7, 7)])
    assert initial_state(0, open_spec).gates_open


def test_slide_is_excluded_from_the_generated_space():
    """Implemented and tested, but measured at 5/40 solvable -- see
    generator.mechanic_space for why it is not generated."""
    from sentinel.gen.generator import mechanic_space

    assert not any(m.slide for m in mechanic_space(wide=True))
    assert not any(m.slide for m in mechanic_space())
