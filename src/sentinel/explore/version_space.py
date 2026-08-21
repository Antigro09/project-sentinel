"""The set of hypotheses still standing, and the action that would split it.

Two gaps close with one piece of machinery.

**Knowing what you don't know.** A single best guess cannot say how sure it
is. The set of hypotheses still consistent with the evidence can: its size
IS the uncertainty, and when it stops shrinking the evidence has run out.
Measured on the compositional benchmark, one episode of arbitrary actions
takes 5,760 candidates down to about two -- so the useful question is never
"which is right" but "what would separate the two that remain".

**Acting to find out.** The identifiability theorem for controlled world
models says a model is unrecoverable without *conditional action
excitation*: action variety GIVEN the state. Random play gives plenty of
marginal action variety and almost none conditional on the rare states that
matter, which is exactly why `ordered_targets` is determined by the evidence
in only 6% of worlds while the hidden counter is determined in 100%. No
amount of training fixes that; only a different choice of actions does.

So the action to take is the one whose outcome the survivors most disagree
about -- maximum disagreement, or Query-by-Committee. Simulating every
survivor one step forward is cheap because the survivor set is small, and
the disagreement count is a direct estimate of the information the action
would yield.

Comparison is on STATE rather than rendered frames, for the reason
documented in `adapt.search.equivalence_search`: rendering 64x64 per
candidate per step does not finish, while the render is a pure function of
a few variables that can be compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sentinel.adapt.hypothesis import mechanics_from_classes
from sentinel.adapt.search import SIMPLICITY_ORDER
from sentinel.env.types import Action, Observation
from sentinel.gen.grid import AGENT, GATE_CLOSED, GATE_OPEN, LEGAL_ACTIONS, TARGET, GridWorldModel
from sentinel.gen.spec import LevelSpec, WorldSpec


def observed_facts(grid, field_size: int):
    """What a frame actually shows: agent, visible targets, gate state."""
    agent = None
    targets = set()
    gates_open = None
    for y in range(field_size):
        row = grid[y]
        for x in range(field_size):
            v = row[x]
            if v == AGENT:
                agent = (x, y)
            elif v == TARGET:
                targets.add((x, y))
            elif v == GATE_OPEN:
                gates_open = True
            elif v == GATE_CLOSED:
                gates_open = False
    return agent, frozenset(targets), gates_open


def state_key(state) -> tuple:
    """Everything about a state that a frame could reveal.

    A target under the agent is occluded, so it must be removed before this
    can be compared with what was observed.
    """
    here = (state.x, state.y)
    visible = frozenset(t for t in state.remaining if t != here)
    return (here, visible, bool(state.gates_open), bool(state.dead))


@dataclass
class VersionSpace:
    """Hypotheses consistent with everything seen so far."""

    field_size: int
    live: list[tuple[tuple[int, ...], object, object]]

    @classmethod
    def over(
        cls,
        observed: LevelSpec,
        field_size: int,
        candidates: Sequence[tuple[int, ...]] | None = None,
    ) -> VersionSpace:
        pool = list(candidates) if candidates is not None else list(SIMPLICITY_ORDER)
        live = []
        for classes in pool:
            spec = WorldSpec(
                world_id="vs", seed=0, field_size=field_size,
                mechanics=mechanics_from_classes(classes), levels=(observed,),
            )
            model = GridWorldModel(spec)
            live.append((classes, model, model.init_state()))
        return cls(field_size=field_size, live=live)

    def __len__(self) -> int:
        return len(self.live)

    @property
    def settled(self) -> bool:
        """True when the evidence can no longer distinguish what remains."""
        return len(self.live) <= 1

    def observe(self, action: Action, frame: Observation) -> int:
        """Drop every hypothesis this transition refutes. Returns survivors.

        **Nothing is merged, and that correction cost a real bug.** An
        earlier version collapsed hypotheses whose predicted state matched,
        keeping one representative per behaviour -- which took 5,760
        candidates to 2 in a single step and looked like a triumph. It was
        wrong: agreeing on step 1 does not mean agreeing on step 5, so
        merging discards hypotheses that would have been distinguished
        later. The check that caught it was asking whether the TRUE rule set
        was still in the set; it was not.

        Two hypotheses are interchangeable only if their mechanics are
        identical, so the only sound pruning is refutation. The saving is
        real anyway: a refuted hypothesis is never simulated again, and most
        die within the first few steps.
        """
        want_agent, want_targets, want_gates = observed_facts(frame.grid, self.field_size)
        survivors = []
        for classes, model, state in self.live:
            try:
                nxt = model.transition(state, action)
            except Exception:
                continue
            here = (nxt.x, nxt.y)
            visible = frozenset(t for t in nxt.remaining if t != here)
            if here != want_agent or visible != want_targets:
                continue
            if want_gates is not None and bool(nxt.gates_open) != want_gates:
                continue
            survivors.append((classes, model, nxt))
        if survivors:
            self.live = survivors
        return len(self.live)

    def disagreement(self, action: Action) -> int:
        """How many distinct outcomes the survivors predict for this action.

        1 means every survivor agrees and the action would teach nothing.
        Higher is a better experiment.
        """
        outcomes = set()
        for _, model, state in self.live:
            try:
                outcomes.add(state_key(model.transition(state, action)))
            except Exception:
                continue
        return len(outcomes)

    def best_action(self, actions: Sequence[int] = LEGAL_ACTIONS) -> tuple[Action, int]:
        """The action the survivors most disagree about.

        This is the whole point: an action every hypothesis predicts
        identically cannot change what is believed, however sensible it
        looks. Ties go to the lower action id so the choice is
        deterministic and reproducible.
        """
        best, best_split = Action(actions[0]), -1
        for aid in actions:
            split = self.disagreement(Action(aid))
            if split > best_split:
                best, best_split = Action(aid), split
        return best, best_split

    def candidates(self) -> list[tuple[int, ...]]:
        return [classes for classes, _, _ in self.live]


def information_gain_history(spec, seed: int = 0, steps: int = 60, epsilon: float = 0.15):
    """Explore by taking the action the surviving hypotheses most disagree about.

    Random play collects evidence about whatever it happens to bump into.
    This collects evidence about what is still UNKNOWN: at every step the
    survivors are simulated one move forward, and the action producing the
    most distinct predictions is taken, because an action every hypothesis
    agrees on cannot change what is believed however sensible it looks.

    `epsilon` keeps a fraction of moves random. Pure disagreement-chasing can
    park the agent in a corner where two hypotheses differ forever about a
    wall it keeps walking into, and the identifiability result is explicit
    that some unconditional action variety is what guarantees coverage in
    the first place.
    """
    import numpy as np

    from sentinel.core.agent import read_layout
    from sentinel.gen.grid import GridWorld

    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)

    observed = read_layout(world.history.last.grid, spec.field_size)
    space = VersionSpace.over(observed, spec.field_size)

    for _ in range(steps):
        if world.done:
            break
        if space.settled or rng.random() < epsilon:
            action = Action(int(rng.integers(1, 6)))
        else:
            action, _ = space.best_action()
        world.step(action)
        space.observe(action, world.history.last)

    return world.history


def planned_information_gain_history(spec, seed: int = 0, steps: int = 60, depth: int = 6):
    """Plan TOWARD disagreement, rather than picking the next splitting move.

    One-step information gain is myopic, and the measurement shows exactly
    where that bites. Choosing the immediately most-splitting action takes
    `edge_mode` identifiability from 91% to 100% -- walking into a wall
    settles it in one move -- while `ordered_targets` stays at 9%, unchanged
    from random play, because nothing is learned about collection order
    until the agent is STANDING on a target, and no single step points that
    way from across the board.

    So search over short action SEQUENCES instead: simulate each under every
    surviving hypothesis and take the one whose end states differ most. A
    walk onto a target emerges by itself, because ordered and unordered
    survivors are precisely the ones that disagree about what happens there
    -- the experiment is discovered from the disagreement rather than
    hand-written, which is what `staged_exploration` had to do.
    """
    import itertools

    import numpy as np

    from sentinel.core.agent import read_layout
    from sentinel.gen.grid import GridWorld

    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)

    observed = read_layout(world.history.last.grid, spec.field_size)
    space = VersionSpace.over(observed, spec.field_size)

    # Sequences to consider. Enumerating all 4^depth is far too many, so
    # take a sample of random walks plus every straight run -- straight runs
    # matter because they are how an agent crosses a board to reach a target.
    # WAIT (action 5) must be in the alphabet. Leaving it out cost
    # `wait_advances_charge` its identifiability outright -- 78% under random
    # play down to 18% -- because whether waiting ticks the hidden counter
    # can only be tested by waiting, and a planner chasing displacement
    # never does. A purposeful explorer that cannot sit still is blind to
    # every rule about sitting still.
    def sequences():
        for aid in (1, 2, 3, 4):
            for n in range(1, depth + 1):
                yield [aid] * n
        for n in (1, 2, 3):
            yield [5] * n
        yield [5, 1]
        yield [5, 4]
        for _ in range(40):
            n = int(rng.integers(2, depth + 1))
            yield [int(a) for a in rng.integers(1, 6, size=n)]

    spent = 0
    while spent < steps and not world.done:
        if space.settled:
            world.step(Action(int(rng.integers(1, 6))))
            spent += 1
            continue

        best_seq, best_split = None, 1
        for seq in sequences():
            ends = set()
            for _, model, state in space.live:
                s = state
                try:
                    for aid in seq:
                        s = model.transition(s, Action(aid))
                except Exception:
                    break
                ends.add(state_key(s))
            if len(ends) > best_split:
                best_seq, best_split = seq, len(ends)

        if best_seq is None:
            best_seq = [int(rng.integers(1, 6))]
        for aid in best_seq:
            if world.done or spent >= steps:
                break
            action = Action(aid)
            world.step(action)
            space.observe(action, world.history.last)
            spent += 1

    return world.history


def disagreement_planning_history(spec, seed: int = 0, steps: int = 60, max_nodes: int = 4000):
    """Search the believed world for a state the survivors disagree about, and go there.

    Enumerating short action sequences is not enough, and the measurement
    says exactly why: `ordered_targets` stays near 18% because collection
    order cannot be learned until the agent is STANDING on a target, and a
    target is routinely eight or more moves away. A depth-6 sequence search
    never arrives.

    So plan. Breadth-first search inside the simplest surviving model
    enumerates reachable states with the paths that reach them; each path is
    then replayed under every survivor and scored by how many distinct end
    states it produces. The winner is the experiment worth running.

    Nothing here mentions targets, or order, or any other mechanic. Walking
    onto a target emerges because ordered and unordered survivors are
    precisely the hypotheses that disagree about what happens there -- the
    experiment is DISCOVERED from disagreement rather than hand-written,
    which is what `staged_exploration` had to do and what makes that
    function narrow.
    """
    from collections import deque

    import numpy as np

    from sentinel.core.agent import read_layout
    from sentinel.gen.grid import GridWorld

    world = GridWorld(spec)
    world.reset()
    rng = np.random.default_rng(seed)

    observed = read_layout(world.history.last.grid, spec.field_size)
    space = VersionSpace.over(observed, spec.field_size)

    spent = 0
    while spent < steps and not world.done:
        if space.settled:
            world.step(Action(int(rng.integers(1, 6))))
            spent += 1
            continue

        # Navigate with the simplest survivor; it only has to propose paths,
        # and every proposal is judged by all survivors.
        guide_classes, guide_model, guide_state = space.live[0]
        # Dedup on a FULLER key than state_key here. state_key deliberately
        # omits the hidden counter because it compares against what a frame
        # shows, and a frame cannot show it -- but waiting changes only that
        # counter, so BFS deduping on the visible key prunes every wait as
        # "already seen". That silently removed wait from every plan and
        # took `wait_advances_charge` from 82% to 26%.
        def search_key(st):
            return state_key(st) + (st.charge,)

        frontier = deque([(guide_state, [])])
        seen = {search_key(guide_state)}
        paths = []
        while frontier and len(seen) < max_nodes:
            state, path = frontier.popleft()
            if len(path) >= 12:
                continue
            for aid in (1, 2, 3, 4, 5):
                try:
                    nxt = guide_model.transition(state, Action(aid))
                except Exception:
                    continue
                key = search_key(nxt)
                if key in seen:
                    continue
                seen.add(key)
                new_path = path + [aid]
                paths.append(new_path)
                frontier.append((nxt, new_path))

        # Score by the ENTROPY of the split, not the number of groups.
        #
        # Counting distinct outcomes sounds like the same thing and is not,
        # and the difference is why `ordered_targets` would not move.
        # Movement rules -- five charge periods, three step distances, four
        # edge modes -- shatter a long walk into dozens of distinct end
        # positions, so a raw count always prefers another movement
        # experiment. But movement is settled within a few moves, and after
        # that those dozens of groups are one survivor apiece: a large count
        # that eliminates almost nothing.
        #
        # Ordered versus unordered is a two-way split, and a BALANCED
        # two-way split halves the survivors. Entropy prefers that, which is
        # the standard result from Bayesian experimental design: what
        # matters is expected reduction in uncertainty, not how many
        # outcomes are possible.
        import math

        best_path, best_score = None, -1.0
        for path in paths:
            groups: dict = {}
            for _, model, state in space.live:
                s = state
                try:
                    for aid in path:
                        s = model.transition(s, Action(aid))
                except Exception:
                    break
                k = state_key(s)
                groups[k] = groups.get(k, 0) + 1
            total = sum(groups.values())
            if total < 2 or len(groups) < 2:
                continue
            entropy = -sum((n / total) * math.log(n / total) for n in groups.values())
            # Among equally informative experiments prefer the shorter walk;
            # a real action is the currency the benchmark charges for.
            score = entropy - 0.01 * len(path)
            if score > best_score:
                best_path, best_score = path, score

        if best_path is None:
            best_path = [int(rng.integers(1, 6))]
        for aid in best_path:
            if world.done or spent >= steps:
                break
            action = Action(aid)
            world.step(action)
            space.observe(action, world.history.last)
            spent += 1

    return world.history
