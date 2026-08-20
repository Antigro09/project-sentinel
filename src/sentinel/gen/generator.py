"""Procedural generation of solvable worlds.

Two properties matter more than variety.

**Every emitted world is solvable**, checked by BFS against its own exact
model before it leaves this module. An unsolvable world in the corpus
would train the teacher that abandoning a problem is sometimes the right
answer, which is the one lesson we cannot afford to teach it.

**The held-out split withholds mechanics, not just seeds.** Holding out
random seeds only measures interpolation: the model has seen every rule
already and merely rearranged. The interesting question is whether a
system that has learned switches and hidden state separately can handle a
world that combines them for the first time. So the split is available
along both axes, and Phase 3's real number should come from the mechanics
holdout.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterator, Sequence

from .grid import solve_world
from .spec import LevelSpec, Mechanics, WorldSpec

MIN_FIELD = 7
MAX_FIELD = 13


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Bounds on what may be produced."""

    min_levels: int = 2
    max_levels: int = 4
    min_targets: int = 1
    max_targets: int = 3
    wall_density: float = 0.12
    hazard_density: float = 0.05
    max_attempts: int = 40
    """Layout retries before giving up on a seed. Sparse walls make most
    layouts solvable, so this is rarely approached."""


def mechanic_space(wide: bool = False) -> list[Mechanics]:
    """The enumerated combinations the generator draws from.

    Kept explicit rather than sampled independently so the train/holdout
    split can withhold *specific combinations* and we can say exactly which
    ones the system had never seen.

    `wide=False` is the original 26-combination space and remains the
    default so existing corpora, splits and measurements keep their
    meaning. `wide=True` is the compositional space.

    **Why the wide space exists.** At 96 encodable rule sets, exhaustive
    verifier search identifies a world's rules in 1.7 seconds and beats the
    trained core on every rule the evidence determines. Nothing there needs
    a learned prior, so the setup cannot test whether one is worth having.
    One verifier replay costs 17.5ms, so against the plan's five-minute
    budget per novel environment, brute force only stops being viable past
    roughly 17,000 hypotheses. That is the number this is walking toward.
    """
    if not wide:
        combos: list[Mechanics] = []
        for charge in (None, 3, 4):
            for hazards in (False, True):
                for switches in (False, True):
                    for ordered in (False, True):
                        combos.append(
                            Mechanics(
                                step_distance=1,
                                charge_period=charge,
                                has_hazards=hazards,
                                has_switches=switches,
                                ordered_targets=ordered,
                            )
                        )
        combos.append(Mechanics(step_distance=2, charge_period=None))
        combos.append(Mechanics(step_distance=1, charge_period=3, wrap_edges=True))
        return combos

    # `slide` is deliberately absent. It is implemented and tested, but a
    # sliding agent cannot choose where to stop, so a target that is not
    # against a wall is usually unreachable: measured at 5/40 solvable
    # against 40/40 without it, and the failing combinations cost 60s each
    # to reject. A mechanic that almost never yields a solvable world is not
    # really in the space, and pretending otherwise would inflate the
    # hypothesis count without making the problem harder. Reinstating it
    # needs a level generator that places targets at wall-adjacent cells.
    wide_combos: list[Mechanics] = []
    for step in (1, 2, 3):
        for charge in (None, 2, 3, 4, 5):
            for edge in ("block", "wrap", "bounce", "respawn"):
                for hazard in (None, "kill", "pushback", "respawn"):
                    for switch in (None, "toggle", "latch"):
                        for ordered in (False, True):
                            for gates_open_at_start in (False, True):
                                for wait_ticks in (True, False):
                                    wide_combos.append(
                                        Mechanics(
                                            step_distance=step,
                                            charge_period=charge,
                                            edge_mode=edge,
                                            has_hazards=hazard is not None,
                                            hazard_effect=hazard or "kill",
                                            has_switches=switch is not None,
                                            switch_mode=switch or "toggle",
                                            ordered_targets=ordered,
                                            gates_start_open=gates_open_at_start,
                                            wait_advances_charge=wait_ticks,
                                        )
                                    )
    return wide_combos


def _free_cells(size: int, taken: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (x, y) for y in range(size) for x in range(size) if (x, y) not in taken
    ]


def _make_level(
    rng: random.Random, size: int, mech: Mechanics, cfg: GeneratorConfig
) -> LevelSpec:
    taken: set[tuple[int, int]] = set()

    start = (rng.randrange(size), rng.randrange(size))
    taken.add(start)

    n_walls = int(size * size * cfg.wall_density)
    walls: set[tuple[int, int]] = set()
    for _ in range(n_walls):
        candidates = _free_cells(size, taken | walls)
        if not candidates:
            break
        walls.add(rng.choice(candidates))

    hazards: set[tuple[int, int]] = set()
    if mech.has_hazards:
        n_haz = max(1, int(size * size * cfg.hazard_density))
        for _ in range(n_haz):
            candidates = _free_cells(size, taken | walls | hazards)
            if not candidates:
                break
            hazards.add(rng.choice(candidates))

    switches: set[tuple[int, int]] = set()
    gates: set[tuple[int, int]] = set()
    if mech.has_switches:
        candidates = _free_cells(size, taken | walls | hazards)
        if candidates:
            switches.add(rng.choice(candidates))
        for _ in range(max(1, size // 4)):
            candidates = _free_cells(size, taken | walls | hazards | switches | gates)
            if not candidates:
                break
            gates.add(rng.choice(candidates))

    n_targets = rng.randint(cfg.min_targets, cfg.max_targets)
    targets: list[tuple[int, int]] = []
    blockedset = taken | walls | hazards | switches | gates
    for _ in range(n_targets):
        candidates = _free_cells(size, blockedset | set(targets))
        if not candidates:
            break
        targets.append(rng.choice(candidates))

    return LevelSpec(
        start=start,
        walls=frozenset(walls),
        hazards=frozenset(hazards),
        targets=tuple(targets),
        switches=frozenset(switches),
        gates=frozenset(gates),
    )


def generate(
    seed: int,
    mechanics: Mechanics | None = None,
    cfg: GeneratorConfig | None = None,
) -> WorldSpec | None:
    """Produce one solvable world, or None if this seed could not yield one.

    Deterministic: the same (seed, mechanics, cfg) always gives the same
    world, which is what makes the corpus reproducible from a seed list.
    """
    cfg = cfg or GeneratorConfig()
    rng = random.Random(seed)

    space = mechanic_space()
    mech = mechanics if mechanics is not None else rng.choice(space)
    size = rng.randint(MIN_FIELD, MAX_FIELD)
    n_levels = rng.randint(cfg.min_levels, cfg.max_levels)

    for attempt in range(cfg.max_attempts):
        levels = tuple(_make_level(rng, size, mech, cfg) for _ in range(n_levels))
        if any(not lv.targets for lv in levels):
            continue
        spec = WorldSpec(
            world_id=f"w{seed:06d}",
            seed=seed,
            field_size=size,
            mechanics=mech,
            levels=levels,
        )
        if solve_world(spec) is not None:
            return spec
    return None


def generate_many(
    count: int,
    start_seed: int = 0,
    mechanics_pool: Sequence[Mechanics] | None = None,
    cfg: GeneratorConfig | None = None,
    max_seed_span: int | None = None,
) -> list[WorldSpec]:
    """Generate `count` solvable worlds, skipping seeds that fail."""
    out: list[WorldSpec] = []
    seed = start_seed
    limit = start_seed + (max_seed_span or count * 10)

    while len(out) < count and seed < limit:
        mech = None
        if mechanics_pool:
            mech = mechanics_pool[seed % len(mechanics_pool)]
        spec = generate(seed, mechanics=mech, cfg=cfg)
        if spec is not None:
            out.append(spec)
        seed += 1
    return out



LABEL_VIEW = (
    ("step_distance", lambda m: m.step_distance),
    ("charge_period", lambda m: m.charge_period or 0),
    ("edge_mode", lambda m: m.effective_edge_mode()),
    ("hazards", lambda m: m.hazard_effect if m.has_hazards else "none"),
    ("switches", lambda m: m.switch_mode if m.has_switches else "none"),
    ("ordered_targets", lambda m: m.ordered_targets),
    ("gates_start_open", lambda m: m.gates_start_open),
    ("wait_advances_charge", lambda m: m.wait_advances_charge),
)
"""How a rule set looks to an evaluation, one entry per thing being judged."""


def _confounding(combos: Sequence[Mechanics]) -> float:
    """Worst pairwise predictability between labels across these rule sets.

    1.0 means some label is perfectly predictable from another, so scoring
    well on it proves nothing about the label itself. This is not a
    theoretical worry: the original random holdout drew four combinations
    in which `charge_period` was *exactly* `has_hazards`, so a model that
    only ever detected hazards -- coloured cells, plainly visible --
    scored as though it had inferred a counter that appears in no frame.
    """
    if len(combos) < 2:
        return 1.0
    worst = 0.0
    views = [[fn(m) for m in combos] for _, fn in LABEL_VIEW]
    for i, a in enumerate(views):
        if len(set(a)) < 2:
            return 1.0  # a constant label measures nothing at all
        for j, b in enumerate(views):
            if i >= j:
                continue
            # Fraction of combos explained by the best map from b to a.
            groups: dict = {}
            for va, vb in zip(a, b):
                groups.setdefault(vb, []).append(va)
            hits = sum(max(Counter(g).values()) for g in groups.values())
            worst = max(worst, hits / len(combos))
    return worst


def balanced_withhold(
    space: Sequence[Mechanics],
    count: int,
    rng: random.Random,
    attempts: int = 4000,
) -> tuple[Mechanics, ...]:
    """Pick holdout combinations whose labels vary independently.

    Random sampling is what produced a benchmark where four of six labels
    were unmeasurable -- two constant, one perfectly confounded, one absent
    from training. Choosing the subset instead costs a few thousand cheap
    evaluations and is the difference between a number that means something
    and a number that does not.
    """
    best: tuple[Mechanics, ...] = tuple(rng.sample(list(space), min(count, len(space))))
    best_score = _confounding(best)
    for _ in range(attempts):
        pick = tuple(rng.sample(list(space), min(count, len(space))))
        score = _confounding(pick)
        if score < best_score:
            best, best_score = pick, score
            if best_score <= 0.5 + 1e-9:
                break
    return best


@dataclass(frozen=True, slots=True)
class Split:
    """Train and held-out worlds, with the holdout basis recorded."""

    train: tuple[WorldSpec, ...]
    holdout_seed: tuple[WorldSpec, ...]
    """Unseen seeds, seen mechanics — measures interpolation."""
    holdout_mechanics: tuple[WorldSpec, ...]
    """Unseen mechanic combinations — measures actual generalization."""
    withheld: tuple[Mechanics, ...]

    def summary(self) -> str:
        return (
            f"train={len(self.train)} "
            f"holdout_seed={len(self.holdout_seed)} "
            f"holdout_mechanics={len(self.holdout_mechanics)} "
            f"withheld_combos={len(self.withheld)}"
        )


def make_split(
    n_train: int,
    n_holdout_seed: int,
    n_holdout_mechanics: int,
    withhold: int = 4,
    seed: int = 0,
    cfg: GeneratorConfig | None = None,
    wide: bool = False,
) -> Split:
    """Build a corpus split that withholds whole mechanic combinations.

    `withhold` combinations never appear in training at any seed. Those are
    the worlds Phase 3 should be judged on: everything else only asks
    whether the system can rearrange rules it has already been taught.
    """
    space = mechanic_space(wide=wide)
    rng = random.Random(seed)
    withheld = balanced_withhold(space, withhold, rng)
    train_pool = [m for m in space if m not in withheld]
    # Shuffle before generation. `generate_many` walks the pool by
    # `seed % len(pool)`, so an unshuffled pool is consumed in enumeration
    # order -- and the space is built with step_distance as the outer loop,
    # which put every step=3 rule set beyond the 2000-world training budget
    # and left that class absent from training entirely.
    rng.shuffle(train_pool)

    train = generate_many(n_train, start_seed=0, mechanics_pool=train_pool, cfg=cfg)

    # Seed holdout starts well past the training span so the seeds cannot
    # collide even when many are skipped as unsolvable.
    seed_holdout_start = 1_000_000
    holdout_seed = generate_many(
        n_holdout_seed,
        start_seed=seed_holdout_start,
        mechanics_pool=train_pool,
        cfg=cfg,
    )

    holdout_mech = generate_many(
        n_holdout_mechanics,
        start_seed=2_000_000,
        mechanics_pool=list(withheld),
        cfg=cfg,
    )

    return Split(
        train=tuple(train),
        holdout_seed=tuple(holdout_seed),
        holdout_mechanics=tuple(holdout_mech),
        withheld=withheld,
    )


def iter_worlds(
    count: int, start_seed: int = 0, cfg: GeneratorConfig | None = None
) -> Iterator[WorldSpec]:
    """Stream worlds without holding them all in memory."""
    produced = 0
    seed = start_seed
    while produced < count:
        spec = generate(seed, cfg=cfg)
        if spec is not None:
            produced += 1
            yield spec
        seed += 1
