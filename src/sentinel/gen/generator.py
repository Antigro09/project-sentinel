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


def mechanic_space() -> list[Mechanics]:
    """The enumerated combinations the generator draws from.

    Kept explicit rather than sampled independently so the train/holdout
    split can withhold *specific combinations* and we can say exactly which
    ones the system had never seen.
    """
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
) -> Split:
    """Build a corpus split that withholds whole mechanic combinations.

    `withhold` combinations never appear in training at any seed. Those are
    the worlds Phase 3 should be judged on: everything else only asks
    whether the system can rearrange rules it has already been taught.
    """
    space = mechanic_space()
    rng = random.Random(seed)
    withheld = tuple(rng.sample(space, min(withhold, len(space))))
    train_pool = [m for m in space if m not in withheld]

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
