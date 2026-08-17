"""Which cells a model is actually judged on.

The 64x64 frame is mostly dead. A generated world occupies a 7-13 cell
corner — roughly 4% of the grid — and the other 96% is background zeros
that never change in any episode.

Scoring over the whole frame therefore rewards knowing nothing. Measured
before this module existed: a model that abstained on the entire playfield
and predicted only background scored **0.93 fitness against the exact
model's 1.00**. It cost itself 4% coverage to avoid every cell where the
dynamics live, and earned a perfect transition match because every
background zero it predicted was trivially correct.

The teacher was not exploiting that hole — it predicts the playfield and
gets it wrong, which is why observed fitness averaged 0.387. But Phase 3
trains a network against this number by gradient descent, and gradient
descent finds holes that a language model politely ignores. A reward
signal that pays 0.93 for silence would have taught the core to abstain on
everything that matters.

So scoring is restricted to the **active region**: cells that are ever
non-background, or that ever change. That is where a world model can be
right or wrong. Everything outside it is free to both a genius and a rock,
and a metric should not pay for it.
"""

from __future__ import annotations

from sentinel.env.history import History
from sentinel.env.types import GRID_SIZE, Grid

BACKGROUND = 0


def active_region(history: History) -> frozenset[tuple[int, int]]:
    """Cells worth scoring: ever non-background, or ever changing.

    Both halves matter. "Ever changes" alone would let a model scribble
    over static walls for free, since walls never move. "Ever non-zero"
    alone would miss a cell that flickers through background. The union is
    the set of cells that carry information about this world.
    """
    frames: list[Grid] = [history.initial.grid] + [s.settled.grid for s in history.steps]

    cells: set[tuple[int, int]] = set()
    first = frames[0]

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if first[y][x] != BACKGROUND:
                cells.add((x, y))

    previous = first
    for grid in frames[1:]:
        for y in range(GRID_SIZE):
            row, prev_row = grid[y], previous[y]
            for x in range(GRID_SIZE):
                if row[x] != BACKGROUND or row[x] != prev_row[x]:
                    cells.add((x, y))
        previous = grid

    return frozenset(cells)


def region_summary(region: frozenset[tuple[int, int]]) -> str:
    total = GRID_SIZE * GRID_SIZE
    if not region:
        return "empty active region"
    xs = [x for x, _ in region]
    ys = [y for _, y in region]
    return (
        f"{len(region)} cells ({len(region) / total:.1%} of frame), "
        f"x {min(xs)}..{max(xs)}, y {min(ys)}..{max(ys)}"
    )
