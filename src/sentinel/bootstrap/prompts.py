"""Prompt construction.

Token cost is the binding constraint on corpus size, and grids are the
expensive part. A 64x64 grid printed naively is ~4k characters; twenty
frames of that is most of a context window spent on background zeros.

Two compressions do almost all the work:

- **Crop to content.** Generated worlds occupy a 7-13 cell corner of the
  64x64 field, so the bounding box of everything non-zero is tiny. The
  offset is stated once and the model works in full coordinates.
- **Diff after the first frame.** Consecutive frames differ in a handful
  of cells, so listing the changes costs a line instead of a grid.

Together these take a 20-step episode from roughly 20k tokens to under
one. That ratio is the difference between a corpus that can be built
overnight and one that cannot.

The repair prompt is the more important of the two. It carries the exact
step where the previous model's story stopped matching reality, which is
the single most informative thing the system knows — and it explicitly
invites a *different hypothesis* rather than a patch, because the failure
mode both published systems report is premature commitment: refining
forever inside a model that was wrong from the start.
"""

from __future__ import annotations

from sentinel.env.history import History
from sentinel.env.types import GRID_SIZE, Grid
from sentinel.verify.report import VerificationReport

SYSTEM = """You are a scientist reverse-engineering an unknown grid-world environment \
from observations. You write executable Python that predicts the environment exactly.

You do not explain. You output one Python code block and nothing else."""

CONTRACT = '''A module-level constant is already defined for you:

    INITIAL_GRID   # tuple of 64 tuples of 64 ints - the exact FRAME 0 above

Use it. Do not retype the layout as a literal. Typical render():

    def render(state):
        g = [list(row) for row in INITIAL_GRID]
        # ... clear the agent's old cell, draw it at its new position ...
        return g

Write exactly these four top-level functions:

    def init_state():
        """Return the state before any action. Any Python value."""

    def transition(state, action):
        """Return the successor state. `action` is an int (1-5).
        Must be deterministic and must not mutate `state`."""

    def render(state):
        """Return a 64x64 list of lists of ints 0-15.
        Use -1 for any cell you are not confident about."""

    def outcome(state):
        """Return "ongoing", "level_complete", or "game_over"."""

RULES
- Plain functions at module level. No classes, no imports beyond the stdlib.
- Keep it SHORT. Long answers get cut off mid-file and score zero.
- `state` may be any value you choose. Use a tuple so it stays hashable.
- CRITICAL: the environment may have HIDDEN STATE. Two identical grids can
  have different successors. If the observations cannot be explained by
  position alone, your state must track something invisible - a counter, a
  toggle, a phase. Check for this explicitly before assuming it is absent.
- Prefer -1 (abstain) over a guess. Abstaining costs coverage; guessing
  wrong costs correctness, which is worse.
- Return the whole 64x64 grid. Cells outside the active area are 0.'''


def content_bounds(grids: list[Grid]) -> tuple[int, int, int, int]:
    """Bounding box of every non-zero cell across all grids."""
    min_x, min_y = GRID_SIZE, GRID_SIZE
    max_x = max_y = -1
    for grid in grids:
        for y in range(GRID_SIZE):
            row = grid[y]
            for x in range(GRID_SIZE):
                if row[x]:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
    if max_x < 0:
        return 0, 0, 0, 0
    return min_x, min_y, max_x, max_y


def grid_to_text(grid: Grid, bounds: tuple[int, int, int, int]) -> str:
    """One hex digit per cell, cropped to `bounds`."""
    min_x, min_y, max_x, max_y = bounds
    return "\n".join(
        "".join(format(grid[y][x], "x") for x in range(min_x, max_x + 1))
        for y in range(min_y, max_y + 1)
    )


def diff_text(before: Grid, after: Grid, limit: int = 40) -> str:
    """Changed cells as `(x,y): old->new`, or a full-grid marker if too many."""
    changes = [
        f"({x},{y}) {before[y][x]:x}->{after[y][x]:x}"
        for y in range(GRID_SIZE)
        for x in range(GRID_SIZE)
        if before[y][x] != after[y][x]
    ]
    if not changes:
        return "no change"
    if len(changes) > limit:
        return f"{len(changes)} cells changed (too many to list)"
    return "  ".join(changes)


def describe_history(history: History, max_steps: int = 40) -> str:
    """Initial frame in full, then per-step diffs."""
    frames = [history.initial.grid] + [s.settled.grid for s in history.steps]
    bounds = content_bounds(frames)
    min_x, min_y, max_x, max_y = bounds

    lines = [
        f"Active region: x {min_x}..{max_x}, y {min_y}..{max_y} "
        f"(all cells outside are 0). Rows below start at x={min_x}, y={min_y}.",
        "",
        "FRAME 0 (initial state):",
        grid_to_text(history.initial.grid, bounds),
        "",
        f"Legal actions: {list(history.initial.available_actions)}",
        f"Levels to win: {history.initial.win_levels}",
        "",
        "TRANSITIONS (each shows the action taken and what changed):",
    ]

    previous = history.initial
    shown = 0
    for step in history.steps:
        if shown >= max_steps:
            lines.append(f"... {len(history.steps) - shown} further steps omitted")
            break
        settled = step.settled
        marks = []
        if settled.levels_completed != previous.levels_completed:
            marks.append("LEVEL COMPLETED -> new layout loaded")
        if settled.state.value == "GAME_OVER":
            marks.append("GAME OVER")
        if settled.state.value == "WIN":
            marks.append("WIN")
        suffix = ("   << " + "; ".join(marks)) if marks else ""
        lines.append(
            f"step {step.index + 1}: action={step.action.action_id}  "
            f"{diff_text(previous.grid, settled.grid)}{suffix}"
        )
        previous = settled
        shown += 1

    return "\n".join(lines)


def build_initial_prompt(history: History, max_steps: int = 40) -> str:
    return f"""Below are observations from an unknown deterministic grid environment.

{describe_history(history, max_steps=max_steps)}

{CONTRACT}

Study the transitions carefully before writing. In particular, check whether \
any two identical grids led to different successors - if so, there is hidden \
state your model must track.

Output one Python code block."""


def build_repair_prompt(
    history: History,
    source: str,
    report: VerificationReport,
    max_steps: int = 40,
) -> str:
    """Prompt built around the exact point the previous model broke."""
    divergence = report.first_divergence
    detail_lines: list[str] = []

    if report.crashed:
        detail_lines.append(f"Your model CRASHED: {report.crash_detail}")
    elif divergence is not None:
        step_result = next((s for s in report.steps if s.index == divergence), None)
        frames = [history.initial.grid] + [s.settled.grid for s in history.steps]
        bounds = content_bounds(frames)

        if divergence == 0:
            detail_lines.append(
                "Your model was already wrong about the INITIAL frame, before "
                "any action was taken."
            )
        else:
            step = history.steps[divergence - 1]
            before = history.observation_before(step.index)
            detail_lines.append(
                f"Your model first went wrong at step {divergence} "
                f"(action={step.action.action_id})."
            )
            detail_lines.append("")
            detail_lines.append("The grid BEFORE that action:")
            detail_lines.append(grid_to_text(before.grid, bounds))
            detail_lines.append("")
            detail_lines.append("What ACTUALLY happened:")
            detail_lines.append(grid_to_text(step.settled.grid, bounds))

        if step_result is not None and step_result.outcome_predicted is not None:
            if not step_result.outcome_correct:
                detail_lines.append("")
                detail_lines.append(
                    f"Your outcome() said {step_result.outcome_predicted.value!r} "
                    f"but it was {step_result.outcome_actual.value!r}."
                )
    else:
        detail_lines.append(
            f"Your model never mispredicted, but only covered "
            f"{report.coverage:.0%} of cells. Reduce the number of -1 cells."
        )

    return f"""You previously proposed this model of an unknown grid environment:

```python
{source}
```

RESULT: {report.summary()}

{chr(10).join(detail_lines)}

Full observations again:

{describe_history(history, max_steps=max_steps)}

{CONTRACT}

Do not simply patch the previous model. If a single rule cannot explain the \
failure above, the underlying hypothesis is probably wrong - consider a \
different one. In particular, if the same visible grid produced different \
successors at different times, position alone cannot be the whole state.

Output one Python code block."""
