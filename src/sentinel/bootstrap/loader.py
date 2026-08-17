"""Turning LLM-written Python into a WorldModel.

The teacher writes four plain functions rather than a class:

    init_state()              -> state
    transition(state, action) -> state
    render(state)             -> 64x64 grid, -1 to abstain
    outcome(state)            -> "ongoing" | "level_complete" | "game_over"

Plain functions because they are what the model writes most reliably: no
import boilerplate, no base class to get wrong, no `self`. The adapter here
supplies the contract around them.

**This is not a sandbox.** Generated code runs in-process with full
privileges, and isolation is deliberately left for later. What *is* here is
a watchdog: a wall-clock timeout on every call, because a single generated
`while True:` would otherwise hang an unattended overnight corpus run with
no recovery. A timeout is recorded as a verification failure, which is the
correct reading — a model that cannot answer in bounded time is not a
usable model regardless of whether it would eventually be right.

Normalisation is deliberately generous. The model returns lists, tuples,
numpy arrays, sometimes a bare int for a whole row. Rejecting those as
malformed would throw away hypotheses that are substantively correct and
merely untidy, so they are coerced where the intent is unambiguous and
rejected only where it is not.
"""

from __future__ import annotations

import signal
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterator

from sentinel.env.types import GRID_SIZE, Action
from sentinel.wm.contract import (
    ABSTAIN,
    ModelError,
    Outcome,
    RenderedGrid,
    WorldModel,
)

REQUIRED = ("init_state", "transition", "render", "outcome")


class ModelTimeout(ModelError):
    """A generated function exceeded its wall-clock budget."""


class LoadError(RuntimeError):
    """Generated source could not be turned into a model at all."""


@contextmanager
def time_guard(seconds: float) -> Iterator[None]:
    """Interrupt the enclosed block after `seconds`.

    SIGALRM only fires on the main thread, so callers running the teacher
    concurrently must parallelise across processes rather than threads.
    That constraint is worth the simplicity: no subprocess round-trip per
    candidate model, and candidates are evaluated thousands of times.
    """
    if seconds <= 0:
        yield
        return

    def _fire(signum: int, frame: Any) -> None:
        raise ModelTimeout(f"exceeded {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def normalize_grid(raw: Any) -> RenderedGrid:
    """Coerce whatever the model returned into a 64x64 grid of ints.

    Accepts lists, tuples, and anything with tolist(). Rejects only what is
    genuinely ambiguous — a wrong row count, a non-integer cell, or a value
    outside 0..15 that is not the abstain sentinel.
    """
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        raise ModelError(f"render returned {type(raw).__name__}, not a grid")
    if len(raw) != GRID_SIZE:
        raise ModelError(f"render returned {len(raw)} rows, expected {GRID_SIZE}")

    rows: list[tuple[int, ...]] = []
    for y, row in enumerate(raw):
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)):
            raise ModelError(f"render row {y} is {type(row).__name__}, not a sequence")
        if len(row) != GRID_SIZE:
            raise ModelError(
                f"render row {y} has {len(row)} cells, expected {GRID_SIZE}"
            )
        cells: list[int] = []
        for x, cell in enumerate(row):
            if isinstance(cell, bool):
                raise ModelError(f"render cell ({x},{y}) is a bool")
            try:
                value = int(cell)
            except (TypeError, ValueError) as exc:
                raise ModelError(
                    f"render cell ({x},{y}) is {type(cell).__name__}, not an int"
                ) from exc
            if value != ABSTAIN and not 0 <= value <= 15:
                raise ModelError(f"render cell ({x},{y}) is {value}; expected 0..15 or -1")
            cells.append(value)
        rows.append(tuple(cells))
    return tuple(rows)


def normalize_outcome(raw: Any) -> Outcome:
    """Accept an Outcome, or any of the strings the model plausibly writes."""
    if isinstance(raw, Outcome):
        return raw
    text = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if text in ("ongoing", "playing", "in_progress", "continue", "none"):
        return Outcome.ONGOING
    if text in ("level_complete", "complete", "win", "solved", "success", "levelcomplete"):
        return Outcome.LEVEL_COMPLETE
    if text in ("game_over", "gameover", "dead", "lose", "lost", "fail", "failed"):
        return Outcome.GAME_OVER
    raise ModelError(f"outcome returned unrecognised value {raw!r}")


def _hashable(state: Any) -> Hashable:
    """Make a returned state hashable so planners can dedup on it.

    Models routinely return dicts and lists. Rather than reject those — the
    hypothesis may be entirely correct — they are frozen into an equivalent
    hashable form.
    """
    try:
        hash(state)
        return state
    except TypeError:
        pass
    if isinstance(state, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in state.items()))
    if isinstance(state, (list, tuple, set, frozenset)):
        return tuple(_hashable(v) for v in state)
    return repr(state)


@dataclass
class LoadedModel(WorldModel):
    """Adapter wrapping four generated functions in the contract."""

    source: str
    fn_init: Callable[[], Any]
    fn_transition: Callable[[Any, Any], Any]
    fn_render: Callable[[Any], Any]
    fn_outcome: Callable[[Any], Any]
    timeout: float = 2.0
    name: str = "generated"

    def init_state(self) -> Hashable:
        with time_guard(self.timeout):
            return _hashable(self.fn_init())

    def transition(self, state: Any, action: Action) -> Hashable:
        with time_guard(self.timeout):
            return _hashable(self.fn_transition(state, action.action_id))

    def render(self, state: Any) -> RenderedGrid:
        with time_guard(self.timeout):
            return normalize_grid(self.fn_render(state))

    def outcome(self, state: Any) -> Outcome:
        with time_guard(self.timeout):
            return normalize_outcome(self.fn_outcome(state))

    def reset_to(self, state: Any) -> Hashable:
        return self.init_state()


def extract_code(text: str) -> str:
    """Pull Python out of a chat response.

    Prefers fenced blocks and takes the longest, since models often emit a
    short illustrative snippet before the real answer.
    """
    if "```" not in text:
        return text.strip()

    blocks: list[str] = []
    parts = text.split("```")
    for i in range(1, len(parts), 2):
        block = parts[i]
        newline = block.find("\n")
        if newline != -1 and block[:newline].strip().lower() in ("python", "py", ""):
            block = block[newline + 1 :]
        blocks.append(block)

    if not blocks:
        return text.strip()
    return max(blocks, key=len).strip()


def load_model(
    source: str,
    timeout: float = 2.0,
    name: str = "generated",
    context: dict[str, Any] | None = None,
) -> LoadedModel:
    """Execute generated source and bind the four required functions.

    `context` pre-populates the namespace — in practice with `INITIAL_GRID`,
    the observed opening frame. That separation matters more than it looks:
    the static layout is *observed data*, not a hypothesis, and making the
    model retype 64 rows of it as a literal was costing thousands of tokens,
    truncating the real logic, and producing models that were wrong about
    frame zero before any action was taken. Handing over the layout lets the
    model spend its output on dynamics, which is the part actually being
    induced.

    Raises LoadError if the source will not run or does not define the
    contract. That is a different failure from "runs but predicts wrongly",
    and the corpus records the two separately — one is a formatting problem
    worth re-prompting, the other is a genuine hypothesis that happens to
    be false.
    """
    namespace: dict[str, Any] = {"__name__": "generated_model"}
    if context:
        namespace.update(context)

    try:
        with time_guard(max(timeout, 5.0)):
            exec(compile(source, "<generated>", "exec"), namespace)  # noqa: S102
    except ModelTimeout as exc:
        raise LoadError(f"module-level code timed out: {exc}") from exc
    except SyntaxError as exc:
        raise LoadError(f"syntax error: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise LoadError(f"{type(exc).__name__} at import: {exc}") from exc

    missing = [fn for fn in REQUIRED if not callable(namespace.get(fn))]
    if missing:
        raise LoadError(f"missing required function(s): {', '.join(missing)}")

    return LoadedModel(
        source=source,
        fn_init=namespace["init_state"],
        fn_transition=namespace["transition"],
        fn_render=namespace["render"],
        fn_outcome=namespace["outcome"],
        timeout=timeout,
        name=name,
    )
