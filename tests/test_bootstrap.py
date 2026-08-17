"""The teacher pipeline, minus the model.

Everything here runs offline. The LLM call is the slow, nondeterministic
part, so the machinery around it — extraction, normalisation, the timeout
watchdog, corpus resume — is tested without it. What remains untested by
this file is whether the model writes *good* models, which is what the
smoke run in `scripts/build_corpus.py --smoke` measures.
"""

from __future__ import annotations

import time

import pytest

from sentinel.bootstrap import (
    CorpusRecord,
    CorpusWriter,
    LoadError,
    ModelTimeout,
    completed_ids,
    extract_code,
    load_model,
    make_training_history,
    normalize_grid,
    normalize_outcome,
    read_corpus,
    time_guard,
)
from sentinel.bootstrap.prompts import build_initial_prompt, content_bounds, diff_text
from sentinel.env.types import GRID_SIZE
from sentinel.gen import generate
from sentinel.verify import evidence_coverage
from sentinel.wm.contract import ABSTAIN, Outcome

GOOD_SOURCE = '''
def init_state():
    return (0, 0)

def transition(state, action):
    x, y = state
    if action == 1: y -= 1
    elif action == 2: y += 1
    elif action == 3: x -= 1
    elif action == 4: x += 1
    return (max(0, min(63, x)), max(0, min(63, y)))

def render(state):
    g = [[0] * 64 for _ in range(64)]
    g[state[1]][state[0]] = 4
    return g

def outcome(state):
    return "ongoing"
'''


# -- code extraction ------------------------------------------------------


def test_extract_code_prefers_longest_block() -> None:
    text = "Here is a sketch:\n```python\nx = 1\n```\nAnd the real answer:\n```python\ndef f():\n    return 42\n```"
    assert "def f()" in extract_code(text)
    assert "x = 1" not in extract_code(text)


def test_extract_code_without_fences() -> None:
    assert extract_code("def f():\n    return 1").startswith("def f()")


def test_extract_code_handles_unlabelled_fence() -> None:
    assert "def g()" in extract_code("```\ndef g():\n    pass\n```")


# -- loading --------------------------------------------------------------


def test_load_good_source() -> None:
    model = load_model(GOOD_SOURCE)
    state = model.init_state()
    from sentinel.env.types import Action

    moved = model.transition(state, Action(4))
    assert moved != state
    assert model.outcome(state) is Outcome.ONGOING
    assert len(model.render(state)) == GRID_SIZE


def test_load_rejects_missing_functions() -> None:
    with pytest.raises(LoadError, match="missing required function"):
        load_model("def init_state():\n    return 0\n")


def test_load_rejects_syntax_error() -> None:
    with pytest.raises(LoadError, match="syntax error"):
        load_model("def init_state(:\n    return 0")


def test_initial_grid_is_injected() -> None:
    """The observed layout is data, not hypothesis.

    Making the model retype 64 rows as a literal cost thousands of tokens,
    truncated the real logic, and produced models wrong about frame zero.
    Handing it over lets the model spend output on dynamics instead.
    """
    source = """
def init_state():
    return 0

def transition(state, action):
    return state + 1

def render(state):
    return [list(row) for row in INITIAL_GRID]

def outcome(state):
    return "ongoing"
"""
    grid = tuple(tuple(7 for _ in range(GRID_SIZE)) for _ in range(GRID_SIZE))
    model = load_model(source, context={"INITIAL_GRID": grid})
    assert model.render(model.init_state())[0][0] == 7


def test_missing_context_surfaces_as_load_failure() -> None:
    """Referencing INITIAL_GRID without it provided must fail loudly."""
    source = "INITIAL_GRID[0]\ndef init_state():\n    return 0\n"
    with pytest.raises(LoadError):
        load_model(source)


def test_unhashable_state_is_accepted() -> None:
    """Models routinely return dicts. Rejecting them would throw away
    hypotheses that may be entirely correct and merely untidy."""
    source = GOOD_SOURCE.replace(
        "def init_state():\n    return (0, 0)",
        "def init_state():\n    return {'x': 0, 'y': 0}",
    ).replace("    x, y = state", "    x, y = (state['x'], state['y']) if isinstance(state, dict) else state")
    model = load_model(source)
    assert hash(model.init_state()) is not None


# -- the watchdog ---------------------------------------------------------


def test_time_guard_interrupts_infinite_loop() -> None:
    """Without this, one generated `while True:` hangs an overnight run."""
    started = time.perf_counter()
    with pytest.raises(ModelTimeout):
        with time_guard(0.3):
            while True:
                pass
    assert time.perf_counter() - started < 3.0


def test_generated_infinite_loop_is_contained() -> None:
    source = GOOD_SOURCE.replace(
        "def outcome(state):\n    return \"ongoing\"",
        "def outcome(state):\n    while True:\n        pass",
    )
    model = load_model(source, timeout=0.3)
    with pytest.raises(ModelTimeout):
        model.outcome(model.init_state())


# -- normalisation --------------------------------------------------------


def test_normalize_accepts_lists_and_tuples() -> None:
    grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    assert normalize_grid(grid)[0][0] == 0
    assert normalize_grid(tuple(tuple(r) for r in grid))[0][0] == 0


def test_normalize_accepts_abstain() -> None:
    grid = [[ABSTAIN] * GRID_SIZE for _ in range(GRID_SIZE)]
    assert normalize_grid(grid)[5][5] == ABSTAIN


def test_normalize_rejects_wrong_shape() -> None:
    from sentinel.wm.contract import ModelError

    with pytest.raises(ModelError, match="rows"):
        normalize_grid([[0] * GRID_SIZE])


def test_normalize_rejects_out_of_range() -> None:
    from sentinel.wm.contract import ModelError

    grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    grid[0][0] = 99
    with pytest.raises(ModelError, match="0..15"):
        normalize_grid(grid)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ongoing", Outcome.ONGOING),
        ("LEVEL_COMPLETE", Outcome.LEVEL_COMPLETE),
        ("game over", Outcome.GAME_OVER),
        ("win", Outcome.LEVEL_COMPLETE),
        ("dead", Outcome.GAME_OVER),
    ],
)
def test_normalize_outcome_accepts_variants(raw: str, expected: Outcome) -> None:
    assert normalize_outcome(raw) is expected


# -- prompts --------------------------------------------------------------


def test_prompt_is_compact() -> None:
    """Grids are the expensive part; cropping and diffing is what makes an
    overnight corpus build affordable."""
    spec = generate(0)
    history = make_training_history(spec)
    prompt = build_initial_prompt(history)

    naive = len(history.steps) * GRID_SIZE * GRID_SIZE
    assert len(prompt) < naive / 4, "compression is not working"
    assert "hidden state" in prompt.lower()


def test_content_bounds_crops_to_active_region() -> None:
    spec = generate(0)
    history = make_training_history(spec)
    min_x, min_y, max_x, max_y = content_bounds([history.initial.grid])
    assert max_x - min_x < GRID_SIZE
    assert max_y - min_y < GRID_SIZE


def test_diff_text_reports_no_change() -> None:
    grid = tuple(tuple(0 for _ in range(GRID_SIZE)) for _ in range(GRID_SIZE))
    assert diff_text(grid, grid) == "no change"


# -- training evidence ----------------------------------------------------


def test_training_history_completes_levels() -> None:
    """Evidence must be able to falsify, or the corpus is built on histories
    that pass wrong models silently."""
    for seed in range(6):
        spec = generate(seed)
        if spec is None:
            continue
        history = make_training_history(spec)
        assert history is not None
        assert evidence_coverage(history).has_level_boundary, (
            f"{spec.world_id}: history never completes a level"
        )


# -- sandbox --------------------------------------------------------------


def test_sandbox_auto_falls_back_without_runtime() -> None:
    """The same call site must work before and after Docker is installed."""
    from sentinel.bootstrap import Sandbox

    box = Sandbox(mode="inprocess")
    assert not box.isolated
    assert "NOT isolated" in box.describe()


def test_sandbox_strict_mode_refuses_to_downgrade() -> None:
    """When isolation is required, silently running unprotected is the worst
    possible outcome — louder to fail."""
    from sentinel.bootstrap import Sandbox, detect_runtime

    if detect_runtime() is not None:
        pytest.skip("a container runtime is available; nothing to refuse")
    with pytest.raises(RuntimeError, match="no working container runtime"):
        Sandbox(mode="docker")


def test_sandbox_returns_live_metrics() -> None:
    """A report crossing a boundary must not rebuild as zeros.

    Every headline metric is computed from `steps`, so transporting a report
    without them would report a perfect-looking 0.0 for everything.
    """
    from sentinel.bootstrap import Sandbox

    spec = generate(0)
    history = make_training_history(spec)
    result = Sandbox(mode="inprocess").verify(GOOD_SOURCE, history, name="probe")

    assert result.ok
    assert result.report is not None
    assert result.report.coverage > 0.0
    assert len(result.report.steps) > 0


def test_report_survives_full_json_roundtrip() -> None:
    """to_json_full/from_json_full is what crosses the container boundary."""
    from sentinel.verify import Verifier
    from sentinel.verify.report import VerificationReport
    from sentinel.wm.reference import StaticModel

    spec = generate(1)
    history = make_training_history(spec)
    original = Verifier().verify(StaticModel(history.initial), history)
    rebuilt = VerificationReport.from_json_full(original.to_json_full())

    assert rebuilt.transition_match == original.transition_match
    assert rebuilt.coverage == original.coverage
    assert rebuilt.accuracy == original.accuracy
    assert rebuilt.outcome_accuracy == original.outcome_accuracy
    assert rebuilt.first_divergence == original.first_divergence
    assert rebuilt.fitness == original.fitness


def test_sandbox_reports_load_failure_not_crash() -> None:
    from sentinel.bootstrap import Sandbox

    spec = generate(0)
    history = make_training_history(spec)
    result = Sandbox(mode="inprocess").verify("def init_state(): return 0", history)

    assert not result.ok
    assert result.kind == "load"
    assert "missing required function" in (result.error or "")


# -- corpus ---------------------------------------------------------------


def test_corpus_roundtrip_and_resume(tmp_path) -> None:
    spec = generate(3)
    record = CorpusRecord(
        world_id=spec.world_id,
        split="train",
        spec=spec,
        source=GOOD_SOURCE,
        fitness=0.75,
        solved=False,
        transition_match=0.8,
        coverage=0.9,
        outcome_accuracy=1.0,
        rounds=2,
        prompt_tokens=100,
        output_tokens=200,
        seconds=12.5,
    )
    path = tmp_path / "c.jsonl"
    with CorpusWriter(path) as writer:
        writer.append(record)

    loaded = read_corpus(path)
    assert len(loaded) == 1
    assert loaded[0].world_id == spec.world_id
    assert loaded[0].spec.to_json() == spec.to_json()
    assert completed_ids(path) == {spec.world_id}


def test_corpus_survives_a_torn_line(tmp_path) -> None:
    """A build killed mid-write must not lose the records before it."""
    path = tmp_path / "c.jsonl"
    spec = generate(4)
    with CorpusWriter(path) as writer:
        writer.append(
            CorpusRecord(
                world_id=spec.world_id, split="train", spec=spec, source=None,
                fitness=0.0, solved=False, transition_match=0.0, coverage=0.0,
                outcome_accuracy=0.0, rounds=1, prompt_tokens=0,
                output_tokens=0, seconds=0.0,
            )
        )
    with path.open("a") as handle:
        handle.write('{"world_id": "torn", "spec": {')

    assert len(read_corpus(path)) == 1
