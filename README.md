# Sentinel

A local research program toward general intelligence.

The bet: generality is an **architecture** problem, not a compute problem.
Datacenter scale buys memorization, and memorization is not what we're
after. If that's right, a single machine is enough to work on it.

The system learns an unfamiliar environment by writing an **executable
world model** — a Python program that is a falsifiable hypothesis about
how the environment works. It runs the program against what actually
happened, repairs it where it mispredicts, simplifies it toward general
rules, and plans by simulating forward instead of spending real actions.

Full plan, including falsifiable claims and kill criteria:
`~/.claude/plans/i-want-you-to-tranquil-church.md`

## Setup

```bash
uv sync
uv run scripts/fetch_games.py   # one-time; the only network access in the project
```

After that, everything runs offline.

## Verify

```bash
uv run pytest tests/ -q          # Phase 0 gate: deterministic replay
uv run scripts/bench_engine.py   # engine throughput on this machine
```

## Measured on an M5 Max (40-core GPU, 128GB)

| | |
|---|---|
| Engine throughput | ~2,550 steps/sec median across 25 games (343 – 3,688) |
| Public environments | 25, all playable offline |
| Observation | 64×64 grid, cell values 0–15 |
| Actions | 7 — five simple, `ACTION6` takes x/y, `ACTION7` undo |

Note the spread. Raw throughput hides how much work a step does: `ft09`
runs at 2,555 steps/sec but only 11% of legal actions change the grid,
so its *effective* rate is ~290/sec. Search budgets should be set from
the effective column, not the raw one.

## Layout

```
src/sentinel/
  env/        environment boundary — typed history, deterministic replay  [built]
  wm/         executable world models — Python as falsifiable hypotheses  [built]
  verify/     replay verifier — the reward signal everything rests on     [built]
  plan/       BFS search inside the model + divergence-checking executor  [built]
  gen/        environment generator — toy world today, thousands later    [seeded]
  core/       tiny recursive reasoner, trained from scratch
  adapt/      test-time training
  memory/     skill library + continual learning
  evolve/     scaffold self-modification
  bootstrap/  LLM teacher, progressively removed
scripts/      fetch_games.py, bench_engine.py
tests/        determinism, verifier gate, planner
```

Nothing above `env/` imports the ARC engine. That boundary is deliberate:
Phase 6 swaps a structurally different environment in behind it, and the
transfer result is only meaningful if nothing upstream had to change.
`gen/toy.py` is the first evidence it holds — it produces the same
`History` type through none of the engine's code.

## Status

**Phase 0 complete.** Engine verified offline, typed interaction history,
deterministic replay proven by test.

**Phase 1 complete.** 54 tests passing.

- World-model contract: `init_state / transition / render / outcome`,
  hidden state mandatory, per-cell abstention.
- Verifier: accuracy, coverage and outcome reported independently.
- **Gate met** — eight injected bugs, all detected.
- End to end: model → BFS plan → act → 3/3 toy levels solved in 70
  actions, zero learning, zero LLM calls.

Three things measurement forced, recorded because they shaped the design:

1. **You cannot falsify a claim your evidence never exercises.** The gate
   first failed on two mutations because random play never completes a
   level, making "levels never end" factually true on that history.
   `verify/evidence.py` now reports untestable channels instead of passing
   them silently.
2. **Cell accuracy is a trap.** A model refusing to posit hidden state
   scores 99.9% of cells and 10.3% of frames — the grid is mostly
   background. Ranking on accuracy preferred a model that predicts nothing
   ever moves over one that tracks the player correctly, so `fitness` is
   built on `transition_match` instead.
3. **Plans must be checked every step.** A model denying hidden state
   produces a confident plan that reality refutes within 3 actions.

Next: Phase 2 — the LLM proposer loop. The agent must now *induce* what
was hand-written here. Reading game source was legitimate for a Phase 1
baseline and is forbidden from here on.
