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
  wm/         executable world models — Python as falsifiable hypotheses
  verify/     replay verifier — the reward signal everything rests on
  core/       tiny recursive reasoner, trained from scratch
  adapt/      test-time training
  memory/     skill library + continual learning
  evolve/     scaffold self-modification
  gen/        procedural environment generator
  bootstrap/  LLM teacher, progressively removed
scripts/      fetch_games.py, bench_engine.py
tests/        Phase 0 determinism gate
```

Nothing above `env/` imports the ARC engine. That boundary is deliberate:
Phase 6 swaps a structurally different environment in behind it, and the
transfer result is only meaningful if nothing upstream had to change.

## Status

**Phase 0 complete.** Engine verified offline, typed interaction history,
deterministic replay proven by test (20 passing).

Next: Phase 1 — the verifier and the world-model contract. The gate is
that the verifier must catch every deliberately injected bug in a
hand-written model. If it can't cheaply tell good models from bad ones,
the program has no reward signal and nothing after this point works.
