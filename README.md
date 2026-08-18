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
  gen/        procedural environment generator, 26 mechanic combinations  [built]
  bootstrap/  LLM teacher: propose → verify → repair, and its removal     [built]
  core/       tiny recursive reasoner, trained from scratch
  adapt/      test-time training
  memory/     skill library + continual learning
  evolve/     scaffold self-modification
scripts/      fetch_games.py, bench_engine.py, build_corpus.py
tests/        determinism, verifier gate, planner, generator, bootstrap
```

Nothing above `env/` imports the ARC engine. That boundary is deliberate:
Phase 6 swaps a structurally different environment in behind it, and the
transfer result is only meaningful if nothing upstream had to change.
`gen/toy.py` is the first evidence it holds — it produces the same
`History` type through none of the engine's code.

## Status

**Phases 0-3 complete.** 101 tests passing.

A 1.4M-parameter recursive network, trained from scratch with no LLM
anywhere in the loop, watches an unknown world, infers rules that appear in
no single frame, builds an executable model of it, plans inside that model,
and acts.

Measured over 3 training seeds x 100 worlds whose rule combinations were
never seen in training:

| condition | solve rate | levels |
|---|---|---|
| true mechanics (ceiling) | 67.0% | 74.4% |
| **core-inferred** | **16.3% +/- 2.5%** | **27.2% +/- 2.7%** |
| default guess (floor) | 5.0% | 12.3% |

Per-label inference on held-out seeds:

| label | core | prior |
|---|---|---|
| has_hazards | 0.999 | 0.553 |
| has_switches | 0.999 | 0.547 |
| wrap_edges | 0.985 | 0.954 |
| **charge_period** (hidden) | **0.795 +/- 0.159** | 0.365 |
| ordered_targets | 0.561 | 0.546 |

`charge_period` is the one that matters: a counter that makes every Nth move
travel two cells, invisible in any single frame, recoverable only from a
pattern across a sequence. Inferring it means positing structure that cannot
be seen.

### What this is, and is not

This infers six mechanic parameters within a known hypothesis space. That is
the smallest testable form of "architecture over scale" -- necessary, not
sufficient, and a long way from the open-ended program synthesis the plan
describes.

### Open problems

- **ordered_targets, ~0.56.** Objective order is unobservable, and a
  solution trajectory never exercises it: measured at literally zero
  failed-collection events until probing was added. Four probing configs
  were tested and every one that made the signal denser cost more on
  charge_period than it gained.
- **The 67% ceiling.** With perfect rules the agent still fails a third of
  these worlds, all of them ordered-target worlds. The loop, not the core,
  is the binding constraint there.
- **charge_period variance, +/- 0.159.** One seed in four still lands near
  chance.

### Lessons that cost the most to learn

1. **A reward signal will be optimised into its blind spot.** Scoring the
   whole 64x64 frame paid 0.93 fitness for predicting only dead background.
2. **You cannot learn a rule your evidence never exercises.** Twice: the
   verifier gate, and ordered_targets.
3. **Early stopping on a mean hides the slowest label.** charge_period read
   0.409 +/- 0.022 -- apparently pinned to chance -- purely because
   saturated heads held the mean flat.
4. **Single runs lie.** The same condition measured 15% and 5% on different
   seeds. Everything above is multi-seed.
