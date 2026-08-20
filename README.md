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
uv run pytest tests/ -q          # all gates: replay, verifier, planner, agent
uv run scripts/bench_engine.py   # engine throughput on this machine
uv run scripts/run_agent.py      # train the core, then solve unseen worlds
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
  core/       tiny recursive reasoner + closed perceive-plan-act loop   [built]
  adapt/      test-time training
  memory/     skill library + continual learning
  evolve/     scaffold self-modification
scripts/      fetch_games.py, bench_engine.py, build_corpus.py,
              preflight.py, rescore_corpus.py, train_core.py, run_agent.py
tests/        determinism, verifier gate, planner, generator, bootstrap
```

Nothing above `env/` imports the ARC engine. That boundary is deliberate:
Phase 6 swaps a structurally different environment in behind it, and the
transfer result is only meaningful if nothing upstream had to change.
`gen/toy.py` is the first evidence it holds — it produces the same
`History` type through none of the engine's code.

## Status

**Phases 0-5 built. 141 tests passing.** The headline numbers in earlier
versions of this file were wrong, and the correction is the most useful
thing here.

### The benchmark could not measure what it claimed

The original held-out set contained four rule combinations. In it,
predicting `charge_period` from `has_hazards` scored **exactly 1.000** --
the four withheld combinations paired `charge=3` with hazards every time,
and hazards are coloured cells anyone can see. So the reported "infers a
counter that appears in no frame" was hazard detection under another name.

Auditing the rest of that split, four of six labels measured nothing:

| label | what it actually measured |
|---|---|
| charge_period | perfectly predictable from has_hazards |
| wrap_edges | constant across the whole holdout |
| has_switches | constant across the whole holdout |
| step_distance | class absent from *training*; 0.751 was both prior and ceiling |

The cause was `make_split` drawing withheld combinations at random. With
four of them, confounding is near certain. The holdout is now *chosen* for
label independence (`balanced_withhold`), which surfaced a structural
result: the original 26-combination space cannot produce a valid holdout at
any size -- 1.00 confounding for every k tried, against 0.54 for the
compositional space. Widening the mechanic space was not optional.

| | old | new |
|---|---|---|
| rule combinations | 26 | 5,760 |
| distinct rule sets in holdout | 4 | 24 |
| worst label confounding | 1.00 | 0.54 |
| exhaustive search cost | 1.7s/world | ~101s/world |

### What the core actually does

It **prunes**, which is the job the plan assigns it. Rank of the true rule
set among 5,760, over 73 held-out worlds -- accuracy is held constant by
construction, since the verifier still decides what is true, so a confident
wrong prior costs replays and nothing else:

| ordering | median rank | mean | top-100 | search time |
|---|---|---|---|---|
| random | 3,734 | 3,244 | 3% | 56.8s |
| simplicity | 3,206 | 2,765 | 0% | 48.4s |
| library (memory/) | 710 | 1,353 | 10% | 23.7s |
| **core** | **173** | **366** | **34%** | **6.4s** |

### What the core does not do

It does not infer hidden state. On a benchmark where `charge_period` is not
confounded with anything:

| label | core | prior | |
|---|---|---|---|
| switches | 0.812 | 0.462 | learned |
| gates_start_open | 0.679 | 0.543 | learned |
| step_distance | 0.542 +/- 0.267 | 0.491 | marginal, unstable |
| hazards | 0.416 +/- 0.239 | 0.342 | marginal, unstable |
| edge_mode | 0.403 | 0.377 | marginal |
| ordered_targets | 0.522 | 0.539 | at prior |
| **charge_period** | **0.203** | **0.298** | **below prior** |
| wait_advances_charge | 0.431 | 0.669 | below prior |

Both results hold at once. Ranking uses the *joint* distribution over eight
heads, and the three labels the core does learn cut the space by about 18x
on their own -- 5760/18 is roughly the observed median of 173. So the core
prunes usefully **without** positing anything it cannot see. That is a
weaker claim than the plan makes, and it is the one the evidence supports.

The failure is not a lack of evidence. The verifier identifies
`charge_period` in **100%** of these worlds with a mean fitness gap of 0.838
-- larger than on the old benchmark. The signal is at full strength and the
network fails to read it.

### Layers

| layer | state |
|---|---|
| `core/` | prunes 18.5x better than simplicity; does not infer hidden state |
| `adapt/` | works -- recovers the exact rule set 36% of the time with no labels |
| `memory/` | works on the wide space (4.5x better ranking); was *worse* than no memory on the narrow one |
| `evolve/` | mechanically correct -- archives every version, gates promotion on held-out worlds, rolls back |

With true mechanics the agent solves **60.0%** of compositional worlds, so
the planner handles the new rules and inference is the binding constraint.

### Open problems

- **charge_period, 0.203 against a 0.298 prior.** The plan's sharpest claim
  rests on this and it is currently unsupported.
- **ordered_targets, still at prior.** Unchanged through every attempt.
- **Search is still cheap enough to win.** 5,760 hypotheses cost ~101s;
  brute force only becomes unusable past ~17,000.

### Lessons that cost the most to learn

1. **A benchmark that cannot fail teaches nothing.** Four of six labels were
   unmeasurable for weeks and every number computed from them was noise.
2. **A reward signal will be optimised into its blind spot.** Scoring the
   whole 64x64 frame paid 0.93 fitness for predicting only dead background.
3. **You cannot learn a rule your evidence never exercises** -- the verifier
   gate, then ordered_targets.
4. **Check the baseline you claim to beat.** The plan asked for a comparison
   against random program search. It was never built, and it turned out
   brute force beat the trained core on the original benchmark outright.
5. **Single runs lie**, and so do confounded ones. Everything above is
   multi-seed on a holdout chosen for label independence.
