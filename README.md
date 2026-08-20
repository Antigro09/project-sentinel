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
uv run pytest tests/ -q                    # all gates: replay, verifier, planner, agent
uv run scripts/bench_engine.py             # engine throughput on this machine
```

Reproducing the numbers below, in order. The first trains and saves cores
that the rest load, so run it first:

```bash
uv run scripts/measure_core.py --seeds 2            # per-label accuracy
uv run scripts/measure_search.py --worlds 60        # ranking quality
uv run scripts/measure_end_to_end.py --worlds 60    # solve rate
uv run scripts/measure_identifiability.py --kind random
```

`measure_identifiability.py` is the one to run first when a label looks
stuck: it says whether the evidence determines that label at all, which is
a different problem from the network failing to read it, and only one of
the two is fixable by training.

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
  explore/    experiments designed from already-inferred rules            [built]
  domains/    a non-spatial environment, for the Phase 6 gate              [built]
scripts/      fetch_games.py, bench_engine.py, build_corpus.py, preflight.py,
              rescore_corpus.py, train_core.py, run_agent.py, and the
              measure_*.py family that reproduces every number below
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

### What the core does

It infers the rules of an unfamiliar world, including a rule that appears in
no single frame, and turns that into solved worlds. Two training seeds on
held-out mechanic combinations:

| label | core | prior |
|---|---|---|
| step_distance | **0.975 +/- 0.001** | 0.491 |
| gates_start_open | 0.758 +/- 0.012 | 0.543 |
| switches | 0.745 +/- 0.018 | 0.462 |
| edge_mode | 0.693 +/- 0.020 | 0.377 |
| **charge_period** (hidden) | **0.559 +/- 0.024** | 0.298 |
| hazards | 0.566 +/- 0.037 | 0.342 |
| ordered_targets | 0.578 +/- 0.026 | 0.539 |
| wait_advances_charge | 0.557 +/- 0.034 | 0.669 |

`charge_period` is the one that matters: a counter that makes every Nth move
travel an extra cell, invisible in any frame, recoverable only from a
pattern across a sequence. At 0.559 against a 0.298 prior it is being
inferred rather than guessed -- and unlike the 0.795 this file used to
report, it is measured on a holdout where the label is confounded with
nothing.

End to end, 60 held-out worlds:

| condition | solve rate | rules exact | verifier replays |
|---|---|---|---|
| true mechanics (ceiling) | 60.0% | 100% | - |
| **core-ordered search** | **58.2%** | 12.7% | **103** |
| simplicity ordering | 49.1% | 32.7% | 2,122 |
| default guess (floor) | 3.6% | 0% | - |

**97% of the ceiling at a twentieth of the search.** Note the middle column,
which is the more interesting result: the core recovers the exact rule set
*less* often than simplicity ordering and solves *more* worlds. Exact match
is the wrong question. A hypothesis that differs from the truth only on
behaviour the episode never exercised -- hazard rules in a level with no
hazards -- plans identically to the truth. The verifier constrains what is
possible, the core chooses among what remains, and the result is a model
that is usable rather than correct.

### How the core was fixed

For most of this project the core learned only the labels visible in a
still frame -- which coloured cells exist -- and nothing about motion. The
diagnosis chain is worth keeping:

1. A one-line rule over the encoded arrays ("the modal nonzero agent
   displacement is the step distance") scored **0.965**; the trained network
   scored **0.542**. The information was never missing.
2. `_coord_grid` spans [-1, 1] over a 16-cell crop, so one cell of
   displacement is 0.133 while `log1p(background mass)` in the same vector
   reaches 5.5. Movement arrived ~40x smaller than the static content it
   competed with. Expressing it in cells lifted step_distance to 0.684.
3. That was not enough, because the rule needs a **mode** and attention
   **averages** -- the mean is dragged around by blocked moves travelling
   zero and charge ticks travelling one extra. One-hot binning by magnitude
   makes the mean of the bins a histogram, and the mode a linear readout.

step_distance then went to 0.975 with a standard deviation of **0.001**,
having been +/- 0.239 before. Removing the variance is the part that says
this is a fix rather than a lucky seed.

### Layers

| layer | state |
|---|---|
| `core/` | infers dynamics including hidden state; 58.2% solve against a 60.0% ceiling |
| `explore/` | designs experiments with already-inferred rules |
| `adapt/` | works -- recovers the exact rule set 36% of the time with no labels |
| `memory/` | works on the wide space (4.5x better ranking); was *worse* than no memory on the narrow one |
| `evolve/` | mechanically correct -- archives every version, gates promotion on held-out worlds, rolls back |

With true mechanics the agent solves **60.0%** of compositional worlds, so
the planner handles the new rules and inference is the binding constraint.

### Phase 6: does any of it generalise?

The plan is blunt about this gate -- if it only works on grid worlds, the
result is an excellent ARC solver rather than a general system. The test
only means something if the second domain differs structurally rather than
cosmetically, which is why `gen/toy.py` does not count: it is still an agent
walking around a board.

`domains/dials.py` shares the observation *type* and nothing else. No agent
occupies a cell, no cell is blocked, no two actions interact through
geometry, and a dial's value is a magnitude rather than a position. It keeps
the property that makes the grid worlds hard -- a hidden coupling means two
identical frames can respond differently to the same action.

Over 24 episodes, with no change to anything above `env/`:

| | |
|---|---|
| verifier ranks the true rule set first | 24/24 |
| median rank of truth | 1 of 24 |
| histories able to test transitions | 24/24 |

So the world-model contract, the verifier and hypothesis search are domain
agnostic in fact and not just in intent. Two honest limits: the dial
hypothesis space holds 24 rule sets, so search is trivial there and this
tests correctness rather than scale; and the **core does not transfer at
all**, which is expected rather than disappointing -- its features are
per-value spatial moments, centroids and displacements, which describe a
board with things on it. A domain with no things and no board offers them
nothing to measure.

### Open problems

- **ordered_targets is under-determined, not unlearned.** The evidence
  distinguishes it in 6% of worlds under random play, against 100% for the
  hidden counter. Sitting near prior is the correct response to evidence
  that does not determine the label. `explore/` lifts it to 19% by inferring
  the movement rule and then planning to land exactly on a target -- greedy
  probing cannot, because collection happens only at the final cell of a
  move and landings fall 2.81 -> 1.19 -> 0.43 as step_distance goes 1 -> 3.
- **wait_advances_charge sits below its prior** (0.557 against 0.669).
- **Search is still cheap enough to be a real baseline.** 5,760 hypotheses
  cost ~101s exhaustively; brute force only becomes unusable past ~17,000.

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
