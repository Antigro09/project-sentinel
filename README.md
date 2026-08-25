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
uv run pytest tests/ -q                    # 227 passing, 1 skipped (~10 min)
uv run scripts/bench_engine.py             # engine throughput on this machine
```

The Level 5 substrates each run standalone and print their own measured
table -- the fastest way to see the current state is:

```bash
uv run python experiments/x47_priced_vocabulary.py   # the depth wall, priced
uv run python experiments/x56_byte_vm.py             # real text, quotiented
uv run python experiments/x58_sweep.py               # 14 parsing tasks
uv run python experiments/x61_working_set.py         # which memory class (~30s)
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
experiments/  the research arc, X1-X61. X17-X34 is the DSL loop; X35-X61
              dissolves the vocabulary anchor, moves the substrate onto real
              text, and measures the memory it needs -- each self-contained,
              with measured results (including the negative ones) in its
              docstring
tests/        determinism, verifier gate, planner, generator, bootstrap, and
              the X46-X61 substrates: interpreter/fast-path agreement,
              certificates, ablations, and the measurement artefacts that
              once produced wrong readings
```

Nothing above `env/` imports the ARC engine. That boundary is deliberate:
Phase 6 swaps a structurally different environment in behind it, and the
transfer result is only meaningful if nothing upstream had to change.
`gen/toy.py` is the first evidence it holds — it produces the same
`History` type through none of the engine's code.

## Status

**227 tests passing, 1 skipped. Level 5 built: the vocabulary anchor is
gone, the substrate runs on real text, and the memory it needs has been
measured rather than assumed (see Level 5 below).** The headline numbers in
earlier versions of this file were wrong, and the correction is the most
useful thing here.

### What this system is, stated precisely

Two claims that appeared in earlier drafts and in outside summaries are
**not** supported and should not be repeated:

- **It is not Turing-complete.** The substrate is finite control, a bounded
  number of finite-alphabet registers, and one pushdown stack. X60's own
  argument -- registers cost `|alphabet|^count` where a writable tape costs
  `|alphabet|^length` -- is exactly the statement that the register half is
  finite-state. Turing-completeness would not be evidence of generality
  anyway; a tiny interpreter has it.
- **It is not label-free discovery.** The defensible version: the system
  receives no semantic rule-class labels during hypothesis selection, but
  every task behaviour and target program in `experiments/` is
  human-authored. That is a real property and a much smaller one.

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
held-out mechanic combinations, scored only where each label is defined:

| label | core | prior | seeds |
|---|---|---|---|
| gates_start_open | **0.997 +/- 0.003** | 0.535 | 0.99, 1.00 |
| step_distance | **0.949 +/- 0.016** | 0.491 | 0.96, 0.93 |
| switches | 0.765 +/- 0.009 | 0.462 | 0.77, 0.76 |
| **charge_period** (hidden) | **0.590 +/- 0.121** | 0.298 | 0.71, 0.47 |
| edge_mode | 0.579 +/- 0.023 | 0.377 | 0.60, 0.56 |
| hazards | 0.563 +/- 0.005 | 0.342 | 0.56, 0.57 |
| wait_advances_charge | 0.563 +/- 0.110 | 0.612 | 0.67, 0.45 |
| ordered_targets | 0.560 +/- 0.036 | 0.539 | 0.60, 0.52 |

`charge_period` is the one that matters: a counter that makes every Nth move
travel an extra cell, invisible in any frame, recoverable only from a
pattern across a sequence. At twice its prior it is being inferred rather
than guessed. Note the spread, though -- 0.71 and 0.47 -- which is the
honest caveat on that claim and the next thing to fix.

Two labels are worth reading carefully rather than at face value.
`ordered_targets` barely clears its prior, and that is close to correct
behaviour: the evidence *determines* it in only 6% of worlds under random
play, so there is usually nothing to infer. `wait_advances_charge` sits
below its prior and is undefined in a quarter of worlds -- whether waiting
ticks a counter is unobservable where no counter exists.

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

step_distance then went to 0.95 with a standard deviation of 0.016, having
been +/- 0.239 before. Removing the variance is the part that says this is a
fix rather than a lucky seed.

4. **Training returned the last epoch's weights, not the best.** Three
   stopping criteria were tuned before noticing that each was really
   selecting a model by selecting when to quit -- the mean lets saturated
   heads hold it flat, the minimum hands a veto to a permanently stuck head,
   and resetting on any improvement lets noise train forever
   (step_distance +/- 0.001 -> +/- 0.306). Snapshotting the best-scoring
   epoch decouples the two: stopping late now costs time and nothing else.

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

## Level 4: the DSL loop (X17-X32)

The label vocabulary is gone. The system now infers **programs** from a
product grammar over transition rules -- 61,440 programs x collection
orders = 368,640 hypotheses, 3.6x past the point where exhaustive search
stops being viable. Every component of the loop speaks DSL; no label
vocabulary exists anywhere.

```
identifiability-aware generation   (X19)
  -> purposeful exploration        (X20: DSL-committee QbC + hazard-seeking)
  -> bulk refutation               (X17: all programs, seconds)
  -> simplicity / learned ranking  (X21-X23)
  -> planning inside the model     (plan/search.py)
  -> divergence-checked execution, re-inference on divergence
```

### The product demonstration

On fresh held-out worlds, the assembled agent:

| agent | solve rate | avg real actions |
|---|---|---|
| random actions | 2/12 | 529 |
| **full agent** | **9/12** | 231 |
| oracle (true model) | 10/12 | 12 |

At 40 worlds: **27/40 = 68% +/- 15**, average 229 actions. The agent wins
by exploring purposefully, inferring an executable model beyond exhaustive
reach, planning inside it, and re-inferring when reality diverges.

### What the experiments established

- The behavioural quotient stays small at scale: signatures ~ K^0.43
  (10,000 sampled programs -> 377 behaviours). Enumeration was never the
  right frame.
- Refutation is exact where fitness is coarse: coordinate ascent fails
  completely (no single-axis gradient); refute-then-select works.
- Identifiability is a property of the world distribution: hazard density
  must scale with step distance, or episodes die before the hidden counter
  can be observed (X18/X19).
- The narrowness migrates: when the explorer's committee was still drawn
  from the old label space, purposeful exploration lost to random walk
  (X18). Every component must speak the current hypothesis language.
- Charge_period at extended ranges (up to 20) is bounded by encoded window
  size, not epochs (X23-X26); three representational fixes failed (X27/
  X28). It remains refutation-resolved only.
- The dominant remaining loss is painting yourself into corners:
  infer-then-solve gathers evidence without regard for whether the task
  stays solvable. X31 diagnosed it; solvability-aware exploration (X32)
  is the fix under test.

### Beyond the grammar anchor (X33-X34)

The eight DSL axes are a closed vocabulary: a world whose dynamics do not
fit them is unlearnable by construction. X33 demonstrated the anchor and
broke it -- an elementary-CA world behind the same env boundary, its rule
recovered exactly by refutation within a new axis. X34 removed the human
from the loop: candidate axis FAMILIES generated from a generic
meta-grammar, scored by version-space collapse, the least expressive
viable family adopted. 3/3 hidden worlds from three different families,
rules recovered exactly, 0.0s. The load-bearing discovery: max-collapse
always picks the most expressive family (it contains the others);
vocabulary growth must be EARNED by the refutation of everything smaller.
This is recursive self-improvement in its smallest measurable form.

### Experiment index

| # | file | question |
|---|---|---|
| X17 | `experiments/x17_dsl_search.py` | does the DSL regress, and does search survive scale? |
| X18 | `experiments/x18_exploration_resolution.py` | which exploration policy raises evidence resolution? |
| X19 | `experiments/x19_identifiable_worlds.py` | does identifiability-aware generation close the loop? |
| X20 | `experiments/x20_dsl_explorer.py` | does a DSL-committee explorer beat random? |
| X21 | `experiments/x21_derivation_core.py` | can a derivation-trained core rank survivors? |
| X22 | `experiments/x22_scaled_loop.py` | does training scale close the per-head gap? |
| X23 | `experiments/x23_charge_recipe.py` | does the full recipe fix charge? |
| X24 | `experiments/x24_charge_window.py` | is charge bounded by window size? |
| X25 | `experiments/x25_long_episodes.py` | does the window fix replicate? |
| X26 | `experiments/x26_seed_metrology.py` | seed noise or configuration signal? |
| X27 | `experiments/x27_autocorr_encoder.py` | mass-change autocorrelation features? |
| X28 | `experiments/x28_centroid_autocorr.py` | centroid-displacement autocorrelation? |
| X29 | `experiments/x29_full_agent.py` | does knowing the rules let it win? |
| X30 | `experiments/x30_agent_taxonomy.py` | solve rate at scale + failure taxonomy |
| X31 | `experiments/x31_no_plan_diagnosis.py` | why do no-plan losses happen? |
| X32 | `experiments/x32_solvability_aware.py` | does solvability-aware exploration fix them? |
| X33 | `experiments/x33_ca_world.py` | can the grammar anchor be broken symbolically? |
| X34 | `experiments/x34_axis_synthesis.py` | can the system generate AND choose its own axes? |

Each docstring carries its full measured results, including the negative
ones -- five of this arc's experiments failed, and each failure named the
next experiment.


## Level 5: dissolving the vocabulary, then finding the memory (X35-X61)

Level 4 ended with a system that could generate and choose its own *axes*.
It still had an anchor underneath: eight hand-written rule menus. This arc
dissolved that, moved the result onto real text, and then measured what
memory the resulting machine actually needs.

```
invented rules      (X44/X45: motion and hazards built from atoms, not chosen)
  -> one substrate  (X46: both domains, one primitive pool)
  -> priced search  (X47: the depth wall was accounting, not expressiveness)
  -> token/byte VM  (X48/X56: derive tests from events; the lattice is gone)
  -> memory         (X49/X50/X60: counter, typed stack, registers)
  -> measurement    (X61: which memory class real tasks actually demand)
```

### The substrate stopped being a menu

X44 and X45 made motion and hazards **generative** -- `slide` is not a
symbol anywhere, it is `(COND FREE (ADD1 REC) ZERO)` recovered uniquely from
displacement evidence. X46 unified both into one pool where no primitive
names a domain, and charged for it: a proximity mine went from 3 nodes to
15, past what enumeration reaches.

X47 showed that wall was **accounting, not expressiveness**. The predicate
lattice closes under OR at 403 truth vectors, so `near` was in the pool the
whole time; size simply charged 7 nodes for one lattice element. Re-pricing
the vocabulary recovered the 15-node rule at 9,000 candidates against the
~2x10^9 that size-first enumeration needs.

| arm | X47 grid | X48 tokens |
|---|---|---|
| cover (coverage + model) | **6/6** | 6/6 |
| learned (model rank) | 5/6 | 6/6 |
| size | 4/6 | 3/6 |
| similar | 3/6 | 3/6 |
| random *(calibration)* | 1/6 | 4/6 |

### The port that wasn't a port

X47's foundation is that the lattice is small. On a token stream it is a
**powerset** -- character tests at one offset are mutually exclusive, so
unions never collapse: 1,024 at one offset, 55,783 at two, 344k+ at three.
The answer was to stop enumerating tests and **derive them from events**:
`is-digit` falls out of "where did the target halt" in 11 comparisons.

X56 carried this to real text (JSON, code, comments) and found the second
scaling wall -- the stack's state space is `|alphabet|^depth`, 14.47 MB per
behaviour on real bytes. The fix is exact rather than approximate: a program
can only inspect the stack through the `TOP(c)` tests it has, so bytes with
no test are indistinguishable and collapse losslessly. 35x-100x smaller,
verified against the interpreter.

### Memory, measured

| experiment | result |
|---|---|
| X49 | window certificate: same window, opposite decisions -- no window-only program can nest. Bounded counter fixes it. |
| X50 | a counter cannot remember *which* bracket. Same window **and** depth, opposite decisions. Typed stack, 5/5, depth-6 generalisation. |
| X59 | two read heads; `SAME` load-bearing on 3/6 by ablation. A read/write scratchpad prices at **7,338 GB per behaviour**. |
| X60 | the proposed fix for that -- collapse states agreeing on current predicates -- is **unsound**, and the counterexample is one `ADV` long. Bounded registers are exact and cheap: 0.0077 MB. 4/5 tasks need them. |
| X61 | **half** of ordinary parsing needs a working set that grows with the input, not "almost all bounded". |

X61's split, measured on the diagonal (growing prefix and suffix together,
because fixing either manufactures a false plateau):

```
strip comment / capture quoted / dedupe adjacent / emit matching first   bounded
capture brackets / only at depth 2 / balanced prefix / reverse           GROWING
```

Every growing task is one the machine already solves -- with the stack.
Registers and stack turned out **complementary for compact synthesis**, and
that is a claim about representation, not expressiveness: a general pushdown
machine could simulate a register.

### What the negative results established

Four experiments produced no positive result and they were the most useful
of the arc. X51 measured a "deceptive valley" and found that widening the
beam (1->8) and enriching the vocabulary both fail. X52 built lookahead
credit assignment: 10x the cost, and it *lost* a target greedy solves.

X53 explained all of it. A monotone route to `copy inside any` **exists**;
the "decoy" that three experiments were built around is its *first step*.
Greedy takes the argmax while the route only needs non-decreasing, and a
per-round beam discards what it needs, permanently. The fix was bookkeeping
-- one global frontier, no discard -- not scoring. 5/5.

X54 then tried to learn a ranker for that frontier and could not: an
efficient search generates no training data, and an inefficient one solves
nothing to label. X55 tried to build a curriculum and measured the band at
**1 usable task in 22** -- roughly 20 hours for a 100-task corpus. Across
X53-X55, **the learned component has no measured role in this substrate.**

### Three shape gaps in three experiments

The most transferable finding. Tasks that scored `--` and looked like search
failures were not:

- **X58** `halt at m` -- needed a lookahead test; offset-0 tests cannot
  express it at all.
- **X59** `zip both` -- needed a depth-2 rule body.
- **X60** `SEQ(LOAD, LOOP(...))` -- needed a prologue shape.

None was a search problem. When a task fails, ask whether its *shape* is in
the language before asking whether the search is strong enough -- probing
shapes is nearly free, since a shape that fits is found in 26-78
evaluations against a 400-state budget.

### Primitives must earn their place

Every new primitive is ablated against the tasks that motivated it:

| primitive | verdict |
|---|---|
| `EQTOP` (X51) | appears in **0 of 4** recovered programs -- did not earn it |
| `SAME` (X59) | **3 of 6** tasks unrecoverable without it |
| register `LOAD`/`MATCH` (X60) | **4 of 5** tasks unrecoverable without it |

### Measurement artefacts caught

- **X61**: two readings said nested brackets need bounded memory. Both were
  horizon artefacts -- a fixed suffix caps the index at what a short suffix
  distinguishes. The wrong measurement looks tidier than the right one.
- **X58**: a `pos-1` clamp made the first byte of every tape look
  halt-associated, deriving a spurious marker.
- **X46**: `STEP` refused to enter walls, so the primitive was doing the
  collision test the program was meant to express.
- **X50**: halting collapsed to one sentinel, so `SEQ(ADV,HALT)` and `HALT`
  had identical tables while the interpreter told them apart.

Every one of these made the system look better than it was.

### Open problems

- **No associative memory.** Registers are bounded, the stack is LIFO.
  Symbol tables, maps, graphs, object identity and cross-references have no
  representation here, and X61 says the memory question is a hierarchy
  rather than one substrate.
- **The learned component still has no measured role** (X53-X55). Any future
  claim for it needs a curriculum that does not yet exist.
- **Tasks and shapes are hand-authored.** Target programs are written by
  hand, and each of the three shape gaps was closed by hand.
- **Scope is small.** Five-byte alphabets, tapes of at most eleven bytes,
  stack depth 2 in search. X56's real-text tasks are the largest thing run.

### Experiment index (X35-X61)

| # | file | question |
|---|---|---|
| X35 | `x35_novelty_trigger.py` | can an agent detect that its grammar is inadequate? |
| X36 | `x36_micro_vm.py` | does a micro-VM substrate beat axis menus? |
| X37 | `x37_micro_rsi_memory.py` | does a learned library compound across worlds? |
| X38 | `x38_atomic_synthesis.py` | can rules be built from atoms rather than chosen? |
| X39 | `x39_probe_scaling.py` | how does probe cost scale with hypothesis count? |
| X40 | `x40_free_recursion.py` | can free recursion be enumerated without divergence? |
| X41 | `x41_primitive_lineage.py` | where does each primitive come from? |
| X42 | `x42_trigger_guard.py` | novelty or corrupted evidence? |
| X43 | `x43_scaled_growth.py` | does the growth loop hold on 20+ worlds? |
| X44 | `x44_invented_motion.py` | can the grid invent its movement rule? |
| X45 | `x45_invented_hazards.py` | can it invent hazard rules the axes cannot express? |
| X46 | `x46_unified_substrate.py` | one pool for motion and hazards -- at what cost? |
| X47 | `x47_priced_vocabulary.py` | is the depth wall expressiveness or accounting? |
| X48 | `x48_token_vm.py` | does the substrate survive a token stream? |
| X49 | `x49_nested_structure.py` | what can no window-only program do? |
| X50 | `x50_stack.py` | what can no counter remember? |
| X51 | `x51_deceptive_valley.py` | how much lookahead does the landscape need? |
| X52 | `x52_lookahead.py` | does scoring what a rule enables beat what it fixes? |
| X53 | `x53_monotone_reachability.py` | unreachable, or merely unfound? |
| X54 | `x54_frontier_ranker.py` | can a ranker pay back the frontier's cost? |
| X55 | `x55_curriculum.py` | can the difficulty band be farmed? |
| X56 | `x56_byte_vm.py` | does it survive real text? |
| X57 | `x57_repair.py` | can a program be generalised after the fact? |
| X58 | `x58_sweep.py` | how often does the one-rule repair bound bind? |
| X59 | `x59_multistream.py` | two read heads, and what a scratchpad costs |
| X60 | `x60_registers.py` | can the machine read what it wrote? |
| X61 | `x61_working_set.py` | which memory class does real parsing need? |

### The next experiment

X62, a **memory-class audit**: measure expressibility, memory growth,
synthesis cost and generalisation *separately* across parameterised task
families -- streaming, register, stack, associative, sequence, set and
graph-shaped. The decision it settles: whether stack plus a few registers
covers realistic work, or whether the next mechanism has to be a sparse
mutable store that executes only the keys a candidate program touches.
Building a POSIX substrate first would wrap the architecture in a larger
environment without answering that.
