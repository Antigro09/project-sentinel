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

**Plan of record: [ROADMAP.md](ROADMAP.md)** -- X63 through X73, in
dependency order, each with the gate it has to pass. Earlier plans, with
their falsifiable claims and kill criteria, are in
`~/.claude/plans/`; the five-gap plan is reconciled at the bottom of the
roadmap rather than discarded.

## Setup

```bash
uv sync
uv run scripts/fetch_games.py   # one-time; the only network access in the project
```

After that, everything runs offline.

## Verify

```bash
uv run pytest tests/ -q                    # 301 passing, 1 skipped (~14 min)
uv run scripts/bench_engine.py             # engine throughput on this machine
```

The Level 5 substrates each run standalone and print their own measured
table -- the fastest way to see the current state is:

```bash
uv run python experiments/x47_priced_vocabulary.py   # the depth wall, priced
uv run python experiments/x56_byte_vm.py             # real text, quotiented
uv run python experiments/x58_sweep.py               # 14 parsing tasks
uv run python experiments/x61_working_set.py         # which memory class (~30s)
uv run python experiments/x63_sparse_price.py        # what the table buys (~1s)
uv run python experiments/x63c_gate.py               # the twelve clauses (~1s)
uv run python experiments/x64a_identify.py           # the eight gates (~70s)
uv run python experiments/x64b1_openworld.py         # open-world, 9 gates (~65s)
uv run python experiments/x64b2_language.py          # language, 10 gates (~55s)
uv run python experiments/x64c_frozen.py             # frozen audit, 10/12 (~65s)
uv run python experiments/x64d_senses.py             # induced senses, 9/10 (~60s)
uv run python experiments/x64e_semantics.py          # posterior semantics, 12/12 (~16s)
uv run python experiments/x64e_audit.py              # the F-1 audit (~15s)
uv run python experiments/x64f_context.py            # contextual, 12/12 (~32 min)
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

**301 tests passing, 1 skipped. Level 5 built: the vocabulary anchor is
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
| X62 | `x62_memory_audit.py` | which memory SHAPE, with four quantities kept apart? |
| X63a | `x63_sparse_price.py` | what does the behaviour table actually buy? |
| X63b | `x63b_cegis_store.py` | a sparse store, searched without any gradient |
| X63c | `x63c_gate.py` | twelve clauses specified from outside, not by me |
| X64A | `x64a_identify.py` | does it know WHICH task it was asked to do? |
| X64B-1 | `x64b1_openworld.py` | can it notice that none of its interpretations fits? |
| X64B-2 | `x64b2_language.py` | can an instruction narrow the space without naming the task? |
| X64C | `x64c_frozen.py` | the same lexicon, frozen, against tasks it has never seen |
| X64D | `x64d_senses.py` | senses induced from evidence; language that cannot delete |
| X64E | `x64e_semantics.py` | a distribution over logical forms; conflict as posterior mass |
| F−1 | `x64e_audit.py` | the X64E audit; two claims come down |
| X64F | `x64f_context.py` | surface language a word-to-slot table cannot decode |

### X62: the memory audit, and what it decided

Eleven tasks across six families, with the four quantities kept apart:

| family | expressible | memory class |
|---|---|---|
| streaming | 2/2 | constant -- registers |
| register | 2/2 | constant -- registers |
| stack | 2/2 | linear -- counter or stack |
| set | **0/2** | converging at 2^\|alphabet\| -- set-shaped |
| sequence | 1/2 | `reverse` is the only task growing with input length |
| associative | **0/1** | converging -- table-shaped |

**Only `reverse` grows with input length.** Set and associative memory are
*bounded* for a fixed alphabet -- 32 and 58 classes -- so the gap is not
capacity but **shape**: a register holds a byte, and neither a subset nor a
key-value map is a byte. X61's "growing working set" framing was too coarse
to see that. The bound is exponential in *alphabet* size, so scaling the
alphabet is what forces the issue, not scaling the input.

The pre-registered rule fired: set 0/2 and associative 0/1 means the next
mechanism is a **sparse mutable store**, keyed and executed lazily -- not a
bigger substrate, and not a writable tape at 7,338 GB per behaviour.

Two findings worth as much as the decision. Reading the index from its last
two values called the set tasks "linear" -- their increments run 10, 10, 5,
1 and 32 is exactly 2^5, so a plateau arriving looks like growth. And **four
of seven expressible tasks were not found** within ~10^6 evaluations, with
one found program failing held-out: expressibility and findability come
apart on more than half the suite, which is why they are separate columns.

### The next experiment, and the eleven after it

The plan of record is **[ROADMAP.md](ROADMAP.md)**: X63 -> X73 in dependency
order, each with an explicit gate. The short version of why it is shaped
that way:

After X62 this project stops adding one primitive at a time and starts
crossing the gaps that separate it from human-level intelligence. The next
step is **not** POSIX, not a larger model, not self-modification. It is to
remove the strongest remaining source of human supervision, which this
README already names two sections above: **we define what a task means by
handing the system a target program.**

- **X63, the sparse store.** X62's pre-registered rule already picked the
  branch -- set 0/2 and associative 0/1 means an exact sparse key-value
  store. **X63a priced it, and priced the wrong risk** (below): the danger
  was never speed. X63b builds the store on concrete execution with
  counterexample-guided repair rather than distance-ranked search.
- **X64, task and goal induction.** The most important transition on the
  list: instruction plus ambiguous demonstrations, no target program, and a
  version space over *interpretations* -- so the system can tell "my
  implementation is wrong" from "my interpretation is wrong" from "the
  request is underspecified" from "this is impossible with these tools."
- **X65 - X73:** lifelong memory, real software engineering, cross-domain
  transfer, uncertainty, grounded causal models, multimodality, long-horizon
  agency, controlled self-improvement, and an integrated evaluation with the
  architecture frozen.

Until X64's gate passes, the honest description of this system stays what it
is now: a synthesiser operated by a human experiment designer.

### X63a: the table is a resolution device, and I priced the wrong risk

A store over 5 keys has (5+1)^5 = 7,776 configurations, so tabulating it
multiplies X62's situation space by the same factor -- 2,232 situations at
0.107 MB per behaviour becomes 17,356,032 at **833 MB**. The store therefore
forces concrete execution, and the pre-registered worry was that execution
could not carry X58-X62's ~10^6-evaluation budgets.

**That worry was backwards.**

| evaluation | us/candidate | vs the search step |
|---|---|---|
| table, rebuilt from atoms | 525.3 | 2.4x |
| table, one rule + re-wrap (what search pays) | 218.4 | 1.0x |
| concrete execution | 6.3 | **0.03x** |
| concrete execution **+ store** | 6.3 | **0.03x** |

Execution is **35x faster** than the step the frontier actually pays, and
the store costs nothing measurable on top -- a run touches the keys it
touches whatever the key space is. The table was never a speed device at
this scale. It is a **resolution** device, bought with memory rather than
time, which is exactly why the store kills it.

What it sells is the search gradient. Over 300 sampled programs the full
signature separates 211 behaviours where outputs alone separate 22. As a
gradient toward `capture brackets`, with four output metrics run as
calibration arms so no single crude metric could manufacture the answer:

| metric | levels | vs table | r (all) | r (top 10%) |
|---|---|---|---|---|
| **full table** | 123 | 1.00x | - | - |
| positional | 10 | 0.08x | 0.167 | 0.398 |
| exact match | 2 | 0.02x | -0.012 | **-0.682** |
| common prefix | 6 | 0.05x | 0.124 | 0.294 |
| character bag | 8 | 0.07x | 0.190 | 0.555 |

None tracks the table, and `exact match` is *anti*-correlated among near
misses: matching one more tape exactly can mean moving further away in the
only ordering the search can act on.

**The resolution is real, not phantom.** The obvious objection is that the
signature scores all 2,232 situations while a run from a fresh start reaches
almost none, so most of that precision would be agreement on dead states.
Only 390 situations (17%) are reachable -- and restricted to those the table
still separates **129** classes, slightly more than the 123 it separates
overall. The precision is about programs, not about states nobody visits.

So clause (a) passed by a factor of 1,600 and clause (b) failed. Speed
survives; the gradient does not. Eight levels cannot order a queue that 123
levels barely ordered -- X62 failed to find 4 of 7 expressible tasks *with*
the table. **X63b is counterexample-guided:** localise the first divergence
between the candidate's output and the evidence and repair at that point
(X57's mechanism), which needs no global gradient at all. That is a
different mechanism, not a flag on this one.

One design consequence fell out of building the store interpreter. Emission
is currently tape-index-valued (`out` holds positions), and the table encodes
it as counts over tape positions -- but `substitute` must emit a value that
need not sit at the head. Keeping index emission would mean storing
*positions* rather than bytes, and positions grow with the tape, so the store
would inherit exactly the unboundedness it was introduced to avoid. X63b
emits bytes.

### X63b/c: the store passes twelve clauses, the search regresses on two

The gate for X63 was specified from outside this project, not by me, and it
is stricter than the two clauses X63b had already declared itself to pass.

**The mechanism passes.** Set 2/2, associative 1/1 -- all three tasks X62
proved inexpressible now have witnesses. Held-out split by axis so a failure
names its axis: longer tapes, unseen symbols, unseen **keys**, unseen
**values**, unseen keys+values, 5/5. Ablation: removing `PUT`/`GET`/`HAS`
fails exactly those three and leaves the other seven untouched. `reverse`
stays a control -- no positional indexing, no store iteration.

Cost follows what is touched, not what is possible, which was the whole
argument for a sparse store:

| key universe | keys touched | us/run | if tabulated |
|---|---|---|---|
| 5 | 1 | 11.6 | 7.78e+03 |
| 50 | 1 | 11.4 | 3.10e+20 |
| 500 | 1 | 11.0 | 2.50e+32 |
| 5,000 | 1 | 11.0 | 2.45e+44 |

1.06x runtime across a 1,000x universe, with 200 keys, a 200-deep stack and
200 emitted bytes all unbounded.

**Two clauses caught real problems.**

*The differential test passed while being vacuous.* Its calibration arm -- an
interpreter crippled the exact way X62's actually was, with a capped `PUSH`
-- was caught on **0 of 400** programs. The cause is a genuine property of
X62's evidence rather than a coding slip: on five-byte tapes a run only
exceeds depth 2 by pushing the *same* byte repeatedly, so `TOP` reads the
same value at every level and no emitted byte can depend on the depth. 129
runs went over the bound and not one changed its output. Rebuilt on
seven-byte tapes with adjacent distinct bytes plus a hand-built
depth-sensitive probe: 2 of 10 probes now distinguish a capped stack, and
the real test passes with 0 unexplained mismatches. **A differential test
without a calibration arm proves nothing, and this one proved nothing while
printing PASS.**

*The search regresses.* X62's table search solved `capture brackets` and
`emit matching first` cleanly; CEGIS loses both and gains `balanced prefix`
and `delayed copy`. Same count, different three -- reported as its own
clause rather than folded into the witness count, which is 7/7 and would
have hidden it.

**And the equivalence key was worth more than the search machinery.**
Output-only merges 4 of 23 behaviour classes, with up to 5 distinct stores
collapsed into one -- three programs that emit nothing on all four training
tapes while holding `{}`, `{'a':'a'}` and `{'(':'('}`. Keying on
`(output, store)` splits 23 classes into 30 and merges none. That one change
took the plateau arm from 8 found to **10 of 10**, and its worst case from
**12,830,685 evaluations to 50,829**, because a plateau move that keeps
`width` alternatives had been keeping `width` copies of a single behaviour.

Against X62, at the end:

| | expressible | found | generalises | worst evals |
|---|---|---|---|---|
| X62, table + frontier | 7 | 4 | 3 | 976,521 |
| X63b arm A, store + strict CEGIS | 10 | 7 | 3 | 16,750 |
| X63b arm B, + plateau | 10 | **10** | 3 | 50,829 |

**A bug the gate found.** `GET` was the only emitting act without an
end-of-tape guard, so a `LOOP` containing it never reached a fixed point:
`substitute` on `'(ab)a'` emitted `'bbbbbb'`. Every other act -- `EMIT`,
`LOAD`, `PUSH`, `PUT` -- already had it.

### And the finding underneath it: fitting is not identifying

CEGIS finds **10 of 10** and **3 of 10 generalise**. That gap is the real
X63 result, so it got four explanations and four experiments rather than a
sentence:

| task | 2 tapes | 4 | 8 | minimised | shape forced |
|---|---|---|---|---|---|
| strip comment | 10 | 10 | 10 | 10 | 10 |
| capture quoted | 5 | 6 | 3 | 3 | 6 |
| dedupe adjacent | 3 | 4 | **10** | 4 | 4 |
| emit matching first | 0 | 3 | 3 | 3 | 3 |
| capture brackets | 6 | 3 | - | 5 | - |
| balanced prefix | 10 | 10 | 10 | 10 | 10 |
| first occurrence only | 1 | 0 | **10** | 0 | 0 |
| emit if seen before | 1 | 1 | - | 1 | 2 |
| delayed copy | 10 | 10 | 10 | 10 | 10 |
| substitute | 6 | 9 | - | 9 | - |
| **TOTAL held-out** | **52** | **56** | **56** | **55** | **45** / 100 |

- **Thin evidence** -- *not monotone, and nearly flat.* 52 → 56 → 56 while
  the evidence doubles twice. Two tasks climb to 10, two collapse. More
  evidence is not a reliable fix.
- **No simplicity bias** -- *refuted.* Deleting every rule that can be
  deleted moves the total from 56 to 55.
- **A missing shape** -- *refuted*, and this is the decisive one. The
  witness for `first occurrence only` is `LOOP(SEQ(LOAD, IF(HAS, …)))`, and
  CEGIS's shape is `LOOP(IF …)` -- so `HAS` could only ever describe the
  *previous* byte. Adding the loop prologue, then **forcing** it, still
  gives 0/10: the returned program even opens with the right rule
  `IF HAS then NOP` and then piles on `AT` tests that fit by accident.
  Forcing the shape lowers the total to 45, because the shapes it displaces
  were carrying other tasks.
- **A weak search** -- *refuted.* It finds 10 of 10.

**A correct program exists in exactly the shape the search was handed, and
the search returns a different one.** Nothing in this machine prefers the
general program, and nothing in it can notice the question is undetermined.
That is precisely what `measure_identifiability.py` reports about
`ordered_targets` at the other end of this repo, and it is the content of
X64 arrived at from the opposite direction.

Three consecutive shape gaps became four, and then the fourth turned out
not to be a shape gap at all. The rule the README has carried since X60 --
*check the shape before blaming the search* -- needs a second clause:
**check whether the evidence identifies the answer before blaming either.**

## X64A: does it know which task it was asked to do?

X63's verdict, stated correctly: **it did not pass its end-to-end gate.**
The sparse store is validated on every mechanism clause; the synthesis is
not, at 3 of 10 held-out; and two tasks X62 solved were traded away. A green
unit suite does not cancel a benchmark regression, and the two should never
be reported as one number.

X64A stops searching for *a* program and starts representing **which tasks
are still possible.** Every task gets one of three states, and has to be
right about which:

| surviving behaviour classes | state |
|---|---|
| 0 | inconsistent or inexpressible |
| 1 | identified |
| 2+ | **underspecified** — and it must say so *before* answering |

Candidates are clustered by **behaviour over a fixed universe of inputs**,
not counted syntactically, so "no legal query can separate these" is a
representable stopping condition rather than an infinite loop. The target is
hidden: it is never inspected, and acts only as a synthetic user answering
one input at a time. The candidate pool is **task-independent** — one pool,
eleven tasks, no labels — so seeding it with every witness leaks nothing
about which task is being asked.

**Nine of eleven tasks are underspecified by the demonstrations alone.**
That is the state X63 could not represent, and answered from anyway, ten
times out of ten.

| arm | answered | queries | held-out | what it is |
|---|---|---|---|---|
| simplest | 1/11 | 0 | 10 | commit to the smallest fit |
| passive examples | 5/11 | 57 | 50 | examples chosen by nobody |
| random queries | 7/11 | 38 | 70 | ask, but ask arbitrarily |
| **disagreement** | **9/11** | **30** | **90** | ask what splits the survivors |
| oracle greedy | 9/11 | 25 | 90 | allowed to know the answers first |

Over 24 seeds, disagreement spends 30.0 queries against random's 36.1 ± 2.9
and scores 90.0 against 79.6 ± 6.8 — outside a standard deviation on both,
and matching the oracle's accuracy for five extra queries. An earlier draft
of that gate passed on a single seed's 13-versus-14 and would have meant
nothing.

**Four kinds of failure, kept apart**, because "it overfitted" is not a
diagnosis and is the only one X63 could report:

| diagnosis | n | |
|---|---|---|
| resolved | 9 | |
| underspecified | 1 | `delayed copy` — budget spent, still open, **reported** open |
| incomplete candidates | 1 | `reverse` — no expressible hypothesis fits |
| search-selection | 0 | the box X63 was in: target fits, something else returned |

### Four of the eight gates could not have failed, so they were calibrated

G1, G2 and G8 are structurally unfailable against a well-behaved system —
`run_arm` only answers when one class survives, so of course it never
answers early. They are only meaningful because they are run against
known-bad arms: **`reckless`** has no ambiguity state and answers the
simplest survivor from the demonstrations (this is X63), and **`paranoid`**
always claims ambiguity. The checks catch reckless on 9 tasks — wrong on 2 —
and paranoid on all 9 identified ones.

### The honest limit

Selecting a hypothesis from a pool containing a correct one is **easier than
synthesising it**. So the identical procedure runs on a **blind** pool built
with no witness seeded: 6 of 11 targets present, 5 resolve correctly, 3
correctly reported inconsistent — and **1 is identified confidently and
wrongly**. When the target is absent, every hypothesis the system can
express may agree, so it converges to one and says `identified`. It cannot
see the outside of its own pool. That failure is undetectable from inside,
which is exactly why `incomplete candidates` is its own diagnosis.

**8/8 gates pass.** The state this machine was missing was never another
byte, stack, or map. It was *"I do not yet know which task you mean."*

## X64B: open-world goal induction

X64A's own limit, attacked in two stages. **B-1** asks whether the system
can notice that *none* of its interpretations is adequate. **B-2** asks
whether an instruction can narrow the space without naming the task.

### B-1: a richer hypothesis pool makes confident wrongness *more* likely

A singleton version space does not imply correctness — it implies
uniqueness inside the current hypothesis class, and no survivor-count rule
can tell the difference, because the count is 1 and the rule is satisfied.
So identification gets three external criticism steps: **confirmation** on
challenge inputs longer than anything queried and carrying symbols the
universe does not contain, an explicit **none-of-the-above**, and
**expansion** that grows the space instead of picking again inside it.

Each rung of the expansion ladder adds exactly one thing, so the rung that
recovers a task *measures* what was missing:

| task | recovered at | what was missing |
|---|---|---|
| strip comment | base | nothing |
| dedupe adjacent | +memory | the `MATCH` test |
| first occurrence only | +shape | the loop prologue |
| emit if seen before | +shape | same |
| delayed copy | +vocabulary | an offset-2 test |
| 5 others | never | nothing expressible is adequate |

With the target removed and every rung pinned in turn:

| rung | naive: wrong | confirm: wrong | naive answered |
|---|---|---|---|
| base | 0 | 0 | 2 |
| +memory | 0 | 0 | 3 |
| +shape | 1 | 0 | 5 |
| +vocabulary | 0 | 0 | 5 |
| +search | **2** | 0 | 6 |

**Richer pools can create wrong singletons that poorer pools avoid by
reporting inconsistency, and the richest rung tested produced the most.**
The sequence is 0, 0, 1, 0, 2 — *not* monotonic, and an earlier draft of
this paragraph said "goes up with pool richness", which the numbers do not
support. What they do support: a poor pool says `inconsistent` and is right
to, while a rich pool can produce a singleton that survives every question
anyone thought to ask and is still wrong. Expansion buys recall and can cost
safety at the same time.

One thing the measurement forced: exact equivalence over the universe is
**stricter** than "produces the intended behaviour on everything anyone will
check". `balanced prefix` has no exact match in any pool and is still right
on every held-out tape, differing from the target only on universe inputs
nobody asked about. The first draft of that gate demanded a wrong
abstention. **Abstention tracks adequacy, not identity.**

**9/9 gates.**

### B-2: words carry constraints, phrases do not carry tasks

The trap: a "language" that maps a phrase to a memorised task label learns
nothing and generalises to no paraphrase. So the lexicon maps **words to
behavioural predicates** and an instruction means the *conjunction* of its
words' constraints. The semantics is authored — that is what "controlled"
means, and it is supervision. What is tested is whether it **composes**.

| instruction | narrows 3,965 → | target kept |
|---|---|---|
| copy what is inside the brackets | 21 | yes |
| keep the symbols seen before | 39 | yes |
| remove repeats in a row | 534 | yes |
| **remove repeats** | **2,461** | *ambiguous — adjacent, or first-occurrence?* |
| replace names using the table | 3,965 | *no constraint — correctly weak* |

Getting there took three real lexical corrections, each caught by a gate:
`within` had inherited a bracket constraint, so *"keep the characters within
the hash"* excluded its own target; `first` had taken the uniqueness reading,
so *"the symbols matching the first"* excluded its own target; and `comment`
alone selected exactly **one** behaviour — a task identity wearing a word's
clothes, the precise trap the design claims to avoid. `first` is genuinely
ambiguous — positional in one phrase, uniqueness in the other — and a
bag-of-words semantics *cannot* disambiguate it, so the constraint has to
live on the words that are not.

| arm | answered | correct | queries | held-out |
|---|---|---|---|---|
| demos only | 10 | 10 | 14 | 100 |
| language only | 10 | 10 | 27 | 100 |
| language + demos, silent | 7 | 7 | 0 | 60 |
| + random queries | 10 | 10 | 17 | 100 |
| **+ disagreement** | **10** | **10** | **8** | **100** |
| oracle (knows the answers) | 10 | 10 | 5 | 100 |

Language and demonstrations each reach 10/11 alone, and together in silence
reach 7. **What language buys is questions** — 8 against 14, the fewest of
any policy without future knowledge, at 1.6× the oracle's floor.

The two negative conditions are the point: with the target removed, **0**
confident errors and 4 none-of-the-above after climbing every rung; with the
instruction contradicting the demonstrations, 8 tasks report **conflict** and
none forces a wrong program.

> **The 24/24 paraphrase figure is development-set performance and is
> retracted as evidence of generalisation.** The lexicon was edited three
> times in response to failures on those exact paraphrases. It shows the
> final lexicon covers the examples used to debug it. X64C tested the
> frozen lexicon on unseen compositions and the result is below.

**10/10 gates — on the suite the lexicon was authored against.**

### X64C: the same lexicon, frozen — and the claim comes down

The lexicon and predicate set are **hashed** into `x64c_frozen.py`. Editing
either makes the experiment refuse to run, so a holdout failure cannot be
quietly repaired. Twelve new task behaviours were built from primitives the
lexicon knows, in combinations it was never authored against, with 48
instruction forms — and all of it was committed *before* the protocol ran.

**10/12 gates. The two failures fire pre-registered falsifiers.**

| arm | answered | correct | wrong | queries | held-out |
|---|---|---|---|---|---|
| **demos + disagreement** | **10** | **10** | 0 | 23 | **100** |
| language + disagreement | 3 | 3 | 0 | 38 | 30 |
| language + demos + dis. | 3 | 3 | 0 | 5 | 30 |
| random clarification | 3 | 3 | 0 | 14 | 30 |
| no confirmation | 4 | 3 | **1** | 4 | 35 |
| oracle | 3 | 3 | 0 | 8 | 30 |

**C1 fails: the query advantage does not survive — it reverses.** On the
development tasks language cut questions from 14 to 8. Here demonstrations
alone answer 10 of 12 and language cuts that to **3**. On unseen
compositions the frozen lexicon is not merely unhelpful, it is harmful.

**C6 fails: the gains are confined to the development families.** 3 of 12
compositional-holdout tasks solved exactly, against 10 of 11 on the tasks
the lexicon was authored against.

The cause is measured, not guessed: **22 of 40 holdout instruction forms
exclude their own target**, against **0 of 30** on the development set.
`brackets` carries *only inside brackets* — authored for "copy what is
inside the brackets" and simply wrong for "remove the brackets". The same
polysemy as `first` and `comment`; the same fix would work; the fix is
exactly what the freeze forbids. **Fitting a lexicon to the evaluation
suite is what X64B-2 did, and this is what it cost.**

What does survive, and it is not nothing:

- **The failure mode is safe.** 22 false exclusions, **0** confident errors.
  A lexicon that excludes the truth makes the system say `CONFLICT`, not
  guess.
- **Conflict detection transfers.** 10 of 12 unseen mismatched pairs
  flagged, 0 forced — it was not memorising the eight authored examples.
- **Confirmation still earns its place.** With it, 0 confident errors;
  without it, 2.
- **Unseen *form* is not the problem — unseen *composition* is.** Where the
  canonical instruction answered at all, 6/6 unseen forms (new word orders,
  44 out-of-vocabulary words) landed in the same behavioural class. Small n,
  and honestly so: an earlier draft of that measurement scored two failures
  as an agreement when both returned nothing, and reported 17/24.

All five planted defects are caught. Three gates had to be repaired first —
the identity-injection was too weak to pin anything, condition 4 treated a
de-seeded target as hopeless when enumeration still reaches it, and the
paraphrase score counted `None == None` as agreement.

**Standing claim after X64C:** Sentinel represents ambiguity, asks
disagreement-maximising questions, detects conflict, abstains when nothing
adequate exists, and fails safely when its language is wrong. **It has not
been shown to understand language that was not authored around its
evaluation set.**

## X64D: senses induced from evidence, and language that cannot delete

X64C falsified the authored lexicon. X64D replaces it. The change is to the
*model*, not the vocabulary:

```
Pi          19 task-independent behavioural predicates
t = (w, r)  a token: a surface word in a syntactic role
Sigma(t)    a SET of senses, induced by clustering the predicate
            signatures of the development examples containing t
I           a reading: one sense chosen per token
viol(b)     sum over tokens of min over senses of |S minus sat(b)|
answer      only when the evidence leaves one candidate
```

**Three choices, each forced by a measured failure.**

*Sense sets, not one sense.* A single intersection is the most-specific
boundary of a version space and fails in both directions — with three
examples it kept an accident (`brackets` acquired "only after the hash");
with thirty-eight it collapsed, and `first`, `hash`, `last` all reduced to
the same generic core so the role stopped mattering. On validation,
clustering matched the single intersection's accuracy (105/105) with fewer
questions (137 vs 153) and four polysemous words instead of one.

*Language ranks, never eliminates.* The empty reading is always available,
so **D5 is a property of the definition rather than something to test for**.

*Evidence decides.* Committing when the language-preferred tier was a
singleton produced four confident errors, and no confirmation against a
fixed list repairs that — the target is always among the survivors, so a
disagreeing rival always exists. Waiting for the evidence drove it to zero,
and cost most of the query saving. That cost is itself the finding.

| arm | retained | answered | correct | **wrong** | queries |
|---|---|---|---|---|---|
| demonstrations only | 126 | 123 | 123 | 0 | 285 |
| X64C hard lexicon | **77** | 76 | 76 | 0 | 176 |
| role-blind, joint | 126 | 125 | 125 | 0 | 263 |
| induced, hard filter | **98** | 101 | 97 | **4** | 179 |
| **induced, joint** | **126** | 125 | 125 | 0 | 267 |
| induced, joint + semantic | 126 | **126** | **126** | 0 | 253 |
| oracle senses | 126 | 126 | 126 | 0 | 246 |

42 held-out tasks, 126 instruction forms, 24 scope×filter compositions
absent from development *and* validation. **9 of 10 gates.**

### D7 fails, and it fails for a reason worth more than a pass

**What cannot eliminate cannot contradict.** Twelve conflict statistics were
tried — set emptiness under the hard reading, the same with uninformative
predicates stripped at four thresholds, violation gaps at three thresholds,
a contrastive z-score against how other instructions rank the same
behaviour, and a discriminating semantic probe:

| statistic | recall | precision |
|---|---|---|
| hard reading empty | 0.31 | 0.62 |
| minus predicates >50% common | **0.83** | **0.51** |
| violation gap ≥ 2 | 0.24 | 0.56 |
| contrastive z ≥ 0 | 0.24 | 0.62 |

Precision sits at chance across the whole family. Induction by intersection
keeps only what examples **share**, so senses are generic, and a generic
constraint is satisfied by the wrong task as readily as the right one.
X64B-2 detected conflict at 8/12 precisely because its senses were
**authored and sharp** — and X64C measured what sharp authored senses cost
on unseen compositions. **D5 and D7 are in tension; this architecture buys
D5.**

### The arm that should temper the headline

Role-blind joint scores **125/126 in 263 queries** against induced joint's
**125/126 in 267**. On this test set the syntactic role earns *nothing
measurable*. Roles are what make polysemy representable; keeping
alternatives is what makes the system work. Those are different claims and
only the second is supported by a performance difference.

## X64E: conflict as posterior mass — 12/12

X64D concluded *"what cannot eliminate cannot contradict."* That is not a
theorem, and X64E refutes it. A model can keep support on every
interpretation and still measure how much probability mass sits outside what
the demonstrations allow. X64D could not, because its language layer
produced **sets** of predicates, and a set has no mass.

```
z          a typed logical form (op, filter, scope) from the task grammar
p_theta    a normalized distribution over all 168 forms given instruction u
C(D)       the forms whose execution fits the demonstrations
conflict   1 - sum over C(D) of p_theta(z | u)
answer     only when the EVIDENCE leaves one behaviour
```

The parser is a log-linear model over `(word, role, slot, value)` indicators
with exact inference — 168 forms, so nothing is approximated — trained by
weak supervision on the behaviourally consistent set:
`L = Σ log Σ_{z ∈ C(D)} p(z|u)`.

### What each arm spends to be right

On the 66 conditions **every** arm covers, all reach 100% correct:

| arm | correct | queries |
|---|---|---|
| demonstrations only | 66 | 150 |
| X64D predicate senses | 66 | 148 |
| uniform logical forms | 66 | 34 |
| authored multi-sense parser | 66 | 8 |
| role-blind induced | 66 | 6 |
| **MAIN induced parser** | **66** | **2** |
| gold logical forms | 66 | 0 |

### Conflict

**AUROC 0.996, AUPRC 0.997, 95% bootstrap CI (0.986, 1.000)**, recall 0.98
and precision 1.00 at a threshold fixed on validation. Calibration is clean:
43 matched pairs score below 0.2, 42 mismatched score above 0.8, one crosses.
X64D reported twelve set-based statistics all at chance. The difference is
not a better statistic — it is that a distribution has mass and a set does
not.

### What learning actually bought — and it is not accuracy

The **authored** parser (weights read straight off the surface realiser, no
learning) scores **1.00 exact-form** against the induced parser's **0.84**;
both reach 1.00 denotation. Learning loses on parsing accuracy and wins
downstream: **2 queries vs 8**, conflict **AUROC 0.996 vs 0.988**. The
induced weights are *calibrated* because they were fitted to a likelihood,
and calibration is exactly what the commit threshold and the conflict mass
consume. An authored map can be right without being confident in proportion.

Also: **28 of 116 behaviours have more than one logical form** (up to 21), so
no behavioural observation can separate them. Denotation accuracy is the
identifiable quantity; exact-form accuracy is reported beside it and should
not be read as the parser being wrong.

### Three controls that were broken and had to be fixed

The authored baseline was first built over *every* logical form — handed the
test split's vocabulary, it scored 1.00 on out-of-vocabulary forms. Rebuilt
over development forms but all three variants, it still had them. Restricted
to what the induced parser sees, it scores 0.00 on variant 2 like everything
else. And E2 first compared 80/86 against 66/66, which compares populations
rather than arms; it now runs on the intersection.

All ten planted defects caught. Freeze digest covers grammar, slots,
role–slot alignment, splits, hyperparameters, confirmation inputs, universe
and held-out set; a test verifies that mutating any of it changes the digest.

### The F−1 audit, which corrects two of the claims above

**The commitment rule was misdescribed.** X64E's text says the system answers
"only when the evidence leaves one behaviour". It does not. The arm whose
numbers were reported also commits when the behaviour posterior exceeds 0.99,
so **language can authorise an answer while behaviourally distinct rivals
remain**. Separating the two sources:

| policy | answered | correct | wrong | queries |
|---|---|---|---|---|
| A demonstrations only | 80 | 80 | 0 | 196 |
| B language ranks queries, **evidence** commits | 80 | 80 | 0 | 150 |
| C language ranks **and** commits at 0.99 (reported) | 80 | 80 | 0 | **2** |

Ranking the questions is worth 196 → 150. Authorising commitment is worth
150 → 2. The headline came from the second; the text described the first.

**The query advantage is not statistically stable.** Paired bootstrap,
resampled *by task meaning*:

| difference | mean | 95% CI | |
|---|---|---|---|
| queries saved vs authored structure | +0.139 | (+0.000, +0.326) | **includes 0** |
| queries saved vs role-blind | +0.119 | (−0.070, +0.302) | **includes 0** |
| conflict margin vs authored | +0.279 | (+0.181, +0.384) | excludes 0 |

So **learning buys calibration — that survives a paired interval — and the
query advantage does not.** E2's pass on query count was a point estimate.

**On the full population, main answers fewer.** 114/129 correct in 12
queries against demonstrations-only's 120/129 in 294. Main trades coverage
for efficiency; it does not dominate. The 66-condition intersection hid that.

**Unknown words: 43 forms, 0 wrong — but 1 interpreted, 29 resolved by
clarification, 9 reported unsupported, and 4 silently ignored.** E8 passes
because nothing is answered wrongly, *not* because unseen vocabulary was
understood, and the 4 silent cases are a partial failure of its spirit.

**What the audit strengthens:** exact-form accuracy is **1.00** on the 29
behaviourally identifiable forms; the 0.84 overall is entirely explained by
forms no observation can separate. Median gold-form rank 1, mean posterior
mass on the gold behaviour 0.987.

**Retracted: "X64 is closed."** The realizer is nearly a serialization of the
logical form — which is why an authored parser reaches 1.00 exact-form — so
the linguistic problem is largely solved by the data generator. X64 remains
open pending X64F, which breaks that one-to-one correspondence.

## X64F: surface language a word-to-slot table cannot decode — 12/12

X64E's realizer was nearly a serialization of the logical form, which is why
an authored word-to-slot table reached 1.00 exact-form without learning
anything. X64F replaces it. The decisive device: **the same nouns serve as
either the filter or the scope delimiter, and only order says which.**

```
remove the brackets before the hash  ->  remove(brackets @ before hash)
remove the hash before the brackets  ->  remove(hashes  @ before brackets)
```

Identical multisets, different meanings. **50 such collisions** cover 46 of
230 live forms. No surface *string* is ambiguous, so word order resolves
them and the denotation ceiling is 1.00.

Three independently seeded frozen splits:

| parser | 101 den/coll | 202 den/coll | 303 den/coll |
|---|---|---|---|
| **contextual** | 0.67 / **0.50** | 0.67 / **0.89** | 0.71 / **0.29** |
| bag-of-words | 0.67 / 0.11 | 0.68 / 0.44 | 0.76 / 0.07 |
| authored structure | 0.11 | 0.18 | 0.02 |
| shuffled | 0.00 | 0.00 | 0.01 |
| gold | 1.00 | 1.00 | 1.00 |

**Pooled and paired by task meaning:** on collisions, contextual 29/50 =
0.58 against bag-of-words 11/50 = 0.22, difference **+0.359, 95% CI
(+0.220, +0.500) — excludes zero**. Across *all* cases the difference is
**−0.026, CI (−0.078, +0.028) — includes zero**.

**Context buys exactly the construction it should and nothing else.**
Reporting only the overall number would have hidden both halves of that.

| arm | answered | correct | wrong | queries |
|---|---|---|---|---|
| demonstrations only | 360 | 360 | 0 | 842 |
| bag-of-words | 360 | 360 | 0 | 727 |
| **contextual (main)** | 360 | 360 | 0 | **721** |
| shuffled language | 356 | 356 | 0 | 1087 |
| main, no confirmation | 382 | 374 | **8** | 721 |
| main, target removed | 6 | 6 | 0 | 0 |

Conflict AUROC **0.943**, CI (0.918, 0.966). F8 paired saving per task
meaning **+0.625**, CI (+0.440, +0.834).

**The authored control collapses to 0.02–0.18**, from 1.00 on X64E's
realizer. That is the whole point: the realizer, not the parser, had been
doing the work.

### What is still weak

- **The operational gain over bag-of-words is negligible** — 721 queries vs
  727. The saving that survives an interval is against *demonstrations
  only*, not against BOW.
- **180 unknown-word cases, 167 correct — and the evidence supplied every
  one.** The parser does not understand those words; it declines to guess.
  Safe, not comprehension.
- With the target removed the system answers **6 of 1080** conditions. Zero
  errors, very little coverage.
- F4's margin is thin: 0.74 vs 0.71.

### Three mid-run corrections, each a measured rejection

**AdaGrad** was tried on the hypothesis that sparse contextual features
needed per-feature rates; it made both arms much worse (0.14 and 0.28 vs
0.51 and 0.59) and stays off. **The gradient was summed**, so the step
scaled with the dataset — 504 examples gave 0.58 on validation and 637 gave
0.04; that was divergence, not overfitting. **The first collision family was
too small to measure**: 22 bags over 188 forms gave 9 training and 9 test
instances, contextual tied BOW, and the phenomenon F1 exists to test was 5%
of the data. Adding `letters` as a third role-swappable noun took it to 50
bags over 46 forms.
