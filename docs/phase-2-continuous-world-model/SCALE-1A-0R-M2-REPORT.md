# Scale 1A-0R-M2 — Temporal Attribution Closure

## Verdict

**The visual ladder is NOT unblocked, and the M1 provisional selection is withdrawn.**

**Decision-rule outcome: T4 fails.** The learned temporal transition is unstable across
seeds, and the rule is explicit — *do not select the filter*.

Two long-outstanding gates are now closed, and one of them changed the picture.

**T2 passes.** The planted sequence-defect suite — flagged as first priority at the end of
the M phase, not built, then flagged again at the end of M1 — exists. Ten defects, ten
guards, every corrupted arm derived from one frozen base batch. All ten caught, no vacuous
guards, none firing on honest data. **Building it immediately caught two broken guards of
my own**, which is the argument for having built it two phases ago.

**T3 passes exhaustively.** Every binary sequence at lengths 1–16 with both initial phases —
262,140 pairs. The generic GRU and the 8-state filter are exactly correct on all 1,020
trained-length pairs *and* all 261,120 extrapolation pairs, stepwise as well as final. The
sampled claim from the M phase is now exact.

**T4 fails, and the same exhaustive sweep is why.** The two-state filter collapses to chance
on 1 of 3 seeds (0.505882 against 1.000000 for the other two) on the *pure parity task*, and
the environmental result was already bimodal at 1 of 5 seeds. A temporal mechanism that
fails outright on a third of initializations, on a task the generic GRU solves perfectly on
every seed, has not earned selection.

## Gate tables

### Prior gates, restated exactly

| M-phase | status | | N-phase (M1) | status |
|---|---|---|---|---|
| M0 | PASS | | N0 | PASS |
| M1 | **PARTIAL** → now **PASS** (T2) | | N1 | PARTIAL |
| M2 | PASS | | N2 | **NOT_RUN** → now **PASS** (T2) |
| M3 | PASS | | N3 | PARTIAL → now **PASS** (T3) |
| M4 | PASS | | N4 | PASS |
| M5 | PASS | | N5 | PARTIAL |
| M6 | **NOT_RUN** → closed in M1 | | N6 | PASS |
| M7 | PASS | | N7 | PASS (weak) |
| M8 | PASS | | N8 | PARTIAL |
| M9 | PASS | | N9 | PASS |
| M10 | PASS | | N10 | NOT_RUN |
| M11 | **PARTIAL** → closed in M1 | | N11 | NOT_RUN |
| M12 | PASS | | N12 | PASS |
| | | | N13 | PASS |

### T gates

| gate | status | basis |
|---|---|---|
| T0 | PASS | provenance below |
| **T1** | **PARTIAL** | one of the five §B leak arms is built (target-in-features, caught). The other four — future observation, event shifted forward, event shifted backward, action misaligned — are **not built** |
| **T2** | **PASS** | 10 of 10 defects caught by their intended guards; 0 vacuous; 0 firing on honest data; 4 regression tests pin it |
| **T3** | **PASS** | exhaustive over 262,140 pairs; GRU and 8-state filter exact on trained *and* extrapolation lengths |
| **T4** | **FAIL** | the two-state filter collapses to chance on 1 of 3 parity seeds and is bimodal on 1 of 5 environmental seeds. Not reliable across seeds |
| T5 | PARTIAL | learned event + exact accumulator beats shuffled events, but the **shifted**-event control was not run and this comparison lacks its own paired interval |
| T6 | PARTIAL | learned event + learned filter beats the trained memoryless baseline, +0.0350 [+0.0028, +0.0672] — but on general transitions, **not on the exact alias-pair population** the gate names |
| T7 | NOT_RUN | results were not conditioned on the number of accumulated phase changes |
| T8 | PASS | factorization retains +0.0794 [+0.0283, +0.1350] under matched supervision |
| T9 | PASS | event supervision contributes +0.0472; permuted labels cost +0.0328, so the labels are informative |
| T10 | PASS | no hidden phase, evaluator event, step, future outcome or provenance in the main input |
| T11 | PASS | every seed retained, including both collapse cases |
| T12 | PARTIAL | the 2×2 is fully matched; the primary comparison did **not** use the exact-alias population |

**4 PASS, 1 FAIL, 5 PARTIAL, 1 NOT_RUN, 2 PASS from prior phases.**

## §C The planted-defect matrix (T2)

Ten defects, ten guards, one frozen base batch of 49 episodes × 9 steps × 26 features.

| planted defect | intended guard | caught |
|---|---|---|
| recurrent state reset every step | state carries within episode | ✓ |
| state detached every step | gradient reaches first step | ✓ |
| hidden state carried across episodes | no cross-episode state | ✓ |
| reset frame omitted | reset present exactly once | ✓ |
| reset frame duplicated | reset present exactly once | ✓ |
| final prediction index shifted | blocked action implies no movement | ✓ |
| padding treated as valid history | mask matches lengths | ✓ |
| histories permuted inside batches | padding is empty | ✓ |
| current step removed | current step present | ✓ |
| target copied into features | no feature equals target | ✓ |

**Vacuous guards: none. Guards firing on the honest pipeline: none.**

### The two guards it caught of mine

1. **`Batch.copy` dropped the planted flags.** A behavioural guard perturbs a batch and
   re-runs the model, but the copy lost the defect flag, so the perturbed run took the
   *honest* code path and a difference always appeared. `state_carries_within_episode`
   passed all eleven arms — a guard with no detection power, and exactly the failure mode
   this suite exists to find.
2. **The blocked-action guard used the wrong columns.** It read 12:16 for the queried
   action's one-hot; that range is goal-direction and previous-action. The guard fired on
   honest data, which the matrix reported as a distinct failure rather than folding into a
   pass count.

Neither would have been visible without the honest-arm column and the vacuity check.

## §D Exhaustive parity (T3)

262,140 (sequence, initial phase) pairs: every binary sequence of lengths 1–16, both
phases. Trained on lengths 1–8 exhaustively (1,020 pairs); lengths 9–16 never trained.

| arm | seed | trained lengths | extrapolation | stepwise trajectory |
|---|---:|---:|---:|---:|
| generic recurrent | 6600/6601/6602 | **1.000000** | **1.000000** | **1.000000** |
| 8-state filter | 6600/6601/6602 | **1.000000** | **1.000000** | **1.000000** |
| 2-state filter | 6600, 6602 | 1.000000 | 1.000000 | 1.000000 |
| **2-state filter** | **6601** | **0.505882** | 0.500000 | 0.533264 |

Exact, not sampled. The generic GRU's competence on this task is now beyond dispute, which
sharpens the environmental result rather than softening it: the same implementation that is
perfect on 261,120 unseen-length parity sequences recovers none of the environmental phase
gap.

## Bugs and corrections

1. **Two of my own guards were broken** and the matrix caught both — see §C.
2. **The M1 provisional selection is withdrawn.** It was made on N7 passing weakly while
   T4's stability question was open. With the exhaustive sweep the two-state filter's
   collapse rate is measured rather than suspected, and the decision rule's T4 branch
   applies.

## Narrow supported claim

> The temporal implementation is correct: ten planted sequence defects are each caught by a
> guard that passes the honest pipeline, with no vacuous guards. The generic recurrent model
> and an 8-state learned filter solve parity **exactly** on all 262,140 sequence/phase pairs
> at lengths 1–16, including 261,120 pairs at lengths never trained on. Under supervision
> matched on both sides, a factorized architecture beats an end-to-end GRU by +0.0794
> [+0.0283, +0.1350] and a trained memoryless baseline by +0.0350 [+0.0028, +0.0672], and
> the gain requires both the factorization and evaluator-derived event labels. The learned
> two-state transition is **not stable**: it collapses to chance on 1 of 3 parity seeds and
> is bimodal on 1 of 5 environmental seeds. No temporal candidate is selected.

## Unsupported claims

- **That the dataflow is leak-free** — four of the five §B leak arms are not built (T1).
- **That the factorized gain holds on the exact alias population** — the primary comparison
  used general transitions (T6, T12).
- **That the gain survives two or more phase changes** — not conditioned on phase-change
  count (T7).
- **That shifted-event controls remove the gain** — not run (T5).
- **Anything about event generalization by horizon, or about visual representations.**

## Selected temporal candidate

**None.** T4's branch is explicit: the learned temporal transition remains unstable, so the
filter is not selected. The factorized *mechanism* is the most promising result in this line
of work — it is the only arm that beats a trained memoryless baseline — but its transition
component fails on a third of initializations on a task the alternative solves perfectly.

The next work is stability, not scale: §I's preregistered initialization arms, chosen on
development and rerun with every seed retained. Model size must not be increased.

