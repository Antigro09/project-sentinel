# Scale 1A-0R-M2C — Environmental Filter Stability and T4 Closure

## Verdict

**U3 and U7 both pass. The temporal path is qualified as a SUPERVISED EVENT-FACTORIZED
TEMPORAL MODEL.** The visual ladder is **not** unblocked, because three closure gates
remain open.

The headline is that a preregistered initialization rule turns an unstable filter into an
exact one, and the fix is legitimately an initialization rule rather than a change of model
class — because the collapse was diagnosed as an optimization basin, not a representational
limit.

## §A Gate ledger correction

**The previous phase's T4 verdict was a ledger error and is withdrawn.** T4 is an
*environmental* gate and admits either a two-state or an eight-state filter. It was marked
FAIL from a **parity** seed collapse in the two-state arm, while the eight-state filter was
exact on every parity seed. Environmental T4 was **NOT_RUN**. Corrected status:

| item | status |
|---|---|
| T2 planted sequence defects | **PASS** |
| T3 exhaustive parity | **PASS** |
| default 2-state parity stability | **FAIL** (1 of 3 seeds collapse) |
| T4 environmental learned filter | **was NOT_RUN** → now **PASS** (this phase) |
| T1 dataflow leak arms | **PARTIAL** (1 of 5 built) |
| T5 shifted-event control | **NOT_RUN** |
| T6 alias-population comparison | **was PARTIAL** → now **PASS** (this phase) |
| T7 two-or-more phase changes on aliases | **NOT_RUN** |
| visual ladder | **BLOCKED** |

## §B Parity collapse: the mechanism

The solved seeds learn the XOR automaton exactly:

| seed | event 0 map | event 1 map | belief entropy | L1 between event maps |
|---|---|---|---:|---:|
| 6600 | `[[0.993, 0.007], [0.008, 0.992]]` stay | `[[0.003, 0.997], [0.997, 0.003]]` flip | 0.172 | 3.960 |
| 6602 | `[[0.991, 0.009], [0.009, 0.991]]` stay | `[[0.007, 0.993], [0.994, 0.006]]` flip | 0.218 | 3.938 |
| **6601** | `[[0.435, 0.565], [0.825, 0.175]]` | `[[0.744, 0.256], [0.370, 0.630]]` | **0.664** | **1.529** |

Maximum entropy for two states is 0.693, so the collapsed seed's belief never becomes
informative and neither event map is a clean stay or flip. **This is an optimization basin,
not a representational limit** — the same model class finds the exact solution on two seeds
in three. That is what makes an initialization rule the right fix rather than a bigger model.

The eight-state filter was exact on 3/3 parity seeds and remained a live candidate
throughout, which the previous phase's verdict obscured.

## §C–§D Environmental filter stability (U3)

Seven preregistered arms, **20 development seeds**, selection on the **10th percentile then
median — never the mean**, then **20 untouched validation seeds**. TRUE public events as
input, isolating transition learning from event extraction.

### Development (20 seeds)

| arm | mean | sd | median | min | p10 | collapsed | after 2+ changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact accumulator (ceiling) | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0/20 | 1.0000 |
| 2-state, default init | 0.7188 | 0.1597 | 0.6375 | 0.6000 | 0.6053 | 0/20 | 0.7178 |
| **2-state, symmetry-broken** | **0.9999** | **0.0006** | **1.0000** | **0.9972** | **1.0000** | **0/20** | **1.0000** |
| 2-state, reset-conditioned | 0.6622 | 0.1006 | 0.6306 | 0.6028 | 0.6136 | 0/20 | 0.6611 |
| 8-state categorical | 0.7625 | 0.1328 | 0.7236 | 0.5917 | 0.6083 | 0/20 | 0.7409 |
| generic GRU | 0.5572 | 0.0193 | 0.5500 | 0.5278 | 0.5386 | 0/20 | 0.5308 |
| trained memoryless | 0.6165 | 0.0171 | 0.6139 | 0.5889 | 0.5917 | 0/20 | 0.6106 |

### Validation (20 untouched seeds)

| arm | mean | sd | min | p10 | collapsed |
|---|---:|---:|---:|---:|---:|
| **2-state, symmetry-broken** (selected) | **1.0000** | **0.0000** | **1.0000** | **1.0000** | **0/20** |
| exact accumulator | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0/20 |
| generic GRU | 0.5614 | 0.0192 | 0.5278 | 0.5358 | 0/20 |
| trained memoryless | 0.6144 | 0.0240 | 0.5778 | 0.5853 | 0/20 |

**U3 passes.** The selected filter matches the exact-accumulator ceiling on every one of
twenty untouched seeds, with zero variance and no collapses.

The perturbation is a tiny seed-derived antisymmetric term added to the transition logits.
It encodes no phase semantics and no XOR structure; it only prevents the two event maps
from starting interchangeable, which is precisely the basin seed 6601 fell into.

**The comparison that matters most is the last two rows.** Given the *same true events*, the
generic GRU reaches 0.5614 — **below the trained memoryless baseline at 0.6144**. The
inductive bias is doing the work, not the information.

## §H Learned-event coupling on exact alias pairs (U7)

Primary population as specified: identical complete public packet, identical proposed
action, different legal histories, different hidden phase, different public next outcome.

| arm | alias pairwise accuracy | vs memoryless | 95% paired interval |
|---|---:|---:|---|
| true phase oracle | 1.0000 | +0.5000 | [+0.5000, +0.5000] |
| true event + accumulator | 1.0000 | +0.5000 | [+0.5000, +0.5000] |
| **learned event + accumulator** | **0.6503** | **+0.1503** | **[+0.0719, +0.2287]** |
| memoryless | 0.5000 | — | exactly chance, by construction |

**U7 passes.** The learned pipeline recovers **+0.1503** of the available 0.5000 headroom
with a paired interval excluding zero, on the population where a memoryless model is
provably at chance.

The memoryless arm landing on exactly 0.5000 is the construction validating itself:
identical packets force identical predictions and therefore an exact tie.

The remaining gap is entirely event extraction. The detector runs at **0.7764 balanced
accuracy**; with true events the same pipeline is perfect.

## §L Closure gates

| gate | status | basis |
|---|---|---|
| U0 | PASS | provenance; suite 522 exact-reference / 456 + 4 skipped Phase-2 |
| U1 | PASS | T2 and T3 remain pinned by regression tests |
| **U2** | **PARTIAL** | 1 of 5 §I leak arms built (target-in-features). Future-observation, event-shifted-forward, event-shifted-backward and action-misaligned are **not built** |
| **U3** | **PASS** | selected filter 1.0000 on 20/20 untouched seeds, 0 collapses, p10 = 1.0000 > memoryless 0.6144 |
| **U4** | **PASS** | on exact alias pairs, +0.1503 [+0.0719, +0.2287] |
| **U5** | **PARTIAL** | after-2-changes reported for the true-event study (1.0000 for the selected filter); **not** conditioned on phase-change count for the alias coupling |
| U6 | PARTIAL | permuted-label control run in M1; **shifted, dropped and constant** event controls not run on the coupling |
| **U7** | **PASS** | learned event + filter beats memoryless on exact alias pairs, interval excluding zero |
| **U8** | **NOT_RUN** | event corruption controls not run on this coupling |
| U9 | PASS | supervision is required — M1 showed factorized-without-labels at 0.5933, below the 0.6056 memoryless baseline |
| U10 | PASS | all 40 seeds retained, including the collapsed parity seed and both filter arms that failed |
| U11 | PARTIAL | populations, supervision and compute matched throughout; the coupling's event-corruption arms are missing |

**7 PASS, 4 PARTIAL/NOT_RUN.**

## Bugs and corrections

1. **The T4 ledger error**, described in §A and withdrawn.
2. **My collapse indicator was miscalibrated.** It compared raw belief entropy against a
   threshold tuned for two states, so all three eight-state seeds were labelled collapsed
   while scoring 1.000000. Now normalised by `log(states)`. Third self-inflicted measurement
   bug this line of work has surfaced, and like the others it was caught by a diagnostic
   that reports mechanism rather than a pass/fail number.
3. **`row_from_state` was imported from the wrong module**, a straightforward error caught
   at import.

## Narrow supported claim

> A two-state learned belief filter with a preregistered antisymmetric initialization,
> selected on twenty development seeds by lower-tail performance and validated on twenty
> untouched seeds, matches the exact-accumulator ceiling on the environmental transition
> task with **zero variance and no collapses** when given true public events. Given the same
> events a generic GRU reaches 0.5614, **below** a trained memoryless baseline at 0.6144, so
> the advantage is inductive bias rather than information. Coupled to a learned public-event
> detector at 0.7764 balanced accuracy, the pipeline beats the memoryless baseline on exact
> public-packet alias pairs by **+0.1503 [+0.0719, +0.2287]**, on a population where a
> memoryless model is provably at chance. The gain requires evaluator-derived event
> supervision, so this is a **supervised event-factorized temporal model and not autonomous
> hidden-state discovery**. The residual gap is event extraction: with true events the same
> pipeline is exact.

## Unsupported claims

- **That the dataflow is leak-free** — four of five §I arms unbuilt (U2).
- **That the alias-pair gain survives two or more phase changes** — not conditioned (U5).
- **That shifted, dropped or constant event corruptions remove the gain** — not run (U6, U8).
- **Anything about visual representations.**

## Selected temporal candidate

**The supervised event-factorized temporal model**: a learned public-event detector feeding
a two-state learned belief filter with the frozen antisymmetric initialization, belief
grounded in the rendered reset stripe. The generic GRU and outcome-only arms are retained as
declared negative controls, as the decision rule requires.

**The visual ladder stays blocked** on U2, U5 and U8. U2 is the one to close first: it is
the same class of gap as the T2 suite, and building that suite caught two broken guards of
my own within minutes.

