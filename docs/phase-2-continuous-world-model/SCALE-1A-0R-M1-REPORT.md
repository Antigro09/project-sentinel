# Scale 1A-0R-M1 — Environmental Finite-State Closure and Attribution Audit

## Verdict

**The visual ladder is NOT unblocked.** The M-phase "Outcome 4" claim survives in weakened
and corrected form, and one of its supports is withdrawn.

**The M-phase comparison was confounded and the specification was right.** The factorized
arm received event supervision the end-to-end arm did not, so calling the difference
"factorization" was unsupported. A matched 2×2 now separates the two, with five seeds and
paired intervals by layout.

### The supervision-matched 2×2 (1024 updates, 5 seeds)

| | outcome loss only | + auxiliary event loss |
|---|---:|---:|
| **memoryless baseline** (no temporal state) | **0.6056** | — |
| end-to-end GRU | 0.5411 | 0.5611 |
| factorized learned filter | 0.5933 | **0.6406** |

| paired comparison (by layout) | Δ | 95% interval | |
|---|---:|---|---|
| factorized+event **vs** end-to-end+event | +0.0794 | [+0.0283, +0.1350] | ✓ |
| factorized+event **vs** factorized outcome-only | +0.0472 | [+0.0100, +0.0861] | ✓ |
| end-to-end+event **vs** end-to-end outcome-only | +0.0200 | [−0.0117, +0.0506] | — |
| factorized+event **vs** permuted event labels | +0.0328 | [+0.0094, +0.0561] | ✓ |
| factorized+event **vs** memoryless baseline | +0.0350 | [+0.0028, +0.0672] | ✓ |
| end-to-end+event **vs** memoryless baseline | −0.0444 | [−0.0983, +0.0056] | — |
| **true event + learned filter vs memoryless** | **+0.0872** | [+0.0622, +0.1117] | ✓ |
| **true event + learned filter vs learned event** | **+0.0522** | [+0.0300, +0.0761] | ✓ |

### What that attributes

- **Factorization contributes.** With supervision matched on both sides, factorized beats
  end-to-end by +0.0794, interval excluding zero.
- **Supervision also contributes**, +0.0472 within the factorized architecture.
- **Supervision alone does not rescue the GRU** — the interval includes zero, and the
  end-to-end arm does not beat the memoryless baseline even with the labels.
- **The labels are informative, not a regularizer** — permuting them costs +0.0328.
- **Neither alone suffices.** Factorized *without* event supervision reaches 0.5933, which
  is **below** the memoryless baseline at 0.6056.

The last point decides how this must be described. Because the gain requires both the
architecture and the event labels, and the labels are evaluator-derived, this is a
**supervised event-factorization result and not autonomous hidden-state discovery.** The
specification anticipated exactly this case and its wording is the one to use.

### Where the remaining loss sits

| arm | accuracy | vs memoryless |
|---|---:|---|
| memoryless baseline | 0.6056 | — |
| end-to-end GRU + event loss | 0.5611 | −0.0444 |
| factorized + learned event + **learned** filter | 0.6406 | +0.0350 |
| factorized + **true** event + learned filter | **0.6928** (sd 0.1470) | +0.0872 |
| factorized + learned event + **exact** accumulator (M phase) | 0.7472 | — |
| phase-aware oracle | 1.0000 | — |

Two bottlenecks, both measured. Event detection costs +0.0522 (true events beat learned
ones with an interval excluding zero). And the **learned transition is worse than the exact
one**: swapping the learned two-state filter for the exact XOR accumulator is the
difference between 0.6406 and 0.7472.

**N5 is only partially met, and the seed spread is the reason.** True event + learned filter
averages 0.6928 with **sd 0.1470**: one seed of five reaches **0.986**, the other four sit
near 0.62. The filter *can* learn the transition and usually does not. A mean with that
spread is not "approaches the ceiling", and reporting only the mean would hide a bimodal
result.

## §A Provenance and exact prior status

| item | value |
|---|---|
| M-phase commit | `eb6567e` |
| branch | `phase-2-continuous-world-model` |
| tracked tree | clean apart from the pre-existing `.claude/worktrees/x35-novelty-trigger` gitlink |
| untracked paths | none |
| exact-reference / Phase-2 tests | 522 / 452 passed + 4 skipped |
| seeds | development 6600–6604; validation layouts 81000+; alias layouts 90000+ |
| visual or final Scale-1 seed opened | **no** |

**The three M-phase partial/not-run gates are not relabelled.** They were M1 (planted
defects), M6 (learned filter on the environment) and M11 (paired intervals). Of these,
**M6 and M11 are now closed** by this phase; **M1 remains open.**

## §L Gates

| gate | status | basis |
|---|---|---|
| N0 | pass | provenance above |
| **N1** | **partial** | the dataflow is ordered `event(X_{t−1},A_{t−1},X_t) → belief → outcome`, and no future observation, target, hidden phase, step or provenance value enters the main input. The **five planted leak arms of §B were not built**, so this is argued from construction rather than demonstrated |
| **N2** | **not run** | the ten M1 planted sequence defects were **not built**. This is the same gap I flagged at the end of the M phase and it is still open |
| **N3** | **partial** | parity was verified by sampling at lengths 1–8 with extrapolation to 9–16 (all arms 1.0000), **not exhaustively** over every sequence |
| N4 | pass | belief is initialised from the **public rendered reset stripe**, not an evaluator bit. The masked/false-stripe variants of §F were not run |
| **N5** | **partial** | true event + learned filter: mean 0.6928, **sd 0.1470**, 1 of 5 seeds at 0.986. +0.0872 over memoryless with interval excluding zero, but not a robust approach to the ceiling |
| N6 | pass | learned event + exact accumulator retains a positive gap; paired interval excludes zero |
| N7 | pass, weakly | learned event + learned filter +0.0350 [+0.0028, +0.0672] — barely excludes zero |
| **N8** | **partial** | permuted event labels reduce the gain (+0.0328, excludes zero). **Shifted, dropped and false-event controls were not run** |
| N9 | pass | the 2×2 above separates factorization, supervision and their interaction |
| **N10** | **not run** | event extraction was not reported by held-out dynamics, per action, or by switch count |
| **N11** | **not run** | results were not conditioned on the number of accumulated phase changes |
| N12 | pass | no hidden phase, evaluator event, step, future outcome or provenance in the main input; event labels are declared auxiliary supervision and never an input |
| N13 | pass | five seeds, paired layout-level intervals on every headline comparison, all seeds retained including the bimodal N5 spread |

**6 pass, 1 pass-weakly, 4 partial, 3 not run.**

## Bugs and corrections

1. **The M-phase attribution was confounded** and is corrected here. One arm received an
   event label the other did not, so "factorization" was not separable from "supervision".
   With matched supervision the factorization effect survives at +0.0794, but the honest
   description changes: the gain needs both, so this is supervised event-factorization.
2. **R_phase was computed against a constant-phase arm** in the M phase. The specification
   forbids substituting that for a trained memoryless model, and it matters: the trained
   memoryless baseline is 0.6056 while factorized-without-supervision is 0.5933, so an arm
   that looked like a gain against a constant is a *loss* against a trained baseline.
3. **The learned filter's environmental result is bimodal** and a mean alone misreports it.

## Narrow supported claim

> With supervision matched on both sides, five seeds and paired layout-level intervals, a
> factorized architecture — public event detector, learned two-state belief filter
> initialised from the rendered reset stripe, shared outcome head — beats an end-to-end GRU
> by +0.0794 [+0.0283, +0.1350] and beats a trained memoryless baseline by +0.0350
> [+0.0028, +0.0672]. The end-to-end GRU does not beat the memoryless baseline even when
> given the identical auxiliary event labels. The gain requires **both** the factorization
> and the event supervision: without the labels the same architecture scores 0.5933, below
> the memoryless baseline. Because the labels are evaluator-derived, this is a **supervised
> event-factorization result, not autonomous hidden-state discovery**. Substituting true
> events for learned ones adds +0.0522, and substituting the exact accumulator for the
> learned filter adds more still, so event detection and transition learning are both
> live bottlenecks.

## Unsupported claims

- **That the temporal dataflow is leak-free** — argued from construction; the five planted
  leak arms were not built (N1).
- **That the sequence plumbing is correct** — the ten planted defects were not built (N2).
  This was the gap I flagged at the end of the M phase, and I did not close it.
- **That the learned filter approaches the environmental ceiling** — it does so in one seed
  of five (N5).
- **That shifted, dropped or false-event controls remove the gain** — not run (N8).
- **Anything about event generalization by horizon or by switch count** — not run (N10, N11).
- **Anything about visual representations** — not run, by rule.

## Selected temporal candidate

**Provisionally the factorized event-detector plus learned finite-state filter**, with the
generic GRU retained as a declared negative arm. The decision rule's condition is met — N7
passes and the GRU fails under matched supervision — but the selection is weak on two
counts that should be closed before it is relied on: the learned transition is bimodal
across seeds, and the exact accumulator still outperforms it.

**The visual ladder remains blocked.** N2 is the gate I would close first: every phase of
this work has had its headline moved by something an unbuilt control would have caught, and
the planted-defect suite is now the oldest outstanding one.

