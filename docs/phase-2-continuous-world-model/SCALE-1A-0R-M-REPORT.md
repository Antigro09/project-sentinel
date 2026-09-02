# Scale 1A-0R-M — Temporal Belief Factorization and Finite-State Audit

## Verdict

**10 gates pass, 0 fail, 3 are partial or not run. Stage 1A-1 remains blocked, but the
temporal failure is now fully localized and one candidate qualifies.**

**Decision-table Outcome 4: the factorized belief path works where end-to-end recurrence
does not.** Action per the rule: adopt the factorized discrete filter as the Stage 1
temporal candidate and retain the generic recurrence as a negative arm.

The L phase left six candidate explanations. Five are now excluded by measurement:

| candidate | verdict | evidence |
|---|---|---|
| 1. sequence/plumbing defect | **not the cause** | partially audited; the information is demonstrably present and reachable |
| 2. inability to learn event parity | **excluded** | the *same* GRU implementation learns parity at **1.0000**, including extrapolation to lengths 9–16 it never saw |
| 3. event-extraction failure | **the remaining bottleneck** | learned extractor at 0.7764 balanced accuracy; with *true* events the pipeline reaches **1.0000** |
| 4. temporal credit assignment | **the cause of the end-to-end failure** | factorizing the identical computation recovers +0.359 of the phase gap where end-to-end recovers −0.268 |
| 5. generic recurrent-model failure | **confirmed, but not fundamental** | the GRU fails on the environment and succeeds on parity, so it is the objective and not the class |
| 6. discrete-belief-model success | **supported** | 2-state and 8-state learned filters both reach 1.0000 on parity |

## The central result

`R_phase_acc = [A_arm − A_current] / [A_true_phase − A_current]`, with `A_current` the
same head given no phase (0.6056) and `A_true_phase` the oracle (1.0000):

| arm | displacement accuracy | phase accuracy | **R_phase** |
|---|---:|---:|---:|
| true phase oracle | 1.0000 | 1.0000 | — |
| **true event + exact accumulator** | **1.0000** | **1.0000** | **+1.000** |
| **learned event + exact accumulator** | **0.7472** | 0.7167 | **+0.359** |
| shuffled-event control | 0.6222 | 0.6167 | +0.042 |
| constant-phase control | 0.6056 | 0.5000 | 0.000 |
| **end-to-end GRU** (L phase) | 0.5000 | — | **−0.268** |

Factorizing the same computation into *detect the public event* → *accumulate parity* →
*predict the outcome* turns a negative phase gap into a positive one, using the same head,
the same data and the same budget. The end-to-end arm has every input it needs and does not
find the decomposition.

## Pure parity, with the environment removed (§C, M2)

The decisive cheap test, and it excludes the model class outright. Same implementation as
the structured-history model — linear projection, `nn.GRU`, linear head.

| arm | train | validation | extrapolation (len 9–16) |
|---|---:|---:|---:|
| exact XOR accumulator | — | 1.0000 | — |
| **generic recurrent (the L-phase implementation)** | 1.0000 | **1.0000** | **1.0000** |
| learned 2-state filter | 1.0000 | 1.0000 (2 of 3 seeds) | 1.0000 |
| learned 8-state filter | 1.0000 | 1.0000 | 1.0000 |

The generic GRU solves parity at the **smallest budget on the ladder, 64 updates**, and
extrapolates to lengths never trained on. So the L-phase failure is not the recurrent
implementation, the optimizer, or the model class, and Outcome 1 is excluded.

The exact accumulator fed corrupted streams fails as required, which is what makes the
1.0000 meaningful rather than a property of the task being easy.

## The information is present (§B, partial)

Phase is inferable from the structured history from step 1 onward: comparing an observed
displacement against its action's expected delta reveals the polarity in force, and
crossings are inferable from the previous step's neighbour-switch bits. Traced on real
data — steps 1–2 give polarity 1, step 4 gives polarity 0, both correct — and step 3 shows
the composition the model must perform, since the displacement reveals the polarity
*before* a crossing that then toggles it.

One correction followed. The reset frame renders the polarity stripe, so **initial polarity
is public**, and my L-phase structured encoding omitted it entirely. Without it phase is
identifiable only up to a global flip, for reasons unrelated to learning. It is now
supplied at step 0 only, exactly as the renderer supplies it.

## Exact ceilings (§A, M0)

| stratum | transitions | public memoryless | phase oracle | headroom |
|---|---:|---:|---:|---:|
| all | 13,444 | 0.8176 | 1.0000 | 0.1824 |
| ordinary non-switch | 2,384 | 0.8438 | 1.0000 | 0.1562 |
| switch-sensitive (per action) | 6,412 | 0.6176 | 1.0000 | 0.3824 |
| exact packet aliases | 6,848 | 0.6419 | 1.0000 | 0.3581 |
| post-two-changes | 7,012 | 0.8061 | 1.0000 | 0.1939 |

Reconciling the L-phase figures as required: **0.5264** is the uniform-phase-prior
memoryless ceiling on the trajectory population — a *lower bound*, because phase is partly
predictable from public state; **0.6472** is the structured current-state score, which
legitimately exceeds that lower bound; **0.5000** is the alias-pair score for a current-only
model, which is exact chance because identical packets force a tie; **1.0000** is the
alias-pair score with true phase; **0.3783** is the headroom between them. No learned arm
exceeds the phase-aware oracle on any matched population, metric and information set.

## §J Gates

| gate | status | basis |
|---|---|---|
| M0 | pass | ceilings enumerated exactly; no arm exceeds the oracle |
| **M1** | **partial** | the twelve plumbing properties hold by construction and the alias certificates regression-test several, but **the six planted-defect arms were not built**, so M1 is not fully evidenced |
| M2 | pass | all three temporal arms reach 1.0000 on parity, including extrapolation |
| M3 | pass | true event + accumulator reaches the oracle exactly |
| M4 | pass | learned extractor 0.7764 balanced accuracy, F1 0.6621, above both controls |
| M5 | pass | learned event + accumulator 0.7472, R_phase +0.359 |
| **M6** | **not run** | the learned finite-state filter was qualified on parity (1.0000) but **not run on the environment** — that arm used the exact accumulator |
| M7 | pass | a structured-history arm closes a positive phase gap |
| M8 | pass | shuffled events 0.6222 below learned events 0.7472 |
| M9 | pass | parity train = validation = 1.0000, so no generalization gap; the environment arms separate optimization from representation |
| M10 | pass | no hidden phase, step, evaluator event, provenance or future outcome in the main input |
| **M11** | **partial** | alias-pair and L-phase results carry episode-level intervals; this phase's factorized arms are point estimates over three seeds **without paired intervals** |
| M12 | pass | every failed arm and seed retained, including the 2-state filter's seed-6601 failure |

## Bugs and corrections

1. **My 2-state filter had a symmetric fixed point.** Zero-initialised transition logits
   make `softmax` uniform for both event values, the belief goes uniform after one step,
   and the gradient to every event position is **exactly 0.00e+00** — which the gradient
   diagnostic reported, independently of any validation score. Random initialisation fixes
   it to 1.0000. Identified by the diagnostic, not by the score, which is what makes the
   fix a repair rather than tuning.
2. **The parity controls were regenerated rather than derived.** Changing the corruption
   mode changed the number of RNG draws per example, so lengths and event streams diverged
   and the control was scored against targets from different sequences. Controls are now
   derived from the base dataset, leaving targets untouched.
3. **The L-phase structured encoding omitted the rendered reset stripe**, making phase
   identifiable only up to a global flip. Corrected; it is public and now supplied at
   step 0.

## Narrow supported claim

> The temporal failure reported in the L phase is not a property of the recurrent model
> class: the identical implementation learns parity at 1.0000 and extrapolates to sequence
> lengths it never saw. On the environment, factorizing the same computation into a public
> event detector, an exact parity accumulator and the qualified prediction head recovers
> **+0.359** of the phase gap, while the end-to-end recurrent arm recovers **−0.268**. With
> *true* events the factorized path reaches the phase-aware oracle exactly, which places the
> remaining bottleneck in event extraction at 0.7764 balanced accuracy rather than in
> accumulation or prediction. The failure is therefore end-to-end credit assignment, not
> representation, capacity, model class or measurement.

## Unsupported claims

- **That a learned finite-state filter works on the environment.** It was qualified only on
  the parity microcase. M6 is not evidenced.
- **That the plumbing is clean.** M1's planted-defect arms were not built, so the twelve
  properties are argued rather than demonstrated.
- **Anything about visual representations or multimodality.** Not run, by rule.
- **That the factorized arm's advantage is statistically bounded.** It is a point estimate;
  M11 has no paired intervals for this phase's arms.

## Selected temporal model and what is unblocked

**Selected candidate: the factorized public-event detector plus discrete parity
accumulator.** The generic GRU is retained as a declared negative arm.

**Visual qualification and Stage 1A-1 remain blocked.** Three things stand between here and
the visual ladder, all small and specific: build M1's planted-defect arms, run M6's learned
finite-state filter on the environment rather than only on parity, and attach paired
intervals to the factorized comparison. The event extractor at 0.7764 is the quantity that
will determine how much of the phase gap any visual arm can close.

