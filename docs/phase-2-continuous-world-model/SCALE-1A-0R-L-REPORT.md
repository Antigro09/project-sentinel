# Scale 1A-0R-L — Learned Dynamics Instrument and Hidden-Belief Qualification

## Verdict

**Stage 1A-1 is NOT unblocked. 8 gates pass, 1 fails, 4 were not run.**

**Decision-table Outcome 3: true phase works, structured recurrence fails.**
Diagnosis: **temporal architecture or history-encoding failure.** Action: repair the
recurrent belief mechanism before the matrix.

The difference from K6 is that this time the instrument is qualified before the negative
is read, and the failure is therefore attributable:

| what | result | meaning |
|---|---:|---|
| public memoryless Bayes on alias classes | 0.6217 | exact reference, enumerated |
| phase-aware oracle | 1.0000 | exact, and a point mass by construction |
| **phase headroom** | **0.3783** | the target does measure hidden state |
| structured current-state model | 0.6472 | clears its reference — the predictor works |
| structured + **true phase** | **1.0000** | at every budget — the head and phase fusion work |
| alias ranking, current packet | 0.5000 | exactly chance, as identical packets force a tie |
| alias ranking, **true phase** | **1.0000** | the metric has 0.5 of headroom and the oracle takes all of it |
| **structured + correct history** | **0.5000** | **worse than the memoryless model** |

Every positive control passes. The same head, the same optimiser, the same data reach
1.000 the moment phase is supplied as one scalar — so nothing is broken about the
predictor, the objective or the fusion. What fails is constructing phase from history:
`R_phase` is **negative**, because the recurrent model does not even match the
current-state model it should strictly dominate.

That is a much narrower and more useful claim than K6's withdrawn one. K6 could not
distinguish "did not construct the state" from "the instrument cannot show success". Here
the instrument demonstrably can show success, and the recurrent path still does not.

## §A Preserved record from Scale 1A-0R-K

| item | value |
|---|---|
| K-phase commit | `1b0413cdfbcd2c2c8640d0701941c452e05aaafd` |
| branch | `phase-2-continuous-world-model` |
| tracked tree | clean apart from the pre-existing `.claude/worktrees/x35-novelty-trigger` gitlink |
| untracked paths | none |
| exact-reference tests | 522 |
| Phase-2 tests | 452 passed, 4 skipped |
| complete repository | 974 collected, 969 passed + 4 skipped at 989.05 s before the L additions |
| K-phase UNKNOWN gates | **K6** (main arm) and **K8** (intervention non-inferiority) |
| Stage 1A-1 training | not begun |
| final Scale-1 seed | not opened |

### Corrections carried forward, all still true

1. **K6 is UNKNOWN, not FAIL.** Its instrument had no positive control.
2. **The one-observation packet guard is now non-vacuous** — it exercises a builder, and
   both no-op and denylist mutations fail the suite.
3. **The old recurrence gap was a 60-update convergence artifact.**
4. **The K-phase learned model never reached the memoryless dynamics ceiling** (0.1492
   against 0.0354).
5. **The derived event/phase pipeline reaches 1.000 but is not a learned world-model
   result** — crossing comes from a parameterless relation over decoded masks.
6. **No event label or hidden phase enters the main learned input.**
7. **No Stage 1A-1 training has begun.**

## §B Exact ceilings

Computed by exhaustive enumeration over the reachable set, not estimated. Given the
packet, the frame fixes the level and the agent's cell; given the cell, the action and the
phase, the successor is determined — so **p(Y | X, A, H) is a point mass and the
phase-aware oracle is exactly 1.000**. The public reference is the mixture over whichever
phases are reachable in that packet class. Weighting is uniform over reachable states
within a class, which is a declared choice.

| stratum | transitions | public memoryless | phase oracle | headroom | public NLL |
|---|---:|---:|---:|---:|---:|
| all | 13,444 | 0.8176 | 1.0000 | 0.1824 | 0.3058 |
| ordinary non-switch | 2,384 | 0.8438 | 1.0000 | 0.1562 | 0.2227 |
| **switch-sensitive** (per action) | 6,412 | 0.6176 | 1.0000 | **0.3824** | 0.6412 |
| **exact packet aliases** | 6,848 | 0.6419 | 1.0000 | **0.3581** | 0.6004 |
| post-first-switch | 11,060 | 0.8120 | 1.0000 | 0.1880 | 0.3238 |
| post-two-changes | 7,012 | 0.8061 | 1.0000 | 0.1939 | 0.3392 |

Headroom is well above zero on the intended population, so the stop condition in §B does
not fire and the target measures hidden state.

## §C–§D Structured calibration

The predictor sees **local** public state — the four neighbours' blocked and switch bits,
the goal direction, the previous action, the public action result, the queried action, and
the normalised position. It does **not** see whether its own cell is a switch, because the
renderer paints the agent over it. The target is the **displacement class** (5 outcomes),
which is the quantity phase actually controls.

| condition | 64 updates | 256 | 1024 | alias ranking |
|---|---:|---:|---:|---:|
| 1 · structured current | 0.5505 | 0.5852 | **0.6472** | 0.5000 |
| 2 · **+ true phase** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 3 · correct history (recurrent) | 0.5630 | 0.5000 | 0.5000 | — |
| 4 · reversed history | 0.3667 | 0.3000 | 0.2500 | — |
| 4 · shuffled history | 0.3167 | 0.3000 | 0.3000 | — |

Condition 1 **exceeds** the 0.5264 uniform-phase-prior ceiling. That is expected and worth
stating precisely: the analytic figure assumes `p(phase | public state) = 0.5`, while phase
is partly predictable from public state in the trajectory distribution, so 0.5264 is a
*lower bound* on the memoryless Bayes rather than the Bayes itself.

Condition 2 reaches the oracle at the smallest budget on the ladder and stays there, with
NLL falling to 0.0002. Supplying phase helps enormously — 0.5000 → 1.0000 on alias
ranking — so the phase-input fusion and the prediction head are both sound, and
decision-table Outcome 2 is excluded.

## §E Budget ladder

64 / 256 / 1024 updates, frozen before validation, identical model, optimiser, batch
construction, seeds and loss. Condition 2 is converged by 64. Condition 1 is still
improving at 1024. Condition 3 peaks at 64 and *declines* thereafter, which is the shape of
a model fitting the marginal rather than acquiring state. Reporting the curve is what
distinguishes this from the K phase, where a single 60-update point was read as a property
of recurrence.

## §H Phase-gap recovery

`R_phase = [L(current) − L(recurrent)] / [L(current) − L(true_phase_oracle)]`

With current 0.6472, recurrent 0.5000 and oracle 1.0000, the numerator is **negative**.
The recurrent model recovers **none** of the phase gap and is worse than the memoryless
model it should strictly dominate. The required conditions for the central learned-history
result — `R_phase > 0` with an interval excluding zero, and recurrent beating
current-frame — are **not met**.

§L7 is reported as **not meaningful** rather than passing. Correct history (0.5000) does
beat reversed (0.3667) and shuffled (0.3167), but with no gain over the memoryless model to
begin with, that comparison measures a sequence model degrading rather than a history
advantage being removed. It only becomes interpretable once L6 passes.

## §G / §I Not run, by the rule

Visual representations and the multimodal ablations were **not attempted**. §G proceeds to
them only after structured-state calibration passes, and L6 fails. Running them now would
produce representation numbers from a temporal mechanism known not to work — which is the
error of the previous three phases in a new costume.

## §K Gates

| gate | status | basis |
|---|---|---|
| L0 | pass | provenance above |
| L1 | pass | headroom 0.3783 on alias classes, 0.3824 per-action |
| L2 | pass | 0.6472 against a 0.5264 lower-bound ceiling |
| L3 | pass | 1.0000 against the exact oracle, at every budget |
| L4 | pass | alias ranking 0.5000 → 1.0000 with true phase |
| L5 | pass | three-point ladder frozen in advance; full curves reported |
| **L6** | **FAIL** | recurrent 0.5000 < memoryless 0.6472; `R_phase` negative |
| L7 | not meaningful | uninterpretable while L6 fails |
| L8, L9, L11 | not run | forbidden by §G until structured calibration passes |
| L10 | pass | no hidden, event or provenance field in the main input |
| L12 | pass | episode-level intervals throughout |

**8 pass, 1 fail, 4 not run.**

## Bugs and corrections

1. **My first ceiling computation inflated the public baseline.** It scored a tie for the
   modal outcome as correct, so a 50/50 class read as 1.0 where the true answer is 0.5.
   Corrected to the modal mass; headroom on alias pairs roughly doubled, 0.1913 → 0.3581.
2. **`switch_sensitive` was defined identically to `post_first_switch`** — both
   `crossings >= 1` — so one of the two strata measured nothing. It is now per-action:
   the transitions whose own packet class contains more than one outcome for that action.
3. **My first structured encoding was a per-layout lookup table.** Absolute 144-way
   one-hots for agent, switches, walls and goal gave train 0.9661 against held-out 0.0655,
   because "the successor is adjacent" is not expressible in a form that survives a new
   wall pattern. Replaced with local, agent-relative features and a displacement target.
4. **The empirical ceiling was a singleton-class artifact.** Grouping by identical input
   bytes over a continuous 24-dimensional encoding makes almost every input unique and
   returns ~0.99. Replaced with an analytic ceiling derived from the dynamics.
5. **The history controls disturbed the step being predicted.** Permuting the whole
   sequence moves the current row too, so the drop measured "lost its own input" rather
   than "lost its history". The controls now corrupt only the prefix, and the recurrent
   conditions are scored at the final step, where every mode has its own row intact.

## Narrow supported claim

> On the reachable state space of the v2 environment, the public memoryless Bayes reference
> and the phase-aware oracle are exactly computable, and the phase headroom on exact
> packet-alias classes is 0.3783. A structured public-state predictor clears its memoryless
> reference, and the same predictor given the true phase as one additional scalar reaches
> the oracle exactly — 1.0000 next-displacement accuracy and 1.0000 alias-pair ranking, at
> every budget on a pre-registered ladder. The identical architecture given **history
> instead of phase** reaches 0.5000, below the memoryless model, so it recovers none of the
> phase gap. Because every positive control passes, this localises the failure to the
> construction of temporal belief and not to the predictor, the objective, the phase
> fusion, the target or the measurement. Nothing is established about any visual
> representation: those arms were not run, by rule. Stage 1A-1 is not unblocked.

