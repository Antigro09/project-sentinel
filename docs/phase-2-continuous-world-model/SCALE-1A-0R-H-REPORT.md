# Scale 1A-0R-H — Slot Resolution and Hidden-State Causal-Chain Audit

## Verdict

**Decision-rule outcome: `stop_testbed_or_readout_invalid`. The 87-workload screen is
NOT unblocked by this audit. No slot geometry is selected.**

The audit was asked which link of

    representation -> switch-event detection -> temporal accumulation
                   -> hidden-phase belief -> same-action outcome prediction

fails. It answers that question, but not in the register the question anticipated. The
chain does break, and it breaks at **link 2, switch-event detection**. The cause is not
slot resolution, and it is not the environment. It is that the shared readout never
recovers the quantity link 2 is built on.

Three results carry the verdict.

**The readout cannot read a prerequisite from a representation that provably contains
it.** Raw pixel blocks are a lossless partition of the frame — the reset-stripe
difference has an identical relative L2 of 0.40323 at every geometry, because a block
partition rearranges scalars without discarding any — and the agent occupies exactly four
uniquely-coloured pixels in every layout tested, including the worst-contrast one. The
shared RFF-ridge probe reads agent position from those pixels at **R² = 0.029**. The
specification anticipates exactly this case: *"If raw/CNN also fail: stop and classify the
audit as testbed/readout invalid."*

**Switch-event detection measures zero under every legitimate condition.** The best
`crossed_now` margin from a condition that does not supply the answer is **+0.001,
CI [−0.026, +0.028]** — and the shuffled controls match it at +0.001. The only non-zero
score, +0.173, comes from `exact_switch_event_history`, which hands the probe the true
crossing indicator; that is predicting a feature from itself, and an earlier version of
the decision logic counted it as recovery.

**The break is quantitative and was predicted before it was observed.** Switch-crossing
requires movement; movement is `position(t) − position(t−1)`. Movement's standard
deviation is 0.21× position's in this environment, so differencing two position estimates
subtracts the signal and adds the errors. At the best measured position fidelity
(R² = 0.548) the implied movement R² is **−18.4**; clipped at zero, that is precisely the
+0.013 observed. Position must clear roughly **R² = 0.99** before a difference carries any
signal at all.

| position R² | implied movement R² |
|---|---:|
| 0.550 (best arm measured) | −18.4 |
| 0.900 | −3.5 |
| 0.950 | −1.1 |
| **0.990** | **+0.57** |
| 0.999 | +0.96 |

What is *not* broken is everything downstream. Given true switch events, the recurrent
readout recovers hidden phase at **1.000**, and the evaluator oracle recovers it at
**1.000**. Accumulation and phase-from-events work perfectly. The chain is starved at its
input, not broken in its logic.

## Two things the audit does establish

The instrument failed at the chain, but two results stand on measurements it *can* make.

**Finer geometries are decisively worse, and must not be adopted.** Qwen position falls
0.548 → 0.366 moving from 4×4×256 to 8×8×64. Intervention non-inferiority fails badly
against the margin of −0.02 R² that was pre-registered in commit `e95f957`, before the run
that produced the numbers it judges. The largest paired losses reach −0.392
[−0.582, −0.207]. Matched capacity makes 8×8×64 a *trade*, not an upgrade: it buys spatial
resolution by projecting each 2560-dimensional token into 64 dimensions instead of pooling
four tokens into 256, and the channel reduction costs more than the spatial detail returns.

**Hidden phase is unidentifiable from a single POST-RESET observation.** 4329 pairs share a
byte-identical frame while holding opposite polarity, both members being genuine rollouts.
Feature identity is verified rather than assumed, under random sampling across layouts:
**0 mismatches** across four geometries × (raw, CNN) and **0 backbone token mismatches**.
In **100%** of pairs some action reaches a different successor, ~97% per individual action.

Two qualifications, both of which an adversarial pass forced and neither of which was in
the first draft of this report.

*The scope is post-reset, not "any observation".* `render_v2` paints polarity as a
full-width one-pixel top stripe whenever `_step == 0`, and **12.93%** of evaluation rows
(120 of 928) are reset frames. On those, a two-line rule comparing frame row 0 against row 1
recovers polarity at **1.0000**; on post-reset rows the same rule sits at 0.5198. Byte-identity
can never pair a reset frame against a post-reset one, so the pair set *structurally
excludes* precisely the observations where phase is single-frame readable. The original claim
quantified over all observations and was false.

*This also explains an apparent leak.* The audit's own `current_frame_only` arm scores
`polarity` above chance (margin +0.098, CI [+0.057, +0.138]) and `initial_polarity` at
+0.144, CI [+0.067, +0.225]. By the harness's own docstring anything above chance "is a
leak" — but it is not one. It is the 12.9% of reset frames doing legitimate work, and the
docstring's rule was stated too broadly.

*The 0.5 is arithmetic, not a measurement.* `evaluate_pairs` sets `delta = np.zeros(...)`
and returns `0.5 if features_identical`. No probe is run on the pairs. Items 7 and 8 are
therefore reported as an **identity proof**, not as probe results — identical features force
identical predictions, hence an exact tie. The artifact now labels them that way.

*"Identical observation" was overstated.* `content_dict` deliberately excludes `step`, so
identical content digests do not mean identical envelopes: **1915 of 4329 pairs (44.2%)**
have members sitting at different steps, which the envelope exposes via `step` and
`timestamp_ns`. The pairs are aliased on the *frame*, not on the observation.

## Where the specification's pins stand

`shuffled controls sit at baseline` **fails**, and the failure is diagnostic rather than a
leak. On every history-dependent target the controls match the legitimate conditions:

| target | best legitimate | best control |
|---|---:|---:|
| `crossed_now` | +0.001 | +0.001 |
| `crossing_count` | +0.469 | +0.460 |
| `crossing_parity` | +0.110 | +0.083 |
| `move_row` | +0.012 | **+0.013** |
| `polarity` | +0.138 | +0.113 |

The correct conditions are not beating the shuffled ones because the ridge readout extracts
nothing from history at all — every temporal condition collapses to the current-frame
content all of them share. `crossing_count` scoring +0.46 under a shuffled control is
position correlating with how far into an episode the agent is, not accumulation.

The recurrent readout is the contrast that makes this diagnostic: given clean events it
reaches 1.000 against 0.098 for current-frame-only, so it *does* use history when the
channel carries signal. Correct and shuffled action sequences score +0.142 and +0.141 —
indistinguishable — which says the action–outcome coupling is never recovered.

`all tested histories legally reachable` is reported honestly rather than asserted: the
correct histories are replayed against the environment and match, while reversed and
shuffled histories are order-destroying controls that deliberately do **not** correspond to
reachable states. Claiming otherwise would be the dishonest reading of the pin.

## Links 2 and 3 are the same event

The specification asks for "whether transition t crossed a switch" and "whether transition t
changed polarity" as separate measurements. In this environment polarity flips exactly when
the agent enters a switch cell, so the two are definitionally identical — verified at
**461/461 rows**. Measuring both is a label-consistency check, not two independent links,
and the identical scores in the chain table reflect that rather than a coincidence.

## Bugs and corrections

Six, four of them in my own instrument, and three of the four are the same failure mode:
a check that passes because it is blind.

1. **The first hidden-state pin inspected nothing.** It looked for observation fields named
   `features`, `metadata`, and `scalar_sensors`. `ObservationEnvelope` has none of them, so
   the pin examined an empty set, found no forbidden names, and reported clean. It now reads
   the fields that exist (`goal_text`, `frame_shape`, `markers_visible`, `audio_present`,
   `image`) and carries a planted-key calibration arm that must fire.

2. **The paired identity check compared `None` to `None`.** `evaluate_pairs` took the pair
   features as a parameter and `main` passed `None`, so `np.array_equal(None, None)`
   returned `True` and "features identical" passed without a feature ever being computed.
   It now builds every geometry from both members' frames and compares elementwise.

3. **The decision rule counted a circular condition as recovery.** `coarse_recovers_switch_events`
   did not filter by condition, so `exact_switch_event_history` — which supplies the true
   crossing indicator — satisfied it. Excluding conditions that hand over the answer flips
   the finding from `True` to `False`, and the outcome from `keep_4x4` to
   `stop_testbed_or_readout_invalid`.

4. **The first paired construction leaked through a non-pixel channel.** It reached the
   opposite phase by changing the phase seed and inverting the action sequence. The frames
   matched, but `LevelV2.digest` embeds `initial_polarity`, so `environment_version` and
   hence `content_digest` differed and the two members were separable through a channel
   unrelated to the image. Pairs are now built inside a single level, reaching one cell by
   two routes of differing crossing parity.

5. **`environment_version` embeds `initial_polarity`** (`procedural_visual_v2.py:309` via
   `LevelV2.digest`), and `content_dict` includes it, so an observation's content digest
   encodes a hidden variable. Traced end to end: `_observation_text` feeds the encoder only
   `structured_observation`, so it never reaches a feature path and this is **not** a
   training leak. It does mean content-digest duplicate detection cannot see two
   frame-identical states that differ in initial polarity.

6. **A first attempt at an adequate-probe control was undertrained and nearly reported as a
   result.** A convolutional probe scored 0.011 on position against ridge's 0.548 — read
   naively that would have said the task was hard. Retrained with minibatches and
   standardised targets it reached **train 0.967 / test 0.037**: not a hard task, an
   overfit one. The agent is drawn as `255 − palette[0]`, a different colour in all 12
   layouts sampled, so cross-layout localisation demands an appearance-invariant rule that
   180 training layouts do not supply.

7. **The report's own headline claim overreached, and adversarial review caught it.** The
   first draft asserted that hidden phase is unidentifiable from a single observation "at
   ANY slot resolution". It is unidentifiable from a single *post-reset* observation; on the
   12.9% of rows that are reset frames it is perfectly readable by a two-line rule. The pair
   construction had selected exactly the subpopulation where the claim holds and the
   conclusion was stated over all of it.

## Adversarial verification

Five load-bearing claims were put to independent refutation agents, two lenses each. Seven
of thirteen agents died on a session limit, so coverage is **partial**: the identifiability
claim was fully adversarially tested, the readout-invalid, geometry-rejection and
decision-rule claims were **not**, and neither completeness critic ran. That is a gap in
this report, not a clean bill of health.

Both agents that did run **refuted the identifiability claim at high confidence**, and both
were right. Every objection was reproduced independently before being accepted: the 12.93%
reset-frame fraction, the 1.0000 stripe-rule accuracy, the `content_dict` step exclusion,
the hardcoded `delta = np.zeros(...)`, and the prefix-sampling flaw in the identity checks
(`pairs[:24]` drew entirely from one layout; the check is now a random sample across all 90
and still returns zero mismatches). The claim has been narrowed accordingly.

One further objection is recorded but not yet acted on: roughly **11% of reachable positions
admit only one polarity**, so position alone fixes phase there, and those states are also
absent from the pair set. This does not change the verdict — the chain is readout-blocked
before it reaches phase — but it further narrows what the pair construction can support.

The three unrefuted claims should be treated as **unverified rather than confirmed**.

## What would unblock the screen

The chain question is not answered and cannot be with this instrument. The binding
requirement is now specific and measurable: **a readout that recovers agent position at
R² ≳ 0.99 across held-out layouts.** Below that, movement, switch-crossing, parity and
phase are all unmeasurable no matter which representation or geometry supplies the features,
and any score reported for them describes the probe.

Nothing here licenses a claim about the interfaces. In particular, per the specification,
**R4 must not be scored as a world-model failure** — it is readout-blocked, and an
interface-blocked gate is not evidence about the world model.

## Provenance

| item | value |
|---|---|
| commit | `2ad6e290b785a8b280365a09c882108fba2a561f` |
| branch | `phase-2-continuous-world-model` |
| wall clock | 21.0 min |
| train / val / test rows | 1385 / 468 / 928 |
| same-observation pairs | 4329 |

### Test suite at this commit

| suite | result |
|---|---|
| exact-reference (`tests/`, excluding `tests/shwm/`) | 522 passed |
| Phase-2 (`tests/shwm/`) | 425 passed, 4 skipped |
| complete repository | 947 passed, 4 skipped |

The 4 skips are backbone-dependent tests that require `mlx-vlm`, absent from the exact
environment by design and passing in `.venv-shwm`.

## Slot geometry, storage and compute

| geometry | slots | width | scalars | GB @100k steps | cells/block | cell-aligned | role |
|---|---|---|---|---|---|---|---|
| g4x4x256 | 16 | 256 | 4096 | 1.64 | 3.0 | yes | current |
| g8x8x64 | 64 | 64 | 4096 | 1.64 | 1.5 | **no** | matched capacity fine |
| g8x8x256 | 64 | 256 | 16384 | 6.55 | 1.5 | **no** | diagnostic high capacity |
| g12x12x64 | 144 | 64 | 9216 | 3.69 | 1.0 | yes | cell aligned diagnostic |

## Causal chain

| link | stratum | window | best margin | 95% CI | arm | geometry | condition | oracle |
|---|---|---|---|---|---|---|---|---|
| movement direction (prerequisite) | all | full | +0.012 | [-0.064, +0.086] | qwen3 vl 4b | g4x4x256 | current frame only | +0.049 |
| movement direction (prerequisite) | all | short | +0.013 | [-0.063, +0.085] | gemma3 4b | g4x4x256 | reversed history | +0.049 |
| agent position (prerequisite) | all | full | +0.548 | [+0.454, +0.630] | qwen3 vl 4b | g4x4x256 | current frame only | +1.000 |
| agent position (prerequisite) | all | short | +0.490 | [+0.403, +0.568] | qwen3 vl 4b | g4x4x256 | correct history | +1.000 |
| 1. switch position / presence | all | full | +0.553 | [+0.458, +0.634] | qwen3 vl 4b | g4x4x256 | current frame only | +1.000 |
| 1. switch position / presence | all | short | +0.464 | [+0.364, +0.555] | qwen3 vl 4b | g4x4x256 | shuffled action sequen | +1.000 |
| 2. transition crossed a switch | all | full | +0.015 | [-0.014, +0.043] | learned cnn | g12x12x64 | exact switch event his | +0.150 |
| 2. transition crossed a switch | all | short | +0.173 | [+0.148, +0.199] | learned cnn | g4x4x256 | exact switch event his | +0.150 |
| 3. transition changed polarity | all | full | +0.015 | [-0.014, +0.043] | learned cnn | g12x12x64 | exact switch event his | +0.150 |
| 3. transition changed polarity | all | short | +0.173 | [+0.148, +0.199] | learned cnn | g4x4x256 | exact switch event his | +0.150 |
| 4. parity / count of crossings | all | full | +0.373 | [+0.276, +0.469] | fixed random spatial | g12x12x64 | exact switch event his | +1.000 |
| 4. parity / count of crossings | all | short | +0.329 | [+0.230, +0.423] | raw lowres spatial | g8x8x256 | exact switch event his | +1.000 |
| 4. parity / count of crossings | post_first_switch | full | +0.413 | [+0.273, +0.537] | gemma3 4b | g4x4x256 | exact switch event his | +1.000 |
| 4. parity / count of crossings | post_first_switch | short | +0.303 | [+0.163, +0.435] | gemma3 4b | g8x8x64 | exact switch event his | +1.000 |
| 4. parity / count of crossings | post_two_changes | full | +0.469 | [+0.354, +0.575] | gemma3 4b | g8x8x64 | correct history | +1.000 |
| 4. parity / count of crossings | post_two_changes | short | +0.420 | [+0.304, +0.530] | learned cnn | g12x12x64 | exact switch event his | +1.000 |
| 5-6. hidden phase | all | full | +0.125 | [+0.094, +0.155] | fixed random spatial | g12x12x64 | correct action sequenc | +0.523 |
| 5-6. hidden phase | all | short | +0.138 | [+0.102, +0.175] | learned cnn | g4x4x256 | correct action sequenc | +0.523 |
| 5-6. hidden phase | post_first_switch | full | +0.066 | [+0.027, +0.106] | qwen3 vl 4b | g4x4x256 | shuffled history | +0.517 |
| 5-6. hidden phase | post_first_switch | short | +0.066 | [+0.026, +0.106] | fixed random spatial | g8x8x64 | correct action sequenc | +0.517 |
| 5-6. hidden phase | post_two_changes | full | +0.099 | [+0.043, +0.155] | qwen3 vl 4b | g8x8x64 | correct history | +0.536 |
| 5-6. hidden phase | post_two_changes | short | +0.118 | [+0.052, +0.181] | learned cnn | g4x4x256 | correct action sequenc | +0.536 |
| 5a. initial polarity (reset stripe) | all | full | +0.433 | [+0.407, +0.457] | fixed random spatial | g12x12x64 | exact switch event his | +0.541 |
| 5a. initial polarity (reset stripe) | all | short | +0.259 | [+0.203, +0.315] | fixed random spatial | g4x4x256 | exact switch event his | +0.541 |
| 9. counterfactual intervention | all | full | +0.547 | [+0.451, +0.633] | qwen3 vl 4b | g4x4x256 | current frame only | +0.947 |
| 9. counterfactual intervention | all | short | +0.483 | [+0.385, +0.569] | qwen3 vl 4b | g4x4x256 | correct action sequenc | +0.947 |
| 9. counterfactual intervention | post_first_switch | full | +0.546 | [+0.441, +0.646] | qwen3 vl 4b | g4x4x256 | current frame only | +0.972 |
| 9. counterfactual intervention | post_first_switch | short | +0.473 | [+0.359, +0.581] | qwen3 vl 4b | g4x4x256 | correct action sequenc | +0.972 |
| 9. counterfactual intervention | post_two_changes | full | +0.508 | [+0.382, +0.619] | qwen3 vl 4b | g4x4x256 | current frame only | +0.966 |
| 9. counterfactual intervention | post_two_changes | short | +0.486 | [+0.356, +0.601] | qwen3 vl 4b | g4x4x256 | correct action sequenc | +0.966 |

## Temporal positive controls

| control | result | verdict |
|---|---|---|
| exact parity accumulator reproduces recorded polarity | 1.0000 | labels valid |
| GRU accumulates parity from events + initial value | 1.0000 | readout capable |

## Recurrent readout on real features

| condition | score | baseline | margin | best arm | geometry |
|---|---|---|---|---|---|
| exact switch event history | 1.000 | 0.477 | +0.523 | learned_cnn | g8x8x256 |
| correct action sequence | 0.620 | 0.477 | +0.142 | fixed_random_spatial | g8x8x64 |
| correct history | 0.619 | 0.477 | +0.141 | fixed_random_spatial | g8x8x256 |
| shuffled action sequence | 0.619 | 0.477 | +0.141 | raw_lowres_spatial | g12x12x64 |
| current frame only | 0.575 | 0.477 | +0.098 | raw_lowres_spatial | g8x8x256 |

## Geometry differences against the 4×4 reference

| geometry | arm | target | condition | stratum | Δ vs 4×4 | 95% CI | direction |
|---|---|---|---|---|---|---|---|
| g8x8x64 | qwen3_vl_4b | successor_3 | correct history | post_two_changes | -0.392 | [-0.582, -0.207] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_1 | correct history | post_two_changes | -0.390 | [-0.583, -0.202] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_2 | correct history | post_two_changes | -0.375 | [-0.573, -0.183] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_0 | correct history | post_two_changes | -0.364 | [-0.524, -0.211] | **worse** |
| g8x8x256 | qwen3_vl_4b | successor_0 | reversed history | all | -0.352 | [-0.502, -0.204] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_3 | exact switch event h | post_two_changes | -0.343 | [-0.569, -0.126] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_3 | correct action seque | post_two_changes | -0.341 | [-0.519, -0.164] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_1 | exact switch event h | post_two_changes | -0.337 | [-0.562, -0.120] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_0 | exact switch event h | post_two_changes | -0.337 | [-0.534, -0.156] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_1 | correct action seque | post_two_changes | -0.336 | [-0.515, -0.160] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_3 | shuffled action sequ | post_two_changes | -0.334 | [-0.510, -0.164] | **worse** |
| g8x8x256 | qwen3_vl_4b | agent_row | shuffled history | all | -0.329 | [-0.449, -0.213] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_1 | shuffled action sequ | post_two_changes | -0.328 | [-0.504, -0.157] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_0 | shuffled action sequ | post_two_changes | -0.327 | [-0.494, -0.163] | **worse** |
| g8x8x256 | qwen3_vl_4b | agent_row | reversed history | all | -0.325 | [-0.474, -0.178] | **worse** |
| g8x8x256 | qwen3_vl_4b | nearest_switch_row | shuffled history | all | -0.325 | [-0.417, -0.232] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_0 | correct action seque | post_two_changes | -0.324 | [-0.496, -0.146] | **worse** |
| g8x8x256 | qwen3_vl_4b | successor_1 | shuffled history | all | -0.322 | [-0.449, -0.202] | **worse** |
| g8x8x256 | qwen3_vl_4b | successor_3 | shuffled history | all | -0.320 | [-0.445, -0.200] | **worse** |
| g8x8x256 | qwen3_vl_4b | successor_0 | correct history | all | -0.320 | [-0.450, -0.197] | **worse** |
| g8x8x64 | qwen3_vl_4b | successor_2 | exact switch event h | post_two_changes | -0.319 | [-0.540, -0.097] | **worse** |
| g8x8x256 | qwen3_vl_4b | successor_3 | reversed history | all | -0.314 | [-0.464, -0.166] | **worse** |
| g8x8x256 | qwen3_vl_4b | agent_row | correct history | all | -0.310 | [-0.441, -0.181] | **worse** |
| g8x8x64 | qwen3_vl_4b | nearest_switch_row | correct history | all | -0.310 | [-0.450, -0.181] | **worse** |

1072 of 5239 paired comparisons have an interval clear of zero.

## Same-observation pairs (items 7 and 8)

**7_phase_discrimination** — 4329 pairs; identity proof, not a probe result; implied value 0.500 against chance 0.500; features identical: True (scope: post-reset observations only).

> 0.5 here is arithmetic, not a measurement: identical features force identical predictions, hence an exact tie. No probe is run. The scope matters -- byte-identity can never pair a reset frame against a post-reset one, so this set excludes precisely the observations where the polarity stripe makes phase readable from one frame.

**8_same_action_outcome_ranking** — 4329 pairs; identity proof, not a probe result; implied value 0.500 against chance 0.500; features identical: True (scope: post-reset observations only).

> the same action reaches a different successor while the observation is identical, so a current-frame predictor is necessarily wrong on one member of every such pair


## Pins

| pin | result |
|---|---|
| public observation fields | `audio_present`, `frame_shape`, `goal_text`, `image`, `markers_visible` |
| forbidden names present | [] |
| planted-key calibration fires | True |
| pairs with differing content digest | 0 of 4329 |
| correct histories replayed clean | 315 rows, 0 mismatches |

## Decision

| clause | value |
|---|---|
| any non oracle recovers events | False |
| coarse recovers switch events | False |
| effect in a pretrained package | True |
| fine grid improves phase or outcome | True |
| fine grid improves switch detection | True |
| high capacity works matched does not | False |
| intervention non inferior | False |
| pixel sources recover events | False |
| position clears differencing threshold | False |
| position r2 best arm | 0.5479 |
| position r2 from lossless pixels | 0.029 |
| readout recovers prerequisites | False |
| **outcome** | **stop_testbed_or_readout_invalid** |
| selected geometry | None |
| pre-registered non-inferiority margin | -0.02 |
| worst intervention CI low at 8×8×64 | -0.4310 |
| **87-workload screen unblocked** | **False** |
