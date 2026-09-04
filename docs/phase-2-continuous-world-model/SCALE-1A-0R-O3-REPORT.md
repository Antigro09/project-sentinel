# Scale 1A-0R-O3 — Palette-Orbit Equivariance, Calibration Sufficiency, and Query-Scoped Uncertainty Closure

Branch `phase-2-continuous-world-model`. Parent commit `87c70e4` (Scale 1A-0R-O2).

The single most important result of this phase is that **O2's Q7 failure was an
input-representation defect, not a perception limit**, and that O2's own stated diagnosis
pointed at the wrong axis. The second most important is a list of things this phase got
wrong before it got them right, all of which are retained below rather than tidied away.

**Closure ledger: 11 PASS, 2 FAIL (R5, R6), 1 SEE_BASIS (R8), 1 PASS_UNQUALIFIED_INSTRUMENT
(R13).** The fresh uncertainty replication — R9, R10, R11 — passes on fresh palettes with
thresholds frozen on development, which was the stated precondition for prospective
prediction. It is necessary and not sufficient: **the decision is to close R5 and R6
first.**

No prospective prediction was begun. No Stage 1A-1 matrix was run. No visual backbone was
added, no model size was increased, and the 87-workload matrix was not launched.

---

## 1. Provenance and the two ledgers

`artifacts/shwm/scale1/o3-gates.json` carries three things that must not be merged.

**The historical O2 ledger is preserved byte for byte** as
`historical_q_ledger_preserved`. No O2 gate code was rewritten. Its tally stands as coded:
13 PASS, 1 FAIL, 1 PARTIAL.

**A separate AUDITED_CLAIM ledger** records, for each gate, what the code computed and
what the evidence supports. Tally: **10 PASS, 3 FAIL, 2 UNSUPPORTED_AS_PALETTE_ROBUSTNESS**.

| Gate | Coded in O2 | Audited | Why |
|---|---|---|---|
| Q4 | PASS | **FAIL** | The binder beat the two learned controls but did not beat the exact count-only Bayes ceiling with an interval excluding zero (+0.0181). O2's own artifact field `cardinality_not_ruled_out_at_the_pooled_level` was already True. |
| Q7 | FAIL | FAIL | Preserved. See item 3 — the *reason* recorded in O2 was wrong, not the verdict. |
| Q8 | PASS | **UNSUPPORTED** | +0.1605 [+0.1296, +0.1957] is a distribution average over rows, not a palette-robust interval. |
| Q9 | PASS | **UNSUPPORTED** | +0.1903 [+0.1125, +0.2694], same objection. |
| Q13 | PARTIAL | **FAIL** | Under the *original* qualitative requirement — unidentifiable cases returned unresolved rather than confidently assimilated — 0.4055 confident assimilation is a failure. The PARTIAL rested on a criterion tightened mid-phase. |

Q8 and Q9 are marked UNSUPPORTED rather than FAIL deliberately. Once the pipeline is
bit-exact palette-equivariant (item 2), per-palette variation on *fixed* semantic content
is zero by construction, so those intervals cannot speak to palette-robust generalisation
in either direction. They are not wrong; they are not evidence for the thing they were
cited for.

---

## 2. The palette-orbit audit, and what it found

`o3-orbit.json`, `o3-route-orbit.json`. Semantics held fixed; rendering varied over an
orbit of **848 role-to-colour permutations** (an exhaustive subgroup plus random
permutations), with outputs mapped back into semantic event space before comparison.

| View | Equivariant | Max logit deviation | Max semantic assignment deviation |
|---|---|---|---|
| `full_token` | **No** | 1.645e-01 | 5.99e-04 |
| `no_rgb` | **Yes** | 0.000e+00 | 0.000e+00 |

**R1 PASS.** Five planted defects were caught: `slot_index_bias`, `rgb_sorted_tokens`,
`cross_palette_memory` (max logit deviation 26.87, decision flip fraction 1.0),
`palette_index_feature`, `memory_key_collision`. The honest pipeline shows 0.0000 spread
over the same orbit, so the test is not vacuous.

**Two planted defects were mis-specified and are retained as RECLASSIFIED, not deleted:**

- `scan_order_registry` — a scan-order registry is only a permutation of slots, and the
  binder is permutation-equivariant, so the orbit spread is exactly 0.00e+00. Not an
  equivariance defect for this architecture. The hazard it stood for is cross-palette
  memory addressing, which `cross_palette_memory` tests directly.
- `role_dependent_colour` — pinning a role to a fixed pool colour breaks *generator
  honesty*, not equivariance; it is caught by O2's leakage guard A at 1.1490 bits.

### The finding

O2 trained its memory arm on `full_token`, which carries the raw RGB block, while **O2's
own factorial had selected `count_plus_motion`**, which does not. On the route leg with
leak-free calibration, `full_token` gives 0.7638 [0.5801, 1.0000] with exactly two
distinct values across 64 palettes — reproducing O2's bimodal split — and `no_rgb` gives a
flat 1.0000.

Classification, as the specification requires before touching calibration or capacity:
**PALETTE-EQUIVARIANCE / CANONICALIZATION DEFECT.**

A precision that matters and that a test in this phase initially got wrong: the property
is **equivariance, not invariance**. Tokens are indexed by colour slot, so relabelling the
palette moves each role's token to a different row; what must hold is that the multiset of
rows is unchanged. Asserting bitwise equality fails on a correct pipeline.
`tests/shwm/test_shwm_o3.py::test_no_rgb_is_equivariant_and_full_token_is_not` pins the
correct statement.

---

## 3. Q7 re-decided (R7)

O2 recorded route parity 0.6491 and diagnosed a calibration-sufficiency question. With the
representation defect removed and calibration leak-free, the same pipeline on `no_rgb`
gives route parity **0.9943 [0.9831, 1.0000]** with **63 of 64** validation palettes above
the frozen 0.75 gate and none collapsed.

**The gate itself was not moved.** Q7's target remains route parity ≥ 0.75.

Two statements must be kept apart here, because only the first is robust.

- **The equivariance finding is structural.** `full_token` is not palette-equivariant and
  `no_rgb` is, bit-exact, over 848 permutations, with a passing negative control and a
  test that pins it. O2's stated diagnosis — calibration sufficiency — pointed at the
  wrong axis, and that conclusion does not depend on any single training run.
- **The route-parity *number* is seed-dependent.** 0.9943 is one training run. Item 13
  measures the same quantity across seeds and finds it spanning 0.71 to 1.00. Q7 is
  re-decided in the sense that the defect O2 blamed was not the defect present; it is not
  re-decided in the sense of a stable number above the gate.

---

## 4. Fresh frozen populations, resampled by palette (R3)

`o3-population.json`. Three disjoint populations of 64 palettes each — development
(20000–20063), validation (21000–21063), replication (22000–22063, held untouched until
every decision was frozen). Each palette draws its **own** calibration layouts, transfer
layouts and alias layouts.

That last point is forced, not stylistic. Once the pipeline is bit-exact
palette-equivariant, two palettes over one semantic trace return *identical* numbers, so
between-palette variance on shared content is exactly zero by construction. The variance
reported is the variance of the content each palette drew, and the artifact says so in
`equivariance_note`.

Validation: contested transfer **0.9969 [0.9908, 1.0000]**, p10 1.0000, min 0.8046. Route
parity **0.9943 [0.9831, 1.0000]**, min 0.6543, 63/64 above gate, 0 collapsed.

These are **single-seed** figures (training seed 53000, 4973 training rows). They reproduce
exactly — verified by re-running section C and comparing palette by palette — but item 13
shows the same measurement under two other seeds lands far lower, and section J reports it
across seeds for that reason. R3 stands as recorded; read it as one draw.

---

## 5. Calibration sufficiency, audited per palette (R2)

`o3-calibration.json`. **63 passes, 1 failure**, classified as
`2_learned_inference_failure` (palette 21034: minimum separation 2, event entropy 0.000 —
the evidence identified the class and the learned arm did not use it).

The exact event class is identified on **64 of 64** palettes; the goal class on **0 of
64**. Category 3 (`equivariance_or_canonicalization_defect`) is **structurally
unreachable** given R1 and is retained in the table so that a regression lands visibly
rather than silently.

---

## 6. Frozen calibration policies at equal budget (R5 — coded FAIL)

`o3-policy.json`. Eight frozen policies, each consuming exactly 48 frame pairs, over 32
validation palettes.

A prediction was written into the file **before** the run: section D had measured the
event class pinned after 2–6 of ~48 interactions on every palette, so the expectation was
that policies would not separate. **That prediction was falsified.**

| Policy | Interactions to pin | Pinned | Balanced accuracy |
|---|---|---|---|
| 7_many_short (12 layouts × 5) | **4.88** | 1.0000 | **0.7051** |
| 1_uniform_spread (6 × 9) | 7.97 | 1.0000 | 0.6240 |
| 2_uniform_concentrated (1 × 49) | 7.97 | 1.0000 | 0.6194 |
| 4_switch_seeking_concentrated | 8.68 | **0.9688** | 0.6232 |
| 5_goal_seeking_spread | 8.94 | 1.0000 | 0.6250 |
| 3_switch_seeking_spread | 9.31 | 1.0000 | 0.6488 |
| 6_goal_seeking_concentrated | 9.97 | 1.0000 | 0.6276 |
| 8_few_long (3 × 17) | **15.50** | 1.0000 | **0.5931** |

**Layout diversity, not interaction count, is what pins the palette.** Twelve short
episodes across twelve layouts pin in 4.88 interactions and score 0.7051; three long
episodes across three layouts take 15.50 and score 0.5931 — at an identical budget.

Under `4_switch_seeking_concentrated`, palette 21023 **never pins**: after the full 48
interactions the surviving class still has 8 members with event mass 0.5. This qualifies
section D's result — "pinned on every palette" was measured under uniform-spread and is a
property of that policy, not of the task.

**R5 is recorded FAIL as coded.** The coded criterion was `all_policies_pin_the_event`, and
one policy does not. The file's own comment states a broader intent ("a real separation OR
a demonstrated absence of headroom both close it"), and a real separation *was* measured —
but the code implemented only the second branch, and loosening a criterion after seeing it
fail is exactly the move this track keeps correcting. The coded value stands; the
substantive finding is recorded beside it.

---

## 7. Persistent-memory replication at the palette level (R4 PASS)

`o3-persistent.json`. Eleven arms, 64 validation palettes, 3 training runs, `no_rgb`.

### The statistic had to be fixed first

O2 drew every palette's transfer rows from one shared layout pool, so the contested
population had a single fixed SWITCH-against-DECOY base rate. Section C requires each
palette to draw its own content, which makes the base rate a **per-palette** quantity:
measured at **0.5801 overall, ranging 0.2667 to 0.8000**.

The first run of this section returned 0.6182 for the stateless frame-pair binder on plain
accuracy — which is that base rate, not a capability, and which would otherwise have
appeared to breach O2's exact count-only ceiling of 0.5000 on contested rows. Every
headline below is **balanced accuracy**, and **arm 0 is the majority-class strategy
carried as the calibration arm**. It lands on exactly 0.5000 balanced while scoring 0.6067
plain, so the trap is visible in the table itself.

| Arm | Balanced | Palette 95% | Row 95% | Plain |
|---|---|---|---|---|
| 0_majority_class *(calibration arm)* | **0.5000** | [0.5000, 0.5000] | [0.5404, 0.5736] | 0.6067 |
| 1_current_frame_binder | 0.5668 | [0.5454, 0.5887] | [0.5592, 0.5860] | 0.5751 |
| 2_frame_pair_binder | 0.5839 | [0.5554, 0.6118] | [0.5723, 0.6038] | 0.6102 |
| **3_recurrent_assignment_memory** | **0.6950** | [0.6375, 0.7503] | [0.6968, 0.7219] | 0.7101 |
| 4_exact_palette_posterior | 1.0000 | [1.0000, 1.0000] | [1.0000, 1.0000] | 1.0000 |
| 5_augmentation_only_detector | 0.5068 | [0.4798, 0.5373] | [0.5006, 0.5221] | 0.4485 |
| 6_memory_reset_before_transfer | 0.5000 | [0.5000, 0.5000] | [0.5000, 0.5000] | 0.4199 |
| 7_shuffled_calibration | 0.5722 | [0.5033, 0.6358] | [0.5623, 0.5933] | 0.5726 |
| 8_wrong_colour_pairings | 0.4745 | [0.4349, 0.5157] | [0.4623, 0.4839] | 0.4069 |
| 9_calibration_from_another_palette | 0.5174 | [0.4615, 0.5762] | [0.5112, 0.5372] | 0.4670 |
| 10_no_persistent_memory | 0.5839 | [0.5556, 0.6121] | [0.5719, 0.6044] | 0.6102 |
| 11_oracle_palette_map | 1.0000 | [1.0000, 1.0000] | [1.0000, 1.0000] | 1.0000 |

**Required contrasts, resampled by palette:**

- memory − current-frame-binder: **+0.1283 [+0.0661, +0.1863]**, wins on **48/64** palettes
- memory − augmentation-only: **+0.1882 [+0.1242, +0.2495]**, wins on **49/64** palettes

Both exclude zero at the palette level. **R4 PASS.**

**The palette interval is 4.5× the row interval** on the memory arm (0.1128 against
0.0251). That ratio is the whole point of the section: thousands of rows from 64 palettes
supply 64 independent tests of palette generalisation, not thousands.

Note also that the memory's 0.6950 is well below O2's reported 0.865. O2's figure was
measured on `full_token` with content shared across palettes; this one is on the
equivariant view with each palette drawing its own.

---

## 8. Query-scoped uncertainty (R9 PASS)

`o3-uncertainty.json`. EVENT, GOAL and FULL are scored separately with per-query
thresholds `{EVENT: 0.40, GOAL: 0.50, FULL: 0.90}`, selected on development and frozen
before validation. No single global per-row confidence margin is used.

GOAL coverage is **0.0000 everywhere**, which is correct: section D found the goal class
identified on 0 of 64 palettes, so a system that never claims to know the goal is behaving
properly.

---

## 9. The per-frame negative control, rebuilt (R10 under the margin rule: FAIL)

O2's `PER_FRAME_PERMUTATION` was not an impossibility control — 0 of 672 legal pairs
carried distinct permutations. Three genuinely distinct regimes were built and verified to
differ:

| Regime | Distinct maps/episode | Colour is a function of role | Exact event identifiable |
|---|---|---|---|
| PERSISTENT_CONVENTION | 1.00 | Yes | 0.5104 |
| PER_FRAME_BIJECTION | 9.00 | Yes | 0.5104 |
| PER_CELL_NOISE | 0.00 | **No** | **0.0** |

Under `PER_CELL_NOISE` the exact audit says nothing is identifiable on any query, and the
confidence-margin rule **answers 100% of rows and is wrong on 0.5598 of them**.

**A confidence-margin abstention rule cannot express "I have no information."** It reads
the model's certainty about its own output, and a degenerate assignment is *saturated* —
so the margin reads maximum confidence exactly where the model knows least. This is a
property of the rule, not a tuning failure, and it is why item 10 exists. The failing rule
is retained on the record rather than replaced silently.

---

## 10. Cause hypotheses, provisional branching, and the evidence-side signal (R10, R11 PASS)

`o3-change.json`. Eleven arms, 32 fresh validation palettes, thresholds selected on
development palettes only and frozen before validation (`promote_after=3`,
`min_components=2`).

### Two signals were tried and refuted first

Both are retained in the artifact's `refuted_signals`.

- **Set-equality on on-screen colours.** A palette is a permutation of *one* eight-colour
  pool, so two palettes share six or seven of their seven colours and the colour set is
  nearly palette-invariant. What it does track is content. Measured: the no-change control
  false-alarmed at **0.25 per palette** and mean detection delay went **negative**.
- **Magnitude of interaction support.** `support <= 0` catches PER_CELL_NOISE (support 0)
  but not PER_FRAME_BIJECTION (support 6), which carries nonzero support and still
  exhibits no colour-to-role map. Any threshold low enough to catch the latter also fires
  on an honest quiet episode.

### What works is consistency, not magnitude

Three anchors are derivable from geometry and behaviour with no model and no role labels:
**BORDER** (modal colour of the outer ring → WALL), **FIELD** (modal interior colour,
border excluded → EMPTY), **MOVER** (colour of the moving singleton → AGENT). Measured over
24 palettes each is single-valued within a palette and equals the true role colour on
**24/24**. Under PER_FRAME_BIJECTION they take 7–8 distinct values inside one episode, and
under PER_CELL_NOISE 7–10.

`FIELD` excludes the border before taking the mode because the plain modal colour flips
from EMPTY to WALL on a dense layout — measured on palette 20002, where it took two values
inside a single palette.

The three causes are **SAME_PALETTE**, **NEW_PALETTE** and **MISSING_APPEARANCE**. A
contradiction opens a provisional branch, and **confirmed memory is never overwritten
until that branch is promoted**.

| Arm | Detect | Delay | False alarms | Corruption | Unresolved | False-confident | Recovery acc. |
|---|---|---|---|---|---|---|---|
| 1_no_change_detector | 0.0000 | – | 0.00 | **1.00** | 0.1684 | 0.2278 | 0.7237 |
| 2_declared_reset | – | – | 0.00 | 0.00 | 0.0776 | 0.3014 | 0.6532 |
| 3_contradiction_detector | **1.0000** | 0.00 | 0.00 | 0.00 | 0.0776 | 0.3014 | 0.6532 |
| 4_exact_change_point_ceiling | 1.0000 | 0.00 | 0.00 | 0.00 | 0.0776 | 0.3014 | 0.6532 |
| 5_false_change_control | – | – | **0.00** | 0.00 | 0.0887 | 0.2000 | 0.7671 |
| 6_silent_palette_change | **1.0000** | 0.00 | 0.00 | 0.00 | 0.0776 | 0.3014 | 0.6532 |
| 7_non_anchor_transposition | **0.0000** | – | 0.00 | 0.00 | 0.0553 | **0.5963** | **0.3659** |
| 8_gradual_drift_reaching_anchor | 1.0000 | 0.96 | 0.00 | 0.00 | 0.8281 | 0.0747 | 0.5267 |
| 9_low_interaction_persistent | – | – | **0.00** | 0.00 | 0.2041 | 0.2764 | 0.6543 |
| 10_missing_appearance_zero_support | **1.0000** | 0.00 | 0.00 | 0.00 | **1.0000** | **0.0000** | – |
| 11_missing_appearance_nonzero_support | **1.0000** | 0.00 | 0.00 | 0.00 | **1.0000** | **0.0000** | – |

**R10 PASS.** Both uninformative regimes are classified MISSING_APPEARANCE on every
palette, return unresolved on every row, and produce zero confident wrong answers — the
zero-support case *and* the nonzero-support case that the magnitude rule could not reach.

**R11 PASS.** A silent change is detected on every palette whose change is detectable in
principle, with zero delay, zero corruption, and zero false alarms on **both** honest
controls — including arm 9, the deliberately quiet persistent episodes (mean support 4.8
against 18.6) that the refuted magnitude rule would have failed.

### Three things inside this pass that are not good news

**(a) The mechanism has a measured, provable blind spot.** Exchanging the SWITCH and DECOY
colours is a legal palette change that leaves BORDER, FIELD and MOVER fixed, so a
model-free signature cannot see it. Detection is 0.0000, the system answers confidently
from a memory bound to the old convention, is false-confident on **0.5963** of rows, and
its recovery accuracy is **0.3659 — below chance**. This is the *adversarial* case:
SWITCH against DECOY is the distinction the whole task turns on. Section I closes the
detection of conventions that move a behaviourally anchored role, and nothing wider.

**(b) The intrinsic ceiling is measured, not assumed.** Two independently drawn palettes
agree on all three anchors on **0.0015** of pairs; such a change is undetectable by
construction and is charged to the ceiling rather than to the mechanism.

**(c) Recalibration by truncation costs accuracy.** Recovery event accuracy: no-change
baseline **0.7671**; no detector at all **0.7237**; the contradiction detector **0.6532**.
The detector is *worse* than ignoring the change. Crucially the exact-change-point oracle
(arm 4) scores **0.6532 as well** — identical — which proves the loss is not a detection
failure but is intrinsic to recalibrating by dropping the contaminated prefix: the clean
history is 23.2 interactions instead of ~46, and a shorter clean history beats a longer
mixed one only on the *palette* question, not on the *event* question. What the mechanism
buys is not accuracy; it is never being confidently wrong about the palette.

---

## 11. Language, replicated on fresh palettes (R13 — PASS through an unqualified instrument)

`o3-language.json`. O2's own goal pipeline, repointed at id spaces no earlier phase has
touched (palettes 23000–23111, layouts 120000+/121000+, demonstrations 116500+); freshness
is verified against every spent pool in the artifact's `provenance` block and asserted in
the test suite. 756 contested keys, 12 palettes, 3 seeds.

| Contrast | Key-level (O2's unit) | Palette-level | Palettes won |
|---|---|---|---|
| correct − shuffled | +0.1182 [+0.1052, +0.1310] | **+0.1182 [+0.1102, +0.1265]** | **12/12** |
| correct − masked | +0.1197 [+0.1076, +0.1316] | **+0.1197 [+0.1118, +0.1276]** | **12/12** |

The contrast replicates in sign and significance on fresh ids and is extremely consistent
across palettes — which is why, unlike section F, the palette interval here is *narrower*
than the key interval: the between-palette variance is small because the effect is
palette-independent. The effect size is roughly **half** O2's (+0.2209 / +0.2161).

**But the instrument did not qualify.** O2's pipeline gates interpretation on Q11: the
semantic-role oracle must clear 0.80 and the grounded exact posterior 0.75.

| Ceiling | Value | Threshold |
|---|---|---|
| semantic oracle, correct language | **0.7910** | 0.80 — **misses** |
| exact posterior + goal mapping | **0.7831** | 0.75 — clears |
| semantic oracle, shuffled language | 0.4993 | (chance, as required) |
| exact posterior, no grounding | 0.5741 | (<0.60, as required) |

The grounding mechanism reproduces exactly as O2 described it — ungrounded 0.5741 near
chance, grounded 0.7831 — but the oracle sits 0.009 below its frozen bar. **The threshold
was not moved.** The ledger records R13 as `PASS_UNQUALIFIED_INSTRUMENT`: a contrast read
through a probe that has not passed its own positive control is not evidence about
language, by this track's own rule.

A reading worth stating and *not* acting on: the oracle (0.7910) and the grounded exact
posterior (0.7831) are close, which suggests the task's intrinsic ceiling on this
population is near 0.79 and that the 0.80 bar was set on a population where it was higher.
That is analysis. Lowering the bar on the strength of it would be fitting.

---

## 12. The initial-state gauge, replicated on fresh seeds (R12 PASS)

`o3-gauge.json`, run with `--seed-base 6000 --layout-offset 500`.

| Variant | Belief accuracy (up to permutation) | Displacement |
|---|---|---|
| 1_authored_public_stripe | 1.0000 | – |
| 4_outcome_trained | **1.0000** | 1.0000 |
| 5_stripe_masked | 0.6851 | 0.4927 |
| 6_reset_omitted | 0.6851 | 0.5024 |
| 7_shuffled_reset_frame | 0.6851 | 0.4908 |
| 8_false_stripe | 1.0000 | 1.0000 |

Outcome-trained matches authored exactly (+0.0000) and the three information ablations
collapse. Arm 8 scoring 1.0000 is correct, not a leak: inverting the stripe globally is a
relabelling of the two polarity states, and the metric is accuracy *up to permutation*.

---

## 13. Two configuration errors that produced false negatives

Both were found by the reserved population and both are recorded in `o3-route.json`.

**Training volume.** The first version of section J hardcoded 8 transfer layouts per
development palette where section C used 20 — 1977 training rows instead of 4973. With
everything else held fixed:

| Training rows | Validation parity | Replication parity |
|---|---|---|
| 1977 (`train_transfer=8`) | 0.8098, 7/12 above gate | 0.7453, 6/12 |
| 4973 (`train_transfer=20`) | **0.9982, 12/12** | **1.0000, 12/12** |

This first presented as "section C does not replicate on the reserved population". Section
C replicates exactly. The configuration did not match.

**Seed.** At matched training volume, route parity on unseen palettes still moves sharply
with the training seed. Training is bit-deterministic within *and across* processes —
verified by comparing digests of the training tensors, the scoring tensors and the output
logits across two separate processes, all identical — so this is genuine seed sensitivity,
not run noise. `train_memory` already selects the best of four restarts by *training
loss*, so **training loss does not predict palette generalisation and the M2F restart rule
does not control this**.

Three independent measurements of the same instability:

| Where | Quantity | Across runs |
|---|---|---|
| Section F | memory contested accuracy | 0.5656 / 0.8301 / 0.7346 |
| Section J | replication route parity | 1.0000 / 0.7079 / 0.7399 |
| Section J | replication palettes above gate | 32/32 / 13/32 / 15/32 |

And the part that makes it a methodological problem rather than a nuisance: **development
route parity is 1.0000 for all three seeds.** The saturated training population gives no
signal whatsoever about which run will generalise, so there is no in-distribution
criterion available to select on.

The consequence applies to the whole phase: **a single-seed route parity number is not a
measurement.** Section J is reported across seeds for that reason, and section C's
figures are labelled as one draw.

---

## 14. Route-parity closure through the controller (R6 FAIL)

`o3-route.json`. The closure condition is chosen on **development** palettes and applied to
the **reserved replication** palettes, because section C's validation numbers had already
been seen and a condition written afterwards would be selected on its own answer. The
validation numbers are carried forward unchanged and explicitly labelled
`OBSERVED BEFORE THE CONDITION WAS FROZEN`. Q7's target of 0.75 is unchanged.

Three training seeds, 16 development and 32 reserved replication palettes each, at matched
training volume (4973 rows).

| Seed | Development | Replication | Replication above gate |
|---|---|---|---|
| 53000 | 1.0000 | **1.0000** | **32/32** |
| 66000 | 1.0000 | 0.7079 | 13/32 |
| 88000 | 1.0000 | 0.7399 | 15/32 |

Frozen from development: fraction above gate ≥ 1.00, collapsed ≤ 0. Pooled replication:
mean 0.8160, min 0.4698, **60/96 above gate, 19 collapsed**. Seed spread **0.2921**.

**R6 FAIL**, and the shape of the failure is the finding. Every seed is *perfect* on
development — 1.0000, all palettes above gate — so development is saturated and carries no
signal at all about which seed will generalise. On the reserved population one seed of
three reproduces that and two collapse to roughly half the palettes below the gate. This is
not a property of the reserved palettes: seed 53000 scores 32/32 on them. It is training-run
instability, and it is exactly what a held-out population reserved until every decision was
frozen exists to catch.

The same decision re-run through section I's controller across all three appearance
regimes, because gating changes coverage and operational accuracy:

| Regime | Coverage | Unresolved | Accuracy given answering |
|---|---|---|---|
| PERSISTENT_CONVENTION | 0.9559 | 0.0441 | 1.0000 |
| PER_FRAME_BIJECTION | **0.0000** | **1.0000** | – |
| PER_CELL_NOISE | **0.0000** | **1.0000** | – |

Gating costs coverage only where the appearance carries no colour-to-role map. On a
persistent convention the controller answers 95.6% of rows and is right on all of them, so
abstention does not purchase its accuracy by refusing the hard cases.

---

## 15. What this closure does *not* cover

- A relabelling that moves no behaviourally anchored role is **invisible** to the section I
  detector, and the unseen case is the adversarial one (item 10a).
- Detecting a change and recalibrating by truncation **lowers** event accuracy at this
  budget (item 10c). The mechanism buys correctness about the palette, not accuracy about
  events.
- The language contrast is measured through a probe that **did not pass its positive
  control** (item 11).
- Route parity and memory generalisation to unseen palettes are **seed-unstable**, and the
  restart rule in use does not control it (item 13).
- The goal class is identified on **0 of 64** palettes; nothing here changes that.
- R5 is recorded FAIL as coded (item 6).
- **R6 FAILS on the reserved population** (item 14). Development is saturated for every
  seed and therefore predicts nothing about which run generalises.

---

## 16. Retention

Every palette, failed route, unresolved case, refuted signal and negative arm is retained
in `artifacts/shwm/scale1/o3-*.json`, including:

- both **REFUTED** change signals with the measurements that refuted them;
- both **RECLASSIFIED** orbit plants with the reason each was mis-specified;
- the confidence-margin uncertainty rule that **failed** R10, kept beside the rule that passes it;
- per-palette values for all 64 validation and all reserved replication palettes;
- per-seed values wherever a claim depends on a training run;
- the low-power section K run, superseded but retained.

`tests/shwm/test_shwm_o3.py` — 18 tests, each pinning one correction from this phase,
including three that pin mistakes this phase made and measured its way out of.
