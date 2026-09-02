# Scale 1A-0R-M2E — Neutral Transition Induction and Event-Fidelity Closure

**A generic procedure does discover the transition — 19 of 20 untouched seeds learn the
XOR automaton exactly, from random orientations, selected on training likelihood alone.
V3 still fails, by 0.0044, on a margin frozen before validation.** The one seed that
fails announces its own failure in the selection signal, and the blocker beyond it is
event fidelity, now quantified rather than described.

---

## 0. Corrected canonical ledger

| claim | corrected status |
|---|---|
| C2 generic symmetry breaking | **FAIL** |
| U3 learned transition | **RETRACTED / NOT SATISFIED** |
| M2D validation coupling | MEASURED, for a **transition-initialized** mechanism — not for a learned transition |
| U7 genuinely learned event + learned transition | **NOT ESTABLISHED** |
| held-out-layout coupling | approximately chance |
| visual ladder | **BLOCKED** |
| Stage 1A-1 matrix | **BLOCKED** |

Permitted label until a neutral arm passes: **TRANSITION-INITIALIZED EVENT-FILTERING
DIAGNOSTIC**. The forbidden label — "learned belief-transition model" — is recorded in
`m2e-gates.json:canonical_corrections` so the ledger is data, not prose.

---

## 1. Provenance

| field | value |
|---|---|
| commit | `b030828` (M2D) — M2E work is uncommitted at time of measurement |
| branch | `phase-2-continuous-world-model` |
| Phase-2 suite | **489 passed, 0 skipped, 38.7 s** (476 before this phase, plus 13 M2E pins) |
| repository suite | **963 passed, 2 failed, 956.4 s** — both failures are `tests/test_planner.py`, `ModuleNotFoundError: No module named 'arc_agi'`, an optional Phase-1 dependency absent from this venv; two further modules fail to *collect* for the same reason |
| seeds | development 13000–13009, validation **14000–14019** — disjoint from every M2C/M2D seed (6600–6602, 7000–7019, 8000–8019, 9000–9009) |

Populations, each with layouts, per-layout row counts and a member digest — and the full
member table written to the npz beside it:

| population | layouts | rows | pairs | classes | member digest |
|---|---|---:|---:|---:|---|
| train episodes | 61000–61039 | 113 episodes | — | — | `2aca2b87e9d112ca` |
| development alias | 91000–91009 | 31 520 | 3 993 | 678 | `9f2bf7eed2ba27e3` |
| validation alias | 90000–90009 | 19 920 | 2 726 | 421 | `5c943da9fabcbbc9` |
| held-out alias | 95000–95009 | 32 032 | 4 093 | 695 | `42b7ebb7768582de` |
| held-out alias 2 | 92000–92009 | 28 044 | 3 605 | 555 | `8f07417904f4c129` |

Frozen predictions: `m2e-transition-predictions.npz` and `m2e-coupling-predictions.npz`,
each carrying per-arm hit matrices, row strata and the member tables.

---

## 2–3. Generic initialisation audit (V2 — PASS)

Exact initial transitions, as constructed:

| source | event 0 | event 1 | d(identity) | d(swap) | d(answer) |
|---|---|---|---:|---:|---:|
| M2D answer-oriented | [0.731, 0.269] / [0.269, 0.731] | [0.269, 0.731] / [0.731, 0.269] | 0.9242 | 0.9242 | 0.0000 |
| generic seed 13000 | — | — | 0.7576 | 0.7576 | 0.2875 |
| generic seed 13001 | [0.284, 0.716] / [0.604, 0.396] | [0.716, 0.284] / [0.396, 0.604] | 0.8101 | 0.8101 | 1.1164 |
| generic seed 13002 | [0.693, 0.307] / [0.612, 0.388] | [0.307, 0.693] / [0.388, 0.612] | 0.7917 | 0.7917 | 0.6904 |

Over 400 draws: mean distance to the answer orientation **0.7201**, mean distance between
two draws **0.5795**. The family is *farther* from the answer than its members are from
each other, which is the property §3 requires. Mean stay-minus-flip diagonal **+0.0317**
(the answer's is **+0.9242**); 53.5 % of draws lean either way.

| check | result |
|---|---|
| 1 latent-state permutation invariance | PASS — −0.0447 [−0.1598, +0.0713] |
| 2 event-label permutation with relabelled init | PASS — +0.0249 [−0.0009, +0.0742] |
| 3 perturbation-sign invariance in expectation | PASS — −0.1188 [−0.2457, +0.0053] |
| 4 no phase labels in initialisation | PASS |
| 5 no stay/swap/XOR matrices in initialisation | PASS |
| 6 no event-name-dependent initialisation | PASS (event-axis reversal is measure-preserving to 1e-9) |
| 7 no transition target serialised before training | PASS |
| 8 no selected initialisation derived from validation | PASS — selection inputs are exactly `{arm, train, seed, updates, event_transform}` |

**One caveat that belongs in the open, not in a footnote.** A supplementary control
replacing the reset-bit→one-hot gauge with a *learned* gauge costs **−0.0750 [−0.1493,
−0.0105]**, an interval excluding zero. Mapping the public reset stripe onto a one-hot
initial belief is a modelling choice that measurably helps. M2D established that the
state labels themselves are anonymous (permuting the gauge is free), so this is not
hidden information — but it is authored structure, and it is not free.

The first version of checks 1–3 used bare tolerances on 10 seeds. With a ~40 % per-draw
success rate the standard error of the mean is near 0.05, so a "delta below 0.10" test
could not tell a real asymmetry from noise — and the sign-reversal delta passed by 0.004.
Replaced with paired intervals on 20 seeds.

---

## 4. True-event transition learning (V3 — **FAIL by 0.0044**)

K chosen on 10 development seeds by the preregistered rule (smallest K within 0.01 of the
best p10): K=2 → 0.4974, K=4 → 0.9374, **K=8 → 0.9886**, K=16 → 0.9948. **K = 8.**
Development gap to the exact accumulator 0.0053 → **frozen margin 0.0253**.

Validation, 20 untouched seeds, exact alias pairs:

| arm | alias | p10 | min | NLL | Brier | phase | occ | T-entropy | collapse | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A exact XOR accumulator | 1.0000 | 1.0000 | 1.0000 | 0.0009 | 0.0000 | 1.0000 | — | — | 0/20 | ceiling |
| B answer-oriented init | 0.9997 | 0.9985 | 0.9982 | 0.0061 | 0.0008 | 1.0000 | 1.00 | 0.144 | 0/20 | **no** |
| C zero/symmetric init | 0.5689 | 0.4977 | 0.4931 | 1.3126 | 0.3382 | 0.5748 | 1.00 | 0.440 | 0/20 | yes |
| D single generic random | 0.6717 | 0.4974 | 0.4967 | 1.0124 | 0.2588 | 0.6761 | 1.00 | 0.341 | 0/20 | yes |
| **E generic restarts (K=8)** | **0.9703** | **0.9813** | 0.5008 | 0.0930 | 0.0237 | 0.9750 | 1.00 | 0.168 | 0/20 | yes |
| F 8-state generic | 0.6514 | 0.5032 | 0.4978 | 1.0738 | 0.2734 | 0.7787 | 0.65 | 0.526 | 0/20 | yes |
| G generic GRU | 0.4981 | 0.4940 | 0.4898 | 2.6900 | 0.4468 | 0.4996 | — | — | 0/20 | yes |
| H trained memoryless | 0.5000 | 0.5000 | 0.5000 | 3.0089 | 0.4477 | — | — | — | 0/20 | yes |

Against the trained memoryless model: E **+0.4703 [+0.4114, +0.4979]**, and +0.4702 after
two or more phase changes. D +0.1717 [+0.0717, +0.2780]. F +0.1514 [+0.0837, +0.2252].
The GRU is at −0.0019 [−0.0073, +0.0034] — chance.

**Why V3 fails.** Four of the five criteria pass for E: beats memoryless, survives 2/3/4+
changes, lower tail stable, no collapse. `within_frozen_margin` fails: 1.0000 − 0.9703 =
**0.0297** against a margin of **0.0253**. A miss of 0.0044.

**What the mean is hiding, in both directions.** Per-seed, 19 of 20 sit between 0.9615
and 1.0000 with phase accuracy exactly 1.0000 and a learned stay-minus-flip diagonal of
**+1.90** — the automaton was discovered, not supplied. One seed, 14017, sits at 0.5008
with phase 0.5000 and a diagonal of +0.0811. So the mean is not measuring partial learning;
it is averaging a bimodal outcome, and a margin rule written on the mean is the wrong
instrument for that. It was nonetheless frozen before validation, so it stands: **V3 is a
FAIL**, and a p10-based rule (0.9813, gap 0.0187 < 0.0253) would have passed. Changing the
statistic after seeing the result is exactly the move these phases exist to prevent.

**The failure is self-announcing.** Seed 14017's eight restarts scored training
log-likelihood −0.0972 … −0.0745, against ≈ −0.0010 for every solved seed — a 70-fold gap
in the only signal the procedure is allowed to see. An adaptive-K rule that keeps drawing
while the training likelihood stays poor uses no privileged information and would very
likely close this. That is the thing to preregister next; it is not claimed here.

---

## 5. Restart compute ledger (V4 — PASS)

| quantity | arm E |
|---|---|
| K | 8 |
| total training runs | 160 (20 seeds × 8) |
| total optimizer updates | 163 840 |
| training-likelihood evaluations | 160 |
| selection cost | 0.11 s total (negligible against 215.5 s of training) |
| wall time | 215.5 s, against 24.7 s for a single generic run (8.7×) |
| memory | restarts run **sequentially**, so peak memory is one model: 4 369 parameters ≈ 17 KB fp32. Peak RSS was not instrumented; the sequential structure is why that is not load-bearing. |
| failed restarts by collapse | **0 of 160** |
| winning restart index | spread across 0–6: counts [6, 3, 3, 4, 3, 0, 1, 0] |

That index spread is the substantive point: in 14 of 20 seeds the winner was not the first
restart, so the restarts are doing work rather than decorating a run that would have
succeeded anyway.

Under **equal cumulative compute** (163 840 updates for every arm):

| arm | mean | p10 |
|---|---:|---:|
| **E generic restarts, K=8** | **0.9703** | 0.9813 |
| D single generic run, 8192 updates | 0.6827 | 0.4990 |
| C zero-init single run, 8192 updates | 0.5746 | 0.4968 |
| G GRU, 8 restarts | 0.4971 | 0.4922 |
| G GRU single run, 8192 updates | 0.4958 | 0.4902 |
| H memoryless, 8 restarts | 0.5000 | 0.5000 |
| H memoryless single run, 8192 updates | 0.5000 | 0.5000 |

**V4 passes.** The restart gain is not bought compute: eight times the updates in one run
buys D 0.011, and the multi-restart controls on the GRU and the memoryless model buy
nothing at all. Restarts help because the failure is a basin, and only a fresh draw leaves
a basin.

An early single-seed smoke test appeared to show the opposite — a single long run beating
restarts. That was one seed of a bimodal distribution and it did not survive twenty.

---

## 6. Event target and detector (V5, V6 — PASS)

**The derivation is published and tested**, not asserted:

```
C_t = 1  iff  position_t != position_{t-1}  and  position_t in switches_{t-1}
```

Across 225 trajectories spanning all four splits it reproduces the training label
**exactly on 225/225**, and is invariant to every field the adapter declares hidden
(`polarity`, `switch_crossings`, `step`) on **225/225**. A calibration arm that consults
`switch_crossings` is caught on 225/225, so the invariance test has power rather than
merely returning True.

Detector scope, stated explicitly: **retrospective**, `p(C_t | X_{t-1}, A_{t-1}, X_t)`.

| split | balanced | precision | recall | F1 | Brier | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| development layouts | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0021 | 0.0021 |
| validation layouts | 0.7982 | 0.7612 | 0.6623 | 0.7083 | 0.1031 | 0.4161 | 0.0916 |
| held-out layouts | 0.8110 | 0.6623 | 0.7391 | 0.6986 | 0.1302 | 0.5894 | 0.1180 |
| held-out visitation policy | 0.7633 | 0.6000 | 0.6667 | 0.6316 | 0.1566 | 0.6540 | 0.1412 |

By switch density on held-out layouts: 0.7352 / 0.8069 / 0.9008 for 0/1/2 adjacent
switches. By event count: 0.7682 (1), 0.9274 (2); the zero-event stratum reports `nan`
balanced accuracy because it contains no positives by definition.

**Held-out dynamics is not instantiable in v2 and is not claimed.** `SWITCH_COUNT` is a
constant and the flip rule never varies, so there is one transition function. What is held
out instead is the visitation policy — goal-directed rather than uniform-random — and it
is labelled as that everywhere it appears.

---

## 7. Coupling rule (V7 — PASS, with a tension worth naming)

Selected on the development alias population under the frozen joint criterion:

| rule | alias | phase-sensitive NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| **hard** | **0.6079** | 3.2178 | 0.3871 | **0.0074** |
| posterior mixture | 0.6019 | **1.8514** | **0.3439** | 0.0851 |
| particle (32 samples) | 0.5986 | 1.8824 | 0.3482 | 0.0809 |
| exact event (ceiling) | 0.9973 | 0.0093 | 0.0022 | 0.0070 |

The primary metric — phase-sensitive NLL — prefers the posterior mixture by 1.37 nats, and
Brier agrees. The calibration **constraint** excluded it: best eligible ECE is hard's
0.0074, and posterior's 0.0851 exceeds 0.0074 + 0.02. So hard is selected.

This is the opposite of what M2D's numbers suggested, and the reason is that the three
calibration measures disagree here: ECE on the outcome probability favours hard, while
Brier and NLL favour posterior. The criterion was frozen on ECE before validation and is
honoured; had it been frozen on Brier, the posterior mixture would have been selected.
Reporting that is more useful than defending the choice.

---

## 8. Event-fidelity requirement

The specification writes the independence diagnostic over phase-changing events n. That
form over-predicts measured parity badly, because a false positive on a step where nothing
happened inverts a route's parity exactly as a miss does — the exponent is the number of
**error opportunities**, which is the route length.

| population | per-step | measured parity | n-based prediction | route-length prediction |
|---|---:|---:|---:|---:|
| development 91k | 0.8626 | 0.6102 | 0.8205 | **0.6119** |
| validation 90k | 0.8389 | 0.6043 | 0.7809 | **0.5849** |
| held-out 92k | 0.8401 | 0.5889 | 0.8002 | **0.5854** |
| held-out 95k | 0.7919 | 0.5159 | 0.7600 | **0.5451** |

Mean absolute error: n-based **0.2106**, route-length **0.0135**. Both forms are reported;
the measurement adjudicates.

By event count on validation, measured parity is non-monotone — 0.7691 (n=0), 0.4830 (1),
0.6355 (2), 0.5847 (3), 0.6614 (4+) — and the route-length form predicts a flat 0.56–0.61.
So the independence model is a good aggregate diagnostic and a poor conditional one, and
the n=1 dip below chance is unexplained.

**The requirement, derived on development:** for expected route parity ≥ 0.55, per-step
accuracy must be ≥ **0.7992**. Held-out layouts sit at **0.7919** — below it. The n-based
form yields a degenerate requirement of 0.5000, because it treats a zero-event route as
automatically correct.

**Classification: EVENT EXTRACTION / GENERALIZATION BOTTLENECK.**

---

## 9. Complete learned coupling — diagnostic only, **not qualifying**

§9 is gated on a generic transition procedure passing §4. None did. These numbers were
run anyway and **cannot satisfy V8, V9 or V10**, which are recorded as NOT_RUN. They are
published because §8 predicts a held-out failure and the prediction is worth testing —
not because a gate can be met out of order. Coupling rule: hard. 20 untouched seeds.

| arm | development | validation | held-out 95k | held-out 92k |
|---|---:|---:|---:|---:|
| 1 learned event + generic learned filter | 0.6026 | 0.5942 | **0.5050** | 0.5869 |
| 2 learned event + exact accumulator | 0.6081 | 0.5999 | 0.5047 | 0.5922 |
| 3 **true** event + generic learned filter | 0.9721 | 0.9703 | **0.9720** | 0.9716 |
| 4 trained memoryless | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 5 generic GRU | 0.5008 | 0.4988 | 0.4976 | 0.5037 |
| 6 detector + no temporal state | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| 7 M2D answer-initialised filter | 0.6082 | 0.5998 | 0.5048 | 0.5921 |
| 8 exact event + exact accumulator | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Learned event + generic filter, against trained memoryless:

| split | delta | interval | 2+ changes |
|---|---:|---|---:|
| development | +0.1026 | [+0.0837, +0.1205] * | +0.1334 * |
| validation | +0.0942 | [+0.0655, +0.1241] * | +0.1289 * |
| **held-out 95k** | **+0.0050** | **[−0.0185, +0.0295]** | +0.0236 |
| held-out 92k | +0.0869 | [+0.0681, +0.1052] * | +0.0362 * |

**§8's prediction is confirmed.** The single population whose detector fidelity falls
below the derived requirement is the single population where the coupling fails, and row 3
is the control that assigns blame: with **true** events the same generic filter scores
0.9720 on held-out 95k — identical to its validation score. The filter transfers perfectly.
Only event extraction does not.

Row 7 also matters for the ledger: the M2D answer-initialised filter and the generic filter
are within 0.006 of each other on every split once coupled, so nothing in M2D's coupling
numbers depended on the initialisation being the answer — the initialisation mattered for
*learning* the transition, not for *using* it.

---

## 10. Event corruption (V11 — PASS)

All derived from one frozen probability sequence, on validation:

| control | alias | delta vs memoryless | interval |
|---|---:|---:|---|
| 1 correct learned probabilities | 0.5942 | +0.0942 | [+0.0655, +0.1241] * |
| 2 one-step forward shift | 0.4974 | −0.0026 | [−0.0287, +0.0217] |
| 3 one-step backward shift | 0.5160 | +0.0160 | [−0.0027, +0.0342] |
| 4 one true event dropped | 0.4446 | −0.0554 | [−0.0819, −0.0316] * |
| 5 one event flipped | 0.4668 | −0.0332 | [−0.0461, −0.0200] * |
| 6 cross-episode matched-prevalence shuffle | 0.4933 | −0.0067 | [−0.0128, −0.0007] * |
| 7 independent example permutation | 0.5027 | +0.0027 | [−0.0032, +0.0082] |
| 8 constant probability | 0.5000 | +0.0000 | [−0.0002, +0.0002] |
| 9 calibrated random detector | 0.5009 | +0.0009 | [−0.0056, +0.0076] |
| 10 action-only detector | 0.4731 | −0.0269 | [−0.0429, −0.0113] * |
| 11 state-only detector | 0.5475 | +0.0475 | [+0.0251, +0.0716] * |

Every misalignment and information-destroying control removes the advantage; dropping or
flipping a single event drives the arm below chance, which is the signature of a real
parity mechanism. The calibrated random detector — right marginal, no information — lands
at +0.0009, which is the cleanest of the set. The state-only detector retains +0.0475
because it retains real information (the neighbour-switch bits); it is a weaker detector,
not a corrupted event, and it is reported separately for that reason.

---

## 11. Prospective boundary

**Demonstrated:** `p(C_t | X_{t-1}, A_{t-1}, X_t)` and the belief update *after* observing
X_t.

**Not demonstrated:** `p(C_{t+1} | B_t, X_t, A_t)` before executing A_t.

M2E is not a planning world model and no prospective predictor was trained. If the
retrospective path is closed, the next phase must build a prospective event/outcome
predictor before any closed-loop planning claim.

---

## 12. Gates

| gate | status | basis |
|---|---|---|
| V0 | PASS | 6 population manifests with layouts, per-layout counts and member digests; Phase-2 489 in 38.7 s |
| V1 | PASS | canonical corrections recorded as data in `m2e-gates.json` |
| V2 | PASS | all eight genericity checks, three of them with paired intervals |
| **V3** | **FAIL** | E misses the frozen margin by 0.0044 (gap 0.0297 vs 0.0253) |
| V4 | PASS | 0.9703 vs 0.6827 for the best equal-cumulative-compute alternative |
| V5 | PASS | derivation exact on 225/225 and hidden-field invariant on 225/225; leaky arm caught on 225/225 |
| V6 | PASS | held-out layouts 0.8110, visitation-policy shift 0.7633 |
| V7 | PASS | hard selected under the frozen joint criterion |
| V8 | NOT_RUN | blocked by V3 — measured value +0.0942 [+0.0655, +0.1241], non-qualifying |
| V9 | NOT_RUN | blocked by V3 — measured value +0.1289, non-qualifying |
| V10 | NOT_RUN | blocked by V3 — measured value +0.0050 [−0.0185, +0.0295], non-qualifying |
| V11 | PASS | every misalignment and information-destroying control removes the advantage |
| V12 | PASS | per-seed records, per-restart likelihoods, collapse counts, frozen predictions |

**9 PASS, 1 FAIL, 3 NOT_RUN.**

Decision per §12: **V3 fails, so transition induction remains blocked.** No model size was
increased at any point; every arm holds 4 369 parameters against a 250 000 ceiling.

---

## 13. Bugs and corrections

1. **The independence exponent was wrong.** Using phase-changing events n gave a
   degenerate requirement (p_min = 0.5000) and over-predicted parity by 0.21 on average,
   because a route with no events is not automatically correct — a false positive inverts
   it. Route length is the right exponent (MAE 0.0135). Both forms are now reported.
2. **`np.clip(x_float32, 1e-9, 1 - 1e-9)` does not clip.** `1 - 1e-9` rounds to exactly
   1.0 in float32, so a confident correct prediction produced log(0) and the development
   NLL was reported as `nan`. Only the `nan` made it visible. Cast to float64 before
   clipping; pinned by a test.
3. **The genericity invariance checks were underpowered.** Bare tolerances on 10 seeds
   against a sampling error near 0.05; the sign-reversal check passed by 0.004. Replaced
   with paired intervals on 20 seeds.
4. **A name-based test that a docstring could fail.** The "selection sees no validation"
   test scanned source text and failed because the docstring contains the word
   "validation". Rewritten over the syntax tree.
5. **`np.math.comb` does not exist in numpy 2.** Caught by the test that used it.
6. **The coupling first retrained per split**, which cost four times as much and would
   have let arms drift between splits for no reason but the RNG. Models are now trained
   once per (arm, seed) and scored everywhere.
7. **A single-seed smoke test suggested a long single run beats restarts.** It did not
   survive twenty seeds (0.6827 vs 0.9703); it was one draw from a bimodal distribution.
8. **`ComputeLedger.selected_rank` is the winning restart's *index*, not its rank** — a
   winner's rank is always 0 and would carry nothing. The field is documented in code and
   described accurately here rather than renamed, which would have required a 33-minute
   re-run to keep the artifact consistent with the code.

---

## 14. The narrow supported claim

With **true** public events supplied, a fixed-K restart procedure over generic,
orientation-free, seed-derived initialisations — selected on training outcome likelihood
alone, with no phase label, no validation score and no transition target — learns the
environmental transition on **19 of 20 untouched validation seeds**, reaching alias-pair
accuracy 0.9703 (p10 0.9813) against a memoryless baseline pinned at exactly 0.5000, and
recovering the XOR automaton exactly (phase accuracy 1.0000, stay-minus-flip diagonal
+1.90). The gain survives equal cumulative-compute accounting against every control.

**This does not clear V3**, which required the validation mean to sit within a
development-frozen 0.0253 of the exact accumulator; it sits 0.0297 away because one seed
of twenty fails outright.

With **learned** events the pathway reaches +0.0942 [+0.0655, +0.1241] on validation and
+0.0050 [−0.0185, +0.0295] on held-out layouts — that is, nothing — while the same filter
with true events scores 0.9720 on those same held-out layouts.

The permitted label is unchanged: **TRANSITION-INITIALIZED EVENT-FILTERING DIAGNOSTIC**.
It is not autonomous hidden-state discovery — the event target and the binary
factorisation are authored — and it is not a planning world model.

---

## 15. Is visual event extraction unblocked?

**No**, and for a sharper reason than last phase.

1. **V3 fails by 0.0044.** The fix is a preregistered adaptive-K rule: seed 14017's eight
   restarts all scored training log-likelihood ≈ −0.08 against ≈ −0.001 for solved seeds,
   so the procedure can tell it has failed using only what it is allowed to see.
2. **Event extraction is the quantified bottleneck.** Per-step accuracy must reach 0.7992
   for the coupling to retain an advantage; held-out layouts deliver 0.7919. The gap is
   0.007 in per-step accuracy and it costs the entire effect.

The second is the one that matters for vision. A visual front end can only lower per-step
event accuracy relative to the structured features used here, and the structured detector
is already 0.007 below the threshold on held-out layouts. Adding an encoder now would
measure the encoder against a floor that has already given way.
