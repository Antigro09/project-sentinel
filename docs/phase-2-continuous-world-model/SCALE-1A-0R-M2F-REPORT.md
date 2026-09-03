# Scale 1A-0R-M2F — Certified Generic Transition Induction and Event-Fidelity Closure

**Both research questions come back yes.** A training-only fit certificate turns rare
catastrophic transition failures into extra restarts at almost no cost, and a
translation-equivariant public event detector closes the route-fidelity gap outright.
All ten F gates and the E gates pass, on a hundred untouched validation seeds.

The M2E ledger is corrected, not reinterpreted: **M2E V3 remains a FAIL.**

---

## 1. Provenance

| field | value |
|---|---|
| commit | `3abcf98022c8fd6c9171347f4caccff5689629a4` (M2E); M2F uncommitted at measurement |
| branch | `phase-2-continuous-world-model` |
| Phase-2 suite | **500 passed, 40.7 s** (`tests/shwm`; 489 before, plus 11 M2F pins) |
| required suite | **974 passed, 4 skipped, 0 failed, 980.4 s** (963 before the 11 M2F pins) |
| optional-dependency tests | 2 marked `optional_dependency`, 2 modules skipped at import |
| visual or final Scale-1 seed opened | **no** |

**The two `arc_agi` failures are gone, and §B is the reason.** `arc_agi` is an optional
Phase-1 dependency. Before this phase two modules failed to *collect* and two planner
tests *failed* when it was absent, so a clean checkout reported "963 passed, 2 failed"
and the failures had to be re-explained in prose every time. `tests/conftest.py` now
registers an `optional_dependency` marker and skips on `importlib.util.find_spec`; the
two dependent modules call `importorskip` and the two dependent tests carry the marker.
Manifests: `-m "not optional_dependency"` selects the required tests; `-m optional_dependency` selects the 2 that need the package.

Seeds, all disjoint from M2C (6600–6602), M2D (7000–7019, 8000–8019, 9000–9009) and
M2E (13000–13019, 14000–14019):

| role | range |
|---|---|
| transition development | **21000–21099** (100) |
| transition validation | **22000–22099** (100, untouched until §F was frozen) |

Populations, each with layouts, per-layout counts and a member digest, with the full
member table in the npz beside it:

| population | layouts | rows | pairs | classes | digest |
|---|---|---:|---:|---:|---|
| train episodes | 61000–61039 | 113 episodes | — | — | `2aca2b87e9d112ca` |
| development alias | 91000–91009 | 31 520 | 3 993 | 678 | `9f2bf7eed2ba27e3` |
| validation alias | 90000–90009 | 19 920 | 2 726 | 421 | `5c943da9fabcbbc9` |
| held-out alias | 95000–95009 | 32 032 | 4 093 | 695 | `42b7ebb7768582de` |
| held-out alias (new) | 97000–97009 | — | — | — | recorded in `m2f-pathway.json` |

Restart tables: 3 200 development rows and 3 200 validation rows, every one retained,
with digests in `m2f-procedures.json`. Frozen predictions in
`m2f-restarts-development.npz` and `m2f-restarts-validation.npz`.

---

## 2. Corrected M2E ledger

| item | status |
|---|---|
| **V3 (M2E)** | **FAIL** — validation mean gap 0.0297 against a development-frozen 0.0253 |
| generic transition discoveries (M2E) | 19/20 |
| visual ladder, at the close of M2E | BLOCKED |
| complete learned pathway, at the close of M2E | NOT ESTABLISHED |

The p10 gap of 0.0187 would have passed and is recorded beside the mean, but the
criterion was frozen on the mean and **is not substituted**. Nothing in M2F converts
M2E's V3 retroactively; M2F is a separate experiment on separate seeds.

---

## 3. Why M2E seed 14017 failed (§C — diagnostic only)

Run out to K=64, every restart retained. This section may not contribute to any M2F
threshold and is written to its own artifact.

- **Restarts 0–7 reproduce the M2E record exactly.** That is the positive control; without
  it the diagnosis would be of some other computation.
- 13 of 64 restarts solve, the first at restart 13. Training log-likelihood −0.0013 for
  solved restarts against −0.075…−0.107 for failures.
- Control seed 14003 solved at restarts 1 and 3.
- The fit curve is **monotone** → not optimizer instability.
- **No intermediate checkpoint beats the final** → not checkpoint selection.
- The seed drives initialisation and minibatch order and nothing else; the trajectories,
  alias population and event channel are byte-identical across seeds, and their digests
  are recorded → a per-seed data or identifiability defect is not something this design
  can produce.

**Diagnosis: unlucky initialisation and insufficient restart count.**

---

## 4. Certified adaptive restarts (§D–§G — F0–F9 all PASS)

### The certificate separates, on development, without phase labels

Over 3 200 development restarts:

```
worst solved restart      log-likelihood  -0.009535
best unsolved restart     log-likelihood  -0.011845
gap                                        0.002310      -> tau = -0.010690
```

No overlap. `certify()` takes exactly `(best_training_log_likelihood, tau)`; there is no
phase accuracy, no transition matrix, no validation score and no event label in scope,
and a test pins the signature.

### Validation, 100 untouched seeds

| procedure | alias | p10 | min | catastrophic | coverage | false cert. | unresolved | fallback | mean K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed K=8 | 0.9799 | 0.9817 | 0.4957 | **0.030** | 0.970 | 0.0000 | 0.030 | 0.9799 | 8.0 |
| fixed K=16 | 0.9979 | 0.9937 | 0.9838 | 0.000 | 1.000 | 0.0000 | 0.000 | 0.9979 | 16.0 |
| fixed K=32 | 0.9989 | 0.9968 | 0.9904 | 0.000 | 1.000 | 0.0000 | 0.000 | 0.9989 | 32.0 |
| **adaptive** | **0.9948** | 0.9843 | 0.9674 | **0.000** | **1.000** | **0.0000** | 0.000 | 0.9948 | **8.24** |
| single long run, 9216 updates | 0.6807 | 0.4984 | — | — | — | — | — | — | 1 |
| GRU multistart, 9 restarts | 0.4975 | 0.4919 | — | — | — | — | — | — | 9 |
| trained memoryless | 0.5000 | 0.5000 | — | — | — | — | — | — | 1 |
| exact accumulator (ceiling) | 1.0000 | 1.0000 | — | — | — | — | — | — | 1 |
| answer-oriented (**ineligible**) | 0.9997 | 0.9990 | — | — | — | — | — | — | 1 |

Against the trained memoryless model, which is pinned at exactly 0.5000 by construction:

| procedure | delta | interval |
|---|---:|---|
| fixed K=8 | +0.4799 | [+0.4600, +0.4949] * |
| adaptive | +0.4948 | [+0.4930, +0.4964] * |
| fixed K=32 | +0.4989 | [+0.4984, +0.4992] * |
| single long run | +0.1807 | [+0.1333, +0.2301] * |
| GRU multistart | −0.0025 | [−0.0050, −0.0001] |

### Compute accounting

| quantity | adaptive | fixed K=8 | fixed K=32 |
|---|---:|---:|---:|
| mean restarts per seed | 8.24 | 8.0 | 32.0 |
| max restarts | 16 | 8 | 32 |
| total optimizer updates (100 seeds) | 843 776 | 819 200 | 3 276 800 |
| catastrophic seeds | **0** | 3 | 0 |

The adaptive rule spends **3 % more compute than fixed K=8 and eliminates all three of
its catastrophic collapses**, because it can tell which seeds have not fitted and only
those get more draws. It reaches fixed-K=32's reliability at a quarter of K=32's compute.

Under **equal cumulative compute** the restart structure is what matters, not the
budget: a single run given the same 9 216 updates reaches 0.6807, and a GRU given nine
restarts reaches 0.4975. Adaptive beats the long run by **+0.3141 [+0.2648, +0.3610]**
and the GRU by **+0.4973 [+0.4944, +0.5002]**.

### F gates

| gate | status | basis |
|---|---|---|
| F0 | PASS | M2E V2 genericity, plus a check that no M2F generic draw equals the answer initialisation |
| F1 | PASS | development separation gap 0.00231, no overlap |
| F2 | PASS | validation false certification **0.0000** vs frozen bound 0.0200 |
| F3 | PASS | adaptive catastrophic rate 0.0000 vs fixed K=8 at 0.0300 |
| F4 | PASS | +0.3141 vs the long run; +0.4973 vs the GRU multistart |
| F5 | PASS | certified-only +0.4948 [+0.4930, +0.4964] |
| F6 | PASS | coverage 1.0000 ≥ 0.9500; overall 0.9948 ≥ 0.9468 |
| F7 | PASS | 2 changes +0.4944; 3 changes +0.4913; 4+ changes +0.4980, all excluding zero |
| F8 | PASS | the answer-oriented arm is `eligible=False` in the `Arm` record, not in prose |
| F9 | PASS | 3 200 + 3 200 restart rows retained, none dropped |

**F0–F9 all pass.**

---

## 5. Initial-state gauge audit (§H)

20 seeds, 8 generic restarts per variant, true events:

| variant | alias | p10 | min | phase | solved | vs authored |
|---|---:|---:|---:|---:|---:|---|
| 1 authored public one-hot | 0.9950 | 0.9861 | 0.9774 | 1.0000 | 20/20 | — |
| 2 learned from outcome only | 0.9034 | 0.5353 | 0.5003 | 0.9314 | 17/20 | **−0.0916 [−0.1684, −0.0257]** |
| 3 phase-supervised oracle *(ineligible)* | 0.9950 | 0.9861 | 0.9774 | 1.0000 | 20/20 | **+0.0000 [+0.0000, +0.0000]** |
| 4 reset stripe masked | 0.6235 | 0.6217 | 0.6199 | 0.6237 | 0/20 | −0.3715 |
| 5 false reset stripe | **0.0074** | 0.0001 | 0.0000 | 0.0000 | 0/20 | −0.9876 |
| 6 random gauge | 0.4938 | 0.4934 | 0.4928 | 0.4938 | 0/20 | −0.5012 |

Two findings, and they point in opposite directions, which is why the arms are kept
separate:

- **The stripe is genuinely public.** Arm 3 is bit-identical to arm 1 — a gauge built
  from the rendered stripe and one built from the evaluator's initial polarity produce
  the same model, which is the demonstration that the renderer draws exactly that number.
  Evaluator phase is not counted as public grounding anywhere; it appears only to
  establish this equivalence.
- **The gauge is not free.** An outcome-trained learned encoder differs by −0.0916 with
  an interval excluding zero.

So the transition result is reported as **conditional on authored initial-state
grounding**. Arm 5 is the sharpest evidence that the belief is load-bearing rather than
decorative: falsifying one public bit drives the model to 0.0074, systematically
*anti*-correct.

---

## 6. Event detector family (§I, §J)

Preregistered, bounded, fixed before validation exposure. Per-step and route-level:

| detector | split | balanced | accuracy | Brier | NLL | ECE | exact-seq | parity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 current structured | development | 1.0000 | 1.0000 | 0.0000 | 0.0021 | 0.0021 | 1.0000 | 1.0000 |
| 1 current structured | validation | 0.8512 | 0.8783 | 0.0947 | 0.3950 | 0.0872 | 0.4211 | 0.6579 |
| 1 current structured | held-out | 0.8317 | 0.8508 | 0.1257 | 0.5366 | 0.1211 | 0.3250 | **0.5250** |
| 1 current structured | policy shift | 0.8069 | 0.8439 | 0.1143 | 0.4235 | 0.1213 | 0.3243 | 0.5135 |
| 2 action-only | held-out | 0.4708 | 0.4698 | 0.2685 | 0.7307 | 0.2357 | 0.0000 | 0.4250 |
| 3 state-pair only | held-out | 0.5983 | 0.6540 | 0.2750 | 1.2478 | 0.2505 | 0.0750 | 0.5750 |
| **4 relational equivariant** | development | **1.0000** | 1.0000 | 0.0000 | 0.0003 | 0.0003 | **1.0000** | **1.0000** |
| **4 relational equivariant** | validation | **1.0000** | 1.0000 | 0.0000 | 0.0004 | 0.0004 | **1.0000** | **1.0000** |
| **4 relational equivariant** | **held-out** | **1.0000** | 1.0000 | 0.0000 | 0.0004 | 0.0004 | **1.0000** | **1.0000** |
| **4 relational equivariant** | **policy shift** | **1.0000** | 1.0000 | 0.0000 | 0.0004 | 0.0004 | **1.0000** | **1.0000** |
| 5 calibrated structured | held-out | 0.8317 | 0.8508 | 0.1010 | 0.3091 | 0.0608 | 0.3250 | 0.5250 |
| 6 exact public derivation (ceiling) | all | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

**The bottleneck was absolute position.** The current detector reads position and goal
direction, so it memorises training layouts — 1.0000 there, 0.83 elsewhere, with route
parity collapsing to 0.5250 because parity is the product of per-step accuracies. The
relational detector sees only the displacement taken, the switch bits of the neighbours
of the cell just left, the previous action and the blocked flag. It is
translation-equivariant by construction and reaches the exact-derivation ceiling on every
split, including layouts and a visitation policy it never saw.

`test_relational_detector_is_translation_equivariant` pins this **with a calibration
arm**: the same test applied to the current structured detector must fail, or it is
measuring nothing.

Structured detector, held-out sequence structure: mean error burst 0.78, max 3, lag-1
error autocorrelation −0.068, and **one layout at a 100 % parity failure rate**. The
non-identical-error diagnostic `[1 + Πₜ(1 − 2eₜ)]/2` predicts 0.7707 against a measured
0.5250 there — it over-predicts because errors are layout-clustered rather than
independent. Balanced accuracy is never plugged into that formula; a test compares the
non-identical form against simulation and against the mean-error form.

Calibration is the one place the structured detector improves: temperature scaling cuts
its held-out NLL from 0.5366 to 0.3091 and ECE from 0.1211 to 0.0608 without moving a
single decision, so its accuracy and parity are unchanged.

---

## 7. Coupling rule (§K)

Selected on the development alias population under a **proper score**, not ECE:

| rule | alias | phase-sensitive NLL | Brier |
|---|---:|---:|---:|
| hard | 0.9972 | 0.0096 | 0.0023 |
| posterior mixture | 0.9971 | 0.0097 | 0.0024 |
| particle (32) | 0.9971 | 0.0097 | 0.0024 |
| exact (ceiling) | 0.9972 | 0.0096 | 0.0023 |

`hard` is selected — but the honest reading is that **the criterion no longer
discriminates**. M2E's tension between hard and posterior existed because the detector
was uncertain; with an exact detector all three rules coincide to within 0.0001 on every
measure. The old choice was not preserved for its own sake: the constraint was rebuilt on
Brier and the selection redone from scratch, and it happened to land in the same place
for a completely different reason.

---

## 8. Complete structured learned pathway (§L)

20 seeds, certified transition models as the frozen adaptive procedure selected them:

| arm | validation | held-out 95k | held-out 97k (new) |
|---|---:|---:|---:|
| 1 exact event + exact accumulator | 1.0000 | 1.0000 | 1.0000 |
| 2 exact event + certified transition | 0.9950 | 0.9971 | 0.9945 |
| 3 learned event + exact accumulator | 1.0000 | 1.0000 | 1.0000 |
| **4 learned event + certified transition** | **0.9950** | **0.9971** | **0.9945** |
| 5 learned event + GRU | 0.4972 | 0.4958 | 0.4957 |
| 6 learned event + no temporal state | 0.5000 | 0.5000 | 0.5000 |
| 7 trained memoryless | 0.5000 | 0.5000 | 0.5000 |
| 8 answer-oriented *(diagnostic)* | 0.9994 | 0.9996 | 0.9991 |

| split | delta vs memoryless | interval | 2+ changes |
|---|---:|---|---:|
| validation | **+0.4950** | [+0.4910, +0.4979] * | +0.4938 * |
| held-out 95k | **+0.4971** | [+0.4947, +0.4989] * | +0.4952 * |
| held-out 97k | **+0.4945** | [+0.4904, +0.4976] * | +0.4965 * |

Arms 2 and 4 are identical, and arms 1 and 3 are identical, **because the relational
detector is exact** — its output equals the true event sequence on these populations.
That is a consequence of the §I result, not a wiring collapse; arm 5 and arm 6 are driven
by the same detector and sit at chance.

The answer-oriented diagnostic (0.9994) and the certified generic model (0.9950) are
within 0.005 of each other, so the authored orientation buys essentially nothing once the
transition has been learned — it mattered for *finding* the automaton, never for using it.

---

## 9. Event corruption (§M / E1)

From one frozen probability sequence, on validation:

| control | alias | delta | interval |
|---|---:|---:|---|
| 1 correct | 0.9950 | +0.4950 | [+0.4910, +0.4979] * |
| 2 shifted one forward | 0.5813 | +0.0813 | [+0.0380, +0.1219] * |
| 3 shifted one backward | 0.6034 | +0.1034 | [+0.0511, +0.1515] * |
| 4 one true event dropped | 0.1561 | −0.3439 | [−0.3680, −0.3202] * |
| 5 one event flipped | 0.3420 | −0.1580 | [−0.1722, −0.1437] * |
| 6 cross-episode shuffle | 0.4959 | −0.0041 | [−0.0097, +0.0013] |
| 7 position-wise permutation | 0.5007 | +0.0007 | [−0.0047, +0.0061] |
| 8 constant | 0.5000 | +0.0000 | [−0.0001, +0.0002] |
| 9 calibrated random | 0.5025 | +0.0025 | [−0.0033, +0.0082] |
| 10 action-only detector | 0.4716 | −0.0284 | [−0.0441, −0.0120] * |
| 11 state-only detector | 0.5505 | +0.0505 | [+0.0276, +0.0743] * |

Dropping or flipping a single event drives the arm far below chance — with an exact
detector the parity mechanism is fully exposed, so one corrupted event inverts the phase
for the remainder of the route. The shifts retain +0.08/+0.10 for a mechanical reason
worth stating: shifting a sequence by one step only changes its parity when an event
falls off the end, so most routes keep the right parity.

---

## 10. E gates

| gate | status | basis |
|---|---|---|
| E0 | PASS | the public derivation is exact on 225/225 trajectories and invariant to every declared hidden field, with a leaky calibration arm caught on 225/225 (M2E §6, carried); the relational detector's inputs lie inside `{X_{t-1}, A_{t-1}, X_t}` |
| E1 | PASS | correct events beat action-only, state-only, shuffled, permuted, constant and calibrated-random controls |
| E2 | PASS | the requirement of 0.7992 per-step is carried from M2E §8, derived on its development population; the relational detector delivers 1.0000, and route parity 1.0000 against a 0.55 target |
| E3 | PASS | validation +0.4950 [+0.4910, +0.4979] |
| E4 | PASS | held-out 95k +0.4971 and held-out 97k +0.4945, both excluding zero |
| **E5** | **NOT INSTANTIABLE** | v2 has one transition function — `SWITCH_COUNT` is constant and the flip rule never varies. The nearest available split is the visitation policy, where the detector scores 1.0000; **this is not claimed as a held-out-dynamics pass** |
| E6 | PASS | 2+ changes +0.4938 on validation, +0.4952 and +0.4965 on the held-out sets |
| E7 | PASS | selected under a Brier constraint, rebuilt from scratch rather than inherited |
| E8 | PASS | every seed, restart, collapse and failed layout retained; 6 400 restart rows and frozen per-row predictions |

---

## 11. Bugs and corrections

1. **The full suite's two failures were an unclassified optional dependency.** Now an
   explicit marker plus `importorskip`, with required and optional manifests. 963 passed,
   4 skipped, 0 failed.
2. **The calibrated-random control was permuting the flattened event array**, which moved
   events into padding and into step 0 — positions the honest detector never fills. It
   showed +0.0106 with an interval excluding zero, an artefact of the control. Rebuilt to
   preserve the structural zeros and the marginal; it now sits at +0.0025 [−0.0033,
   +0.0082].
3. **`ComputeLedger.selected_rank` holds the winning restart's index**, not its rank; a
   winner's rank is always 0 and carries nothing. Documented in code rather than renamed,
   which would have desynchronised the M2E artifact from its source.
4. The validation restart pass initially scored every restart on two held-out alias
   populations. Only the ~4 selectable models per seed ever reach the pathway stage, so
   this was removed and the pathway module scores those directly — the same numbers for a
   third of the time.

---

## 12. The narrow supported claim

On exact complete-public-packet alias pairs, where a trained memoryless model is pinned
at exactly 0.5000 by construction:

- A **fixed-K=8-plus-adaptive-blocks restart procedure over generic, orientation-free,
  seed-derived initialisations**, selecting on training outcome likelihood alone and
  certifying on a development-frozen threshold, learns the environmental transition on
  **100 of 100 untouched validation seeds** (0.9948, p10 0.9843, zero catastrophic
  failures, zero false certifications) at 8.24 restarts per seed.
- A **translation-equivariant public retrospective event detector** reaches the exact
  derivation ceiling — per-step 1.0000, route parity 1.0000 — on held-out layouts and
  under a visitation-policy shift.
- The complete pathway, **learned event + certified generic learned transition**, beats
  trained memoryless by **+0.4950 [+0.4910, +0.4979]** on validation, **+0.4971** and
  **+0.4945** on two held-out layout sets, and survives two or more phase changes on all
  three.

Qualified, per §M, as a **SUPERVISED RETROSPECTIVE EVENT-FACTORIZED GENERICALLY LEARNED
BELIEF MODEL**, with three standing conditions:

- **conditional on authored initial-state grounding** — the reset-stripe gauge is public
  but not free (−0.0916 without it);
- the event *target* and the binary factorisation remain **authored**, so this is not
  autonomous event-ontology discovery;
- it is **retrospective**. Demonstrated: `p(C_t | X_{t-1}, A_{t-1}, X_t)` and the belief
  update after observing X_t. Not demonstrated: `p(C_{t+1} | B_t, X_t, A_t)` before
  executing A_t. **M2F is not a planning world model.**

---

## 13. Is the visual event-extraction ladder unblocked?

**Yes**, for the first time in this line of work, and with a specific target rather than
a hope.

The two blockers M2E named are closed. Transition induction is reliable and self-auditing
— zero catastrophic failures and zero false certifications across 100 untouched seeds,
with a certificate that declares UNRESOLVED instead of acting on a transition it has not
fitted. Event extraction is no longer the bottleneck: the relational detector reaches the
exact-derivation ceiling on held-out layouts.

What the visual phase must now reproduce is precise. The relational detector needs three
quantities from pixels: the agent's displacement between consecutive frames, and the
switch status of the cell entered, read from the frame *before* entry because the renderer
paints the agent over the switch beneath it. Per-step accuracy must clear **0.7992** for
the coupling to retain any advantage at all, and route parity is the product of per-step
accuracies — so the visual target is not "good event detection" but a per-step error rate
low enough to survive six multiplications.

The next phase must also build a **prospective** predictor before any closed-loop planning
claim; nothing here licenses one.
