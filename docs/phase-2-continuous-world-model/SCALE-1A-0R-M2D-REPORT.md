# Scale 1A-0R-M2D — Learned Event/Belief Coupling Closure

**The U7 pass is withdrawn and re-earned, and the U3 pass does not survive the audit
that §C asked for.** Both changes come from the same discipline: read what the code did,
not what the report said.

---

## 1. Commit and provenance

| field | value |
|---|---|
| commit | `85eb314e80f513a6abc414d955b68de9fb768f71` |
| branch | `phase-2-continuous-world-model` |
| tracked modified | `.claude/worktrees/x35-novelty-trigger` only |
| untracked | 7 new `experiments/shwm/m2d_*.py`, 1 new `tests/shwm/test_shwm_m2d.py` |
| Phase-2 suite | **476 passed, 0 skipped, 37.3 s** (`tests/shwm`; 460 before this phase, plus 16 new M2D pins) |
| repository suite | **934 passed, 2 failed, 1057.0 s** |
| the 2 failures | `tests/test_planner.py` — `ModuleNotFoundError: No module named 'arc_agi'`, an optional Phase-1 dependency absent from this venv. Two further modules (`test_env_determinism`, `test_verifier`) fail to *collect* for the same reason. Not caused by this work, and not fixed by it. |

Seeds: symmetry and filters use validation seeds 8000–8019; the coupling uses development
seeds 7000–7004 for the hard/posterior choice and validation seeds 8000–8019 for every
reported number. Alias layouts: development 91000–91009, validation 90000–90009,
held-out 95000–95009 and 92000–92009.

**Saved predictions now exist for every reported arm.** `m2d-coupling-predictions.npz`
(`c737e67c34496e45`) holds per-arm hit matrices of shape (20 seeds × 19920 rows) plus
row strata and both event arrays; `m2d-filters-predictions.npz` holds the same for §D.
Recomputing U7 from the frozen file alone reproduces the artifact exactly — accuracy
0.599869, interval +0.0999 [+0.0710, +0.1287], and the 2+-changes stratum +0.1371 — with
no retraining. That is the property the M2C artifact did not have.

---

## 2. Gate ledgers

| U gate | status | causing field |
|---|---|---|
| U0 | PASS | provenance above; artifact digests recorded |
| U1 | PASS | `test_shwm_planted_defects.py`, `test_shwm_m2d.py` |
| **U2** | **PASS** | `m2d-dataflow.json:u2_dataflow_clean=true`; 12 defects, each caught by its own guard, all guards passing honest |
| **U3** | **PARTIAL** | stable (`arms.1_original.stats.p10=1.0000`, 0/20 collapses) but `c2_symmetry_breaking_is_generic=false` |
| **U4** | **PASS** | `c3_true_event_filter_beats_memoryless=true`; p10 0.9987 vs 0.5000, +0.4997 [+0.4992, +0.5000] |
| **U5** | **PASS** | `survival.changes_2plus` +0.1371 [+0.0997, +0.1739], 149 200 rows |
| U6 | PASS | `corruption` holds all shifted/dropped/flipped/shuffled/constant arms |
| **U7** | **PASS** | `u7_arm_key=2_learned_event_learned_filter_2state::hard`, +0.0999 [+0.0710, +0.1287] |
| **U8** | **PASS** | `c8_corruptions_remove_the_advantage=true` |
| U9 | PASS | event target, binary factorisation **and now the transition itself** are authored |
| U10 | PASS | per-seed records plus frozen per-row predictions |
| U11 | PASS | one padded alias tensor shared by every arm; identical data, budget, ceiling |

| C gate | status | causing field |
|---|---|---|
| C0 | PASS | provenance and suite above |
| **C1** | **FAIL** | `m2d-arm-identity.json:temporal_mechanism=exact_accumulator` vs the M2C narrative |
| **C2** | **FAIL** | `c2_symmetry_breaking_is_generic=false`, `c2_orientation_invariant=false` |
| C3 | PASS | as U4 |
| C4 | PASS | as U2 |
| C5 | PASS | correct events beat every corruption |
| C6 | PASS | as U7 |
| C7 | PASS | as U5 |
| C8 | PASS | as U8 |
| C9 | PASS | matched populations, supervision, evidence, compute |
| C10 | PASS | every seed and failure retained |

**10 PASS, 1 PARTIAL for U; 9 PASS, 2 FAIL for C.**

---

## 3. The U7 arm identity, resolved

Resolved by parsing the frozen source, not by reading either label:

```
filter aliases imported from filter_stability : ['pad', 'run_filter']
of those, ever CALLED                          : NONE
phase_from_learned_events XOR-accumulates      : True
=> temporal_mechanism                          : exact_accumulator
```

`SELECTED_FILTER = "3_two_state_symmetry_broken"` reached the artifact as a **string
field** describing an object that was never instantiated. Per §B, **U7 is marked
NOT_RUN for M2C** and re-run here.

Nine of thirteen identity fields were recoverable only from source; four —
`checkpoint_hash`, `initialization_rule`, `seed`, `query_action_budget` — are absent
from the artifact entirely.

**A second provenance defect, found while reproducing the population.** The artifact
records `alias_examples: 6396`. Re-running the same function with its documented
defaults gives **9960**. 6396 corresponds to **six** alias layouts, not ten — and the
artifact records no layout set, so its population is not reproducible from it. Every
M2D artifact now records its layout list.

M2D arms carry an `ArmIdentity` built from the trained object: `temporal_mechanism`
comes from the class holding the recurrence and `checkpoint_hash` from the parameter
bytes. A stale label is no longer expressible.

---

## 4. Symmetry-breaking audit — the M2C claim is refuted

The M2C docstring said the perturbation "encodes no phase semantics and no XOR
structure". Softmaxing it takes one line:

```
event 0 -> [[0.7311, 0.2689],    event 1 -> [[0.2689, 0.7311],
            [0.2689, 0.7311]]                [0.7311, 0.2689]]
             = STAY                            = FLIP
```

That is the XOR automaton at 73 % confidence before a single gradient step. Twenty
untouched validation seeds, every random control matched to the frozen perturbation's
Frobenius norm of 1.4142:

| arm | held-out | p10 | min | phase (up to permutation) |
|---|---:|---:|---:|---:|
| 1 original (XOR-oriented) | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 2 sign-reversed | 0.6258 | 0.6108 | 0.6056 | 0.5300 |
| 3a latent-state permutation | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3b gauge permuted | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3c gauge learned | 0.9510 | 0.9175 | 0.6111 | 0.9751 |
| 4 event labels permuted | 0.6246 | 0.6072 | 0.5944 | 0.5332 |
| 4b event labels permuted, init relabelled too | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| 5 matched random antisymmetric | 0.7207 | 0.6139 | 0.6028 | 0.6271 |
| 5b matched random independent | 0.7208 | 0.6081 | 0.6000 | 0.6440 |
| 6 zero symmetry breaking | 0.6599 | 0.6050 | 0.6000 | 0.5569 |
| 7 eight-state overcomplete | 0.7443 | 0.6297 | 0.6056 | 0.8060 |
| 8 random restarts *(exploratory)* | 0.9797 | 0.9917 | 0.6528 | 0.9751 |

**Symmetry breaking as such does nothing.** A perturbation of identical magnitude and
identical event-antisymmetry, differing only in orientation, sits at the lower tail of
the no-perturbation baseline (p10 0.6139 vs 0.6050). Only the orientation that *is* the
answer reaches the ceiling, and reversing its sign destroys it.

What the controls do establish:

- **Latent states are anonymous.** 3a and 3b are both exactly 1.0000, so relabelling
  states changes nothing once the transitions can compensate.
- **Event relabelling is an invariance, once applied consistently.** Arm 4 collapses and
  4b is exact, which localises the whole asymmetry in the initialisation's binding to
  the event *labels* — it names which event means "flip".
- **The overcomplete 8-state control does not rescue it** (p10 0.6297), so this is not
  about supplying the exact hidden cardinality.
- Arm 8 is **exploratory and carries no gate** — it was added after validation exposure,
  which the M2C rule forbids for a candidate. It is reported because it answers the
  question the failure raises: eight matched-magnitude random restarts kept by *training
  loss alone* — no phase label, no held-out set, no automaton — solve 19 of 20 seeds.
  That is the generic route, and it is what the next phase should preregister.

**C2 fails.** U3 becomes PARTIAL: the filter is stable, but it did not learn the
environmental transition — it was initialised at it.

---

## 5. True-event closure (§D, U4), 20 seeds

Population: 2726 pairs, 19 920 directed rows, 421 alias classes, layouts 90000–90009.

| arm | alias acc | p10 | NLL | Brier | margin | phase | collapse | vs memoryless |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| exact accumulator | 1.0000 | 1.0000 | 0.0009 | 0.0000 | 10.54 | 1.0000 | 0.00 | +0.5000 [+0.5000, +0.5000] |
| learned filter, 2-state | 0.9997 | 0.9987 | 0.0060 | 0.0008 | 8.59 | 1.0000 | 0.00 | +0.4997 [+0.4992, +0.5000] |
| learned filter, 8-state | 0.5873 | 0.5029 | 1.3059 | 0.3277 | 0.78 | 0.6504 | 0.00 | +0.0873 [+0.0359, +0.1500] |
| generic GRU | 0.4976 | 0.4914 | 2.6855 | 0.4465 | −0.01 | 0.5019 | 0.00 | −0.0024 [−0.0083, +0.0032] |
| trained memoryless | **0.5000** | 0.5000 | 3.0456 | 0.4489 | 0.0000 | — | 0.00 | — |
| constant-event filter | 0.4999 | 0.4987 | 1.4560 | 0.3876 | 0.00 | 0.5005 | 0.90 | −0.0001 [−0.0010, +0.0009] |

The memoryless arm landing on **exactly** 0.5000 with zero variance is the construction
validating itself: identical packets force identical logits, so the two directions of a
pair must tie. Every point above it is history and nothing else.

The GRU, given the same true events and the same budget, remains at chance. The
constant-event filter confirms the recurrence contributes nothing without events.
**U4 passes.**

---

## 6. Hard and posterior-mixture coupling (§E, U7)

Coupling chosen on the **development** alias population (layouts 91000–91009,
seeds 7000–7004): hard 0.6081, posterior 0.6024 → **hard selected**. Then, on validation:

| arm | alias acc | p10 | NLL | Brier | vs memoryless |
|---|---:|---:|---:|---:|---|
| learned event + exact accumulator, hard | 0.5999 | 0.5999 | 4.116 | 0.3993 | +0.0999 [+0.0710, +0.1286] |
| learned event + exact accumulator, posterior | 0.6058 | 0.6024 | 2.275 | 0.3491 | +0.1058 [+0.0764, +0.1351] |
| **learned event + learned 2-state filter, hard** | **0.5999** | 0.5997 | 3.367 | 0.3961 | **+0.0999 [+0.0710, +0.1287]** |
| learned event + learned 2-state filter, posterior | 0.6018 | 0.5985 | 1.871 | 0.3411 | +0.1018 [+0.0726, +0.1311] |
| learned event + learned 8-state filter, hard | 0.5201 | 0.4961 | 1.615 | 0.3882 | +0.0201 [+0.0050, +0.0395] |
| learned event + generic GRU, hard | 0.4995 | 0.4954 | 2.676 | 0.4452 | −0.0005 [−0.0063, +0.0051] |
| learned event + no temporal state | 0.5000 | 0.5000 | 3.046 | 0.4489 | +0.0000 |
| memoryless | 0.5000 | 0.5000 | 3.046 | 0.4489 | — |

**U7 passes on the arm it names.** The learned detector coupled to the learned 2-state
filter beats the trained memoryless model by +0.0999 with a paired hierarchical interval
excluding zero, and beats the generic GRU by +0.1003 [+0.0704, +0.1306].

Two honest qualifications:

- The U7 arm is **statistically indistinguishable from the exact accumulator**:
  −0.0000 [−0.0003, +0.0002]. The learned filter converged to the automaton it was
  initialised at, so "learned filter" and "hand-written XOR" are the same function here.
- **Hard versus posterior is a wash on accuracy** (−0.0020 [−0.0067, +0.0026]) and the
  selection rule picked hard by 0.006 on development. On *calibration* the posterior
  mixture clearly wins — NLL 1.871 vs 3.367, Brier 0.3411 vs 0.3961 — because it does
  not discretise an uncertain event before propagating it. The preregistered rule is
  honoured and hard is reported as selected; a rule keyed on NLL would have chosen
  differently, and that is worth knowing rather than burying.

---

## 7. Dataflow and planted leaks (§F, U2)

Legal order, enforced and audited:

```
C_hat_t = g(X_{t-1}, A_{t-1}, X_t)      B_t = F(B_{t-1}, C_hat_t)      Y_hat = P(X_t, B_t, A_t)
```

The guards are **information tests, not name tests**: the three legal inputs are held
byte-identical, one forbidden channel is moved, and the estimate must not move. Every
mutated arm keeps the honest architecture — the seven channels always occupy their slots
and are zero-filled unless planted — so a guard firing means the channel carries
information, not that a tensor changed shape.

| planted defect | guard | passes honest | catches |
|---|---|---|---|
| 1 X_{t+1} to detector | no_future_observation | yes | yes |
| 2 Y_{t+1} to detector | no_future_outcome | yes | yes |
| 3 target displacement | no_target_displacement | yes | yes |
| 4 future action result | no_future_action_result | yes | yes |
| 5 evaluator phase | no_evaluator_phase | yes | yes |
| 6 simulator step | no_simulator_step | yes | yes |
| 7 provenance digest | no_provenance_digest | yes | yes |
| 8 event shifted forward | alignment peaks at lag zero | yes | yes |
| 9 event shifted backward | alignment peaks at lag zero | yes | yes |
| 10 action misaligned | action matches scored transition | yes | yes |
| 11 query action visible | query action not read | yes | yes |
| 12 belief reads observation | belief isolated | yes | yes |

**12 of 12. U2 passes** — the oldest open gate in this line of work.

Two things the build itself surfaced:

- **Defect 11 is the M2C detector.** `sequence_features` writes the query one-hot into
  every row, so the M2C event estimate moved when you asked about a different action.
  A_t is not in the legal input set for C_t. The M2D detector masks it; that repair is
  in every coupling number above.
- **The wiring guards need an untrained detector and the alignment guard needs a trained
  one**, and running both under one detector is what a first version did. A trained model
  can learn to ignore a live channel, hiding a real wire behind a converged weight; a
  random projection responds to anything connected. Conversely, asking where the lag-zero
  peak of random noise lies is meaningless. The matrix is therefore run twice and each
  guard is judged under the detector that can answer it.

---

## 8. Event-detector generalisation (§I)

The detector is explicitly **retrospective**: `p(C_t | X_{t-1}, A_{t-1}, X_t)`. It is not
`p(C_{t+1} | B_t, X_t, A_t)` and no prospective predictor was trained, so it is not a
prospective world model.

| split | balanced | precision | recall | F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| development layouts | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0021 |
| held-out layouts | 0.7982 | 0.7612 | 0.6623 | 0.7083 | 0.1031 | 0.0916 |
| far held-out layouts | 0.8110 | 0.6623 | 0.7391 | 0.6986 | 0.1302 | 0.1180 |
| held-out visitation policy | 0.7633 | 0.6000 | 0.6667 | 0.6316 | 0.1566 | 0.1412 |

Held-out layouts, by action: 0.8269 / 0.6905 / 0.8747 / 0.8571 — action 1 is markedly
worse. By phase-change count: 0.7610 (1), 0.8029 (2), 0.8750 (3). The `changes_0`
stratum reports `nan` balanced accuracy because it contains no positive events by
definition, which is correct rather than missing.

**A perfect 1.0000 on development layouts against 0.7982 held out is the whole
fragility.** Restricted-input controls confirm where the signal is: an action-only
predictor reaches 0.4564 held out (nothing), a state-only predictor 0.6455.

**"Held-out dynamics" is not instantiable in v2 and is not claimed.** `SWITCH_COUNT` is
a constant and the flip rule never varies, so there is one transition function. What is
held out instead is the *visitation policy* — a goal-directed driver rather than the
uniform-random one training saw — and it is labelled as that.

---

## 9. Survival by number of phase changes (§G, U5)

Strata use the **pair minimum**, deliberately. Aliases share a layout and therefore an
initial polarity, so their own crossing counts always differ in parity; stratifying on a
row's own count would split the two directions of every pair and move the memoryless
baseline off 0.5. The pair minimum keeps both directions together and still means "both
histories have had at least n changes".

| stratum | rows | memoryless | learned event + learned filter | delta | interval |
|---|---:|---:|---:|---:|---|
| 0 changes | 121 680 | 0.5000 | 0.6155 | +0.1155 | [+0.0766, +0.1505] * |
| 1 change | 127 520 | 0.5000 | 0.5414 | +0.0414 | [−0.0110, +0.0973] |
| 2 changes | 103 520 | 0.5000 | 0.6436 | +0.1436 | [+0.1007, +0.1854] * |
| 3 changes | 40 720 | 0.5000 | 0.6195 | +0.1195 | [+0.0632, +0.1733] * |
| 4+ changes | 4 960 | 0.5000 | 0.6452 | +0.1452 | [+0.0939, +0.2114] * |
| **2+ changes** | **149 200** | **0.5000** | **0.6371** | **+0.1371** | **[+0.0997, +0.1739] \*** |

**U5 passes**, and zero- and one-change cases are excluded from the gate rather than
pooled into it. The one-change stratum is the only one whose interval crosses zero; it
is reported as measured rather than smoothed, and it has no obvious mechanism — one
crossing is exactly where a single detector error flips the parity outright.

---

## 10. Event corruption (§H, U8)

Every control derives from one frozen base event array.

| control | alias acc | delta vs memoryless | interval |
|---|---:|---:|---|
| 1 correct learned events | 0.5999 | +0.0999 | [+0.0710, +0.1287] * |
| 2 shifted one forward | 0.4968 | −0.0032 | [−0.0297, +0.0219] |
| 3 shifted one backward | 0.5166 | +0.0166 | [−0.0027, +0.0351] |
| 4 one genuine event dropped | 0.4409 | −0.0591 | [−0.0852, −0.0350] * |
| 5 one event flipped | 0.4648 | −0.0352 | [−0.0479, −0.0223] * |
| 6 cross-episode matched-prevalence shuffle | 0.4930 | −0.0070 | [−0.0131, −0.0008] * |
| 7 position-wise permutation, marginal preserved | 0.5026 | +0.0026 | [−0.0034, +0.0083] |
| 8 constant event | 0.5000 | +0.0000 | [−0.0000, +0.0001] |

**U8 passes.** Every misalignment and information-destroying control removes the
advantage, and dropping or flipping a single event drives the arm *below* chance —
which is the signature of a genuine parity mechanism, since one corrupted event inverts
the phase for the remainder of the route.

Controls 9 and 10 are reported separately as **detector-input ablations, not
corruptions**: an action-only predictor gives −0.0284 and a state-only predictor
+0.0508. §H is explicit that U8 is judged on the misalignment and information-destroying
set; a restricted-input predictor is a weaker detector that legitimately keeps whatever
information it still sees, and scoring it as a corruption would fail the gate for the
wrong reason.

---

## 11. Paired intervals (§J)

All intervals are nested paired resampling, validation seed → layout → alias class,
4000 resamples.

| comparison | delta | interval |
|---|---:|---|
| true-event learned filter − memoryless | +0.4997 | [+0.4992, +0.5000] |
| learned-event exact accumulator − memoryless | +0.0999 | [+0.0710, +0.1286] |
| learned-event learned filter − memoryless | +0.0999 | [+0.0710, +0.1287] |
| learned-event learned filter − generic GRU | +0.1003 | [+0.0704, +0.1306] |
| learned-event learned filter − exact accumulator | −0.0000 | [−0.0003, +0.0002] |
| hard − posterior coupling | −0.0020 | [−0.0067, +0.0026] |
| 2+ phase changes | +0.1371 | [+0.0997, +0.1739] |
| correct events − constant events | +0.0999 | [+0.0710, +0.1287] |
| 2-state − 8-state filter (true events) | +0.4124 | [+0.3497, +0.4638] |

---

## 12. Transfer — and the actual blocker

Four alias layout sets, with true events as the positive control:

| population | memoryless | true event + filter | learned event + filter | final-parity accuracy |
|---|---:|---:|---:|---:|
| development 91k | 0.5000 | 0.9998 | 0.6080 | 0.6102 |
| validation 90k | 0.5000 | 0.9998 | 0.5998 | 0.6043 |
| held-out 92k | 0.5000 | 0.9998 | 0.5921 | 0.5889 |
| held-out 95k | 0.5000 | 0.9999 | **0.5048** | **0.5159** |

The true-event arm is at ceiling on all four, so the populations are sound — 95k is not
a broken construction. What varies is event fidelity, and **the coupling's accuracy
tracks final-parity accuracy to within 0.01 on every set**.

That is the mechanism, and it is multiplicative rather than average: parity over a route
is the product of the per-step accuracies, so a per-step drop from 0.8626 to 0.7919
collapses parity from 0.6102 to 0.5159 and the entire advantage with it. Reporting the
per-step figure alone would have made an arm that cannot work look nearly correct.

**Event fidelity and horizon are the blocker**, exactly as the §K decision table
anticipates. Not the belief filter, not the head, not the model class.

---

## 13. Bugs and corrections

1. **The M2C U7 arm was not the arm reported.** `run_filter` imported, never called;
   XOR accumulation in its place. Resolved by AST, not by reading. U7 withdrawn, re-run.
2. **The M2C artifact's population is not reproducible from it.** `alias_examples: 6396`
   is six layouts; the documented defaults give 9960 over ten. No layout set recorded.
3. **The M2C symmetry-breaking docstring was false.** The perturbation *is* the XOR
   automaton at 73 % confidence. Self-caught by printing the matrix before writing the
   controls that would have caught it anyway.
4. **The M2C detector reads the query action**, which is not in the legal input set for
   C_t. Caught by guard 11; repaired by masking in all M2D couplings.
5. **`anti[::-1]` produced a negative-strided view** that MLX refuses. Caught mid-run.
6. **A first dataflow build ran every guard under one detector**, so the alignment guard
   failed the honest pipeline — random weights have no lag-zero peak. Split into wiring
   (untrained) and behavioural (trained) matrices.
7. **A first stratifier used each row's own crossing count.** Because aliases share an
   initial polarity their counts always differ in parity, so that split the two
   directions of every pair and would have moved the memoryless baseline off 0.5.
   Replaced with the pair minimum, and pinned by a test.
8. **C8 was first computed over all ten §H controls**, failing on the state-only
   predictor. §H judges U8 on the misalignment and information-destroying set; the two
   restricted-input predictors are reported separately.
9. **The nested bootstrap was first written to gather row indices** and would have taken
   hours. Rewritten over per-group sums and counts; a paired difference of means needs
   no index gathering.
10. **`alias_class` was keyed on Python's `hash`**, which is salted per process and would
    have made the bootstrap grouping unreproducible across runs. Replaced with SHA-256.

---

## 14. The narrow supported claim

On exact complete-public-packet alias pairs, where a memoryless model is at exactly
0.5000 by construction, a retrospective learned event detector coupled to a two-state
learned belief filter recovers **+0.0999 [+0.0710, +0.1287]** of the available 0.5000
headroom, survives two or more phase changes at **+0.1371 [+0.0997, +0.1739]**, and
loses that advantage under every misalignment and information-destroying control.

It is **not** autonomous hidden-state discovery, and the reasons have grown by one:

- the event target is authored;
- the binary factorisation is authored;
- **and the transition matrix is supplied at initialisation** — every matched-magnitude
  generic initialisation sits at the memoryless baseline in the lower tail.

So the honest label is a **supervised, event-factorised, transition-initialised belief
model**, weaker than M2C's "supervised event-factorized temporal model". It is also
functionally indistinguishable from a hand-written XOR accumulator on this population
(−0.0000 [−0.0003, +0.0002]).

The gain does not transfer to one of four alias layout sets, and where it fails, final
parity is at 0.5159.

---

## 15. Is the visual ladder unblocked?

**No.** Two things must close first, and neither is about the visual interface.

1. **C2.** A generic initialisation rule, preregistered before validation exposure. The
   exploratory restart procedure — restarts selected on training loss alone — reaches
   19/20 seeds and is the obvious candidate, but it was added after exposure and cannot
   carry the gate in this phase.
2. **Event fidelity.** At 0.79–0.86 per-step accuracy, route parity runs at 0.52–0.61 and
   the advantage is worth between +0.0048 and +0.1080 depending on which layouts you
   draw. Putting a visual front end in front of a detector that already fails to
   transfer would measure the front end against a moving floor.

U2, U4, U5, U7 and U8 are closed. C1 and C2 are not, and C2 is the one that matters:
it is the difference between a filter that learns the environment's temporal structure
and a filter that is told it.
