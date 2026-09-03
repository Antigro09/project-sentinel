# Scale 1A-0R-O1 — Global Visual Role Binding and Hidden-Palette Memory Closure

**A stateless, permutation-equivariant global role binder solves the hidden palette that
phase O's local detector could not — including where cardinality provably cannot help.**
The outcome-trained initial-state gauge, owed since M2F, passes: grounding is no longer
authored. The multimodal gate fails on a properly paired population, and phase N's N12 is
withdrawn.

**7 PASS, 3 PARTIAL, 4 NOT_RUN, 1 FAIL.** Prospective prediction is **not** unblocked.

---

## 1. Provenance (P0 — PASS)

| field | value |
|---|---|
| commit | `953f0520efeaf6d69eeb7e8adb1a0cb7c876ca99` |
| branch | `phase-2-continuous-world-model` |
| tracked modified | `.claude/worktrees/x35-novelty-trigger` only |
| Phase-2 suite | 510 passed, 38.3 s |
| required suite | **984 passed, 4 skipped, 0 failed, 988.0 s** |
| final Scale-1 seed opened | no |
| prospective model started | no |
| Stage 1A-1 matrix run | no |

Palettes: development 9300–9315, unseen 9400–9415 (binding); 7101 (equivalence).
Layouts: binding train 110000–110023, test 111000–111011; gauge train 110000–110039, test
111000–111019; multimodal train 113000–113199, test 114000–114139.

### The O-ledger inconsistency, resolved

Phase O reported "6 PASS + 2 PARTIAL + 7 NOT_RUN" while calling O12 and O13 *failures* in
prose. The counts were right (15 gates) but the statuses were not distinguishable: O12 and
O13 carried the same `NOT_RUN` as O6–O10, which were genuinely blocked. Every `NOT_RUN`
now carries a `reason_class`:

| gate | status | reason_class | mandatory in O | superseded |
|---|---|---|---|---|
| O6–O10 | NOT_RUN | `BLOCKED_UPSTREAM` | no | — |
| **O12** | NOT_RUN | **`NOT_DELIVERED`** | **yes** | **P11: PASS** |
| **O13** | NOT_RUN | **`NOT_DELIVERED`** | **yes** | **P12: FAIL** |

The status field still describes the measurement — there was none. The new field says
why, so the prose no longer has to.

---

## 2. Residual class histogram (P1 — PASS, and it refutes phase O)

All 720 permutations enumerated per stage, 47 grounded episodes. The quotients have an
exact closed form: `~event` iff π fixes AGENT and SWITCH; `~goal` iff π fixes AGENT and
both markers; `~full` iff π is the identity.

| stage | mean | med | min | max | entropy | full mass | event mass | goal mass | class > 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one frame | 12.00 | 12 | 12 | 12 | 3.5850 | 0.0833 | 0.3333 | 0.1667 | 47 |
| frame pair + action | 5.19 | 4 | 4 | 12 | 2.2361 | 0.2252 | 0.9007 | 0.4504 | 47 |
| short legal history | 4.51 | 4 | 4 | 12 | 2.1012 | 0.2394 | 0.9574 | 0.4787 | 47 |
| **grounded calibration** | **2.47** | 2 | 2 | **12** | 1.1313 | 0.4770 | 0.9716 | 0.4858 | **3** |
| complete history | 2.47 | 2 | 2 | 12 | 1.1313 | 0.4770 | 0.9716 | 0.4858 | 3 |
| + language goal | 2.47 | 2 | 2 | 12 | 1.1313 | 0.4770 | 0.9716 | 0.4858 | 3 |

**Grounded histogram: `{2: 44, 4: 1, 12: 2}` — phase O's claim that the only residual is
GOAL_ALPHA ↔ GOAL_BETA is refuted.** Three episodes exceed class 2, and two of them are
still at the *one-frame* class of 12:

- layout 110002, class 4 — an extra `WALL ↔ EMPTY` swap: the agent was never blocked, so
  wall grounding never fired;
- layouts 110015 and 110047, class 12 — `AGENT ↔ GOAL_BETA` unresolved: the agent never
  moved, so motion never pinned it.

O's claim came from three sampled certificates, not a histogram. This is what §B was for.

**Goal-equivalence is never identified at any stage (0.000), language included.** The
instruction names the target *role*; it binds a *colour* only if the episode also shows
the agent reaching that marker, which under this policy it never does.

---

## 3. Removing the cardinality shortcut (P3 — PASS)

A DECOY role renders as its own colour and behaves exactly like EMPTY — walkable, no
polarity flip — so semantic dynamics are untouched and only the informativeness of a cell
count changes.

| stratum | construction | switch : decoy counts |
|---|---|---|
| COUNT_INFORMATIVE | no decoy | 7 : — |
| COUNT_VARIED | decoy count ~ U{4..10} | 7 : 4–10 |
| COUNT_COLLISION | decoy count = switch count | **7 : 7** |

---

## 4. Global role binding (P5 — PARTIAL, P7 — PARTIAL)

A DeepSets over per-colour tokens — RGB, cell count, spatial moments, motion — emitting a
soft colour-to-role assignment, with the event computed by the M2F relational expression
lifted onto soft roles and supervised by the **public event label only**. Unseen palettes,
held-out layouts, 8 palettes × 24 layouts, 2 seeds.

| stratum | count-only | motion-only | count+motion | full token | local conv (O baseline) |
|---|---:|---:|---:|---:|---:|
| COUNT_INFORMATIVE | 0.5109 | 0.6131 | **0.9048** | 0.8793 | 0.5868 |
| COUNT_VARIED | 0.5219 | 0.6103 | 0.9000 | **0.9239** | 0.5776 |
| **COUNT_COLLISION** | 0.5274 | 0.6197 | 0.9017 | **0.9015** | 0.5510 |

Three things follow.

- **The binder transfers to unseen palettes; the local detector does not.** ~0.90 against
  0.55–0.59, reproducing phase O's negative result for the local arm and overturning it
  for a global one. This is the prediction O ended on, and it holds.
- **It is not a cardinality lookup.** Count-only is at chance in *every* stratum,
  including COUNT_INFORMATIVE, and the binder holds 0.9015 under a provable cardinality
  collision where SWITCH and DECOY have identical counts.
- **Motion is the load-bearing feature.** count+motion ≈ full token; adding RGB and
  spatial moments buys almost nothing.

**P5 is PARTIAL, not PASS.** What works is *stateless* — a current-frame-pair binder. The
persistent-memory arms §E preregisters (recurrent explicit assignment, Sinkhorn binder,
implicit recurrent) were **not built**, so "appearance memory" is untested and P6 has no
gain to destroy. **P7 is PARTIAL** for the same class of reason: event prediction transfers,
route parity under unseen palettes was not measured.

---

## 5. O12 — outcome-trained initial-state gauge (P11 — PASS)

Owed since M2F. The gauge sees the reset frame, emits a two-state initial belief, and is
trained **only** through the likelihood of later displacement outcomes — no phase target,
no phase input. The belief is the sole path from the reset frame to the loss.

| variant | belief accuracy (up to permutation) | displacement |
|---|---:|---:|
| authored public stripe | 1.0000 | — |
| stripe-supervised | 1.0000 | 1.0000 |
| phase-supervised (diagnostic) | 1.0000 | 1.0000 |
| **outcome-trained** | **1.0000** | 1.0000 |
| stripe masked | 0.5085 | 0.6051 |
| reset frame omitted | 0.5085 | 0.5696 |
| shuffled reset frame | 0.5085 | 0.5838 |
| false stripe | 0.7543 | 0.7969 |

Paired difference against the authored gauge: **+0.0000**.

**Initial-state grounding is no longer conditional on authored evidence.** M2F, N and O all
had to carry that condition; it is discharged here.

---

## 6. O13 — multimodal grounding (P12 — FAIL)

The scene is held byte-identical between the two goals: same layout, same palette, same
shared action plan, same history. Only the instruction differs. **374 contested keys** over
748 rows, from 2520 test rows.

| arm | contested accuracy |
|---|---:|
| vision + language + history | **0.5000** |
| shuffled language | 0.4960 |
| masked language | 0.5000 |
| wrong goal convention | 0.4996 |
| no history | 0.4964 |
| shuffled history | 0.5040 |

Paired intervals by contested key:

| comparison | delta | interval |
|---|---:|---|
| correct − shuffled language | +0.0040 | [−0.0071, +0.0156] |
| correct − masked language | +0.0000 | [−0.0089, +0.0094] |
| correct − wrong goal convention | +0.0004 | [−0.0111, +0.0116] |
| correct − no history | +0.0036 | [−0.0076, +0.0152] |
| correct − shuffled history | −0.0040 | [−0.0152, +0.0071] |

**P12 fails, and phase N's N12 pass is withdrawn.** N measured +0.021 on *all* held-out
rows with a point estimate and no interval; on the keys where language is actually
decisive, there is nothing.

**But the precise reading matters.** The *correct* arm is itself at 0.5000 on contested
keys. A test whose treatment arm is at chance has no power to detect an effect, so this is
a **capability failure of the readout, not evidence that language is uninformative**. The
information is present by construction — a contested key is one where the two goals
genuinely disagree.

---

## 7. Gates

| gate | status | basis |
|---|---|---|
| P0 | PASS | provenance; O ledger reissued with `reason_class` |
| P1 | PASS | full histogram; **refutes** O's residual claim |
| P2 | PASS | oracle spread 0.0113 across five regimes (carried) |
| P3 | PASS | count-only at chance everywhere; binder 0.9015 under collision |
| P4 | PARTIAL | entropy 3.585 → 1.149 bits; event identified 0.958 vs a pre-stated 0.99 |
| **P5** | **PARTIAL** | stateless binder beats local by ~0.35; persistent-memory arms `NOT_DELIVERED` |
| P6 | NOT_RUN | `BLOCKED_UPSTREAM` — no persistent-memory gain exists to destroy |
| P7 | PARTIAL | event transfers; route parity under unseen palettes `NOT_DELIVERED` |
| P8 | NOT_RUN | `NOT_DELIVERED` — binder never coupled to the certified transition |
| P9 | NOT_RUN | `NOT_DELIVERED` — no alias-pair or phase-change result in this regime |
| P10 | PASS | per-frame at 0.4903, reported UNRESOLVED (carried) |
| **P11** | **PASS** | outcome-trained gauge 1.0000, +0.0000 against authored |
| **P12** | **FAIL** | 374 contested keys; all intervals include zero; correct arm at chance |
| P13 | PASS | public per-colour tokens only; no palette id, role label or evaluator state |
| P14 | PASS | every palette, stratum, seed and failed arm retained |

**Decision.** None of §L's failure branches fit cleanly. The exact posterior passes P4 *and*
learned global role binding works, so "learned global role binding is the blocker" no
longer holds. Count-varied and count-collision both pass, so "cardinality lookup" is ruled
out. O12 passes, so the authored-grounding condition is discharged. But the interface is
**not frozen** and prospective prediction is **not unblocked**, because §L requires P5–P12
and P12 fails with P6, P8, P9 unrun.

---

## 8. Bugs and corrections

1. **Phase O's residual claim was not supported by a histogram.** Printing it shows
   `{2: 44, 4: 1, 12: 2}` — three episodes exceed class 2 and two are at 12. Withdrawn.
2. **Phase N's N12 is withdrawn.** It passed on a +0.021 point estimate over all rows; on
   374 contested keys with paired intervals, every interval includes zero.
3. **My cardinality check rejected the truth** in the O identifiability audit (the agent
   can occlude a goal marker) — caught by asserting the truth must survive its own
   evidence, and carried into this phase's stage code.
4. **The O ledger could not distinguish "blocked" from "not built".** Both were `NOT_RUN`.
   Fixed with `reason_class`, and four of this phase's own gaps are labelled
   `NOT_DELIVERED` under the same rule rather than hidden behind the same word.

---

## 9. The narrow supported claim

Under a per-episode hidden bijection from seven semantic roles to a colour pool, with
supervision from the public event label alone and no palette id, role name or evaluator
state in scope:

- a **stateless permutation-equivariant global binder** over per-colour public tokens
  reaches **0.9015–0.9239 balanced accuracy on unseen palettes and held-out layouts**,
  against **0.5510–0.5868** for the local detector phase O tested;
- it holds under a **provable cardinality collision** (SWITCH and DECOY both at seven
  cells), and **count-only features are at chance in every stratum**, so the mechanism is
  not a cardinality lookup;
- an **outcome-trained initial-state gauge** matches the authored stripe map exactly
  (**+0.0000**), discharging a condition carried since M2F;
- the exact posterior concentrates from 9.4919 to **1.1494** bits, and the residual class
  after grounding is **{2: 44, 4: 1, 12: 2}** — not uniformly the two goal markers.

Standing conditions: persistent appearance **memory** is untested (the working binder is
stateless); the binder is **not coupled** to the certified transition, so no alias-pair,
route-parity or phase-change result exists under unseen palettes; and the **multimodal
contribution is not established**, with the caveat that the test lacked power.

---

## 10. Is prospective prediction unblocked?

**No**, and the remaining work is now four specific arms rather than an open problem.

Three of the four are mine to build and were not: the persistent-memory binders (P5/P6),
the coupling of the binder to the certified transition on the alias population under
unseen palettes (P8/P9), and route parity in that regime (P7). The fourth is P12, where the
obstacle is a readout that cannot solve contested keys at all — the same class of failure
as phase N's scalar event head, and the same fix is likely: supervise the goal-relative
decision spatially rather than through a pooled coordinate comparison.

What has changed is the diagnosis. After phase O the question was whether appearance
conventions could be inferred at all. They can, from public evidence, by a small stateless
set model with no memory and no privileged information.
