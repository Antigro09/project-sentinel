# Scale 1A-0R-N — Visual Event Extraction and Initial-State Grounding

**Perception recovers the event from pixels, and the complete visual pathway works — but
only while the palette is held fixed.** All fifteen N gates pass. Two 4B pretrained
backbones lose to a frozen random projection. And every interface, including the oracle
diagnostic, collapses to chance under appearance shift.

---

## 1. Provenance

| field | value |
|---|---|
| commit | `f65a7468bf637c4c010a3d5fbc3686ed71cce86f` (M2F); N work uncommitted at measurement |
| branch | `phase-2-continuous-world-model` |
| Phase-2 suite | **510 passed, 40.4 s** (500 before, plus 10 N pins) |
| required suite | clean since M2F: 974 passed, 4 skipped, 0 failed |
| optional manifest | 2 tests + 2 modules, `arc_agi`, skipped explicitly |
| visual or final Scale-1 seed opened | **no** |
| Stage 1A-1 matrix run | **no** |

New layout ranges claimed, disjoint from every prior phase: train **110000–110059**,
held-out **111000–111029**, crossed **112000–112029**. Appearance-shift seed 777001 vs
the canonical seed. Alias populations reuse the frozen 90000–90009 and 95000–95009.
Every split records its layouts, episode count, event rate and a content digest.

---

## 2. M2F gate ledger, carried

| gate | status | gate | status |
|---|---|---|---|
| F0 | PASS | F5 | PASS |
| F1 | PASS | F6 | PASS |
| F2 | PASS | F7 | PASS |
| F3 | PASS | F8 | PASS |
| F4 | PASS | F9 | PASS |
| E0 | PASS | E4 | PASS |
| E1 | PASS | **E5** | **NOT_INSTANTIABLE** |
| E2 | PASS | E6 | PASS |
| E3 | PASS | E7 | PASS |
| | | E8 | PASS |

E5 stays NOT_INSTANTIABLE: v2 has one transition function. M2F's gauge finding is carried
unchanged — the stripe equals the initial polarity and is public, the learned structured
gauge did **not** match the authored one, so the structured result remains conditional on
authored initial-state grounding.

---

## 3. Structured ceilings, frozen and unaltered

| ceiling | value |
|---|---|
| exact public relational event derivation | 1.0000 per-step, 1.0000 route parity |
| relational learned detector, structured features | 1.0000 per-step held out |
| exact event + exact accumulator | alias 1.0000 |
| exact event + adaptive generic transition learner | alias 0.9999 |
| authored public reset-state gauge | 1.0000 |
| true phase oracle | alias 1.0000 |
| trained structured memoryless | alias exactly 0.5000 |

---

## 4–5. Visual interfaces and the public auxiliary heads

Held-out layouts. One head shape, one parameter budget, one optimizer, one update count
for every interface; the head is fully convolutional so its parameter count does not
depend on the slot grid, and each interface keeps its **native** grid.

| interface | eligible | agent f1 | switch f1 | event | event via masks | head params |
|---|---|---:|---:|---:|---:|---:|
| 1 raw pixels + equivariant CNN | y | 1.0000 | 0.9999 | 0.9818 | 0.9992 | 53 841 |
| 2 fixed random projection | y | 1.0000 | 1.0000 | **0.9945** | **1.0000** | 24 337 |
| 3 learned CNN slots | y | 1.0000 | 0.9981 | **1.0000** | 0.9945 | 101 713 |
| 4 Qwen3-VL 4B spatial slots | y | 0.3128 | 0.4579 | 0.5760 | 0.5828 | 24 337 |
| 5 Gemma 3 4B spatial slots | y | 0.9952 | 0.9170 | 0.8820 | 0.8973 | 24 337 |
| 6 Qwen mean-pool | y | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 24 337 |
| 7 Gemma mean-pool | y | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 24 337 |
| 8 cell-aligned | **N** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 24 337 |

Compute: Qwen 80 ms/frame, Gemma 491 ms/frame, both deduplicated by frame content and
projected 2560→32 per frame by a frozen declared projection before caching. Neither
backbone receives a trainable adapter, and the two cheap baselines carry 0 and 53 841
trainable encoder parameters respectively.

**Both 4B backbones lose to a frozen random projection of the raw pixels.** On a 24×24
frame whose cells are 2×2 pixels, an exact 2×2 block mean is a near-lossless encoding,
while the towers resample through 896×896 (Gemma) or their own patch grid (Qwen) and
arrive at 8×8 or 16×16 tokens that no longer line up with the 12×12 cell grid. Qwen's 8×8
grid cannot even represent the cell lattice.

**Event by split, every interface:**

| interface | held-out layouts | policy shift | **appearance shift** | crossed |
|---|---:|---:|---:|---:|
| 1 CNN | 0.9818 | 0.9906 | **0.4987** | 0.5000 |
| 2 random projection | 0.9945 | 0.9925 | **0.5000** | 0.5000 |
| 3 learned CNN slots | 1.0000 | 1.0000 | **0.5020** | 0.5017 |
| 5 Gemma slots | 0.8820 | 0.8784 | **0.5044** | 0.5024 |
| 8 cell-aligned (oracle) | 1.0000 | 1.0000 | **0.5000** | 0.5000 |

**This is the phase's main negative result and it is not a detail.** Held-out *layouts*
transfer perfectly because the appearance seed fixes the palette, so walls, switches and
the agent keep the same colours in every layout. Change the palette and every interface —
including the cell-aligned oracle — falls to chance. What the detectors learned is a
colour test, not an object.

**Two targets are not public everywhere**, and saying so is part of the result.
`entered_cell_switch` is invisible when the agent does not move: it occludes its own cell
in the before frame *and* the after frame, so nothing in the pair says what is underneath.
`reset_stripe_state` is only rendered on the reset frame. Both are trained and scored on
their identifiable subsets, where they reach 0.9945–1.0000, and reported unconditioned as
well.

---

## 6. Initial-state gauge (N9, reported separately)

| variant | held-out accuracy | eligible |
|---|---:|---|
| A authored public stripe map | 1.0000 | yes |
| B stripe-supervised visual reader | 1.0000 | yes |
| C reset stripe masked | 0.5056 | yes |
| D reset frame omitted | 0.4607 | yes |
| E false stripe | 1.0000 | yes |
| F shuffled reset frames | 0.5468 | yes |
| G phase-supervised visual decoder | 1.0000 | **no** |

The stripe is perfectly readable from pixels, and masking it, removing the frame or
shuffling frames all destroy that — so the information is in the frame and nowhere else.

Two honest qualifications. Arm E scores 1.0000 because the reader is **retrained** on the
inverted stripe and simply learns the inverted convention; it is a positive invariance
check, not an information-destroying control, and C/D/F are the destroying ones. And arms
B and G are the *same computation*: both are supervised on a number that equals the
initial polarity. **The outcome-only-trained visual gauge that §F asks for is NOT_RUN**,
so no claim is made that visual grounding is outcome-learnable. The result stays
**conditional on authored initial-state grounding**.

---

## 7–8. Per-step and route-level event fidelity

Measured on the alias routes, interface 2:

| | validation | held-out |
|---|---:|---:|
| per-step accuracy | 0.9977 | 0.9902 |
| exact route-event-sequence accuracy | 0.9886 | 0.9570 |
| **final event parity** | **0.9886** | **0.9620** |
| mean first error position | 2.41 | 3.65 |
| mean / max error burst | 0.011 / 1 | 0.043 / 1 |
| error autocorrelation lag-1 | −0.003 | −0.011 |
| worst-layout parity failure rate | 0.106 | 0.242 |
| routes | 7 708 | 12 940 |

The M2E development-frozen requirement was per-step ≥ 0.7992 for parity ≥ 0.55. Visual
extraction delivers 0.9902 per-step and 0.9620 parity held out, so **N5 passes with
room**. The independence diagnostic reads 1.0000 against a measured 0.9620 — it
over-predicts because the detector is confident when wrong. Per the specification it is
used only as a diagnostic and the measured sequence result decides.

---

## 9. Complete visual temporal pathway

Only the event source varies; the belief filter is the frozen M2F certified adaptive
learner and the outcome head is the frozen M2F head.

| arm | validation | held-out |
|---|---:|---:|
| 1 visual event + exact accumulator | 0.9896 | 0.9684 |
| **2 visual event + certified adaptive transition** | **0.9895** | **0.9683** |
| 3 visual event + generic GRU | 0.4995 | 0.5003 |
| 4 visual event + no temporal state | 0.5000 | 0.5000 |
| 5 visual memoryless baseline | **0.5000** | **0.5000** |
| 6 structured event + certified transition (ceiling) | 0.9999 | 0.9999 |
| 7 exact event + exact accumulator (ceiling) | 1.0000 | 1.0000 |

| split | delta vs visual memoryless | interval | 2+ changes |
|---|---:|---|---:|
| validation | **+0.4895** | [+0.4679, +0.5000] * | +0.4758 * |
| held-out | **+0.4683** | [+0.4333, +0.4929] * | +0.4817 * |

The memoryless baseline sits at exactly 0.5000 with zero variance, verified rather than
assumed: alias pair members share a byte-identical packet *and* a byte-identical frame, so
any memoryless model must tie. The GRU remains at chance even given good visual events,
as in every prior phase.

---

## 10. Multimodal ablations (N12)

The alias outcome metric is displacement, which is goal-independent, so it can never show
language contributing. The target here is instead **did the action move the agent toward
the language-named goal** — needing the frame for positions, the language for *which*
marker (the frame renders both and marks neither), and the history because polarity
negates the action delta.

| arm | held-out balanced | F1 |
|---|---:|---:|
| vision + language + history | **0.5317** | 0.2840 |
| vision + shuffled language + history | 0.5111 | 0.2589 |
| vision + masked language + history | 0.5106 | 0.2843 |
| vision + language + no history | 0.5035 | 0.0930 |
| vision + language + shuffled action history | 0.5053 | 0.1195 |

Both channels are non-vacuous (language +0.021, history +0.028), on 65 frame-action keys
that carry two different correct answers under two different goals. Audio remains declared
absent.

**The margins are small and the absolute level is low, and there is an environment reason.**
`reset` accepts exactly one dynamic and defaults every other seed, so appearance cannot be
held fixed while the goal varies. The controlled language test therefore necessarily runs
with the palette changing per layout — the regime §4 shows is hardest. History's
contribution is robust; language's is real but marginal.

---

## 11. Gates

| gate | status | basis |
|---|---|---|
| N0 | PASS | provenance; Phase-2 510 in 40.4 s; required suite clean |
| N1 | PASS | 16 planted visual defects, 16 caught, all guards pass honest |
| N2 | PASS | agent mask f1 1.0000 from raw pixels held out |
| N3 | PASS | best non-oracle interface 1.0000 vs controls at 0.5000 |
| N4 | PASS | held-out layouts 1.0000 — **appearance shift fails at 0.5020** |
| N5 | PASS | route parity 0.9620 vs the frozen 0.55 requirement |
| N6 | PASS | +0.4895 [+0.4679, +0.5000] |
| N7 | PASS | +0.4758 [+0.4304, +0.5000] |
| N8 | PASS | +0.4683 [+0.4333, +0.4929] |
| N9 | PASS | gauge reported separately; outcome-trained arm NOT_RUN |
| N10 | PASS | slots 0.8820 vs mean-pool 0.5000 |
| N11 | PASS | cheap baselines retained and they **win** |
| N12 | PASS | language +0.021, history +0.028, 65 contested keys |
| N13 | PASS | no dynamics split exists and none is claimed |
| N14 | PASS | every interface, split, seed and failure retained |

**Decision (§L): pretrained slot/resampling loss.** Both pretrained interfaces lose to the
cheap baselines, so the raw/CNN path continues and no large backbone is forced. Selected
interface **2 fixed random projection**, frozen alongside cheap baseline **1 raw pixels +
equivariant CNN**.

---

## 12. Bugs and corrections

1. **A misaligned shared slot grid.** Forcing every interface through 8×8 meant
   zero-padding a 12×12 map to 16×16 and pooling 2×2: six slot rows real, two padding, no
   slot boundary on a cell boundary. Both pixel interfaces sat at majority-class on every
   spatial target. Each interface now keeps its native grid.
2. **The event head could not fit its own training set.** Supervised as a bare scalar
   behind a spatial max, exactly one location per example got gradient: train balanced
   accuracy 0.5301 at 2500 updates, 0.5788 at 8000. Supervised spatially, the identical
   head reaches 1.0000 on train and held out. Every interface number taken before that
   was measuring the readout.
3. **An attempted fix collapsed every head to a constant** — a Linear on an unbounded
   post-max activation — and to the *same* constant for three very different interfaces,
   which is what gave it away.
4. **`pool_to_slots` reported an upsample as a divisibility error**, because the modulo
   check ran before the upsampling check. Reordered; pinned by a test.
5. **Training two targets where they are unidentifiable** poisoned their heads.
   `entered_cell_switch` and `reset_stripe_state` are now fitted and scored on their
   identifiable subsets and reported both ways.
6. **The §K target was degenerate**: "land on the goal" fires on 0.67 % of rows and left
   2 of 738 contested keys, so every arm predicted all-negative at exactly 0.5000.
   Replaced with the dense toward-the-goal target.
7. **The §K readout could not express the task.** A 3×3 convolutional head cannot compare
   an agent cell with a marker eleven cells away; it scored 0.5378 with correct language
   and 0.5547 with *shuffled* language. Replaced with a soft-argmax coordinate readout.
8. **Both §K "goal draws" produced the same goal, twice over.** First because a second
   `reset` call defaults the goal back; then because draws 0 and 1 both hash to the alpha
   marker. Fixed to a single `goal:` dynamic with draws 0 and 2, and pinned by a test that
   asserts two goals and one identical frame.
9. **I downloaded 17.5 GB of model weights that were already on disk**, having checked the
   HF cache but not `artifacts/shwm/backbones/`, which is the directory the code uses.
   Interfaces 4–7 were never blocked.
10. **The scout workflow lost 9 of 10 agents to a usage limit**; the remaining subsystems
    were mapped inline instead.

---

## 13. The narrow supported claim

From frame pairs alone — no structured coordinates, no switch bits, no true displacement,
event, phase, step, seed, layout id, provenance digest, future frame, future action result
or target outcome, each planted and caught — a fixed random projection of the raw 24×24
pixels supports a retrospective event detector reaching **0.9945 balanced accuracy** and
**0.9620 route parity** on held-out layouts, and the complete pathway of visual event plus
the M2F certified adaptive transition beats a visual memoryless baseline pinned at exactly
0.5000 by **+0.4895 [+0.4679, +0.5000]** on validation and **+0.4683 [+0.4333, +0.4929]**
on held-out layouts, surviving two or more phase changes on both.

Standing conditions:

- **it does not survive appearance shift** — every interface, oracle included, falls to
  chance at 0.4987–0.5044 when the palette changes;
- **conditional on authored initial-state grounding** — the outcome-only-trained visual
  gauge is NOT_RUN;
- the event target and the binary factorisation remain **authored**;
- it is **retrospective**: `p(C_t | O_{t-1}, A_{t-1}, O_t)` and the belief update after
  observing O_t. Not `p(C_{t+1} | B_t, O_t, A_t)` before acting. **This is not a planning
  world model.**
- **no dynamics-generalization claim**; v2 has one transition function.

---

## 14. Selected interface

**2 fixed random projection**, with **1 raw pixels + equivariant CNN** frozen alongside as
the required cheap baseline. Both pretrained slot interfaces are retained in the record as
losses, not removed.

---

## 15. Is prospective world-model training unblocked?

**Yes for the retrospective precondition, no for the representation.**

N3–N8 pass, so the next phase may build the prospective predictor
`p(C_{t+1}, O_{t+1} | B_t, O_t, A_t, language_goal)` against the comparisons §M names:
action-conditioned, no-action, shuffled-action, reactive policy, oracle simulator.

But it should be built on the appearance problem, not around it. A detector that reads
palette colours is not perception, and the crossed split says so at 0.5000. The obvious
next control is appearance-randomised *training* rather than a fixed canonical seed —
cheap to run, and it is the difference between an event detector and a colour lookup.

The 87-workload Stage 1A-1 matrix was not launched and remains blocked.
