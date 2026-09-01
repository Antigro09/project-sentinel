# Scale 1A-0R-J — Full-Observation Aliasing and Event-Readout Qualification

## Verdict

**The `b19b3b6` diagnosis is overturned. The chain is not broken; the readout was not
qualified.** With a qualified readout, every link from pixels to hidden phase recovers
end-to-end at **1.000** on held-out layouts, using **35,650 parameters**.

The narrowed diagnosis requested in §0 is the correct one, and the evidence is now direct
rather than inferential:

| link | previous audit | this audit |
|---|---:|---:|
| agent position (exact cell) | R² 0.029 from raw pixels | **1.0000** |
| switch mask (F1) | not measured | **0.981** |
| switch crossing (balanced accuracy) | +0.001, CI spanning zero | **1.0000** |
| hidden phase, end-to-end | unmeasurable | **1.0000** |
| hidden phase, post-two-changes | unmeasurable | **1.0000** |

Nothing about the environment or the representations changed between the two audits. The
readout did.

## §0 Canonical corrections, recorded

1. **The global unidentifiability claim is retracted.** Hidden phase is not unidentifiable
   from every single observation.
2. **The supported construct is existential**, and it holds: legally reachable states exist
   with the same complete public observation, different hidden phases, and different
   same-action outcomes. **14** such pairs were constructed and certified.
3. **Reset frames reveal initial polarity** and are **12.93%** of the previous evaluation
   rows. A two-line rule reads polarity off them at **1.0000**.
4. The current-frame baseline is reported separately per stratum (below).
5. **The previous readout's negative representation findings are not accepted as
   attribution results.** They are reported as properties of that probe.
6. **Geometry rejection is provisional** and the earlier claim is withdrawn pending the
   qualified-readout comparison.
7. **No Stage 1A world-model training has begun.** No final Scale-1 seed was opened.

## §1 The public packet, and what aliases under it

The complete agent-visible packet is `visual`, `language_goal_tokens`, `scalar_sensors`,
`previous_action`, `action_result`, `timestamp_ns`, `modality_masks`, `audio_slots`.
Evaluator-only state is `polarity`, `switch_crossings`, `initial_polarity`, `position`,
`simulator_step`, `last_blocked`.

Over **5007** legally reachable states from 60 layouts at depth 7:

| level | fields hashed | classes | pairs | same step | diff step | post/post | diff phase | diff phase + outcome |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | frame only | 2758 | 2249 | 1228 | 1021 | 2249 | 2249 | **2249** |
| B | + scalar sensors | 2758 | 2249 | 1228 | 1021 | 2249 | 2249 | 2249 |
| C | complete packet | 4993 | 14 | 14 | 0 | 14 | 14 | **14** |
| C− | complete minus timestamp | 4947 | 60 | 14 | 46 | 60 | 60 | 60 |

No reset/reset or reset/post-reset pairs occur at any level: a reset frame carries the
polarity stripe and therefore cannot alias against anything.

Two things this settles. **Scalar sensors carry nothing** — level B is identical to level A.
And **`previous_action` and `action_result` do most of the disambiguating**, collapsing 2249
pairs to 60. Partial observability in this environment is real but rare; the packet already
carries most of what fixes phase.

### The timestamp channel (J1)

`procedural_visual_v2.py:297` sets `timestamp_ns = self._step`, so the public packet carries
the simulator step. `ObservationPacket.canonical_dict` omits `timestamp_ns`, so the packet
*digest* does not see it — a pair hash built from that digest would alias states an agent
reading the packet can separate.

This is reported at both hash levels rather than resolved by choosing one. The channel
destroys **46** of 60 alias pairs, and it is exactly the different-step ones, as it must be.
Step correlates **0.268** with accumulated crossing count and **−0.006** with phase directly.
The specification's pin — that the simulator step must not enter the public packet — is
therefore **violated**, and the packet or environment should be versioned before training.
The recurrence certificate survives either way, so this does not block J2.

## §2 The identifiability claim, stated correctly

**A. Reset observability.** Initial phase is directly visible on reset frames; a two-line
rule recovers it at 1.0000, and on post-reset frames the same rule sits at 0.5198.

**B. Partial-observability certificate.** At least one legally reachable complete-public-
observation equivalence class contains multiple hidden phases with different action
outcomes — 14 of them, all four actions differing in 13 of the 14. Example: layout 90000,
cell (7,5), both members at step 7, crossings 1 versus 2, polarity 0 versus 1, successors
`[77,90,101,89]` against `[101,89,77,90]`.

**C. Empirical predictability.** Phase remains statistically predictable on other frames
because prior hidden dynamics changed visible state. This is why the previous audit's
`current_frame_only` arm scored above chance and why calling that a leak was wrong.

**D. History value.** Correct history beats both controls on the accumulation task:
1.0000 against 0.5506 reversed and 0.6315 shuffled-events.

The recurrence gate is B plus D. Both hold.

## §3 The bounded readout family

Parameter cap **250,000**, fixed before any validation exposure.

| # | readout | class | parameters | role |
|---|---|---|---:|---|
| 1 | hand-coded renderer-aware oracle | raw pixel | 0 | calibration only |
| 2 | translation-equivariant CNN, cell heatmap | raw pixel | 35,601 | position |
| 3 | object/relation decoder, frozen relation head | raw pixel | 35,650 | events |
| 4 | shared slotwise MLP with upsampling | slot | ≤ cap | events from slots |
| 7 | two-frame event classifier | temporal | 35,633 | negative result, retained |
| 8 | GRU over decoded events | temporal | 62,912 | accumulation |
| 9 | exact parity accumulator | temporal | 0 | positive control |

Readouts 5 and 6 (slot cross-attention; matched token-grid CNN) were **not run**. That is a
gap, not a pass: the slotwise MLP is the only slot readout exercised, so a slot-arm failure
cannot yet be attributed to the interface rather than to that one readout.

## §5 Direct causal-chain results

Split A is information presence (same layout families, held-out trajectories); split B is
systematic generalisation (held-out layouts, appearance fixed at the canonical seed).

| target | metric | A | B | control |
|---|---|---:|---:|---|
| agent cell, hand-coded | exact accuracy | 1.0000 | 1.0000 | — |
| agent cell, learned CNN | exact accuracy | 1.0000 | 1.0000 | chance 0.0069 |
| agent mask | F1 | 1.0000 | 1.0000 | — |
| switch mask | F1 | 0.9789 | 0.9809 | — |
| switch crossing, naive two-frame | balanced accuracy | 0.5872 | 0.5854 | retained negative |
| switch crossing, relation decoder | balanced accuracy | **1.0000** | **1.0000** | shuffled labels 0.4534 |
| switch crossing, relation decoder | F1 / Brier | 1.0000 / 0.0000 | 1.0000 / 0.0000 | constant 0.5000 |
| hidden phase, end-to-end | accuracy | — | **1.0000** | reversed 0.5506 |
| hidden phase after ≥1 switch | accuracy | — | **1.0000** | baseline 0.502 |
| hidden phase after ≥2 switches | accuracy | — | **1.0000** | baseline 0.546 |

Position R² is retired as a gate. It was a prerequisite for one particular differencing
readout, not a property of the task, and exact-cell accuracy is the honest measure.

### Appearance shift, reported separately

Both the hand-coded oracle and the learned CNN collapse to **0.0000** under appearance shift.
Both are tuned to a fixed palette, and this is a real limitation of the qualified readout —
it is a *within-appearance* result. It is reported here rather than folded into the
generalisation number.

## §6 Required event fidelity

Under the independent-error model, `P(correct parity | n) = (1 + (2p−1)^n)/2`. Over the
empirical validation distribution of `n` (mean 1.094, max 5; histogram
`{0:437, 1:366, 2:223, 3:104, 4:24, 5:11}`), reaching the pre-registered phase target of
**0.90** requires event accuracy **0.8964**.

| event accuracy p | closed form | simulated |
|---|---:|---:|
| 0.70 | 0.7689 | 0.7682 |
| 0.80 | 0.8276 | 0.8395 |
| 0.90 | 0.9031 | 0.9082 |
| 0.95 | 0.9485 | 0.9528 |
| 0.99 | 0.9892 | 0.9897 |

Measured event accuracy is 1.0000, implying phase 1.0000 — and the end-to-end run confirms
1.0000 rather than assuming it. The independence model is optimistic and is used as a design
target, not a gate.

## Bugs, corrections and environment findings

1. **Renderer occlusion.** `render_v2` draws switches and then paints the agent over them, so
   the switch a agent is standing on is absent from the current frame: measured directly, a
   switch cell reads `(177,44,147)` when the agent is away and `(41,161,141)` — the agent's
   colour — when it is on it. "Am I on a switch" is not a single-frame question in this
   environment. Reading the switch mask from the previous frame, and excluding the occluded
   cell from the decoder's supervision so it is not trained to hallucinate a colour that is
   not present, moved event balanced accuracy from **0.4525 to 1.0000**.
2. **Simulator step in the public packet.** `timestamp_ns = self._step`, omitted from
   `canonical_dict`. Recorded above; the packet should be versioned before training.
3. **The previous audit's R² ≥ 0.99 gate was wrong in kind.** It was derived from the
   noise-amplification behaviour of a differencing readout over regression estimates. The
   qualified readout does not difference regression estimates; it classifies cells. Exact-cell
   accuracy of 1.0000 makes the differencing analysis moot, and the gate is retired.
4. **Appearance was confounded with layout.** The agent is drawn as `255 − palette[0]` and the
   palette was tied to the layout seed, so every layout had a different agent colour and
   cross-layout localisation required an appearance-invariant rule. Holding appearance at the
   canonical seed — as §4 directs — gives one agent colour across layouts and is what makes
   the generalisation split answer the intended question.
5. **The naive two-frame classifier is retained as a negative result.** At 0.587 balanced
   accuracy with the same parameter budget, it shows that the object-relation *structure*,
   not capacity, is what recovers the event.

## §8 Geometry: still UNKNOWN, and why

The geometry comparison was run with the qualified relation head but with a **slot** decoder
(readout 4: a shared slotwise MLP, then nearest upsampling to the cell grid). Its `raw@geometry`
control is what disqualifies it.

| arm | agent exact-cell (B) | switch F1 (B) | event balanced accuracy (B) | CI vs majority |
|---|---:|---:|---:|---|
| qwen @ 4×4×256 | 0.1114 | 0.0000 | 0.5000 | [−0.054, +0.054] |
| qwen @ 8×8×64 | 0.3657 | 0.5053 | 0.5958 | [−0.032, +0.089] |
| qwen @ 8×8×256 | 0.3686 | 0.5392 | 0.6372 | [−0.040, +0.103] |
| gemma @ 4×4×256 | 0.1029 | 0.0297 | 0.5165 | [−0.040, +0.063] |
| gemma @ 8×8×64 | 0.3714 | 0.4627 | 0.5992 | [−0.069, +0.046] |
| gemma @ 8×8×256 | 0.3600 | 0.4937 | 0.6485 | [−0.009, +0.097] |
| **raw @ 4×4×256** (control) | 0.0943 | 0.1241 | 0.5307 | [−0.043, +0.063] |
| **raw @ 8×8×64** (control) | **0.3714** | 0.5848 | 0.6797 | [−0.043, +0.094] |
| **raw @ 8×8×256** (control) | 0.3714 | 0.5538 | 0.6451 | [−0.057, +0.080] |

**The raw control fails.** The same pixels that the qualified convolutional readout decodes at
exact-cell **1.0000** are decoded at **0.3714** by the slot readout. The information is
identical; the readout is not.

And the failure is structural, not a matter of training. A slotwise MLP followed by nearest
upsampling assigns every cell inside a slot the *same* logit, so the argmax is arbitrary
within a slot and accuracy is capped at one over the cells per slot:

| geometry | cells per slot | ceiling | observed |
|---|---:|---:|---:|
| 4×4 | 9.00 | 0.111 | **0.111** |
| 8×8 | 2.25 | 0.444 | 0.371 |

The 4×4 arms sit exactly on their ceiling. So these numbers measure the upsampling scheme,
not the interfaces.

By the specification's own rule — *"a readout family that cannot recover raw-pixel position or
switch location on dynamics_clean is not qualified to judge slot representations"* — **J6 and
J8 are UNKNOWN, not FAIL**, and the geometry-rejection claim from `b19b3b6` stays withdrawn
rather than being replaced with a new one. Readouts 5 (slot cross-attention) and 6 (matched
token-grid CNN) were not run and are the specific missing work.

This is the same error the previous audit made, caught this time by a control that was put in
deliberately to catch it.

## §10 Gates

| gate | status | basis |
|---|---|---|
| J0 | pass | suite and provenance below |
| J1 | pass | packet aliased with and without the timestamp; the channel is reported, and the leak recorded |
| J2 | pass | 14 legally reachable full-packet certificates |
| J3 | pass | hand-coded 1.0000 / learned 1.0000 exact-cell on held-out layouts |
| J4 | pass | event balanced accuracy 1.0000, CI [+0.246, +0.246], never trained on event labels |
| J5 | pass | end-to-end phase 1.0000 against a pre-registered 0.90 |
| J6 | **unknown** | slot readout fails its own raw-pixel calibration (0.371 vs 1.000) |
| J7 | pass | correct 1.0000 vs reversed 0.5506, shuffled-events 0.6315 |
| J8 | **unknown** | geometry cannot be decided from a disqualified readout |
| J9 | **unknown** | intervention non-inferiority withdrawn, not re-measured |
| J10 | pass | negative results and prior objections retained |

**8 pass, 0 fail, 3 unknown.**

## Decision

The specification's branches do not cover this case exactly, because the raw-pixel family
passed while the *slot* readout family failed its own calibration. The nearest correct
classification is **readout architecture failure, confined to the slot readout family** — not
slot/interface information loss, which would require a qualified slot readout to establish.

**Stage 1A-1 is NOT unblocked.** The 87-workload screen is not launched.

What changed is that the blocker is now small and specific. The chain is fully recoverable
from raw pixels; what remains is to build a slot readout that clears the same raw-pixel
calibration the pixel readout cleared — slot cross-attention or a matched token-grid CNN,
either of which can localise within a slot — and then rerun §8. Two further items are
outstanding before training: the `timestamp_ns` packet leak should be versioned, and the
qualified readout is currently within-appearance only.

## Narrow supported claim

> With appearance held fixed, agent position, switch location, switch-crossing events and
> hidden phase are all recoverable end-to-end from raw frames on held-out layouts, at exact-cell
> accuracy 1.000, event balanced accuracy 1.000 and phase accuracy 1.000, using a
> 35,650-parameter object-relation readout that is never shown a hidden value or an event
> label. Legally reachable states nevertheless exist — 14 of them here — that share a complete
> public observation, differ in hidden phase, and lead to different outcomes under the same
> action, so the environment is genuinely partially observable at those states. Nothing is
> established about the slot interfaces or about slot geometry.

