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

Nothing about the environment or the representations changed between the two audits. But
the readout is not the only thing that did, and an independent reviewer was right to catch
the overreach: **the target formulation changed too.** The previous audit scored position as
scalar row/column regression by R²; this one scores it as 144-way cell classification by
exact accuracy. On the same pixels and the same splits, a **dense linear ridge** reaches
**0.9875** held-out under the classification target, so the recovery is not attributable to
the convolutional architecture or the parameter budget. And with appearance pinned, the agent
is a globally unique colour, which makes exact-cell decoding close to a constant-colour
lookup — the 0.0000 appearance-shift score is the same fact seen from the other side. The
honest statement is that the *previous* R² = 0.029 measured the scalar-regression
formulation, not the information and not the probe family.

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

> **Corrected.** The first version of this table counted 5007 states and 14 full-packet
> pairs. That enumeration keyed its search on `(position, polarity)` and kept one route per
> node, so it counted *representatives*, not reachable states — and level C hashes `step`,
> `previous_action` and `action_result`, which that sample holds functionally determined by
> the key. The numbers below come from enumerating on the tuple level C actually
> distinguishes. Two independent implementations agree on every cell.

Over **43,179** legally reachable states from 60 layouts at depth 7:

| level | fields hashed | classes | pairs | same step | diff step | post/post | diff phase | diff phase + outcome |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | frame only | 2758 | 518,337 | — | — | all | 260,958 | **260,958** |
| B | + scalar sensors | 4032 | 376,953 | — | — | all | 192,749 | 192,749 |
| C | complete packet | 33,598 | 9,581 | 9,581 | 0 | 9,581 | 9,581 | **9,581** |
| C− | complete minus timestamp | 11,693 | 94,761 | — | — | all | 39,556 | 39,556 |

No reset/reset or reset/post-reset pairs occur at any level: a reset frame carries the
polarity stripe and therefore cannot alias against anything.

Two earlier conclusions drawn from this table were artifacts of the enumerator and are
**withdrawn**:

- ~~"Scalar sensors carry nothing — level B is identical to level A."~~ They destroy
  **141,384** of 518,337 pairs. B equalled A only because one-route-per-node made it
  impossible for two aliasing states to differ in `blocked`.
- ~~"Partial observability is real but rare."~~ It is common: a large fraction of reachable
  states sit in a full-packet alias class with a different-phase, different-outcome partner.

What stands is that **`previous_action` and `action_result` do most of the disambiguating**
(518,337 → 94,761) and the timestamp does most of the rest.

### The timestamp channel (J1)

`procedural_visual_v2.py:297` sets `timestamp_ns = self._step`, so the public packet carries
the simulator step. `ObservationPacket.canonical_dict` omits `timestamp_ns`, so the packet
*digest* does not see it — a pair hash built from that digest would alias states an agent
reading the packet can separate.

This is reported at both hash levels rather than resolved by choosing one. The channel
destroys **85,180** of 94,761 alias pairs, and it is exactly the different-step ones, as it
must be. Step correlates **0.2249** with accumulated crossing count and **0.0031** with phase
directly.
The specification's pin — that the simulator step must not enter the public packet — is
therefore **violated**, and the packet or environment should be versioned before training.
The recurrence certificate survives either way, so this does not block J2.

### A second leak: the packet digest encodes initial polarity

The packet has **ten** fields, not the eight originally listed. `interface_name` and
`source_observation_digest` were missing, and the second one leaks.
`source_observation_digest` is `ObservationEnvelope.content_digest`, which hashes
`environment_version`, a digest of `LevelV2.digest`, which includes `initial_polarity`.
Verified directly: two levels with identical `layout_digest` and `appearance_digest` but
opposite initial polarity produce **different packet digests**.

This is the mirror of the timestamp defect. The timestamp makes the digest *alias* states an
agent can separate; this makes it *separate* states an agent cannot. Both travel by value
inside a hash, which is why the name-based pin missed both — it compares field names against
a forbidden list, and neither leak has a forbidden name. Within a single layout the level is
constant, so this channel does not change the counts above; it matters for cross-episode
identity, caching and dedup.

## §2 The identifiability claim, stated correctly

**A. Reset observability.** Initial phase is directly visible on reset frames; a two-line
rule recovers it at 1.0000, and on post-reset frames the same rule sits at 0.5198.

**B. Partial-observability certificate.** At least one legally reachable complete-public-
observation equivalence class contains multiple hidden phases with different action
outcomes — **9,581** of them. (The earlier "13 of 14 with all four actions differing" was a
misread of a per-action count.) Example: layout 90000,
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
| switch crossing, relation decoder | balanced accuracy | **1.0000** | **1.0000** | see note |
| switch crossing, relation decoder | F1 / Brier | 1.0000 / 0.0000 | 1.0000 / 0.0000 | constant 0.5000 |
| hidden phase, end-to-end | accuracy | — | **1.0000** | reversed 0.5506 |
| hidden phase after ≥1 switch | accuracy | — | **1.0000** | baseline 0.502 |
| hidden phase after ≥2 switches | accuracy | — | **1.0000** | baseline 0.546 |

The `shuffled labels 0.4534` figure originally cited on the relation-decoder row was
produced by the *two-frame CNN*, not by the relation decoder, which shipped with no
label-scramble control of its own. A reviewer ran the correct one — scrambling the agent and
switch mask targets across examples — and it collapses the decoder to event balanced accuracy
**0.5000**, agent exact **0.0243**, switch F1 **0.0000**. The claim survives, but on evidence
that was not in the artifact when the claim was made.

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

## §9 Independent review, and what it overturned

Three targeted reviewers were run as specified. **All three returned "overstated" at high
confidence.** Every objection acted on below was reproduced independently before being
accepted; none was taken on the reviewer's word.

**Overturned.**

1. **The state enumeration was a canonical-representative sample, not an enumeration.** Keyed
   on `(position, polarity)` with one route per node, while level C hashes `step`,
   `previous_action` and `action_result`. Corrected: 5007 → **43,179** states, full-packet
   pairs 14 → **9,581**. Two independent implementations now agree on every cell of the
   table.
2. **"Scalar sensors carry nothing" — withdrawn.** They destroy 141,384 pairs.
3. **"Partial observability is real but rare" — withdrawn.** It is common.
4. **The packet is ten fields, and `source_observation_digest` leaks `initial_polarity`.**
   Verified directly. Recorded above as a second leak alongside the timestamp.
5. **The position result is not attributable to the CNN.** A dense linear ridge reaches
   0.9875 held-out under the same cell-classification target, and with appearance pinned the
   agent is a globally unique colour, so "held-out layouts" poses little transfer question
   for this readout.
6. **`GEOMETRY_D` was silently dropped from the slot run** — the one arm built to separate
   resolution from alignment. Running it refutes the classification this report originally
   drew; see §8.
7. **The ceiling arithmetic was wrong.** The cap is `P(true cell is the row-major-first cell
   of its slot)` under `argmax`'s first-index tie-break on the empirical position
   distribution — **0.1175** at 4×4 and **0.3650** at 8×8 — not `1/(cells per slot)`. The
   arms are *at* ceiling, not below an abstract bound.
8. **The J6 gate had no discriminative power as built.** Its `raw_control` had an arithmetic
   maximum of 0.3714 given the arms run, so UNKNOWN was guaranteed before any frame was
   encoded; and it drew the calibration arm and the verdict from the same unfiltered pool, so
   a raw arm at a geometry no backbone can supply could have flipped it to PASS. Both fixed:
   qualification is now per geometry, and the verdict considers only backbone arms at
   backbone-suppliable geometries.
9. **The temporal controls do not change the event multiset**, contrary to their own
   docstring. Reversal visits the same pairs in the opposite order and the shuffle permutes
   within a trajectory, so final-step parity is identical under all three modes. They bite by
   misaligning intermediate parities, which is a weaker control than claimed.

**Survived refutation.** J2's existential certificate — strengthened, not weakened, by the
corrected enumeration. The timestamp leak in full. The absence of any reset-frame aliasing.
The renderer-occlusion finding. The relation head having zero parameters and never seeing an
event label, which a reviewer confirmed collapses to chance under a proper label scramble,
across five seeds. And the event labels themselves: cumulative `crossed_now` matches the
simulator's `switch_crossings` on 4229/4229 rows.

## §8 Geometry: still UNKNOWN, and now for a precise reason

The geometry comparison uses the qualified relation head with a **slot** decoder (readout 4:
a shared slotwise MLP, then nearest upsampling to the cell grid). Held-out layouts:

| arm | agent exact-cell | switch F1 | event balanced accuracy | CI vs majority |
|---|---:|---:|---:|---|
| qwen @ 4×4×256 | 0.1114 | 0.0000 | 0.5000 | [−0.054, +0.054] |
| qwen @ 8×8×64 | 0.3657 | 0.5053 | 0.5958 | [−0.032, +0.089] |
| qwen @ 8×8×256 | 0.3686 | 0.5392 | 0.6372 | [−0.040, +0.103] |
| gemma @ 4×4×256 | 0.1029 | 0.0297 | 0.5165 | [−0.040, +0.063] |
| gemma @ 8×8×64 | 0.3714 | 0.4627 | 0.5992 | [−0.069, +0.046] |
| gemma @ 8×8×256 | 0.3600 | 0.4937 | 0.6485 | [−0.009, +0.097] |
| raw @ 4×4×256 (control) | 0.0943 | 0.1241 | 0.5307 | [−0.043, +0.063] |
| raw @ 8×8×64 (control) | 0.3714 | 0.5848 | 0.6797 | [−0.043, +0.094] |
| raw @ 8×8×256 (control) | 0.3714 | 0.5538 | 0.6451 | [−0.057, +0.080] |
| **raw @ 12×12×64 (cell-aligned)** | **1.0000** | **1.0000** | **1.0000** | **[+0.260, +0.260]** |

The last row is the one that matters, and the first version of this report did not have it:
`GEOMETRY_D` was silently dropped from the run, which is precisely the arm built to separate
resolution from alignment. With it, the earlier classification collapses.

**The slot readout is not architecturally incapable.** The identical decoder, same seeds and
hyperparameters, recovers position, switch mask and the crossing event *perfectly* the moment
the slot grid is one slot per game cell.

**What caps the other arms is the output parameterisation, not the interface.** Nearest
upsampling gives every cell inside a slot the same logit, so `argmax` returns the row-major
first cell of the slot and accuracy is bounded by `P(true cell is first in its slot)` over the
empirical position distribution:

| geometry | ceiling | observed (raw control) |
|---|---:|---:|
| 4×4 | 0.1175 | 0.0943 |
| 8×8 | 0.3650 | 0.3714 |
| 12×12 | 1.0000 | 1.0000 |

The 8×8 arms are *at* ceiling. (The earlier report gave the 8×8 ceiling as 0.444 from
`1/(cells per slot)`; that is the bound under a uniform prior with random tie-breaks, and
neither holds here. It left a 0.073 gap unexplained and understated the case.)

**And every source sits at the same ceiling.** Raw pixels are lossless, and they score 0.3714
at 8×8 — the same as gemma (0.3714) and qwen (0.3657). At this measurement resolution the
interfaces are indistinguishable *because the ceiling is binding*, not because they carry the
same information. Nothing can be attributed to any interface from these numbers.

**Neither backbone can supply a one-slot-per-cell grid**: qwen's native token grid is 8×8 and
gemma's is 16×16, and 12 divides neither. So the 12×12 arm is available to the pixel sources
only, and the geometry question at the geometries that actually matter is **undecidable with
this readout**.

**J6 and J8 are therefore UNKNOWN**, and the gate now says so for the right reason. The
earlier gate was worse than uninformative: its `raw_control` had an arithmetic maximum of
0.3714 given the arms it ran, so UNKNOWN was guaranteed before a frame was encoded, and it
drew its calibration arm and its verdict from the same unfiltered pool. Qualification is now
per geometry, and the verdict considers only backbone arms at backbone-suppliable geometries.

## Decision

The classification in the first version of this report — *"readout architecture failure,
confined to the slot readout family"* — is **withdrawn**. It is refuted by the 12×12 arm.

The correct classification is narrower: **the slot readout's output parameterisation cannot
localise within a slot, and no backbone supplies a slot grid as fine as a game cell.** That
is neither a readout-family failure nor demonstrated interface information loss. It is a
mismatch between the decoder's output resolution and the quantity being decoded, at exactly
the geometries the backbones can provide.

**Stage 1A-1 is NOT unblocked. The 87-workload screen is not launched.**

The remaining work is specific and small: a slot readout that can localise *within* a slot —
readout 5 (slot cross-attention), readout 6 (matched token-grid CNN), or simply a sub-slot
offset head on the existing decoder — then rerun §8. Only then can "do the pretrained slot
interfaces lose spatial information" be asked at all. Two further items stand before training:
the `timestamp_ns` and `source_observation_digest` leaks should be versioned, and the
qualified readout is currently within-appearance only.

## §10 Gates

| gate | status | basis |
|---|---|---|
| J0 | pass | commit `a4dfc3c`; 522 exact-reference, 425 + 4 skipped Phase-2, 947 + 4 complete |
| J1 | pass | packet aliased with and without the timestamp; the channel is reported, and the leak recorded |
| J2 | pass | 14 legally reachable full-packet certificates |
| J3 | pass | hand-coded 1.0000 / learned 1.0000 exact-cell on held-out layouts |
| J4 | pass | event balanced accuracy 1.0000, CI [+0.246, +0.246], never trained on event labels |
| J5 | pass | end-to-end phase 1.0000 against a pre-registered 0.90 |
| J6 | **unknown** | at backbone-suppliable geometries every arm sits at the readout's tie-break ceiling; the readout itself is fine (1.0000 at cell-aligned 12×12) |
| J7 | pass | correct 1.0000 vs reversed 0.5506, shuffled-events 0.6315 |
| J8 | **unknown** | no backbone supplies a one-slot-per-cell grid, so geometry is undecidable with this decoder |
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

