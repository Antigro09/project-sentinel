# Scale 1A-0R-O — Appearance Identifiability, Visual Convention Learning, and Palette-Robust Event Extraction

**Appearance shift is two different problems, and this phase separates them.** Photometric
jitter is solved outright by training on it. The hidden per-episode palette convention is
*identifiable in principle* — an exact posterior concentrates from 9.49 to 1.15 bits on
exactly the evidence the learned arms receive — and **none of the learned detectors
recover it**. Per-frame permutation is unidentifiable by construction and is reported
UNRESOLVED rather than forced.

**6 PASS, 2 PARTIAL, 7 NOT_RUN.** Prospective training is **not** unblocked.

---

## 1. Provenance and the clean N reproduction (O0 — PASS)

| field | value |
|---|---|
| commit | `69b4cfcc934fcb3e372c3c6dcc5f8598ee75b802` |
| branch | `phase-2-continuous-world-model` |
| tracked modified | `.claude/worktrees/x35-novelty-trigger` only |
| **N source digests** | **all 10 `n_*.py` match the committed bytes exactly** |
| Phase-2 suite | 510 passed, 40.4 s |
| **required suite** | **984 passed, 4 skipped, 0 failed, 1061.3 s** |
| final Scale-1 seed opened | no |
| prospective model started | no |
| Stage 1A-1 matrix run | no |

**N reproduces bit-exact from the clean committed tree.** Re-running every module and
comparing against the committed artifacts:

| module | check | result |
|---|---|---|
| `n_dataflow` | N1, wiring, behavioural matrices | match |
| `n_gauge` | authored 1.0000, masked 0.5056, shuffled 0.5468, outcome-trained NOT_RUN | match |
| `n_multimodal` | N12, 65 contested keys, 0.5316901408450704 | match |
| `n_pathway` | N6/N7/N8; validation +0.489491, held-out +0.468313 | match |
| `n_aux` | 8 interfaces × 3 targets × held-out | **0 differing cells** |

No divergence, so no development-evidence split is needed.

---

## 2. Corrected interface terminology (§B)

`cell_aligned oracle` → **`cell_aligned_color_grid`**. It was never an oracle: it read the
environment's cell lattice out of *pixels* and was exactly as palette-bound as everything
else, which is why it also fell to 0.5000 under appearance shift in N.

The fixed-projection interface, stated exactly rather than called "random":

| property | value |
|---|---|
| spatial aggregation | exact 2×2 pixel-block mean, 24×24 → 12×12 |
| aggregation lossless | **yes** — the renderer emits each cell as a solid 2×2 block, so the block mean inverts the upsample byte for byte |
| projection | frozen `normal(0, 1/√in)`, shape (6, 64), seed 20002 |
| output grid / width | 12×12 / 64 |
| bytes per pair | 3456 → 864, ratio 4.0 |
| trainable parameters | 0 |
| honest name | `environment_aligned_pixel_aggregation_plus_frozen_projection` |

**A real oracle was added**: it consumes role one-hots (AGENT, SWITCH, WALL, GOAL_ALPHA,
GOAL_BETA, EMPTY) and no colour at all.

---

## 3. Appearance regimes (O2 — PASS)

All three render from one semantic role grid, with layout, policy and palette seeds
factored. 32 development / 32 validation / 32 replication palettes.

| regime | construction |
|---|---|
| PHOTOMETRIC_JITTER | fixed role→colour map; per-cell texture ±18 and global brightness ±25 |
| HIDDEN_PALETTE_CONVENTION | per-episode bijection role → colour pool, fixed within the episode |
| PER_FRAME_PERMUTATION | a fresh bijection every frame — the impossibility control |

**O1 — the semantic oracle is invariant to palette:** spread **0.0113** across all five
splits at level 0.9890, with the *seen* palette the lowest of the five. An absolute
threshold would have failed this for a reason unrelated to appearance; invariance is what
§B asks and what is tested. The split does not leak.

---

## 4. Palette identifiability (O3 — PASS)

The renderer is a per-cell lookup, so identical pixels means the two semantic trajectories
differ by a **role permutation**. The observational equivalence class is therefore exactly
the set of permutations whose permuted trajectory is still legal — enumerable over all
**720** permutations of six roles, not sampled.

| evidence | mean class | median | max | SWITCH pinned |
|---|---:|---:|---:|---:|
| one frame | 12.00 | 12 | 12 | 1.000 |
| frame pair + action | 6.33 | 4 | 12 | 1.000 |
| reset frame | 12.00 | 12 | 12 | 1.000 |
| short legal history | 4.33 | 4 | 12 | 1.000 |
| grounded calibration episode | **2.08** | 2 | 4 | 1.000 |
| complete permitted history | 2.08 | 2 | 4 | 1.000 |

The one-frame class of exactly 12 decomposes as 3! (agent and the two markers are all
singletons) × 2! (wall/empty). **SWITCH is pinned from a single frame by cardinality
alone**, because seven switch cells is a generator constant. Walls are pinned
behaviourally once the polarity sign is known: a blocked move says the target cell is a
wall, a completed move says it is not.

**The residual class after calibration is exactly `GOAL_ALPHA ↔ GOAL_BETA`**, with
`changes_the_event: False`. No visual or behavioural evidence in the permitted set
separates the two markers — only the language channel names them, which ties this audit
directly to §K.

**A correction.** I stated in-session that the event target is identifiable from one
frame. It is not: SWITCH is pinned there, but AGENT is not, and the event needs both. That
claim is withdrawn; the posterior curve below is the correct account.

---

## 5. Exact palette posterior (O5 — PARTIAL)

Prior: 720 permutations, 9.4919 bits. The likelihood is an indicator, so the posterior is
uniform over survivors and its entropy is exactly log₂ of the class size.

| calibration steps | class size | entropy (bits) | true-class mass | event identified |
|---:|---:|---:|---:|---:|
| 0 | 12.00 | 3.5850 | 0.0833 | 0.000 |
| 1 | 6.38 | 2.2819 | 0.2656 | 0.625 |
| 2 | 4.62 | 1.7087 | 0.3802 | 0.750 |
| 3 | 3.25 | 1.3231 | 0.4479 | 0.875 |
| 8 | **2.50** | **1.1494** | **0.4722** | **0.958** |

**O5 is PARTIAL, not PASS.** The pre-stated criterion was the event identified in ≥99 % of
episodes; it reaches 0.958. The posterior does concentrate hard — entropy 3.585 → 1.149
bits, true mass 0.083 → 0.472 — but loosening the threshold after seeing the curve is the
move this project keeps having to correct, so both numbers are recorded and the gate is
PARTIAL. The full mapping is **never** resolved within an episode, because the two goal
markers are permanently exchangeable.

---

## 6. Detection across regimes (O4 — PARTIAL, O11 — PASS)

32 palettes, 40 training layouts, 20 held-out layouts, 2 seeds. Balanced accuracy on the
retrospective event.

| arm | seen palette | unseen (validation) | unseen (replication) | photometric jitter | per-frame |
|---|---:|---:|---:|---:|---:|
| **14 semantic-role oracle** | 0.9803 | 0.9916 | 0.9901 | 0.9916 | 0.9916 |
| 1 fixed-palette detector | **1.0000** | 0.5766 | 0.5704 | 0.5781 | 0.5064 |
| 2 palette-augmented detector | 0.9474 | **0.5720** | 0.6054 | 0.5288 | 0.4903 |
| 2b photometric-jitter-trained | 0.5000 | 0.5839 | 0.5043 | **1.0000** | 0.4928 |

- **Photometric jitter is SOLVED** by training on it: 1.0000. Invariant appearance
  learning works when the invariance exists in the pixels.
- **Hidden-palette augmentation does not transfer**: 0.5720 against the fixed detector's
  0.5766 — no improvement at all, over 32 training palettes.
- **Per-frame permutation stays at chance** (0.4903) and the audit marks it unresolvable,
  so it is reported UNRESOLVED rather than as a confident semantic claim (O11).
- The fixed-palette detector reproduces N exactly: perfect on its own palette, chance off
  it.

**Why augmentation fails while the posterior succeeds.** The audit shows SWITCH is pinned
by *cardinality* — a global count over the frame. The detectors are local convolutional
heads over a frozen projection of colours; they cannot count cells. The information is
present on exactly the evidence they receive, and the architecture cannot use it. That is
a learner failure, not a testbed failure, and it is the phase's central diagnosis.

---

## 7. Gates

| gate | status | basis |
|---|---|---|
| O0 | PASS | sources match; every N module reproduces bit-exact; suite 984/4/0 |
| O1 | PASS | oracle spread 0.0113 across five regimes |
| O2 | PASS | regimes factored from one role grid; oracle invariance is the leak test |
| O3 | PASS | exhaustive 720-permutation audit with collision certificates |
| **O4** | **PARTIAL** | jitter solved (1.0000); hidden palette not (0.5720 vs 0.5766) |
| **O5** | **PARTIAL** | concentrates 3.585→1.149 bits; event identified 0.958 vs a pre-stated 0.99 |
| O6 | NOT_RUN | no recurrent appearance-memory arm built |
| O7 | NOT_RUN | no convention-transfer gain exists to destroy |
| O8 | NOT_RUN | blocked by O4/O6 |
| O9 | NOT_RUN | blocked by O4/O6 |
| O10 | NOT_RUN | blocked by O4/O6 |
| O11 | PASS | per-frame at 0.4903, reported UNRESOLVED |
| **O12** | **NOT_RUN** | the outcome-trained gauge is mandatory this phase and was not built |
| **O13** | **NOT_RUN** | strengthened multimodal population with paired intervals not built |
| O14 | PASS | every palette, regime, seed and unresolved example retained |

Two of the NOT_RUNs are failures to deliver rather than blocked gates. **O12 was mandatory
in this phase and I did not build it**, so the N gauge finding stands unchanged and
initial-state grounding remains conditional on authored evidence. **O13** was required to
replace N's +0.02 point estimates with paired intervals; N's estimates are carried and
explicitly do not satisfy the gate.

**Decision (§L), and two branches apply at once:**

- *photometric jitter passes but hidden palette convention fails* → **invariant appearance
  learning works, episode-level grounding does not**;
- *exact palette inference concentrates but learned appearance memory fails* → **the
  testbed is valid; learned convention inference is the blocker**.

The appearance-aware interface is **not** frozen and prospective training is **not**
unblocked.

---

## 8. Bugs and corrections

1. **My cardinality check rejected the truth.** The agent occludes whatever it stands on,
   including a goal marker, so a marker count of 0 is legal. Requiring exactly one
   rejected the true trajectory on every episode where the agent stepped on a marker —
   caught only by the assertion that the truth must survive its own evidence.
2. **The audit left WALL and EMPTY exchangeable** because nothing tested blocking. Adding
   wall grounding — a blocked move implies a wall, a completed move implies not — took the
   calibrated class from 4.00 to 2.08.
3. **O1's threshold tested the wrong thing.** An absolute >0.99 on the oracle failed at 32
   palettes for a reason unrelated to appearance: a larger, more varied evaluation set. The
   spread across regimes is the quantity §B asks about, and the *seen* palette scores
   lowest of the five.
4. **I over-claimed single-frame identifiability of the event** in-session. SWITCH is
   pinned from one frame; AGENT is not; the event needs both. Withdrawn above and recorded
   in the artifact.
5. **O5's criterion was met only on a softer reading.** Recorded PARTIAL with both the
   pre-stated 0.99 test it fails and the concentration evidence it passes.

---

## 9. The narrow supported claim

Under a per-episode hidden bijection from six semantic roles to a fixed colour pool:

- the observational equivalence class of the palette mapping is **exactly enumerable**,
  falling from 720 to 12 on one frame and to **2.08** after grounded calibration, with the
  residual being precisely the two exchangeable goal markers;
- an **exact posterior** over that family concentrates from 9.4919 to **1.1494** bits and
  identifies the event in **0.958** of episodes;
- a **semantic-role oracle** is invariant to palette (spread 0.0113), so the construction
  does not leak;
- **photometric jitter is solved** by training on it (1.0000);
- **no learned detector transfers to an unseen palette convention** (0.5720 with 32
  training palettes, against 0.5766 for a detector that never saw one).

Standing conditions: initial-state grounding remains **conditional on authored evidence**
(O12 not run); the multimodal contribution remains a **point estimate without an
interval** (O13 not run); no prospective prediction or planning result exists; no
dynamics-generalization claim is made.

---

## 10. Is prospective world-model training unblocked?

**No.** §L gates it on O0–O14 and seven are NOT_RUN, two of them — the outcome-trained
gauge and the interval-backed multimodal test — because this phase did not build them
rather than because they were blocked.

The substantive obstacle is sharper than "appearance is hard". The information needed to
pin the event under a hidden palette **is present in the evidence the learners receive** —
the exact posterior proves it on the same frames. What fails is the readout: SWITCH is
identified by a *global cardinality*, and a local convolutional head over a colour
projection has no way to count. The next phase should test a detector with a global
counting or set-level readout before adding any capacity, and should build the two arms
this phase owed: the outcome-trained initial-belief encoder, and the multimodal population
with paired intervals.
