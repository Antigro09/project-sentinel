# Scale 1A-0 — observation interface and hidden-state testbed

Status: **complete and frozen.** The feature gate passes on `dynamics_clean`, the
v2 environment carries a hidden variable that is genuinely hidden and genuinely
exercised, and Stage 1A-1 has a defined matrix. No pretrained encoder is declared
a winner, and no world-model capability is claimed.

Freeze manifest digest: `sha256:24e2df3eabbcf7c1ac71bc22a48e4e2650cfa4c726138ed10f6ce75382096dcf`
Artefacts: `artifacts/shwm/scale1/`

## A. The record

| item | status |
|---|---|
| X64 | closed at the controlled hidden-convention level, 2026-08-27 |
| Phase-1 exact reference | frozen at `5205543` |
| X65A | latent-identity work reached; not closed as a whole |
| Scale 0 | passed at `f694c23` |
| `c890702` | Scale-0 audit and feature-sufficiency investigation |
| first S1.2 stop verdict | **retracted** — the probe failed on raw pixels, an input that provably contains the target |
| second S1.2 reading | **confounded** by palette shift; every representation collapsed, pixels included |
| v1 `charge` probe | **invalid** as hidden-state evidence; charge is recoverable from public time |
| Scale-1 world-model training | not begun |

v1's generator and every Scale-0 digest are unchanged, and a test asserts it.

## B. Token geometry, reconciled

Both figures I reported earlier were wrong, in different ways.

| | processor patches | vision tower | after projector | token grid | sequence with image |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL 4B | 256 | 256 | **64** | 8×8 | 87 (64 visual + 23 language) |
| Gemma 3 4B | 4,096 | 4,096 | **256** | 16×16 | 287 (256 visual + 31 language) |

- **"49" was wrong**: it assumed the processor keeps the 224×224 image. It upscales to meet a 65,536-pixel minimum, giving a 16×16 patch grid and 64 tokens after the 2×2 merge.
- **"4096" was wrong**: that is Gemma's patch count. The multimodal projector pools it to `mm_tokens_per_image` = 256.
- **"87/287" were right but conflated** visual with language tokens.
- **The 16× encode-cost claim survives**: it is driven by vision-tower patches, 4,096 against 256.

The enabling consequence: both backbones emit a **square** token grid with a
recoverable row-major coordinate mapping, so a coordinate-preserving slot
interface is buildable for both without guessing a layout.

## C. v2 hidden state

`polarity` is drawn from a seed stream independent of layout and appearance,
flips on entering a switch cell, is rendered only on the reset frame, and shows
itself only through mirrored movement. No step, horizon, or remaining-step field
exists anywhere in the observation.

**Reachable certificate** (executed, not synthesised):

```text
history_a [2,1] -> polarity 0        history_b [3,2] -> polarity 1
same public observation, equal-length histories, different observation traces
same probe action -> different observable successors
```

**Language certificate**: at a reachable state the instruction changes the best
action from 0 to 3 with the pixels identical, so the goal cannot be read off the
frame.

Two defects were caught building it, and a third after:

1. Switches generated too far from the start to ever be crossed, so polarity never flipped.
2. The language test compared only the start state and concluded the task did not need language — both markers simply lie in the same direction from there.
3. **At three switches only 27.3% of episodes ever changed polarity.** In the rest it equalled its initial value, which the reset frame shows, so "recovering hidden state from history" was mostly reading an indicator. Density was raised to seven and the rate went to **76%**, with an audit now requiring ≥50% and a calibration arm that rejects the old 27% figure.

That third change was made after seeing a result, so the test for it is whether
it would have been made had the numbers been high. It would — a high
hidden-phase score with 27% exercise would have been *more* alarming, since it
would have meant the probe was reading the indicator rather than tracking state.
The crossing rate is a property of the testbed independent of any probe outcome,
and the change landed before the freeze.

## D. Splits

| stratum | appearance | layout | role |
|---|---|---|---|
| `dynamics_clean` | canonical, pinned | held out | **the attribution stratum** |
| `appearance_shift` | held out | trained | perception diagnostic; may not veto |
| `crossed_shift` | held out | held out | reported, never primary |
| `legacy_v1_replication` | — | — | replication only |

Lineage is what Scale 0 lacked: it checked whether a branch *sibling* crossed a
split, but a state reached by continuing from one branch is not a sibling and
still carries that clone's lineage. Crossing with it is now caught.

Writing the audit exposed my own rule as wrong. I first flagged any layout in
both `dynamics_clean` and `crossed_shift` — but those strata share held-out
layouts *by design*, which is what makes crossed a shift of the same layouts. The
real violation is a trained layout reaching a held-out stratum.

## E. Multimodal packet

Visual slots, tokenised language goal, scalar sensors, previous action, action
result, timestamp, modality masks, and a **declared-absent audio channel** — so a
later audio modality is a mask bit rather than an interface change after a gate
has passed. The goal uses a shared word-level vocabulary rather than each
backbone's tokenizer; otherwise comparing interfaces would partly be comparing
tokenizers.

## F. Eight interfaces, one shape

All emit `(16, 256)`. A matched shape is what makes the comparison a comparison:
more slots or a wider slot is more capacity, and more capacity cannot be told
apart from a better representation.

| interface | learned | trainable |
|---|---|---:|
| `raw_lowres_spatial` | no | 0 |
| `fixed_random_spatial_projection` | no | 0 |
| `small_learned_cnn` | **yes** | 167,104 |
| `qwen3_vl_4b_mean_pool` | no | 0 |
| `qwen3_vl_4b_spatial_slots` | no | 0 |
| `gemma3_4b_mean_pool` | no | 0 |
| `gemma3_4b_spatial_slots` | no | 0 |
| `oracle_structured_state` | no (evaluator-only) | 0 |

A fixed random projection is a frozen matrix drawn once and is not called a
learned representation. Mean pooling broadcasts its vector across all sixteen
slots, which is what pooling leaves once slots exist: the spatial arm with
position deleted.

**A defect worth naming**: Gemma 3's `processor.image_token_id` returns
`<start_of_image>`, a marker appearing once, while its 256 visual slots carry
`<image_soft_token>` from the config. Trusting that attribute isolated **one**
visual token instead of 256 and would have fed the slot interface a marker. The
span is now chosen by which candidate id actually repeats.

## G. Feature qualification

Mean margin over the frozen baseline on `dynamics_clean`, by group:

| interface | position | relation | legal actions | **intervention** | action effect | **hidden phase** |
|---|---:|---:|---:|---:|---:|---:|
| raw_lowres_spatial | −0.012 | −0.021 | −0.001 | −0.038 | +0.094 | +0.078 |
| fixed_random_spatial_projection | +0.027 | −0.000 | −0.001 | −0.000 | +0.110 | +0.119 |
| small_learned_cnn | +0.001 | −0.209 | −0.002 | −0.000 | +0.058 | +0.090 |
| qwen3_vl_4b_mean_pool | −0.028 | −0.003 | −0.004 | +0.004 | +0.135 | +0.108 |
| **qwen3_vl_4b_spatial_slots** | +0.112 | +0.052 | −0.003 | **+0.400** | +0.094 | +0.070 |
| gemma3_4b_mean_pool | +0.000 | −0.121 | −0.001 | −0.081 | +0.130 | +0.043 |
| **gemma3_4b_spatial_slots** | +0.102 | +0.059 | −0.001 | **+0.369** | +0.061 | −0.018 |
| `oracle` (evaluator-only) | +0.565 | +0.999 | +0.221 | +1.000 | +0.345 | +0.413 |

Controls: the oracle wins everywhere, so the probes can read the variables. The
shuffled-label arm sits at zero or below for every interface, so the protocol
does not leak.

**The clean result is intervention.** Only the coordinate-preserving slot
interfaces predict what a counterfactual action would do — +0.400 and +0.369,
consistent across all four actions individually. Every pooled interface, raw
pixels, the random projection, and the learned CNN sit at zero or below. That is
Decision 4 vindicated on its own terms: pooling does not merely lose accuracy, it
removes the capability entirely.

**The weak result is hidden phase**, and it should not be oversold. Margins run
+0.04 to +0.12 against an oracle at +0.413, so admissible interfaces capture
roughly a quarter of what is recoverable — and the slot interfaces are not the
best at it. The gate threshold is met; the channel is thin, and Stage 1A-1 should
not assume its belief state has much to work with.

`appearance_shift` shows *higher* intervention margins (+0.554, +0.493) than
`dynamics_clean`, because it reuses trained layouts by design and only the palette
moves. That is the strata behaving correctly, and it is also why that stratum
cannot veto the screen.

### Gate S1.2 (v2): **RETRACTED — see the 1A-0R verdict**

This section originally read **PASSED**. It does not survive the 1A-0R audit and
is corrected in
[`SCALE-1A-0R-REPORT.md`](SCALE-1A-0R-REPORT.md). Two things were wrong:

1. The gate checked its two clauses **independently**, so a pass could be — and
   was — assembled from two different interfaces. A Stage 1A-1 model receives one
   interface, not the union of several.
2. It judged on point estimates alone. With episode-level intervals the
   hidden-phase margin for the only interface clearing both thresholds is
   `+0.070 [-0.009, +0.141]`, which includes zero; conditioning on post-switch
   states drops it to `-0.028` against an oracle at `+0.543`.

The intervention result is unaffected and strengthens under intervals. The
hidden-phase result does not survive. **S1.2 passes on intervention only.**

## H. Frozen, and the Stage 1A-1 matrix

Frozen by source digest: the v2 environment, the packet schema, the splits, the
interfaces, the probe configuration, the controls, and the gate thresholds. Both
certificates are recomputed at freeze time rather than copied, since a
certificate true of an earlier environment is worse than none.

**Stage 1A-1: the 50M action-conditioned screen.**

```text
4 interfaces x 6 arms x 3 seeds = 72 workloads at 50M
```

| interfaces | why |
|---|---|
| `qwen3_vl_4b_spatial_slots` | passed intervention; encoder package A |
| `gemma3_4b_spatial_slots` | passed intervention; encoder package B, so S1.9 encoder scope is answerable |
| `fixed_random_spatial_projection` | the floor that matched both backbones at S1.2 and must be beaten, or a pretrained encoder is unjustified |
| `qwen3_vl_4b_mean_pool` | the Scale-0 interface, as the pooling ablation |

| arms |
|---|
| continuous, discrete, hybrid |
| no-action control |
| shuffled-action control |
| reactive frozen-feature control |

Seeds 6600, 6601, 6602. Identical transitions, optimiser budget, batch shape,
trainable target, planner, rollout budget, and stopping rule across every arm.

## What is not claimed

No pretrained encoder winner. No representation winner. No world-model
capability. The intervention result is a statement about **interfaces**, not
about a learned dynamics model — it says which interfaces could support one, not
that one works. Hidden-phase recovery is weak and is reported as weak. No Scale-1
training has begun.
