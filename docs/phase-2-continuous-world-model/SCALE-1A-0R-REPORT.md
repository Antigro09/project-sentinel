# Scale 1A-0R — matrix attribution audit

Status: **S1.2 passes on intervention only. The hidden-phase clause is
retracted.** No non-oracle interface clears both clauses, and the interfaces that
carry intervention information carry no measurable hidden-state information.

Authoritative verdict: `artifacts/shwm/scale1/s1-2-verdict.json`

## The correction

The Scale-1A-0 report said **Gate S1.2 (v2): PASSED**. It does not survive this
audit, for two reasons that are both mine.

**The gate checked its clauses independently.** "Some interface supports
intervention" and "some interface recovers hidden phase" were satisfied by
*different* interfaces, and reported as a pass. A Stage 1A-1 model receives one
interface, not the union of several, so that pass described nothing a model could
inherit. The gate now requires one interface to clear both.

**It judged on point estimates.** `qwen3_vl_4b_spatial_slots` clears a 0.05
hidden-phase threshold at +0.070 — and the paired episode interval for that same
number is `[-0.009, +0.141]`.

## B. The qualification table, with episode intervals

`dynamics_clean`, 95% paired intervals bootstrapped over **episodes** — frames
within an episode share a level, a palette and a trajectory, so resampling frames
would report an interval far narrower than the evidence supports.

| interface | intervention | 95% interval | excl. 0 | hidden phase | 95% interval | excl. 0 |
|---|---:|:--|:--:|---:|:--|:--:|
| **qwen3_vl_4b_spatial_slots** | **+0.402** | [+0.294, +0.503] | **yes** | +0.070 | [−0.009, +0.141] | no |
| **gemma3_4b_spatial_slots** | **+0.384** | [+0.249, +0.496] | **yes** | −0.018 | [−0.073, +0.031] | no |
| fixed_random_spatial_projection | −0.000 | [−0.000, +0.000] | no | +0.119 | [+0.031, +0.196] | yes |
| qwen3_vl_4b_mean_pool | +0.012 | [−0.004, +0.025] | no | +0.108 | [+0.049, +0.162] | yes |
| small_learned_cnn | −0.000 | [−0.000, +0.000] | no | +0.090 | [+0.009, +0.161] | yes |
| raw_lowres_spatial | −0.032 | [−0.072, +0.001] | no | +0.078 | [+0.002, +0.155] | yes |
| gemma3_4b_mean_pool | −0.087 | [−0.222, +0.039] | no | +0.043 | [−0.007, +0.089] | no |
| `oracle` (evaluator-only) | +1.000 | [+1.000, +1.000] | yes | +0.413 | [+0.350, +0.480] | yes |

Shuffled-label controls sit within ±0.06 of zero for every cell. Full table with
absolute scores, baselines, episode and lineage counts, development and
validation sizes, probe architecture and solver configuration, and degeneracy
flags is in `artifacts/shwm/scale1/audit-bc.json`, reported separately for
`dynamics_clean`, `appearance_shift` and `crossed_shift`.

`legacy_v1_replication` is **not evaluated**: its 36.3% transition-tuple overlap
makes a capability number from it uninterpretable. It is retained for reproducing
Scale 0, not for qualifying an interface.

## C. The hidden-phase construct audit

Polarity margin over baseline, `qwen3_vl_4b_spatial_slots`:

| feature variant | all episodes | switch-crossing | **post-first-switch** | ≥2 changes |
|---|---:|---:|---:|---:|
| current_frame_only | +0.041 | +0.059 | +0.052 | +0.042 |
| **reset_frame_only** | **+0.149** | +0.091 | **−0.064** | +0.218 |
| correct_history | +0.067 | +0.062 | **−0.028** | −0.028 |
| shuffled_history | +0.005 | +0.003 | +0.004 | +0.004 |
| correct_actions | +0.081 | +0.071 | −0.004 | +0.028 |
| shuffled_actions | +0.097 | +0.093 | +0.028 | +0.032 |
| **structured_oracle** | **+0.537** | +0.543 | **+0.543** | +0.540 |

Read the `reset_frame_only` row first. It scores **+0.149 overall and −0.064
post-switch** — which is what an indicator-reader looks like, since post-switch
the initial value is provably the wrong answer. The overall hidden-phase margins
are substantially that indicator.

Then read the oracle row: **+0.543 post-switch**, flat across every condition. The
information exists and is readable. An interface failing there is failing to
carry it, not being asked something impossible.

Then everything between: on post-switch states, `correct_history` gives **−0.028**
and no admissible variant exceeds +0.052. Shuffled actions score *no worse* than
correct actions (+0.097 against +0.081), so the action history contributes
nothing either.

Pins, all passing: the reset indicator cannot solve the post-switch subset;
polarity is absent from the scalar sensors; `action_result` correlates with
polarity below 0.2; every invariance pair is reachable through executed legal
histories; the construct is nonvacuous post-switch.

## D. Visual token extraction

**The test that mattered: the visual slots are bit-identical across a language
goal change, max difference 0.000e+00, for both backbones.** They are
**vision-only**. Had they moved, the +0.402 intervention margin could have been
partly a language effect and the raw and CNN controls — which have no language
path — would have been handicapped by construction. No language-fusion module is
owed to them.

Extraction stage: **multimodal-projector output written into the language model's
input embedding sequence, before any decoder layer runs.** Not vision-tower output
(1024/1152 wide), not a transformer hidden state.

| | patches | markers in sequence | soft visual tokens | selected | grid |
|---|---:|---:|---:|---:|---|
| Qwen3-VL | 256 | 64 | 64 | 64 | 8×8 |
| Gemma 3 | 4,096 | **1** | **256** | 256 | 16×16 |

That asymmetry is the defect corrected in `106d899`: Gemma's
`processor.image_token_id` is the `<start_of_image>` marker appearing once, while
its slots carry `<image_soft_token>`. The span is the set of positions whose id
equals the candidate that actually repeats, so language, marker and padding
tokens are excluded by construction.

Slot resampling: 8×8 → 2×2 blocks → 4×4 slots; 16×16 → 4×4 blocks → 4×4 slots.
Projection: one frozen random matrix per interface, never trained. Cached slot
shape (16, 256).

## E. Multimodal necessity

Both directions certified on reachable states:

- **vision alone is insufficient** — same pixels, different instruction, best action 0 → 3
- **language alone is insufficient** — same instruction, different visual state, different best action
- shuffled instruction tokenises differently and changes the correct action
- action history is required on the hidden-phase case: same frame, same action, different successor

Audio remains declared and absent.

## S1.2 verdict

| clause | result | basis |
|---|---|---|
| intervention | **PASSES** | `qwen3_vl_4b_spatial_slots`, `gemma3_4b_spatial_slots`; intervals exclude zero |
| hidden phase | **FAILS** | no interface with an interval excluding zero survives post-switch conditioning |
| one interface clears both | **none** | — |
| **S1.2** | **intervention-only pass** | |

## What this predicts about the 50M screen

Stated before the screen runs, so the result is a test of a prediction rather
than a rationalisation after it:

- **R3 (action conditioning beats no-action and shuffled-action on intervention)
  is likely to pass.** Intervention information is robustly present in both slot
  interfaces with intervals excluding zero.
- **R4 (recurrence contributes on same-frame/different-history cases) is
  predicted to FAIL** on these interfaces. There is direct evidence: on
  post-switch states, correct history scores −0.028 while the oracle scores
  +0.543, and shuffled actions score no worse than correct ones.
- **R5 (planning beats reactive) is unknown** and the screen is the right
  instrument for it.

If R4 fails as predicted, the correct response is not a larger model. It is an
interface that carries hidden state — the oracle proves the information is there
in the same 16×256 slot budget, so the loss is in the slot encoding, not in the
task. That is a 1A-0-class fix, not a Scale-1 one.

## Narrow supported claim

> On the v2 qualification task, coordinate-preserving pretrained visual slots
> retain counterfactual-action information that mean pooling removes, with
> episode-level intervals excluding zero for both backbone packages.

Not supported: any hidden-belief claim, any world-model claim, any encoder or
representation winner, any planning claim. The intervention result is a statement
about **interfaces** — which ones could support a dynamics model, not that one
works.
