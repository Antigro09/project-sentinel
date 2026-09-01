# S1.2 Feature Sufficiency — result

Status: **the stop does not fire, but Stage 1A is not started.** The pooled
encoder interface does preserve action-relevant state, so the brief's stop
condition is not met. Three findings below are design questions the frozen
Scale-1 spec has to answer first, and one of them undercuts the reason for using
a pretrained backbone at all.

Artefact: `artifacts/shwm/scale1/feature-sufficiency.json`

## The instrument, and why the first answer was wrong

The first pass used a linear probe and reported that nothing — not the encoder,
not raw pixels — could recover the agent's position. That was an artefact of the
instrument. Raw low-resolution pixels certainly contain the agent, which is drawn
as a coloured cell, so a probe that cannot read it there is not calibrated and
its negatives elsewhere mean nothing.

Two things were added. A **positive control** whose features contain the spatial
answers by construction: it now scores +1.000 on agent and goal position and
+0.998 on relative offsets, so the machinery demonstrably works. And a
**random-Fourier ridge**, giving nonlinear capacity while staying closed-form, so
nothing differs between representations except the features themselves.

A run whose oracle control does not clear 0.95 prints `NOT CALIBRATED` and its
table is not read.

## Result 1 — the interface preserves controllable state

Held-out **steps** within training levels: appearance and layout fixed, agent
position and step unseen. Margin over the frozen baseline:

| target | raw pixels | random projection | pooled Qwen | pooled Gemma | oracle |
|---|---:|---:|---:|---:|---:|
| agent_row | +0.338 | +0.278 | +0.354 | +0.368 | +0.847 |
| agent_col | +0.335 | +0.295 | +0.336 | +0.365 | +0.851 |
| goal_row | +0.797 | +0.797 | +0.794 | +0.797 | +0.797 |
| goal_col | +0.712 | +0.712 | +0.708 | +0.712 | +0.712 |
| delta_row | +0.923 | +0.915 | +0.881 | +0.900 | +0.999 |
| delta_col | +0.930 | +0.929 | +0.886 | +0.919 | +0.999 |
| manhattan | +0.781 | +0.777 | +0.756 | +0.817 | +0.746 |
| legal_action_count | +0.291 | +0.305 | +0.286 | +0.462 | +0.344 |

Relative offsets at R² 0.88–0.92 and goal position essentially perfect. **S1.2's
stop condition is not met**: the pooled interface supports action-relevant state
prediction well above baseline.

## Result 2 — the pretrained backbones buy nothing here

The column that matters is `random projection`: a fixed random Gaussian map of
raw pixels, no pretraining, no learning. It matches or beats the 4B backbones on
almost every variable, and on `delta_row` it beats them outright (+0.915 against
Qwen's +0.881).

So on the variables that govern control in this environment, **4.4 billion
inherited parameters are worth approximately zero against a random projection of
the pixels**. That is not an argument against the backbone in general — it is an
argument that *this environment* cannot distinguish one, and that a Scale-1
attribution claim resting on inherited perception would have nothing to rest on.
The required control arm "random frozen encoder" is not a formality here; it is
currently winning.

## Result 3 — held-out appearance defeats every representation equally

On held-out **levels**, with unseen colour palettes, every representation
collapses to baseline on position — including raw pixels:

| target | raw pixels | pooled Qwen | tokens Qwen | pooled Gemma | tokens Gemma |
|---|---:|---:|---:|---:|---:|
| agent_row | −0.042 | +0.000 | +0.160 | −0.044 | +0.137 |
| delta_row | −0.034 | −0.014 | +0.001 | −0.003 | +0.007 |

This is not encoder loss. "Where is the agent" means "which cell matches *this
level's* agent colour", and no fixed readout can answer that for a palette it has
never seen. It affects pixels and backbones alike.

The consequence for Scale 1 is concrete: **evaluation stratum B (held-out layouts
and appearances) will measure appearance generalisation, not world-model
quality.** Reporting a world-model result on stratum B without this caveat would
attribute a perception failure to the dynamics model.

## Result 4 — pooling does lose information, measurably

Within one encoder and one forward pass, the only difference between `pooled` and
`spatial_tokens` is whether the sequence was collapsed. On held-out levels the
tokens win on three of four position variables (agent_row +0.160 against +0.000)
and heavily on the history variable. That comparison is internally valid whatever
the appearance confound does, because both arms face it identically.

So "pooling loss" is a supported classification, and a spatial-token adapter is
indicated — though Result 1 shows it is not *required* to clear the stop, and
Result 2 raises the prior question of whether a pretrained backbone is the right
input at all.

## Result 5 — the hidden variable is not hidden

`charge` is a deterministic function of the step count, and `steps_remaining` is
in the observation. Measured directly: `charge == step % 3` at every step.

A Scale-0 test asserted the observation is invariant to `charge` and passed —
correctly, because it varies charge while holding step fixed. That combination
never occurs in a real trajectory, where the two move together. The test is true
and vacuous, which is a worse failure than a red one.

The procedural visual adapter therefore provides **no genuine partial
observability**, and any Scale-1 claim about recurrent belief inferring hidden
state would be unsupported there. Fixing it in place would change the environment
generator digest and alter Scale-0's inputs, which is forbidden; it needs a
versioned v2 adapter chosen and frozen in the Scale-1 spec.

## What this means for the gate

**S1.2: the stop condition does not fire.** At least one frozen-encoder interface
preserves enough controllable state to support action-effect prediction above the
frozen baseline, by a wide margin, when appearance is held fixed.

**Stage 1A is nonetheless not started**, because three of the five results are
design questions that §13 requires frozen before evaluation, and deciding them
after seeing Scale-1 numbers would retract those numbers:

1. Does the Scale-1 design keep a pretrained backbone at all, given that a random
   projection matches it (Result 2)?
2. Does stratum B stay in the primary claim, given that it measures appearance
   generalisation rather than dynamics (Result 3)?
3. Does Scale 1 adopt a v2 adapter with a genuinely hidden variable (Result 5)?

None of these is answerable by more compute, which is why the response is a
question rather than a training run.
