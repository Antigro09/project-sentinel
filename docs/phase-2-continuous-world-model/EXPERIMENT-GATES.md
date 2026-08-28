# SHWM Experiment Gates and Falsifiers

## Evidence policy

Every gate is frozen before its final split is generated. Development success
is not a phase pass. A phase report must include commit, full-suite result,
tracked-tree status, seeds, task counts, controls, uncertainty, failures, and
the exact claim that is and is not unblocked.

## Scale matrix

| Scale | Question | Central comparison | Pass gate | Main falsifier |
|---:|---|---|---|---|
| 0 | Can the pipeline run and be audited locally? | frozen 48-workload matrix: two sizes × three representations × two encoders plus dimension controls | every cell/seed completes under the singular matching and hard-resource contract | resource, leakage, restart, matching, or attribution failure |
| 1 | Does representation affect useful behavior? | continuous vs discrete vs hybrid under a separately frozen equal-cumulative-compute contract | selected arm improves planning/calibration/transfer, not loss only | prediction difference without behavioral difference |
| 2 | Does the world model earn its place? | reactive, no-action, model-free, DreamerV3, SHWM, oracle | higher success at equal interactions or fewer interactions to matched success | no interval-supported control gain |
| 3 | Does verification earn its place? | WM-only, symbolic-only, hybrid no verifier, hybrid verifier, verifier+memory | fewer real failures or better calibration without erasing planning gain | verifier only adds cost or abstention |
| 4 | Are action effects robust under interventions? | passive correlational vs action/intervention model | better interventional prediction and policy transfer | ordinary prediction improves while intervention fails |
| 5 | Does a shared core transfer? | frozen shared core vs reset/from-scratch | fewer interactions/steps in held-out environment; reset removes gain | adapter-specific or memorized gain |
| 6 | Does continual learning survive high-dimensional state? | structured memory vs replay/no-memory/shuffled/stale | transfer, retention, revision, bounded growth, restart | retrieval-only or replay-only apparent gain |
| 7 | Does one core work in GUI/code domains? | shared hybrid vs domain-specific/reset controls | transferable improvement without harmful cross-repository/domain reuse | separate specialist core needed |
| 8 | What scales? | three model sizes × three data sizes | monotone or interpretable capability curve with uncertainty | loss improves but capability stays flat |

## Required arms

At each relevant scale:

1. random frozen encoder;
2. pretrained frozen encoder only;
3. reactive policy on frozen features;
4. no action conditioning;
5. no recurrence;
6. model-free control;
7. matched Dreamer-style baseline;
8. continuous latent;
9. discrete latent;
10. hybrid latent;
11. world model without verifier;
12. verifier without learned world model where meaningful;
13. hybrid without memory;
14. hybrid with structured memory;
15. shuffled memory;
16. reset shared core;
17. oracle simulator;
18. larger-budget diagnostic;
19. equal cumulative compute control;
20. final no-leakage audit.

Not every arm runs in every scale, but every omission is justified before
results.

## Metrics

### Prediction and representation

- one-step latent/event/reward/terminal prediction;
- 5-, 10-, 25-step rollout error and coverage;
- action-effect discrimination;
- representation collapse/code utilization;
- observation-versus-intervention prediction gap;
- calibration and proper scoring rules;
- model-inadequacy detection.

### Control and planning

- real task success;
- real interactions to matched success;
- plan completion and repair;
- model exploitation gap: imagined minus real return;
- catastrophic confident errors;
- planner nodes/model calls/wall time;
- abstention and unnecessary-question rate.

### Transfer and continual learning

- forward and backward transfer;
- retention and forgetting;
- revision accuracy and collateral damage;
- negative transfer;
- memory bytes, growth slope, and useful performance per byte;
- retrieval precision/recall and stale/poisoned retrieval;
- restart delta;
- gain removed by core/memory reset.

### Resource attribution

- frozen and trainable parameter counts separately;
- training tokens/transitions and real interactions;
- optimizer steps and cumulative FLOP estimate;
- peak resident unified memory;
- cache payload, metadata, and index bytes;
- wall time, planner calls, and verifier calls;
- backbone-only performance.

## Statistical protocol

- Scale 0 uses exactly development seeds 6600–6602 and the workload in
  [SCALE-0-RUN-MATRIX.md](SCALE-0-RUN-MATRIX.md); later decisive gates use at
  least four untouched seeds;
- paired task or stream resampling when conditions share generated tasks;
- report point estimate and 95% interval;
- freeze noninferiority/equivalence margins on validation;
- correct or hierarchically model families of related metrics rather than
  selecting the best post hoc;
- report all preregistered seeds, including failures;
- use per-domain floors in integrated evaluation, not average only;
- distinguish accuracy-at-budget from compute-unmatched accuracy;
- publish no Phase-2 final seed until the freeze manifest is committed.

## Causal audit

The causal label is permitted only if the model is tested on forced actions or
mechanism changes not determined by the behavior policy. Required strata:

- same observed state, different forced action;
- training correlation reversed while action mechanism stays fixed;
- mechanism changes while visual appearance stays fixed;
- irrelevant appearance changes;
- policy changes while environment dynamics remain fixed;
- missing action support.

Report whether failure means insufficient coverage, incorrect parameters,
incorrect mechanism factorization, or missing representation.

## Verifier audit

Measure both correctness and probe coverage. Plant:

- observable mismatches the verifier must catch;
- latent mismatches that current probes cannot distinguish;
- incorrect event decoders;
- corrupted tool results;
- delayed failures;
- verifier timeouts;
- actions whose consequences are irreversible.

The verifier passes only if it improves real decisions, not merely detects
contrived errors after the task.

## No-answer-leakage gate

Before each task, audit that persistent state contains no future:

- target action sequence or program;
- expected observations or rewards;
- hidden mechanics or convention;
- evaluator predicates not available to the agent;
- clarification answers;
- final split labels;
- branch siblings from another split.

Frozen encoders are an inherited-data channel. Their known training scope and
contamination risk are disclosed; performance is not labeled autonomous
Sentinel learning merely because the encoder already represents an answer.

## Global falsifiers

The architecture hypothesis is not supported if:

- a reactive backbone-only agent matches SHWM under equal interactions;
- action conditioning does not improve intervention outcomes;
- imagined return is systematically higher than realized return and the
  verifier does not control exploitation;
- verification removes all benefit by universal abstention;
- representation arms differ only because budgets are mismatched;
- same-environment repetition explains transfer;
- gains vanish under new goals or action combinations;
- reset/shuffle controls retain the gain;
- memory creates large negative transfer;
- world-model scaling improves prediction but not behavior;
- exact Phase-1 capabilities regress;
- final failures cause edits to the generator, thresholds, or evaluator.

## Advancement rule

Passing one scale unlocks only its stated successor dependency. It does not
prove AGI progress in percentage terms. Scale 0 unlocks a representation
experiment; Scale 2 supports only a learned-planning contribution; Scale 5
supports only bounded cross-environment transfer; Scale 6 is the first place a
continuous-scale continual-learning claim can be evaluated.
