# Claude Code Implementation Specification: SHWM Scale 0

Status: authorized architecture scaffold only after Anthony explicitly starts
implementation. Do not infer authorization to train or run final tests from
this document.

## 1. Exact objective

Implement an additive, dependency-light contract and reproducible preflight for
learned action-conditioned latent world models. Preserve the existing exact
`sentinel.wm.WorldModel` and verifier behavior. End with measured throughput,
memory, restart, and leakage evidence; do not claim planning or transfer.

The singular workload, matching, seed, dimension, optimizer, planner, and
resource contract is
`docs/phase-2-continuous-world-model/SCALE-0-RUN-MATRIX.md`. This specification
does not override that matrix.

## 2. Base and branch

```text
repository: /Users/anthonycavero/Documents/Startup/project-sentinel-phase-2-continuous-world-model
branch: phase-2-continuous-world-model
exact base: 5205543b110ba6da2e3f6da30630809941f821c4
reference branch: phase-1-exact-reference
```

Do not edit the separate dirty `phase-1-verifier` checkout. Do not push unless
Anthony asks.

## 3. Required modules

```text
src/sentinel/wm/latent_contract.py
src/sentinel/wm/representations.py
src/sentinel/wm/belief.py
src/sentinel/wm/dynamics.py
src/sentinel/wm/events.py
src/sentinel/wm/uncertainty.py
src/sentinel/wm/cache.py
src/sentinel/wm/dataset.py
src/sentinel/wm/planner_bridge.py
src/sentinel/wm/verifier_bridge.py
src/sentinel/wm/versioning.py
src/sentinel/env/adapters/base.py
src/sentinel/env/adapters/synthetic_control.py
```

Add the first real procedural adapter only after checking dependency health and
license. Adapter implementation must not expose hidden simulator state.

## 4. Minimum data types

Use frozen dataclasses and typed enums. Canonical serialization rejects NaN,
Inf, mutable mappings, unknown fields, noncanonical float encodings where
hashes matter, missing source digests, and model-version mismatches.

Required types:

- `RepresentationKind`;
- `EncoderIdentity`;
- `ObservationEnvelope`;
- `LatentObservation`;
- `BeliefState`;
- `TransitionRecord`;
- `SequenceBatch`;
- `StructuredEvent`;
- `TransitionPrediction`;
- `UncertaintyDecomposition`;
- `VerificationResult`;
- `ModelUpdateEvidence`;
- `FreezeManifest`.

## 5. Representation arms

Implement deterministic fake and parameter-countable neural stubs for:

```text
continuous: real-valued deterministic + stochastic state
discrete: categorical stochastic state
hybrid: continuous perceptual state + categorical event/mechanism state
```

All expose the same `BeliefUpdater` and `ActionConditionedDynamics` protocols.
Parameter matching uses actual trainable tensors. Frozen parameters, buffers,
optimizer state, and estimated activations are separate fields.

## 6. Frozen encoder preflight

Frozen family A: Qwen3-VL 4B.
Frozen family B: Gemma 3 4B.

This document does not itself authorize a download, but these are the frozen
families for the matrix. Before acquisition record:

- official source and license;
- exact revision and weight digest;
- native preprocessing;
- MLX or alternative local-runtime fidelity;
- precision and quantization behavior;
- modality support;
- expected storage and memory.

If a faithful adapter is not locally supported, stop and record the
incompatibility before any matrix cell runs. A replacement requires a reviewed
pre-run matrix amendment; after a cell runs, the full matrix must restart. Do
not silently substitute a community conversion or larger model.

## 7. Cache contract

The cache is content addressed by raw input, full encoder identity,
preprocessing, precision, projector, and modality mask. It supports atomic
writes, integrity check, version rejection, size accounting, and restart.

Never load a cache entry whose key omits a factor that can alter the latent.
Measure payload, metadata, index, resident memory, and cache construction time
separately.

## 8. Dataset and split contract

`TransitionRecord` includes action propensity, collector policy, branch group,
taint, and environment generator identity. Assign split before branch
collection. All siblings stay in one split. Add tests for duplicate frames,
latent hashes, environment seeds, branch groups, task goals, and evaluator
fields across splits.

No target plan, hidden mechanic, future observation, expected test result, or
clarification answer enters model input.

## 9. Synthetic environment

Create a deterministic partial-observation environment with two independent
fixtures:

- restore the identical complete simulator state, force two different actions,
  and produce different observable successors; this is the action-conditioning
  fixture;
- produce the same current observation from two different hidden histories,
  apply the same action, and produce different observable successors; this is
  the belief/history fixture;
- observable success/failure probes;
- one state distinction outside the initial probe set;
- branch/restore support;
- a hidden-state field unavailable to model code.

This environment validates plumbing and falsifiers, not capability.

## 10. Model configurations

Create 50M and 200M target configurations for each representation. Unit tests
may instantiate tiny versions, but those are not matrix runs. Reject an arm
outside ±1% of the target trainable count.

The 12-cell primary matrix uses width 512. The required sensitivity control
uses widths 256 and 1,024 for the 50M hybrid arm under both encoders while
remaining within the same ±1% parameter tolerance. Do not label dimensions as
small/AGI scales.

## 11. Objective plumbing

Expose separately:

- next latent;
- multi-step latent;
- reward/progress;
- terminal/failure;
- inverse action;
- structured event;
- uncertainty calibration;
- observed/imagined consistency;
- boundary-separation loss.

The total is the exact configured weighted sum. Log all values and gradients.
Disable each term independently. NaN/Inf is a hard failure.

## 12. Planner and verifier dry run

Implement a pure rollout API and deterministic fake planner. Count every model
call, node, depth, and wall-time unit. Every proposed external action crosses
the existing authority/verifier path. Feed observable mismatches back as
counterexamples with provenance. Include a planted latent mismatch invisible to
current probes so coverage is not conflated with verifier correctness.

## 13. Uncertainty

Persist separate aleatoric, epistemic, and model-inadequacy values. A combined
decision score is derived but not stored in place of components. Include
`MISSING_REPRESENTATION` and `UNKNOWN_EVENT` as distinct states.

## 14. Restart

At a fixed midpoint:

1. persist declared state;
2. terminate process;
3. launch a clean process;
4. reload only declared files;
5. continue and compare with uninterrupted execution.

Plant an undeclared global cache and require the test to catch it. Corrupt each
checkpoint field and fail closed.

## 15. Reproducibility

- Hydra for configuration;
- MLflow local SQLite for run parameters/metrics;
- DVC for deterministic evidence artifacts;
- SHA-256 checksums;
- seed registry;
- dependency lock;
- no network during frozen run unless the manifest explicitly includes an
  immutable local model artifact.

## 16. Required tests

At minimum:

```text
test_exact_world_model_contract_unchanged
test_latent_record_round_trip
test_rejects_noncanonical_latent
test_encoder_identity_invalidates_cache
test_precision_invalidates_cache
test_branch_group_never_crosses_split
test_no_hidden_state_in_model_input
test_no_future_target_fields
test_action_blind_collision_fails
test_action_conditioned_collision_representable
test_observation_only_fails_hidden_history_alias
test_recurrent_belief_representable_on_hidden_history_alias
test_representation_parameter_budgets_match
test_loss_sum_exact
test_all_action_paths_cross_verifier
test_probe_coverage_separate_from_correctness
test_restart_matches_uninterrupted
test_corrupted_checkpoint_fails_closed
test_forbidden_global_state_detected
test_final_seed_locked_before_freeze
```

Run focused tests after each slice and the complete existing suite before the
Scale-0 verdict.

## 17. Three-seed preflight report

Report every one of the 48 mandatory workloads in the frozen run matrix; no
failed seed may be dropped. Report per encoder/arm/size/width/seed:

- seed and task count;
- actual frozen/trainable parameters;
- cold load and cache-build time;
- transition/model/planner throughput;
- peak unified memory;
- serialized cache payload/metadata/index;
- restart equality;
- exact-suite result;
- every error, retry, and excluded condition.

The report is `MEASURED: systems throughput only`.

## 18. Gate and stop

Scale 0 passes only if all 48 frozen development workloads complete locally
under one unchanged evaluator with the matrix's exact evidence/budget rules and
hard 112 GiB peak-memory, 200 GiB artifact, two-hour per-run, 72-hour total,
and eight-hour cache ceilings; exact tests remain intact, restart is verified,
and leakage is absent.

Stop on any prerequisite failure. Do not sample final seeds or begin Scale 1 in
the same task. Return explicit `Scale 1 unblocked` or `not unblocked`.
