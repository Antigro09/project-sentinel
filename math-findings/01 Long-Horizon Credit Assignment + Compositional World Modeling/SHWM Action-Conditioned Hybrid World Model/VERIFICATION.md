# SHWM Math and Setup Verification

Date: 2026-08-28
Branch: `phase-2-continuous-world-model`
Frozen base: `5205543b110ba6da2e3f6da30630809941f821c4`
Scope: theory, finite formal checks, reproducibility scaffolding, and repository
regression safety. No neural world model was trained and no capability gate was
run.

## Verdict

The SHWM research package is internally reproducible on the current machine,
its finite helper theorems compile in Lean 4 with Mathlib, and the complete
pre-existing Sentinel repository suite remains green in the isolated Phase-2
worktree.

This establishes only that:

- the proposed finite definitions and helper claims are mechanically coherent;
- the arithmetic, exact counterexamples, and toy numerical diagnostics produce
  the logged results;
- the Phase-2 documentation package does not regress the frozen codebase;
- the proposed Scale-0 experiment is sufficiently specified to implement.

It does **not** establish that SHWM improves planning, transfer, continual
learning, multimodal grounding, causal reasoning, or AGI-relevant capability.
Those claims remain `UNKNOWN` until the preregistered experimental gates run.

## Executed toolchain

The dedicated math environment was reused rather than modifying Sentinel's
application environment:

```text
/Users/anthonycavero/Documents/Startup/project-sentinel/.venv-math-research/bin/python
```

| Resource | Executed version | Purpose |
|---|---:|---|
| Lean 4 + Mathlib | Lean 4.34.0-rc2 | mechanically checked finite theorems |
| SymPy | 1.14.0 | symbolic recurrence, posterior, and arithmetic checks |
| NumPy / SciPy | 2.5.2 / 1.18.1 | exact arrays and rollout-parameter recovery |
| JAX | 0.11.1 | differentiable action-conditioning collision toy |
| CVXPy | 1.9.2 | constrained parameter-allocation diagnostic |
| NetworkX | 3.6.1 | verifier-path graph invariant |
| NumPyro | 0.21.0 | finite intervention posterior diagnostic |
| Matplotlib | 3.11.1 | required figures |
| Hydra | 1.3.5 | deterministic configuration loading |
| MLflow | 3.15.2 | local experiment trace |
| Hypothesis | 6.165.10 | derandomized finite helper properties |
| DVC | 3.67.1 | reproducible artifact pipeline |

Coq/Rocq, Isabelle/HOL, Mathematica, and Maple were unavailable. No secondary
formal or symbolic verification using those systems is claimed.

## Lean verification

Executed from the existing Mathlib workspace:

```text
/Users/anthonycavero/Documents/Startup/project-sentinel/.math-research-tools/elan/bin/lake env lean \
  "/Users/anthonycavero/Documents/Startup/project-sentinel-phase-2-continuous-world-model/math-findings/01 Long-Horizon Credit Assignment + Compositional World Modeling/SHWM Action-Conditioned Hybrid World Model/formal/SHWM.lean"
```

Result: exit code `0`; no `sorry`, `admit`, or unfinished proof.

Mechanically checked claims:

| Lean declaration | Exact finite claim | Status |
|---|---|---|
| `finitePrecisionLatent_cardinality` | a `d`-coordinate, `q`-bit code space has cardinality `(2^q)^d` | `REPRODUCED — mechanically checked in Lean` |
| `actionSequence_cardinality` | length-`h` words over `B` actions have cardinality `B^h` | `REPRODUCED — mechanically checked in Lean` |
| `verifierQuotient_sufficient_for_traceScore` | verifier-equivalent states have equal finite-horizon probe-trace scores | `REPRODUCED — mechanically checked in Lean` |
| passive-kernel lemmas | two kernels may agree on the observed action yet disagree under an intervention | `REPRODUCED — mechanically checked in Lean` |
| `weightedLoss_nonnegative` | nonnegative weighted component losses have nonnegative total | `REPRODUCED — mechanically checked in Lean` |
| `squaredDisagreement_nonnegative` | squared ensemble disagreement is nonnegative | `REPRODUCED — mechanically checked in Lean` |
| rollout recurrence theorem | repeated one-step error satisfies the finite geometric-sum bound | `REPRODUCED — mechanically checked in Lean` |
| `observableMismatch_rejected` | exact observable inequality is rejected | `REPRODUCED — mechanically checked in Lean` |
| non-injective-probe theorem | a missing probe can hide a latent mismatch | `REPRODUCED — mechanically checked in Lean` |

These theorems are intentionally finite and assumption-explicit. They do not
prove learnability or performance of a neural implementation.

## Symbolic checks

Command:

```text
python shwm_symbolic_checks.py
```

Result: pass. Logged in `results/symbolic-checks.json` and pinned by `dvc.lock`.

| Check | Result |
|---|---:|
| rollout recurrence residual | `0` |
| finite posterior normalization | `1` |
| action words for branching `4`, horizon `25` | `1,125,899,906,842,624` |
| one million 512-dimensional fp16 latents | `1.024 GB` decimal / `0.953674 GiB` binary |
| 15B parameters at 12 bytes per parameter | `180 GB` |
| 15B parameters at 16 bytes per parameter | `240 GB` |
| 70B raw int4 packed weights | `35 GB` |

The model-memory figures exclude activations, allocator overhead, framework
buffers, quantization scales, and runtime caches where applicable. They are
resource arithmetic, not evidence of a theoretical model-size requirement.

## Exact finite enumeration

Command:

```text
python shwm_exact_checks.py
```

Result: pass. Logged in `results/exact-checks.json`.

- Repeating the passive action for 0, 1, 2, 8, or 64 observations leaves the
  two-kernel posterior at `(0.5, 0.5)` in the constructed counterexample.
- One noiseless distinguishing intervention concentrates the posterior exactly
  on the compatible kernel.
- Under the deliberately narrow two-model, conditionally independent noise
  model with reliability `0.9`, two agreeing interventions yield posterior
  `0.987804878...`, above `0.95`.
- Action-sequence counts and rollout-error recurrences match exact arithmetic.

This is an existence counterexample to passive identifiability, not a claim
that all passive datasets are non-identifying.

## Property checks

Command:

```text
python shwm_property_checks.py
```

Result: six derandomized properties, 200 examples each, **1,200 configured
examples passed**. The properties cover finite code cardinality, action-word
growth, cache arithmetic, posterior normalization, rollout recurrence, and
exact verifier equality.

## Numerical and resource diagnostics

Command:

```text
python shwm_resource_checks.py
```

Result: pass. Logged in `results/resource-checks.json`.

| Diagnostic | Result | Scope limit |
|---|---|---|
| frozen Scale-0 matrix | 12 cells, 36 primary runs, 12 dimension runs, 48 workloads; 204,800 transition positions and 19,200 planner candidates per run | configuration arithmetic; workloads not executed |
| JAX action collision | conditioned MSE `0`; best action-blind MSE `1` | two transitions from one state |
| JAX belief-history alias | history-conditioned MSE `0`; best observation-only MSE `1` | same current observation/action, two histories |
| SciPy rollout fit | recovered `L = 0.9000000000004997`; residual `1.09e-12` | synthetic known recurrence |
| CVXPy allocation | `optimal`; total `200,000,000` parameters | engineering prior, not capability optimization |
| NetworkX authority graph | both proposal-to-action paths include verifier | declared graph only |
| NumPyro posterior | `(0.012195..., 0.987804...)`; normalized | finite two-model diagnostic |
| Hydra | seed, representation arms, dimensions loaded | configuration only |
| MLflow | local SQLite run recorded | local database intentionally uncommitted |

The five Matplotlib figures were generated and visually inspected:

- `training-state-memory-lower-bound.png`;
- `latent-cache-footprint.png`;
- `planning-sequence-growth.png`;
- `rollout-error-stability-map.png`;
- `intervention-identifiability.png`.

## Reproducibility checks

| Check | Result |
|---|---|
| `dvc repro` | all four stages passed |
| `dvc status` | `Data and pipelines are up to date.` |
| Python byte compilation | all four Python check files passed |
| JSON parse audit | four result JSON files parsed successfully |
| strict report structure | exactly one section each for A through I |
| whitespace audit | `git diff --check` passed |

The result JSON and PNG files are DVC-managed local outputs, not Git blobs. No
DVC remote was configured or used. A fresh checkout runs `dvc repro` against
the committed scripts/config/lock to regenerate them.

Three tooling corrections are retained in this record:

1. `dvc init --subdir --no-scm` was rejected because those flags cannot be used
   together. The repository-backed correction was `dvc init --subdir`.
2. MLflow 3.15 rejected the legacy file backend. The diagnostic was changed to
   a local SQLite tracking store and then passed. No mathematical result
   depended on either of those first two failed setup attempts.
3. After the independent reader revisions, `dvc repro` exposed that a bare
   `python` command was unavailable in a clean shell. Every DVC stage was pinned
   to the documented dedicated math interpreter; all four stages then reran
   successfully and updated `dvc.lock`. No mathematical result depended on the
   failed invocation.

## Repository regression verification

### First isolated full-suite run

Command:

```text
uv run pytest -q
```

Result: **486 passed, 5 skipped, 7 failed, 9 errors in 967.16 seconds**.

All 16 non-passing cases traced to one worktree setup defect: the new isolated
worktree did not contain the repository's ignored offline `environment_files`
asset bundle. No failure implicated a Phase-2 document, theorem, or script.

Correction: temporarily link the already-present local offline asset bundle
from the primary checkout into the isolated worktree. The link was removed
after the final suite and is not part of the commit; the source assets were not
modified.

### Targeted correction check

```text
uv run pytest -q \
  tests/test_env_determinism.py \
  tests/test_planner.py \
  tests/test_verifier.py
```

Result: **58 passed in 21.09 seconds**.

### Final full-suite rerun

```text
uv run pytest -q
```

Result: **521 passed, 1 skipped in 940.11 seconds (15:40)**.

This is the regression result for the Phase-2 setup branch. It is not a SHWM
capability result because Phase-2 runtime implementation has not begun.

## Evidence ledger

| Claim | Evidence label | Boundary |
|---|---|---|
| finite helper theorems compile | `REPRODUCED` | mechanically checked exact definitions in `formal/SHWM.lean` |
| symbolic and finite diagnostics pass | `MEASURED` | deterministic scripts and logged artifacts |
| original Sentinel suite remains green | `REPRODUCED` | 521 passed, 1 skipped in isolated worktree |
| Scale-0 package is implementable | `INFERRED` | detailed interfaces and gates; not yet coded |
| VQ-IBD/SHWM composition is novel | `HYPOTHESIS` | bounded audit only; global novelty unknown |
| SHWM improves planning or transfer | `UNKNOWN` | requires Scale 2–5 experiments |
| SHWM supports continuous-scale continual learning | `UNKNOWN` | requires committed X65 contract and Scale 6 |
| system is closer to AGI by a measurable percentage | `UNKNOWN` | no valid percentage metric is defined |

## Remaining disconfirmation conditions

The direction should be stopped or revised if any of the following occurs:

- action conditioning improves prediction metrics but not real planning;
- continuous, discrete, and hybrid arms tie on capability under equal budgets;
- the pretrained backbone explains the full gain in ablation;
- the verifier does not reduce real rollout failures or improve calibration;
- intervention performance does not beat the correlational control;
- transfer survives core/memory reset, indicating leakage or task repetition;
- final-seed access occurs before freeze;
- exact Phase-1 reference behavior is altered to make Phase 2 look better;
- local throughput cannot support the preregistered three-seed runs.

The next authorized implementation unit is Scale 0 only. No Phase-2/SHWM final
seed has been sampled; inherited X64H artifacts are outside this statement.
