# Scale-0 Frozen Development Run Matrix

Status: canonical Scale-0 development-throughput contract
Applies to: `phase-2-continuous-world-model`
Final-evaluation status: none; these are development seeds and workloads

## Purpose

This file is the single source of truth for the Scale-0 execution matrix. Scale
0 measures interface correctness, attribution, throughput, restart, and local
resource feasibility. It does not select a winning representation and does not
support a capability claim.

Any change to this matrix must be committed before the first affected workload
runs. Once any cell has run, changing a factor, budget, tolerance, seed, or
resource ceiling invalidates all earlier cells in that matrix; the entire
matrix must restart under a new version.

## Frozen factors

### Frozen encoders

| Encoder ID | Initial model family | Role |
|---|---|---|
| `qwen3_vl_4b` | Qwen3-VL 4B | multimodal instruction/visual feature path |
| `gemma3_4b` | Gemma 3 4B | independent multimodal control path |

S0.2 must pin the exact official model revision, license record, preprocessing
digest, tokenizer/processor revision, dtype, and local runtime before the
dataset is encoded. If either family cannot run faithfully, Scale 0 stops. A
replacement is permitted only through a reviewed pre-run amendment to this
file; it is never selected after comparing matrix results.

Both encoders are frozen. Their parameters are inherited capability and are
reported separately from SHWM trainable parameters.

### Representation arms

```text
continuous
discrete
hybrid
```

### Trainable sizes

```text
50,000,000 parameters ±1%
200,000,000 parameters ±1%
```

The count includes every trainable projector, representation parameter,
recurrent/dynamics parameter, uncertainty module, and prediction head used in
the workload. It excludes frozen encoders, nonparametric exact verifiers, and
non-trainable planner code. The actual tensor count is authoritative; names
such as “50M” and “200M” are rejected outside the tolerance.

### Primary latent interface width

The primary 12-cell matrix uses width `512`. Width is not treated as semantic
capacity or an AGI scale.

The required dimension-sensitivity control uses widths `256` and `1,024` for
the 50M hybrid arm under both encoders. Internal widths must be adjusted so the
total trainable count remains within the same ±1% 50M tolerance. The existing
512-width hybrid/50M cells serve as the middle control.

### Development seeds

```text
6600
6601
6602
```

These seeds control initialization and minibatch order only. They are not
Phase-2 final seeds. The raw development dataset and split manifest are fixed
across all cells.

## Required cells

The primary matrix is:

```text
2 frozen encoders
× 3 representation arms
× 2 trainable sizes
= 12 cells

12 cells × 3 development seeds = 36 throughput runs
```

The dimension control adds:

```text
2 frozen encoders
× 1 hybrid representation
× 1 trainable size (50M)
× 2 added widths (256, 1,024)
× 3 development seeds
= 12 additional runs
```

Total mandatory Scale-0 training workloads: **48**. Unit tests and one-batch
schema checks are not counted as matrix runs.

## Fixed development data

Generate and seal exactly `100,000` transitions before encoder caching:

| Environment | Transitions |
|---|---:|
| deterministic controlled fixtures | 50,000 |
| one procedural visual adapter | 50,000 |

Within each environment, collection is:

```text
30% random                 15,000
25% scripted/oracle        12,500
25% current Sentinel       12,500
20% uncertainty-seeking    10,000
```

All arms receive the same raw transition IDs, branch groups, split manifest,
actions, outcomes, and order before seed-controlled minibatch permutation.
Encoder arms differ only in their frozen feature transform and required
projector. There are no per-run online environment interactions.

The two controlled fixtures are independent:

1. **Action intervention fixture:** restore the identical full simulator state,
   force two different legal actions, and require different observable
   successors. This isolates action conditioning.
2. **Belief aliasing fixture:** construct the same current observation from two
   different hidden histories, apply the same action, and require different
   observable successors. This isolates recurrent belief/history sufficiency.

Hidden simulator state is used only by the generator/evaluator and never enters
model input.

## Fixed optimization workload

Every matrix run uses:

```text
sequence length:                  32 transitions
sequences per batch:              32
optimizer updates:                200
transition positions processed:   204,800
optimizer:                        AdamW
learning rate:                    3e-4 constant
betas:                            (0.9, 0.95)
epsilon:                          1e-8
weight decay:                     0.01
gradient clipping:                global norm 1.0
trainable weights/activations:    bf16
optimizer accumulators:           fp32
```

All nine declared SHWM loss components are computed with plumbing weight `1.0`
for this throughput workload, including boundary separation. These coefficients
are not claimed to be optimal and no capability comparison may be drawn from
the resulting loss values.

If a backend cannot implement the frozen precision or optimizer semantics, the
matrix stops before that cell; it does not silently substitute another regime.

## Fixed planning dry run

After each training workload, use the same deterministic planner adapter and
evaluator-required probe set:

```text
horizons:                    5, 10, 25
planner invocations:         100 per horizon
candidate sequences:         64 per invocation
online environment actions:  0
```

Every proposed external-action path must pass the exact authority/verifier
bridge. Required evaluator probes are frozen independently of any probes the
model requests. Log exact model calls, expanded nodes, verifier calls,
rejections, abstentions, and wall time.

## Matching rule

“Matched” means all of the following simultaneously:

| Quantity | Rule |
|---|---|
| trainable parameters | target size within ±1% |
| raw transitions and split | exact same IDs and manifest |
| optimizer updates | exactly 200 |
| sequence length and batch | exactly 32 and 32 |
| per-run online interactions | exactly 0 |
| planner invocations/candidates | exact same counts |
| verifier-required probes | exact same frozen set |
| development seeds | all three; no dropped failed seed |

Actual FLOPs, wall time, peak memory, serialized state, and cache size are
**measured outcomes**, not quantities silently equalized after execution. Scale
0 reports them and makes no representation-quality comparison. Scale 1 must
define a separate equal-cumulative-compute capability protocol before it runs.

## Hard local resource envelope

Measured after model weights are locally available:

```text
peak unified memory per process:  ≤112 GiB
total Phase-2 artifact storage:   ≤200 GiB
hard timeout per matrix run:      2 hours
total 48-run matrix wall clock:   ≤72 hours
latent-cache build wall clock:    ≤8 hours
```

Run cells sequentially unless a separate resource proof shows parallel runs
remain inside the same peak-memory and total-compute accounting. Thermal
throttling, retries, compilation time, and failed runs remain in the report.

Exceeding a ceiling is a Scale-0 stop, not permission to shrink one arm after
seeing its results. A revised envelope requires a new pre-run matrix version
and a complete rerun.

## Gate

Scale 0 passes only if:

- all 48 mandatory workloads complete;
- all three seeds are retained for every cell;
- parameter and evidence matching rules hold;
- no Phase-2 final seed or evaluator answer leaks;
- restart and artifact checks pass;
- the exact Sentinel full suite remains green;
- every hard resource ceiling holds;
- the tracked tree is clean at the reported commit.

Passing unblocks a preregistered Scale-1 experiment design. It does not prove a
world-model capability gain.
