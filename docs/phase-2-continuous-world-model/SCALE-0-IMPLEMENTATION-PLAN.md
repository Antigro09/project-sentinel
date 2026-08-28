# Scale-0 Implementation Plan

## Outcome

Scale 0 ends with a reproducible throughput and attribution preflight. It does
not end with a claim that Sentinel understands a world, plans causally, or has
cross-domain capability.

## Inputs

- exact base commit `5205543b110ba6da2e3f6da30630809941f821c4`;
- existing exact `sentinel.wm` contract and verifier tests;
- this Phase-2 architecture contract;
- the singular frozen workload contract in
  [`SCALE-0-RUN-MATRIX.md`](SCALE-0-RUN-MATRIX.md);
- Qwen3-VL 4B and Gemma 3 4B as the two named frozen-backbone candidates,
  subject to exact-revision, license, and local-runtime verification before any
  matrix cell runs;
- two environment adapters: one tiny generated controlled environment and one
  procedural visual environment;
- no Phase-2 final test seeds; inherited X64H seed artifacts are out of scope.

## Non-goals

- no 15B trainable model;
- no 70B training or teacher requirement;
- no OSWorld or SWE-bench development;
- no audio/full video training;
- no continual-memory capability claim;
- no self-improvement;
- no changes to exact Phase-1 semantics;
- no final representation winner;
- no final benchmark score.

## Work packages

### S0.0 — Freeze and provenance

1. Record base commit, branch, Python/MLX versions, hardware, and worktree
   cleanliness.
2. Add a machine-readable freeze schema, but do not produce the final freeze.
3. Add taint labels for development, validation, final, oracle, and inherited
   pretrained data.
4. Test that a final seed cannot be loaded before a committed manifest grants
   access.

Acceptance: provenance test passes and the exact suite is unchanged.

### S0.1 — Latent contracts

Implement typed records and Protocols from `ARCHITECTURE.md` with canonical
serialization. Add deterministic fake continuous, discrete, and hybrid
representations.

Acceptance:

- round-trip equality;
- content hashes stable across restart;
- malformed masks, dimensions, or model versions fail closed;
- no array payload is accepted without source digest;
- exact `WorldModel` contract remains unchanged.

### S0.2 — Frozen encoder adapters

Implement the adapter, identity digest, preprocessing manifest, and health
check. Preflight the two frozen candidates: Qwen3-VL 4B and Gemma 3 4B, subject
to official license, exact revision, conversion, precision, and local support
verification. If one cannot be run faithfully, stop before encoding data. A
replacement requires a reviewed pre-run matrix amendment; after any cell runs,
the entire matrix must restart under the new version.

Measure:

- cold load time;
- peak unified memory;
- images/frames/tokens per second;
- deterministic repeatability at fixed precision;
- cache hit/miss throughput;
- raw feature size;
- behavior under missing modalities.

Acceptance: at least two independent encoder paths satisfy the same adapter
without evaluator changes. This is a systems gate only.

### S0.3 — Content-addressed latent cache

Cache key:

```text
sha256(raw observation)
+ encoder weight revision and digest
+ preprocessing digest
+ precision
+ projector digest
+ modality mask
```

Test stale cache rejection after changing any field. Report raw payload and
index/metadata separately.

Acceptance: one million 512-dimensional fp16 entries are estimated at 1.024
decimal GB raw; the implementation report must measure actual serialized and
resident sizes instead of repeating that estimate.

### S0.4 — Transition dataset

Implement immutable `TransitionRecord` and `SequenceBatch` schemas, branch
groups, collector propensities, and split manifests.

Required split order:

1. sample environment family/seed/dynamic;
2. assign split;
3. collect all branches inside the assigned split;
4. hash and seal the episode;
5. never split individual branches across train/test.

Acceptance: leakage tests detect duplicate raw frames, duplicate latent hashes,
shared branch groups, reused environment seeds, and target/evaluator fields.

### S0.5 — Environment adapters

Adapter A is deterministic and tiny, with two separate fixtures:

1. restore the same complete simulator state, force two different legal
   actions, and produce different observable successors;
2. produce the same current observation from two different hidden histories,
   apply the same action, and produce different observable successors.

The first isolates action conditioning. The second isolates recurrent belief
state/history sufficiency. Neither may expose hidden simulator state to model
input.

Adapter B is a procedural visual domain selected after compatibility preflight.
It must expose reset, step, legal actions, observable probes, and branch/restore
when the environment supports it.

Acceptance: exact replay matches adapter output; an action-blind model fails the
first fixture; a current-observation-only model fails the second fixture; no
hidden state is copied into model input.

### S0.6 — Frozen 50M/200M workload configurations

Implement configuration and parameter-count validation for:

- continuous state;
- categorical state;
- hybrid state.

Tiny implementations may be used only for unit tests and one-batch schema
checks; they are not matrix runs. A workload may be called 50M or 200M only
when its actual trainable tensor count is within the matrix's ±1% tolerance.
Frozen backbone parameters are reported separately.

The primary matrix uses width 512. The 50M hybrid sensitivity control also uses
256 and 1,024 while holding total trainable parameters within ±1%. Acceptance:
every required cell uses the exact data, optimizer, seed, interaction, planner,
probe, precision, and resource contract in `SCALE-0-RUN-MATRIX.md`. No
representation or training-quality claim is permitted.

### S0.7 — Objective and metrics plumbing

Implement every declared component, including boundary separation. For the
throughput workload use the frozen plumbing coefficient `1.0` for every term;
this is not a final coefficient selection. Log every component, rollout
horizon, event accuracy/coverage, reward/terminal metrics, calibration,
gradient norm, and peak memory.

Acceptance: disabling a component makes only its declared metric/path disappear;
the loss total exactly matches the weighted components; NaN/Inf fails the run.

### S0.8 — Planner/verifier dry run

Use fake dynamics to test CEM/beam/MCTS accounting, uncertainty penalties,
probe requests, rejection, counterexample storage, and abstention.

Acceptance: every path to external action passes the verifier/authority gate;
the evaluator-required probe set is independent of model-requested probes; the
exact 100 invocations × 64 candidates at each of horizons 5, 10, and 25 are
used for every run.

### S0.9 — Restart and artifact pipeline

Run mid-epoch and mid-episode restart tests. Persist only declared state. Use
Hydra configuration, MLflow local tracking, DVC artifact stages, and SHA-256
checksums.

Acceptance: uninterrupted and restarted deterministic fake runs match;
corrupted state fails closed; undeclared cached state is detected.

### S0.10 — Three-seed throughput preflight

Run all 12 primary cells on development seeds 6600, 6601, and 6602, then the 12
additional dimension-sensitivity runs specified by the canonical matrix: 48
training workloads total. Each run uses exactly 100,000 sealed development
transitions, 200 updates, batch 32, sequence length 32, no online interactions,
and the frozen planning dry run. Do not use Phase-2 final seeds.

Report:

- wall time and throughput per stage;
- peak and steady unified memory;
- cache size and hit ratio;
- model/optimizer/activation estimates versus measured resident memory;
- rollout and planning calls per second;
- failures, retries, thermal throttling observations;
- exact full-suite result;
- tracked-tree cleanliness.

Gate: all 48 workloads complete without changing the evaluator or matching
rules, no seed is dropped, and each run stays within 112 GiB peak unified
memory, 200 GiB total artifacts, two hours per run, 72 hours for the matrix,
and eight hours for cache construction.

## Suggested commit sequence

1. `SHWM S0.0: contracts and provenance`
2. `SHWM S0.1: frozen encoder adapters and cache`
3. `SHWM S0.2: transition dataset and leak checks`
4. `SHWM S0.3: environment adapters`
5. `SHWM S0.4: representation and model configs`
6. `SHWM S0.5: objective, planner, and verifier plumbing`
7. `SHWM S0.6: restart and reproducibility`
8. `SHWM S0.7: three-seed throughput report`

Each commit includes focused tests. The final Scale-0 handoff includes full
suite output, commit hash, dirty-tree status, measured limits, and an explicit
unblocked/not-unblocked verdict for Scale 1.

## Stop conditions

Stop Scale 0 and diagnose if:

- exact Phase-1 tests regress;
- the adapter exposes hidden simulator state;
- cache identity is not reproducible;
- action-blind and action-conditioned arms receive different evidence;
- action intervention and hidden-history aliasing are collapsed into one test;
- branch groups cross splits;
- any parameter, data, optimizer, interaction, planner, probe, or seed rule in
  the frozen matrix cannot be matched;
- one candidate backbone consumes the local budget before a trainable model;
- restart depends on undeclared state;
- a final seed or evaluator answer becomes visible;
- a claimed 50M/200M model does not match actual trainable parameters;
- any hard matrix resource ceiling is exceeded.

The response to a stop is a narrower design or corrected premise, not automatic
scaling.
