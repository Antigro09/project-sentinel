# SHWM Scale-0 Handoff

Status: **Scale 0 passed.** All 48 mandatory workloads ran against both named
frozen backbones under the frozen matching contract, and all ten gate clauses
hold. This unblocks a preregistered Scale-1 experiment design and nothing else:
Scale 0 measures interface correctness, attribution, throughput, restart, and
local feasibility. It measured no capability, and no representation arm was
shown better than another.

Branch: `phase-2-continuous-world-model`
Base commit: `5205543b110ba6da2e3f6da30630809941f821c4`
Generated resource report: [`SCALE-0-RESOURCE-REPORT.md`](SCALE-0-RESOURCE-REPORT.md)

## 1. The gate

| clause | result |
|---|---|
| all 48 workloads complete | PASS |
| all three seeds retained for every cell | PASS |
| matching rules hold | PASS |
| no leakage detected | PASS |
| restart and artifact checks pass | PASS |
| hard resource ceilings hold | PASS |
| no undeclared process state | PASS |
| both frozen encoders runnable | PASS |
| is a matrix run | PASS |
| tracked tree clean for run inputs | PASS |

Two clauses cannot be settled inside the driver and are not asserted by it. The
exact full suite is run separately at the reported commit and its result is in
§2b. The no-final-seed condition is enforced structurally by
`FinalSeedGuard`, which refuses to load a final seed without a committed
post-freeze manifest; no such manifest exists.

### The stop that resolved

An earlier version of this document reported Scale 0 as **stopped** because
`google/gemma-3-4b-it` was licence-gated. That was a misdiagnosis, and it is
written up as a defect in §3.12 rather than deleted. The account had already
accepted the Gemma terms; the preflight saw `gated: manual` and inferred an
outstanding acceptance, when `gated` says only that the repository is gated and
the acceptance state is not observable without a credential. The real blocker was
a missing local access token. Once one was installed, both families preflighted
runnable at their pinned revisions and the matrix ran.

| Encoder | Repository | Revision | Licence | Frozen parameters |
|---|---|---|---|---:|
| `qwen3_vl_4b` | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec` | apache-2.0 | 4,437,815,808 |
| `gemma3_4b` | `google/gemma-3-4b-it` | `093f9f38` | gemma | 4,300,079,472 |

Those parameters are inherited capability. They are reported apart from the 50M
or 200M trainable budget, which they never enter, and every workload's trainable
count excludes them entirely.

## 2. Headline measurements

| quantity | measured | ceiling |
|---|---:|---:|
| workloads completed | 48 / 48 | — |
| cold latent-cache build (both backbones) | **3.67 h** | 8 h |
| matrix wall clock, warm cache | 25.9 min | 72 h |
| peak device memory, 200M arm | 3.94 GiB | — |
| process resident high-water | 12.15 GiB | 112 GiB |
| artifact storage | 0.97 GiB | 200 GiB |
| worst parameter drift from target | +0.0074% | ±1% |

The cold cache figure is the one the eight-hour ceiling is about. A confirmation
re-run against a warm cache rebuilt the dataset in 82 s, which says nothing about
that ceiling and is not the number to quote; see §3.17, where the fact that the
warm run overwrote the cold run's artefact is recorded as a defect.

## 2a. Implemented file map

Additive only. `sentinel.wm.contract` — the exact executable-world-model
contract — is untouched, and nothing in the Phase-1 path imports any of this.

**`src/sentinel/wm/`** — 24 new modules, 6,433 lines

| module | role |
|---|---|
| `versioning.py` | canonical JSON, content hashes, `ArrayDigest`, file digests |
| `latent_contract.py` | every typed record and Protocol; taint; the hidden-field tripwire |
| `events.py` | the frozen 12-member verifier-facing event vocabulary |
| `uncertainty.py` | calibration tables and ensemble disagreement |
| `provenance.py` | freeze manifest, taint ledger, final-seed guard |
| `matrix.py` | `SCALE-0-RUN-MATRIX.md` transcribed into executable form |
| `backbones.py` | preflight for the two named frozen families |
| `encoder.py` | frozen-encoder adapter, bf16 packing, the random-encoder control |
| `cache.py` | content-addressed feature cache with stale and corruption rejection |
| `dataset.py` | transition/sequence schemas, split manifest, leak audits |
| `collect.py` | the four collector policies, feature table, batch sampler |
| `authority.py` | the single-use token gate every external action crosses |
| `sizing.py` | parameter closed form and the 50M/200M solver |
| `models.py` | the three MLX representation arms behind one forward interface |
| `representations.py` | deterministic fake projections for all three arms |
| `belief.py` | deterministic fake recurrent belief |
| `dynamics.py` | deterministic fake action-conditioned dynamics, plus the action-blind control |
| `objective.py` | the nine loss components, coverage, and the finiteness check |
| `metrics.py` | rollout divergence, code utilisation, action-effect discrimination |
| `planner_bridge.py` | beam, CEM, and MCTS over a pure rollout with exact accounting |
| `verifier_bridge.py` | required-versus-requested probes, counterexamples, controller |
| `restart.py` | declared state, checksums, undeclared-global detection |
| `resource.py` | measured resources beside their estimates |
| `trainer.py` | the frozen training loop and the fp32-accumulator optimizer |

**`src/sentinel/env/adapters/`** — 4 files, 988 lines: the adapter contract with
hidden-state invariance, the deterministic controlled domain with both causal
fixtures, and the procedural visual domain with independent appearance and
mechanics seeds.

**`experiments/shwm/`** — 12 files, 1,648 lines: the Hydra config, the backbone
preflight, the dataset builder, the workload runner, the driver, the report
generator, the Qwen runtime probe, and the DVC pipeline.

**`tests/shwm/`** — 16 files, 4,012 lines.


## 2b. Test results and tree status

The gate clause the driver cannot settle itself. Run at the reported commit:

```text
uv run pytest -q
839 passed in 943.35s (0:15:43)
```

| suite | tests | result |
|---|---:|---|
| exact Phase-1 (`tests/`, excluding `tests/shwm/`) | 522 | all passed |
| Phase-2 Scale-0 (`tests/shwm/`) | 317 | all passed |
| **total** | **839** | **0 failed, 0 skipped** |

The exact half is unchanged in composition. `VERIFICATION.md` records the
pre-Phase-2 baseline as "521 passed, 1 skipped in 940.11s" -- the same 522 tests,
in the same wall time to within four seconds. The one skip passes here because
this checkout has the ignored offline asset bundle that the isolated setup
worktree did not; no test was modified, deleted, or weakened.

The exact reference branch was not touched. `phase-1-exact-reference` still
resolves to `5205543b110ba6da2e3f6da30630809941f821c4` and its reflog contains
only the entry that created it.

Tracked tree: **clean for run inputs**. One entry is dirty outside them --
`.claude/worktrees/x35-novelty-trigger`, a gitlink that was already modified
before this work began and that this task is forbidden to stage. It cannot
affect a run; §3.15 explains why the clause is scoped rather than absolute, and
the report lists the entry by name so the exemption is visible.

Untracked entries in tracked scope: none. The 18 GiB under `artifacts/` -- both
backbones, the latent caches, restart checkpoints, and the MLflow database -- is
gitignored and content-addressed by `artifacts/shwm/scale0/checksums.json`,
reproducible from the committed code, config, and DVC pipeline.

## 3. Negative findings and corrected defects

The pipeline was run rather than reasoned about, and running it is what produced
this list. Each entry is a defect that a green test suite did not catch.

### 3.1 The encoder cache could never hit, and the leak check could never fire

`ObservationEnvelope.digest` hashed the episode id along with the content. Two
identical frames reached by different routes therefore had different digests, so
the content-addressed cache missed every time and its hit ratio measured nothing
— and the duplicate-frame leakage check, which compares those same digests
across splits, was structurally incapable of finding anything.

Content and positional identity are now separate hashes. The cache and the leak
audit use content; provenance uses position. Across the sealed 100,000-transition
set the cache hit ratio is **0.728** — 46,443 distinct observations behind
100,000 transitions — where before it could only ever have been zero.

### 3.2 Transition-level split disjointness is unattainable in a finite domain

With disjointness now measurable, it turned out to be impossible for one of the
two families. The controlled adapter reaches a small set of observations, so
identical `(observation, action, successor)` tuples appear in both splits, and no
split procedure can prevent it.

Raising there would be a check satisfiable only by enlarging the environment. So
the audit now **raises** on the four structural violations (branch group across
splits, branch group across episodes, seed across splits, one episode step in two
splits) and **measures** content overlap, with a stricter gate that the
generative family asserts and passes at **0.00%**.

### 3.2b The controlled family's held-out split is not a generalisation test

Measuring the overlap rather than asserting it away produced a number that
matters more than the check did. At the sealed scale the controlled family shows
**97.7% observation overlap** and **36.3% transition-tuple overlap** between
train and held-out, because 50,000 transitions are drawn from an environment with
only **11,883 distinct transitions** in total. Its 9,241 held-out transitions are
therefore mostly present in training by arithmetic.

That is not a leak the split procedure caused and it is not fixable by splitting
differently. It is a statement about the environment: **the controlled adapter is
too small to support a held-out generalisation test at this data volume.** It
remains sound for what Scale 0 uses it for — exact replay, the two causal
fixtures, restart, and throughput — and Scale 1 must either enlarge it or treat
its held-out split as a replication check rather than a generalisation one. The
procedural visual family, at 0.00% overlap on both measures, is the one that
carries a generalisation claim.

### 3.3 The boundary loss was vacuous

`L_boundary` is a hinge, `max(0, m − d)`, at margin 1.0. In an unnormalised
512-dimensional space, two random projections sit about **17** apart, so the
hinge was satisfied by every pair and the term reported zero forever while
appearing to work.

The metric space is now the unit sphere, where the distance lies in [0, 2] and a
margin of 1.0 means something. The term also now reports pair count, active
pairs, and mean pair distance, because a zero from "no branch pairs", a zero from
"pairs present, none collapsed", and a real penalty are three different facts.

### 3.4 Two loss components can be negative, so the Lean lemma may not apply

`weightedLoss_nonnegative` is conditional on every component being non-negative.
`L_reward` and `L_calibration` are Gaussian negative log-likelihoods and can go
below zero. Replacing them with non-negative surrogates would let the variance
head win by reporting certainty — the exact failure a proper scoring rule
prevents — so the precondition is now **checked at runtime and reported per
step** instead of assumed.

### 3.5 MLX keeps optimizer moments in the parameter's dtype

The matrix freezes bf16 weights with fp32 accumulators. MLX's `AdamW` allocates
its moments with `zeros_like(parameter)` and does its arithmetic in the
gradient's dtype, so on a bf16 model every accumulator was bf16: eight
significand bits holding a running average across two hundred updates.

The matrix says a backend that cannot implement the frozen precision stops the
cell rather than substituting another regime. Implementing it was cheaper than
stopping, so the moments are float32, the update is computed in float32, and
only the resulting parameter is cast back.

### 3.6 A non-deterministic backward that a matching loss concealed

Two identical runs of the continuous arm diverged in one process. The loss
matched exactly for four updates and the gradient norm matched at every step;
only the weights moved apart.

The cause was the action embedding. As a gather, its backward is a scatter-add,
and the action vector is consumed once per dynamics block and again for every
imagined multi-step — so four rows accumulated thousands of contributions whose
atomic ordering is not fixed. The float32 loss reduction rounded the difference
away; the bf16 weights did not.

It is now a one-hot matmul. Same parameter shape, deterministic backward, and
all three arms are bit-reproducible on repeat and across a restart.

This one is worth dwelling on, because the restart gate would have failed with
every visible number agreeing.

### 3.7 MLX cannot differentiate through scatter indices

The straight-through categorical was written the obvious way, scattering a one
into a zero tensor. `mx.put_along_axis` has no VJP with respect to indices, so
the discrete arm raised at the first backward pass — and had the error been
softer, the arm would have trained with no gradient reaching its codebook. It is
now a broadcast equality against the argmax.

### 3.8 The latent cache costs 69% more than the plan's arithmetic

The plan estimates one million 512-wide fp16 latents at 1.024 decimal GB. That
covers the raw payload and nothing else. Measured over 46,443 real entries:

| component | bytes per entry | projected at 1e6 entries |
|---|---:|---:|
| payload (plan's estimate) | 1,024 | 1.024 GB |
| payload (measured, incl. `.npy` header) | 1,152 | 1.152 GB |
| index and metadata | 575 | 0.575 GB |
| **total** | **1,727** | **1.727 GB** |

So the real footprint is **69% above** the figure the plan quotes. The estimate
was not wrong; it was the payload term alone, and a 128-byte per-file header plus
a JSON index that costs more than half the payload again are what turn it into a
storage number. The report measures this rather than repeating the arithmetic.

### 3.9 A forced branch action does not carry the policy's propensity

Branch siblings are actions the collector forced, not actions the policy chose.
Recording the policy's propensity on them would misstate the behaviour
distribution — which Theorem 1 makes part of what any later interventional claim
rests on. Forced actions now carry uniform propensity and chosen actions carry
the policy's, and a test separates the two.

### 3.10 Smaller corrections

- The frozen matrix keyed its transition counts by a prose name
  (`deterministic_controlled`) while the adapter was `synthetic_control`, so the
  config and the contract could disagree silently. Now keyed by adapter name.
- `assert_no_hidden_state` compared every snapshot field against the observation
  and flagged legitimately observable ones. Hidden state is now tested by
  **invariance** — vary the hidden field across its domain, require one
  observation digest — with a planted leaky adapter as the calibration arm.
- The verifier dry run sampled a prefix of the records, which are collected one
  family at a time, so the visual family was never verified. Now strided.
- Two instrumentation costs were sitting inside the measured loop: the sampler
  recomputed its permutation (one hash per sequence) on every update, and the
  cache size report walked tens of thousands of payload files per workload. Both
  are hoisted, so the throughput numbers measure the model rather than the
  harness.
- Peak memory was reported as the larger of MLX's device peak and `ru_maxrss`.
  `ru_maxrss` is a process high-water mark and monotonic by construction, so
  across the 48-workload sequence it climbed to 12.15 GiB and attributed that to
  a 50M model whose device peak was 1.63 GiB. The two are now reported
  separately: the device peak per workload, the process high-water against the
  per-process ceiling.

### 3.11 The verifier's coverage metric has no headroom yet

Probe coverage comes back at exactly 1.000 for every verification, and that is a
limitation rather than a result: the recorded probe set is precisely the
evaluator-required set, so there is nothing a model could fail to probe. Lemma 4
says accuracy and coverage must be separate numbers, and they are separate in the
implementation — but until the environments expose probes beyond the required
six, the coverage column cannot move and is not yet evidence of anything. The
detection rate is meaningful: 256 deliberately corrupted predictions out of 512,
all 256 caught.


### 3.12 The preflight named the wrong blocker

The first version of this handoff reported that Scale 0 was stopped because
`google/gemma-3-4b-it` needed its licence accepted. That was wrong. The account
had **already** accepted it -- the model page reads "You have been granted access
to this model", and the authenticated Hub connector could read the repository's
`config.json` throughout.

The preflight had seen `gated: manual` and concluded the terms were outstanding.
`gated` says only that the repository is gated; whether *this account* has
accepted is a different fact, and one that is not observable without a
credential. Conflating them sent a reader to accept a licence they already held
while the real blocker -- no access token on this machine -- went unnamed.

The two facts are now distinguished. With no credential present the preflight
reports the missing token and says explicitly that acceptance is unobservable
from where it stands, rather than guessing.

### 3.13 The two frozen families differ in encode cost by 16x

Not a defect, but a measured asymmetry large enough that a reader should not
meet it by surprise. Under each model's own official preprocessing:

| family | image size | patch | merge | vision tokens | measured rate |
|---|---|---:|---:|---:|---:|
| Qwen3-VL 4B | dynamic | 16 | 2 | ~49 | 70.2 obs/s |
| Gemma 3 4B | fixed 896x896 | 14 | none | 4,096 | 3.4 obs/s |

Gemma does roughly 84x the vision-token work per image, and its half of the
sealed cache takes hours where Qwen's takes minutes. Shrinking Gemma's input
would close the gap and was not done: preprocessing is part of the encoder
identity, and tuning it to improve a number is the substitution the matrix
forbids. Wall time is a measured outcome rather than a matched quantity, so this
does not threaten the matching rule -- every matched quantity is fixed after
caching. It does consume about half the eight-hour cache ceiling at this data
volume, which is a fact Scale 1 needs before it chooses a larger one.

### 3.14 The gate never checked restart equivalence

The Scale-0 verdict was a single boolean over failures, matching, resource
envelope, and undeclared state. Restart was not among them -- so a run whose
restarted weights disagreed with the uninterrupted ones would have reported a
pass, against a matrix that lists restart among its gate conditions.

All ten clauses are now evaluated and reported by name. Two of them cannot be
settled inside the driver and say so instead of being assumed: the exact full
suite is run separately at the reported commit, and the no-final-seed condition
is enforced structurally by the seed guard.

### 3.15 Tree cleanliness failed on a file that cannot affect a run

The cleanliness clause asked whether the tracked tree was clean, full stop, which
an unrelated worktree pointer this task is forbidden to stage would have failed.
Cleanliness is now evaluated over the paths whose state can change what the run
does, and anything dirty outside them is listed in the report by name so the
exemption is visible rather than silent.

Writing the test for that exposed a worse one underneath. `git_state` stripped
the whole `git status --porcelain` output before splitting it into lines.
Porcelain v1 is `XY<space>PATH`, so an unstaged entry begins with a space, and
stripping shifted every path two characters: `src/thing.py` was being recorded as
`rc/thing.py`. Every dirty path in every provenance record was misreported, and
nothing downstream noticed because only the boolean was ever read.

### 3.16 An interrupted cache build lost all of its work

The cache index became durable only when the caller flushed it, and a cache build
flushes once at the end. Measured against the real Gemma build, an interruption
at hour three would have left roughly 40,000 payload files on disk with no index
referencing any of them, and the restart would have re-encoded every one --
turning a transient failure into a blown eight-hour ceiling.

Writes now append to a journal that the loader replays, so a restart loses at
most the entry in flight. Flush folds the journal into the index and removes it,
keeping total bytes written linear in the entry count rather than quadratic.

### 3.17 A confirmation re-run destroyed the artefact it was confirming

The driver wrote its report to a fixed filename per mode. The first matrix run
built both caches cold in **13,221.2 s (3.67 h)**; the confirmation re-run, which
existed only to recompute the verdict under corrected gate logic, hit a fully
warm cache, rebuilt the dataset in 82 s, and overwrote the artefact carrying the
cold figure. The number survived in a run log rather than in evidence.

That matters beyond tidiness: the cold build is the measurement the eight-hour
cache ceiling is actually about, and the warm 82 s says nothing about it. Reports
are now also written to a content-addressed copy under `runs/`, so a re-run adds
an artefact instead of replacing one. The cold figure quoted throughout this
document is sourced from the first matrix run's log, which is a weaker provenance
than the artefact it should have come from.

The fix arrived after the matrix ran, and verifying it cost one more instance of
the same defect: a one-workload smoke test exercising the archive path overwrote
the full dry-run report. That artefact was already superseded by the matrix, so
nothing downstream depended on it, but it is the third time this filename
collision destroyed something in one session -- which is the argument for the
fix, made three times. The matrix deliverable has since been archived under
`runs/matrix-f3f16c289e54b31b.json` and every artefact re-checksummed, so the
result this document reports cannot be lost the same way.

## 4. Evidence labels

| Claim | Label | Boundary |
|---|---|---|
| All 48 mandatory workloads completed under the frozen matching contract, with both named backbones | `MEASURED` | one machine, one run, confirmed by a second run against a warm cache |
| Every arm's actual trainable parameter count is within 0.008% of target | `MEASURED` | counted off the built MLX model at every width and both targets |
| A run split across two fresh interpreters produces bit-identical weights | `REPRODUCED` | subprocess restart test on all three arms, and in-run at update 100 of 200 |
| Cold latent-cache build for both 4B backbones takes 3.67 h | `MEASURED` | this machine, 100,000 transitions, each model's official preprocessing; the artefact was overwritten and the figure is sourced from the run log (§3.17) |
| The two frozen families differ in encode cost by ~16x | `MEASURED` | 4,096 vision tokens against ~49, under each model's own preprocessing |
| `gemma3_4b` requires the account holder to accept its licence | `RETRACTED` | the account had already accepted; the blocker was a missing local token (§3.12) |
| The controlled family's held-out split supports a generalisation claim | `RETRACTED` | measured at 36.3% transition-tuple overlap; it is a replication check (§3.2b) |
| Any representation arm is better than another | `UNKNOWN` | Scale 0 makes no capability comparison, and 200 updates at plumbing weights cannot support one |
| The learned world model contributes anything to planning or control | `UNKNOWN` | requires Scale 2 against a reactive frozen-backbone baseline at equal interactions |
| SHWM improves transfer, causal reasoning, or continual learning | `UNKNOWN` | requires Scales 4-6 and, for the last, the committed X65A/X65B contract |
| The project is measurably closer to human-level capability | `UNKNOWN` | no valid metric for that distance is defined, here or in the strategy document |

## 5. Scale-0 gate

**PASSED.** All 48 mandatory workloads completed against both named frozen
encoder families, all three development seeds retained for every one of the 16
cells, every matching rule held, and every hard ceiling held with room to spare.
The per-clause table is in §1 and the full measurements are in
[`SCALE-0-RESOURCE-REPORT.md`](SCALE-0-RESOURCE-REPORT.md).

What that does **not** mean, stated because the matrix says so in the same
breath as the pass: no capability was measured. Two hundred optimizer updates at
plumbing weight 1.0 is a stopwatch on a pipeline. The 48 models are throwaway.
No representation arm was shown better than another, and the matching rules exist
precisely to stop a winner being read out of those loss values.

## 6. Scale 1

**Scale 1 is unblocked.**

It is unblocked for exactly one thing: a preregistered Scale-1 experiment design,
which must be committed before it runs and must define its own
equal-cumulative-compute protocol, which Scale 0 deliberately did not. Nothing
about planning, transfer, causality, continual learning, or capability follows
from this pass.

Three things Scale 1 has to settle before it can mean anything, all of them
findings from this run rather than general caution:

1. **The controlled family's held-out split is not a generalisation test.** At
   36.3% transition-tuple overlap it is a replication check. Either enlarge the
   environment or say so in the Scale-1 design; deciding afterwards invalidates
   the result (§3.2b).
2. **Both backbones' text paths are nearly as weak as the control's.** The shared
   adapter is an embedding lookup for structured observations and the full vision
   tower for images, so inherited capability enters through vision almost
   exclusively. An attribution argument that ignores this will over-credit the
   backbone on the visual family and under-credit it on the controlled one
   (§3.13, and the module docstring in `backbone_encoder.py`).
3. **The two families cost 16x differently to encode.** That is a measured
   outcome and does not threaten the matching rule, but at a larger data volume
   it threatens the cache ceiling: Gemma alone consumed 3.5 of the 8 hours at
   100,000 transitions.

## 7. Cheapest next falsifier

1. **Write the Scale-1 design and freeze it.** Costs nothing but thought, and
   deciding the three questions above afterwards would retract whatever Scale 1
   produces. This is the only step that must come first.
2. **Run the Scale-2 comparison before the Scale-1 contest, if forced to choose.**
   Scale 1 asks which representation is best; Scale 2 asks whether a learned
   world model beats a reactive frozen-backbone policy at equal interactions. If
   the answer to Scale 2 is no, the Scale-1 result is a comparison among options
   that do not matter. The strategy document already orders them 1 then 2, and
   that ordering is worth revisiting now that the backbone is cheap to run and a
   reactive baseline is nearly free to build.
3. **Re-measure the cache ceiling at the Scale-1 data volume.** Gemma's 3.4
   observations per second is the binding constraint on every later scale, and it
   is a property of its official preprocessing rather than of anything tunable.

## 8. What is explicitly not claimed

No capability result. No representation winner. No statement that continuous,
discrete, or hybrid state is preferable. No planning, transfer, causal, or
continual-learning claim. No Phase-2 final seed has been sampled, and the
final-seed guard refuses to load one without a committed post-freeze manifest.
The frozen backbones remain inherited capability. Every workload trained 50M or
200M parameters of its own on top of 4,437,815,808 or 4,300,079,472 frozen ones,
and the two counts are reported in separate columns throughout. Roughly 99% of
the parameters involved in any Scale-0 workload were trained by someone else, on
data this project has not audited; that ratio is the reason attribution controls
exist at every later scale and the reason a Scale-0 pass licenses none of them.
