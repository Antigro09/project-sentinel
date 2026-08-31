# SHWM Scale-0 Handoff

Status: **Scale 0 is not passed.** The gate stopped where the run matrix says it
must, at the S0.2 encoder preflight, on a licence the account holder has not
accepted. Everything downstream of that gate was built, tested, and measured, so
that granting access is the only remaining step before the matrix can run.

Branch: `phase-2-continuous-world-model`
Base commit: `5205543b110ba6da2e3f6da30630809941f821c4`
Generated resource report: [`SCALE-0-RESOURCE-REPORT.md`](SCALE-0-RESOURCE-REPORT.md)

## 1. The stop

`SCALE-0-RUN-MATRIX.md` names two frozen encoder families and says: "If either
family cannot run faithfully, Scale 0 stops." One of them cannot.

| Encoder | Repository | Verdict | Licence | Gated |
|---|---|---|---|---|
| `qwen3_vl_4b` | `Qwen/Qwen3-VL-4B-Instruct` | runnable | apache-2.0 | no |
| `gemma3_4b` | `google/gemma-3-4b-it` | **blocked** | gemma | `manual` |

`qwen3_vl_4b` was taken further than the metadata check, because whether a 4B
backbone is affordable on this machine is itself a Scale-0 stop condition and is
cheaper to answer now than after access is granted. Loaded at the pinned
revision `ebb281ec` in MLX at bfloat16:

| quantity | measured |
|---|---:|
| cold load | 1.0 s |
| total parameters | 4,437,815,808 |
| vision-tower parameters | 415,347,712 |
| parameter bytes | 8.27 GiB |
| peak resident | 8.55 GiB |
| encode rate | 26.4 frames/s |
| feature width | 2,560 |
| projected 50,000 visual observations | **0.53 h** against an 8 h ceiling |

So the backbone premise holds: one frozen 4B encoder can build its half of the
latent cache in about half an hour, at under 9 GiB, with 7% of the ceiling spent.
Those parameters are inherited capability and are reported apart from the 50M or
200M Sentinel-trainable count, which they never enter.

The block is not a network failure and not a missing dependency. An anonymous
read of `config.json` at the pinned revision returns **HTTP 401**, the Hugging
Face API reports `gated: manual`, and this machine has no access token in any of
the four places one can live. Google's Gemma Terms of Use must be reviewed and
accepted by the account holder on the model page, after which a token has to be
made available locally.

Two things this handoff deliberately does not do:

- **Accept the licence.** Agreeing to terms on someone's behalf is not a step an
  implementation is entitled to take, whatever the deadline.
- **Substitute a model.** A community re-upload of the same weights would route
  around the gate rather than satisfy it, and a different family would be a
  replacement. The matrix permits a replacement only through a reviewed pre-run
  amendment, and never one selected after seeing results.

The consequence is narrow and worth stating precisely: **no matrix cell has
run**, so nothing has been invalidated and no restart is owed. The matrix is
waiting, not broken.

## 2. What was built and audited anyway

The encoder gate blocks the *matrix*. It does not block the pipeline the matrix
runs on, all of which is backbone-independent. So the remaining twelve build
items were implemented and tested, and the 48 workload *shapes* were executed
end to end against the deterministic control encoder as a dry run.

Every artefact from that run is stamped `is_matrix_run: false`, and the driver
is written so that a dry run **cannot** report a Scale-0 pass however well it
goes. That is enforced in code, not in a footnote.


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

Full suite, from the repository root:

```text
uv run pytest -q
829 passed in 939.27s (0:15:39)
```

| suite | tests | result |
|---|---:|---|
| exact Phase-1 (`tests/`, excluding `tests/shwm/`) | 522 | all passed |
| Phase-2 Scale-0 (`tests/shwm/`) | 307 | all passed |
| **total** | **829** | **0 failed, 0 skipped** |

The exact half is unchanged in composition. `VERIFICATION.md` records the
pre-Phase-2 baseline as "521 passed, 1 skipped in 940.11s" — the same 522 tests,
in the same wall time to within a second. The one skip now passes because this
checkout has the ignored offline asset bundle that the isolated setup worktree
did not; no test was modified, deleted, or weakened to achieve it.

The exact reference branches were not touched. `phase-1-exact-reference` still
resolves to `5205543b110ba6da2e3f6da30630809941f821c4` and its reflog contains
only the entry that created it.

Tracked tree at the reported commit: **clean**. Untracked entries, exactly:

```text
(none — artifacts/, .venv-shwm/, and the downloaded backbone are gitignored)
```

The 8.5 GiB of run artefacts and backbone weights under `artifacts/` are
deliberately not tracked; they are content-addressed by
`artifacts/shwm/scale0/checksums.json` and reproducible from the committed code,
config, and DVC pipeline.

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

### 3.8 The latent cache costs 12.5% more than the plan's arithmetic

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

## 4. Evidence labels

| Claim | Label | Boundary |
|---|---|---|
| Both named encoder families were checked for licence, revision, gating, and local access | `MEASURED` | HTTP probes at pinned revisions on 2026-08-31; a re-check may differ if access changes |
| `gemma3_4b` is not runnable on this machine | `MEASURED` | anonymous read returns 401; no token present; resolvable by the account holder |
| `qwen3_vl_4b` is licence-clear, locally readable, and encodes at 26.4 frames/s in 8.55 GiB | `MEASURED` | one load at the pinned revision, 24 procedural frames at 224x224; a different resolution or batch shape gives a different rate |
| A frozen 4B backbone can build its half of the latent cache inside the 8 h ceiling | `INFERRED` | linear extrapolation from 24 frames to 50,000; assumes no thermal throttling and the same preprocessing |
| The Scale-0 pipeline runs end to end at the matrix's workload shape | `MEASURED` | with the control encoder; not a matrix run |
| Every arm's actual trainable parameter count is within 0.01% of target | `MEASURED` | counted off the built MLX model, all widths and both targets |
| A run split across two fresh interpreters produces bit-identical weights | `REPRODUCED` | subprocess restart test, all three arms |
| Recorded training throughput would hold with real backbone features | `UNKNOWN` | the control encoder is a hash projection; features of the same width but different statistics may train at a different rate |
| The controlled family's held-out split supports a generalisation claim | `RETRACTED` | measured at 36.3% transition-tuple overlap; see 3.2b |
| Any representation arm is better than another | `UNKNOWN` | Scale 0 makes no capability comparison and 200 updates at plumbing weights cannot support one |
| SHWM improves planning, transfer, or causal reasoning | `UNKNOWN` | requires Scales 2-5 |

## 5. Scale-0 gate

The gate requires all 48 mandatory workloads to complete under both named
frozen encoders. Zero have. The gate result is therefore **not passed**, for one
reason and one reason only: the `gemma3_4b` licence gate.

Every other gate clause was exercised against the dry run and is reported in
[`SCALE-0-RESOURCE-REPORT.md`](SCALE-0-RESOURCE-REPORT.md).

## 6. Scale 1

**Scale 1 is not unblocked.**

Passing Scale 0 would unblock a preregistered Scale-1 design and nothing else.
Scale 0 has not passed, and the dry run cannot substitute: it measures a
pipeline, not a representation, and the control encoder carries no perceptual
grounding for a representation contest to be about.

## 7. Cheapest next falsifier

In order of cost, and each one can retire the step after it:

1. **Accept the Gemma Terms of Use and provide a token.** Minutes of human time,
   no compute. It is now the only thing standing between the current state and a
   matrix run.

   The falsifier that would have come before it has already been run and passed:
   the Qwen3-VL runtime probe shows a frozen 4B backbone encoding at 26.4
   frames/s in 8.55 GiB, which puts the visual half of the cache at 0.53 h
   against an 8 h ceiling. Had that failed, granting access would have unblocked
   nothing and the right answer would have been a narrower design rather than a
   larger download. It did not fail, so the licence really is the whole blocker.

2. **Run the matrix.** The dry run puts 48 workloads at 26.2 minutes of training
   and planning; adding two backbone encode passes at roughly half an hour each
   leaves the 72-hour ceiling untouched. Peak device memory per workload is 3.94
   GiB at 200M and the process high-water across the whole sequence is 12.15 GiB,
   against a 112 GiB ceiling — the matrix is not close to any resource limit.

3. **Decide what the controlled family's held-out split is for** (§3.2b). Its
   36.3% transition-tuple overlap is a property of an 11,883-transition
   environment, not of the split procedure. Scale 1 should either enlarge it or
   declare it a replication check, before it is used to support a generalisation
   claim it cannot carry. This costs nothing to decide and invalidates a Scale-1
   result if decided afterwards.

## 8. What is explicitly not claimed

No capability result. No representation winner. No statement that continuous,
discrete, or hybrid state is preferable. No planning, transfer, causal, or
continual-learning claim. No Phase-2 final seed has been sampled, and the
final-seed guard refuses to load one without a committed post-freeze manifest.
The frozen backbone remains inherited capability and is reported separately from
Sentinel-trainable parameters, which for every workload here is zero frozen and
50M or 200M trainable.
