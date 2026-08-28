# SHWM Decision Log

## D-001 — No parameter-count AGI threshold

Decision: treat parameters as an independent experimental variable.
Evidence: bounded primary-source audit; Chinchilla is compute-optimal scaling,
not an AGI theorem; official multimodal families include sub-15B models.
Status: `INFERRED`, not proof that small models suffice.
Revisit: only with a formal lower-bound theorem whose capability definition and
computational model apply to Sentinel.

## D-002 — Preserve exact Sentinel

Decision: freeze `phase-1-exact-reference` and add neural components through
new interfaces.
Reason: exact execution, counterexamples, uncertainty states, provenance,
revision, and abstention are useful controls and cannot be reconstructed from
latent confidence.
Revisit: only if an ablation shows the exact layer contributes no safety,
calibration, or task value at its measured cost.

## D-003 — Compare representation families

Decision: continuous, discrete, and hybrid arms use the exact matching contract
in `SCALE-0-RUN-MATRIX.md`. The primary matrix uses width 512; the 50M hybrid
dimension control adds 256 and 1,024 under both encoders while retaining the
same parameter tolerance, data, updates, seeds, and planner workload.
Rejected: assuming continuous 1,024D is required.
Revisit: after Scale 1 planning, calibration, and intervention results.

## D-004 — Action conditioning is mandatory

Decision: the main dynamics contract includes action; no-action is a baseline.
Evidence: formally checked finite counterexample and exact enumeration show
passive action support can leave interventions non-identifiable.
Scope: this does not prove high-dimensional causal identification.

## D-005 — Observable verification, not latent equality

Decision: verify events, tests, rewards, constraints, tool results, and state
probes.
Reason: latent coordinates are not semantically canonical.
Boundary: a non-injective probe can hide a latent mismatch; report coverage.

## D-006 — Frozen encoder capability is inherited

Decision: use pretrained encoders for feasible local grounding, but report
their standalone performance and training-contamination risk.
Rejected: calling pretrained concepts autonomous Sentinel learning.

## D-007 — Scale 0 is a systems preflight

Decision: contracts, adapters, cache, dataset, fake models, resource accounting,
and restart before real representation experiments.
Rejected: starting with 15B training, OSWorld, SWE-bench, or final seeds.

## D-008 — Scale only after behavioral contribution

Decision: prediction loss alone never unlocks a larger model.
Gate: control, intervention, calibration, or transfer gain with uncertainty and
matched resources.

## D-009 — X65B-core is mandatory for X66 transfer

Decision: context-scoped validity, negative-transfer control, compositional
reuse, revision locality, and restart are required before repository-memory
transfer claims.
Optional: full sheaf/category machinery.
Mandatory: the operational scoped-transfer gate.

## D-010 — Parallelize, then rejoin

Decision: SHWM Scales 0–5 may run in parallel with X65A/X65B exact-memory work.
Scale 6 must consume a committed memory contract; Scale 7/X66 requires both
lines.
Reason: this shortens wall-clock time without collapsing causal attribution.

## D-011 — Freeze one Scale-0 workload

Decision: execute 12 primary encoder/representation/size cells at width 512 on
seeds 6600–6602 plus 12 dimension-control runs, for 48 workloads total. Each
uses the frozen 100,000-transition dataset, 200 optimizer updates, zero online
interactions, fixed planning calls, ±1% parameter tolerance, and explicit
memory/storage/time ceilings.
Reason: “equal budget,” “prototype,” and “three-seed preflight” were otherwise
ambiguous and could enable post-result resource changes.
Revisit: only through a committed pre-run matrix revision followed by a full
rerun; never by dropping a failed cell or seed.

## Rejected architecture candidates

### Monolithic unified latent agent

One large multimodal policy predicts and acts without a separate model or
verifier. Rejected for the first pilot because attribution, counterfactual
inspection, model-inadequacy diagnosis, and exact regression control are weak.
It remains a baseline.

### Pure latent world-model agent

A learned dynamics/planner acts without exact observable verification or
structured revision. Rejected as the main architecture because model
exploitation and shared misspecification can remain confidently wrong. It
remains a Scale-3 ablation.

### Fully symbolic high-dimensional extension

Manually define objects/events and extend the exact hypothesis grammar to raw
multimodal observations. Rejected as the sole Phase-2 direction because it
continues to hand the system the representation. It remains the exact control
where finite abstraction is available.

## Open decisions

- initial procedural environment adapter;
- RSSM, transformer, or state-space belief implementation;
- categorical code structure and hybrid interface;
- planner per action space;
- event vocabulary sufficiency;
- uncertainty approximation;
- final Scale-1 metrics and margins before validation freeze.

Qwen3-VL 4B and Gemma 3 4B are frozen for Scale 0, subject to exact-revision,
license, and runtime verification. Incompatibility is a stop; a replacement
requires a pre-run matrix amendment. The Scale-0 resource envelope is frozen
in `SCALE-0-RUN-MATRIX.md`; measured throughput informs Scale 1, not a post hoc
Scale-0 budget change.
