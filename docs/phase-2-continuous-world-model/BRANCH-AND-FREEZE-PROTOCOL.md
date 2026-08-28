# Branch, Freeze, and Evidence Protocol

## Branch topology

```text
5205543b110ba6da2e3f6da30630809941f821c4
├── phase-1-verifier
├── phase-1-exact-reference       # immutable exact reference branch
└── phase-2-continuous-world-model
```

The Phase-2 worktree is:

```text
/Users/anthonycavero/Documents/Startup/project-sentinel-phase-2-continuous-world-model
```

The original Phase-1 checkout remains:

```text
/Users/anthonycavero/Documents/Startup/project-sentinel
```

It contained unrelated unfinished X65A-L1 edits when the new branches were
created. Those files were not staged, moved, reset, copied, or committed by
this setup.

## Role of each branch

### `phase-1-exact-reference`

- The branch ref is immutable at
  `5205543b110ba6da2e3f6da30630809941f821c4`; it receives no commits of any
  kind, including documentation-only commits.
- Never receives neural dependencies.
- Never changes exact interpreter or verifier semantics for a Phase-2 result.
- Supplies reproducible behavioral, provenance, memory, and abstention controls.
- New provenance is recorded on `phase-1-verifier`, the consuming Phase-2
  branch, or a separate versioned record without advancing this ref.

### `phase-1-verifier`

- Continues the exact X64/X65 evidence line.
- Must close active audits with full suite, commit, clean tracked tree, and
  explicit phase verdict.
- Does not inherit Phase-2 neural results as evidence for X65.

### `phase-2-continuous-world-model`

- Adds high-dimensional perception and learned dynamics behind additive
  interfaces.
- Imports exact behavior by commit or reviewed merge, never by copying dirty
  files.
- Maintains separate development/validation/final manifests.
- Cannot redefine a failed gate after seeing final results.

## Merge discipline

1. Exact-reference fixes originate on `phase-1-verifier` and pass the full
   exact suite.
2. The exact fix is committed with a narrow evidence statement.
3. Phase 2 imports that commit through a reviewed merge or cherry-pick.
4. If Phase-2 dependencies prevent the exact suite from running unchanged,
   the import stops.
5. Phase-2 neural code never flows back into `phase-1-exact-reference`.
6. Shared interfaces may flow back only when they are dependency-free and
   behavior-preserving, with equivalence tests.

## Freeze levels

### Development freeze

Freezes schema, metric names, seed ranges, and artifact layout. It may be
revised, but every revision is logged.

### Validation freeze

Freezes architecture family, representation budgets, comparison arms, query
and planning budgets, thresholds, and statistical procedure. Validation can
select among preregistered variants.

### Final freeze

Before final seeds are sampled, commit and hash:

- source tree and dependency lock;
- environment/task generators;
- backbone identities and preprocessing;
- data collection policy and split logic;
- model architectures and parameter-count checks;
- training/inference code and configs;
- planner and verifier;
- uncertainty and commitment thresholds;
- memory/retrieval/revision policy;
- evaluators and statistical analysis;
- gates, falsifiers, and reporting templates.

Only then sample final environment seeds, mechanics, goals, and task streams.

## Manifest minimum

```json
{
  "phase": "SHWM-SCALE-N",
  "base_commit": "...",
  "implementation_commit": "...",
  "dirty_tracked": false,
  "dependency_lock_sha256": "...",
  "encoder_identities": [],
  "environment_generator_sha256": "...",
  "split_procedure_sha256": "...",
  "evaluator_sha256": "...",
  "config_sha256": "...",
  "gate_document_sha256": "...",
  "final_seed_file": null,
  "created_before_final_seed": true
}
```

The final seed file is a separate post-freeze artifact whose hash is added
without changing prior fields.

## Evidence ledger entry

Every result records:

```text
MEASURED / REPRODUCED / INFERRED / HYPOTHESIS / SPECULATIVE / RETRACTED / UNKNOWN
experiment and title
commit and manifest hashes
full-suite result
task distribution, seeds, and sample count
arms and resource matching
metrics and intervals
positive and negative results
bugs and corrections
scope limits
unmeasured claims
next cheapest falsifier
unblocked / not unblocked
```

## Rollback

- Neural checkpoints and datasets are content-addressed artifacts, not Git
  history.
- Every promoted model references a parent, manifest, and evaluation report.
- A regression restores the prior promoted model and retains the failed model
  as negative evidence.
- Evaluators and failed tests are never deleted to improve a score.
- Irreversible environment actions require explicit authority and cannot rely
  on model rollout confidence alone.

## Branch setup verification

The initial setup is valid only if:

- both branch refs resolve to the recorded base before the Phase-2 commit;
- the primary checkout's existing dirty files are byte-unchanged;
- the Phase-2 worktree contains only scoped setup artifacts;
- the Phase-1 exact full suite at the pin is either reproduced or explicitly
  recorded as historical and scheduled before use as a live control;
- no remote is pushed without Anthony's explicit request.
