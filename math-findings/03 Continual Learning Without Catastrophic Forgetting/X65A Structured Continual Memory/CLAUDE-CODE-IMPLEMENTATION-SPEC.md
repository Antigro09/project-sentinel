# Claude Code Implementation Specification — X65A

Status: blocked implementation handoff  
Experiment: X65A — Structured Continual Memory  
Primary theory: `x65a-theory-cycle.md`

## 1. Non-negotiable prerequisite

Do not create production X65 files or sample X65 final seeds until the **final frozen X64H run passes** and its manifest, commit, gates, and untouched seeds are auditable. Provisional X64H-0B results are insufficient.

The X65 runner must begin with a fail-closed prerequisite check:

```python
assert x64h_manifest.phase == "final"
assert x64h_manifest.freeze_valid
assert x64h_manifest.all_required_gates_passed
assert current_x64h_digest == x64h_manifest.frozen_digest
```

If any field is absent, ambiguous, or false, exit without creating an X65 result artifact. Theory checks under `math-findings/` may run independently because they do not alter Sentinel or inspect hidden final seeds.

## 2. Objective and non-goals

Implement an exact finite reference system that tests whether verified experience from earlier *different* tasks causes better performance on later unseen compositions while preserving old competence, correcting planted false beliefs, resisting irrelevant memory, controlling serialized growth, and surviving process restart.

Do not:

- call storage, replay, or retrieval alone continual learning;
- modify frozen X64H semantics while measuring X65;
- store future target programs, logical forms, outputs, conventions, or answers;
- merge programs by visible output when continuation effects differ;
- grant any memory arm extra candidate expansions, queries, wall-time policy, or current-task evidence;
- implement a learned retriever before the exact reference arm is pinned;
- claim global submodularity, AGI capability, or categorical novelty;
- open final stream seeds before the freeze manifest is committed.

## 3. Proposed package layout

```text
experiments/x65a/
  types.py                 immutable IDs, taint labels, schemas, enums
  episodic.py              immutable verified evidence ledger
  semantic.py              finite factors, X64H adapter, calibration state
  procedural.py            typed program contracts and composition
  negative.py              counterexamples, defeat, supersession, staleness
  graph.py                 typed provenance/dependency DAG and closures
  posterior.py             exact finite hierarchical Bayesian update
  reliability.py           source/context/fault latent variables
  retrieval.py             candidate frontier and exact budgeted selection
  decision.py              act/ask/abstain Bayes risk
  consolidation.py         deterministic two-part MDL proposals and guards
  persistence.py           atomic schema-only write, reload, hash audit
  leakage.py               taint enforcement, snapshots, forbidden canaries
  streams.py               A–G task streams and dependency counterfactuals
  arms.py                  all 20 arms behind one interface
  budgets.py               bytes, entries, retrieval, expansions, time ledger
  metrics.py               paired transfer/retention/revision/growth metrics
  statistics.py            frozen intervals, paired bootstrap, gate decisions
  protocol.py              prerequisite, freeze, seed release, taint status
experiments/x65a_structured_memory.py
tests/test_x65a_*.py
```

Keep these boundaries even if repository conventions lead to fewer physical files.

## 4. Immutable schemas

Use frozen dataclasses or equivalent immutable records. IDs are content-addressed where possible.

```python
Taint = PUBLIC | OBSERVED | ORACLE_ONLY | TARGET_ONLY | FUTURE
MemoryKind = EPISODIC | SEMANTIC | PROCEDURAL | NEGATIVE
EdgeKind = SUPPORTS | CONTRADICTS | DERIVES | COMPOSES | CONTEXT_GATES | INVALIDATES
OpenWorld = RELEVANT | NONE | UNCERTAIN | CONTRADICTED | MISSING_REPRESENTATION | UNKNOWN_TASK
Decision = EXECUTE | ASK | ABSTAIN | EXPAND

ProvenanceRef = (source_id, observation_hash, task_index, context_id, taint)

EpisodicEntry = (
    id, instruction, demonstrations, questions, answers,
    selected_interpretation, executed_program, complete_trace,
    outcome, counterexamples, context_id, provenance, acquired_at,
)

SemanticEntry = (
    id, typed_claim, posterior_table_or_sufficient_statistic,
    calibration_state, validity_context, provenance,
    created_at, version, supersedes, status,
)

ProcedureEntry = (
    id, program_ast, precondition, postcondition,
    continuation_effects, known_failures, resource_contract,
    behavioral_signature, confidence, validity_context,
    provenance, created_at, version, supersedes, status,
)

NegativeEntry = (
    id, defeated_claim_or_program, scoped_context, counterexample,
    rejection_reason, source_reliability_state, provenance,
    created_at, version, supersedes, status,
)

DependencyEdge = (source_id, target_id, edge_kind, context_predicate, provenance)

PersistentState = (
    schema_version, frozen_model_digest, task_index,
    episodic_entries, semantic_entries, procedure_entries,
    negative_entries, dependency_edges, exact_posterior,
    calibration_state, audit_chain_hash,
)
```

Do not serialize evaluator-only dependency truth, final seed metadata, target labels, future candidate pools, Python RNG objects, function closures, caches, MLflow objects, or the raw task generator.

## 5. Exact finite posterior

Represent the pilot latent state as a canonical finite enumeration `LambdaState`. Include frozen X64H convention factors, reusable semantic factors, procedure-validity factors, context boundary, source reliability, and fault cause.

For each task, compute in log space:

```text
task_likelihood[lambda] = sum over finite c,z,b
    p(c | lambda)
  * p(z,b | c,lambda)
  * p(observed task evidence | z,b,c,lambda)

new_log_q[lambda] = old_log_q[lambda] + log(task_likelihood[lambda])
new_log_q -= logsumexp(new_log_q)
```

Cross-check every new factor against rational or high-precision brute-force enumeration on microcases. A zero normalizer produces `MISSING_REPRESENTATION` or `UNKNOWN_TASK`; never silently renormalize a surviving in-class candidate.

Persistent sufficient statistics are allowed only when a test proves they induce the same posterior predictive as the complete observed history under the frozen model. Provenance and unique counterexamples remain separately persisted.

## 6. Memory write and revision

Evidence-authoritative update order:

1. receive only public/current evidence;
2. execute through the trusted interpreter;
3. verify trace and outcome;
4. append an immutable episodic record;
5. enumerate `OLD_WRONG`, `NEW_CORRUPT`, `CONTEXT_SHIFT`, and `PARSE_ERROR` fault causes;
6. update source reliability and semantic/procedural posteriors;
7. create negative/supersession entries rather than deleting old versions;
8. compute the context-compatible descendant closure of changed nodes;
9. invalidate/recompute only that closure;
10. run independent-memory probes before committing;
11. atomically persist and reopen/hash-check.

An update outside the dependency closure is a protocol violation unless the audit event names a newly observed global dependency and creates its edge before revision.

Support later reversal: new trustworthy evidence may restore a superseded claim as a new version. Preserve the complete evidence chain.

## 7. Procedural contracts and compounding

A stored program must carry:

- typed input/output domains;
- preconditions and failure predicates;
- postconditions;
- stack, register, store, pointer, sparse-memory, and external effects;
- deterministic resource cost in the finite VM;
- trusted-trace hashes and counterexamples.

Composition `compose(p, q)` is admissible only when `post(p)` establishes `pre(q)` and effect composition is defined. Test both the trusted and fast interpreter. Do not store the later composite target before its task.

The search enumerator must count every candidate expansion identically across arms. Record raw and macro description lengths, target rank, candidate count at discovery, and timeout. For A9, require:

```text
macro_search(target, B) succeeds
primitive_search(target, B) fails
primitive_search(target, B_large) succeeds
target not in persisted programs before task
```

This supports only a resource-bounded capability statement.

## 8. Retrieval

Construct the candidate frontier from current-task typed requirements, observed contexts, graph edges, and open-world status. Surface similarity may propose candidates but cannot create dependency edges or override context/negative gates.

For main X65A, enumerate every subset with at most four nodes and at most 512 serialized bytes from a frontier capped at 20 entries. Evaluate:

```text
expected_value(S) =
    expected reduction in current Bayes risk
  - lambda_interference * expected wrong-context loss
  - lambda_compute * counted expansion cost
  - lambda_query * expected extra questions
```

Use deterministic content-hash tie-breaking. Log all subset values. If the frontier cap truncates candidates, set `incomplete_retrieval=True`; an exact arm may not report exactness for that task.

Implement the independent weighted-coverage greedy selector only as a restricted diagnostic. Do not use its guarantee for the main complementary/nonmonotone utility.

## 9. Consolidation

Use a deterministic proposal order and a computable code with frozen coding tables:

```text
score(M) = L(model_and_components) + sum L(observed_task | M)
```

Candidate proposals may:

- replace repeated episode fragments with a semantic component;
- replace repeated verified program subtrees with a procedure;
- merge truly equivalent version nodes;
- remove redundant raw fields while retaining provenance references;
- evict low-value duplicative evidence under the byte budget.

Accept a proposal only when:

1. total code length strictly decreases;
2. all trusted finite replay results are identical;
3. continuation-relevant state is identical on frozen probes;
4. unique counterexamples and provenance remain reachable;
5. validation-frozen held-out transfer is not degraded beyond tolerance;
6. byte count is measured from canonical serialized bytes, not Python object size.

Compression without task utility does not pass A6.

## 10. Persistence and hidden-state canary

Implement canonical JSON or another fully specified deterministic encoding. Write to a new file, `fsync`, atomically rename, reopen, validate schema, recompute every content hash, and verify posterior normalization.

At midpoint:

1. set `X65A_FORBIDDEN_TARGET=future_target_answer` in the parent and place the same string in a runtime-only cache;
2. serialize permitted state;
3. terminate and confirm the parent PID is gone;
4. launch the child with a scrubbed environment, external next-task input, and no inherited random state;
5. fail if the forbidden value appears in bytes, environment, memory keys, logs, MLflow parameters, DVC metadata, or child process state;
6. compare exact posterior, predictions, entries, and audit hash before/after.

No `pickle` for scientific state. No global singleton memory.

## 11. Stream generator

Generate A–G streams from a hidden evaluator-only dependency DAG. The agent-facing task object must not contain future edges or labels.

- **A semantic:** early grounded convention/context tasks; later held-out combinations.
- **B procedural:** early primitive subprograms; later new macro composition.
- **C retention:** delayed old-family probes after interference.
- **D revision:** plausible false contextual claim, trusted refutation, optional later reversal.
- **E interference:** similar irrelevant, stale, and poisoned entries.
- **F no reuse:** no shared latent components; memory should not help.
- **G switch:** explicit convention/context boundary.

Generate dependency-respecting, reverse, random, A→B, B→A, and shuffled-memory order counterfactuals from the same underlying instances. The reverse order must place composite targets before their prerequisites without changing current-task evidence.

## 12. Arms

Implement all 20 theory-report arms through one immutable `ArmConfig` and one runner. Only permitted differences may vary. Log the fully resolved arm config.

1. no memory;
2. raw full replay;
3. random retrieval;
4. recent retrieval;
5. surface similarity;
6. dependency-oracle retrieval;
7. episodic only;
8. semantic only;
9. procedural only;
10. negative/revision only;
11. semantic plus procedural;
12. all types, no consolidation;
13. full VDFM;
14. no revision;
15. no provenance;
16. shuffled memory;
17. stale memory;
18. poisoned memory;
19. oracle relevance;
20. unlimited-memory diagnostic.

Oracle arms receive evaluator-only fields through a separately typed interface that cannot be serialized by the agent writer.

## 13. Freeze and leakage protocol

`freeze_digest()` must include:

- X64H manifest/digest and adapter version;
- task/semantic/program grammar;
- latent state enumeration and priors;
- source reliability/fault model;
- all memory schemas and coding tables;
- graph edge semantics;
- retrieval utility, caps, tie-breakers, and budgets;
- program enumeration order and search budget;
- consolidation proposals and acceptance tests;
- stream generator and order-counterfactual logic;
- all 20 arms;
- metrics, confidence intervals, thresholds, gates, and stopping rules;
- restart and taint implementation;
- code commit and resolved dependency lock.

Final seeds may be requested only after a manifest containing this digest is committed. Any frozen-field mutation invalidates those seeds. Never regenerate a final stream based on a non-oracle failure.

Take a canonical memory snapshot immediately before every task and run:

```python
assert no_target_or_future_taint(snapshot)
assert all(entry.acquired_at < current_task_index for entry in snapshot.entries)
assert current_target_program not in snapshot.program_asts
assert current_target_logical_form not in snapshot.semantic_claims
assert current_expected_outputs not in snapshot.episodic_payloads
```

The last three checks detect direct leakage. Also run structural canaries to catch aliases, hashes, or evaluator IDs that encode targets.

## 14. Metrics and statistics

Emit one row per arm × stream × task, plus stream and run summaries. Required fields:

- identifiers and all digests;
- task family, dependency role, order condition, and target-absence result;
- accuracy, denotation/program result, log loss, confidence, abstention;
- questions, candidate expansions, wall time, and peak memory;
- retrieved IDs/types/bytes, precision/recall against evaluator dependencies;
- irrelevant/stale/poison rates;
- posterior hashes and calibration quantities;
- revision target, latency, source state, supersession IDs, collateral total variation;
- old-task probe results before/after;
- serialized bytes, entry counts, graph nodes/edges, growth slope inputs;
- process IDs, restart hashes, hidden-channel result;
- compounding raw/macro lengths and larger-budget memoryless outcome.

Use stream-level paired analysis. Freeze a hierarchical bootstrap with 10,000 replicates and Holm correction for preregistered primary contrasts. Include every timeout, error, and abstention. Generate gate decisions solely from machine-readable results.

## 15. Mandatory tests before final seeds

At minimum, pin these tests:

1. X65 exits unless final frozen X64H passed.
2. Every frozen-field mutation changes the digest.
3. Final seeds cannot be requested pre-freeze.
4. Exact posterior matches rational enumeration and normalizes.
5. Same sufficient statistic yields the same posterior predictive.
6. Zero evidence returns an open-world state, not an in-class singleton.
7. Every memory type round-trips canonically.
8. Target/oracle/future taint is rejected at every writer boundary.
9. Direct target snapshots stay clean before every task.
10. Graph closure contains all planted dependents and excludes planted independent nodes.
11. Factorized revision leaves an unrelated marginal unchanged.
12. Missing dependency and spurious dependency defects are detected.
13. Superseded claims, counterexamples, and reversals retain provenance.
14. Source corruption and context shift are distinguishable on hand-enumerated cases.
15. Output-equal but continuation-distinct programs never merge.
16. Contract composition rejects incompatible pre/postconditions.
17. The later composite target is absent while its components are present.
18. Raw/macro searches count candidates identically and obey the same budget.
19. The complementary-memory counterexample defeats greedy additive retrieval.
20. The restricted coverage utility passes exhaustive submodularity checks.
21. Consolidation lowers code length and preserves every frozen trusted trace.
22. Unique counterexamples cannot be evicted by consolidation.
23. Raw replay growth is at least one serialized nonempty record per task.
24. All arms receive identical current-task evidence, query, and compute budgets.
25. Reset, shuffle, stale, poison, and no-reuse controls alter only their declared field.
26. Reverse order removes prerequisite-based transfer.
27. Process restart changes PID and preserves only permitted state.
28. The forbidden target canary is absent after restart and causes a deliberate contaminated fixture to fail.
29. Random state, candidate caches, and full history are absent after reload.
30. Every arm emits the complete metric schema and every gate is machine-derived.

Keep the existing X64 regression suite unchanged and green.

## 16. Development sequence

After the prerequisite passes:

1. add schemas, canonical serialization, taint labels, and fail-closed X64H check;
2. port the exact finite theory microcases from this package into repository tests;
3. implement exact posterior and source/fault enumeration;
4. implement four ledgers and typed dependency graph;
5. implement process restart before retrieval or consolidation;
6. implement procedural contracts and bounded-search accounting;
7. implement exact retrieval and restricted diagnostic selectors;
8. implement graph-local revision and supersession;
9. implement deterministic MDL proposals and verifier guards;
10. implement A–G streams and leakage snapshots;
11. implement all arms and equal-budget audit;
12. run development streams, then validation streams, then freeze thresholds;
13. commit freeze manifest;
14. sample final streams once;
15. run all paired arms and produce an immutable results bundle;
16. audit failures without changing frozen code or regenerating seeds.

## 17. Required final artifact bundle

```text
x65a-freeze-manifest.json
x65a-resolved-config.yaml
x65a-final-seed-manifest-target-only.json
x65a-task-results.jsonl
x65a-stream-results.jsonl
x65a-gates.json
x65a-budget-audit.json
x65a-leakage-audit.json
x65a-restart-audit.json
x65a-memory-snapshots/          # observed/public fields only
x65a-figures/
x65a-report.md
```

Every result must include commit, freeze digest, environment lock digest, X64H digest, stream seed hash, arm config hash, and complete failure records.

## 18. Stop conditions

Stop and report rather than patching around the result if:

- final X64H is absent or fails;
- exact posterior exceeds the measured local resource limit;
- the stream generator leaks future dependencies;
- static/raw replay solves every hidden stream under the same budget;
- the oracle procedure/convention arm fails;
- a final seed or target is exposed before freeze;
- any final failure would require changing a frozen field;
- a clean restart cannot exclude hidden state;
- the main result depends on unequal compute or stopping rules.

The cheapest valid negative result is preferable to an unfalsifiable “memory improvement.”
