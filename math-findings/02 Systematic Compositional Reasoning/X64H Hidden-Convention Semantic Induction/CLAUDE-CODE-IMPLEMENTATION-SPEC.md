# Claude Code Implementation Specification — X64H

Status: implementation handoff, not authorization to run final hidden tests before freeze  
Experiment: X64H — Hidden-Convention Semantic Induction  
Primary theory: `x64h-theory-research-brief.md`

## 1. Objective and non-goals

Implement a finite exact reference mechanism that jointly infers the current typed semantic form and an episode-persistent hidden language convention from an instruction, trusted behavioral demonstrations, prior same-convention history, and clarification answers.

Do not replace Sentinel's trusted executor, existing behavioral version-space states, confirmation logic, or controlled logical forms. Do not claim natural-language induction, AGI capability, or global novelty. Do not expose the sampled convention to any arm except the oracle-convention arm.

## 2. Required architecture

Create an isolated X64H package and one runner:

```text
experiments/x64h/
  types.py             immutable public, target-only, and oracle-only records
  semantic.py          adapter around existing X64 logical forms and executor
  convention.py        frozen finite convention tuple and post-freeze sampler
  grammar.py           FT-SPCFG generation and exact inside likelihood
  posterior.py         joint, conflict, behavior, and OTHER posteriors
  queries.py           semantic/behavioral answer models and information gain
  decision.py          execution, ask, abstain, expand Bayes risks
  persistence.py       convention posterior and observation-only audit log
  arms.py              all 14 arms through one interface
  metrics.py           convention-level metrics and confidence intervals
  protocol.py          freeze manifest, seed release, gates, and taint audit
experiments/x64h_hidden_convention.py
tests/test_x64h_*.py
```

If repository conventions favor fewer files, preserve these module boundaries conceptually and in tests.

Core immutable types:

```python
ConventionSpec = (
    lexical_map,
    order_rules,
    contextual_senses,
    phrase_rules,
    optional_word_rules,
    argument_drop_rules,
    attachment_rules,
)

Evidence = (utterance, demonstrations, observed_questions_and_answers)
PosteriorState = (log_p_phi, model_hash, observation_hashes)
OpenWorldKind = IN | UNKNOWN_REALIZATION | UNKNOWN_MEANING | UNKNOWN_PROGRAM
Decision = EXECUTE | ASK_SEMANTIC | ASK_BEHAVIORAL | ABSTAIN | EXPAND
```

The convention identifier and target logical form are evaluator-only fields. They must not occur in the agent-facing episode object or persistent state.

## 3. Exact generative model

Reuse existing typed logical forms `x64e_semantics.Z`, `ALL_Z`, `execute`, `denote`, and the trusted behavioral replay path unless a typed extension is necessary. Every new semantic production must have a test showing that it is executable and represented in the oracle arm.

Implement a finite typed synchronous PCFG conditioned on `ConventionSpec`. Each rule couples a typed semantic constructor or bounded fragment to a surface expansion. The finite rule inventory must support:

- atom-to-phrase synonym selection;
- the same phrase for different atoms in different parent/role contexts;
- child-order permutations;
- multi-token phrase rules;
- optional function-word alternatives;
- explicitly marked child omission;
- local attachment state.

For each `(phi, z, u)`, compute

```text
language_likelihood(phi, z, u)
  = sum over all valid synchronous derivations yielding u
      product of frozen rule and lexical probabilities
```

Use a binarized, acyclic epsilon-safe grammar. Verify the inside result against brute-force derivation enumeration on tiny grammars. Exact means no beam, particle truncation, top-k pruning, or unreported candidate cap.

## 4. Posterior and state updates

Compute in log space:

```text
log_joint[phi,z] =
    log_p_phi_from_history[phi]
  + log_p_semantic[z]
  + log_language_likelihood[phi,z]
  + log_trusted_behavior_likelihood[z]
```

Normalize only inside the `IN, M=0` component. Separately compute and normalize:

- shared-meaning likelihood `L0`;
- incompatible-meaning likelihood `L1`;
- unknown-realization likelihood;
- unknown-meaning likelihood;
- unknown-program likelihood.

Push the in-class joint posterior through the semantic-to-program map for immediate behavioral decisions. Preserve exact semantic forms or continuation-equivalence classes in stored compositional programs.

Update `p(phi | history)` only after an observation is public. The update may marginalize over `z`, but it may not read the target convention, target logical form, future task, future answer, or generator derivation.

## 5. Conflict, open world, and commitment

Implement the match/mismatch Bayes factor from Equations (8)–(10) of the theory report. Calibrate priors and thresholds on validation only. Keep ambiguity, conflict, and open-world probabilities as separate outputs.

Implement a finite top-level mixture with normalized, frozen base likelihoods for:

1. unknown word/phrase/construction;
2. intended semantic form outside the grammar;
3. intended executable program outside the candidate pool.

Never discard `OTHER` mass and renormalize the in-class posterior before deciding. Add an explicit regression test where one in-class candidate survives but `OTHER` dominates; the expected action is ask, expand, or abstain—not execute.

Execution requires a calibrated conflict gate, a conditional open-world gate, a conditional leading-behavior gate, and minimum expected risk. Implement costs for wrong execution, each query type, abstention, and expansion in the frozen configuration.

## 6. Clarification

Use one question interface for:

- behavioral questions: selected input and requested output;
- semantic questions: sense, attachment, omitted argument, or paraphrase choice.

For each question, enumerate its answer distribution under `(Z, Phi, O, M)` and compute mutual information minus frozen query cost. Implement random-disagreement and information-gain policies with identical question pools, budgets, stopping rules, and answer access.

Do not claim a greedy adaptive-submodularity guarantee. Record posterior entropy, expected answer entropy, expected decision-risk reduction, realized answer, and posterior change for every question.

## 7. Convention generation and equivalence audit

The development generator may sample convention tuples from the meta-grammar. Before freeze, implement a structural audit that:

- computes lexical incidence signatures in the finite subset submodel;
- detects duplicate signatures and explicit grammar automorphisms where feasible;
- reports the observational equivalence class;
- verifies that the oracle-convention parser can represent every generated item;
- rejects malformed grammar draws using only frozen structural criteria.

Do not reject or regenerate a final convention because a non-oracle arm fails. The final evaluator scores posterior mass on the true equivalence class.

## 8. Freeze and leakage controls

`freeze_digest()` must hash every field that can affect generation, inference, querying, decisions, or evaluation:

- semantic grammar and executor version;
- convention meta-grammar and structural filter;
- all priors and probability tables;
- candidate limits and exactness flags;
- query pool, costs, budget, and stopping rule;
- conflict and OTHER models;
- commitment risks and thresholds;
- gate definitions and confidence-interval procedure;
- development/validation generators;
- code commit.

Final convention seeds are unavailable until a freeze manifest with this digest is committed. After release, write a target-only manifest containing final seeds and convention hashes. Mutating any frozen field must change the digest and invalidate those seeds.

Add taint labels `PUBLIC`, `OBSERVED`, `ORACLE_ONLY`, `TARGET_ONLY`, and `FUTURE`. The persistence writer accepts only `PUBLIC` and `OBSERVED`. Test this at serialization boundaries and after process restart.

### Reproducibility stack

- Put all non-secret generation, inference, query, cost, and gate values in Hydra configuration; serialize the fully resolved config into every run artifact.
- Log arm, seed, convention hash, model hash, metrics, runtime, peak memory, and artifact paths to the project-local MLflow SQLite store.
- Version generated non-secret manifests and result tables with DVC after the repository owner approves initialization; never place hidden targets, credentials, or future answers in a remote DVC store.
- Keep the exact JSON outputs as the authoritative machine-readable result and generate the Markdown report from them.
- Implement the amortized arm in JAX only after the exact arm is pinned. NumPyro may represent the hierarchical convention prior if used, but it must be checked against exact finite marginals on microcases.

## 9. Required arms

Implement all arms behind the same evaluator:

1. demonstrations only;
2. static family-aware authored parser;
3. static default-convention parser;
4. exact Bayesian convention inference;
5. learned/amortized convention inference;
6. joint task-and-convention posterior;
7. joint + random queries;
8. joint + information-gain queries;
9. no convention memory;
10. shuffled convention history;
11. oracle convention;
12. oracle task meaning;
13. no open-world component;
14. no confirmation.

Only arm 11 receives `ConventionSpec`; only arm 12 receives target meaning. Arms 2 and 3 receive no final convention fields.

## 10. Test requirements before any final run

At minimum, pin:

1. freeze digest changes for every frozen-field mutation;
2. final seeds cannot be requested before a committed freeze manifest;
3. inside likelihood equals brute-force derivation sum on microgrammars;
4. every per-`(phi,z)` utterance likelihood is normalized on its bounded support;
5. exact behavioral likelihood uses the trusted executor;
6. joint posterior normalizes and matches hand-enumerated cases;
7. convention posterior improves on a separating context family;
8. duplicate signatures preserve a non-singleton equivalence class;
9. no-memory and shuffled-history controls remove persistent evidence;
10. information gain and random use identical pools and stopping rules;
11. the `k=4` query microcase reproduces optimal/greedy `2.0` and random `2.857142857`;
12. conflict posterior matches a hand-computed Bayes factor;
13. ambiguity, conflict, and `OTHER` are not conflated;
14. dominant `OTHER` prevents confident singleton execution;
15. continuation-distinct programs are not merged by current-output equivalence;
16. process restart restores only observed convention evidence;
17. taint tests reject target, oracle-only, and future fields;
18. every arm emits the same complete metric schema;
19. oracle-convention failures identify grammar/executor errors;
20. hidden caps set `incomplete_candidates=True` and cannot report exact inference.

Run the existing X64E/X64G regression tests unchanged.

## 11. Experiment sequence

1. Reproduce the included finite bijection and noise results.
2. Implement one four-atom, two-constructor FT-SPCFG microcase.
3. Prove exact posterior values by hand and pin them in tests.
4. Scale to development conventions and profile exact inference.
5. Add validation-only calibration, amortized inference, and gate tuning.
6. Run all reset, shuffle, oracle, no-OTHER, and no-confirmation controls.
7. Produce and commit the freeze manifest.
8. Release post-freeze final seeds.
9. Run four untouched final seeds without code or threshold edits.
10. Report every gate, failure, runtime, and confidence interval; a failure remains a result.

## 12. Acceptance criteria for the implementation slice

The implementation slice is complete only when:

- exact microcases and leakage tests pass;
- the 14 arms share one evaluator and metric schema;
- state survives process restart without target leakage;
- equivalence classes, `OTHER`, conflict, and incomplete candidates are visible in outputs;
- a committed freeze manifest can deterministically reproduce development and validation runs;
- no final hidden seed has been sampled or inspected before freeze.

Passing these software criteria does not pass X64H. The research result is determined only by H1–H10 on untouched post-freeze conventions.
