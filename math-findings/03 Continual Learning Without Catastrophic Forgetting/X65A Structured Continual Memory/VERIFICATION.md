# X65A verification record

**Run date:** 2026-08-27  
**Environment:** dedicated `.venv-math-research`, project-local Lean/Mathlib workspace, and cycle-local DVC subproject.

## Verdict

All checks included in this theory cycle passed after correcting the working directory of one audit command. They establish finite algebraic claims, explicit counterexamples, a toy dependency construction, and schema-limited restart persistence. They do **not** establish continual learning in Sentinel and do not pass any X65A gate.

Production implementation remains blocked until final frozen X64H passes.

## Lean 4 + Mathlib

Command:

```bash
source "/Users/anthonycavero/Documents/Startup/project-sentinel/math-findings/activate-math-research.sh"
cd "/Users/anthonycavero/Documents/Startup/project-sentinel/.math-research-tools/lean/sentinel_math"
lake env lean "/Users/anthonycavero/Documents/Startup/project-sentinel/math-findings/03 Continual Learning Without Catastrophic Forgetting/X65A Structured Continual Memory/formal/X65A.lean"
```

Result: exit code 0 under Lean 4.34.0-rc2 with no warnings and no `sorry`.

Mechanically checked:

- a smaller finite memory space cannot injectively encode every history;
- a past-only store does not literally contain a current or future indexed target;
- exact finite Bayesian updating normalizes when evidence mass is nonzero;
- left-only evidence preserves the unrelated right marginal under factorization;
- the executable local update preserves the unrelated component;
- equal finite posteriors induce equal posterior predictives;
- nonempty raw replay grows at least linearly;
- the stated component-cost threshold implies a shorter two-part code.

Not mechanically checked:

- the entropy/error lower bound;
- a history-to-posterior sufficiency theorem with all conditional-independence premises internalized;
- typed dependency-closure revision over the complete memory graph;
- finite-state procedural contract composition;
- submodularity of independent stochastic coverage;
- any X65A transfer, retention, revision, or bounded-growth gate.

Rocq/Coq and Isabelle were not installed. No secondary formal proof is claimed.

## SymPy

`results/symbolic-checks.json` records:

- exact two-state posterior normalization: 1;
- reliable refuting evidence lowers an interior belief exactly when source reliability exceeds one half;
- factorized unrelated marginal: unchanged;
- MDL saving: `n*(raw-residual)-component`;
- posterior odds depend on evidence order only through finite positive/negative counts;
- macro/raw full-tree depth-layer ratio: `d**(length-compressed_length)`.

These are derivation checks, not performance evidence.

## Exact finite and graph checks

`results/exact-checks.json` records:

- 16 four-bit histories versus 8 three-bit memory states, with a concrete collided pair separated by index 3;
- 32 length-five histories collapsed to six count-statistic classes with identical exact posteriors in each class;
- exact expected MAP accuracy from persistent binary-convention evidence: `1/2`, `4/5`, `4/5`, `112/125`, `112/125`, `2944/3125`, `2944/3125`, `15104/15625`, `15104/15625`;
- general complementary/stale retrieval: nonmonotone and nonsubmodular;
- independent weighted coverage: monotone and submodular under exhaustive subset checks;
- planted claim revision: `4/5 -> 4/13 -> 76/85`, while an unrelated factor remains exactly `9/10`;
- dependency-respecting order solves both toy composites; reverse order solves neither;
- resource-bounded compounding layer check: `6^3 = 216 <= 1000 < 6^8 = 1,679,616`, with the target composite absent from memory;
- raw toy-code slope approximately 128 bytes/task versus 22.38 for the specified consolidated code;
- clean child PID differs from parent, posterior remains exactly `(1/5, 4/5)`, and the planted forbidden answer channel is absent.

The restart hash is deterministic for the permitted state. PID values are run metadata and are not interpreted scientifically.

## Property-based checks

Hypothesis generated 300 examples for each of five properties, 1,500 total:

- posterior normalization;
- factorized revision locality;
- raw replay's linear lower bound;
- past-only direct-target exclusion;
- deterministic coverage submodularity.

All passed. These tests search finite arithmetic/input boundaries; they are not substitutes for proofs or Sentinel experiments.

## Targeted resource checks

`results/resource-checks.json` records:

- SciPy solved the restricted binary retrieval knapsack exactly;
- CVXPy returned an optimal LP-relaxation upper bound of approximately 5.5 versus integer value 5.0;
- JAX produced finite gradients for the independent-coverage surrogate;
- NumPyro represented the finite posterior `(0.2, 0.8)` and normalized it;
- Hydra loaded `configs/x65a.yaml`;
- MLflow recorded each resource run in the project-local store. Random MLflow run IDs are deliberately excluded from DVC-tracked scientific JSON.

This restricted additive/coverage diagnostic does not reinstate a greedy guarantee for VDFM's complementary, interference-sensitive utility.

## DVC and Jupyter

Cycle-local DVC 3.67.1 reproduced four stages:

1. symbolic;
2. exact/figures;
3. properties;
4. resource checks.

`dvc status` reports: `Data and pipelines are up to date.`

The Jupyter notebook parses and validates under nbformat 5.11.1 with stable cell IDs. It is a launcher; the authoritative outputs are the DVC-tracked JSON files and PNG figures.

## Plot audit

All six PNG files were opened and inspected:

- `semantic-transfer-curve.png`;
- `memory-growth.png`;
- `stability-plasticity-frontier.png`;
- `retrieval-counterexample.png`;
- `revision-dependency-region.png`;
- `compounding-reachability-phase.png`.

Axes, labels, legends, state regions, and trends match the JSON construction. The dependency-graph plot's initial rightmost label was clipped; the plotting bounds were expanded, the DVC exact stage was rerun, and the corrected image was reinspected.

## Prior-art audit

The bounded primary-source audit was written independently in `prior-art-audit.md`. It found strong component and integration collisions, especially PlugMem, Rosenbloom/Soar, DreamCoder, TMS/ATMS, MACLA, TRUSTMEM/MemGuard, AgentCL, and classical Bayesian/MDL/submodular work. It did not locate the full X65A conjunction, but this is only a provisional bounded-search observation. No categorical novelty claim is supported.

## Corrections and retained negative results

- An aggregate audit command initially ran `lake env lean` from the cycle directory, so Lean could not resolve Mathlib. The exact same proof file was rerun from the project-local Lean workspace and passed with exit code 0; this was an invocation error, not a proof failure.
- The symbolic source-reliability check initially asserted the sign of a cancelled numerator without retaining the denominator's sign. It was replaced by an equality for the full posterior-minus-prior expression and then passed.
- General retrieval utility is neither monotone nor submodular; the main pilot must use exact bounded-frontier selection and cannot cite an ordinary greedy guarantee.
- Direct no-leakage checks exclude stored targets but cannot prove that no covert encoding channel exists.
- Posterior sufficiency is model-relative and does not protect against a misspecified latent family.
- The two-part byte curve proves only that one specified toy code is shorter, not that it preserves useful knowledge.
- The procedural search count is a depth-layer construction, not an end-to-end exponential-speedup result.
- Recent prior art narrows any potential contribution to an unproven systems integration plus its causal protocol.
