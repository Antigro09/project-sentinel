# OCI verification record

**Run date:** 2026-08-26  
**Environment:** dedicated `.venv-math-research` and pinned project-local Lean/Mathlib workspace.

## Verdict

All checks included in this cycle passed. They establish a finite structural core and expose its boundaries. They do not establish that the full OCI induction architecture works in Sentinel.

## Lean 4 + Mathlib

Command:

```bash
source "/Users/anthonycavero/Documents/Startup/project-sentinel/math-findings/activate-math-research.sh"
cd "/Users/anthonycavero/Documents/Startup/project-sentinel/.math-research-tools/lean/sentinel_math"
lake env lean "/Users/anthonycavero/Documents/Startup/project-sentinel/math-findings/02 Systematic Compositional Reasoning/OCI Operadic Contract Induction/formal/OCI.lean"
```

Result: exit code 0 under Lean 4.34.0-rc2. The file contains no `sorry`.

Mechanically checked:

- uniqueness of the constructor-respecting fold;
- exact evaluation for a term assumed outside an arbitrary training set;
- context-closed one-step rewrite preservation;
- finite rewrite-chain preservation;
- semantic soundness of a reachable normalizer result;
- finite-sum form of the scalar depth-error recurrence.

Not mechanically checked:

- the full dependent colored-operad representation;
- identifiability modulo finite automorphism groups;
- termination or confluence of an induced rewrite system;
- the heterogeneous metric path-product bound;
- any full OCI learning or adaptation result.

Rocq/Coq and Isabelle were not installed. No secondary formal proof is claimed.

## SymPy

`results/symbolic-checks.json` records:

- recurrence residual: 0;
- affine composition associativity: passed;
- two-level heterogeneous path-sum identity: passed;
- false associativity defect:

\[
\eta xyz(\eta x^2+1)(\eta xy+2);
\]

- binary signature counting values through 64 candidates.

## Bounded numerical falsification

`results/falsification-results.json` records seed `20260826`.

- Signature-separation probability for 0, 1, 2 probes: 0.000, 0.160, 1.000.
- Mean observational class size: 8.000, 1.385, 1.000.
- OOD-word commitment coverage: 0.000, 0.283981, 1.000.
- Conditional exact accuracy after commitment: 1.000.
- False commitments: 0 in the exact singleton-gated toy.
- Contract bound violations: 0 across 1,980 generated graphs.
- Depth-10 bound: 0.083331; median actual error: 0.012853.
- False-rule maximum sampled defect: 42.030835.
- Exact finite fixed-family minimum: one probe, with two minimal separating probe sets.

The flat memorizer's OOD coverage was zero by construction. This is a support sanity check, not a strong comparative result.

## Property-based checks

Hypothesis checked 500 finite affine composition cases and found the counterexample `(1, 1, 1)` for the perturbed false associativity law, with defect 0.231 at `eta=0.1`.

## Targeted resource checks

`results/resource-checks.json` records:

- SciPy recovered the exact hidden generator permutation.
- CVXPy recovered sensitivities approximately `(0.3, 0.4)` with optimal objective `0.7000000000465147`.
- JAX differentiated the unseen word `(0, 2, 1, 2, 0)` and produced finite gradients.
- NumPyro constructed a normalized finite symbol posterior.
- Hydra loaded `experiments/oci.yaml`.
- MLflow logged local run `7bc5d6e46fec4b1787b11f09f2bd4804` to the project-local store.

## Plot audit

All five PNG files were opened and inspected:

- `signature-identifiability.png`;
- `ood-composition-transfer.png`;
- `error-bound-vs-depth.png`;
- `stability-region.png`;
- `invalid-rewrite-boundary.png`.

Axes, labels, legends, and numerical trends match the JSON outputs.

## Corrections and retained negative results

- The exact OOD theorem is a conditional free-algebra result, not evidence of autonomous semantic discovery.
- A sampled equation is not a universal certificate.
- A semantics-preserving rewrite chain does not establish termination or confluence.
- Conditional accuracy of 1.0 follows from exact singleton binding in a deliberately favorable finite family.
- The experiment has not compared OCI against the decisive static-interpreter-plus-active-learning baseline.
- The novelty audit found strong component collisions; novelty remains a provisional integration hypothesis.
