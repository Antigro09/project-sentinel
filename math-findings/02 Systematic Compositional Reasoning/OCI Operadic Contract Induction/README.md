# OCI — Operadic Contract Induction

This cycle develops and falsifies a proof-scoped mechanism for executing unseen typed composition trees after their primitive symbols have been identified.

## Start here

- `oci-research-cycle.md` — complete strict A–I research cycle.
- `CLAUDE-CODE-IMPLEMENTATION-SPEC.md` — isolated OCI-1 experiment contract.
- `prior-art-audit.md` — bounded primary-source novelty audit and candidate rejection.
- `VERIFICATION.md` — exact checks run, results, and formal/non-formal status.

## Reproducible artifacts

- `formal/OCI.lean` — mechanically checked binary-tree core.
- `oci_symbolic_checks.py` — SymPy derivation checks.
- `oci_falsification.py` — exact finite and graph-based stress tests plus Matplotlib figures.
- `oci_resource_checks.py` — SciPy, JAX, CVXPy, NumPyro, Hydra, and local MLflow checks.
- `oci_property_checks.py` — Hypothesis properties and counterexample generation.
- `experiments/oci.yaml` — frozen resource-check configuration.
- `results/` — machine-readable outputs.
- `figures/` — five generated plots.

## Scientific status

- **REPRODUCED:** Lean compiled six core results with no `sorry`.
- **MEASURED:** finite symbol signatures, structurally held-out affine words, contractive trees, false rewrite boundaries, and resource checks ran locally.
- **HYPOTHESIS:** the full OCI learning loop outperforms matched active-symbolic baselines.
- **UNKNOWN:** effectiveness on natural language, learned semantic parsers, unrestricted novel primitives, or integrated Sentinel tasks.
- **SPECULATIVE:** broad AGI relevance.

The exact guarantee is conditional: identified primitive semantics plus a law-respecting typed evaluator determine every legal unseen composition tree. It is not evidence that arbitrary new meanings can be inferred without distinguishing observations.

## Re-run

From the `project-sentinel` root:

```bash
source math-findings/activate-math-research.sh
python "math-findings/02 Systematic Compositional Reasoning/OCI Operadic Contract Induction/oci_symbolic_checks.py"
python "math-findings/02 Systematic Compositional Reasoning/OCI Operadic Contract Induction/oci_falsification.py"
python "math-findings/02 Systematic Compositional Reasoning/OCI Operadic Contract Induction/oci_resource_checks.py"
python "math-findings/02 Systematic Compositional Reasoning/OCI Operadic Contract Induction/oci_property_checks.py"
cd .math-research-tools/lean/sentinel_math
lake env lean "/Users/anthonycavero/Documents/Startup/project-sentinel/math-findings/02 Systematic Compositional Reasoning/OCI Operadic Contract Induction/formal/OCI.lean"
```
