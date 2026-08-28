# X65A — Structured Continual Memory

This cycle develops Verified Dependency-Factored Memory (VDFM), an exact finite hypothesis for continual semantic and procedural transfer, local belief revision, bounded consolidation, and clean-restart persistence.

Production implementation is **blocked until final frozen X64H passes**. The completed artifacts are theory, formal proofs, finite falsification checks, and an implementation handoff.

## Start here

- `x65a-theory-cycle.md` — complete strict A–I research cycle and smallest decisive pilot.
- `CLAUDE-CODE-IMPLEMENTATION-SPEC.md` — fail-closed implementation contract for Claude Code.
- `prior-art-audit.md` — bounded primary-source novelty audit and strongest component collisions.
- `VERIFICATION.md` — exact checks, formal status, limitations, and reproduction commands.

## Reproducible artifacts

- `formal/X65A.lean` — mechanically checked finite theorems.
- `x65a_symbolic_checks.py` — SymPy derivation checks.
- `x65a_exact_checks.py` — exact enumeration, dependency/revision/restart checks, and Matplotlib figures.
- `x65a_property_checks.py` — 1,500 Hypothesis-generated boundary cases.
- `x65a_resource_checks.py` — SciPy, CVXPy, JAX, NumPyro, Hydra, and local MLflow checks.
- `configs/x65a.yaml` — frozen toy-check configuration.
- `x65a_reproducibility.ipynb` — Jupyter launcher and result summary.
- `dvc.yaml` and `dvc.lock` — local DVC reproduction graph.
- `results/` — authoritative machine-readable finite-check outputs.
- `figures/` — six generated plots.

## Scientific status

- **REPRODUCED:** Lean compiled eight finite claims without `sorry`; DVC reproduced all four computational stages.
- **MEASURED:** symbolic, exact finite, process-restart, graph, optimization, probabilistic, and property checks passed in the toy model.
- **HYPOTHESIS:** VDFM can beat budget-matched replay on new compositions while retaining and revising knowledge.
- **UNKNOWN:** every X65A gate in Sentinel.
- **BLOCKED:** production X65 work until final X64H passes.

The prior-art audit found strong collisions, especially PlugMem, DreamCoder, Rosenbloom/Soar, TMS/ATMS, verifier-governed memory, and AgentCL. No categorical novelty claim is warranted.

## Re-run

From the `project-sentinel` root:

```bash
source math-findings/activate-math-research.sh
cd "math-findings/03 Continual Learning Without Catastrophic Forgetting/X65A Structured Continual Memory"
dvc repro
cd "/Users/anthonycavero/Documents/Startup/project-sentinel/.math-research-tools/lean/sentinel_math"
lake env lean "/Users/anthonycavero/Documents/Startup/project-sentinel/math-findings/03 Continual Learning Without Catastrophic Forgetting/X65A Structured Continual Memory/formal/X65A.lean"
```

