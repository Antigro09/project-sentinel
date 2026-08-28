# Phase-2 Math-Research Toolchain

The SHWM setup reused the dedicated environment already installed beside the
Phase-1 checkout. It did not modify Sentinel's application `.venv`.

## Executed environment

| Purpose | Version |
|---|---|
| Lean / Mathlib | Lean 4.34.0-rc2; existing Mathlib workspace |
| SymPy | 1.14.0 |
| NumPy / SciPy | 2.5.2 / 1.18.1 |
| JAX | 0.11.1 |
| CVXPy | 1.9.2 |
| NetworkX | 3.6.1 |
| NumPyro | 0.21.0 |
| Matplotlib | 3.11.1 |
| Hydra | 1.3.5 |
| MLflow | 3.15.2 |
| Hypothesis | 6.165.10 |
| DVC | 3.67.1 |

Python executable used:

```text
/Users/anthonycavero/Documents/Startup/project-sentinel/.venv-math-research/bin/python
```

Lean workspace used:

```text
/Users/anthonycavero/Documents/Startup/project-sentinel/.math-research-tools/lean/sentinel_math
```

The phase-independent environment lock remains in the primary checkout's
`math-findings/requirements-math-research.lock`. The SHWM cycle also records a
short exact-version list. Coq/Rocq, Isabelle, Mathematica, and Maple were not
available and were not claimed as executed.

The first MLflow attempt used the legacy file backend and failed because MLflow
3.15 places that backend in maintenance mode. The script was corrected to a
local SQLite store and then passed. This tooling failure changed no math result.

An independent clean-shell rerun later showed that DVC stages using a bare
`python` were not self-contained. `dvc.yaml` now pins every stage to the
dedicated interpreter above. All four stages reran successfully; the failed
invocation changed no math result.
