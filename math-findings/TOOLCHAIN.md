# Sentinel Math-Research Toolchain

Status: installed and smoke-tested on 2026-08-26.

## Activate

From a terminal:

    source "/Users/anthonycavero/Documents/Startup/project-sentinel/math-findings/activate-math-research.sh"

The script activates the dedicated Python environment, the project-local Elan installation, and a local SQLite MLflow store. It does not modify shell startup files or Sentinel's pre-existing .venv.

The Jupyter kernel is registered as **Sentinel Math Research (Python 3.12)**.

## Installed locations

| Component | Location |
|---|---|
| Python 3.12 environment | /Users/anthonycavero/Documents/Startup/project-sentinel/.venv-math-research |
| Elan and Lean toolchains | /Users/anthonycavero/Documents/Startup/project-sentinel/.math-research-tools/elan |
| Lean + Mathlib workspace | /Users/anthonycavero/Documents/Startup/project-sentinel/.math-research-tools/lean/sentinel_math |
| Persistent MLflow database after first real run | /Users/anthonycavero/Documents/Startup/project-sentinel/.math-research-tools/mlflow.db |

Both generated environment directories are excluded locally from Git. The Python environment occupies about 1.4 GB; the Lean/Mathlib cache occupies about 11 GB.

## Verified core versions

| Purpose | Installed implementation |
|---|---|
| Formal verification | Elan 4.2.4; Lean 4.34.0-rc2; Lake 5.0.0; Mathlib revision 85e3a25e006c35636f0e53b0e9296caca2685bc0 |
| Symbolic mathematics | SymPy 1.14.0 |
| Numerical computing | NumPy 2.5.2; SciPy 1.18.1 |
| Differentiable mechanisms | JAX/JAXlib 0.11.1; Optax 0.2.8 |
| Constrained optimization | CVXPy 1.9.2 with Clarabel, OSQP, SCS, and HiGHS backends |
| Graph and probabilistic work | NetworkX 3.6.1; NumPyro 0.21.0 |
| Visualization | Matplotlib 3.11.1; Seaborn 0.13.2; Plotly 7.0.0 |
| Reproducibility | JupyterLab 4.6.3; Papermill 2.7.0; Hydra 1.3.5; MLflow 3.15.2; DVC 3.67.1 |
| Robustness/testing | Pytest 9.1.1; Hypothesis 6.165.10; scikit-learn 1.9.0 |

The complete transitive version lock is in requirements-math-research.lock. The shorter human-maintained package list is in requirements-math-research.in.

## Verification performed

The offline Python smoke test passed all of:

- symbolic polynomial identity simplification;
- NumPy linear solve;
- SciPy ODE integration against a closed form;
- CVXPy constrained optimization with Clarabel;
- JAX automatic differentiation;
- NumPyro distribution construction;
- NetworkX acyclic-graph traversal;
- scikit-learn fitting;
- Hypothesis counterexample generation;
- Matplotlib headless rendering;
- MLflow experiment and metric logging to a temporary SQLite backend.

pip check reported no broken dependencies.

The Lean workspace compiled two Mathlib-backed theorems using ring and finite-sum simplification. The build completed successfully.

Run the checks again with:

    source "/Users/anthonycavero/Documents/Startup/project-sentinel/math-findings/activate-math-research.sh"
    python "/Users/anthonycavero/Documents/Startup/project-sentinel/math-findings/toolchain-smoke-test.py"
    cd "/Users/anthonycavero/Documents/Startup/project-sentinel/.math-research-tools/lean/sentinel_math"
    lake build

## Deliberate substitutions and deferred optional tools

- JAX + NumPyro were selected instead of also installing PyTorch + Pyro. The protocol requires one differentiable framework and one probabilistic framework, and this avoids a redundant heavy stack.
- MLflow was selected instead of Weights & Biases, so experiment tracking works offline and does not require an external account.
- Markdown reports satisfy the protocol's Quarto-or-Markdown path; Quarto is not required for the current workflow.
- Hypothesis is installed. Foolbox remains optional until an adversarial differentiable model warrants it.
- Lean 4 + Mathlib is installed as the primary proof system. Rocq/Coq or Isabelle is intentionally deferred until a theorem is selected for an independent cross-check; neither belongs in a Python venv, and installing one preemptively would add a second large compiler ecosystem.
- Mathematica and Maple are proprietary optional secondary checks and were not installed.

These substitutions are recorded so later reports cannot imply that an unavailable tool was executed.
