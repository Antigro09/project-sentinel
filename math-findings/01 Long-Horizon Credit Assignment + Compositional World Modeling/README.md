# Long-Horizon Memory Math Findings

This folder records a mathematical research direction for Project Sentinel: **Boundary Contract Dual Ledger (BCDL)**.

## What it is

BCDL is best classified as a **long-horizon causal-credit and world-model revision mechanism**. It is relevant to long-horizon memory because it tries to preserve information about which intervention changed an outcome and which model component conflicts with verified evidence across a long chain of modules.

It is **not yet a complete memory architecture**. The supplied work does not measure long-term retention, forgetting, memory growth, retrieval quality, or cross-domain reuse. Those remain open.

## Current verdict

- **PROVEN (paper level):** cut-invariant causal-credit conservation and the greatest-safe local revision theorem, under explicit assumptions.
- **MEASURED:** exhaustive finite checks and a 4,000-trial numerical localization study.
- **RETRACTED / REJECTED:** Counterfactual Cutset Dividends as a novel mechanism; it reduces to existing Harsanyi/Shapley-style attribution.
- **UNKNOWN:** mechanical proof in Lean or Coq, integrated novelty, learnability of compact sufficient boundaries, and practical value for Sentinel.
- **HYPOTHESIS:** keeping causal credit and epistemic model correction in separate ledgers improves safe continual correction and compositional transfer.
- **NOT ESTABLISHED:** AGI, human-level reasoning, general long-term memory, or superiority over existing approaches.

## Files

- `Boundary Contract Dual Ledger - Math Findings.docx` — polished research note for reading and sharing inside the project.
- `boundary-contract-dual-ledger-math-findings.md` — text-first canonical version with equations and evidence labels.
- `original-research-report.txt` — exact copy of the user-provided source report for provenance.

The decisive next step is a controlled hidden-state-aliasing experiment that compares fixed boundaries with query-preserving boundary refinement and preregisters failure conditions.
