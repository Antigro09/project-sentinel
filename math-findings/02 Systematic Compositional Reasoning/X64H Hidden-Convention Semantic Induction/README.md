# X64H — Hidden-Convention Semantic Induction

This folder contains the mathematical theory, formal checks, prior-art audit, finite falsification evidence, and Claude Code implementation handoff for X64H.

## Start here

- `x64h-theory-research-brief.md` — complete A–I research cycle and all 17 requested deliverables.
- `CLAUDE-CODE-IMPLEMENTATION-SPEC.md` — direct implementation contract.
- `prior-art-audit.md` — bounded audit of 27 primary sources and the novelty delta.
- `VERIFICATION.md` — executed checks, regression result, corrected numerical pitfall, and explicit non-results.

## Reproducible evidence

- `formal/X64H.lean` — checked Lean 4 + Mathlib proofs.
- `x64h_symbolic_checks.py` — SymPy checks for posterior, conflict, and commitment identities.
- `x64h_exact_enumeration.py` — exact finite identifiability, noise, and query calculations plus Matplotlib figures.
- `results/symbolic-checks.json` — executed symbolic results.
- `results/exact-enumeration.json` — executed exact finite results.
- `figures/identifiability-probability.png`
- `figures/noisy-signature-recovery.png`
- `figures/query-policy-comparison.png`

## Scientific status

- The deterministic authored-inverse theorem and finite separating-signature results are mechanically checked.
- The symbolic identities and finite enumerations were executed.
- The full FT-SPCFG mechanism and X64H gates have not been implemented or empirically tested.
- The literature result is bounded: no audited source covered the full composition, but every individual ingredient has prior art.

## Re-run local checks

From the `project-sentinel` root:

```bash
source math-findings/activate-math-research.sh
python "math-findings/02 Systematic Compositional Reasoning/X64H Hidden-Convention Semantic Induction/x64h_symbolic_checks.py"
python "math-findings/02 Systematic Compositional Reasoning/X64H Hidden-Convention Semantic Induction/x64h_exact_enumeration.py"
cd .math-research-tools/lean/sentinel_math
lake env lean "/Users/anthonycavero/Documents/Startup/project-sentinel/math-findings/02 Systematic Compositional Reasoning/X64H Hidden-Convention Semantic Induction/formal/X64H.lean"
```
