# X64H Verification Record

Date: 2026-08-26

## Passed checks

- Lean 4 + Mathlib accepted `formal/X64H.lean` directly with exit code 0.
- The same theorem module compiled in the local Mathlib workspace.
- `x64h_symbolic_checks.py` completed under SymPy 1.14.0.
- `x64h_exact_enumeration.py` completed with Python warnings treated as errors.
- Exact finite separating probabilities matched the falling-factorial formula.
- Every majority-error sum matched `scipy.stats.binom` within the asserted tolerance.
- Constructive context families attained `ceil(log2(k))` for every `k` from 2 through 16.
- The exact `k=4` clarification calculation returned 2.0 questions for optimal and greedy information gain and 2.857142857 for random disagreement.
- Existing X64E/X64G regression tests passed: 13 tests in 10.54 seconds.
- All three generated figures were visually inspected.

## Corrected during verification

Using a NumPy fixed-width integer as the exponent input to the plotting formula overflowed for large `k`. The exact theorem values and stored `k=4` enumeration were unaffected, but the larger-`k` plot could have been wrong. The implementation now coerces the exponent to an unbounded Python integer, the script passes with warnings promoted to errors, and the figures were regenerated and reinspected.

## Not verified

- No FT-SPCFG X64H implementation exists yet.
- No hidden convention, adaptation, conflict, open-world, or gate result has been run.
- The stochastic Bayes-decoder, Fano, concentration, and calibration results remain paper derivations rather than Lean proofs.
- No empirical claim extends beyond the included toy finite checks.

