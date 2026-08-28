"""SymPy checks for the X65A finite structured-memory derivations.

These are algebra checks, not empirical evidence that the proposed continual
learner works.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "symbolic-checks.json"


def main() -> None:
    p, l_true, l_false = sp.symbols(
        "p l_true l_false", positive=True, finite=True
    )
    evidence = sp.expand(p * l_true + (1 - p) * l_false)
    post_true = sp.cancel(p * l_true / evidence)
    post_false = sp.cancel((1 - p) * l_false / evidence)
    normalization = sp.simplify(post_true + post_false)
    assert normalization == 1

    # A counter-observation reports that h is false.  A reliable source is
    # correct with probability r, so P(counter | h)=1-r and
    # P(counter | not h)=r.
    r = sp.symbols("r", positive=True, finite=True)
    revision_denominator = p * (1 - r) + (1 - p) * r
    revised = sp.cancel(p * (1 - r) / revision_denominator)
    revision_delta = sp.factor(
        p * (1 - p) * (1 - 2 * r) / revision_denominator
    )
    # For 0<p<1, the sign is the sign of 1-2r: trusted counterevidence lowers
    # the belief exactly when r>1/2.
    assert sp.simplify((revised - p) - revision_delta) == 0

    # Factorized revision locality: update A, retain B's marginal.
    a0, a1, b = sp.symbols("a0 a1 b", nonnegative=True, finite=True)
    za = a0 + a1
    marginal_b = sp.simplify((a0 / za) * b + (a1 / za) * b)
    assert marginal_b == b

    # Two-part MDL threshold.
    n, raw, residual, component = sp.symbols(
        "n raw residual component", positive=True, finite=True
    )
    raw_code = n * raw
    consolidated_code = component + n * residual
    mdl_saving = sp.factor(raw_code - consolidated_code)
    threshold = sp.solve_univariate_inequality(
        consolidated_code < raw_code, component
    )
    assert sp.simplify(mdl_saving - (n * (raw - residual) - component)) == 0

    # Posterior odds depend on the history only through finite counts when
    # observations are conditionally i.i.d. given the latent component.
    odds0, lr_pos, lr_neg = sp.symbols(
        "odds0 lr_pos lr_neg", positive=True, finite=True
    )
    positives, negatives = sp.symbols("positives negatives", integer=True, nonnegative=True)
    odds_closed = odds0 * lr_pos**positives * lr_neg**negatives
    odds_reordered = odds0 * lr_neg**negatives * lr_pos**positives
    assert sp.simplify(odds_closed - odds_reordered) == 0

    # Search-count reduction from verified macros.  The ratio is exact for the
    # deliberately simple full d-ary enumeration model.
    d, length, compressed_length = sp.symbols(
        "d length compressed_length", positive=True, finite=True
    )
    search_ratio = sp.simplify(d**length / d**compressed_length)
    assert search_ratio == d ** (length - compressed_length)

    result = {
        "status": "MEASURED: SymPy derivation checks executed; no capability claim",
        "posterior": {
            "evidence": str(evidence),
            "true": str(post_true),
            "false": str(post_false),
            "normalization": str(normalization),
        },
        "reliable_counterevidence": {
            "posterior": str(revised),
            "posterior_minus_prior": str(revision_delta),
            "interpretation": "for 0<p<1, posterior decreases iff r>1/2",
        },
        "revision_locality_marginal": str(marginal_b),
        "mdl": {
            "saving": str(mdl_saving),
            "acceptance_region": str(threshold),
        },
        "finite_sufficient_statistic_odds": str(odds_closed),
        "macro_search_ratio": str(search_ratio),
        "sympy_version": sp.__version__,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
