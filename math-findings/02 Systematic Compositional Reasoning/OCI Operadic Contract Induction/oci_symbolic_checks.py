"""Symbolic checks for Operadic Contract Induction (OCI).

These checks validate algebra used in the paper derivations.  They do not
constitute evidence that the proposed learning mechanism works empirically.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "symbolic-checks.json"


def affine(a: sp.Expr, b: sp.Expr, value: sp.Expr) -> sp.Expr:
    return a * value + b


def compose_affine(
    outer: tuple[sp.Expr, sp.Expr], inner: tuple[sp.Expr, sp.Expr]
) -> tuple[sp.Expr, sp.Expr]:
    """Return coefficients of outer(inner(x))."""
    outer_a, outer_b = outer
    inner_a, inner_b = inner
    return outer_a * inner_a, outer_a * inner_b + outer_b


def main() -> None:
    n = sp.symbols("n", integer=True, nonnegative=True)
    rho = sp.symbols("rho", nonnegative=True)
    epsilon, initial = sp.symbols("epsilon initial", nonnegative=True)

    # Closed form for E_{d+1} = epsilon + rho E_d.
    closed = rho**n * initial + epsilon * (1 - rho**n) / (1 - rho)
    recurrence_residual = sp.factor(
        closed.subs(n, n + 1) - epsilon - rho * closed
    )
    initial_residual = sp.simplify(closed.subs(n, 0) - initial)
    assert recurrence_residual == 0
    assert initial_residual == 0

    # Operadic grafting specializes to associative function composition for
    # unary affine generators.  Parenthesization cannot change denotation.
    x = sp.symbols("x")
    a1, a2, a3, b1, b2, b3 = sp.symbols("a1 a2 a3 b1 b2 b3")
    f1, f2, f3 = (a1, b1), (a2, b2), (a3, b3)
    left_coeffs = compose_affine(compose_affine(f1, f2), f3)
    right_coeffs = compose_affine(f1, compose_affine(f2, f3))
    assert all(
        sp.expand(left - right) == 0
        for left, right in zip(left_coeffs, right_coeffs, strict=True)
    )
    left_assoc = affine(left_coeffs[0], left_coeffs[1], x)
    right_assoc = affine(right_coeffs[0], right_coeffs[1], x)
    assert sp.expand(left_assoc - right_assoc) == 0

    # Path-sum form of the heterogeneous tree error theorem for a two-level
    # binary tree.  Each local error is transported by ancestor sensitivity.
    er, ec = sp.symbols("epsilon_root epsilon_child", nonnegative=True)
    lr1, lr2, lc1, lc2 = sp.symbols(
        "L_root_1 L_root_2 L_child_1 L_child_2", nonnegative=True
    )
    e1, e2, e3 = sp.symbols("e1 e2 e3", nonnegative=True)
    recurrence_tree = er + lr1 * (ec + lc1 * e1 + lc2 * e2) + lr2 * e3
    path_sum_tree = er + lr1 * ec + lr1 * lc1 * e1 + lr1 * lc2 * e2 + lr2 * e3
    assert sp.expand(recurrence_tree - path_sum_tree) == 0

    # A false associativity certificate must be rejectable.  This polynomial
    # operation is associative only on a restricted algebraic boundary.
    eta, y, z = sp.symbols("eta y z")

    def star(left: sp.Expr, right: sp.Expr) -> sp.Expr:
        return left + right + eta * left**2 * right

    associativity_defect = sp.factor(star(star(x, y), z) - star(x, star(y, z)))
    assert sp.simplify(associativity_defect.subs(eta, 0)) == 0
    assert associativity_defect != 0

    # The binary-signature counting lower bound is m >= ceil(log_2 k).
    # Enumerate a finite sanity range rather than pretending SymPy proves the
    # underlying information-theoretic theorem.
    signature_lower_bounds = {
        str(k): next(m for m in range(32) if 2**m >= k) for k in range(1, 65)
    }

    result = {
        "status": "MEASURED: symbolic identities executed successfully",
        "sympy_version": sp.__version__,
        "depth_recurrence_closed_form": str(closed),
        "depth_recurrence_residual": str(recurrence_residual),
        "affine_composition_associativity": True,
        "heterogeneous_path_sum_identity": True,
        "nonassociative_star_defect": str(associativity_defect),
        "binary_signature_lower_bounds_k_1_to_64": signature_lower_bounds,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
