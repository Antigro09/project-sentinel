"""SymPy checks for the finite SHWM theory claims.

The script checks algebra and resource arithmetic only. It does not train a
world model and cannot establish planning, transfer, causality, or AGI.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "results" / "symbolic-checks.json"


def main() -> None:
    lipschitz, epsilon = sp.symbols("L epsilon", real=True)
    horizon = sp.symbols("n", integer=True, nonnegative=True)

    # Closed form for e_0 = 0 and e_{n+1} = epsilon + L e_n, for L != 1.
    closed = epsilon * (1 - lipschitz**horizon) / (1 - lipschitz)
    closed_next = epsilon * (1 - lipschitz ** (horizon + 1)) / (1 - lipschitz)
    recurrence_residual = sp.factor(
        closed_next - (epsilon + lipschitz * closed)
    )
    assert recurrence_residual == 0

    # Finite Bayesian posterior normalization for two candidate dynamics.
    p0, p1, l0, l1 = sp.symbols("p0 p1 l0 l1", positive=True)
    evidence = p0 * l0 + p1 * l1
    posterior_sum = sp.factor(p0 * l0 / evidence + p1 * l1 / evidence)
    assert posterior_sum == 1

    # A finite precision d-dimensional latent with p bits per coordinate.
    bits, dimension = sp.symbols("p d", integer=True, nonnegative=True)
    coordinate_states = 2**bits
    latent_states = sp.expand_power_base(coordinate_states**dimension, force=True)
    assert sp.simplify(latent_states - 2 ** (bits * dimension)) == 0

    # Open-loop search with b actions and horizon H.
    branching, planning_horizon = sp.symbols(
        "b H", integer=True, nonnegative=True
    )
    sequence_count = branching**planning_horizon
    assert sp.simplify(sequence_count.subs({branching: 4, planning_horizon: 25})) == 4**25

    # Resource arithmetic used in the report. Decimal GB and binary GiB are
    # both emitted so the units cannot be silently conflated.
    transition_count = 1_000_000
    cache_dimension = 512
    bytes_per_coordinate = 2
    cache_bytes = transition_count * cache_dimension * bytes_per_coordinate
    cache_gb_decimal = sp.Rational(cache_bytes, 10**9)
    cache_gib_binary = sp.Rational(cache_bytes, 2**30)
    assert cache_gb_decimal == sp.Rational(128, 125)  # 1.024 GB

    params_15b = 15_000_000_000
    params_70b = 70_000_000_000
    adam_12_bytes = params_15b * 12  # bf16 weights + grads + fp32 m/v
    adam_16_bytes = params_15b * 16  # plus fp32 master weights
    int4_raw_bytes = params_70b * sp.Rational(4, 8)
    assert adam_12_bytes == 180_000_000_000
    assert adam_16_bytes == 240_000_000_000
    assert int4_raw_bytes == 35_000_000_000

    result = {
        "status": "MEASURED: symbolic identities and arithmetic passed; no capability claim",
        "checks": {
            "rollout_recurrence_residual": str(recurrence_residual),
            "finite_posterior_sum": str(posterior_sum),
            "finite_precision_latent_states": "2**(bits*dimension)",
            "open_loop_sequences_b4_h25": int(4**25),
            "latent_cache_bytes_1m_x_512_x_fp16": cache_bytes,
            "latent_cache_gb_decimal": float(cache_gb_decimal),
            "latent_cache_gib_binary": float(cache_gib_binary),
            "15b_adam_state_12_bytes_per_parameter_gb": adam_12_bytes / 10**9,
            "15b_adam_state_16_bytes_per_parameter_gb": adam_16_bytes / 10**9,
            "70b_int4_raw_weight_gb": float(int4_raw_bytes / 10**9),
        },
        "assumptions": [
            "The rollout closed form shown uses L != 1; L = 1 is the linear n*epsilon case.",
            "Training-memory arithmetic excludes activations, allocator overhead, and framework buffers.",
            "The int4 figure is raw packed weight storage and excludes scales, metadata, cache, and runtime overhead.",
            "Finite machine codes do not bound the number of context-dependent descriptions that can refer to a code.",
        ],
        "versions": {"sympy": sp.__version__},
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
