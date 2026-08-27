"""X65A-0: the frozen latent-factor table.

The addendum requires this table to be emitted BEFORE the posterior is
implemented, with

    K = product(cardinality[f] for f in latent_factors)
    assert K <= 256

and it forbids hiding a larger factorized product behind separate arrays
while calling the joint exact. So the joint really is enumerated as one
list of K states, and `ExactPosterior` refuses any state list it did not
get from here.

Verified deterministic procedures are NOT latent factors. The four
primitives of the pilot are verified over a declared finite domain, so
their validity carries no modelled uncertainty and contributes no
cardinality. Uncertainty over procedure validity may be modelled, but only
by paying for it: two 2-level validity factors take K from 64 to 256, which
saturates the budget exactly. A third would be 512 and is refused.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .types import content_id

K_MAX = 256


@dataclass(frozen=True)
class Factor:
    name: str
    values: tuple
    origin: str
    note: str

    @property
    def cardinality(self) -> int:
        return len(self.values)


CORE = (
    Factor("phi_0", (0, 1), "x64h",
           "frozen X64H convention bit; the final posterior p(phi|H) enters "
           "as ONE semantic factor, not as the memory system"),
    Factor("phi_1", (0, 1), "x64h", "second frozen X64H convention bit"),
    Factor("sigma_0", (0, 1), "semantic", "reusable semantic atom"),
    Factor("sigma_1", (0, 1), "semantic", "reusable semantic atom"),
    Factor("context", (0, 1), "context",
           "context boundary / explicit convention switch"),
    Factor("source_reliability", (0, 1), "reliability",
           "reliable | unreliable source; confounded with claim truth "
           "without repeated sources or differing contexts"),
)

PROCEDURE_VALIDITY = (
    Factor("pi_0_valid", (0, 1), "procedural",
           "modelled uncertainty over a procedure's validity; OFF by default"),
    Factor("pi_1_valid", (0, 1), "procedural", "as above"),
)

# Per-task nuisance variables. Marginalised INSIDE the task likelihood and
# deliberately not part of Lambda, so they never inflate K.
NUISANCE = {
    "fault": ("OLD_WRONG", "NEW_CORRUPT", "CONTEXT_SHIFT", "PARSE_ERROR"),
    "composition": ("c", "z", "b"),
}


def factors(procedure_validity: bool = False) -> tuple:
    return CORE + (PROCEDURE_VALIDITY if procedure_validity else ())


def cardinality(procedure_validity: bool = False) -> int:
    k = 1
    for f in factors(procedure_validity):
        k *= f.cardinality
    return k


def states(procedure_validity: bool = False) -> tuple:
    """The canonical finite enumeration. One list of K joint states."""
    fs = factors(procedure_validity)
    k = cardinality(procedure_validity)
    if k > K_MAX:
        raise ValueError(f"latent cardinality {k} exceeds the frozen "
                         f"budget {K_MAX}; the pilot may not silently grow")
    return tuple(tuple(zip((f.name for f in fs), combo))
                 for combo in itertools.product(*(f.values for f in fs)))


def table(procedure_validity: bool = False) -> dict:
    fs = factors(procedure_validity)
    return {
        "factors": [{"name": f.name, "cardinality": f.cardinality,
                     "origin": f.origin, "note": f.note} for f in fs],
        "K": cardinality(procedure_validity),
        "K_max": K_MAX,
        "within_budget": cardinality(procedure_validity) <= K_MAX,
        "nuisance_marginalised_in_likelihood": {k: len(v)
                                                for k, v in NUISANCE.items()},
        "procedure_validity_modelled": procedure_validity,
        "headroom_factor": K_MAX // cardinality(procedure_validity),
        "digest": content_id("latent", [
            (f.name, f.cardinality, f.origin) for f in fs]),
    }
