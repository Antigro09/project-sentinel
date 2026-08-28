"""Red-capable regressions for the X65A-L1 audit.

These begin at the evaluator seams that produced the misleading X65A-L
headline.  Each test names the semantic failure, rather than merely checking
that a runner completes.
"""

import sys

sys.path.insert(0, "experiments")

from x65a import latent_id as LI
from x65a_l_latent import summarise


def _row(*, kind: str, outcome: str) -> dict:
    return {
        "kind": kind,
        "outcome": outcome,
        "correct": False,
        "literal": False,
        "equivalent": False,
        "rank": None,
        "queries": 0,
        "bytes_scanned": 0,
        "bytes_retrieved": 0,
        "nodes": 0,
        "wrote": False,
        "units": 1,
    }


def test_unresolved_new_identity_is_not_recall():
    """Permanent abstention prevents damage but does not learn an identity."""
    got = summarise([_row(kind="new", outcome=LI.UNRESOLVED_IDENTITY)])
    assert got["new_identity_recall"] == 0.0


def test_create_new_identity_is_recall():
    got = summarise([_row(kind="new", outcome=LI.CREATE_NEW)])
    assert got["new_identity_recall"] == 1.0
