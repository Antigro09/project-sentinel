"""Preflight the two named frozen encoder families and record the verdict.

Run before anything is encoded. If either family is not faithfully runnable the
matrix stops here, and the reason is written down rather than worked around: a
replacement requires a reviewed pre-run amendment to the matrix, and after any
cell has run the whole matrix restarts.

    uv run python experiments/shwm/backbone_preflight.py [--offline] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from sentinel.wm.backbones import matrix_may_run, preflight_all  # noqa: E402
from sentinel.wm.provenance import environment_state, git_state  # noqa: E402
from sentinel.wm.versioning import digest_of  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="skip network checks")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts/shwm/scale0/backbone-preflight.json")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="exit 0 even when a family is blocked, so a pipeline stage still records the verdict",
    )
    arguments = parser.parse_args()

    preflights = preflight_all(allow_network=not arguments.offline)
    permitted, reason = matrix_may_run(preflights)

    document = {
        "git": git_state(REPO),
        "environment": environment_state(),
        "preflights": [p.canonical_dict() for p in preflights],
        "matrix_may_run": permitted,
        "reason": reason,
    }
    document["digest"] = digest_of(
        {"preflights": document["preflights"], "matrix_may_run": permitted}
    )

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    for preflight in preflights:
        candidate = preflight.candidate
        print(f"{candidate.encoder_id:14s} {candidate.repository:28s} {preflight.verdict.value}")
        print(f"    revision : {preflight.revision}")
        print(f"    licence  : {preflight.licence}   gated: {preflight.gated}")
        print(f"    params   : {preflight.total_parameters:,}" if preflight.total_parameters else "")
        print(f"    detail   : {preflight.detail}")
    print()
    print(f"matrix may run: {permitted}")
    print(f"reason        : {reason}")
    print(f"written       : {arguments.out}")
    if permitted or arguments.allow_blocked:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
