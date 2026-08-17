"""Container-side worker. Reads a job on stdin, writes a report on stdout.

Runs inside the sandbox with no network and a memory cap, so it must import
only the pure-Python parts of the package — no arcengine, no numpy, no
matplotlib. `sentinel.verify` and `sentinel.wm` are deliberately free of
those dependencies, which is what makes a plain python:3.12-slim image
sufficient and keeps the image pull small.

Job on stdin:
    {"source": "...", "history": {...}, "timeout": 2.0, "name": "..."}

Report on stdout (always valid JSON, even on failure — the caller must
never have to distinguish a crashed worker from a crashed model):
    {"ok": true,  "report": {...}}
    {"ok": false, "error": "...", "kind": "load|verify|worker"}
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        job = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "kind": "worker", "error": f"bad job: {exc}"}))
        return 0

    try:
        from sentinel.env.history import History
        from sentinel.bootstrap.loader import LoadError, load_model
        from sentinel.verify import Verifier
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "kind": "worker", "error": f"import: {exc}"}))
        return 0

    try:
        history = History.from_json(job["history"])
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "kind": "worker", "error": f"history: {exc}"}))
        return 0

    try:
        model = load_model(
            job["source"],
            timeout=float(job.get("timeout", 2.0)),
            name=str(job.get("name", "generated")),
            context={"INITIAL_GRID": history.initial.grid},
        )
    except LoadError as exc:
        print(json.dumps({"ok": False, "kind": "load", "error": str(exc)}))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "kind": "load", "error": f"{type(exc).__name__}: {exc}"}))
        return 0

    try:
        report = Verifier(stop_on_crash=True).verify(model, history)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "kind": "verify", "error": f"{type(exc).__name__}: {exc}"}))
        return 0

    # Full form, not the summary: every headline metric is computed from
    # `steps`, so a report sent without them would rebuild as all zeros.
    print(json.dumps({"ok": True, "report": report.to_json_full()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
