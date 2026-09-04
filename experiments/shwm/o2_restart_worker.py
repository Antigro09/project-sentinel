"""Reload a persisted palette memory in a FRESH process and re-score it.

Section H asks whether the memory survives a process restart. Nothing about that is
answered by keeping an object alive in the process that built it, so the check is done
from the other side: this script imports nothing from the run that produced the file,
loads the assignment and the transfer rows from disk, and prints the accuracy it gets.

    .venv-shwm/bin/python experiments/shwm/o2_restart_worker.py <state.npz>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import o2_core as C  # noqa: E402


def main() -> int:
    state = np.load(sys.argv[1])
    assignment = state["assignment"]
    before, after = state["before"], state["after"]
    rows = np.arange(len(before))[:, None, None]
    pb, pa = assignment[rows, before], assignment[rows, after]
    evidence = (pa[..., C.AGENT] * (1.0 - pb[..., C.AGENT])
                * pb[..., C.SWITCH]).reshape(len(before), -1).max(axis=1)
    predicted = (evidence > 0.5).astype(float)
    truth = state["event"]
    per_class = [float((predicted[truth == v] == v).mean()) for v in (0.0, 1.0)
                 if (truth == v).any()]
    print(json.dumps({"rows": int(len(truth)),
                      "balanced_accuracy": float(np.mean(per_class)),
                      "accuracy": float((predicted == truth).mean()),
                      "assignment_bytes": int(assignment.nbytes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
