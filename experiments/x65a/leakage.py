"""X65A-0: leakage snapshots and canaries.

The direct checks catch literal storage of a future answer. They do NOT
establish statistical independence of a future answer from memory, and they
must not be described that way: legitimate compositional transfer is
supposed to make some future answers inferable. Claiming otherwise would
contradict the experiment's own goal.

Structural canaries cover the case the literal check misses -- an alias, a
content hash, or an evaluator id that encodes a target without containing
it.
"""

from __future__ import annotations

import hashlib

from .types import FORBIDDEN_KEYS, Taint, TaintError, canon, encode

FORBIDDEN_CANARY = "future_target_answer"


def snapshot(state) -> dict:
    return canon(state)


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def assert_clean(snap: dict, task_index: int, target_program=None,
                 target_logical_form=None, expected_outputs=None) -> dict:
    """Run before every task. Every failure is fatal."""
    found = []
    blob = encode(snap)
    text = blob.decode("utf-8")

    for k in FORBIDDEN_KEYS:
        if f'"{k}"' in text:
            found.append(f"forbidden key {k}")
    for t in (Taint.ORACLE_ONLY, Taint.TARGET_ONLY, Taint.FUTURE):
        if t.value in text:
            found.append(f"non-persistable taint {t.value}")
    if FORBIDDEN_CANARY in text:
        found.append("forbidden canary value")

    for e in snap.get("active", []):
        at = e.get("acquired_at", e.get("created_at"))
        if at is not None and at >= task_index:
            found.append(f"entry acquired_at {at} >= current task {task_index}")

    # direct target checks
    for label, obj in (("program", target_program),
                       ("logical form", target_logical_form),
                       ("expected outputs", expected_outputs)):
        if obj is None:
            continue
        needle = encode(obj).decode("utf-8").strip('"')
        if needle and needle in text:
            found.append(f"current target {label} present verbatim")
        # structural canary: the target's content hash appearing as an alias
        h = hashlib.sha256(encode(obj)).hexdigest()
        for s in _strings(snap):
            if h[:16] in s:
                found.append(f"current target {label} present as a content hash")
                break

    return {"clean": not found, "violations": found,
            "checked_bytes": len(blob),
            "note": "absence of a literal target does not establish "
                    "statistical independence of a future answer from memory"}


def contaminated_fixture(snap: dict) -> dict:
    """The calibration arm. A clean snapshot check is worthless unless a
    deliberately contaminated one fails."""
    bad = dict(snap)
    bad["planted"] = FORBIDDEN_CANARY
    return bad
