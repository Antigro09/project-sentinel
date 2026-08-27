"""X65A-S: a genuine restart in the middle of a stream.

The parent runs the stream up to the declared restart point, persists the
semantic store, and dies. A child with a scrubbed environment reloads only
that file, rebuilds the stream deterministically from the seed, and runs the
delayed returns. If the delayed-return effect survives, it survived a real
process boundary and not a round trip inside one interpreter.

Dispatch: `python -m x65a.restart_s parent|child <path> <family> <seed>`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from .types import encode

FORBIDDEN_ENV = "X65A_FORBIDDEN_TARGET"
FORBIDDEN_VALUE = "future_target_answer"
_RUNTIME_CACHE: dict = {}


def _setup(overlap: str, seed: int):
    sys.path.insert(0, "experiments")
    from x64h import episode as EP
    from x64h import family as F
    from x65a import arms_s as A
    from x65a import streams as SR
    fam = F.Family(F.FamilySpec(overlap=overlap))
    beh = EP.behaviour_table(fam.forms)
    cfg = EP.Config(overlap=overlap)
    st = SR.build_stream(fam, beh, cfg, seed)
    return fam, beh, st, A


def run_parent(path: Path, overlap: str, seed: int) -> dict:
    fam, beh, st, A = _setup(overlap, seed)
    _RUNTIME_CACHE["answer"] = os.environ.get(FORBIDDEN_ENV, FORBIDDEN_VALUE)
    arm = A.Arm("main", fam, beh, random.Random(seed))
    for i, app in enumerate(st.appearances[:st.restart_before]):
        arm.observe_episode(app, i)
    payload = {"schema": 1, "overlap": overlap, "seed": seed,
               "records": arm.store.canon(),
               "evidence": arm.evidence.canon()}
    blob = encode(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return {"pid": os.getpid(), "sha256": hashlib.sha256(blob).hexdigest(),
            "records": len(arm.store.records),
            "bytes": arm.store.bytes(),
            "surviving": {r.identity: r.surviving
                          for r in arm.store.records.values()}}


def run_child(path: Path, overlap: str, seed: int) -> dict:
    fam, beh, st, A = _setup(overlap, seed)
    from x65a import semantic_mem as SM
    blob = path.read_bytes()
    d = json.loads(blob.decode())
    arm = A.Arm("main", fam, beh, random.Random(seed))
    for r in d["records"]:
        arm.store.put(SM.SemanticRecord(
            r["identity"],
            tuple(SM.GroundedObservation(g["z"], g["u"], g["e"])
                  for g in r["grounded"]),
            r["n_confirming"], r["n_contradicting"], r["context_id"],
            __import__("x65a.types", fromlist=["Status"]).Status(r["status"]),
            r["last_verified_use"], r["created_at"], r["version"],
            r["supersedes"], None, r["surviving"]))
    ok = n = 0
    for i, app in enumerate(st.appearances):
        if i < st.restart_before:
            continue
        p = arm.prior_for(app)
        if app.kind != "return":
            continue
        for t in app.transfer:
            c, _q = A.solve(fam, beh, p, t, arm.ledger, 0)
            ok += c
            n += 1
    return {"pid": os.getpid(),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "records": len(arm.store.records),
            "surviving": {r.identity: r.surviving
                          for r in arm.store.records.values()},
            "return_transfer": ok / max(1, n), "return_tasks": n,
            "forbidden_in_bytes": FORBIDDEN_VALUE in blob.decode(),
            "forbidden_in_env": any(FORBIDDEN_VALUE in v
                                    for v in os.environ.values()),
            "forbidden_in_globals": FORBIDDEN_VALUE in json.dumps(
                _RUNTIME_CACHE),
            "env_size": len(os.environ)}


def cycle(path: Path, overlap: str, seed: int) -> dict:
    here = str(Path("experiments").resolve())
    penv = dict(os.environ)
    penv[FORBIDDEN_ENV] = FORBIDDEN_VALUE
    penv["PYTHONPATH"] = here
    p = subprocess.run([sys.executable, "-m", "x65a.restart_s", "parent",
                        str(path), overlap, str(seed)], capture_output=True,
                       text=True, env=penv)
    if p.returncode != 0:
        return {"ok": False, "stage": "parent", "error": p.stderr[-400:]}
    parent = json.loads(p.stdout)
    alive = True
    try:
        os.kill(parent["pid"], 0)
    except (ProcessLookupError, PermissionError):
        alive = False
    cenv = {"PATH": "/usr/bin:/bin", "PYTHONPATH": here,
            "PYTHONDONTWRITEBYTECODE": "1"}
    c = subprocess.run([sys.executable, "-m", "x65a.restart_s", "child",
                        str(path), overlap, str(seed)], capture_output=True,
                       text=True, env=cenv)
    if c.returncode != 0:
        return {"ok": False, "stage": "child", "error": c.stderr[-400:]}
    child = json.loads(c.stdout)
    return {"ok": (not alive and parent["pid"] != child["pid"]
                   and parent["sha256"] == child["sha256"]
                   and parent["surviving"] == child["surviving"]
                   and not child["forbidden_in_bytes"]
                   and not child["forbidden_in_env"]
                   and not child["forbidden_in_globals"]),
            "parent_pid": parent["pid"], "child_pid": child["pid"],
            "parent_pid_gone": not alive,
            "audit_hash_identical": parent["sha256"] == child["sha256"],
            "sufficient_statistic_identical":
                parent["surviving"] == child["surviving"],
            "records": parent["records"], "bytes": parent["bytes"],
            "post_restart_return_transfer": child["return_transfer"],
            "post_restart_tasks": child["return_tasks"],
            "env_size": child["env_size"],
            "forbidden_channel_closed": not any(
                (child["forbidden_in_bytes"], child["forbidden_in_env"],
                 child["forbidden_in_globals"]))}


def main(argv) -> int:
    mode, path, overlap, seed = argv[1], Path(argv[2]), argv[3], int(argv[4])
    print(json.dumps(run_parent(path, overlap, seed) if mode == "parent"
                     else run_child(path, overlap, seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
