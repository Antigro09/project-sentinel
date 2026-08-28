"""X65A-S2: both tiers must survive a genuine restart.

The provisional branch is the part that is easy to lose: it is transient by
design, so a system that persists only ConfirmedState would look correct on
every steady-state check and silently drop every open question at a restart.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .provisional import ConfirmedState, ProvisionalBranch
from .semantic_mem import GroundedObservation
from .types import encode

FORBIDDEN_ENV = "X65A_FORBIDDEN_TARGET"
FORBIDDEN_VALUE = "future_target_answer"
_RUNTIME_CACHE: dict = {}


def _state(overlap: str, seed: int):
    sys.path.insert(0, "experiments")
    import random
    from x64h import family as F
    from x65a import provisional as P
    from x65a import s2_suite as S2
    fam = F.Family(F.FamilySpec(overlap=overlap))
    cases = S2.build_suite(fam, seed, 3, 3)
    rng = random.Random(seed)
    out = []
    for c in cases:
        outcome, conf, branch, used = P.resolve(
            fam, c.confirmed, c.event, c.phi_true, "main",
            list(range(fam.m)), rng)
        out.append((c, outcome, conf, branch, used))
    return fam, out


def payload(rows) -> dict:
    return {"schema": 1,
            "confirmed": [conf.canon() for _c, _o, conf, _b, _u in rows],
            "provisional": [b.canon() for _c, _o, _cf, b, _u in rows
                            if b is not None],
            "outcomes": [o for _c, o, _cf, _b, _u in rows]}


def run_parent(path: Path, overlap: str, seed: int) -> dict:
    _RUNTIME_CACHE["answer"] = os.environ.get(FORBIDDEN_ENV, FORBIDDEN_VALUE)
    _fam, rows = _state(overlap, seed)
    blob = encode(payload(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return {"pid": os.getpid(), "sha256": hashlib.sha256(blob).hexdigest(),
            "confirmed": len(payload(rows)["confirmed"]),
            "provisional": len(payload(rows)["provisional"]),
            "bytes": len(blob)}


def run_child(path: Path, overlap: str, seed: int) -> dict:
    blob = path.read_bytes()
    d = json.loads(blob.decode())
    _fam, rows = _state(overlap, seed)
    # compare through the canonical encoder: the parent's bytes carry
    # Fractions as {"__frac__": [n, d]} and a live dict carries Fraction
    # objects, so a direct comparison would always differ
    live = json.loads(encode(payload(rows)).decode())
    return {"pid": os.getpid(), "sha256": hashlib.sha256(blob).hexdigest(),
            "confirmed_identical": live["confirmed"] == d["confirmed"],
            "provisional_identical": live["provisional"] == d["provisional"],
            "outcomes_identical": live["outcomes"] == d["outcomes"],
            "provisional_branches": len(d["provisional"]),
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
    p = subprocess.run([sys.executable, "-m", "x65a.restart_s2", "parent",
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
    c = subprocess.run([sys.executable, "-m", "x65a.restart_s2", "child",
                        str(path), overlap, str(seed)], capture_output=True,
                       text=True, env=cenv)
    if c.returncode != 0:
        return {"ok": False, "stage": "child", "error": c.stderr[-400:]}
    ch = json.loads(c.stdout)
    return {"ok": (not alive and parent["pid"] != ch["pid"]
                   and parent["sha256"] == ch["sha256"]
                   and ch["confirmed_identical"]
                   and ch["provisional_identical"]
                   and ch["outcomes_identical"]
                   and not ch["forbidden_in_bytes"]
                   and not ch["forbidden_in_env"]
                   and not ch["forbidden_in_globals"]),
            "parent_pid": parent["pid"], "child_pid": ch["pid"],
            "parent_pid_gone": not alive,
            "hash_identical": parent["sha256"] == ch["sha256"],
            "confirmed_identical": ch["confirmed_identical"],
            "provisional_identical": ch["provisional_identical"],
            "outcomes_identical": ch["outcomes_identical"],
            "provisional_branches": ch["provisional_branches"],
            "bytes": parent["bytes"], "env_size": ch["env_size"],
            "forbidden_channel_closed": not any(
                (ch["forbidden_in_bytes"], ch["forbidden_in_env"],
                 ch["forbidden_in_globals"]))}


def main(argv) -> int:
    mode, path, overlap, seed = argv[1], Path(argv[2]), argv[3], int(argv[4])
    print(json.dumps(run_parent(path, overlap, seed) if mode == "parent"
                     else run_child(path, overlap, seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
