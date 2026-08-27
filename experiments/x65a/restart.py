"""X65A-0: a genuine process restart, not a round trip.

Serializing and reloading inside one interpreter proves nothing about
hidden state: caches, module globals, RNG state and environment variables
all survive it. So the parent is a SEPARATE PROCESS that writes state and
dies, and the child is another process with a scrubbed environment that
receives its next input externally.

A forbidden value is planted in the parent's environment and in a
runtime-only cache. The child must not see it anywhere -- not in the
serialized bytes, not in its environment, not in its own module globals.

Dispatch: `python -m x65a.restart parent <path> <obs>` / `... child <path> <obs>`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FORBIDDEN_ENV = "X65A_FORBIDDEN_TARGET"
FORBIDDEN_VALUE = "future_target_answer"

# A runtime-only cache. Exactly the kind of channel that must not survive.
_RUNTIME_CACHE: dict = {}


def _state_for(observations):
    from .microcases import convention_posterior
    from .posterior import ExactPosterior
    from .store import ActiveMemory, AuditArchive, PersistentState, SCHEMA_VERSION
    post = convention_posterior(observations)
    am, ar = ActiveMemory(), AuditArchive()
    for i, o in enumerate(observations):
        ar.append({"evidence_id": f"obs{i}", "observation": int(o),
                   "acquired_at": i})
    return PersistentState(SCHEMA_VERSION, "x64h:e39153b7", len(observations),
                           am, ar, post)


def _report(path: Path, observations):
    from .store import load
    st = _state_for(observations)
    d = load(path) if path.exists() else None
    return st, d


def run_parent(path: Path, observations) -> dict:
    from .store import save
    _RUNTIME_CACHE["answer"] = os.environ.get(FORBIDDEN_ENV, FORBIDDEN_VALUE)
    st = _state_for(observations)
    sha = save(path, st)
    return {"pid": os.getpid(), "sha256": sha,
            "posterior": {str(k): str(v) for k, v in st.posterior.q.items()},
            "archive_chain": st.archive.chain,
            "bytes": st.report_bytes(),
            "cache_had_forbidden_value": FORBIDDEN_VALUE
                                         in json.dumps(_RUNTIME_CACHE)}


def run_child(path: Path, next_obs) -> dict:
    from .store import load
    d = load(path)
    blob = path.read_bytes().decode("utf-8")
    env_hit = any(FORBIDDEN_VALUE in v for v in os.environ.values())
    glob_hit = FORBIDDEN_VALUE in json.dumps(_RUNTIME_CACHE)
    q = d["posterior"]["q"]
    return {"pid": os.getpid(),
            "sha256": d["_sha256"],
            "posterior": {str(k): str(v) for k, v in q.items()},
            "archive_chain": d["archive_chain"],
            "forbidden_in_bytes": FORBIDDEN_VALUE in blob,
            "forbidden_in_env": env_hit,
            "forbidden_in_globals": glob_hit,
            "env_size": len(os.environ),
            "next_observation_received": int(next_obs),
            "random_state_inherited": os.environ.get("PYTHONHASHSEED") == "0"}


def restart_cycle(path: Path, observations, next_obs: int = 1) -> dict:
    """Parent writes and dies; child reads with a scrubbed environment."""
    here = str(Path("experiments").resolve())
    penv = dict(os.environ)
    penv[FORBIDDEN_ENV] = FORBIDDEN_VALUE
    penv["PYTHONPATH"] = here
    obs = ",".join(str(o) for o in observations)
    p = subprocess.run([sys.executable, "-m", "x65a.restart", "parent",
                        str(path), obs], capture_output=True, text=True,
                       env=penv, cwd=str(Path.cwd()))
    if p.returncode != 0:
        return {"ok": False, "stage": "parent", "error": p.stderr[-500:]}
    parent = json.loads(p.stdout)

    alive = True
    try:
        os.kill(parent["pid"], 0)
    except (ProcessLookupError, PermissionError):
        alive = False

    cenv = {"PATH": "/usr/bin:/bin", "PYTHONPATH": here,
            "PYTHONDONTWRITEBYTECODE": "1"}
    c = subprocess.run([sys.executable, "-m", "x65a.restart", "child",
                        str(path), str(next_obs)], capture_output=True,
                       text=True, env=cenv, cwd=str(Path.cwd()))
    if c.returncode != 0:
        return {"ok": False, "stage": "child", "error": c.stderr[-500:]}
    child = json.loads(c.stdout)

    ok = (not alive
          and parent["pid"] != child["pid"]
          and parent["sha256"] == child["sha256"]
          and parent["posterior"] == child["posterior"]
          and parent["archive_chain"] == child["archive_chain"]
          and not child["forbidden_in_bytes"]
          and not child["forbidden_in_env"]
          and not child["forbidden_in_globals"])
    return {"ok": ok, "parent": parent, "child": child,
            "parent_pid_gone": not alive,
            "distinct_process": parent["pid"] != child["pid"],
            "posterior_exactly_preserved":
                parent["posterior"] == child["posterior"],
            "hash_preserved": parent["sha256"] == child["sha256"],
            "forbidden_channel_closed": not any(
                (child["forbidden_in_bytes"], child["forbidden_in_env"],
                 child["forbidden_in_globals"]))}


def contaminated_cycle(path: Path, observations) -> dict:
    """Calibration arm: the SAME child, run with the forbidden value left in
    its environment, must be caught. A canary that never fires is decoration."""
    here = str(Path("experiments").resolve())
    from .store import save
    save(path, _state_for(observations))
    cenv = {"PATH": "/usr/bin:/bin", "PYTHONPATH": here,
            FORBIDDEN_ENV: FORBIDDEN_VALUE}
    c = subprocess.run([sys.executable, "-m", "x65a.restart", "child",
                        str(path), "1"], capture_output=True, text=True,
                       env=cenv, cwd=str(Path.cwd()))
    child = json.loads(c.stdout)
    return {"detected": child["forbidden_in_env"], "child": child}


def main(argv) -> int:
    mode, path, arg = argv[1], Path(argv[2]), argv[3]
    if mode == "parent":
        obs = [int(x) for x in arg.split(",") if x != ""]
        print(json.dumps(run_parent(path, obs)))
    elif mode == "child":
        print(json.dumps(run_child(path, int(arg))))
    else:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
