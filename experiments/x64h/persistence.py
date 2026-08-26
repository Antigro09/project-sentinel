"""Convention posterior across tasks and restarts, with taint enforced at
the serialisation boundary.

Only PUBLIC and OBSERVED fields may be written. The writer checks, rather
than trusting the caller: a leak here would make every persistence result
meaningless, and the check is cheap.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .types import PERSISTABLE, PosteriorState, Taint, TaintError

FORBIDDEN_KEYS = ("phi", "convention", "convention_id", "z_true", "target",
                  "gold", "future", "answer_key", "seed_secret")


def _check(payload: dict) -> None:
    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if str(k).lower() in FORBIDDEN_KEYS:
                    raise TaintError(f"forbidden key {k!r} at {path}")
                walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
    walk(payload)


def save(path: Path, state: PosteriorState, taints=None) -> None:
    taints = taints or {}
    for k, t in taints.items():
        if t not in PERSISTABLE:
            raise TaintError(f"field {k!r} has taint {t.value}; only "
                             f"PUBLIC and OBSERVED may be persisted")
    lp = list(state.log_p_phi)
    if lp:
        m = max(lp)
        tot = m + math.log(sum(math.exp(x - m) for x in lp))
        lp = [x - tot for x in lp]
    payload = {
        "log_p_phi": lp,
        "model_hash": state.model_hash,
        "observation_hashes": list(state.observation_hashes),
    }
    _check(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))


def load(path: Path, n_phi: int, model_hash: str) -> PosteriorState:
    if not path.exists():
        return PosteriorState(tuple([-math.log(n_phi)] * n_phi), model_hash)
    d = json.loads(path.read_text())
    lp = list(d["log_p_phi"])
    m = max(lp)
    tot = m + math.log(sum(math.exp(x - m) for x in lp)) if lp else 0.0
    if abs(tot) > 1e-6:
        raise TaintError("persisted convention prior is not normalised; a "
                         "prior must be a probability distribution")
    if d["model_hash"] != model_hash:
        raise TaintError("persisted state was written under a different "
                         "model hash; the freeze has changed")
    return PosteriorState(tuple(d["log_p_phi"]), d["model_hash"],
                          tuple(d["observation_hashes"]))


def observation_hash(evidence) -> str:
    return hashlib.sha256(repr(evidence.key()).encode()).hexdigest()[:16]
