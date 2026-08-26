"""Freeze manifest, seed release, and the taint audit.

The rule this file exists to enforce: a final convention seed may not be
sampled or inspected until a freeze manifest carrying the full digest has
been committed. `release_final_seeds` refuses otherwise, and mutating any
frozen field changes the digest and invalidates seeds already released.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import convention as C
from . import decision as DE
from . import posterior as PO
from . import semantic as S
from .types import Taint, TaintError

def _arms():
    from . import arms
    return arms


def _metrics():
    from . import metrics
    return metrics


MANIFEST = Path("experiments/x64h/freeze_manifest.json")
TARGET_ONLY_MANIFEST = Path("experiments/x64h/.target_only_manifest.json")


def resolved_config(cfg: PO.Config, costs: DE.Costs, gates: DE.Gates,
                    budget: int, query_universe_size: int) -> dict:
    """Every non-secret value that can affect generation, inference,
    querying, decisions or evaluation. Serialised into every artifact, so a
    run can be reproduced from its own output."""
    return {
        "semantic": {
            "filters": list(S.X64H_FILTERS), "scopes": list(S.X64H_SCOPES),
            "polarities": list(S.POLARITIES),
            "universe": list(S.UNIVERSE), "held_out": list(S.HELD_OUT),
            "n_forms": len(S.x64h_forms()),
        },
        "convention_meta": {
            "words": list(C.WORDS), "function_words": list(C.FUNCTION_WORDS),
            "phrase_pool": [list(p) for p in C.PHRASE_POOL],
            "filter_atoms": list(C.FILTER_ATOMS),
            "scope_atoms": list(C.SCOPE_ATOMS),
            "op_atoms": list(C.OP_ATOMS),
        },
        "other_model": asdict(cfg.other),
        "inference": {"prior_conflict": cfg.prior_conflict, "rho": cfg.rho,
                      "exact": cfg.exact},
        "costs": asdict(costs),
        "gates": asdict(gates),
        "query": {"budget": budget, "universe_size": query_universe_size},
        "arms": list(_arms().ARMS),
        "metrics_schema": list(_metrics().SCHEMA),
        "bootstrap": {"resamples": 2000, "unit": "task_meaning", "seed": 13},
    }


def freeze_digest(cfg=None, costs=None, gates=None, budget=6,
                  universe_size=16) -> str:
    payload = resolved_config(cfg or PO.Config(), costs or DE.Costs(),
                              gates or DE.Gates(), budget, universe_size)
    payload["commit"] = _git_commit()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()
                          ).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class FreezeManifest:
    digest: str
    commit: str
    created: str
    python: str
    note: str = ""

    def write(self, path: Path = MANIFEST) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))


def write_manifest(note: str = "", path: Path = MANIFEST) -> FreezeManifest:
    m = FreezeManifest(freeze_digest(), _git_commit(),
                       time.strftime("%Y-%m-%dT%H:%M:%S"),
                       sys.version.split()[0], note)
    m.write(path)
    return m


def manifest_committed(path: Path = MANIFEST) -> tuple[bool, str]:
    """A manifest is committed when it exists, matches the current digest,
    and is tracked by git with no uncommitted change."""
    if not path.exists():
        return False, "no freeze manifest"
    d = json.loads(path.read_text())
    if d["digest"] != freeze_digest():
        return False, ("the frozen configuration changed after the manifest "
                       "was written; the digest no longer matches")
    try:
        r = subprocess.run(["git", "status", "--porcelain", str(path)],
                           capture_output=True, text=True, check=True)
        if r.stdout.strip():
            return False, "the freeze manifest is not committed"
    except Exception:
        return False, "git unavailable; cannot verify the manifest is committed"
    return True, "committed"


def release_final_seeds(n: int = 4, path: Path = MANIFEST):
    """Refuses unless a matching manifest is committed. This is the only
    function permitted to produce final convention seeds, and it writes a
    TARGET_ONLY record of what it produced."""
    ok, why = manifest_committed(path)
    if not ok:
        raise TaintError(
            f"final seeds may not be released: {why}. Sampling or "
            f"inspecting a final convention before the freeze is committed "
            f"invalidates the experiment.")
    digest = freeze_digest()
    seeds = [int(hashlib.sha256(f"{digest}:final:{i}".encode()
                                ).hexdigest()[:8], 16) for i in range(n)]
    TARGET_ONLY_MANIFEST.write_text(json.dumps(
        {"digest": digest, "seeds": seeds,
         "convention_hashes": [C.sample_convention(s).digest() for s in seeds],
         "taint": Taint.TARGET_ONLY.value}, indent=2, sort_keys=True))
    return seeds


def taint_audit(obj, label: str = "root") -> list[str]:
    """Walk an agent-facing object and report anything that must not be
    there. Used at every serialisation boundary and after restart."""
    bad = []
    forbidden = ("phi", "convention", "convention_id", "z_true", "target",
                 "gold", "future", "answer_key")

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                if str(k).lower() in forbidden:
                    bad.append(f"{path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif hasattr(o, "__dataclass_fields__"):
            for k in o.__dataclass_fields__:
                if str(k).lower() in forbidden:
                    bad.append(f"{path}.{k}")
                walk(getattr(o, k), f"{path}.{k}")

    walk(obj, label)
    return bad


def environment() -> dict:
    """The reproducibility stack the specification names is not installed:
    hydra, omegaconf, mlflow, dvc, jax and numpyro are all absent, and DVC
    initialisation additionally needs the repository owner's approval. The
    PROPERTIES those tools provide are implemented here with the standard
    library -- a fully resolved config serialised into every artifact, a
    structured per-run record, and JSON as the authoritative output with any
    prose generated from it."""
    present = {}
    for m in ("hydra", "omegaconf", "mlflow", "dvc", "jax", "numpyro"):
        try:
            __import__(m)
            present[m] = True
        except Exception:
            present[m] = False
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "optional_tools": present}
