"""X64H freeze: hash everything that can move the result, then refuse to
produce a final seed until that hash is committed.

The existing `protocol.freeze_digest` folds `git rev-parse HEAD` into the
digest. That makes a committed manifest self-invalidating: committing the
manifest moves HEAD, so the recomputed digest no longer matches what the
manifest records and `manifest_committed` can never return True. The digest
here is a function of the FROZEN ARTIFACTS ONLY. The commit is recorded
beside it, not inside it, so mutating any frozen file breaks the seal and
committing the manifest does not.

Final seeds are derived from the digest itself:

    seed_i = sha256(digest : family : i)

so they are a deterministic function of the frozen artifacts. They cannot be
chosen to flatter a result without inverting SHA-256, and the function that
computes them refuses to run before the manifest is committed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import audit0c as A
from . import episode as EP
from . import family as F
from . import decision as DE
from . import posterior as PO
from . import protocol as PR
from . import semantic as S
from .types import Taint, TaintError

ROOT = Path("experiments")
MANIFEST = ROOT / "x64h" / "freeze_manifest_0c.json"
SEEDS = ROOT / "x64h" / "final_seeds_0c.json"

SOURCES = {
    "alphabet_families": ["x64h/family.py"],
    "calibration_transfer_schedule": ["x64h/episode.py"],
    "teacher_selection_likelihood": ["x64h/audit0c.py"],
    "exact_inference": ["x64h/episode.py", "x64h/audit0c.py",
                        "x64h/posterior.py", "x64h/grammar.py"],
    "query_policy": ["x64h/queries.py", "x64h/episode.py"],
    "conflict_model": ["x64h/episode.py"],
    "open_world_other": ["x64h/posterior.py", "x64h/types.py"],
    "evaluator": ["x64h/semantic.py", "x64e_semantics.py", "x64d_senses.py",
                  "x64a_identify.py"],
    "gates": ["x64h_0c_audit.py", "x64h_0b_validity.py", "x64h_final.py"],
    "bootstrap": ["x64h/audit0c.py"],
    "persistence": ["x64h/persistence.py", "x64h/protocol.py"],
}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file(p: str) -> str:
    return _sha((ROOT / p).read_bytes())


def structural(cfg: EP.Config, arms) -> dict:
    """The values that decide the experiment but do not live in one file."""
    fams = {ov: F.Family(F.FamilySpec(overlap=ov))
            for ov in ("shared", "disjoint_op")}
    return {
        "families": {ov: {**f.accounting(),
                          "signature_sha": _sha(f.signatures().tobytes()),
                          "minimal_separating_k":
                              f.minimal_separating_size()["k"]}
                     for ov, f in fams.items()},
        "config": asdict(cfg),
        "prior": "uniform over the convention family; "
                 "p(z | D) uniform over demo-consistent behaviours",
        "arms": [list(a) for a in arms],
        "likelihoods": ["naive", "aware", "selection_only"],
        "semantic": {"ops": list(F.OPS), "filters": list(F.FILTERS_0B),
                     "scopes": list(F.SCOPES_0B),
                     "universe": list(S.UNIVERSE)},
        "other_model": asdict(PO.Config().other),
        "bootstrap": {"resamples": 10000, "unit": "episode", "seed": 20260827},
        "thresholds": {"oracle": 0.98, "static": 0.95, "R": 0.50,
                       "late": 0.95, "margin": 0.02, "theta_commit": 0.99},
        # the implementation slice's resolved config WITHOUT its commit
        # field. `protocol.freeze_digest` appends `git rev-parse HEAD`, and
        # embedding that here reintroduced the very self-invalidation this
        # module exists to avoid: committing the manifest moved HEAD, the
        # structural digest moved with it, and the seal broke on a tree
        # whose files were byte-identical.
        "implementation_slice_config": _sha(json.dumps(
            PR.resolved_config(PO.Config(), DE.Costs(), DE.Gates(), 6, 16),
            sort_keys=True, default=str).encode()),
    }


def component_digests(cfg: EP.Config, arms) -> dict:
    out = {k: _sha("".join(sorted(_file(p) for p in v)).encode())
           for k, v in SOURCES.items()}
    out["structural"] = _sha(
        json.dumps(structural(cfg, arms), sort_keys=True, default=str).encode())
    return out


def freeze_digest(cfg: EP.Config, arms) -> str:
    d = component_digests(cfg, arms)
    return _sha(json.dumps(d, sort_keys=True).encode())


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def write_manifest(cfg: EP.Config, arms, note: str = "") -> dict:
    m = {
        "digest": freeze_digest(cfg, arms),
        "components": component_digests(cfg, arms),
        "structural": structural(cfg, arms),
        "commit_at_write": _commit(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.version.split()[0],
        "note": note,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True, default=str))
    return m


def manifest_committed(cfg: EP.Config, arms) -> tuple[bool, str]:
    if not MANIFEST.exists():
        return False, "no freeze manifest"
    d = json.loads(MANIFEST.read_text())
    now = component_digests(cfg, arms)
    moved = [k for k in now if d["components"].get(k) != now[k]]
    if moved:
        return False, ("frozen components changed after the manifest was "
                       f"written: {', '.join(sorted(moved))}")
    if d["digest"] != freeze_digest(cfg, arms):
        return False, "the freeze digest no longer matches"
    try:
        r = subprocess.run(["git", "status", "--porcelain", str(MANIFEST)],
                           capture_output=True, text=True, check=True)
        if r.stdout.strip():
            return False, "the freeze manifest is not committed"
    except Exception:
        return False, "git unavailable; cannot verify the manifest is committed"
    return True, "committed"


def release_final_seeds(cfg: EP.Config, arms, n: int = 6,
                        families=("shared", "disjoint_op")) -> dict:
    """The ONLY function permitted to produce final convention seeds. It
    refuses unless the freeze is committed and every frozen component still
    hashes to what the manifest recorded."""
    ok, why = manifest_committed(cfg, arms)
    if not ok:
        raise TaintError(
            f"final seeds may not be released: {why}. Sampling or inspecting "
            f"a final convention before the freeze is committed invalidates "
            f"the experiment.")
    digest = freeze_digest(cfg, arms)
    out = {fam: [int(_sha(f"{digest}:{fam}:{i}".encode())[:8], 16)
                 for i in range(n)] for fam in families}
    SEEDS.write_text(json.dumps(
        {"digest": digest, "released_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "seeds": out, "n_per_family": n, "taint": Taint.TARGET_ONLY.value,
         "note": "derived from the freeze digest; not chosen"},
        indent=2, sort_keys=True))
    return out
