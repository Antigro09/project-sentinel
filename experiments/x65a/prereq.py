"""X65A-0: the fail-closed X64H prerequisite.

X65A may not produce a result artifact unless the FINAL frozen X64H run is
present, intact, and consistent. This module re-derives everything it can
rather than trusting a recorded value: the freeze digest is recomputed from
the frozen sources, the manifest's committed status is re-checked against
git, the artifact hashes are recomputed from bytes on disk, the released
seeds are re-derived from the digest, and the closure status is read from
the final result bundle for BOTH alphabet strata.

Pinned hashes are the second half of the check, not the first. They catch
the case where everything recomputes consistently but against the wrong
history.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "experiments")

from .types import TaintError

ROOT = Path("experiments/x64h")

PINNED = {
    "final_result_commit": "74701e335d69a73ab39307a98d1a41c916bce693",
    "freeze_manifest_commit": "db263b1834dccf3acd7a64af728ff3e84cb9977c",
    "freeze_digest":
        "e39153b7369a9fc8c9e14546181097ee81a0f27450b2804b2a23fbcfbef7178b",
    "manifest_sha256":
        "9d8c940e65a7fbdb5fbfc7f2790e3ea61c24c9c0894536ba4f92dcd4d43d9258",
    "seeds_sha256":
        "4589e1b95b3eab45ba8af5d712feaa11106f8992dc77a1768aeca99aaf52da16",
    "final_result_sha256":
        "8d71c381a24d473d2369155ea95c4672de8b95a635c8afe00cb6c6a11dff8f29",
    "strata": ("shared", "disjoint_op"),
    "closure_conditions": 6,
}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


@dataclass(frozen=True)
class Prerequisite:
    ok: bool
    checks: dict
    failures: tuple

    def require(self) -> None:
        if not self.ok:
            raise TaintError(
                "X65A prerequisite failed; no X65A artifact may be produced: "
                + "; ".join(self.failures))


def check() -> Prerequisite:
    c: dict = {}
    fail = []

    def need(name: str, cond: bool, detail=""):
        c[name] = {"pass": bool(cond), "detail": detail}
        if not cond:
            fail.append(f"{name} ({detail})" if detail else name)

    man = ROOT / "freeze_manifest_0c.json"
    seeds = ROOT / "final_seeds_0c.json"
    final = ROOT / "results" / "x64h_final.json"
    need("manifest_present", man.exists(), str(man))
    need("seeds_present", seeds.exists(), str(seeds))
    need("final_result_present", final.exists(), str(final))
    if fail:
        return Prerequisite(False, c, tuple(fail))

    # 1. the freeze digest, recomputed from the frozen sources
    from x64h import freeze0c as FZ
    from x64h_0c_audit import ARMS_0C, BASE
    live = FZ.freeze_digest(BASE, ARMS_0C)
    m = json.loads(man.read_text())
    need("freeze_digest_recomputes", live == m["digest"],
         f"live {live[:16]} vs manifest {m['digest'][:16]}")
    need("freeze_digest_pinned", m["digest"] == PINNED["freeze_digest"],
         m["digest"][:16])
    committed, why = FZ.manifest_committed(BASE, ARMS_0C)
    need("freeze_valid", committed, why)

    # 2. artifact hashes, recomputed from bytes
    need("manifest_sha", _sha(man) == PINNED["manifest_sha256"], _sha(man)[:16])
    need("seeds_sha", _sha(seeds) == PINNED["seeds_sha256"], _sha(seeds)[:16])
    need("final_result_sha", _sha(final) == PINNED["final_result_sha256"],
         _sha(final)[:16])

    # 3. commits
    head_final = _git("rev-parse", PINNED["final_result_commit"][:7])
    need("final_result_commit", head_final == PINNED["final_result_commit"],
         head_final[:12])
    mc = _git("log", "-1", "--format=%H", "--", str(man))
    need("freeze_manifest_commit", mc == PINNED["freeze_manifest_commit"],
         mc[:12])

    # 4. the seeds are the ones the digest implies, for both strata
    sd = json.loads(seeds.read_text())
    need("seed_digest_matches", sd["digest"] == m["digest"], sd["digest"][:16])
    derived_ok = True
    for fam, ss in sd["seeds"].items():
        for i, s in enumerate(ss):
            if s != int(hashlib.sha256(
                    f"{sd['digest']}:{fam}:{i}".encode()).hexdigest()[:8], 16):
                derived_ok = False
    need("seeds_derived_from_digest", derived_ok)
    need("seeds_per_stratum", all(len(v) >= 4 for v in sd["seeds"].values()),
         str({k: len(v) for k, v in sd["seeds"].items()}))

    # 5. both strata closed
    fr = json.loads(final.read_text())
    need("final_digest_matches", fr.get("digest") == m["digest"])
    need("manifest_intact_at_run", fr.get("manifest_intact") is True)
    need("both_strata_present",
         set(fr.get("families", {})) == set(PINNED["strata"]),
         str(sorted(fr.get("families", {}))))
    closure = fr.get("closure", {})
    need("closure_complete", len(closure) == PINNED["closure_conditions"],
         f"{len(closure)} conditions")
    need("all_gates_passed", bool(closure) and all(closure.values()),
         str({k: v for k, v in closure.items() if not v}) if closure else "absent")
    for st in PINNED["strata"]:
        f = fr.get("families", {}).get(st, {})
        iv = f.get("intervals", {}).get("persist/aware - static [whole]", {})
        need(f"effect_in_{st}", bool(iv) and iv.get("lo", 0) > 0,
             f"lo {iv.get('lo')}")

    return Prerequisite(not fail, c, tuple(fail))


def simulate_broken(field: str) -> Prerequisite:
    """Calibration arm. A prerequisite that cannot fail is not a check."""
    import copy
    saved = dict(PINNED)
    try:
        PINNED[field] = "0" * 64 if isinstance(saved[field], str) else 999
        return check()
    finally:
        PINNED.update(saved)
