"""Freeze manifests, taint ledgers, and the final-seed guard.

The protocol document asks for a machine-readable freeze schema and an explicit
rule that a final seed cannot be read before the freeze is committed. Both are
here, and the guard is a runtime object rather than a convention, because
"remember not to look at the final seeds" is not a control.

`FreezeManifest` deliberately carries `final_seed_file: None` at Scale 0. The
field exists so that the shape of the committed document does not change when
the final freeze happens -- the seed file's hash is *added*, and every prior
field keeps its value, which is what makes the two documents comparable.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from sentinel.wm.latent_contract import ContractViolation, Taint
from sentinel.wm.versioning import digest_file, digest_of


class FinalSeedAccessError(RuntimeError):
    """Something tried to read a final seed before the freeze permitted it."""


@dataclass(frozen=True, slots=True)
class FreezeManifest:
    """The minimum manifest from `BRANCH-AND-FREEZE-PROTOCOL.md`."""

    phase: str
    base_commit: str
    implementation_commit: str
    dirty_tracked: bool
    dependency_lock_sha256: str
    encoder_identities: tuple[str, ...]
    environment_generator_sha256: str
    split_procedure_sha256: str
    evaluator_sha256: str
    config_sha256: str
    gate_document_sha256: str
    final_seed_file: str | None = None
    created_before_final_seed: bool = True

    def __post_init__(self) -> None:
        if self.final_seed_file is not None and self.created_before_final_seed:
            raise ContractViolation(
                "a manifest that names a final seed file cannot also claim it was "
                "created before final seeds existed"
            )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "base_commit": self.base_commit,
            "implementation_commit": self.implementation_commit,
            "dirty_tracked": self.dirty_tracked,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "encoder_identities": list(self.encoder_identities),
            "environment_generator_sha256": self.environment_generator_sha256,
            "split_procedure_sha256": self.split_procedure_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "config_sha256": self.config_sha256,
            "gate_document_sha256": self.gate_document_sha256,
            "final_seed_file": self.final_seed_file,
            "created_before_final_seed": self.created_before_final_seed,
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())

    def write(self, path: Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.canonical_dict(), indent=2, sort_keys=True) + "\n")
        return self.digest

    @property
    def grants_final_seed_access(self) -> bool:
        """Only a committed manifest that names a seed file opens the gate."""
        return self.final_seed_file is not None and not self.created_before_final_seed


@dataclass
class FinalSeedGuard:
    """The only supported way to read a Phase-2 final seed file.

    Refuses unless a manifest that names the seed file has been committed and no
    longer claims to predate it. Development seeds 6600-6602 are not final seeds
    and are readable without a manifest -- the guard says so explicitly rather
    than blocking all seed access, because a guard that blocks everything gets
    routed around.
    """

    manifest: FreezeManifest | None = None
    development_seeds: frozenset[int] = frozenset({6600, 6601, 6602})

    def check_development_seed(self, seed: int) -> int:
        if seed not in self.development_seeds:
            raise FinalSeedAccessError(
                f"seed {seed} is not one of the frozen development seeds "
                f"{sorted(self.development_seeds)}"
            )
        return seed

    def load_final_seeds(self, path: Path) -> tuple[int, ...]:
        if self.manifest is None:
            raise FinalSeedAccessError(
                "no freeze manifest is loaded; final seeds cannot be read before "
                "the final freeze is committed"
            )
        if not self.manifest.grants_final_seed_access:
            raise FinalSeedAccessError(
                f"manifest {self.manifest.phase} does not grant final-seed access "
                f"(final_seed_file={self.manifest.final_seed_file!r}, "
                f"created_before_final_seed={self.manifest.created_before_final_seed})"
            )
        path = Path(path)
        actual = digest_file(path)
        if actual != self.manifest.final_seed_file:
            raise FinalSeedAccessError(
                f"final seed file digest {actual[:24]}... does not match the "
                f"manifest's {str(self.manifest.final_seed_file)[:24]}..."
            )
        return tuple(int(line) for line in path.read_text().split() if line.strip())


@dataclass
class TaintLedger:
    """Which taints each consumer is permitted to see, and what it actually saw."""

    permitted: Mapping[str, frozenset[Taint]]
    observed: dict[str, set[Taint]] = field(default_factory=dict)

    def record(self, consumer: str, taints: frozenset[Taint]) -> None:
        allowed = self.permitted.get(consumer)
        if allowed is None:
            raise ContractViolation(f"consumer {consumer!r} has no declared taint permission")
        violation = taints - allowed
        if violation:
            raise ContractViolation(
                f"{consumer} received {sorted(t.value for t in violation)}, "
                f"permitted {sorted(t.value for t in allowed)}"
            )
        self.observed.setdefault(consumer, set()).update(taints)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "permitted": {k: sorted(t.value for t in v) for k, v in sorted(self.permitted.items())},
            "observed": {k: sorted(t.value for t in v) for k, v in sorted(self.observed.items())},
        }


RUN_INPUT_PREFIXES: tuple[str, ...] = (
    "src/",
    "tests/",
    "experiments/",
    "docs/",
    "pyproject.toml",
    "uv.lock",
)
"""Paths whose state can change what a run does.

The matrix asks for a clean tracked tree, and the intent is that the reported
commit fully describes the code that ran. A dirty entry outside these prefixes
-- an unrelated worktree pointer, say -- cannot affect the run, and failing the
gate on it would report a Scale-0 failure for a bookkeeping reason. Such entries
are listed by name in the report instead of being folded into the verdict, so
the exemption is visible rather than assumed.
"""


def git_state(repo: Path) -> dict[str, Any]:
    """Commit, branch, and tracked-tree cleanliness at report time."""

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
            return ""

    # Deliberately not `run`, which strips the output: porcelain's first two
    # columns are status flags and the third is a space, so stripping the leading
    # blank of an unstaged entry (" M path") shifts every path two characters and
    # silently misclassifies it.
    try:
        porcelain = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain=v1"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):  # pragma: no cover
        porcelain = ""
    lines = [line for line in porcelain.split("\n") if line]
    tracked_dirty = [line for line in lines if not line.startswith("??")]
    untracked = [line[3:] for line in lines if line.startswith("??")]
    def path_of(entry: str) -> str:
        """Porcelain v1 is `XY<space>PATH`; a rename is `PATH -> NEWPATH`."""
        path = entry[3:].strip().strip('"')
        return path.split(" -> ")[-1] if " -> " in path else path

    run_input_dirty = [
        entry
        for entry in tracked_dirty
        if path_of(entry).startswith(RUN_INPUT_PREFIXES)
    ]
    other_dirty = [entry for entry in tracked_dirty if entry not in run_input_dirty]
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tracked": bool(tracked_dirty),
        "dirty_tracked_entries": tracked_dirty,
        "dirty_run_inputs": run_input_dirty,
        "dirty_outside_run_inputs": other_dirty,
        "clean_for_run_inputs": not run_input_dirty,
        "untracked_entries": untracked,
    }


def environment_state() -> dict[str, Any]:
    """Interpreter, platform, and accelerator facts recorded with every report."""
    state: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import mlx.core as mx

        state["mlx"] = getattr(mx, "__version__", "unknown")
        state["metal_available"] = bool(mx.metal.is_available())
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        state["metal_device"] = {
            k: (int(v) if isinstance(v, (int, float)) else str(v)) for k, v in info.items()
        }
    except Exception as exc:  # pragma: no cover - environment probe
        state["mlx"] = f"unavailable: {exc}"
    return state
