"""Container isolation for LLM-written code.

Generated code is executed to be verified — that is the whole mechanism —
and by default it runs in-process with full privileges. This module moves
that execution into a container with no network, a memory cap, and a
read-only mount of the repository.

**Auto-detecting by design.** `mode="auto"` uses a container runtime when
one is present and falls back to in-process when none is, so the same code
path works before and after Docker is installed. Nothing changes on a
machine without it, and nothing needs changing on a machine that gains it.
`mode="docker"` refuses to fall back, for when isolation is a requirement
rather than a preference.

The container needs only the pure-Python parts of the package, so the
stock `python:3.12-slim` image is enough — no build step, no Dockerfile,
no image to maintain. That is deliberate: a sandbox nobody has to
provision is a sandbox that actually gets used.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sentinel.env.history import History
from sentinel.verify.report import VerificationReport

from .loader import LoadError, load_model

Mode = Literal["auto", "docker", "inprocess"]

RUNTIMES = ("docker", "finch", "podman", "nerdctl")
IMAGE = "python:3.12-slim"


def detect_runtime() -> str | None:
    """First working container runtime on PATH, or None.

    Presence on PATH is not enough — Docker Desktop installs the CLI but
    the daemon may be stopped, and a stopped daemon fails per-invocation
    rather than at import. Each candidate is probed.
    """
    for name in RUNTIMES:
        if shutil.which(name) is None:
            continue
        try:
            probe = subprocess.run(
                [name, "info"], capture_output=True, timeout=20, check=False
            )
        except Exception:  # noqa: BLE001
            continue
        if probe.returncode == 0:
            return name
    return None


def repo_root() -> Path:
    """Directory containing `src/`, mounted read-only into the container."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class SandboxResult:
    ok: bool
    report: VerificationReport | None
    error: str | None
    kind: str | None
    """load | verify | worker | runtime — which stage failed."""
    isolated: bool
    """False when this ran in-process because no runtime was available."""


@dataclass
class Sandbox:
    """Verifies generated models, in a container where possible."""

    mode: Mode = "auto"
    runtime: str | None = None
    image: str = IMAGE
    memory: str = "512m"
    cpus: str = "1"
    wall_timeout: float = 120.0
    """Hard ceiling on one container, including startup.

    Deliberately larger than it first appears necessary. Verification makes
    three model calls per recorded step, so a 52-step history is ~156 calls;
    at the 2s per-call guard a slow-but-finite model could legitimately need
    minutes. A fixed 90s would have killed those containers and recorded
    them as failures, which reads as "the model is broken" rather than "the
    model is slow". `timeout_for` scales this with history length.
    """
    model_timeout: float = 2.0
    """Per-call guard inside the container, enforced by SIGALRM."""

    def __post_init__(self) -> None:
        if self.mode == "inprocess":
            self.runtime = None
        elif self.runtime is None:
            self.runtime = detect_runtime()
        if self.mode == "docker" and self.runtime is None:
            raise RuntimeError(
                "mode='docker' but no working container runtime was found. "
                "Install Docker Desktop and ensure the daemon is running, or "
                "use mode='auto' to fall back to in-process execution."
            )

    @property
    def isolated(self) -> bool:
        return self.runtime is not None

    def describe(self) -> str:
        if not self.isolated:
            return "in-process (NOT isolated — generated code runs with full privileges)"
        return f"{self.runtime} + {self.image} (network=none, mem={self.memory})"

    def timeout_for(self, history: History) -> float:
        """Wall budget scaled to how much work this verification actually is.

        A long history means more model calls, so a flat timeout punishes
        long episodes for being long. Capped so one pathological model
        cannot eat an hour of an overnight run.
        """
        return min(600.0, max(self.wall_timeout, 20.0 + 1.5 * len(history.steps)))

    def image_present(self) -> bool:
        if not self.runtime:
            return False
        probe = subprocess.run(
            [self.runtime, "image", "inspect", self.image],
            capture_output=True,
            check=False,
        )
        return probe.returncode == 0

    def pull_image(self) -> tuple[bool, str]:
        """Fetch the base image. Called once; every later run is offline."""
        if not self.runtime:
            return False, "no container runtime"
        result = subprocess.run(
            [self.runtime, "pull", self.image],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0, (result.stderr or result.stdout).strip()[-400:]

    def verify(
        self, source: str, history: History, name: str = "generated"
    ) -> SandboxResult:
        if not self.isolated:
            return self._verify_inprocess(source, history, name)
        return self._verify_container(source, history, name)

    def _verify_inprocess(
        self, source: str, history: History, name: str
    ) -> SandboxResult:
        from sentinel.verify import Verifier

        try:
            model = load_model(
                source,
                timeout=self.model_timeout,
                name=name,
                context={"INITIAL_GRID": history.initial.grid},
            )
        except LoadError as exc:
            return SandboxResult(False, None, str(exc), "load", isolated=False)

        report = Verifier(stop_on_crash=True).verify(model, history)
        return SandboxResult(True, report, None, None, isolated=False)

    def _verify_container(
        self, source: str, history: History, name: str
    ) -> SandboxResult:
        assert self.runtime is not None
        job = json.dumps(
            {
                "source": source,
                "history": history.to_json(),
                "timeout": self.model_timeout,
                "name": name,
            }
        )
        root = repo_root()
        command = [
            self.runtime,
            "run",
            "--rm",
            "--interactive",
            "--network=none",
            f"--memory={self.memory}",
            f"--cpus={self.cpus}",
            "--pids-limit=128",
            "--read-only",
            "--tmpfs=/tmp:size=32m",
            "--workdir=/app",
            "-v",
            f"{root}:/app:ro",
            "-e",
            "PYTHONPATH=/app/src",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            self.image,
            "python",
            "/app/src/sentinel/bootstrap/_sandbox_worker.py",
        ]

        budget = self.timeout_for(history)
        try:
            completed = subprocess.run(
                command,
                input=job,
                capture_output=True,
                text=True,
                timeout=budget,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # --rm cleans the container up. A model that cannot answer within
            # the wall budget is not a usable model, so this is a result
            # rather than an error.
            return SandboxResult(
                False, None, f"container exceeded {budget:g}s", "verify", True
            )
        except Exception as exc:  # noqa: BLE001
            return SandboxResult(False, None, f"{type(exc).__name__}: {exc}", "runtime", True)

        if completed.returncode != 0 and not completed.stdout.strip():
            detail = (completed.stderr or "").strip()[-400:] or "no output"
            return SandboxResult(False, None, detail, "runtime", True)

        try:
            payload: dict[str, Any] = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            detail = (completed.stdout or completed.stderr).strip()[-400:]
            return SandboxResult(False, None, f"unparseable worker output: {detail}", "worker", True)

        if not payload.get("ok"):
            return SandboxResult(
                False, None, payload.get("error"), payload.get("kind", "worker"), True
            )

        return SandboxResult(
            True,
            VerificationReport.from_json_full(payload["report"]),
            None,
            None,
            isolated=True,
        )
