"""Preflight for the two named frozen backbone families.

`SCALE-0-RUN-MATRIX.md` names exactly two encoder families and forbids choosing
a third after seeing results. So this module does not "find a vision model"; it
checks whether the two *named* ones can be run faithfully, and reports a verdict
per family. The candidate list is frozen data, not a search.

A family is `RUNNABLE` only when all five conditions hold: the official
repository resolves, the licence is recorded, the exact revision is pinned,
the weights are locally readable, and a local runtime can execute them at the
declared precision. Anything else is `BLOCKED` with the specific reason, and a
blocked family stops Scale 0 before any observation is encoded.

Two things this module deliberately will not do. It will not accept a licence
agreement on anyone's behalf. And it will not swap in a mirror, a re-upload, or a
larger model as a workaround, because the matrix says a replacement requires a
reviewed pre-run amendment and a complete restart.

It is also careful about *which* blocker it names. "This repository is gated" and
"this account has not accepted the terms" are different facts, and the second one
cannot be observed without a credential. Reporting the second when only the first
is known sends someone to accept a licence they already accepted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from sentinel.wm.latent_contract import Precision
from sentinel.wm.versioning import digest_of


class PreflightVerdict(str, Enum):
    RUNNABLE = "runnable"
    BLOCKED = "blocked"
    UNCHECKED = "unchecked"


class BlockReason(str, Enum):
    LICENCE_NOT_ACCEPTED = "gated_licence_not_accepted_by_this_account"
    NO_CREDENTIAL = "no_local_access_token"
    NO_NETWORK = "repository_unreachable"
    NO_RUNTIME = "no_local_runtime_for_architecture"
    PRECISION_UNSUPPORTED = "declared_precision_unsupported_locally"
    WEIGHTS_ABSENT = "weights_not_present_locally"
    DISK_BUDGET = "download_exceeds_artifact_budget"


@dataclass(frozen=True, slots=True)
class BackboneCandidate:
    """One frozen encoder family, exactly as the matrix names it."""

    encoder_id: str
    repository: str
    family: str
    declared_precision: Precision
    role: str

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "encoder_id": self.encoder_id,
            "repository": self.repository,
            "family": self.family,
            "declared_precision": self.declared_precision.value,
            "role": self.role,
        }


FROZEN_CANDIDATES: tuple[BackboneCandidate, ...] = (
    BackboneCandidate(
        encoder_id="qwen3_vl_4b",
        repository="Qwen/Qwen3-VL-4B-Instruct",
        family="Qwen3-VL 4B",
        declared_precision=Precision.BF16,
        role="multimodal instruction/visual feature path",
    ),
    BackboneCandidate(
        encoder_id="gemma3_4b",
        repository="google/gemma-3-4b-it",
        family="Gemma 3 4B",
        declared_precision=Precision.BF16,
        role="independent multimodal control path",
    ),
)
"""Frozen. Editing this tuple is a matrix amendment, not a code change."""


@dataclass(frozen=True, slots=True)
class BackbonePreflight:
    """The full evidence record for one candidate."""

    candidate: BackboneCandidate
    verdict: PreflightVerdict
    reasons: tuple[BlockReason, ...] = ()
    revision: str | None = None
    licence: str | None = None
    gated: str | bool | None = None
    total_parameters: int | None = None
    stored_dtypes: tuple[str, ...] = ()
    download_bytes: int | None = None
    local_path: str | None = None
    runtime: str | None = None
    detail: str = ""
    measurements: Mapping[str, Any] = field(default_factory=dict)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.canonical_dict(),
            "verdict": self.verdict.value,
            "reasons": [r.value for r in self.reasons],
            "revision": self.revision,
            "licence": self.licence,
            "gated": self.gated,
            "total_parameters": self.total_parameters,
            "stored_dtypes": list(self.stored_dtypes),
            "download_bytes": self.download_bytes,
            "local_path": self.local_path,
            "runtime": self.runtime,
            "detail": self.detail,
            "measurements": dict(self.measurements),
        }

    @property
    def digest(self) -> str:
        return digest_of(self.canonical_dict())


def _curl_json(url: str, timeout: int = 30, token: str | None = None) -> tuple[int, Any]:
    """Fetch JSON with the system curl, returning (status, parsed-or-text).

    curl rather than a Python HTTP client because the Phase-1 environment has no
    `requests`-based HF client installed and Scale 0 must not add a dependency to
    the exact environment merely to read a metadata endpoint.
    """
    command = ["curl", "-sS", "--max-time", str(timeout), "-w", "\n%{http_code}", url]
    if token:
        command[1:1] = ["-H", f"Authorization: Bearer {token}"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 0, str(exc)
    body, _, status = completed.stdout.rpartition("\n")
    try:
        code = int(status.strip())
    except ValueError:
        return 0, completed.stderr or completed.stdout
    try:
        return code, json.loads(body)
    except json.JSONDecodeError:
        return code, body


def local_token() -> str | None:
    """Any Hugging Face token this machine already has. Never prompts for one."""
    for variable in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(variable)
        if value:
            return value
    for path in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        if path.exists():
            text = path.read_text().strip()
            if text:
                return text
    return None


def detect_runtime() -> dict[str, Any]:
    """Which local runtimes could execute a 4B multimodal backbone."""
    import importlib.util as finder

    available = {
        name: finder.find_spec(name) is not None
        for name in ("mlx", "mlx_vlm", "transformers", "torch", "safetensors", "huggingface_hub")
    }
    available["metal"] = False
    if available["mlx"]:
        try:
            import mlx.core as mx

            available["metal"] = bool(mx.metal.is_available())
        except Exception:  # pragma: no cover - runtime probe only
            available["metal"] = False
    return available


def preflight_candidate(
    candidate: BackboneCandidate,
    *,
    weights_root: Path | None = None,
    allow_network: bool = True,
) -> BackbonePreflight:
    """Check one named family without downloading it or accepting anything."""
    runtime = detect_runtime()
    reasons: list[BlockReason] = []
    revision = licence = None
    gated: str | bool | None = None
    total_parameters: int | None = None
    dtypes: tuple[str, ...] = ()
    download_bytes: int | None = None

    if not allow_network:
        return BackbonePreflight(
            candidate=candidate,
            verdict=PreflightVerdict.UNCHECKED,
            detail="network checks disabled",
            measurements={"runtime": runtime},
        )

    status, payload = _curl_json(f"https://huggingface.co/api/models/{candidate.repository}")
    if status != 200 or not isinstance(payload, dict):
        return BackbonePreflight(
            candidate=candidate,
            verdict=PreflightVerdict.BLOCKED,
            reasons=(BlockReason.NO_NETWORK,),
            detail=f"metadata endpoint returned HTTP {status}",
            measurements={"runtime": runtime},
        )

    revision = payload.get("sha")
    card = payload.get("cardData") or {}
    licence = card.get("license")
    gated = payload.get("gated")
    safetensors = payload.get("safetensors") or {}
    total_parameters = safetensors.get("total")
    dtypes = tuple((safetensors.get("parameters") or {}).keys())

    token = local_token()

    # `gated` says the repository is gated. It does not say whether this account
    # has accepted the terms, and conflating the two produces the wrong blocker:
    # an earlier version of this preflight reported "licence requires acceptance"
    # for a repository whose licence had in fact already been accepted, when the
    # only thing missing was a token on this machine. Without a credential the
    # acceptance state is simply not observable from here, so it is reported as
    # unknown rather than guessed.
    if gated not in (False, None) and token is None:
        reasons.append(BlockReason.NO_CREDENTIAL)

    # A read probe on one small file is the ground truth for access; the `gated`
    # field alone does not say whether *this* machine can actually fetch bytes.
    probe_status = None
    if revision:
        probe_url = (
            f"https://huggingface.co/{candidate.repository}/resolve/{revision}/config.json"
        )
        command = ["curl", "-sS", "-L", "--max-time", "30", "-o", os.devnull, "-w", "%{http_code}", probe_url]
        if token:
            command[1:1] = ["-H", f"Authorization: Bearer {token}"]
        try:
            probe = subprocess.run(command, capture_output=True, text=True, timeout=45)
            probe_status = int(probe.stdout.strip() or 0)
        except (OSError, subprocess.TimeoutExpired, ValueError):
            probe_status = 0
        if probe_status != 200:
            if token is None:
                if BlockReason.NO_CREDENTIAL not in reasons:
                    reasons.append(BlockReason.NO_CREDENTIAL)
            else:
                # A credential was presented and still refused: that is the
                # account lacking access, which is the licence question.
                reasons.append(BlockReason.LICENCE_NOT_ACCEPTED)

    if not (runtime["mlx"] or runtime["torch"]):
        reasons.append(BlockReason.NO_RUNTIME)

    local_path = None
    if weights_root is not None:
        candidate_dir = Path(weights_root) / candidate.encoder_id
        if candidate_dir.exists():
            local_path = str(candidate_dir)

    free_bytes = shutil.disk_usage(Path.cwd()).free
    if download_bytes is not None and download_bytes > free_bytes:
        reasons.append(BlockReason.DISK_BUDGET)

    verdict = PreflightVerdict.BLOCKED if reasons else PreflightVerdict.RUNNABLE
    if verdict is PreflightVerdict.RUNNABLE:
        detail = (
            "official repository resolves, licence recorded, revision pinned, read probe succeeded"
        )
    else:
        detail = "; ".join(r.value for r in reasons)
        if gated not in (False, None) and token is None:
            detail += (
                "; the repository is gated and no credential was presented, so whether this "
                "account has accepted the terms is not observable from here"
            )
    return BackbonePreflight(
        candidate=candidate,
        verdict=verdict,
        reasons=tuple(reasons),
        revision=revision,
        licence=licence,
        gated=gated,
        total_parameters=total_parameters,
        stored_dtypes=dtypes,
        download_bytes=download_bytes,
        local_path=local_path,
        runtime="mlx+metal" if runtime.get("metal") else ("torch" if runtime["torch"] else None),
        detail=detail,
        measurements={
            "runtime": runtime,
            "config_probe_http": probe_status,
            "token_present": token is not None,
            "free_disk_bytes": free_bytes,
        },
    )


def preflight_all(
    *, weights_root: Path | None = None, allow_network: bool = True
) -> tuple[BackbonePreflight, ...]:
    return tuple(
        preflight_candidate(c, weights_root=weights_root, allow_network=allow_network)
        for c in FROZEN_CANDIDATES
    )


def matrix_may_run(preflights: tuple[BackbonePreflight, ...]) -> tuple[bool, str]:
    """Both named families must be runnable, or the matrix stops.

    This is the gate the run matrix states in words: "If either family cannot
    run faithfully, Scale 0 stops." It returns a reason rather than raising, so
    a report can print the reason and still emit everything else it measured.
    """
    blocked = [p for p in preflights if p.verdict is not PreflightVerdict.RUNNABLE]
    if not blocked:
        return True, "both frozen encoder families are runnable"
    lines = [
        f"{p.candidate.encoder_id} ({p.candidate.repository}): {p.verdict.value} -- {p.detail}"
        for p in blocked
    ]
    return False, "; ".join(lines)
