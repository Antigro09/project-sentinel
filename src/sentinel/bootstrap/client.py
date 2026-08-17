"""Local model client.

Talks to Ollama over HTTP, which is what is already running on this
machine. Deliberately thin — no SDK, no framework — because the only thing
the corpus build needs is "send a prompt, get text, tell me the token
counts", and the token counts are what let us cost a run before committing
a night to it.

Measured on this machine, gpt-oss:120b at ~81 tok/s decode and ~560 tok/s
prefill. The workload is prefill-heavy (long histories in, short programs
out), which is the favourable direction on Apple silicon.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_HOST = "http://localhost:11434"
TEACHER_MODEL = "gpt-oss:120b"
STUDENT_MODEL = "qwen3-coder:30b"


class LLMError(RuntimeError):
    """The model could not be reached or refused to answer."""


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    model: str
    truncated: bool = False
    """True when generation stopped at the token cap rather than finishing.

    Worth surfacing: a truncated response usually presents as a syntax
    error deep in the file, which reads like the model writing bad code
    when it actually wrote fine code and was cut off.
    """
    thinking: str = ""
    """Reasoning-channel output, returned separately by harmony-format models.

    Kept because it is where the answer ends up when the model runs out of
    budget mid-thought — see `OllamaClient.complete`.
    """

    @property
    def decode_rate(self) -> float:
        return self.output_tokens / self.seconds if self.seconds else 0.0

    def summary(self) -> str:
        return (
            f"{self.model}: {self.prompt_tokens}p + {self.output_tokens}o "
            f"in {self.seconds:.1f}s ({self.decode_rate:.0f} tok/s)"
        )


@dataclass
class UsageTally:
    """Running cost of a corpus build, in tokens and wall-clock."""

    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    failures: int = 0

    def add(self, completion: Completion) -> None:
        self.calls += 1
        self.prompt_tokens += completion.prompt_tokens
        self.output_tokens += completion.output_tokens
        self.seconds += completion.seconds

    def summary(self) -> str:
        rate = self.output_tokens / self.seconds if self.seconds else 0.0
        return (
            f"{self.calls} calls, {self.failures} failed, "
            f"{self.prompt_tokens:,}p + {self.output_tokens:,}o tokens, "
            f"{self.seconds / 60:.1f} min, {rate:.0f} tok/s avg"
        )


@dataclass
class OllamaClient:
    """Minimal Ollama HTTP client."""

    model: str = TEACHER_MODEL
    host: str = DEFAULT_HOST
    temperature: float = 0.2
    """Low but non-zero. The repair loop benefits from the model trying a
    genuinely different hypothesis on retry rather than re-deriving the
    same wrong one."""
    num_predict: int = 9000
    """Must be generous. gpt-oss splits output between a reasoning channel
    and the answer, and at 2600 the reasoning consumed the entire budget and
    the answer came back empty — a silent failure that looks like the model
    refusing to respond."""
    think: str | None = "low"
    """Reasoning effort for harmony-format models. Measured on this machine:
    "low" spends 13 characters on reasoning and 1.7k tokens total; "high"
    spends 10.6k characters and 3.6k tokens. Low is the right default for
    bulk generation; the teacher escalates on repair rounds, where the
    extra thought is actually buying something."""
    keep_alive: str = "2h"
    """How long Ollama keeps the model resident between calls.

    The default is 5 minutes. A world takes ~3 minutes, so a single slow
    one — a long container verify, a retry — can let a 65GB model unload
    and pay a cold reload on the next call. Holding it resident for the
    length of a run costs nothing extra: the memory is already committed.
    """
    timeout: float = 900.0
    usage: UsageTally = field(default_factory=UsageTally)

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/version", timeout=5):
                return True
        except Exception:  # noqa: BLE001
            return False

    def installed_models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as resp:
                data = json.load(resp)
            return sorted(m["name"] for m in data.get("models", []))
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"could not list models: {exc}") from exc

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        think: str | None | bool = "__default__",
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.num_predict,
            },
        }
        if system:
            payload["system"] = system
        effort = self.think if think == "__default__" else think
        if effort is not None:
            payload["think"] = effort

        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.URLError as exc:
            self.usage.failures += 1
            raise LLMError(f"{self.model} unreachable at {self.host}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            self.usage.failures += 1
            raise LLMError(f"{self.model} request failed: {exc}") from exc

        text = data.get("response") or ""
        thinking = data.get("thinking") or ""

        # If the budget ran out mid-thought the answer is empty while the
        # reasoning channel holds whatever was produced. Falling back to it
        # recovers a usable program often enough to be worth doing, and the
        # alternative is discarding a full generation as a silent blank.
        if not text.strip() and thinking.strip():
            text = thinking

        completion = Completion(
            text=text,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            seconds=time.perf_counter() - started,
            model=self.model,
            truncated=data.get("done_reason") == "length",
            thinking=thinking,
        )
        self.usage.add(completion)
        return completion
