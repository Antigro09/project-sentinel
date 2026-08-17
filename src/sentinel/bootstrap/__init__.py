"""Bootstrap layer — the LLM teacher, and its eventual removal.

The teacher writes world models; the verifier scores them automatically;
the corpus records both. Nothing here is meant to be permanent. It exists
to produce the training data for `core/`, and the metric that matters is
the share of episodes the system can eventually solve with no LLM call at
all.
"""

from .client import (
    STUDENT_MODEL,
    TEACHER_MODEL,
    Completion,
    LLMError,
    OllamaClient,
    UsageTally,
)
from .corpus import (
    CorpusRecord,
    CorpusWriter,
    completed_ids,
    corpus_stats,
    iter_usable,
    read_corpus,
)
from .loader import (
    LoadedModel,
    LoadError,
    ModelTimeout,
    extract_code,
    load_model,
    normalize_grid,
    normalize_outcome,
    time_guard,
)
from .prompts import build_initial_prompt, build_repair_prompt, describe_history
from .teacher import Attempt, InductionResult, Teacher, make_training_history

__all__ = [
    "STUDENT_MODEL",
    "TEACHER_MODEL",
    "Attempt",
    "Completion",
    "CorpusRecord",
    "CorpusWriter",
    "InductionResult",
    "LLMError",
    "LoadError",
    "LoadedModel",
    "ModelTimeout",
    "OllamaClient",
    "Teacher",
    "UsageTally",
    "build_initial_prompt",
    "build_repair_prompt",
    "completed_ids",
    "corpus_stats",
    "describe_history",
    "extract_code",
    "iter_usable",
    "load_model",
    "make_training_history",
    "normalize_grid",
    "normalize_outcome",
    "read_corpus",
    "time_guard",
]
