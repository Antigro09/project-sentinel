"""One metric schema for every arm, and task-level confidence intervals."""

from __future__ import annotations

import random

SCHEMA = ("arm", "n", "executed", "correct", "wrong", "abstained",
          "expanded", "asked_semantic", "asked_behavioral", "conflict_flag",
          "mean_p_in", "mean_ambiguity", "incomplete_candidates",
          "convention_class_mass")


def blank(arm: str) -> dict:
    d = {k: 0 for k in SCHEMA}
    d["arm"] = arm
    d["incomplete_candidates"] = False
    d["mean_p_in"] = 0.0
    d["mean_ambiguity"] = 0.0
    d["convention_class_mass"] = 0.0
    return d


def accumulate(acc: dict, verdict, correct: bool) -> None:
    from .types import Decision
    acc["n"] += 1
    acc["asked_behavioral"] += verdict.asked
    acc["asked_semantic"] += verdict.semantic_asked
    acc["mean_p_in"] += verdict.open_world.get("IN", 0.0)
    acc["mean_ambiguity"] += verdict.ambiguity
    acc["incomplete_candidates"] = (acc["incomplete_candidates"]
                                    or verdict.incomplete_candidates)
    if verdict.decision == Decision.EXECUTE:
        acc["executed"] += 1
        acc["correct"] += bool(correct)
        acc["wrong"] += not correct
    elif verdict.decision == Decision.ABSTAIN:
        acc["abstained"] += 1
    elif verdict.decision == Decision.EXPAND:
        acc["expanded"] += 1
    else:
        acc["conflict_flag"] += 1


def finish(acc: dict) -> dict:
    n = max(1, acc["n"])
    acc["mean_p_in"] /= n
    acc["mean_ambiguity"] /= n
    return acc


def paired_ci(pairs, n=2000, seed=13):
    rg = random.Random(seed)
    out = []
    for _ in range(n):
        s = [pairs[rg.randrange(len(pairs))] for _ in pairs]
        out.append(sum(a - b for a, b in s) / len(s))
    out.sort()
    return out[int(0.025 * n)], sum(out) / len(out), out[int(0.975 * n)]
