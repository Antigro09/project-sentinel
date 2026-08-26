"""Execute, ask, abstain or expand -- by minimum expected Bayes risk.

Execution requires four things at once: a calibrated conflict gate, a
conditional open-world gate, a conditional leading-behaviour gate, and
minimum expected risk. Any one of them failing sends the decision
elsewhere. The costs are frozen configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import Decision


@dataclass(frozen=True)
class Costs:
    wrong_execution: float = 20.0
    behavioral_query: float = 1.0
    semantic_query: float = 1.0
    abstain: float = 4.0
    expand: float = 6.0


@dataclass(frozen=True)
class Gates:
    """Calibrated on validation only."""
    max_conflict: float = 0.25
    min_p_in: float = 0.60
    min_leading_behaviour: float = 0.90


def decide(post, costs: Costs, gates: Gates, budget_left: int,
           can_ask: bool = True) -> tuple[Decision, dict]:
    b, pb = post.top_behaviour()
    p_in = post.p_other.get("IN", 0.0)
    risk_execute = costs.wrong_execution * (1.0 - pb * p_in)
    risk_abstain = costs.abstain
    risk_expand = costs.expand
    risk_ask = costs.behavioral_query if (can_ask and budget_left > 0) \
        else math.inf

    gate_conflict = post.p_conflict <= gates.max_conflict
    gate_open = p_in >= gates.min_p_in
    gate_lead = pb >= gates.min_leading_behaviour
    detail = {
        "p_top_behaviour": pb, "p_in": p_in, "p_conflict": post.p_conflict,
        "gate_conflict": gate_conflict, "gate_open_world": gate_open,
        "gate_leading": gate_lead,
        "risk_execute": risk_execute, "risk_ask": risk_ask,
        "risk_abstain": risk_abstain, "risk_expand": risk_expand,
    }
    if not (gate_conflict and gate_open and gate_lead):
        # OTHER dominating must not be answerable by executing the single
        # surviving in-class candidate, however concentrated it is.
        worst = max(("UNKNOWN_REALIZATION", "UNKNOWN_MEANING",
                     "UNKNOWN_PROGRAM"),
                    key=lambda k: post.p_other.get(k, 0.0))
        if not gate_open and post.p_other.get(worst, 0.0) >= 0.5:
            if worst == "UNKNOWN_PROGRAM":
                return Decision.EXPAND, detail
            return (Decision.ASK_SEMANTIC if budget_left > 0
                    else Decision.ABSTAIN), detail
        if budget_left > 0 and can_ask:
            return (Decision.ASK_BEHAVIORAL if gate_conflict
                    else Decision.ASK_SEMANTIC), detail
        return Decision.ABSTAIN, detail

    options = {Decision.EXECUTE: risk_execute,
               Decision.ABSTAIN: risk_abstain,
               Decision.EXPAND: risk_expand}
    if risk_ask < math.inf:
        options[Decision.ASK_BEHAVIORAL] = risk_ask
    best = min(options, key=options.get)
    return best, detail
