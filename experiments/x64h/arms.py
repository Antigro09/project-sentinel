"""All fourteen arms behind one interface.

Every arm receives the same Episode and the same persistent state, returns
the same Verdict, and emits the same metric schema. Only arm 11 receives
the ConventionSpec and only arm 12 receives the target meaning; both are
passed through an oracle channel that the other twelve cannot read.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from . import decision as DE
from . import posterior as PO
from . import queries as Q
from . import semantic as S
from .types import Decision, Taint, TaintError, Verdict

ARMS = (
    "demonstrations_only",
    "static_family_aware",
    "static_default_convention",
    "exact_bayesian_convention",
    "amortized_convention",
    "joint_task_and_convention",
    "joint_random_queries",
    "joint_infogain_queries",
    "no_convention_memory",
    "shuffled_convention_history",
    "oracle_convention",
    "oracle_task_meaning",
    "no_open_world",
    "no_confirmation",
)

ORACLE_ARMS = {"oracle_convention", "oracle_task_meaning"}


@dataclass
class Context:
    family: tuple
    forms: tuple
    cfg: PO.Config
    costs: DE.Costs
    gates: DE.Gates
    budget: int = 6
    query_universe: tuple = ()
    amortized_topk: int = 3


class Oracle:
    """The only channel through which a target-only field may travel, and it
    refuses every arm that is not entitled to it."""

    def __init__(self, phi, z_true):
        self._phi, self._z = phi, z_true

    def convention(self, arm):
        if arm != "oracle_convention":
            raise TaintError(f"arm {arm!r} requested the convention")
        return self._phi

    def meaning(self, arm):
        if arm != "oracle_task_meaning":
            raise TaintError(f"arm {arm!r} requested the target meaning")
        return self._z


def _prior(arm, state, ctx, rng):
    n = len(ctx.family)
    if arm == "no_convention_memory":
        return [-math.log(n)] * n
    if arm == "shuffled_convention_history":
        lp = list(state.log_p_phi)
        rng.shuffle(lp)
        return lp
    return list(state.log_p_phi)


def run_arm(arm, episode, state, ctx, oracle, rng):
    ev = episode.evidence
    fam, forms, cfg = ctx.family, ctx.forms, ctx.cfg
    incomplete = False

    if arm == "demonstrations_only":
        keep = [z for z in forms
                if S.behavioral_loglik(z, ev.demonstrations, cfg.rho)
                > -math.inf]
        beh = {}
        for z in keep:
            beh[S.denote(z)] = beh.get(S.denote(z), 0.0) + 1.0 / max(1, len(keep))
        post = PO.Posterior({}, beh, 0.0, {"IN": 1.0}, 0.0, False, 0.0)
        dec, _d = DE.decide(post, ctx.costs, ctx.gates, 0, can_ask=False)
        b, pb = post.top_behaviour()
        return _verdict(dec, b, post, 0, 0, incomplete), state

    if arm == "oracle_task_meaning":
        z = oracle.meaning(arm)
        return Verdict(Decision.EXECUTE, S.denote(z), {"IN": 1.0}, 0.0, 0.0,
                       0, 0, False, ("oracle meaning",)), state

    if arm == "oracle_convention":
        phis, lp = (oracle.convention(arm),), [0.0]
    elif arm == "static_default_convention":
        phis, lp = (fam[0],), [0.0]
    elif arm == "static_family_aware":
        phis, lp = tuple(fam), [-math.log(len(fam))] * len(fam)
    elif arm == "amortized_convention":
        base = _prior(arm, state, ctx, rng)
        order = sorted(range(len(fam)), key=lambda i: -base[i])
        keep = order[: ctx.amortized_topk]
        phis = tuple(fam[i] for i in keep)
        lp = [base[i] for i in keep]
        incomplete = True          # a truncated candidate set is not exact
    else:
        phis, lp = tuple(fam), _prior(arm, state, ctx, rng)

    use_cfg = cfg
    if arm == "no_open_world":
        use_cfg = replace(cfg, other=PO.OtherModel(
            prior_in=1.0, prior_unknown_realization=0.0,
            prior_unknown_meaning=0.0, prior_unknown_program=0.0))

    post = PO.joint(ev, phis, lp, forms, use_cfg, incomplete=incomplete)
    asked = sem_asked = 0
    policy = {"joint_random_queries": "random",
              "joint_infogain_queries": "infogain"}.get(arm)
    if policy:
        pool = (Q.behavioral_pool(ctx.query_universe, set(), ctx.costs.behavioral_query)
                + Q.semantic_pool(forms, set(), ctx.costs.semantic_query))
        lj = dict(post.log_joint)
        for _ in range(ctx.budget):
            q = Q.choose(policy, pool, lj, forms, phis, rng)
            if q is None:
                break
            z_true = oracle._z
            ans = (S.execute(z_true)(q.payload) if q.kind == "behavioral"
                   else getattr(z_true, q.payload[0]) == q.payload[1])
            lj = Q.restrict(q, ans, lj, forms)
            asked += q.kind == "behavioral"
            sem_asked += q.kind == "semantic"
            pool = [x for x in pool if x is not q]
            if len(lj) <= 1:
                break
        tot = PO.logsumexp(list(lj.values()))
        beh = {}
        for (i, z), v in lj.items():
            beh[S.denote(z)] = beh.get(S.denote(z), 0.0) + math.exp(v - tot)
        post = PO.Posterior(lj, beh, post.p_conflict, post.p_other,
                            post.ambiguity, incomplete, tot)

    gates = ctx.gates
    if arm == "no_confirmation":
        gates = DE.Gates(max_conflict=1.0, min_p_in=0.0,
                         min_leading_behaviour=0.0)
    dec, _d = DE.decide(post, ctx.costs, gates, 0, can_ask=False)
    b, _pb = post.top_behaviour()
    new_state = state
    if arm not in ("no_convention_memory", "demonstrations_only",
                   "oracle_task_meaning") and post.log_joint:
        cp = PO.convention_posterior(post.log_joint, len(phis))
        # NORMALISE. A prior has to be a probability distribution. Persisting
        # the raw marginal let accumulated evidence leak into the prior's
        # SCALE, so by the second task log p(phi) was large and negative,
        # the IN component of the open-world mixture was compared against
        # frozen OTHER priors on the wrong scale, and p_in collapsed to
        # 0.05. The exact arm executed 1 task of 24 while the memoryless
        # control executed all 24 -- the persistent arm was strictly worse
        # than having no memory, which is what exposed it.
        tot = PO.logsumexp(cp)
        if len(cp) == len(state.log_p_phi) and tot > -math.inf:
            new_state = replace(state, log_p_phi=tuple(c - tot for c in cp))
    return _verdict(dec, b, post, asked, sem_asked, incomplete), new_state


def _verdict(dec, b, post, asked, sem, incomplete):
    return Verdict(dec, b if dec == Decision.EXECUTE else None,
                   dict(post.p_other), post.p_conflict, post.ambiguity,
                   asked, sem, incomplete)
