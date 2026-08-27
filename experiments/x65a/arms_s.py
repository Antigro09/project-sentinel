"""X65A-S: the twelve arms, one runner, one ledger.

Every arm sees identical current-task evidence, the same candidate pool, the
same clarification budget, the same interpreter and the same stopping rule.
The only permitted difference is WHAT PRIOR over conventions it brings to the
task, and where that prior came from. The opaque identity label is public to
every non-oracle arm; the main arm's only retrieval privilege is a dictionary
lookup by that label -- no scoring, no frontier, no selection.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from x64h import episode as EP
from x64h import family as FAM

from . import evidence as EV
from . import semantic_mem as SM
from .types import Status, byte_cost, canon

LIKELIHOOD = "aware"          # X64H-0C's trusted, correctly specified model
QUERY_CAP = 8

ARMS = ("none", "within_episode", "raw_replay", "most_recent",
        "random_record", "shuffled_ids", "wrong_similar", "main",
        "main_no_quarantine", "unlimited", "oracle", "bigger_query_budget")


@dataclass
class Ledger:
    posterior_evals: int = 0
    likelihood_evals: int = 0
    queries: int = 0
    interpreter_execs: int = 0
    replayed_episodes: int = 0
    serializations: int = 0
    archive_reads: int = 0
    wall_s: float = 0.0

    def total_units(self) -> int:
        return (self.posterior_evals + self.likelihood_evals + self.queries
                + self.interpreter_execs + self.replayed_episodes
                + self.serializations + self.archive_reads)

    def canon(self):
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()} | {
            "total_units": self.total_units()}


def _uniform(fam):
    return np.full(fam.n, 1.0 / fam.n)


def _delta(fam, i):
    v = np.zeros(fam.n)
    v[i] = 1.0
    return v


def solve(fam, beh, prior, task, led: Ledger, queries: int = 0):
    """One task. Returns (correct, queries_used). Clarification probes are
    charged to the ledger and capped identically for every arm."""
    live = list(task.live)
    asked = set(task.demos)
    led.posterior_evals += 1
    led.likelihood_evals += 1
    b, _c, best = EP._infer_by(LIKELIHOOD, fam, prior, task.u, task.pool,
                               live, task.tie)
    used = 0
    while used < queries and best != task.z and len(live) > 1:
        cand = [k for k in range(len(EP.UNIVERSE)) if k not in asked]
        split = []
        for k in cand:
            d: dict = {}
            for j in live:
                d[beh[j][k]] = d.get(beh[j][k], 0.0) + b[j]
            if len(d) > 1:
                split.append((-sum(p * math.log2(p)
                                   for p in d.values() if p > 0), k))
        if not split:
            break
        k = max(split)[1]
        asked.add(k)
        led.queries += 1
        led.interpreter_execs += 1
        used += 1
        y = beh[task.z][k]
        live = [j for j in live if beh[j][k] == y]
        led.posterior_evals += 1
        b, _c, best = EP._infer_by(LIKELIHOOD, fam, prior, task.u, task.pool,
                                   live, task.tie)
    return best == task.z, used


def queries_to_correct(fam, beh, prior, task, led: Ledger, cap=QUERY_CAP):
    ok, used = solve(fam, beh, prior, task, led, queries=cap)
    return used if ok else cap + 1


class Arm:
    """State carried across episodes, per arm."""

    def __init__(self, name, fam, beh, rng, compute_ceiling=None):
        self.name, self.fam, self.beh, self.rng = name, fam, beh, rng
        self.store = SM.SemanticStore(budget_bytes=4096)
        self.ledger = Ledger()
        self.evidence = EV.EvidenceLedger()
        self.raw: list = []                  # raw_replay only
        self.raw_budget = 4096
        self.episode_grounded: tuple = ()    # within_episode only
        self.shuffle_map: dict = {}
        self.written_order: list = []
        self.evicted = 0
        self.compute_ceiling = compute_ceiling
        self.truncated_replays = 0

    # ------------------------------------------------------------ writing

    def derive_meaning(self, task):
        """What the DEMONSTRATIONS identify, using only public evidence:
        the input tapes and the outputs observed on them. Reading the
        generator's `task.z` instead would give the semantic arm information
        no raw episode contains, and the unlimited-replay equivalence
        diagnostic would be comparing two different observers."""
        live = list(range(self.fam.m))
        for d in task.demos:
            y = self.beh[task.z][d]
            live = [j for j in live if self.beh[j][d] == y]
        return live[0] if len(live) == 1 else None

    def observe_episode(self, app, task_index: int) -> None:
        obs = []
        for t in app.cal:
            # THE SHARED COST. Identifying the meaning from demonstrations is
            # paid once by every arm, structured or not. What differs is
            # whether the arm has to pay it AGAIN on every later read; that
            # difference is exactly what structure is supposed to buy, so it
            # must not be hidden by charging only one side.
            live_n = self.fam.m
            for d in t.demos:
                self.ledger.interpreter_execs += live_n
                live_n = max(1, live_n // 2)
            z = self.derive_meaning(t)
            if z is None:
                continue          # the demonstrations did not identify it
            k = EV.ExternalEvidenceKey(
                "teacher", f"ep{app.index}", len(obs),
                EV.observation_hash((z, t.u)), app.label)
            bid, _new = self.evidence.absorb(k)
            obs.append(SM.GroundedObservation(z, t.u, bid))
        self.episode_grounded = tuple(obs)
        if self.name in ("none", "oracle", "bigger_query_budget",
                         "within_episode"):
            return
        if self.name == "raw_replay":
            # GENUINELY RAW. The episode as observed: the utterance and the
            # input/output demonstrations. NOT the derived meaning index --
            # storing `z` would make this semantic memory with extra steps,
            # which is what the first version of this arm accidentally was.
            rec = {"identity": app.label, "episode": app.index,
                   "cal": [{"u": t.u,
                            "demos": [[EP.UNIVERSE[d], self.beh[t.z][d]]
                                      for d in t.demos]}
                           for t in app.cal]}
            if rec["cal"]:
                self.raw.append(rec)
                self.ledger.serializations += 1
                while byte_cost(self.raw) > self.raw_budget and self.raw:
                    self.raw.pop(0)          # oldest first, under the budget
                    self.evicted += 1
            return
        if not obs:
            return
        quarantine = self.name != "main_no_quarantine"
        rec = self.store.get(app.label) or SM.SemanticRecord(app.label)
        before = rec.grounded
        for o in obs:
            rec, why = SM.absorb(self.fam, rec, o, task_index, self.evidence,
                                 quarantine=quarantine)
        self.store.put(rec)
        self.ledger.serializations += 1
        self.written_order.append(app.label)
        if self.name != "unlimited":
            while self.store.over_budget() and len(self.store.records) > 1:
                oldest = self.written_order.pop(0)
                self.store.records.pop(oldest, None)
                self.evicted += 1

    # ------------------------------------------------------------ reading

    def prior_for(self, app):
        fam = self.fam
        n = self.name
        if n in ("none", "bigger_query_budget"):
            return _uniform(fam)
        if n == "oracle":
            return _delta(fam, app.phi)
        if n == "within_episode":
            g = self.episode_grounded
            return SM.prior_from(fam, g) if g else _uniform(fam)
        if n == "raw_replay":
            mine = [r for r in self.raw if r["identity"] == app.label]
            self.ledger.replayed_episodes += len(self.raw)
            self.ledger.archive_reads += 1
            # the meaning must be RE-DERIVED by replaying the demonstrations
            # through the trusted interpreter, at the same cost the original
            # grounding paid
            g = []
            for r in mine:
                if (self.compute_ceiling is not None
                        and self.ledger.total_units() >= self.compute_ceiling):
                    self.truncated_replays += 1
                    break
                for c in r["cal"]:
                    live = list(range(fam.m))
                    for x, y in c["demos"]:
                        k = EP.UNIVERSE.index(x)
                        self.ledger.interpreter_execs += len(live)
                        live = [j for j in live if self.beh[j][k] == y]
                    if len(live) == 1:
                        g.append(SM.GroundedObservation(live[0], c["u"],
                                                        "replay"))
            g = tuple(g)
            if not g:
                return _uniform(fam)
            try:
                return SM.prior_from(fam, g)
            except Exception:
                return _uniform(fam)
        if n == "most_recent":
            if not self.written_order:
                return _uniform(fam)
            r = self.store.get(self.written_order[-1])
            return SM.prior_from(fam, r.grounded) if r and r.grounded \
                else _uniform(fam)
        if n == "random_record":
            keys = list(self.store.records)
            if not keys:
                return _uniform(fam)
            r = self.store.get(keys[self.rng.randrange(len(keys))])
            return SM.prior_from(fam, r.grounded) if r and r.grounded \
                else _uniform(fam)
        if n == "shuffled_ids":
            keys = sorted(self.store.records)
            if not keys:
                return _uniform(fam)
            if app.label not in self.shuffle_map:
                self.shuffle_map[app.label] = keys[
                    self.rng.randrange(len(keys))]
            r = self.store.get(self.shuffle_map[app.label])
            return SM.prior_from(fam, r.grounded) if r and r.grounded \
                else _uniform(fam)
        if n == "wrong_similar":
            from .streams import near_convention
            return _delta(fam, near_convention(fam, app.phi, self.rng))
        # main | main_no_quarantine | unlimited
        r = self.store.get(app.label)
        self.ledger.archive_reads += 1
        if r is None or not r.grounded:
            return _uniform(fam)
        try:
            return SM.prior_from(fam, r.grounded)
        except Exception:
            return _uniform(fam)

    def query_budget(self) -> int:
        return 4 if self.name == "bigger_query_budget" else 1

    def active_bytes(self) -> int:
        if self.name == "raw_replay":
            return byte_cost(self.raw)
        return self.store.bytes()
