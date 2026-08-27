"""X64H-0B: calibration tasks that ground the convention, transfer tasks
that require it.

X64H-0 failed V1 and V10 because one demonstration schedule was asked to do
two incompatible jobs at once: keep the task ambiguous so language is
needed, and reveal what the codewords mean so the convention is learnable.
The sweep showed the coupling directly -- with the withholding weight at
zero the oracle reached 0.997 and recovery 0.84 but ambiguity collapsed to
1.70 classes; with it positive ambiguity rose to 4.25 and recovery fell to
0.14.

Here the two jobs are given to two disjoint task sets.

    CALIBRATION  demonstrations identify the meaning on their own, and the
                 utterance exposes all three roles. The learner grounds
                 (meaning, utterance) and the convention posterior sharpens.
                 No oracle label is used: the demonstrations ARE the
                 grounding.
    TRANSFER     the meaning is unseen, the demonstrations deliberately
                 leave 2-8 behaviours open, and the utterance exposes fewer
                 roles. Only a learned convention closes the gap.

The generator SELECTS the exposure pattern so that the true convention makes
the transfer task identifiable. That is a constructed oracle ceiling and is
reported as such: `rejected_tasks` counts the candidates discarded. The
demonstrations themselves are chosen without reference to the convention,
so the only convention-dependent choice in a transfer task is which of the
three two-role patterns is spoken -- a teacher choice the specification
grants explicitly.

The learner's observation model is uniform over the pool. The teacher's
pattern choice is not uniform, so the model is MISSPECIFIED against the
generator. That is conservative: it can only cost the arms that use the
model, which are the treatment arms.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

import numpy as np

from . import family as FAM
from . import semantic as S

UNIVERSE = list(S.UNIVERSE)


def behaviour_table(forms) -> list[tuple]:
    return [tuple(S.execute(z)(x) for x in UNIVERSE) for z in forms]


@dataclass(frozen=True)
class Task:
    kind: str                       # cal | transfer
    z: int                          # index of the true form
    demos: tuple                    # universe indices
    live: tuple                     # form indices consistent with the demos
    u: int                          # the utterance code
    pool: tuple                     # the exposure pool it was drawn from
    tie: tuple                      # a task-local order for argmax ties


@dataclass
class Episode:
    phi: int
    phi_alt: int
    tasks: list                     # in presentation order
    cal_idx: list                   # positions of calibration tasks
    tr_idx: list                    # positions of transfer tasks
    u_alt: dict = field(default_factory=dict)     # position -> phi_alt code
    u_wrong: dict = field(default_factory=dict)   # position -> mispaired code
    rejected: int = 0
    reject_band: int = 0
    reject_oracle: int = 0
    oracle_unselected: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    boundary: int = 0


@dataclass(frozen=True)
class Config:
    overlap: str = "shared"
    n_cal: int = 6
    n_transfer: int = 16
    demos_cal_cap: int = 6
    demos_transfer_cap: int = 4
    ambiguity: tuple = (2, 8)
    exposure_mix: tuple = (0.0, 1.0, 0.0)     # P(1 role), P(2 roles), P(3)
    order_p: float = 0.5
    schedule: str = "interleaved"             # interleaved | front
    retries: int = 24
    queries: int = 0

    def label(self) -> str:
        return (f"{self.overlap}/cal{self.n_cal}/tr{self.n_transfer}/"
                f"dc{self.demos_cal_cap}/dt{self.demos_transfer_cap}/"
                f"amb{self.ambiguity[0]}-{self.ambiguity[1]}/"
                f"exp{self.exposure_mix}/ord{self.order_p}/{self.schedule}")


# ------------------------------------------------------- demonstrations

def _consistent(beh, live, k, y):
    return tuple(j for j in live if beh[j][k] == y)


def pick_demos(beh, m, j_true, target, cap, rng, forbid=()):
    """Greedy probes towards a target number of surviving behaviours. The
    convention plays no part: a demonstration is an input and an output."""
    live = tuple(range(m))
    demos: list[int] = []
    order = [k for k in range(len(UNIVERSE)) if k not in forbid]
    rng.shuffle(order)
    while len(live) > target and len(demos) < cap:
        best, score = None, None
        for k in order:
            if k in demos:
                continue
            nxt = _consistent(beh, live, k, beh[j_true][k])
            if len(nxt) == len(live):
                continue
            # closest to the target from ABOVE; never undershoot if some
            # probe can avoid it
            key = (0 if len(nxt) >= target else 1, abs(len(nxt) - target))
            if score is None or key < score:
                best, score = k, key
        if best is None:
            break
        demos.append(best)
        live = _consistent(beh, live, best, beh[j_true][best])
    return tuple(demos), live


# ------------------------------------------------------ episode building

def _covering_calibration(rng, n_cal, nF=4, nS=4, nO=2):
    """A balanced covering design, not a random sample: every filter value,
    every scope value and every operator value appears, and the (F, S)
    pairing is a random Latin row so the design is not a fixed diagonal."""
    tasks = []
    blocks = max(1, math.ceil(n_cal / max(nF, nS)))
    for b in range(blocks):
        fs = list(range(nF))
        ss = list(range(nS))
        rng.shuffle(fs)
        rng.shuffle(ss)
        ops = [(i + b) % nO for i in range(max(nF, nS))]
        rng.shuffle(ops)
        block = [(ops[i % len(ops)], fs[i % nF], ss[i % nS])
                 for i in range(max(nF, nS))]
        rng.shuffle(block)
        tasks += block
    # the FIRST block is a complete covering of every filter and scope
    # value; truncating a shuffle of all blocks destroyed that, and the
    # transfer meanings then contained atoms calibration had never grounded
    return tasks[:n_cal]


def _form_index(fam, o, f, s):
    return o * len(FAM.FILTERS_0B) * len(FAM.SCOPES_0B) + f * len(FAM.SCOPES_0B) + s


def _pools(cfg):
    p1, p2, p3 = cfg.exposure_mix
    out = []
    if p1 > 0:
        out.append((p1, FAM.P1))
    if p2 > 0:
        out.append((p2, FAM.P2))
    if p3 > 0:
        out.append((p3, FAM.CAL_POOL))
    return out


def build_episode(fam, beh, cfg: Config, seed: int) -> Episode:
    rng = random.Random(seed)
    order_want = 1 if rng.random() < cfg.order_p else 0
    cand = np.where(fam.ORD == order_want)[0]
    phi = int(cand[rng.randrange(len(cand))])
    # a GENUINELY different convention: drawn from the whole family, not
    # from the same order-bit group, or the shuffled control would inherit
    # one correct bit for free
    phi_alt = int(rng.randrange(fam.n))
    while phi_alt == phi:
        phi_alt = int(rng.randrange(fam.n))

    m = fam.m
    cal_triples = _covering_calibration(rng, cfg.n_cal)
    cal_forms = [_form_index(fam, *t) for t in cal_triples]
    seen = set(cal_forms)
    pool_tr = [j for j in range(m) if j not in seen]
    rng.shuffle(pool_tr)
    tr_forms = pool_tr[:cfg.n_transfer]

    cal_tasks = []
    for j in cal_forms:
        d, live = pick_demos(beh, m, j, 1, cfg.demos_cal_cap, rng)
        u = fam.realise(phi, j, ("O", "F", "S"))
        tie = tuple(rng.sample(range(m), m))
        cal_tasks.append(Task("cal", j, d, live, u, FAM.CAL_POOL, tie))

    tr_tasks, rejected = [], 0
    reject_band = reject_oracle = 0
    unselected = []
    lo, hi = cfg.ambiguity
    for j in tr_forms:
        made = None
        for _ in range(cfg.retries):
            target = rng.randint(lo, hi)
            d, live = pick_demos(beh, m, j, target, cfg.demos_transfer_cap, rng)
            if not (lo <= len(live) <= hi) or j not in live:
                rejected += 1
                reject_band += 1
                continue
            w = rng.random()
            acc, pool = 0.0, FAM.P2
            for p, pl in _pools(cfg):
                acc += p
                if w <= acc:
                    pool = pl
                    break
            good = []
            for pat in pool:
                u = fam.realise(phi, j, pat)
                surv = [k for k in live
                        if any(fam.realise(phi, k, q) == u for q in pool)]
                if surv == [j]:
                    good.append(u)
            # what an UNSELECTED exposure would have given the oracle:
            # the ceiling is constructed, and this is what it cost
            pat0 = pool[rng.randrange(len(pool))]
            u0 = fam.realise(phi, j, pat0)
            surv0 = [k for k in live
                     if any(fam.realise(phi, k, q) == u0 for q in pool)]
            unselected.append(surv0 == [j])
            if not good:
                rejected += 1
                reject_oracle += 1
                continue
            u = good[rng.randrange(len(good))]
            tie = tuple(rng.sample(range(m), m))
            made = Task("transfer", j, d, live, u, pool, tie)
            break
        if made is not None:
            tr_tasks.append(made)

    if cfg.schedule == "front":
        tasks = cal_tasks + tr_tasks
    else:
        tasks, ci = [], 0
        n = len(cal_tasks) + len(tr_tasks)
        slots = set(round(i * n / max(1, len(cal_tasks)))
                    for i in range(len(cal_tasks)))
        ti = 0
        for pos in range(n):
            if pos in slots and ci < len(cal_tasks):
                tasks.append(cal_tasks[ci]); ci += 1
            elif ti < len(tr_tasks):
                tasks.append(tr_tasks[ti]); ti += 1
            elif ci < len(cal_tasks):
                tasks.append(cal_tasks[ci]); ci += 1
        tasks += cal_tasks[ci:] + tr_tasks[ti:]

    cov = {"op": {t_[0] for t_ in cal_triples},
           "filt": {t_[1] for t_ in cal_triples},
           "scope": {t_[2] for t_ in cal_triples}}
    ep = Episode(phi, phi_alt, tasks,
                 [i for i, t in enumerate(tasks) if t.kind == "cal"],
                 [i for i, t in enumerate(tasks) if t.kind == "transfer"],
                 rejected=rejected, reject_band=reject_band,
                 reject_oracle=reject_oracle, oracle_unselected=unselected,
                 coverage=cov, boundary=len(tasks) // 2)
    cal_pos = ep.cal_idx
    shifted = cal_pos[1:] + cal_pos[:1]
    for a, b in zip(cal_pos, shifted):
        ep.u_alt[a] = fam.realise(phi_alt, tasks[a].z, ("O", "F", "S"))
        ep.u_wrong[a] = fam.realise(phi, tasks[b].z, ("O", "F", "S"))
    return ep


def repeat_episode(fam, beh, cfg, seed):
    """V6 control: transfer tasks whose complete meanings ARE calibration
    meanings, so any advantage could come from memorising a task."""
    ep = build_episode(fam, beh, cfg, seed)
    rng = random.Random(seed + 99991)
    cal_forms = [ep.tasks[i].z for i in ep.cal_idx]
    tasks = list(ep.tasks)
    lo, hi = cfg.ambiguity
    m = fam.m
    for pos in ep.tr_idx:
        j = cal_forms[rng.randrange(len(cal_forms))]
        made = None
        for _ in range(cfg.retries):
            target = rng.randint(lo, hi)
            d, live = pick_demos(beh, m, j, target, cfg.demos_transfer_cap, rng)
            if not (lo <= len(live) <= hi) or j not in live:
                continue
            good = []
            for pat in FAM.P2:
                u = fam.realise(ep.phi, j, pat)
                surv = [k for k in live
                        if any(fam.realise(ep.phi, k, q) == u for q in FAM.P2)]
                if surv == [j]:
                    good.append(u)
            if not good:
                continue
            made = Task("transfer", j, d, live,
                        good[rng.randrange(len(good))], FAM.P2,
                        tuple(rng.sample(range(m), m)))
            break
        if made is not None:
            tasks[pos] = made
    ep.tasks = tasks
    return ep


# ------------------------------------------------------------ inference

NEG = -math.inf


def _norm(v):
    t = v.sum()
    if t <= 0:
        return None
    return v / t


def infer(fam, p_phi, u, pool, live, tie):
    """Exact joint over (convention, form).

        p(phi, z | u, D) proportional to p(phi) p(z | D) p(u | phi, z)

    Returns the form posterior, the convention marginal and the argmax."""
    C = fam.counts(u, pool) / len(pool)
    mask = np.zeros(fam.m)
    mask[list(live)] = 1.0
    joint = (p_phi[:, None] * C) * mask[None, :]
    tot = joint.sum()
    if tot <= 0:
        joint = C * mask[None, :]
        tot = joint.sum()
        if tot <= 0:
            joint = np.tile(mask, (fam.n, 1))
            tot = joint.sum()
    b = joint.sum(axis=0) / tot
    conv = joint.sum(axis=1) / tot
    best, bs = None, -1.0
    for j in tie:
        if b[j] > bs + 1e-12:
            best, bs = j, b[j]
    return b, conv, best


def entropy_bits(p) -> float:
    q = p[p > 0]
    return float(-(q * np.log2(q)).sum())


# ----------------------------------------------------------------- arms

ARMS = ("oracle", "persist", "static", "reset", "shuffled", "phi_change",
        "wrong_pairing", "repeat_task", "default", "demos_only",
        "query_random", "query_infogain", "selection_aware")


def _uniform(fam):
    return np.full(fam.n, 1.0 / fam.n)


def _delta(fam, i):
    v = np.zeros(fam.n)
    v[i] = 1.0
    return v


def _query(fam, beh, arm, task, p_phi, rng, budget):
    """A behavioural question: name an input, receive its output. `random`
    picks uniformly among questions that split the posterior; `infogain`
    maximises the exact answer entropy under the current joint. Same pool,
    same budget, same answer channel."""
    live = list(task.live)
    asked = set(task.demos)
    b, conv, best = infer(fam, p_phi, task.u, task.pool, live, task.tie)
    for _ in range(budget):
        cand = [k for k in range(len(UNIVERSE)) if k not in asked]
        splits = []
        for k in cand:
            d: dict = {}
            for j in live:
                d[beh[j][k]] = d.get(beh[j][k], 0.0) + b[j]
            if len(d) > 1:
                h = -sum(p * math.log2(p) for p in d.values() if p > 0)
                splits.append((h, k))
        if not splits:
            break
        if arm == "query_random":
            k = splits[rng.randrange(len(splits))][1]
        else:
            k = max(splits)[1]
        asked.add(k)
        y = beh[task.z][k]
        live = [j for j in live if beh[j][k] == y]
        b, conv, best = infer(fam, p_phi, task.u, task.pool, live, task.tie)
    return b, conv, best, len(asked) - len(task.demos)


def run_arm(fam, beh, ep: Episode, arm: str, cfg: Config, seed: int) -> dict:
    rng = random.Random(seed * 7919 + hash(arm) % 100003)
    if arm == "oracle":
        p = _delta(fam, ep.phi)
    elif arm == "default":
        p = _delta(fam, 0)
    else:
        p = _uniform(fam)

    phi_after = (ep.phi_alt if arm == "phi_change" else ep.phi)
    cal_correct, tr_correct, tr_pos = [], [], []
    ent, mass, ncl, queries = [], [], [], []
    prior_H, norm_err = [], 0.0
    trace = []
    cls = fam.class_of()
    true_cls = int(cls[ep.phi])

    ent.append(entropy_bits(_uniform(fam)))
    mass.append(float(_uniform(fam)[cls == true_cls].sum()))

    for pos, t in enumerate(ep.tasks):
        u = t.u
        if t.kind == "cal":
            if arm == "shuffled":
                u = ep.u_alt.get(pos, u)
            elif arm == "wrong_pairing":
                u = ep.u_wrong.get(pos, u)
            elif arm == "phi_change" and pos >= ep.boundary:
                u = fam.realise(phi_after, t.z, t.pool[0])
        else:
            if arm == "phi_change" and pos >= ep.boundary:
                u = fam.realise(phi_after, t.z, t.pool[rng.randrange(len(t.pool))])

        use = p
        if arm in ("static", "demos_only"):
            use = _uniform(fam)
        if arm == "reset" and t.kind == "transfer":
            use = _uniform(fam)

        prior_H.append(entropy_bits(use))
        norm_err = max(norm_err, abs(float(use.sum()) - 1.0))

        if arm == "selection_aware":
            # ADVERSARIAL CONTROL against the generator itself. The teacher
            # picks the exposure so the true convention identifies the task,
            # and that choice depends on the convention. A learner that
            # modelled the SELECTION RULE rather than a uniform pool could
            # exploit it without ever learning the convention. This arm does
            # exactly that: it scores each candidate meaning by how many
            # conventions would have made the utterance select it uniquely.
            # If the generator leaks through its own selection, this arm
            # beats chance and the gap is not attributable to convention
            # learning.
            L = list(t.live)
            hit = fam.counts(u, t.pool)[:, L] > 0
            uniq = hit.sum(axis=1) == 1
            w = np.zeros(fam.m)
            if uniq.any():
                for wi in hit[uniq].argmax(axis=1):
                    w[L[int(wi)]] += 1.0
            if w.sum() <= 0:
                w[L] = 1.0
            b = w / w.sum()
            conv = _uniform(fam)
            best, bs = None, -1.0
            for j in t.tie:
                if b[j] > bs + 1e-12:
                    best, bs = j, b[j]
            nq = 0
        elif arm == "demos_only":
            b = np.zeros(fam.m)
            b[list(t.live)] = 1.0 / len(t.live)
            conv = _uniform(fam)
            best = next(j for j in t.tie if j in set(t.live))
            nq = 0
        elif arm in ("query_random", "query_infogain") and t.kind == "transfer":
            b, conv, best, nq = _query(fam, beh, arm, t, use, rng, cfg.queries)
        else:
            b, conv, best = infer(fam, use, u, t.pool, t.live, t.tie)
            nq = 0

        ok = (best == t.z)
        if t.kind == "cal":
            cal_correct.append(ok)
        else:
            tr_correct.append(ok)
            tr_pos.append(pos)
            ncl.append(len(t.live))
            queries.append(nq)

        if arm in ("persist", "shuffled", "phi_change", "wrong_pairing",
                   "repeat_task", "reset", "query_random", "query_infogain"):
            if arm == "reset" and t.kind == "transfer":
                pass
            else:
                nz = _norm(conv)
                if nz is not None:
                    p = nz
        ent.append(entropy_bits(p))
        mass.append(float(p[cls == true_cls].sum()))
        trace.append({"pos": pos, "kind": t.kind, "correct": bool(ok),
                      "live": len(t.live), "H": ent[-1], "mass": mass[-1]})

        norm_err = max(norm_err, abs(float(p.sum()) - 1.0))
    return {"arm": arm, "cal": cal_correct, "transfer": tr_correct,
            "transfer_pos": tr_pos, "entropy": ent, "mass": mass,
            "classes": ncl, "queries": queries, "trace": trace,
            "prior_H": prior_H, "max_normalisation_error": norm_err}


def first_task_indexed(fam, ep: Episode) -> dict:
    """Entropy and true-class mass at the times the brief names: the prior,
    after the first utterance alone, and after the first demonstrations."""
    cls = fam.class_of()
    true_cls = int(cls[ep.phi])
    p0 = _uniform(fam)
    t = ep.tasks[0]
    _b, c_u, _ = infer(fam, p0, t.u, t.pool, tuple(range(fam.m)), t.tie)
    _b, c_ud, _ = infer(fam, p0, t.u, t.pool, t.live, t.tie)
    return {
        "H_prior": entropy_bits(p0),
        "H_after_first_utterance": entropy_bits(_norm(c_u)),
        "H_after_first_demonstrations": entropy_bits(_norm(c_ud)),
        "mass_prior": float(p0[cls == true_cls].sum()),
        "mass_after_first_utterance": float(_norm(c_u)[cls == true_cls].sum()),
        "mass_after_first_demonstrations":
            float(_norm(c_ud)[cls == true_cls].sum()),
    }


def questions_to_identify(fam, beh, ep: Episode, policy: str, cfg: Config,
                          seed: int, cap: int = 8) -> list:
    """V11 on the CURRENT transfer distribution rather than the k=4
    microcase: from the memoryless prior, how many behavioural questions
    does each policy need before one behaviour is left? Run without history
    so the number measures the QUERY POLICY and not the convention
    posterior."""
    rng = random.Random(seed * 104729 + (0 if policy == "random" else 1))
    p0 = _uniform(fam)
    out = []
    for pos in ep.tr_idx:
        t = ep.tasks[pos]
        live = list(t.live)
        asked = set(t.demos)
        n = 0
        while len(live) > 1 and n < cap:
            b, _c, _best = infer(fam, p0, t.u, t.pool, live, t.tie)
            splits = []
            for k in range(len(UNIVERSE)):
                if k in asked:
                    continue
                d: dict = {}
                for j in live:
                    d[beh[j][k]] = d.get(beh[j][k], 0.0) + b[j]
                if len(d) > 1:
                    splits.append((-sum(x * math.log2(x)
                                        for x in d.values() if x > 0), k))
            if not splits:
                break
            k = (splits[rng.randrange(len(splits))][1] if policy == "random"
                 else max(splits)[1])
            asked.add(k)
            y = beh[t.z][k]
            live = [j for j in live if beh[j][k] == y]
            n += 1
        out.append(n)
    return out


def collapse_point(fam, beh, ep: Episode, cfg: Config, seed: int,
                   bits: float = 1.0) -> dict:
    """How much grounding the convention actually costs: the first task
    after which the convention posterior is below `bits`, and how many
    calibration tasks that took."""
    r = run_arm(fam, beh, ep, "persist", cfg, seed)
    ent = r["entropy"]
    for i, h in enumerate(ent):
        if h < bits:
            seen = sum(1 for pos in range(i) if ep.tasks[pos].kind == "cal")
            return {"tasks": i, "calibration_tasks": seen, "H": h}
    return {"tasks": None, "calibration_tasks": None, "H": ent[-1]}
