"""X65A-S: stable-identity streams.

An EPISODE is one appearance of one identity. The first appearance carries
grounded calibration tasks and then immediate transfer tasks; later
appearances carry only NEW transfer meanings, so a returning-identity gain
cannot come from a repeated complete meaning. Meanings are drawn without
replacement per identity, and no calibration meaning is ever used as a
transfer target.

The schedule places distractor identities and an out-of-family event between
first appearances and delayed returns, so the returns are separated by
10-20 intervening episodes rather than by the 7 that plain round-robin over
eight identities would give.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from x64h import episode as EP
from x64h import family as FAM


@dataclass
class Appearance:
    identity: int                 # generator slot, never exposed
    label: str                    # the opaque public ID
    index: int                    # position in the stream
    kind: str                     # first | return | distractor | unknown
    cal: tuple = ()               # grounded calibration tasks
    transfer: tuple = ()          # NEW meanings, the measured tasks
    phi: int = -1
    gap: int = 0                  # intervening episodes since last appearance


@dataclass
class Stream:
    family: str
    seed: int
    appearances: list
    identities: tuple
    phis: tuple
    restart_before: int
    meanings_used: dict = field(default_factory=dict)

    def returns(self):
        return [a for a in self.appearances if a.kind == "return"]

    def firsts(self):
        return [a for a in self.appearances if a.kind == "first"]


def _tasks_for(fam, beh, cfg, phi, meanings, rng, kind):
    """Build calibration or transfer tasks for specific meanings under a
    specific convention, reusing the frozen X64H generator."""
    out = []
    m = fam.m
    for j in meanings:
        if kind == "cal":
            d, live = EP.pick_demos(beh, m, j, 1, cfg.demos_cal_cap, rng)
            u = fam.realise(phi, j, ("O", "F", "S"))
            out.append(EP.Task("cal", j, d, live, u, FAM.CAL_POOL,
                               tuple(rng.sample(range(m), m))))
            continue
        made = None
        lo, hi = cfg.ambiguity
        for _ in range(cfg.retries):
            target = rng.randint(lo, hi)
            d, live = EP.pick_demos(beh, m, j, target, cfg.demos_transfer_cap,
                                    rng)
            if not (lo <= len(live) <= hi) or j not in live:
                continue
            good = [fam.realise(phi, j, p) for p in FAM.P2
                    if [k for k in live
                        if any(fam.realise(phi, k, q) == fam.realise(phi, j, p)
                               for q in FAM.P2)] == [j]]
            if not good:
                continue
            made = EP.Task("transfer", j, d, live,
                           good[rng.randrange(len(good))], FAM.P2,
                           tuple(rng.sample(range(m), m)))
            break
        if made is not None:
            out.append(made)
    return tuple(out)


def near_convention(fam, phi: int, rng) -> int:
    """A convention differing by one mapping or the order bit. The
    'wrong but similar' arm needs a genuinely near miss, not a random one."""
    cands = []
    for i in range(fam.n):
        d = int((fam.PO[i] != fam.PO[phi]).any()) \
            + int((fam.PF[i] != fam.PF[phi]).any()) \
            + int((fam.PS[i] != fam.PS[phi]).any()) \
            + int(fam.ORD[i] != fam.ORD[phi])
        if d == 1:
            cands.append(i)
        if len(cands) > 64:
            break
    return cands[rng.randrange(len(cands))] if cands else (phi + 1) % fam.n


def build_stream(fam, beh, cfg, seed: int, n_identities: int = 8,
                 n_distractors: int = 4, n_cal: int = 6, n_immediate: int = 3,
                 n_transfer: int = 3, order: str = "dependency") -> Stream:
    from .identity import assign
    rng = random.Random(seed)
    total = n_identities + n_distractors
    phis = tuple(rng.randrange(fam.n) for _ in range(total))
    ids = assign(seed, total, phis)

    meanings: dict = {}
    for i in range(total):
        pool = list(range(fam.m))
        rng.shuffle(pool)
        need = n_cal + n_immediate + 2 * n_transfer
        meanings[i] = pool[:need]

    def mk(i, kind, cal_ms, tr_ms, idx, gap):
        return Appearance(i, ids[i].label, idx, kind,
                          _tasks_for(fam, beh, cfg, phis[i], cal_ms, rng, "cal"),
                          _tasks_for(fam, beh, cfg, phis[i], tr_ms, rng,
                                     "transfer"),
                          phis[i], gap)

    apps: list = []
    pos = 0
    for i in range(n_identities):                      # first appearances
        ms = meanings[i]
        apps.append(mk(i, "first", ms[:n_cal],
                       ms[n_cal:n_cal + n_immediate], pos, 0))
        pos += 1
    for k in range(n_distractors):                     # irrelevant identities
        i = n_identities + k
        ms = meanings[i]
        apps.append(mk(i, "distractor", ms[:2], ms[2:4], pos, 0))
        pos += 1

    # an out-of-family event attributed to identity 0: a grounded pair
    # spoken under a DIFFERENT convention. With quarantine the record is
    # untouched; without it the record is overwritten.
    # the alien must genuinely DISAGREE on this meaning: two different
    # conventions can happen to realise one form identically, and an
    # "unknown" event that is accidentally consistent tests nothing
    z_unknown = meanings[0][0]
    alien = rng.randrange(fam.n)
    while (alien == phis[0]
           or fam.u3[alien, z_unknown] == fam.u3[phis[0], z_unknown]):
        alien = rng.randrange(fam.n)
    unk = Appearance(0, ids[0].label, pos, "unknown",
                     _tasks_for(fam, beh, cfg, alien, [z_unknown], rng,
                                "cal"), (), alien, 0)
    apps.append(unk)
    pos += 1
    restart_before = pos

    last = {a.identity: a.index for a in apps if a.kind == "first"}
    b = n_cal + n_immediate
    for wave in (0, 1):                                # two waves of returns
        seq = list(range(n_identities))
        if order == "reverse":
            seq = seq[::-1]
        elif order == "random":
            rng.shuffle(seq)
        elif order == "grouped":
            seq = sorted(seq)
        for i in seq:
            ms = meanings[i]
            tr = ms[b + wave * n_transfer: b + (wave + 1) * n_transfer]
            apps.append(mk(i, "return", (), tr, pos, pos - last[i]))
            last[i] = pos
            pos += 1

    if order == "reverse_recurrence":
        firsts = [a for a in apps if a.kind == "first"]
        rest = [a for a in apps if a.kind != "first"]
        apps = rest + firsts
        for k, a in enumerate(apps):
            a.index = k

    return Stream(fam.spec.overlap, seed, apps, ids, phis, restart_before,
                  meanings)


def schedule_summary(st: Stream) -> dict:
    gaps = [a.gap for a in st.returns()]
    return {"episodes": len(st.appearances),
            "identities": len({a.identity for a in st.appearances
                               if a.kind in ("first", "return")}),
            "distractors": len([a for a in st.appearances
                                if a.kind == "distractor"]),
            "unknown_events": len([a for a in st.appearances
                                   if a.kind == "unknown"]),
            "returns": len(gaps),
            "gap_min": min(gaps) if gaps else 0,
            "gap_max": max(gaps) if gaps else 0,
            "gap_mean": round(sum(gaps) / len(gaps), 2) if gaps else 0,
            "long_gap_returns": len([g for g in gaps if g >= 10]),
            "restart_before_episode": st.restart_before,
            "transfer_tasks": sum(len(a.transfer) for a in st.appearances),
            "calibration_tasks": sum(len(a.cal) for a in st.appearances),
            "meaning_overlap_cal_vs_transfer": sum(
                len({t.z for t in a.cal} & {t.z for a2 in st.appearances
                                            if a2.identity == a.identity
                                            for t in a2.transfer})
                for a in st.appearances)}
