"""X64H-0 validity: does the overlapping-codebook testbed actually require
convention inference?

Exact enumeration throughout: 1152 codebooks x 32 forms = 36,864 pairs per
task, so nothing is approximated and no candidate is capped.

    p(u | phi, z) = (number of exposure patterns realising u) / |PATTERNS|

which is a proper distribution over utterances, and

    p(phi, z | u, D) proportional to p(phi) p(z) p(u | phi, z) p(D | z).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from . import codebook as K
from . import semantic as S

NEG = -math.inf


def precompute(family, fs):
    """utt[i][j] : the utterance -> count map for codebook i and form j."""
    utt = []
    for phi in family:
        row = []
        for z in fs:
            c: dict[tuple[str, ...], int] = {}
            for p in K.PATTERNS:
                u = K.realise(phi, z, p)
                c[u] = c.get(u, 0) + 1
            row.append(c)
        utt.append(row)
    beh = [S.denote(z) for z in fs]
    return utt, beh


def demo_mask(fs, demos):
    return [all(S.execute(z)(x) == y for x, y in demos) for z in fs]


def posterior(utt, beh, log_p_phi, u, demos, fs, live_phi=None):
    """Exact joint over (phi, z). Returns log joint, the behaviour posterior
    and the convention marginal."""
    ok = demo_mask(fs, demos)
    lp = math.log(1.0 / 3.0)
    lj: dict[tuple[int, int], float] = {}
    idx = live_phi if live_phi is not None else range(len(log_p_phi))
    for i in idx:
        if log_p_phi[i] == NEG:
            continue
        row = utt[i]
        for j in range(len(fs)):
            if not ok[j]:
                continue
            c = row[j].get(u, 0)
            if c == 0:
                continue
            lj[(i, j)] = log_p_phi[i] + math.log(c) + lp
    if not lj:
        return {}, {}, [NEG] * len(log_p_phi)
    m = max(lj.values())
    tot = m + math.log(sum(math.exp(v - m) for v in lj.values()))
    bpost: dict = {}
    cpost = [NEG] * len(log_p_phi)
    for (i, j), v in lj.items():
        w = math.exp(v - tot)
        bpost[beh[j]] = bpost.get(beh[j], 0.0) + w
        cpost[i] = v if cpost[i] == NEG else _lse(cpost[i], v)
    cpost = [c - tot if c != NEG else NEG for c in cpost]
    return lj, bpost, cpost


def _lse(a, b):
    if a == NEG:
        return b
    if b == NEG:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def entropy(logps) -> float:
    h = 0.0
    for lp in logps:
        if lp == NEG:
            continue
        p = math.exp(lp)
        if p > 0:
            h -= p * math.log2(p)
    return h


@dataclass
class EpisodeResult:
    correct: list[bool]
    entropy: list[float]
    true_class_mass: list[float]
    n_classes: list[int]
    withholding: list[bool] = None

    def __post_init__(self):
        if self.withholding is None:
            self.withholding = []


def _agrees_on(z, target, roles) -> bool:
    for r in roles:
        a = {"O": "op", "F": "filt", "S": "scope"}[r]
        if getattr(z, a) != getattr(target, a):
            return False
    return True


def _exposed_roles(fs, utt, phi_idx, z_idx, u):
    """Which two roles this utterance actually showed. Known to the TEACHER
    only; no arm receives it."""
    from . import codebook as K
    phi_row = utt[phi_idx][z_idx]
    if phi_row.get(u, 0) == 0:
        return ("O", "F")
    return ("O", "F")


def teacher_demo(fs, utt, phi_idx, z_idx, u, n_demos, universe,
                 exposed=("O", "F"), alpha=1.0):
    """The teacher shows an example that resolves what the utterance leaves
    open.

    With a fixed random demonstration the ORACLE only reached 0.771, because
    an utterance exposes two of three roles and one demonstration did not
    pin the omitted one. The specification asks for demonstrations that
    leave ambiguity `primarily in the omitted or permuted role`, which means
    the example has to be chosen against that role rather than at random --
    a property of the teacher, not of the learner, and available to every
    arm equally.
    """
    # The demonstration resolves the OMITTED role and is deliberately
    # uninformative about the exposed ones. Discriminating across all roles
    # made the demonstrations alone leave 1.33 behaviour classes, so the
    # task was nearly solved before language was consulted and V10 failed:
    # the specification asks for 2-8 classes BEFORE language and history.
    target = fs[z_idx]
    live = [j for j in range(len(fs))
            if _agrees_on(fs[j], target, exposed)]
    f = S.execute(target)
    chosen: list[tuple[str, str]] = []
    everything = list(range(len(fs)))
    for step in range(n_demos):
        best, score = None, None
        for x in universe:
            if any(x == c[0] for c in chosen):
                continue
            inner = len({S.execute(fs[j])(x) for j in live})
            outer = len({S.execute(fs[j])(x) for j in everything})
            # resolve the omitted role (high `inner`) while saying as little
            # as possible about the exposed ones (low `outer`). Maximising
            # `inner` alone let the demonstrations incidentally resolve the
            # exposed roles too, leaving 1.70 classes before language and
            # failing V10.
            # Penalising only the FIRST demonstration was tried: it lifted
            # the oracle from 0.960 to 0.972, still short of 0.98, and cost
            # V10 -- the later demonstrations then revealed the exposed
            # roles and memoryless ambiguity fell from 4.15 classes to 1.82.
            # Rejected; the penalty applies throughout.
            k = (inner, -alpha * outer)
            if score is None or k > score:
                best, score = x, k
        if best is None:
            break
        chosen.append((best, f(best)))
        live = [j for j in live if S.execute(fs[j])(best) == f(best)]
    return tuple(chosen)


def run_episode(family, fs, utt, beh, classes, phi_idx, zs, arm,
                n_demos=1, rng=None, alt_phi=None, changing=None,
                alpha=1.0, teacher_universe=24, schedule=0):
    """One episode under one convention.

    arm: oracle | static | persist | reset | shuffled | default
    """
    rng = rng or random.Random(0)
    n = len(family)
    uniform = [-math.log(n)] * n
    true_class = next(v for v in classes.values() if phi_idx in v)

    if arm == "oracle":
        prior = [NEG] * n
        prior[phi_idx] = 0.0
    elif arm == "default":
        prior = [NEG] * n
        prior[0] = 0.0
    elif arm == "shuffled":
        prior = [NEG] * n
        prior[alt_phi] = 0.0
    else:
        prior = list(uniform)

    out = EpisodeResult([], [], [], [], [])
    for t, z in enumerate(zs):
        pidx = phi_idx if changing is None else changing[t]
        phi = family[pidx]
        f = S.execute(z)
        pat = K.PATTERNS[rng.randrange(len(K.PATTERNS))]
        u = K.realise(phi, z, pat)
        # A MIXED SCHEDULE. Teaching tasks (alpha = 0) let the
        # demonstrations reveal what the codewords mean, so the codebook can
        # be learned; withholding tasks (alpha > 0) keep the exposed roles
        # open, so the memoryless task stays ambiguous and the utterance has
        # to carry them. A uniform schedule cannot have both: the sweep
        # showed alpha = 0 giving learnability with no ambiguity and
        # alpha > 0 giving ambiguity with recovery collapsing to 0.14.
        a_t = 0.0 if (schedule and t % schedule == 0) else alpha
        demos = teacher_demo(fs, utt, pidx, fs.index(z), u, n_demos,
                             S.UNIVERSE[:teacher_universe], exposed=pat,
                             alpha=a_t)
        out.withholding.append(a_t > 0)
        use = prior if arm in ("oracle", "persist", "default", "shuffled") \
            else list(uniform)
        lj, bpost, cpost = posterior(utt, beh, use, u, demos, fs)
        if not bpost:
            lj, bpost, cpost = posterior(utt, beh, list(uniform), u, demos, fs)
        top = max(bpost, key=bpost.get) if bpost else None
        out.correct.append(top == S.denote(z))
        out.n_classes.append(len(bpost))
        out.entropy.append(entropy(cpost))
        out.true_class_mass.append(
            sum(math.exp(cpost[i]) for i in true_class if cpost[i] != NEG))
        if arm == "persist":
            prior = list(cpost)
    return out
